"""Deterministic doubles used by the unit tests.

The vector store deliberately ignores metadata filters so that the collection
isolation tests exercise the service's own guard rather than a cooperative
backend.
"""

import hashlib
from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from app.parsing import ConvertedDocument, DocumentSegment, build_converted_document
from app.repository import (
    DocumentRecord,
    EmbeddingState,
    JobRecord,
    Repository,
    SourceObjectRecord,
    SourceRecord,
    merge_sources,
)
from app.storage import SourceObjectStore, StoredObject, StoredObjectMissingError

VOCAB = (
    "alarm",
    "battery",
    "calibration",
    "expiratory",
    "filter",
    "inspiratory",
    "maintenance",
    "oxygen",
    "pressure",
    "sensor",
    "valve",
    "ventilator",
)


class DeterministicEmbedding(BaseEmbedding):
    """Keyword-presence embeddings: no network, identical text -> identical vector."""

    _vocab: tuple[str, ...] = PrivateAttr()

    def __init__(self, vocab: tuple[str, ...] = VOCAB) -> None:
        super().__init__(model_name="deterministic", embed_batch_size=8)
        self._vocab = vocab

    @classmethod
    def class_name(cls) -> str:
        return "deterministic"

    @property
    def dim(self) -> int:
        return len(self._vocab)

    def _vectorize(self, text: str) -> list[float]:
        lowered = text.lower()
        return [1.0 if token in lowered else 0.0 for token in self._vocab]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vectorize(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vectorize(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(text) for text in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._get_text_embeddings(texts)


class RecordingVectorStore(BasePydanticVectorStore):
    """In-memory vector store that records queries and ignores metadata filters."""

    stores_text: bool = True
    is_embedding_query: bool = True

    _nodes: dict[str, BaseNode] = PrivateAttr(default_factory=dict)
    _queries: list[VectorStoreQuery] = PrivateAttr(default_factory=list)
    _cleared: int = PrivateAttr(default=0)
    _deleted_refs: list[str] = PrivateAttr(default_factory=list)

    @property
    def client(self) -> None:
        return None

    @property
    def queries(self) -> list[VectorStoreQuery]:
        return self._queries

    @property
    def deleted_refs(self) -> list[str]:
        return self._deleted_refs

    @property
    def clear_count(self) -> int:
        return self._cleared

    @property
    def nodes(self) -> dict[str, BaseNode]:
        return self._nodes

    def add(self, nodes: list[BaseNode], **kwargs: Any) -> list[str]:
        for node in nodes:
            self._nodes[str(node.node_id)] = node
        return [str(node.node_id) for node in nodes]

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        self._deleted_refs.append(ref_doc_id)
        for node_id in [
            node_id
            for node_id, node in self._nodes.items()
            if node.ref_doc_id == ref_doc_id or node.metadata.get("document_id") == ref_doc_id
        ]:
            self._nodes.pop(node_id, None)

    def clear(self) -> None:
        self._cleared += 1
        self._nodes.clear()

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        self._queries.append(query)
        if not query.query_embedding:
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        scored: list[tuple[float, BaseNode]] = []
        for node in self._nodes.values():
            if not node.embedding:
                continue
            score = sum(a * b for a, b in zip(node.embedding, query.query_embedding, strict=False))
            scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        top = scored[: (query.similarity_top_k or len(scored))]
        return VectorStoreQueryResult(
            nodes=[node for _, node in top],
            similarities=[score for score, _ in top],
            ids=[str(node.node_id) for _, node in top],
        )


class RecordingConverter:
    """Document converter double: records calls and can be made to fail.

    ``segments`` is what a provenance-carrying converter returns; setting it to
    ``None`` models the Markdown-only fallback, which reports no page or section.
    """

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.markdown = "# Converted\n\nThe expiratory valve needs calibration."
        self.segments: list[DocumentSegment] | None = None
        self.error: Exception | None = None

    async def convert(
        self, *, filename: str, media_type: str, content: bytes
    ) -> ConvertedDocument:
        self.calls.append({"filename": filename, "media_type": media_type, "content": content})
        if self.error is not None:
            raise self.error
        if self.segments is not None:
            return build_converted_document(list(self.segments))
        return build_converted_document([DocumentSegment(text=self.markdown)])


class InMemorySourceStore(SourceObjectStore):
    """Source storage double that keeps objects in a dict."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.put_error: Exception | None = None

    @property
    def backend(self) -> str:
        return "memory"

    async def put(self, key: str, content: bytes) -> StoredObject:
        if self.put_error is not None:
            raise self.put_error
        self.objects[key] = content
        return StoredObject(
            key=key, byte_size=len(content), checksum=hashlib.sha256(content).hexdigest()
        )

    async def stat(self, key: str) -> StoredObject | None:
        content = self.objects.get(key)
        if content is None:
            return None
        return StoredObject(key=key, byte_size=len(content), checksum="")

    async def read(
        self, key: str, *, offset: int = 0, length: int | None = None
    ) -> AsyncIterator[bytes]:
        content = self.objects.get(key)
        if content is None:
            raise StoredObjectMissingError("The stored source file is no longer available.")
        window = content[offset:] if length is None else content[offset : offset + length]
        # Deliberately chunked so range assembly is exercised, not just sliced.
        for start in range(0, len(window), 8):
            yield window[start : start + 8]

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class InMemoryRepository(Repository):
    """Repository double whose state survives being handed to a new application."""

    def __init__(self) -> None:
        self.collections: dict[str, datetime] = {}
        self.documents: dict[tuple[str, str], DocumentRecord] = {}
        self.jobs: dict[UUID, JobRecord] = {}
        self.source_objects: dict[tuple[str, str], SourceObjectRecord] = {}
        self.embedding_state: EmbeddingState | None = None
        self.probe_error: Exception | None = None

    async def ensure_collection(self, collection_id: str) -> None:
        self.collections[collection_id] = datetime.now(UTC)

    async def get_document(self, collection_id: str, external_id: str) -> DocumentRecord | None:
        return self.documents.get((collection_id, external_id))

    async def upsert_document(self, record: DocumentRecord) -> None:
        now = datetime.now(UTC)
        existing = self.documents.get((record.collection_id, record.external_id))
        self.documents[(record.collection_id, record.external_id)] = replace(
            record,
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

    async def list_sources(self, collection_id: str, *, limit: int) -> list[SourceRecord]:
        return merge_sources(
            [
                document
                for (collection, _), document in self.documents.items()
                if collection == collection_id
            ],
            [job for job in self.jobs.values() if job.collection_id == collection_id],
            [
                record
                for (collection, _), record in self.source_objects.items()
                if collection == collection_id
            ],
            limit=limit,
        )

    async def upsert_source_object(self, record: SourceObjectRecord) -> None:
        now = datetime.now(UTC)
        key = (record.collection_id, record.external_id)
        existing = self.source_objects.get(key)
        self.source_objects[key] = replace(
            record,
            # An upsert refreshes the bytes but keeps whatever preview the last
            # successful conversion produced, matching the SQL implementation.
            preview_key=record.preview_key or (existing.preview_key if existing else None),
            preview_bytes=record.preview_bytes or (existing.preview_bytes if existing else None),
            preview_checksum=record.preview_checksum or (existing.preview_checksum if existing else None),
            page_count=record.page_count if record.page_count is not None else (existing.page_count if existing else None),
            created_at=existing.created_at if existing else now,
            updated_at=now,
        )

    async def set_source_preview(
        self,
        document_id: UUID,
        *,
        preview_key: str | None,
        preview_bytes: int | None,
        preview_checksum: str | None,
        page_count: int | None,
    ) -> None:
        for key, record in self.source_objects.items():
            if record.document_id == document_id:
                self.source_objects[key] = replace(
                    record,
                    preview_key=preview_key,
                    preview_bytes=preview_bytes,
                    preview_checksum=preview_checksum,
                    page_count=page_count,
                    updated_at=datetime.now(UTC),
                )
                return

    async def get_source_object(
        self, collection_id: str, external_id: str
    ) -> SourceObjectRecord | None:
        return self.source_objects.get((collection_id, external_id))

    async def create_job(self, record: JobRecord) -> None:
        now = datetime.now(UTC)
        self.jobs[record.job_id] = replace(record, created_at=now, updated_at=now)

    async def update_job(
        self,
        job_id: UUID,
        *,
        status: str,
        document_id: UUID | None = None,
        chunk_count: int | None = None,
        unchanged: bool | None = None,
        detail: str | None = None,
    ) -> None:
        job = self.jobs[job_id]
        self.jobs[job_id] = replace(
            job,
            status=status,
            document_id=document_id if document_id is not None else job.document_id,
            chunk_count=chunk_count if chunk_count is not None else job.chunk_count,
            unchanged=unchanged if unchanged is not None else job.unchanged,
            detail=detail,
            updated_at=datetime.now(UTC),
        )

    async def get_job(self, job_id: UUID) -> JobRecord | None:
        return self.jobs.get(job_id)

    async def get_latest_job(self, collection_id: str, external_id: str) -> JobRecord | None:
        matching = [
            job
            for job in self.jobs.values()
            if job.collection_id == collection_id and job.external_id == external_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda job: job.created_at or datetime.fromtimestamp(0, tz=UTC))

    async def fail_interrupted_jobs(self, detail: str) -> int:
        failed = 0
        for job_id, job in list(self.jobs.items()):
            if job.status in ("accepted", "processing"):
                self.jobs[job_id] = replace(
                    job, status="failed", detail=detail, updated_at=datetime.now(UTC)
                )
                failed += 1
        return failed

    async def get_embedding_state(self) -> EmbeddingState | None:
        return self.embedding_state

    async def set_embedding_state(self, state: EmbeddingState) -> None:
        self.embedding_state = state

    async def probe(self) -> None:
        if self.probe_error is not None:
            raise self.probe_error
