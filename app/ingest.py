"""Idempotent text-document ingestion with durable job state."""

import asyncio
import hashlib
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4, uuid5, NAMESPACE_URL

from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import NodeRelationship, ObjectType, RelatedNodeInfo, TextNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from app.embeddings.pipeline import delete_document_nodes, run_ingestion
from app.log import logger
from app.models import RESERVED_METADATA_KEYS, TextDocumentRequest
from app.parsing import (
    ConvertedDocument,
    DocumentError,
    DocumentNormalizer,
    DocumentSegment,
    UploadedFile,
    resolve_format,
)
from app.repository import DocumentRecord, JobRecord, Repository, SourceObjectRecord
from app.storage import (
    ORIGINAL_VARIANT,
    PREVIEW_VARIANT,
    SourceObjectStore,
    StorageError,
    storage_key,
)

MAXIMUM_JOB_DETAIL_CHARACTERS = 500
INTERRUPTED_JOB_DETAIL = "Ingestion was interrupted by a service restart."
UPLOAD_EXTERNAL_ID_PREFIX = "file-"
UNEXPECTED_JOB_DETAIL = "Document processing failed unexpectedly. Try again or ask the service owner to check the logs."
# Left on a job whose converter task cannot be picked up again because this
# deployment now talks to a different converter than the one holding it.
CONVERTER_CHANGED_DETAIL = (
    "The document converter that was processing this file is no longer configured on this "
    "service. Upload the file again."
)
UNRESUMABLE_UPLOAD_DETAIL = (
    "The upload this conversion belongs to can no longer be identified. Upload the file again."
)
_UNFINISHED_JOB_STATUSES = ("accepted", "processing")


class MetadataConflictError(ValueError):
    """Raised when caller metadata would overwrite a service-owned key."""


def document_uuid(collection_id: str, external_id: str) -> UUID:
    """Stable document identity for a collection-scoped external ID."""
    # Keep the namespace stable so moving the service does not orphan indexed documents.
    return uuid5(NAMESPACE_URL, f"athena:{collection_id}:{external_id}")


def text_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def upload_external_id(content: bytes, extension: str) -> str:
    """Content-derived identity, so re-uploading the same file is idempotent.

    The extension is part of the identity because the same bytes parsed as two
    different formats are two different documents.
    """
    digest = hashlib.sha256(f"{extension.lower()}\0".encode("utf-8") + content).hexdigest()
    return f"{UPLOAD_EXTERNAL_ID_PREFIX}{digest}"


@dataclass(frozen=True, slots=True)
class UploadSubmission:
    """One accepted file upload awaiting normalization and indexing."""

    collection_id: str
    external_id: str
    title: str
    upload: UploadedFile


@dataclass(slots=True)
class _DocumentLock:
    """A per-document lock plus the number of holders waiting on it."""

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


class IngestionService:
    """Owns the ingest lifecycle: job creation, change detection, and indexing."""

    def __init__(
        self,
        *,
        repository: Repository,
        vector_store: BasePydanticVectorStore,
        pipeline: IngestionPipeline,
        normalizer: DocumentNormalizer | None = None,
        source_store: SourceObjectStore | None = None,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._pipeline = pipeline
        self._normalizer = normalizer
        self._source_store = source_store
        self._locks: dict[tuple[str, str], _DocumentLock] = {}
        # Held only while an upload decides between reusing a pending job and
        # creating one. It is separate from the per-document lock because that one
        # is held for the whole conversion, which can be an hour long.
        self._accept_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def _document_lock(self, key: tuple[str, str]) -> AsyncIterator[None]:
        """Serialize ingestion per document without leaking a lock per key."""
        entry = self._locks.get(key)
        if entry is None:
            entry = self._locks[key] = _DocumentLock()
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                self._locks.pop(key, None)

    async def submit(self, request: TextDocumentRequest) -> JobRecord:
        """Record an accepted job and index the document in the background."""
        _reject_reserved_metadata(request.metadata)
        return await self._accept(
            request.collection_id,
            request.external_id,
            lambda job_id: self.run(job_id, request),
            title=request.title,
            source_type=request.source_type,
        )

    async def retain_original(self, submission: UploadSubmission) -> None:
        """Persist the uploaded bytes durably, before any background work starts.

        Bytes are written first and the row second: a crash between the two leaves
        an unreferenced file that the next upload of the same content overwrites,
        whereas the other order would leave a row pointing at nothing. Both steps
        are keyed by the document UUID, so re-uploading identical content replaces
        the same object rather than accumulating copies.
        """
        if self._source_store is None:
            return
        document_id = document_uuid(submission.collection_id, submission.external_id)
        key = storage_key(document_id, ORIGINAL_VARIANT)
        upload = submission.upload
        stored = await self._source_store.put(key, upload.content)
        await self._repository.upsert_source_object(
            SourceObjectRecord(
                document_id=document_id,
                collection_id=submission.collection_id,
                external_id=submission.external_id,
                filename=upload.filename,
                media_type=upload.media_type,
                byte_size=stored.byte_size,
                checksum=stored.checksum,
                storage_backend=self._source_store.backend,
                storage_key=stored.key,
            )
        )

    async def submit_upload(self, submission: UploadSubmission) -> JobRecord:
        """Record an accepted job, then convert and index the file in the background.

        Conversion happens inside the job rather than inside the HTTP request so a
        slow PDF conversion is observable as ``processing`` instead of holding the
        upload connection open.

        Re-uploading a file whose job is still running returns that job instead of
        starting a second one. Uploads are identified by their content, so the two
        requests are the same work; converting twice would double the cost of the
        conversion and, for a remote converter, the bill for it.
        """
        async with self._accept_lock:
            pending = await self._repository.get_latest_job(
                submission.collection_id, submission.external_id
            )
            if pending is not None and pending.status in _UNFINISHED_JOB_STATUSES:
                logger.info(
                    "Upload %s/%s is already being processed by job %s",
                    submission.collection_id,
                    submission.external_id,
                    pending.job_id,
                )
                return pending
            return await self._accept(
                submission.collection_id,
                submission.external_id,
                lambda job_id: self.run_upload(job_id, submission),
                title=submission.title,
                source_type=submission.upload.format.source_type,
                filename=submission.upload.filename,
                media_type=submission.upload.media_type,
            )

    async def _accept(
        self,
        collection_id: str,
        external_id: str,
        start: Callable[[UUID], Coroutine[Any, Any, None]],
        *,
        title: str = "",
        source_type: str = "text",
        filename: str | None = None,
        media_type: str | None = None,
    ) -> JobRecord:
        job = JobRecord(
            job_id=uuid4(),
            collection_id=collection_id,
            external_id=external_id,
            status="accepted",
            title=title,
            source_type=source_type,
            filename=filename,
            media_type=media_type,
        )
        await self._repository.create_job(job)
        self._spawn(job.job_id, start)
        return job

    async def resume_conversions(self) -> int:
        """Pick up converter tasks that outlived the last Athena process.

        Returns the number of jobs resumed. A job whose task belongs to a
        converter this deployment no longer uses is failed here with a retryable
        reason rather than left waiting on something nobody will poll.
        """
        jobs = await self._repository.list_resumable_jobs()
        converter = (
            self._normalizer.resumable_converter if self._normalizer is not None else None
        )
        resumed = 0
        for job in jobs:
            if converter is None or converter.name != job.converter_name:
                logger.warning(
                    "Cannot resume conversion %s for job %s: converter %r is not configured",
                    job.converter_task_id,
                    job.job_id,
                    job.converter_name,
                )
                await self._repository.update_job(
                    job.job_id, status="failed", detail=CONVERTER_CHANGED_DETAIL
                )
                continue
            logger.info(
                "Resuming converter task %s for job %s", job.converter_task_id, job.job_id
            )
            self._spawn(job.job_id, lambda _job_id, job=job: self.resume_upload(job))
            resumed += 1
        return resumed

    def _spawn(
        self,
        job_id: UUID,
        start: Callable[[UUID], Coroutine[Any, Any, None]],
    ) -> None:
        task = asyncio.create_task(self._run_guarded(job_id, start))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_for_pending(self) -> None:
        """Await in-flight ingestion so shutdown does not orphan job state."""
        if not self._tasks:
            return
        await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _run_guarded(
        self,
        job_id: UUID,
        start: Callable[[UUID], Coroutine[Any, Any, None]],
    ) -> None:
        try:
            await start(job_id)
        except Exception:  # pragma: no cover - run()/run_upload() record failures
            logger.exception("Ingestion job %s crashed", job_id)

    async def run(self, job_id: UUID, request: TextDocumentRequest) -> None:
        """Execute one ingestion job, recording every state transition."""
        async with self._document_lock((request.collection_id, request.external_id)):
            await self._repository.update_job(job_id, status="processing")
            try:
                await self._index(job_id, request)
            except Exception as exc:
                await self._record_failure(
                    job_id, request.collection_id, request.external_id, exc
                )

    async def run_upload(self, job_id: UUID, submission: UploadSubmission) -> None:
        """Normalize an uploaded file and index the text it produced."""
        async with self._document_lock((submission.collection_id, submission.external_id)):
            await self._repository.update_job(job_id, status="processing")
            try:
                converted = await self._convert(job_id, submission)
                await self._finish_upload(job_id, submission, converted)
            except Exception as exc:
                await self._record_failure(
                    job_id, submission.collection_id, submission.external_id, exc
                )

    async def resume_upload(self, job: JobRecord) -> None:
        """Continue a conversion that was submitted before the last restart.

        Nothing is resubmitted: the converter still holds the file under the task
        id recorded on the job, and its own deadline still runs from when it was
        submitted, so a task that has since expired fails here rather than being
        waited on forever.
        """
        async with self._document_lock((job.collection_id, job.external_id)):
            await self._repository.update_job(job.job_id, status="processing")
            try:
                submission = self._resumed_submission(job)
                converted = await self._await_conversion(job)
                await self._finish_upload(job.job_id, submission, converted)
            except Exception as exc:
                await self._record_failure(
                    job.job_id, job.collection_id, job.external_id, exc
                )

    def _resumed_submission(self, job: JobRecord) -> UploadSubmission:
        """Rebuild the upload identity a resumed job needs, from its own row.

        The file's bytes are deliberately not reloaded: the converter already has
        them, and everything left to do -- metadata, the preview, the index --
        needs only the identity and the format.
        """
        if self._normalizer is None or not job.filename:  # pragma: no cover - defensive
            raise DocumentError(UNRESUMABLE_UPLOAD_DETAIL)
        upload_format = resolve_format(job.filename, job.media_type)
        return UploadSubmission(
            collection_id=job.collection_id,
            external_id=job.external_id,
            title=job.title,
            upload=UploadedFile(
                filename=job.filename,
                media_type=job.media_type or upload_format.canonical_media_type,
                content=b"",
                format=upload_format,
            ),
        )

    async def _convert(
        self, job_id: UUID, submission: UploadSubmission
    ) -> ConvertedDocument:
        """Produce normalized text for *submission*, recording resumable work first."""
        if self._normalizer is None:  # pragma: no cover - defensive
            raise RuntimeError("File uploads are not enabled on this service.")
        converter = self._normalizer.resumable_converter
        if not submission.upload.format.needs_conversion or converter is None:
            return await self._normalizer.to_document(submission.upload)

        task_id = await self._normalizer.submit_conversion(submission.upload)
        submitted_at = datetime.now(UTC)
        # Persisted before the first poll, so a restart one moment later still
        # finds the remote task instead of resubmitting the document.
        await self._repository.set_job_conversion_task(
            job_id,
            converter_name=converter.name,
            task_id=task_id,
            submitted_at=submitted_at,
        )
        logger.info(
            "Submitted %s to %s as task %s",
            submission.upload.filename,
            converter.name,
            task_id,
        )
        return await self._normalizer.await_conversion(task_id, submitted_at=submitted_at)

    async def _await_conversion(self, job: JobRecord) -> ConvertedDocument:
        if self._normalizer is None:  # pragma: no cover - defensive
            raise RuntimeError("File uploads are not enabled on this service.")
        if not job.converter_task_id or job.converter_submitted_at is None:  # pragma: no cover
            raise DocumentError(UNRESUMABLE_UPLOAD_DETAIL)
        return await self._normalizer.await_conversion(
            job.converter_task_id, submitted_at=job.converter_submitted_at
        )

    async def _finish_upload(
        self,
        job_id: UUID,
        submission: UploadSubmission,
        converted: ConvertedDocument,
    ) -> None:
        request = _upload_request(submission, converted)
        # Recorded before indexing so a source stays viewable even when
        # embedding later fails: the tester can still open what was read.
        await self._record_normalized_form(submission, converted)
        await self._index(job_id, request, converted.segments)

    async def _record_failure(
        self,
        job_id: UUID,
        collection_id: str,
        external_id: str,
        exc: Exception,
    ) -> None:
        # A DocumentError is an expected outcome for a bad or unconvertible file,
        # so it is logged as a warning rather than as a service fault.
        if isinstance(exc, DocumentError):
            logger.warning("Ingestion rejected %s/%s: %s", collection_id, external_id, exc)
        else:
            logger.exception("Ingestion failed for %s/%s", collection_id, external_id)
        await self._repository.update_job(job_id, status="failed", detail=_job_detail(exc))

    async def _record_normalized_form(
        self, submission: UploadSubmission, converted: ConvertedDocument
    ) -> None:
        """Store the normalized text and page reach for the viewer.

        A format that needs conversion gets a stored preview, because its original
        bytes cannot be shown safely inline. Text and Markdown do not: their
        original *is* the readable form, so a second copy would only drift.
        """
        if self._source_store is None:
            return
        document_id = document_uuid(submission.collection_id, submission.external_id)
        preview_key: str | None = None
        preview_bytes: int | None = None
        preview_checksum: str | None = None
        if submission.upload.format.needs_conversion:
            key = storage_key(document_id, PREVIEW_VARIANT)
            try:
                stored = await self._source_store.put(key, converted.text.encode("utf-8"))
            except StorageError:
                # A missing preview degrades the viewer; it must not fail the
                # ingest that already produced searchable text.
                logger.warning(
                    "Could not store the normalized preview for %s/%s",
                    submission.collection_id,
                    submission.external_id,
                    exc_info=True,
                )
            else:
                preview_key = stored.key
                preview_bytes = stored.byte_size
                # Recorded so the preview's HTTP validator tracks the extracted
                # text rather than the original: converting the same file again can
                # produce different text, and a caller must not be told otherwise.
                preview_checksum = stored.checksum
        await self._repository.set_source_preview(
            document_id,
            preview_key=preview_key,
            preview_bytes=preview_bytes,
            preview_checksum=preview_checksum,
            page_count=converted.page_count,
        )

    async def _index(
        self,
        job_id: UUID,
        request: TextDocumentRequest,
        segments: tuple[DocumentSegment, ...] = (),
    ) -> None:
        checksum = request.checksum or text_checksum(request.text)
        document_id = document_uuid(request.collection_id, request.external_id)

        await self._repository.ensure_collection(request.collection_id)
        existing = await self._repository.get_document(
            request.collection_id, request.external_id
        )

        if (
            existing is not None
            and existing.checksum == checksum
            and existing.chunk_count > 0
            and not request.force_reindex
        ):
            logger.info(
                "Document %s/%s is unchanged; skipping re-embedding",
                request.collection_id,
                request.external_id,
            )
            await self._repository.update_job(
                job_id,
                status="completed",
                document_id=existing.document_id,
                chunk_count=existing.chunk_count,
                unchanged=True,
                detail=None,
            )
            return

        pending_record = DocumentRecord(
            document_id=document_id,
            collection_id=request.collection_id,
            external_id=request.external_id,
            title=request.title,
            source_type=request.source_type,
            source_uri=request.source_uri,
            checksum=checksum,
            version=request.version,
            page=request.page,
            section=request.section,
            metadata=dict(request.metadata),
            chunk_count=0,
        )
        # Persist the invalid state before writing vectors. If deletion,
        # embedding, or the process then fails, a retry cannot incorrectly
        # treat an empty or partially written index as unchanged.
        await self._repository.upsert_document(pending_record)

        if existing is not None:
            # Scoped to one document; unrelated collections are never touched.
            await delete_document_nodes(self._vector_store, str(document_id))

        updated_at = datetime.now(UTC)
        nodes = _build_nodes(request, document_id, checksum, updated_at, segments)
        chunk_count = await run_ingestion(self._pipeline, nodes)

        await self._repository.upsert_document(replace(pending_record, chunk_count=chunk_count))
        await self._repository.update_job(
            job_id,
            status="completed",
            document_id=document_id,
            chunk_count=chunk_count,
            unchanged=False,
            detail=None,
        )
        logger.info(
            "Indexed %d chunk(s) for %s/%s",
            chunk_count,
            request.collection_id,
            request.external_id,
        )


def _upload_request(
    submission: UploadSubmission, converted: ConvertedDocument
) -> TextDocumentRequest:
    """The normalized-text request an uploaded file reduces to."""
    upload = submission.upload
    return TextDocumentRequest(
        collection_id=submission.collection_id,
        external_id=submission.external_id,
        text=converted.text,
        title=submission.title,
        source_type=upload.format.source_type,
        source_uri=f"upload:{submission.external_id}",
        metadata={"filename": upload.filename, "media_type": upload.media_type},
    )


def _build_nodes(
    request: TextDocumentRequest,
    document_id: UUID,
    checksum: str,
    updated_at: datetime,
    segments: tuple[DocumentSegment, ...] = (),
) -> list[TextNode]:
    """Build one input node per located segment of this document.

    Every node points at the same source id, so the chunks the splitter derives
    from them are all deleted together and never touch another document. What
    differs per node is only the location: a segment's own page and section.
    """
    located = segments or (
        DocumentSegment(text=request.text, page=request.page, section=request.section),
    )
    return [
        _build_node(
            request,
            document_id,
            checksum,
            updated_at,
            text=segment.text,
            page=segment.page if segment.page is not None else request.page,
            section=segment.section or request.section,
        )
        for segment in located
    ]


def _build_node(
    request: TextDocumentRequest,
    document_id: UUID,
    checksum: str,
    updated_at: datetime,
    *,
    text: str,
    page: int | None,
    section: str,
) -> TextNode:
    metadata: dict[str, object] = {
        "collection_id": request.collection_id,
        "document_id": str(document_id),
        "external_id": request.external_id,
        "title": request.title,
        "source_type": request.source_type,
        "source_uri": request.source_uri,
        "checksum": checksum,
        "version": request.version,
        "section": section,
        "updated_at": updated_at.isoformat(),
        "updated_at_ts": int(updated_at.timestamp()),
    }
    if page is not None:
        metadata["page"] = page
    metadata.update(request.metadata)

    node = TextNode(text=text, metadata=metadata)
    node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=str(document_id), node_type=ObjectType.DOCUMENT
    )
    # Metadata is bookkeeping, not content: embeddings and prompts use chunk text
    # only, so retrieval stays deterministic for identical text.
    node.excluded_embed_metadata_keys = list(metadata)
    node.excluded_llm_metadata_keys = list(metadata)
    return node


def _reject_reserved_metadata(metadata: dict[str, object]) -> None:
    conflicts = sorted(set(metadata) & RESERVED_METADATA_KEYS)
    if conflicts:
        raise MetadataConflictError(
            "metadata may not contain service-owned keys: " + ", ".join(conflicts)
        )


def _job_detail(exc: Exception) -> str:
    # DocumentError messages are deliberately user-safe. Unexpected exceptions
    # stay in service logs; provider URLs and database diagnostics must not be
    # persisted into API-visible job state.
    detail = str(exc) if isinstance(exc, DocumentError) else UNEXPECTED_JOB_DETAIL
    if len(detail) > MAXIMUM_JOB_DETAIL_CHARACTERS:
        return f"{detail[:MAXIMUM_JOB_DETAIL_CHARACTERS]}…"
    return detail
