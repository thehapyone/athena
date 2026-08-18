"""Durable state: collection identity, document identity, and ingestion jobs.

All ingestion state lives in PostgreSQL so a service restart never loses job or
document status. The in-process layer above this module keeps no authoritative
state of its own.
"""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import asyncpg

from app.models import JobStatus, SourceStatus


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    document_id: UUID
    collection_id: str
    external_id: str
    title: str
    source_type: str
    source_uri: str
    checksum: str
    version: str
    page: int | None
    section: str
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_count: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class JobRecord:
    job_id: UUID
    collection_id: str
    external_id: str
    status: JobStatus
    title: str = ""
    source_type: str = "text"
    filename: str | None = None
    media_type: str | None = None
    document_id: UUID | None = None
    chunk_count: int = 0
    unchanged: bool = False
    detail: str | None = None
    # Identity of the converter task this upload was handed to, recorded before
    # any polling starts. It is what makes a conversion resumable: after a
    # restart the job is continued against the same remote task instead of the
    # file being converted, and paid for, a second time.
    converter_name: str | None = None
    converter_task_id: str | None = None
    converter_submitted_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def resumable(self) -> bool:
        """Whether this job is waiting on remote work that outlived Athena."""
        return bool(
            self.status in ("accepted", "processing")
            and self.converter_task_id
            and self.converter_submitted_at is not None
        )


@dataclass(frozen=True, slots=True)
class SourceObjectRecord:
    """The original bytes kept for one uploaded source, plus what describes them.

    This row is written before conversion starts, so it exists for an upload that
    is still ``processing`` and for one whose indexing later failed. ``preview_key``
    is filled in afterwards, once normalization has produced text worth showing.
    """

    document_id: UUID
    collection_id: str
    external_id: str
    filename: str
    media_type: str
    byte_size: int
    checksum: str
    storage_backend: str
    storage_key: str
    preview_key: str | None = None
    preview_bytes: int | None = None
    # Hash of the preview's own bytes. It is separate from ``checksum`` because
    # reprocessing one original can produce different extracted text — a converter
    # upgrade, for instance — and a caller revalidating the preview has to be told
    # that it changed even though the original did not.
    preview_checksum: str | None = None
    page_count: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One selectable document source, as a UI needs to render it."""

    external_id: str
    title: str
    source_type: str
    status: SourceStatus
    chunk_count: int
    detail: str | None = None
    filename: str | None = None
    media_type: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    # Viewer state. ``viewable`` is false for every source ingested before
    # originals were retained, and for the manual, which has no uploaded file.
    viewable: bool = False
    byte_size: int | None = None
    page_count: int | None = None
    preview_available: bool = False


@dataclass(frozen=True, slots=True)
class EmbeddingState:
    model_name: str
    model_dim: int


def derive_source_status(
    document: DocumentRecord | None,
    job: JobRecord | None,
) -> tuple[SourceStatus, str | None]:
    """Collapse document and latest-job state into one status plus a detail.

    A document keeps its ``ready`` status while a re-index runs, because its
    existing chunks stay searchable until the new ones replace them. Re-indexing
    resets ``chunk_count`` to zero first, so an in-flight ingest reads as
    ``processing`` rather than as a ready but empty source.
    """
    if document is not None and document.chunk_count > 0:
        return "ready", None
    if job is not None and job.status in ("accepted", "processing"):
        return "processing", None
    if job is not None and job.status == "failed":
        return "failed", job.detail
    if job is None and document is None:  # pragma: no cover - defensive
        return "failed", None
    return "failed", "Indexing produced no searchable content."


def build_source_record(
    document: DocumentRecord | None,
    job: JobRecord | None,
    source_object: SourceObjectRecord | None = None,
) -> SourceRecord:
    """Merge a document row, its latest job, and any stored original into one record."""
    if document is None and job is None:  # pragma: no cover - defensive
        raise ValueError("A source needs either a document or a job.")
    status, detail = derive_source_status(document, job)
    external_id = document.external_id if document is not None else job.external_id  # type: ignore[union-attr]
    created_at = _earliest(
        document.created_at if document else None, job.created_at if job else None
    )
    updated_at = _latest(
        document.updated_at if document else None, job.updated_at if job else None
    )
    return SourceRecord(
        external_id=external_id,
        title=(document.title if document else job.title) or external_id,  # type: ignore[union-attr]
        source_type=document.source_type if document else job.source_type,  # type: ignore[union-attr]
        status=status,
        chunk_count=document.chunk_count if document else 0,
        detail=detail,
        filename=(
            _optional_text(document.metadata.get("filename"))
            if document is not None
            else job.filename  # type: ignore[union-attr]
        )
        or (source_object.filename if source_object is not None else None),
        media_type=(
            _optional_text(document.metadata.get("media_type"))
            if document is not None
            else job.media_type  # type: ignore[union-attr]
        )
        or (source_object.media_type if source_object is not None else None),
        created_at=created_at,
        updated_at=updated_at,
        viewable=source_object is not None,
        byte_size=source_object.byte_size if source_object is not None else None,
        page_count=source_object.page_count if source_object is not None else None,
        preview_available=bool(source_object is not None and source_object.preview_key),
    )


class Repository(ABC):
    """Storage contract used by the ingestion and search paths."""

    @abstractmethod
    async def ensure_collection(self, collection_id: str) -> None: ...

    @abstractmethod
    async def get_document(self, collection_id: str, external_id: str) -> DocumentRecord | None: ...

    @abstractmethod
    async def upsert_document(self, record: DocumentRecord) -> None: ...

    @abstractmethod
    async def list_sources(self, collection_id: str, *, limit: int) -> list[SourceRecord]: ...

    @abstractmethod
    async def upsert_source_object(self, record: SourceObjectRecord) -> None: ...

    @abstractmethod
    async def set_source_preview(
        self,
        document_id: UUID,
        *,
        preview_key: str | None,
        preview_bytes: int | None,
        preview_checksum: str | None,
        page_count: int | None,
    ) -> None: ...

    @abstractmethod
    async def get_source_object(
        self, collection_id: str, external_id: str
    ) -> SourceObjectRecord | None: ...

    @abstractmethod
    async def create_job(self, record: JobRecord) -> None: ...

    @abstractmethod
    async def update_job(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        document_id: UUID | None = None,
        chunk_count: int | None = None,
        unchanged: bool | None = None,
        detail: str | None = None,
    ) -> None: ...

    @abstractmethod
    async def set_job_conversion_task(
        self,
        job_id: UUID,
        *,
        converter_name: str,
        task_id: str,
        submitted_at: datetime,
    ) -> None: ...

    @abstractmethod
    async def get_job(self, job_id: UUID) -> JobRecord | None: ...

    @abstractmethod
    async def get_latest_job(self, collection_id: str, external_id: str) -> JobRecord | None: ...

    @abstractmethod
    async def list_resumable_jobs(self) -> list[JobRecord]: ...

    @abstractmethod
    async def fail_interrupted_jobs(self, detail: str) -> int: ...

    @abstractmethod
    async def get_embedding_state(self) -> EmbeddingState | None: ...

    @abstractmethod
    async def set_embedding_state(self, state: EmbeddingState) -> None: ...

    @abstractmethod
    async def probe(self) -> None: ...


def schema_ddl(schema: str) -> tuple[str, ...]:
    """Return idempotent DDL statements for *schema*.

    ``schema`` is validated as a SQL identifier in :mod:`app.config` before it
    reaches this module; it is interpolated because PostgreSQL does not accept
    identifiers as bind parameters.
    """
    return (
        f"CREATE SCHEMA IF NOT EXISTS {schema}",
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.collections (
            collection_id text PRIMARY KEY,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.documents (
            document_id uuid PRIMARY KEY,
            collection_id text NOT NULL REFERENCES {schema}.collections (collection_id),
            external_id text NOT NULL,
            title text NOT NULL DEFAULT '',
            source_type text NOT NULL DEFAULT 'text',
            source_uri text NOT NULL DEFAULT '',
            checksum text NOT NULL,
            version text NOT NULL DEFAULT '',
            page integer,
            section text NOT NULL DEFAULT '',
            metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            chunk_count integer NOT NULL DEFAULT 0,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (collection_id, external_id)
        )
        """,
        f"CREATE INDEX IF NOT EXISTS documents_collection_idx "
        f"ON {schema}.documents (collection_id)",
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.ingest_jobs (
            job_id uuid PRIMARY KEY,
            collection_id text NOT NULL,
            external_id text NOT NULL,
            title text NOT NULL DEFAULT '',
            source_type text NOT NULL DEFAULT 'text',
            filename text,
            media_type text,
            document_id uuid,
            status text NOT NULL
                CHECK (status IN ('accepted', 'processing', 'completed', 'failed')),
            chunk_count integer NOT NULL DEFAULT 0,
            unchanged boolean NOT NULL DEFAULT false,
            detail text,
            converter_name text,
            converter_task_id text,
            converter_submitted_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
        f"ALTER TABLE {schema}.ingest_jobs ADD COLUMN IF NOT EXISTS title text NOT NULL DEFAULT ''",
        f"ALTER TABLE {schema}.ingest_jobs ADD COLUMN IF NOT EXISTS source_type text NOT NULL DEFAULT 'text'",
        f"ALTER TABLE {schema}.ingest_jobs ADD COLUMN IF NOT EXISTS filename text",
        f"ALTER TABLE {schema}.ingest_jobs ADD COLUMN IF NOT EXISTS media_type text",
        # Nullable and unconstrained, so an existing deployment upgrades in place:
        # every job written before asynchronous conversion simply has no converter
        # task, which is exactly how a non-resumable job reads.
        f"ALTER TABLE {schema}.ingest_jobs ADD COLUMN IF NOT EXISTS converter_name text",
        f"ALTER TABLE {schema}.ingest_jobs ADD COLUMN IF NOT EXISTS converter_task_id text",
        f"ALTER TABLE {schema}.ingest_jobs "
        f"ADD COLUMN IF NOT EXISTS converter_submitted_at timestamptz",
        f"CREATE INDEX IF NOT EXISTS ingest_jobs_status_idx "
        f"ON {schema}.ingest_jobs (status)",
        # Startup reads exactly this set to decide what to resume, so it is worth
        # a partial index rather than a scan of every job ever recorded.
        f"CREATE INDEX IF NOT EXISTS ingest_jobs_resumable_idx "
        f"ON {schema}.ingest_jobs (status) WHERE converter_task_id IS NOT NULL",
        f"CREATE INDEX IF NOT EXISTS ingest_jobs_identity_idx "
        f"ON {schema}.ingest_jobs (collection_id, external_id, created_at DESC)",
        # Written before conversion starts, so there is no foreign key to
        # documents: the original is durable even for an upload whose indexing is
        # still running or has failed.
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.source_objects (
            document_id uuid PRIMARY KEY,
            collection_id text NOT NULL,
            external_id text NOT NULL,
            filename text NOT NULL,
            media_type text NOT NULL,
            byte_size bigint NOT NULL,
            checksum text NOT NULL,
            storage_backend text NOT NULL,
            storage_key text NOT NULL,
            preview_key text,
            preview_bytes bigint,
            preview_checksum text,
            page_count integer,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            UNIQUE (collection_id, external_id)
        )
        """,
        f"ALTER TABLE {schema}.source_objects "
        f"ADD COLUMN IF NOT EXISTS preview_checksum text",
        f"CREATE INDEX IF NOT EXISTS source_objects_collection_idx "
        f"ON {schema}.source_objects (collection_id)",
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.embedding_state (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            model_name text NOT NULL,
            model_dim integer NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """,
    )


class PostgresRepository(Repository):
    """asyncpg-backed implementation."""

    def __init__(self, pool: asyncpg.Pool, schema: str) -> None:
        self._pool = pool
        self._schema = schema

    async def ensure_schema(self) -> None:
        async with self._pool.acquire() as connection:
            for statement in schema_ddl(self._schema):
                await connection.execute(statement)

    async def probe(self) -> None:
        async with self._pool.acquire() as connection:
            await connection.execute("SELECT 1")

    async def ensure_collection(self, collection_id: str) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.collections (collection_id)
            VALUES ($1)
            ON CONFLICT (collection_id) DO UPDATE SET updated_at = now()
            """,
            collection_id,
        )

    async def get_document(self, collection_id: str, external_id: str) -> DocumentRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT document_id, collection_id, external_id, title, source_type, source_uri,
                   checksum, version, page, section, metadata, chunk_count, created_at, updated_at
            FROM {self._schema}.documents
            WHERE collection_id = $1 AND external_id = $2
            """,
            collection_id,
            external_id,
        )
        return _document_from_row(row) if row else None

    async def upsert_document(self, record: DocumentRecord) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.documents (
                document_id, collection_id, external_id, title, source_type, source_uri,
                checksum, version, page, section, metadata, chunk_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12)
            ON CONFLICT (document_id) DO UPDATE SET
                title = EXCLUDED.title,
                source_type = EXCLUDED.source_type,
                source_uri = EXCLUDED.source_uri,
                checksum = EXCLUDED.checksum,
                version = EXCLUDED.version,
                page = EXCLUDED.page,
                section = EXCLUDED.section,
                metadata = EXCLUDED.metadata,
                chunk_count = EXCLUDED.chunk_count,
                updated_at = now()
            """,
            record.document_id,
            record.collection_id,
            record.external_id,
            record.title,
            record.source_type,
            record.source_uri,
            record.checksum,
            record.version,
            record.page,
            record.section,
            json.dumps(record.metadata),
            record.chunk_count,
        )

    async def list_sources(self, collection_id: str, *, limit: int) -> list[SourceRecord]:
        """List every document and in-flight ingest in *collection_id*.

        Jobs are included on their own so an upload is visible as ``processing``
        before its document row exists. Both halves are bounded by *limit*; a
        collection larger than that is truncated to the most recent entries.
        """
        documents = await self._pool.fetch(
            f"""
            SELECT document_id, collection_id, external_id, title, source_type, source_uri,
                   checksum, version, page, section, metadata, chunk_count, created_at, updated_at
            FROM {self._schema}.documents
            WHERE collection_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            collection_id,
            limit,
        )
        jobs = await self._pool.fetch(
            f"""
            SELECT job_id, collection_id, external_id, title, source_type, filename,
                   media_type, document_id, status, chunk_count, unchanged, detail,
                   converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM (
                SELECT DISTINCT ON (external_id)
                       job_id, collection_id, external_id, title, source_type, filename,
                       media_type, document_id, status, chunk_count, unchanged, detail,
                       converter_name, converter_task_id, converter_submitted_at,
                       created_at, updated_at
                FROM {self._schema}.ingest_jobs
                WHERE collection_id = $1
                ORDER BY external_id, created_at DESC, job_id DESC
            ) AS latest_jobs
            ORDER BY updated_at DESC, job_id DESC
            LIMIT $2
            """,
            collection_id,
            limit,
        )
        source_objects = await self._pool.fetch(
            f"""
            SELECT document_id, collection_id, external_id, filename, media_type, byte_size,
                   checksum, storage_backend, storage_key, preview_key, preview_bytes,
                   preview_checksum, page_count, created_at, updated_at
            FROM {self._schema}.source_objects
            WHERE collection_id = $1
            ORDER BY updated_at DESC
            LIMIT $2
            """,
            collection_id,
            limit,
        )
        return merge_sources(
            [_document_from_row(row) for row in documents],
            [_job_from_row(row) for row in jobs],
            [_source_object_from_row(row) for row in source_objects],
            limit=limit,
        )

    async def upsert_source_object(self, record: SourceObjectRecord) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.source_objects (
                document_id, collection_id, external_id, filename, media_type, byte_size,
                checksum, storage_backend, storage_key, preview_key, preview_bytes,
                preview_checksum, page_count
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            ON CONFLICT (document_id) DO UPDATE SET
                filename = EXCLUDED.filename,
                media_type = EXCLUDED.media_type,
                byte_size = EXCLUDED.byte_size,
                checksum = EXCLUDED.checksum,
                storage_backend = EXCLUDED.storage_backend,
                storage_key = EXCLUDED.storage_key,
                updated_at = now()
            """,
            record.document_id,
            record.collection_id,
            record.external_id,
            record.filename,
            record.media_type,
            record.byte_size,
            record.checksum,
            record.storage_backend,
            record.storage_key,
            record.preview_key,
            record.preview_bytes,
            record.preview_checksum,
            record.page_count,
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
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.source_objects
            SET preview_key = $2,
                preview_bytes = $3,
                preview_checksum = $4,
                page_count = $5,
                updated_at = now()
            WHERE document_id = $1
            """,
            document_id,
            preview_key,
            preview_bytes,
            preview_checksum,
            page_count,
        )

    async def get_source_object(
        self, collection_id: str, external_id: str
    ) -> SourceObjectRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT document_id, collection_id, external_id, filename, media_type, byte_size,
                   checksum, storage_backend, storage_key, preview_key, preview_bytes,
                   preview_checksum, page_count, created_at, updated_at
            FROM {self._schema}.source_objects
            WHERE collection_id = $1 AND external_id = $2
            """,
            collection_id,
            external_id,
        )
        return _source_object_from_row(row) if row else None

    async def create_job(self, record: JobRecord) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.ingest_jobs (
                job_id, collection_id, external_id, title, source_type, filename,
                media_type, document_id, status, chunk_count, unchanged, detail,
                converter_name, converter_task_id, converter_submitted_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
            """,
            record.job_id,
            record.collection_id,
            record.external_id,
            record.title,
            record.source_type,
            record.filename,
            record.media_type,
            record.document_id,
            record.status,
            record.chunk_count,
            record.unchanged,
            record.detail,
            record.converter_name,
            record.converter_task_id,
            record.converter_submitted_at,
        )

    async def update_job(
        self,
        job_id: UUID,
        *,
        status: JobStatus,
        document_id: UUID | None = None,
        chunk_count: int | None = None,
        unchanged: bool | None = None,
        detail: str | None = None,
    ) -> None:
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.ingest_jobs
            SET status = $2,
                document_id = COALESCE($3, document_id),
                chunk_count = COALESCE($4, chunk_count),
                unchanged = COALESCE($5, unchanged),
                detail = $6,
                updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            status,
            document_id,
            chunk_count,
            unchanged,
            detail,
        )

    async def set_job_conversion_task(
        self,
        job_id: UUID,
        *,
        converter_name: str,
        task_id: str,
        submitted_at: datetime,
    ) -> None:
        """Record the converter task this job is waiting on.

        Written before the first poll, so a crash between submission and the
        first status read still leaves the remote task findable.
        """
        await self._pool.execute(
            f"""
            UPDATE {self._schema}.ingest_jobs
            SET converter_name = $2,
                converter_task_id = $3,
                converter_submitted_at = $4,
                updated_at = now()
            WHERE job_id = $1
            """,
            job_id,
            converter_name,
            task_id,
            submitted_at,
        )

    async def get_job(self, job_id: UUID) -> JobRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT job_id, collection_id, external_id, document_id, status,
                   title, source_type, filename, media_type, chunk_count, unchanged,
                   detail, converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM {self._schema}.ingest_jobs
            WHERE job_id = $1
            """,
            job_id,
        )
        return _job_from_row(row) if row else None

    async def get_latest_job(self, collection_id: str, external_id: str) -> JobRecord | None:
        row = await self._pool.fetchrow(
            f"""
            SELECT job_id, collection_id, external_id, document_id, status,
                   title, source_type, filename, media_type, chunk_count, unchanged,
                   detail, converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM {self._schema}.ingest_jobs
            WHERE collection_id = $1 AND external_id = $2
            ORDER BY created_at DESC, job_id DESC
            LIMIT 1
            """,
            collection_id,
            external_id,
        )
        return _job_from_row(row) if row else None

    async def list_resumable_jobs(self) -> list[JobRecord]:
        rows = await self._pool.fetch(
            f"""
            SELECT job_id, collection_id, external_id, document_id, status,
                   title, source_type, filename, media_type, chunk_count, unchanged,
                   detail, converter_name, converter_task_id, converter_submitted_at,
                   created_at, updated_at
            FROM {self._schema}.ingest_jobs
            WHERE status IN ('accepted', 'processing')
              AND converter_task_id IS NOT NULL
              AND converter_submitted_at IS NOT NULL
            ORDER BY converter_submitted_at
            """
        )
        return [_job_from_row(row) for row in rows]

    async def fail_interrupted_jobs(self, detail: str) -> int:
        """Fail unfinished jobs that hold no resumable converter task.

        A job whose converter task is recorded is deliberately left alone: the
        remote conversion is still running, and failing it here would throw away
        work Athena has already paid for.
        """
        rows = await self._pool.fetch(
            f"""
            UPDATE {self._schema}.ingest_jobs
            SET status = 'failed', detail = $1, updated_at = now()
            WHERE status IN ('accepted', 'processing')
              AND (converter_task_id IS NULL OR converter_submitted_at IS NULL)
            RETURNING job_id
            """,
            detail,
        )
        return len(rows)

    async def get_embedding_state(self) -> EmbeddingState | None:
        row = await self._pool.fetchrow(
            f"SELECT model_name, model_dim FROM {self._schema}.embedding_state WHERE singleton"
        )
        if row is None:
            return None
        return EmbeddingState(model_name=row["model_name"], model_dim=row["model_dim"])

    async def set_embedding_state(self, state: EmbeddingState) -> None:
        await self._pool.execute(
            f"""
            INSERT INTO {self._schema}.embedding_state (singleton, model_name, model_dim)
            VALUES (true, $1, $2)
            ON CONFLICT (singleton) DO UPDATE SET
                model_name = EXCLUDED.model_name,
                model_dim = EXCLUDED.model_dim,
                updated_at = now()
            """,
            state.model_name,
            state.model_dim,
        )


def merge_sources(
    documents: Iterable[DocumentRecord],
    jobs: Iterable[JobRecord],
    source_objects: Iterable[SourceObjectRecord] = (),
    *,
    limit: int,
) -> list[SourceRecord]:
    """Pair documents with their latest job and stored original, newest first."""
    latest_jobs: dict[str, JobRecord] = {}
    for job in jobs:
        current = latest_jobs.get(job.external_id)
        if current is None or _sorts_after(job.created_at, current.created_at):
            latest_jobs[job.external_id] = job
    originals = {record.external_id: record for record in source_objects}

    records: list[SourceRecord] = []
    seen: set[str] = set()
    for document in documents:
        seen.add(document.external_id)
        records.append(
            build_source_record(
                document,
                latest_jobs.get(document.external_id),
                originals.get(document.external_id),
            )
        )
    for external_id, job in latest_jobs.items():
        if external_id not in seen:
            records.append(build_source_record(None, job, originals.get(external_id)))

    records.sort(key=lambda record: (record.updated_at or _EPOCH), reverse=True)
    return records[:limit]


_EPOCH = datetime.fromtimestamp(0, tz=UTC)


def _sorts_after(candidate: datetime | None, current: datetime | None) -> bool:
    return (candidate or _EPOCH) >= (current or _EPOCH)


def _earliest(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None or right is None:
        return left or right
    return min(left, right)


def _latest(left: datetime | None, right: datetime | None) -> datetime | None:
    if left is None or right is None:
        return left or right
    return max(left, right)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _document_from_row(row: Any) -> DocumentRecord:
    metadata = row["metadata"]
    return DocumentRecord(
        document_id=row["document_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        title=row["title"],
        source_type=row["source_type"],
        source_uri=row["source_uri"],
        checksum=row["checksum"],
        version=row["version"],
        page=row["page"],
        section=row["section"],
        metadata=json.loads(metadata) if isinstance(metadata, str) else dict(metadata or {}),
        chunk_count=row["chunk_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _source_object_from_row(row: Any) -> SourceObjectRecord:
    return SourceObjectRecord(
        document_id=row["document_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        filename=row["filename"],
        media_type=row["media_type"],
        byte_size=row["byte_size"],
        checksum=row["checksum"],
        storage_backend=row["storage_backend"],
        storage_key=row["storage_key"],
        preview_key=row["preview_key"],
        preview_bytes=row["preview_bytes"],
        preview_checksum=row["preview_checksum"],
        page_count=row["page_count"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _job_from_row(row: Any) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        collection_id=row["collection_id"],
        external_id=row["external_id"],
        status=row["status"],
        title=row["title"],
        source_type=row["source_type"],
        filename=row["filename"],
        media_type=row["media_type"],
        document_id=row["document_id"],
        chunk_count=row["chunk_count"],
        unchanged=row["unchanged"],
        detail=row["detail"],
        converter_name=row["converter_name"],
        converter_task_id=row["converter_task_id"],
        converter_submitted_at=row["converter_submitted_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
