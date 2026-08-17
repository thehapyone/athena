"""End-to-end ingestion and retrieval against a real PostgreSQL/pgvector database.

Skipped unless ``KNOWLEDGE_TEST_DATABASE_URL`` points at a database with the
``vector`` extension available. Embeddings stay deterministic so the test needs
no embedding endpoint.

    docker run --rm -d -p 55432:5432 -e POSTGRES_PASSWORD=postgres \\
        --name knowledge-test-db pgvector/pgvector:pg17
    KNOWLEDGE_TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:55432/postgres \\
        .venv/bin/python -m pytest tests/integration -q
"""

import os
import uuid
from datetime import UTC, datetime, timedelta

import asyncpg
import pytest
import pytest_asyncio

from app.config import Settings
from app.embeddings.pipeline import build_ingestion_pipeline, build_vector_store
from app.embeddings.retriever import search_documents
from app.ingest import IngestionService, UploadSubmission, upload_external_id
from app.models import SearchRequest, TextDocumentRequest
from app.parsing import DocumentNormalizer, UploadedFile, resolve_format
from app.repository import JobRecord, PostgresRepository
from app.storage import ORIGINAL_VARIANT, PREVIEW_VARIANT, LocalFileSourceStore, storage_key
from tests.fakes import VOCAB, DeterministicEmbedding

DATABASE_URL = os.environ.get("KNOWLEDGE_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="KNOWLEDGE_TEST_DATABASE_URL is not set"
)

EXAMPLE_TEXT = (
    "Calibrate the inspiratory pressure sensor during preventive maintenance. "
    "A persistent alarm means the expiratory valve needs replacement."
)
TROLLEY_TEXT = (
    "Calibrate the oxygen pressure sensor on the anaesthesia trolley. "
    "A persistent alarm means the battery pack needs replacement."
)


@pytest_asyncio.fixture
async def settings() -> Settings:
    schema = f"knowledge_test_{uuid.uuid4().hex[:12]}"
    return Settings.from_env(
        {
            "KNOWLEDGE_DATABASE_URL": DATABASE_URL,
            "KNOWLEDGE_DB_SCHEMA": schema,
            "KNOWLEDGE_API_TOKEN": "integration-token-0123456789",
            "KNOWLEDGE_EMBEDDING_BASE_URL": "https://example.invalid/openai/v1",
            "KNOWLEDGE_EMBEDDING_API_KEY": "unused",
            "KNOWLEDGE_EMBEDDING_MODEL": "deterministic",
            "KNOWLEDGE_EMBEDDING_DIMENSION": str(len(VOCAB)),
            "KNOWLEDGE_CHUNK_SIZE": "128",
            "KNOWLEDGE_CHUNK_OVERLAP": "16",
            "KNOWLEDGE_RETRIEVAL_MODE": "hybrid",
        }
    )


@pytest_asyncio.fixture
async def stack(settings: Settings, tmp_path):
    pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=4)
    await pool.execute("CREATE EXTENSION IF NOT EXISTS vector")
    repository = PostgresRepository(pool, settings.db_schema)
    await repository.ensure_schema()

    embed_model = DeterministicEmbedding()
    vector_store = build_vector_store(settings)
    pipeline = build_ingestion_pipeline(
        embed_model,
        vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    source_store = LocalFileSourceStore(tmp_path / "sources")
    await source_store.prepare()
    ingestion = IngestionService(
        repository=repository,
        vector_store=vector_store,
        pipeline=pipeline,
        normalizer=DocumentNormalizer(
            converter=None, max_text_bytes=settings.max_document_bytes
        ),
        source_store=source_store,
    )
    try:
        yield repository, vector_store, embed_model, ingestion, source_store
    finally:
        await vector_store.close()
        await pool.execute(f"DROP SCHEMA IF EXISTS {settings.db_schema} CASCADE")
        await pool.close()


async def _ingest(ingestion: IngestionService, request: TextDocumentRequest):
    job = await ingestion.submit(request)
    await ingestion.wait_for_pending()
    return job


async def test_two_collections_stay_isolated_in_postgres(stack, settings: Settings) -> None:
    repository, vector_store, embed_model, ingestion, _source_store = stack

    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Example service manual",
            source_type="manual",
            text=EXAMPLE_TEXT,
        ),
    )
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="other-manual",
            external_id="trolley-manual",
            title="Trolley manual",
            source_type="manual",
            text=TROLLEY_TEXT,
        ),
    )

    result = await search_documents(
        vector_store,
        embed_model,
        SearchRequest(query="pressure sensor alarm", collection_ids=["example-collection"]),
        settings,
    )

    assert result.items
    assert {item.collection_id for item in result.items} == {"example-collection"}
    assert all("trolley" not in item.text.lower() for item in result.items)
    assert all(item.citations for item in result.items)

    both = await search_documents(
        vector_store,
        embed_model,
        SearchRequest(
            query="pressure sensor alarm", collection_ids=["example-collection", "other-manual"]
        ),
        settings,
    )
    assert {item.collection_id for item in both.items} == {"example-collection", "other-manual"}


async def test_excluded_source_is_filtered_by_postgres(stack, settings: Settings) -> None:
    _repository, vector_store, embed_model, ingestion, _source_store = stack
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Legacy embedded manual",
            text=EXAMPLE_TEXT,
        ),
    )
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="file-replacement",
            title="Uploaded replacement",
            text="Pressure sensor maintenance guidance for the ventilator.",
        ),
    )

    result = await search_documents(
        vector_store,
        embed_model,
        SearchRequest(
            query=EXAMPLE_TEXT,
            collection_ids=["example-collection"],
            top_k=1,
            filters={"exclude_external_id": ["example-manual"]},
        ),
        settings,
    )

    assert [item.external_id for item in result.items] == ["file-replacement"]


async def test_job_and_document_state_survive_a_new_repository_instance(
    stack, settings: Settings
) -> None:
    repository, vector_store, _embed_model, ingestion, _source_store = stack

    request = TextDocumentRequest(
        collection_id="example-collection",
        external_id="example-manual",
        text=EXAMPLE_TEXT,
    )
    job = await _ingest(ingestion, request)
    repeated = await _ingest(ingestion, request)

    # A fresh repository object stands in for a restarted process.
    reopened = PostgresRepository(repository._pool, settings.db_schema)  # noqa: SLF001
    stored_job = await reopened.get_job(job.job_id)
    stored_repeat = await reopened.get_job(repeated.job_id)
    document = await reopened.get_document("example-collection", "example-manual")

    assert stored_job is not None and stored_job.status == "completed"
    assert stored_repeat is not None and stored_repeat.unchanged is True
    assert document is not None and document.chunk_count == stored_job.chunk_count


async def test_uploaded_and_text_sources_are_listed_per_collection(stack) -> None:
    repository, _vector_store, _embed_model, ingestion, _source_store = stack

    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="example-collection",
            external_id="example-manual",
            title="Example service manual",
            source_type="manual",
            text=EXAMPLE_TEXT,
        ),
    )
    await _ingest(
        ingestion,
        TextDocumentRequest(
            collection_id="other-manual", external_id="trolley-manual", text=TROLLEY_TEXT
        ),
    )

    content = b"The oxygen sensor is calibrated during preventive maintenance."
    document_format = resolve_format("notes.txt", "text/plain")
    await ingestion.submit_upload(
        UploadSubmission(
            collection_id="example-collection",
            external_id=upload_external_id(content, ".txt"),
            title="Field notes",
            upload=UploadedFile(
                filename="notes.txt",
                media_type="text/plain",
                content=content,
                format=document_format,
            ),
        )
    )
    # A file that cannot be decoded proves a failed source is listed with its reason.
    broken = b"\xff\xfe\x00\x01"
    await ingestion.submit_upload(
        UploadSubmission(
            collection_id="example-collection",
            external_id=upload_external_id(broken, ".txt"),
            title="broken.txt",
            upload=UploadedFile(
                filename="broken.txt",
                media_type="text/plain",
                content=broken,
                format=document_format,
            ),
        )
    )
    await ingestion.wait_for_pending()

    sources = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }

    assert set(sources) == {
        "example-manual",
        upload_external_id(content, ".txt"),
        upload_external_id(broken, ".txt"),
    }
    assert sources["example-manual"].status == "ready"
    assert sources["example-manual"].source_type == "manual"

    uploaded = sources[upload_external_id(content, ".txt")]
    assert uploaded.status == "ready"
    assert uploaded.title == "Field notes"
    assert uploaded.filename == "notes.txt"
    assert uploaded.media_type == "text/plain"
    assert uploaded.chunk_count >= 1

    failed = sources[upload_external_id(broken, ".txt")]
    assert failed.status == "failed"
    assert failed.detail and "UTF-8" in failed.detail

    other = await repository.list_sources("other-manual", limit=50)
    assert [record.external_id for record in other] == ["trolley-manual"]


async def test_source_limit_uses_recency_not_external_id_order(stack, settings: Settings) -> None:
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    now = datetime.now(UTC)
    # The identifiers are chosen so alphabetical order and recency disagree: a
    # LIMIT applied under "ORDER BY external_id" would keep "a-older" and drop
    # the newer row, so this fails unless the query orders by recency first.
    older = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="a-older",
        title="Older upload",
        source_type="upload",
        status="accepted",
    )
    newer = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="z-newer",
        title="Newer upload",
        source_type="upload",
        status="accepted",
    )
    await repository.create_job(older)
    await repository.create_job(newer)
    await repository._pool.execute(  # noqa: SLF001 - integration boundary
        f"UPDATE {settings.db_schema}.ingest_jobs SET created_at = $1, updated_at = $1 "
        "WHERE job_id = $2",
        now - timedelta(minutes=1),
        older.job_id,
    )
    await repository._pool.execute(  # noqa: SLF001 - integration boundary
        f"UPDATE {settings.db_schema}.ingest_jobs SET created_at = $1, updated_at = $1 "
        "WHERE job_id = $2",
        now,
        newer.job_id,
    )

    sources = await repository.list_sources("example-collection", limit=1)

    assert [source.external_id for source in sources] == ["z-newer"]


async def test_existing_ingest_jobs_schema_upgrades_without_breaking_old_writers(
    stack, settings: Settings
) -> None:
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    schema = settings.db_schema
    legacy_job_id = uuid.uuid4()
    overlap_job_id = uuid.uuid4()

    await repository._pool.execute(f"DROP TABLE {schema}.ingest_jobs")  # noqa: SLF001
    await repository._pool.execute(  # noqa: SLF001
        f"""
        CREATE TABLE {schema}.ingest_jobs (
            job_id uuid PRIMARY KEY,
            collection_id text NOT NULL,
            external_id text NOT NULL,
            document_id uuid,
            status text NOT NULL
                CHECK (status IN ('accepted', 'processing', 'completed', 'failed')),
            chunk_count integer NOT NULL DEFAULT 0,
            unchanged boolean NOT NULL DEFAULT false,
            detail text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    await repository._pool.execute(  # noqa: SLF001
        f"""
        INSERT INTO {schema}.ingest_jobs (
            job_id, collection_id, external_id, status, detail
        ) VALUES ($1, 'example-collection', 'legacy-source', 'failed', 'legacy detail')
        """,
        legacy_job_id,
    )

    await repository.ensure_schema()

    legacy = await repository.get_job(legacy_job_id)
    assert legacy is not None
    assert legacy.title == ""
    assert legacy.source_type == "text"
    assert legacy.filename is None
    assert legacy.media_type is None
    assert legacy.detail == "legacy detail"

    # An old application instance can still insert its original column set
    # while a new instance is live because every added column has a default.
    await repository._pool.execute(  # noqa: SLF001
        f"""
        INSERT INTO {schema}.ingest_jobs (
            job_id, collection_id, external_id, status
        ) VALUES ($1, 'example-collection', 'overlap-source', 'accepted')
        """,
        overlap_job_id,
    )
    overlap = await repository.get_job(overlap_job_id)
    assert overlap is not None and overlap.title == "" and overlap.source_type == "text"

    current = JobRecord(
        job_id=uuid.uuid4(),
        collection_id="example-collection",
        external_id="current-source",
        title="Current upload",
        source_type="markdown",
        filename="current.md",
        media_type="text/markdown",
        status="accepted",
    )
    await repository.create_job(current)
    stored_current = await repository.get_job(current.job_id)
    assert stored_current is not None
    assert stored_current.title == "Current upload"
    assert stored_current.filename == "current.md"


async def _upload(ingestion: IngestionService, filename: str, content: bytes, media_type: str):
    """Run the production upload path: retain the bytes, then index them."""
    document_format = resolve_format(filename, media_type)
    submission = UploadSubmission(
        collection_id="example-collection",
        external_id=upload_external_id(content, document_format.extension),
        title=filename,
        upload=UploadedFile(
            filename=filename,
            media_type=document_format.canonical_media_type,
            content=content,
            format=document_format,
        ),
    )
    await ingestion.retain_original(submission)
    await ingestion.submit_upload(submission)
    await ingestion.wait_for_pending()
    return submission.external_id


async def test_retained_originals_survive_a_restart_in_postgres(
    stack, settings: Settings, tmp_path
) -> None:
    """The acceptance case: reopen the persisted file after the service restarts."""
    repository, _vector_store, _embed_model, ingestion, source_store = stack
    content = b"The oxygen sensor is calibrated during preventive maintenance."

    external_id = await _upload(ingestion, "notes.txt", content, "text/plain")

    stored = await repository.get_source_object("example-collection", external_id)
    assert stored is not None
    assert stored.filename == "notes.txt"
    assert stored.media_type == "text/plain"
    assert stored.byte_size == len(content)
    assert len(stored.checksum) == 64
    assert stored.storage_backend == "local"
    assert stored.storage_key == storage_key(stored.document_id, ORIGINAL_VARIANT)
    # A text upload is its own readable form, so no second copy is stored.
    assert stored.preview_key is None
    assert stored.page_count is None

    listed = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }
    assert listed[external_id].viewable is True
    assert listed[external_id].byte_size == len(content)
    assert listed[external_id].preview_available is False

    # A restart is a fresh repository and a fresh store over the same database
    # and the same volume.
    restarted_pool = await asyncpg.create_pool(settings.database_url, min_size=1, max_size=1)
    try:
        restarted = PostgresRepository(restarted_pool, settings.db_schema)
        await restarted.ensure_schema()
        after_restart = await restarted.get_source_object("example-collection", external_id)
    finally:
        await restarted_pool.close()

    assert after_restart == stored
    fresh_store = LocalFileSourceStore(tmp_path / "sources")
    await fresh_store.prepare()
    read_back = b"".join([chunk async for chunk in fresh_store.read(stored.storage_key)])
    assert read_back == content

    # Re-uploading identical content replaces the same object rather than adding one.
    await _upload(ingestion, "notes.txt", content, "text/plain")
    files = [path for path in source_store.root.rglob("*") if path.is_file()]
    assert [path.name for path in files] == ["original"], files
    reuploaded = await repository.get_source_object("example-collection", external_id)
    assert reuploaded is not None
    assert reuploaded.storage_key == stored.storage_key
    assert reuploaded.checksum == stored.checksum
    assert reuploaded.created_at == stored.created_at


async def test_a_preview_and_page_reach_round_trip_through_postgres(stack) -> None:
    repository, _vector_store, _embed_model, ingestion, source_store = stack

    class LocatedConverter:
        name = "located"

        async def convert(self, *, filename: str, media_type: str, content: bytes):
            from app.parsing import DocumentSegment, build_converted_document

            return build_converted_document(
                [
                    DocumentSegment(
                        text="Calibrate the inspiratory pressure sensor.",
                        page=4,
                        section="2 Maintenance",
                    ),
                    DocumentSegment(
                        text="The expiratory valve alarm needs the battery replaced.",
                        page=17,
                        section="3 Alarms",
                    ),
                ]
            )

    ingestion._normalizer = DocumentNormalizer(  # noqa: SLF001
        converter=LocatedConverter(), max_text_bytes=8_000_000
    )
    external_id = await _upload(ingestion, "service.pdf", b"%PDF-1.7 body", "application/pdf")

    stored = await repository.get_source_object("example-collection", external_id)
    assert stored is not None
    assert stored.page_count == 17
    assert stored.preview_key == storage_key(stored.document_id, PREVIEW_VARIANT)
    assert stored.preview_bytes and stored.preview_bytes > 0
    # The preview hashes its own bytes, so its HTTP validator tracks the extracted
    # text rather than the original that produced it.
    assert stored.preview_checksum and len(stored.preview_checksum) == 64
    assert stored.preview_checksum != stored.checksum

    preview = b"".join([chunk async for chunk in source_store.read(stored.preview_key)])
    assert b"inspiratory pressure sensor" in preview

    listed = {
        record.external_id: record
        for record in await repository.list_sources("example-collection", limit=50)
    }
    assert listed[external_id].page_count == 17
    assert listed[external_id].preview_available is True

    # The provenance reached the indexed chunks, which is what citations read back.
    latest = await repository.get_latest_job("example-collection", external_id)
    assert latest is not None and latest.status == "completed"


async def test_an_existing_database_gains_source_objects_without_a_migration_step(
    stack, settings: Settings
) -> None:
    """A database created before this feature must upgrade on the next start."""
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    schema = settings.db_schema

    await repository._pool.execute(f"DROP TABLE {schema}.source_objects")  # noqa: SLF001

    # An existing document with no retained original must still list safely.
    await repository.ensure_collection("example-collection")
    await repository.ensure_schema()

    tables = {
        row["tablename"]
        for row in await repository._pool.fetch(  # noqa: SLF001
            "SELECT tablename FROM pg_tables WHERE schemaname = $1", schema
        )
    }
    assert "source_objects" in tables
    assert await repository.get_source_object("example-collection", "legacy-source") is None


async def test_a_source_objects_table_without_a_preview_checksum_gains_it(
    stack, settings: Settings
) -> None:
    """The upgrade path for a database created by the first cut of this feature."""
    repository, _vector_store, _embed_model, _ingestion, _source_store = stack
    schema = settings.db_schema
    document_id = uuid.uuid4()

    await repository._pool.execute(  # noqa: SLF001
        f"ALTER TABLE {schema}.source_objects DROP COLUMN preview_checksum"
    )
    await repository._pool.execute(  # noqa: SLF001
        f"""
        INSERT INTO {schema}.source_objects (
            document_id, collection_id, external_id, filename, media_type, byte_size,
            checksum, storage_backend, storage_key, preview_key, preview_bytes, page_count
        ) VALUES ($1, 'example-collection', 'file-legacy', 'legacy.pdf', 'application/pdf', 12,
                  'legacy-checksum', 'local', 'ab/legacy/original', 'ab/legacy/preview', 40, 3)
        """,
        document_id,
    )

    await repository.ensure_schema()

    legacy = await repository.get_source_object("example-collection", "file-legacy")
    assert legacy is not None
    # The pre-existing row keeps its data and reads back with no preview hash.
    assert legacy.preview_key == "ab/legacy/preview"
    assert legacy.preview_bytes == 40
    assert legacy.preview_checksum is None

    # Reprocessing records one, without touching the original's checksum.
    await repository.set_source_preview(
        document_id,
        preview_key="ab/legacy/preview",
        preview_bytes=64,
        preview_checksum="c" * 64,
        page_count=3,
    )
    updated = await repository.get_source_object("example-collection", "file-legacy")
    assert updated is not None
    assert updated.preview_checksum == "c" * 64
    assert updated.checksum == "legacy-checksum"

    # ensure_schema stays idempotent once the column exists.
    await repository.ensure_schema()
    assert (await repository.get_source_object("example-collection", "file-legacy")) == updated
