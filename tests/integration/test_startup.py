"""The production startup path against a real database.

Proves the service starts with only documented environment variables plus
PostgreSQL: no Athena repository content, configuration file, or embedding call
is involved. Skipped unless ``KNOWLEDGE_TEST_DATABASE_URL`` is set.
"""

import os
import uuid

import asyncpg
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import create_app

DATABASE_URL = os.environ.get("KNOWLEDGE_TEST_DATABASE_URL", "")

pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="KNOWLEDGE_TEST_DATABASE_URL is not set"
)

API_TOKEN = "integration-token-0123456789"


@pytest_asyncio.fixture
async def documented_env(tmp_path):
    schema = f"knowledge_start_{uuid.uuid4().hex[:12]}"
    env = {
        "KNOWLEDGE_DATABASE_URL": DATABASE_URL,
        "KNOWLEDGE_DB_SCHEMA": schema,
        "KNOWLEDGE_API_TOKEN": API_TOKEN,
        "KNOWLEDGE_EMBEDDING_BASE_URL": "https://example.invalid/openai/v1",
        "KNOWLEDGE_EMBEDDING_API_KEY": "unused-during-startup",
        "KNOWLEDGE_EMBEDDING_MODEL": "text-embedding-3-large",
        "KNOWLEDGE_EMBEDDING_DIMENSION": "1536",
        # The deployment default lives inside the container; a test run needs a
        # directory it can actually write to.
        "KNOWLEDGE_SOURCE_STORAGE_DIR": str(tmp_path / "sources"),
    }
    try:
        yield env
    finally:
        pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def test_service_starts_and_serves_health_and_auth(documented_env: dict[str, str]) -> None:
    settings = Settings.from_env(documented_env)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://knowledge") as client:
            live = await client.get("/health/live")
            ready = await client.get("/health/ready")
            unauthorized = await client.post(
                "/v1/search", json={"query": "x", "collection_ids": ["example-collection"]}
            )
            missing_job = await client.get(
                f"/v1/jobs/{uuid.uuid4()}", headers={"Authorization": f"Bearer {API_TOKEN}"}
            )

    assert live.status_code == 200
    assert ready.status_code == 200 and ready.json() == {
        "status": "ready",
        "database": "ok",
        "detail": None,
    }
    assert unauthorized.status_code == 401
    assert missing_job.status_code == 404


async def test_startup_creates_the_schema_and_vector_table(documented_env: dict[str, str]) -> None:
    settings = Settings.from_env(documented_env)
    app = create_app(settings)

    async with app.router.lifespan_context(app):
        pass

    pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=1)
    try:
        tables = {
            row["tablename"]
            for row in await pool.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = $1", settings.db_schema
            )
        }
    finally:
        await pool.close()

    assert {
        "collections",
        "documents",
        "ingest_jobs",
        "source_objects",
        "embedding_state",
    } <= tables
    assert any(name.endswith(settings.vector_table) for name in tables), tables
