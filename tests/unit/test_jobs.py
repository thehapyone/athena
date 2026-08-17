"""Job lifecycle, durability across restarts, and readiness behaviour."""

from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import Settings
from app.main import EmbeddingModelMismatchError, create_app
from app.repository import EmbeddingState, JobRecord
from tests.conftest import auth_headers, ingest
from tests.fakes import InMemoryRepository

DOCUMENT = {
    "collection_id": "example-collection",
    "external_id": "example-manual",
    "text": "Calibrate the inspiratory pressure sensor.",
}


async def test_unknown_job_returns_404(client: AsyncClient) -> None:
    response = await client.get(f"/v1/jobs/{uuid4()}", headers=auth_headers())
    assert response.status_code == 404


async def test_job_transitions_are_persisted(
    client: AsyncClient, repository: InMemoryRepository
) -> None:
    job = await ingest(client, DOCUMENT)

    stored = repository.jobs[UUID(job["job_id"])]
    assert stored.status == "completed"
    assert stored.chunk_count == job["chunk_count"]
    assert job["created_at"] and job["updated_at"]


async def test_job_status_survives_a_restart(
    settings: Settings, runtime_factory, repository: InMemoryRepository
) -> None:
    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://knowledge") as http_client:
            http_client.app = app  # type: ignore[attr-defined]
            job = await ingest(http_client, DOCUMENT)

    # A second process starting against the same durable state.
    restarted = create_app(settings, runtime_factory=runtime_factory)
    async with restarted.router.lifespan_context(restarted):
        transport = ASGITransport(app=restarted)
        async with AsyncClient(transport=transport, base_url="http://knowledge") as http_client:
            response = await http_client.get(f"/v1/jobs/{job['job_id']}", headers=auth_headers())

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["chunk_count"] == job["chunk_count"]


async def test_interrupted_jobs_are_reconciled_on_startup(
    settings: Settings, runtime_factory, repository: InMemoryRepository
) -> None:
    interrupted = JobRecord(
        job_id=uuid4(),
        collection_id="example-collection",
        external_id="example-manual",
        status="processing",
    )
    await repository.create_job(interrupted)

    app = create_app(settings, runtime_factory=runtime_factory)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://knowledge") as http_client:
            response = await http_client.get(
                f"/v1/jobs/{interrupted.job_id}", headers=auth_headers()
            )

    body = response.json()
    assert body["status"] == "failed"
    assert "restart" in body["detail"]


async def test_failed_ingestion_is_reported_on_the_job(
    client: AsyncClient, repository: InMemoryRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = client.app.state.runtime  # type: ignore[attr-defined]

    async def explode(*_args, **_kwargs):
        raise RuntimeError("embedding endpoint unreachable")

    monkeypatch.setattr(runtime.ingestion, "_index", explode)

    response = await client.post("/v1/documents/text", json=DOCUMENT, headers=auth_headers())
    job_id = response.json()["job_id"]
    await runtime.ingestion.wait_for_pending()

    job = (await client.get(f"/v1/jobs/{job_id}", headers=auth_headers())).json()
    assert job["status"] == "failed"
    assert "failed unexpectedly" in job["detail"]
    assert "embedding endpoint unreachable" not in job["detail"]


async def test_readiness_reports_database_failures(
    client: AsyncClient, repository: InMemoryRepository
) -> None:
    assert (await client.get("/health/ready")).json()["status"] == "ready"

    repository.probe_error = RuntimeError("connection refused")
    response = await client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert (await client.get("/health/live")).status_code == 200


async def test_embedding_state_is_recorded_on_first_start(
    client: AsyncClient, repository: InMemoryRepository, settings: Settings
) -> None:
    assert repository.embedding_state == EmbeddingState(
        model_name=settings.embedding_model, model_dim=settings.embedding_dimension
    )


async def test_startup_fails_when_the_indexed_embedding_model_changed(
    settings: Settings, runtime_factory, repository: InMemoryRepository
) -> None:
    await repository.set_embedding_state(EmbeddingState(model_name="other-model", model_dim=12))

    app = create_app(settings, runtime_factory=runtime_factory)
    with pytest.raises(EmbeddingModelMismatchError, match="does not match"):
        async with app.router.lifespan_context(app):
            pass
