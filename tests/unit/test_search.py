"""Search: collection isolation, filtering, citations, and bounds."""

from llama_index.core.vector_stores.types import FilterOperator
from httpx import AsyncClient

from tests.conftest import auth_headers, ingest
from tests.fakes import RecordingVectorStore

EXAMPLE_TEXT = (
    "Calibrate the inspiratory pressure sensor during preventive maintenance. "
    "A persistent alarm means the expiratory valve needs replacement."
)
OTHER_TEXT = (
    "Calibrate the oxygen pressure sensor on the anaesthesia trolley. "
    "A persistent alarm means the battery pack needs replacement."
)


async def seed_two_collections(client: AsyncClient) -> None:
    await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "example-manual",
            "title": "Example service manual",
            "source_type": "manual",
            "source_uri": "file:///data/example-manual.txt",
            "section": "6.5.2 Pre-use check",
            "page": 151,
            "text": EXAMPLE_TEXT,
        },
    )
    await ingest(
        client,
        {
            "collection_id": "other-manual",
            "external_id": "trolley-manual",
            "title": "Trolley manual",
            "source_type": "manual",
            "text": OTHER_TEXT,
        },
    )


async def search(client: AsyncClient, body: dict) -> dict:
    response = await client.post("/v1/search", json=body, headers=auth_headers())
    response.raise_for_status()
    return response.json()


async def test_search_never_returns_another_collection(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    await seed_two_collections(client)

    result = await search(
        client, {"query": "pressure sensor alarm", "collection_ids": ["example-collection"]}
    )

    assert result["items"], "the requested collection should match"
    assert {item["collection_id"] for item in result["items"]} == {"example-collection"}
    assert all("trolley" not in item["text"].lower() for item in result["items"])
    assert all("anaesthesia" not in item["text"].lower() for item in result["items"])
    # The fake store ignores metadata filters, so isolation here is the service's
    # own post-retrieval guard rather than backend cooperation.
    assert len(vector_store.nodes) > len(result["items"])


async def test_search_can_span_several_named_collections(client: AsyncClient) -> None:
    await seed_two_collections(client)

    result = await search(
        client,
        {"query": "pressure sensor alarm", "collection_ids": ["example-collection", "other-manual"]},
    )

    assert {item["collection_id"] for item in result["items"]} == {"example-collection", "other-manual"}


async def test_search_sends_a_collection_filter_to_the_backend(
    client: AsyncClient, vector_store: RecordingVectorStore
) -> None:
    await seed_two_collections(client)
    await search(client, {"query": "pressure", "collection_ids": ["example-collection"]})

    applied = vector_store.queries[-1].filters
    assert applied is not None
    collection_filter = next(f for f in applied.filters if f.key == "collection_id")
    assert collection_filter.operator == FilterOperator.IN
    assert collection_filter.value == ["example-collection"]


async def test_search_requires_at_least_one_collection(client: AsyncClient) -> None:
    response = await client.post(
        "/v1/search", json={"query": "pressure", "collection_ids": []}, headers=auth_headers()
    )
    assert response.status_code == 422

    response = await client.post(
        "/v1/search", json={"query": "pressure"}, headers=auth_headers()
    )
    assert response.status_code == 422


async def test_results_carry_document_metadata_and_citations(client: AsyncClient) -> None:
    await seed_two_collections(client)

    result = await search(
        client, {"query": "pressure sensor alarm", "collection_ids": ["example-collection"]}
    )
    item = result["items"][0]

    assert item["external_id"] == "example-manual"
    assert item["title"] == "Example service manual"
    assert item["source_type"] == "manual"
    assert item["source_uri"] == "file:///data/example-manual.txt"
    assert item["page"] == 151
    assert item["section"] == "6.5.2 Pre-use check"
    assert item["document_id"]
    assert item["chunk_id"]
    assert item["updated_at"]
    citation = item["citations"][0]
    assert citation["label"] == "Example service manual"
    assert citation["locator"] == "section:6.5.2 Pre-use check"
    assert citation["page"] == 151


async def test_filters_narrow_results_within_a_collection(client: AsyncClient) -> None:
    await seed_two_collections(client)
    await ingest(
        client,
        {
            "collection_id": "example-collection",
            "external_id": "release-note",
            "source_type": "note",
            "text": "The pressure sensor alarm threshold changed in this release.",
        },
    )

    everything = await search(
        client, {"query": "pressure sensor alarm", "collection_ids": ["example-collection"]}
    )
    assert {item["source_type"] for item in everything["items"]} == {"manual", "note"}

    notes = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "filters": {"source_type": ["note"]},
        },
    )
    assert {item["source_type"] for item in notes["items"]} == {"note"}

    by_document = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "filters": {"external_id": ["release-note"]},
        },
    )
    assert {item["external_id"] for item in by_document["items"]} == {"release-note"}

    without_legacy = await search(
        client,
        {
            "query": "pressure sensor alarm",
            "collection_ids": ["example-collection"],
            "filters": {"exclude_external_id": ["example-manual"]},
        },
    )
    assert {item["external_id"] for item in without_legacy["items"]} == {"release-note"}


async def test_empty_results_report_a_warning(client: AsyncClient) -> None:
    result = await search(
        client, {"query": "pressure sensor", "collection_ids": ["example-collection"]}
    )

    assert result["items"] == []
    assert result["warnings"] == ["no_results"]
    assert result["stats"]["returned"] == 0


async def test_top_k_is_bounded_by_configuration(client: AsyncClient) -> None:
    await seed_two_collections(client)

    result = await search(
        client,
        {"query": "pressure sensor alarm", "collection_ids": ["example-collection"], "top_k": 1},
    )
    assert len(result["items"]) == 1

    response = await client.post(
        "/v1/search",
        json={"query": "pressure", "collection_ids": ["example-collection"], "top_k": 500},
        headers=auth_headers(),
    )
    assert response.status_code == 422
