"""The authenticated source-metadata and source-content endpoints."""

import pytest
from httpx import AsyncClient

from app.ingest import document_uuid, upload_external_id
from app.storage import ORIGINAL_VARIANT, PREVIEW_VARIANT, StorageError, storage_key
from tests.conftest import API_TOKEN, auth_headers
from tests.fakes import (
    DocumentSegment,
    InMemoryRepository,
    InMemorySourceStore,
    RecordingConverter,
)

COLLECTION = "example-collection"
TEXT = b"The expiratory valve needs calibration every year."
PDF = b"%PDF-1.7 binary body that is long enough to slice"


async def upload(
    client: AsyncClient,
    *,
    filename: str = "notes.txt",
    content: bytes = TEXT,
    media_type: str = "text/plain",
    collection_id: str = COLLECTION,
    token: str = API_TOKEN,
):
    return await client.post(
        "/v1/documents/file",
        files={"file": (filename, content, media_type)},
        data={"collection_id": collection_id},
        headers=auth_headers(token),
    )


async def settle(client: AsyncClient, accepted) -> dict:
    body = accepted.json()
    await client.app.state.runtime.ingestion.wait_for_pending()  # type: ignore[attr-defined]
    return body


async def source(
    client: AsyncClient,
    external_id: str,
    *,
    collection_id: str = COLLECTION,
    token: str = API_TOKEN,
):
    return await client.get(
        "/v1/documents/source",
        params={"collection_id": collection_id, "external_id": external_id},
        headers=auth_headers(token),
    )


async def content(
    client: AsyncClient,
    external_id: str,
    *,
    collection_id: str = COLLECTION,
    variant: str = ORIGINAL_VARIANT,
    headers: dict[str, str] | None = None,
    token: str = API_TOKEN,
):
    return await client.get(
        "/v1/documents/source/content",
        params={
            "collection_id": collection_id,
            "external_id": external_id,
            "variant": variant,
        },
        headers={**auth_headers(token), **(headers or {})},
    )


async def test_an_upload_is_retained_before_indexing_finishes(
    client: AsyncClient, source_store: InMemorySourceStore, repository: InMemoryRepository
) -> None:
    accepted = await upload(client)
    external_id = accepted.json()["external_id"]

    # The bytes are durable as soon as the upload is accepted — before the
    # background job has run at all.
    key = storage_key(document_uuid(COLLECTION, external_id), ORIGINAL_VARIANT)
    assert source_store.objects[key] == TEXT
    assert repository.source_objects[(COLLECTION, external_id)].checksum

    await settle(client, accepted)
    described = await source(client, external_id)

    assert described.status_code == 200
    body = described.json()
    assert body["filename"] == "notes.txt"
    assert body["media_type"] == "text/plain"
    assert body["byte_size"] == len(TEXT)
    assert body["status"] == "ready"
    assert body["page_count"] is None
    # Text is its own readable form, so no second normalized copy is stored.
    assert body["preview_available"] is False


async def test_the_original_bytes_are_served_back_verbatim(client: AsyncClient) -> None:
    accepted = await upload(client)
    await settle(client, accepted)

    served = await content(client, accepted.json()["external_id"])

    assert served.status_code == 200
    assert served.content == TEXT
    assert served.headers["content-type"] == "text/plain; charset=utf-8"
    assert served.headers["content-disposition"].startswith("inline;")
    assert served.headers["x-content-type-options"] == "nosniff"
    assert served.headers["accept-ranges"] == "bytes"
    assert served.headers["cache-control"] == "private, no-cache"


async def test_a_range_request_returns_only_that_window(client: AsyncClient) -> None:
    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    partial = await content(client, external_id, headers={"Range": "bytes=5-14"})

    assert partial.status_code == 206
    assert partial.content == PDF[5:15]
    assert partial.headers["content-range"] == f"bytes 5-14/{len(PDF)}"
    assert partial.headers["content-length"] == "10"

    unsatisfiable = await content(client, external_id, headers={"Range": "bytes=9000-9100"})
    assert unsatisfiable.status_code == 416
    assert unsatisfiable.headers["content-range"] == f"bytes */{len(PDF)}"


async def test_an_unchanged_source_is_revalidated_rather_than_re_sent(
    client: AsyncClient,
) -> None:
    """A PDF page jump reloads the frame, so repeat requests must be cheap."""
    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    first = await content(client, external_id)
    validator = first.headers["etag"]

    assert first.headers["cache-control"] == "private, no-cache"
    assert validator.startswith('"') and validator.endswith('-original"')

    unchanged = await content(client, external_id, headers={"If-None-Match": validator})
    assert unchanged.status_code == 304
    assert unchanged.content == b""
    assert unchanged.headers["etag"] == validator

    # A validator from the other representation must not satisfy this one.
    preview_validator = (await content(client, external_id, variant=PREVIEW_VARIANT)).headers["etag"]
    assert preview_validator != validator
    crossed = await content(client, external_id, headers={"If-None-Match": preview_validator})
    assert crossed.status_code == 200

    # A conditional range request still returns the range, not a 304.
    ranged = await content(
        client, external_id, headers={"If-None-Match": validator, "Range": "bytes=0-3"}
    )
    assert ranged.status_code == 206
    assert ranged.content == PDF[:4]


async def test_a_changed_preview_is_not_served_from_a_stale_validator(
    client: AsyncClient, converter: RecordingConverter, repository: InMemoryRepository
) -> None:
    """Reprocessing the same original can change the extracted text.

    The original's bytes are unchanged, so its validator must not change. The
    preview's did change, so a browser holding the old preview validator has to be
    sent the new text rather than a 304.
    """
    converter.segments = [DocumentSegment(text="Battery interval: three years.", page=4)]
    first = await upload(client, filename="service.pdf", content=PDF, media_type="application/pdf")
    await settle(client, first)
    external_id = first.json()["external_id"]

    original_validator = (await content(client, external_id)).headers["etag"]
    stale_preview = await content(client, external_id, variant=PREVIEW_VARIANT)
    stale_validator = stale_preview.headers["etag"]
    assert stale_preview.text == "Battery interval: three years."
    assert stale_validator != original_validator

    # The same file again, but the converter now extracts different text.
    converter.segments = [DocumentSegment(text="Battery interval: five years.", page=4)]
    second = await upload(client, filename="service.pdf", content=PDF, media_type="application/pdf")
    await settle(client, second)
    assert second.json()["external_id"] == external_id

    refreshed = await content(
        client, external_id, variant=PREVIEW_VARIANT, headers={"If-None-Match": stale_validator}
    )
    assert refreshed.status_code == 200, "a changed preview must not be revalidated away"
    assert refreshed.text == "Battery interval: five years."
    fresh_validator = refreshed.headers["etag"]
    assert fresh_validator != stale_validator

    # The new validator does revalidate, so caching still works.
    unchanged = await content(
        client, external_id, variant=PREVIEW_VARIANT, headers={"If-None-Match": fresh_validator}
    )
    assert unchanged.status_code == 304

    # The original's bytes never changed, so its validator must not have either.
    assert (await content(client, external_id)).headers["etag"] == original_validator
    assert (
        await content(client, external_id, headers={"If-None-Match": original_validator})
    ).status_code == 304

    stored = repository.source_objects[(COLLECTION, external_id)]
    assert stored.preview_checksum and stored.preview_checksum != stored.checksum


async def test_a_preview_row_written_before_checksums_still_validates(
    client: AsyncClient, converter: RecordingConverter, repository: InMemoryRepository
) -> None:
    """An upgraded deployment has preview rows with no stored hash.

    Those must still produce a validator that is distinct from the original's and
    that revalidates, rather than failing or colliding.
    """
    from dataclasses import replace

    converter.segments = [DocumentSegment(text="Legacy extracted text.", page=2)]
    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    key = (COLLECTION, external_id)
    repository.source_objects[key] = replace(repository.source_objects[key], preview_checksum=None)

    preview = await content(client, external_id, variant=PREVIEW_VARIANT)
    original = await content(client, external_id)

    assert preview.status_code == 200
    assert preview.headers["etag"] != original.headers["etag"]
    assert (
        await content(
            client,
            external_id,
            variant=PREVIEW_VARIANT,
            headers={"If-None-Match": preview.headers["etag"]},
        )
    ).status_code == 304


async def test_a_pdf_is_offered_inline_and_office_files_are_not(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    pdf = await upload(client, filename="service.pdf", content=PDF, media_type="application/pdf")
    await settle(client, pdf)
    docx = await upload(
        client,
        filename="report.docx",
        content=b"PK\x03\x04 word",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    await settle(client, docx)

    served_pdf = await content(client, pdf.json()["external_id"])
    served_docx = await content(client, docx.json()["external_id"])

    assert served_pdf.headers["content-type"] == "application/pdf"
    assert served_pdf.headers["content-disposition"].startswith("inline;")
    # An Office file downloads; it is never handed to the browser as renderable.
    assert served_docx.headers["content-type"] == "application/octet-stream"
    assert served_docx.headers["content-disposition"].startswith("attachment;")


async def test_uploaded_html_is_never_served_inline(client: AsyncClient) -> None:
    accepted = await upload(
        client,
        filename="page.html",
        content=b"<script>alert(1)</script><p>Body</p>",
        media_type="text/html",
    )
    await settle(client, accepted)

    served = await content(client, accepted.json()["external_id"])

    assert served.headers["content-type"] == "application/octet-stream"
    assert served.headers["content-disposition"].startswith("attachment;")


async def test_a_converted_source_gets_a_plain_text_preview(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    converter.segments = [
        DocumentSegment(text="Battery replacement interval", page=4, section="2 Maintenance"),
        DocumentSegment(text="Alarm code 1", page=17, section="3 Alarms"),
    ]
    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    described = await source(client, external_id)
    preview = await content(client, external_id, variant=PREVIEW_VARIANT)

    assert described.json()["preview_available"] is True
    assert described.json()["page_count"] == 17
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "text/plain; charset=utf-8"
    assert preview.text == "Battery replacement interval\n\nAlarm code 1"
    assert "extracted.txt" in preview.headers["content-disposition"]


async def test_a_source_without_a_preview_reports_it_rather_than_guessing(
    client: AsyncClient,
) -> None:
    accepted = await upload(client)
    await settle(client, accepted)

    missing = await content(client, accepted.json()["external_id"], variant=PREVIEW_VARIANT)

    assert missing.status_code == 404
    assert "preview" in missing.json()["detail"]


async def test_page_provenance_reaches_the_indexed_chunks(
    client: AsyncClient, converter: RecordingConverter, vector_store
) -> None:
    converter.segments = [
        DocumentSegment(text="The battery module is replaced.", page=4, section="2 Maintenance"),
        DocumentSegment(text="The pressure sensor is calibrated.", page=17, section="3 Alarms"),
    ]

    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)

    located = {
        (node.metadata.get("page"), node.metadata.get("section"))
        for node in vector_store.nodes.values()
    }
    assert located == {(4, "2 Maintenance"), (17, "3 Alarms")}


async def test_a_converter_without_provenance_reports_no_pages(
    client: AsyncClient, converter: RecordingConverter, vector_store
) -> None:
    converter.segments = None

    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)

    described = await source(client, accepted.json()["external_id"])
    assert described.json()["page_count"] is None
    assert all(node.metadata.get("page") is None for node in vector_store.nodes.values())


async def test_a_source_in_another_collection_cannot_be_reached(client: AsyncClient) -> None:
    accepted = await upload(client, collection_id="other-collection")
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    assert (await source(client, external_id)).status_code == 404
    assert (await content(client, external_id)).status_code == 404
    # The same identity in its own collection is reachable, which is what makes
    # the two 404s above isolation rather than absence.
    assert (
        await source(client, external_id, collection_id="other-collection")
    ).status_code == 200


@pytest.mark.parametrize(
    ("collection_id", "external_id"),
    [
        ("Bad Collection", "file-abc"),
        ("../other", "file-abc"),
        (COLLECTION, ""),
        (COLLECTION, "x" * 257),
    ],
)
async def test_malformed_identifiers_are_refused_before_any_lookup(
    client: AsyncClient, collection_id: str, external_id: str
) -> None:
    described = await source(client, external_id, collection_id=collection_id)
    fetched = await content(client, external_id, collection_id=collection_id)

    assert described.status_code == 422
    assert fetched.status_code == 422


async def test_an_unknown_variant_is_refused(client: AsyncClient) -> None:
    accepted = await upload(client)
    await settle(client, accepted)

    rejected = await content(client, accepted.json()["external_id"], variant="../../etc/passwd")

    assert rejected.status_code == 422


async def test_source_endpoints_require_the_service_token(client: AsyncClient) -> None:
    accepted = await upload(client)
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    assert (await source(client, external_id, token="wrong")).status_code == 401
    assert (await content(client, external_id, token="wrong")).status_code == 401


async def test_a_source_ingested_without_a_retained_original_reports_unavailable(
    client: AsyncClient,
) -> None:
    """A text document ingested through /v1/documents/text has no stored file.

    This is also the shape of every upload indexed before originals were kept, so
    it must answer with a clear state rather than an error the UI cannot explain.
    """
    ingested = await client.post(
        "/v1/documents/text",
        json={
            "collection_id": COLLECTION,
            "external_id": "example-manual",
            "text": "The ventilator alarm list starts here.",
        },
        headers=auth_headers(),
    )
    ingested.raise_for_status()
    await client.app.state.runtime.ingestion.wait_for_pending()  # type: ignore[attr-defined]

    described = await source(client, "example-manual")
    listed = await client.get(
        "/v1/documents", params={"collection_id": COLLECTION}, headers=auth_headers()
    )

    assert described.status_code == 404
    assert "uploaded again" in described.json()["detail"]
    item = listed.json()["items"][0]
    assert item["status"] == "ready"
    assert item["viewable"] is False
    assert item["byte_size"] is None


async def test_a_listed_upload_advertises_that_it_can_be_opened(client: AsyncClient) -> None:
    accepted = await upload(client)
    await settle(client, accepted)

    listed = await client.get(
        "/v1/documents", params={"collection_id": COLLECTION}, headers=auth_headers()
    )

    item = listed.json()["items"][0]
    assert item["viewable"] is True
    assert item["byte_size"] == len(TEXT)
    assert item["preview_available"] is False


async def test_an_upload_is_refused_when_its_bytes_cannot_be_stored(
    client: AsyncClient, source_store: InMemorySourceStore, repository: InMemoryRepository
) -> None:
    source_store.put_error = StorageError("volume is read-only")

    refused = await upload(client)

    assert refused.status_code == 503
    assert "could not be stored" in refused.json()["detail"]
    # No job is created, so the drawer never shows an upload that has no bytes.
    assert repository.jobs == {}


async def test_re_uploading_identical_content_reuses_one_stored_object(
    client: AsyncClient, source_store: InMemorySourceStore
) -> None:
    first = await upload(client)
    await settle(client, first)
    second = await upload(client)
    await settle(client, second)

    assert second.json()["external_id"] == first.json()["external_id"]
    assert len(source_store.objects) == 1


async def test_a_row_whose_file_vanished_answers_404_rather_than_a_short_body(
    client: AsyncClient, source_store: InMemorySourceStore
) -> None:
    accepted = await upload(client)
    await settle(client, accepted)
    source_store.objects.clear()

    served = await content(client, accepted.json()["external_id"])

    assert served.status_code == 404


async def test_a_failed_upload_is_still_viewable(
    client: AsyncClient, converter: RecordingConverter
) -> None:
    """A conversion failure must not cost the tester the file they uploaded."""
    from app.parsing import ConversionFailedError

    converter.error = ConversionFailedError("The document converter could not read this file.")
    accepted = await upload(
        client, filename="service.pdf", content=PDF, media_type="application/pdf"
    )
    await settle(client, accepted)
    external_id = accepted.json()["external_id"]

    described = await source(client, external_id)
    served = await content(client, external_id)

    assert described.status_code == 200
    assert described.json()["status"] == "failed"
    assert served.status_code == 200
    assert served.content == PDF
