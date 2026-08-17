"""Docling adapter behaviour, driven by a mock transport rather than a live server."""

import json

import httpx
import pytest

from app.parsing import (
    ConversionFailedError,
    ConversionUnavailableError,
    DoclingClient,
    DocumentTooLargeError,
)
from app.parsing.docling import CHUNK_PATH, CONVERT_PATH

BASE_URL = "http://docling:5001"


def client_with(handler, *, max_response_bytes: int = 1_000_000) -> DoclingClient:
    transport = httpx.MockTransport(handler)
    return DoclingClient(
        BASE_URL,
        httpx.AsyncClient(transport=transport, timeout=5.0),
        max_response_bytes=max_response_bytes,
    )


async def convert(client: DoclingClient):
    return await client.convert(
        filename="report.pdf", media_type="application/pdf", content=b"%PDF-1.7"
    )


def chunk_response(chunks: list[dict], *, status: str = "success") -> dict:
    return {
        "chunks": chunks,
        "documents": [{"kind": "ExportResult", "status": status, "content": {"filename": "report.pdf"}}],
        "processing_time": 0.1,
    }


def route_absent(status_code: int = 404):
    """A transport where only the chunk route is missing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_PATH:
            return httpx.Response(status_code, json={"detail": "Not Found"})
        return httpx.Response(200, json={"status": "success", "document": {"md_content": "# Ok"}})

    return handler


async def test_prefers_the_chunk_endpoint_and_keeps_page_provenance() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["body"] = request.content
        return httpx.Response(
            200,
            json=chunk_response(
                [
                    {
                        "text": "Chapter 1\nReplace the battery module every three years.",
                        "page_numbers": [4, 5],
                        "headings": ["Service manual", "1 Preventive maintenance"],
                    },
                    {
                        "text": "Monitoring technical error code 1.",
                        "page_numbers": [17],
                        "headings": ["2 Alarms"],
                    },
                ]
            ),
        )

    converted = await convert(client_with(handler))

    assert seen["method"] == "POST"
    assert seen["url"] == f"{BASE_URL}{CHUNK_PATH}"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="files"; filename="report.pdf"' in body
    assert b"%PDF-1.7" in body

    # The page a chunk starts on, and the most specific heading above it.
    assert [(segment.page, segment.section) for segment in converted.segments] == [
        (4, "1 Preventive maintenance"),
        (17, "2 Alarms"),
    ]
    assert converted.page_count == 17
    assert converted.has_provenance is True
    assert "Replace the battery module" in converted.text
    assert "Monitoring technical error code 1." in converted.text


async def test_chunks_without_pages_report_no_pagination() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chunk_response([{"text": "Body text.", "page_numbers": None, "headings": None}]),
        )

    converted = await convert(client_with(handler))

    assert converted.page_count is None
    assert converted.segments[0].page is None
    assert converted.segments[0].section == ""
    assert converted.has_provenance is False


@pytest.mark.parametrize("pages", [[], [0], ["4"], [True], "4", {"page": 4}])
async def test_unusable_page_numbers_are_not_turned_into_pages(pages: object) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chunk_response([{"text": "Body.", "page_numbers": pages}]))

    converted = await convert(client_with(handler))

    assert converted.segments[0].page is None
    assert converted.page_count is None


async def test_empty_and_malformed_chunks_are_skipped() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=chunk_response(
                [
                    "not an object",
                    {"text": "   "},
                    {"text": None},
                    {"text": "Real content.", "page_numbers": [2]},
                ]
            ),
        )

    converted = await convert(client_with(handler))

    assert [segment.text for segment in converted.segments] == ["Real content."]


async def test_a_chunk_response_with_no_usable_text_fails_the_job() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chunk_response([{"text": "  "}]))

    with pytest.raises(ConversionFailedError, match="no readable text"):
        await convert(client_with(handler))


async def test_a_failed_conversion_status_fails_the_job() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chunk_response([{"text": "Body."}], status="failure"))

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_partial_success_is_accepted() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=chunk_response([{"text": "Some text"}], status="partial_success")
        )

    converted = await convert(client_with(handler))

    assert converted.text == "Some text"


@pytest.mark.parametrize("status_code", [404, 405, 501])
async def test_a_missing_chunk_route_falls_back_to_markdown(status_code: int) -> None:
    client = client_with(route_absent(status_code))

    converted = await convert(client)

    assert converted.text == "# Ok"
    assert converted.page_count is None
    assert converted.has_provenance is False


async def test_the_markdown_fallback_is_used_for_every_later_document() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == CHUNK_PATH:
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json={"status": "success", "document": {"md_content": "# Ok"}})

    client = client_with(handler)
    await convert(client)
    await convert(client)

    # The chunk route is probed once, not once per document.
    assert calls == [CHUNK_PATH, CONVERT_PATH, CONVERT_PATH]


async def test_the_markdown_fallback_posts_the_documented_contract() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_PATH:
            return httpx.Response(404, json={"detail": "Not Found"})
        seen["url"] = str(request.url)
        seen["body"] = request.content
        return httpx.Response(200, json={"status": "success", "document": {"md_content": "# Ok"}})

    await convert(client_with(handler))

    assert seen["url"] == f"{BASE_URL}{CONVERT_PATH}"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'name="to_formats"' in body
    assert b"md" in body


@pytest.mark.parametrize(
    "payload",
    [
        {"chunks": "not a list"},
        {},
        [],
    ],
)
async def test_unusable_chunk_responses_fail_the_job(payload: object) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "failure", "document": {"md_content": "ignored"}},
        {"status": "success", "document": {}},
        {"status": "success", "document": {"md_content": "   "}},
        {"status": "success", "document": {"md_content": 12}},
        {"status": "success"},
        {"status": "success", "document": "not an object"},
        [],
    ],
)
async def test_unusable_markdown_fallbacks_fail_the_job(payload: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == CHUNK_PATH:
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json=payload)

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_non_json_responses_fail_the_job() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_client_errors_fail_the_job_and_server_errors_report_unavailability() -> None:
    def rejecting(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "unsupported"})

    with pytest.raises(ConversionFailedError):
        await convert(client_with(rejecting))

    def failing(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(failing))


async def test_server_timeout_reports_that_conversion_took_too_long() -> None:
    def timing_out(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(504, json={"detail": "maximum wait exceeded"})

    with pytest.raises(ConversionUnavailableError, match="took too long"):
        await convert(client_with(timing_out))


async def test_an_unreachable_converter_reports_unavailability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConversionUnavailableError, match="unavailable"):
        await convert(client_with(handler))


async def test_oversized_responses_are_cut_off_before_parsing() -> None:
    body = json.dumps(chunk_response([{"text": "x" * 5_000}])).encode()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    with pytest.raises(DocumentTooLargeError):
        await convert(client_with(handler, max_response_bytes=64))


async def test_no_message_leaks_the_converter_address() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConversionUnavailableError) as failure:
        await convert(client_with(handler))
    assert "docling:5001" not in str(failure.value)
