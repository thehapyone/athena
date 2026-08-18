"""Azure Document Intelligence adapter behaviour, driven by a mock transport."""

import asyncio
import json
import time

import httpx
import pytest

from app.parsing import (
    AzureDocumentIntelligenceClient,
    ConversionFailedError,
    ConversionUnavailableError,
    DocumentTooLargeError,
)
from app.parsing.azure_di import API_VERSION

BASE_URL = "https://resource.cognitiveservices.azure.com"
ANALYZE_PATH = "/documentintelligence/documentModels/prebuilt-layout:analyze"
OPERATION_URL = f"{BASE_URL}/documentintelligence/documentModels/prebuilt-layout/analyzeResults/abc123"
API_KEY = "test-azure-key"


def client_with(
    handler,
    *,
    max_response_bytes: int = 1_000_000,
    timeout_seconds: int = 30,
) -> AzureDocumentIntelligenceClient:
    transport = httpx.MockTransport(handler)
    return AzureDocumentIntelligenceClient(
        BASE_URL,
        API_KEY,
        httpx.AsyncClient(transport=transport, timeout=5.0),
        model_id="prebuilt-layout",
        timeout_seconds=timeout_seconds,
        max_response_bytes=max_response_bytes,
    )


async def convert(client: AzureDocumentIntelligenceClient):
    return await client.convert(
        filename="report.pdf", media_type="application/pdf", content=b"%PDF-1.7"
    )


def accepted_response() -> httpx.Response:
    return httpx.Response(202, headers={"operation-location": OPERATION_URL})


def succeeded_response(result: dict) -> httpx.Response:
    return httpx.Response(200, json={"status": "succeeded", "analyzeResult": result})


def paragraph(text: str, *, page: int | None = None, role: str | None = None) -> dict:
    paragraph: dict = {"content": text}
    if page is not None:
        paragraph["boundingRegions"] = [{"pageNumber": page}]
    if role is not None:
        paragraph["role"] = role
    return paragraph


async def test_submits_and_polls_then_keeps_page_and_section_provenance() -> None:
    seen: dict[str, object] = {}
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.method == "POST":
            seen["url"] = str(request.url.copy_with(query=None))
            seen["params"] = dict(request.url.params)
            seen["headers"] = dict(request.headers)
            seen["body"] = request.content
            return accepted_response()
        assert str(request.url) == OPERATION_URL
        return succeeded_response(
            {
                "content": "ignored when paragraphs are present",
                "paragraphs": [
                    paragraph("1 Preventive maintenance", page=4, role="sectionHeading"),
                    paragraph("Replace the battery module every three years.", page=4),
                    paragraph("2 Alarms", page=17, role="sectionHeading"),
                    paragraph("Monitoring technical error code 1.", page=17),
                ],
            }
        )

    converted = await convert(client_with(handler))

    assert seen["url"] == f"{BASE_URL}{ANALYZE_PATH}"
    assert seen["params"]["api-version"] == API_VERSION
    assert seen["params"]["outputContentFormat"] == "markdown"
    assert seen["headers"]["ocp-apim-subscription-key"] == API_KEY
    assert seen["body"] == b"%PDF-1.7"
    assert calls == [ANALYZE_PATH, "/documentintelligence/documentModels/prebuilt-layout/analyzeResults/abc123"]

    assert [(segment.page, segment.section) for segment in converted.segments] == [
        (4, "1 Preventive maintenance"),
        (4, "1 Preventive maintenance"),
        (17, "2 Alarms"),
        (17, "2 Alarms"),
    ]
    assert converted.page_count == 17
    assert converted.has_provenance is True
    assert "Replace the battery module" in converted.text
    assert "Monitoring technical error code 1." in converted.text


async def test_no_paragraphs_falls_back_to_whole_document_content() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return succeeded_response({"content": "# Whole document markdown"})

    converted = await convert(client_with(handler))

    assert converted.text == "# Whole document markdown"
    assert converted.page_count is None
    assert converted.has_provenance is False


async def test_blank_paragraphs_are_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return succeeded_response(
            {
                "content": "fallback",
                "paragraphs": [{"content": "   "}, {"content": None}, paragraph("Real content.")],
            }
        )

    converted = await convert(client_with(handler))

    assert [segment.text for segment in converted.segments] == ["Real content."]


async def test_no_readable_text_anywhere_fails_the_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return succeeded_response({"content": "   ", "paragraphs": []})

    with pytest.raises(ConversionFailedError, match="no readable text"):
        await convert(client_with(handler))


async def test_polling_continues_while_the_operation_is_running(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.parsing import azure_di

    monkeypatch.setattr(azure_di, "_POLL_INTERVAL_SECONDS", 0.001)
    poll_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.method == "POST":
            return accepted_response()
        poll_count += 1
        if poll_count < 3:
            return httpx.Response(200, json={"status": "running"})
        return succeeded_response({"content": "done"})

    converted = await convert(client_with(handler))

    assert poll_count == 3
    assert converted.text == "done"


@pytest.mark.parametrize("status", ["failed", "canceled", "cancelled"])
async def test_a_terminal_failure_status_fails_the_job(status: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return httpx.Response(200, json={"status": status})

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_an_operation_url_outside_the_configured_endpoint_is_rejected() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(
            202, headers={"operation-location": "https://evil.example.com/steal-the-key"}
        )

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler))

    # The GET to the attacker-controlled URL, which would carry the API key,
    # must never have been attempted.
    assert len(calls) == 1


async def test_polling_past_the_timeout_reports_unavailability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return httpx.Response(200, json={"status": "running"})

    with pytest.raises(ConversionUnavailableError, match="took too long"):
        await convert(client_with(handler, timeout_seconds=0))


async def test_the_timeout_bounds_real_elapsed_time_even_with_slow_poll_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow-but-responding poll request must still count against the deadline.

    Counting only the sleep between polls, and not the polls themselves, would
    let a server that always takes just under its own timeout to answer turn a
    short configured timeout into an effectively unbounded wait.
    """
    from app.parsing import azure_di

    monkeypatch.setattr(azure_di, "_POLL_INTERVAL_SECONDS", 0.01)
    poll_request_seconds = 0.1
    timeout_seconds = 0.2

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        await asyncio.sleep(poll_request_seconds)
        return httpx.Response(200, json={"status": "running"})

    started = time.monotonic()
    with pytest.raises(ConversionUnavailableError, match="took too long"):
        await convert(client_with(handler, timeout_seconds=timeout_seconds))
    real_elapsed = time.monotonic() - started

    # Generous margin over the configured timeout: this fails outright under
    # the bug this test guards against, which took several seconds instead.
    assert real_elapsed < timeout_seconds + 5 * poll_request_seconds


@pytest.mark.parametrize("status_code", [401, 403, 408, 429])
async def test_auth_rate_limit_and_timeout_failures_report_unavailability(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json={"error": "denied"})

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler))


async def test_server_errors_on_submit_report_unavailability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="overloaded")

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler))


async def test_client_errors_on_submit_fail_the_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "unsupported content"})

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_a_missing_operation_location_reports_unavailability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(202)

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler))


@pytest.mark.parametrize("status_code", [400, 500])
async def test_poll_errors_report_unavailability(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return httpx.Response(status_code, text="broken")

    with pytest.raises(ConversionUnavailableError):
        await convert(client_with(handler))


async def test_non_json_poll_responses_fail_the_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return httpx.Response(200, content=b"<html>gateway</html>")

    with pytest.raises(ConversionFailedError):
        await convert(client_with(handler))


async def test_an_unreachable_endpoint_reports_unavailability() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConversionUnavailableError, match="unavailable"):
        await convert(client_with(handler))


async def test_oversized_poll_responses_are_cut_off_before_parsing() -> None:
    body = json.dumps(
        {"status": "succeeded", "analyzeResult": {"content": "x" * 5_000}}
    ).encode()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return accepted_response()
        return httpx.Response(200, content=body)

    with pytest.raises(DocumentTooLargeError):
        await convert(client_with(handler, max_response_bytes=64))


async def test_oversized_submit_responses_are_cut_off_before_parsing() -> None:
    """The submit response is size-capped too, not just the poll response.

    A 202 body is normally empty, but a misbehaving or compromised endpoint
    could still send an oversized error body, and it must not be read in full
    before that is noticed.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return succeeded_response({"content": "unreachable"})
        return httpx.Response(400, content=b"x" * 5_000)

    with pytest.raises(DocumentTooLargeError):
        await convert(client_with(handler, max_response_bytes=64))


async def test_no_message_leaks_the_endpoint_or_key() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConversionUnavailableError) as failure:
        await convert(client_with(handler))
    assert "resource.cognitiveservices.azure.com" not in str(failure.value)
    assert API_KEY not in str(failure.value)
