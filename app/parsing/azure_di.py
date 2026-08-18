"""Minimal client for Azure AI Document Intelligence.

Analysis is asynchronous: submitting a document returns 202 with an
``Operation-Location`` header, and the result is fetched by polling that URL
until the operation reports ``succeeded`` or ``failed``. The prebuilt layout
model is used by default so a scanned PDF gets real OCR instead of the "no
readable text" failure a text-only converter would report, while still
returning the page and heading provenance citations need.

``paragraphs`` in the result carry per-paragraph page numbers and, for
headings, a ``role``. That is used to build the same page/section-located
segments Docling produces, rather than just the flattened ``content`` field:
a citation that cannot name a page cannot open one. When Azure returns
paragraphs, they are preferred; otherwise the whole document's markdown
``content`` is used as a single unlocated segment.

Azure is never exposed to browsers. It is reached from this service only,
over a bounded per-request timeout, and every response is size-capped before
it is parsed so a misbehaving or compromised endpoint cannot exhaust memory
here.
"""

import asyncio
import json
import time
from typing import Any
from urllib.parse import urlsplit

import httpx

from app.log import logger
from app.parsing.errors import (
    ConversionFailedError,
    ConversionUnavailableError,
    DocumentTooLargeError,
)
from app.parsing.segments import ConvertedDocument, DocumentSegment, build_converted_document

API_VERSION = "2024-11-30"
_ANALYZE_PATH_TEMPLATE = "/documentintelligence/documentModels/{model_id}:analyze"
_POLL_INTERVAL_SECONDS = 2.0
_HEADING_ROLES = frozenset({"title", "sectionHeading"})

_FAILED_MESSAGE = "The document converter could not read this file."
_TIMEOUT_MESSAGE = (
    "Document conversion took too long. Try a smaller document or increase the "
    "Azure Document Intelligence conversion timeout."
)
_UNAVAILABLE_MESSAGE = (
    "The document converter is unavailable, so PDF and Office uploads cannot be "
    "processed right now. Plain text and Markdown uploads still work."
)
_NO_TEXT_MESSAGE = (
    "The document converter returned no readable text for this file."
)


class AzureDocumentIntelligenceClient:
    """Converts one binary document into indexable, located text via Azure."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        client: httpx.AsyncClient,
        *,
        model_id: str,
        timeout_seconds: int,
        max_response_bytes: int,
    ) -> None:
        endpoint = endpoint.rstrip("/")
        self._analyze_url = f"{endpoint}{_ANALYZE_PATH_TEMPLATE.format(model_id=model_id)}"
        # The operation URL Azure hands back is followed with the API key attached;
        # it must resolve to the same endpoint we were configured with, or a
        # misbehaving or compromised responder could redirect the key elsewhere.
        parsed_endpoint = urlsplit(endpoint)
        self._trusted_origin = (parsed_endpoint.scheme, parsed_endpoint.netloc)
        self._api_key = api_key
        self._client = client
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

    @property
    def name(self) -> str:
        return "azure-document-intelligence"

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument:
        # One deadline for the whole conversion, started before submission: a
        # slow submit must count against AZURE_OCR_TIMEOUT_SECONDS the same as
        # slow polling does, or the two together could exceed it unnoticed.
        deadline = time.monotonic() + self._timeout_seconds
        operation_url = await self._submit(filename, media_type, content, deadline)
        payload = await self._poll(filename, operation_url, deadline)
        return _extract_segments(payload)

    async def _submit(
        self, filename: str, media_type: str, content: bytes, deadline: float
    ) -> str:
        # Streamed like polling, and for the same reason: a 202 body is normally
        # empty, but an error response from a misbehaving endpoint must still be
        # size-capped before it is read into memory.
        try:
            async with self._client.stream(
                "POST",
                self._analyze_url,
                params={"api-version": API_VERSION, "outputContentFormat": "markdown"},
                headers={
                    "Ocp-Apim-Subscription-Key": self._api_key,
                    "Content-Type": media_type or "application/octet-stream",
                },
                content=content,
                timeout=_remaining(deadline),
            ) as response:
                if response.status_code in (401, 403, 408, 429):
                    logger.warning(
                        "Azure Document Intelligence rejected %s with HTTP %d",
                        filename,
                        response.status_code,
                    )
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                if response.status_code >= 500:
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                await _read_bounded(response, self._max_response_bytes)
                if response.status_code != 202:
                    logger.warning(
                        "Azure Document Intelligence rejected %s with HTTP %d",
                        filename,
                        response.status_code,
                    )
                    raise ConversionFailedError(_FAILED_MESSAGE)
                operation_url = response.headers.get("operation-location")
        except DocumentTooLargeError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Azure Document Intelligence submission timed out for %s", filename)
            raise ConversionUnavailableError(_TIMEOUT_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.warning("Azure Document Intelligence request failed: %s", type(exc).__name__)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc

        if not operation_url or not self._is_trusted(operation_url):
            logger.warning(
                "Azure Document Intelligence accepted %s with no usable operation URL", filename
            )
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
        return operation_url

    def _is_trusted(self, operation_url: str) -> bool:
        parsed = urlsplit(operation_url)
        return (parsed.scheme, parsed.netloc) == self._trusted_origin

    async def _poll(self, filename: str, operation_url: str, deadline: float) -> dict[str, Any]:
        while True:
            if time.monotonic() >= deadline:
                logger.warning("Azure Document Intelligence timed out analyzing %s", filename)
                raise ConversionUnavailableError(_TIMEOUT_MESSAGE)
            body = await self._poll_once(operation_url, deadline)
            payload = _payload(body)
            raw_status = payload.get("status")
            status = raw_status.lower() if isinstance(raw_status, str) else ""
            if status == "succeeded":
                return payload
            if status in ("failed", "canceled", "cancelled"):
                logger.warning("Azure Document Intelligence failed to analyze %s", filename)
                raise ConversionFailedError(_FAILED_MESSAGE)
            if time.monotonic() >= deadline:
                logger.warning("Azure Document Intelligence timed out analyzing %s", filename)
                raise ConversionUnavailableError(_TIMEOUT_MESSAGE)
            await asyncio.sleep(min(_POLL_INTERVAL_SECONDS, _remaining(deadline)))

    async def _poll_once(self, operation_url: str, deadline: float) -> bytes:
        try:
            async with self._client.stream(
                "GET",
                operation_url,
                headers={"Ocp-Apim-Subscription-Key": self._api_key},
                timeout=_remaining(deadline),
            ) as response:
                if response.status_code >= 500:
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                if response.status_code >= 400:
                    logger.warning(
                        "Azure Document Intelligence polling failed with HTTP %d",
                        response.status_code,
                    )
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                return await _read_bounded(response, self._max_response_bytes)
        except DocumentTooLargeError:
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Azure Document Intelligence polling request timed out")
            raise ConversionUnavailableError(_TIMEOUT_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.warning("Azure Document Intelligence polling request failed: %s", type(exc).__name__)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc


def _remaining(deadline: float) -> float:
    """Seconds left before *deadline*, floored so httpx never sees zero or negative.

    Used as each request's own timeout, so a single slow request cannot by
    itself push the whole conversion past its configured deadline.
    """
    return max(deadline - time.monotonic(), 0.001)


async def _read_bounded(response: httpx.Response, limit: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    async for chunk in response.aiter_bytes():
        received += len(chunk)
        if received > limit:
            raise DocumentTooLargeError(
                "The converted document is larger than this service accepts."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _payload(body: bytes) -> dict[str, Any]:
    try:
        payload: Any = json.loads(body)
    except ValueError as exc:
        raise ConversionFailedError(_FAILED_MESSAGE) from exc
    if not isinstance(payload, dict):
        raise ConversionFailedError(_FAILED_MESSAGE)
    return payload


def _extract_segments(payload: dict[str, Any]) -> ConvertedDocument:
    result = payload.get("analyzeResult")
    if not isinstance(result, dict):
        raise ConversionFailedError(_FAILED_MESSAGE)

    raw_paragraphs = result.get("paragraphs")
    if isinstance(raw_paragraphs, list) and raw_paragraphs:
        segments = _segments_from_paragraphs(raw_paragraphs)
        if segments:
            return build_converted_document(segments)

    content = result.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ConversionFailedError(_NO_TEXT_MESSAGE)
    return build_converted_document([DocumentSegment(text=content)])


def _segments_from_paragraphs(raw_paragraphs: list[Any]) -> list[DocumentSegment]:
    """Located paragraphs, carrying forward the most recent heading as section.

    Azure reports no chunk hierarchy the way Docling does, so the last heading
    or title paragraph seen is used as the section for the paragraphs that
    follow it -- the same "most specific heading above this text" locator.
    """
    segments: list[DocumentSegment] = []
    section = ""
    for raw in raw_paragraphs:
        if not isinstance(raw, dict):
            continue
        text = raw.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        if raw.get("role") in _HEADING_ROLES:
            section = text.strip()
        segments.append(
            DocumentSegment(text=text, page=_first_page(raw.get("boundingRegions")), section=section)
        )
    return segments


def _first_page(value: Any) -> int | None:
    """The lowest page a paragraph's bounding regions touch."""
    if not isinstance(value, list):
        return None
    pages = [
        region.get("pageNumber")
        for region in value
        if isinstance(region, dict)
        and isinstance(region.get("pageNumber"), int)
        and not isinstance(region.get("pageNumber"), bool)
        and region.get("pageNumber") >= 1
    ]
    return min(pages) if pages else None
