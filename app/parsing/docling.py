"""Minimal client for a docling-serve instance.

Two endpoints of the pinned v1.30.0 contract are used:

``POST {base}/v1/chunk/hierarchical/file``
    The preferred path. It answers ``{"chunks": [...], "documents": [...]}`` where
    each chunk carries ``text`` plus the structural provenance the viewer needs —
    ``page_numbers`` and ``headings``. That is why this is used instead of
    ``md_content`` alone: Markdown flattens away the page a passage came from, and
    a citation that cannot name a page cannot open one.

``POST {base}/v1/convert/file``
    The fallback, used when the chunk route is absent (an older or trimmed
    deployment). It yields ``md_content`` with no provenance, so citations then
    open the document without a location rather than inventing one.

Docling is never exposed to browsers. It is reached from this service only, over
a bounded timeout, and its response is size-capped before it is parsed so a
misbehaving or compromised converter cannot exhaust memory here.
"""

import json
from typing import Any

import httpx

from app.log import logger
from app.parsing.errors import (
    ConversionFailedError,
    ConversionUnavailableError,
    DocumentTooLargeError,
)
from app.parsing.segments import ConvertedDocument, DocumentSegment, build_converted_document

CONVERT_PATH = "/v1/convert/file"
CHUNK_PATH = "/v1/chunk/hierarchical/file"

_ACCEPTED_STATUSES = frozenset({"success", "partial_success"})
# Statuses that mean "this deployment has no chunk route", as opposed to "this
# file could not be read". Only these fall back to plain Markdown conversion.
_ROUTE_ABSENT_STATUSES = frozenset({404, 405, 501})
_FAILED_MESSAGE = "The document converter could not read this file."
_TIMEOUT_MESSAGE = (
    "Document conversion took too long. Try a smaller document or increase the "
    "Docling conversion timeout."
)
_UNAVAILABLE_MESSAGE = (
    "The document converter is unavailable, so PDF and Office uploads cannot be "
    "processed right now. Plain text and Markdown uploads still work."
)
_NO_TEXT_MESSAGE = (
    "The document converter returned no readable text for this file. A scanned "
    "document needs OCR, which this evaluation service does not run."
)
# Chunk responses repeat per-chunk metadata, so they run larger than the Markdown
# they describe. The cap still bounds memory; it is just not the same cap.
_CHUNK_RESPONSE_FACTOR = 4


class DoclingClient:
    """Converts one binary document into indexable, located text."""

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        *,
        max_response_bytes: int,
    ) -> None:
        base = base_url.rstrip("/")
        self._convert_url = f"{base}{CONVERT_PATH}"
        self._chunk_url = f"{base}{CHUNK_PATH}"
        self._client = client
        self._max_response_bytes = max_response_bytes
        self._chunk_supported = True

    @property
    def name(self) -> str:
        return "docling"

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument:
        """Convert *content*, preferring the structured chunk endpoint."""
        if self._chunk_supported:
            body = await self._post(
                self._chunk_url,
                filename=filename,
                media_type=media_type,
                content=content,
                data={"include_converted_doc": "false"},
                limit=self._max_response_bytes * _CHUNK_RESPONSE_FACTOR,
                allow_route_absent=True,
            )
            if body is not None:
                return _extract_segments(body)
            # Latched so one probe per process is enough: a deployment's route
            # table does not change while the service runs.
            self._chunk_supported = False
            logger.warning(
                "Docling has no %s route; falling back to Markdown conversion without "
                "page provenance",
                CHUNK_PATH,
            )

        body = await self._post(
            self._convert_url,
            filename=filename,
            media_type=media_type,
            content=content,
            data={"to_formats": "md"},
            limit=self._max_response_bytes,
            allow_route_absent=False,
        )
        return build_converted_document([DocumentSegment(text=_extract_markdown(body or b""))])

    async def _post(
        self,
        url: str,
        *,
        filename: str,
        media_type: str,
        content: bytes,
        data: dict[str, str],
        limit: int,
        allow_route_absent: bool,
    ) -> bytes | None:
        """POST one file. Returns ``None`` only when the route itself is absent."""
        try:
            async with self._client.stream(
                "POST",
                url,
                files={"files": (filename, content, media_type)},
                data=data,
                headers={"Accept": "application/json"},
            ) as response:
                if allow_route_absent and response.status_code in _ROUTE_ABSENT_STATUSES:
                    return None
                if response.status_code == 504:
                    logger.warning("Docling timed out while converting %s", filename)
                    raise ConversionUnavailableError(_TIMEOUT_MESSAGE)
                if response.status_code >= 500:
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                body = await _read_bounded(response, limit)
                if response.status_code >= 400:
                    logger.warning(
                        "Docling rejected %s with HTTP %d", filename, response.status_code
                    )
                    raise ConversionFailedError(_FAILED_MESSAGE)
        except DocumentTooLargeError:
            raise
        except httpx.HTTPError as exc:
            logger.warning("Docling request failed: %s", type(exc).__name__)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc
        return body


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


def _reject_failed_status(status: Any) -> None:
    if isinstance(status, str) and status.lower() not in _ACCEPTED_STATUSES:
        logger.warning("Docling reported conversion status %r", status)
        raise ConversionFailedError(_FAILED_MESSAGE)


def _extract_markdown(body: bytes) -> str:
    payload = _payload(body)
    _reject_failed_status(payload.get("status"))

    document = payload.get("document")
    markdown = document.get("md_content") if isinstance(document, dict) else None
    if not isinstance(markdown, str) or not markdown.strip():
        raise ConversionFailedError(_NO_TEXT_MESSAGE)
    return markdown


def _extract_segments(body: bytes) -> ConvertedDocument:
    """Turn a chunk response into located segments.

    A chunk with no usable text is dropped rather than failing the document: a
    figure-only or empty region is normal in a real service manual.
    """
    payload = _payload(body)
    for document in payload.get("documents") or ():
        if isinstance(document, dict):
            _reject_failed_status(document.get("status"))

    raw_chunks = payload.get("chunks")
    if not isinstance(raw_chunks, list):
        raise ConversionFailedError(_FAILED_MESSAGE)

    segments: list[DocumentSegment] = []
    for raw in raw_chunks:
        if not isinstance(raw, dict):
            continue
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        segments.append(
            DocumentSegment(
                text=text,
                page=_first_page(raw.get("page_numbers")),
                section=_deepest_heading(raw.get("headings")),
            )
        )
    if not segments:
        raise ConversionFailedError(_NO_TEXT_MESSAGE)
    return build_converted_document(segments)


def _first_page(value: Any) -> int | None:
    """The page a chunk starts on, or ``None`` for a format without pagination."""
    if not isinstance(value, list):
        return None
    pages = [
        item
        for item in value
        if isinstance(item, int) and not isinstance(item, bool) and item >= 1
    ]
    return min(pages) if pages else None


def _deepest_heading(value: Any) -> str:
    """The most specific heading above a chunk, which is the useful locator."""
    if not isinstance(value, list):
        return ""
    for heading in reversed(value):
        if isinstance(heading, str) and heading.strip():
            return heading.strip()
    return ""
