"""Client for a docling-serve instance, driven through its asynchronous task API.

Three endpoints of the pinned v1.30.0 contract are used:

``POST {base}/v1/chunk/hybrid/file/async``
    Submits one file and answers ``{"task_id": ..., "task_status": ...}``. This is
    the asynchronous form of the hybrid chunk route, so it keeps the structural
    provenance the viewer needs -- per-chunk ``page_numbers``, ``headings``,
    ``captions`` and ``doc_items`` -- and unlike the hierarchical route it honours
    a token budget, so Athena does not have to re-split what Docling returns.
    Markdown conversion is deliberately not used as a fallback: it flattens away
    the page a passage came from, and a citation that cannot name a page cannot
    open one.

    ``chunking_use_markdown_tables`` is requested because the chunkers otherwise
    serialize a table as triplets. Markdown rows are what makes a table that still
    exceeds the budget splittable on row boundaries, and ``doc_items`` is what says
    a chunk is a table in the first place: its entries are document self-references
    such as ``#/tables/0``.

``GET {base}/v1/status/poll/{task_id}``
    Reports ``pending``, ``started``, ``success``, ``partial_success`` or
    ``failure`` for a submitted task.

``GET {base}/v1/result/{task_id}``
    Returns the completed ``{"chunks": [...], "documents": [...]}`` payload.

The synchronous routes are not used. Docling answers them with HTTP 504 once its
own ``DOCLING_SERVE_MAX_SYNC_WAIT`` elapses even though its worker keeps running,
which loses work that in fact completed. Submitting a task and polling it means
the conversion deadline is Athena's to set, and the task id is durable, so a
restart resumes the same conversion instead of paying for it twice.

Docling is never exposed to browsers. It is reached from this service only, every
request is bounded by its own timeout as well as by the overall conversion
deadline, and every response is size-capped before it is parsed so a misbehaving
or compromised converter cannot exhaust memory here.
"""

import asyncio
import json
import time
from datetime import UTC, datetime
from typing import Any

import httpx

from app.log import logger
from app.parsing.errors import (
    ConversionDeadlineExceededError,
    ConversionFailedError,
    ConversionResultUnavailableError,
    ConversionSubmissionError,
    ConversionTaskLostError,
    ConversionUnavailableError,
    DocumentTooLargeError,
)
from app.parsing.segments import ConvertedDocument, DocumentSegment, build_converted_document

CHUNK_ASYNC_PATH = "/v1/chunk/hybrid/file/async"
STATUS_PATH = "/v1/status/poll"
RESULT_PATH = "/v1/result"

_TABLE_REFERENCE_PREFIX = "#/tables/"

_PENDING_STATUSES = frozenset({"pending", "started"})
_ACCEPTED_STATUSES = frozenset({"success", "partial_success"})
# Statuses that mean "this deployment has no asynchronous chunk route". The
# route exists in the pinned image, so this is an explicit failure rather than a
# silent downgrade to a Markdown conversion that carries no page provenance.
_ROUTE_ABSENT_STATUSES = frozenset({404, 405, 501})
# Statuses that say "not now" rather than "not ever": an auth, throttling or
# gateway answer is about the converter's state, not about this document.
_UNAVAILABLE_STATUSES = frozenset({401, 403, 408, 429})
# How long the converter may keep answering transiently before the conversion is
# given up on. It is a duration rather than a count because the cost of the two
# mistakes is asymmetric: a conversion can run for an hour, so discarding one
# over a brief gateway blip wastes all of that work and makes the next upload pay
# for it again, while a converter that has genuinely gone away is still given up
# on promptly. The overall deadline bounds this too.
_TRANSIENT_FAILURE_GRACE_SECONDS = 300.0
# A submit or status response is a short JSON object. Capping it far below the
# result cap means a converter that answers a status poll with a document body
# is refused before it is buffered.
_STATUS_RESPONSE_BYTES = 64 * 1024
# Chunk results repeat per-chunk metadata, so they run larger than the Markdown
# they describe. The cap still bounds memory; it is just not the same cap.
_RESULT_RESPONSE_FACTOR = 4

_FAILED_MESSAGE = "The document converter could not read this file."
_SUBMIT_REJECTED_MESSAGE = (
    "The document converter refused this file. Check that it is a readable document of "
    "a supported type."
)
_DEADLINE_MESSAGE = (
    "Document conversion did not finish within the conversion deadline. Try a smaller "
    "document or increase the Docling conversion deadline."
)
_TASK_LOST_MESSAGE = (
    "The document converter no longer holds this conversion, so it could not be "
    "finished. Upload the file again to start a new conversion."
)
_RESULT_MESSAGE = (
    "The document converter finished but its result could not be retrieved. Upload the "
    "file again."
)
_UNAVAILABLE_MESSAGE = (
    "The document converter is unavailable, so PDF and Office uploads cannot be "
    "processed right now. Plain text and Markdown uploads still work."
)
_ROUTE_ABSENT_MESSAGE = (
    "The document converter does not offer the asynchronous chunking API this service "
    "requires. Ask the service owner to run a supported docling-serve version."
)
_NO_TEXT_MESSAGE = (
    "The document converter returned no readable text for this file. A scanned "
    "document needs OCR, which this evaluation service does not run."
)


def _monotonic() -> float:
    """Indirection so tests can drive a long conversion without waiting for one."""
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


class DoclingClient:
    """Converts one binary document into indexable, located text.

    Conversion is two steps the caller can persist between: :meth:`submit` hands
    the file to Docling and returns the task id, and :meth:`await_result` follows
    that task to its end. Keeping them apart is what makes a conversion survive
    an Athena restart.
    """

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        *,
        max_response_bytes: int,
        request_timeout_seconds: float,
        deadline_seconds: float,
        poll_interval_seconds: float,
        max_chunk_tokens: int,
    ) -> None:
        base = base_url.rstrip("/")
        self._submit_url = f"{base}{CHUNK_ASYNC_PATH}"
        self._status_url = f"{base}{STATUS_PATH}"
        self._result_url = f"{base}{RESULT_PATH}"
        self._client = client
        self._max_response_bytes = max_response_bytes
        self._request_timeout_seconds = request_timeout_seconds
        self._deadline_seconds = deadline_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_chunk_tokens = max_chunk_tokens

    @property
    def name(self) -> str:
        return "docling"

    @property
    def deadline_seconds(self) -> float:
        return self._deadline_seconds

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument:
        """Submit and follow one conversion without persisting the task id.

        Used by callers that keep no durable job state of their own. The ingest
        path uses :meth:`submit` and :meth:`await_result` instead, so a restart
        does not resubmit work Docling is still doing.
        """
        task_id = await self.submit(filename=filename, media_type=media_type, content=content)
        return await self.await_result(task_id, submitted_at=datetime.now(UTC))

    async def submit(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> str:
        """Enqueue one conversion and return the converter's task id."""
        body = await self._request(
            "POST",
            self._submit_url,
            limit=_STATUS_RESPONSE_BYTES,
            timeout=self._request_timeout_seconds,
            files={"files": (filename, content, media_type)},
            data={
                "include_converted_doc": "false",
                # Docling's own budget, so chunks need no re-splitting here by a
                # splitter that cannot see the structure.
                "chunking_max_tokens": str(self._max_chunk_tokens),
                "chunking_use_markdown_tables": "true",
            },
            rejected=ConversionSubmissionError(_SUBMIT_REJECTED_MESSAGE),
            route_absent=ConversionUnavailableError(_ROUTE_ABSENT_MESSAGE),
        )
        payload = _payload(body)
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            logger.warning("Docling accepted %s without returning a task id", filename)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
        _reject_terminal_status(payload.get("task_status"))
        return task_id.strip()

    async def await_result(
        self, task_id: str, *, submitted_at: datetime
    ) -> ConvertedDocument:
        """Poll *task_id* to completion and return the converted document.

        The deadline runs from *submitted_at*, not from this call, so resuming a
        task after a restart cannot silently grant the conversion a second full
        deadline.
        """
        deadline = _monotonic() + self._remaining_seconds(submitted_at)
        # Set on the first transient answer and cleared by the next good one, so the
        # grace window measures one continuous outage rather than a tally of blips
        # spread over an hour.
        failing_since: float | None = None
        while True:
            if _monotonic() >= deadline:
                logger.warning("Docling conversion %s exceeded its deadline", task_id)
                raise ConversionDeadlineExceededError(_DEADLINE_MESSAGE)
            try:
                if await self._is_complete(task_id, deadline):
                    # Retrieval is inside the retried region on purpose: by this
                    # point the conversion has succeeded, and dropping it because
                    # one fetch got a 503 would throw that away. A result the
                    # converter has since discarded answers 404 instead, which is
                    # a lost task rather than a transient one and is not retried.
                    return await self._result(task_id)
            except ConversionUnavailableError:
                now = _monotonic()
                failing_since = now if failing_since is None else failing_since
                if now - failing_since >= _TRANSIENT_FAILURE_GRACE_SECONDS:
                    logger.warning(
                        "Docling has been unreachable for %.0fs while converting %s; giving up",
                        now - failing_since,
                        task_id,
                    )
                    raise
                logger.warning("Docling is answering transiently for %s; retrying", task_id)
            else:
                failing_since = None
            await _sleep(min(self._poll_interval_seconds, _remaining(deadline)))

    def _remaining_seconds(self, submitted_at: datetime) -> float:
        # A naive timestamp would otherwise raise here; treating it as UTC keeps a
        # resume working against a row written by an older revision.
        anchored = submitted_at if submitted_at.tzinfo else submitted_at.replace(tzinfo=UTC)
        elapsed = (datetime.now(UTC) - anchored).total_seconds()
        remaining = self._deadline_seconds - max(elapsed, 0.0)
        if remaining <= 0:
            raise ConversionDeadlineExceededError(_DEADLINE_MESSAGE)
        return remaining

    async def _is_complete(self, task_id: str, deadline: float) -> bool:
        body = await self._request(
            "GET",
            f"{self._status_url}/{task_id}",
            limit=_STATUS_RESPONSE_BYTES,
            timeout=min(self._request_timeout_seconds, _remaining(deadline)),
            rejected=ConversionTaskLostError(_TASK_LOST_MESSAGE),
            route_absent=ConversionTaskLostError(_TASK_LOST_MESSAGE),
        )
        status = _status_of(_payload(body).get("task_status"))
        if status in _ACCEPTED_STATUSES:
            return True
        if status in _PENDING_STATUSES:
            return False
        logger.warning("Docling reported task status %r for %s", status, task_id)
        raise ConversionFailedError(_FAILED_MESSAGE)

    async def _result(self, task_id: str) -> ConvertedDocument:
        # Deliberately not clipped to the conversion deadline: the conversion is
        # already finished by the time this runs, and cutting the fetch short at
        # the boundary would throw away work that succeeded.
        body = await self._request(
            "GET",
            f"{self._result_url}/{task_id}",
            limit=self._max_response_bytes * _RESULT_RESPONSE_FACTOR,
            timeout=self._request_timeout_seconds,
            rejected=ConversionResultUnavailableError(_RESULT_MESSAGE),
            route_absent=ConversionTaskLostError(_TASK_LOST_MESSAGE),
        )
        return _extract_segments(_payload(body))

    async def _request(
        self,
        method: str,
        url: str,
        *,
        limit: int,
        timeout: float,
        rejected: Exception,
        route_absent: Exception,
        files: dict[str, tuple[str, bytes, str]] | None = None,
        data: dict[str, str] | None = None,
    ) -> bytes:
        """Perform one bounded request, mapping transport and status to a document error.

        *rejected* is raised for a 4xx that is about this document or task, and
        *route_absent* for the subset that means "this endpoint does not exist
        here". A 5xx, a throttling or auth answer, or a transport failure is
        always converter unavailability, because retrying later can succeed.
        """
        try:
            async with self._client.stream(
                method,
                url,
                files=files,
                data=data,
                headers={"Accept": "application/json"},
                timeout=timeout,
            ) as response:
                if response.status_code in _ROUTE_ABSENT_STATUSES:
                    await _drain(response, _STATUS_RESPONSE_BYTES)
                    logger.warning(
                        "Docling answered %s with HTTP %d", _url_path(url), response.status_code
                    )
                    raise route_absent
                if (
                    response.status_code >= 500
                    or response.status_code in _UNAVAILABLE_STATUSES
                ):
                    await _drain(response, _STATUS_RESPONSE_BYTES)
                    logger.warning(
                        "Docling answered %s with HTTP %d", _url_path(url), response.status_code
                    )
                    raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE)
                body = await _read_bounded(response, limit)
                if response.status_code >= 400:
                    logger.warning(
                        "Docling answered %s with HTTP %d", _url_path(url), response.status_code
                    )
                    raise rejected
                return body
        except (DocumentTooLargeError, ConversionUnavailableError):
            raise
        except httpx.TimeoutException as exc:
            logger.warning("Docling request to %s timed out", _url_path(url))
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc
        except httpx.HTTPError as exc:
            logger.warning("Docling request failed: %s", type(exc).__name__)
            raise ConversionUnavailableError(_UNAVAILABLE_MESSAGE) from exc


def _url_path(url: str) -> str:
    """The path of *url*, so a log line never carries the converter's address."""
    _, _, rest = url.partition("://")
    _, slash, path = rest.partition("/")
    return f"{slash}{path}"


def _remaining(deadline: float) -> float:
    """Seconds left before *deadline*, floored so httpx never sees zero or negative."""
    return max(deadline - _monotonic(), 0.001)


def _status_of(value: Any) -> str:
    return value.lower() if isinstance(value, str) else ""


def _reject_terminal_status(value: Any) -> None:
    """Refuse a submission that came back already failed, before any polling."""
    status = _status_of(value)
    if status and status not in _PENDING_STATUSES and status not in _ACCEPTED_STATUSES:
        logger.warning("Docling rejected the submission with status %r", status)
        raise ConversionFailedError(_FAILED_MESSAGE)


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


async def _drain(response: httpx.Response, limit: int) -> None:
    """Read and discard an error body, so the connection can be reused."""
    try:
        await _read_bounded(response, limit)
    except (DocumentTooLargeError, httpx.HTTPError):
        pass


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


def _extract_segments(payload: dict[str, Any]) -> ConvertedDocument:
    """Turn a chunk result into located segments.

    A chunk with no usable text is dropped rather than failing the document: a
    figure-only or empty region is normal in a real service manual.
    """
    if payload.get("kind") == "TaskFailureResult":
        logger.warning("Docling returned a task failure result")
        raise ConversionFailedError(_FAILED_MESSAGE)
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
                is_table=_is_table(raw.get("doc_items")),
                caption=_first_caption(raw.get("captions")),
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


def _is_table(value: Any) -> bool:
    """Whether this chunk came from a table, per the converter's own references.

    ``doc_items`` holds self-references such as ``#/tables/0``, so a table is
    recognized by where its content came from, not by how it was serialized.
    """
    if not isinstance(value, list):
        return False
    return any(
        isinstance(reference, str) and reference.startswith(_TABLE_REFERENCE_PREFIX)
        for reference in value
    )


def _first_caption(value: Any) -> str:
    """The first caption the converter attached, which names the table."""
    if not isinstance(value, list):
        return ""
    for caption in value:
        if isinstance(caption, str) and caption.strip():
            return caption.strip()
    return ""


def _deepest_heading(value: Any) -> str:
    """The most specific heading above a chunk, which is the useful locator."""
    if not isinstance(value, list):
        return ""
    for heading in reversed(value):
        if isinstance(heading, str) and heading.strip():
            return heading.strip()
    return ""
