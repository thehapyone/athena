"""Turn an uploaded file into the normalized, located text the ingest path expects.

Text and Markdown are decoded in-process so the feature works on a small VM with
no converter running. Everything else goes through a ``DocumentConverter``, which
today is Docling and could later be an Azure or Mistral OCR adapter without the
ingest path changing.
"""

from dataclasses import dataclass
from typing import Protocol

from app.parsing.errors import (
    ConversionUnavailableError,
    DocumentDecodeError,
    DocumentTooLargeError,
)
from app.parsing.formats import UploadFormat
from app.parsing.segments import (
    ConvertedDocument,
    DocumentSegment,
    build_converted_document,
    clean_text,
)


class DocumentConverter(Protocol):
    """Converts a binary document to located text."""

    @property
    def name(self) -> str: ...

    async def convert(
        self,
        *,
        filename: str,
        media_type: str,
        content: bytes,
    ) -> ConvertedDocument: ...


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """One accepted upload, already validated against a supported format."""

    filename: str
    media_type: str
    content: bytes
    format: UploadFormat


class DocumentNormalizer:
    """Produces indexable, located text for an uploaded file."""

    def __init__(
        self,
        *,
        converter: DocumentConverter | None,
        max_text_bytes: int,
    ) -> None:
        self._converter = converter
        self._max_text_bytes = max_text_bytes

    @property
    def conversion_available(self) -> bool:
        return self._converter is not None

    async def to_document(self, upload: UploadedFile) -> ConvertedDocument:
        if upload.format.needs_conversion:
            if self._converter is None:
                raise ConversionUnavailableError(
                    f"{upload.format.label} uploads need a document converter, which is not "
                    "configured on this deployment. Plain text and Markdown uploads still work."
                )
            converted = await self._converter.convert(
                filename=upload.filename,
                media_type=upload.media_type,
                content=upload.content,
            )
        else:
            # A decoded text or Markdown file is its own normalized form, and it
            # has no pagination to report.
            converted = build_converted_document(
                [DocumentSegment(text=clean_text(_decode_text(upload.content)))]
            )

        if not converted.text:
            raise DocumentDecodeError("The document contains no readable text to index.")
        if len(converted.text.encode("utf-8")) > self._max_text_bytes:
            raise DocumentTooLargeError(
                "The document's text exceeds the size this service indexes."
            )
        return converted


def _decode_text(content: bytes) -> str:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DocumentDecodeError(
            "The file is not valid UTF-8 text. Save it as UTF-8 and upload it again."
        ) from exc
    return decoded
