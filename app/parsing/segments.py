"""Located pieces of a normalized document.

A segment is one run of text plus wherever it came from. Pagination is optional
on purpose: a PDF has stable page numbers, a Markdown file does not, and a
citation must never claim a page the format cannot have.
"""

from dataclasses import dataclass

# Matches the section field length the ingest contract accepts.
MAXIMUM_SECTION_CHARACTERS = 512

_CONTROL_CHARACTERS = {code: None for code in range(32) if code not in (9, 10)}


@dataclass(frozen=True, slots=True)
class DocumentSegment:
    """One located run of text."""

    text: str
    page: int | None = None
    section: str = ""
    # Read from the converter's document structure, never inferred from the text:
    # how a table is serialized is the converter's choice.
    is_table: bool = False
    caption: str = ""


@dataclass(frozen=True, slots=True)
class ConvertedDocument:
    """Normalized text plus the provenance carried alongside it.

    ``text`` is the whole document as one string: it is what a preview shows and
    what the checksum is taken over, so change detection stays independent of how
    the converter happened to split the document this time.
    """

    text: str
    segments: tuple[DocumentSegment, ...]
    page_count: int | None = None

    @property
    def has_provenance(self) -> bool:
        return any(segment.page is not None or segment.section for segment in self.segments)


def clean_text(text: str) -> str:
    """Normalize newlines and drop control characters that break chunking."""
    return text.replace("\r\n", "\n").replace("\r", "\n").translate(_CONTROL_CHARACTERS).strip()


def build_converted_document(segments: list[DocumentSegment]) -> ConvertedDocument:
    """Clean and bound *segments*, dropping the ones that hold no text."""
    cleaned: list[DocumentSegment] = []
    pages: list[int] = []
    for segment in segments:
        text = clean_text(segment.text)
        if not text:
            continue
        if segment.page is not None:
            pages.append(segment.page)
        cleaned.append(
            DocumentSegment(
                text=text,
                page=segment.page,
                section=clean_text(segment.section)[:MAXIMUM_SECTION_CHARACTERS],
                is_table=segment.is_table,
                caption=clean_text(segment.caption)[:MAXIMUM_SECTION_CHARACTERS],
            )
        )
    return ConvertedDocument(
        text="\n\n".join(segment.text for segment in cleaned),
        segments=tuple(cleaned),
        # The highest page that produced text, which is what page navigation can
        # actually reach. It is not claimed to be the document's physical length.
        page_count=max(pages) if pages else None,
    )
