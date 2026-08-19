"""Chunking that keeps a service manual's structure retrievable.

Docling reports which chunks came from a table and honours a token budget, so
structure is never inferred from text here. What remains is the part no converter
does: a chunk is embedded on its own, so a terse row such as
``| E-142 | Flow sensor drift | Replace SV-3 |`` carries no trace of the manual or
page it came from and loses to any paragraph that merely discusses flow sensors.
Every chunk therefore opens with a provenance line, which the lexical half of
hybrid retrieval indexes as well.

Central splitting stays because plain text, Markdown, and Azure Document
Intelligence uploads carry no table structure to read; their tables are still
split blindly.
"""

import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    TextNode,
    TransformComponent,
)
from llama_index.core.utils import get_tokenizer

TABLE_METADATA_KEYS = ("table_part", "table_parts")

# How far into a chunk a heading may sit and still count as already present.
_HEADING_LOOKAHEAD_CHARACTERS = 200

_SENTENCE_PATTERN = re.compile(r"[^.!?\n]+(?:[.!?]+|\n+)|[^.!?\n]+$")


def split_sentences(text: str) -> list[str]:
    """Split sentences and manual lines without NLTK or downloaded corpora."""
    return _SENTENCE_PATTERN.findall(text)


def context_header(metadata: dict[str, Any], text: str = "") -> str:
    """The provenance line prepended to a chunk, empty when nothing is known.

    A section already carried by *text* is left out, since Docling contextualizes
    its own chunks with their headings.
    """
    title = str(metadata.get("title") or "").strip()
    section = str(metadata.get("section") or "").strip()
    if section and section in text[: len(section) + _HEADING_LOOKAHEAD_CHARACTERS]:
        section = ""
    located = " > ".join(part for part in (title, section) if part)
    page = metadata.get("page")
    if isinstance(page, int) and not isinstance(page, bool):
        located = f"{located} - page {page}" if located else f"page {page}"
    return f"[{located}]" if located else ""


def split_table(table: str, *, budget: int, tokenizer: Any) -> list[str]:
    """Split one Markdown table into parts of at most *budget* tokens.

    A row is never split and the header is repeated in every part, so a part may
    exceed *budget* when a single row does. Rows that lose their header also lose
    what each column means.
    """
    rows = table.split("\n")
    header, body = rows[:2], rows[2:]
    if not body:
        return [table]

    header_tokens = len(tokenizer("\n".join(header)))
    parts: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for row in body:
        row_tokens = len(tokenizer(row))
        if current and header_tokens + current_tokens + row_tokens > budget:
            parts.append("\n".join(header + current))
            current = []
            current_tokens = 0
        current.append(row)
        current_tokens += row_tokens
    parts.append("\n".join(header + current))
    return parts


class StructuralChunker(TransformComponent):
    """Split located segments into chunks that can still be found.

    A segment the converter identified as a table is kept whole when it fits and
    split on row boundaries when it does not. Everything else is delegated to the
    sentence splitter unchanged.
    """

    chunk_size: int
    splitter: SentenceSplitter

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        super().__init__(
            chunk_size=chunk_size,
            splitter=SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_tokenizer_fn=split_sentences,
            ),
        )

    @classmethod
    def class_name(cls) -> str:
        return "StructuralChunker"

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[BaseNode]:
        chunks: list[BaseNode] = []
        for node in nodes:
            chunks.extend(self._split_node(node))
        return chunks

    def _split_node(self, node: BaseNode) -> list[BaseNode]:
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        if not text.strip():
            return []

        header = context_header(node.metadata, text)
        if node.metadata.get("is_table"):
            chunks = self._table_chunks(node, text, header=header)
        else:
            chunks = list(self.splitter([node]))

        for chunk in chunks:
            _prepend(chunk, header)
        return chunks

    def _table_chunks(self, node: BaseNode, text: str, *, header: str) -> list[BaseNode]:
        caption = str(node.metadata.get("caption") or "").strip()
        prefix = "" if not caption or caption in text else f"{caption}\n"

        tokenizer = get_tokenizer()
        # The header row, caption and provenance line ride along in every part, so
        # the row budget has to leave room for them.
        reserved = len(tokenizer(f"{header}\n{prefix}")) if (header or prefix) else 0
        parts = split_table(
            text,
            budget=max(self.chunk_size - reserved, 1),
            tokenizer=tokenizer,
        )

        chunks = [_derived(node, f"{prefix}{part}") for part in parts]
        if len(chunks) > 1:
            for position, chunk in enumerate(chunks, start=1):
                chunk.metadata["table_part"] = position
                chunk.metadata["table_parts"] = len(chunks)
                _exclude_metadata(chunk, TABLE_METADATA_KEYS)
        return chunks


def _derived(node: BaseNode, text: str) -> TextNode:
    """One chunk of *node*, carrying its metadata, exclusions and provenance.

    The source relationship is copied because deletion is scoped by ``ref_doc_id``:
    a chunk that named an intermediate node would survive its document's re-ingest.
    """
    chunk = TextNode(text=text, metadata=dict(node.metadata))
    chunk.relationships = dict(node.relationships)
    chunk.excluded_embed_metadata_keys = list(node.excluded_embed_metadata_keys)
    chunk.excluded_llm_metadata_keys = list(node.excluded_llm_metadata_keys)
    return chunk


def _prepend(chunk: BaseNode, header: str) -> None:
    if not header:
        return
    text = chunk.get_content(metadata_mode=MetadataMode.NONE)
    if text.startswith(header):
        return
    chunk.set_content(f"{header}\n{text}")


def _exclude_metadata(chunk: BaseNode, keys: Sequence[str]) -> None:
    for key in keys:
        if key in chunk.metadata:
            if key not in chunk.excluded_embed_metadata_keys:
                chunk.excluded_embed_metadata_keys.append(key)
            if key not in chunk.excluded_llm_metadata_keys:
                chunk.excluded_llm_metadata_keys.append(key)
