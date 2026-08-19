"""Structural chunking: provenance prefixes and row-safe table splitting."""

import pytest
from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    ObjectType,
    RelatedNodeInfo,
    TextNode,
)

from app.embeddings.chunking import StructuralChunker, context_header, split_table

DOCUMENT_ID = "11111111-2222-3333-4444-555555555555"

TABLE = "\n".join(
    ["| Alarm | Cause | Action |", "| --- | --- | --- |"]
    + [f"| E-{code} | Cause {code} | Replace SV-{code} |" for code in range(100, 160)]
)


def node(text: str, **metadata: object) -> TextNode:
    """An ingest-shaped input node: located, with metadata kept out of the text."""
    full = {
        "document_id": DOCUMENT_ID,
        "title": "Servo Service Manual",
        "section": "7.4 Alarm troubleshooting",
        "page": 112,
        **metadata,
    }
    built = TextNode(text=text, metadata=full)
    built.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(
        node_id=DOCUMENT_ID, node_type=ObjectType.DOCUMENT
    )
    built.excluded_embed_metadata_keys = list(full)
    built.excluded_llm_metadata_keys = list(full)
    return built


def contents(chunks: list) -> list[str]:
    return [chunk.get_content(metadata_mode=MetadataMode.NONE) for chunk in chunks]


def test_context_header_names_the_document_and_page() -> None:
    assert (
        context_header({"title": "Servo Manual", "section": "7.4 Alarms", "page": 112})
        == "[Servo Manual > 7.4 Alarms - page 112]"
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, ""),
        ({"title": "Servo Manual"}, "[Servo Manual]"),
        ({"section": "7.4"}, "[7.4]"),
        ({"page": 9}, "[page 9]"),
        ({"title": "", "section": "", "page": None}, ""),
        # A boolean is an int in Python; it is not a page.
        ({"page": True}, ""),
    ],
)
def test_context_header_omits_what_is_not_known(metadata: dict, expected: str) -> None:
    assert context_header(metadata) == expected


def test_context_header_does_not_repeat_a_heading_the_converter_already_added() -> None:
    """Docling contextualizes its own chunks with their headings."""
    header = context_header(
        {"title": "Servo Manual", "section": "7.4 Alarms", "page": 112},
        text="7.4 Alarms\nWhen the alarm is active, check the flow sensor.",
    )
    assert header == "[Servo Manual - page 112]"


def test_every_chunk_carries_its_provenance() -> None:
    chunker = StructuralChunker(chunk_size=64, chunk_overlap=8)
    chunks = chunker([node("Purge the circuit. " * 200)])

    assert len(chunks) > 1
    for text in contents(chunks):
        assert text.startswith("[Servo Service Manual > 7.4 Alarm troubleshooting - page 112]\n")


def test_provenance_stays_out_of_the_metadata_text() -> None:
    """The prefix is content; metadata itself must not leak into embeddings."""
    chunker = StructuralChunker(chunk_size=256, chunk_overlap=16)
    chunk = chunker([node("Purge the circuit.")])[0]

    embedded = chunk.get_content(metadata_mode=MetadataMode.EMBED)
    assert embedded.startswith("[Servo Service Manual")
    assert "document_id" not in embedded


def test_chunks_keep_the_document_as_their_source() -> None:
    """Deletion is scoped by ref_doc_id, so every chunk must keep the document id."""
    chunker = StructuralChunker(chunk_size=64, chunk_overlap=8)
    chunks = chunker([node(TABLE, is_table=True)]) + chunker([node("Prose. " * 100)])

    assert len(chunks) > 2
    assert {chunk.ref_doc_id for chunk in chunks} == {DOCUMENT_ID}


def test_table_chunks_keep_the_metadata_retrieval_depends_on() -> None:
    """A chunk without collection_id is dropped by the isolation guard at search."""
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True, collection_id="manuals")])

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["collection_id"] == "manuals"
        assert chunk.metadata["document_id"] == DOCUMENT_ID
        assert chunk.metadata["title"] == "Servo Service Manual"
        assert chunk.metadata["page"] == 112
        assert chunk.excluded_embed_metadata_keys
        assert "collection_id" not in chunk.get_content(metadata_mode=MetadataMode.EMBED)


def test_an_empty_segment_produces_no_chunks() -> None:
    chunker = StructuralChunker(chunk_size=256, chunk_overlap=16)
    assert chunker([node("   \n  ")]) == []


def test_a_table_the_converter_flagged_repeats_its_header_in_every_part() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True)])

    assert len(chunks) > 1
    for text in contents(chunks):
        body = text.split("\n", 1)[1]
        assert body.startswith("| Alarm | Cause | Action |\n| --- | --- | --- |")


def test_table_rows_are_never_split() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True)])

    rows = [
        line for text in contents(chunks) for line in text.split("\n") if line.startswith("| E-")
    ]
    assert rows == [line for line in TABLE.split("\n") if line.startswith("| E-")]


def test_a_table_that_fits_stays_whole() -> None:
    small = "| Alarm | Action |\n| --- | --- |\n| E-142 | Replace SV-3 |"
    chunker = StructuralChunker(chunk_size=800, chunk_overlap=120)
    chunks = chunker([node(small, is_table=True)])

    assert len(chunks) == 1
    assert "table_part" not in chunks[0].metadata


def test_split_table_parts_are_numbered() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True)])

    total = len(chunks)
    assert [chunk.metadata["table_part"] for chunk in chunks] == list(range(1, total + 1))
    assert {chunk.metadata["table_parts"] for chunk in chunks} == {total}


def test_table_metadata_never_reaches_the_embedding() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunk = chunker([node(TABLE, is_table=True)])[0]

    assert "table_part" not in chunk.get_content(metadata_mode=MetadataMode.EMBED)
    assert "table_part" not in chunk.get_content(metadata_mode=MetadataMode.LLM)


def test_the_converters_caption_is_carried_into_every_part() -> None:
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE, is_table=True, caption="Table 7-4 Alarm codes")])

    assert len(chunks) > 1
    for text in contents(chunks):
        assert "Table 7-4 Alarm codes" in text


def test_a_caption_already_in_the_text_is_not_repeated() -> None:
    caption = "Table 7-4 Alarm codes"
    chunker = StructuralChunker(chunk_size=800, chunk_overlap=120)
    small = f"{caption}\n| Alarm | Action |\n| --- | --- |\n| E-142 | Replace SV-3 |"
    chunk = chunker([node(small, is_table=True, caption=caption)])[0]

    assert chunk.get_content(metadata_mode=MetadataMode.NONE).count(caption) == 1


def test_an_unflagged_table_goes_to_the_sentence_splitter() -> None:
    """Structure comes from the converter: tabular-looking text is not a table."""
    chunker = StructuralChunker(chunk_size=200, chunk_overlap=16)
    chunks = chunker([node(TABLE)])

    assert len(chunks) > 1
    assert "table_part" not in chunks[0].metadata
    bodies = [text.split("\n", 1)[1] for text in contents(chunks)]
    assert not all(body.startswith("| Alarm | Cause | Action |") for body in bodies)


def test_split_table_keeps_an_oversized_row_intact() -> None:
    table = "| A | B |\n| --- | --- |\n| " + "x" * 400 + " | y |"
    parts = split_table(table, budget=8, tokenizer=lambda text: text.split())

    assert len(parts) == 1
    assert parts[0] == table
