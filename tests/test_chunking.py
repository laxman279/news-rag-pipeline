import pytest

from src.news_rag.chunking import chunks_for_record, split_text
from src.news_rag.models import ArticleRecord


def make_record() -> ArticleRecord:
    return ArticleRecord(
        id="1234567890abcdef",
        url="https://economictimes.indiatimes.com/tech/startups/example/articleshow/1.cms",
        title="Example",
        author="",
        published_at="2026-10-14T00:00:00Z",
        body="Paragraph 1 content.\n\nParagraph 2 is slightly longer content.\n\nParagraph 3 final words.",
        summary="Example summary.",
        companies=[],
        people=[],
        locations=["Bengaluru"],
        category="general",
        funding=None,
    )


def test_split_text_preserves_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = split_text(text, chunk_size=30, overlap=0)
    assert chunks == [
        "First paragraph here.",
        "Second paragraph here.",
        "Third paragraph here.",
    ]


def test_split_text_falls_back_recursively() -> None:
    # A single paragraph that exceeds chunk_size falls back to sentences / words
    text = "First sentence. Second sentence is here."
    chunks = split_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    # Check that sentences / words are kept together
    assert all(len(c) <= 20 for c in chunks)


def test_split_text_uses_overlap() -> None:
    text = "Sentence one. Sentence two. Sentence three."
    chunks = split_text(text, chunk_size=28, overlap=10)
    assert len(chunks) >= 2
    # Ensure overlap creates shared text between adjacent chunks
    assert any(word in chunks[1] for word in chunks[0].split() if len(word) > 3)


def test_chunk_ids_are_idempotent() -> None:
    first = chunks_for_record(make_record(), chunk_size=40, overlap=10)
    second = chunks_for_record(make_record(), chunk_size=40, overlap=10)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert first[0].metadata["article_id"] == "1234567890abcdef"
    assert first[0].metadata["published_yyyymmdd"] == 20261014
    assert first[0].metadata["chunk_index"] == 0


def test_split_text_empty_and_whitespace() -> None:
    assert split_text("") == []
    assert split_text("   \n\n  \t  ") == []


def test_split_text_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        split_text("Sample text", chunk_size=0, overlap=0)

    with pytest.raises(ValueError, match="overlap must be non-negative and smaller"):
        split_text("Sample text", chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="overlap must be non-negative and smaller"):
        split_text("Sample text", chunk_size=100, overlap=-5)
