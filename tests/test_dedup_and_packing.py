from unittest.mock import MagicMock, patch
import pytest

from index import create_payload_indexes
from src.news_rag.chunking import Chunk, chunks_for_record
from src.news_rag.langchain_rag import (
    build_rag_context,
    parse_user_query,
    retrieve_unique_articles,
)
from src.news_rag.models import ArticleRecord, ParsedNewsQuery


def make_sample_record(
    article_id: str = "art_12345678",
    title: str = "Sample AI Article",
    categories: list[str] | None = None,
) -> ArticleRecord:
    """Create a sample ArticleRecord for testing metadata enrichment."""
    return ArticleRecord(
        id=article_id,
        url="https://economictimes.indiatimes.com/tech/artificial-intelligence/sample/articleshow/100.cms",
        title=title,
        author="Tech Reporter",
        published_at="2026-10-14T08:30:00Z",
        body="First paragraph of the article.\n\nSecond paragraph discusses deep learning.\n\nThird paragraph concludes with future outlook.",
        summary="Summary of the article.",
        companies=["Anthropic"],
        people=["Dario Amodei"],
        locations=["Bengaluru"],
        category="AI",
        categories=categories or ["AI", "technology"],
        funding=None,
    )


def test_chunk_metadata_enrichment() -> None:
    """Test that every chunk receives structured metadata in extract/chunking stage."""
    record = make_sample_record()
    chunks = chunks_for_record(record, chunk_size=40, overlap=10)

    assert len(chunks) > 0
    for idx, chunk in enumerate(chunks):
        assert chunk.metadata["article_id"] == record.id
        assert chunk.metadata["title"] == record.title
        assert chunk.metadata["url"] == str(record.url)
        assert chunk.metadata["published_date"] == record.published_at
        assert chunk.metadata["categories"] == ["ai", "technology"]
        assert chunk.metadata["chunk_index"] == idx


def test_parsed_news_query_clamping_and_normalization() -> None:
    """Test ParsedNewsQuery defaults, category normalization, and count clamping."""
    # Default count is 5
    q1 = ParsedNewsQuery(semantic_query="general updates")
    assert q1.target_article_count == 5
    assert q1.category_filter is None

    # Clamping between 1 and 25
    q_large = ParsedNewsQuery(semantic_query="big count", target_article_count=100)
    assert q_large.target_article_count == 25

    q_small = ParsedNewsQuery(semantic_query="small count", target_article_count=-5)
    assert q_small.target_article_count == 1

    # Category normalization (lowercase)
    q_cat = ParsedNewsQuery(semantic_query="ai news", category_filter="  AI  ")
    assert q_cat.category_filter == "ai"

    # Category None normalization for empty/null strings
    q_none = ParsedNewsQuery(semantic_query="tech", category_filter="none")
    assert q_none.category_filter is None


def test_parse_user_query_with_mocked_llm() -> None:
    """Test that query parsing using mocked LLM structured output parses requested count and category."""
    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    mock_structured.invoke.return_value = ParsedNewsQuery(
        semantic_query="mentioned AI",
        category_filter="ai",
        target_article_count=10,
    )

    query = "Tell me about 10 news articles that mentioned AI"
    parsed = parse_user_query(query, llm=mock_llm)

    assert parsed.target_article_count == 10
    assert parsed.category_filter == "ai"
    assert "AI" in parsed.semantic_query or "mentioned AI" in parsed.semantic_query
    assert mock_llm.with_structured_output.called
    assert mock_structured.invoke.called


def test_parse_user_query_deterministic_fallback() -> None:
    """Test fallback regex-based parser when LLM is unavailable."""
    query = "Tell me about 10 news articles that mentioned AI"
    parsed = parse_user_query(query, api_key=None, llm=None)

    assert parsed.target_article_count == 10
    assert parsed.category_filter == "ai"
    assert "AI" in parsed.semantic_query


def test_retrieve_unique_articles_deduplication_30_chunks_top_8_same() -> None:
    """Test that when vector store returns 30 chunks where the top 8 chunks belong

    to the same article_id, the deduplication logic still returns 10 distinct article_id groups.
    """
    mock_results = []
    # Chunks 0..7 belong to article_1 (scores 0.99 down to 0.92)
    for i in range(8):
        doc = MagicMock()
        doc.page_content = f"Article 1 paragraph {i}"
        doc.metadata = {
            "article_id": "article_1",
            "title": "Article One",
            "url": "https://example.com/art1",
            "published_date": "2026-10-14T00:00:00Z",
            "categories": ["ai"],
            "chunk_index": i,
        }
        mock_results.append((doc, 0.01 + i * 0.01))  # distance -> high score

    # Chunks 8..29 belong to article_2 through article_23
    for i in range(2, 24):
        doc = MagicMock()
        doc.page_content = f"Article {i} content"
        doc.metadata = {
            "article_id": f"article_{i}",
            "title": f"Article {i}",
            "url": f"https://example.com/art{i}",
            "published_date": "2026-10-14T00:00:00Z",
            "categories": ["ai"],
            "chunk_index": 0,
        }
        mock_results.append((doc, 0.2 + (i * 0.01)))

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search_with_score.return_value = mock_results

    parsed_query = ParsedNewsQuery(
        semantic_query="ai news",
        category_filter="ai",
        target_article_count=10,
    )

    grouped_articles = retrieve_unique_articles(parsed_query, vectorstore=mock_vectorstore)

    # Must return exactly 10 distinct article_id groups
    assert len(grouped_articles) == 10
    assert "article_1" in grouped_articles
    # article_1 should have collected its multiple chunks
    assert len(grouped_articles["article_1"]) == 8
    # All 10 distinct article IDs should be present
    expected_ids = [f"article_{i}" for i in range(1, 11)]
    assert list(grouped_articles.keys()) == expected_ids


def test_retrieve_unique_articles_fallback_on_empty_filter() -> None:
    """Test fallback to unfiltered search if category_filter returns fewer than target_article_count."""
    filtered_results = [
        (
            MagicMock(
                page_content="Only one match",
                metadata={"article_id": "art_only_1", "chunk_index": 0},
            ),
            0.1,
        )
    ]
    unfiltered_results = [
        (
            MagicMock(
                page_content=f"Match {i}",
                metadata={"article_id": f"art_unfiltered_{i}", "chunk_index": 0},
            ),
            0.1 + i * 0.05,
        )
        for i in range(1, 6)
    ]

    mock_vectorstore = MagicMock()

    # When called with filter, return only 1 match (< target_article_count 5)
    # When called without filter, return 5 matches
    def mock_search_fn(query, k, filter=None):
        if filter is not None:
            return filtered_results
        return unfiltered_results

    mock_vectorstore.similarity_search_with_score.side_effect = mock_search_fn

    parsed_query = ParsedNewsQuery(
        semantic_query="broad search",
        category_filter="obscure_tag",
        target_article_count=5,
    )

    grouped = retrieve_unique_articles(parsed_query, vectorstore=mock_vectorstore)
    assert len(grouped) == 5
    assert "art_unfiltered_1" in grouped


def test_adaptive_context_packing_switches_multi_to_single_chunk() -> None:
    """Test that adaptive context packing switches from multi-chunk to single-chunk mode when target_article_count > 5."""
    # Build 6 sample articles, each with 3 chunks with different scores and indices
    articles = {}
    for a_idx in range(1, 7):
        aid = f"art_{a_idx}"
        chunks = [
            Chunk(
                id=f"{aid}_c2",
                text=f"Article {a_idx} - Third paragraph (index 2)",
                metadata={
                    "article_id": aid,
                    "title": f"Title {a_idx}",
                    "published_date": "2026-10-14T00:00:00Z",
                    "url": f"https://example.com/{aid}",
                    "categories": ["ai", "tech"],
                    "chunk_index": 2,
                },
                score=0.75,
            ),
            Chunk(
                id=f"{aid}_c0",
                text=f"Article {a_idx} - First paragraph (index 0, highest score)",
                metadata={
                    "article_id": aid,
                    "title": f"Title {a_idx}",
                    "published_date": "2026-10-14T00:00:00Z",
                    "url": f"https://example.com/{aid}",
                    "categories": ["ai", "tech"],
                    "chunk_index": 0,
                },
                score=0.95,
            ),
            Chunk(
                id=f"{aid}_c1",
                text=f"Article {a_idx} - Second paragraph (index 1)",
                metadata={
                    "article_id": aid,
                    "title": f"Title {a_idx}",
                    "published_date": "2026-10-14T00:00:00Z",
                    "url": f"https://example.com/{aid}",
                    "categories": ["ai", "tech"],
                    "chunk_index": 1,
                },
                score=0.85,
            ),
        ]
        articles[aid] = chunks

    # Case 1: target_article_count <= 5 (multi-chunk mode: up to 3 chunks sorted by chunk_index)
    articles_small = {k: v for i, (k, v) in enumerate(articles.items()) if i < 3}
    context_multi = build_rag_context(articles_small, target_article_count=3)

    # Should contain all paragraphs for each article
    assert "First paragraph" in context_multi
    assert "Second paragraph" in context_multi
    assert "Third paragraph" in context_multi
    # Should maintain coherent reading order (index 0 before index 1 before index 2)
    pos_0 = context_multi.find("Article 1 - First paragraph")
    pos_1 = context_multi.find("Article 1 - Second paragraph")
    pos_2 = context_multi.find("Article 1 - Third paragraph")
    assert pos_0 < pos_1 < pos_2
    assert '<article index="1" id="art_1">' in context_multi

    # Case 2: target_article_count > 5 (single-chunk mode: header + top 1 highest-scoring chunk)
    context_single = build_rag_context(articles, target_article_count=6)

    # Should contain the top 1 highest-scoring chunk (First paragraph with score 0.95)
    assert "First paragraph (index 0, highest score)" in context_single
    # Should NOT contain lower-scoring chunks in single-chunk mode
    assert "Second paragraph" not in context_single
    assert "Third paragraph" not in context_single
    # Headers must be present
    assert "<title>Title 1</title>" in context_single
    assert "<url>https://example.com/art_1</url>" in context_single
    assert "<categories>ai, tech</categories>" in context_single
    # Should number each article up to 6
    assert '<article index="6" id="art_6">' in context_single


def test_create_payload_indexes_invokes_store_method() -> None:
    """Test create_payload_indexes calls create_payload_index if supported."""
    mock_store = MagicMock()
    create_payload_indexes(mock_store)

    assert mock_store.create_payload_index.call_count == 2
    mock_store.create_payload_index.assert_any_call(field_name="article_id")
    mock_store.create_payload_index.assert_any_call(field_name="categories")
