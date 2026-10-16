from unittest.mock import MagicMock, patch

from query import generate_with_gemini, repeated_investors_answer, synthesize
from src.news_rag.models import ArticleRecord
from src.news_rag.planning import plan_query


def test_generate_with_gemini_formats_and_calls_langchain() -> None:
    fake_contexts = [
        {
            "text": "Sarvam AI announced a new funding round of $50M.",
            "metadata": {"title": "Sarvam Fundraise", "url": "https://example.com/sarvam"},
            "score": 0.85,
        }
    ]

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Sarvam AI raised $50M in funding."

    with patch("src.news_rag.langchain_rag.build_rag_chain", return_value=mock_chain) as mock_builder:
        ans = generate_with_gemini(
            "How much did Sarvam raise?",
            fake_contexts,
            model="gemini-3.8-flash",
            api_key="test-api-key",
        )
        assert ans == "Sarvam AI raised $50M in funding."
        assert mock_builder.called
        assert mock_chain.invoke.called


def test_synthesize_uses_gemini_and_includes_sources() -> None:
    fake_contexts = [
        {
            "text": "Sarvam AI announced a new funding round of $50M.",
            "metadata": {"title": "Sarvam Fundraise", "url": "https://example.com/sarvam"},
            "score": 0.85,
        }
    ]
    plan = plan_query("How much did Sarvam raise?")

    with patch("query.generate_with_gemini", return_value="Sarvam AI raised $50M in funding."):
        result = synthesize(
            "How much did Sarvam raise?",
            fake_contexts,
            plan,
            model="gemini-3.8-flash",
            api_key="test-api-key",
        )
        assert "Sarvam AI raised $50M in funding." in result
        assert "Sources:\n- https://example.com/sarvam" in result


def test_synthesize_uses_extractive_fallback_without_api_key() -> None:
    fake_contexts = [
        {
            "text": "Sarvam AI announced a new funding round of $50M.",
            "metadata": {"title": "Sarvam Fundraise", "url": "https://example.com/sarvam"},
            "score": 0.85,
        }
    ]
    plan = plan_query("How much did Sarvam raise?")

    with patch("config.Settings.get_api_key", return_value=""):
        result = synthesize("How much did Sarvam raise?", fake_contexts, plan)
        assert "Matching corpus excerpts:" in result
        assert "Sarvam AI announced a new funding round" in result
        assert "Sources:\n- https://example.com/sarvam" in result


def test_repeated_investors_are_calculated_from_structured_funding_data() -> None:
    first = ArticleRecord.model_validate(
        {
            "id": "first-record",
            "url": "https://example.com/first",
            "title": "First funding round",
            "published_at": "2026-10-01T00:00:00Z",
            "body": "First article body.",
            "summary": "First summary.",
            "companies": [],
            "people": [],
            "locations": [],
            "category": "funding",
            "funding": {"investors": ["Alpha Ventures"]},
        }
    )
    second = first.model_copy(
        update={"id": "second-record", "url": "https://example.com/second"}
    )
    plan = plan_query("Which investors are appearing in >1 funding round?")

    result = repeated_investors_answer([first, second], plan)

    assert "Alpha Ventures — 2 records" in result
    assert "https://example.com/first" in result

