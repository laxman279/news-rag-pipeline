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


def test_synthesize_filters_sources_to_only_cited_article() -> None:
    fake_contexts = [
        {
            "text": "NTT Data in final talks to build Anthropic's data center.",
            "metadata": {
                "title": "NTT Data in final talks to build Anthropic's first data centre in Japan",
                "url": "https://example.com/ntt",
            },
            "score": 0.90,
        },
        {
            "text": "Vishal Sikka letter to Infosys employees.",
            "metadata": {"title": "Vishal Sikka Letter", "url": "https://example.com/infosys"},
            "score": 0.40,
        },
        {
            "text": "AMD CEO Lisa Su talks chips.",
            "metadata": {"title": "AMD Chips", "url": "https://example.com/amd"},
            "score": 0.35,
        },
    ]
    plan = plan_query("list articles that talk about NTT.")
    llm_output = "The article that talks about NTT is: NTT Data in final talks (Source 1)."

    with patch("query.generate_with_gemini", return_value=llm_output):
        result = synthesize(
            "list articles that talk about NTT.",
            fake_contexts,
            plan,
            model="gemini-3.8-flash",
            api_key="test-api-key",
        )
        assert "https://example.com/ntt" in result
        assert "https://example.com/infosys" not in result
        assert "https://example.com/amd" not in result



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


def test_answer_outputs_notice_when_chroma_fails_and_lexical_retrieve_is_used(capsys) -> None:
    from query import answer

    fake_record = ArticleRecord.model_validate(
        {
            "id": "record-001",
            "url": "https://example.com/rec1",
            "title": "Fintech Report",
            "published_at": "2026-10-14T00:00:00Z",
            "body": "Fintech funding dropped 41 percent.",
            "summary": "Fintech funding dropped 41 percent.",
            "companies": [],
            "people": [],
            "locations": [],
            "category": "general",
            "funding": None,
        }
    )

    with patch("query.load_records", return_value=[fake_record]), \
         patch("query.chroma_retrieve", return_value=[]), \
         patch("config.Settings.get_api_key", return_value=""):
        res = answer("What happened to fintech funding?")
        captured = capsys.readouterr()
        assert "[Retrieval Warning]" in captured.out
        assert "ChromaDB vector search was unavailable" in res


