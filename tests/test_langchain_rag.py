from unittest.mock import MagicMock, patch

from src.news_rag.langchain_rag import (
    build_rag_chain,
    format_context_chunks,
    get_llm,
    langchain_generate_answer,
)
from src.news_rag.planning import plan_query


def test_format_context_chunks() -> None:
    contexts = [
        {
            "text": "Infosys announced AI revenue growth.",
            "metadata": {"title": "Infosys AI", "published_at": "2026-10-14T10:00:00Z"},
        }
    ]
    formatted = format_context_chunks(contexts)
    assert "[Source 1] Title: Infosys AI | Published: 2026-10-14T10:00:00Z" in formatted
    assert "Infosys announced AI revenue growth." in formatted


def test_get_llm_instantiates_gemini() -> None:
    llm = get_llm(model="gemini-3.8-flash", api_key="fake-test-key")
    assert llm.model == "gemini-3.8-flash"
    assert llm.temperature == 0.0


def test_build_rag_chain_returns_runnable() -> None:
    plan = plan_query("What funding rounds occurred in the last 7 days?")
    chain = build_rag_chain(model="gemini-3.8-flash", api_key="fake-test-key", plan=plan)
    assert hasattr(chain, "invoke")


def test_langchain_generate_answer_invokes_chain() -> None:
    fake_contexts = [
        {
            "text": "Tracxn report notes funding dropped 41%.",
            "metadata": {"title": "Fintech Report", "url": "https://example.com/fintech"},
        }
    ]
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Fintech funding dropped 41% according to Tracxn."

    with patch("src.news_rag.langchain_rag.build_rag_chain", return_value=mock_chain):
        ans = langchain_generate_answer(
            question="What happened to fintech funding?",
            contexts=fake_contexts,
            model="gemini-3.8-flash",
            api_key="fake-test-key",
        )
        assert ans == "Fintech funding dropped 41% according to Tracxn."
        assert mock_chain.invoke.called

