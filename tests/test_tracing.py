import os
from unittest.mock import patch

from config import Settings
from src.news_rag.tracing import traceable


def test_traceable_preserves_function_behavior() -> None:
    @traceable(name="test_addition", run_type="tool", tags=["test"])
    def add(a: int, b: int) -> int:
        return a + b

    assert add(3, 4) == 7


def test_setup_langsmith_configures_env_vars() -> None:
    settings = Settings(
        langsmith_api_key="lsv2_pt_testkey_12345",
        langsmith_project="test-project",
        langsmith_tracing=True,
    )

    clean_env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("LANGSMITH_", "LANGCHAIN_"))
    }
    with patch.dict(os.environ, clean_env, clear=True):
        settings.setup_langsmith()
        assert os.environ.get("LANGSMITH_TRACING") == "true"
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGSMITH_API_KEY") == "lsv2_pt_testkey_12345"
        assert os.environ.get("LANGSMITH_PROJECT") == "test-project"


def test_setup_langsmith_noop_when_key_is_missing() -> None:
    settings = Settings(
        langsmith_api_key="",
        langsmith_project="test-project",
        langsmith_tracing=False,
    )

    with patch.dict(os.environ, {}, clear=True):
        settings.setup_langsmith()
        assert "LANGSMITH_API_KEY" not in os.environ
