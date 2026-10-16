from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - makes error messages friendlier before install
    def load_dotenv() -> None:
        return None


load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    target_site_url: str = os.getenv(
        "TARGET_SITE_URL", "https://economictimes.indiatimes.com/tech"
    )
    target_rss_url: str = os.getenv(
        "TARGET_RSS_URL",
        "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
    )
    raw_dir: Path = Path(os.getenv("RAW_DIR", "data/raw"))
    records_dir: Path = Path(os.getenv("RECORDS_DIR", "data/records"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "data/chroma"))
    collection_name: str = os.getenv("COLLECTION_NAME", "et_tech_articles")
    request_delay_seconds: float = _float_env("REQUEST_DELAY_SECONDS", 1.5)
    request_timeout_seconds: float = _float_env("REQUEST_TIMEOUT_SECONDS", 20.0)
    max_articles: int = _int_env("MAX_ARTICLES", 60)
    min_articles: int = _int_env("MIN_ARTICLES", 55)
    user_agent: str = os.getenv(
        "USER_AGENT", "NewsRAGPipeline/1.0 (+https://example.com/contact)"
    )
    gemini_api_key: str = (
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    ).strip()
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash").strip()

    def get_api_key(self, override: str | None = None) -> str:
        """Return the active Gemini API key from explicit override, GEMINI_API_KEY, or GOOGLE_API_KEY."""
        if override and override.strip():
            return override.strip()
        return (
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or self.gemini_api_key or ""
        ).strip()

    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size_chars: int = _int_env(
        "CHUNK_SIZE_CHARS", _int_env("CHUNK_SIZE_TOKENS", 1200)
    )
    chunk_overlap_chars: int = _int_env(
        "CHUNK_OVERLAP_CHARS", _int_env("CHUNK_OVERLAP_TOKENS", 200)
    )
    chunk_size_tokens: int = _int_env(
        "CHUNK_SIZE_TOKENS", _int_env("CHUNK_SIZE_CHARS", 1200)
    )
    chunk_overlap_tokens: int = _int_env(
        "CHUNK_OVERLAP_TOKENS", _int_env("CHUNK_OVERLAP_CHARS", 200)
    )
    top_k: int = _int_env("TOP_K", 5)

    def ensure_dirs(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        Path("results").mkdir(parents=True, exist_ok=True)


settings = Settings()
