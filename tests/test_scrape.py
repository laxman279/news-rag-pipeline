from pathlib import Path
from types import SimpleNamespace
from urllib import robotparser

import pytest
import requests

import scrape


def test_allowed_fails_closed_when_robots_cannot_be_loaded() -> None:
    assert scrape.allowed(None, "https://example.com/article") is False


def test_load_robots_returns_none_after_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scrape, "fetch", lambda *_args: (_ for _ in ()).throw(requests.ConnectionError()))

    assert scrape.load_robots(object()) is None


def test_scrape_rejects_an_insufficient_discovery_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir = tmp_path / "raw"
    fake_settings = SimpleNamespace(
        raw_dir=raw_dir,
        min_articles=2,
        ensure_dirs=lambda: raw_dir.mkdir(parents=True, exist_ok=True),
    )
    robots = robotparser.RobotFileParser()
    robots.parse(["User-agent: *", "Allow: /"])
    monkeypatch.setattr(scrape, "settings", fake_settings)
    monkeypatch.setattr(scrape, "build_session", lambda: object())
    monkeypatch.setattr(scrape, "load_robots", lambda _session: robots)
    monkeypatch.setattr(scrape, "discover_urls", lambda _session, _robots: ["https://example.com/one"])

    with pytest.raises(RuntimeError, match="expected at least 2"):
        scrape.scrape()

    assert "INSUFFICIENT_CORPUS" in (raw_dir / "failed_urls.log").read_text(encoding="utf-8")
