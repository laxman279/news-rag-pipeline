from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from urllib3.util.retry import Retry

from config import settings
from src.news_rag.tracing import traceable
from src.news_rag.url_utils import article_id, canonicalize_url, looks_like_et_tech_article


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": settings.user_agent})
    retry_config = Retry(
        total=3,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry_config)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def load_robots(session: requests.Session) -> robotparser.RobotFileParser | None:
    """Fetch robots.txt through the configured session and fail closed if unavailable."""
    parsed = urlparse(settings.target_site_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    robots = robotparser.RobotFileParser()
    robots.set_url(robots_url)
    try:
        response = fetch(session, robots_url)
    except requests.RequestException as exc:
        print(f"Could not fetch robots.txt; no pages will be requested: {exc}")
        return None
    robots.parse(response.text.splitlines())
    return robots


def allowed(robots: robotparser.RobotFileParser | None, url: str) -> bool:
    return robots is not None and robots.can_fetch(settings.user_agent, url)


@traceable(name="scrape_fetch_url", run_type="tool", tags=["scraping"])
@retry(
    retry=retry_if_exception_type(requests.RequestException),
    wait=wait_exponential(multiplier=1, min=1, max=12),
    stop=stop_after_attempt(3),
    reraise=True,
)
def fetch(session: requests.Session, url: str) -> requests.Response:
    response = session.get(url, timeout=settings.request_timeout_seconds)
    response.raise_for_status()
    return response


def extract_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for anchor in soup.find_all("a", href=True):
        canonical = canonicalize_url(anchor["href"], base_url)
        if looks_like_et_tech_article(canonical):
            links.add(canonical)
    return links


def discover_from_rss(session: requests.Session) -> set[str]:
    response = fetch(session, settings.target_rss_url)
    soup = BeautifulSoup(response.text, "xml")
    links = set()
    for item in soup.find_all("item"):
        link = item.find("link")
        if link and link.text:
            canonical = canonicalize_url(link.text)
            if looks_like_et_tech_article(canonical):
                links.add(canonical)
    return links


def category_pages() -> list[str]:
    base = settings.target_site_url.rstrip("/")
    return [
        base,
        f"{base}/startups",
        f"{base}/funding",
        f"{base}/technology",
        f"{base}/artificial-intelligence",
        f"{base}/internet",
    ] + [f"{base}/{page}" for page in range(2, 8)]


@traceable(name="scrape_discover_urls", run_type="tool", tags=["scraping"])
def discover_urls(session: requests.Session, robots: robotparser.RobotFileParser | None) -> list[str]:
    urls = discover_from_rss(session) if allowed(robots, settings.target_rss_url) else set()
    for page in category_pages():
        if len(urls) >= settings.max_articles:
            break
        if not allowed(robots, page):
            continue
        try:
            response = fetch(session, page)
        except requests.RequestException:
            continue
        urls.update(extract_links(response.text, page))
        time.sleep(settings.request_delay_seconds)
    return sorted(urls)[: settings.max_articles]


def load_manifest(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")


@traceable(name="scrape_pipeline", run_type="chain", tags=["scraping"])
def scrape(limit: int | None = None) -> None:
    settings.ensure_dirs()
    session = build_session()
    robots = load_robots(session)
    failed_path = settings.raw_dir / "failed_urls.log"
    if robots is None:
        with failed_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} ROBOTS_UNAVAILABLE\n")
        return
    urls = discover_urls(session, robots)
    if limit:
        urls = urls[:limit]
    if limit is None and len(urls) < settings.min_articles:
        message = (
            f"Discovered only {len(urls)} allowed article URLs; expected at least "
            f"{settings.min_articles}. No fetches were performed."
        )
        with failed_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(timezone.utc).isoformat()} INSUFFICIENT_CORPUS {message}\n")
        raise RuntimeError(message)

    manifest_path = settings.raw_dir / "manifest.json"
    manifest = {row["id"]: row for row in load_manifest(manifest_path)}

    for url in urls:
        canonical = canonicalize_url(url)
        aid = article_id(canonical)
        output_path = settings.raw_dir / f"{aid}.html"
        if output_path.exists():
            manifest.setdefault(
                aid,
                {
                    "id": aid,
                    "url": canonical,
                    "fetched_at": datetime.fromtimestamp(
                        output_path.stat().st_mtime, timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    "http_status": 200,
                },
            )
            continue
        if not allowed(robots, canonical):
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()} ROBOTS {canonical}\n")
            continue
        try:
            response = fetch(session, canonical)
            output_path.write_text(response.text, encoding=response.encoding or "utf-8")
            manifest[aid] = {
                "id": aid,
                "url": canonical,
                "fetched_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "http_status": response.status_code,
            }
        except requests.RequestException as exc:
            with failed_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{datetime.now(timezone.utc).isoformat()} ERROR {canonical} {exc}\n")
        time.sleep(settings.request_delay_seconds)

    write_manifest(manifest_path, sorted(manifest.values(), key=lambda row: str(row["id"])))
    print(f"Discovered {len(urls)} URLs; manifest has {len(manifest)} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape Economic Times Tech HTML.")
    parser.add_argument("--limit", type=int, help="Optional smaller fetch count for smoke tests.")
    args = parser.parse_args()
    scrape(limit=args.limit)


if __name__ == "__main__":
    main()
