from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse


TRACKING_PREFIXES = ("utm_",)
TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid"}
EXCLUDED_TECH_PATHS = ("/tech/newsletters/",)


def canonicalize_url(url: str, base_url: str | None = None) -> str:
    joined = urljoin(base_url, url) if base_url else url
    parsed = urlparse(joined)
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") or "/"
    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=False)
        if key not in TRACKING_KEYS and not key.startswith(TRACKING_PREFIXES)
    ]
    query = urlencode(sorted(query_pairs))
    return urlunparse((scheme, netloc, path, "", query, ""))


def article_id(url: str, length: int = 16) -> str:
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:length]


def looks_like_et_tech_article(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    return (
        "economictimes.indiatimes.com" in parsed.netloc.lower()
        and "/tech/" in path
        and not any(excluded_path in path for excluded_path in EXCLUDED_TECH_PATHS)
        and path.endswith(".cms")
        and ("articleshow" in path or "/news/" in path or "/startup" in path)
    )
