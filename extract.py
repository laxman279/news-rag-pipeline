from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from pydantic import ValidationError

from config import settings
from src.news_rag.io import read_json, write_json
from src.news_rag.models import ArticleRecord, Funding
from src.news_rag.url_utils import looks_like_et_tech_article

try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None


CATEGORY_RULES = {
    "funding": ("funding", "raised", "raises", "seed", "series a", "investor"),
    "product launch": ("launch", "launched", "unveils", "introduces"),
    "policy": ("policy", "regulation", "government", "rule", "rbi", "meity"),
    "M&A": ("acquire", "acquisition", "merger", "stake"),
    "layoffs": ("layoff", "job cuts", "fired", "workforce reduction"),
    "AI": (" ai ", "artificial intelligence", "genai", "generative ai"),
    "earnings": ("earnings", "profit", "loss", "revenue"),
}

LOCATION_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "delhi": "Delhi",
    "mumbai": "Mumbai",
    "gurugram": "Gurugram",
    "gurgaon": "Gurugram",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
    "chennai": "Chennai",
    "india": "India",
    "singapore": "Singapore",
}


def _text_or_empty(soup: BeautifulSoup, selector: str, attr: str | None = None) -> str:
    node = soup.select_one(selector)
    if not node:
        return ""
    if attr:
        return node.get(attr, "").strip()
    return node.get_text(" ", strip=True)


def _json_ld_items(payload: Any) -> list[dict[str, Any]]:
    """Return every object in a JSON-LD document, including graph members."""
    if isinstance(payload, list):
        return [item for value in payload for item in _json_ld_items(value)]
    if not isinstance(payload, dict):
        return []

    items = [payload]
    for key in ("@graph", "mainEntity", "mainEntityOfPage"):
        if key in payload:
            items.extend(_json_ld_items(payload[key]))
    return items


def parse_json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            payload = json.loads(script.string or "{}")
        except json.JSONDecodeError:
            continue
        items = _json_ld_items(payload)
        for item in items:
            article_types = item.get("@type", "")
            type_names = article_types if isinstance(article_types, list) else [article_types]
            if isinstance(item, dict) and any(
                str(name).lower() in {"newsarticle", "article"} for name in type_names
            ):
                return item
    return {}


def clean_body(html: str) -> str:
    if trafilatura:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            output_format="txt",
        )
        if extracted:
            return re.sub(r"\s+", " ", extracted).strip()
    soup = BeautifulSoup(html, "html.parser")
    for selector in ("script", "style", "nav", "footer", "aside", "iframe"):
        for node in soup.select(selector):
            node.decompose()
    article = soup.find("article") or soup.find("main") or soup.body or soup
    return re.sub(r"\s+", " ", article.get_text(" ", strip=True)).strip()


def normalize_datetime(value: str | None) -> str:
    """Normalize a publisher-supplied timestamp, never substituting scrape time."""
    if not value:
        raise ValueError("Article has no publisher-supplied publication timestamp")
    value = value.strip()
    normalized = value.replace("Z", "+00:00")
    # Some Economic Times JSON-LD timestamps use a one-digit hour (for example T6:00:00).
    normalized = re.sub(r"T(\d):", r"T0\1:", normalized)
    # ET also emits midnight as hour 24, which is the next calendar day in ISO 8601.
    rolls_to_next_day = bool(re.search(r"T24:", normalized))
    normalized = normalized.replace("T24:", "T00:")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            for date_format in ("%A, %d %B, %Y, %I:%M:%S %p", "%d %B %Y"):
                try:
                    dt = datetime.strptime(value, date_format)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unparseable publisher publication timestamp: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if rolls_to_next_day:
        dt += timedelta(days=1)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def sentence_summary(body: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", body)
    selected = [sentence.strip() for sentence in sentences if sentence.strip()][:3]
    return " ".join(selected) or body[:280]


def category_for(text: str) -> str:
    lowered = f" {text.lower()} "
    for category, keywords in CATEGORY_RULES.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return "general"


def extract_named_items(text: str) -> tuple[list[str], list[str], list[str]]:
    locations = []
    lowered = text.lower()
    for needle, normalized in LOCATION_ALIASES.items():
        if re.search(rf"\b{re.escape(needle)}\b", lowered) and normalized not in locations:
            locations.append(normalized)

    capitalized_phrases = re.findall(
        r"\b(?:[A-Z][A-Za-z&.-]+)(?:\s+[A-Z][A-Za-z&.-]+){0,3}\b", text
    )
    stop = {"The", "A", "An", "India", "Economic Times", "ETtech", "ET Tech"}
    companies = []
    people = []
    for phrase in capitalized_phrases[:80]:
        cleaned = phrase.strip()
        if cleaned in stop or cleaned in locations:
            continue
        if any(token in cleaned for token in ("Ltd", "Limited", "Inc", "AI", "Tech")):
            if cleaned not in companies:
                companies.append(cleaned)
        elif len(cleaned.split()) >= 2 and cleaned not in people:
            people.append(cleaned)
    return companies[:15], people[:15], locations


def funding_for(text: str, category: str) -> Funding | None:
    if category != "funding":
        return None
    amount_match = re.search(
        r"(?P<currency>\$|Rs\.?|INR|USD)\s?(?P<amount>[\d,.]+)\s?(?P<unit>crore|million|billion|mn|m)?",
        text,
        flags=re.IGNORECASE,
    )
    round_match = re.search(r"\b(seed|pre-seed|series [a-z]|debt)\b", text, flags=re.IGNORECASE)
    investors_match = re.search(r"(?:from|led by)\s+([^.;]+)", text, flags=re.IGNORECASE)
    investors = []
    if investors_match:
        investors = [part.strip() for part in re.split(r",| and ", investors_match.group(1))[:6]]
    return Funding(
        amount=" ".join(part for part in [
            amount_match.group("amount") if amount_match else None,
            amount_match.group("unit") if amount_match else None,
        ] if part),
        currency=amount_match.group("currency") if amount_match else None,
        round=round_match.group(1).title() if round_match else None,
        investors=[investor for investor in investors if investor],
    )


_GEMINI_CHAIN = None


def _get_gemini_chain(api_key: str | None = None):
    global _GEMINI_CHAIN
    if _GEMINI_CHAIN is not None:
        return _GEMINI_CHAIN
    active_key = settings.get_api_key(api_key)
    if not active_key:
        return None
    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=active_key,
            temperature=0.0,
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an information extraction assistant. Extract JSON for an Indian tech news article. "
                        "Return ONLY valid JSON with keys exactly: summary, companies, people, locations, category, funding. "
                        "Category must be one of funding, product launch, policy, M&A, layoffs, AI, earnings, general."
                    ),
                ),
                ("human", "Article text:\n{article_text}"),
            ]
        )
        _GEMINI_CHAIN = prompt | llm | JsonOutputParser()
        return _GEMINI_CHAIN
    except Exception:
        return None


def ask_gemini(
    html_text: str,
    heuristic_payload: dict[str, Any],
    api_key: str | None = None,
) -> dict[str, Any]:
    """Enrich heuristic extraction using LangChain ChatGoogleGenerativeAI JSON chain."""
    chain = _get_gemini_chain(api_key=api_key)
    if chain is None:
        return heuristic_payload
    try:
        parsed = chain.invoke({"article_text": html_text[:6000]})
    except Exception:
        return heuristic_payload

    merged = dict(heuristic_payload)
    for key in ("summary", "companies", "people", "locations", "category", "funding"):
        value = parsed.get(key)
        if value:
            merged[key] = value
    return merged


# Backward-compatibility alias
ask_ollama = ask_gemini


def extract_article(
    path: Path,
    manifest_row: dict[str, Any],
    api_key: str | None = None,
    use_llm: bool = False,
) -> ArticleRecord:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    ld = parse_json_ld(soup)
    body = clean_body(html)
    title = (
        ld.get("headline")
        or _text_or_empty(soup, 'meta[property="og:title"]', "content")
        or _text_or_empty(soup, "title")
        or "Untitled"
    )
    author = ld.get("author") or _text_or_empty(soup, 'meta[name="author"]', "content")
    if isinstance(author, list):
        author = [item.get("name", str(item)) if isinstance(item, dict) else str(item) for item in author]
    elif isinstance(author, dict):
        author = author.get("name", "")
    published = (
        ld.get("datePublished")
        or _text_or_empty(soup, 'meta[property="article:published_time"]', "content")
        or _text_or_empty(soup, 'meta[name="datePublished"]', "content")
        or _text_or_empty(soup, 'meta[itemprop="datePublished"]', "content")
    )
    companies, people, locations = extract_named_items(f"{title}. {body}")
    category = category_for(f"{title}. {body}")
    heuristic_payload = {
        "summary": sentence_summary(body),
        "companies": companies,
        "people": people,
        "locations": locations,
        "category": category,
        "funding": funding_for(body, category).model_dump() if funding_for(body, category) else None,
    }
    enriched = (
        ask_gemini(body, heuristic_payload, api_key=api_key)
        if use_llm
        else heuristic_payload
    )
    return ArticleRecord(
        id=str(manifest_row["id"]),
        url=str(manifest_row["url"]),
        title=str(title).strip(),
        author=author or "",
        published_at=normalize_datetime(published),
        body=body,
        summary=str(enriched["summary"]),
        companies=list(enriched.get("companies") or []),
        people=list(enriched.get("people") or []),
        locations=list(enriched.get("locations") or []),
        category=str(enriched.get("category") or "general"),
        funding=enriched.get("funding"),
    )


def run_extract(
    force: bool = False,
    api_key: str | None = None,
    use_llm: bool = False,
) -> None:
    settings.ensure_dirs()
    manifest_path = settings.raw_dir / "manifest.json"
    manifest = {row["id"]: row for row in read_json(manifest_path)} if manifest_path.exists() else {}
    error_path = settings.records_dir / "extraction_errors.log"
    if force:
        error_path.write_text("", encoding="utf-8")

    html_files = sorted(settings.raw_dir.glob("*.html"))
    total = len(html_files)
    extracted = 0
    skipped = 0

    mode_label = "with Gemini refinement" if use_llm else "fast deterministic mode"
    print(f"Starting extraction for {total} raw articles ({mode_label})...")

    for idx, html_path in enumerate(html_files, start=1):
        article_id = html_path.stem
        output_path = settings.records_dir / f"{article_id}.json"
        row = manifest.get(article_id)
        if not row:
            continue
        if not str(row.get("url", "")) or not looks_like_et_tech_article(str(row["url"])):
            if output_path.exists():
                output_path.unlink()
            continue
        if output_path.exists() and not force:
            skipped += 1
            continue
        try:
            record = extract_article(html_path, row, api_key=api_key, use_llm=use_llm)
            write_json(output_path, record.model_dump(mode="json"))
            extracted += 1
            if extracted % 10 == 0 or extracted == 1:
                print(f"  [{idx}/{total}] Extracted: {record.title[:50]}...", flush=True)
        except (ValidationError, ValueError, OSError) as exc:
            if force and output_path.exists():
                output_path.unlink()
            with error_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{article_id}: {exc}\n")

    print(
        f"Extraction complete: {extracted} records extracted, "
        f"{skipped} existing records preserved in {settings.records_dir}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract validated article records.")
    parser.add_argument("--force", action="store_true", help="Re-extract existing records.")
    parser.add_argument(
        "--use-gemini",
        action="store_true",
        help="Use Gemini LLM to refine extracted fields (slower due to remote API rate limits).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API Key override (enables Gemini refinement if provided).",
    )
    args = parser.parse_args()
    use_llm = args.use_gemini or bool(args.api_key)
    run_extract(force=args.force, api_key=args.api_key, use_llm=use_llm)


if __name__ == "__main__":
    main()

