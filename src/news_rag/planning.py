from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


from typing import Iterable


CATEGORY_KEYWORDS = {
    "funding": ("funding", "funded", "raised", "round", "investor", "investment"),
    "product launch": ("launch", "launched", "unveiled", "released"),
    "policy": ("policy", "regulation", "rules", "government", "rbi", "meity"),
    "M&A": ("acquisition", "acquired", "merger", "stake", "bought"),
    "layoffs": ("layoff", "layoffs", "fired", "job cuts", "workforce"),
    "AI": (" ai ", "artificial intelligence", "genai", "generative ai"),
    "earnings": ("earnings", "profit", "loss", "revenue", "quarter"),
}

LOCATION_ALIASES = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "mumbai": "Mumbai",
    "bombay": "Mumbai",
    "gurugram": "Gurugram",
    "gurgaon": "Gurugram",
    "hyderabad": "Hyderabad",
    "pune": "Pune",
    "chennai": "Chennai",
    "madras": "Chennai",
    "kolkata": "Kolkata",
    "calcutta": "Kolkata",
    "noida": "Noida",
    "ahmedabad": "Ahmedabad",
    "india": "India",
    "us": "US",
    "usa": "USA",
    "singapore": "Singapore",
}


@dataclass(frozen=True)
class QueryPlan:
    question: str
    date_range: tuple[int | None, int | None] | None = None
    category: str | None = None
    company: str | None = None
    location: str | None = None
    title_match: str | None = None
    mentioned_terms: tuple[str, ...] = ()
    is_aggregation: bool = False
    list_all: bool = False
    repeated_investors: bool = False
    direct_records: bool = False
    limit: int | None = None
    low_relevance_threshold: float = 0.18
    notes: list[str] = field(default_factory=list)


def _yyyymmdd(dt: datetime) -> int:
    return int(dt.strftime("%Y%m%d"))


def plan_query(
    question: str,
    now: datetime | None = None,
    known_locations: Iterable[str] | None = None,
) -> QueryPlan:
    now = now or datetime.now(timezone.utc)
    lowered = f" {question.lower()} "
    category = None
    for name, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            category = name
            break

    # Build dynamic location candidate map:
    # 1. Base alias normalization table (e.g. Bangalore -> Bengaluru)
    # 2. Dynamic locations discovered from the extracted corpus records
    location_candidates = dict(LOCATION_ALIASES)
    if known_locations:
        for loc in known_locations:
            loc_clean = loc.strip()
            if loc_clean and loc_clean.lower() not in location_candidates:
                location_candidates[loc_clean.lower()] = loc_clean

    location = None
    # Sort by length descending to match multi-word names first ("new delhi" before "delhi")
    for candidate, normalized in sorted(
        location_candidates.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if re.search(rf"\b{re.escape(candidate)}\b", lowered):
            location = normalized
            break

    # Syntactic pattern fallback: "in <City>", "from <City>", "at <City>"
    if not location:
        prep_match = re.search(r"\b(?:in|at|from|near)\s+([A-Z][a-z]+)\b", question)
        if prep_match:
            candidate = prep_match.group(1)
            stop_words = {
                "The", "This", "Last", "Recent", "September", "October",
                "November", "December", "January", "February", "March", "April",
                "May", "June", "July", "August"
            }
            if candidate not in stop_words:
                location = location_candidates.get(candidate.lower(), candidate)

    date_range = None
    last_days_match = re.search(r"last\s+(\d+)\s+days?", lowered)
    if last_days_match:
        days = int(last_days_match.group(1))
        date_range = (_yyyymmdd(now - timedelta(days=days)), _yyyymmdd(now))

    title_match = None
    quoted = re.search(r'"([^"]+)"', question)
    if quoted:
        title_match = quoted.group(1)
        category = None

    company = None
    company_match = re.search(r"\b(?:company|startup)\s+([A-Z][\w&.-]+)", question)
    if company_match:
        company = company_match.group(1)

    mention_match = re.search(
        r"\bmention(?:ing|s|ed)?\s+(.+?)(?:\?|$)", question, flags=re.IGNORECASE
    )
    if mention_match and not location:
        raw_target = mention_match.group(1).strip().strip("\"'").rstrip(".,!?")
        if raw_target.lower() in location_candidates:
            location = location_candidates[raw_target.lower()]

    mentioned_terms: tuple[str, ...] = ()
    if mention_match and not location:
        raw_terms = mention_match.group(1).strip().strip("\"'")
        stop_terms = {"the", "a", "an", "article", "articles", "news", "tech"}
        terms = [
            term.lower()
            for term in re.findall(r"[A-Za-z0-9&.-]+", raw_terms)
            if len(term) > 2 and term.lower() not in stop_terms
        ]
        mentioned_terms = tuple(dict.fromkeys(terms))

    # Extract explicitly requested quantity (e.g. "tell 10 news", "top 5 startups", "list 7 articles")
    limit = None
    limit_match = re.search(
        r"\b(?:top|tell|give|list|show|latest)?\s*(\d{1,2})\s+(?:most|articles|news|startups|stories|events|rounds|companies)\b",
        lowered,
    )
    if not limit_match:
        limit_match = re.search(r"\b(?:top|latest)\s+(\d{1,2})\b", lowered)
    if limit_match:
        num = int(limit_match.group(1))
        if 1 <= num <= 30:
            limit = num

    list_all = bool(re.search(r"\b(?:list|show|give)\s+all\b|\ball articles\b", lowered))
    repeated_investors = bool(
        re.search(r"\binvestors?\b.*\bappear\w*\b.*(?:more than\s+1|>\s*1)", lowered)
    )
    is_aggregation = any(
        phrase in lowered
        for phrase in (
            "all articles",
            "list",
            "how many",
            "count",
            "appearing in >1",
            "appearing in more than 1",
            "investors appearing",
            "top ",
            "tell ",
            "give ",
            "show ",
        )
    ) or (limit is not None and limit > 1)
    # Direct records retrieval is only for explicit structured filters where vector similarity
    # is unnecessary (e.g. strict location matching, quoted title lookup, or specific mentioned terms).
    # Open-ended aggregation queries without a structured filter (e.g. "list articles that talk about UPI")
    # must go through Chroma vector search.
    has_structured_filter = (
        location is not None
        or title_match is not None
        or bool(mentioned_terms)
        or (list_all and category is not None)
    )
    direct_records = has_structured_filter

    return QueryPlan(
        question=question,
        date_range=date_range,
        category=category,
        company=company,
        location=location,
        title_match=title_match,
        mentioned_terms=mentioned_terms,
        is_aggregation=is_aggregation,
        list_all=list_all,
        repeated_investors=repeated_investors,
        direct_records=direct_records,
        limit=limit,
    )


def chroma_where(plan: QueryPlan) -> dict[str, object] | None:
    clauses: list[dict[str, object]] = []
    if plan.category:
        clauses.append({"category": plan.category})
    if plan.date_range:
        start, end = plan.date_range
        if start is not None:
            clauses.append({"published_yyyymmdd": {"$gte": start}})
        if end is not None:
            clauses.append({"published_yyyymmdd": {"$lte": end}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
