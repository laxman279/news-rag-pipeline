from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import re

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


_COMPANY_LEGAL_SUFFIX = re.compile(
    r"(?:[,\s]+)(?:pvt\.?\s+)?(?:private\s+)?(?:limited|ltd\.?|incorporated|inc\.?|corporation|corp\.?|company|co\.?|llc|plc|lp|llp|gmbh|ag|sa|nv|pte\.?|有限責任公司)+\.?$",
    re.IGNORECASE,
)
_COMPANY_NOISE = re.compile(
    r"\b(?:latest news|news updates?|subscribe now|read more|click here|funding round|ipo|synopsis|the economic times|ettech)\b",
    re.IGNORECASE,
)
_NON_COMPANY_VALUES = {
    "ai", "artificial intelligence", "genai", "fintech", "deeptech", "startup",
    "startups", "company", "companies", "the company", "the firm", "the group",
    "news", "technology", "technology news", "funding", "ipo", "product launch",
    "retail", "consumer packaged goods", "cpg", "healthcare", "banking",
    "financial services", "insurance", "financial services and insurance",
    "telecommunications", "telecom", "automotive", "e-commerce", "ecommerce",
    "logistics", "real estate", "manufacturing", "education",
    "return on investment", "roi", "revenue", "profit", "ebitda", "cagr",
    "market cap", "valuation", "success", "trust", "data", "scale",
    "intelligence", "personalisation", "personalization", "leadership",
    "techleadership", "future of commerce", "enterprise", "enterprises",
    "et spotlight", "et spotlight team", "spotlight", "spotlight team",
    "agent", "agents", "ai agent", "ai agents",
}


def normalize_company_name(value: str) -> str | None:
    """Clean obvious extraction noise and normalize legal suffix variants."""
    if not isinstance(value, str):
        return None
    name = re.sub(r"\s+", " ", value).strip(" \t\r\n,;:|\"'`“”‘’()[]{}#")
    name = re.sub(r"^(?:the|a|an|many|several|all|various|most|some)\s+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s*[-–—]\s*(?:backed|funded|led|owned|driven|focused|centric|based)\b.*$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"\s+(?:latest news|news updates?|updates|funding round|fundraise|funding|synopsis|now playing)\b.*$", "", name, flags=re.IGNORECASE)
    name = name.strip(" .,:;-/#")
    if not name or any(char in name for char in (":", ";", ",", "|", "#", "@")) or re.search(r"\.[a-z]{2,}\b", name, flags=re.IGNORECASE):
        return None
    key = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    if not name or key in _NON_COMPANY_VALUES or _COMPANY_NOISE.search(name):
        return None
    if len(name) < 2 or len(name.split()) > 7 or re.search(r"[.!?].+", name):
        return None
    # A legal suffix is not part of the canonical display name; this also makes
    # variants such as "Google" and "Google LLC" collapse to one entity.
    name = _COMPANY_LEGAL_SUFFIX.sub("", name).strip(" .,:;-/#")
    key = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
    if not name or key in _NON_COMPANY_VALUES:
        return None
    return name


def deduplicate_companies(values: object) -> list[str]:
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, (list, tuple, set)):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        name = normalize_company_name(value) if isinstance(value, str) else None
        if not name:
            continue
        key = re.sub(r"[^a-z0-9]+", " ", name.casefold()).strip()
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


class Funding(BaseModel):
    amount: str | None = None
    currency: str | None = None
    round: str | None = None
    investors: list[str] = Field(default_factory=list)
    startup: str | None = None


class ArticleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=8)
    url: HttpUrl
    title: str = Field(min_length=1)
    author: str | list[str] = ""
    published_at: str
    body: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    companies: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    category: str
    categories: list[str] = Field(default_factory=list)
    funding: Funding | None = None

    @field_validator("companies", mode="before")
    @classmethod
    def validate_companies(cls, value: Any) -> list[str]:
        return deduplicate_companies(value)

    @field_validator("published_at")
    @classmethod
    def validate_datetime_utc(cls, value: str) -> str:
        normalized = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        aliases = {
            "m&a": "M&A",
            "merger": "M&A",
            "acquisition": "M&A",
            "artificial intelligence": "AI",
            "ai": "AI",
        }
        normalized = value.strip()
        normalized = aliases.get(normalized.lower(), normalized)
        return normalized or "general"

    @field_validator("categories", mode="before")
    @classmethod
    def validate_categories(cls, value: Any) -> list[str]:
        if not value:
            return []
        if isinstance(value, str):
            value = [value]
        aliases = {
            "m&a": "M&A",
            "merger": "M&A",
            "acquisition": "M&A",
            "artificial intelligence": "AI",
            "ai": "AI",
        }
        cleaned: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                raw = item.strip()
                norm = aliases.get(raw.lower(), raw)
                if norm not in cleaned:
                    cleaned.append(norm)
        return cleaned

    def metadata(self) -> dict[str, Any]:
        dt = datetime.fromisoformat(self.published_at.replace("Z", "+00:00"))
        cats = self.categories if self.categories else [self.category]
        return {
            "article_id": self.id,
            "url": str(self.url),
            "title": self.title,
            "published_at": self.published_at,
            "published_epoch": int(dt.timestamp()),
            "published_yyyymmdd": int(dt.strftime("%Y%m%d")),
            "category": self.category,
            "categories": ", ".join(cats),
            "companies": ", ".join(self.companies),
            "locations": ", ".join(self.locations),
            "has_funding": self.funding is not None,
        }
