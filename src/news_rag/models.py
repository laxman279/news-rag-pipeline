from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class Funding(BaseModel):
    amount: str | None = None
    currency: str | None = None
    round: str | None = None
    investors: list[str] = Field(default_factory=list)


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
