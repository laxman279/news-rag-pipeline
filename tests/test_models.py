import pytest
from pydantic import ValidationError

from src.news_rag.models import ArticleRecord


def valid_payload() -> dict:
    return {
        "id": "1234567890abcdef",
        "url": "https://economictimes.indiatimes.com/tech/startups/example/articleshow/1.cms",
        "title": "Startup raises funding",
        "author": "",
        "published_at": "2026-10-14T01:00:00+05:30",
        "body": "A Bengaluru startup raised a seed round from investors.",
        "summary": "A Bengaluru startup raised a seed round.",
        "companies": ["Example Tech"],
        "people": [],
        "locations": ["Bengaluru"],
        "category": "funding",
        "funding": {
            "amount": "5 million",
            "currency": "USD",
            "round": "Seed",
            "investors": ["Alpha Ventures"],
        },
    }


def test_article_record_normalizes_utc_datetime() -> None:
    record = ArticleRecord.model_validate(valid_payload())

    assert record.published_at == "2026-10-13T19:30:00Z"
    assert record.category == "funding"


def test_article_record_rejects_unknown_fields() -> None:
    payload = valid_payload()
    payload["unexpected"] = "nope"

    with pytest.raises(ValidationError):
        ArticleRecord.model_validate(payload)
