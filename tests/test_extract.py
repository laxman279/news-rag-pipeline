from bs4 import BeautifulSoup
import pytest

from extract import normalize_datetime, parse_json_ld


def test_parse_json_ld_reads_articles_inside_a_graph() -> None:
    soup = BeautifulSoup(
        '''<script type="application/ld+json">
        {"@graph": [{"@type": ["WebPage"]}, {"@type": ["NewsArticle"],
        "headline": "Example", "datePublished": "2026-02-02T6:00:00+05:30"}]}
        </script>''',
        "html.parser",
    )

    article = parse_json_ld(soup)

    assert article["headline"] == "Example"


def test_normalize_datetime_handles_one_digit_hour_without_inventing_a_date() -> None:
    assert normalize_datetime("2026-02-02T6:00:00+05:30") == "2026-02-02T00:30:00Z"
    assert normalize_datetime("2026-02-02T24:00:00+05:30") == "2026-02-02T18:30:00Z"
    with pytest.raises(ValueError, match="publisher-supplied"):
        normalize_datetime(None)
