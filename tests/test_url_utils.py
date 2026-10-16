from src.news_rag.url_utils import article_id, canonicalize_url, looks_like_et_tech_article


def test_article_id_is_stable_for_tracking_params() -> None:
    first = "https://economictimes.indiatimes.com/tech/startups/example/articleshow/1.cms?utm_source=x"
    second = "https://economictimes.indiatimes.com/tech/startups/example/articleshow/1.cms"

    assert canonicalize_url(first) == canonicalize_url(second)
    assert article_id(first) == article_id(second)
    assert len(article_id(first)) == 16


def test_newsletter_is_not_treated_as_an_article() -> None:
    assert not looks_like_et_tech_article(
        "https://economictimes.indiatimes.com/tech/newsletters/morning-dispatch/example/articleshow/1.cms"
    )
