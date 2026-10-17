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


def test_extract_named_items_spacy_isolates_names_and_companies() -> None:
    from extract import extract_named_items

    sample_text = (
        "Tata Sons Chairman Natarajan Chandrasekaran met with Satya Nadella in Bengaluru. "
        "Tata Group and Bloomberg New Economy reported the event in May. "
        "The End. Latest News Updates on The Economic Times. Subscribe Now."
    )
    companies, people, locations = extract_named_items(sample_text)

    # Valid human names identified
    assert "Natarajan Chandrasekaran" in people
    assert "Satya Nadella" in people

    # Boilerplate / noise discarded
    assert "The End" not in people
    assert "Subscribe Now" not in people
    assert "May. The" not in people
    assert "The Economic Times" not in people

    # Companies properly classified
    assert any("Tata" in c for c in companies)
    assert "Bengaluru" in locations


def test_classify_article_categories_multi_domain() -> None:
    from extract import classify_article_categories

    title = "Sarvam plans big fundraise from venture capital investors"
    body = (
        "GenAI startup Sarvam AI is in talks to raise $50 million from venture capital investors "
        "to scale its generative artificial intelligence platform and LLM research."
    )
    url = "https://economictimes.indiatimes.com/tech/newsletters/morning-dispatch/sarvam-plans-big-fundraise/articleshow/134905940.cms"
    cats = classify_article_categories(title, body, url=url)

    # Should detect both AI and funding
    assert "funding" in cats
    assert "AI" in cats
    assert len(cats) >= 2


def test_classify_article_categories_out_of_taxonomy() -> None:
    from extract import classify_article_categories

    title = "India's first superconducting quantum computer to be commissioned next month"
    body = (
        "Scientists at the Department of Science and Technology announced that the 25-qubit "
        "superconducting quantum processor and quantum computing system will be operational soon."
    )
    url = "https://economictimes.indiatimes.com/tech/technology/quantum-computer-india/articleshow/67890.cms"
    cats = classify_article_categories(title, body, url=url)

    # Should detect quantum computing as an out-of-taxonomy category
    assert "quantum computing" in cats


def test_classify_article_categories_publisher_metadata_signals() -> None:
    from extract import classify_article_categories

    soup = BeautifulSoup(
        '''
        <html>
        <head>
            <meta name="keywords" content="crypto exchange hack, malware, cyberattack, digital payments" />
            <meta property="article:section" content="technology" />
        </head>
        </html>
        ''',
        "html.parser",
    )
    title = "Hackers breach payment gateway in major cyberattack"
    body = "Malware infected servers of the payment gateway, compromising user security credentials."
    url = "https://economictimes.indiatimes.com/tech/catalysts/breach-alert/articleshow/1234.cms"
    cats = classify_article_categories(title, body, url=url, soup=soup)

    assert "cybersecurity" in cats
    assert "fintech" in cats


def test_classify_article_categories_fallback_general() -> None:
    from extract import classify_article_categories

    title = "Random weekly musings on everyday office life"
    body = "The sun was shining outside and the coffee was warm on a regular Wednesday morning."
    url = "https://economictimes.indiatimes.com/tech/articleshow/000.cms"
    cats = classify_article_categories(title, body, url=url)

    assert cats == ["general"]


def test_clean_body_preserves_paragraph_boundaries_and_prunes_boilerplate() -> None:
    from extract import clean_body

    sample_html = (
        "<html><body><article>"
        "<p>Clueso raised $10 million led by Dharana Capital.</p>"
        "<p>The round saw participation from new investors Founders Future.</p>"
        "\n\nSynopsis Clueso raised $10 million.\n\n"
        "(Catch all the Technology News News, and Latest News Updates on The Economic Times.)\n"
        "Subscribe Now"
        "</article></body></html>"
    )
    body = clean_body(sample_html)

    # Paragraph boundaries are preserved
    assert "\n\n" in body
    assert "Clueso raised $10 million led by Dharana Capital." in body
    assert "The round saw participation from new investors Founders Future." in body

    # Trailing boilerplate is pruned
    assert "Catch all the Technology News" not in body
    assert "Subscribe Now" not in body


def test_extract_named_items_strips_relationship_modifiers_and_noise_suffixes() -> None:
    from extract import extract_named_items

    sample_text = (
        "Clueso, an AI video creation startup, has raised $10 million in a round led by Dharana Capital.\n\n"
        "YC-backed AI video creation startup Clueso announced the round with Founders Future and Neotribe Ventures.\n\n"
        "The company filed an IPO with the SEC. Catch all the Technology News News."
    )
    companies, people, locations = extract_named_items(sample_text)

    # Clean company names without cross-sentence merging
    assert "Dharana Capital" in companies
    assert "Founders Future" in companies
    assert "Neotribe Ventures" in companies
    assert "Clueso" in companies

    # Verified no merged entity like 'Dharana Capital YC'
    assert "Dharana Capital YC" not in companies
    assert "Business NewsTechFundingYC" not in companies

    # Relationship modifiers stripped
    assert not any(c.endswith("-backed") for c in companies)

    # Noise suffixes and generic terms pruned
    assert "the Technology News News" not in companies
    assert "Technology News News" not in companies
    assert "IPO" not in companies
    assert "AI" not in companies

