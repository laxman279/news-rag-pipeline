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


def test_extract_named_items_gliner_isolates_names_and_companies() -> None:
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


def test_extract_named_items_gliner_filters_business_metrics_and_sectors() -> None:
    from extract import extract_named_items

    sample_text = (
        "ADA CEO Srinivas Gattamneni stated that Return on Investment and Healthcare solutions "
        "rely heavily on Trust and Data. #TechLeadership #ADA"
    )
    companies, people, locations = extract_named_items(sample_text)

    assert "ADA" in companies
    assert "Srinivas Gattamneni" in people
    assert "Return on Investment" not in companies
    assert "Healthcare" not in companies
    assert "Trust" not in companies
    assert "TechLeadership" not in companies


def test_extract_named_items_raises_when_gliner_fails(monkeypatch) -> None:
    import extract
    from extract import extract_named_items

    monkeypatch.setattr(extract, "get_gliner_model", lambda: None)
    with pytest.raises(RuntimeError, match="GLiNER not working"):
        extract_named_items("Some company raised funding in Bengaluru.")


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


def test_stage_1_isolate_funding_paragraphs_filters_irrelevant_text() -> None:
    from extract import isolate_funding_paragraphs

    text = (
        "The sun was shining outside and the team was having morning coffee in Bengaluru.\n\n"
        "Clueso, an AI video creation startup, has raised $10 million in a round led by Dharana Capital.\n\n"
        "Engineers at the firm deployed a new Kubernetes cluster on AWS last week.\n\n"
        "The round saw participation from new investors Founders Future and Neotribe Ventures.\n\n"
        "Weekly status meetings will now be scheduled on Thursdays."
    )
    paras = isolate_funding_paragraphs(text)

    # Stage 1: Only the 2 paragraphs mentioning funding keywords are isolated
    assert len(paras) == 2
    assert "Clueso, an AI video creation startup, has raised $10 million" in paras[0]
    assert "Founders Future and Neotribe Ventures" in paras[1]
    assert "morning coffee" not in " ".join(paras)
    assert "Kubernetes cluster" not in " ".join(paras)
    assert "Weekly status meetings" not in " ".join(paras)


def test_stage_1_isolate_funding_paragraphs_supports_html() -> None:
    from extract import isolate_funding_paragraphs

    html = (
        "<html><body><article>"
        "<p>The tech industry saw several macro policy changes yesterday.</p>"
        "<p>GenAI startup Sarvam AI announced a new funding round of $50M from venture capital investors.</p>"
        "<p>Office attendance was reported at 85 percent across all branches.</p>"
        "</article></body></html>"
    )
    paras = isolate_funding_paragraphs(html)

    assert len(paras) == 1
    assert "Sarvam AI announced a new funding round of $50M" in paras[0]
    assert "macro policy changes" not in paras[0]


def test_stage_2_extract_funding_gliner_semantic_labels() -> None:
    from extract import extract_funding

    text = (
        "General market trends were discussed earlier this morning.\n\n"
        "Clueso, an AI video creation startup, has raised $10 million in a round led by Dharana Capital.\n\n"
        "The round saw participation from new investors Founders Future and Neotribe Ventures."
    )
    funding = extract_funding(text)

    assert funding is not None
    # Stage 2 specific semantic labels:
    # "startup receiving funding"
    assert funding.startup == "Clueso"
    # "venture capital or angel investor"
    assert "Dharana Capital" in funding.investors
    assert any("Founders Future" in inv for inv in funding.investors)
    # "money amount raised"
    assert funding.currency == "$"
    assert "10 million" in str(funding.amount)


def test_stage_2_extract_funding_returns_none_when_no_funding() -> None:
    from extract import extract_funding

    text = "The weather was pleasant today and workers walked through the park."
    assert extract_funding(text) is None


def test_stage_2_extract_funding_raises_when_gliner_fails(monkeypatch) -> None:
    import extract
    from extract import extract_funding

    text = "Startup X raised $15 million in a Series A round from Sequoia."
    monkeypatch.setattr(extract, "get_gliner_model", lambda: None)
    with pytest.raises(RuntimeError, match="GLiNER not working"):
        extract_funding(text)


def test_funding_for_respects_category() -> None:
    from extract import funding_for

    text = "Sarvam AI raised $50 million in a round led by Lightspeed India."

    # When category is non-funding, returns None
    assert funding_for(text, category="semiconductors") is None
    assert funding_for(text, category=["semiconductors", "policy"]) is None

    # When category includes funding, extracts funding
    res = funding_for(text, category="funding")
    assert res is not None
    assert res.startup == "Sarvam AI"


def test_greedy_packing_accumulates_paragraphs_under_limit() -> None:
    from extract import pack_paragraphs_greedy

    # Paragraphs of length 300, 300, 300: 300 + 1 + 300 + 1 + 300 = 902 chars <= 1000
    p1 = "A" * 300
    p2 = "B" * 300
    p3 = "C" * 300
    # A fourth paragraph of 200 chars would make 902 + 1 + 200 = 1103 > 1000, so it must start a new chunk
    p4 = "D" * 200

    text = f"{p1}\n{p2}\n{p3}\n{p4}"
    chunks = pack_paragraphs_greedy(text, max_chars=1000)

    assert len(chunks) == 2
    assert chunks[0] == f"{p1}\n{p2}\n{p3}"
    assert len(chunks[0]) == 902
    assert chunks[1] == p4
    assert len(chunks[1]) == 200


def test_greedy_packing_splits_oversized_paragraphs() -> None:
    from extract import pack_paragraphs_greedy

    # A single paragraph of 1200+ chars with sentence break
    sentence1 = "A" * 600 + "."
    sentence2 = "B" * 600 + "."
    long_para = f"{sentence1} {sentence2}"
    chunks = pack_paragraphs_greedy(long_para, max_chars=1000)

    assert len(chunks) >= 2
    assert all(len(c) <= 1000 for c in chunks)


def test_batch_inference_extract_named_items_calls_batch_predict_entities() -> None:
    from unittest.mock import MagicMock
    from extract import extract_named_items

    mock_model = MagicMock()
    mock_model.batch_predict_entities.return_value = [
        [{"text": "Acme", "label": "company", "score": 0.9}],
        [{"text": "John Doe", "label": "person", "score": 0.95}],
    ]

    # Two paragraphs that each have 600 characters, so they cannot fit in one 1000-char chunk
    p1 = "Acme announced a major product line expansion.\n" * 15  # ~700 chars
    p2 = "John Doe joined the firm as leadership executive.\n" * 15 # ~750 chars
    text = f"{p1}\n{p2}"

    companies, people, locations = extract_named_items(text, model=mock_model)

    assert mock_model.batch_predict_entities.call_count == 1
    call_args, call_kwargs = mock_model.batch_predict_entities.call_args
    assert isinstance(call_args[0], list)
    assert len(call_args[0]) >= 2
    assert call_args[1] == ["company", "person", "location"]
    assert "threshold" in call_kwargs


def test_batch_inference_extract_funding_calls_batch_predict_entities() -> None:
    from unittest.mock import MagicMock
    from extract import extract_funding

    mock_model = MagicMock()
    mock_model.batch_predict_entities.return_value = [
        [
            {"text": "Acme", "label": "startup receiving funding", "score": 0.9},
            {"text": "Sequoia Capital", "label": "venture capital or angel investor", "score": 0.85},
            {"text": "Series A", "label": "funding round type", "score": 0.9},
            {"text": "$10 million", "label": "money amount raised", "score": 0.95},
        ]
    ]

    text = "Acme raised $10 million in Series A funding from Sequoia Capital."
    funding = extract_funding(text, model=mock_model)

    assert mock_model.batch_predict_entities.call_count == 1
    call_args, call_kwargs = mock_model.batch_predict_entities.call_args
    assert isinstance(call_args[0], list)
    assert len(call_args[0]) >= 1
    assert "threshold" in call_kwargs
    assert funding is not None
    assert funding.startup == "Acme"
    assert "Sequoia Capital" in funding.investors
    assert funding.currency == "$"
    assert "10 million" in str(funding.amount)



