from datetime import datetime, timezone

from src.news_rag.planning import chroma_where, plan_query


def test_plan_extracts_location_and_aggregation() -> None:
    plan = plan_query("List all articles mentioning Bengaluru")

    assert plan.location == "Bengaluru"
    assert plan.mentioned_terms == ()
    assert plan.is_aggregation is True
    assert plan.direct_records is True


def test_plan_builds_metadata_filter_for_recent_funding() -> None:
    now = datetime(2026, 10, 14, tzinfo=timezone.utc)
    plan = plan_query("funding rounds in the last 7 days", now=now)

    assert plan.category == "funding"
    assert plan.date_range == (20261007, 20261014)
    assert chroma_where(plan) == {
        "$and": [
            {"category": "funding"},
            {"published_yyyymmdd": {"$gte": 20261007}},
            {"published_yyyymmdd": {"$lte": 20261014}},
        ]
    }


def test_weather_query_planning() -> None:
    plan = plan_query("What is the weather today?")
    assert plan.category is None
    assert plan.location is None
    assert plan.company is None


def test_title_match_does_not_infer_category_from_title_words() -> None:
    plan = plan_query('Find the article titled "Sarvam plans big fundraise"')

    assert plan.title_match == "Sarvam plans big fundraise"
    assert plan.category is None


def test_plan_extracts_mentioned_terms() -> None:
    plan = plan_query("List all articles mentioning AWS tech stack")

    assert plan.mentioned_terms == ("aws", "stack")
    assert plan.direct_records is True


def test_plan_handles_multi_sentence_query_with_formatting_instructions() -> None:
    plan = plan_query(
        "tell about articles which mention acquisition.tell 1 line about each article."
    )

    assert plan.mentioned_terms == ("acquisition",)
    assert plan.category == "M&A"
    assert plan.direct_records is True


def test_plan_resolves_location_aliases_and_prepositions() -> None:
    # Alias normalization
    plan_alias = plan_query("List startups in Bangalore")
    assert plan_alias.location == "Bengaluru"

    # Prepositional pattern extraction
    plan_prep = plan_query("Which startups in Noida raised funding?")
    assert plan_prep.location == "Noida"
    assert plan_prep.category == "funding"


def test_plan_supports_dynamic_corpus_locations() -> None:
    # Dynamic location passed from extracted corpus
    plan = plan_query("News from Munich", known_locations={"Munich", "Berlin"})
    assert plan.location == "Munich"


def test_plan_extracts_requested_limit_and_aggregation() -> None:
    plan = plan_query("tell 10 most important tech news from last 7 days")
    assert plan.limit == 10
    assert plan.is_aggregation is True
    assert plan.date_range is not None


def test_plan_identifies_exact_list_and_repeated_investor_aggregations() -> None:
    list_plan = plan_query("List all articles mentioning Bengaluru")
    investor_plan = plan_query("Which investors are appearing in >1 funding round?")

    assert list_plan.list_all is True
    assert investor_plan.repeated_investors is True
