from __future__ import annotations

import argparse
import re
from pathlib import Path

from config import settings
from src.news_rag.chunking import chunks_for_record
from src.news_rag.io import load_records
from src.news_rag.langchain_rag import langchain_chroma_retrieve, langchain_generate_answer
from src.news_rag.models import ArticleRecord
from src.news_rag.planning import QueryPlan, plan_query
from src.news_rag.tracing import traceable



def record_matches(record: ArticleRecord, plan: QueryPlan) -> bool:
    if plan.category:
        article_cats = {record.category.lower(), *(c.lower() for c in getattr(record, "categories", []))}
        if plan.category.lower() not in article_cats:
            return False
    if plan.location and plan.location.lower() not in {loc.lower() for loc in record.locations}:
        return False 
    if plan.company and plan.company.lower() not in {company.lower() for company in record.companies}:
        return False
    if plan.title_match and plan.title_match.lower() not in record.title.lower():
        return False
    if plan.mentioned_terms:
        haystack = " ".join(
            [
                record.title,
                record.summary,
                record.body,
                " ".join(record.companies),
                " ".join(record.people),
                " ".join(record.locations),
            ]
        ).lower()
        if not all(
            re.search(rf"\b{re.escape(term)}\b", haystack)
            for term in plan.mentioned_terms
        ):
            return False
    if plan.date_range:
        yyyymmdd = int(record.metadata()["published_yyyymmdd"])
        start, end = plan.date_range
        if start is not None and yyyymmdd < start:
            return False
        if end is not None and yyyymmdd > end:
            return False
    return True


def lexical_score(question: str, text: str) -> float:
    q_terms = {term for term in re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower())}
    if not q_terms:
        return 0.0
    t_terms = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", text.lower()))
    return len(q_terms & t_terms) / len(q_terms)


def chunk_matches_plan(text: str, metadata: dict[str, object], plan: QueryPlan) -> bool:
    text_lower = text.lower()
    if plan.location:
        loc = plan.location.lower()
        meta_locs = str(metadata.get("locations", "")).lower()
        aliases = {loc}
        if loc in {"bengaluru", "bangalore"}:
            aliases = {"bengaluru", "bangalore"}
        if not any(alias in text_lower or alias in meta_locs for alias in aliases):
            return False
    if plan.company:
        comp = plan.company.lower()
        meta_comps = str(metadata.get("companies", "")).lower()
        if comp not in text_lower and comp not in meta_comps:
            return False
    if plan.mentioned_terms:
        if not all(term in text_lower for term in plan.mentioned_terms):
            return False
    return True


@traceable(name="chroma_retrieval", run_type="retriever", tags=["querying"])
def chroma_retrieve(question: str, plan: QueryPlan) -> list[dict[str, object]]:
    """Retrieve top relevant chunks from ChromaDB using LangChain vector store."""
    try:
        return langchain_chroma_retrieve(question, plan, chunk_matches_fn=chunk_matches_plan)
    except Exception as exc:
        print(f"[ChromaDB Error] ChromaDB query failed: {exc}", flush=True)
        return []


@traceable(name="lexical_retrieval", run_type="retriever", tags=["querying", "fallback"])
def lexical_retrieve(question: str, records: list[ArticleRecord], plan: QueryPlan) -> list[dict[str, object]]:
    print(
        "[Retrieval Warning] ChromaDB is unavailable or returned an error / 0 vector matches. "
        "Falling back to lexical keyword retrieval.",
        flush=True,
    )
    rows = []
    target_count = plan.limit or (max(settings.top_k, 5) if plan.is_aggregation else settings.top_k)
    for record in records:
        if not record_matches(record, plan):
            continue
        for chunk in chunks_for_record(record, settings.chunk_size_tokens, settings.chunk_overlap_tokens):
            score = lexical_score(question, chunk.text)
            if score:
                rows.append({"text": chunk.text, "metadata": chunk.metadata, "score": score})
    return sorted(rows, key=lambda row: float(row["score"]), reverse=True)[:target_count]


@traceable(name="direct_metadata_retrieval", run_type="retriever", tags=["querying"])
def direct_record_retrieve(records: list[ArticleRecord], plan: QueryPlan) -> list[dict[str, object]]:
    """Return exact record matches for structured filters without semantic-ranking loss."""
    rows = [
        {"text": record.summary, "metadata": record.metadata(), "score": 1.0}
        for record in records
        if record_matches(record, plan)
    ]
    rows.sort(
        key=lambda row: int((row["metadata"] or {}).get("published_yyyymmdd", 0)),
        reverse=True,
    )
    if plan.list_all:
        return rows
    return rows[: plan.limit or max(settings.top_k, 5)]


def extractive_answer(contexts: list[dict[str, object]]) -> str:
    """Provide a deterministic, source-backed answer when Gemini is unavailable."""
    excerpts = []
    for ctx in contexts:
        metadata = ctx.get("metadata") or {}
        title = str(metadata.get("title", "Untitled"))
        published = str(metadata.get("published_at", ""))
        text = " ".join(str(ctx.get("text", "")).split())
        if len(text) > 500:
            text = text[:500].rsplit(" ", 1)[0] + "..."
        excerpts.append(f"- {title} ({published}): {text}")
    return "Matching corpus excerpts:\n" + "\n".join(excerpts)


@traceable(name="repeated_investors_analysis", run_type="tool", tags=["querying"])
def repeated_investors_answer(records: list[ArticleRecord], plan: QueryPlan) -> str:
    """Answer repeated-investor questions from validated funding metadata, not retrieval snippets."""
    investor_sources: dict[str, tuple[str, set[str]]] = {}
    for record in records:
        if not record_matches(record, plan) or not record.funding:
            continue
        for investor in record.funding.investors:
            display_name = investor.strip()
            if not display_name:
                continue
            key = display_name.casefold()
            saved_name, urls = investor_sources.get(key, (display_name, set()))
            urls.add(str(record.url))
            investor_sources[key] = (saved_name, urls)

    repeated = [entry for entry in investor_sources.values() if len(entry[1]) > 1]
    if not repeated:
        return "No investor appears in more than one validated funding record.\n\nSources: none"

    repeated.sort(key=lambda entry: (-len(entry[1]), entry[0].casefold()))
    lines = ["Investors appearing in more than one validated funding record:"]
    source_urls: set[str] = set()
    for name, urls in repeated:
        lines.append(f"- {name} — {len(urls)} records")
        source_urls.update(urls)
    sources = "\n".join(f"- {url}" for url in sorted(source_urls))
    answer_text = "\n".join(lines)
    return f"{answer_text}\n\nSources:\n{sources}"


@traceable(name="generate_with_gemini", run_type="llm", tags=["querying"])
def generate_with_gemini(
    question: str,
    contexts: list[dict[str, object]],
    model: str | None = None,
    api_key: str | None = None,
    plan: QueryPlan | None = None,
) -> str | None:
    """Generate an answer using LangChain LCEL RAG chain (ChatGoogleGenerativeAI) grounded strictly on retrieved chunks."""
    return langchain_generate_answer(
        question, contexts, model=model, api_key=api_key, plan=plan
    )


# Backward-compatibility alias
generate_with_llama3 = generate_with_gemini


def extract_cited_sources(
    llm_answer: str,
    contexts: list[dict[str, object]],
) -> list[str]:
    """Extract URLs of only the sources that the LLM actually cited or referenced in its answer."""
    # If the LLM indicates no answer could be found, return empty list (no sources used)
    if "cannot be found in the corpus" in llm_answer.lower():
        return []

    cited_indices = set()
    matches = re.findall(
        r"(?:\[Source\s*(\d+)\]|\(Source\s*(\d+)\)|\bSource\s*(\d+)\b|\[(\d+)\])",
        llm_answer,
        flags=re.IGNORECASE,
    )
    for m in matches:
        for g in m:
            if g and g.isdigit():
                cited_indices.add(int(g))

    cited_urls = []
    for idx in sorted(cited_indices):
        if 1 <= idx <= len(contexts):
            url = str((contexts[idx - 1].get("metadata") or {}).get("url", ""))
            if url and url not in cited_urls:
                cited_urls.append(url)

    # Also check if any context title is explicitly mentioned in the LLM answer
    for ctx in contexts:
        meta = ctx.get("metadata") or {}
        title = str(meta.get("title", "")).strip()
        url = str(meta.get("url", ""))
        if title and len(title) > 10 and title.lower() in llm_answer.lower():
            if url and url not in cited_urls:
                cited_urls.append(url)

    # Fallback: if no specific citation or title was detected in the text, include all context URLs
    if not cited_urls:
        for ctx in contexts:
            url = str((ctx.get("metadata") or {}).get("url", ""))
            if url and url not in cited_urls:
                cited_urls.append(url)

    return cited_urls


@traceable(name="answer_synthesis", run_type="chain", tags=["querying"])
def synthesize(
    question: str,
    contexts: list[dict[str, object]],
    plan: QueryPlan,
    model: str | None = None,
    api_key: str | None = None,
    retrieval_notice: str | None = None,
) -> str:
    prefix = f"{retrieval_notice}\n\n" if retrieval_notice else ""
    if not contexts or float(contexts[0]["score"]) < plan.low_relevance_threshold:
        return f"{prefix}The answer cannot be found in the corpus.\n\nSources: none"

    chosen_model = model or getattr(settings, "gemini_model", "gemini-3.8-flash")
    active_key = settings.get_api_key(api_key)

    if not active_key:
        urls = []
        for ctx in contexts:
            metadata = ctx.get("metadata") or {}
            url = str(metadata.get("url", ""))
            if url and url not in urls:
                urls.append(url)
        source_lines = "\n".join(f"- {url}" for url in sorted(urls)) or "none"
        return f"{prefix}{extractive_answer(contexts)}\n\nSources:\n{source_lines}"

    # Generate grounded answer with Gemini via LangChain
    llm_answer = generate_with_gemini(
        question, contexts, model=chosen_model, api_key=active_key, plan=plan
    )
    if llm_answer:
        # Extract only the sources that the LLM actually used or cited in its answer
        used_urls = extract_cited_sources(llm_answer, contexts)
        source_lines = "\n".join(f"- {url}" for url in sorted(used_urls)) or "none"
        return f"{prefix}{llm_answer}\n\nSources:\n{source_lines}"

    urls = []
    for ctx in contexts:
        metadata = ctx.get("metadata") or {}
        url = str(metadata.get("url", ""))
        if url and url not in urls:
            urls.append(url)
    source_lines = "\n".join(f"- {url}" for url in sorted(urls)) or "none"
    return (
        f"{prefix}Error: Unable to generate response with Gemini model '{chosen_model}'. "
        "Please check your API key, model quota, or network connection.\n\n"
        f"Sources:\n{source_lines}"
    )


@traceable(name="rag_query_pipeline", run_type="chain", tags=["querying"])
def answer(
    question: str,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    records = load_records(settings.records_dir)
    corpus_locations = {loc for r in records for loc in r.locations if loc}
    plan = plan_query(question, known_locations=corpus_locations)
    if plan.repeated_investors:
        return repeated_investors_answer(records, plan)

    retrieval_notice = None
    if plan.direct_records:
        contexts = direct_record_retrieve(records, plan)
    else:
        contexts = chroma_retrieve(question, plan)
        if not contexts:
            retrieval_notice = (
                "[Retrieval Notice: ChromaDB vector search was unavailable, returned an error, "
                "or yielded 0 vector matches. Retrieved using lexical keyword search fallback.]"
            )
            contexts = lexical_retrieve(question, records, plan)
    return synthesize(
        question,
        contexts,
        plan,
        model=model,
        api_key=api_key,
        retrieval_notice=retrieval_notice,
    )


def sample_questions(records: list[ArticleRecord]) -> list[str]:
    first_company = next((company for record in records for company in record.companies), "startup")
    first_title = records[0].title if records else "sample title"
    return [
        "What funding rounds are mentioned in the corpus?",
        "List all articles mentioning Bengaluru.",
        f"What does the corpus say about {first_company}?",
        f'Find the article titled "{first_title}"',
        "Which investors are appearing in >1 funding round?",
        "What AI stories are covered?",
        "Summarize layoffs in the corpus.",
        "What is the weather today?",
    ]


def run_samples(model: str | None = None, api_key: str | None = None) -> None:
    settings.ensure_dirs()
    records = load_records(settings.records_dir)
    output = Path("results/sample_queries.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    sections = ["# Sample Queries\n"]
    for idx, question in enumerate(sample_questions(records), start=1):
        response = answer(question, model=model, api_key=api_key)
        note = "Sources are corpus-backed; without a Gemini key the answer is deterministic and extractive."
        sections.append(f"## {idx}. {question}\n\n{response}\n\nCorrectness note: {note}\n")
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {output}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the local news RAG corpus using Gemini.")
    parser.add_argument("question", nargs="?", help="Natural language question.")
    parser.add_argument(
        "--model",
        default=getattr(settings, "gemini_model", "gemini-3.8-flash"),
        help="Gemini model name for generation (default: gemini-3.8-flash).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Gemini API Key (optional, defaults to GEMINI_API_KEY from environment or .env).",
    )
    parser.add_argument("--run-samples", action="store_true", help="Run required sample queries.")
    args = parser.parse_args()
    if args.run_samples:
        run_samples(model=args.model, api_key=args.api_key)
        return
    if not args.question:
        raise SystemExit('Usage: python query.py "your question" [--api-key YOUR_KEY]')
    print(answer(args.question, model=args.model, api_key=args.api_key))


if __name__ == "__main__":
    main()

