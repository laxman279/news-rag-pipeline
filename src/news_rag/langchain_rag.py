from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from config import settings
from src.news_rag.chunking import Chunk
from src.news_rag.models import ParsedNewsQuery
from src.news_rag.planning import QueryPlan, chroma_where
from src.news_rag.tracing import traceable


_EMBEDDINGS: Any = None


def get_embeddings():
    """Return LangChain HuggingFaceEmbeddings using the configured embedding model."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS
    from langchain_huggingface import HuggingFaceEmbeddings

    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
    )
    return _EMBEDDINGS


def get_vectorstore():
    """Return a LangChain Chroma vector store connected to the persistent directory."""
    import chromadb
    from langchain_chroma import Chroma

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
    )


def get_llm(model: str | None = None, api_key: str | None = None):
    """Return a LangChain ChatGoogleGenerativeAI instance."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    chosen_model = model or settings.gemini_model or "gemini-3.8-flash"
    active_key = settings.get_api_key(api_key)
    if not active_key:
        raise ValueError(
            "Gemini API key is required. Set GEMINI_API_KEY in your .env file or pass via --api-key."
        )

    return ChatGoogleGenerativeAI(
        model=chosen_model,
        google_api_key=active_key,
        temperature=0.0,
    )


def format_context_chunks(
    contexts: list[dict[str, Any]], max_chars_per_chunk: int = 2500
) -> str:
    """Format retrieved context chunks with source headers for the prompt."""
    formatted_chunks = []
    for idx, ctx in enumerate(contexts, start=1):
        meta = ctx.get("metadata") or {}
        title = meta.get("title", "Untitled")
        pub = meta.get("published_at", "") or str(meta.get("published_yyyymmdd", ""))
        text = str(ctx.get("text", "")).strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rsplit(" ", 1)[0] + "..."
        header = f"[Source {idx}] Title: {title}"
        if pub:
            header += f" | Published: {pub}"
        formatted_chunks.append(f"{header}\nContent:\n{text}")
    return "\n\n".join(formatted_chunks)


def build_rag_chain(
    model: str | None = None,
    api_key: str | None = None,
    plan: QueryPlan | None = None,
):
    """Build a LangChain LCEL (LangChain Expression Language) RAG answering chain with Gemini."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    date_guidance = f"Note: Today's reference date is {today_str}."
    if plan and plan.date_range and plan.date_range != (None, None):
        start_d, end_d = plan.date_range
        date_guidance += (
            f" The context chunks below are already pre-filtered for the requested date window ({start_d} to {end_d})."
        )

    system_instruction = (
        "You are an AI assistant answering questions based strictly on the provided news article chunks.\n\n"
        "Instructions:\n"
        "1. Base your answer ONLY on the provided context chunks below.\n"
        "2. If the answer cannot be found or deduced from the context chunks, state: 'The answer cannot be found in the corpus.'\n"
        "3. Do not invent, assume, or hallucinate facts that are not present in the context.\n"
        "4. Always cite the specific source number(s) you used (e.g. [Source 1], [Source 2]). Only cite sources that directly support your answer.\n"
        f"5. {date_guidance}\n"
        "6. Be concise, direct, and factual in your response."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_instruction),
            (
                "human",
                "Context Chunks:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )

    llm = get_llm(model=model, api_key=api_key)
    return prompt | llm | StrOutputParser()


def build_rag_context(
    articles: dict[str, list[Chunk]],
    target_article_count: int | None = None,
) -> str:
    """Adaptively pack context chunks into formatted XML article blocks."""
    if not articles:
        return ""

    effective_count = target_article_count if target_article_count is not None else len(articles)
    is_compact = effective_count > 5

    article_blocks = []
    for idx, (aid, chunks) in enumerate(articles.items(), start=1):
        if not chunks:
            continue

        meta = chunks[0].metadata if isinstance(chunks[0].metadata, dict) else {}
        title = str(meta.get("title") or "Untitled").strip()
        pub_date = str(meta.get("published_date") or meta.get("published_at") or "").strip()
        url = str(meta.get("url") or "").strip()
        raw_cats = meta.get("categories") or meta.get("category") or []
        if isinstance(raw_cats, list):
            cats_str = ", ".join(str(c) for c in raw_cats if c)
        else:
            cats_str = str(raw_cats)

        if is_compact:
            top_chunk = max(chunks, key=lambda c: getattr(c, "score", 0.0) or 0.0)
            content_text = top_chunk.text.strip()
        else:
            top_3_chunks = sorted(
                chunks,
                key=lambda c: getattr(c, "score", 0.0) or 0.0,
                reverse=True,
            )[:3]
            sorted_chunks = sorted(
                top_3_chunks,
                key=lambda c: int(c.metadata.get("chunk_index", 0) if isinstance(c.metadata, dict) else 0),
            )
            content_text = "\n\n".join(c.text.strip() for c in sorted_chunks)

        block = (
            f'<article index="{idx}" id="{aid}">\n'
            f"  <header>\n"
            f"    <title>{title}</title>\n"
            f"    <published_date>{pub_date}</published_date>\n"
            f"    <url>{url}</url>\n"
            f"    <categories>{cats_str}</categories>\n"
            f"  </header>\n"
            f"  <content>\n"
            f"{content_text}\n"
            f"  </content>\n"
            f"</article>"
        )
        article_blocks.append(block)

    return "\n\n".join(article_blocks)


def _fallback_parse_query(raw_query: str) -> ParsedNewsQuery:
    """Deterministic regex-based fallback query parser."""
    count = 5
    count_match = re.search(
        r"\b(?:top\s+|first\s+|show\s+(?:me\s+)?|get\s+|find\s+)?(\d+)\s*(?:news\s+)?articles?\b",
        raw_query,
        flags=re.IGNORECASE,
    )
    if not count_match:
        count_match = re.search(r"\barticles?\b.*?(\d+)", raw_query, flags=re.IGNORECASE)
    if count_match:
        try:
            count = int(count_match.group(1))
        except (ValueError, TypeError):
            count = 5

    known_categories = [
        "artificial intelligence",
        "generative ai",
        "genai",
        "ai",
        "funding",
        "startups",
        "product launch",
        "policy",
        "m&a",
        "mergers and acquisitions",
        "layoffs",
        "earnings",
        "fintech",
        "semiconductors",
        "cybersecurity",
        "enterprise tech",
        "technology",
    ]
    category = None
    lowered = f" {raw_query.lower()} "
    for cat in known_categories:
        pattern = rf"(?:\b|\s){re.escape(cat)}(?:\b|\s)"
        if re.search(pattern, lowered):
            if cat in {"artificial intelligence", "generative ai", "genai", "ai"}:
                category = "ai"
            elif cat in {"mergers and acquisitions", "m&a"}:
                category = "m&a"
            else:
                category = cat.lower()
            break

    cleaned = re.sub(
        r"\b(?:tell\s+me\s+about|show\s+me|find|list|get|give\s+me)\b",
        "",
        raw_query,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:top\s+|first\s+)?\d+\s*(?:news\s+)?articles?\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\b(?:that\s+mentioned|mentioning|covering|about|related\s+to)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = " ".join(cleaned.split()).strip()
    if not cleaned:
        cleaned = raw_query.strip()

    return ParsedNewsQuery(
        semantic_query=cleaned,
        category_filter=category,
        target_article_count=count,
    )


@traceable(name="parse_user_query", run_type="parser", tags=["querying"])
def parse_user_query(
    raw_query: str,
    model: str | None = None,
    api_key: str | None = None,
    llm: Any = None,
) -> ParsedNewsQuery:
    """Parse user query into structured search parameters using Gemini or fallback."""
    active_key = settings.get_api_key(api_key)
    if llm is None and active_key:
        try:
            llm = get_llm(model=model, api_key=active_key)
        except Exception:
            llm = None

    if llm is not None:
        try:
            structured_llm = llm.with_structured_output(ParsedNewsQuery)
            prompt = (
                "You are an expert news search query parser. Extract structured search parameters from the query:\n"
                "- semantic_query: The core topic or search subject stripped of article count requests and filter syntax.\n"
                "- category_filter: Normalized lowercase category or topic tag if explicitly requested "
                "(e.g., 'ai', 'funding', 'fintech', 'semiconductors', 'policy', 'layoffs', 'm&a'), or null if unspecified.\n"
                "- target_article_count: The number of distinct articles requested by the user (integer between 1 and 25, default 5).\n\n"
                f"Query: {raw_query}"
            )
            result = structured_llm.invoke(prompt)
            if isinstance(result, ParsedNewsQuery):
                return result
            if isinstance(result, dict):
                return ParsedNewsQuery.model_validate(result)
        except Exception as exc:
            print(f"[QueryParser Warning] LLM parsing failed: {exc}. Using fallback parser.", flush=True)

    return _fallback_parse_query(raw_query)


def _convert_to_chunk(doc: Any, dist_or_score: Any) -> Chunk:
    if isinstance(dist_or_score, (int, float)):
        val = float(dist_or_score)
        score = max(0.0, 1.0 - val) if 0.0 <= val <= 2.0 else val
    else:
        score = 0.0

    if hasattr(doc, "page_content"):
        content = doc.page_content
        metadata = dict(getattr(doc, "metadata", {}) or {})
        cid = getattr(doc, "id", None) or str(metadata.get("chunk_id", ""))
    elif isinstance(doc, dict):
        content = str(doc.get("text") or doc.get("page_content") or "")
        metadata = dict(doc.get("metadata") or {})
        cid = str(doc.get("id") or metadata.get("chunk_id") or "")
    else:
        content = str(doc)
        metadata = {}
        cid = ""

    aid = str(metadata.get("article_id") or metadata.get("id") or "")
    c_idx = metadata.get("chunk_index", 0)
    if not cid:
        cid = f"{aid}_chunk_{c_idx}" if aid else f"chunk_{id(doc)}"

    return Chunk(id=cid, text=content, metadata=metadata, score=score)


def _fetch_from_vectorstore(
    vstore: Any,
    query: str,
    k: int,
    category_filter: str | None = None,
) -> list[Chunk]:
    filter_dict = None
    if category_filter:
        filter_dict = {"categories": {"$contains": category_filter.lower()}}

    raw_results = None
    if filter_dict is not None:
        try:
            raw_results = vstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict,
            )
        except Exception:
            try:
                raw_results = vstore.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter={"category": category_filter.lower()},
                )
            except Exception:
                raw_results = None

    if raw_results is None:
        try:
            raw_results = vstore.similarity_search_with_score(query=query, k=k)
        except Exception as exc:
            print(f"[Vectorstore Error] similarity search failed: {exc}", flush=True)
            return []

    chunks = []
    for item in raw_results:
        if isinstance(item, tuple) and len(item) == 2:
            d, s = item
        else:
            d, s = item, 0.0
        chunks.append(_convert_to_chunk(d, s))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def _group_chunks_by_article(chunks: list[Chunk], limit: int) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        aid = str(chunk.metadata.get("article_id") or chunk.metadata.get("id") or "")
        if not aid:
            continue
        if aid not in grouped:
            if len(grouped) >= limit:
                continue
            grouped[aid] = []
        grouped[aid].append(chunk)
    return grouped


@traceable(name="retrieve_unique_articles", run_type="retriever", tags=["querying"])
def retrieve_unique_articles(
    parsed_query: ParsedNewsQuery,
    vectorstore: Any = None,
) -> dict[str, list[Chunk]]:
    """Retrieve chunks grouped into target_article_count unique articles."""
    if vectorstore is None:
        try:
            vectorstore = get_vectorstore()
        except Exception as exc:
            print(f"[Vectorstore Error] Could not initialize vectorstore: {exc}", flush=True)
            return {}

    target_count = parsed_query.target_article_count
    category_filter = parsed_query.category_filter
    semantic_query = parsed_query.semantic_query or ""

    # Check for Native Grouping support (e.g. Qdrant, Weaviate, Milvus)
    native_method = getattr(vectorstore, "query_points_groups", None) or getattr(
        getattr(vectorstore, "_client", None), "query_points_groups", None
    )
    if native_method is not None:
        try:
            groups_resp = native_method(
                query=semantic_query,
                group_by="article_id",
                group_limit=target_count,
                group_size=3,
                filter=category_filter,
            )
            if groups_resp:
                native_dict: dict[str, list[Chunk]] = {}
                for g in groups_resp:
                    aid = getattr(g, "id", None) or str(g.get("id", ""))
                    points = getattr(g, "hits", None) or g.get("hits", [])
                    native_dict[aid] = [
                        _convert_to_chunk(p, getattr(p, "score", 0.0)) for p in points
                    ]
                if len(native_dict) >= target_count or not category_filter:
                    return native_dict
        except Exception as exc:
            print(f"[Vectorstore Info] Native grouping query failed: {exc}. Using iterative fallback.", flush=True)

    # Fallback (Iterative Over-Fetch & Deduplicate):
    # Step 1: Pre-filter by category_filter if present (fetch k = target_count * 5)
    grouped: dict[str, list[Chunk]] = {}
    if category_filter:
        chunks = _fetch_from_vectorstore(
            vectorstore,
            query=semantic_query,
            k=target_count * 5,
            category_filter=category_filter,
        )
        grouped = _group_chunks_by_article(chunks, limit=target_count)

    # Fallback to unfiltered search if strict filtering returned fewer than target_count results
    if len(grouped) < target_count:
        unfiltered_chunks = _fetch_from_vectorstore(
            vectorstore,
            query=semantic_query,
            k=target_count * 5,
            category_filter=None,
        )
        grouped = _group_chunks_by_article(unfiltered_chunks, limit=target_count)

    # If still fewer than target_count, perform up to one pagination/expansion step (k = target_count * 10)
    if len(grouped) < target_count:
        expanded_chunks = _fetch_from_vectorstore(
            vectorstore,
            query=semantic_query,
            k=target_count * 10,
            category_filter=None,
        )
        grouped = _group_chunks_by_article(expanded_chunks, limit=target_count)

    return grouped


@traceable(name="langchain_generate_answer", run_type="chain", tags=["querying"])
def langchain_generate_answer(
    question: str,
    contexts: list[dict[str, Any]] | dict[str, list[Chunk]],
    model: str | None = None,
    api_key: str | None = None,
    plan: QueryPlan | None = None,
    target_article_count: int | None = None,
) -> str | None:
    """Generate an answer using the LangChain LCEL RAG chain with Gemini."""
    if not contexts:
        return None

    if isinstance(contexts, dict):
        context_str = build_rag_context(contexts, target_article_count=target_article_count)
    else:
        is_large_context = len(contexts) > 5
        max_chars = 1200 if is_large_context else 2500
        context_str = format_context_chunks(contexts, max_chars_per_chunk=max_chars)

    try:
        chain = build_rag_chain(model=model, api_key=api_key, plan=plan)
        response = chain.invoke({"context": context_str, "question": question})
        return str(response).strip() if response else None
    except Exception as exc:
        print(f"[Gemini Error] Generation failed: {exc}", flush=True)
        return None




@traceable(name="langchain_chroma_retrieve", run_type="retriever", tags=["querying"])
def langchain_chroma_retrieve(
    question: str,
    plan: QueryPlan,
    chunk_matches_fn: Any = None,
) -> list[dict[str, Any]]:
    """Retrieve top relevant chunks from Chroma using LangChain's Chroma vector store."""
    try:
        vectorstore = get_vectorstore()
    except Exception as exc:
        print(f"[ChromaDB Error] Could not initialize Chroma vectorstore: {exc}", flush=True)
        return []

    target_count = plan.limit or (max(settings.top_k, 5) if plan.is_aggregation else settings.top_k)
    fetch_k = min(45, max(target_count * 4, 15))

    try:
        results = vectorstore.similarity_search_with_score(
            query=question,
            k=fetch_k,
            filter=chroma_where(plan),
        )
    except Exception as exc:
        print(f"[ChromaDB Error] Similarity search failed: {exc}", flush=True)
        return []

    if not results:
        print("[ChromaDB Info] Vector similarity search returned 0 results.", flush=True)

    rows = []
    seen_urls = set()
    for doc, distance in results:
        metadata = doc.metadata or {}
        if chunk_matches_fn and not chunk_matches_fn(doc.page_content, metadata, plan):
            continue
        url = metadata.get("url")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        rows.append({
            "text": doc.page_content,
            "metadata": metadata,
            "score": max(0.0, 1.0 - float(distance)),
        })
        if len(rows) >= target_count:
            break
    return rows
