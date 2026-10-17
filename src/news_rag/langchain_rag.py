from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from config import settings
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


@traceable(name="langchain_generate_answer", run_type="chain", tags=["querying"])
def langchain_generate_answer(
    question: str,
    contexts: list[dict[str, Any]],
    model: str | None = None,
    api_key: str | None = None,
    plan: QueryPlan | None = None,
) -> str | None:
    """Generate an answer using the LangChain LCEL RAG chain with Gemini."""
    if not contexts:
        return None

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
