from __future__ import annotations

import argparse

from langchain_core.documents import Document

from config import settings
from src.news_rag.chunking import chunks_for_record
from src.news_rag.io import load_records
from src.news_rag.langchain_rag import get_vectorstore
from src.news_rag.tracing import traceable


@traceable(name="index_pipeline", run_type="chain", tags=["indexing"])
def index_records() -> None:
    settings.ensure_dirs()
    records = load_records(settings.records_dir)
    vectorstore = get_vectorstore()
    expected_ids = {
        chunk.id
        for record in records
        for chunk in chunks_for_record(
            record,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
    }
    existing_ids = vectorstore.get(include=[]).get("ids", [])
    stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in expected_ids]
    if stale_ids:
        vectorstore.delete(ids=stale_ids)
    inserted = 0
    for record in records:
        existing = vectorstore.get(where={"article_id": record.id}, include=[])
        ids = existing.get("ids", [])
        if ids:
            vectorstore.delete(ids=ids)
        chunks = chunks_for_record(
            record,
            chunk_size=settings.chunk_size_chars,
            overlap=settings.chunk_overlap_chars,
        )
        if not chunks:
            continue
        docs = [
            Document(page_content=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]
        doc_ids = [chunk.id for chunk in chunks]
        vectorstore.add_documents(documents=docs, ids=doc_ids)
        inserted += len(chunks)
    print(
        f"Indexed {inserted} chunks from {len(records)} records into {settings.chroma_dir} using LangChain Chroma "
        f"({len(stale_ids)} stale chunks removed)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk records and upsert into ChromaDB using LangChain.")
    parser.parse_args()
    index_records()


if __name__ == "__main__":
    main()
