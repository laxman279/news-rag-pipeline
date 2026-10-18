from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import ArticleRecord


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, object]
    score: float = 0.0



def split_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """Recursively split text based on hierarchical separators."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    if not text or not text.strip():
        return []

    split_separators = (
        list(separators)
        if separators is not None
        else ["\n\n", "\n", ". ", " ", ""]
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=split_separators,
    )
    return splitter.split_text(text)


def chunks_for_record(
    record: ArticleRecord,
    chunk_size: int = 1000,
    overlap: int = 200,
    prepend_summary: bool = True,
) -> list[Chunk]:
    chunks = split_text(record.body, chunk_size=chunk_size, overlap=overlap)
    metadata = record.metadata()
    summary = (record.summary or "").strip()
    prefix = f"Summary: {summary}\n\n" if (prepend_summary and summary) else ""
    return [
        Chunk(
            id=f"{record.id}_chunk_{idx}",
            text=f"{prefix}{text}",
            metadata={**metadata, "chunk_index": idx},
        )
        for idx, text in enumerate(chunks)
    ]
