from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import ArticleRecord


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_records(records_dir: Path) -> list[ArticleRecord]:
    records = []
    for path in sorted(records_dir.glob("*.json")):
        records.append(ArticleRecord.model_validate(read_json(path)))
    return records


def urls_from_records(records: Iterable[ArticleRecord]) -> list[str]:
    return sorted({str(record.url) for record in records})
