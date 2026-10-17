from __future__ import annotations

import functools
from typing import Any, Callable

from config import settings

# Ensure LangSmith environment variables are initialized
settings.setup_langsmith()

try:
    from langsmith import traceable as _langsmith_traceable

    HAS_LANGSMITH = True
except ImportError:  # pragma: no cover
    HAS_LANGSMITH = False
    _langsmith_traceable = None


def traceable(
    name: str | None = None,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Safe wrapper for @langsmith.traceable that falls back to a transparent decorator if langsmith is not installed."""
    if HAS_LANGSMITH and _langsmith_traceable is not None:
        return _langsmith_traceable(
            name=name,
            run_type=run_type,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper

    return decorator
