"""Safe wrapper for all external calls (yfinance, FinBERT, ChromaDB, Ollama).

Design-doc rule: every call that touches a network or GPU must go through
safe_external_call so the pipeline degrades gracefully instead of crashing.
"""

from __future__ import annotations

from typing import Callable, Optional, TypeVar

from src.xbra.utils.logging import logger

_T = TypeVar("_T")


def safe_external_call(
    fn: Callable[[], _T],
    *,
    fallback: Optional[_T] = None,
    label: str = "",
) -> Optional[_T]:
    """Execute fn(); return fallback on any exception, logging the failure.

    fn must be a zero-argument callable (use a closure to capture context).

    Args:
        fn:       Zero-argument callable to run.
        fallback: Value returned when fn raises. Defaults to None.
        label:    Human-readable name for log messages (e.g. "yfinance/AAPL").
    """
    try:
        return fn()
    except Exception as exc:
        tag = label or getattr(fn, "__name__", "external_call")
        logger.warning(
            f"[safe_external_call] {tag} failed: {type(exc).__name__}: {exc}"
        )
        return fallback
