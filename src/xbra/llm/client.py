"""Ollama LLM client — shared singleton used by all agents.

Two clients:
  - router_llm  : llama3.2:3b  (fast, intent routing only)
  - analysis_llm: llama3.3:70b (full reasoning, per-agent LLM step)

All agents call `get_analysis_llm()` or `get_router_llm()` rather than
instantiating their own clients, so model + temperature are set once.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from config.settings import (
    ANALYSIS_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    OLLAMA_BASE_URL,
    ORCHESTRATOR_MODEL,
)


def _make_ollama_llm(model: str, temperature: float = LLM_TEMPERATURE) -> Any:
    """Lazily import langchain_ollama to avoid import-time errors if not installed."""
    try:
        from langchain_ollama import OllamaLLM
    except ImportError:
        raise RuntimeError(
            "langchain-ollama not installed — run: pip install langchain-ollama"
        )
    return OllamaLLM(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=LLM_MAX_TOKENS,
    )


@lru_cache(maxsize=1)
def get_router_llm() -> Any:
    """Return the fast routing LLM (llama3.2:3b), cached."""
    return _make_ollama_llm(ORCHESTRATOR_MODEL, temperature=0.0)


@lru_cache(maxsize=1)
def get_analysis_llm() -> Any:
    """Return the analysis LLM (llama3.3:70b), cached."""
    return _make_ollama_llm(ANALYSIS_MODEL, temperature=LLM_TEMPERATURE)


def invoke(llm: Any, prompt: str) -> str:
    """Invoke an LLM and return the response string.

    Wraps the call so we can add retry logic or logging here later.
    """
    response = llm.invoke(prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()
