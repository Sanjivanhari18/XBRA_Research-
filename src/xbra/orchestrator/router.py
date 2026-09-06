"""Phase 2 — Intent router using llama3.2:3b (fast, cached).

Classifies the user's natural-language query into a pipeline mode so the
orchestrator can skip unnecessary agents when only partial analysis is needed.
"""

from __future__ import annotations

from src.xbra.llm.client import get_router_llm, invoke
from src.xbra.llm.prompts import INTENT_ROUTER_PROMPT, parse_intent

INTENT_LABELS = [
    "full_analysis",
    "bias_only",
    "risk_only",
    "explain_trade",
    "report",
]


def route_intent(user_query: str) -> str:
    """Classify user_query → one of INTENT_LABELS.

    Falls back to "full_analysis" if the LLM is unavailable.
    """
    # TODO Phase 2: uncomment when Ollama is running
    # llm = get_router_llm()
    # prompt = INTENT_ROUTER_PROMPT.format(user_query=user_query)
    # response = invoke(llm, prompt)
    # return parse_intent(response)
    raise NotImplementedError("Phase 2 — route_intent (needs Ollama running)")
