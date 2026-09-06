"""Phase 2 — Orchestrator: LangGraph state definition.

XBRAState is defined in src/xbra/schemas.py (single source of truth).
This module re-exports it and provides the LangGraph-compatible TypedDict
wrapper needed for graph construction.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, TypedDict

from src.xbra.schemas import XBRAState  # noqa: F401  (re-export)


class XBRAStateDict(TypedDict, total=False):
    """TypedDict version for LangGraph graph node signatures.

    LangGraph requires dict-based state; we convert XBRAState ↔ dict at the
    node boundary using .model_dump() / XBRAState(**dict).

    `stage` and `errors` use Annotated reducers so parallel nodes can write
    to them without triggering INVALID_CONCURRENT_GRAPH_UPDATE.
    """
    investor_id:      str
    investor_profile: Dict[str, Any]
    user_query:       str
    behavior_output:  Dict[str, Any]
    market_output:    Dict[str, Any]
    risk_output:      Dict[str, Any]
    strategy_output:  Dict[str, Any]
    fused_bias:       Dict[str, Any]
    explanation:      Dict[str, Any]
    # last writer wins for stage (parallel nodes both write it)
    stage:            Annotated[str, lambda a, b: b]
    # errors accumulate across parallel nodes
    errors:           Annotated[List[str], operator.add]
