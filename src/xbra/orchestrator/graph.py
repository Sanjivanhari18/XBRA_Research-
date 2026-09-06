"""Phase 2 — Orchestrator: LangGraph DAG definition.

DAG topology (matches paper architecture):

  START
    │
  [ingestion_node]     — normalize raw trades into InvestorProfile
    │
  [orchestrator_node]  — intent routing, validates input, sets stage
    │
  ┌─────────────┐
  │             │     ← parallel branches (LangGraph fans out on multiple edges)
[behavior]  [market]
  │             │
  └──────┬──────┘
         │
      [risk]           ← waits for both (reads state.behavior_output + state.market_output)
         │
     [strategy]        ← also reads behavior_output (can start after behavior is done,
         │               LangGraph allows conditional sequencing via state checks)
         │
     [fusion]          ← collects behavior + risk + strategy
         │
  [explainability]     — SHAP + Ollama narrative + hallucination check
         │
     [report]          — Streamlit page + PDF
         │
       END

State object (XBRAStateDict) flows through the entire graph.
Each node reads what it needs from state, returns an update dict.
LangGraph merges updates back; nodes only write their own output keys.

Parallel execution note:
  LangGraph runs multiple outgoing edges from one node in parallel threads.
  behavior_node and market_node both receive state from orchestrator_node
  and write to different state keys → no write conflict.

  risk_node uses a conditional: it only runs after BOTH outputs are present.
  This is implemented via a gate node that checks state completeness.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Node imports (all implemented in their respective modules)
# Imported here so the graph has a single dependency point.
# ---------------------------------------------------------------------------

# Phase 3a
from src.xbra.agents.behavior_agent import behavior_node
# Phase 3b
from src.xbra.agents.market_agent import market_node
# Phase 3c
from src.xbra.agents.risk_agent import risk_node
# Phase 3d
from src.xbra.agents.strategy_agent import strategy_node
# Phase 4
from src.xbra.fusion.signal_fusion import fusion_node
# Phase 5
from src.xbra.explainability.narrative import explainability_node
# Phase 6
from src.xbra.report.report_node import report_node

from src.xbra.orchestrator.state import XBRAStateDict


# ---------------------------------------------------------------------------
# Local nodes defined here (ingestion gate and orchestrator logic)
# ---------------------------------------------------------------------------

def ingestion_node(state: XBRAStateDict) -> dict:
    """Validate that investor_profile is present; load from registry if only
    investor_id was provided."""
    from src.xbra.data.registry import DataRegistry
    from src.xbra.ingestion.loaders import load_from_parquet
    from config.settings import SYNTHETIC_DIR

    investor_id = state.get("investor_id", "")
    if not state.get("investor_profile") and investor_id:
        profile = load_from_parquet(investor_id, SYNTHETIC_DIR / "trades.parquet")
        return {"investor_profile": profile.model_dump(), "stage": "ingested"}
    return {"stage": "ingested"}


def orchestrator_node(state: XBRAStateDict) -> dict:
    """Route the user query to determine pipeline mode.
    For now: always full_analysis (router will add intent-based branching in Phase 2)."""
    from src.xbra.orchestrator.router import route_intent

    query = state.get("user_query") or "full_analysis"
    try:
        intent = route_intent(query)
    except NotImplementedError:
        intent = "full_analysis"
    return {"stage": f"routing:{intent}"}


def _both_agents_done(state: XBRAStateDict) -> str:
    """Conditional edge: block risk node until behavior AND market are ready."""
    b_done = state.get("behavior_output") is not None
    m_done = state.get("market_output") is not None
    return "risk" if (b_done and m_done) else "wait_gate"


def wait_gate_node(state: XBRAStateDict) -> dict:
    """No-op node that LangGraph routes to while waiting for parallel branches."""
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the XBRA LangGraph StateGraph.

    Returns a compiled graph with .invoke(state_dict) and .stream(state_dict).
    Raises ImportError if langgraph is not installed.
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise RuntimeError(
            "langgraph not installed — run: pip install langgraph"
        )

    graph = StateGraph(XBRAStateDict)

    # --- Register nodes ---
    graph.add_node("ingestion",      ingestion_node)
    graph.add_node("orchestrator",   orchestrator_node)
    graph.add_node("behavior",       behavior_node)
    graph.add_node("market",         market_node)
    graph.add_node("wait_gate",      wait_gate_node)
    graph.add_node("risk",           risk_node)
    graph.add_node("strategy",       strategy_node)
    graph.add_node("fusion",         fusion_node)
    graph.add_node("explainability", explainability_node)
    graph.add_node("report",         report_node)

    # --- Wire edges ---
    graph.add_edge(START,          "ingestion")
    graph.add_edge("ingestion",    "orchestrator")

    # Fan-out: behavior and market run in parallel
    graph.add_edge("orchestrator", "behavior")
    graph.add_edge("orchestrator", "market")

    # Strategy runs after behavior (reads behavior_output)
    graph.add_edge("behavior",     "strategy")

    # Risk needs both — use conditional routing through a gate
    graph.add_conditional_edges(
        "behavior",
        _both_agents_done,
        {"risk": "risk", "wait_gate": "wait_gate"},
    )
    graph.add_conditional_edges(
        "market",
        _both_agents_done,
        {"risk": "risk", "wait_gate": "wait_gate"},
    )
    graph.add_conditional_edges(
        "wait_gate",
        _both_agents_done,
        {"risk": "risk", "wait_gate": "wait_gate"},
    )

    # Linear from risk onward
    graph.add_edge("risk",           "fusion")
    graph.add_edge("strategy",       "fusion")
    graph.add_edge("fusion",         "explainability")
    graph.add_edge("explainability", "report")
    graph.add_edge("report",         END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_pipeline(investor_id: str, user_query: str | None = None) -> dict[str, Any]:
    """Run the full XBRA pipeline for one investor. Returns final state dict."""
    graph = build_graph()
    initial_state: XBRAStateDict = {
        "investor_id": investor_id,
        "user_query":  user_query or "full_analysis",
        "errors":      [],
        "stage":       "init",
    }
    return graph.invoke(initial_state)
