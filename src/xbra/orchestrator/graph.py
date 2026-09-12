"""Phase 2 — Orchestrator: LangGraph DAG definition.

Sequential execution topology (matches paper architecture):

  START
    │
  [ingestion_node]     — normalize raw trades into InvestorProfile
    │
  [orchestrator_node]  — intent routing, validates input, sets stage
    │
  [behavior]           — compute bias scores from trade patterns
    │
  [market]             — fetch OHLCV/news, compute indicators & regime per trade
    │
  [strategy]           — infer trading archetype, consistency flags
    │
  [risk]               — drawdown, Sharpe, VaR, behavioral risk decomposition
    │
  [fusion]             — fuse all agent outputs into FusedBiasVector
    │
  [explainability]     — SHAP + Ollama narrative + hallucination check
    │
  [report]             — Streamlit page + PDF
    │
   END

Sequential order ensures each agent can read the outputs of all prior agents
from the shared XBRAStateDict. No parallel branches.
"""

from __future__ import annotations

from typing import Any

from src.xbra.agents.behavior_agent import behavior_node
from src.xbra.agents.market_agent import market_node
from src.xbra.agents.risk_agent import risk_node
from src.xbra.agents.strategy_agent import strategy_node
from src.xbra.fusion.signal_fusion import fusion_node
from src.xbra.explainability.narrative import explainability_node
from src.xbra.report.report_node import report_node
from src.xbra.orchestrator.state import XBRAStateDict


# ---------------------------------------------------------------------------
# Local nodes
# ---------------------------------------------------------------------------

def ingestion_node(state: XBRAStateDict) -> dict:
    """Validate investor_profile is present; load from registry if only investor_id given."""
    from src.xbra.data.registry import DataRegistry
    from src.xbra.ingestion.loaders import load_from_parquet
    from config.settings import SYNTHETIC_DIR

    investor_id = state.get("investor_id", "")
    if not state.get("investor_profile") and investor_id:
        profile, _ = load_from_parquet(investor_id, SYNTHETIC_DIR / "trades.parquet")
        return {"investor_profile": profile.model_dump(), "stage": "ingested"}
    return {"stage": "ingested"}


def orchestrator_node(state: XBRAStateDict) -> dict:
    """Route the user query to determine pipeline mode."""
    from src.xbra.orchestrator.router import route_intent

    query = state.get("user_query") or "full_analysis"
    try:
        intent = route_intent(query)
    except NotImplementedError:
        intent = "full_analysis"
    return {"stage": f"routing:{intent}"}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the XBRA LangGraph StateGraph.

    Returns a compiled graph with .invoke(state_dict) and .stream(state_dict).
    """
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        raise RuntimeError("langgraph not installed — run: pip install langgraph")

    graph = StateGraph(XBRAStateDict)

    graph.add_node("ingestion",      ingestion_node)
    graph.add_node("orchestrator",   orchestrator_node)
    graph.add_node("behavior",       behavior_node)
    graph.add_node("market",         market_node)
    graph.add_node("strategy",       strategy_node)
    graph.add_node("risk",           risk_node)
    graph.add_node("fusion",         fusion_node)
    graph.add_node("explainability", explainability_node)
    graph.add_node("report",         report_node)

    graph.add_edge(START,            "ingestion")
    graph.add_edge("ingestion",      "orchestrator")
    graph.add_edge("orchestrator",   "behavior")
    graph.add_edge("behavior",       "market")
    graph.add_edge("market",         "strategy")
    graph.add_edge("strategy",       "risk")
    graph.add_edge("risk",           "fusion")
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
