"""Phase 6 — Report LangGraph node."""

from __future__ import annotations

from src.xbra.orchestrator.state import XBRAStateDict


def report_node(state: XBRAStateDict) -> dict:
    from src.xbra.report.charts import build_all_charts, fig_to_png_bytes
    from src.xbra.report.pdf_generator import generate_pdf
    from src.xbra.schemas import (
        ExplanationOutput, FusedBiasVector, InvestorProfile,
        RiskAgentOutput, StrategyAgentOutput,
    )

    profile_dict  = state.get("investor_profile")
    risk_dict     = state.get("risk_output")
    fused_dict    = state.get("fused_bias")
    strategy_dict = state.get("strategy_output")
    expl_dict     = state.get("explanation")

    if not all([profile_dict, risk_dict, fused_dict, expl_dict]):
        return {"errors": state.get("errors", []) + ["report_node: missing inputs"]}

    profile  = InvestorProfile(**profile_dict)      if isinstance(profile_dict, dict)   else profile_dict
    risk     = RiskAgentOutput(**risk_dict)         if isinstance(risk_dict, dict)       else risk_dict
    fused    = FusedBiasVector(**fused_dict)        if isinstance(fused_dict, dict)      else fused_dict
    strategy = StrategyAgentOutput(**strategy_dict) if isinstance(strategy_dict, dict)   else strategy_dict
    expl     = ExplanationOutput(**expl_dict)       if isinstance(expl_dict, dict)       else expl_dict

    trades = [t.model_dump() for t in profile.trades]

    charts   = build_all_charts(fused.model_dump(), risk.model_dump(), trades, profile.portfolio_value)
    chart_pngs = {k: fig_to_png_bytes(v) for k, v in charts.items()}

    pdf_path = generate_pdf(
        investor_id     = profile.investor_id,
        explanation     = expl.model_dump(),
        risk_output     = risk.model_dump(),
        fused_bias      = fused.model_dump(),
        strategy_output = strategy.model_dump() if strategy else {},
        trades          = trades,
        chart_pngs      = chart_pngs,
    )

    return {
        "stage":       "done",
        "report_path": str(pdf_path),
        "charts":      {k: v for k, v in charts.items()},  # Plotly figs for Streamlit
    }
