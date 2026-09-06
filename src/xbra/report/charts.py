"""Phase 6 — Report: chart builders using Plotly."""

from __future__ import annotations

import io
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        raise RuntimeError("plotly not installed — run: pip install plotly")


def bias_radar_chart(fused_bias: Dict[str, float]) -> Any:
    go = _plotly()
    categories = ["Loss Aversion", "Overconfidence", "Herding", "Disposition"]
    keys       = ["loss_aversion", "overconfidence", "herding", "disposition"]
    values     = [fused_bias.get(k, 0.0) for k in keys]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=cats_closed,
        fill="toself", name="Bias Profile",
        line=dict(color="#E74C3C", width=2),
        fillcolor="rgba(231, 76, 60, 0.2)",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5] * (len(categories) + 1), theta=cats_closed,
        mode="lines", name="Population Avg",
        line=dict(color="#3498DB", width=1, dash="dash"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Behavioral Bias Profile",
        template="plotly_white",
        height=400,
    )
    return fig


def risk_decomposition_bar(r_behavioral: float, r_market: float, r_interaction: float) -> Any:
    go = _plotly()
    fig = go.Figure()
    colors = ["#E74C3C", "#3498DB", "#F39C12"]
    labels = ["Behavioural", "Market", "Interaction"]
    values = [r_behavioral, r_market, r_interaction]
    for label, val, color in zip(labels, values, colors):
        fig.add_trace(go.Bar(
            name=label, x=[val], y=["Risk Attribution"],
            orientation="h", marker_color=color,
            text=f"{val:.0f}%", textposition="inside",
        ))
    fig.update_layout(
        barmode="stack", title="Risk Decomposition",
        xaxis_title="% of Total Drawdown Explained",
        template="plotly_white", height=200,
        xaxis=dict(range=[0, 100]),
    )
    return fig


def regime_heatmap(regime_decomposition: Dict[str, Dict[str, float]]) -> Any:
    go = _plotly()
    if not regime_decomposition:
        fig = go.Figure()
        fig.update_layout(title="Regime Analysis (no data)")
        return fig

    regimes = list(regime_decomposition.keys())
    dims    = ["r_behavioral_pct", "r_market_pct", "r_interaction_pct"]
    labels  = ["Behavioural %", "Market %", "Interaction %"]
    z = [[regime_decomposition[r].get(d, 0) for d in dims] for r in regimes]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=labels, y=regimes,
        colorscale="RdBu_r", zmid=33,
        text=[[f"{v:.0f}%" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig.update_layout(title="Risk Attribution by Market Regime", template="plotly_white", height=300)
    return fig


def equity_curve_chart(trades: List[Dict], portfolio_value: float = 100_000.0) -> Any:
    go = _plotly()
    if not trades:
        return go.Figure()

    df = pd.DataFrame(trades).sort_values("exit_date")
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["cum_pnl"]   = df["realized_pnl"].cumsum()
    df["portfolio"] = portfolio_value + df["cum_pnl"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["exit_date"], y=df["portfolio"],
        mode="lines", name="Portfolio",
        line=dict(color="#2ECC71", width=2),
    ))
    fig.add_hline(y=portfolio_value, line_dash="dash", line_color="grey", annotation_text="Initial Capital")
    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        template="plotly_white", height=350,
    )
    return fig


def fig_to_png_bytes(fig: Any) -> bytes:
    """Convert Plotly figure to PNG bytes for embedding in PDF."""
    try:
        return fig.to_image(format="png", width=800, height=400)
    except Exception:
        return b""


def build_all_charts(
    fused_bias: Dict[str, float],
    risk_output: Dict[str, Any],
    trades: List[Dict],
    portfolio_value: float = 100_000.0,
) -> Dict[str, Any]:
    """Build all report charts. Returns {name: plotly_figure}."""
    return {
        "bias_radar":     bias_radar_chart(fused_bias),
        "risk_decomp":    risk_decomposition_bar(
            risk_output.get("r_behavioral_pct", 33),
            risk_output.get("r_market_pct", 33),
            risk_output.get("r_interaction_pct", 33),
        ),
        "regime_heatmap": regime_heatmap(risk_output.get("regime_decomposition", {})),
        "equity_curve":   equity_curve_chart(trades, portfolio_value),
    }
