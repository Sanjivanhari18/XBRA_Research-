"""UI component: renders the analysis report from session state."""

from __future__ import annotations

import streamlit as st


def render_report_view() -> None:
    result = st.session_state.get("pipeline_result")
    investor_id = st.session_state.get("selected_investor", "")

    if not result or not isinstance(result, dict):
        st.info("Run an analysis from the Analysis tab first.")
        return

    errors = result.get("errors", [])
    if errors:
        for e in errors:
            st.error(e)

    fused    = result.get("fused_bias", {})
    risk     = result.get("risk_output", {})
    strategy = result.get("strategy_output", {})
    expl     = result.get("explanation", {})
    charts   = result.get("charts", {})  # Plotly figures from report_node

    st.subheader(f"Report — {investor_id}")

    # ── KPI row ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dominant Bias",   fused.get("dominant_bias", "—").replace("_", " ").title())
    c2.metric("Max Drawdown",    f"{risk.get('max_drawdown', 0):.1%}")
    c3.metric("Sharpe Ratio",    f"{risk.get('sharpe_ratio', 0):.2f}")
    c4.metric("Strategy",        (strategy.get("archetype") or "—").replace("_", " ").title())

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Behavioral Bias Profile")
        if "bias_radar" in charts:
            st.plotly_chart(charts["bias_radar"], use_container_width=True)
        else:
            from src.xbra.report.charts import bias_radar_chart
            st.plotly_chart(bias_radar_chart(fused), use_container_width=True)

    with col2:
        st.markdown("#### Risk Decomposition")
        if "risk_decomp" in charts:
            st.plotly_chart(charts["risk_decomp"], use_container_width=True)
        else:
            from src.xbra.report.charts import risk_decomposition_bar
            st.plotly_chart(
                risk_decomposition_bar(
                    risk.get("r_behavioral_pct", 33),
                    risk.get("r_market_pct", 33),
                    risk.get("r_interaction_pct", 33),
                ),
                use_container_width=True,
            )

    # ── Regime heatmap ────────────────────────────────────────────────────
    if risk.get("regime_decomposition"):
        st.markdown("#### Regime Analysis")
        if "regime_heatmap" in charts:
            st.plotly_chart(charts["regime_heatmap"], use_container_width=True)

    # ── Equity curve ─────────────────────────────────────────────────────
    if "equity_curve" in charts:
        st.markdown("#### Portfolio Equity Curve")
        st.plotly_chart(charts["equity_curve"], use_container_width=True)

    st.divider()

    # ── Bias explanations ─────────────────────────────────────────────────
    st.markdown("#### Behavioral Bias Deep-Dives")
    per_bias = expl.get("per_bias", {})
    if per_bias:
        for bias_name, parts in per_bias.items():
            with st.expander(f"**{bias_name.replace('_',' ').title()}**", expanded=True):
                st.markdown(f"🔍 **Pattern:** {parts.get('pattern', '')}")
                st.markdown(f"📋 **Evidence:** {parts.get('evidence', '')}")
                st.markdown(f"⚠️ **Consequence:** {parts.get('consequence', '')}")
    else:
        st.info("No significant biases detected above the reporting threshold.")

    # ── Recommendations ───────────────────────────────────────────────────
    recs = expl.get("full_report", "")
    if recs:
        st.markdown("#### Recommendations")
        st.markdown(recs)

    # ── PDF Download ──────────────────────────────────────────────────────
    report_path = result.get("report_path")
    if report_path:
        import pathlib
        p = pathlib.Path(report_path)
        if p.exists():
            with open(p, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF Report",
                    data=f,
                    file_name=p.name,
                    mime="application/pdf",
                    type="primary",
                )
    else:
        if st.button("Generate PDF"):
            st.warning("Ensure Phase 6 (report_node) completed successfully.")
