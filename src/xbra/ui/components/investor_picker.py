"""UI component: synthetic investor picker + pipeline trigger."""

from __future__ import annotations

import streamlit as st


def render_investor_picker() -> None:
    st.subheader("Demo — Synthetic Investor Analysis")

    try:
        from src.xbra.data.registry import DataRegistry
        investor_ids = DataRegistry.investor_ids()
    except FileNotFoundError:
        st.error(
            "Synthetic dataset not found. Generate it first:\n"
            "```\npython scripts/generate_synthetic.py\n```"
        )
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        selected = st.selectbox("Select investor profile", investor_ids)
    with col2:
        bias = DataRegistry.bias_for(selected) if selected else "—"
        st.metric("Ground-truth bias", bias)

    if st.button("Run Analysis", type="primary"):
        st.session_state["selected_investor"] = selected
        st.session_state["pipeline_result"] = None
        with st.spinner("Running XBRA pipeline..."):
            try:
                from src.xbra.orchestrator.graph import run_pipeline
                result = run_pipeline(selected)
                st.session_state["pipeline_result"] = result
            except NotImplementedError as e:
                st.warning(f"Pipeline stub reached: {e}\n\nImplement the relevant phase to proceed.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ── Display results ────────────────────────────────────────────────────
    result = st.session_state.get("pipeline_result")
    if result and st.session_state.get("selected_investor") == selected:
        st.success("Analysis complete.")
        st.divider()

        beh = result.get("behavior_output", {})
        mkt = result.get("market_output", {})
        risk = result.get("risk_output", {})
        strat = result.get("strategy_output", {})
        fused = result.get("fused_bias", {})
        expl = result.get("explanation", {})

        # ── Bias scores ────────────────────────────────────────────────────
        st.subheader("Detected Biases")
        if beh:
            scores = {
                "Loss Aversion":  beh.get("loss_aversion_score", 0),
                "Disposition":    beh.get("disposition_score", 0),
                "Herding":        beh.get("herding_score", 0),
                "Overconfidence": beh.get("overconfidence_score", 0),
            }
            cols = st.columns(4)
            for col, (label, score) in zip(cols, scores.items()):
                col.metric(label, f"{score:.2f}")

            dominant = fused.get("dominant_bias", beh.get("dominant_bias", "—"))
            st.info(f"**Dominant bias:** {dominant}")

        # ── Risk metrics ───────────────────────────────────────────────────
        st.subheader("Risk Profile")
        if risk:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Drawdown",   f"{risk.get('max_drawdown', 0):.3f}")
            c2.metric("Sharpe Ratio",   f"{risk.get('sharpe_ratio', 0):.3f}")
            c3.metric("Sortino Ratio",  f"{risk.get('sortino_ratio', 0):.3f}")
            c4.metric("VaR 95%",        f"{risk.get('var_95', 0):.3f}")

            st.markdown("**Risk Decomposition (SHAP)**")
            r_beh  = risk.get("r_behavioral_pct", 0)
            r_mkt  = risk.get("r_market_pct", 0)
            r_int  = risk.get("r_interaction_pct", 0)
            peer   = risk.get("peer_percentile", 50)
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("R_behavioral", f"{r_beh:.1f}%")
            d2.metric("R_market",     f"{r_mkt:.1f}%")
            d3.metric("R_interaction",f"{r_int:.1f}%")
            d4.metric("Peer %ile",    f"{peer:.0f}th")

        # ── Strategy ───────────────────────────────────────────────────────
        if strat:
            st.subheader("Strategy Archetype")
            st.markdown(f"**{strat.get('archetype', '—')}** "
                        f"(turnover {strat.get('turnover_rate', 0):.4f})")

        # ── Narratives ─────────────────────────────────────────────────────
        if expl and expl.get("per_bias"):
            st.subheader("Explanations")
            for bias_key, info in expl["per_bias"].items():
                pattern  = info.get("pattern", info.get("narrative", "")) if isinstance(info, dict) else str(info)
                evidence = info.get("evidence", "") if isinstance(info, dict) else ""
                with st.expander(f"**{bias_key.replace('_', ' ').title()}**"):
                    st.write(pattern)
                    if evidence:
                        st.caption(evidence[:400])

        # ── Recommendations ────────────────────────────────────────────────
        full_report = expl.get("full_report", "") if expl else ""
        if full_report:
            st.subheader("Recommendations")
            recs = [r.strip() for r in full_report.split("\n") if r.strip()]
            for r in recs[:5]:
                st.markdown(f"- {r}")

        # ── Report link ────────────────────────────────────────────────────
        report_path = result.get("report_path")
        if report_path:
            st.divider()
            st.caption(f"PDF report saved to: `{report_path}`")
