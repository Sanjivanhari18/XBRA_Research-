"""UI component: evaluation dashboard — paper result tables."""

from __future__ import annotations

import streamlit as st


def render_eval_dashboard() -> None:
    st.subheader("Evaluation — Paper Results")
    st.caption("Runs after the full pipeline has been executed on all 60 synthetic investors.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total investors", "60", help="10 per bias class")
    col2.metric("Phase", "7 (pending)", help="Requires Phase 3–6 complete")
    col3.metric("Ground truth", "Available", help="Known injected bias labels")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Table 1 · Bias F1", "Table 3 · Ablation", "Table 2 · SHAP Stability"])

    with tab1:
        st.markdown("**Per-class Precision / Recall / F1**")
        try:
            import pandas as pd
            from config.settings import DATA_DIR
            p = DATA_DIR / "paper_results" / "table1_bias_classification.csv"
            if p.exists():
                st.dataframe(pd.read_csv(p), use_container_width=True)
            else:
                st.info("Run Phase 7 evaluation to generate this table.")
        except Exception as e:
            st.error(str(e))

    with tab2:
        st.markdown("**Ablation: macro-F1 when each agent is removed**")
        try:
            import pandas as pd
            from config.settings import DATA_DIR
            p = DATA_DIR / "paper_results" / "table3_ablation.csv"
            if p.exists():
                st.dataframe(pd.read_csv(p), use_container_width=True)
            else:
                st.info("Run Phase 7 ablation study to generate this table.")
        except Exception as e:
            st.error(str(e))

    with tab3:
        st.markdown("**SHAP feature stability across rolling windows**")
        st.info("Run Phase 7 SHAP stability analysis to generate this table.")
