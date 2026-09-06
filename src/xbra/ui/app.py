"""Phase 6 — Streamlit UI: main application entry point.

Run:
    streamlit run src/xbra/ui/app.py

Page layout:
  Sidebar:  mode selector, model status indicator, investor picker
  Main:     tabbed view — Analysis | Report | Evaluation | About
"""

from __future__ import annotations

import streamlit as st

from src.xbra.utils.logging import setup_logging

setup_logging()

st.set_page_config(
    page_title="XBRA — Explainable Behavioral Risk Attribution",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("XBRA")
    st.caption("Explainable Behavioral Risk Attribution")
    st.divider()

    mode = st.radio(
        "Mode",
        ["Demo — Synthetic Investor", "Upload Trade Log"],
        index=0,
    )

    st.divider()
    st.subheader("Model Status")

    def _check_ollama() -> bool:
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    if _check_ollama():
        st.success("Ollama: running")
    else:
        st.warning("Ollama: offline (narratives disabled)")

    st.divider()
    st.caption("llama3.3:70b · llama3.2:3b · FinBERT")
    st.caption("ChromaDB · LangGraph · SHAP")

# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

tab_analysis, tab_report, tab_eval, tab_about = st.tabs(
    ["🔍 Analysis", "📄 Report", "📊 Evaluation", "ℹ️ About"]
)

# ── Analysis tab ──────────────────────────────────────────────────────────
with tab_analysis:
    if mode == "Demo — Synthetic Investor":
        from src.xbra.ui.components.investor_picker import render_investor_picker
        render_investor_picker()
    else:
        from src.xbra.ui.components.upload import render_upload_panel
        render_upload_panel()

# ── Report tab ────────────────────────────────────────────────────────────
with tab_report:
    from src.xbra.ui.components.report_view import render_report_view
    render_report_view()

# ── Evaluation tab ────────────────────────────────────────────────────────
with tab_eval:
    from src.xbra.ui.components.eval_dashboard import render_eval_dashboard
    render_eval_dashboard()

# ── About tab ─────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ### XBRA — Explainable Behavioral Risk Attribution

    **Research paper proof-of-concept.** An agentic AI framework that detects,
    quantifies, and explains behavioural biases in retail investor trading data.

    #### Pipeline
    ```
    Trade data → Ingestion → Orchestrator (LangGraph)
                                   │
                   ┌───────────────┼───────────────┐
               Behavior          Market         (parallel)
                   │               │
                   └───────┬───────┘
                           │
                          Risk (SHAP decomposition)
                           │
                       Strategy
                           │
                      Signal Fusion
                           │
                      Explainability (Ollama narrative)
                           │
                         Report (PDF)
    ```

    #### Paper Hypotheses
    - **H1** — Behavioural features explain a significant % of portfolio drawdown
    - **H2** — Attribution varies meaningfully across market regimes
    - **H3** — Agent-based decomposition outperforms a monolithic baseline

    #### Tech Stack
    Streamlit · LangGraph · Ollama (llama3.3:70b) · ChromaDB ·
    FinBERT · XGBoost · SHAP · yfinance · Zero cost.
    """)
