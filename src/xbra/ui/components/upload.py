"""UI component: trade log upload panel."""

from __future__ import annotations

import streamlit as st


def render_upload_panel() -> None:
    st.subheader("Upload Your Trade Log")

    uploaded = st.file_uploader(
        "Upload CSV or PDF broker export",
        type=["csv", "pdf"],
        help="CSV columns expected: date, symbol, side, qty, price. "
             "Exit trades can be paired by trade_id or matched heuristically.",
    )

    if uploaded:
        st.success(f"Uploaded: **{uploaded.name}** ({uploaded.size:,} bytes)")

        investor_id = st.text_input("Investor ID (for your records)", value="MY_PORTFOLIO")

        if st.button("Analyse Portfolio", type="primary"):
            # TODO Phase 1+6: pass to ingestion layer
            st.warning("Upload ingestion is implemented in Phase 1. Coming soon.")
