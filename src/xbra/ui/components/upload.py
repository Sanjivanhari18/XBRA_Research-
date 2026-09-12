"""UI component: trade log upload panel."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st


def render_upload_panel():
    """Render the CSV upload panel.

    Returns (investor_id, InvestorProfile, NormalizationReport) when the user
    uploads and clicks Analyse, or (None, None, None) otherwise.
    """
    st.subheader("Upload Your Trade Log")

    uploaded = st.file_uploader(
        "Upload CSV broker export",
        type=["csv"],
        help=(
            "Any broker CSV export. Accepted column names include: "
            "Date / Ticker / Action (BUY/SELL) / Shares / Price — "
            "the system maps these automatically."
        ),
    )

    if not uploaded:
        return None, None, None

    st.caption(f"File: **{uploaded.name}** ({uploaded.size:,} bytes)")

    if st.button("Analyse Portfolio", type="primary"):
        investor_id = f"INV_{uuid.uuid4().hex[:8].upper()}"

        # Write the upload to a temp file so loaders can read it from disk
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = Path(tmp.name)

        try:
            from src.xbra.ingestion.loaders import load_from_csv
            with st.spinner("Normalising fills and reconstructing positions…"):
                profile, report = load_from_csv(investor_id, tmp_path)
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
            return None, None, None
        finally:
            tmp_path.unlink(missing_ok=True)

        _show_ingestion_summary(report)
        return investor_id, profile, report

    return None, None, None


def _show_ingestion_summary(report) -> None:
    st.success(
        f"Loaded **{report.closed_position_count}** closed positions "
        f"(quality score: **{report.data_quality_score:.2f}**)"
    )
    cols = st.columns(4)
    cols[0].metric("Raw fills",   report.raw_fill_count)
    cols[1].metric("Accepted",    report.accepted_fill_count)
    cols[2].metric("Duplicates",  report.duplicate_fill_count)
    cols[3].metric("Rejected",    report.rejected_fill_count)

    if report.rejected_reasons:
        with st.expander("Rejected row details"):
            for r in report.rejected_reasons:
                st.warning(r)

    if report.open_position_count > 0:
        st.info(
            f"{report.open_position_count} position(s) are still open at the end of "
            "the upload window and are excluded from analysis."
        )
