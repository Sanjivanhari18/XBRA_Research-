"""Phase 6 — PDF Generator using ReportLab."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List

from config.settings import PDF_OUTPUT_DIR


def generate_pdf(
    investor_id: str,
    explanation: Dict[str, Any],
    risk_output: Dict[str, Any],
    fused_bias: Dict[str, Any],
    strategy_output: Dict[str, Any],
    trades: List[Dict],
    chart_pngs: Dict[str, bytes] | None = None,
) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
        )
    except ImportError:
        raise RuntimeError("reportlab not installed — run: pip install reportlab")

    out_path = PDF_OUTPUT_DIR / f"{investor_id}_xbra_report.pdf"
    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()
    story: List[Any] = []

    H1 = styles["Heading1"]
    H2 = styles["Heading2"]
    BodyText = styles["BodyText"]
    BodyText.leading = 14

    def add_heading(text: str, level: int = 1) -> None:
        story.append(Paragraph(text, H1 if level == 1 else H2))
        story.append(Spacer(1, 0.3*cm))

    def add_para(text: str) -> None:
        story.append(Paragraph(text, BodyText))
        story.append(Spacer(1, 0.2*cm))

    def add_hr() -> None:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Spacer(1, 0.3*cm))

    # ── Cover ──────────────────────────────────────────────────────────────
    add_heading("XBRA — Explainable Behavioral Risk Attribution")
    add_para(f"<b>Investor:</b> {investor_id}")
    add_para(f"<b>Report date:</b> {date.today().isoformat()}")
    add_hr()

    # ── Executive Summary ──────────────────────────────────────────────────
    add_heading("1. Executive Summary", level=2)
    dominant = fused_bias.get("dominant_bias", "neutral").replace("_", " ").title()
    max_dd   = risk_output.get("max_drawdown", 0)
    r_beh    = risk_output.get("r_behavioral_pct", 0)
    summary  = (
        f"Analysis of {investor_id}'s trading history reveals a dominant behavioural pattern of "
        f"<b>{dominant}</b>. Behavioural factors explain <b>{r_beh:.0f}%</b> of the portfolio's "
        f"maximum drawdown of <b>{max_dd:.1%}</b>. "
        f"This report provides a detailed breakdown of detected biases, "
        f"market context, and actionable recommendations."
    )
    add_para(summary)
    add_hr()

    # ── Investor Risk Profile ──────────────────────────────────────────────
    add_heading("2. Investor Risk Profile", level=2)
    arch     = strategy_output.get("archetype", "unknown").replace("_", " ").title()
    sharpe   = risk_output.get("sharpe_ratio", 0)
    sortino  = risk_output.get("sortino_ratio", 0)
    peer_pct = risk_output.get("peer_percentile", 50)
    profile_data = [
        ["Metric", "Value"],
        ["Dominant Bias",       dominant],
        ["Strategy Archetype",  arch],
        ["Max Drawdown",        f"{max_dd:.1%}"],
        ["Sharpe Ratio",        f"{sharpe:.2f}"],
        ["Sortino Ratio",       f"{sortino:.2f}"],
        ["Peer Risk Percentile",f"{peer_pct:.0f}th"],
        ["Behavioural Risk %",  f"{r_beh:.0f}%"],
    ]
    tbl = Table(profile_data, colWidths=[7*cm, 7*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F3F4")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))
    add_hr()

    # ── Charts ─────────────────────────────────────────────────────────────
    if chart_pngs:
        add_heading("3. Charts", level=2)
        for chart_name, png_bytes in chart_pngs.items():
            if png_bytes:
                import io
                story.append(Image(io.BytesIO(png_bytes), width=14*cm, height=7*cm))
                story.append(Spacer(1, 0.3*cm))
        add_hr()

    # ── Bias Deep-Dives ────────────────────────────────────────────────────
    add_heading("4. Behavioral Bias Deep-Dives", level=2)
    per_bias = explanation.get("per_bias", {})
    if per_bias:
        for bias_name, parts in per_bias.items():
            add_heading(f"  {bias_name.replace('_',' ').title()}", level=2)
            for part_name in ["pattern", "evidence", "consequence"]:
                text = parts.get(part_name, "")
                if text:
                    add_para(f"<b>{part_name.title()}:</b> {text}")
            story.append(Spacer(1, 0.3*cm))
    else:
        add_para("No significant biases detected above the reporting threshold.")
    add_hr()

    # ── Trade Evidence Table ───────────────────────────────────────────────
    add_heading("5. Trade Evidence", level=2)
    if trades:
        trade_sample = sorted(trades, key=lambda x: abs(x.get("realized_pnl", 0)), reverse=True)[:10]
        tdata = [["Trade ID", "Symbol", "Entry", "Exit", "Days", "P&L"]]
        for t in trade_sample:
            tdata.append([
                t.get("trade_id", "")[:12],
                t.get("symbol", ""),
                str(t.get("entry_date", ""))[:10],
                str(t.get("exit_date", ""))[:10],
                str(t.get("holding_days", "")),
                f"{t.get('realized_pnl', 0):.0f}",
            ])
        ttbl = Table(tdata, colWidths=[3.5*cm, 2*cm, 2.5*cm, 2.5*cm, 1.5*cm, 2*cm])
        ttbl.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 8),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.grey),
            ("PADDING",     (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F3F4")]),
        ]))
        story.append(ttbl)
    add_hr()

    # ── Recommendations ────────────────────────────────────────────────────
    add_heading("6. Recommendations", level=2)
    recs = explanation.get("full_report", "")
    if recs:
        for line in recs.split("\n"):
            line = line.strip()
            if line:
                add_para(line)

    doc.build(story)
    return out_path
