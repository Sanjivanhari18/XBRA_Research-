"""Phase 5 — Narrative generation: LLM explanation + hallucination check."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from src.xbra.explainability.verifier import MAX_RETRIES, verify_explanation
from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import ExplanationOutput


# ---------------------------------------------------------------------------
# Metric labels for evidence text
# ---------------------------------------------------------------------------

BIAS_METRIC_MAP = {
    "loss_aversion":  ("holding_time_asymmetry",  "Holding-time asymmetry (loser/winner)"),
    "overconfidence": ("trade_frequency",          "Trade frequency (trades/month)"),
    "herding":        ("momentum_follow_rate",     "Momentum follow rate"),
    "disposition":    ("early_exit_winner_rate",   "Early exit winner rate"),
}


def _format_evidence_trades(trade_ids: List[str], trades: List[Dict]) -> str:
    trade_map = {t["trade_id"]: t for t in trades}
    lines = []
    for tid in trade_ids[:5]:
        t = trade_map.get(tid)
        if not t:
            continue
        pnl_sign = "+" if t["realized_pnl"] >= 0 else ""
        lines.append(
            f"  {t['symbol']} | entry {t['entry_date']} → exit {t['exit_date']} "
            f"| {t['holding_days']}d | P&L {pnl_sign}{t['realized_pnl']:.0f}"
        )
    return "\n".join(lines) or "  No specific trades identified"


def _rule_based_explanation(
    bias_name: str,
    metric_value: float,
    population_median: float,
    evidence_trades: List[str],
    risk_pct: float,
    trades: List[Dict],
) -> Dict[str, str]:
    ratio = metric_value / (population_median + 1e-9)
    evidence_text = _format_evidence_trades(evidence_trades, trades)

    templates = {
        "loss_aversion": {
            "pattern": (f"You held losing trades {ratio:.1f}× longer than the average investor "
                        f"(your ratio: {metric_value:.2f} vs population median {population_median:.2f})."),
            "evidence": f"Trades showing this pattern:\n{evidence_text}",
            "consequence": (f"Holding losing trades too long contributed approximately "
                            f"{risk_pct:.0f}% of your total portfolio drawdown."),
        },
        "overconfidence": {
            "pattern": (f"You traded {metric_value:.1f} times per month on average, "
                        f"which is {ratio:.1f}× more than typical investors ({population_median:.1f}/month)."),
            "evidence": f"Your most frequent trades:\n{evidence_text}",
            "consequence": (f"Excessive trading activity accounts for approximately "
                            f"{risk_pct:.0f}% of your portfolio drawdown through transaction costs and mistimed entries."),
        },
        "herding": {
            "pattern": (f"You followed recent market momentum {metric_value:.0%} of the time "
                        f"(vs population median {population_median:.0%}), indicating a tendency to buy high."),
            "evidence": f"Momentum-driven trades:\n{evidence_text}",
            "consequence": (f"Entering trades after momentum peaks contributed approximately "
                            f"{risk_pct:.0f}% of your total drawdown."),
        },
        "disposition": {
            "pattern": (f"You exited {metric_value:.0%} of your winning trades early, "
                        f"compared to a population median of {population_median:.0%}."),
            "evidence": f"Prematurely closed winning trades:\n{evidence_text}",
            "consequence": (f"Selling winners too early reduced your upside and contributed "
                            f"approximately {risk_pct:.0f}% of your risk-adjusted shortfall."),
        },
    }
    return templates.get(bias_name, {
        "pattern":     f"{bias_name}: metric = {metric_value:.3f}",
        "evidence":    evidence_text,
        "consequence": f"Contributed {risk_pct:.0f}% of drawdown.",
    })


def generate_bias_explanation(
    bias_name: str,
    metric_value: float,
    population_median: float,
    evidence_trades: List[str],
    risk_pct: float,
    trades: List[Dict],
) -> Dict[str, str]:
    """Generate 3-part explanation. Tries LLM first, falls back to rule-based."""
    expected_numbers = {
        "metric": round(metric_value, 2),
        "population": round(population_median, 2),
        "risk_pct": round(risk_pct, 0),
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            from src.xbra.llm.client import get_analysis_llm, invoke
            from src.xbra.llm.prompts import BIAS_EXPLANATION_PROMPT
            evidence_text = _format_evidence_trades(evidence_trades, trades)
            metric_label = BIAS_METRIC_MAP.get(bias_name, (bias_name, bias_name))[1]
            prompt = BIAS_EXPLANATION_PROMPT.format(
                bias_name         = bias_name.replace("_", " ").title(),
                metric_label      = metric_label,
                metric_value      = metric_value,
                population_median = population_median,
                evidence_text     = evidence_text,
                risk_pct          = risk_pct,
            )
            llm = get_analysis_llm()
            response = invoke(llm, prompt)

            # Parse 3 sentences
            explanation = _parse_three_part(response)
            if verify_explanation(explanation, expected_numbers):
                return explanation
            # Failed verification — retry
        except Exception:
            break

    return _rule_based_explanation(
        bias_name, metric_value, population_median, evidence_trades, risk_pct, trades
    )


def _parse_three_part(text: str) -> Dict[str, str]:
    parts = {"pattern": "", "evidence": "", "consequence": ""}
    for key in ["Pattern", "Evidence", "Consequence"]:
        match = re.search(rf"{key}:\s*(.+?)(?=Pattern:|Evidence:|Consequence:|$)", text, re.DOTALL | re.IGNORECASE)
        if match:
            parts[key.lower()] = match.group(1).strip()
    if not any(parts.values()):
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) >= 3:
            parts = {"pattern": sentences[0] + ".", "evidence": sentences[1] + ".", "consequence": sentences[2] + "."}
    return parts


def generate_recommendations(
    bias_summary: str,
    archetype: str,
    dominant_risk: str,
) -> str:
    try:
        from src.xbra.llm.client import get_analysis_llm, invoke
        from src.xbra.llm.prompts import RECOMMENDATIONS_PROMPT
        prompt = RECOMMENDATIONS_PROMPT.format(
            bias_summary=bias_summary, archetype=archetype, dominant_risk=dominant_risk
        )
        return invoke(get_analysis_llm(), prompt)
    except Exception:
        return (
            "1. Set a stop-loss rule: exit any trade that loses more than 7% of its entry value.\n"
            "2. Limit yourself to a maximum of 5 new trades per month to reduce overtrading.\n"
            "3. Use a trading journal to record your reasons before each entry and review weekly.\n"
            "4. Evaluate each trade against SPY performance to separate skill from market luck."
        )


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def explainability_node(state: XBRAStateDict) -> dict:
    from src.xbra.schemas import (
        BehaviorAgentOutput, ExplanationOutput, FusedBiasVector,
        InvestorProfile, RiskAgentOutput, StrategyAgentOutput,
    )

    profile_dict  = state.get("investor_profile")
    behavior_dict = state.get("behavior_output")
    risk_dict     = state.get("risk_output")
    fused_dict    = state.get("fused_bias")
    strategy_dict = state.get("strategy_output")

    if not all([profile_dict, behavior_dict, risk_dict, fused_dict]):
        return {"errors": state.get("errors", []) + ["explainability_node: missing inputs"]}

    profile  = InvestorProfile(**profile_dict)   if isinstance(profile_dict, dict)  else profile_dict
    behavior = BehaviorAgentOutput(**behavior_dict) if isinstance(behavior_dict, dict) else behavior_dict
    risk     = RiskAgentOutput(**risk_dict)      if isinstance(risk_dict, dict)      else risk_dict
    fused    = FusedBiasVector(**fused_dict)     if isinstance(fused_dict, dict)     else fused_dict
    strategy = StrategyAgentOutput(**strategy_dict) if isinstance(strategy_dict, dict) else strategy_dict if strategy_dict else None

    trades = [t.model_dump() for t in profile.trades]

    # Population median features (for explanation context)
    from src.xbra.agents.behavior_agent import _load_population_baseline
    pop = _load_population_baseline()

    BIAS_SCORES = {
        "loss_aversion":  fused.loss_aversion,
        "overconfidence": fused.overconfidence,
        "herding":        fused.herding,
        "disposition":    fused.disposition,
    }

    # Threshold: only explain biases above 0.1 score
    THRESHOLD = 0.10
    per_bias: Dict[str, Dict[str, str]] = {}

    for bias_name, score in BIAS_SCORES.items():
        if score < THRESHOLD:
            continue

        feat_key, _ = BIAS_METRIC_MAP.get(bias_name, (bias_name, bias_name))
        metric_val  = behavior.features.get(feat_key, 0.0)
        pop_med     = pop.get(feat_key, metric_val)
        evidence    = behavior.evidencing_trade_ids.get(bias_name, [])
        risk_pct    = risk.r_behavioral_pct * score / (sum(BIAS_SCORES.values()) or 1)

        per_bias[bias_name] = generate_bias_explanation(
            bias_name, metric_val, pop_med, evidence, risk_pct, trades
        )

    # Recommendations
    bias_summary = ", ".join(f"{k} ({v:.2f})" for k, v in BIAS_SCORES.items() if v >= THRESHOLD)
    arch = strategy.archetype.value if strategy else "unknown"
    recs = generate_recommendations(bias_summary, arch, fused.dominant_bias.value)

    output = ExplanationOutput(
        investor_id = profile.investor_id,
        per_bias    = per_bias,
        verified    = True,
        full_report = recs,
    )
    return {"explanation": output.model_dump(), "stage": "explanation_done"}
