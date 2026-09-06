"""All LLM prompt templates for XBRA — single source of truth.

Templates use Python str.format() style placeholders.
Each template is paired with a parse_* function that extracts structured
output from the LLM response string.
"""

from __future__ import annotations

import json
import re


# ---------------------------------------------------------------------------
# Intent Router (llama3.2:3b)
# ---------------------------------------------------------------------------

INTENT_ROUTER_PROMPT = """\
You are an intent classifier for a financial analysis assistant.
Classify the user message into exactly one of these labels:
  full_analysis  — run complete bias + risk + strategy analysis
  bias_only      — only detect behavioural biases
  risk_only      — only compute risk metrics and decomposition
  explain_trade  — explain a specific trade (trade_id will be in the query)
  report         — regenerate the report from existing results

Respond with ONLY the label, nothing else.

User message: {user_query}
Label:"""


def parse_intent(response: str) -> str:
    valid = {"full_analysis", "bias_only", "risk_only", "explain_trade", "report"}
    clean = response.strip().lower().replace("-", "_")
    return clean if clean in valid else "full_analysis"


# ---------------------------------------------------------------------------
# Behavior Agent (llama3.3:70b)
# ---------------------------------------------------------------------------

BEHAVIOR_ANALYSIS_PROMPT = """\
You are a behavioural finance analyst. Based on the computed metrics below,
assess the investor's behavioural biases. Be precise and cite the numbers.

Investor ID: {investor_id}
Holding-time asymmetry (loser/winner ratio): {holding_time_asymmetry:.2f}
Post-loss re-entry speed (days): {post_loss_reentry_speed:.1f}
Position-size coefficient of variation: {position_size_cv:.2f}
Trade frequency (trades/month): {trade_frequency:.1f}
Early exit winner rate: {early_exit_winner_rate:.2%}
Momentum follow rate: {momentum_follow_rate:.2%}

Population medians:
  holding_time_asymmetry: {pop_holding_asymmetry:.2f}
  trade_frequency:        {pop_trade_frequency:.1f}
  early_exit_winner_rate: {pop_early_exit_rate:.2%}

Respond in JSON with exactly this structure:
{{
  "dominant_bias": "<loss_averse|overconfident|herding|disposition|mixed|neutral>",
  "bias_scores": {{
    "loss_aversion": <0.0-1.0>,
    "overconfidence": <0.0-1.0>,
    "herding": <0.0-1.0>,
    "disposition": <0.0-1.0>
  }},
  "reasoning": "<one paragraph, cite specific numbers>"
}}"""


def parse_behavior_response(response: str) -> dict:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {"dominant_bias": "neutral", "bias_scores": {}, "reasoning": response}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"dominant_bias": "neutral", "bias_scores": {}, "reasoning": response}


# ---------------------------------------------------------------------------
# Market Agent (llama3.3:70b)
# ---------------------------------------------------------------------------

MARKET_CONTEXT_PROMPT = """\
Summarize the market conditions that surrounded the investor's most significant trades.

Market regime during analysis window: {regime_summary}
Average sentiment score for traded symbols: {avg_sentiment:.3f} (range -1 to +1)
Trades flagged for sentiment proximity (within 48h of sentiment spike): {n_sentiment_flagged}
Cross-asset correlation with SPY: {spy_correlation:.2f}

Flagged trades:
{flagged_trades_text}

In 2-3 sentences, explain whether market conditions could explain some of the
investor's behaviour, or whether their decisions were largely market-independent."""


# ---------------------------------------------------------------------------
# Risk Agent (llama3.3:70b)
# ---------------------------------------------------------------------------

RISK_DECOMPOSITION_PROMPT = """\
Explain the following risk decomposition to a non-technical retail investor.
Use plain English. Avoid jargon.

Portfolio statistics:
  Max drawdown:       {max_drawdown:.1%}
  Sharpe ratio:       {sharpe_ratio:.2f}
  Sortino ratio:      {sortino_ratio:.2f}
  Peer percentile:    {peer_percentile:.0f}th (higher = riskier than peers)

Risk attribution:
  Behavioural factors: {r_behavioral_pct:.0f}% of drawdown
  Market factors:      {r_market_pct:.0f}% of drawdown
  Interaction effects: {r_interaction_pct:.0f}% of drawdown

Regime breakdown:
{regime_text}

Write 2-3 sentences suitable for a retail investor with no finance background."""


# ---------------------------------------------------------------------------
# Strategy Agent (llama3.3:70b)
# ---------------------------------------------------------------------------

STRATEGY_CLASSIFICATION_PROMPT = """\
Based on the trading pattern analysis below, classify this investor's implicit
trading strategy and describe it in plain English.

Turnover rate (trades/month):        {turnover_rate:.1f}
Sector concentration (Herfindahl):   {sector_herfindahl:.3f}  (1.0 = fully concentrated)
Median holding period (days):        {median_holding_days:.0f}
Archetype oscillation detected:      {archetype_oscillation}
k-means assigned cluster:            {cluster_label}

Respond in JSON:
{{
  "archetype": "<day_trader|swing_trader|buy_hold_drifter|momentum_chaser>",
  "description": "<one sentence describing the investor's implicit strategy>",
  "consistency": <true|false>
}}"""


def parse_strategy_response(response: str) -> dict:
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        return {"archetype": "swing_trader", "description": response, "consistency": True}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"archetype": "swing_trader", "description": response, "consistency": True}


# ---------------------------------------------------------------------------
# Explainability Layer (llama3.3:70b)
# ---------------------------------------------------------------------------

BIAS_EXPLANATION_PROMPT = """\
You are writing a plain-English explanation for a retail investor.
Do NOT use finance jargon. Do NOT invent any statistics.

Bias: {bias_name}
Key metric: {metric_label} = {metric_value} (population median: {population_median})
Evidence trades (ID, symbol, entry, exit, holding days, P&L):
{evidence_text}
Risk contribution: {risk_pct:.0f}% of total portfolio drawdown

Write EXACTLY 3 sentences labelled Pattern, Evidence, and Consequence.
Use ONLY the numbers given above.

Pattern: [describe what the metric shows about behaviour]
Evidence: [name the specific trades and dates]
Consequence: [state the portfolio impact in % terms]"""


RECOMMENDATIONS_PROMPT = """\
Based on this investor's behavioural analysis, write 4 specific, actionable
recommendations a retail investor can follow. No jargon.

Detected biases: {bias_summary}
Strategy archetype: {archetype}
Dominant risk factor: {dominant_risk}

Write a numbered list of 4 recommendations. Each: 1 sentence, concrete, positive tone."""
