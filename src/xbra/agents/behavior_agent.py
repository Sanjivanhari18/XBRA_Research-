"""Phase 3a — Behavior Agent: feature engineering, deviation scoring, bias detection."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.xbra.llm.client import get_analysis_llm, invoke
from src.xbra.llm.prompts import BEHAVIOR_ANALYSIS_PROMPT, parse_behavior_response
from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import BehaviorAgentOutput, InvestorProfile


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(profile: InvestorProfile) -> Dict[str, float]:
    trades = [t.model_dump() for t in profile.trades]
    if not trades:
        return {}

    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df = df.sort_values("entry_date").reset_index(drop=True)

    winners = df[df["is_winner"]]
    losers  = df[~df["is_winner"]]

    # Holding-time asymmetry
    med_w = winners["holding_days"].median() if not winners.empty else 1
    med_l = losers["holding_days"].median()  if not losers.empty  else 1
    holding_time_asymmetry = med_l / max(med_w, 1)

    # Trade frequency (trades per 30 days)
    span_days = (df["exit_date"].max() - df["entry_date"].min()).days or 1
    trade_frequency = len(df) / (span_days / 30.0)

    # Position-size coefficient of variation
    df["pos_value"] = df["quantity"] * df["entry_price"]
    pos_mean = df["pos_value"].mean()
    position_size_cv = df["pos_value"].std() / pos_mean if pos_mean > 0 else 0.0

    # Early-exit winner rate: winner trades with holding < median_winner_hold
    early_exit_winner_rate = 0.0
    if not winners.empty:
        early_exit_winner_rate = float((winners["holding_days"] < med_w).mean())

    # Post-loss re-entry speed (days from losing exit to next entry)
    losers_sorted = losers.sort_values("exit_date")
    reentry_speeds: List[float] = []
    for _, lt in losers_sorted.iterrows():
        nxt = df[df["entry_date"] > lt["exit_date"]]
        if not nxt.empty:
            reentry_speeds.append((nxt.iloc[0]["entry_date"] - lt["exit_date"]).days)
    post_loss_reentry_speed = float(np.mean(reentry_speeds)) if reentry_speeds else 30.0

    # Winner ratio
    winner_ratio = float(df["is_winner"].mean())

    # Avg gain / loss magnitude (as % return)
    avg_gain = 0.0
    avg_loss = 0.0
    if not winners.empty:
        avg_gain = float(((winners["exit_price"] - winners["entry_price"]) / winners["entry_price"]).mean())
    if not losers.empty:
        avg_loss = float(abs(((losers["exit_price"] - losers["entry_price"]) / losers["entry_price"]).mean()))

    # Momentum follow rate: proxy — did the stock rise in 10d before entry?
    # Approximated here as proportion of trades where exit_price > entry_price
    # (true momentum requires market data — filled in by Market Agent later)
    momentum_follow_rate_proxy = winner_ratio  # placeholder; Market Agent refines

    return {
        "holding_time_asymmetry":  round(holding_time_asymmetry, 4),
        "post_loss_reentry_speed": round(post_loss_reentry_speed, 2),
        "position_size_cv":        round(position_size_cv, 4),
        "trade_frequency":         round(trade_frequency, 3),
        "early_exit_winner_rate":  round(early_exit_winner_rate, 4),
        "winner_ratio":            round(winner_ratio, 4),
        "avg_gain_magnitude":      round(avg_gain, 4),
        "avg_loss_magnitude":      round(avg_loss, 4),
        "n_trades":                len(df),
        "momentum_follow_rate":    round(momentum_follow_rate_proxy, 4),
    }


# ---------------------------------------------------------------------------
# Population baseline (loaded lazily from DataRegistry)
# ---------------------------------------------------------------------------

_pop_baseline: Dict[str, float] | None = None


def _load_population_baseline() -> Dict[str, float]:
    global _pop_baseline
    if _pop_baseline is not None:
        return _pop_baseline

    try:
        from src.xbra.data.registry import DataRegistry
        gt = DataRegistry.ground_truth()
        all_features = []
        for inv_id in gt["investor_id"].tolist():
            try:
                profile = _quick_load_profile(inv_id)
                all_features.append(engineer_features(profile))
            except Exception:
                continue
        if all_features:
            df = pd.DataFrame(all_features)
            _pop_baseline = df.median().to_dict()
        else:
            _pop_baseline = {}
    except Exception:
        _pop_baseline = {}
    return _pop_baseline


def _quick_load_profile(investor_id: str) -> InvestorProfile:
    from src.xbra.data.registry import DataRegistry
    from src.xbra.ingestion.loaders import load_from_parquet
    from config.settings import SYNTHETIC_DIR
    return load_from_parquet(investor_id, SYNTHETIC_DIR / "trades.parquet")


# ---------------------------------------------------------------------------
# Deviation scoring
# ---------------------------------------------------------------------------

def compute_deviation_scores(
    features: Dict[str, float],
    pop_baseline: Dict[str, float],
) -> Dict[str, float]:
    """Z-score each feature against population median (MAD-based robust z-score)."""
    scores: Dict[str, float] = {}
    for feat, val in features.items():
        if feat not in pop_baseline or pop_baseline[feat] == 0:
            scores[f"{feat}_dev"] = 0.0
            continue
        deviation = (val - pop_baseline[feat]) / (pop_baseline[feat] + 1e-9)
        scores[f"{feat}_dev"] = round(float(np.clip(deviation, -5, 5)), 4)
    return scores


# ---------------------------------------------------------------------------
# Rule-based bias scoring
# ---------------------------------------------------------------------------

# (feature_deviation_key, direction, threshold) → (bias, weight)
BIAS_RULES: List[tuple] = [
    ("holding_time_asymmetry_dev",  "positive", 0.3,  "loss_aversion",  0.80),
    ("holding_time_asymmetry_dev",  "positive", 0.3,  "disposition",    0.50),
    ("early_exit_winner_rate_dev",  "positive", 0.2,  "disposition",    0.85),
    ("early_exit_winner_rate_dev",  "positive", 0.2,  "loss_aversion",  0.40),
    ("position_size_cv_dev",        "positive", 0.3,  "overconfidence", 0.70),
    ("trade_frequency_dev",         "positive", 0.3,  "overconfidence", 0.80),
    ("post_loss_reentry_speed_dev", "negative", -0.2, "loss_aversion",  0.60),
    ("momentum_follow_rate_dev",    "positive", 0.2,  "herding",        0.90),
    ("winner_ratio_dev",            "negative", -0.2, "overconfidence", 0.40),
]


def apply_bias_rules(deviation_scores: Dict[str, float]) -> Dict[str, float]:
    accumulator: Dict[str, List[float]] = {
        "loss_aversion": [], "overconfidence": [], "herding": [], "disposition": []
    }
    for feat_key, direction, threshold, bias, weight in BIAS_RULES:
        val = deviation_scores.get(feat_key, 0.0)
        triggered = (direction == "positive" and val >= threshold) or \
                    (direction == "negative" and val <= threshold)
        if triggered:
            # Scale 0→1 based on how far past threshold
            intensity = min(abs(val - threshold) / (abs(threshold) + 1e-9), 1.0)
            accumulator[bias].append(intensity * weight)

    result: Dict[str, float] = {}
    for bias, scores in accumulator.items():
        result[bias] = round(float(np.mean(scores)) if scores else 0.0, 4)
    return result


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence(bias_scores: Dict[str, float], n_trades: int) -> Dict[str, float]:
    """Confidence = bias score × data sufficiency factor."""
    data_factor = min(n_trades / 100.0, 1.0)  # ramps to 1.0 at 100+ trades
    return {k: round(v * data_factor, 4) for k, v in bias_scores.items()}


# ---------------------------------------------------------------------------
# Evidencing trade IDs
# ---------------------------------------------------------------------------

def find_evidencing_trades(profile: InvestorProfile, bias_scores: Dict[str, float]) -> Dict[str, List[str]]:
    """Return the trade IDs that most strongly exemplify each bias."""
    evidence: Dict[str, List[str]] = {}
    df = pd.DataFrame([t.model_dump() for t in profile.trades])
    if df.empty:
        return evidence

    losers  = df[~df["is_winner"]].sort_values("holding_days", ascending=False)
    winners = df[df["is_winner"]].sort_values("holding_days")

    if bias_scores.get("loss_aversion", 0) > 0.1:
        evidence["loss_aversion"] = losers["trade_id"].head(5).tolist()

    if bias_scores.get("disposition", 0) > 0.1:
        evidence["disposition"] = winners["trade_id"].head(5).tolist()

    if bias_scores.get("overconfidence", 0) > 0.1:
        large = df.nlargest(5, "quantity")
        evidence["overconfidence"] = large["trade_id"].tolist()

    if bias_scores.get("herding", 0) > 0.1:
        evidence["herding"] = df["trade_id"].head(5).tolist()

    return evidence


# ---------------------------------------------------------------------------
# LLM interpretation step
# ---------------------------------------------------------------------------

def llm_interpret(
    investor_id: str,
    features: Dict[str, float],
    pop_baseline: Dict[str, float],
) -> Dict[str, Any]:
    try:
        llm = get_analysis_llm()
        prompt = BEHAVIOR_ANALYSIS_PROMPT.format(
            investor_id               = investor_id,
            holding_time_asymmetry    = features.get("holding_time_asymmetry", 0),
            post_loss_reentry_speed   = features.get("post_loss_reentry_speed", 0),
            position_size_cv          = features.get("position_size_cv", 0),
            trade_frequency           = features.get("trade_frequency", 0),
            early_exit_winner_rate    = features.get("early_exit_winner_rate", 0),
            momentum_follow_rate      = features.get("momentum_follow_rate", 0),
            pop_holding_asymmetry     = pop_baseline.get("holding_time_asymmetry", 1.0),
            pop_trade_frequency       = pop_baseline.get("trade_frequency", 5.0),
            pop_early_exit_rate       = pop_baseline.get("early_exit_winner_rate", 0.3),
        )
        response = invoke(llm, prompt)
        return parse_behavior_response(response)
    except Exception:
        return {"dominant_bias": "neutral", "bias_scores": {}, "reasoning": "LLM unavailable"}


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def behavior_node(state: XBRAStateDict) -> dict:
    from src.xbra.schemas import BehaviorAgentOutput, InvestorProfile

    profile_dict = state.get("investor_profile")
    if not profile_dict:
        return {"errors": state.get("errors", []) + ["behavior_node: no investor_profile in state"]}

    profile = InvestorProfile(**profile_dict) if isinstance(profile_dict, dict) else profile_dict
    investor_id = profile.investor_id

    features     = engineer_features(profile)
    pop_baseline = _load_population_baseline()
    dev_scores   = compute_deviation_scores(features, pop_baseline)
    bias_scores  = apply_bias_rules(dev_scores)
    confidence   = compute_confidence(bias_scores, features.get("n_trades", 0))
    evidence     = find_evidencing_trades(profile, bias_scores)
    llm_result   = llm_interpret(investor_id, features, pop_baseline)

    # Merge LLM bias scores with rule-based (LLM refines if available)
    if llm_result.get("bias_scores"):
        for b, s in llm_result["bias_scores"].items():
            if b in bias_scores:
                bias_scores[b] = round((bias_scores[b] + float(s)) / 2, 4)

    output = BehaviorAgentOutput(
        investor_id          = investor_id,
        loss_aversion_score  = bias_scores.get("loss_aversion", 0.0),
        overconfidence_score = bias_scores.get("overconfidence", 0.0),
        herding_score        = bias_scores.get("herding", 0.0),
        disposition_score    = bias_scores.get("disposition", 0.0),
        confidence_per_bias  = confidence,
        evidencing_trade_ids = evidence,
        llm_summary          = llm_result.get("reasoning"),
        features             = features,
    )

    return {"behavior_output": output.model_dump(), "stage": "behavior_done"}
