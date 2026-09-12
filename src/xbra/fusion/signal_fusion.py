"""Phase 4 — Signal Fusion Layer: weighted combination + silhouette validation."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import (
    BiasType,
    BehaviorAgentOutput,
    FusedBiasVector,
    MarketAgentOutput,
    StrategyAgentOutput,
)

# BiasType already imported above; also needed inside fusion_node (imported locally for clarity)

FUSION_WEIGHTS = {
    "behavior": 0.60,
    "strategy": 0.25,
    "market":   0.15,
}


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return scores
    max_val = max(abs(v) for v in scores.values()) or 1.0
    return {k: round(float(v) / max_val, 4) for k, v in scores.items()}


# ---------------------------------------------------------------------------
# Weighted fusion
# ---------------------------------------------------------------------------

def weighted_combine(
    behavior: BehaviorAgentOutput,
    market: MarketAgentOutput,
    strategy: StrategyAgentOutput,
) -> Dict[str, float]:
    bw = FUSION_WEIGHTS["behavior"]
    sw = FUSION_WEIGHTS["strategy"]

    behavior_raw = {
        "loss_aversion":  behavior.loss_aversion_score,
        "overconfidence": behavior.overconfidence_score,
        "herding":        behavior.herding_score,
        "disposition":    behavior.disposition_score,
    }

    strategy_adj = strategy.strategy_adjusted_biases or {}

    fused: Dict[str, float] = {}
    for bias in ["loss_aversion", "overconfidence", "herding", "disposition"]:
        b_score = behavior_raw.get(bias, 0.0)
        s_score = strategy_adj.get(bias, b_score)   # strategy adjusts or falls back
        fused[bias] = round(bw * b_score + sw * s_score, 4)

    # Market context dampens scores when market explains the behaviour
    n_market_explains = sum(1 for v in market.market_explains_flags.values() if v)
    n_total = max(len(market.market_explains_flags), 1)
    market_explain_rate = n_market_explains / n_total
    market_dampener = 1.0 - (FUSION_WEIGHTS["market"] * market_explain_rate)

    fused = {k: round(v * market_dampener, 4) for k, v in fused.items()}
    return fused


def pick_dominant_bias(scores: Dict[str, float]) -> BiasType:
    if not scores or all(v < 0.05 for v in scores.values()):
        return BiasType.NEUTRAL
    dominant = max(scores, key=scores.get)  # type: ignore[arg-type]
    bias_map = {
        "loss_aversion": BiasType.LOSS_AVERSE,
        "overconfidence": BiasType.OVERCONFIDENT,
        "herding": BiasType.HERDING,
        "disposition": BiasType.DISPOSITION,
    }
    return bias_map.get(dominant, BiasType.NEUTRAL)


# ---------------------------------------------------------------------------
# Silhouette validation (population-level check)
# ---------------------------------------------------------------------------

def validate_with_silhouette(bias_vectors_df: pd.DataFrame) -> float:
    """Compute silhouette score on the population bias matrix.

    bias_vectors_df: rows = investors, cols = [loss_aversion, overconfidence, herding, disposition]
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        if len(bias_vectors_df) < 6:
            return 0.0

        X = bias_vectors_df[["loss_aversion", "overconfidence", "herding", "disposition"]].fillna(0)
        km = KMeans(n_clusters=4, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = float(silhouette_score(X, labels))
        return round(score, 4)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def fusion_node(state: XBRAStateDict) -> dict:
    from src.xbra.schemas import (
        BehaviorAgentOutput, FusedBiasVector,
        MarketAgentOutput, StrategyAgentOutput,
    )

    behavior_dict = state.get("behavior_output")
    market_dict   = state.get("market_output")
    strategy_dict = state.get("strategy_output")

    if not all([behavior_dict, market_dict, strategy_dict]):
        return {"errors": state.get("errors", []) + ["fusion_node: missing agent outputs"]}

    behavior = BehaviorAgentOutput(**behavior_dict) if isinstance(behavior_dict, dict) else behavior_dict
    market   = MarketAgentOutput(**market_dict)     if isinstance(market_dict, dict)   else market_dict
    strategy = StrategyAgentOutput(**strategy_dict) if isinstance(strategy_dict, dict) else strategy_dict

    fused_scores = weighted_combine(behavior, market, strategy)
    rule_dominant = pick_dominant_bias(fused_scores)
    silhouette    = 0.0   # population silhouette computed in Phase 7

    # Use the XGBoost classifier prediction when available (preferred over rule-based)
    if behavior.predicted_bias and behavior.classifier_confidence > 0.0:
        try:
            predicted = BiasType(behavior.predicted_bias)
        except ValueError:
            predicted = rule_dominant
    else:
        predicted = rule_dominant

    output = FusedBiasVector(
        investor_id      = behavior.investor_id,
        loss_aversion    = fused_scores.get("loss_aversion", 0.0),
        overconfidence   = fused_scores.get("overconfidence", 0.0),
        herding          = fused_scores.get("herding", 0.0),
        disposition      = fused_scores.get("disposition", 0.0),
        dominant_bias    = predicted,
        silhouette_score = silhouette,
        predicted_bias   = predicted,
    )
    return {"fused_bias": output.model_dump(), "stage": "fusion_done"}
