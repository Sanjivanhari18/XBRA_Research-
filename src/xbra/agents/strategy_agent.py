"""Phase 3d — Strategy Agent: archetype clustering, consistency, bias adjustment."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config.settings import KMEANS_MAX_ITER, KMEANS_N_INIT, MODELS_DIR, N_STRATEGY_CLUSTERS
from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import StrategyAgentOutput, StrategyArchetype

_KMEANS_PATH = MODELS_DIR / "strategy_kmeans.pkl"
_SCALER_PATH = MODELS_DIR / "strategy_scaler.pkl"

ARCHETYPE_MAP = {
    0: StrategyArchetype.DAY_TRADER,
    1: StrategyArchetype.SWING_TRADER,
    2: StrategyArchetype.BUY_HOLD_DRIFTER,
    3: StrategyArchetype.MOMENTUM_CHASER,
}

# Per-archetype "expected" overconfidence threshold (turnover/trades per month)
ARCHETYPE_NORMS: Dict[str, Dict[str, float]] = {
    "day_trader":       {"trade_frequency": 30.0, "position_size_cv": 0.8},
    "swing_trader":     {"trade_frequency": 8.0,  "position_size_cv": 0.5},
    "buy_hold_drifter": {"trade_frequency": 2.0,  "position_size_cv": 0.3},
    "momentum_chaser":  {"trade_frequency": 10.0, "position_size_cv": 0.6},
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_strategy_features(trades: List[Dict]) -> Dict[str, float]:
    if not trades:
        return {}
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"]  = pd.to_datetime(df["exit_date"])
    df["pos_value"]  = df["quantity"] * df["entry_price"]

    span_days = (df["exit_date"].max() - df["entry_date"].min()).days or 1
    trade_frequency = len(df) / (span_days / 30.0)

    # Sector concentration proxy — use symbol as sector stand-in
    sym_freq = df["symbol"].value_counts(normalize=True)
    herfindahl = float((sym_freq ** 2).sum())

    # Holding-period distribution
    med_hold = float(df["holding_days"].median())
    std_hold = float(df["holding_days"].std()) if len(df) > 1 else 0.0
    p25_hold = float(df["holding_days"].quantile(0.25))
    p75_hold = float(df["holding_days"].quantile(0.75))

    # Turnover rate (value traded / portfolio value proxy)
    avg_pos = df["pos_value"].mean()
    turnover = trade_frequency * avg_pos / 100_000.0

    return {
        "trade_frequency": round(trade_frequency, 3),
        "sector_herfindahl": round(herfindahl, 4),
        "median_holding_days": round(med_hold, 1),
        "std_holding_days": round(std_hold, 1),
        "p25_holding_days": round(p25_hold, 1),
        "p75_holding_days": round(p75_hold, 1),
        "turnover_rate": round(turnover, 4),
        "n_unique_symbols": float(df["symbol"].nunique()),
    }


# ---------------------------------------------------------------------------
# k-means model: population-level training
# ---------------------------------------------------------------------------

def train_or_load_kmeans():
    if _KMEANS_PATH.exists() and _SCALER_PATH.exists():
        with open(_KMEANS_PATH, "rb") as f:
            km = pickle.load(f)
        with open(_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        return km, scaler

    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        raise RuntimeError("scikit-learn not installed")

    from src.xbra.data.registry import DataRegistry
    from src.xbra.ingestion.loaders import load_from_parquet
    from config.settings import SYNTHETIC_DIR

    gt = DataRegistry.ground_truth()
    rows = []
    for inv_id in gt["investor_id"].tolist():
        try:
            profile = load_from_parquet(inv_id, SYNTHETIC_DIR / "trades.parquet")
            trades  = [t.model_dump() for t in profile.trades]
            feat    = extract_strategy_features(trades)
            feat["investor_id"] = inv_id
            rows.append(feat)
        except Exception:
            continue

    if len(rows) < N_STRATEGY_CLUSTERS:
        raise RuntimeError("Not enough investor data to fit k-means")

    df = pd.DataFrame(rows).set_index("investor_id").fillna(0)
    feat_cols = [c for c in df.columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feat_cols])

    km = KMeans(
        n_clusters=N_STRATEGY_CLUSTERS,
        n_init=KMEANS_N_INIT,
        max_iter=KMEANS_MAX_ITER,
        random_state=42,
    )
    km.fit(X)

    with open(_KMEANS_PATH, "wb") as f:
        pickle.dump(km, f)
    with open(_SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return km, scaler


def assign_archetype(strategy_features: Dict[str, float]) -> StrategyArchetype:
    """Assign archetype label using the pre-trained k-means model."""
    try:
        km, scaler = train_or_load_kmeans()
        feat_names = ["trade_frequency", "sector_herfindahl", "median_holding_days",
                      "std_holding_days", "p25_holding_days", "p75_holding_days",
                      "turnover_rate", "n_unique_symbols"]
        row = np.array([[strategy_features.get(f, 0.0) for f in feat_names]])
        row_scaled = scaler.transform(row)
        label = int(km.predict(row_scaled)[0])
        return ARCHETYPE_MAP.get(label % 4, StrategyArchetype.SWING_TRADER)
    except Exception:
        # Rule-based fallback
        freq = strategy_features.get("trade_frequency", 5)
        med  = strategy_features.get("median_holding_days", 10)
        if freq > 20:
            return StrategyArchetype.DAY_TRADER
        if med > 30:
            return StrategyArchetype.BUY_HOLD_DRIFTER
        if strategy_features.get("sector_herfindahl", 0) < 0.15:
            return StrategyArchetype.MOMENTUM_CHASER
        return StrategyArchetype.SWING_TRADER


# ---------------------------------------------------------------------------
# Strategy stability
# ---------------------------------------------------------------------------

def detect_archetype_oscillation(trades: List[Dict], window_months: int = 3) -> bool:
    if not trades:
        return False
    df = pd.DataFrame(trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df = df.sort_values("entry_date")

    span_months = (df["entry_date"].max() - df["entry_date"].min()).days / 30
    if span_months < window_months * 2:
        return False

    archetypes = []
    for i in range(0, int(span_months), window_months):
        start = df["entry_date"].min() + pd.DateOffset(months=i)
        end   = start + pd.DateOffset(months=window_months)
        window = df[(df["entry_date"] >= start) & (df["entry_date"] < end)]
        if len(window) < 5:
            continue
        feat = extract_strategy_features(window.to_dict("records"))
        arc  = assign_archetype(feat)
        archetypes.append(arc)

    if len(archetypes) < 2:
        return False
    return len(set(a.value for a in archetypes)) > 1


# ---------------------------------------------------------------------------
# Strategy-adjusted bias severity
# ---------------------------------------------------------------------------

def adjust_bias_for_strategy(
    bias_scores: Dict[str, float],
    archetype: StrategyArchetype,
    strategy_features: Dict[str, float],
) -> Dict[str, float]:
    """Scale down biases that are expected for the investor's archetype."""
    norms = ARCHETYPE_NORMS.get(archetype.value, {})
    adjusted = dict(bias_scores)

    # Overconfidence: penalise less if high trade_frequency is normal for archetype
    expected_freq = norms.get("trade_frequency", 8.0)
    actual_freq   = strategy_features.get("trade_frequency", 8.0)
    if actual_freq <= expected_freq * 1.5:
        adjusted["overconfidence"] = adjusted.get("overconfidence", 0) * 0.6

    return {k: round(float(v), 4) for k, v in adjusted.items()}


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def strategy_node(state: XBRAStateDict) -> dict:
    from src.xbra.schemas import BehaviorAgentOutput, InvestorProfile, StrategyAgentOutput

    profile_dict  = state.get("investor_profile")
    behavior_dict = state.get("behavior_output")
    if not profile_dict:
        return {"errors": state.get("errors", []) + ["strategy_node: no investor_profile"]}

    profile  = InvestorProfile(**profile_dict) if isinstance(profile_dict, dict) else profile_dict
    behavior = BehaviorAgentOutput(**behavior_dict) if isinstance(behavior_dict, dict) else behavior_dict if behavior_dict else None

    trades = [t.model_dump() for t in profile.trades]
    feat   = extract_strategy_features(trades)
    arch   = assign_archetype(feat)
    osc    = detect_archetype_oscillation(trades)

    raw_bias_scores = {}
    if behavior:
        raw_bias_scores = {
            "loss_aversion": behavior.loss_aversion_score,
            "overconfidence": behavior.overconfidence_score,
            "herding": behavior.herding_score,
            "disposition": behavior.disposition_score,
        }
    adj_biases = adjust_bias_for_strategy(raw_bias_scores, arch, feat)

    # LLM summary
    llm_summary = None
    try:
        from src.xbra.llm.client import get_analysis_llm, invoke
        from src.xbra.llm.prompts import STRATEGY_CLASSIFICATION_PROMPT, parse_strategy_response
        prompt = STRATEGY_CLASSIFICATION_PROMPT.format(
            turnover_rate=feat.get("turnover_rate", 0),
            sector_herfindahl=feat.get("sector_herfindahl", 0),
            median_holding_days=feat.get("median_holding_days", 0),
            archetype_oscillation=osc,
            cluster_label=arch.value,
        )
        llm = get_analysis_llm()
        resp = invoke(llm, prompt)
        parsed = parse_strategy_response(resp)
        llm_summary = parsed.get("description")
    except Exception:
        pass

    output = StrategyAgentOutput(
        investor_id             = profile.investor_id,
        archetype               = arch,
        consistency_flag        = not osc,
        turnover_rate           = feat.get("turnover_rate", 0.0),
        sector_herfindahl       = feat.get("sector_herfindahl", 0.0),
        archetype_oscillation   = osc,
        strategy_adjusted_biases= adj_biases,
        llm_summary             = llm_summary,
    )
    return {"strategy_output": output.model_dump(), "stage": "strategy_done"}
