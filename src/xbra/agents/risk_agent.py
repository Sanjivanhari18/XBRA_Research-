"""Phase 3c — Risk Agent: metrics, XGBoost, SHAP decomposition (paper core)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, SHAP_STABILITY_THRESHOLD, XGBOOST_PARAMS
from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import MarketRegime, RiskAgentOutput

_MODEL_PATH = MODELS_DIR / "risk_xgb.pkl"
_FEATURE_NAMES_PATH = MODELS_DIR / "risk_feature_names.pkl"


# ---------------------------------------------------------------------------
# Risk metrics (no empyrical dependency — pure pandas/numpy)
# ---------------------------------------------------------------------------

def compute_standard_metrics(trades: List[Dict], portfolio_value: float = 100_000.0) -> Dict[str, float]:
    if not trades:
        return {"max_drawdown": 0, "sharpe_ratio": 0, "sortino_ratio": 0, "var_95": 0, "concentration": 0}

    df = pd.DataFrame(trades).sort_values("exit_date")
    df["cum_pnl"] = df["realized_pnl"].cumsum()
    df["portfolio"] = portfolio_value + df["cum_pnl"]
    df["running_max"] = df["portfolio"].cummax()
    df["drawdown"] = (df["portfolio"] - df["running_max"]) / df["running_max"]
    max_drawdown = float(df["drawdown"].min())

    # Daily returns proxy from trade-level P&L
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    daily = df.groupby("exit_date")["realized_pnl"].sum()
    returns = daily / portfolio_value
    mean_r  = returns.mean()
    std_r   = returns.std() or 1e-9

    sharpe = float((mean_r / std_r) * np.sqrt(252)) if std_r > 0 else 0.0

    neg_returns = returns[returns < 0]
    downside_std = neg_returns.std() or 1e-9
    sortino = float((mean_r / downside_std) * np.sqrt(252)) if downside_std > 0 else 0.0

    var_95 = float(np.percentile(returns, 5)) if len(returns) > 1 else 0.0

    # Position concentration: Herfindahl index over symbols
    sym_counts = pd.DataFrame(trades)["symbol"].value_counts(normalize=True)
    concentration = float((sym_counts ** 2).sum())

    return {
        "max_drawdown":  round(max_drawdown, 4),
        "sharpe_ratio":  round(sharpe, 4),
        "sortino_ratio": round(sortino, 4),
        "var_95":        round(var_95, 6),
        "concentration": round(concentration, 4),
    }


# ---------------------------------------------------------------------------
# Feature matrix builder (behavioral + market features per investor)
# ---------------------------------------------------------------------------

def build_feature_row(
    behavior_features: Dict[str, float],
    market_ctx: Dict[str, Dict],
    regime_labels: Dict[str, str],
) -> Dict[str, float]:
    """Combine behavior + market features into one flat feature vector."""
    row = dict(behavior_features)

    # Market aggregate features
    if market_ctx:
        alphas    = [v["alpha"]     for v in market_ctx.values()]
        sym_rets  = [v["sym_return"] for v in market_ctx.values()]
        row["avg_alpha"]          = float(np.mean(alphas))
        row["avg_sym_return"]     = float(np.mean(sym_rets))
        row["pct_negative_alpha"] = float(np.mean([a < 0 for a in alphas]))

    # Regime distribution
    if regime_labels:
        labels = list(regime_labels.values())
        row["pct_bull"]     = labels.count(MarketRegime.BULL)     / len(labels)
        row["pct_bear"]     = labels.count(MarketRegime.BEAR)     / len(labels)
        row["pct_sideways"] = labels.count(MarketRegime.SIDEWAYS) / len(labels)
    else:
        row.update({"pct_bull": 0.33, "pct_bear": 0.33, "pct_sideways": 0.34})

    return {k: float(v) for k, v in row.items()}


# ---------------------------------------------------------------------------
# Population-level XGBoost risk model
# ---------------------------------------------------------------------------

def _load_population_features() -> Optional[pd.DataFrame]:
    """Load or compute feature matrix for all investors."""
    cache = MODELS_DIR / "population_features.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    try:
        from src.xbra.data.registry import DataRegistry
        from src.xbra.agents.behavior_agent import engineer_features
        from src.xbra.ingestion.loaders import load_from_parquet
        from config.settings import SYNTHETIC_DIR

        gt = DataRegistry.ground_truth()
        rows = []
        for inv_id in gt["investor_id"].tolist():
            try:
                profile = load_from_parquet(inv_id, SYNTHETIC_DIR / "trades.parquet")
                trades  = [t.model_dump() for t in profile.trades]
                feat    = engineer_features(profile)
                metrics = compute_standard_metrics(trades)
                row     = {**feat, "investor_id": inv_id, "max_drawdown": metrics["max_drawdown"]}
                rows.append(row)
            except Exception:
                continue

        if not rows:
            return None
        df = pd.DataFrame(rows).set_index("investor_id")
        df.to_parquet(cache)
        return df
    except Exception:
        return None


def train_or_load_risk_model():
    """Return (model, feature_names). Trains XGBoost if not cached."""
    if _MODEL_PATH.exists() and _FEATURE_NAMES_PATH.exists():
        with open(_MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(_FEATURE_NAMES_PATH, "rb") as f:
            feat_names = pickle.load(f)
        return model, feat_names

    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise RuntimeError("xgboost not installed — run: pip install xgboost")

    pop_df = _load_population_features()
    if pop_df is None or len(pop_df) < 5:
        raise RuntimeError("Not enough population data to train risk model. Run Phase 0 first.")

    target = "max_drawdown"
    features = [c for c in pop_df.columns if c != target]
    X = pop_df[features].fillna(0)
    y = pop_df[target]

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X, y)

    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(_FEATURE_NAMES_PATH, "wb") as f:
        pickle.dump(features, f)

    return model, features


# ---------------------------------------------------------------------------
# SHAP decomposition
# ---------------------------------------------------------------------------

BEHAVIORAL_FEATURES = {
    "holding_time_asymmetry", "post_loss_reentry_speed", "position_size_cv",
    "trade_frequency", "early_exit_winner_rate", "winner_ratio",
    "avg_gain_magnitude", "avg_loss_magnitude", "n_trades", "momentum_follow_rate",
}
MARKET_FEATURES = {
    "avg_alpha", "avg_sym_return", "pct_negative_alpha",
    "pct_bull", "pct_bear", "pct_sideways",
}


def decompose_risk_with_shap(
    model,
    feature_row: Dict[str, float],
    feature_names: List[str],
) -> Dict[str, Any]:
    try:
        import shap
    except ImportError:
        raise RuntimeError("shap not installed — run: pip install shap")

    X = pd.DataFrame([feature_row])[feature_names].fillna(0)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)[0]

    shap_dict = {feature_names[i]: float(shap_vals[i]) for i in range(len(feature_names))}
    total_abs = sum(abs(v) for v in shap_dict.values()) or 1e-9

    r_behavioral = sum(abs(v) for k, v in shap_dict.items() if k in BEHAVIORAL_FEATURES)
    r_market     = sum(abs(v) for k, v in shap_dict.items() if k in MARKET_FEATURES)
    r_interaction = total_abs - r_behavioral - r_market

    return {
        "r_behavioral_pct":  round(r_behavioral  / total_abs * 100, 1),
        "r_market_pct":      round(r_market      / total_abs * 100, 1),
        "r_interaction_pct": round(max(r_interaction, 0) / total_abs * 100, 1),
        "shap_values":       {k: round(v, 6) for k, v in shap_dict.items()},
    }


# ---------------------------------------------------------------------------
# Regime-conditioned decomposition
# ---------------------------------------------------------------------------

def regime_conditioned_decomp(
    trades: List[Dict],
    behavior_features: Dict[str, float],
    market_ctx: Dict[str, Dict],
    regime_labels: Dict[str, str],
    model,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    regime_decomp: Dict[str, Dict[str, float]] = {}
    for regime in [MarketRegime.BULL, MarketRegime.BEAR, MarketRegime.SIDEWAYS]:
        regime_trade_ids = {tid for tid, r in regime_labels.items() if r == regime}
        if not regime_trade_ids:
            continue
        ctx_subset = {tid: v for tid, v in market_ctx.items() if tid in regime_trade_ids}
        row = build_feature_row(behavior_features, ctx_subset, {t: regime for t in regime_trade_ids})
        try:
            decomp = decompose_risk_with_shap(model, row, feature_names)
            regime_decomp[regime] = {
                "r_behavioral_pct":  decomp["r_behavioral_pct"],
                "r_market_pct":      decomp["r_market_pct"],
                "r_interaction_pct": decomp["r_interaction_pct"],
            }
        except Exception:
            pass
    return regime_decomp


# ---------------------------------------------------------------------------
# Peer benchmarking
# ---------------------------------------------------------------------------

def compute_peer_percentile(investor_drawdown: float) -> float:
    pop_df = _load_population_features()
    if pop_df is None or "max_drawdown" not in pop_df.columns:
        return 50.0
    all_dd = pop_df["max_drawdown"].dropna().tolist()
    below = sum(1 for d in all_dd if d >= investor_drawdown)  # worse drawdown = more negative
    return round(below / len(all_dd) * 100, 1)


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def risk_node(state: XBRAStateDict) -> dict:
    from src.xbra.schemas import BehaviorAgentOutput, InvestorProfile, MarketAgentOutput, RiskAgentOutput

    profile_dict  = state.get("investor_profile")
    behavior_dict = state.get("behavior_output")
    market_dict   = state.get("market_output")

    if not all([profile_dict, behavior_dict, market_dict]):
        return {"errors": state.get("errors", []) + ["risk_node: missing inputs"]}

    profile  = InvestorProfile(**profile_dict)   if isinstance(profile_dict, dict)  else profile_dict
    behavior = BehaviorAgentOutput(**behavior_dict) if isinstance(behavior_dict, dict) else behavior_dict
    market   = MarketAgentOutput(**market_dict)  if isinstance(market_dict, dict)   else market_dict

    trades = [t.model_dump() for t in profile.trades]
    metrics = compute_standard_metrics(trades, profile.portfolio_value)

    try:
        model, feat_names = train_or_load_risk_model()
        feature_row = build_feature_row(
            behavior.features, market.per_trade_context, market.regime_labels
        )
        decomp = decompose_risk_with_shap(model, feature_row, feat_names)
        regime_decomp = regime_conditioned_decomp(
            trades, behavior.features,
            market.per_trade_context, market.regime_labels,
            model, feat_names,
        )
    except Exception as ex:
        decomp = {"r_behavioral_pct": 33.3, "r_market_pct": 33.3, "r_interaction_pct": 33.3, "shap_values": {}}
        regime_decomp = {}

    peer_pct = compute_peer_percentile(metrics["max_drawdown"])

    # LLM summary
    llm_summary = None
    try:
        from src.xbra.llm.client import get_analysis_llm, invoke
        from src.xbra.llm.prompts import RISK_DECOMPOSITION_PROMPT
        regime_text = "\n".join(
            f"  {r}: behavioral={v.get('r_behavioral_pct',0):.0f}% market={v.get('r_market_pct',0):.0f}%"
            for r, v in regime_decomp.items()
        ) or "  Not computed"
        prompt = RISK_DECOMPOSITION_PROMPT.format(
            max_drawdown=metrics["max_drawdown"], sharpe_ratio=metrics["sharpe_ratio"],
            sortino_ratio=metrics["sortino_ratio"], peer_percentile=peer_pct,
            r_behavioral_pct=decomp["r_behavioral_pct"],
            r_market_pct=decomp["r_market_pct"],
            r_interaction_pct=decomp["r_interaction_pct"],
            regime_text=regime_text,
        )
        llm = get_analysis_llm()
        llm_summary = invoke(llm, prompt)
    except Exception:
        pass

    output = RiskAgentOutput(
        investor_id         = profile.investor_id,
        max_drawdown        = metrics["max_drawdown"],
        sharpe_ratio        = metrics["sharpe_ratio"],
        sortino_ratio       = metrics["sortino_ratio"],
        var_95              = metrics["var_95"],
        concentration       = metrics["concentration"],
        r_behavioral_pct    = decomp["r_behavioral_pct"],
        r_market_pct        = decomp["r_market_pct"],
        r_interaction_pct   = decomp["r_interaction_pct"],
        regime_decomposition= regime_decomp,
        peer_percentile     = peer_pct,
        llm_summary         = llm_summary,
        shap_values         = decomp.get("shap_values", {}),
    )
    return {"risk_output": output.model_dump(), "stage": "risk_done"}
