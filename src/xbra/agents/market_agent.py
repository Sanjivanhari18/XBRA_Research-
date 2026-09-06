"""Phase 3b — Market Agent: regime classification, sentiment, context alignment."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import MarketAgentOutput, MarketRegime


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regime(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling regime: bull / bear / sideways based on return + volatility."""
    ret  = close.pct_change().rolling(window).mean()
    vol  = close.pct_change().rolling(window).std()

    def label(r: float, v: float) -> str:
        if pd.isna(r) or pd.isna(v):
            return MarketRegime.SIDEWAYS.value
        if r > 0.005 and v < 0.015:
            return MarketRegime.BULL.value
        if r < -0.005:
            return MarketRegime.BEAR.value
        return MarketRegime.SIDEWAYS.value

    return pd.Series(
        [label(r, v) for r, v in zip(ret, vol)],
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Sentiment (FinBERT or price-momentum proxy)
# ---------------------------------------------------------------------------

_finbert_pipeline = None


def _load_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        _finbert_pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,   # CPU
        )
    except Exception:
        _finbert_pipeline = None
    return _finbert_pipeline


def run_finbert_sentiment(headlines: List[str]) -> List[float]:
    """Return [-1, 1] sentiment score per headline.

    Falls back to 0.0 (neutral) if FinBERT is not available.
    """
    pipe = _load_finbert()
    if pipe is None or not headlines:
        return [0.0] * len(headlines)

    LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    batch_size = 32
    scores: List[float] = []
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i: i + batch_size]
        try:
            results = pipe(batch, truncation=True, max_length=512)
            for r in results:
                scores.append(LABEL_MAP.get(r["label"].lower(), 0.0) * r["score"])
        except Exception:
            scores.extend([0.0] * len(batch))
    return scores


def _price_momentum_sentiment(prices: pd.DataFrame, symbols: List[str], as_of: pd.Timestamp) -> float:
    """Proxy: 5-day return of traded symbols as sentiment signal (no headlines needed)."""
    past = as_of - timedelta(days=7)
    vals = []
    for sym in symbols:
        if sym not in prices.columns:
            continue
        seg = prices[sym].dropna()
        seg_past  = seg[seg.index <= past]
        seg_now   = seg[seg.index <= as_of]
        if seg_past.empty or seg_now.empty:
            continue
        vals.append((seg_now.iloc[-1] - seg_past.iloc[-1]) / (seg_past.iloc[-1] + 1e-9))
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Context alignment (as-of join)
# ---------------------------------------------------------------------------

def align_market_context(
    trades: List[Dict],
    prices: pd.DataFrame,
    spy_prices: pd.Series,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, float], Dict[str, bool]]:
    """For each trade, attach regime label, sentiment, and context flags.

    Returns:
        per_trade_context, regime_labels, sentiment_scores, market_explains_flags
    """
    spy_regime = classify_regime(spy_prices)

    per_trade_context: Dict[str, Dict]  = {}
    regime_labels:     Dict[str, str]   = {}
    sentiment_scores:  Dict[str, float] = {}
    market_explains:   Dict[str, bool]  = {}

    traded_symbols = list({t["symbol"] for t in trades})

    for t in trades:
        tid     = t["trade_id"]
        sym     = t["symbol"]
        e_ts    = pd.Timestamp(t["entry_date"])

        # Regime at entry
        reg_idx = spy_regime.index[spy_regime.index <= e_ts]
        regime  = str(spy_regime.loc[reg_idx[-1]]) if not reg_idx.empty else MarketRegime.SIDEWAYS.value
        regime_labels[tid] = regime

        # Sentiment proxy (no Kaggle dataset wired yet — use price momentum)
        sent = _price_momentum_sentiment(prices, [sym], e_ts)
        sentiment_scores[tid] = round(sent, 4)

        # Context flag: trade entered during high sentiment spike (|sent| > 0.05)
        market_explains[tid] = abs(sent) > 0.05

        # Symbol return during holding period
        x_ts = pd.Timestamp(t["exit_date"])
        if sym in prices.columns:
            seg  = prices[sym].dropna()
            p_e  = seg[seg.index <= e_ts]
            p_x  = seg[seg.index <= x_ts]
            sym_ret = (p_x.iloc[-1] / p_e.iloc[-1] - 1) if (not p_e.empty and not p_x.empty) else 0.0
        else:
            sym_ret = 0.0

        # SPY return during same holding period
        spy_e = spy_prices[spy_prices.index <= e_ts]
        spy_x = spy_prices[spy_prices.index <= x_ts]
        spy_ret = (spy_x.iloc[-1] / spy_e.iloc[-1] - 1) if (not spy_e.empty and not spy_x.empty) else 0.0

        per_trade_context[tid] = {
            "regime":        regime,
            "sentiment":     sent,
            "sym_return":    round(sym_ret, 4),
            "spy_return":    round(spy_ret, 4),
            "alpha":         round(sym_ret - spy_ret, 4),
        }

    return per_trade_context, regime_labels, sentiment_scores, market_explains


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def market_node(state: XBRAStateDict) -> dict:
    from src.xbra.data.registry import DataRegistry
    from src.xbra.schemas import InvestorProfile, MarketAgentOutput

    profile_dict = state.get("investor_profile")
    if not profile_dict:
        return {"errors": state.get("errors", []) + ["market_node: no investor_profile"]}

    profile = InvestorProfile(**profile_dict) if isinstance(profile_dict, dict) else profile_dict
    trades = [t.model_dump() for t in profile.trades]

    try:
        prices = DataRegistry.prices()
        spy = prices["SPY"] if "SPY" in prices.columns else prices.iloc[:, 0]
    except Exception:
        spy = pd.Series(dtype=float)
        prices = pd.DataFrame()

    ctx, regimes, sentiments, explains = align_market_context(trades, prices, spy)

    # Market LLM summary (best-effort)
    llm_summary = None
    try:
        from src.xbra.llm.client import get_analysis_llm, invoke
        from src.xbra.llm.prompts import MARKET_CONTEXT_PROMPT
        regime_counts = pd.Series(list(regimes.values())).value_counts().to_dict()
        regime_summary = ", ".join(f"{k}: {v} trades" for k, v in regime_counts.items())
        avg_sent = float(np.mean(list(sentiments.values()))) if sentiments else 0.0
        n_flagged = sum(explains.values())
        flagged_text = "\n".join(
            f"  {tid}: sym={ctx[tid]['sym_return']:.1%}, spy={ctx[tid]['spy_return']:.1%}"
            for tid, flag in list(explains.items())[:5] if flag
        ) or "None"
        prompt = MARKET_CONTEXT_PROMPT.format(
            regime_summary=regime_summary, avg_sentiment=avg_sent,
            n_sentiment_flagged=n_flagged, spy_correlation=0.5,
            flagged_trades_text=flagged_text,
        )
        llm = get_analysis_llm()
        llm_summary = invoke(llm, prompt)
    except Exception:
        pass

    output = MarketAgentOutput(
        investor_id         = profile.investor_id,
        per_trade_context   = ctx,
        regime_labels       = regimes,
        sentiment_scores    = sentiments,
        market_explains_flags= explains,
        llm_summary         = llm_summary,
    )
    return {"market_output": output.model_dump(), "stage": "market_done"}
