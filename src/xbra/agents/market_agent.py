"""Phase 3b — Market Agent: yfinance OHLCV fetch, technical indicators, HMM regime, FinBERT sentiment.

Per-trade context produced:
  Short-term: RSI-14, MACD(12/26/9), Bollinger-20 (%B), ATR-14, volume z-score
  Long-term:  SMA-50, SMA-200, EMA-50, 52-week high/low proximity
  Regime:     3-state HMM (bull/bear/sideways) fit on 252-day daily returns
  Sentiment:  FinBERT on yfinance headlines; falls back to 0.0 if unavailable

All external calls use safe_external_call so the pipeline degrades gracefully.
Data fetch window: earliest_entry - 400 days → latest_exit + 30 days (per unique symbol).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.xbra.integrations.safe_call import safe_external_call
from src.xbra.orchestrator.state import XBRAStateDict
from src.xbra.schemas import InvestorProfile, MarketAgentOutput, MarketRegime, Trade
from src.xbra.utils.logging import logger


# ---------------------------------------------------------------------------
# OHLCV fetching
# ---------------------------------------------------------------------------

def fetch_ohlcv(symbol: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV from yfinance for one symbol.

    Returns a DataFrame with lowercase column names (open/high/low/close/volume)
    indexed by date, or None if the fetch fails or returns empty data.
    """
    def _do_fetch() -> Optional[pd.DataFrame]:
        import yfinance as yf
        df = yf.download(
            symbol,
            start=str(start),
            end=str(end),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None
        # Flatten MultiIndex that some yfinance versions produce for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        return df

    return safe_external_call(_do_fetch, fallback=None, label=f"yfinance/{symbol}")


# ---------------------------------------------------------------------------
# Technical indicators (via `ta` library)
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append short-term and long-term indicators to a copy of the OHLCV DataFrame.

    Short-term: RSI-14, MACD(12/26/9), Bollinger-20 (%B), ATR-14, volume z-score
    Long-term:  SMA-50, SMA-200, EMA-50, 52-week high proximity, 52-week low proximity

    Requires at least ~30 rows for short-term indicators and ~250 rows for long-term.
    Rows without enough history are filled with NaN.
    """
    import ta

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"].astype(float)

    result = df[["open", "high", "low", "close", "volume"]].copy()

    # Short-term ---------------------------------------------------------------
    result["rsi_14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

    macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    result["macd"]        = macd.macd()
    result["macd_signal"] = macd.macd_signal()
    result["macd_diff"]   = macd.macd_diff()

    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    result["bb_upper"] = bb.bollinger_hband()
    result["bb_lower"] = bb.bollinger_lband()
    result["bb_pband"] = bb.bollinger_pband()   # 0 = at lower band, 1 = at upper band

    result["atr_14"] = ta.volatility.AverageTrueRange(
        high=high, low=low, close=close, window=14
    ).average_true_range()

    # Volume anomaly: z-score vs 20-day rolling window
    vol_mean = vol.rolling(20).mean()
    vol_std  = vol.rolling(20).std().replace(0.0, np.nan)
    result["vol_zscore"] = (vol - vol_mean) / vol_std

    # Long-term ----------------------------------------------------------------
    result["sma_50"]  = ta.trend.SMAIndicator(close=close, window=50).sma_indicator()
    result["sma_200"] = ta.trend.SMAIndicator(close=close, window=200).sma_indicator()
    result["ema_50"]  = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

    # 52-week (252 trading-day) high/low proximity
    rolling_high = close.rolling(252, min_periods=1).max()
    rolling_low  = close.rolling(252, min_periods=1).min()
    result["52wk_high_proximity"] = close / rolling_high  # 1.0 = at all-time 252d high
    result["52wk_low_proximity"]  = close / rolling_low   # higher = further from 52d low

    return result


# ---------------------------------------------------------------------------
# HMM regime detection
# ---------------------------------------------------------------------------

def hmm_regimes(close: pd.Series) -> pd.Series:
    """Fit a 3-state GaussianHMM on daily returns and map states to bull/bear/sideways.

    States are mapped by sorting mean returns: lowest → bear, middle → sideways, highest → bull.
    Falls back to rolling-mean heuristic if hmmlearn is unavailable or fitting fails.
    """
    try:
        from hmmlearn.hmm import GaussianHMM

        returns = close.pct_change().fillna(0.0).values
        vol_5d  = pd.Series(returns).rolling(5).std().fillna(0.0).values
        X = np.column_stack([returns, vol_5d])

        model = GaussianHMM(n_components=3, covariance_type="diag",
                            n_iter=200, tol=1e-4, min_covar=1e-3,
                            init_params="km", params="stmc", random_state=42)
        model.fit(X)

        # Guard: reject model if startprob_ is degenerate (NaN or zero-sum)
        sp = model.startprob_
        if np.isnan(sp).any() or sp.sum() < 0.99:
            raise ValueError("HMM startprob_ degenerate after fit")

        states = model.predict(X)

        # Map integer states → regime labels by mean return of each state.
        # Guard against degenerate states (0 samples) by defaulting their mean to 0.0.
        state_means = {}
        for s in range(3):
            mask = states == s
            state_means[s] = float(X[mask, 0].mean()) if mask.sum() > 0 else 0.0
        sorted_states = sorted(state_means, key=state_means.get)  # ascending
        label_map = {
            sorted_states[0]: MarketRegime.BEAR.value,
            sorted_states[1]: MarketRegime.SIDEWAYS.value,
            sorted_states[2]: MarketRegime.BULL.value,
        }
        return pd.Series([label_map[s] for s in states], index=close.index)

    except Exception as exc:
        logger.warning(f"[MarketAgent] HMM failed, using rolling fallback: {exc}")
        return _rolling_regime_fallback(close)


def _rolling_regime_fallback(close: pd.Series, window: int = 20) -> pd.Series:
    """Heuristic fallback when hmmlearn is unavailable or fitting fails."""
    ret = close.pct_change().rolling(window).mean()
    vol = close.pct_change().rolling(window).std()
    labels = []
    for r, v in zip(ret, vol):
        if pd.isna(r) or pd.isna(v):
            labels.append(MarketRegime.SIDEWAYS.value)
        elif r > 0.005 and v < 0.015:
            labels.append(MarketRegime.BULL.value)
        elif r < -0.005:
            labels.append(MarketRegime.BEAR.value)
        else:
            labels.append(MarketRegime.SIDEWAYS.value)
    return pd.Series(labels, index=close.index)


# ---------------------------------------------------------------------------
# News + FinBERT sentiment
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
            device=-1,   # CPU only
        )
    except Exception:
        _finbert_pipeline = None
    return _finbert_pipeline


def fetch_news_headlines(symbol: str, limit: int = 10) -> List[str]:
    """Fetch recent headlines via yfinance Ticker.news (free, no API key)."""
    def _do_fetch() -> List[str]:
        import yfinance as yf
        news = yf.Ticker(symbol).news or []
        titles = []
        for item in news[:limit]:
            # yfinance >= 0.2.50 nests headline under item["content"]["title"]
            content = item.get("content") or {}
            title = content.get("title") or item.get("title") or ""
            if title:
                titles.append(title)
        return titles

    return safe_external_call(_do_fetch, fallback=[], label=f"yfinance_news/{symbol}") or []


def finbert_sentiment(headlines: List[str]) -> float:
    """Mean FinBERT sentiment in [-1, 1]. Returns 0.0 if headlines empty or model unavailable."""
    if not headlines:
        return 0.0
    pipe = _load_finbert()
    if pipe is None:
        return 0.0
    LABEL_MAP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
    try:
        results = pipe(headlines, truncation=True, max_length=512)
        scores = [LABEL_MAP.get(r["label"].lower(), 0.0) * r["score"] for r in results]
        return float(np.mean(scores)) if scores else 0.0
    except Exception as exc:
        logger.warning(f"[MarketAgent] FinBERT inference failed: {exc}")
        return 0.0


# ---------------------------------------------------------------------------
# Per-trade context assembly
# ---------------------------------------------------------------------------

def _row_as_of(df: pd.DataFrame, target: date) -> Optional[pd.Series]:
    """Return the last row with DatetimeIndex <= target, or None."""
    ts  = pd.Timestamp(target)
    idx = df.index[df.index <= ts]
    return df.loc[idx[-1]] if not idx.empty else None


def _safe_float(row: pd.Series, key: str) -> Optional[float]:
    """Extract a float from a Series row, returning None for NaN."""
    v = row.get(key)
    if v is None:
        return None
    try:
        fv = float(v)
        return round(fv, 4) if not np.isnan(fv) else None
    except (TypeError, ValueError):
        return None


def build_trade_context(
    trade: Trade,
    sym_indicators: pd.DataFrame,
    sym_regimes: pd.Series,
    sentiment: float,
    spy_indicators: Optional[pd.DataFrame],
) -> Tuple[Dict[str, Any], str, bool]:
    """Build a context dict for one trade.

    Returns:
        (context_dict, regime_label, market_explains_flag)
    """
    entry_row  = _row_as_of(sym_indicators, trade.entry_date)
    regime_row = _row_as_of(sym_regimes.rename("r").to_frame(), trade.entry_date)
    regime     = regime_row["r"] if regime_row is not None else MarketRegime.SIDEWAYS.value

    # Symbol return over the holding period
    exit_row   = _row_as_of(sym_indicators, trade.exit_date)
    e_close    = float(entry_row["close"]) if entry_row is not None else np.nan
    x_close    = float(exit_row["close"])  if exit_row  is not None else np.nan
    sym_ret    = (x_close / e_close - 1) if (not np.isnan(e_close) and e_close != 0 and not np.isnan(x_close)) else 0.0

    # SPY return over the same period (benchmark)
    spy_ret = 0.0
    if spy_indicators is not None:
        spy_e = _row_as_of(spy_indicators, trade.entry_date)
        spy_x = _row_as_of(spy_indicators, trade.exit_date)
        if spy_e is not None and spy_x is not None:
            ec, xc = float(spy_e["close"]), float(spy_x["close"])
            spy_ret = (xc / ec - 1) if (ec and ec != 0) else 0.0

    ctx: Dict[str, Any] = {
        "regime":    regime,
        "sentiment": round(sentiment, 4),
        "sym_return": round(sym_ret, 4),
        "spy_return": round(spy_ret, 4),
        "alpha":      round(sym_ret - spy_ret, 4),
    }

    if entry_row is not None:
        ctx.update({
            "rsi_14":               _safe_float(entry_row, "rsi_14"),
            "macd":                 _safe_float(entry_row, "macd"),
            "macd_signal":          _safe_float(entry_row, "macd_signal"),
            "bb_pband":             _safe_float(entry_row, "bb_pband"),
            "atr_14":               _safe_float(entry_row, "atr_14"),
            "vol_zscore":           _safe_float(entry_row, "vol_zscore"),
            "sma_50":               _safe_float(entry_row, "sma_50"),
            "sma_200":              _safe_float(entry_row, "sma_200"),
            "ema_50":               _safe_float(entry_row, "ema_50"),
            "52wk_high_proximity":  _safe_float(entry_row, "52wk_high_proximity"),
            "52wk_low_proximity":   _safe_float(entry_row, "52wk_low_proximity"),
        })
        sma50  = _safe_float(entry_row, "sma_50")
        sma200 = _safe_float(entry_row, "sma_200")
        ctx["entry_above_sma50"]  = (float(entry_row["close"]) > sma50)  if sma50  is not None else None
        ctx["entry_above_sma200"] = (float(entry_row["close"]) > sma200) if sma200 is not None else None

    market_explains = abs(sentiment) > 0.3 or regime == MarketRegime.BEAR.value
    return ctx, regime, market_explains


# ---------------------------------------------------------------------------
# LangGraph node
# ---------------------------------------------------------------------------

def market_node(state: XBRAStateDict) -> dict:
    """Market Agent node: fetches OHLCV + news, computes indicators and regime per trade."""
    profile_dict = state.get("investor_profile")
    if not profile_dict:
        return {"errors": state.get("errors", []) + ["market_node: no investor_profile"]}

    profile: InvestorProfile = (
        InvestorProfile(**profile_dict) if isinstance(profile_dict, dict) else profile_dict
    )
    trades = profile.trades

    if not trades:
        output = MarketAgentOutput(investor_id=profile.investor_id, confidence=0.0)
        return {"market_output": output.model_dump(), "stage": "market_done"}

    # ── 1. Compute fetch window per unique symbol ──────────────────────────────
    sym_lo: Dict[str, date] = {}
    sym_hi: Dict[str, date] = {}
    for t in trades:
        sym_lo[t.symbol] = min(sym_lo.get(t.symbol, t.entry_date), t.entry_date)
        sym_hi[t.symbol] = max(sym_hi.get(t.symbol, t.exit_date),  t.exit_date)

    global_lo = min(sym_lo.values())
    global_hi = max(sym_hi.values())
    fetch_start = global_lo - timedelta(days=400)   # ≈ 252 trading days buffer
    fetch_end   = global_hi + timedelta(days=30)

    # ── 2. Fetch OHLCV per unique symbol + SPY benchmark ─────────────────────
    ohlcv: Dict[str, Optional[pd.DataFrame]] = {}
    for sym in sym_lo:
        ohlcv[sym] = fetch_ohlcv(sym, fetch_start, fetch_end)
    ohlcv["SPY"] = fetch_ohlcv("SPY", fetch_start, fetch_end)

    # ── 3. Compute indicators and HMM regimes ─────────────────────────────────
    sym_indicators: Dict[str, Optional[pd.DataFrame]] = {}
    sym_regimes:    Dict[str, Optional[pd.Series]]    = {}
    for sym, df in ohlcv.items():
        if df is not None and not df.empty:
            try:
                sym_indicators[sym] = compute_indicators(df)
                sym_regimes[sym]    = hmm_regimes(df["close"])
            except Exception as exc:
                logger.warning(f"[MarketAgent] processing failed for {sym}: {exc}")
                sym_indicators[sym] = None
                sym_regimes[sym]    = None
        else:
            sym_indicators[sym] = None
            sym_regimes[sym]    = None

    # ── 4. Fetch news + compute FinBERT sentiment (one per symbol) ────────────
    sym_sentiment: Dict[str, float] = {}
    for sym in sym_lo:
        headlines          = fetch_news_headlines(sym)
        sym_sentiment[sym] = finbert_sentiment(headlines)

    # ── 5. Build per-trade context ────────────────────────────────────────────
    per_trade_context:    Dict[str, Dict[str, Any]] = {}
    regime_labels:        Dict[str, MarketRegime]   = {}
    sentiment_scores_out: Dict[str, float]          = {}
    market_explains_out:  Dict[str, bool]           = {}

    spy_df = sym_indicators.get("SPY")

    for t in trades:
        sym    = t.symbol
        ind_df = sym_indicators.get(sym)
        reg_s  = sym_regimes.get(sym)
        sent   = sym_sentiment.get(sym, 0.0)

        if ind_df is not None and reg_s is not None:
            ctx, regime, explains = build_trade_context(t, ind_df, reg_s, sent, spy_df)
        else:
            ctx     = {"regime": MarketRegime.SIDEWAYS.value, "sentiment": 0.0}
            regime  = MarketRegime.SIDEWAYS.value
            explains = False

        per_trade_context[t.trade_id]    = ctx
        regime_labels[t.trade_id]        = MarketRegime(regime)
        sentiment_scores_out[t.trade_id] = round(sent, 4)
        market_explains_out[t.trade_id]  = explains

    # ── 6. Confidence: fraction of traded symbols with valid OHLCV ───────────
    n_syms = len(sym_lo)
    n_ok   = sum(1 for s in sym_lo if sym_indicators.get(s) is not None)
    confidence = round(n_ok / n_syms if n_syms else 0.0, 3)

    logger.info(
        f"[MarketAgent] {profile.investor_id}: {n_ok}/{n_syms} symbols ok, "
        f"confidence={confidence:.3f}"
    )

    output = MarketAgentOutput(
        investor_id           = profile.investor_id,
        per_trade_context     = per_trade_context,
        regime_labels         = regime_labels,
        sentiment_scores      = sentiment_scores_out,
        market_explains_flags = market_explains_out,
        confidence            = confidence,
    )
    return {"market_output": output.model_dump(), "stage": "market_done"}
