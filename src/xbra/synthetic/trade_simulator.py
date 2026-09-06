"""Trade simulator — generates realistic trade sequences for one investor.

Given:
  - A BiasProfile (controls decision-making parameters)
  - Real OHLCV price data for the trade universe
  - A date range

Produces a list of Trade objects whose P&L is calculated from real prices
but whose timing/sizing/exit decisions follow the bias profile.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd

from src.xbra.schemas import Trade
from src.xbra.synthetic.bias_profiles import BiasProfile


class TradeSimulator:
    def __init__(
        self,
        investor_id: str,
        profile: BiasProfile,
        prices: pd.DataFrame,      # columns = tickers, index = date (business days)
        start_date: date,
        end_date: date,
        n_trades_target: int,
        rng: np.random.Generator,
    ) -> None:
        self.investor_id = investor_id
        self.profile = profile
        self.prices = prices
        self.start_date = start_date
        self.end_date = end_date
        self.n_trades_target = n_trades_target
        self.rng = rng

        self._trading_dates: list[date] = [
            d.date() for d in prices.index if start_date <= d.date() <= end_date
        ]
        self._tickers = list(prices.columns)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> List[Trade]:
        trades: List[Trade] = []
        cursor = self._trading_dates[0]

        while len(trades) < self.n_trades_target and cursor < self.end_date:
            symbol = self._pick_symbol(cursor)
            if symbol is None:
                break

            entry_date = cursor
            entry_price = self._get_price(symbol, entry_date)
            if entry_price is None:
                cursor = self._advance_cursor(cursor)
                continue

            # Determine exit date from holding-day distribution
            exit_date, is_winner_at_exit = self._determine_exit(
                symbol, entry_date, entry_price
            )
            exit_price = self._get_price(symbol, exit_date)
            if exit_price is None:
                cursor = self._advance_cursor(cursor)
                continue

            holding_days = (exit_date - entry_date).days
            realized_pnl_pct = (exit_price - entry_price) / entry_price
            is_winner = realized_pnl_pct > 0

            quantity = max(1, int(
                self.rng.uniform(*self.profile.position_size_range) * 100_000 / entry_price
            ))
            realized_pnl = realized_pnl_pct * entry_price * quantity

            trades.append(Trade(
                trade_id    = f"{self.investor_id}_{uuid.uuid4().hex[:8]}",
                investor_id = self.investor_id,
                symbol      = symbol,
                entry_date  = entry_date,
                exit_date   = exit_date,
                entry_price = round(entry_price, 4),
                exit_price  = round(exit_price, 4),
                quantity    = quantity,
                realized_pnl= round(realized_pnl, 2),
                holding_days= holding_days,
                is_winner   = is_winner,
            ))

            # Next trade starts after a gap
            gap_days = int(self.rng.integers(*self.profile.inter_trade_days_range))
            cursor = self._advance_cursor(exit_date, gap_days)

        return trades

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_symbol(self, as_of: date) -> str | None:
        if not self._tickers:
            return None

        # Herding: with momentum_follow_prob, pick a recently-up stock
        if self.rng.random() < self.profile.momentum_follow_prob:
            return self._pick_momentum_stock(as_of)

        return str(self.rng.choice(self._tickers))

    def _pick_momentum_stock(self, as_of: date) -> str:
        """Pick the ticker with the highest 10-day return up to as_of."""
        past = as_of - timedelta(days=14)
        eligible = [
            t for t in self._tickers
            if self._get_price(t, past) is not None and self._get_price(t, as_of) is not None
        ]
        if not eligible:
            return str(self.rng.choice(self._tickers))

        returns = {
            t: (self._get_price(t, as_of) / self._get_price(t, past) - 1)  # type: ignore[operator]
            for t in eligible
        }
        # Among top-3 by momentum, pick randomly (adds noise)
        top3 = sorted(returns, key=returns.get, reverse=True)[:3]  # type: ignore[arg-type]
        return str(self.rng.choice(top3))

    def _determine_exit(
        self, symbol: str, entry_date: date, entry_price: float
    ) -> tuple[date, bool]:
        """Simulate day-by-day holding and apply profile exit rules.

        Returns (exit_date, is_winner).
        """
        p = self.profile
        # Draw a maximum holding window
        max_hold_days = int(self.rng.integers(60, 180))
        cursor = entry_date + timedelta(days=1)
        last_valid = entry_date

        for _ in range(max_hold_days):
            if cursor > self.end_date:
                break
            price = self._get_price(symbol, cursor)
            if price is None:
                cursor += timedelta(days=1)
                continue

            ret = (price - entry_price) / entry_price
            last_valid = cursor

            is_winner = ret > 0

            # --- Exit rules based on bias profile ---

            # Profit-take: exit winners that exceeded threshold
            if is_winner and ret >= p.profit_take_threshold:
                # Disposition / loss-averse: high prob of exiting winners early
                if self.rng.random() < p.early_exit_winner_prob:
                    return cursor, True

            # Loss-cut: exit losers that breached stop-loss
            if not is_winner and p.loss_cut_threshold is not None:
                if ret <= p.loss_cut_threshold:
                    return cursor, False

            # Bias-specific holding limits
            if is_winner:
                max_w = p.winner_hold_days[1]
                if (cursor - entry_date).days >= max_w:
                    return cursor, True
            else:
                max_l = p.loser_hold_days[1]
                if (cursor - entry_date).days >= max_l:
                    return cursor, False

            cursor += timedelta(days=1)

        return last_valid, (self._get_price(symbol, last_valid) or entry_price) > entry_price

    def _get_price(self, symbol: str, as_of: date) -> float | None:
        """Return the closing price for a symbol on or before as_of."""
        if symbol not in self.prices.columns:
            return None
        series = self.prices[symbol].dropna()
        # as-of: last available price ≤ as_of
        candidates = series[series.index.date <= as_of]  # type: ignore[attr-defined]
        if candidates.empty:
            return None
        return float(candidates.iloc[-1])

    def _advance_cursor(self, from_date: date, days: int = 1) -> date:
        target = from_date + timedelta(days=days)
        valids = [d for d in self._trading_dates if d >= target]
        return valids[0] if valids else self.end_date
