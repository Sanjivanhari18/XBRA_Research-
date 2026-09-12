"""Phase 1 — Position Builder: FIFO lot matching to reconstruct closed positions.

Converts individual fill records into position-level Trade objects.

Position logic (per symbol, chronological FIFO):
  BUY  → first covers any open short lots; remainder opens new long lots
  SELL → first closes any open long lots; remainder opens new short lots

P&L conventions (no commissions — retail trader scope):
  Long:  realized_pnl = (exit_price  - entry_price) * qty
  Short: realized_pnl = (entry_price - exit_price)  * qty

Handles:
  - Long positions:  partial fills, averaging down, partial closes
  - Short positions: initial short, covering (partial or full)
  - Mixed transitions: sell-into-short, buy-to-cover into long

Output: List[Trade] (one per matched close event) + List[str] warnings.
"""

from __future__ import annotations

import uuid
from collections import deque
from datetime import date
from typing import List, Tuple

import pandas as pd

from src.xbra.schemas import Trade


def build_positions(
    fills_df: pd.DataFrame,
    investor_id: str,
) -> Tuple[List[Trade], List[str]]:
    """Reconstruct closed positions from individual fills using FIFO lot matching.

    Inputs:
        fills_df: normalised fill DataFrame with columns:
                  fill_id, fill_date, symbol, action, quantity, price
                  (sorted chronologically — normalizer guarantees this)
        investor_id: used to populate Trade.investor_id

    Outputs:
        (closed_positions, warnings)
        closed_positions — one Trade per matched open→close event
        warnings         — unmatched sells, uncovered shorts, or open positions
                           remaining at end of the data window

    Side effects: none. Pure function.
    """
    closed: List[Trade] = []
    warnings: List[str] = []

    for symbol, group in fills_df.groupby("symbol"):
        group = group.sort_values("fill_date").reset_index(drop=True)
        open_longs:  deque = deque()   # lots opened by BUY
        open_shorts: deque = deque()   # lots opened by SELL (short positions)

        for _, row in group.iterrows():
            qty       = int(row["quantity"])
            price     = float(row["price"])
            fill_date = _to_date(row["fill_date"])
            fill_id   = str(row.get("fill_id", uuid.uuid4().hex[:8]))

            if row["action"] == "buy":
                qty_remaining = qty

                # First: cover any open short positions (FIFO)
                if open_shorts and qty_remaining > 0:
                    matched, qty_remaining = _match_lots(open_shorts, qty_remaining)
                    if matched:
                        closed.append(_make_trade(
                            matched_lots=matched,
                            close_price=price,
                            close_date=fill_date,
                            investor_id=investor_id,
                            symbol=str(symbol),
                            side="short",
                        ))

                # Remainder opens a new long lot
                if qty_remaining > 0:
                    open_longs.append({
                        "date":    fill_date,
                        "qty":     qty_remaining,
                        "price":   price,
                        "fill_id": fill_id,
                    })

            elif row["action"] == "sell":
                qty_remaining = qty

                # First: close any open long positions (FIFO)
                if open_longs and qty_remaining > 0:
                    matched, qty_remaining = _match_lots(open_longs, qty_remaining)
                    if matched:
                        closed.append(_make_trade(
                            matched_lots=matched,
                            close_price=price,
                            close_date=fill_date,
                            investor_id=investor_id,
                            symbol=str(symbol),
                            side="long",
                        ))

                # Remainder opens a new short position
                if qty_remaining > 0:
                    open_shorts.append({
                        "date":    fill_date,
                        "qty":     qty_remaining,
                        "price":   price,
                        "fill_id": fill_id,
                    })

        # Surface any positions still open at end of window
        if open_longs:
            remaining = sum(lot["qty"] for lot in open_longs)
            warnings.append(
                f"Open long position at end of window: {symbol} "
                f"{remaining} shares (oldest lot opened {open_longs[0]['date']})"
            )
        if open_shorts:
            remaining = sum(lot["qty"] for lot in open_shorts)
            warnings.append(
                f"Open short position at end of window: {symbol} "
                f"{remaining} shares short (opened {open_shorts[0]['date']})"
            )

    return closed, warnings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_date(val) -> date:
    if isinstance(val, date):
        return val
    return pd.Timestamp(val).date()


def _match_lots(
    open_lots: deque,
    qty_to_close: int,
) -> Tuple[List[dict], int]:
    """Consume lots FIFO until qty_to_close is filled or lots are exhausted.

    Mutates open_lots in-place (partial lots updated, exhausted lots popped).

    Returns:
        (consumed_slices, unmatched_qty)
        unmatched_qty == 0 means the close order was fully matched.
    """
    consumed: List[dict] = []

    while qty_to_close > 0 and open_lots:
        lot  = open_lots[0]
        take = min(lot["qty"], qty_to_close)

        consumed.append({
            "date":  lot["date"],
            "qty":   take,
            "price": lot["price"],
        })

        lot["qty"]  -= take
        qty_to_close -= take

        if lot["qty"] == 0:
            open_lots.popleft()

    return consumed, qty_to_close


def _make_trade(
    matched_lots: List[dict],
    close_price:  float,
    close_date:   date,
    investor_id:  str,
    symbol:       str,
    side:         str,   # "long" or "short"
) -> Trade:
    """Build a Trade (closed position) from matched lot slices and the close event.

    P&L (no commissions):
      long  → (close_price - avg_entry) * qty
      short → (avg_entry - close_price) * qty
    """
    total_qty = sum(lot["qty"] for lot in matched_lots)
    avg_entry = sum(lot["qty"] * lot["price"] for lot in matched_lots) / total_qty
    open_date = matched_lots[0]["date"]

    if side == "long":
        realized_pnl = (close_price - avg_entry) * total_qty
    else:
        realized_pnl = (avg_entry - close_price) * total_qty

    holding_days = (close_date - open_date).days
    trade_id     = f"{investor_id}_{symbol}_{open_date}_{close_date}_{uuid.uuid4().hex[:6]}"

    return Trade(
        trade_id     = trade_id,
        investor_id  = investor_id,
        symbol       = symbol,
        entry_date   = open_date,
        exit_date    = close_date,
        entry_price  = round(avg_entry, 4),
        exit_price   = round(close_price, 4),
        quantity     = total_qty,
        side         = side,
        realized_pnl = round(realized_pnl, 2),
        holding_days = max(holding_days, 0),
        is_winner    = realized_pnl > 0,
    )
