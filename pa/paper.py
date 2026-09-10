"""Paper-trading simulator maths.

Holdings, cash and P&L are *derived* by replaying the trade log, so the only
thing stored is the list of trades. The equity curve is a real day-by-day
reconstruction of the account against actual historical prices.

Fake money, real prices. No fees, spreads, slippage or taxes are modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Position:
    ticker: str
    shares: float
    avg_cost: float
    last_price: float
    cost_basis: float
    market_value: float
    unrealized: float
    unrealized_pct: float
    weight: float


@dataclass
class PaperState:
    starting_cash: float
    cash: float
    invested: float
    total_value: float
    realized_pnl: float
    total_pnl: float
    total_pnl_pct: float
    positions: list[Position] = field(default_factory=list)
    n_trades: int = 0
    missing_prices: list[str] = field(default_factory=list)


def _replay(trades: pd.DataFrame):
    """Return (holdings {ticker: {shares, cost}}, cash_delta, realized)."""
    holdings: dict[str, dict] = {}
    cash_delta = 0.0
    realized = 0.0
    for _, t in trades.iterrows():
        tk, sh, pr = t["ticker"], float(t["shares"]), float(t["price"])
        h = holdings.setdefault(tk, {"shares": 0.0, "cost": 0.0})
        if t["side"] == "buy":
            h["shares"] += sh
            h["cost"] += sh * pr
            cash_delta -= sh * pr
        else:  # sell
            sell = min(sh, h["shares"])
            avg = h["cost"] / h["shares"] if h["shares"] > 1e-9 else pr
            realized += sell * (pr - avg)
            h["shares"] -= sell
            h["cost"] -= sell * avg
            cash_delta += sell * pr
    return holdings, cash_delta, realized


def compute_state(
    starting_cash: float, trades: pd.DataFrame, latest_prices: dict[str, float]
) -> PaperState:
    if trades is None or trades.empty:
        return PaperState(starting_cash, starting_cash, 0.0, starting_cash,
                          0.0, 0.0, 0.0, [], 0)

    holdings, cash_delta, realized = _replay(trades)
    cash = starting_cash + cash_delta

    positions, invested, missing = [], 0.0, []
    for tk, h in holdings.items():
        if h["shares"] <= 1e-6:
            continue
        lp = latest_prices.get(tk)
        if lp is None or lp != lp:
            missing.append(tk)
            lp = h["cost"] / h["shares"]  # fall back to cost so value isn't lost
        mv = h["shares"] * lp
        invested += mv
        avg = h["cost"] / h["shares"]
        positions.append(Position(
            ticker=tk, shares=h["shares"], avg_cost=avg, last_price=lp,
            cost_basis=h["cost"], market_value=mv,
            unrealized=mv - h["cost"],
            unrealized_pct=(mv - h["cost"]) / h["cost"] if h["cost"] else 0.0,
            weight=0.0,
        ))

    total = cash + invested
    for p in positions:
        p.weight = p.market_value / total if total else 0.0
    positions.sort(key=lambda p: p.market_value, reverse=True)

    return PaperState(
        starting_cash=starting_cash,
        cash=cash,
        invested=invested,
        total_value=total,
        realized_pnl=realized,
        total_pnl=total - starting_cash,
        total_pnl_pct=(total - starting_cash) / starting_cash if starting_cash else 0.0,
        positions=positions,
        n_trades=len(trades),
        missing_prices=missing,
    )


def equity_curve(
    starting_cash: float,
    trades: pd.DataFrame,
    price_history: pd.DataFrame,
    benchmark: pd.Series | None = None,
    benchmark_name: str = "All-in S&P 500",
) -> pd.DataFrame:
    """Daily account value since the first trade, next to a buy-and-hold
    benchmark of the same starting cash."""
    if trades is None or trades.empty or price_history is None or price_history.empty:
        return pd.DataFrame()

    tr = trades.copy()
    ts = pd.to_datetime(tr["ts"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_localize(None)
    tr["day"] = ts.dt.normalize()

    idx = pd.to_datetime(price_history.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    ph = price_history.copy()
    ph.index = idx

    start = tr["day"].min() - pd.Timedelta(days=4)
    hist = ph[ph.index >= start].ffill()
    if hist.empty:  # trades newer than any price bar - show the last ~10 bars flat
        hist = ph.tail(10).ffill()
    if hist.empty:
        return pd.DataFrame()

    rows = []
    for day in hist.index:
        upto = tr[tr["day"] <= day]
        holdings, cash_delta, _ = _replay(upto)
        cash = starting_cash + cash_delta
        mv = 0.0
        for tk, h in holdings.items():
            if h["shares"] <= 1e-6:
                continue
            px = hist.loc[day].get(tk, np.nan)
            mv += h["shares"] * (px if px == px else h["cost"] / max(h["shares"], 1e-9))
        rows.append((day, cash + mv))

    out = pd.DataFrame(rows, columns=["date", "Your portfolio"]).set_index("date")
    if benchmark is not None and not benchmark.empty:
        b = benchmark.reindex(out.index).ffill().bfill()
        out[benchmark_name] = starting_cash * b / b.iloc[0]
    return out
