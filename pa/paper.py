"""Paper-trading simulator maths.

Holdings, cash and P&L are *derived* by replaying the trade log, so the only
thing stored is the list of trades. The equity curve is a real day-by-day
reconstruction of the account against actual historical prices.

Fake money, real prices. Broker commission and an FX conversion fee are
modelled per trade (see ``FeeModel``); spreads, slippage, dividends and taxes
are not. Prices are the security's native quote — not converted to one
currency — so the FX fee stands in for the cost of holding foreign names.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeeModel:
    """A simple broker cost model, applied to every buy and sell.

    ``flat`` euros per trade, plus ``rate_bps`` basis points of the trade
    value, plus an extra ``fx_bps`` basis points when the security is not
    quoted in the account currency (EUR).
    """

    flat: float = 0.0
    rate_bps: float = 0.0
    fx_bps: float = 0.0

    def fee(self, notional: float, foreign: bool) -> float:
        bps = self.rate_bps + (self.fx_bps if foreign else 0.0)
        return self.flat + abs(notional) * bps / 1e4

    @property
    def active(self) -> bool:
        return self.flat > 0 or self.rate_bps > 0 or self.fx_bps > 0


PRESETS: dict[str, FeeModel] = {
    "No fees": FeeModel(),
    "Low-cost broker (0.25% FX only)": FeeModel(fx_bps=25.0),
    "Flat 2 / trade + 0.25% FX": FeeModel(flat=2.0, fx_bps=25.0),
    "0.1% commission + 0.35% FX": FeeModel(rate_bps=10.0, fx_bps=35.0),
}


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
    country: str = ""
    currency: str = ""
    sector: str = ""
    name: str = ""


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
    fees_paid: float = 0.0


def _replay(trades: pd.DataFrame):
    """Return (holdings {ticker: {shares, cost}}, cash_delta, realized, fees).

    A buy's fee is folded into its cost basis; a sell's fee comes straight off
    the proceeds and the realized gain. Either way it leaves the account, so it
    always reduces ``cash_delta``.
    """
    holdings: dict[str, dict] = {}
    cash_delta = 0.0
    realized = 0.0
    fees = 0.0
    for _, t in trades.iterrows():
        tk, sh, pr = t["ticker"], float(t["shares"]), float(t["price"])
        fee = float(t["fee"]) if "fee" in t and pd.notna(t["fee"]) else 0.0
        fees += fee
        h = holdings.setdefault(tk, {"shares": 0.0, "cost": 0.0})
        if t["side"] == "buy":
            h["shares"] += sh
            h["cost"] += sh * pr + fee
            cash_delta -= sh * pr + fee
        else:  # sell
            sell = min(sh, h["shares"])
            avg = h["cost"] / h["shares"] if h["shares"] > 1e-9 else pr
            realized += sell * (pr - avg) - fee
            h["shares"] -= sell
            h["cost"] -= sell * avg
            cash_delta += sell * pr - fee
    return holdings, cash_delta, realized, fees


def compute_state(
    starting_cash: float, trades: pd.DataFrame, latest_prices: dict[str, float]
) -> PaperState:
    if trades is None or trades.empty:
        return PaperState(starting_cash, starting_cash, 0.0, starting_cash,
                          0.0, 0.0, 0.0, [], 0)

    holdings, cash_delta, realized, fees = _replay(trades)
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
        fees_paid=fees,
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
        holdings, cash_delta, _, _ = _replay(upto)
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
