"""Beginner-oriented 'explore from cash' tools.

Everything here works from a simple two-sleeve mix (a broad stock fund + a broad
bond fund) so a newcomer can feel the risk/return trade-off and see how regular
contributions might compound. Illustrative, historical, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import data, metrics

# Sensible defaults for the two sleeves.
DEFAULT_STOCK = "VTI"   # US total stock market (long history)
DEFAULT_BOND = "BND"    # US aggregate bonds
CASH_PROXY = "BIL"      # 1-3 month T-bills, used only if a cash sleeve is added

PRESETS: dict[str, int] = {
    "Very cautious": 20,
    "Cautious": 40,
    "Balanced": 60,
    "Adventurous": 80,
    "All-in on stocks": 100,
}


@dataclass
class MixResult:
    stock_pct: int
    stock_ticker: str
    bond_ticker: str
    ann_return: float
    ann_vol: float
    max_drawdown: float
    worst_12m: float
    best_12m: float
    growth: pd.DataFrame            # €10k growth: columns mix / all-stocks / all-bonds / cash
    frontier: pd.DataFrame          # vol/ret traced across every stock/bond split
    monthly_returns: pd.Series      # of the chosen mix, for the projection
    start: date = None
    end: date = None
    notes: list[str] = field(default_factory=list)


def _mix_daily_returns(px: pd.DataFrame, stock: str, bond: str, stock_w: float) -> pd.Series:
    r = px[[stock, bond]].pct_change().dropna()
    return stock_w * r[stock] + (1 - stock_w) * r[bond]


def analyze_mix(
    stock_pct: int,
    stock_ticker: str = DEFAULT_STOCK,
    bond_ticker: str = DEFAULT_BOND,
    lookback_years: float = 18.0,
    cash_rate: float = 0.03,
    start_value: float = 10_000.0,
    account_ccy: str = "",
) -> MixResult:
    px = data.fetch_prices([stock_ticker, bond_ticker], lookback_years)
    missing = [t for t in (stock_ticker, bond_ticker) if t not in px.columns]
    if missing:
        raise ValueError(f"No price data for {', '.join(missing)}.")
    px = px[[stock_ticker, bond_ticker]].dropna()
    if len(px) < 250:
        raise ValueError("Not enough overlapping history for these two funds.")

    if account_ccy:
        # Convert both funds into the account currency so returns, volatility
        # and drawdowns are what a holder of that currency actually experienced.
        from pa import fx

        f = data.fetch_fundamentals([stock_ticker, bond_ticker])
        tc = {t: (f.loc[t].get("currency") if t in f.index else account_ccy)
              for t in px.columns}
        px = fx.convert_frame(px, tc, account_ccy).dropna()

    sw = stock_pct / 100.0
    mix_r = _mix_daily_returns(px, stock_ticker, bond_ticker, sw)

    # Growth of `start_value`
    idx = mix_r.index
    daily_cash = (1 + cash_rate) ** (1 / 252) - 1
    growth = pd.DataFrame(
        {
            f"{stock_pct}/{100 - stock_pct} mix": start_value * (1 + mix_r).cumprod(),
            "100% stocks": start_value * (1 + px[stock_ticker].pct_change().reindex(idx).fillna(0)).cumprod(),
            "100% bonds": start_value * (1 + px[bond_ticker].pct_change().reindex(idx).fillna(0)).cumprod(),
            "Cash / savings": start_value * (1 + pd.Series(daily_cash, index=idx)).cumprod(),
        }
    )

    # Frontier across all splits
    rows = []
    for p in range(0, 101, 5):
        rr = _mix_daily_returns(px, stock_ticker, bond_ticker, p / 100.0)
        rows.append(
            {"stock_pct": p, "vol": metrics.annualized_vol(rr), "ret": metrics.annualized_return(rr)}
        )
    frontier = pd.DataFrame(rows)

    roll_12m = (1 + mix_r).rolling(252).apply(np.prod, raw=True) - 1
    monthly = px.resample("ME").last().pct_change().dropna()
    mix_monthly = sw * monthly[stock_ticker] + (1 - sw) * monthly[bond_ticker]

    return MixResult(
        stock_pct=stock_pct,
        stock_ticker=stock_ticker,
        bond_ticker=bond_ticker,
        ann_return=metrics.annualized_return(mix_r),
        ann_vol=metrics.annualized_vol(mix_r),
        max_drawdown=metrics.max_drawdown(mix_r),
        worst_12m=float(roll_12m.min()),
        best_12m=float(roll_12m.max()),
        growth=growth,
        frontier=frontier,
        monthly_returns=mix_monthly,
        start=idx[0].date(),
        end=idx[-1].date(),
    )


@dataclass
class Projection:
    months: np.ndarray
    contributed: np.ndarray
    percentiles: dict[str, np.ndarray]   # 'p10','p25','p50','p75','p90'
    end_low: float
    end_median: float
    end_high: float
    total_contributed: float
    # low-fee vs high-fee median ending values
    fee_low_median: float
    fee_high_median: float
    fee_low_pct: float
    fee_high_pct: float
    method: str = "bootstrap"
    faithfulness: dict | None = None


def _block_bootstrap_paths(
    monthly: np.ndarray, n_months: int, n_paths: int, block: int, rng: np.random.Generator
) -> np.ndarray:
    """Return an (n_paths, n_months) array of resampled monthly returns."""
    if len(monthly) == 0:
        return np.zeros((n_paths, n_months))
    n_blocks = int(np.ceil(n_months / block))
    max_start = max(len(monthly) - block, 0)
    out = np.empty((n_paths, n_blocks * block))
    for i in range(n_paths):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        out[i] = np.concatenate([monthly[s : s + block] for s in starts])
    return out[:, :n_months]


def project_contributions(
    monthly_returns: pd.Series,
    start_value: float = 1_000.0,
    monthly_contribution: float = 100.0,
    years: int = 10,
    n_paths: int = 1_000,
    block: int = 6,
    fee_low: float = 0.0015,
    fee_high: float = 0.015,
    seed: int = 0,
    method: str = "bootstrap",
) -> Projection:
    """Monte-Carlo of a contribution plan.

    ``method``: "bootstrap" resamples real 6-month chunks of history;
    "neural" fits a small mixture-density network and rolls it forward. The
    bootstrap is the trustworthy default; the neural option is a teaching
    comparison and falls back to bootstrap when history is too short.

    Returns percentile bands of portfolio value over time, plus a low-fee vs
    high-fee comparison on the median path.
    """
    rng = np.random.default_rng(seed)
    m = monthly_returns.dropna().to_numpy()
    n_months = years * 12

    faith = None
    method_used = method
    if method == "neural":
        from . import neural

        paths_r, faith = neural.generate_paths(m, n_months, n_paths, seed)
        if paths_r is None:
            method_used = "bootstrap"
    if method_used == "bootstrap":
        paths_r = _block_bootstrap_paths(m, n_months, n_paths, block, rng)

    def run(fee_annual: float) -> np.ndarray:
        fee_m = fee_annual / 12.0
        vals = np.full(n_paths, start_value, dtype=float)
        series = np.empty((n_paths, n_months + 1))
        series[:, 0] = vals
        for t in range(n_months):
            vals = (vals + monthly_contribution) * (1 + paths_r[:, t] - fee_m)
            series[:, t + 1] = vals
        return series

    base = run(fee_low)
    high = run(fee_high)

    months = np.arange(n_months + 1)
    contributed = start_value + monthly_contribution * months
    pct = {
        "p10": np.percentile(base, 10, axis=0),
        "p25": np.percentile(base, 25, axis=0),
        "p50": np.percentile(base, 50, axis=0),
        "p75": np.percentile(base, 75, axis=0),
        "p90": np.percentile(base, 90, axis=0),
    }
    total_contrib = float(contributed[-1])
    return Projection(
        months=months,
        contributed=contributed,
        percentiles=pct,
        end_low=float(pct["p10"][-1]),
        end_median=float(pct["p50"][-1]),
        end_high=float(pct["p90"][-1]),
        total_contributed=total_contrib,
        fee_low_median=float(np.median(base[:, -1])),
        fee_high_median=float(np.median(high[:, -1])),
        fee_low_pct=fee_low,
        fee_high_pct=fee_high,
        method=method_used,
        faithfulness=faith,
    )
