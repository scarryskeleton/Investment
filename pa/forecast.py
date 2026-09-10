"""Per-asset 'outlook' - naive return estimates + simulated scenario ranges.

There is no reliable way to forecast an individual stock or bond. What this
module produces is deliberately modest:

  * three standard textbook estimates of expected return (historical mean, CAPM,
    exponentially-weighted mean) - shown side by side precisely because they
    disagree, which is the honest signal;
  * a distribution of what-if outcomes over a chosen horizon, built by
    resampling the asset's own history (or the small neural generator);
  * the asset's actual best and worst rolling-12-month returns for context.

Everything is framed as a range, never a point prediction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pypfopt import expected_returns

from . import explore, metrics, neural


@dataclass
class AssetOutlook:
    ticker: str
    n_months: int
    ann_vol: float
    est_hist_mean: float
    est_capm: float
    est_ema: float
    est_blend: float
    hist_best_12m: float
    hist_worst_12m: float
    horizon_years: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    method: str
    note: str

    @property
    def est_spread(self) -> float:
        e = [x for x in (self.est_hist_mean, self.est_capm, self.est_ema) if x == x]
        return (max(e) - min(e)) if len(e) > 1 else 0.0


def _scenario_totals(monthly: np.ndarray, horizon_m: int, n_paths: int, method: str, seed: int):
    used = method
    paths = None
    if method == "neural":
        paths, _ = neural.generate_paths(monthly, horizon_m, n_paths, seed)
        if paths is None:
            used = "bootstrap"
    if paths is None:
        rng = np.random.default_rng(seed)
        block = min(6, max(1, len(monthly) // 4))
        paths = explore._block_bootstrap_paths(monthly, horizon_m, n_paths, block, rng)
    total = (1 + paths).prod(axis=1) - 1.0
    return total, used


def asset_outlook(
    ticker: str,
    prices: pd.Series,
    market_prices: pd.Series,
    rf: float = 0.03,
    horizon_years: float = 1.0,
    n_paths: int = 2000,
    method: str = "bootstrap",
    seed: int = 0,
) -> AssetOutlook:
    prices = prices.dropna()
    df = prices.to_frame(ticker)
    daily = metrics.daily_returns(prices)

    def _first(fn):
        try:
            return float(fn().iloc[0])
        except Exception:
            return float("nan")

    est_mean = _first(lambda: expected_returns.mean_historical_return(df))
    est_ema = _first(lambda: expected_returns.ema_historical_return(df))
    try:
        mkt = market_prices.reindex(prices.index).dropna()
        common = df.reindex(mkt.index).dropna()
        est_capm = float(
            expected_returns.capm_return(
                common,
                market_prices=mkt.reindex(common.index).to_frame("mkt"),
                risk_free_rate=rf,
            ).iloc[0]
        )
    except Exception:
        est_capm = float("nan")
    est_blend = float(np.nanmean([est_capm, est_ema]))

    monthly = prices.resample("ME").last().pct_change().dropna().to_numpy()
    roll12 = pd.Series((1 + monthly)).rolling(12).apply(np.prod, raw=True) - 1
    best = float(roll12.max()) if roll12.notna().any() else float("nan")
    worst = float(roll12.min()) if roll12.notna().any() else float("nan")

    horizon_m = max(1, int(round(horizon_years * 12)))
    total, used = _scenario_totals(monthly, horizon_m, n_paths, method, seed)
    p10, p25, p50, p75, p90 = np.percentile(total, [10, 25, 50, 75, 90])

    est = [x for x in (est_mean, est_capm, est_ema) if x == x]
    spread = (max(est) - min(est)) if len(est) > 1 else 0.0
    if spread > 0.10:
        note = (f"The textbook estimates disagree by {spread * 100:.0f} percentage "
                "points — normal for a single asset, and a measure of how little "
                "is actually knowable.")
    elif len(est) > 1:
        note = ("The textbook estimates roughly agree here, but the what-if range "
                "below is still very wide.")
    else:
        note = "Too little data for the textbook estimates; only the scenario range is shown."

    return AssetOutlook(
        ticker=ticker,
        n_months=len(monthly),
        ann_vol=metrics.annualized_vol(daily),
        est_hist_mean=est_mean,
        est_capm=est_capm,
        est_ema=est_ema,
        est_blend=est_blend,
        hist_best_12m=best,
        hist_worst_12m=worst,
        horizon_years=horizon_years,
        p10=float(p10), p25=float(p25), p50=float(p50), p75=float(p75), p90=float(p90),
        method=used,
        note=note,
    )
