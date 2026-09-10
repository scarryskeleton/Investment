"""Correlation- and concentration-based scoring of candidate additions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import (
    annualized_return,
    annualized_vol,
    daily_returns,
    portfolio_return_series,
    sector_exposure,
    sharpe,
)


def _renormalize(weights: dict[str, float], new_ticker: str, add_weight: float) -> dict:
    scaled = {k: v * (1 - add_weight) for k, v in weights.items()}
    scaled[new_ticker] = scaled.get(new_ticker, 0.0) + add_weight
    return scaled


def candidate_scores(
    holdings_prices: pd.DataFrame,
    holdings_weights: dict[str, float],
    candidate_prices: pd.DataFrame,
    add_weight: float = 0.10,
    rf: float = 0.02,
) -> pd.DataFrame:
    """Score each candidate on how it would change the portfolio if added at
    ``add_weight`` (existing positions scaled down pro-rata).

    Columns:
      corr_to_portfolio   - return correlation with current portfolio (lower better)
      vol_after / vol_delta      - annualized vol before vs after
      sharpe_after / sharpe_delta
      div_ratio_delta     - change in diversification ratio
      cand_return / cand_vol     - the candidate's own history
    """
    all_prices = holdings_prices.join(
        candidate_prices[[c for c in candidate_prices.columns if c not in holdings_prices.columns]],
        how="inner",
    )
    rets = daily_returns(all_prices)
    port_r = portfolio_return_series(rets, holdings_weights)

    base_vol = annualized_vol(port_r)
    base_sharpe = sharpe(port_r, rf)
    base_div = _div_ratio(rets, holdings_weights)

    out = []
    for c in candidate_prices.columns:
        if c not in rets.columns or c in holdings_weights and holdings_weights[c] > 0:
            continue
        cr = rets[c]
        joined = pd.concat([port_r, cr], axis=1, join="inner").dropna()
        if len(joined) < 60:
            continue
        corr = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))

        new_w = _renormalize(holdings_weights, c, add_weight)
        new_r = portfolio_return_series(rets, new_w)
        v_after = annualized_vol(new_r)
        s_after = sharpe(new_r, rf)
        d_after = _div_ratio(rets, new_w)

        out.append(
            {
                "ticker": c,
                "corr_to_portfolio": corr,
                "vol_after": v_after,
                "vol_delta": v_after - base_vol,
                "sharpe_after": s_after,
                "sharpe_delta": s_after - base_sharpe,
                "div_ratio_delta": d_after - base_div,
                "cand_return": annualized_return(cr),
                "cand_vol": annualized_vol(cr),
            }
        )
    df = pd.DataFrame(out).set_index("ticker") if out else pd.DataFrame()
    df.attrs["base_vol"] = base_vol
    df.attrs["base_sharpe"] = base_sharpe
    df.attrs["base_div"] = base_div
    return df


def _div_ratio(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    cols = [c for c in returns.columns if weights.get(c, 0.0) != 0.0]
    if len(cols) < 2:
        return 1.0
    w = np.array([weights[c] for c in cols])
    w = w / w.sum()
    sub = returns[cols]
    vols = sub.std(ddof=1).values
    port_vol = np.sqrt(w @ sub.cov().values @ w)
    return float((w @ vols) / port_vol) if port_vol else 1.0


def composite_ranking(scores: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Combine the sub-scores into a single 0-100 'diversification fit' score.

    Rewards: low/negative correlation, vol reduction, Sharpe improvement,
    higher diversification ratio.
    """
    if scores.empty:
        return scores
    w = {"corr": 0.35, "vol": 0.25, "sharpe": 0.25, "div": 0.15}
    if weights:
        w.update(weights)

    def z(s, invert=False):
        s = s.astype(float)
        if s.std(ddof=0) == 0 or s.isna().all():
            return pd.Series(0.0, index=s.index)
        zz = (s - s.mean()) / s.std(ddof=0)
        return -zz if invert else zz

    raw = (
        w["corr"] * z(scores["corr_to_portfolio"], invert=True)
        + w["vol"] * z(scores["vol_delta"], invert=True)
        + w["sharpe"] * z(scores["sharpe_delta"])
        + w["div"] * z(scores["div_ratio_delta"])
    )
    ranked = scores.copy()
    lo, hi = raw.min(), raw.max()
    ranked["fit_score"] = 50.0 if hi == lo else (raw - lo) / (hi - lo) * 100.0
    return ranked.sort_values("fit_score", ascending=False)


def sector_gaps(
    holdings_weights: dict[str, float],
    fundamentals: pd.DataFrame,
    target_max: float = 0.35,
) -> pd.DataFrame:
    """Sectors that are over- or under-represented vs an equal-ish target."""
    exp = sector_exposure(holdings_weights, fundamentals)
    if exp.empty:
        return pd.DataFrame()
    n = max(len(exp), 3)
    target = 1.0 / n
    df = exp.to_frame("weight")
    df["target"] = target
    df["gap"] = df["weight"] - target
    df["over_concentrated"] = df["weight"] > target_max
    return df


def rationale(row: pd.Series, cand_meta: pd.Series | None = None) -> str:
    bits = []
    corr = row.get("corr_to_portfolio", np.nan)
    if corr < 0:
        bits.append(f"negatively correlated with your portfolio ({corr:+.2f})")
    elif corr < 0.4:
        bits.append(f"low correlation to your portfolio ({corr:.2f})")
    elif corr > 0.8:
        bits.append(f"highly correlated ({corr:.2f}) - limited diversification")

    vd = row.get("vol_delta", np.nan)
    if vd < -0.002:
        bits.append(f"cuts portfolio volatility by {abs(vd) * 100:.1f} pp")
    elif vd > 0.005:
        bits.append(f"raises volatility by {vd * 100:.1f} pp")

    sd = row.get("sharpe_delta", np.nan)
    if sd > 0.02:
        bits.append(f"improves Sharpe by {sd:+.2f}")
    elif sd < -0.05:
        bits.append(f"lowers Sharpe by {sd:+.2f}")

    dd = row.get("div_ratio_delta", np.nan)
    if dd > 0.02:
        bits.append("increases diversification ratio")

    if cand_meta is not None:
        cat = cand_meta.get("category")
        if cat and cat not in ("Watchlist",):
            bits.append(f"adds {cat} exposure")

    return "; ".join(bits) if bits else "neutral impact on current risk/return"
