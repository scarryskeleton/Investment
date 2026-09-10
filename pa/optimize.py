"""Portfolio optimizers: mean-variance (MPT), risk parity, HRP, Black-Litterman.

Every function takes a price DataFrame (adjusted close, one column per asset)
and returns a plain ``dict[ticker, weight]`` that sums to ~1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pypfopt import (
    BlackLittermanModel,
    EfficientFrontier,
    black_litterman,
    expected_returns,
    objective_functions,
    risk_models,
)
from scipy.cluster.hierarchy import linkage, to_tree
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from .metrics import daily_returns

warnings.filterwarnings(
    "ignore",
    message="max_sharpe transforms the optimization problem",
    category=UserWarning,
)


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #
def expected_return_vector(prices: pd.DataFrame, method: str = "blend") -> pd.Series:
    """Annualized expected returns.

    method:
      - "mean"  : historical mean (noisy, included for comparison)
      - "capm"  : CAPM-implied given market beta
      - "ema"   : exponentially-weighted historical mean
      - "blend" : 50/50 CAPM + EMA, a cheap shrinkage toward a defensible prior
    """
    if method == "mean":
        return expected_returns.mean_historical_return(prices)
    if method == "capm":
        return expected_returns.capm_return(prices)
    if method == "ema":
        return expected_returns.ema_historical_return(prices)
    capm = expected_returns.capm_return(prices)
    ema = expected_returns.ema_historical_return(prices)
    return 0.5 * capm + 0.5 * ema


def covariance(prices: pd.DataFrame, shrink: bool = True) -> pd.DataFrame:
    if shrink:
        return risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    return risk_models.sample_cov(prices)


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class OptResult:
    method: str
    weights: dict[str, float]
    exp_return: float = float("nan")
    exp_vol: float = float("nan")
    exp_sharpe: float = float("nan")
    notes: list[str] = field(default_factory=list)

    def nonzero(self, threshold: float = 1e-4) -> dict[str, float]:
        return {k: v for k, v in self.weights.items() if abs(v) > threshold}


def _perf(weights: dict[str, float], mu: pd.Series, S: pd.DataFrame, rf: float):
    idx = list(mu.index)
    w = np.array([weights.get(t, 0.0) for t in idx])
    ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ S.values @ w))
    sharpe = (ret - rf) / vol if vol else float("nan")
    return ret, vol, sharpe


# --------------------------------------------------------------------------- #
# Mean-variance
# --------------------------------------------------------------------------- #
def mean_variance(
    prices: pd.DataFrame,
    objective: str = "max_sharpe",
    rf: float = 0.02,
    target_vol: float | None = None,
    target_return: float | None = None,
    return_method: str = "blend",
    min_weights: dict[str, float] | None = None,
    max_weight: float = 0.25,
    l2_gamma: float = 0.1,
) -> OptResult:
    mu = expected_return_vector(prices, return_method)
    S = covariance(prices, shrink=True)

    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    if l2_gamma:
        ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)

    notes = []
    if min_weights:
        tickers = list(mu.index)
        floor = np.array([min_weights.get(t, 0.0) for t in tickers])
        if floor.sum() > 1:
            floor = floor / floor.sum() * 0.95
            notes.append("Existing-position floors exceeded 100%; scaled down.")
        ef.add_constraint(lambda w: w >= floor)

    try:
        if objective == "min_vol":
            ef.min_volatility()
        elif objective == "efficient_risk" and target_vol:
            ef.efficient_risk(target_vol)
        elif objective == "efficient_return" and target_return:
            ef.efficient_return(target_return)
        else:
            ef.max_sharpe(risk_free_rate=rf)
    except Exception as e:  # infeasible target, solver failure
        notes.append(f"{objective} failed ({e}); fell back to min_volatility.")
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        ef.min_volatility()

    w = ef.clean_weights()
    ret, vol, shp = ef.portfolio_performance(risk_free_rate=rf)
    return OptResult("Mean-Variance", dict(w), ret, vol, shp, notes)


def efficient_frontier_points(
    prices: pd.DataFrame,
    rf: float = 0.02,
    return_method: str = "blend",
    n: int = 40,
    max_weight: float = 0.25,
) -> pd.DataFrame:
    mu = expected_return_vector(prices, return_method)
    S = covariance(prices, shrink=True)

    lo = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    lo.min_volatility()
    min_ret = lo.portfolio_performance()[0]
    hi_ret = float(mu.max()) * 0.99

    rows = []
    for tr in np.linspace(min_ret, hi_ret, n):
        ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
        try:
            ef.efficient_return(tr)
            r, v, s = ef.portfolio_performance(risk_free_rate=rf)
            rows.append({"ret": r, "vol": v, "sharpe": s})
        except Exception:
            continue
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Risk parity (equal risk contribution)
# --------------------------------------------------------------------------- #
def risk_parity(prices: pd.DataFrame, rf: float = 0.02) -> OptResult:
    S = covariance(prices, shrink=True)
    cov = S.values
    n = cov.shape[0]

    def objective(w):
        mrc = cov @ w
        rc = w * mrc
        rc = rc / rc.sum() if rc.sum() else rc
        return np.sum((rc - 1.0 / n) ** 2)

    res = minimize(
        objective,
        np.repeat(1.0 / n, n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = dict(zip(S.index, np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()))
    mu = expected_return_vector(prices)
    ret, vol, shp = _perf(w, mu, S, rf)
    return OptResult("Risk Parity", w, ret, vol, shp)


def hrp(prices: pd.DataFrame, rf: float = 0.02) -> OptResult:
    """Hierarchical Risk Parity (Lopez de Prado 2016).

    Self-contained implementation - clusters assets by correlation distance,
    quasi-diagonalizes, then allocates by recursive inverse-variance bisection.
    Uses no expected-return estimates.
    """
    rets = daily_returns(prices).dropna()
    cov = rets.cov()
    corr = rets.corr()
    tickers = list(cov.index)

    dist = np.sqrt(np.clip((1.0 - corr.values) / 2.0, 0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")

    # Quasi-diagonal order = leaf order of the tree.
    order = [tickers[i] for i in to_tree(link).pre_order()]

    cov_o = cov.loc[order, order].values
    w = pd.Series(1.0, index=order)
    clusters = [list(range(len(order)))]
    while clusters:
        clusters = [
            c[j:k]
            for c in clusters
            for j, k in ((0, len(c) // 2), (len(c) // 2, len(c)))
            if len(c) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            var_l = _cluster_var(cov_o, left)
            var_r = _cluster_var(cov_o, right)
            alpha = 1.0 - var_l / (var_l + var_r)
            w.iloc[left] *= alpha
            w.iloc[right] *= 1 - alpha

    weights = {t: float(w.get(t, 0.0)) for t in tickers}
    S = covariance(prices, shrink=True)
    mu = expected_return_vector(prices)
    ret, vol, shp = _perf(weights, mu, S, rf)
    return OptResult("Hierarchical Risk Parity", weights, ret, vol, shp)


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def risk_contributions(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    S = covariance(prices, shrink=True)
    idx = list(S.index)
    w = np.array([weights.get(t, 0.0) for t in idx])
    if w.sum():
        w = w / w.sum()
    mrc = S.values @ w
    rc = w * mrc
    total = rc.sum()
    return pd.Series(rc / total if total else rc, index=idx).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Black-Litterman
# --------------------------------------------------------------------------- #
def black_litterman_weights(
    prices: pd.DataFrame,
    market_prices: pd.Series,
    market_caps: dict[str, float],
    views: dict[str, float] | None = None,
    view_confidence: float = 0.5,
    rf: float = 0.02,
    max_weight: float = 0.25,
) -> OptResult:
    """Blend market-implied (equilibrium) returns with optional absolute views.

    ``views`` maps ticker -> expected annual return. With no views, the model
    just surfaces what current market caps imply - already a useful sanity
    check against naive historical means.
    """
    S = covariance(prices, shrink=True)
    caps = {t: market_caps[t] for t in S.index if t in market_caps}
    notes = []
    missing = [t for t in S.index if t not in caps]
    if missing:
        notes.append(f"No market cap for {', '.join(missing)}; excluded from BL prior.")
    S = S.loc[list(caps), list(caps)]
    prices = prices[list(caps)]

    delta = black_litterman.market_implied_risk_aversion(market_prices)
    prior = black_litterman.market_implied_prior_returns(caps, delta, S)

    views = {t: v for t, v in (views or {}).items() if t in caps}
    if views:
        bl = BlackLittermanModel(
            S,
            pi=prior,
            absolute_views=views,
            omega="idzorek",
            view_confidences=[view_confidence] * len(views),
        )
        notes.append(f"Applied {len(views)} view(s) at {view_confidence:.0%} confidence.")
    else:
        bl = BlackLittermanModel(S, pi=prior, absolute_views={})
        notes.append("No views supplied - showing market-implied equilibrium.")

    bl_ret = bl.bl_returns()
    bl_cov = bl.bl_cov()
    ef = EfficientFrontier(bl_ret, bl_cov, weight_bounds=(0, max_weight))
    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    try:
        ef.max_sharpe(risk_free_rate=rf)
    except Exception:
        ef.min_volatility()
    w = ef.clean_weights()
    ret, vol, shp = ef.portfolio_performance(risk_free_rate=rf)
    return OptResult("Black-Litterman", dict(w), ret, vol, shp, notes)
