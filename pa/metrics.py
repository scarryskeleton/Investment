"""Return, risk and exposure metrics for a portfolio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return prices.pct_change().dropna(how="all")


def align_weights(weights: dict[str, float], columns) -> np.ndarray:
    w = np.array([weights.get(c, 0.0) for c in columns], dtype=float)
    total = w.sum()
    return w / total if total else w


def portfolio_return_series(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    cols = [c for c in returns.columns if weights.get(c, 0.0) != 0.0]
    if not cols:
        return pd.Series(dtype=float)
    w = align_weights({c: weights[c] for c in cols}, cols)
    return returns[cols].mul(w, axis=1).sum(axis=1)


def annualized_return(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    return (1 + r).prod() ** (TRADING_DAYS / len(r)) - 1


def annualized_vol(r: pd.Series) -> float:
    return r.std(ddof=1) * np.sqrt(TRADING_DAYS) if not r.empty else float("nan")


def sharpe(r: pd.Series, rf: float = 0.02) -> float:
    vol = annualized_vol(r)
    if not vol or np.isnan(vol):
        return float("nan")
    return (annualized_return(r) - rf) / vol


def sortino(r: pd.Series, rf: float = 0.02) -> float:
    downside = r[r < 0]
    dd = downside.std(ddof=1) * np.sqrt(TRADING_DAYS)
    if not dd or np.isnan(dd):
        return float("nan")
    return (annualized_return(r) - rf) / dd


def max_drawdown(r: pd.Series) -> float:
    if r.empty:
        return float("nan")
    curve = (1 + r).cumprod()
    return (curve / curve.cummax() - 1).min()


def beta_to_market(r: pd.Series, market_r: pd.Series) -> float:
    df = pd.concat([r, market_r], axis=1, join="inner").dropna()
    if len(df) < 20:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    return cov[0, 1] / cov[1, 1]


def diversification_ratio(returns: pd.DataFrame, weights: dict[str, float]) -> float:
    """Weighted average of asset vols divided by portfolio vol.

    1.0 means no diversification benefit; higher is better.
    """
    cols = [c for c in returns.columns if weights.get(c, 0.0) != 0.0]
    if len(cols) < 2:
        return 1.0
    w = align_weights({c: weights[c] for c in cols}, cols)
    vols = returns[cols].std(ddof=1).values
    port_vol = np.sqrt(w @ returns[cols].cov().values @ w)
    return float((w @ vols) / port_vol) if port_vol else 1.0


@dataclass
class PortfolioSnapshot:
    weights: dict[str, float]
    ann_return: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    beta: float
    diversification_ratio: float
    n_holdings: int

    def as_dict(self) -> dict:
        return {
            "Annualized return": self.ann_return,
            "Annualized volatility": self.ann_vol,
            "Sharpe ratio": self.sharpe,
            "Sortino ratio": self.sortino,
            "Max drawdown": self.max_drawdown,
            "Beta to market": self.beta,
            "Diversification ratio": self.diversification_ratio,
            "Holdings": self.n_holdings,
        }


def snapshot(
    prices: pd.DataFrame,
    weights: dict[str, float],
    market_prices: pd.Series | None = None,
    rf: float = 0.02,
) -> PortfolioSnapshot:
    rets = daily_returns(prices)
    pr = portfolio_return_series(rets, weights)
    market_r = daily_returns(market_prices) if market_prices is not None else None
    return PortfolioSnapshot(
        weights={k: v for k, v in weights.items() if v},
        ann_return=annualized_return(pr),
        ann_vol=annualized_vol(pr),
        sharpe=sharpe(pr, rf),
        sortino=sortino(pr, rf),
        max_drawdown=max_drawdown(pr),
        beta=beta_to_market(pr, market_r) if market_r is not None else float("nan"),
        diversification_ratio=diversification_ratio(rets, weights),
        n_holdings=sum(1 for v in weights.values() if v),
    )


def sector_exposure(
    weights: dict[str, float], fundamentals: pd.DataFrame
) -> pd.Series:
    s = pd.Series(weights, dtype=float)
    s = s[s != 0]
    if s.empty or "sector" not in fundamentals.columns:
        return pd.Series(dtype=float)
    sectors = fundamentals["sector"].reindex(s.index).fillna("Unknown")
    return s.groupby(sectors).sum().sort_values(ascending=False)


def concentration(weights: dict[str, float]) -> dict[str, float]:
    w = np.array([v for v in weights.values() if v], dtype=float)
    if w.size == 0:
        return {"hhi": float("nan"), "top_weight": float("nan"), "effective_n": float("nan")}
    w = w / w.sum()
    hhi = float((w**2).sum())
    return {"hhi": hhi, "top_weight": float(w.max()), "effective_n": 1.0 / hhi}
