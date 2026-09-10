"""Top-level orchestration: from holdings + universe to ranked suggestions."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import data, diversification, metrics, optimize
from .universe import build_candidate_universe

MARKET_TICKER = "SPY"


@dataclass
class Analysis:
    holdings_weights: dict[str, float]
    holdings_prices: pd.DataFrame
    market_prices: pd.Series
    fundamentals: pd.DataFrame
    snapshot: metrics.PortfolioSnapshot
    sector_exposure: pd.Series
    concentration: dict
    candidate_meta: pd.DataFrame
    candidate_prices: pd.DataFrame
    scores: pd.DataFrame
    ranked: pd.DataFrame
    sector_gaps: pd.DataFrame
    warnings: list[str] = field(default_factory=list)


def run_analysis(
    holdings_weights: dict[str, float],
    watchlist: list[str] | None = None,
    include_curated_etfs: bool = True,
    include_sp500: bool = False,
    lookback_years: float = 5.0,
    rf: float = 0.02,
    add_weight: float = 0.10,
) -> Analysis:
    warnings: list[str] = []
    held = [t for t, w in holdings_weights.items() if w]

    prices_all = data.fetch_prices(held + [MARKET_TICKER], lookback_years)
    missing_held = [t for t in held if t not in prices_all.columns]
    if missing_held:
        warnings.append(f"No price data for holdings: {', '.join(missing_held)} (dropped).")
    held = [t for t in held if t in prices_all.columns]
    if not held:
        raise ValueError("None of the supplied holdings returned price data.")

    holdings_weights = {t: holdings_weights[t] for t in held}
    tot = sum(holdings_weights.values())
    holdings_weights = {t: w / tot for t, w in holdings_weights.items()}

    market_prices = prices_all[MARKET_TICKER]
    holdings_prices = prices_all[held]

    fundamentals = data.fetch_fundamentals(held)
    snap = metrics.snapshot(holdings_prices, holdings_weights, market_prices, rf)
    sec_exp = metrics.sector_exposure(holdings_weights, fundamentals)
    conc = metrics.concentration(holdings_weights)

    meta = build_candidate_universe(
        watchlist=watchlist,
        include_curated_etfs=include_curated_etfs,
        include_sp500=include_sp500,
        exclude=held,
    )
    cand_prices = data.fetch_prices(list(meta.index), lookback_years) if len(meta) else pd.DataFrame()
    if not cand_prices.empty:
        got = [t for t in meta.index if t in cand_prices.columns]
        meta = meta.loc[got]

    if cand_prices.empty:
        scores = pd.DataFrame()
        ranked = pd.DataFrame()
        warnings.append("No candidate price data retrieved.")
    else:
        scores = diversification.candidate_scores(
            holdings_prices, holdings_weights, cand_prices, add_weight, rf
        )
        ranked = diversification.composite_ranking(scores)
        if not ranked.empty:
            ranked = ranked.join(meta[["name", "category", "source"]], how="left")
            ranked["rationale"] = [
                diversification.rationale(
                    ranked.loc[t],
                    ranked.loc[t] if "category" in ranked.columns else None,
                )
                for t in ranked.index
            ]

    gaps = diversification.sector_gaps(holdings_weights, fundamentals)

    return Analysis(
        holdings_weights=holdings_weights,
        holdings_prices=holdings_prices,
        market_prices=market_prices,
        fundamentals=fundamentals,
        snapshot=snap,
        sector_exposure=sec_exp,
        concentration=conc,
        candidate_meta=meta,
        candidate_prices=cand_prices,
        scores=scores,
        ranked=ranked,
        sector_gaps=gaps,
        warnings=warnings,
    )


def optimize_with_candidates(
    analysis: Analysis,
    extra_tickers: list[str],
    methods: list[str],
    objective: str = "max_sharpe",
    rf: float = 0.02,
    keep_existing: bool = True,
    target_vol: float | None = None,
    views: dict[str, float] | None = None,
    return_method: str = "blend",
    max_weight: float = 0.25,
) -> dict[str, optimize.OptResult]:
    """Run the selected optimizers over (holdings + chosen candidates)."""
    held = list(analysis.holdings_weights)
    extra = [t for t in extra_tickers if t not in held and t in analysis.candidate_prices.columns]
    tickers = held + extra

    prices = analysis.holdings_prices.join(
        analysis.candidate_prices[extra], how="inner"
    ) if extra else analysis.holdings_prices.copy()
    prices = prices.dropna()

    min_weights = None
    if keep_existing:
        # Keep at least half of each current position's (rescaled) weight.
        min_weights = {t: analysis.holdings_weights[t] * 0.5 for t in held}

    results: dict[str, optimize.OptResult] = {}

    if "mean_variance" in methods:
        results["mean_variance"] = optimize.mean_variance(
            prices, objective=objective, rf=rf, target_vol=target_vol,
            return_method=return_method, min_weights=min_weights, max_weight=max_weight,
        )
    if "risk_parity" in methods:
        results["risk_parity"] = optimize.risk_parity(prices, rf=rf)
    if "hrp" in methods:
        results["hrp"] = optimize.hrp(prices, rf=rf)
    if "black_litterman" in methods:
        caps = data.market_caps(tickers)
        try:
            results["black_litterman"] = optimize.black_litterman_weights(
                prices, analysis.market_prices, caps, views=views, rf=rf, max_weight=max_weight,
            )
        except Exception as e:
            results["black_litterman"] = optimize.OptResult(
                "Black-Litterman", {}, notes=[f"Failed: {e}"]
            )

    return results


def weights_table(
    current: dict[str, float], results: dict[str, optimize.OptResult]
) -> pd.DataFrame:
    """Tidy comparison of current vs each optimizer's target weights."""
    cols = {"Current": pd.Series(current)}
    for key, res in results.items():
        cols[res.method] = pd.Series(res.nonzero())
    df = pd.DataFrame(cols).fillna(0.0)
    df = df.loc[df.abs().sum(axis=1).sort_values(ascending=False).index]
    return df
