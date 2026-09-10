"""Command-line version of the analysis, for quick runs without the dashboard.

    python cli.py --holdings sample_portfolio.csv --watchlist COST,JNJ,XOM --top 15
"""

from __future__ import annotations

import argparse

import pandas as pd

from pa import portfolio, suggest

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank portfolio additions.")
    ap.add_argument("--holdings", required=True, help="CSV file or inline 'AAPL:40,MSFT:25'")
    ap.add_argument("--mode", default="shares", choices=["shares", "dollars", "weight"])
    ap.add_argument("--watchlist", default="", help="comma-separated tickers")
    ap.add_argument("--no-etfs", action="store_true", help="exclude curated ETF universe")
    ap.add_argument("--sp500", action="store_true", help="include full S&P 500")
    ap.add_argument("--lookback", type=float, default=5.0)
    ap.add_argument("--rf", type=float, default=0.04)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if args.holdings.endswith(".csv"):
        text = open(args.holdings).read()
    else:
        text = "\n".join(p.replace(":", " ") for p in args.holdings.split(","))
    parsed = portfolio.parse_holdings_text(text)
    weights = portfolio.to_weights(parsed, mode=args.mode)

    watchlist = [t.strip().upper() for t in args.watchlist.replace(",", " ").split() if t.strip()]

    A = suggest.run_analysis(
        holdings_weights=weights,
        watchlist=watchlist,
        include_curated_etfs=not args.no_etfs,
        include_sp500=args.sp500,
        lookback_years=args.lookback,
        rf=args.rf,
    )

    for w in A.warnings:
        print(f"! {w}")

    s = A.snapshot
    print("\n=== Current portfolio ===")
    print(f"  holdings            : {s.n_holdings}")
    print(f"  annualized return   : {s.ann_return:6.1%}")
    print(f"  annualized vol      : {s.ann_vol:6.1%}")
    print(f"  Sharpe / Sortino    : {s.sharpe:.2f} / {s.sortino:.2f}")
    print(f"  max drawdown        : {s.max_drawdown:6.1%}")
    print(f"  beta to market      : {s.beta:.2f}")
    print(f"  diversification rat.: {s.diversification_ratio:.2f}")
    print(f"  effective # names   : {A.concentration['effective_n']:.1f}")

    if not A.sector_exposure.empty:
        print("\n=== Sector exposure ===")
        for sec, wt in A.sector_exposure.items():
            print(f"  {sec:<26} {wt:6.1%}")

    if not A.ranked.empty:
        print(f"\n=== Top {args.top} suggested additions ===")
        cols = ["fit_score", "corr_to_portfolio", "vol_delta", "sharpe_delta", "rationale"]
        table = A.ranked.head(args.top)[[c for c in cols if c in A.ranked.columns]].copy()
        table["vol_delta"] = (table["vol_delta"] * 100).round(2)
        table = table.round(3)
        print(table.to_string())


if __name__ == "__main__":
    main()
