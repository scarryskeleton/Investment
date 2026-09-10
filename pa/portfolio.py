"""Parse user holdings input into normalized weights."""

from __future__ import annotations

import io

import pandas as pd

from . import data


def parse_holdings_text(text: str) -> pd.DataFrame:
    """Accept lines like 'AAPL, 50', 'MSFT 30', 'VTI,0.25' or CSV with headers.

    Returns a DataFrame indexed by ticker with a single 'amount' column.
    The amount's meaning (shares / dollars / weight) is decided later.
    """
    text = (text or "").strip()
    if not text:
        return pd.DataFrame(columns=["amount"])

    # Try CSV with headers first.
    try:
        df = pd.read_csv(io.StringIO(text))
        cols = {c.lower().strip(): c for c in df.columns}
        tcol = next((cols[k] for k in ("ticker", "symbol", "stock") if k in cols), None)
        acol = next(
            (cols[k] for k in ("amount", "shares", "quantity", "qty", "weight", "value", "dollars") if k in cols),
            None,
        )
        if tcol is not None and acol is not None:
            out = df[[tcol, acol]].copy()
            out.columns = ["ticker", "amount"]
            out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
            out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
            return out.dropna().groupby("ticker").sum()
    except Exception:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip().replace("\t", " ")
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if not parts:
            continue
        ticker = parts[0].upper()
        amount = 1.0
        if len(parts) > 1:
            try:
                amount = float(parts[1].replace("$", "").replace("%", ""))
            except ValueError:
                amount = 1.0
        rows.append((ticker, amount))
    if not rows:
        return pd.DataFrame(columns=["amount"])
    return pd.DataFrame(rows, columns=["ticker", "amount"]).groupby("ticker").sum()


def normalize_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a free-form edited grid (Ticker / Amount columns, any case) into the
    canonical frame: indexed by upper-case ticker with a single 'amount' column.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["amount"])
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[k] for k in ("ticker", "symbol", "stock") if k in cols), None)
    acol = next(
        (cols[k] for k in ("amount", "shares", "quantity", "qty", "weight", "value") if k in cols),
        None,
    )
    if tcol is None or acol is None:
        return pd.DataFrame(columns=["amount"])
    out = df[[tcol, acol]].copy()
    out.columns = ["ticker", "amount"]
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[(out["ticker"] != "") & out["ticker"].ne("NAN") & out["amount"].notna() & (out["amount"] > 0)]
    return out.groupby("ticker").sum()


def to_weights(
    holdings: pd.DataFrame,
    mode: str = "shares",
    latest_prices: pd.Series | None = None,
) -> dict[str, float]:
    """Convert parsed holdings to weights summing to 1.

    mode: 'shares' | 'dollars' | 'weight'
    """
    if holdings.empty:
        return {}
    amt = holdings["amount"].astype(float)

    if mode == "weight":
        vals = amt
    elif mode == "dollars":
        vals = amt
    else:  # shares
        if latest_prices is None:
            latest_prices = _latest_prices(list(amt.index))
        vals = amt * latest_prices.reindex(amt.index)

    vals = vals.dropna()
    total = vals.sum()
    if not total:
        return {}
    return (vals / total).to_dict()


def _latest_prices(tickers: list[str]) -> pd.Series:
    px = data.fetch_prices(tickers, lookback_years=0.2)
    if px.empty:
        return pd.Series(dtype=float)
    return px.iloc[-1]


SAMPLE = """# ticker, shares  -  a tech-heavy example portfolio
AAPL, 40
MSFT, 25
NVDA, 15
GOOGL, 20
AMZN, 12
META, 10
TSLA, 8
"""
