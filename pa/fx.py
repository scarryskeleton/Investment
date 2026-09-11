"""Currency conversion via Yahoo FX pairs.

Yahoo quotes spot and daily-history FX as ``BASEQUOTE=X`` (e.g. ``EURUSD=X`` is
"US dollars per 1 euro"). We reuse :func:`pa.data.fetch_prices` so the same
disk cache and retry logic covers FX too.

``convert(native, account)`` gives the number you multiply a *native*-currency
amount by to get an *account*-currency amount, either at spot (a float) or as a
date-indexed Series for historical reconstructions.
"""

from __future__ import annotations

import pandas as pd

from pa import data

# Account currencies offered in the UI -> display symbol. The symbol is a
# prefix; multi-character codes carry a trailing space.
SYMBOLS: dict[str, str] = {
    "EUR": "€", "USD": "$", "GBP": "£", "JPY": "¥", "CHF": "CHF ",
    "CAD": "C$", "AUD": "A$", "NZD": "NZ$", "SEK": "kr ", "NOK": "kr ",
    "DKK": "kr ", "PLN": "zł ", "CZK": "Kč ", "HUF": "Ft ",
    "SGD": "S$", "HKD": "HK$", "INR": "₹", "CNY": "CN¥", "KRW": "₩",
    "TWD": "NT$", "BRL": "R$", "MXN": "MX$", "ZAR": "R ", "ILS": "₪",
    "AED": "AED ", "SAR": "SAR ", "TRY": "₺",
}

CURRENCIES: list[str] = list(SYMBOLS)


def symbol(ccy: str) -> str:
    return SYMBOLS.get((ccy or "").strip().upper(), f"{(ccy or '').strip().upper()} ")


def normalize(raw: str) -> tuple[str, float]:
    """Map a raw Yahoo currency code to (ISO code, scale).

    Yahoo quotes some London lines in pence as ``GBp`` / ``GBX`` and a few
    others in minor units; ``scale`` converts a raw quote into the major unit.
    """
    c = (raw or "").strip()
    if c in ("GBp", "GBX", "GBX="):
        return "GBP", 0.01
    if c in ("ZAc", "ZAX"):
        return "ZAR", 0.01
    if c in ("ILA",):
        return "ILS", 0.01
    return c.upper(), 1.0


def _pair(base: str, quote: str, lookback_years: float) -> pd.Series | None:
    """Series of 'quote units per 1 base unit', or None if Yahoo has neither leg."""
    if base == quote:
        return None
    direct, inverse = f"{base}{quote}=X", f"{quote}{base}=X"
    df = data.fetch_prices([direct], lookback_years=lookback_years)
    if direct in df.columns and not df[direct].dropna().empty:
        return df[direct].dropna()
    df = data.fetch_prices([inverse], lookback_years=lookback_years)
    if inverse in df.columns and not df[inverse].dropna().empty:
        return (1.0 / df[inverse].dropna()).rename(direct)
    return None


def convert(
    native: str,
    account: str,
    index: pd.Index | None = None,
    lookback_years: float = 15.0,
) -> float | pd.Series:
    """Multiplier turning a ``native``-currency amount into ``account`` currency.

    With ``index`` given, returns a Series reindexed (ffill/bfill) onto it; the
    fallback when Yahoo has no data for the pair is a flat 1.0 * scale, which at
    least keeps magnitudes sane.
    """
    nat, scale = normalize(native)
    acc, _ = normalize(account)

    if not nat or nat == acc:
        series_val = scale
        if index is None:
            return series_val
        return pd.Series(series_val, index=index)

    s = _pair(nat, acc, lookback_years)
    if s is None:
        if index is None:
            return scale  # unknown pair: assume parity
        return pd.Series(scale, index=index)

    s = s * scale
    if index is None:
        return float(s.iloc[-1])

    idx = pd.to_datetime(index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s.reindex(s.index.union(idx)).sort_index().ffill().bfill().reindex(idx)


def convert_frame(
    prices: pd.DataFrame, ticker_ccy: dict[str, str], account: str
) -> pd.DataFrame:
    """Convert each price column from its native currency to ``account``,
    using that pair's daily history aligned to ``prices.index``."""
    if prices is None or prices.empty:
        return prices
    out = prices.copy()
    for col in out.columns:
        mult = convert(ticker_ccy.get(col, account), account, index=out.index)
        out[col] = out[col] * (mult.values if hasattr(mult, "values") else mult)
    return out
