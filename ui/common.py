"""Shared state, formatting and cached data helpers used by two or more of
the app's four modes.

The one thing worth understanding about this module: ``ACCOUNT_CCY`` is a
plain string that app.py *reassigns* every rerun (from the sidebar currency
picker), not just mutates. A mode module must read it as ``common.ACCOUNT_CCY``
(module-qualified) rather than ``from ui.common import ACCOUNT_CCY`` — the
latter would freeze a stale copy at import time. ``EUR``/``EUR2`` don't have
this problem since app.py only ever mutates their ``.ccy`` attribute in
place, so importing those two by name is safe.
"""

from __future__ import annotations

import streamlit as st

from pa import education, fx

H = education.METRIC_HELP
PCT = "{:.1%}"
NEW_PROFILE = "➕ New profile…"
UNSAVED = "— unsaved / new —"


class _Fmt:
    """Currency-aware money formatter.

    ``.format(x)`` is kept so the many existing call sites need no change; the
    bound currency is swapped at runtime once the user's choice is known.
    """

    def __init__(self, ccy: str = "EUR", dp: int = 0) -> None:
        self.ccy = ccy
        self.dp = dp

    @property
    def sym(self) -> str:
        return fx.symbol(self.ccy)

    def format(self, x) -> str:
        try:
            return f"{self.sym}{float(x):,.{self.dp}f}"
        except (TypeError, ValueError):
            return f"{self.sym}{x}"

    def m(self, x) -> str:
        """Markdown-safe: escape ``$`` so Streamlit doesn't read it as LaTeX."""
        return self.format(x).replace("$", r"\$")


# Module-level singletons; their ``.ccy`` is rebound in app.py's sidebar to
# the account currency, which propagates to every mode automatically since
# they're the same objects everywhere they're imported.
EUR = _Fmt("EUR", 0)
EUR2 = _Fmt("EUR", 2)

# Reassigned (not mutated) each rerun by app.py's sidebar — read this as
# ``common.ACCOUNT_CCY``, never via a bare `from ui.common import ACCOUNT_CCY`.
ACCOUNT_CCY = "EUR"


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _latest_prices(tickers):
    from pa import data

    px_ = data.fetch_prices(list(tickers), lookback_years=0.2)
    return {} if px_.empty else px_.iloc[-1].to_dict()


@st.cache_data(show_spinner=False, ttl=60 * 15)
def _intraday(ticker):
    from pa import data

    return data.fetch_intraday(ticker)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _origins(tickers):
    """{ticker: {country, currency, name, sector, industry}} for a set of tickers."""
    from pa import data

    if not tickers:
        return {}
    f = data.fetch_fundamentals(list(tickers))
    out = {}
    for t in tickers:
        row = f.loc[t].to_dict() if t in f.index else {}
        out[t] = {
            "country": (row.get("country") or "").strip(),
            "currency": (row.get("currency") or "").strip(),
            "name": row.get("name") or t,
            "sector": (row.get("sector") or "").strip() or "Unknown",
            "industry": (row.get("industry") or "").strip() or "Unknown",
            "quote_type": (row.get("quote_type") or "").strip(),
        }
    return out


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _fx_spot(native, account):
    """Multiplier: 1 unit of `native` currency -> how many `account` units, now."""
    if not native or native.upper() == account.upper():
        return 1.0
    try:
        return float(fx.convert(native, account))
    except Exception:
        return 1.0


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _fx_convert_history(prices, ticker_ccy, account):
    """Convert a price-history DataFrame to `account` currency, per-column, with
    each pair's own daily FX history."""
    try:
        return fx.convert_frame(prices, dict(ticker_ccy), account)
    except Exception:
        return prices


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _universe_prices(tickers, lookback):
    from pa import data

    return data.fetch_prices(list(tickers), lookback)


# country name -> flag emoji, for the names yfinance returns most often
_FLAGS = {
    "United States": "🇺🇸", "United Kingdom": "🇬🇧", "Netherlands": "🇳🇱",
    "Germany": "🇩🇪", "France": "🇫🇷", "Switzerland": "🇨🇭", "Ireland": "🇮🇪",
    "Canada": "🇨🇦", "Japan": "🇯🇵", "China": "🇨🇳", "Hong Kong": "🇭🇰",
    "Taiwan": "🇹🇼", "South Korea": "🇰🇷", "India": "🇮🇳", "Australia": "🇦🇺",
    "Spain": "🇪🇸", "Italy": "🇮🇹", "Sweden": "🇸🇪", "Denmark": "🇩🇰",
    "Norway": "🇳🇴", "Finland": "🇫🇮", "Belgium": "🇧🇪", "Brazil": "🇧🇷",
    "Mexico": "🇲🇽", "Israel": "🇮🇱", "Singapore": "🇸🇬", "Luxembourg": "🇱🇺",
    "Austria": "🇦🇹", "Portugal": "🇵🇹", "New Zealand": "🇳🇿", "South Africa": "🇿🇦",
}


def _flag(country: str) -> str:
    return _FLAGS.get((country or "").strip(), "🌐")
