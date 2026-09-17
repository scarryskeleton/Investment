"""Candidate universes: a curated ETF set plus the current S&P 500."""

from __future__ import annotations

import io

import pandas as pd
import requests

# A compact set of liquid, broad exposures - "steady market" building blocks
# useful for diversifying a concentrated single-stock portfolio.
CURATED_ETFS: dict[str, tuple[str, str]] = {
    # Broad equity
    "VTI": ("US Total Market", "US Equity"),
    "SPY": ("S&P 500", "US Equity"),
    "RSP": ("S&P 500 Equal Weight", "US Equity"),
    "VTV": ("US Large-Cap Value", "US Equity"),
    "VUG": ("US Large-Cap Growth", "US Equity"),
    "IJR": ("US Small-Cap", "US Equity"),
    "MDY": ("US Mid-Cap", "US Equity"),
    # International
    "VEA": ("Developed ex-US", "Intl Equity"),
    "VWO": ("Emerging Markets", "Intl Equity"),
    "EFA": ("EAFE", "Intl Equity"),
    "EWJ": ("Japan", "Intl Equity"),
    # Sectors
    "XLK": ("Technology", "US Sector"),
    "XLV": ("Health Care", "US Sector"),
    "XLF": ("Financials", "US Sector"),
    "XLE": ("Energy", "US Sector"),
    "XLU": ("Utilities", "US Sector"),
    "XLP": ("Consumer Staples", "US Sector"),
    "XLY": ("Consumer Discretionary", "US Sector"),
    "XLI": ("Industrials", "US Sector"),
    "XLB": ("Materials", "US Sector"),
    "XLRE": ("Real Estate", "US Sector"),
    "XLC": ("Communication Services", "US Sector"),
    "ITA": ("Aerospace & Defense", "US Sector"),
    # Fixed income
    "BND": ("US Aggregate Bonds", "Fixed Income"),
    "IEF": ("7-10Y Treasuries", "Fixed Income"),
    "TLT": ("20Y+ Treasuries", "Fixed Income"),
    "SHY": ("1-3Y Treasuries", "Fixed Income"),
    "TIP": ("TIPS (inflation-linked)", "Fixed Income"),
    "LQD": ("Investment-Grade Credit", "Fixed Income"),
    "HYG": ("High-Yield Credit", "Fixed Income"),
    # Real assets / alternatives
    "GLD": ("Gold", "Real Assets"),
    "SLV": ("Silver", "Real Assets"),
    "DBC": ("Broad Commodities", "Real Assets"),
    "VNQ": ("US REITs", "Real Assets"),
    # Crypto
    "IBIT": ("Bitcoin", "Crypto"),
}


# A hand-picked spread of large, liquid single stocks across sectors and
# regions - a browsable "stocks" list without the slowness of the full S&P 500.
# (Non-US names use their US-listed line where one exists.)
POPULAR_STOCKS: list[str] = [
    # US tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "CRM",
    "ADBE", "ORCL", "CSCO", "IBM", "QCOM", "TXN", "INTC", "PLTR",
    # Financials (incl. fintech)
    "JPM", "BAC", "V", "MA", "GS", "BRK-B", "PYPL",
    # Health care
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO",
    # Energy
    "XOM", "CVX", "COP", "SLB",
    # Industrials
    "CAT", "GE", "HON", "DE", "UPS",
    # Defense & aerospace ("war" stocks)
    "LMT", "RTX", "NOC", "GD",
    # Aviation (airlines + planemaker)
    "BA", "DAL", "UAL", "LUV",
    # Mining, metals & other resources
    "BHP", "RIO", "FCX", "NEM", "VALE", "SCCO",
    # Real estate (REITs)
    "AMT", "PLD", "SPG", "O", "EQIX",
    # Consumer staples & discretionary
    "KO", "PEP", "PG", "COST", "WMT", "HD", "MCD", "NKE", "SBUX", "DIS", "NFLX",
    "T", "VZ",
    # Retail specifically
    "TGT", "LOW", "TJX",
    # Crypto-related equities (the coins themselves are in CRYPTO_FX)
    "COIN", "MSTR", "MARA",
    # International (US listings / ADRs)
    "ASML", "TSM", "NVO", "SAP", "SHEL", "AZN", "TM", "BABA", "TTE", "UL", "BP",
    "SONY", "HSBC",
]

# Spot cryptocurrencies and major currency pairs - not companies or funds, so
# they show up in Research with no fundamentals (no P/E, no sector), but their
# price history, volatility and vs-benchmark comparison all work the same way.
# A separate list because mixing them into POPULAR_STOCKS would misleadingly
# imply they're equities.
CRYPTO_FX: list[str] = [
    # Crypto (priced in USD)
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
    # Major currency pairs
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCNY=X",
]


def curated_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [(t, n, c) for t, (n, c) in CURATED_ETFS.items()],
        columns=["ticker", "name", "category"],
    ).set_index("ticker")


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents, scraped from Wikipedia.

    ``pd.read_html(url)`` fetches with urllib's default User-Agent, which
    Wikipedia's edge returns a 403 for - fetch the page ourselves with a
    real one and hand the HTML to read_html instead.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    resp = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PortfolioResearchTool/1.0)"},
        timeout=10,
    )
    resp.raise_for_status()
    tables = pd.read_html(io.StringIO(resp.text))
    syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
    return sorted(syms.str.strip().str.upper().tolist())


def build_candidate_universe(
    watchlist: list[str] | None = None,
    include_curated_etfs: bool = True,
    include_sp500: bool = False,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the pool of tickers to evaluate as additions.

    Returns a DataFrame indexed by ticker with 'name', 'category', 'source'.
    """
    exclude = {t.upper() for t in (exclude or [])}
    rows: dict[str, dict] = {}

    def add(ticker, name, category, source):
        ticker = ticker.upper().strip()
        if not ticker or ticker in exclude or ticker in rows:
            return
        rows[ticker] = {"name": name, "category": category, "source": source}

    for t in watchlist or []:
        add(t, t, "Watchlist", "watchlist")

    if include_curated_etfs:
        for t, (n, c) in CURATED_ETFS.items():
            add(t, n, c, "curated_etf")

    if include_sp500:
        try:
            for t in sp500_tickers():
                add(t, t, "S&P 500", "sp500")
        except Exception:
            pass

    return pd.DataFrame.from_dict(rows, orient="index")
