"""Price and fundamentals data access, backed by yfinance with a disk cache."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

_PRICE_CACHE = CACHE_DIR / "prices"
_PRICE_CACHE.mkdir(exist_ok=True)
_INFO_CACHE = CACHE_DIR / "info_v2.json"
_EST_CACHE = CACHE_DIR / "estimates.json"
_EST_TTL = timedelta(days=2)

# How long a cached daily-price file stays fresh.
_PRICE_TTL = timedelta(hours=12)
_INFO_TTL = timedelta(days=7)


def _norm(tickers) -> list[str]:
    seen, out = set(), []
    for t in tickers:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _price_path(ticker: str, start: str, end: str) -> Path:
    safe = ticker.replace("/", "-").replace("^", "_")
    return _PRICE_CACHE / f"{safe}__{start}__{end}.parquet"


def fetch_prices(
    tickers,
    lookback_years: float = 5.0,
    end: date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame of adjusted close prices, one column per ticker.

    Tickers that return no data are silently dropped; check the columns of the
    result against your request to see which succeeded.
    """
    tickers = _norm(tickers)
    if not tickers:
        return pd.DataFrame()

    end = end or date.today()
    start = end - timedelta(days=int(365.25 * lookback_years) + 5)
    start_s, end_s = start.isoformat(), end.isoformat()

    frames: dict[str, pd.Series] = {}
    missing: list[str] = []

    for t in tickers:
        p = _price_path(t, start_s, end_s)
        if use_cache and p.exists() and p.stat().st_size > 0:
            age = datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)
            if age < _PRICE_TTL:
                try:
                    s = pd.read_parquet(p)["price"]
                    s.name = t
                    frames[t] = s
                    continue
                except Exception:
                    try:
                        p.unlink()
                    except OSError:
                        pass
        missing.append(t)

    if missing:
        raw = yf.download(
            missing,
            start=start_s,
            end=end_s,
            auto_adjust=True,
            progress=False,
            group_by="column",
        )
        if not raw.empty:
            close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            if isinstance(close, pd.Series):  # single ticker
                close = close.to_frame(missing[0])
            for t in missing:
                if t in close.columns:
                    s = close[t].dropna()
                    if len(s) < 30:
                        continue
                    s.name = t
                    frames[t] = s
                    if use_cache:
                        try:
                            dest = _price_path(t, start_s, end_s)
                            tmp = dest.with_suffix(".parquet.tmp")
                            s.to_frame("price").to_parquet(tmp)
                            tmp.replace(dest)
                        except Exception:
                            pass

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames.values(), axis=1)
    df = df.sort_index().ffill().dropna(how="all")
    # Keep only rows where at least the longest-history column is present, then
    # drop leading rows with any NaN so every series starts together.
    df = df.dropna()
    return df


def _load_info_cache() -> dict:
    if _INFO_CACHE.exists():
        try:
            return json.loads(_INFO_CACHE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _save_info_cache(cache: dict) -> None:
    try:
        tmp = _INFO_CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=0))
        tmp.replace(_INFO_CACHE)
    except OSError:
        pass


def fetch_fundamentals(tickers) -> pd.DataFrame:
    """Return per-ticker sector, industry, market cap and name.

    yfinance's info endpoint is slow and occasionally flaky, so results are
    cached on disk for a week and failures degrade to 'Unknown'.
    """
    tickers = _norm(tickers)
    cache = _load_info_cache()
    now = time.time()
    rows = {}
    dirty = False

    for t in tickers:
        entry = cache.get(t)
        if entry and (now - entry.get("_ts", 0)) < _INFO_TTL.total_seconds():
            rows[t] = entry
            continue
        info = {}
        try:
            info = yf.Ticker(t).get_info() or {}
        except Exception:
            info = {}
        # Current yfinance reports these yield fields as percent numbers
        # (0.34 = 0.34%, 4.04 = 4.04%), so convert to a fraction. The older
        # `trailingAnnualDividendYield` is already a fraction.
        div = info.get("dividendYield")
        div = div / 100.0 if div else info.get("trailingAnnualDividendYield")
        fund_yield = info.get("yield")  # already a fraction in current yfinance
        entry = {
            "name": info.get("shortName") or info.get("longName") or t,
            "long_name": info.get("longName") or info.get("shortName") or t,
            "sector": info.get("sector") or "Unknown",
            "industry": info.get("industry") or info.get("category") or "Unknown",
            "market_cap": info.get("marketCap") or float("nan"),
            "quote_type": info.get("quoteType") or "Unknown",
            "summary": info.get("longBusinessSummary") or "",
            "country": info.get("country") or "",
            "website": info.get("website") or "",
            "employees": info.get("fullTimeEmployees") or None,
            "currency": info.get("currency") or "",
            "category": info.get("category") or "",
            "trailing_pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "dividend_yield": div,
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "return_on_equity": info.get("returnOnEquity"),
            "beta": info.get("beta"),
            "fifty_two_low": info.get("fiftyTwoWeekLow"),
            "fifty_two_high": info.get("fiftyTwoWeekHigh"),
            "target_mean": info.get("targetMeanPrice"),
            "n_analysts": info.get("numberOfAnalystOpinions"),
            "recommendation": info.get("recommendationKey") or "",
            "ytd_return": info.get("ytdReturn"),
            "yield": fund_yield,
            "_ts": now,
        }
        cache[t] = entry
        rows[t] = entry
        dirty = True

    if dirty:
        _save_info_cache(cache)

    df = pd.DataFrame.from_dict(rows, orient="index")
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


def fetch_profile(ticker: str) -> dict:
    """The full cached info entry for one ticker (see fetch_fundamentals)."""
    df = fetch_fundamentals([ticker])
    t = str(ticker).strip().upper()
    if t in df.index:
        return df.loc[t].to_dict()
    return {}


def market_caps(tickers) -> dict[str, float]:
    f = fetch_fundamentals(tickers)
    return {t: float(v) for t, v in f["market_cap"].items() if pd.notna(v) and v > 0}


def _num_or_none(v):
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def fetch_estimates(ticker: str) -> dict:
    """Analyst consensus for the upcoming quarter and fiscal year.

    Returns {} for funds / tickers Yahoo has no estimates for. Structure:
      next_earnings: ISO date or None
      periods: {"+1q"|"+1y"|"0y": {rev_avg, rev_growth, eps_avg, eps_low,
               eps_high, eps_growth, eps_yago, n_analysts}}
      eps_trend_90d: {period: pct change in the estimate over ~90 days}
      revisions_30d: {period: {up, down}}
    """
    t = str(ticker).strip().upper()
    cache = {}
    if _EST_CACHE.exists():
        try:
            cache = json.loads(_EST_CACHE.read_text())
        except json.JSONDecodeError:
            cache = {}
    entry = cache.get(t)
    if entry and (time.time() - entry.get("_ts", 0)) < _EST_TTL.total_seconds():
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    out: dict = {"periods": {}, "eps_trend_90d": {}, "revisions_30d": {}}
    try:
        yt = yf.Ticker(t)
        cal = yt.calendar or {}
        ed = cal.get("Earnings Date")
        if isinstance(ed, list) and ed:
            out["next_earnings"] = ed[0].isoformat()
        elif ed:
            out["next_earnings"] = str(ed)

        ee = yt.earnings_estimate
        re_ = yt.revenue_estimate
        et = yt.eps_trend
        rev = yt.eps_revisions
        for p in ("+1q", "+1y", "0y"):
            row: dict = {}
            if ee is not None and p in ee.index:
                r = ee.loc[p]
                row.update(
                    eps_avg=_num_or_none(r.get("avg")),
                    eps_low=_num_or_none(r.get("low")),
                    eps_high=_num_or_none(r.get("high")),
                    eps_yago=_num_or_none(r.get("yearAgoEps")),
                    eps_growth=_num_or_none(r.get("growth")),
                    n_analysts=_num_or_none(r.get("numberOfAnalysts")),
                )
            if re_ is not None and p in re_.index:
                r = re_.loc[p]
                row.update(
                    rev_avg=_num_or_none(r.get("avg")),
                    rev_growth=_num_or_none(r.get("growth")),
                )
            if row:
                out["periods"][p] = row
            if et is not None and p in et.index:
                now = _num_or_none(et.loc[p].get("current"))
                then = _num_or_none(et.loc[p].get("90daysAgo"))
                if now and then:
                    out["eps_trend_90d"][p] = now / then - 1.0
            if rev is not None and p in rev.index:
                out["revisions_30d"][p] = {
                    "up": _num_or_none(rev.loc[p].get("upLast30days")) or 0,
                    "down": _num_or_none(rev.loc[p].get("downLast30days")) or 0,
                }
    except Exception:
        pass

    if not out["periods"] and "next_earnings" not in out:
        out = {}

    cache[t] = {**out, "_ts": time.time()}
    try:
        tmp = _EST_CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=0, default=str))
        tmp.replace(_EST_CACHE)
    except OSError:
        pass
    return out


_DOCS_CACHE = CACHE_DIR / "docs.json"
_DOCS_TTL = timedelta(hours=18)


def fetch_company_docs(ticker: str) -> dict:
    """Where to read the company's own plans: IR site, recent SEC filings, news.

    Returns {ir_website, filings: [{date, type, title, url}], news: [{title,
    publisher, date, url}]}. Empty pieces for funds / when Yahoo has nothing.
    """
    t = str(ticker).strip().upper()
    cache = {}
    if _DOCS_CACHE.exists():
        try:
            cache = json.loads(_DOCS_CACHE.read_text())
        except json.JSONDecodeError:
            cache = {}
    entry = cache.get(t)
    if entry and (time.time() - entry.get("_ts", 0)) < _DOCS_TTL.total_seconds():
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    out = {"ir_website": "", "filings": [], "news": []}
    try:
        yt = yf.Ticker(t)
        info = yt.get_info() or {}
        out["ir_website"] = info.get("irWebsite") or ""

        # words to keep a news headline as "relevant"
        nm = (info.get("shortName") or info.get("longName") or "").lower()
        kw = {t.lower()} | {w.strip(",.") for w in nm.split()
                            if len(w) > 3 and w.lower() not in
                            ("corp", "corporation", "inc", "company", "the", "holdings",
                             "group", "plc", "ltd", "limited", "class")}

        for f in (yt.sec_filings or [])[:30]:
            d = f.get("date")
            out["filings"].append({
                "date": d.isoformat() if hasattr(d, "isoformat") else str(d),
                "type": f.get("type") or "",
                "title": f.get("title") or "",
                "url": f.get("edgarUrl") or "",
            })

        seen = set()
        for a in (yt.news or [])[:20]:
            c = a.get("content", a) or {}
            title = c.get("title")
            if not title or title in seen:
                continue
            if kw and not any(k in title.lower() for k in kw):
                continue
            seen.add(title)
            prov = c.get("provider")
            pub = prov.get("displayName") if isinstance(prov, dict) else c.get("publisher")
            url = c.get("canonicalUrl")
            url = url.get("url") if isinstance(url, dict) else (c.get("link") or "")
            out["news"].append({
                "title": title,
                "publisher": pub or "",
                "date": str(c.get("pubDate") or c.get("displayTime") or "")[:10],
                "url": url,
            })
            if len(out["news"]) >= 6:
                break
    except Exception:
        pass

    cache[t] = {**out, "_ts": time.time()}
    try:
        tmp = _DOCS_CACHE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cache, indent=0, default=str))
        tmp.replace(_DOCS_CACHE)
    except OSError:
        pass
    return out


def latest_report_links(docs: dict) -> dict:
    """Pick the most recent annual and interim report (US or foreign filer)."""
    annual = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F"}
    interim = {"10-Q", "10-Q/A", "6-K"}
    out = {}
    for f in docs.get("filings", []):
        ty = (f.get("type") or "").upper()
        if ty in annual and "annual" not in out:
            out["annual"] = f
        elif ty in interim and "quarterly" not in out:
            out["quarterly"] = f
    return out


def search_symbols(query: str, limit: int = 10) -> list[dict]:
    """Look up tickers by company / fund name (or partial ticker).

    Returns [{symbol, name, type, exchange}], best match first. Uses Yahoo's
    search endpoint via yfinance, with a direct HTTP fallback.
    """
    query = (query or "").strip()
    if len(query) < 2:
        return []

    quotes: list = []
    try:
        s = yf.Search(query, max_results=limit)
        quotes = list(getattr(s, "quotes", []) or [])
    except Exception:
        quotes = []
    if not quotes:
        try:
            import requests

            r = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query, "quotesCount": limit, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=6,
            )
            quotes = r.json().get("quotes", [])
        except Exception:
            quotes = []

    out: list[dict] = []
    for q in quotes:
        sym = q.get("symbol")
        if not sym or q.get("quoteType") in ("OPTION", "FUTURE"):
            continue
        out.append({
            "symbol": sym,
            "name": q.get("longname") or q.get("shortname") or sym,
            "type": str(q.get("quoteType") or "").title(),
            "exchange": q.get("exchDisp") or q.get("exchange") or "",
        })
        if len(out) >= limit:
            break
    return out
