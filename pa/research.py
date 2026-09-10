"""Explore a single stock or bond against the market, with business context.

Pulls the yfinance business profile (what the company does, key valuation and
quality stats, analyst view) and measures the security against a benchmark -
relative performance, beta, correlation, up/down capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RelativeStats:
    benchmark: str
    asset_return: float
    bench_return: float
    excess: float
    beta: float
    corr: float
    up_capture: float
    down_capture: float
    rel_strength: pd.Series      # cumulative asset / benchmark, rebased to 1.0
    rebased: pd.DataFrame        # asset and benchmark, both rebased to 100


def vs_market(prices: pd.Series, bench_prices: pd.Series, benchmark: str = "SPY") -> RelativeStats:
    a = prices.dropna()
    b = bench_prices.reindex(a.index).dropna()
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]
    ra, rb = a.pct_change().dropna(), b.pct_change().dropna()
    df = pd.concat([ra, rb], axis=1, join="inner").dropna()
    df.columns = ["a", "b"]

    tot_a = (1 + df["a"]).prod() - 1
    tot_b = (1 + df["b"]).prod() - 1
    cov = np.cov(df["a"], df["b"])
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else float("nan")
    corr = float(df["a"].corr(df["b"]))

    # capture ratios on monthly returns (the usual convention)
    m = pd.concat(
        [a.resample("ME").last().pct_change(), b.resample("ME").last().pct_change()],
        axis=1, join="inner",
    ).dropna()
    m.columns = ["a", "b"]
    up, dn = m[m["b"] > 0], m[m["b"] < 0]

    def _cap(g):
        gb = (1 + g["b"]).prod() - 1
        ga = (1 + g["a"]).prod() - 1
        return float(ga / gb) if abs(gb) > 1e-9 else float("nan")

    rel = ((1 + df["a"]).cumprod()) / ((1 + df["b"]).cumprod())
    rel = rel / rel.iloc[0]
    rebased = pd.DataFrame({
        prices.name or "asset": a / a.iloc[0] * 100,
        benchmark: b / b.iloc[0] * 100,
    })

    return RelativeStats(
        benchmark=benchmark,
        asset_return=float(tot_a),
        bench_return=float(tot_b),
        excess=float(tot_a - tot_b),
        beta=beta,
        corr=corr,
        up_capture=_cap(up) if len(up) else float("nan"),
        down_capture=_cap(dn) if len(dn) else float("nan"),
        rel_strength=rel,
        rebased=rebased,
    )


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def browse_table(
    tickers: list[str],
    prices: pd.DataFrame,
    bench_prices: pd.Series,
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per ticker: fast price-derived stats plus any cached fundamentals.

    Columns are pre-formatted strings except the numeric sort helpers, which are
    dropped before display by the caller if desired.
    """
    f = fundamentals if fundamentals is not None else pd.DataFrame()
    br = bench_prices.pct_change().dropna() if bench_prices is not None else None
    rows = []
    for t in tickers:
        if t not in prices.columns:
            continue
        s = prices[t].dropna()
        if len(s) < 60:
            continue
        r = s.pct_change().dropna()
        yr = 252
        ret_1y = s.iloc[-1] / s.iloc[-yr] - 1 if len(s) > yr else s.iloc[-1] / s.iloc[0] - 1
        vol = r.std() * (252 ** 0.5)
        dd = float((s / s.cummax() - 1).min())
        from_high = s.iloc[-1] / s.iloc[-yr:].max() - 1 if len(s) > 20 else 0.0
        beta = float("nan")
        excess = float("nan")
        if br is not None:
            j = pd.concat([r, br], axis=1, join="inner").dropna()
            if len(j) > 40:
                cov = j.cov().values
                beta = cov[0, 1] / cov[1, 1] if cov[1, 1] else float("nan")
                bret_1y = (bench_prices.iloc[-1] / bench_prices.iloc[-yr] - 1
                           if len(bench_prices) > yr else float("nan"))
                excess = ret_1y - bret_1y if bret_1y == bret_1y else float("nan")

        fr = f.loc[t].to_dict() if t in f.index else {}
        qt = str(fr.get("quote_type", "")).upper()
        typ = "ETF" if qt in ("ETF", "MUTUALFUND") else "Stock" if qt == "EQUITY" else "—"

        rows.append({
            "Ticker": t,
            "Name": (fr.get("name") or t)[:34],
            "Type": typ,
            "Sector / category": fr.get("sector") if fr.get("sector") not in (None, "Unknown")
            else (fr.get("industry") or "—"),
            "1y return": ret_1y,
            "vs bench (1y)": excess,
            "Volatility": vol,
            "Max drawdown": dd,
            "From 12mo high": from_high,
            "P/E": _num(fr.get("trailing_pe")),
            "Yield": _num(fr.get("dividend_yield")) if fr.get("dividend_yield") else _num(fr.get("yield")),
            "Beta": beta,
        })
    df = pd.DataFrame(rows)
    return df.set_index("Ticker") if not df.empty else df


def key_stats(profile: dict, last_price: float | None = None) -> list[tuple[str, str]]:
    """A tidy (label, formatted value) list for a stats table."""
    p = profile
    out: list[tuple[str, str]] = []

    def pct(v, d=1):
        v = _num(v)
        return f"{v * 100:.{d}f}%" if v == v else "—"

    def num(v, d=1):
        v = _num(v)
        return f"{v:.{d}f}" if v == v else "—"

    mc = _num(p.get("market_cap"))
    if mc == mc and mc > 0:
        out.append(("Market cap", f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.1f}B"))
    out += [
        ("Trailing P/E", num(p.get("trailing_pe"))),
        ("Forward P/E", num(p.get("forward_pe"))),
        ("Price / book", num(p.get("price_to_book"))),
        ("Dividend yield", pct(p.get("dividend_yield"), 2)),
        ("Profit margin", pct(p.get("profit_margin"))),
        ("Revenue growth (yoy)", pct(p.get("revenue_growth"))),
        ("Return on equity", pct(p.get("return_on_equity"))),
        ("Beta (Yahoo)", num(p.get("beta"), 2)),
    ]
    lo, hi = _num(p.get("fifty_two_low")), _num(p.get("fifty_two_high"))
    if lo == lo and hi == hi:
        tag = ""
        if last_price and last_price == last_price:
            pos = (last_price - lo) / (hi - lo) if hi > lo else float("nan")
            if pos == pos:
                tag = f"  (now {pos:.0%} of the way up)"
        out.append(("52-week range", f"{lo:,.2f} – {hi:,.2f}{tag}"))
    tgt = _num(p.get("target_mean"))
    if tgt == tgt and tgt > 0:
        up = f"  ({(tgt/last_price - 1)*100:+.0f}% vs now)" if last_price else ""
        out.append((f"Analyst target ({int(_num(p.get('n_analysts')) or 0)} analysts)",
                    f"{tgt:,.2f}{up}"))
    if p.get("recommendation"):
        out.append(("Analyst consensus", str(p["recommendation"]).replace("_", " ").title()))
    y = _num(p.get("yield"))
    if y == y and y > 0:
        out.append(("Fund yield", pct(y, 2)))
    return [(k, v) for k, v in out if v not in ("—", "")]


def _money(v):
    v = _num(v)
    if v != v:
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"${v/1e12:.2f}T"
    if a >= 1e9:
        return f"${v/1e9:.1f}B"
    if a >= 1e6:
        return f"${v/1e6:.0f}M"
    return f"${v:,.2f}"


def estimates_summary(est: dict) -> tuple[list[tuple[str, str, str]], list[str]]:
    """(table rows, insight lines) for the 'What's expected next' section.

    Rows are (label, next-quarter value, next-year value).
    """
    if not est or not est.get("periods"):
        return [], []
    per = est["periods"]
    q, y = per.get("+1q", {}), per.get("+1y", {})

    def g(v):
        v = _num(v)
        return f"{v*100:+.0f}%" if v == v else "—"

    def eps(v):
        v = _num(v)
        return f"${v:.2f}" if v == v else "—"

    rows = [
        ("Revenue (consensus)", _money(q.get("rev_avg")), _money(y.get("rev_avg"))),
        ("  …growth vs a year ago", g(q.get("rev_growth")), g(y.get("rev_growth"))),
        ("Earnings per share", eps(q.get("eps_avg")), eps(y.get("eps_avg"))),
        ("  …growth vs a year ago", g(q.get("eps_growth")), g(y.get("eps_growth"))),
        ("  …range (low–high)",
         f"{eps(q.get('eps_low'))}–{eps(q.get('eps_high'))}" if q.get("eps_low") else "—",
         f"{eps(y.get('eps_low'))}–{eps(y.get('eps_high'))}" if y.get("eps_low") else "—"),
        ("Analysts contributing",
         f"{int(q['n_analysts'])}" if q.get("n_analysts") else "—",
         f"{int(y['n_analysts'])}" if y.get("n_analysts") else "—"),
    ]

    tips: list[str] = []
    yg_r, yg_e = _num(y.get("rev_growth")), _num(y.get("eps_growth"))
    if yg_r == yg_r and yg_e == yg_e:
        tips.append(f"Consensus has **revenue ~{yg_r*100:+.0f}%** and **EPS ~{yg_e*100:+.0f}%** "
                    "next fiscal year vs this one.")
    qg = _num(q.get("eps_growth"))
    if qg == qg and qg < -0.03:
        tips.append(f"Earnings are expected to **fall ~{abs(qg)*100:.0f}%** next quarter "
                    "versus the same quarter a year ago — analysts see a soft patch.")

    tr = est.get("eps_trend_90d", {}).get("+1y")
    tr = _num(tr)
    if tr == tr:
        if tr > 0.02:
            tips.append(f"Next year's EPS estimate has been **revised up ~{tr*100:.0f}%** "
                        "over the last 90 days — a good sign.")
        elif tr < -0.02:
            tips.append(f"Next year's EPS estimate has been **cut ~{abs(tr)*100:.0f}%** over "
                        "the last 90 days — analysts are turning more cautious.")

    rv = est.get("revisions_30d", {}).get("+1y", {})
    up, dn = _num(rv.get("up")), _num(rv.get("down"))
    if up == up and dn == dn and (up + dn) >= 3:
        tips.append(f"Last 30 days: **{int(up)} upward** vs **{int(dn)} downward** revisions to "
                    "the next-year estimate.")

    return rows, tips


def business_insights(profile: dict, rel: RelativeStats | None = None) -> list[str]:
    p = profile
    qt = str(p.get("quote_type", "")).upper()
    out: list[str] = []

    _cat = str(p.get("category") or p.get("industry") or "").lower()
    is_bond = any(w in _cat for w in ("bond", "treasury", "fixed income", "muni",
                                      "credit", "high yield", "tips", "aggregate"))
    if qt == "ETF" or qt == "MUTUALFUND" or is_bond:
        cat = p.get("category") or p.get("industry") or "fund"
        out.append(f"This is a **fund** ({cat}), not a single company — you're buying a "
                   "basket, so single-stock risk doesn't apply.")
        y = _num(p.get("yield")) or _num(p.get("dividend_yield"))
        if y == y and y > 0:
            if is_bond:
                out.append(f"Yields about **{y*100:.1f}%**. It's a bond fund — the price "
                           "moves *opposite* to interest rates (rates up, price down).")
            else:
                out.append(f"Yields about **{y*100:.1f}%** in dividends.")
    else:
        pe = _num(p.get("trailing_pe"))
        if pe == pe and pe > 0:
            if pe > 30:
                out.append(f"Trades at a **P/E of {pe:.0f}** — well above the ~20 long-run "
                           "market average, so a lot of future growth is already priced in.")
            elif pe < 12:
                out.append(f"Trades at a **P/E of {pe:.0f}** — cheap on the surface; check "
                           "whether that's a bargain or a warning.")
            else:
                out.append(f"P/E of **{pe:.0f}**, roughly in line with the market.")
        pm = _num(p.get("profit_margin"))
        if pm == pm:
            word = "very profitable" if pm > 0.2 else "profitable" if pm > 0.05 else \
                   "thin margins" if pm > 0 else "lossmaking"
            out.append(f"Net margin **{pm*100:.0f}%** — {word}.")
        rg = _num(p.get("revenue_growth"))
        if rg == rg:
            out.append(f"Revenue {'grew' if rg >= 0 else 'shrank'} **{abs(rg)*100:.0f}%** "
                       "over the last reported year.")
        dy = _num(p.get("dividend_yield"))
        if dy == dy and dy > 0:
            out.append(f"Pays a **{dy*100:.1f}%** dividend.")
        elif dy == 0 or dy != dy:
            out.append("Pays little or no dividend — return would come from the price.")

    tgt = _num(p.get("target_mean"))
    if tgt == tgt and tgt > 0 and p.get("recommendation"):
        out.append(f"Analyst consensus is **{str(p['recommendation']).replace('_',' ')}** "
                   f"with an average target of {tgt:,.2f}. Targets are frequently wrong; "
                   "treat them as one opinion.")

    if rel is not None and rel.beta == rel.beta:
        if rel.excess > 0.05:
            out.append(f"Has **beaten {rel.benchmark}** by {rel.excess*100:.0f} points over "
                       "this window — but past out-performance doesn't persist reliably.")
        elif rel.excess < -0.05:
            out.append(f"Has **lagged {rel.benchmark}** by {abs(rel.excess)*100:.0f} points "
                       "over this window.")
        if rel.down_capture == rel.down_capture:
            if rel.down_capture > 1.1:
                out.append(f"In down months it fell **{rel.down_capture:.0%}** as much as the "
                           "market — more painful on the way down.")
            elif rel.down_capture < 0.8:
                out.append(f"In down months it only fell **{rel.down_capture:.0%}** as much "
                           "as the market — relatively defensive.")

    return out
