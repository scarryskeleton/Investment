"""Portfolio suggestion dashboard.

Run with:  streamlit run app.py

Not investment advice. This is a decision-support tool that describes what
would have been efficient given historical data and explicit assumptions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pa import (
    education,
    explore,
    forecast,
    fx,
    ideacard,
    metrics,
    optimize,
    paper,
    plan as planning,
    portfolio,
    store,
    suggest,
    timing,
)

H = education.METRIC_HELP

st.set_page_config(page_title="Portfolio Suggestions", page_icon="📊", layout="wide")

PCT = "{:.1%}"
NEW_PROFILE = "➕ New profile…"
UNSAVED = "— unsaved / new —"

store.init()


# --------------------------------------------------------------------------- #
# Cached wrappers
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _run_analysis(holdings_weights, watchlist, curated, sp500, lookback, rf, add_w):
    return suggest.run_analysis(
        holdings_weights=dict(holdings_weights),
        watchlist=list(watchlist),
        include_curated_etfs=curated,
        include_sp500=sp500,
        lookback_years=lookback,
        rf=rf,
        add_weight=add_w,
    )


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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _explore_mix(stock_pct, stock_ticker, bond_ticker, account_ccy="EUR"):
    return explore.analyze_mix(stock_pct, stock_ticker, bond_ticker,
                               account_ccy=account_ccy)


@st.cache_data(show_spinner=False)
def _explore_projection(monthly_values, start_value, monthly_contribution, years, method):
    s = pd.Series(monthly_values)
    return explore.project_contributions(
        s, start_value, monthly_contribution, years, method=method
    )


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _outlooks(tickers, rf, horizon_years, method):
    from pa import data

    # Always use a long window here - scenario breadth needs history, and this
    # is independent of the optimizer's "years of history" setting.
    px = data.fetch_prices(list(tickers) + ["SPY"], lookback_years=12)
    if px.empty or "SPY" not in px.columns:
        return []
    mkt = px["SPY"]
    out = []
    for t in tickers:
        if t in px.columns:
            try:
                out.append(forecast.asset_outlook(
                    t, px[t], mkt, rf=rf, horizon_years=horizon_years, method=method
                ))
            except Exception:
                pass
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _timing_prices(ticker):
    from pa import data

    px = data.fetch_prices([ticker], lookback_years=15)
    return px[ticker].dropna() if ticker in px.columns else pd.Series(dtype=float)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _idea_context(ticker, horizon_years, rf):
    from pa import data

    px = data.fetch_prices([ticker, "SPY"], lookback_years=12)
    if ticker not in px.columns or "SPY" not in px.columns:
        return None
    f = data.fetch_fundamentals([ticker])
    frow = f.loc[ticker].to_dict() if ticker in f.index else {}
    try:
        return ideacard.build_context(
            ticker, px[ticker], px["SPY"], frow, rf=rf, horizon_years=float(horizon_years)
        )
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Session defaults + helpers
# --------------------------------------------------------------------------- #
_DEFAULTS = dict(
    amount_mode="shares",
    watchlist_text="COST, JNJ, XOM, BRK-B",
    use_curated=True,
    use_sp500=False,
    lookback=5.0,
    rf_rate=0.04,
    add_weight=0.10,
    editor_nonce=0,
    profile_nonce=0,
    port_nonce=0,
    loaded_portfolio=UNSAVED,
    profile=None,
)
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _set_loaded(name: str) -> None:
    """Point the 'Saved portfolios' picker at ``name`` (survives Streamlit's
    sticky widget state by rotating the selectbox key)."""
    st.session_state.loaded_portfolio = name
    st.session_state.port_nonce += 1


def _starter_df() -> pd.DataFrame:
    p = portfolio.parse_holdings_text(portfolio.SAMPLE)
    return p.rename_axis("Ticker").reset_index().rename(columns={"amount": "Amount"})


def _reset_editor(df: pd.DataFrame) -> None:
    df = df.copy()
    ren = {}
    for c in df.columns:
        cl = str(c).lower()
        if cl in ("ticker", "symbol", "stock"):
            ren[c] = "Ticker"
        elif cl in ("amount", "shares", "quantity", "qty", "weight", "value"):
            ren[c] = "Amount"
    df = df.rename(columns=ren)
    if "Ticker" not in df.columns:
        df["Ticker"] = ""
    if "Amount" not in df.columns:
        df["Amount"] = 0.0
    st.session_state.editor_df = df[["Ticker", "Amount"]].reset_index(drop=True)
    st.session_state.editor_nonce += 1


if "editor_df" not in st.session_state:
    st.session_state.editor_df = _starter_df()

_STOCK_SLEEVES = {
    "VTI — US total stock market": "VTI",
    "VT — whole world": "VT",
    "VEA — developed markets ex-US": "VEA",
    "VWO — emerging markets": "VWO",
}
_BOND_SLEEVES = {
    "BND — US total bond market": "BND",
    "IEF — 7-10 year treasuries": "IEF",
    "SHY — 1-3 year treasuries (very safe)": "SHY",
    "TLT — 20+ year treasuries (volatile)": "TLT",
    "TIP — inflation-linked bonds": "TIP",
}
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


# Module-level singletons; their ``.ccy`` is rebound in the sidebar to the
# account currency, which propagates to every render function.
EUR = _Fmt("EUR", 0)
EUR2 = _Fmt("EUR", 2)
ACCOUNT_CCY = "EUR"


def render_explore() -> None:
    """The beginner 'explore from cash' page: a risk slider + contribution sim."""
    sb = st.sidebar
    st.session_state.setdefault("explore_stock_pct", 60)

    sb.subheader("Your mix")
    sb.caption("Or jump to a starting point:")
    _pcols = sb.columns(3)
    for _c, (_short, _val) in zip(_pcols, [("Calm", 40), ("Mixed", 60), ("Bold", 85)]):
        if _c.button(_short, use_container_width=True,
                     help=f"Set the mix to {_val}% stocks"):
            st.session_state.explore_stock_pct = _val
            st.rerun()

    stock_pct = sb.slider(
        "% in stocks  (the rest goes to bonds)", 0, 100, step=5, key="explore_stock_pct",
        help="Left = safer and calmer, lower growth. Right = more growth, much bigger drops.",
    )
    _near = min(explore.PRESETS.items(), key=lambda kv: abs(kv[1] - stock_pct))[0]
    sb.caption(f"≈ **{_near}**")

    with sb.expander("Change the two funds"):
        stock_label = st.selectbox("Stock sleeve", list(_STOCK_SLEEVES), index=0)
        bond_label = st.selectbox("Bond sleeve", list(_BOND_SLEEVES), index=0)
    stock_ticker, bond_ticker = _STOCK_SLEEVES[stock_label], _BOND_SLEEVES[bond_label]

    sb.subheader("A monthly savings plan")
    start_value = sb.number_input(f"Starting amount ({ACCOUNT_CCY})", 0, 5_000_000, 1_000, 250)
    monthly_contribution = sb.number_input(f"Added every month ({ACCOUNT_CCY})", 0, 200_000,
                                           150, 25)
    years = sb.slider("For how many years", 1, 40, 15)
    _sim_choice = sb.radio(
        "How to imagine the future",
        ["Resample real history", "Neural generator (experimental)"],
        key="sim_method",
        help="Resampling shuffles real 6-month chunks of the past. The neural "
             "generator trains a small mixture-density network and rolls it "
             "forward — interesting to compare, but with ~15 years of data the "
             "resample method is more trustworthy.",
    )
    sim_method = "neural" if _sim_choice.startswith("Neural") else "bootstrap"

    st.title("🌱 Explore: how a simple mix behaves")
    st.markdown(
        "See how a simple **stocks + bonds** portfolio would have behaved over the "
        "last ~18 years. **Move the *% in stocks* slider on the left** (or use the "
        "buttons) and watch every number below react. Nothing here is real money "
        "or a recommendation — it's a sandbox for building intuition."
    )

    try:
        m = _explore_mix(stock_pct, stock_ticker, bond_ticker, ACCOUNT_CCY)
    except Exception as e:
        st.error(f"Couldn't build that mix ({e}). Try different funds in the sidebar.")
        return

    c = st.columns(4)
    c[0].metric("Historical return / yr", PCT.format(m.ann_return), help=H["Annualized return"])
    c[1].metric("Volatility / yr", PCT.format(m.ann_vol), help=H["Annualized volatility"])
    c[2].metric("Worst drop (peak→trough)", PCT.format(m.max_drawdown), help=H["Max drawdown"])
    c[3].metric("Worst 12-month stretch", PCT.format(m.worst_12m),
                help="The lowest return over any rolling one-year window in this history.")

    st.info(education.interpret_mix(m))

    left, right = st.columns(2)
    with left:
        st.subheader("The trade-off")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=m.frontier["vol"], y=m.frontier["ret"], mode="lines",
                                 line=dict(color="#888"), name="every stock/bond split"))
        _cur = m.frontier.loc[m.frontier["stock_pct"] == stock_pct]
        if not _cur.empty:
            fig.add_trace(go.Scatter(x=_cur["vol"], y=_cur["ret"], mode="markers+text",
                                     text=[f"  {stock_pct}% stocks"], textposition="middle right",
                                     marker=dict(size=14, color="#e4572e"), name="your mix"))
        fig.update_layout(height=340, xaxis_title="volatility (bumpiness)",
                          yaxis_title="historical return / yr",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                          showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Slide the stocks % in the sidebar and watch the dot move. More "
                   "stocks buys more return — and more bumpiness.")

    with right:
        st.subheader(f"Growth of {EUR.m(10_000)}")
        g = m.growth.copy()
        g.index.name = "date"
        fig = px.line(g)
        fig.update_layout(height=340, yaxis_title="", xaxis_title="", legend_title="",
                          margin=dict(t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{m.start:%b %Y} – {m.end:%b %Y}, in **{ACCOUNT_CCY}** "
                   "(funds converted at historical rates). The 'Cash / savings' line "
                   "assumes a steady 3%/yr — notice how inflation-era savings barely move.")

    st.divider()
    st.subheader(f"If you add {EUR.m(monthly_contribution)} every month for {years} years")

    with st.spinner("Simulating…" if sim_method == "bootstrap" else "Training the generator…"):
        proj = _explore_projection(
            tuple(m.monthly_returns.values), start_value, monthly_contribution, years, sim_method
        )
    yrs = proj.months / 12.0
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=yrs, y=proj.percentiles["p90"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yrs, y=proj.percentiles["p10"], fill="tonexty",
                             fillcolor="rgba(46,134,171,0.15)", line=dict(width=0),
                             name="10th–90th percentile"))
    fig.add_trace(go.Scatter(x=yrs, y=proj.percentiles["p75"], line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=yrs, y=proj.percentiles["p25"], fill="tonexty",
                             fillcolor="rgba(46,134,171,0.30)", line=dict(width=0),
                             name="25th–75th percentile"))
    fig.add_trace(go.Scatter(x=yrs, y=proj.percentiles["p50"], line=dict(color="#2e86ab", width=2.5),
                             name="typical (median)"))
    fig.add_trace(go.Scatter(x=yrs, y=proj.contributed, line=dict(color="#999", dash="dash"),
                             name="money you put in"))
    fig.update_layout(height=380, xaxis_title="years from now",
                      yaxis_title=f"portfolio value ({ACCOUNT_CCY})",
                      margin=dict(t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    d = st.columns(4)
    d[0].metric("You'd put in", EUR.format(proj.total_contributed))
    d[1].metric("Typical outcome", EUR.format(proj.end_median))
    d[2].metric("Unlucky (10th pct)", EUR.format(proj.end_low))
    d[3].metric("Lucky (90th pct)", EUR.format(proj.end_high))

    if proj.method == "neural" and proj.faithfulness and proj.faithfulness.get("ok"):
        f = proj.faithfulness
        st.caption(
            f"🧠 **Neural generator** — a 2-component mixture-density network "
            f"(~100 weights) trained on {f['n_train']} months, then rolled forward "
            f"1,000 times. Generated months swing **{f['gen_std']:.2%}** vs "
            f"**{f['hist_std']:.2%}** in history, with a worst month of "
            f"**{f['gen_min']:.1%}** vs **{f['hist_min']:.1%}** (the average is "
            "anchored to history on purpose). It picks up volatility clustering "
            "and a fatter crash tail — which is why its outlook is often a bit "
            "gloomier — but on ~15 years of data it's a demo. Trust the resample."
        )
    elif proj.method == "bootstrap" and sim_method == "neural":
        st.warning("Not enough history to train the generator here — showing the "
                   "resample method instead.")
    else:
        st.caption(
            "Each run resamples real 6-month chunks of this mix's history, 1,000 "
            "times. The spread is the point: the future is a **range**, not a "
            "number. Longer horizons and steady monthly adding narrow the bad cases."
        )

    _gap = proj.fee_low_median - proj.fee_high_median
    st.warning(
        f"**Fees:** the same plan in a fund charging **{proj.fee_high_pct:.2%}/yr** "
        f"instead of **{proj.fee_low_pct:.2%}/yr** ends near {EUR.m(proj.fee_high_median)} "
        f"vs {EUR.m(proj.fee_low_median)} — about **{EUR.m(_gap)} lost to fees** "
        f"over {years} years, and the gap compounds the longer you invest. Broad "
        "index ETFs are typically at the low end."
    )

    st.divider()
    st.subheader("Things to try")
    st.markdown(
        """
        - Drag the slider to **100% stocks** — see the return rise and the *worst
          drop* get much deeper. Then to **20%** — calmer, but the growth chart
          flattens.
        - Set the plan to **40 years** instead of 15 — notice how the unlucky
          case improves relative to what you put in. Time is the biggest lever.
        - Switch *How to imagine the future* to the **neural generator** and back.
          They should roughly agree — where they don't, that gap is model
          uncertainty, and neither one is "the answer".
        - Read the **📖 Learn** ideas below, then come back to this page.
        """
    )
    with st.expander("📖 New to this? Read the primer"):
        st.markdown(education.GETTING_STARTED)

    st.caption("Not advice. Historical data only — the future won't match the past.")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _research_bundle(ticker, benchmark, lookback):
    from pa import data, forecast, research, timing

    px = data.fetch_prices([ticker, benchmark], lookback)
    if ticker not in px.columns:
        return None
    prices = px[ticker].dropna()
    prices.name = ticker
    bench = px[benchmark].dropna() if benchmark in px.columns else None
    profile = data.fetch_profile(ticker)
    try:
        estimates = data.fetch_estimates(ticker)
    except Exception:
        estimates = {}
    try:
        docs = data.fetch_company_docs(ticker)
    except Exception:
        docs = {}
    rel = research.vs_market(prices, bench, benchmark) if bench is not None else None
    tc = timing.trend_context(ticker, prices)
    try:
        outlook = forecast.asset_outlook(
            ticker, prices, bench if bench is not None else prices,
            horizon_years=1.0, method="bootstrap",
        )
    except Exception:
        outlook = None
    return {
        "prices": prices, "bench": bench, "profile": profile, "rel": rel,
        "trend": tc, "outlook": outlook, "last": float(prices.iloc[-1]),
        "estimates": estimates, "docs": docs,
    }


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _universe_prices(tickers, lookback):
    from pa import data

    return data.fetch_prices(list(tickers), lookback)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def _universe_fundamentals(tickers):
    from pa import data

    return data.fetch_fundamentals(list(tickers))


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _symbol_search(query):
    from pa import data

    return data.search_symbols(query, limit=10)


def render_research() -> None:
    """Mode: search / browse stocks & bonds, then focus one for business detail."""
    from pa import data as data_mod
    from pa import research, universe

    sb = st.sidebar
    sb.subheader("🔎 Find any company")
    query = sb.text_input("Search by name or ticker  (press Enter)",
                          placeholder="e.g. ASML, Apple, Novo Nordisk").strip()
    searched = None
    if query:
        matches = _symbol_search(query)
        if matches:
            opts = {
                f"{m['symbol']} · {m['name'][:32]}"
                + (f" — {m['exchange']}" if m['exchange'] else ""): m['symbol']
                for m in matches
            }
            choice = sb.radio(f"{len(opts)} matches — pick one", list(opts), key="search_match")
            searched = opts.get(choice)
        else:
            sb.warning(
                "No matches. Press Enter after typing, try fewer words, or the "
                "exact ticker. **Privately held companies** (e.g. Gymshark, IKEA, "
                "Bosch) aren't listed on any exchange, so they won't appear — "
                "there's nothing to buy or price."
            )

    sb.divider()
    sb.subheader("Browse a list")
    src = sb.selectbox(
        "List", ["Stocks & bonds (curated ETFs)", "Popular stocks", "S&P 500", "My own list"],
        help="Scan the table, click a row to focus that name below.",
    )
    my_list_txt = ""
    if src == "My own list":
        my_list_txt = sb.text_area("Tickers (comma / space separated)",
                                   value="AAPL, MSFT, JNJ, XOM, TLT, GLD", height=90)

    bench_label = sb.selectbox(
        "Compare against", ["S&P 500 (SPY)", "US bonds (BND)", "World stocks (VT)",
                            "Nasdaq-100 (QQQ)"], index=0)
    benchmark = {"S&P 500 (SPY)": "SPY", "US bonds (BND)": "BND",
                 "World stocks (VT)": "VT", "Nasdaq-100 (QQQ)": "QQQ"}[bench_label]
    lookback = sb.slider("Years of history", 1.0, 15.0, 8.0, 0.5)
    jump = sb.text_input("…or jump straight to a ticker",
                         placeholder="e.g. NVDA").upper().strip()
    extra = sb.text_input("On the focused name, also compare",
                          placeholder="e.g. AAPL, GOOGL")

    st.title("🔎 Research a stock or bond")
    st.caption("Scan a list and click a row, **or type a company name in "
               "*🔎 Find any company* in the sidebar** (works for anything listed — "
               "ASML, Shell, Toyota…). Data from Yahoo Finance — not advice, "
               "backward-looking.")

    # ---- resolve the universe ----
    if src == "Stocks & bonds (curated ETFs)":
        uni = list(universe.CURATED_ETFS)
    elif src == "Popular stocks":
        uni = list(universe.POPULAR_STOCKS)
    elif src == "S&P 500":
        try:
            uni = universe.sp500_tickers()
        except Exception:
            uni = list(universe.POPULAR_STOCKS)
            st.warning("Couldn't fetch the S&P 500 list — showing popular stocks instead.")
    else:
        uni = [t.strip().upper() for t in my_list_txt.replace(",", " ").split() if t.strip()]

    uni = [t for t in dict.fromkeys(uni) if t]
    focus = None

    if uni:
        with st.spinner("Loading the list…"):
            pxu = _universe_prices(tuple(sorted(set(uni + [benchmark]))), lookback)
            fund = _universe_fundamentals(tuple(sorted(uni))) if len(uni) <= 70 else None
            btbl = research.browse_table(
                uni, pxu, pxu[benchmark] if benchmark in pxu.columns else None, fund
            )
        if btbl.empty:
            st.info("No price data for that list.")
        else:
            st.markdown(f"**{len(btbl)} names** — click a row to focus it. "
                        "Sort by clicking a column header.")
            if fund is None:
                st.caption("Big list: P/E, yield and sector load only when you focus a name.")
            disp = btbl.copy()
            for c in ("1y return", "vs bench (1y)", "Volatility", "Max drawdown", "From 12mo high"):
                disp[c] = disp[c].map(lambda v: f"{v:+.0%}" if v == v else "—")
            disp["Yield"] = disp["Yield"].map(lambda v: f"{v*100:.1f}%" if v == v and v > 0 else "—")
            disp["P/E"] = disp["P/E"].map(lambda v: f"{v:.0f}" if v == v and v > 0 else "—")
            disp["Beta"] = disp["Beta"].map(lambda v: f"{v:.2f}" if v == v else "—")
            ev = st.dataframe(
                disp, use_container_width=True, height=380,
                on_select="rerun", selection_mode="single-row", key="browse_sel",
            )
            picked = ev.selection.rows if ev and ev.selection else []
            if picked:
                focus = btbl.index[picked[0]]

    focus = (jump or searched or focus or (uni[0] if uni else "MSFT")).upper()

    st.divider()
    if searched and not jump:
        st.caption(f"Showing your search pick — **{focus}**. Pick a different match "
                   "in the sidebar, or clear the search box to go back to the list.")
    b = _research_bundle(focus, benchmark, lookback)
    if b is None:
        st.error(f"No price data for **{focus}** — check the symbol.")
        return

    p, rel, tc, o = b["profile"], b["rel"], b["trend"], b["outlook"]
    ticker = focus
    name = p.get("long_name") or ticker
    st.header(f"📄 {ticker} — {name}")

    meta = " · ".join(x for x in [
        p.get("sector") if p.get("sector") not in (None, "Unknown") else None,
        p.get("industry") if p.get("industry") not in (None, "Unknown") else None,
        p.get("country") or None,
        (f"{int(p['employees']):,} employees" if p.get("employees") else None),
    ] if x)
    if meta:
        st.caption(meta)

    # ---- Business ----
    if p.get("summary"):
        st.subheader("The business")
        st.write(p["summary"])
        if p.get("website"):
            st.caption(p["website"])

    # ---- Key stats ----
    st.subheader("Key stats")
    _sec_sym = fx.symbol(p.get("currency") or "USD")
    stats = research.key_stats(p, last_price=b["last"], sym=_sec_sym)
    if p.get("currency") and fx.normalize(p["currency"])[0] not in ("USD", ""):
        st.caption(f"Figures below are in **{fx.normalize(p['currency'])[0]}**, "
                   "the security's own currency — not your account currency.")
    extra_tickers = [t.strip().upper() for t in extra.replace(",", " ").split() if t.strip()]
    extra_tickers = [t for t in extra_tickers if t != ticker][:2]
    if extra_tickers:
        cols = {ticker: dict(stats)}
        for et in extra_tickers:
            eb = _research_bundle(et, benchmark, lookback)
            if eb:
                cols[et] = dict(research.key_stats(
                    eb["profile"], last_price=eb["last"],
                    sym=fx.symbol(eb["profile"].get("currency") or "USD")))
        labels = list(dict.fromkeys(k for c in cols.values() for k in c))
        table = pd.DataFrame(
            {tk: [c.get(lbl, "—") for lbl in labels] for tk, c in cols.items()},
            index=labels,
        )
        st.dataframe(table, use_container_width=True)
    elif stats:
        half = (len(stats) + 1) // 2
        cc = st.columns(2)
        for col, chunk in zip(cc, (stats[:half], stats[half:])):
            col.dataframe(pd.DataFrame(chunk, columns=["", " "]).set_index(""),
                          use_container_width=True)
    else:
        st.caption("Yahoo returned no fundamentals for this ticker (common for bond "
                   "ETFs and non-US listings).")

    # ---- Insights ----
    tips = research.business_insights(p, rel)
    if tips:
        st.subheader("In plain language")
        for t in tips:
            st.markdown(f"- {t}")

    # ---- What's expected next ----
    est = b.get("estimates") or {}
    est_rows, est_tips = research.estimates_summary(est, sym=_sec_sym)
    if est.get("next_earnings") or est_rows:
        st.subheader("What's expected next")
        if est.get("next_earnings"):
            try:
                _d = pd.to_datetime(est["next_earnings"]).strftime("%d %b %Y")
            except Exception:
                _d = est["next_earnings"]
            st.markdown(f"**Next earnings report:** {_d}")
        if est_rows:
            st.dataframe(
                pd.DataFrame(est_rows, columns=["", "Next quarter", "Next fiscal year"]).set_index(""),
                use_container_width=True,
            )
        for t in est_tips:
            st.markdown(f"- {t}")
        st.caption("These are **analyst consensus estimates**, not the company's own "
                   "official guidance — they shift constantly and are often off. "
                   "'Next fiscal year' may not line up with the calendar year.")
    elif p.get("quote_type", "").upper() in ("ETF", "MUTUALFUND"):
        st.subheader("What's expected next")
        st.caption("This is a fund — there are no company earnings estimates. What "
                   "matters for a fund is the direction of its whole market and, for "
                   "bond funds, interest rates.")

    # ---- The company's own plans (pointers to primary sources) ----
    docs = b.get("docs") or {}
    reports = data_mod.latest_report_links(docs) if docs.get("filings") else {}
    if docs.get("ir_website") or reports or docs.get("news"):
        st.subheader("The company's own plans — straight from the source")
        st.caption("The dashboard can't summarise management's strategy or guidance "
                   "— that lives in the earnings call and the filings. Here's where "
                   "to read it in the company's own words:")
        lb = st.columns(3)
        if docs.get("ir_website"):
            lb[0].link_button("📊 Investor relations site", docs["ir_website"],
                              use_container_width=True,
                              help="Guidance, strategy decks, earnings-call replays")
        if reports.get("annual"):
            lb[1].link_button(
                f"📄 Annual report · {reports['annual']['date'][:7]}",
                reports["annual"]["url"], use_container_width=True,
                help="10-K / 20-F — the 'Business' and MD&A sections lay out the outlook")
        if reports.get("quarterly"):
            lb[2].link_button(
                f"📄 Latest {reports['quarterly']['type']} · {reports['quarterly']['date'][:7]}",
                reports["quarterly"]["url"], use_container_width=True,
                help="The most recent quarterly filing")
        if est.get("next_earnings"):
            try:
                _ed = pd.to_datetime(est["next_earnings"]).strftime("%d %b %Y")
            except Exception:
                _ed = est["next_earnings"]
            st.markdown(f"🗓️ **Next earnings call:** {_ed} — that's when fresh guidance drops.")

        if docs.get("news"):
            st.markdown("**Recent headlines** (guidance / product / strategy news usually breaks here first)")
            for n in docs["news"][:5]:
                bits = " · ".join(x for x in [n.get("publisher"), n.get("date")] if x)
                if n.get("url"):
                    st.markdown(f"- [{n['title']}]({n['url']}) — {bits}")
                else:
                    st.markdown(f"- {n['title']} — {bits}")
            st.caption("Headlines via Yahoo Finance — a rough feed, relevance varies.")

    # ---- Against the market ----
    if rel is not None:
        st.subheader(f"Against {benchmark}")
        m = st.columns(4)
        m[0].metric(f"{ticker} total return", f"{rel.asset_return:+.0%}")
        m[1].metric(f"{benchmark} total return", f"{rel.bench_return:+.0%}")
        m[2].metric("Beta", f"{rel.beta:.2f}", help=H["Beta to market"])
        m[3].metric("Correlation", f"{rel.corr:.2f}",
                    help="1.0 = moves in lockstep with the benchmark; 0 = unrelated.")
        m = st.columns(4)
        m[0].metric("Excess return", f"{rel.excess:+.0%}",
                    help="How much more (or less) than the benchmark, over this window.")
        m[1].metric("Up-market capture", f"{rel.up_capture:.0%}" if rel.up_capture == rel.up_capture else "—",
                    help="In months the benchmark rose, how much of that rise this captured.")
        m[2].metric("Down-market capture", f"{rel.down_capture:.0%}" if rel.down_capture == rel.down_capture else "—",
                    help="In months the benchmark fell, how much of that fall this took. Lower = more defensive.")

        fig = px.line(rel.rebased)
        fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title="growth of 100",
                          xaxis_title="", legend_title="",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Relative strength** (rising = beating the benchmark)")
        fig = px.area(rel.rel_strength)
        fig.update_traces(line_color="#2e86ab", fillcolor="rgba(46,134,171,0.15)")
        fig.add_hline(y=1.0, line=dict(color="#888", dash="dot"))
        fig.update_layout(height=200, margin=dict(t=6, b=6), showlegend=False,
                          yaxis_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    # ---- Trend + outlook (reused, honest framing) ----
    st.subheader("Where it stands & a range of what-ifs")
    st.info(tc.status)
    if o is not None:
        oc = st.columns(4)
        oc[0].metric("Volatility / yr", f"{o.ann_vol:.0%}", help=H["Annualized volatility"])
        oc[1].metric("Worst 12 months", f"{o.hist_worst_12m:+.0%}")
        oc[2].metric("Return estimates (hist/CAPM/EMA)",
                     f"{o.est_hist_mean:+.0%}/{o.est_capm:+.0%}/{o.est_ema:+.0%}")
        oc[3].metric("1-year scenario range", f"{o.p10:+.0%} … {o.p90:+.0%}")
        st.caption(o.note + "  This is **not a forecast** — individual securities "
                   "can't be reliably predicted. For a full idea write-up use the "
                   "💡 Idea worksheet in Analyze mode.")

    st.divider()
    st.caption("Not investment advice. Fundamentals via Yahoo Finance can be stale or "
               "wrong; all return/risk figures are historical.")


def _intraday_chart(ticker: str) -> None:
    """A compact 'how it's been moving' line — hourly bars when Yahoo has
    them for this ticker, daily bars over a longer window otherwise."""
    hist, gran = _intraday(ticker)
    if hist.empty or len(hist) < 2:
        return
    unit = "hour" if gran == "hourly" else "day"
    chg = (float(hist.iloc[-1]) / float(hist.iloc[0]) - 1) * 100
    up = chg >= 0
    st.caption(f"📈 {ticker} — last {len(hist)} {unit}s "
               f"({'+' if up else ''}{chg:.1f}%)"
               + ("" if gran == "hourly" else " — hourly data unavailable for this one"))
    fig = px.line(hist)
    fig.update_traces(line=dict(color="#2e86ab" if up else "#e4572e", width=2))
    fig.update_layout(height=130, margin=dict(t=0, b=0, l=0, r=0), showlegend=False,
                      xaxis_title="", yaxis_title="")
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _trade_context_prices(ticker, lookback_years):
    from pa import data

    return data.fetch_prices([ticker], lookback_years=lookback_years)


def _trade_history_chart(ticker: str, trade_date, trade_price: float,
                         side: str, ccy: str) -> None:
    """Daily price history around one specific past trade, with a marker at
    the date/price it executed — 'was this a good entry (or exit)?' at a
    glance. Uses the security's own native currency, matching the price
    history itself, so there's no FX distortion to second-guess."""
    trade_date = pd.Timestamp(trade_date)
    pad = pd.Timedelta(days=45)
    days_needed = max((pd.Timestamp.now() - trade_date).days, 0) + 45 + 5
    lookback_years = max(0.3, days_needed / 365.25)

    hist = _trade_context_prices(ticker, round(lookback_years, 2))
    if hist.empty or ticker not in hist.columns:
        st.caption(f"No historical price data available for {ticker}.")
        return
    s = hist[ticker]
    window = s[(s.index >= trade_date - pad) & (s.index <= trade_date + pad)]
    if window.empty:
        window = s.tail(90)

    fig = px.line(window)
    fig.update_traces(line=dict(color="#9aa0a6", width=1.5), showlegend=False)
    fig.add_scatter(
        x=[trade_date], y=[trade_price], mode="markers", showlegend=False,
        marker=dict(size=13, color="#2e86ab" if side == "buy" else "#e4572e",
                   symbol="triangle-up" if side == "buy" else "triangle-down"),
    )
    fig.update_layout(height=260, margin=dict(t=10, b=10), xaxis_title="",
                      yaxis_title=f"price ({ccy})")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{side.title()} marker at {trade_date:%d %b %Y}, "
               f"{trade_price:,.2f} {ccy} — ±45 days of {ticker}'s own price history.")


def _paper_fee_model() -> paper.FeeModel:
    """Render the 'Trading costs' expander and return the chosen fee model."""
    names = list(paper.PRESETS) + ["Custom…"]
    default = "Low-cost broker (0.25% FX only)"
    with st.expander("⚙️ Trading costs", expanded=False):
        choice = st.selectbox(
            "Cost model", names, index=names.index(default), key="paper_fee_choice",
            help="Applied to every buy and sell. The FX fee is charged on any "
                 f"holding not quoted in {ACCOUNT_CCY}, on top of the currency "
                 "conversion already reflected in the price.",
        )
        if choice == "Custom…":
            fc = st.columns(3)
            m = paper.FeeModel(
                flat=fc[0].number_input(f"Flat {ACCOUNT_CCY}/trade", 0.0, 100.0, 0.0, 0.5,
                                        key="paper_fee_flat"),
                rate_bps=fc[1].number_input("Commission (bps)", 0.0, 200.0, 0.0, 1.0,
                                            key="paper_fee_bps",
                                            help="1 bp = 0.01% of the trade value"),
                fx_bps=fc[2].number_input("FX fee on foreign (bps)", 0.0, 200.0, 25.0,
                                          1.0, key="paper_fee_fx"),
            )
        else:
            m = paper.PRESETS[choice]
        if m.active:
            bits = []
            if m.flat:
                bits.append(f"{m.flat:g} {ACCOUNT_CCY} per trade")
            if m.rate_bps:
                bits.append(f"{m.rate_bps:g} bps commission")
            if m.fx_bps:
                bits.append(f"{m.fx_bps:g} bps FX fee on non-{ACCOUNT_CCY} names")
            st.caption("Charging " + ", ".join(bits)
                       + ". Fees are folded into cost basis on buys and taken off "
                       "the proceeds on sells.")
        else:
            st.caption("No fees — trades execute at the quoted price.")
    return m


def _convert_trades(trades: pd.DataFrame, account: str) -> pd.DataFrame:
    """Return the trade log with ``price`` and ``fee`` converted from each
    trade's native currency (``ccy``) into ``account`` at that trade date's
    historical exchange rate."""
    if trades is None or trades.empty:
        return trades
    out = trades.copy()
    days = pd.to_datetime(out["ts"]).dt.normalize()
    mult = pd.Series(1.0, index=out.index)
    natv = out["ccy"].fillna("").replace("", account)
    for nat in sorted(set(natv)):
        code, _ = fx.normalize(nat)
        if not code or code == account.upper():
            continue
        uniq = pd.DatetimeIndex(sorted(days.unique()))
        s = fx.convert(nat, account, index=uniq)
        lookup = dict(zip(uniq, list(s.values)))
        sel = natv == nat
        mult.loc[sel] = days.loc[sel].map(lookup).astype(float).values
    out["price"] = out["price"] * mult.values
    out["fee"] = out["fee"] * mult.values
    return out


def _render_paper_overview(state: paper.PaperState, account: str) -> None:
    """Where the practice money sits — by business market (sector) and by
    country — with the tickers that make up each slice, plus FX exposure."""
    invested = state.invested or 1.0
    rows = [{"ticker": p.ticker,
             "name": p.name or p.ticker,
             "value": p.market_value,
             "sector": p.sector or "Unknown",
             "country": p.country or "Unknown",
             "currency": (fx.normalize(p.currency)[0] or "—")} for p in state.positions]
    df = pd.DataFrame(rows)

    def _slice(col: str) -> pd.DataFrame:
        g = df.groupby(col).agg(
            pct=("value", lambda s: s.sum() / invested * 100),
            tickers=("ticker", lambda s: ", ".join(sorted(s))),
        ).sort_values("pct", ascending=False)
        return g

    by_sector = _slice("sector")
    by_country = _slice("country")
    by_ccy = (df.groupby("currency")["value"].sum().sort_values(ascending=False)
              / invested * 100)
    foreign_pct = float(by_ccy[[c for c in by_ccy.index
                                if c not in (account.upper(), "—")]].sum())

    st.subheader("Portfolio overview")

    bar = px.bar(
        by_sector.reset_index(), x="pct", y="sector", orientation="h",
        text=[f"{v:.0f}%" for v in by_sector["pct"]],
        hover_data={"tickers": True, "pct": ":.1f", "sector": False},
        labels={"pct": "% of holdings", "sector": ""},
    )
    bar.update_traces(marker_color="#4c78a8", textposition="outside", cliponaxis=False)
    bar.update_layout(height=max(180, 40 * len(by_sector) + 70),
                      margin=dict(t=10, b=10, l=10, r=30), xaxis_title="% of holdings")
    st.caption("By business market (sector)")
    st.plotly_chart(bar, use_container_width=True)

    oc = st.columns([3, 2])
    sdf = by_sector.reset_index()
    sdf["% of holdings"] = sdf["pct"].round(1)
    oc[0].caption("Sectors and their holdings")
    oc[0].dataframe(sdf[["sector", "% of holdings", "tickers"]]
                    .rename(columns={"sector": "Market", "tickers": "Tickers"}),
                    hide_index=True, use_container_width=True)

    cdf = pd.DataFrame({"Currency": by_ccy.index, "% of holdings": by_ccy.values.round(1)})
    oc[1].caption("By currency")
    oc[1].dataframe(cdf, hide_index=True, use_container_width=True)
    oc[1].metric("Foreign-currency exposure", f"{foreign_pct:.0f}%",
                 help=f"Share of invested value in securities not quoted in "
                      f"{account}. That slice carries currency risk (already "
                      "reflected in your P&L) and an FX fee on every trade.")

    n_sec = df["sector"].nunique()
    lead = by_sector.index[0] if len(by_sector) else "—"
    ctry = by_country.index[0] if len(by_country) else "—"
    st.caption(
        f"Holdings span **{n_sec}** market{'' if n_sec == 1 else 's'}, most in "
        f"**{lead}** ({by_sector['pct'].iloc[0]:.0f}%); biggest country is "
        f"**{ctry}** ({by_country['pct'].iloc[0]:.0f}%). Sector, country and "
        "currency come from Yahoo Finance and can be rough for funds — an ETF is "
        "tagged where it is *domiciled*, not where it invests."
    )


def _account_snapshot(profile: str) -> dict | None:
    """Cheap performance snapshot of one profile's practice account, for the
    leaderboard. None if that profile has no practice account yet, or has
    never traded — nothing to rank."""
    acc = store.practice_get(profile)
    if acc is None:
        return None
    trades = store.practice_trades(acc["id"])
    if trades.empty:
        return None

    stored_ccy = (acc["currency"] or "EUR").upper()
    held = sorted(set(trades["ticker"]))
    origins = _origins(tuple(held))
    ticker_ccy = {t: (origins.get(t, {}).get("currency") or stored_ccy) for t in held}

    tr = trades.copy()
    tr["ccy"] = [c or ticker_ccy.get(tk, stored_ccy) for c, tk in zip(tr["ccy"], tr["ticker"])]
    tr = _convert_trades(tr, stored_ccy)
    latest = {t: v * _fx_spot(ticker_ccy.get(t, stored_ccy), stored_ccy)
             for t, v in _latest_prices(tuple(held)).items()}
    state = paper.compute_state(float(acc["starting_cash"]), tr, latest)

    return {
        "profile": profile,
        "currency": stored_ccy,
        "total_value": state.total_value,
        "pnl_pct": state.total_pnl_pct,
        "n_trades": state.n_trades,
        "started": trades["ts"].min(),
        "locked": store.profile_has_password(profile),
    }


def _render_leaderboard(current_profile: str, acct_ccy: str) -> None:
    """Every profile's practice account, ranked by return on its own starting
    cash — visible to everyone, since viewing was never behind a password.

    Always renders (even with zero or one active trader) so it's a permanent,
    predictable fixture rather than something that appears once a threshold
    is met and otherwise looks like it's missing.
    """
    st.subheader("🏆 Leaderboard")
    rows = []
    for p in store.list_profiles():
        try:
            snap = _account_snapshot(p)
        except Exception:
            continue  # one account's bad data shouldn't blank the board for everyone
        if snap:
            rows.append(snap)

    if not rows:
        st.caption("No one's placed a practice trade yet — be the first, and this "
                   "fills in.")
        return

    for r in rows:
        r["value_here"] = r["total_value"] * _fx_spot(r["currency"], acct_ccy)
    rows.sort(key=lambda r: r["pnl_pct"], reverse=True)

    st.caption(
        ("Every practice account, ranked by return since its own starting "
         "cash — currency-neutral, so it's fair across accounts in different "
         "currencies. Only its own password can change an account; anyone "
         "can see where it stands.")
        if len(rows) > 1 else
        "You're the only one who's traded so far — share the link and this "
        "turns into a real leaderboard."
    )

    lb = pd.DataFrame([{
        "Rank": i + 1,
        "Profile": ("👉 " if r["profile"] == current_profile else "") + r["profile"]
                   + (" 🔒" if r["locked"] else ""),
        "Return": f"{r['pnl_pct']:+.1%}",
        f"Value ({acct_ccy})": EUR.format(r["value_here"]),
        "Started": pd.to_datetime(r["started"]).strftime("%d %b %Y"),
        "Trades": r["n_trades"],
    } for i, r in enumerate(rows)])
    st.dataframe(lb, hide_index=True, use_container_width=True)

    if len(rows) > 1:
        bar_df = pd.DataFrame({
            "profile": [r["profile"] for r in rows],
            "pnl_pct": [r["pnl_pct"] * 100 for r in rows],
            "who": ["You" if r["profile"] == current_profile else "Other" for r in rows],
        }).sort_values("pnl_pct")
        fig = px.bar(bar_df, x="pnl_pct", y="profile", orientation="h", color="who",
                    color_discrete_map={"You": "#e4572e", "Other": "#4c78a8"},
                    labels={"pnl_pct": "return %", "profile": ""})
        fig.update_layout(height=max(160, 34 * len(rows) + 60), showlegend=False,
                          margin=dict(t=10, b=10, l=10, r=30))
        st.plotly_chart(fig, use_container_width=True)


def _paper_try_unlock(profile: str, key: str, pw_key: str) -> None:
    """Check whatever's currently sitting in the password field and update
    the unlock state. Used both as the field's ``on_change`` (so Enter
    submits it) and called directly from the Unlock button.

    Deliberately *not* an ``st.form`` — forms in the sidebar don't reliably
    submit on Enter (confirmed: typing the right password and pressing
    Enter did nothing until you clicked the button), which is exactly what
    read as "buggy". A plain widget's own ``on_change`` fires on Enter *or*
    blur, so both work now.
    """
    pw = st.session_state.get(pw_key, "")
    st.session_state[f"{key}::err"] = False
    if pw:
        if store.profile_check_password(profile, pw):
            st.session_state[key] = True
        else:
            st.session_state[f"{key}::err"] = True
    st.session_state[pw_key] = ""  # clear the field either way


def _paper_access(profile: str) -> bool:
    """Sidebar password gate for one practice profile's write actions.

    Viewing a profile never needs a password — everyone can see everyone's
    practice account. This only decides whether trade/undo/reset are allowed
    to run this rerun, returning True when they are.
    """
    sb = st.sidebar
    key = f"unlocked::{profile}"
    has_pw = store.profile_has_password(profile)

    if has_pw and not st.session_state.get(key):
        pw_key = f"pwval::{profile}"
        sb.caption(f"🔒 **{profile}** is password-protected. Anyone can "
                   "view it — you need the password to trade, undo or reset.")
        sb.text_input(
            "Password", type="password", label_visibility="collapsed",
            placeholder="Password to make changes, then press Enter", key=pw_key,
            on_change=_paper_try_unlock, args=(profile, key, pw_key),
        )
        if sb.button("Unlock", use_container_width=True):
            _paper_try_unlock(profile, key, pw_key)
            st.rerun()
        if st.session_state.get(f"{key}::err"):
            sb.error("Wrong password.")
        return bool(st.session_state.get(key))

    if has_pw:
        sb.caption(f"🔓 Unlocked for **{profile}** this session.")
        if sb.button("Lock again", use_container_width=True):
            st.session_state[key] = False
            st.rerun()
        return True

    with sb.expander("🔓 Add a password"):
        st.caption("Anyone can already view this profile. A password stops "
                   "*others* from trading, undoing or resetting it — they can "
                   "still watch.")
        p1 = st.text_input("New password", type="password", key=f"newpw1_{profile}")
        p2 = st.text_input("Confirm", type="password", key=f"newpw2_{profile}")
        if st.button("Set password", key=f"setpw_{profile}"):
            if not p1:
                st.error("Type a password first.")
            elif p1 != p2:
                st.error("Passwords don't match.")
            else:
                store.profile_set_password(profile, p1)
                st.session_state[key] = True
                st.toast("Password set — this profile is now protected.", icon="🔒")
                st.rerun()
    return True


def render_paper() -> None:
    """Mode: paper-trade fake money at real prices, track it vs the market."""
    sb = st.sidebar
    sb.subheader("Practice account")

    _profs = store.list_profiles()
    _opts = _profs + [NEW_PROFILE]
    _cur = st.session_state.profile if st.session_state.profile in _profs else None
    _idx = _profs.index(_cur) if _cur else (0 if _profs else len(_opts) - 1)
    _ch = sb.selectbox("Profile", _opts, index=_idx,
                       help="Everyone can see every profile's practice account. "
                            "Only its own password can change one.",
                       key=f"paper_prof_{st.session_state.profile_nonce}")

    st.title("🎮 Practice portfolio")

    if _ch == NEW_PROFILE:
        _n = sb.text_input("New profile name", key="paper_new_prof",
                           placeholder="anything — e.g. your name")
        _p = sb.text_input("Password (optional, protects it from others)",
                           type="password", key="paper_new_prof_pw")
        if sb.button("Create", use_container_width=True):
            if _n.strip():
                store.get_or_create_profile(_n.strip(), password=_p)
                if _p:
                    st.session_state[f"unlocked::{_n.strip()}"] = True
                st.session_state.profile = _n.strip()
                st.session_state.profile_nonce += 1
                st.rerun()
            else:
                sb.error("Type a name first.")
        st.info("Pick or create a **profile** in the sidebar — your practice "
                "portfolio is saved under it, so you can trade over days and weeks "
                "and watch how your picks do. A password is optional but keeps "
                "others from changing it; anyone can still view it.")
        return

    profile = _ch
    st.session_state.profile = _ch
    can_edit = _paper_access(profile)
    acc = store.practice_get_or_create(profile, currency=ACCOUNT_CCY)
    aid = acc["id"]
    acct_ccy = ACCOUNT_CCY
    stored_ccy = (acc["currency"] or "EUR").upper()
    # starting cash was typed in `stored_ccy`; show it in the active currency
    starting = float(acc["starting_cash"]) * _fx_spot(stored_ccy, acct_ccy)

    trades_raw = store.practice_trades(aid)
    held = sorted(set(trades_raw["ticker"])) if not trades_raw.empty else []
    origins = _origins(tuple(held))
    ticker_ccy = {t: (origins.get(t, {}).get("currency") or acct_ccy) for t in held}

    # fill in the native currency on any trade that predates the ccy column
    trades = trades_raw.copy()
    if not trades.empty:
        trades["ccy"] = [c or ticker_ccy.get(tk, acct_ccy)
                         for c, tk in zip(trades["ccy"], trades["ticker"])]
    trades = _convert_trades(trades, acct_ccy)

    latest_raw = _latest_prices(tuple(held + ["SPY"])) if held else {}
    latest = {t: v * _fx_spot("USD" if t == "SPY" else ticker_ccy.get(t, acct_ccy),
                              acct_ccy)
              for t, v in latest_raw.items()}
    state = paper.compute_state(starting, trades, latest)

    for p in state.positions:
        o = origins.get(p.ticker, {})
        p.country = o.get("country", "")
        p.currency = o.get("currency", "")
        p.sector = o.get("sector", "")
        p.name = o.get("name", p.ticker)

    _mix = (" Converted to your currency at historical rates, so exchange-rate "
            "moves are part of the result." if any(
                fx.normalize(c)[0] not in (acct_ccy, "") for c in ticker_ccy.values())
            else "")
    st.caption(f"Fake money, **real prices**. Profile **{profile}**, started with "
               f"**{EUR.m(starting)}**, all in **{acct_ccy}**.{_mix} "
               "A place to learn, not a broker.")
    if state.missing_prices:
        st.warning("No current price for: " + ", ".join(state.missing_prices)
                   + " — valued at cost for now.")

    fees = _paper_fee_model()

    # ---- headline ----
    ec = pd.DataFrame()
    if not trades.empty:
        hist_raw = _universe_prices(tuple(sorted(set(held + ["SPY"]))), 3.0)
        hist = _fx_convert_history(
            hist_raw, tuple({**ticker_ccy, "SPY": "USD"}.items()), acct_ccy)
        cols = [t for t in held if t in hist.columns]
        ec = paper.equity_curve(
            starting, trades, hist[cols] if cols else hist.iloc[:, :0],
            hist["SPY"] if "SPY" in hist.columns else None,
        )

    c = st.columns(4)
    c[0].metric("Total value", EUR.format(state.total_value),
                f"{state.total_pnl:+,.0f}  ({state.total_pnl_pct:+.1%})")
    c[1].metric("Cash to invest", EUR.format(state.cash))
    c[2].metric("In the market", EUR.format(state.invested))
    if not ec.empty and "All-in S&P 500" in ec.columns:
        spy_val = float(ec["All-in S&P 500"].iloc[-1])
        c[3].metric("If you'd bought only S&P 500", EUR.format(spy_val),
                    f"{spy_val - starting:+,.0f}",
                    help="Same starting cash, all in SPY on your first trade date.")
    else:
        c[3].metric("Realized P&L", EUR.format(state.realized_pnl),
                    help="Profit/loss you've locked in by selling.")

    with st.spinner("Tallying every account…"):
        _render_leaderboard(profile, acct_ccy)

    # ---- trade ticket ----
    st.subheader("Place a trade")
    tc = st.columns([2, 1, 1])
    tk = tc[0].text_input("Ticker", key="paper_tk",
                          placeholder="AAPL, MSFT, BND, VWO, GLD…").upper().strip()
    side = tc[1].radio("Side", ["Buy", "Sell"], horizontal=True, key="paper_side")
    by = tc[2].radio("Amount as", ["shares", acct_ccy], horizontal=True, key="paper_by")

    price_native = None
    if tk:
        price_native = _latest_prices((tk, "SPY")).get(tk)
    if tk and not price_native:
        st.warning(f"No price found for **{tk}** — check the ticker.")
    elif tk:
        origin = _origins((tk,)).get(tk, {})
        cur = origin.get("currency", "")
        nat_code = fx.normalize(cur)[0]
        foreign = bool(nat_code) and nat_code != acct_ccy
        rate = _fx_spot(cur or acct_ccy, acct_ccy)
        price_now = price_native * rate           # in account currency
        qc = st.columns([1, 3])
        if by == "shares":
            qty = qc[0].number_input("Shares", 0.0, 1e7, 1.0, 1.0, key="paper_qty")
            shares = float(qty)
        else:
            amt = qc[0].number_input(f"Amount ({acct_ccy})", 0.0, 1e9, 500.0, 50.0,
                                     key="paper_amt")
            shares = float(amt) / price_now
        value = shares * price_now
        fee = fees.fee(value, foreign)
        cash_out = value + fee if side == "Buy" else value - fee
        _fx_bit = (f" *(≈ {price_native:,.2f} {cur or nat_code})*" if foreign else "")
        _origin_bit = (f" · {_flag(origin.get('country',''))} "
                       f"{origin.get('country','')}" if origin.get("country") else "")
        _fee_bit = (f" · fee **{EUR2.m(fee)}**"
                    + (" *(incl. FX)*" if foreign and fees.fx_bps else "")) if fee else ""
        qc[1].markdown(
            f"&nbsp;\n\n**{tk}** at **{EUR.m(price_now)}**{_fx_bit}{_origin_bit} → "
            f"{side.lower()} **{shares:,.4f}** shares = **{EUR.m(value)}**{_fee_bit}"
            + (f" → **{EUR2.m(cash_out)}** {'out' if side == 'Buy' else 'in'}"
               if fee else "")
        )
        _intraday_chart(tk)
        pos = next((p for p in state.positions if p.ticker == tk), None)
        if st.button(f"{side} {tk}", type="primary", disabled=not can_edit):
            if not can_edit:
                st.error("🔒 This profile is locked — enter its password in the sidebar.")
            elif shares <= 0:
                st.error("Enter a positive amount.")
            elif side == "Buy" and value + fee > state.cash + 1e-6:
                st.error(f"Not enough cash — {EUR2.m(value)} + {EUR2.m(fee)} "
                         f"fee, you have {EUR2.m(state.cash)}.")
            elif side == "Sell" and (pos is None or shares > pos.shares + 1e-6):
                st.error(f"You only hold {pos.shares:,.4f} {tk}." if pos
                         else f"You don't hold any {tk}.")
            else:
                # store native price + its currency; fee back in native units so a
                # later currency switch reconverts both consistently
                store.practice_record_trade(
                    aid, side.lower(), tk, shares, price_native,
                    fee / rate if rate else fee, cur or acct_ccy)
                st.toast(f"{side} {shares:,.4f} {tk} @ {EUR.m(price_now)}"
                         + (f"  (fee {EUR2.m(fee)})" if fee else ""), icon="🎮")
                st.rerun()

    # ---- holdings ----
    if state.positions:
        st.subheader("Your holdings")
        hdf = pd.DataFrame([{
            "Ticker": p.ticker,
            "Market": p.sector or "—",
            "Origin": f"{_flag(p.country)} {p.country or '—'}",
            "Ccy": fx.normalize(p.currency)[0] or "—",
            "Shares": round(p.shares, 4),
            f"Avg cost ({acct_ccy})": round(p.avg_cost, 2),
            f"Price now ({acct_ccy})": round(p.last_price, 2),
            f"Value ({acct_ccy})": round(p.market_value, 0),
            f"P&L ({acct_ccy})": round(p.unrealized, 0),
            "P&L %": round(p.unrealized_pct * 100, 1),
            "Weight %": round(p.weight * 100, 1),
        } for p in state.positions])
        st.dataframe(hdf, hide_index=True, use_container_width=True)
        d = st.columns(4)
        d[0].metric("Unrealized P&L", EUR.format(sum(p.unrealized for p in state.positions)))
        d[1].metric("Realized P&L", EUR.format(state.realized_pnl))
        d[2].metric("Fees paid", EUR2.format(state.fees_paid),
                    help="Total commission + FX fees across every trade so far.")
        d[3].metric("Trades made", state.n_trades)

        st.subheader("Price movement")
        _pick = st.selectbox("Holding", [p.ticker for p in state.positions],
                             key="paper_movement_pick")
        _intraday_chart(_pick)

        _render_paper_overview(state, acct_ccy)
    elif trades.empty:
        st.info("No trades yet. Pick a ticker above and buy something to get started — "
                "try a broad ETF like **VTI** or **BND**, or a company you know.")

    # ---- equity curve ----
    if not ec.empty:
        st.subheader("How you're doing vs the market")
        fig = px.line(ec)
        fig.add_hline(y=starting, line=dict(color="#888", dash="dot"),
                      annotation_text="you started here")
        fig.update_layout(height=360, margin=dict(t=10, b=10),
                          yaxis_title=f"value ({acct_ccy})",
                          xaxis_title="", legend_title="",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        if "All-in S&P 500" in ec.columns:
            you, spy = ec["Your portfolio"].iloc[-1], ec["All-in S&P 500"].iloc[-1]
            diff = you - spy
            st.markdown(
                f"Since your first trade you're at **{EUR.m(you)}** vs "
                f"**{EUR.m(spy)}** if you'd just bought the S&P 500 — "
                f"**{EUR.m(abs(diff))} {'ahead' if diff >= 0 else 'behind'}**. "
                "Beating the market consistently is *hard* — that's the lesson."
            )

    # ---- trade journal ----
    if not trades_raw.empty:
        st.subheader("📝 Trade journal")
        st.caption("Every trade, with room for the *why* — jot down your thesis so "
                   "future-you can check it against what actually happened. Anyone "
                   "can read these notes; only this profile's password can edit them.")

        log = trades.copy()
        log["Date"] = log["ts"].dt.strftime("%Y-%m-%d %H:%M")
        log["Side"] = log["side"].str.upper()
        log["Ticker"] = log["ticker"]
        log["Shares"] = log["shares"].round(4)
        log[f"Price ({acct_ccy})"] = log["price"].round(2)
        log[f"Fee ({acct_ccy})"] = log["fee"].round(2)
        log["Note"] = log["note"].fillna("").apply(
            lambda n: (n[:57] + "…") if len(n) > 60 else n)
        cols = ["Date", "Side", "Ticker", "Shares", f"Price ({acct_ccy})",
                f"Fee ({acct_ccy})", "Note"]
        view = log[["id", *cols]].iloc[::-1].reset_index(drop=True)
        st.dataframe(view[cols], hide_index=True, use_container_width=True)
        st.caption("Prices shown converted to " + acct_ccy
                   + " at each trade date's exchange rate. Notes are truncated "
                     "above — write and read the full thing below.")

        st.markdown("**📝 Note & 📊 chart for one trade**")
        _labels = {
            int(r["id"]): f"{r['side'].upper()} {r['ticker']} — {r['shares']:.4g} sh "
                          f"@ {r['price']:,.2f} {(r['ccy'] or acct_ccy)} "
                          f"({r['ts']:%d %b %Y})"
            for _, r in trades_raw.iloc[::-1].iterrows()
        }
        _tid = st.selectbox("Trade", list(_labels), format_func=lambda i: _labels[i],
                            key="paper_chart_trade")
        if _tid is not None:
            _row = trades_raw.loc[trades_raw["id"] == _tid].iloc[0]
            note_key = f"note_edit_{aid}_{_tid}"
            db_note = _row["note"] or ""
            if note_key not in st.session_state:
                st.session_state[note_key] = db_note

            st.text_area(
                "Your thesis, the catalyst, what would change your mind — as much "
                "as you want",
                key=note_key, height=180, max_chars=4000, disabled=not can_edit,
                placeholder="Why this trade?",
            )
            dirty = st.session_state[note_key].strip() != db_note.strip()
            bc = st.columns([1, 5])
            if bc[0].button("💾 Save note", disabled=not can_edit or not dirty,
                            key=f"save_note_{_tid}"):
                store.practice_set_trade_note(_tid, st.session_state[note_key])
                st.toast("Note saved.", icon="📝")
                st.rerun()
            if not can_edit:
                bc[1].caption("🔒 Locked — enter the password in the sidebar to edit.")
            elif dirty:
                bc[1].caption("Unsaved changes.")

            _trade_history_chart(_row["ticker"], _row["ts"], float(_row["price"]),
                                 _row["side"], _row["ccy"] or acct_ccy)

        if st.button("Undo last trade", disabled=not can_edit):
            if can_edit:
                store.practice_undo_last(aid)
                st.rerun()
        if not can_edit:
            st.caption("🔒 Locked — enter the password in the sidebar to undo.")

    with st.expander("Start over"):
        newcash = st.number_input(f"Fresh starting cash ({acct_ccy})", 100.0, 1e8,
                                  float(round(starting)), 500.0)
        if st.button("Reset this practice portfolio", type="secondary",
                     disabled=not can_edit):
            if can_edit:
                store.practice_reset(aid, newcash)
                store.practice_set_currency(aid, acct_ccy)
                st.toast("Practice portfolio reset.", icon="♻️")
                st.rerun()
        if not can_edit:
            st.caption("🔒 Locked — enter the password in the sidebar to reset.")

    st.divider()
    st.caption(f"Paper trading — no real money. Prices are delayed closes, converted "
               f"to {acct_ccy} at historical exchange rates. Commission and an FX fee "
               "are modelled; spreads, slippage, dividends and taxes are not, so real "
               "results would still differ. This is a learning sandbox.")


# --------------------------------------------------------------------------- #
# Sidebar - mode switch, then account / portfolio editor / inputs
# --------------------------------------------------------------------------- #
sb = st.sidebar
sb.title("📊 Portfolio Suggestions")
sb.caption("Decision-support only — not investment advice.")

_saved_ccy = store.get_setting("currency", "EUR")
_ccy_idx = fx.CURRENCIES.index(_saved_ccy) if _saved_ccy in fx.CURRENCIES else 0
ACCOUNT_CCY = sb.selectbox(
    "Currency", fx.CURRENCIES, index=_ccy_idx,
    format_func=lambda c: f"{c} · {fx.symbol(c).strip()}",
    help="Everything below is shown and calculated in this currency. Prices in "
         "other currencies are converted with historical exchange rates, so "
         "currency moves show up in your returns.",
)
if ACCOUNT_CCY != _saved_ccy:
    store.set_setting("currency", ACCOUNT_CCY)
EUR.ccy = ACCOUNT_CCY
EUR2.ccy = ACCOUNT_CCY
CCY_SYM = fx.symbol(ACCOUNT_CCY).strip() or ACCOUNT_CCY
sb.divider()

app_mode = sb.radio(
    "Mode",
    ["🌱 Explore from cash", "🔎 Research a stock or bond",
     "🎮 Practice portfolio", "📈 Analyze a portfolio"],
    key="app_mode",
    help="Explore = beginner risk/mix sandbox. Research = look up one security. "
         "Practice = paper-trade with fake money at real prices. "
         "Analyze = optimise a portfolio you already have.",
)
sb.divider()

if app_mode.startswith("🌱"):
    render_explore()
    st.stop()

if app_mode.startswith("🔎"):
    render_research()
    st.stop()

if app_mode.startswith("🎮"):
    render_paper()
    st.stop()

sb.checkbox(
    "Beginner mode", key="beginner_mode", value=True,
    help="Keeps things simple: hides the advanced optimizer knobs and uses sensible defaults.",
)
sb.divider()

_AMOUNT_LABELS = {
    "shares": "Number of shares",
    "dollars": "Dollar value ($)",
    "weight": "Percentages (%)",
}

# ---- Holdings (always available - no sign-in needed) ----
sb.subheader("1 · What you hold")
sb.caption("Type the tickers you own. Don't own anything yet? Leave the example "
           "in, or switch to **🌱 Explore from cash** above.")
edited = sb.data_editor(
    st.session_state.editor_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key=f"editor_{st.session_state.editor_nonce}",
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker", required=True, max_chars=12,
                                              help="e.g. AAPL, VTI, MSFT"),
        "Amount": st.column_config.NumberColumn("Amount", min_value=0.0, format="%.4f"),
    },
)
st.session_state.editor_df = edited

sb.radio(
    "Those numbers are…",
    ["shares", "dollars", "weight"],
    format_func=lambda x: _AMOUNT_LABELS[x],
    key="amount_mode",
    help="Shares get priced at the latest close. Percentages are used as-is.",
)

with sb.expander("Paste a list instead"):
    _imp = st.text_area(
        "One `TICKER, amount` per line, or CSV with ticker/amount columns",
        height=110,
        key="import_txt",
    )
    if st.button("Fill the table with this"):
        _parsed = portfolio.parse_holdings_text(_imp)
        if _parsed.empty:
            st.error("Couldn't read any tickers from that.")
        else:
            _reset_editor(
                _parsed.rename_axis("Ticker").reset_index().rename(columns={"amount": "Amount"})
            )
            st.rerun()

# ---- Optional account: save & load between visits ----
_profiles = store.list_profiles()
profile = st.session_state.profile if st.session_state.profile in _profiles else None
run = False

with sb.expander("💾 Save & load  (optional)", expanded=bool(profile)):
    st.caption("Profiles remember your portfolios between visits. Stored only on "
               "this computer — no account, no password.")
    _opts = _profiles + [NEW_PROFILE]
    _idx = _profiles.index(profile) if profile else (0 if _profiles else len(_opts) - 1)
    _choice = st.selectbox(
        "Profile", _opts, index=_idx, key=f"profile_select_{st.session_state.profile_nonce}"
    )

    if _choice == NEW_PROFILE:
        _new = st.text_input("Pick a name", key="new_profile_name",
                             placeholder="anything — e.g. your first name")
        if st.button("Create", use_container_width=True):
            if _new.strip():
                store.get_or_create_profile(_new.strip())
                st.session_state.profile = _new.strip()
                st.session_state.profile_nonce += 1
                st.session_state.pop("new_profile_name", None)
                _set_loaded(UNSAVED)
                st.rerun()
            else:
                st.error("Type a name first.")
        profile = st.session_state.profile if st.session_state.profile in _profiles else None
    else:
        if _choice != st.session_state.profile:
            st.session_state.profile = _choice
            _set_loaded(UNSAVED)
            st.rerun()
        profile = _choice

    if profile:
        _plist = store.list_portfolios(profile)
        _saved = _plist["name"].tolist() if not _plist.empty else []
        _poptions = [UNSAVED] + _saved
        if st.session_state.loaded_portfolio not in _poptions:
            st.session_state.loaded_portfolio = UNSAVED
        _sel = st.selectbox(
            "Saved portfolios", _poptions,
            index=_poptions.index(st.session_state.loaded_portfolio),
            key=f"port_select_{st.session_state.port_nonce}",
        )
        if _sel != st.session_state.loaded_portfolio:
            _set_loaded(_sel)
            st.rerun()
        lc = st.columns(3)
        if lc[0].button("Load", disabled=_sel == UNSAVED, use_container_width=True,
                        help="Put this saved portfolio into the table above"):
            d = store.load_portfolio(profile, _sel)
            if d:
                _reset_editor(d["positions"])
                _set_loaded(_sel)
                st.session_state.amount_mode = d["amount_mode"]
                st.session_state.watchlist_text = d["watchlist"] or st.session_state.watchlist_text
                st.toast(f"Loaded '{_sel}'", icon="📂")
                st.rerun()
        if lc[1].button("Copy", disabled=_sel == UNSAVED, use_container_width=True,
                        help="Save a duplicate under a new name"):
            n, i = f"{_sel} copy", 2
            while n in _saved:
                n, i = f"{_sel} copy {i}", i + 1
            store.duplicate_portfolio(profile, _sel, n)
            _set_loaded(n)
            st.rerun()
        if lc[2].button("Delete", disabled=_sel == UNSAVED, use_container_width=True,
                        help="Delete this saved portfolio"):
            store.delete_portfolio(profile, _sel)
            _set_loaded(UNSAVED)
            st.rerun()

        _default_name = "" if st.session_state.loaded_portfolio == UNSAVED else st.session_state.loaded_portfolio
        _save_name = st.text_input(
            "Name", value=_default_name, key=f"save_name_{st.session_state.port_nonce}",
            placeholder="e.g. My portfolio",
        )
        if st.button("💾 Save the table above", type="primary", use_container_width=True):
            try:
                _pos = portfolio.normalize_editor_df(st.session_state.editor_df).reset_index()
                store.save_portfolio(
                    profile, _save_name, _pos,
                    amount_mode=st.session_state.amount_mode,
                    watchlist=st.session_state.watchlist_text,
                )
                _set_loaded(_save_name.strip())
                st.toast(f"Saved '{_save_name.strip()}'", icon="💾")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

        if st.button("Delete this profile", help="Removes the profile and all its portfolios"):
            st.session_state.confirm_delete_profile = True
        if st.session_state.get("confirm_delete_profile"):
            st.warning(f"Delete **{profile}** and everything saved in it?")
            dc = st.columns(2)
            if dc[0].button("Yes, delete", use_container_width=True):
                store.delete_profile(profile)
                st.session_state.profile = None
                st.session_state.confirm_delete_profile = False
                st.session_state.profile_nonce += 1
                _set_loaded(UNSAVED)
                st.rerun()
            if dc[1].button("Keep it", use_container_width=True):
                st.session_state.confirm_delete_profile = False
                st.rerun()
    else:
        st.caption("↑ Create a profile to unlock saving.")

# ---- What to compare against ----
sb.subheader("2 · Ideas to compare against")
sb.caption("The tool scores these on how well they'd diversify what you hold.")
sb.text_input("Tickers you're curious about (comma-separated)", key="watchlist_text",
              placeholder="e.g. JNJ, XOM, GLD")
sb.checkbox("Include a set of broad ETFs (recommended)", key="use_curated")
sb.checkbox("Also scan the whole S&P 500 (slower)", key="use_sp500")

with sb.expander("Settings"):
    st.slider("Years of history to use", 1.0, 15.0, step=0.5, key="lookback",
              help="More years = includes older crashes; fewer years = more recent behaviour.")
    st.number_input("Risk-free rate (savings / T-bill yield)", 0.0, 0.10, step=0.005,
                    format="%.3f", key="rf_rate")
    st.slider("Hypothetical position size for the scan", 0.02, 0.25, step=0.01, key="add_weight",
              help="When scoring an idea, the tool pretends you add this much of it.")

sb.divider()
run = sb.button("📊 Analyze my portfolio", type="primary", use_container_width=True)

if run:
    parsed = portfolio.normalize_editor_df(st.session_state.editor_df)
    if parsed.empty:
        st.sidebar.error("Add at least one holding with a positive amount.")
        st.stop()
    mode = st.session_state.amount_mode
    latest = _latest_prices(tuple(sorted(parsed.index))) if mode == "shares" else None
    if mode == "shares" and latest:
        _meta = _origins(tuple(sorted(parsed.index)))
        latest = {t: v * _fx_spot(_meta.get(t, {}).get("currency") or ACCOUNT_CCY,
                                  ACCOUNT_CCY)
                  for t, v in latest.items()}
    weights = portfolio.to_weights(
        parsed, mode=mode, latest_prices=pd.Series(latest) if latest else None
    )
    if not weights:
        st.sidebar.error("Could not price these tickers — check the symbols.")
        st.stop()

    # A real money value for the holdings, when we can derive one.
    if mode == "dollars":
        st.session_state.portfolio_value = float(parsed["amount"].sum())
    elif mode == "shares" and latest:
        _lp = pd.Series(latest)
        st.session_state.portfolio_value = float(
            (parsed["amount"] * _lp.reindex(parsed.index)).dropna().sum()
        )
    else:
        st.session_state.portfolio_value = None
    watchlist = [
        t.strip().upper()
        for t in st.session_state.watchlist_text.replace(",", " ").split()
        if t.strip()
    ]
    try:
        with st.spinner("Pulling prices and crunching numbers…"):
            st.session_state.analysis = _run_analysis(
                tuple(sorted(weights.items())),
                tuple(watchlist),
                st.session_state.use_curated,
                st.session_state.use_sp500,
                st.session_state.lookback,
                st.session_state.rf_rate,
                st.session_state.add_weight,
            )
            st.session_state.rf = st.session_state.rf_rate
            st.session_state.pop("opt_results", None)
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.stop()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
st.title("📈 Analyze a portfolio")

if "analysis" not in st.session_state:
    st.info("Fill in the table on the left (or keep the example), then press "
            "**📊 Analyze my portfolio**. No sign-in needed.")
    intro, learn = st.tabs(["What you'll get", "📖 New to investing? Start here"])
    with intro:
        st.markdown(
            """
            1. **A health check** of what you hold — how much it swings, its worst
               historical drop, how concentrated it is, and which sectors carry
               the risk. Written up in plain language.
            2. **Ideas that would balance it out** — your watchlist plus a set of
               broad ETFs, ranked by how well each one would diversify what you
               already own.
            3. **Four "what a computer would pick" portfolios** — different methods
               for choosing weights, shown next to your current mix so you can see
               where they agree and disagree.

            It's all based on past data. Treat it as *"what would have worked"*,
            not a prediction.
            """
        )
    with learn:
        st.markdown(education.GETTING_STARTED)
        st.divider()
        st.markdown(education.HOW_TO_USE_FROM_CASH)
    st.stop()

A: suggest.Analysis = st.session_state.analysis
rf = st.session_state.get("rf", 0.04)

_FRIENDLY_WARN = {
    "No candidate price data retrieved.":
        "No ideas to compare against — tick **Include a set of broad ETFs** in "
        "the sidebar, or add a few tickers to the watchlist, then Analyze again.",
}
for w in A.warnings:
    st.warning(_FRIENDLY_WARN.get(w, w))

(tab_overview, tab_scan, tab_opt, tab_plan, tab_outlook, tab_timing,
 tab_idea, tab_summary, tab_learn) = st.tabs(
    ["Portfolio overview", "Diversification scan", "Optimizers", "Action plan",
     "Outlook", "Timing check", "💡 Idea worksheet", "Suggestion summary", "📖 Learn"]
)

# --------------------------------------------------------------------------- #
# Tab 1 - Overview
# --------------------------------------------------------------------------- #
with tab_overview:
    snap = A.snapshot
    c = st.columns(4)
    c[0].metric("Annualized return", PCT.format(snap.ann_return), help=H["Annualized return"])
    c[1].metric("Annualized volatility", PCT.format(snap.ann_vol), help=H["Annualized volatility"])
    c[2].metric("Sharpe ratio", f"{snap.sharpe:.2f}", help=H["Sharpe ratio"])
    c[3].metric("Beta to market", f"{snap.beta:.2f}", help=H["Beta to market"])
    c = st.columns(4)
    c[0].metric("Sortino ratio", f"{snap.sortino:.2f}", help=H["Sortino ratio"])
    c[1].metric("Max drawdown", PCT.format(snap.max_drawdown), help=H["Max drawdown"])
    c[2].metric("Diversification ratio", f"{snap.diversification_ratio:.2f}",
                help=H["Diversification ratio"])
    c[3].metric("Effective # holdings", f"{A.concentration['effective_n']:.1f}",
                help=H["Effective # holdings"]
                + f"\n\nHHI {A.concentration['hhi']:.3f}; largest position "
                f"{A.concentration['top_weight']:.0%}")

    st.caption("ℹ️ Hover any metric's ? for a plain-language explanation. Full "
               "glossary and method notes are in the **Learn** tab.")

    _top_sector = A.sector_exposure.index[0] if not A.sector_exposure.empty and \
        A.sector_exposure.iloc[0] > 0.35 else None
    st.info("**What this means — ** " + education.interpret_snapshot(
        snap, A.concentration, _top_sector))

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Current allocation")
        wser = pd.Series(A.holdings_weights).sort_values(ascending=False)
        fig = px.pie(values=wser.values, names=wser.index, hole=0.45)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Sector exposure")
        if not A.sector_exposure.empty:
            se = A.sector_exposure.sort_values()
            fig = px.bar(se, orientation="h", labels={"value": "weight", "index": ""})
            fig.update_layout(showlegend=False, height=360, margin=dict(t=10, b=10),
                              xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Sector data unavailable for these tickers.")

    st.subheader("Growth of $1 (in-sample)")
    rets = metrics.daily_returns(A.holdings_prices)
    pr = metrics.portfolio_return_series(rets, A.holdings_weights)
    mkt = metrics.daily_returns(A.market_prices)
    curve = pd.DataFrame(
        {"Your portfolio": (1 + pr).cumprod(), "SPY": (1 + mkt.reindex(pr.index)).cumprod()}
    ).dropna()
    fig = px.line(curve)
    fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis_title="", xaxis_title="",
                      legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Holdings correlation")
    corr = rets.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=420, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if not A.sector_gaps.empty:
        over = A.sector_gaps[A.sector_gaps["over_concentrated"]]
        if not over.empty:
            st.info(
                "Concentrated in: "
                + ", ".join(f"**{s}** ({w:.0%})" for s, w in over["weight"].items())
                + " — the scan favors names that add different exposure."
            )

# --------------------------------------------------------------------------- #
# Tab 2 - Diversification scan
# --------------------------------------------------------------------------- #
with tab_scan:
    if A.ranked.empty:
        st.warning("No candidates were scored.")
    else:
        st.caption(
            f"Ranking {len(A.ranked)} candidates by how a {A.scores.attrs.get('base_vol', 0):.0%}-vol "
            "portfolio changes when each is added at the test weight. "
            f"Baseline Sharpe {A.scores.attrs.get('base_sharpe', float('nan')):.2f}, "
            f"diversification ratio {A.scores.attrs.get('base_div', float('nan')):.2f}."
        )
        src = st.multiselect(
            "Filter by source",
            sorted(A.ranked["source"].dropna().unique()) if "source" in A.ranked else [],
            default=sorted(A.ranked["source"].dropna().unique()) if "source" in A.ranked else [],
        )
        view = A.ranked[A.ranked["source"].isin(src)] if src else A.ranked
        n_view = len(view)
        if n_view > 6:
            top_n = st.slider("Show top N", 5, min(50, n_view),
                              min(15, n_view), key="scan_top_n")
        else:
            top_n = n_view
        view = view.head(top_n)

        show = view.reset_index().rename(columns={"index": "ticker"})
        disp = pd.DataFrame(
            {
                "Ticker": show["ticker"],
                "Name": show.get("name", show["ticker"]),
                "Category": show.get("category", ""),
                "Fit score": show["fit_score"].round(1),
                "Corr to portfolio": show["corr_to_portfolio"].round(2),
                "Vol Δ (pp)": (show["vol_delta"] * 100).round(2),
                "Sharpe Δ": show["sharpe_delta"].round(3),
                "Div-ratio Δ": show["div_ratio_delta"].round(3),
                "Why": show["rationale"],
            }
        )
        st.dataframe(
            disp,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Fit score": st.column_config.ProgressColumn(
                    "Fit score", min_value=0, max_value=100, format="%.0f"
                )
            },
        )

        fig = px.bar(
            view.sort_values("corr_to_portfolio"),
            x="corr_to_portfolio",
            y=view.sort_values("corr_to_portfolio").index,
            color="vol_delta",
            color_continuous_scale="RdYlGn_r",
            labels={"corr_to_portfolio": "correlation to your portfolio", "y": ""},
        )
        fig.update_layout(height=max(300, 22 * len(view)), margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download full ranking (CSV)",
            A.ranked.to_csv().encode(),
            "diversification_ranking.csv",
            "text/csv",
        )

# --------------------------------------------------------------------------- #
# Tab 3 - Optimizers
# --------------------------------------------------------------------------- #
with tab_opt:
    st.caption("Pick candidates to add, choose methods, and compare target weights.")
    default_adds = list(A.ranked.head(5).index) if not A.ranked.empty else []
    pool = list(A.candidate_prices.columns) if not A.candidate_prices.empty else []
    adds = st.multiselect("Candidate additions to include", pool, default=default_adds)

    _beginner = st.session_state.get("beginner_mode", True)

    cc = st.columns(4)
    m_mv = cc[0].checkbox("Mean-Variance (MPT)", value=True)
    m_rp = cc[1].checkbox("Risk Parity", value=True)
    m_hrp = cc[2].checkbox("Hierarchical RP", value=True)
    m_bl = cc[3].checkbox("Black-Litterman", value=True)

    if _beginner:
        objective, target_vol, return_method, views = "max_sharpe", 0.15, "blend", {}
        keep_existing = st.checkbox(
            "Keep my existing positions (don't sell them down much)", value=True
        )
        st.caption("Advanced knobs (objective, return estimate, Black-Litterman "
                   "views) are hidden — turn off **Beginner mode** in the sidebar to show them.")
    else:
        oc = st.columns(3)
        objective = oc[0].selectbox("MPT objective", ["max_sharpe", "min_vol", "efficient_risk"])
        target_vol = oc[1].slider("Target vol (efficient_risk)", 0.05, 0.40, 0.15, 0.01)
        return_method = oc[2].selectbox(
            "Expected-return estimate", ["blend", "capm", "ema", "mean"],
            help="'blend' = 50/50 CAPM + EMA, a light shrinkage. 'mean' is the noisy raw historical mean.",
        )
        keep_existing = st.checkbox(
            "Keep existing positions (floor at 50% of current weight)", value=True
        )
        views_text = st.text_area(
            "Black-Litterman views (optional) — `TICKER: annual_return`, one per line",
            value="",
            height=80,
            placeholder="NVDA: 0.15\nXLU: 0.06",
        )
        views = {}
        for line in views_text.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                try:
                    views[k.strip().upper()] = float(v)
                except ValueError:
                    pass

    methods = []
    if m_mv:
        methods.append("mean_variance")
    if m_rp:
        methods.append("risk_parity")
    if m_hrp:
        methods.append("hrp")
    if m_bl:
        methods.append("black_litterman")

    if st.button("Run optimizers", type="primary"):
        with st.spinner("Optimizing…"):
            st.session_state.opt_results = suggest.optimize_with_candidates(
                A, adds, methods, objective=objective, rf=rf,
                keep_existing=keep_existing, target_vol=target_vol,
                views=views, return_method=return_method,
            )

    results = st.session_state.get("opt_results")
    if results:
        perf_rows = []
        for key, res in results.items():
            perf_rows.append(
                {
                    "Method": res.method,
                    "Exp. return": res.exp_return,
                    "Exp. volatility": res.exp_vol,
                    "Exp. Sharpe": res.exp_sharpe,
                    "# positions": len(res.nonzero()),
                }
            )
        pdf = pd.DataFrame(perf_rows)
        st.dataframe(
            pdf.style.format(
                {"Exp. return": PCT.format, "Exp. volatility": PCT.format, "Exp. Sharpe": "{:.2f}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        for res in results.values():
            for n in res.notes:
                st.caption(f"· {res.method}: {n}")

        wt = suggest.weights_table(A.holdings_weights, results)
        st.subheader("Target weights vs current")
        st.dataframe(
            wt.style.format("{:.1%}").background_gradient(cmap="Blues", axis=None),
            use_container_width=True,
        )

        melt = wt.reset_index(names="ticker").melt("ticker", var_name="portfolio", value_name="weight")
        fig = px.bar(melt, x="ticker", y="weight", color="portfolio", barmode="group")
        fig.update_layout(height=380, yaxis_tickformat=".0%", margin=dict(t=10, b=10),
                          legend_title="", xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

        # Efficient frontier with current + optimized portfolios marked
        st.subheader("Efficient frontier")
        try:
            held_plus = list(A.holdings_weights) + [t for t in adds if t in A.candidate_prices.columns]
            fprices = A.holdings_prices
            if adds:
                fprices = A.holdings_prices.join(
                    A.candidate_prices[[t for t in adds if t in A.candidate_prices.columns]],
                    how="inner",
                ).dropna()
            front = optimize.efficient_frontier_points(fprices, rf=rf, return_method=return_method)
            fig = go.Figure()
            if not front.empty:
                fig.add_trace(go.Scatter(x=front["vol"], y=front["ret"], mode="lines",
                                         name="Efficient frontier"))
            cur = metrics.snapshot(A.holdings_prices, A.holdings_weights, A.market_prices, rf)
            fig.add_trace(go.Scatter(x=[cur.ann_vol], y=[cur.ann_return], mode="markers+text",
                                     text=["Current"], textposition="top center",
                                     marker=dict(size=13, symbol="x"), name="Current"))
            for res in results.values():
                if not np.isnan(res.exp_vol):
                    fig.add_trace(go.Scatter(x=[res.exp_vol], y=[res.exp_return],
                                             mode="markers+text", text=[res.method],
                                             textposition="bottom center",
                                             marker=dict(size=11), name=res.method))
            fig.update_layout(height=420, xaxis_title="volatility", yaxis_title="expected return",
                              xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                              margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.caption(f"Frontier unavailable: {e}")

        # Risk contribution for the chosen MPT / RP portfolio
        pick = results.get("risk_parity") or next(iter(results.values()))
        rc = optimize.risk_contributions(
            A.holdings_prices.join(
                A.candidate_prices[[t for t in adds if t in A.candidate_prices.columns]], how="inner"
            ).dropna() if adds else A.holdings_prices,
            pick.weights,
        )
        st.subheader(f"Risk contribution — {pick.method}")
        fig = px.bar(rc.sort_values(), orientation="h", labels={"value": "share of portfolio risk", "index": ""})
        fig.update_layout(height=max(280, 22 * len(rc)), showlegend=False,
                          xaxis_tickformat=".0%", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 3b - Action plan (drift, rebalancing trades, contribution steering)
# --------------------------------------------------------------------------- #
with tab_plan:
    st.caption("The genuinely useful bit: turn the analysis into moves you can "
               "actually make — how far your holdings have drifted from a target "
               "mix, and the trades (or where to point new money) to fix it. "
               "No forecasting, just arithmetic.")

    _pv = st.session_state.get("portfolio_value")
    if _pv:
        st.metric("Portfolio value (from your holdings)", EUR.format(_pv))
    else:
        _pv = st.number_input(
            f"Your portfolio's total value ({ACCOUNT_CCY})", min_value=100.0,
            max_value=1e9, value=10_000.0, step=500.0,
            help="You entered weights, so the tool doesn't know the cash amount.")

    _held = list(A.holdings_weights)
    _optres = st.session_state.get("opt_results") or {}
    _tgt_labels = ["Equal weight across my holdings",
                   "Risk parity (each holding = equal risk)"]
    _tgt_labels += [f"Match optimizer: {r.method}" for r in _optres.values() if r.nonzero()]
    _tgt_labels += ["Type my own target %"]
    _tchoice = st.selectbox("Target mix to steer toward", _tgt_labels)

    if _tchoice.startswith("Equal"):
        target = planning.equal_weight(_held)
    elif _tchoice.startswith("Risk parity"):
        try:
            target = optimize.risk_parity(A.holdings_prices).nonzero()
        except Exception:
            target = planning.equal_weight(_held)
            st.warning("Risk parity failed here — using equal weight.")
    elif _tchoice.startswith("Match optimizer"):
        _mname = _tchoice.split(": ", 1)[1]
        target = next(r.nonzero() for r in _optres.values() if r.method == _mname)
    else:
        _te = st.data_editor(
            pd.DataFrame({"Ticker": _held,
                          "Target %": [round(A.holdings_weights[t] * 100, 1) for t in _held]}),
            hide_index=True, use_container_width=True, key="plan_target_editor",
            num_rows="dynamic")
        target = {}
        for _, row in _te.iterrows():
            try:
                target[str(row["Ticker"]).upper().strip()] = float(row["Target %"]) / 100.0
            except (ValueError, TypeError):
                pass

    oc = st.columns(3)
    _band_pp = oc[0].slider("Rebalance when drift exceeds", 2, 15, 5, 1, format="%d pp",
                            help="A 5-point band is the common '5/25 rule'. "
                                 "Wider = fewer, larger trades.")
    _band = _band_pp / 100.0
    _contrib = oc[1].number_input(
        f"New money to add now ({ACCOUNT_CCY}) — for Option B", 0.0, 1e8, 0.0, 100.0,
        help="A lump sum or a monthly contribution. Option B below shows how to "
             "split it across your underweight holdings.")

    if not target or sum(target.values()) <= 0:
        st.info("Pick or enter a target mix.")
    else:
        pl = planning.build_plan(A.holdings_weights, target, float(_pv),
                                 band_pp=_band, contribution=float(_contrib))
        for n in pl.notes:
            st.markdown("• " + n)

        st.subheader("Current vs target")
        rows = []
        for r in sorted(pl.rows, key=lambda r: -abs(r.drift_pp)):
            rows.append({
                "Ticker": r.ticker,
                "Now": f"{r.current_weight:.1%}",
                "Target": f"{r.target_weight:.1%}",
                "Drift": f"{r.drift_pp:+.1%}",
                "": "⚠︎ outside band" if r.breach else "ok",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

        _melt = pd.DataFrame({
            "ticker": [r.ticker for r in pl.rows] * 2,
            "weight": [r.current_weight for r in pl.rows] + [r.target_weight for r in pl.rows],
            "which": ["now"] * len(pl.rows) + ["target"] * len(pl.rows),
        })
        fig = px.bar(_melt, x="ticker", y="weight", color="which", barmode="group",
                     color_discrete_map={"now": "#9aa0a6", "target": "#2e86ab"})
        fig.update_layout(height=300, yaxis_tickformat=".0%", margin=dict(t=10, b=10),
                          xaxis_title="", yaxis_title="", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

        if pl.needs_rebalance:
            st.subheader("Option A — rebalance now (involves selling)")
            tr = [{"Action": "SELL", "Ticker": r.ticker,
                   "Amount": EUR.format(abs(r.trade_value))} for r in pl.sells]
            tr += [{"Action": "BUY", "Ticker": r.ticker,
                    "Amount": EUR.format(r.trade_value)} for r in pl.buys]
            st.dataframe(pd.DataFrame(tr), hide_index=True, use_container_width=True)
            st.caption("Selling winners in a taxable account can trigger capital-gains "
                       "tax — in a tax-sheltered account it's free. Option B avoids it.")
        else:
            st.success("Nothing is outside its band — no rebalancing trade needed.")

        st.subheader("Option B — add new money instead (no selling)")
        if _contrib <= 0:
            st.info("Set **New money to add now** above (e.g. a monthly contribution) "
                    "and this shows how to split it across your underweight holdings — "
                    "drifting toward target with **no selling and no capital-gains tax**. "
                    "Done with every contribution, it can replace Option A entirely.")
        else:
            cs = [{"Ticker": t, "Add": EUR.format(v),
                   "→ new weight": f"{(A.holdings_weights.get(t, 0) * _pv + v) / (_pv + _contrib):.1%}"}
                  for t, v in sorted(pl.contribution_split.items(), key=lambda kv: -kv[1]) if v > 1]
            st.markdown(f"Split your **{EUR.m(_contrib)}** like this:")
            st.dataframe(pd.DataFrame(cs), hide_index=True, use_container_width=True)
            st.caption("New money flows to whatever is most underweight first, nudging "
                       "you toward target with zero tax drag. Do this with every "
                       "contribution and you may rarely need Option A.")

    st.divider()
    st.caption("Not advice. Rebalancing bands are a standard, non-predictive "
               "discipline; the target mix is your choice, not the tool's.")


# --------------------------------------------------------------------------- #
# Tab 4 - Outlook (per-asset scenario ranges, NOT predictions)
# --------------------------------------------------------------------------- #
with tab_outlook:
    st.error(
        "**This is not a prediction.** Individual stocks and bonds can't be "
        "reliably forecast. Below are (a) three textbook estimates of expected "
        "return — shown together *because they disagree* — and (b) a range of "
        "what-if outcomes from resampling each asset's own history. Read the "
        "width of the ranges, not the middle."
    )

    _wl = sorted(
        A.candidate_meta.index[A.candidate_meta.get("source", pd.Series(dtype=str)) == "watchlist"]
    ) if not A.candidate_meta.empty else []
    _anchor = [t for t in ("SPY", "BND") if t]
    _all_opts = sorted(set(_wl) | set(_anchor) | set(A.holdings_weights))
    _default = _wl or _anchor

    pick = st.multiselect("Assets to look at", _all_opts, default=_default,
                          help="Your watchlist tickers, plus your holdings and two "
                               "market anchors (SPY = US stocks, BND = US bonds).")
    oc = st.columns([2, 2, 3])
    horizon = oc[0].slider("Horizon (years)", 1, 10, 1,
                           help="Longer horizons make the range explode — that itself is the lesson.")
    _eng = oc[1].radio("Scenario engine", ["Resample history", "Neural generator"],
                       horizontal=False, key="outlook_engine")
    o_method = "neural" if _eng.startswith("Neural") else "bootstrap"

    if not pick:
        st.info("Pick at least one asset. Add tickers to your watchlist in the sidebar "
                "for them to show up here.")
    else:
        with st.spinner("Building scenario ranges…"):
            outs = _outlooks(tuple(pick), rf, float(horizon), o_method)
        if not outs:
            st.warning("Couldn't get data for those tickers.")
        else:
            def _pct(x, signed=True):
                if x != x:
                    return "—"
                return f"{x:+.0%}" if signed else f"{x:.0%}"

            rows = []
            for o in outs:
                rows.append({
                    "Ticker": o.ticker,
                    "History": f"{o.n_months} mo",
                    "Volatility / yr": _pct(o.ann_vol, signed=False),
                    "Estimate: hist mean": _pct(o.est_hist_mean),
                    "Estimate: CAPM": _pct(o.est_capm),
                    "Estimate: EMA": _pct(o.est_ema),
                    f"{horizon}y low (10th)": _pct(o.p10),
                    f"{horizon}y middle": _pct(o.p50),
                    f"{horizon}y high (90th)": _pct(o.p90),
                    "Own worst / best 12mo": f"{_pct(o.hist_worst_12m)} / {_pct(o.hist_best_12m)}",
                })
            st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)

            fig = go.Figure()
            for o in outs:
                fig.add_trace(go.Scatter(
                    x=[o.p10, o.p90], y=[o.ticker, o.ticker], mode="lines",
                    line=dict(color="rgba(46,134,171,0.45)", width=10),
                    showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[o.p25, o.p75], y=[o.ticker, o.ticker], mode="lines",
                    line=dict(color="rgba(46,134,171,0.9)", width=10),
                    showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(
                    x=[o.p50], y=[o.ticker], mode="markers",
                    marker=dict(symbol="line-ns", size=18, color="white",
                                line=dict(width=2)),
                    name="scenario midpoint", showlegend=False))
                for label, val, sym in [("hist mean", o.est_hist_mean, "circle"),
                                        ("CAPM", o.est_capm, "diamond"),
                                        ("EMA", o.est_ema, "square")]:
                    if val == val:
                        fig.add_trace(go.Scatter(
                            x=[val], y=[o.ticker], mode="markers",
                            marker=dict(symbol=sym, size=9, color="#e4572e"),
                            name=f"estimate: {label}",
                            showlegend=(o is outs[0])))
            fig.add_vline(x=0, line=dict(color="#888", dash="dot"))
            fig.update_layout(
                height=max(240, 90 * len(outs)),
                xaxis_title=f"total return over {horizon} year(s)",
                xaxis_tickformat="+.0%", margin=dict(t=30, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Bar = 10th–90th percentile of resampled scenarios (darker = "
                       "25th–75th), white tick = midpoint. Orange marks = the three "
                       "textbook estimates. When the orange marks are scattered and "
                       "the bar is wide, that's the honest picture: no one knows.")

            for o in outs:
                prefix = ("⚠️ only ~%d months of history — treat this as noise. " % o.n_months
                          if o.n_months < 30 else "")
                st.markdown(f"**{o.ticker}** — {prefix}{o.note}")

            if o_method == "neural" and any(x.method == "bootstrap" for x in outs):
                st.caption("Some assets fell back to resampling — not enough history "
                           "to train the generator.")

    st.divider()
    st.caption("Not advice. These are statistical what-ifs on past data, not a "
               "view on any company or bond.")


# --------------------------------------------------------------------------- #
# Tab 5 - Timing check (there is no buy signal — this shows the trade-offs)
# --------------------------------------------------------------------------- #
with tab_timing:
    st.error(
        "**The buy/sell markers and the projection cone below are not forecasts.** "
        "The markers are a mechanical rule reacting to the *past*; the cone is a "
        "range of paths from historical volatility. This tab exists to show what "
        "such rules have actually delivered — historically smaller drawdowns but "
        "**lower** returns — not where a price is headed."
    )

    _topts = sorted(set(A.holdings_weights) | {"SPY", "QQQ", "BND", "TLT"}
                    | set(A.candidate_meta.index[
                        A.candidate_meta.get("source", pd.Series(dtype=str)) == "watchlist"
                    ] if not A.candidate_meta.empty else []))
    t_pick = st.selectbox("Asset", _topts, index=_topts.index("SPY") if "SPY" in _topts else 0)

    _series = _timing_prices(t_pick)
    if _series.empty or len(_series) < 260:
        st.warning("Not enough price history for this ticker.")
    else:
        # ---- Where things stand ----
        st.subheader("Where things stand")
        tc = timing.trend_context(t_pick, _series)
        st.info(tc.status)
        k = st.columns(4)
        k[0].metric("Price vs 200-day avg", f"{tc.last_price / tc.sma200 - 1:+.1%}"
                    if tc.sma200 == tc.sma200 else "—")
        k[1].metric("From 12-month high", f"{tc.pct_from_52w_high:+.1%}")
        k[2].metric("From all-time high (window)", f"{tc.pct_from_high:+.1%}")
        k[3].metric("In current trend for", f"{max(1, tc.regime_trading_days) // 21} mo"
                    if tc.regime_trading_days >= 21 else f"{tc.regime_trading_days} days")

        # buy/sell markers (50/200 crossover) + a forward projection cone
        cs = timing.crossover_signals(_series)
        proj_m = st.slider("Show a projection cone this many months ahead", 0, 36, 12, 3,
                           help="A resampled range of where the price could go — "
                                "widens fast because uncertainty compounds. Not a forecast.")

        _px = tc.price
        _recent = _px.iloc[-min(len(_px), 252 * 6):]  # last ~6 years for readability
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=_recent.index, y=_recent, name=t_pick, line=dict(color="#2e86ab")))
        fig.add_trace(go.Scatter(x=_recent.index, y=tc.sma50_series.reindex(_recent.index),
                                 name="50-day avg", line=dict(color="#f5b700", width=1)))
        fig.add_trace(go.Scatter(x=_recent.index, y=tc.sma200_series.reindex(_recent.index),
                                 name="200-day avg", line=dict(color="#e4572e", width=1.5)))

        _b = [(d, p) for d, p in zip(cs.buy_dates, cs.buy_prices) if d >= _recent.index[0]]
        _s = [(d, p) for d, p in zip(cs.sell_dates, cs.sell_prices) if d >= _recent.index[0]]
        if _b:
            fig.add_trace(go.Scatter(x=[d for d, _ in _b], y=[p for _, p in _b], mode="markers",
                                     name="rule: buy", marker=dict(symbol="triangle-up", size=12,
                                                                   color="#2ca02c")))
        if _s:
            fig.add_trace(go.Scatter(x=[d for d, _ in _s], y=[p for _, p in _s], mode="markers",
                                     name="rule: sell", marker=dict(symbol="triangle-down", size=12,
                                                                    color="#d62728")))

        if proj_m > 0:
            cone = timing.projection_cone(_series, months=proj_m)
            fig.add_trace(go.Scatter(x=cone.index, y=cone["p90"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cone.index, y=cone["p10"], fill="tonexty",
                                     fillcolor="rgba(46,134,171,0.13)", line=dict(width=0),
                                     name="range of paths (10–90%)"))
            fig.add_trace(go.Scatter(x=cone.index, y=cone["p75"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cone.index, y=cone["p25"], fill="tonexty",
                                     fillcolor="rgba(46,134,171,0.22)", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cone.index, y=cone["p50"], name="if it kept drifting",
                                     line=dict(color="#2e86ab", dash="dot")))

        fig.update_layout(height=380, margin=dict(t=10, b=10), yaxis_title="", xaxis_title="",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        _state = "**in the market**" if cs.currently_in else "**in cash**"
        st.markdown(
            f"The **50/200 crossover rule** (golden cross = buy, death cross = sell) is "
            f"currently {_state}. Its last signal was a **{cs.last_signal}** on "
            f"**{cs.last_signal_date:%d %b %Y}** at {cs.last_signal_price:.2f}. Over the "
            f"whole history it fired **{len(cs.buy_dates)} buys and {len(cs.sell_dates)} sells** "
            "— it flips whenever the trend turns, always a step late."
        )
        if proj_m > 0:
            st.caption(
                f"The shaded cone is where **{t_pick}** could land over the next "
                f"{proj_m} months if the future resembled its own past — the dotted "
                "line is its historical drift. It is **not a forecast**: it widens "
                "fast, and the real price finishes outside it more often than you'd think."
            )

        st.markdown("**Drop from its running peak**")
        fig = px.area(tc.drawdown_series, labels={"value": "", "index": ""})
        fig.update_traces(line_color="#c1121f", fillcolor="rgba(193,18,31,0.25)")
        fig.update_layout(height=180, margin=dict(t=6, b=6), showlegend=False,
                          yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Moving averages are a lagging *description* of trend, not a "
                   "prediction. Price crossing below its 200-day average has "
                   "sometimes preceded a bad stretch and sometimes just whipsawed.")

        # ---- Does the 200-day rule help? ----
        st.subheader("Does the “200-day rule” actually help?")
        st.caption("Rule: hold the asset while it's above its 200-day average, "
                   "move to cash when it drops below. Acting on the prior day's "
                   "close, with a small trading cost.")
        bt = timing.trend_follow_backtest(_series)
        comp = pd.DataFrame(
            {
                "Annual return": [f"{bt.ann_return_bh:+.1%}", f"{bt.ann_return_rule:+.1%}"],
                "Volatility": [f"{bt.vol_bh:.1%}", f"{bt.vol_rule:.1%}"],
                "Worst drawdown": [f"{bt.maxdd_bh:.0%}", f"{bt.maxdd_rule:.0%}"],
                "# trades": ["0", f"{bt.n_trades}"],
                "Time invested": ["100%", f"{bt.time_in_market:.0%}"],
            },
            index=["Buy & hold", "200-day rule"],
        )
        st.dataframe(comp, use_container_width=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt.equity_bh.index, y=bt.equity_bh, name="Buy & hold",
                                 line=dict(color="#2e86ab")))
        fig.add_trace(go.Scatter(x=bt.equity_rule.index, y=bt.equity_rule, name="200-day rule",
                                 line=dict(color="#e4572e")))
        fig.update_layout(height=320, margin=dict(t=10, b=10), yaxis_title="growth of 1",
                          xaxis_title="", legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        _verdict = ("cut the worst drawdown" if bt.maxdd_rule > bt.maxdd_bh else "did not reduce drawdown")
        st.markdown(
            f"Over this history the rule **{_verdict}** "
            f"({bt.maxdd_rule:.0%} vs {bt.maxdd_bh:.0%}) but returned "
            f"**{bt.ann_return_rule:+.1%}/yr vs {bt.ann_return_bh:+.1%}/yr** buy & hold — "
            f"with **{bt.n_trades} trades** and **{bt.missed_while_out:+.0%}** of the "
            "asset's gains earned while the rule had you sitting in cash. Smoother, "
            "not richer — and that's *before* taxes."
        )

        # ---- Track record of the buy signal ----
        st.subheader("What actually happened after every “buy” signal?")
        st.caption("A buy signal = the price closing back **above** its 200-day "
                   "average. Below: the return over the following 3 / 6 / 12 months "
                   "after each signal, next to the return over the same horizons "
                   "from a **randomly chosen day**. If the signal predicted "
                   "anything, the two would look clearly different.")
        tr = timing.signal_track_record(_series)
        if tr.n_signals < 5:
            st.info("Too few signals in this history to say anything.")
        else:
            st.markdown(f"This rule has flashed **{tr.n_signals} buy signals** over the period.")
            trrows = []
            for hm in (3, 6, 12):
                a, b = tr.after_signal[hm], tr.after_any_day[hm]
                trrows.append({
                    "Next…": f"{hm} months",
                    "After a signal — median": f"{a['median']:+.1%}",
                    "After a signal — up how often": f"{a['win']:.0%}",
                    "After any day — median": f"{b['median']:+.1%}",
                    "After any day — up how often": f"{b['win']:.0%}",
                })
            st.dataframe(pd.DataFrame(trrows).set_index("Next…"), use_container_width=True)

            fig = go.Figure()
            for hm in (3, 6, 12):
                fig.add_trace(go.Box(
                    y=tr.fwd_any[hm], x=[f"{hm} months"] * len(tr.fwd_any[hm]),
                    name="after any random day", marker_color="#9aa0a6",
                    legendgroup="any", showlegend=(hm == 3), boxpoints=False))
                fig.add_trace(go.Box(
                    y=tr.fwd_signal[hm], x=[f"{hm} months"] * len(tr.fwd_signal[hm]),
                    name="after a buy signal", marker_color="#2e86ab",
                    legendgroup="sig", showlegend=(hm == 3), boxpoints="outliers"))
            fig.update_layout(boxmode="group", height=360, yaxis_tickformat="+.0%",
                              yaxis_title="return over the following period", xaxis_title="",
                              margin=dict(t=10, b=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            s12, b12 = tr.after_signal[12], tr.after_any_day[12]
            d3s, d3b = tr.after_signal[3], tr.after_any_day[3]
            diff = s12["median"] - b12["median"]
            if abs(diff) < 0.03:
                _line = ("**The signal barely moved the needle** — you'd have done "
                         "about as well buying on a random day.")
            elif diff > 0:
                _line = (f"The signal's 12-month median was {diff:+.1%} higher — a small "
                         f"edge, on just {tr.n_signals} samples with an enormous spread. "
                         "Not something to bet on.")
            else:
                _line = (f"The signal's 12-month median was actually {diff:+.1%} *lower* "
                         "than a random day.")
            st.markdown(
                f"12 months after a buy signal: median **{s12['median']:+.1%}**, up "
                f"**{s12['win']:.0%}** of the time. After a random day: "
                f"**{b12['median']:+.1%}**, up **{b12['win']:.0%}**. "
                f"At 3 months the signal was **{d3s['median']:+.1%}** vs "
                f"**{d3b['median']:+.1%}** for a random day — you often buy right "
                f"after a bounce that then wobbles. {_line}"
            )

        # ---- Invest now vs wait for a dip ----
        st.subheader("Invest now, or wait for a dip?")
        dc = st.columns(2)
        dip_pct = dc[0].slider("Wait for a dip of…", 5, 30, 10, 5, format="%d%%")
        dip = dip_pct / 100.0
        dh = dc[1].slider("Judge over the next… (years)", 1, 7, 3)
        dw = timing.dip_wait_study(_series, dip_pct=dip, horizon_years=float(dh))
        if dw.n_windows == 0:
            st.warning("Not enough history for that horizon.")
        else:
            m = st.columns(3)
            m[0].metric("Investing now beat waiting", f"{dw.now_better_share:.0%}",
                        help=f"across {dw.n_windows} historical start dates")
            m[1].metric("The dip never came", f"{dw.never_dipped_share:.0%}",
                        help=f"within {dh} years — you'd still be sitting in cash")
            m[2].metric("Avg time spent waiting", f"{dw.avg_wait_months:.0f} mo")

            _clip = np.clip(dw.diffs, -1.5, 3.0)
            fig = px.histogram(_clip, nbins=28)
            fig.add_vline(x=0, line=dict(color="#bbb", dash="dot"))
            fig.update_traces(marker_color="#2e86ab")
            fig.update_layout(
                height=240, showlegend=False, margin=dict(t=6, b=6),
                xaxis_title="growth-multiple advantage of investing now  ( >0 = now won )",
                yaxis_title="# of start dates")
            st.plotly_chart(fig, use_container_width=True)
            if dw.now_better_share >= 0.5:
                _msg = ("Investing now won more often — waiting mostly means sitting "
                        "in cash while the price drifts up, then buying higher anyway.")
            else:
                _msg = (f"For **{t_pick}**, waiting for a dip actually won more often — "
                        "it's an asset that has repeatedly fallen back, so dip-buyers "
                        "got rewarded. That's the exception: for a broad rising market "
                        "(try **SPY**) investing now usually wins. It depends on the asset.")
            st.caption(
                f"Median over {dh}y: {EUR2.m(1)} → **{EUR2.m(dw.median_now_multiple)}** "
                f"now vs **{EUR2.m(dw.median_wait_multiple)}** waiting for a "
                f"{dip_pct}% dip. " + _msg
            )

    st.divider()
    st.caption("Not advice. Historical, single-asset, before taxes and costs. "
               "For a long horizon, the honest move is usually to invest steadily "
               "(a bit each month) rather than to time an entry.")


# --------------------------------------------------------------------------- #
# Tab - Idea worksheet
# --------------------------------------------------------------------------- #
with tab_idea:
    st.caption(
        "A worksheet for a single-stock idea (e.g. an *Investment Idea Generation* "
        "brief). The tool auto-fills the **quantitative** context and turns it into "
        "risk flags and research questions — **you** write the pitch, the catalyst "
        "and the drivers. It has no fundamentals, filings or news."
    )

    _wl = [t for t in A.candidate_meta.index
           if str(A.candidate_meta.loc[t].get("source", "")) == "watchlist"] \
        if not A.candidate_meta.empty else []
    ic = st.columns([2, 1, 1])
    idea_ticker = ic[0].text_input(
        "Ticker", value=(_wl[0] if _wl else list(A.holdings_weights)[0]),
    ).upper().strip()
    stance = ic[1].radio("Stance", ["Long", "Short"], horizontal=True)
    horizon = ic[2].slider("Thesis horizon (yrs)", 1, 5, 1)
    name = st.text_input("Name / Student ID", placeholder="e.g. C. — s1234567")

    ctx = _idea_context(idea_ticker, horizon, rf) if idea_ticker else None
    if ctx is None:
        st.warning("No price data for that ticker — check the symbol.")
    else:
        g = st.columns(4)
        g[0].metric("Sector", ctx.sector)
        g[1].metric("Beta to market", f"{ctx.beta:.2f}" if ctx.beta == ctx.beta else "—",
                    help=H["Beta to market"])
        g[2].metric("Volatility / yr", f"{ctx.ann_vol:.0%}", help=H["Annualized volatility"])
        g[3].metric("Worst drawdown", f"{ctx.max_drawdown:.0%}", help=H["Max drawdown"])
        g = st.columns(4)
        g[0].metric("Trend", "above 200-day" if ctx.above_200 else "below 200-day")
        g[1].metric("From 12-mo high", f"{ctx.pct_from_52w_high:+.0%}")
        g[2].metric(f"{horizon}y scenario midpoint", f"{ctx.scen_p50:+.0%}")
        g[3].metric(f"{horizon}y range (10–90%)", f"{ctx.scen_p10:+.0%} … {ctx.scen_p90:+.0%}")

        st.markdown("#### Your inputs — the parts the tool can't do")
        concept = st.text_area(
            "1. Core concept — what it does & why it's attractive now (3–4 sentences)",
            key="idea_concept", height=90)
        catalyst = st.text_area(
            "2. Catalyst — the recent event / trend / mispricing that triggered this",
            key="idea_catalyst", height=70)
        dcx = st.columns(2)
        d1 = dcx[0].text_area("3. Driver 1", key="idea_d1", height=70)
        d2 = dcx[1].text_area("3. Driver 2", key="idea_d2", height=70)

        flags = ideacard.risk_flags(ctx, stance)
        questions = ideacard.research_questions(ctx, stance)

        st.markdown("#### Auto-generated risk flags")
        for _x in flags:
            st.markdown(f"- {_x}")
        st.markdown("#### Suggested research priorities")
        for _x in questions:
            st.markdown(f"- {_x}")

        filled = ideacard.render_template(
            name, stance, ctx, concept, catalyst, d1, d2, flags, questions
        )
        st.download_button(
            "⬇ Download the filled template (Markdown)",
            filled, file_name=f"idea_{idea_ticker or 'draft'}.md", mime="text/markdown",
            type="primary",
        )
        with st.expander("Preview the filled template"):
            st.markdown(filled)

    st.divider()
    st.caption("Not investment advice. The quantitative context is historical and "
               "illustrative — it does not forecast returns or tell you whether the "
               "idea is good. That's your analysis to make.")


# --------------------------------------------------------------------------- #
# Tab 6 - Summary
# --------------------------------------------------------------------------- #
with tab_summary:
    st.subheader("Top suggested additions")
    if A.ranked.empty:
        st.write("Run an analysis with a candidate universe to see suggestions.")
    else:
        for t, row in A.ranked.head(6).iterrows():
            name = row.get("name", t)
            st.markdown(
                f"**{t} — {name}**  ·  fit score {row['fit_score']:.0f}/100  "
                f"·  corr {row['corr_to_portfolio']:.2f}  "
                f"·  vol Δ {row['vol_delta'] * 100:+.1f} pp  "
                f"·  Sharpe Δ {row['sharpe_delta']:+.3f}"
            )
            st.caption(row.get("rationale", ""))

    st.divider()
    st.markdown(
        """
        ### How to read this

        - **Fit score** blends four things: low correlation to your current
          book, volatility reduction, Sharpe improvement, and a higher
          diversification ratio. It is relative to the other candidates in this
          run, not an absolute quality rating.
        - **Correlation** is measured over your chosen lookback. Correlations
          move — especially in a crisis, when they tend toward 1.
        - The **optimizers** disagree on purpose. MPT chases the historical
          risk/return trade-off (and is sensitive to the return estimate);
          risk parity and HRP ignore return estimates and balance risk;
          Black-Litterman starts from what market caps imply and nudges toward
          your views.
        - Expected returns are the weak link. If two methods broadly agree on a
          name, that is more informative than any single number.

        *This tool is for building and testing your own process. It is not
        investment advice and does not account for taxes, transaction costs,
        liquidity, or your personal circumstances.*
        """
    )


# --------------------------------------------------------------------------- #
# Tab 5 - Learn
# --------------------------------------------------------------------------- #
with tab_learn:
    st.info(
        "This is general background to make the numbers meaningful — **not "
        "advice**. This tool doesn't know your goals, income, debts, taxes or "
        "time horizon, and it can't tell you what to buy."
    )

    learn_a, learn_b, learn_c = st.tabs(
        ["Starting from cash", "Glossary", "What each method assumes"]
    )

    with learn_a:
        st.markdown(education.GETTING_STARTED)
        st.divider()
        st.markdown(education.HOW_TO_USE_FROM_CASH)

    with learn_b:
        q = st.text_input("Filter terms", key="glossary_filter").strip().lower()
        st.markdown("#### Metrics on the dashboard")
        for term, text in education.METRIC_HELP.items():
            if q and q not in term.lower() and q not in text.lower():
                continue
            st.markdown(f"**{term}** — {text}")
        st.markdown("#### General terms")
        for term, text in education.GLOSSARY.items():
            if q and q not in term.lower() and q not in text.lower():
                continue
            st.markdown(f"**{term}** — {text}")

    with learn_c:
        for method, notes in education.METHOD_NOTES.items():
            st.markdown(f"### {method}")
            st.markdown(f"- **What it does:** {notes['does']}")
            st.markdown(f"- **What it assumes:** {notes['assumes']}")
            st.markdown(f"- **Where it breaks down:** {notes['breaks']}")
        st.caption(
            "The optimizers disagree by design. When several land on the same "
            "name, that agreement means more than any single number."
        )
