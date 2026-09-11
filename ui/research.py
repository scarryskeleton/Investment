"""Mode: 🔎 Research a stock or bond — search/browse, then a full write-up."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pa import fx
from ui import common as C


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _research_bundle(ticker, benchmark, lookback):
    from pa import data, forecast, research, timing

    px_ = data.fetch_prices([ticker, benchmark], lookback)
    if ticker not in px_.columns:
        return None
    prices = px_[ticker].dropna()
    prices.name = ticker
    bench = px_[benchmark].dropna() if benchmark in px_.columns else None
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
            pxu = C._universe_prices(tuple(sorted(set(uni + [benchmark]))), lookback)
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
        m[2].metric("Beta", f"{rel.beta:.2f}", help=C.H["Beta to market"])
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
        oc[0].metric("Volatility / yr", f"{o.ann_vol:.0%}", help=C.H["Annualized volatility"])
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
