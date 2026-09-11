"""Mode: 📈 Analyze a portfolio — the full optimization workflow.

This is the one mode still shaped as a single large function rather than
having been split further (it was originally ~1,270 lines of top-level
script code, never wrapped in a function at all) — Portfolio overview,
Diversification scan, Optimizers, Action plan, Outlook, Timing check, the
Idea worksheet, Suggestion summary and the Learn tab all live here. A
further split (e.g. one helper per tab) is a reasonable next step but a
separate, more careful pass than the mechanical move that created this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pa import (
    education,
    forecast,
    ideacard,
    metrics,
    optimize,
    plan as planning,
    portfolio,
    store,
    suggest,
    timing,
)
from ui import common as C
from ui.common import EUR, EUR2, H, NEW_PROFILE, PCT, UNSAVED


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


def render_analyze() -> None:
    """Mode: the full holdings -> optimizers -> action plan workflow."""
    sb = st.sidebar
    if "editor_df" not in st.session_state:
        st.session_state.editor_df = _starter_df()

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
        latest = C._latest_prices(tuple(sorted(parsed.index))) if mode == "shares" else None
        if mode == "shares" and latest:
            _meta = C._origins(tuple(sorted(parsed.index)))
            latest = {t: v * C._fx_spot(_meta.get(t, {}).get("currency") or C.ACCOUNT_CCY,
                                        C.ACCOUNT_CCY)
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
                f"Your portfolio's total value ({C.ACCOUNT_CCY})", min_value=100.0,
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
            f"New money to add now ({C.ACCOUNT_CCY}) — for Option B", 0.0, 1e8, 0.0, 100.0,
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
