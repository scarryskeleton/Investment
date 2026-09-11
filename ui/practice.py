"""Mode: 🎮 Practice portfolio — paper trading, leaderboard, trade journal."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from pa import fx, paper, store
from ui import common as C


def _intraday_chart(ticker: str) -> None:
    """A compact 'how it's been moving' line — hourly bars when Yahoo has
    them for this ticker, daily bars over a longer window otherwise."""
    hist, gran = C._intraday(ticker)
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
                 f"holding not quoted in {C.ACCOUNT_CCY}, on top of the currency "
                 "conversion already reflected in the price.",
        )
        if choice == "Custom…":
            fc = st.columns(3)
            m = paper.FeeModel(
                flat=fc[0].number_input(f"Flat {C.ACCOUNT_CCY}/trade", 0.0, 100.0, 0.0, 0.5,
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
                bits.append(f"{m.flat:g} {C.ACCOUNT_CCY} per trade")
            if m.rate_bps:
                bits.append(f"{m.rate_bps:g} bps commission")
            if m.fx_bps:
                bits.append(f"{m.fx_bps:g} bps FX fee on non-{C.ACCOUNT_CCY} names")
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
    origins = C._origins(tuple(held))
    ticker_ccy = {t: (origins.get(t, {}).get("currency") or stored_ccy) for t in held}

    tr = trades.copy()
    tr["ccy"] = [c or ticker_ccy.get(tk, stored_ccy) for c, tk in zip(tr["ccy"], tr["ticker"])]
    tr = _convert_trades(tr, stored_ccy)
    latest = {t: v * C._fx_spot(ticker_ccy.get(t, stored_ccy), stored_ccy)
             for t, v in C._latest_prices(tuple(held)).items()}
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
        r["value_here"] = r["total_value"] * C._fx_spot(r["currency"], acct_ccy)
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
        f"Value ({acct_ccy})": C.EUR.format(r["value_here"]),
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
    _opts = _profs + [C.NEW_PROFILE]
    _cur = st.session_state.profile if st.session_state.profile in _profs else None
    _idx = _profs.index(_cur) if _cur else (0 if _profs else len(_opts) - 1)
    _ch = sb.selectbox("Profile", _opts, index=_idx,
                       help="Everyone can see every profile's practice account. "
                            "Only its own password can change one.",
                       key=f"paper_prof_{st.session_state.profile_nonce}")

    st.title("🎮 Practice portfolio")

    if _ch == C.NEW_PROFILE:
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
    acc = store.practice_get_or_create(profile, currency=C.ACCOUNT_CCY)
    aid = acc["id"]
    acct_ccy = C.ACCOUNT_CCY
    stored_ccy = (acc["currency"] or "EUR").upper()
    # starting cash was typed in `stored_ccy`; show it in the active currency
    starting = float(acc["starting_cash"]) * C._fx_spot(stored_ccy, acct_ccy)

    trades_raw = store.practice_trades(aid)
    held = sorted(set(trades_raw["ticker"])) if not trades_raw.empty else []
    origins = C._origins(tuple(held))
    ticker_ccy = {t: (origins.get(t, {}).get("currency") or acct_ccy) for t in held}

    # fill in the native currency on any trade that predates the ccy column
    trades = trades_raw.copy()
    if not trades.empty:
        trades["ccy"] = [c or ticker_ccy.get(tk, acct_ccy)
                         for c, tk in zip(trades["ccy"], trades["ticker"])]
    trades = _convert_trades(trades, acct_ccy)

    latest_raw = C._latest_prices(tuple(held + ["SPY"])) if held else {}
    latest = {t: v * C._fx_spot("USD" if t == "SPY" else ticker_ccy.get(t, acct_ccy),
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
               f"**{C.EUR.m(starting)}**, all in **{acct_ccy}**.{_mix} "
               "A place to learn, not a broker.")
    if state.missing_prices:
        st.warning("No current price for: " + ", ".join(state.missing_prices)
                   + " — valued at cost for now.")

    fees = _paper_fee_model()

    # ---- headline ----
    ec = pd.DataFrame()
    if not trades.empty:
        hist_raw = C._universe_prices(tuple(sorted(set(held + ["SPY"]))), 3.0)
        hist = C._fx_convert_history(
            hist_raw, tuple({**ticker_ccy, "SPY": "USD"}.items()), acct_ccy)
        cols = [t for t in held if t in hist.columns]
        ec = paper.equity_curve(
            starting, trades, hist[cols] if cols else hist.iloc[:, :0],
            hist["SPY"] if "SPY" in hist.columns else None,
        )

    c = st.columns(4)
    c[0].metric("Total value", C.EUR.format(state.total_value),
                f"{state.total_pnl:+,.0f}  ({state.total_pnl_pct:+.1%})")
    c[1].metric("Cash to invest", C.EUR.format(state.cash))
    c[2].metric("In the market", C.EUR.format(state.invested))
    if not ec.empty and "All-in S&P 500" in ec.columns:
        spy_val = float(ec["All-in S&P 500"].iloc[-1])
        c[3].metric("If you'd bought only S&P 500", C.EUR.format(spy_val),
                    f"{spy_val - starting:+,.0f}",
                    help="Same starting cash, all in SPY on your first trade date.")
    else:
        c[3].metric("Realized P&L", C.EUR.format(state.realized_pnl),
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
        price_native = C._latest_prices((tk, "SPY")).get(tk)
    if tk and not price_native:
        st.warning(f"No price found for **{tk}** — check the ticker.")
    elif tk:
        origin = C._origins((tk,)).get(tk, {})
        cur = origin.get("currency", "")
        nat_code = fx.normalize(cur)[0]
        foreign = bool(nat_code) and nat_code != acct_ccy
        rate = C._fx_spot(cur or acct_ccy, acct_ccy)
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
        _origin_bit = (f" · {C._flag(origin.get('country',''))} "
                       f"{origin.get('country','')}" if origin.get("country") else "")
        _fee_bit = (f" · fee **{C.EUR2.m(fee)}**"
                    + (" *(incl. FX)*" if foreign and fees.fx_bps else "")) if fee else ""
        qc[1].markdown(
            f"&nbsp;\n\n**{tk}** at **{C.EUR.m(price_now)}**{_fx_bit}{_origin_bit} → "
            f"{side.lower()} **{shares:,.4f}** shares = **{C.EUR.m(value)}**{_fee_bit}"
            + (f" → **{C.EUR2.m(cash_out)}** {'out' if side == 'Buy' else 'in'}"
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
                st.error(f"Not enough cash — {C.EUR2.m(value)} + {C.EUR2.m(fee)} "
                         f"fee, you have {C.EUR2.m(state.cash)}.")
            elif side == "Sell" and (pos is None or shares > pos.shares + 1e-6):
                st.error(f"You only hold {pos.shares:,.4f} {tk}." if pos
                         else f"You don't hold any {tk}.")
            else:
                # store native price + its currency; fee back in native units so a
                # later currency switch reconverts both consistently
                store.practice_record_trade(
                    aid, side.lower(), tk, shares, price_native,
                    fee / rate if rate else fee, cur or acct_ccy)
                st.toast(f"{side} {shares:,.4f} {tk} @ {C.EUR.m(price_now)}"
                         + (f"  (fee {C.EUR2.m(fee)})" if fee else ""), icon="🎮")
                st.rerun()

    # ---- holdings ----
    if state.positions:
        st.subheader("Your holdings")
        hdf = pd.DataFrame([{
            "Ticker": p.ticker,
            "Market": p.sector or "—",
            "Origin": f"{C._flag(p.country)} {p.country or '—'}",
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
        d[0].metric("Unrealized P&L", C.EUR.format(sum(p.unrealized for p in state.positions)))
        d[1].metric("Realized P&L", C.EUR.format(state.realized_pnl))
        d[2].metric("Fees paid", C.EUR2.format(state.fees_paid),
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
                f"Since your first trade you're at **{C.EUR.m(you)}** vs "
                f"**{C.EUR.m(spy)}** if you'd just bought the S&P 500 — "
                f"**{C.EUR.m(abs(diff))} {'ahead' if diff >= 0 else 'behind'}**. "
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
