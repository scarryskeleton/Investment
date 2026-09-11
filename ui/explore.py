"""Mode: 🌱 Explore from cash — a beginner risk-slider + contribution simulator."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pa import education, explore
from ui import common as C

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


@st.cache_data(show_spinner=False, ttl=60 * 60 * 6)
def _explore_mix(stock_pct, stock_ticker, bond_ticker, account_ccy="EUR"):
    return explore.analyze_mix(stock_pct, stock_ticker, bond_ticker,
                               account_ccy=account_ccy)


@st.cache_data(show_spinner=False)
def _explore_projection(monthly_values, start_value, monthly_contribution, years, method):
    import pandas as pd

    s = pd.Series(monthly_values)
    return explore.project_contributions(
        s, start_value, monthly_contribution, years, method=method
    )


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
    start_value = sb.number_input(f"Starting amount ({C.ACCOUNT_CCY})", 0, 5_000_000, 1_000, 250)
    monthly_contribution = sb.number_input(f"Added every month ({C.ACCOUNT_CCY})", 0, 200_000,
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
        m = _explore_mix(stock_pct, stock_ticker, bond_ticker, C.ACCOUNT_CCY)
    except Exception as e:
        st.error(f"Couldn't build that mix ({e}). Try different funds in the sidebar.")
        return

    c = st.columns(4)
    c[0].metric("Historical return / yr", C.PCT.format(m.ann_return), help=C.H["Annualized return"])
    c[1].metric("Volatility / yr", C.PCT.format(m.ann_vol), help=C.H["Annualized volatility"])
    c[2].metric("Worst drop (peak→trough)", C.PCT.format(m.max_drawdown), help=C.H["Max drawdown"])
    c[3].metric("Worst 12-month stretch", C.PCT.format(m.worst_12m),
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
        st.subheader(f"Growth of {C.EUR.m(10_000)}")
        g = m.growth.copy()
        g.index.name = "date"
        fig = px.line(g)
        fig.update_layout(height=340, yaxis_title="", xaxis_title="", legend_title="",
                          margin=dict(t=10, b=10),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"{m.start:%b %Y} – {m.end:%b %Y}, in **{C.ACCOUNT_CCY}** "
                   "(funds converted at historical rates). The 'Cash / savings' line "
                   "assumes a steady 3%/yr — notice how inflation-era savings barely move.")

    st.divider()
    st.subheader(f"If you add {C.EUR.m(monthly_contribution)} every month for {years} years")

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
                      yaxis_title=f"portfolio value ({C.ACCOUNT_CCY})",
                      margin=dict(t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    d = st.columns(4)
    d[0].metric("You'd put in", C.EUR.format(proj.total_contributed))
    d[1].metric("Typical outcome", C.EUR.format(proj.end_median))
    d[2].metric("Unlucky (10th pct)", C.EUR.format(proj.end_low))
    d[3].metric("Lucky (90th pct)", C.EUR.format(proj.end_high))

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
        f"instead of **{proj.fee_low_pct:.2%}/yr** ends near {C.EUR.m(proj.fee_high_median)} "
        f"vs {C.EUR.m(proj.fee_low_median)} — about **{C.EUR.m(_gap)} lost to fees** "
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
