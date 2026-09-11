"""Portfolio suggestion dashboard.

Run with:  streamlit run app.py

Not investment advice. This is a decision-support tool that describes what
would have been efficient given historical data and explicit assumptions.
"""

from __future__ import annotations

import streamlit as st

from pa import fx, store
from ui import common as C
from ui.analyze import render_analyze
from ui.common import EUR, EUR2, UNSAVED
from ui.explore import render_explore
from ui.practice import render_paper
from ui.research import render_research

st.set_page_config(page_title="Portfolio Suggestions", page_icon="📊", layout="wide")

store.init()


# --------------------------------------------------------------------------- #
# Session defaults shared across modes
# --------------------------------------------------------------------------- #
# `profile` / `profile_nonce` are a single cross-mode identity (🎮 Practice and
# the 📈 Analyze "saved portfolios" picker both read/write them), so these
# defaults must be set before *either* mode dispatches — hence living here
# rather than in ui/practice.py or ui/analyze.py. The Analyze-only keys ride
# along in the same dict; setting them unconditionally is cheap.
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
# Every ui.* mode module reads the account currency as C.ACCOUNT_CCY
# (module-qualified) rather than importing the name directly, since — unlike
# EUR/EUR2, which are only ever mutated in place — this one is reassigned
# each rerun and a plain `from ui.common import ACCOUNT_CCY` would freeze a
# stale copy at import time.
C.ACCOUNT_CCY = ACCOUNT_CCY
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

render_analyze()
