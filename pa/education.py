"""Plain-language explanations for the dashboard's metrics, terms and methods.

Nothing here is advice. It is background so the numbers on the screen mean
something to a reader who is new to investing.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# One-line helper text for the metric tiles (used as Streamlit `help=`)
# --------------------------------------------------------------------------- #
METRIC_HELP: dict[str, str] = {
    "Annualized return": (
        "The average yearly growth rate of this portfolio over the lookback "
        "window, if you compounded it. Past performance — not a prediction."
    ),
    "Annualized volatility": (
        "How much the portfolio's value bounced around per year (standard "
        "deviation of returns). Higher = a rougher ride. ~15% is roughly a "
        "broad stock market; cash is near 0%."
    ),
    "Sharpe ratio": (
        "Return above the risk-free rate, divided by volatility — i.e. how "
        "much reward you got per unit of risk. Higher is better. Under ~0.5 is "
        "weak, ~1 is decent, above ~1.5 is strong (and often too good to last)."
    ),
    "Sortino ratio": (
        "Like Sharpe, but only counts downside wobble as 'risk'. Rewards "
        "portfolios whose volatility is mostly to the upside."
    ),
    "Beta to market": (
        "How much the portfolio moves when the market (SPY) moves. 1.0 = moves "
        "with the market; 1.3 = tends to rise/fall ~30% more; below 1 = calmer "
        "than the market."
    ),
    "Max drawdown": (
        "The worst peak-to-trough drop over the window — the deepest paper "
        "loss you'd have had to sit through. Ask yourself honestly whether you "
        "could hold through a drop that size."
    ),
    "Diversification ratio": (
        "Weighted-average volatility of the holdings divided by the "
        "portfolio's actual volatility. 1.0 = no diversification benefit; "
        "higher means the holdings partly cancel each other out."
    ),
    "Effective # holdings": (
        "How many *equally-weighted* positions would give the same "
        "concentration you actually have. If you hold 8 names but one is huge, "
        "this might say 3 — you're less diversified than the count suggests."
    ),
    "Fit score": (
        "A 0–100 blend of: low correlation to your current portfolio, "
        "volatility reduction, Sharpe improvement, and a better diversification "
        "ratio. It ranks candidates *against each other* in this run — it is "
        "not an absolute 'good investment' rating."
    ),
    "Correlation to portfolio": (
        "Do this candidate's returns move with your current holdings? +1 = "
        "lockstep (adds little diversification), 0 = unrelated, negative = "
        "tends to zig when your portfolio zags."
    ),
}

# --------------------------------------------------------------------------- #
# Glossary
# --------------------------------------------------------------------------- #
GLOSSARY: dict[str, str] = {
    "Asset class": (
        "A family of investments that behaves similarly: cash, bonds, stocks, "
        "real assets (property, gold, commodities). Spreading across classes is "
        "the first layer of diversification."
    ),
    "Diversification": (
        "Owning things that don't all move together, so a bad patch in one is "
        "cushioned by others. It reduces risk without necessarily reducing "
        "expected return — the closest thing to a free lunch in investing."
    ),
    "Index fund / ETF": (
        "A single fund that holds a whole basket of assets (e.g. every large US "
        "company) at very low cost. Buying one gives instant diversification. "
        "An ETF is an index fund that trades like a stock."
    ),
    "Expense ratio / fee": (
        "The percentage a fund charges you per year. 0.03%–0.20% is normal for "
        "broad index ETFs; 1%+ is expensive. Fees compound against you, so this "
        "matters a lot over decades."
    ),
    "Risk-free rate": (
        "What you can earn with (near) zero risk — short-term government bonds "
        "or a savings account. It's the benchmark every risky investment is "
        "measured against."
    ),
    "Time horizon": (
        "How long until you need the money. Longer horizons can ride out stock "
        "market drops; money needed within a few years usually shouldn't be at "
        "market risk at all."
    ),
    "Volatility / risk": (
        "How much an investment's value swings. Often used as shorthand for "
        "'risk', though the real risk is being forced to sell low or not "
        "meeting your goal."
    ),
    "Covariance matrix": (
        "A table of how every pair of assets moves together. It's the raw "
        "material every optimizer here uses to estimate portfolio risk."
    ),
    "Shrinkage (Ledoit-Wolf)": (
        "Raw historical estimates are noisy. Shrinkage pulls them toward a "
        "simpler, more stable structure — trading a little bias for a lot less "
        "noise. Used here for the covariance matrix."
    ),
    "Efficient frontier": (
        "The set of portfolios with the highest expected return for each level "
        "of risk. Anything below the curve is 'inefficient' — you could get "
        "more return for the same risk."
    ),
    "Rebalancing": (
        "Periodically trimming what's grown and topping up what's lagged to "
        "return to your target mix. Enforces 'sell high, buy low' mechanically."
    ),
    "Dollar-cost averaging": (
        "Investing a fixed amount on a schedule regardless of price, instead of "
        "all at once. Removes the pressure of timing and smooths your entry."
    ),
    "Monte Carlo simulation": (
        "Running a plan through thousands of possible futures instead of one, to "
        "see the range of outcomes. The 'futures' here come either from "
        "reshuffling real history or from a small trained model."
    ),
    "Mixture-density network": (
        "The 'neural generator' option: a tiny neural network that, given the "
        "last few months, outputs a *blend* of two bell curves for next month's "
        "return — one calm, one turbulent. Sampling it repeatedly produces "
        "market-like paths with quiet spells and occasional crashes. On ~15 "
        "years of data it's a teaching demo, not a better crystal ball."
    ),
    "Emergency fund": (
        "Cash set aside (often ~3–6 months of expenses) for surprises, kept "
        "safe and instantly accessible — separate from anything invested."
    ),
}

# --------------------------------------------------------------------------- #
# What each optimizer assumes and where it breaks
# --------------------------------------------------------------------------- #
METHOD_NOTES: dict[str, dict[str, str]] = {
    "Mean-Variance (MPT)": {
        "does": "Finds the mix on the efficient frontier that best trades "
                "expected return against risk (e.g. maximum Sharpe).",
        "assumes": "That your estimates of future return and risk are roughly "
                   "right, and that only mean and variance matter.",
        "breaks": "Expected-return estimates are very noisy, so it can pile "
                  "into whatever looked best recently. Shrinkage and a per-name "
                  "cap help, but treat its weights as one opinion, not truth.",
    },
    "Risk Parity": {
        "does": "Sizes positions so each contributes an equal share of "
                "portfolio risk.",
        "assumes": "Only that the risk estimates are reasonable — it ignores "
                   "expected returns entirely.",
        "breaks": "Can lean heavily on low-volatility assets (like bonds), and "
                  "real-world versions sometimes use leverage to compensate "
                  "(this one does not).",
    },
    "Hierarchical Risk Parity": {
        "does": "Groups assets into a tree by correlation, then splits risk "
                "down the branches.",
        "assumes": "That the correlation structure is meaningful and fairly "
                   "stable.",
        "breaks": "Correlations shift, especially in crises when many things "
                  "fall together and the tree you built no longer applies.",
    },
    "Black-Litterman": {
        "does": "Starts from the mix the whole market already holds (implied by "
                "market caps), then nudges it toward any views you enter.",
        "assumes": "That the market's aggregate positioning is a sensible "
                   "starting point and your views come with honest confidence "
                   "levels.",
        "breaks": "With no views it just echoes the market; with over-confident "
                  "views it swings hard toward them. It needs market-cap data, "
                  "which plain ETFs often don't report.",
    },
}

# --------------------------------------------------------------------------- #
# Getting-started background (markdown). General education, not advice.
# --------------------------------------------------------------------------- #
GETTING_STARTED = """
### Starting from cash — how people usually think about this

**This is general background, not advice for your situation. This tool is not a
financial adviser and doesn't know your goals, income, debts or taxes.**

#### 1. Before investing at all
Most beginner-focused guidance suggests, in rough order: clear high-interest
debt, build a small emergency fund in a plain savings account, and only invest
money you won't need for *at least* five years.

#### 2. What "good to invest in" actually depends on
There's no universal answer — it hinges on your **time horizon** (when you need
the money) and how large a temporary drop you could hold through without selling.
Money for next year and money for 20 years from now belong in very different
places.

#### 3. The common building blocks

| Block | Role | Rough yearly swing |
|---|---|---|
| Cash / savings / money-market | safety, spending money, emergency fund | ~0% |
| Government & high-grade bonds | ballast, income, cushions stock crashes | ~3–8% |
| Broad stock index funds (whole-market or whole-world) | long-term growth engine | ~15–20% |
| Real assets (gold, REITs, commodities) | situational diversifiers | varies, often high |

#### 4. Why broad index funds come up so often for beginners
- One fund can hold thousands of companies → instant diversification
- Fees are tiny (often under 0.20%/yr); high fees compound against you badly
- Over long periods, most professional stock-pickers fail to beat a cheap index
- Less to get wrong than choosing individual stocks

#### 5. The knobs that matter most
- Your **stock vs. bond split** — more stocks = more expected growth *and* more
  volatility. This single choice drives most of your outcome.
- **Keep fees low.**
- **Automate contributions** and add regularly rather than trying to time the market.
- **Don't panic-sell** in a downturn — the max-drawdown number on the Overview
  tab is there to help you pick a mix you could actually hold through.

#### 6. Things to be wary of
Hot tips, "can't lose" pitches, putting core savings into a single stock or a
single coin, leverage, and any product promising high return with low risk.

#### 7. Where to learn more (free, non-commercial)
- **Bogleheads wiki** — bogleheads.org/wiki (start with "Getting started")
- Your national regulator's investor-education pages. In the Netherlands: the
  **AFM** consumer site and **Wijzer in geldzaken** (wijzeringeldzaken.nl).
- Books often recommended for beginners: *The Psychology of Money* (Housel),
  *The Little Book of Common Sense Investing* (Bogle), *A Random Walk Down Wall
  Street* (Malkiel).

Region matters for **taxes and account types** (e.g. Dutch box 3 wealth tax,
accumulating vs. distributing UCITS ETFs) — check local rules before acting.
"""

def interpret_snapshot(snap, concentration: dict, top_sector: str | None = None) -> str:
    """A short plain-language read-out of a portfolio's own numbers."""
    bits = []

    vol = snap.ann_vol
    if vol != vol:  # nan
        return "Not enough data to interpret this portfolio."
    if vol < 0.08:
        bits.append(
            f"This is a **low-volatility** mix (~{vol:.0%} a year) — calmer than the "
            "stock market, closer to a bond-heavy portfolio."
        )
    elif vol < 0.16:
        bits.append(
            f"This mix swings about **{vol:.0%} a year**, roughly in line with a "
            "broad stock market."
        )
    else:
        bits.append(
            f"This is a **high-volatility** mix (~{vol:.0%} a year) — a bumpier ride "
            "than the overall market."
        )

    if snap.beta == snap.beta:
        if snap.beta > 1.15:
            bits.append(
                f"It tends to move about **{snap.beta:.1f}×** the market — amplifying "
                "both gains and losses."
            )
        elif snap.beta < 0.85:
            bits.append(
                f"It moves only about **{snap.beta:.1f}×** the market — steadier when "
                "the market drops."
            )

    if snap.max_drawdown == snap.max_drawdown:
        bits.append(
            f"Its worst stretch in this window was a **{snap.max_drawdown:.0%} drop**. "
            "Picture that happening to your money — could you hold on without selling?"
        )

    if snap.sharpe == snap.sharpe:
        if snap.sharpe < 0.4:
            bits.append(
                f"The Sharpe ratio ({snap.sharpe:.2f}) is on the low side — you were "
                "not paid much extra return for the risk taken."
            )
        elif snap.sharpe > 1.2:
            bits.append(
                f"The Sharpe ratio ({snap.sharpe:.2f}) looks strong — historically a "
                "good return for the risk. Strong numbers rarely repeat, so don't "
                "count on it."
            )

    eff = concentration.get("effective_n", float("nan"))
    top = concentration.get("top_weight", float("nan"))
    if eff == eff and eff < 4:
        bits.append(
            f"It is **concentrated**: behaves like only ~{eff:.0f} equally-sized "
            f"positions, with the largest at {top:.0%}. One bad name would hurt a lot."
        )
    if top_sector:
        bits.append(f"Most of the risk sits in **{top_sector}** — adding something "
                    "different would spread it out.")

    return " ".join(bits)


def interpret_mix(mix) -> str:
    """Plain-language read-out for the 'explore from cash' two-sleeve mix."""
    s = mix.stock_pct
    label = {20: "very cautious", 40: "cautious", 60: "balanced", 80: "adventurous",
             100: "all-in on stocks"}.get(s, "custom")
    out = [
        f"A **{s}/{100 - s} mix** ({s}% {mix.stock_ticker} stocks, {100 - s}% "
        f"{mix.bond_ticker} bonds) — a {label} allocation.",
        f"Over {mix.start:%b %Y}–{mix.end:%b %Y} it grew about **{mix.ann_return:.1%} "
        f"a year** on average, with year-to-year swings of ~{mix.ann_vol:.0%}.",
        f"Its worst 12-month stretch lost **{mix.worst_12m:.0%}**; the deepest drop "
        f"from a peak was **{mix.max_drawdown:.0%}**.",
    ]
    if s >= 80:
        out.append("Mostly stocks: highest expected growth, but expect drops of "
                   "30–50% every decade or so. Best suited to money you won't touch "
                   "for 10+ years.")
    elif s <= 40:
        out.append("Bond-heavy: much smoother, but lower expected growth — over long "
                   "horizons it may barely stay ahead of inflation.")
    else:
        out.append("A middle-of-the-road split: some growth, drops that are "
                   "uncomfortable but usually survivable.")
    out.append("*Historical, illustrative, not a forecast or a recommendation.*")
    return " ".join(out)


HOW_TO_USE_FROM_CASH = """
**Using this dashboard with no holdings yet**

You can still explore. In the sidebar set **Amounts are… → weight** and type a
hypothetical mix, for example:

```
VTI, 60
BND, 30
GLD, 10
```

Press **Analyze** and read the Overview tab: the **max drawdown** tells you the
worst drop that mix lived through, **annualized volatility** how bumpy it was,
and the **growth of $1** chart how it compared to the S&P 500. Try a few mixes
(more bonds, fewer bonds) and watch how the risk numbers move. It's a sandbox —
no real money involved.
"""
