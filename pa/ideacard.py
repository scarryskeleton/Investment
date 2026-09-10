"""Single-stock idea worksheet - scaffolding for an 'Investment Idea Generation'
style brief.

The tool auto-fills the *quantitative* context (trend, volatility, beta,
drawdown history, a scenario range, where the textbook return estimates
disagree) and turns the numbers into risk flags and research questions. It does
**not** write the pitch or the catalyst - it has no fundamentals, filings or
news. Those fields stay with the student.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass
class IdeaContext:
    ticker: str
    sector: str
    industry: str
    market_cap: float
    last_price: float
    ann_return: float
    ann_vol: float
    beta: float
    max_drawdown: float
    worst_12m: float
    above_200: bool
    pct_from_52w_high: float
    current_drawdown: float
    est_hist_mean: float
    est_capm: float
    est_ema: float
    est_spread: float
    scen_p10: float
    scen_p50: float
    scen_p90: float
    horizon_years: float
    history_months: int


def build_context(
    ticker: str,
    prices,
    market_prices,
    fundamentals_row: dict | None = None,
    rf: float = 0.03,
    horizon_years: float = 1.0,
) -> IdeaContext:
    from . import forecast, metrics, timing

    o = forecast.asset_outlook(
        ticker, prices, market_prices, rf=rf, horizon_years=horizon_years, method="bootstrap"
    )
    tc = timing.trend_context(ticker, prices)
    daily = metrics.daily_returns(prices.dropna())
    mkt_daily = metrics.daily_returns(market_prices.dropna())
    fr = fundamentals_row or {}

    return IdeaContext(
        ticker=ticker,
        sector=fr.get("sector") or "Unknown",
        industry=fr.get("industry") or "Unknown",
        market_cap=float(fr.get("market_cap") or float("nan")),
        last_price=tc.last_price,
        ann_return=metrics.annualized_return(daily),
        ann_vol=o.ann_vol,
        beta=metrics.beta_to_market(daily, mkt_daily),
        max_drawdown=metrics.max_drawdown(daily),
        worst_12m=o.hist_worst_12m,
        above_200=tc.above_200,
        pct_from_52w_high=tc.pct_from_52w_high,
        current_drawdown=tc.current_drawdown,
        est_hist_mean=o.est_hist_mean,
        est_capm=o.est_capm,
        est_ema=o.est_ema,
        est_spread=o.est_spread,
        scen_p10=o.p10,
        scen_p50=o.p50,
        scen_p90=o.p90,
        horizon_years=horizon_years,
        history_months=o.n_months if hasattr(o, "n_months") else 0,
    )


def _pct(x, d=0):
    return f"{x * 100:+.{d}f}%" if x == x else "n/a"


def risk_flags(ctx: IdeaContext, stance: str) -> list[str]:
    f: list[str] = []
    b = ctx.beta
    if b == b and b > 1.2:
        f.append(f"High beta ({b:.1f}): the idea is partly a bet on the whole market "
                 "going up. Check how the name behaves in a broad sell-off.")
    elif b == b and b < 0.7:
        f.append(f"Low beta ({b:.1f}): it moves fairly independently of the market, so "
                 "the thesis rests on company-specific factors — those need to be solid.")
    if ctx.max_drawdown == ctx.max_drawdown and ctx.max_drawdown < -0.4:
        f.append(f"Has dropped {_pct(ctx.max_drawdown)} peak-to-trough before — size the "
                 "position for that happening again.")
    if ctx.pct_from_52w_high < -0.2:
        f.append(f"Trading {_pct(ctx.pct_from_52w_high)} from its 12-month high — is this a "
                 "genuine discount or a broken story? The priority is finding out *why* it fell.")
    elif ctx.pct_from_52w_high > -0.03:
        f.append("Near its 12-month high — for a long, make sure you're not just chasing "
                 "momentum; for a short, the price trend is against you.")
    if ctx.est_spread > 0.15:
        f.append(f"The textbook return estimates disagree by ~{ctx.est_spread * 100:.0f} "
                 "percentage points — the expected return is essentially unknown from price "
                 "history alone. Anchor it with analyst estimates and company guidance.")
    if ctx.ann_vol == ctx.ann_vol and ctx.ann_vol > 0.4:
        f.append(f"Very volatile ({ctx.ann_vol:.0%}/yr): the {ctx.horizon_years:g}-year "
                 f"scenario band runs {_pct(ctx.scen_p10)} to {_pct(ctx.scen_p90)} — a huge "
                 "range of outcomes.")
    if stance == "Short":
        f.append("Short-specific: borrow cost and availability, short interest / squeeze "
                 "risk, and the asymmetric payoff (gain capped at 100%, loss unbounded).")
    if not f:
        f.append("No loud statistical warnings — but low volatility is not low risk. The "
                 "real risks here are fundamental (see the research questions).")
    return f


def research_questions(ctx: IdeaContext, stance: str) -> list[str]:
    q = [
        "Valuation vs peers and vs its own 5-year history (P/E, EV/EBITDA, P/S) — is the "
        "entry price actually attractive, or just lower than it was?",
        "Revenue and earnings growth trend, plus the most recent management guidance — does "
        "it support the thesis?",
        "Balance sheet: debt load, interest coverage, cash runway.",
        "Competitive position — the one or two things that could break the moat.",
    ]
    if ctx.pct_from_52w_high < -0.1:
        q.append(f"What specifically drove the {_pct(ctx.pct_from_52w_high)} move off the "
                 "12-month high — a miss, a downgrade, a sector rotation, macro?")
    else:
        q.append("What is the market pricing in that you think is wrong?")
    if ctx.est_spread > 0.15:
        q.append("Reconcile the return estimates: what do sell-side price targets and the "
                 "company's own outlook imply for the next 12 months?")
    if stance == "Short":
        q.append("What is the catalyst and the timeline for the thesis to play out, and what "
                 "would force you to cover?")
    return q


_STANCE_BOX = {"Long": "[x] Long    [ ] Short", "Short": "[ ] Long    [x] Short"}


def render_template(
    name: str,
    stance: str,
    ctx: IdeaContext,
    concept: str,
    catalyst: str,
    driver1: str,
    driver2: str,
    flags: list[str],
    questions: list[str],
) -> str:
    def _f(s, fallback):
        return s.strip() if s and s.strip() else f"_{fallback}_"

    if ctx.market_cap == ctx.market_cap and ctx.market_cap > 0:
        mc = ctx.market_cap
        cap = f"${mc / 1e12:.2f}T" if mc >= 1e12 else f"${mc / 1e9:.1f}B"
    else:
        cap = "n/a"
    trend = "above" if ctx.above_200 else "below"

    return f"""# Investment Idea Generation Template

**Name / Student ID:** {_f(name, 'your name / ID')}
**Proposed Stance:** {_STANCE_BOX.get(stance, stance)}
**Target Asset & Ticker:** {ctx.ticker}
**Sector / Industry:** {ctx.sector} / {ctx.industry}

## 1. Core Concept (The Elevator Pitch)
{_f(concept, 'What does the company do, and why is it attractive right now? 3-4 sentences.')}

## 2. The Catalyst or Spark (Why Now?)
{_f(catalyst, 'What recent event, industry trend, or market mispricing triggered this idea?')}

## 3. Key Drivers & Tailwinds
- **Driver 1:** {_f(driver1, 'primary factor supporting growth or valuation upside')}
- **Driver 2:** {_f(driver2, 'primary factor supporting growth or valuation upside')}

## 4. Initial Risks & Questions for Additional Research
- **Immediate Risk:** {flags[0] if flags else '_[main threat to the thesis]_'}
- **Research Priority:** {questions[0] if questions else '_[data or metric to verify first]_'}

---

### Quantitative context (auto-filled from price history — *not* a forecast)
| | |
|---|---|
| Last price | {ctx.last_price:,.2f} |
| Market cap | {cap} |
| Beta to market | {ctx.beta:.2f} |
| Volatility | {ctx.ann_vol:.0%} / yr |
| Historical return | {_pct(ctx.ann_return)} / yr |
| Worst drawdown (in window) | {_pct(ctx.max_drawdown)} |
| Worst rolling 12 months | {_pct(ctx.worst_12m)} |
| Trend | {trend} its 200-day average; {_pct(ctx.pct_from_52w_high)} from 12-month high |
| Return estimates (hist / CAPM / EMA) | {_pct(ctx.est_hist_mean)} / {_pct(ctx.est_capm)} / {_pct(ctx.est_ema)} |
| {ctx.horizon_years:g}-year scenario range (10th–90th pct) | {_pct(ctx.scen_p10)} to {_pct(ctx.scen_p90)}, midpoint {_pct(ctx.scen_p50)} |

Source: Yahoo Finance daily prices via `yfinance`. Historical and illustrative
only — this is not investment advice and does not forecast returns.

### All risk flags
{chr(10).join(f'- {x}' for x in flags)}

### All research questions
{chr(10).join(f'- {x}' for x in questions)}

*Generated {date.today().isoformat()} with the Portfolio dashboard. The pitch,
catalyst and drivers are the author's own; the tool supplies only the
quantitative context and prompts.*
"""
