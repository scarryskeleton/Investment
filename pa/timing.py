"""Timing & trend context - deliberately honest about what timing can and can't do.

Three pieces:
  * trend_context  - where a price sits vs its moving averages / recent highs
  * trend_follow_backtest - the classic "hold above the 200-day average, sell
    below" rule vs simply buying and holding
  * dip_wait_study  - investing a lump sum now vs sitting in cash waiting for a dip

None of this is a buy signal. Trend-following historically reduced drawdowns
rather than raising returns, and waiting for a dip usually cost more than it saved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .explore import _block_bootstrap_paths
from .metrics import TRADING_DAYS


# --------------------------------------------------------------------------- #
# 1. Trend context
# --------------------------------------------------------------------------- #
@dataclass
class TrendContext:
    ticker: str
    last_price: float
    sma50: float
    sma200: float
    above_200: bool
    regime_trading_days: int
    pct_from_52w_high: float
    pct_from_high: float
    current_drawdown: float
    price: pd.Series
    sma50_series: pd.Series
    sma200_series: pd.Series
    drawdown_series: pd.Series
    status: str


def trend_context(ticker: str, prices: pd.Series) -> TrendContext:
    prices = prices.dropna()
    sma50 = prices.rolling(50).mean()
    sma200 = prices.rolling(200).mean()
    last = float(prices.iloc[-1])

    above = last > float(sma200.iloc[-1]) if sma200.notna().iloc[-1] else last > prices.mean()
    flags = (prices > sma200).dropna()
    regime = 0
    if not flags.empty:
        cur = flags.iloc[-1]
        for v in flags.iloc[::-1]:
            if v == cur:
                regime += 1
            else:
                break

    win52 = prices.iloc[-TRADING_DAYS:]
    dd = prices / prices.cummax() - 1.0

    pct_52w = last / float(win52.max()) - 1.0
    pct_high = last / float(prices.max()) - 1.0
    cur_dd = float(dd.iloc[-1])

    months = round(regime / 21.0)
    _mtxt = "about a month" if months == 1 else f"~{months} months"
    parts = [
        f"{ticker} is trading **{'above' if above else 'below'} its 200-day average**",
        f"and has been for {_mtxt}." if regime else ".",
    ]
    if pct_52w < -0.01:
        parts.append(f"It's **{abs(pct_52w):.0%} below** its 12-month high")
    else:
        parts.append("It's **at or near** its 12-month high")
    if cur_dd < -0.02:
        parts.append(f"and **{abs(cur_dd):.0%} below** its all-time high in this window.")
    else:
        parts.append("and near its all-time high in this window.")
    status = " ".join(parts)

    return TrendContext(
        ticker=ticker,
        last_price=last,
        sma50=float(sma50.iloc[-1]) if sma50.notna().iloc[-1] else float("nan"),
        sma200=float(sma200.iloc[-1]) if sma200.notna().iloc[-1] else float("nan"),
        above_200=bool(above),
        regime_trading_days=int(regime),
        pct_from_52w_high=pct_52w,
        pct_from_high=pct_high,
        current_drawdown=cur_dd,
        price=prices,
        sma50_series=sma50,
        sma200_series=sma200,
        drawdown_series=dd,
        status=status,
    )


# --------------------------------------------------------------------------- #
# 2. 200-day trend-following rule vs buy & hold
# --------------------------------------------------------------------------- #
@dataclass
class TrendFollowBacktest:
    slow: int
    ann_return_rule: float
    ann_return_bh: float
    vol_rule: float
    vol_bh: float
    maxdd_rule: float
    maxdd_bh: float
    n_trades: int
    time_in_market: float
    missed_while_out: float
    equity_rule: pd.Series
    equity_bh: pd.Series


def _ann_ret(equity: pd.Series) -> float:
    if len(equity) < 2:
        return float("nan")
    return float(equity.iloc[-1]) ** (TRADING_DAYS / len(equity)) - 1.0


def _maxdd(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def trend_follow_backtest(
    prices: pd.Series, slow: int = 200, cost: float = 0.0005
) -> TrendFollowBacktest:
    prices = prices.dropna()
    ret = prices.pct_change().fillna(0.0)
    sma = prices.rolling(slow).mean()
    signal = (prices > sma).astype(float)
    tradeable = signal.shift(1).fillna(0.0)  # act on yesterday's close

    mask = sma.notna()
    ret, tradeable = ret[mask], tradeable[mask]

    switches = tradeable.diff().abs().fillna(0.0) > 0
    strat_ret = tradeable * ret - switches.astype(float) * cost
    eq_rule = (1 + strat_ret).cumprod()
    eq_bh = (1 + ret).cumprod()

    missed = float((1 + ret[tradeable == 0]).prod() - 1.0)

    return TrendFollowBacktest(
        slow=slow,
        ann_return_rule=_ann_ret(eq_rule),
        ann_return_bh=_ann_ret(eq_bh),
        vol_rule=float(strat_ret.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        vol_bh=float(ret.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        maxdd_rule=_maxdd(eq_rule),
        maxdd_bh=_maxdd(eq_bh),
        n_trades=int(switches.sum()),
        time_in_market=float(tradeable.mean()),
        missed_while_out=missed,
        equity_rule=eq_rule,
        equity_bh=eq_bh,
    )


# --------------------------------------------------------------------------- #
# 1b. Mechanical buy/sell markers (50/200-day crossover) + a projection cone
# --------------------------------------------------------------------------- #
@dataclass
class CrossoverSignals:
    fast: int
    slow: int
    buy_dates: list
    buy_prices: list
    sell_dates: list
    sell_prices: list
    currently_in: bool
    last_signal: str          # "BUY" / "SELL"
    last_signal_date: object
    last_signal_price: float


def crossover_signals(prices: pd.Series, fast: int = 50, slow: int = 200) -> CrossoverSignals:
    """The textbook 'golden cross / death cross': the fast average crossing
    **up** through the slow one is a buy, crossing down is a sell. A mechanical
    rule - it reacts to trend changes after the fact, it does not predict them.
    """
    prices = prices.dropna()
    f = prices.rolling(fast).mean()
    s = prices.rolling(slow).mean()
    spread = (f - s).dropna()
    sign = np.sign(spread)
    flip = sign.diff().fillna(0.0)

    buys = spread.index[flip > 0]
    sells = spread.index[flip < 0]

    events = [(d, "BUY") for d in buys] + [(d, "SELL") for d in sells]
    events.sort()
    last_d, last_k = events[-1] if events else (prices.index[-1], "—")

    return CrossoverSignals(
        fast=fast, slow=slow,
        buy_dates=list(buys), buy_prices=[float(prices.loc[d]) for d in buys],
        sell_dates=list(sells), sell_prices=[float(prices.loc[d]) for d in sells],
        currently_in=bool(sign.iloc[-1] > 0),
        last_signal=last_k,
        last_signal_date=last_d,
        last_signal_price=float(prices.loc[last_d]) if last_k != "—" else float("nan"),
    )


def projection_cone(
    prices: pd.Series, months: int = 12, n_paths: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """Where the price could plausibly go, by resampling its own daily history
    forward. Returns a DataFrame indexed by future business days with p10/p25/
    p50/p75/p90 price levels. NOT a forecast - the band widens fast and the real
    price often finishes outside it.
    """
    r = prices.dropna().pct_change().dropna().to_numpy()
    last = float(prices.dropna().iloc[-1])
    h = max(5, months * 21)
    rng = np.random.default_rng(seed)
    paths = _block_bootstrap_paths(r, h, n_paths, 10, rng)
    levels = last * np.cumprod(1 + paths, axis=1)
    idx = pd.bdate_range(prices.dropna().index[-1], periods=h + 1, inclusive="right")
    return pd.DataFrame(
        {f"p{q}": np.percentile(levels, q, axis=0) for q in (10, 25, 50, 75, 90)},
        index=idx,
    )


# --------------------------------------------------------------------------- #
# 2b. Track record of the "buy" signal
# --------------------------------------------------------------------------- #
def _dist(a: np.ndarray) -> dict:
    if len(a) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "win": float("nan")}
    return {
        "n": int(len(a)),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "win": float((a > 0).mean()),
    }


@dataclass
class SignalTrackRecord:
    slow: int
    n_signals: int
    signal_dates: list
    after_signal: dict          # horizon-months -> _dist
    after_any_day: dict         # horizon-months -> _dist
    fwd_signal: dict            # horizon-months -> np.ndarray of forward returns
    fwd_any: dict


def signal_track_record(
    prices: pd.Series, slow: int = 200, horizons_months=(3, 6, 12)
) -> SignalTrackRecord:
    """For every 'buy signal' (price closes back above its 200-day average),
    what did the next 3 / 6 / 12 months look like - next to the same horizons
    measured from a *randomly chosen* day. If the signal predicted anything, the
    two distributions would differ.
    """
    prices = prices.dropna()
    p = prices.to_numpy()
    sma = prices.rolling(slow).mean().to_numpy()
    valid = ~np.isnan(sma)
    above = np.zeros(len(p), dtype=bool)
    above[valid] = p[valid] > sma[valid]

    cross_up = np.zeros(len(p), dtype=bool)
    cross_up[1:] = above[1:] & ~above[:-1] & valid[1:]
    sig_idx = np.where(cross_up)[0]
    any_idx = np.where(valid)[0]

    after_sig, after_any, fwd_sig, fwd_any = {}, {}, {}, {}
    for hm in horizons_months:
        h = hm * 21

        def _fwd(idxs):
            idxs = idxs[idxs + h < len(p)]
            return p[idxs + h] / p[idxs] - 1.0 if len(idxs) else np.array([])

        fs, fa = _fwd(sig_idx), _fwd(any_idx)
        fwd_sig[hm], fwd_any[hm] = fs, fa
        after_sig[hm], after_any[hm] = _dist(fs), _dist(fa)

    return SignalTrackRecord(
        slow=slow,
        n_signals=int(len(sig_idx)),
        signal_dates=[prices.index[i] for i in sig_idx],
        after_signal=after_sig,
        after_any_day=after_any,
        fwd_signal=fwd_sig,
        fwd_any=fwd_any,
    )


# --------------------------------------------------------------------------- #
# 3. Invest now vs wait for a dip
# --------------------------------------------------------------------------- #
@dataclass
class DipWaitStudy:
    dip_pct: float
    horizon_years: float
    n_windows: int
    now_better_share: float
    median_now_multiple: float
    median_wait_multiple: float
    avg_wait_months: float
    never_dipped_share: float
    diffs: np.ndarray  # now_multiple - wait_multiple, per window


def dip_wait_study(
    prices: pd.Series,
    dip_pct: float = 0.10,
    horizon_years: float = 3.0,
    cash_rate: float = 0.03,
    step_days: int = 21,
) -> DipWaitStudy:
    p_all = prices.dropna().to_numpy()
    H = int(round(horizon_years * TRADING_DAYS))
    if len(p_all) < H + step_days:
        return DipWaitStudy(dip_pct, horizon_years, 0, float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), np.array([]))

    daily_cash = (1 + cash_rate) ** (1 / TRADING_DAYS) - 1.0
    now_m, wait_m, wait_d, never = [], [], [], 0

    for s in range(0, len(p_all) - H, step_days):
        seg = p_all[s : s + H + 1]
        p0 = seg[0]
        now_m.append(seg[-1] / p0)

        target = p0 * (1 - dip_pct)
        hit = np.where(seg <= target)[0]
        if len(hit) == 0:
            never += 1
            wait_m.append((1 + daily_cash) ** H)      # never invested - just cash
            wait_d.append(H)
        else:
            d = int(hit[0])
            wait_m.append((1 + daily_cash) ** d * (seg[-1] / seg[d]))
            wait_d.append(d)

    now_m = np.array(now_m)
    wait_m = np.array(wait_m)
    return DipWaitStudy(
        dip_pct=dip_pct,
        horizon_years=horizon_years,
        n_windows=len(now_m),
        now_better_share=float((now_m >= wait_m).mean()),
        median_now_multiple=float(np.median(now_m)),
        median_wait_multiple=float(np.median(wait_m)),
        avg_wait_months=float(np.mean(wait_d) / 21.0),
        never_dipped_share=float(never / len(now_m)),
        diffs=now_m - wait_m,
    )
