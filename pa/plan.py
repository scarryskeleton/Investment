"""Turn the analysis into concrete action - drift, rebalancing trades, and how
to steer new contributions. No forecasting: this is arithmetic on a target mix.

Band rebalancing (act only when a holding drifts more than a few points, or a
large fraction, from target) is standard practice - it enforces "sell high, buy
low" mechanically without constant tinkering.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftRow:
    ticker: str
    current_weight: float
    target_weight: float
    current_value: float
    target_value: float
    drift_pp: float          # current - target, in weight points
    drift_rel: float         # (current - target) / target
    breach: bool
    trade_value: float       # + = buy this much, - = sell this much


@dataclass
class ActionPlan:
    total_value: float
    rows: list
    needs_rebalance: bool
    contribution: float
    contribution_split: dict
    band_pp: float
    band_rel: float
    notes: list = field(default_factory=list)

    @property
    def sells(self):
        return sorted((r for r in self.rows if r.breach and r.trade_value < -1),
                      key=lambda r: r.trade_value)

    @property
    def buys(self):
        return sorted((r for r in self.rows if r.breach and r.trade_value > 1),
                      key=lambda r: -r.trade_value)


def build_plan(
    current_weights: dict[str, float],
    target_weights: dict[str, float],
    total_value: float,
    band_pp: float = 0.05,
    band_rel: float = 0.25,
    contribution: float = 0.0,
) -> ActionPlan:
    tickers = sorted(set(current_weights) | set(target_weights))

    def _norm(d):
        s = sum(max(v, 0.0) for v in d.values())
        return {t: max(d.get(t, 0.0), 0.0) / s for t in tickers} if s else {t: 0.0 for t in tickers}

    cw, tw = _norm(current_weights), _norm(target_weights)

    rows = []
    for t in tickers:
        cv, tv = cw[t] * total_value, tw[t] * total_value
        dpp = cw[t] - tw[t]
        if tw[t] > 1e-9:
            drel = dpp / tw[t]
            breach = abs(dpp) > band_pp or abs(drel) > band_rel
        else:
            drel = np.inf if cw[t] > 1e-9 else 0.0
            breach = cw[t] > band_pp            # holding something not in the target
        if cw[t] < 1e-9 and tw[t] > band_pp:    # missing a target holding entirely
            breach = True
        rows.append(DriftRow(t, cw[t], tw[t], cv, tv, dpp, drel, breach, tv - cv))

    needs = any(r.breach for r in rows)

    # ---- steer new cash to the underweight names first (no selling) ----
    split = {t: 0.0 for t in tickers}
    if contribution > 0:
        grand = total_value + contribution
        gaps = {t: max(tw[t] * grand - cw[t] * total_value, 0.0) for t in tickers}
        gsum = sum(gaps.values())
        if 0 < gsum <= contribution:
            split = dict(gaps)
            rem = contribution - gsum
            for t in tickers:
                split[t] += tw[t] * rem
        elif gsum > 0:
            split = {t: gaps[t] * contribution / gsum for t in tickers}
        else:
            split = {t: tw[t] * contribution for t in tickers}

    notes = [f"Portfolio value ≈ €{total_value:,.0f}."]
    breaches = [r for r in rows if r.breach]
    if not breaches:
        notes.append("Every position is within its rebalancing band — no trades needed today.")
    else:
        worst = max(breaches, key=lambda r: abs(r.drift_pp))
        notes.append(
            f"{len(breaches)} position(s) sit outside the ±{band_pp:.0%}-point / "
            f"{band_rel:.0%}-relative band. Biggest gap: **{worst.ticker}** is "
            f"{worst.drift_pp:+.1%} vs its target."
        )
    if contribution > 0:
        notes.append(
            f"Putting your €{contribution:,.0f} into the underweight names moves you "
            "toward target **without selling** — which avoids realising any capital gains."
        )
    return ActionPlan(total_value, rows, needs, contribution, split, band_pp, band_rel, notes)


# --------------------------------------------------------------------------- #
# Target-mix builders
# --------------------------------------------------------------------------- #
def equal_weight(tickers) -> dict[str, float]:
    tickers = list(tickers)
    return {t: 1.0 / len(tickers) for t in tickers} if tickers else {}


def blend_targets(a: dict, b: dict, w: float = 0.5) -> dict:
    keys = set(a) | set(b)
    return {k: w * a.get(k, 0.0) + (1 - w) * b.get(k, 0.0) for k in keys}
