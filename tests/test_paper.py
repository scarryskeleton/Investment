"""Tests for pa.paper — the practice-portfolio replay/P&L math.

Pure functions, no network: every case here is built from synthetic trades
and price series, so it runs anywhere in well under a second.
"""

from __future__ import annotations

import unittest

import pandas as pd

from pa import paper


def _trades(*rows) -> pd.DataFrame:
    """rows of (ts, side, ticker, shares, price, fee, ccy)."""
    cols = ["ts", "side", "ticker", "shares", "price", "fee", "ccy"]
    df = pd.DataFrame(rows, columns=cols)
    df["ts"] = pd.to_datetime(df["ts"])
    return df


class FeeModelTests(unittest.TestCase):
    def test_no_fees_by_default(self):
        m = paper.FeeModel()
        self.assertFalse(m.active)
        self.assertEqual(m.fee(10_000, foreign=True), 0.0)

    def test_flat_plus_bps(self):
        m = paper.FeeModel(flat=2.0, rate_bps=10.0, fx_bps=25.0)
        self.assertTrue(m.active)
        # domestic: flat + rate_bps only
        self.assertAlmostEqual(m.fee(1_000, foreign=False), 2.0 + 1_000 * 10 / 1e4)
        # foreign: flat + rate_bps + fx_bps
        self.assertAlmostEqual(m.fee(1_000, foreign=True), 2.0 + 1_000 * 35 / 1e4)

    def test_fee_scales_with_notional_sign_agnostic(self):
        m = paper.FeeModel(rate_bps=25.0)
        self.assertAlmostEqual(m.fee(-500, foreign=False), m.fee(500, foreign=False))


class ComputeStateTests(unittest.TestCase):
    def test_empty_trades_returns_starting_cash(self):
        s = paper.compute_state(10_000.0, pd.DataFrame(), {})
        self.assertEqual(s.cash, 10_000.0)
        self.assertEqual(s.total_value, 10_000.0)
        self.assertEqual(s.total_pnl, 0.0)
        self.assertEqual(s.positions, [])

    def test_buy_then_partial_sell_matches_hand_calculation(self):
        # Same scenario verified by hand earlier: buy 10 @ 100 (fee 2.25),
        # sell 4 @ 120 (fee 2.12), mark the rest at 130.
        trades = _trades(
            ("2024-01-02", "buy", "AAPL", 10, 100.0, 2.25, "USD"),
            ("2024-02-02", "sell", "AAPL", 4, 120.0, 2.12, "USD"),
        )
        s = paper.compute_state(10_000.0, trades, {"AAPL": 130.0})

        self.assertAlmostEqual(s.fees_paid, 4.37)
        self.assertAlmostEqual(s.cash, 9_475.63, places=2)
        self.assertAlmostEqual(s.invested, 780.0)
        self.assertAlmostEqual(s.realized_pnl, 76.98, places=2)
        self.assertAlmostEqual(s.total_value, 10_255.63, places=2)
        self.assertAlmostEqual(s.total_pnl, 255.63, places=2)

        self.assertEqual(len(s.positions), 1)
        pos = s.positions[0]
        self.assertEqual(pos.ticker, "AAPL")
        self.assertAlmostEqual(pos.shares, 6.0)
        self.assertAlmostEqual(pos.avg_cost, 100.225, places=3)
        self.assertAlmostEqual(pos.market_value, 780.0)
        self.assertAlmostEqual(pos.weight, 780.0 / 10_255.63, places=6)

    def test_selling_more_than_held_is_clamped_not_negative(self):
        # _replay clamps a sell to whatever's actually held, so an
        # over-sell can't manufacture a short position or negative shares.
        trades = _trades(
            ("2024-01-01", "buy", "MSFT", 2, 300.0, 0.0, "USD"),
            ("2024-01-05", "sell", "MSFT", 5, 310.0, 0.0, "USD"),
        )
        s = paper.compute_state(1_000.0, trades, {"MSFT": 320.0})
        self.assertEqual(s.positions, [])  # fully closed, not negative
        self.assertGreater(s.cash, 1_000.0)  # sold 2 shares at a gain

    def test_missing_price_falls_back_to_cost_not_lost(self):
        trades = _trades(("2024-01-01", "buy", "ZZZZ", 3, 50.0, 0.0, "USD"))
        s = paper.compute_state(1_000.0, trades, {})  # no quote for ZZZZ
        self.assertIn("ZZZZ", s.missing_prices)
        self.assertEqual(len(s.positions), 1)
        self.assertAlmostEqual(s.positions[0].market_value, 150.0)  # valued at cost

    def test_multiple_holdings_weights_sum_to_one(self):
        trades = _trades(
            ("2024-01-01", "buy", "A", 10, 10.0, 0.0, "USD"),
            ("2024-01-01", "buy", "B", 5, 20.0, 0.0, "USD"),
        )
        s = paper.compute_state(1_000.0, trades, {"A": 10.0, "B": 20.0})
        self.assertAlmostEqual(sum(p.weight for p in s.positions), 200.0 / 1000.0)


class EquityCurveTests(unittest.TestCase):
    def test_tracks_holdings_value_day_by_day(self):
        trades = _trades(("2024-01-02", "buy", "X", 10, 10.0, 0.0, "USD"))
        idx = pd.date_range("2024-01-01", "2024-01-05", freq="D")
        hist = pd.DataFrame({"X": [10.0, 10.0, 11.0, 12.0, 13.0]}, index=idx)

        ec = paper.equity_curve(1_000.0, trades, hist)
        self.assertIn("Your portfolio", ec.columns)
        # before the trade: still all cash
        self.assertAlmostEqual(ec["Your portfolio"].iloc[0], 1_000.0)
        # after: cash (900) + 10 shares at that day's price
        self.assertAlmostEqual(ec["Your portfolio"].iloc[-1], 900.0 + 10 * 13.0)

    def test_benchmark_column_rebased_to_starting_cash(self):
        trades = _trades(("2024-01-01", "buy", "X", 1, 10.0, 0.0, "USD"))
        idx = pd.date_range("2024-01-01", "2024-01-03", freq="D")
        hist = pd.DataFrame({"X": [10.0, 10.0, 10.0]}, index=idx)
        bench = pd.Series([100.0, 110.0, 120.0], index=idx)

        ec = paper.equity_curve(1_000.0, trades, hist, benchmark=bench,
                                benchmark_name="SPY")
        self.assertIn("SPY", ec.columns)
        self.assertAlmostEqual(ec["SPY"].iloc[0], 1_000.0)          # rebased
        self.assertAlmostEqual(ec["SPY"].iloc[-1], 1_000.0 * 120 / 100)

    def test_empty_trades_returns_empty_frame(self):
        self.assertTrue(paper.equity_curve(1_000.0, pd.DataFrame(), pd.DataFrame()).empty)


if __name__ == "__main__":
    unittest.main()
