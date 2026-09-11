"""Tests for pa.fx — currency normalization and conversion.

Network calls (through pa.data.fetch_prices) are monkeypatched out with
synthetic FX series, so these run offline and deterministically.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from pa import fx


class NormalizeTests(unittest.TestCase):
    def test_plain_code_passthrough(self):
        self.assertEqual(fx.normalize("USD"), ("USD", 1.0))
        self.assertEqual(fx.normalize("eur"), ("EUR", 1.0))

    def test_pence_scaled_to_pounds(self):
        code, scale = fx.normalize("GBp")
        self.assertEqual(code, "GBP")
        self.assertAlmostEqual(scale, 0.01)

    def test_empty_string(self):
        self.assertEqual(fx.normalize(""), ("", 1.0))


class SymbolTests(unittest.TestCase):
    def test_known_currency(self):
        self.assertEqual(fx.symbol("USD"), "$")
        self.assertEqual(fx.symbol("EUR"), "€")

    def test_unknown_currency_falls_back_to_code(self):
        self.assertEqual(fx.symbol("XYZ"), "XYZ ")


class ConvertTests(unittest.TestCase):
    def test_same_currency_is_identity(self):
        self.assertEqual(fx.convert("EUR", "EUR"), 1.0)

    def test_pence_converts_via_scale_even_with_no_pair_data(self):
        # normalize() alone should apply the 0.01 scale for GBp->GBP; with
        # no mocked pair data (unknown to Yahoo in this test), convert()
        # falls back to the scale rather than erroring.
        with mock.patch("pa.fx._pair", return_value=None):
            self.assertAlmostEqual(fx.convert("GBp", "GBP"), 0.01)

    def test_direct_pair_used_when_available(self):
        with mock.patch("pa.data.fetch_prices") as m:
            idx = pd.date_range("2024-01-01", periods=3)
            m.return_value = pd.DataFrame({"EURUSD=X": [1.10, 1.11, 1.12]}, index=idx)
            rate = fx.convert("EUR", "USD")
            self.assertAlmostEqual(rate, 1.12)  # last value = spot

    def test_inverse_pair_used_when_direct_missing(self):
        with mock.patch("pa.data.fetch_prices") as m:
            idx = pd.date_range("2024-01-01", periods=2)

            def fake_fetch(tickers, lookback_years=15.0):
                if tickers == ["USDEUR=X"]:
                    return pd.DataFrame()  # direct pair unavailable
                if tickers == ["EURUSD=X"]:
                    return pd.DataFrame({"EURUSD=X": [1.0, 1.25]}, index=idx)
                return pd.DataFrame()

            m.side_effect = fake_fetch
            rate = fx.convert("USD", "EUR")  # asks for the inverse of EURUSD
            self.assertAlmostEqual(rate, 1 / 1.25)

    def test_historical_series_reindexed_to_requested_dates(self):
        with mock.patch("pa.data.fetch_prices") as m:
            src_idx = pd.date_range("2024-01-01", periods=5, freq="D")
            m.return_value = pd.DataFrame(
                {"EURUSD=X": [1.00, 1.01, 1.02, 1.03, 1.04]}, index=src_idx
            )
            want_idx = pd.date_range("2024-01-02", periods=2, freq="D")
            s = fx.convert("EUR", "USD", index=want_idx)
            self.assertEqual(len(s), 2)
            self.assertAlmostEqual(s.iloc[0], 1.01)

    def test_unknown_pair_falls_back_to_flat_one(self):
        with mock.patch("pa.fx._pair", return_value=None):
            idx = pd.date_range("2024-01-01", periods=3)
            s = fx.convert("ZZZ", "USD", index=idx)
            self.assertTrue((s == 1.0).all())


class ConvertFrameTests(unittest.TestCase):
    def test_converts_each_column_by_its_own_currency(self):
        idx = pd.date_range("2024-01-01", periods=3)
        prices = pd.DataFrame({"AAPL": [100, 101, 102], "SU.PA": [50, 51, 52]}, index=idx)

        def fake_convert(native, account, index=None, lookback_years=15.0):
            if native == account:
                return pd.Series(1.0, index=index)
            return pd.Series(0.9, index=index)  # pretend USD->EUR is flat 0.9

        with mock.patch("pa.fx.convert", side_effect=fake_convert):
            out = fx.convert_frame(prices, {"AAPL": "USD", "SU.PA": "EUR"}, "EUR")
        self.assertAlmostEqual(out["AAPL"].iloc[0], 90.0)   # converted
        self.assertAlmostEqual(out["SU.PA"].iloc[0], 50.0)  # already EUR, untouched

    def test_empty_frame_passthrough(self):
        out = fx.convert_frame(pd.DataFrame(), {}, "EUR")
        self.assertTrue(out.empty)


if __name__ == "__main__":
    unittest.main()
