"""Tests for pa.portfolio — parsing free-form holdings text into weights."""

from __future__ import annotations

import unittest

import pandas as pd

from pa import portfolio


class ParseHoldingsTextTests(unittest.TestCase):
    def test_space_separated_lines(self):
        df = portfolio.parse_holdings_text("AAPL 40\nMSFT 25")
        self.assertEqual(df.loc["AAPL", "amount"], 40.0)
        self.assertEqual(df.loc["MSFT", "amount"], 25.0)

    def test_comma_separated_lines(self):
        df = portfolio.parse_holdings_text("AAPL, 40\nMSFT, 25")
        self.assertEqual(sorted(df.index), ["AAPL", "MSFT"])
        self.assertEqual(df.loc["AAPL", "amount"], 40.0)

    def test_csv_with_header(self):
        df = portfolio.parse_holdings_text("ticker,shares\nAAPL,40\nMSFT,25")
        self.assertEqual(df.loc["AAPL", "amount"], 40.0)

    def test_ticker_only_defaults_to_one_share(self):
        df = portfolio.parse_holdings_text("AAPL")
        self.assertEqual(df.loc["AAPL", "amount"], 1.0)

    def test_duplicate_tickers_sum(self):
        df = portfolio.parse_holdings_text("AAPL 10\nAAPL 5")
        self.assertEqual(df.loc["AAPL", "amount"], 15.0)

    def test_comment_and_blank_lines_ignored(self):
        df = portfolio.parse_holdings_text("# a comment\n\nAAPL 10\n")
        self.assertEqual(list(df.index), ["AAPL"])

    def test_lowercase_ticker_uppercased(self):
        df = portfolio.parse_holdings_text("aapl 10")
        self.assertEqual(list(df.index), ["AAPL"])

    def test_empty_input(self):
        df = portfolio.parse_holdings_text("")
        self.assertTrue(df.empty)

    def test_dollar_and_percent_signs_stripped(self):
        df = portfolio.parse_holdings_text("AAPL $500\nMSFT 25%")
        self.assertEqual(df.loc["AAPL", "amount"], 500.0)
        self.assertEqual(df.loc["MSFT", "amount"], 25.0)


class NormalizeEditorDfTests(unittest.TestCase):
    def test_renames_flexible_headers(self):
        df = pd.DataFrame({"Symbol": ["aapl", "msft"], "Shares": [10, 20]})
        out = portfolio.normalize_editor_df(df)
        self.assertEqual(sorted(out.index), ["AAPL", "MSFT"])
        self.assertEqual(out.loc["AAPL", "amount"], 10.0)

    def test_drops_zero_and_negative_and_blank_rows(self):
        df = pd.DataFrame({"Ticker": ["AAPL", "", "MSFT", "GOOG"],
                           "Amount": [10, 5, 0, -3]})
        out = portfolio.normalize_editor_df(df)
        self.assertEqual(list(out.index), ["AAPL"])

    def test_empty_or_none_input(self):
        self.assertTrue(portfolio.normalize_editor_df(None).empty)
        self.assertTrue(portfolio.normalize_editor_df(pd.DataFrame()).empty)

    def test_missing_recognizable_columns_returns_empty(self):
        df = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
        self.assertTrue(portfolio.normalize_editor_df(df).empty)


class ToWeightsTests(unittest.TestCase):
    def test_weight_mode_normalizes_to_one(self):
        holdings = pd.DataFrame({"amount": [60.0, 40.0]}, index=["AAPL", "BND"])
        w = portfolio.to_weights(holdings, mode="weight")
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertAlmostEqual(w["AAPL"], 0.6)

    def test_dollars_mode_normalizes_to_one(self):
        holdings = pd.DataFrame({"amount": [750.0, 250.0]}, index=["AAPL", "BND"])
        w = portfolio.to_weights(holdings, mode="dollars")
        self.assertAlmostEqual(w["AAPL"], 0.75)

    def test_shares_mode_uses_supplied_prices(self):
        holdings = pd.DataFrame({"amount": [10.0, 5.0]}, index=["AAPL", "BND"])
        prices = pd.Series({"AAPL": 100.0, "BND": 80.0})  # 1000 vs 400
        w = portfolio.to_weights(holdings, mode="shares", latest_prices=prices)
        self.assertAlmostEqual(w["AAPL"], 1000 / 1400)
        self.assertAlmostEqual(w["BND"], 400 / 1400)

    def test_empty_holdings_returns_empty_dict(self):
        self.assertEqual(portfolio.to_weights(pd.DataFrame(columns=["amount"])), {})

    def test_ticker_missing_a_price_is_dropped_not_nan(self):
        holdings = pd.DataFrame({"amount": [10.0, 5.0]}, index=["AAPL", "GHOST"])
        prices = pd.Series({"AAPL": 100.0})  # no quote for GHOST
        w = portfolio.to_weights(holdings, mode="shares", latest_prices=prices)
        self.assertEqual(set(w), {"AAPL"})
        self.assertAlmostEqual(w["AAPL"], 1.0)


if __name__ == "__main__":
    unittest.main()
