"""Tests for pa.store — the SQLite persistence layer.

Each test gets its own throwaway DB file (never userdata/portfolios.db) and
forces the local-SQLite backend, so these never touch a real Turso database
even if this machine happens to have those env vars set.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pa import store


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._patches = [
            mock.patch.object(store, "DB_PATH", self._db_path),
            mock.patch.object(store, "_turso_creds", return_value=None),
        ]
        for p in self._patches:
            p.start()
        store.init()

    def tearDown(self):
        for p in self._patches:
            p.stop()
        self._tmpdir.cleanup()


class ProfileTests(StoreTestCase):
    def test_create_then_list(self):
        store.get_or_create_profile("Alice")
        self.assertIn("Alice", store.list_profiles())

    def test_get_or_create_is_idempotent(self):
        id1 = store.get_or_create_profile("Alice")
        id2 = store.get_or_create_profile("Alice")
        self.assertEqual(id1, id2)

    def test_rename_and_delete(self):
        store.get_or_create_profile("Alice")
        store.rename_profile("Alice", "Alicia")
        self.assertIn("Alicia", store.list_profiles())
        self.assertNotIn("Alice", store.list_profiles())
        store.delete_profile("Alicia")
        self.assertNotIn("Alicia", store.list_profiles())


class PasswordTests(StoreTestCase):
    def test_no_password_by_default(self):
        store.get_or_create_profile("Alice")
        self.assertFalse(store.profile_has_password("Alice"))
        # an unset password never blocks a check
        self.assertTrue(store.profile_check_password("Alice", "anything"))

    def test_password_set_at_creation(self):
        store.get_or_create_profile("Bob", password="secret1")
        self.assertTrue(store.profile_has_password("Bob"))
        self.assertTrue(store.profile_check_password("Bob", "secret1"))
        self.assertFalse(store.profile_check_password("Bob", "wrong"))

    def test_password_not_stored_in_plaintext(self):
        store.get_or_create_profile("Bob", password="secret1")
        with store._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM profiles WHERE name='Bob'"
            ).fetchone()
        self.assertNotEqual(row["password_hash"], "secret1")

    def test_set_password_on_existing_profile(self):
        store.get_or_create_profile("Carol")
        store.profile_set_password("Carol", "newpw")
        self.assertTrue(store.profile_check_password("Carol", "newpw"))

    def test_clearing_password_reopens_profile(self):
        store.get_or_create_profile("Carol", password="pw")
        store.profile_set_password("Carol", "")
        self.assertFalse(store.profile_has_password("Carol"))

    def test_second_creation_call_does_not_overwrite_password(self):
        store.get_or_create_profile("Dave", password="first")
        store.get_or_create_profile("Dave", password="second")  # already exists
        self.assertTrue(store.profile_check_password("Dave", "first"))


class PracticeAccountTests(StoreTestCase):
    def test_get_or_create_defaults(self):
        acc = store.practice_get_or_create("Alice")
        self.assertEqual(acc["starting_cash"], 10_000.0)
        self.assertEqual(acc["currency"], "EUR")

    def test_practice_get_is_read_only(self):
        # looking up a profile that's never opened Practice mode must not
        # silently create an account for it (the leaderboard relies on this)
        store.get_or_create_profile("Alice")
        self.assertIsNone(store.practice_get("Alice"))
        store.practice_get_or_create("Alice")
        self.assertIsNotNone(store.practice_get("Alice"))

    def test_currency_can_be_set(self):
        acc = store.practice_get_or_create("Alice", currency="USD")
        self.assertEqual(acc["currency"], "USD")
        store.practice_set_currency(acc["id"], "GBP")
        self.assertEqual(store.practice_get("Alice")["currency"], "GBP")


class TradeTests(StoreTestCase):
    def setUp(self):
        super().setUp()
        self.aid = store.practice_get_or_create("Alice")["id"]

    def test_record_and_read_back(self):
        store.practice_record_trade(self.aid, "buy", "aapl", 10, 150.0,
                                    fee=1.5, ccy="USD")
        df = store.practice_trades(self.aid)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["ticker"], "AAPL")  # upper-cased
        self.assertEqual(row["shares"], 10.0)
        self.assertEqual(row["fee"], 1.5)
        self.assertEqual(row["ccy"], "USD")
        self.assertIn("id", df.columns)

    def test_rejects_bad_side_or_nonpositive_amounts(self):
        with self.assertRaises(ValueError):
            store.practice_record_trade(self.aid, "hold", "AAPL", 1, 100.0)
        with self.assertRaises(ValueError):
            store.practice_record_trade(self.aid, "buy", "AAPL", 0, 100.0)
        with self.assertRaises(ValueError):
            store.practice_record_trade(self.aid, "buy", "AAPL", 1, -5.0)

    def test_undo_last_removes_only_the_most_recent(self):
        store.practice_record_trade(self.aid, "buy", "AAPL", 1, 100.0)
        store.practice_record_trade(self.aid, "buy", "MSFT", 1, 200.0)
        store.practice_undo_last(self.aid)
        df = store.practice_trades(self.aid)
        self.assertEqual(list(df["ticker"]), ["AAPL"])

    def test_reset_clears_trades_and_can_update_starting_cash(self):
        store.practice_record_trade(self.aid, "buy", "AAPL", 1, 100.0)
        store.practice_reset(self.aid, starting_cash=5_000.0)
        self.assertTrue(store.practice_trades(self.aid).empty)
        self.assertEqual(store.practice_get("Alice")["starting_cash"], 5_000.0)

    def test_note_round_trip(self):
        store.practice_record_trade(self.aid, "buy", "AAPL", 1, 100.0)
        tid = int(store.practice_trades(self.aid).iloc[0]["id"])
        store.practice_set_trade_note(tid, "  bought the dip  ")
        note = store.practice_trades(self.aid).iloc[0]["note"]
        self.assertEqual(note, "bought the dip")  # trimmed

    def test_note_length_is_capped(self):
        store.practice_record_trade(self.aid, "buy", "AAPL", 1, 100.0)
        tid = int(store.practice_trades(self.aid).iloc[0]["id"])
        store.practice_set_trade_note(tid, "x" * 5000)
        note = store.practice_trades(self.aid).iloc[0]["note"]
        self.assertEqual(len(note), 4000)


class SettingsTests(StoreTestCase):
    def test_default_when_unset(self):
        self.assertEqual(store.get_setting("currency", "EUR"), "EUR")

    def test_set_then_get(self):
        store.set_setting("currency", "USD")
        self.assertEqual(store.get_setting("currency", "EUR"), "USD")

    def test_upsert_overwrites(self):
        store.set_setting("currency", "USD")
        store.set_setting("currency", "GBP")
        self.assertEqual(store.get_setting("currency"), "GBP")


class MigrationTests(StoreTestCase):
    def test_init_is_idempotent(self):
        store.init()
        store.init()  # must not raise on a second pass
        store.get_or_create_profile("Alice")
        self.assertIn("Alice", store.list_profiles())


if __name__ == "__main__":
    unittest.main()
