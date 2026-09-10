"""Local persistence for user profiles and their saved portfolios.

A single SQLite file under ``userdata/`` (git-ignored). No passwords - profiles
are just named namespaces, appropriate for a dashboard that runs on your own
machine. Every call opens its own short-lived connection so it is safe to use
from Streamlit's script threads.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "userdata" / "portfolios.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolios (
    id          INTEGER PRIMARY KEY,
    profile_id  INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    amount_mode TEXT NOT NULL DEFAULT 'shares',
    watchlist   TEXT NOT NULL DEFAULT '',
    updated_at  TEXT NOT NULL,
    UNIQUE (profile_id, name)
);
CREATE TABLE IF NOT EXISTS positions (
    id           INTEGER PRIMARY KEY,
    portfolio_id INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    amount       REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS practice_accounts (
    id            INTEGER PRIMARY KEY,
    profile_id    INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    name          TEXT NOT NULL DEFAULT 'Practice',
    starting_cash REAL NOT NULL DEFAULT 10000,
    created_at    TEXT NOT NULL,
    UNIQUE (profile_id, name)
);
CREATE TABLE IF NOT EXISTS practice_trades (
    id         INTEGER PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES practice_accounts(id) ON DELETE CASCADE,
    ts         TEXT NOT NULL,
    side       TEXT NOT NULL,
    ticker     TEXT NOT NULL,
    shares     REAL NOT NULL,
    price      REAL NOT NULL,
    fee        REAL NOT NULL DEFAULT 0
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Additive column adds for databases created by an older version."""
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(practice_trades)")}
    if "fee" not in cols:
        conn.execute("ALTER TABLE practice_trades ADD COLUMN fee REAL NOT NULL DEFAULT 0")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        _migrate(conn)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def list_profiles() -> list[str]:
    with _connect() as conn:
        return [r["name"] for r in conn.execute("SELECT name FROM profiles ORDER BY name")]


def get_or_create_profile(name: str) -> int:
    name = name.strip()
    if not name:
        raise ValueError("Profile name cannot be empty.")
    with _connect() as conn:
        row = conn.execute("SELECT id FROM profiles WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        cur = conn.execute(
            "INSERT INTO profiles (name, created_at) VALUES (?, ?)", (name, _now())
        )
        return cur.lastrowid


def _profile_id(conn: sqlite3.Connection, name: str) -> int | None:
    row = conn.execute("SELECT id FROM profiles WHERE name = ?", (name.strip(),)).fetchone()
    return row["id"] if row else None


def delete_profile(name: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM profiles WHERE name = ?", (name.strip(),))


def rename_profile(old: str, new: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE profiles SET name = ? WHERE name = ?", (new.strip(), old.strip())
        )


# --------------------------------------------------------------------------- #
# Portfolios
# --------------------------------------------------------------------------- #
def list_portfolios(profile: str) -> pd.DataFrame:
    with _connect() as conn:
        pid = _profile_id(conn, profile)
        if pid is None:
            return pd.DataFrame(columns=["name", "amount_mode", "n_positions", "updated_at"])
        rows = conn.execute(
            """
            SELECT p.name, p.amount_mode, p.updated_at,
                   COUNT(pos.id) AS n_positions
            FROM portfolios p
            LEFT JOIN positions pos ON pos.portfolio_id = p.id
            WHERE p.profile_id = ?
            GROUP BY p.id
            ORDER BY p.updated_at DESC
            """,
            (pid,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def save_portfolio(
    profile: str,
    name: str,
    positions: pd.DataFrame,
    amount_mode: str = "shares",
    watchlist: str = "",
) -> None:
    """Create or overwrite a named portfolio.

    ``positions`` needs columns 'ticker' and 'amount' (any case). Blank tickers
    and non-numeric / non-positive amounts are dropped.
    """
    name = name.strip()
    if not name:
        raise ValueError("Portfolio name cannot be empty.")

    clean = _clean_positions(positions)
    if clean.empty:
        raise ValueError("Portfolio has no valid positions to save.")

    with _connect() as conn:
        pid = _profile_id(conn, profile)
        if pid is None:
            pid = conn.execute(
                "INSERT INTO profiles (name, created_at) VALUES (?, ?)",
                (profile.strip(), _now()),
            ).lastrowid

        row = conn.execute(
            "SELECT id FROM portfolios WHERE profile_id = ? AND name = ?", (pid, name)
        ).fetchone()
        if row:
            port_id = row["id"]
            conn.execute(
                "UPDATE portfolios SET amount_mode = ?, watchlist = ?, updated_at = ? WHERE id = ?",
                (amount_mode, watchlist, _now(), port_id),
            )
            conn.execute("DELETE FROM positions WHERE portfolio_id = ?", (port_id,))
        else:
            port_id = conn.execute(
                "INSERT INTO portfolios (profile_id, name, amount_mode, watchlist, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (pid, name, amount_mode, watchlist, _now()),
            ).lastrowid

        conn.executemany(
            "INSERT INTO positions (portfolio_id, ticker, amount) VALUES (?, ?, ?)",
            [(port_id, t, a) for t, a in clean.itertuples(index=False)],
        )


def load_portfolio(profile: str, name: str) -> dict | None:
    """Return {'positions': DataFrame[ticker, amount], 'amount_mode', 'watchlist'}."""
    with _connect() as conn:
        pid = _profile_id(conn, profile)
        if pid is None:
            return None
        port = conn.execute(
            "SELECT id, amount_mode, watchlist FROM portfolios WHERE profile_id = ? AND name = ?",
            (pid, name),
        ).fetchone()
        if not port:
            return None
        rows = conn.execute(
            "SELECT ticker, amount FROM positions WHERE portfolio_id = ? ORDER BY ticker",
            (port["id"],),
        ).fetchall()
    return {
        "positions": pd.DataFrame([dict(r) for r in rows], columns=["ticker", "amount"]),
        "amount_mode": port["amount_mode"],
        "watchlist": port["watchlist"],
    }


def delete_portfolio(profile: str, name: str) -> None:
    with _connect() as conn:
        pid = _profile_id(conn, profile)
        if pid is None:
            return
        conn.execute(
            "DELETE FROM portfolios WHERE profile_id = ? AND name = ?", (pid, name)
        )


def duplicate_portfolio(profile: str, name: str, new_name: str) -> None:
    data = load_portfolio(profile, name)
    if data is None:
        raise ValueError(f"Portfolio '{name}' not found.")
    save_portfolio(
        profile, new_name, data["positions"], data["amount_mode"], data["watchlist"]
    )


def _clean_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ticker", "amount"])
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[k] for k in ("ticker", "symbol", "stock") if k in cols), None)
    acol = next(
        (cols[k] for k in ("amount", "shares", "quantity", "qty", "weight", "value") if k in cols),
        None,
    )
    if tcol is None or acol is None:
        return pd.DataFrame(columns=["ticker", "amount"])
    out = df[[tcol, acol]].copy()
    out.columns = ["ticker", "amount"]
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out = out[(out["ticker"] != "") & out["amount"].notna() & (out["amount"] > 0)]
    return out.groupby("ticker", as_index=False)["amount"].sum()


# --------------------------------------------------------------------------- #
# Practice ("paper trading") portfolio
# --------------------------------------------------------------------------- #
def practice_get_or_create(
    profile: str, name: str = "Practice", starting_cash: float = 10_000.0
) -> dict:
    with _connect() as conn:
        pid = _profile_id(conn, profile)
        if pid is None:
            pid = conn.execute(
                "INSERT INTO profiles (name, created_at) VALUES (?, ?)",
                (profile.strip(), _now()),
            ).lastrowid
        row = conn.execute(
            "SELECT * FROM practice_accounts WHERE profile_id = ? AND name = ?", (pid, name)
        ).fetchone()
        if row is None:
            aid = conn.execute(
                "INSERT INTO practice_accounts (profile_id, name, starting_cash, created_at) "
                "VALUES (?, ?, ?, ?)",
                (pid, name, float(starting_cash), _now()),
            ).lastrowid
            row = conn.execute("SELECT * FROM practice_accounts WHERE id = ?", (aid,)).fetchone()
    return dict(row)


def practice_record_trade(
    account_id: int, side: str, ticker: str, shares: float, price: float,
    fee: float = 0.0,
) -> None:
    if side not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")
    if shares <= 0 or price <= 0:
        raise ValueError("shares and price must be positive")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO practice_trades (account_id, ts, side, ticker, shares, price, fee) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (account_id, _now(), side, ticker.strip().upper(), float(shares),
             float(price), max(0.0, float(fee))),
        )


def practice_trades(account_id: int) -> pd.DataFrame:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, side, ticker, shares, price, fee FROM practice_trades "
            "WHERE account_id = ? ORDER BY ts, id",
            (account_id,),
        ).fetchall()
    df = pd.DataFrame([dict(r) for r in rows],
                      columns=["ts", "side", "ticker", "shares", "price", "fee"])
    if not df.empty:
        # stored as UTC ISO; drop the tz so it compares cleanly with naive price indexes
        df["ts"] = pd.to_datetime(df["ts"], utc=True).dt.tz_localize(None)
    return df


def practice_reset(account_id: int, starting_cash: float | None = None) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM practice_trades WHERE account_id = ?", (account_id,))
        if starting_cash is not None:
            conn.execute(
                "UPDATE practice_accounts SET starting_cash = ? WHERE id = ?",
                (float(starting_cash), account_id),
            )


def practice_undo_last(account_id: int) -> None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id FROM practice_trades WHERE account_id = ? ORDER BY ts DESC, id DESC LIMIT 1",
            (account_id,),
        ).fetchone()
        if row:
            conn.execute("DELETE FROM practice_trades WHERE id = ?", (row["id"],))
