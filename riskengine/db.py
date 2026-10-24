"""SQLite storage layer: a small star-schema warehouse for prices, returns and results."""
from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import pandas as pd

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS dim_asset (
    ticker       TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    asset_class  TEXT NOT NULL,
    sub_class    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_price (
    date      TEXT NOT NULL,
    ticker    TEXT NOT NULL REFERENCES dim_asset(ticker),
    adj_close REAL NOT NULL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS fact_return (
    date    TEXT NOT NULL,
    ticker  TEXT NOT NULL REFERENCES dim_asset(ticker),
    ret     REAL NOT NULL,
    PRIMARY KEY (date, ticker)
);
CREATE TABLE IF NOT EXISTS fact_factor (
    date   TEXT PRIMARY KEY,
    MKT_RF REAL, SMB REAL, HML REAL, RMW REAL, CMA REAL, RF REAL, MOM REAL
);
CREATE TABLE IF NOT EXISTS fact_portfolio_nav (
    date      TEXT NOT NULL,
    strategy  TEXT NOT NULL,
    ret       REAL NOT NULL,
    nav       REAL NOT NULL,
    PRIMARY KEY (date, strategy)
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
CREATE INDEX IF NOT EXISTS ix_return_ticker ON fact_return(ticker, date);
CREATE INDEX IF NOT EXISTS ix_nav_strategy  ON fact_portfolio_nav(strategy, date);
"""


def connect(path: Path = config.DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    # SQLite builds don't always ship math functions; register the ones our SQL uses.
    conn.create_function("LN", 1, lambda x: math.log(x) if x and x > 0 else None, deterministic=True)
    conn.create_function("EXP", 1, lambda x: math.exp(x) if x is not None else None, deterministic=True)
    conn.create_function("SQRT", 1, lambda x: math.sqrt(x) if x is not None and x >= 0 else None,
                         deterministic=True)
    conn.executescript(SCHEMA)
    return conn


def _to_long(wide: pd.DataFrame, value_name: str) -> pd.DataFrame:
    long = wide.rename_axis("date").reset_index().melt(id_vars="date", var_name="ticker",
                                                       value_name=value_name)
    long["date"] = pd.to_datetime(long["date"]).dt.strftime("%Y-%m-%d")
    return long.dropna()


def load_market_data(conn: sqlite3.Connection, prices: pd.DataFrame, returns: pd.DataFrame,
                     factors: pd.DataFrame | None, source: str) -> None:
    cur = conn.cursor()
    for tbl in ("fact_price", "fact_return", "fact_factor", "dim_asset", "meta"):
        cur.execute(f"DELETE FROM {tbl}")
    cur.executemany("INSERT INTO dim_asset VALUES (?,?,?,?)",
                    [(t, *v) for t, v in config.UNIVERSE.items()])
    cur.executemany("INSERT INTO fact_price VALUES (?,?,?)",
                    _to_long(prices, "adj_close").itertuples(index=False, name=None))
    cur.executemany("INSERT INTO fact_return VALUES (?,?,?)",
                    _to_long(returns, "ret").itertuples(index=False, name=None))
    if factors is not None:
        f = factors.reindex(columns=["MKT_RF", "SMB", "HML", "RMW", "CMA", "RF", "MOM"])
        f.index = pd.to_datetime(f.index).strftime("%Y-%m-%d")
        rows = [(d, *[None if pd.isna(v) else float(v) for v in r])
                for d, r in zip(f.index, f.itertuples(index=False, name=None))]
        cur.executemany("INSERT INTO fact_factor VALUES (?,?,?,?,?,?,?,?)", rows)
    cur.executemany("INSERT INTO meta VALUES (?,?)", [
        ("data_source", source),
        ("start_date", str(prices.index.min().date())),
        ("end_date", str(prices.index.max().date())),
    ])
    conn.commit()


def load_nav(conn: sqlite3.Connection, navs: dict[str, pd.Series], rets: dict[str, pd.Series]) -> None:
    conn.execute("DELETE FROM fact_portfolio_nav")
    rows = []
    for name, nav in navs.items():
        r = rets[name].reindex(nav.index).fillna(0.0)
        for d, n, x in zip(nav.index.strftime("%Y-%m-%d"), nav.values, r.values):
            rows.append((d, name, float(x), float(n)))
    conn.executemany("INSERT INTO fact_portfolio_nav VALUES (?,?,?,?)", rows)
    conn.commit()


def run_sql_file(conn: sqlite3.Connection, path: Path) -> dict[str, pd.DataFrame]:
    """Execute a .sql file whose queries are separated by '-- name: <query_name>' markers."""
    text = path.read_text()
    results: dict[str, pd.DataFrame] = {}
    for block in text.split("-- name:")[1:]:
        name, _, sql = block.partition("\n")
        results[name.strip()] = pd.read_sql_query(sql, conn)
    return results


def read_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM meta").fetchall())
