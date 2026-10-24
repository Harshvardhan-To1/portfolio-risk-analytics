"""Market data ingestion with a fallback chain: yfinance -> Yahoo chart API -> synthetic.

Fama-French factor data is pulled from Kenneth French's data library.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile

import numpy as np
import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

FF5_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")
MOM_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Momentum_Factor_daily_CSV.zip")
HEADERS = {"User-Agent": "Mozilla/5.0 (portfolio-risk-analytics research project)"}


# --------------------------------------------------------------------------- prices
def _from_yahoo(tickers: list[str], start: str) -> pd.DataFrame:
    import warnings

    import yfinance as yf

    with warnings.catch_warnings():  # yfinance emits pandas deprecation noise
        warnings.simplefilter("ignore")
        df = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                         threads=False)
    if df is None or df.empty:
        raise RuntimeError("Yahoo returned no data")
    prices = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]]
    missing = set(tickers) - set(prices.dropna(axis=1, how="all").columns)
    if missing:
        raise RuntimeError(f"Yahoo missing tickers: {sorted(missing)}")
    return prices[tickers]


def _from_yahoo_chart_api(tickers: list[str], start: str) -> pd.DataFrame:
    """Direct call to Yahoo's v8 chart endpoint (dividend/split-adjusted close)."""
    period1 = int(pd.Timestamp(start).timestamp())
    period2 = int(pd.Timestamp.today().timestamp())
    frames = {}
    for t in tickers:
        url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{t}"
               f"?period1={period1}&period2={period2}&interval=1d&events=div,split")
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        res = resp.json()["chart"]["result"][0]
        idx = pd.to_datetime(res["timestamp"], unit="s").normalize()
        adj = res["indicators"]["adjclose"][0]["adjclose"]
        frames[t] = pd.Series(adj, index=idx, dtype=float).groupby(level=0).last()
        time.sleep(0.3)
    prices = pd.DataFrame(frames)
    return prices.loc[prices.index >= start, tickers]


def _synthetic(tickers: list[str], start: str, seed: int = 7) -> pd.DataFrame:
    """Correlated fat-tailed returns with crisis regimes. Used only when offline."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, pd.Timestamp.today().normalize())
    n = len(dates)
    vol = {"IVV": .18, "EFA": .20, "EEM": .25, "IYR": .26, "AGG": .045, "TLT": .15,
           "LQD": .08, "HYG": .10, "TIP": .06, "IAU": .16}
    mu = {"IVV": .10, "EFA": .06, "EEM": .05, "IYR": .07, "AGG": .03, "TLT": .035,
          "LQD": .04, "HYG": .055, "TIP": .03, "IAU": .07}
    # one equity/risk factor + one duration factor drive the correlation structure
    eq_load = {"IVV": .95, "EFA": .85, "EEM": .80, "IYR": .75, "AGG": .0, "TLT": -.3,
               "LQD": .25, "HYG": .70, "TIP": .05, "IAU": .05}
    dur_load = {"IVV": .0, "EFA": .0, "EEM": .0, "IYR": .2, "AGG": .9, "TLT": .9,
                "LQD": .8, "HYG": .2, "TIP": .75, "IAU": .25}
    regime = np.ones(n)
    for s, e in config.HISTORICAL_SCENARIOS.values():
        regime[(dates >= s) & (dates <= e)] = 2.5
    f_eq = rng.standard_t(4, n) / np.sqrt(2) * regime
    f_dur = rng.standard_t(5, n) / np.sqrt(5 / 3)
    out = {}
    for t in tickers:
        idio_w = np.sqrt(max(1 - eq_load[t] ** 2 - dur_load[t] ** 2, 0.05))
        z = eq_load[t] * f_eq + dur_load[t] * f_dur + idio_w * rng.standard_normal(n)
        z = z / z.std()
        daily = mu[t] / 252 + vol[t] / np.sqrt(252) * z
        daily = np.where(regime > 1, daily - 0.0015 * eq_load[t], daily)
        out[t] = 100 * np.exp(np.cumsum(np.log1p(daily)))
    return pd.DataFrame(out, index=dates)


def fetch_prices(tickers: list[str] = config.TICKERS,
                 start: str = config.START_DATE) -> tuple[pd.DataFrame, str]:
    """Return (adjusted close prices, source name)."""
    for name, fn in (("yahoo", _from_yahoo), ("yahoo-chart-api", _from_yahoo_chart_api)):
        try:
            prices = fn(tickers, start)
            prices = prices.sort_index().ffill(limit=3).dropna()
            if len(prices) > 1000:
                log.info("Loaded %d days of prices from %s", len(prices), name)
                return prices, name
        except Exception as exc:  # network failure, rate limit, schema change...
            log.warning("Price source %s failed: %s", name, exc)
    log.warning("All live sources failed - generating SYNTHETIC data for demo purposes")
    return _synthetic(tickers, start), "synthetic"


# --------------------------------------------------------------------------- factors
def _read_french_zip(url: str) -> pd.DataFrame:
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0]).decode("latin-1")
    lines = raw.splitlines()
    # header = the line right before the first row that starts with a YYYYMMDD date
    first_row = next(i for i, ln in enumerate(lines)
                     if ln.split(",")[0].strip().isdigit() and len(ln.split(",")[0].strip()) == 8)
    header_idx = first_row - 1
    rows = []
    for ln in lines[header_idx + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)
    cols = ["Date"] + [c.strip() for c in lines[header_idx].split(",")[1:]]
    df = pd.DataFrame(rows, columns=cols)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    return df.set_index("Date").astype(float) / 100.0


def fetch_factors(start: str = config.START_DATE) -> pd.DataFrame | None:
    """Daily Fama-French 5 factors + Momentum (decimal returns), or None if unavailable."""
    try:
        ff5 = _read_french_zip(FF5_URL)
        try:
            mom = _read_french_zip(MOM_URL)
            mom.columns = ["MOM"]
            ff5 = ff5.join(mom, how="inner")
        except Exception as exc:
            log.warning("Momentum factor unavailable: %s", exc)
        ff5 = ff5.rename(columns={"Mkt-RF": "MKT_RF"})
        return ff5.loc[ff5.index >= start]
    except Exception as exc:
        log.warning("Fama-French factors unavailable: %s", exc)
        return None
