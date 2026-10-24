"""Central configuration: investment universe, model portfolio, benchmark and scenarios."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMG_DIR = ROOT / "docs" / "images"
DB_PATH = DATA_DIR / "risk.db"

START_DATE = "2007-06-01"   # captures the 2008 Global Financial Crisis
TRADING_DAYS = 252
RISK_FREE_FALLBACK = 0.02   # used only if Fama-French RF series is unavailable
NOTIONAL = 100_000_000      # $100M model client portfolio

# Investment universe: iShares ETFs spanning the main asset classes.
UNIVERSE = {
    "IVV": ("iShares Core S&P 500", "Equity", "US Large Cap"),
    "EFA": ("iShares MSCI EAFE", "Equity", "Developed ex-US"),
    "EEM": ("iShares MSCI Emerging Markets", "Equity", "Emerging Markets"),
    "IYR": ("iShares US Real Estate", "Real Assets", "REITs"),
    "AGG": ("iShares Core US Aggregate Bond", "Fixed Income", "Core Aggregate"),
    "TLT": ("iShares 20+ Year Treasury Bond", "Fixed Income", "Long Treasuries"),
    "LQD": ("iShares iBoxx $ IG Corporate Bond", "Fixed Income", "IG Credit"),
    "HYG": ("iShares iBoxx $ High Yield Corporate Bond", "Fixed Income", "HY Credit"),
    "TIP": ("iShares TIPS Bond", "Fixed Income", "Inflation-Linked"),
    "IAU": ("iShares Gold Trust", "Real Assets", "Gold"),
}
TICKERS = list(UNIVERSE)

# Model client portfolio (strategic allocation) — weights sum to 1.
PORTFOLIO = {
    "IVV": 0.30, "EFA": 0.10, "EEM": 0.05, "IYR": 0.05,
    "AGG": 0.20, "TLT": 0.05, "LQD": 0.10, "HYG": 0.05, "TIP": 0.05,
    "IAU": 0.05,
}

# Benchmark: classic 60/40 (US equity / US aggregate bonds).
BENCHMARK = {"IVV": 0.60, "AGG": 0.40}

# Historical stress windows (peak-to-trough of the S&P 500 or the defining move).
HISTORICAL_SCENARIOS = {
    "Global Financial Crisis": ("2008-09-01", "2009-03-09"),
    "Euro Sovereign Debt Crisis": ("2011-07-22", "2011-10-03"),
    "Taper Tantrum": ("2013-05-22", "2013-06-24"),
    "China Devaluation / Oil Crash": ("2015-08-10", "2016-02-11"),
    "COVID-19 Crash": ("2020-02-19", "2020-03-23"),
    "2022 Inflation & Rate Shock": ("2022-01-03", "2022-10-12"),
    "SVB Regional Bank Stress": ("2023-03-08", "2023-03-17"),
}

# Hypothetical shocks on "driver" assets; all other assets respond through their
# conditional expectation under a covariance matrix from the chosen correlation regime:
#   "stressed" = days when the S&P 500 was >10% below its peak (crisis correlations)
#   "recent"   = trailing 3 years (post-2022 regime: stocks and bonds positively correlated)
HYPOTHETICAL_SCENARIOS = {
    "Equity Crash (S&P -25%)": ({"IVV": -0.25}, "stressed"),
    "Rates +150bp (Long UST -20%)": ({"TLT": -0.20}, "recent"),
    "Credit Spread Blowout (HY -15%)": ({"HYG": -0.15}, "stressed"),
    "Stagflation (Equity -15%, UST -10%)": ({"IVV": -0.15, "TLT": -0.10}, "recent"),
    "EM Crisis (EM -30%)": ({"EEM": -0.30}, "stressed"),
    "Flight to Safety (Gold +15%, UST +10%)": ({"IAU": 0.15, "TLT": 0.10}, "stressed"),
}

VAR_CONFIDENCE = 0.99
VAR_WINDOW = 500          # look-back for rolling VaR backtest (trading days)
TRANSACTION_COST_BPS = 5  # one-way cost applied to turnover in backtests
MAX_WEIGHT = 0.40         # single-asset cap for optimizers
