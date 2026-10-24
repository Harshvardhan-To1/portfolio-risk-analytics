"""Value-at-Risk / Expected Shortfall models and statistical backtesting.

All VaR / ES figures are returned as positive loss fractions (0.023 = 2.3% loss).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- models
def historical_var(r: pd.Series, conf: float = 0.99) -> float:
    return float(-np.quantile(r, 1 - conf))


def historical_es(r: pd.Series, conf: float = 0.99) -> float:
    cutoff = np.quantile(r, 1 - conf)
    return float(-r[r <= cutoff].mean())


def parametric_var(r: pd.Series, conf: float = 0.99) -> float:
    z = stats.norm.ppf(1 - conf)
    return float(-(r.mean() + z * r.std(ddof=1)))


def parametric_es(r: pd.Series, conf: float = 0.99) -> float:
    a = 1 - conf
    return float(-(r.mean() - r.std(ddof=1) * stats.norm.pdf(stats.norm.ppf(a)) / a))


def cornish_fisher_var(r: pd.Series, conf: float = 0.99) -> float:
    """Modified VaR: adjusts the normal quantile for skewness and excess kurtosis."""
    z = stats.norm.ppf(1 - conf)
    s, k = stats.skew(r), stats.kurtosis(r)  # kurtosis() is excess kurtosis
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36)
    return float(-(r.mean() + z_cf * r.std(ddof=1)))


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA volatility forecast for day t using information up to t-1."""
    var = np.empty(len(r))
    var[0] = r.iloc[:20].var()
    x = r.values
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=r.index)


def monte_carlo_var(asset_returns: pd.DataFrame, weights: np.ndarray, conf: float = 0.99,
                    n_sims: int = 100_000, horizon: int = 1, dof: float = 5.0,
                    seed: int = 42) -> dict[str, float | np.ndarray]:
    """Multivariate Student-t simulation (captures fat tails + cross-asset correlation)."""
    rng = np.random.default_rng(seed)
    mu = asset_returns.mean().values
    cov = asset_returns.cov().values
    # scale so the simulated t-distribution has the sample covariance
    scale = cov * (dof - 2) / dof
    L = np.linalg.cholesky(scale)
    z = rng.standard_normal((n_sims, len(mu))) @ L.T
    chi = rng.chisquare(dof, size=(n_sims, 1))
    sims = mu * horizon + z * np.sqrt(dof / chi) * np.sqrt(horizon)
    pnl = sims @ weights
    var = -np.quantile(pnl, 1 - conf)
    es = -pnl[pnl <= -var].mean()
    return {"var": float(var), "es": float(es), "pnl": pnl}


def var_table(r: pd.Series, asset_returns: pd.DataFrame, weights: np.ndarray,
              levels=(0.95, 0.99, 0.995)) -> pd.DataFrame:
    rows = []
    for c in levels:
        mc = monte_carlo_var(asset_returns, weights, conf=c)
        rows.append({
            "Confidence": f"{c:.1%}",
            "Historical VaR": historical_var(r, c),
            "Parametric VaR": parametric_var(r, c),
            "Cornish-Fisher VaR": cornish_fisher_var(r, c),
            "Monte Carlo (t) VaR": mc["var"],
            "Historical ES": historical_es(r, c),
            "Parametric ES": parametric_es(r, c),
            "Monte Carlo (t) ES": mc["es"],
        })
    return pd.DataFrame(rows).set_index("Confidence")


# --------------------------------------------------------------------------- backtesting
def rolling_var_forecasts(r: pd.Series, conf: float = 0.99, window: int = 500) -> pd.DataFrame:
    """Out-of-sample 1-day VaR forecasts: each day's VaR uses only the prior `window` days."""
    hist = -r.rolling(window).quantile(1 - conf).shift(1)
    mu = r.rolling(window).mean().shift(1)
    sd = r.rolling(window).std().shift(1)
    z = stats.norm.ppf(1 - conf)
    param = -(mu + z * sd)
    sigma = ewma_vol(r)
    ewma = -(z * sigma)  # EWMA assumes zero mean, the RiskMetrics convention
    # Filtered Historical Simulation (Hull & White 1998): standardise past returns by their
    # EWMA vol, take the empirical quantile of those residuals, rescale by today's vol.
    # Keeps the fat-tailed empirical shape while reacting to the current vol regime.
    resid = r / sigma
    fhs = -(resid.rolling(window).quantile(1 - conf).shift(1) * sigma)
    df = pd.DataFrame({"return": r, "Historical": hist, "Parametric": param, "EWMA": ewma,
                       "Filtered HS": fhs})
    return df.dropna()


def kupiec_pof(exceptions: int, n: int, conf: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test. Returns (LR, p-value)."""
    p = 1 - conf
    x = exceptions
    if x == 0:
        lr = -2 * n * np.log(1 - p)
    elif x == n:
        lr = -2 * n * np.log(p)
    else:
        phat = x / n
        lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p)
                   - (n - x) * np.log(1 - phat) - x * np.log(phat))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def christoffersen_independence(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen (1998) test that VaR exceptions are not clustered in time."""
    h = hits.astype(int)
    prev, curr = h[:-1], h[1:]
    n00 = np.sum((prev == 0) & (curr == 0)); n01 = np.sum((prev == 0) & (curr == 1))
    n10 = np.sum((prev == 1) & (curr == 0)); n11 = np.sum((prev == 1) & (curr == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p, a, b):  # log-likelihood of a non-events and b events with prob p
        return (a * np.log(1 - p) if a else 0.0) + (b * np.log(p) if b else 0.0)

    if pi in (0, 1):
        return 0.0, 1.0
    lr = -2 * (ll(pi, n00 + n10, n01 + n11) - ll(pi0, n00, n01) - ll(pi1, n10, n11))
    return float(lr), float(1 - stats.chi2.cdf(lr, 1))


def basel_zone(exceptions_250: int) -> str:
    """Basel traffic-light zone for exceptions in the latest 250 days at 99% VaR."""
    if exceptions_250 <= 4:
        return "Green"
    if exceptions_250 <= 9:
        return "Yellow"
    return "Red"


def backtest(forecasts: pd.DataFrame, conf: float = 0.99) -> pd.DataFrame:
    rows = []
    r = forecasts["return"]
    for model in [c for c in forecasts.columns if c != "return"]:
        hits = (r < -forecasts[model]).values
        n, x = len(hits), int(hits.sum())
        lr_pof, p_pof = kupiec_pof(x, n, conf)
        lr_ind, p_ind = christoffersen_independence(hits)
        lr_cc = lr_pof + lr_ind
        rows.append({
            "Model": model,
            "Observations": n,
            "Exceptions": x,
            "Expected": round(n * (1 - conf), 1),
            "Exception Rate": x / n,
            "Kupiec p-value": p_pof,
            "Independence p-value": p_ind,
            "Cond. Coverage p-value": float(1 - stats.chi2.cdf(lr_cc, 2)),
            "Last-250d Exceptions": int(hits[-250:].sum()),
            "Basel Zone": basel_zone(int(hits[-250:].sum())),
            "Verdict": "PASS" if p_pof > 0.05 and p_ind > 0.05 else "FAIL",
        })
    return pd.DataFrame(rows).set_index("Model")
