"""Portfolio construction: covariance shrinkage, optimizers, efficient frontier and a
walk-forward (out-of-sample) backtest with transaction costs.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import MAX_WEIGHT, TRADING_DAYS, TRANSACTION_COST_BPS


# --------------------------------------------------------------------------- estimation
def ledoit_wolf(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit & Wolf (2004) shrinkage toward a scaled identity. Returns (cov, intensity)."""
    X = returns.values - returns.values.mean(axis=0)
    n, p = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / p
    F = mu * np.eye(p)
    d2 = np.linalg.norm(S - F, "fro") ** 2 / p
    b2_bar = sum(np.linalg.norm(np.outer(x, x) - S, "fro") ** 2 for x in X) / n**2 / p
    b2 = min(b2_bar, d2)
    delta = b2 / d2 if d2 > 0 else 1.0
    return delta * F + (1 - delta) * S, float(delta)


def annualize(mu_daily: np.ndarray, cov_daily: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return mu_daily * TRADING_DAYS, cov_daily * TRADING_DAYS


# --------------------------------------------------------------------------- optimizers
def _solve(objective: Callable, n: int, max_w: float = MAX_WEIGHT,
           extra_constraints: list | None = None) -> np.ndarray:
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}] + (extra_constraints or [])
    res = minimize(objective, np.full(n, 1 / n), method="SLSQP",
                   bounds=[(0.0, max_w)] * n, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(cov: np.ndarray, max_w: float = MAX_WEIGHT) -> np.ndarray:
    return _solve(lambda w: w @ cov @ w, len(cov), max_w)


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0,
               max_w: float = MAX_WEIGHT) -> np.ndarray:
    def neg_sharpe(w):
        return -(w @ mu - rf) / np.sqrt(w @ cov @ w)
    return _solve(neg_sharpe, len(cov), max_w)


def risk_parity(cov: np.ndarray) -> np.ndarray:
    """Equal Risk Contribution via the convex log-barrier formulation (Spinu 2013)."""
    n = len(cov)
    b = np.full(n, 1 / n)
    c = cov / np.mean(np.diag(cov))  # rescale: weights are scale-invariant, solver isn't
    obj = lambda y: 0.5 * y @ c @ y - b @ np.log(y)
    grad = lambda y: c @ y - b / y
    res = minimize(obj, np.full(n, 1.0), jac=grad, method="L-BFGS-B",
                   bounds=[(1e-10, None)] * n, options={"ftol": 1e-15, "gtol": 1e-12})
    return res.x / res.x.sum()


def max_diversification(cov: np.ndarray, max_w: float = MAX_WEIGHT) -> np.ndarray:
    vol = np.sqrt(np.diag(cov))
    return _solve(lambda w: -(w @ vol) / np.sqrt(w @ cov @ w), len(cov), max_w)


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 40,
                       max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    w_min = min_variance(cov, max_w)
    r_lo = w_min @ mu
    # highest achievable return under the weight cap: fill best assets up to the cap
    order, w_hi, left = np.argsort(-mu), np.zeros(len(mu)), 1.0
    for i in order:
        w_hi[i] = min(max_w, left)
        left -= w_hi[i]
    r_hi = w_hi @ mu
    rows = []
    for target in np.linspace(r_lo, r_hi * 0.999, n_points):
        w = _solve(lambda w: w @ cov @ w, len(cov), max_w,
                   [{"type": "eq", "fun": lambda w, t=target: w @ mu - t}])
        rows.append({"return": w @ mu, "volatility": np.sqrt(w @ cov @ w), **dict(enumerate(w))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- backtest
def _strategy_weights(name: str, window: pd.DataFrame, static: dict | None) -> np.ndarray:
    cov_d, _ = ledoit_wolf(window)
    mu, cov = annualize(window.mean().values, cov_d)
    n = window.shape[1]
    if name == "Equal Weight":
        return np.full(n, 1 / n)
    if name == "Min Variance":
        return min_variance(cov)
    if name == "Max Sharpe":
        # shrink expected returns toward the cross-sectional mean: raw historical means are
        # notoriously noisy and push mean-variance toward corner solutions
        mu_shrunk = 0.5 * mu + 0.5 * mu.mean()
        return max_sharpe(mu_shrunk, cov)
    if name == "Risk Parity":
        return risk_parity(cov)
    if name == "Max Diversification":
        return max_diversification(cov)
    if static is not None:
        return pd.Series(static).reindex(window.columns).fillna(0).values
    raise ValueError(name)


def walk_forward(returns: pd.DataFrame, strategies: list[str], lookback: int = 756,
                 static: dict[str, dict] | None = None,
                 cost_bps: float = TRANSACTION_COST_BPS) -> dict[str, dict]:
    """Monthly-rebalanced backtest. Weights at each month-end use only the trailing
    `lookback` days (no look-ahead). Turnover is charged `cost_bps` per side.
    """
    static = static or {}
    idx = returns.index
    month_ends = returns.groupby(idx.to_period("M")).tail(1).index
    month_ends = [d for d in month_ends if idx.get_loc(d) >= lookback]
    results = {}
    for name in strategies + list(static):
        daily, weights_hist, turnover_hist = [], {}, []
        w_cur = None
        for k, rebal in enumerate(month_ends):
            loc = idx.get_loc(rebal)
            window = returns.iloc[loc - lookback + 1: loc + 1]
            w_new = _strategy_weights(name, window, static.get(name))
            turnover = np.abs(w_new - (w_cur if w_cur is not None else 0)).sum()
            turnover_hist.append(turnover)
            weights_hist[rebal] = w_new
            end = idx.get_loc(month_ends[k + 1]) if k + 1 < len(month_ends) else len(idx) - 1
            period = returns.iloc[loc + 1: end + 1]
            w = w_new.copy()
            for j, (d, r) in enumerate(zip(period.index, period.values)):
                pr = float(w @ r)
                if j == 0:
                    pr -= turnover * cost_bps / 1e4
                daily.append((d, pr))
                w = w * (1 + r) / (1 + w @ r)
            w_cur = w
        ser = pd.Series(dict(daily)).sort_index()
        results[name] = {
            "returns": ser,
            "weights": pd.DataFrame(weights_hist, index=returns.columns).T,
            "avg_turnover": float(np.mean(turnover_hist[1:])) if len(turnover_hist) > 1 else 0.0,
        }
    return results
