"""Historical and hypothetical stress testing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def historical_scenarios(returns: pd.DataFrame, weights: dict,
                         bench_w: dict | None = None,
                         scenarios: dict = config.HISTORICAL_SCENARIOS,
                         notional: float = config.NOTIONAL) -> pd.DataFrame:
    """Replay the realized asset moves of each crisis window on today's portfolio."""
    w = pd.Series(weights)
    rows = []
    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end]
        if len(window) < 3:
            continue
        cum = (1 + window).prod() - 1
        port = float((w * cum.reindex(w.index)).sum())
        worst = (cum.reindex(w.index) * w).idxmin()
        row = {
            "Scenario": name, "Start": start, "End": end, "Trading Days": len(window),
            "Portfolio Return": port, "P&L ($)": port * notional,
            "Largest Detractor": worst,
        }
        if bench_w:
            b = pd.Series(bench_w)
            row["Benchmark Return"] = float((b * cum.reindex(b.index)).sum())
            row["Relative"] = port - row["Benchmark Return"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Scenario")


def scenario_asset_moves(returns: pd.DataFrame,
                         scenarios: dict = config.HISTORICAL_SCENARIOS) -> pd.DataFrame:
    out = {}
    for name, (start, end) in scenarios.items():
        window = returns.loc[start:end]
        if len(window) >= 3:
            out[name] = (1 + window).prod() - 1
    return pd.DataFrame(out).T


def regime_sample(returns: pd.DataFrame, regime: str = "stressed") -> pd.DataFrame:
    """Return the slice of history whose correlations a scenario should use."""
    if regime == "recent":
        return returns.iloc[-3 * config.TRADING_DAYS:]
    if regime == "stressed" and "IVV" in returns:
        eq = (1 + returns["IVV"]).cumprod()
        stressed = returns[(eq / eq.cummax() - 1) < -0.10]
        if len(stressed) > 250:
            return stressed
    return returns


def conditional_shock(returns: pd.DataFrame, shocks: dict[str, float],
                      regime: str = "stressed", horizon_days: int = 21) -> pd.Series:
    """Propagate shocks on driver assets to all others using the conditional mean of a
    multivariate normal:  E[r_o | r_d = s] = mu_o + S_od S_dd^-1 (s - mu_d).

    Covariance is scaled to the scenario horizon and estimated on the chosen regime:
    "stressed" (S&P drawdown > 10%), "recent" (last 3 years) or "full".
    """
    sample = regime_sample(returns, regime)
    cov = sample.cov() * horizon_days
    mu = sample.mean() * horizon_days

    drivers = list(shocks)
    others = [c for c in returns.columns if c not in drivers]
    s = pd.Series(shocks)
    S_dd = cov.loc[drivers, drivers].values
    S_od = cov.loc[others, drivers].values
    cond = mu[others].values + S_od @ np.linalg.solve(S_dd, (s - mu[drivers]).values)
    return pd.concat([s, pd.Series(cond, index=others)]).reindex(returns.columns)


def hypothetical_scenarios(returns: pd.DataFrame, weights: dict,
                           scenarios: dict = config.HYPOTHETICAL_SCENARIOS,
                           notional: float = config.NOTIONAL) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = pd.Series(weights).reindex(returns.columns).fillna(0)
    moves, rows = {}, []
    for name, (shocks, regime) in scenarios.items():
        m = conditional_shock(returns, shocks, regime)
        moves[name] = m
        port = float((w * m).sum())
        rows.append({"Scenario": name, "Correlation Regime": regime,
                     "Portfolio Return": port, "P&L ($)": port * notional})
    return pd.DataFrame(rows).set_index("Scenario"), pd.DataFrame(moves).T
