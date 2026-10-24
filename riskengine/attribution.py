"""Risk decomposition and Brinson-Fachler performance attribution."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .config import TRADING_DAYS


def risk_contributions(weights: pd.Series, cov_daily: pd.DataFrame) -> pd.DataFrame:
    """Euler decomposition: each asset's contribution to total portfolio volatility.

    MCR_i = (Sigma w)_i / sigma_p ; CTR_i = w_i * MCR_i ; sum(CTR) = sigma_p.
    """
    w = weights.reindex(cov_daily.index).fillna(0).values
    cov = cov_daily.values * TRADING_DAYS
    sigma = np.sqrt(w @ cov @ w)
    mcr = cov @ w / sigma
    ctr = w * mcr
    df = pd.DataFrame({
        "Weight": w,
        "Standalone Vol": np.sqrt(np.diag(cov)),
        "Marginal Contribution": mcr,
        "Contribution to Vol": ctr,
        "% of Risk": ctr / sigma,
    }, index=cov_daily.index)
    df["Risk / Capital Ratio"] = df["% of Risk"] / df["Weight"].replace(0, np.nan)
    return df


def component_var(weights: pd.Series, asset_returns: pd.DataFrame, conf: float = 0.99) -> pd.Series:
    """Parametric component VaR (sums to total portfolio VaR) in return units."""
    from scipy.stats import norm

    w = weights.reindex(asset_returns.columns).fillna(0).values
    cov = asset_returns.cov().values
    sigma = np.sqrt(w @ cov @ w)
    z = -norm.ppf(1 - conf)
    return pd.Series(w * (cov @ w) / sigma * z, index=asset_returns.columns)


def brinson_fachler(returns: pd.DataFrame, port_w: dict, bench_w: dict,
                    freq: str = "M") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asset-class level Brinson-Fachler attribution, computed each period and summed.

    Segment benchmark return for a class the benchmark doesn't hold uses the equal-weight
    return of that class's assets, so allocation effects stay well defined.
    """
    classes = {t: v[1] for t, v in config.UNIVERSE.items() if t in returns.columns}
    pw, bw = pd.Series(port_w), pd.Series(bench_w)
    period_r = (1 + returns).groupby(returns.index.to_period(freq)).prod() - 1

    rows = []
    for period, r in period_r.iterrows():
        bench_total = float((bw * r.reindex(bw.index)).sum())
        for cls in sorted(set(classes.values())):
            members = [t for t, c in classes.items() if c == cls]
            wp, wb = pw.reindex(members).fillna(0), bw.reindex(members).fillna(0)
            Wp, Wb = wp.sum(), wb.sum()
            Rp = float((wp * r[members]).sum() / Wp) if Wp > 0 else 0.0
            Rb = float((wb * r[members]).sum() / Wb) if Wb > 0 else float(r[members].mean())
            rows.append({
                "period": period.to_timestamp(), "Asset Class": cls,
                "Allocation": (Wp - Wb) * (Rb - bench_total),
                "Selection": Wb * (Rp - Rb),
                "Interaction": (Wp - Wb) * (Rp - Rb),
            })
    detail = pd.DataFrame(rows)
    detail["Total Active"] = detail[["Allocation", "Selection", "Interaction"]].sum(axis=1)
    totals = detail.groupby("Asset Class")[["Allocation", "Selection", "Interaction",
                                             "Total Active"]].sum()
    totals.loc["Total"] = totals.sum()
    return totals, detail
