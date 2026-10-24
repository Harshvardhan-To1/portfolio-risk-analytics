"""Fama-French 5-factor + Momentum attribution of portfolio returns."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import TRADING_DAYS

FACTOR_LABELS = {
    "MKT_RF": "Market (Mkt-RF)",
    "SMB": "Size (SMB)",
    "HML": "Value (HML)",
    "RMW": "Profitability (RMW)",
    "CMA": "Investment (CMA)",
    "MOM": "Momentum (MOM)",
}


def factor_regression(port: pd.Series, factors: pd.DataFrame, hac_lags: int = 5) -> dict:
    """OLS of portfolio excess returns on factors with Newey-West (HAC) standard errors."""
    cols = [c for c in FACTOR_LABELS if c in factors.columns]
    df = pd.concat([port.rename("port"), factors], axis=1, join="inner").dropna()
    y = df["port"] - df["RF"]
    X = sm.add_constant(df[cols])
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    coef = model.params.rename({"const": "Alpha"})
    table = pd.DataFrame({
        "Coefficient": coef,
        "t-stat": model.tvalues.rename({"const": "Alpha"}),
        "p-value": model.pvalues.rename({"const": "Alpha"}),
    })
    table.loc["Alpha", "Coefficient"] *= TRADING_DAYS  # report alpha annualized

    # Variance decomposition: how much of portfolio variance each factor explains.
    betas = model.params[cols].values
    F = df[cols]
    cov_f = F.cov().values
    total_var = y.var()
    factor_contrib = betas * (cov_f @ betas)
    risk_decomp = pd.Series(factor_contrib / total_var, index=cols)
    risk_decomp["Idiosyncratic"] = 1 - risk_decomp.sum()

    # Return attribution: average factor premium x exposure (annualized).
    ret_attr = pd.Series(betas * F.mean().values * TRADING_DAYS, index=cols)
    ret_attr["Alpha"] = model.params["const"] * TRADING_DAYS
    ret_attr["Risk-free"] = df["RF"].mean() * TRADING_DAYS

    return {
        "table": table.rename(index=FACTOR_LABELS),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "n_obs": int(model.nobs),
        "risk_decomposition": risk_decomp.rename(index=FACTOR_LABELS),
        "return_attribution": ret_attr.rename(index=FACTOR_LABELS),
    }


def rolling_factor_betas(port: pd.Series, factors: pd.DataFrame, window: int = 252,
                         step: int = 21) -> pd.DataFrame:
    """Rolling OLS factor exposures, sampled every `step` days."""
    cols = [c for c in FACTOR_LABELS if c in factors.columns]
    df = pd.concat([port.rename("port"), factors], axis=1, join="inner").dropna()
    y = (df["port"] - df["RF"]).values
    X = np.column_stack([np.ones(len(df)), df[cols].values])
    out = {}
    for end in range(window, len(df) + 1, step):
        b, *_ = np.linalg.lstsq(X[end - window:end], y[end - window:end], rcond=None)
        out[df.index[end - 1]] = b[1:]
    return pd.DataFrame(out, index=cols).T.rename(columns=FACTOR_LABELS)
