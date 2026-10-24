"""Performance and risk statistics for return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TRADING_DAYS


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna(how="all")


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float],
                      rebalance: str | None = "M") -> pd.Series:
    """Portfolio returns, rebalanced to target weights each period ("M", "Q") and
    drifting with market moves in between. rebalance=None means constant weights.
    """
    w_target = pd.Series(weights).reindex(returns.columns).fillna(0.0).values
    if rebalance is None:
        return pd.Series(returns.values @ w_target, index=returns.index, name="portfolio")
    period = returns.index.to_period(rebalance)
    out = np.empty(len(returns))
    w, prev = w_target.copy(), None
    for i, (p, r) in enumerate(zip(period, returns.values)):
        if p != prev:
            w, prev = w_target.copy(), p
        out[i] = w @ r
        w = w * (1 + r) / (1 + out[i])
    return pd.Series(out, index=returns.index, name="portfolio")


def nav(returns: pd.Series, start: float = 100.0) -> pd.Series:
    return start * (1 + returns).cumprod()


def drawdown(returns: pd.Series) -> pd.Series:
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown_details(returns: pd.Series) -> dict:
    dd = drawdown(returns)
    trough = dd.idxmin()
    peak = (1 + returns).cumprod().loc[:trough].idxmax()
    after = dd.loc[trough:]
    recovered = after[after >= 0]
    recovery = recovered.index[0] if len(recovered) else None
    return {
        "max_drawdown": float(dd.min()),
        "peak": peak, "trough": trough, "recovery": recovery,
        "days_to_trough": int((trough - peak).days),
        "days_to_recover": int((recovery - trough).days) if recovery is not None else None,
    }


def annualized_return(r: pd.Series) -> float:
    years = len(r) / TRADING_DAYS
    return float((1 + r).prod() ** (1 / years) - 1)


def annualized_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = r - (rf.reindex(r.index).fillna(0.0) if isinstance(rf, pd.Series) else rf / TRADING_DAYS)
    return float(ex.mean() / ex.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sortino(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    ex = r - (rf.reindex(r.index).fillna(0.0) if isinstance(rf, pd.Series) else rf / TRADING_DAYS)
    downside = np.sqrt((np.minimum(ex, 0) ** 2).mean())
    return float(ex.mean() / downside * np.sqrt(TRADING_DAYS))


def beta(r: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([r, bench], axis=1).dropna()
    c = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)
    return float(c[0, 1] / c[1, 1])


def tracking_error(r: pd.Series, bench: pd.Series) -> float:
    return annualized_vol((r - bench).dropna())


def information_ratio(r: pd.Series, bench: pd.Series) -> float:
    active = (r - bench).dropna()
    return float(active.mean() / active.std(ddof=1) * np.sqrt(TRADING_DAYS))


def summary(r: pd.Series, bench: pd.Series | None = None,
            rf: pd.Series | float = 0.0) -> dict[str, float]:
    ann_ret = annualized_return(r)
    mdd = max_drawdown_details(r)["max_drawdown"]
    out = {
        "Annualized Return": ann_ret,
        "Annualized Volatility": annualized_vol(r),
        "Sharpe Ratio": sharpe(r, rf),
        "Sortino Ratio": sortino(r, rf),
        "Max Drawdown": mdd,
        "Calmar Ratio": ann_ret / abs(mdd) if mdd else np.nan,
        "Skewness": float(r.skew()),
        "Excess Kurtosis": float(r.kurt()),
        "Best Day": float(r.max()),
        "Worst Day": float(r.min()),
        "% Positive Days": float((r > 0).mean()),
    }
    if bench is not None:
        out.update({
            "Beta vs Benchmark": beta(r, bench),
            "Tracking Error": tracking_error(r, bench),
            "Information Ratio": information_ratio(r, bench),
        })
    return out


def rolling_vol(r: pd.Series, window: int = 63) -> pd.Series:
    return r.rolling(window).std() * np.sqrt(TRADING_DAYS)


def rolling_beta(r: pd.Series, bench: pd.Series, window: int = 252) -> pd.Series:
    return r.rolling(window).cov(bench) / bench.rolling(window).var()


def calendar_returns(r: pd.Series) -> pd.DataFrame:
    """Monthly return grid (years x months) plus a full-year column."""
    monthly = (1 + r).resample("ME").prod() - 1
    grid = monthly.to_frame("ret")
    grid["Year"], grid["Month"] = grid.index.year, grid.index.strftime("%b")
    table = grid.pivot(index="Year", columns="Month", values="ret")
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table = table.reindex(columns=months)
    table["Year"] = (1 + r).groupby(r.index.year).prod() - 1
    return table
