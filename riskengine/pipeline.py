"""End-to-end analytics pipeline: ingest -> warehouse -> analytics -> results bundle."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import attribution, config, db, factors, metrics, optimize, stress
from .data import fetch_factors, fetch_prices
from . import var as varlib

log = logging.getLogger(__name__)

CLIENT, BENCH = "Client Portfolio", "Benchmark 60/40"
OPT_STRATEGIES = ["Equal Weight", "Min Variance", "Max Sharpe", "Risk Parity", "Max Diversification"]


@dataclass
class Results:
    source: str
    prices: pd.DataFrame
    returns: pd.DataFrame
    factors: pd.DataFrame | None
    port: pd.Series
    bench: pd.Series
    rf: pd.Series | float
    summary: pd.DataFrame
    drawdown: dict
    calendar: pd.DataFrame
    var_table: pd.DataFrame
    var_forecasts: pd.DataFrame
    var_backtest: pd.DataFrame
    mc_pnl: np.ndarray
    risk_contrib: pd.DataFrame
    component_var: pd.Series
    shrinkage: float
    frontier: pd.DataFrame
    point_portfolios: dict
    walk_forward: dict
    wf_summary: pd.DataFrame
    hist_stress: pd.DataFrame
    stress_moves: pd.DataFrame
    hypo_stress: pd.DataFrame
    hypo_moves: pd.DataFrame
    brinson: pd.DataFrame
    factor_model: dict | None
    rolling_betas: pd.DataFrame | None
    sql: dict = field(default_factory=dict)


def run(refresh: bool = True) -> Results:
    # ---------------------------------------------------------------- 1. ingest
    conn = db.connect()
    if refresh or not conn.execute("SELECT COUNT(*) FROM fact_price").fetchone()[0]:
        prices, source = fetch_prices()
        ff = fetch_factors()
        rets = metrics.simple_returns(prices)
        db.load_market_data(conn, prices, rets, ff, source)
    prices, ff, source = read_warehouse(conn)
    rets = metrics.simple_returns(prices)
    log.info("Universe: %d assets, %d days (%s to %s), source=%s", rets.shape[1], len(rets),
             rets.index[0].date(), rets.index[-1].date(), source)

    # ---------------------------------------------------------------- 2. performance
    port = metrics.portfolio_returns(rets, config.PORTFOLIO, "M").rename(CLIENT)
    bench = metrics.portfolio_returns(rets, config.BENCHMARK, "M").rename(BENCH)
    rf = ff["RF"] if ff is not None else config.RISK_FREE_FALLBACK
    summary = pd.DataFrame({
        CLIENT: metrics.summary(port, bench, rf),
        BENCH: metrics.summary(bench, None, rf),
    })

    # ---------------------------------------------------------------- 3. VaR
    w_vec = pd.Series(config.PORTFOLIO).reindex(rets.columns).fillna(0).values
    var_tbl = varlib.var_table(port, rets, w_vec)
    fc = varlib.rolling_var_forecasts(port, config.VAR_CONFIDENCE, config.VAR_WINDOW)
    bt = varlib.backtest(fc, config.VAR_CONFIDENCE)
    mc = varlib.monte_carlo_var(rets, w_vec, config.VAR_CONFIDENCE)

    # ---------------------------------------------------------------- 4. risk decomposition
    cov_lw, delta = optimize.ledoit_wolf(rets)
    cov_lw = pd.DataFrame(cov_lw, index=rets.columns, columns=rets.columns)
    rc = attribution.risk_contributions(pd.Series(config.PORTFOLIO), cov_lw)
    cvar = attribution.component_var(pd.Series(config.PORTFOLIO), rets) * config.NOTIONAL

    # ---------------------------------------------------------------- 5. optimization
    mu, cov = optimize.annualize(rets.mean().values, cov_lw.values)
    frontier = optimize.efficient_frontier(mu, cov)
    frontier = frontier.rename(columns=dict(enumerate(rets.columns)))
    rf_ann = float(ff["RF"].iloc[-252:].mean() * config.TRADING_DAYS) if ff is not None \
        else config.RISK_FREE_FALLBACK
    pts = {
        CLIENT: w_vec,
        BENCH: pd.Series(config.BENCHMARK).reindex(rets.columns).fillna(0).values,
        "Min Variance": optimize.min_variance(cov),
        "Max Sharpe": optimize.max_sharpe(mu, cov, rf_ann),
        "Risk Parity": optimize.risk_parity(cov),
        "Equal Weight": np.full(len(mu), 1 / len(mu)),
    }
    point_portfolios = {k: {"weights": pd.Series(w, index=rets.columns), "return": float(w @ mu),
                            "volatility": float(np.sqrt(w @ cov @ w))} for k, w in pts.items()}

    wf = optimize.walk_forward(rets, OPT_STRATEGIES,
                               static={CLIENT: config.PORTFOLIO, BENCH: config.BENCHMARK})
    common_start = max(v["returns"].index[0] for v in wf.values())
    wf_rows = {}
    for name, res in wf.items():
        r = res["returns"].loc[common_start:]
        s = metrics.summary(r, wf[BENCH]["returns"].loc[common_start:] if name != BENCH else None, rf)
        s["Avg Monthly Turnover"] = res["avg_turnover"]
        wf_rows[name] = s
    wf_summary = pd.DataFrame(wf_rows)

    # ---------------------------------------------------------------- 6. stress tests
    hist = stress.historical_scenarios(rets, config.PORTFOLIO, config.BENCHMARK)
    moves = stress.scenario_asset_moves(rets)
    hypo, hypo_moves = stress.hypothetical_scenarios(rets, config.PORTFOLIO)

    # ---------------------------------------------------------------- 7. attribution
    brinson, _ = attribution.brinson_fachler(rets, config.PORTFOLIO, config.BENCHMARK)

    fm = rb = None
    if ff is not None:
        fm = factors.factor_regression(port, ff)
        rb = factors.rolling_factor_betas(port, ff)

    # ---------------------------------------------------------------- 8. SQL layer
    navs = {name: metrics.nav(res["returns"].loc[common_start:]) for name, res in wf.items()}
    rets_map = {name: res["returns"].loc[common_start:] for name, res in wf.items()}
    navs[CLIENT], rets_map[CLIENT] = metrics.nav(port), port
    navs[BENCH], rets_map[BENCH] = metrics.nav(bench), bench
    db.load_nav(conn, navs, rets_map)
    sql_results = db.run_sql_file(conn, config.ROOT / "sql" / "analytics.sql")
    conn.close()

    return Results(
        source=source, prices=prices, returns=rets, factors=ff, port=port, bench=bench, rf=rf,
        summary=summary, drawdown=metrics.max_drawdown_details(port),
        calendar=metrics.calendar_returns(port), var_table=var_tbl, var_forecasts=fc,
        var_backtest=bt, mc_pnl=mc["pnl"], risk_contrib=rc, component_var=cvar,
        shrinkage=delta, frontier=frontier, point_portfolios=point_portfolios,
        walk_forward=wf, wf_summary=wf_summary, hist_stress=hist, stress_moves=moves,
        hypo_stress=hypo, hypo_moves=hypo_moves, brinson=brinson, factor_model=fm,
        rolling_betas=rb, sql=sql_results,
    )


def read_warehouse(conn) -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    prices = pd.read_sql_query("SELECT date, ticker, adj_close FROM fact_price", conn,
                               parse_dates=["date"]).pivot(index="date", columns="ticker",
                                                           values="adj_close")
    prices = prices[[t for t in config.TICKERS if t in prices.columns]]
    prices.columns.name = None
    ff = pd.read_sql_query("SELECT * FROM fact_factor", conn, parse_dates=["date"],
                           index_col="date")
    ff = None if ff.empty else ff.dropna(axis=1, how="all")
    return prices, ff, db.read_meta(conn).get("data_source", "unknown")
