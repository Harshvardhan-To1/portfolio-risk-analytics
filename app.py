"""Interactive risk dashboard.   Run:  streamlit run app.py

Reads the SQLite warehouse built by run_pipeline.py and recomputes all analytics live
for whatever portfolio the user builds in the sidebar.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from riskengine import attribution, config, db, factors, metrics, optimize, stress
from riskengine import var as varlib
from riskengine.pipeline import read_warehouse
from riskengine.report import COLORS, CRITICAL

st.set_page_config(page_title="Portfolio Risk Analytics", layout="wide")


@st.cache_data
def load():
    conn = db.connect()
    prices, ff, source = read_warehouse(conn)
    conn.close()
    return metrics.simple_returns(prices), ff, source


try:
    rets, ff, source = load()
except Exception:
    st.error("No data found. Run `python run_pipeline.py` first to build the warehouse.")
    st.stop()
if rets.empty:
    st.error("Warehouse is empty. Run `python run_pipeline.py` first.")
    st.stop()

# ------------------------------------------------------------------ sidebar: build portfolio
st.sidebar.header("Build a portfolio")
preset = st.sidebar.selectbox("Start from", ["Client Portfolio", "60/40", "Equal Weight",
                                             "Risk Parity", "Min Variance"])
cov_lw = optimize.ledoit_wolf(rets)[0] * config.TRADING_DAYS
presets = {
    "Client Portfolio": pd.Series(config.PORTFOLIO),
    "60/40": pd.Series(config.BENCHMARK),
    "Equal Weight": pd.Series(1 / len(rets.columns), index=rets.columns),
    "Risk Parity": pd.Series(optimize.risk_parity(cov_lw), index=rets.columns),
    "Min Variance": pd.Series(optimize.min_variance(cov_lw), index=rets.columns),
}
base = presets[preset].reindex(rets.columns).fillna(0)
raw = {t: st.sidebar.slider(f"{t} · {config.UNIVERSE[t][2]}", 0, 100, int(round(base[t] * 100)),
                            key=f"{preset}-{t}") for t in rets.columns}
total = sum(raw.values())
if total == 0:
    st.sidebar.error("Allocate at least one asset.")
    st.stop()
weights = {t: v / total for t, v in raw.items()}
st.sidebar.caption(f"Raw total {total}% — normalised to 100%.")
conf = st.sidebar.select_slider("VaR confidence", [0.95, 0.975, 0.99, 0.995], value=0.99)
notional = st.sidebar.number_input("Notional ($M)", 1, 10_000, 100) * 1e6
lo, hi = rets.index[0].date(), rets.index[-1].date()
period = st.sidebar.slider("Analysis window", lo, hi, (lo, hi))

r = rets.loc[str(period[0]):str(period[1])]
port = metrics.portfolio_returns(r, weights, "M")
bench = metrics.portfolio_returns(r, config.BENCHMARK, "M")
rf = ff["RF"] if ff is not None else config.RISK_FREE_FALLBACK
w_vec = pd.Series(weights).reindex(r.columns).values

# ------------------------------------------------------------------ header + KPIs
st.title("Multi-Asset Portfolio Risk Analytics")
st.caption(f"{len(r):,} trading days · {period[0]} → {period[1]} · data source: {source}")
if source == "synthetic":
    st.warning("Running on synthetic data — re-run the pipeline with internet access.")

s, sb = metrics.summary(port, bench, rf), metrics.summary(bench, None, rf)
hvar = varlib.historical_var(port, conf)
c = st.columns(6)
c[0].metric("Ann. return", f"{s['Annualized Return']:.2%}",
            f"{s['Annualized Return'] - sb['Annualized Return']:+.2%} vs 60/40")
c[1].metric("Volatility", f"{s['Annualized Volatility']:.2%}",
            f"{s['Annualized Volatility'] - sb['Annualized Volatility']:+.2%}", delta_color="inverse")
c[2].metric("Sharpe", f"{s['Sharpe Ratio']:.2f}", f"{s['Sharpe Ratio'] - sb['Sharpe Ratio']:+.2f}")
c[3].metric("Max drawdown", f"{s['Max Drawdown']:.1%}",
            f"{s['Max Drawdown'] - sb['Max Drawdown']:+.1%}")
c[4].metric(f"1d VaR {conf:.1%}", f"${hvar * notional / 1e6:,.2f}M")
c[5].metric(f"1d ES {conf:.1%}", f"${varlib.historical_es(port, conf) * notional / 1e6:,.2f}M")


def style(fig, h=380, pct_y=True):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=-0.15))
    if pct_y:
        fig.update_yaxes(tickformat=".0%")
    return fig


tabs = st.tabs(["Performance", "VaR & Backtest", "Risk Decomposition", "Factors",
                "Stress Tests", "Optimizer"])

with tabs[0]:
    fig = go.Figure()
    for name, x in (("Your portfolio", port), ("60/40", bench)):
        fig.add_trace(go.Scatter(x=x.index, y=(1 + x).cumprod() - 1, name=name,
                                 line=dict(color=COLORS["Client Portfolio" if name != "60/40"
                                                          else "Benchmark 60/40"])))
    st.plotly_chart(style(fig), width="stretch")
    dd = metrics.drawdown(port)
    fig = go.Figure(go.Scatter(x=dd.index, y=dd, fill="tozeroy", name="Drawdown",
                               line=dict(color=COLORS["Client Portfolio"])))
    st.plotly_chart(style(fig, 240), width="stretch")
    st.dataframe(pd.DataFrame({"Your portfolio": s, "60/40": sb}).style.format("{:.3f}"),
                 width="stretch")
    st.subheader("Monthly returns")
    st.dataframe(metrics.calendar_returns(port).iloc[::-1].style.format("{:.1%}", na_rep="")
                 .background_gradient(cmap="RdBu", vmin=-0.08, vmax=0.08), width="stretch")

with tabs[1]:
    st.dataframe(varlib.var_table(port, r, w_vec).style.format("{:.2%}"), width="stretch")
    window = st.slider("Backtest look-back (days)", 250, 1000, config.VAR_WINDOW, 50)
    if len(port) > window + 250:
        fc = varlib.rolling_var_forecasts(port, conf, window)
        bt = varlib.backtest(fc, conf)
        st.dataframe(bt, width="stretch")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=fc.index, y=fc["return"], name="Daily P&L", marker_color="#c3c2b7"))
        fig.add_trace(go.Scatter(x=fc.index, y=-fc["Historical"], name="Historical VaR",
                                 line=dict(color=COLORS["Client Portfolio"])))
        fig.add_trace(go.Scatter(x=fc.index, y=-fc["Filtered HS"], name="Filtered HS VaR",
                                 line=dict(color=COLORS["Max Diversification"])))
        hits = fc[fc["return"] < -fc["Historical"]]
        fig.add_trace(go.Scatter(x=hits.index, y=hits["return"], mode="markers", name="Exception",
                                 marker=dict(color=CRITICAL, size=7)))
        st.plotly_chart(style(fig, 420), width="stretch")
    else:
        st.info("Widen the analysis window to run a backtest.")

with tabs[2]:
    cov = pd.DataFrame(optimize.ledoit_wolf(r)[0], index=r.columns, columns=r.columns)
    rc = attribution.risk_contributions(pd.Series(weights), cov)
    rcs = rc.sort_values("% of Risk")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=rcs.index, x=rcs["Weight"], orientation="h", name="Capital",
                         marker_color="#86b6ef"))
    fig.add_trace(go.Bar(y=rcs.index, x=rcs["% of Risk"], orientation="h", name="Risk",
                         marker_color=COLORS["Client Portfolio"]))
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(style(fig, 420, pct_y=False), width="stretch")
    st.dataframe(rc.style.format("{:.3f}"), width="stretch")
    cv = attribution.component_var(pd.Series(weights), r, conf) * notional
    st.subheader(f"Component VaR ({conf:.1%}) — sums to total parametric VaR")
    st.bar_chart(cv)

with tabs[3]:
    if ff is None:
        st.info("Fama-French factors unavailable.")
    else:
        fm = factors.factor_regression(port, ff)
        a, b = st.columns(2)
        a.metric("R²", f"{fm['r_squared']:.1%}")
        a.dataframe(fm["table"].style.format("{:.4f}"), width="stretch")
        b.dataframe(fm["risk_decomposition"].to_frame("Share of variance").style.format("{:.1%}"),
                    width="stretch")
        rb = factors.rolling_factor_betas(port, ff)
        fig = go.Figure()
        for i, col in enumerate(rb.columns):
            fig.add_trace(go.Scatter(x=rb.index, y=rb[col], name=col,
                                     line=dict(color=list(COLORS.values())[i])))
        st.plotly_chart(style(fig, pct_y=False), width="stretch")

with tabs[4]:
    hs = stress.historical_scenarios(rets, weights, config.BENCHMARK, notional=notional)
    st.subheader("Historical scenarios")
    st.dataframe(hs.style.format({"Portfolio Return": "{:.1%}", "Benchmark Return": "{:.1%}",
                                  "Relative": "{:+.1%}", "P&L ($)": "${:,.0f}"}),
                 width="stretch")
    st.subheader("Custom shock")
    cols = st.columns(4)
    driver = cols[0].selectbox("Shock asset", list(rets.columns), index=0)
    size = cols[1].slider("Shock size (%)", -50, 50, -20) / 100
    regime = cols[2].radio("Correlation regime", ["stressed", "recent", "full"],
                           help="stressed = S&P >10% below peak · recent = last 3 years")
    moves = stress.conditional_shock(rets, {driver: size}, regime)
    pnl = float((pd.Series(weights) * moves).sum())
    cols[3].metric("Portfolio impact", f"{pnl:.2%}", f"${pnl * notional:,.0f}")
    fig = go.Figure(go.Bar(x=moves.index, y=moves, marker_color=[
        "#e34948" if v < 0 else "#2a78d6" for v in moves]))
    st.plotly_chart(style(fig, 320), width="stretch")

with tabs[5]:
    cov_d = optimize.ledoit_wolf(r)[0]
    mu, cov = optimize.annualize(r.mean().values, cov_d)
    cap = st.slider("Max weight per asset", 0.15, 1.0, config.MAX_WEIGHT, 0.05)
    fr = optimize.efficient_frontier(mu, cov, 30, cap)
    pts = {"Your portfolio": w_vec, "Min Variance": optimize.min_variance(cov, cap),
           "Max Sharpe": optimize.max_sharpe(mu, cov, 0.0, cap),
           "Risk Parity": optimize.risk_parity(cov)}
    color_map = {"Your portfolio": COLORS["Client Portfolio"], "Min Variance": COLORS["Min Variance"],
                 "Max Sharpe": COLORS["Max Sharpe"], "Risk Parity": COLORS["Risk Parity"]}
    fig = go.Figure(go.Scatter(x=fr["volatility"], y=fr["return"], name="Frontier",
                               line=dict(color="#52514e", width=2.5)))
    for name, w in pts.items():
        fig.add_trace(go.Scatter(x=[np.sqrt(w @ cov @ w)], y=[w @ mu], mode="markers", name=name,
                                 marker=dict(size=14, color=color_map[name])))
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style(fig, 440), width="stretch")
    st.dataframe(pd.DataFrame(pts, index=r.columns).style.format("{:.1%}"),
                 width="stretch")
