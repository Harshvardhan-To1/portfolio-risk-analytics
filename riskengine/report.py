"""Static charts (PNG, for the README) and an interactive HTML risk report (Plotly)."""
from __future__ import annotations

import html
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config
from .pipeline import BENCH, CLIENT, Results

# Validated categorical palette (fixed slot order; colour follows the entity, never rank).
COLORS = {
    CLIENT: "#2a78d6", BENCH: "#eb6834", "Equal Weight": "#1baf7a", "Min Variance": "#eda100",
    "Max Sharpe": "#e87ba4", "Risk Parity": "#008300", "Max Diversification": "#6250d6",
}
NEG, POS, NEUTRAL = "#e34948", "#2a78d6", "#f0efec"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 1.6,
})


def _pct(ax, axis="y", decimals=0):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def _save(fig, name):
    config.IMG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.IMG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


# =========================================================================== static PNGs
def save_static_charts(res: Results) -> None:
    # 1. Walk-forward growth of $100
    fig, ax = plt.subplots(figsize=(10, 5))
    start = max(v["returns"].index[0] for v in res.walk_forward.values())
    for name, out in res.walk_forward.items():
        nav = 100 * (1 + out["returns"].loc[start:]).cumprod()
        lw = 2.4 if name == CLIENT else 1.4
        ax.plot(nav.index, nav, color=COLORS[name], lw=lw, label=name)
        ax.annotate(f"{nav.iloc[-1]:.0f}", (nav.index[-1], nav.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK2, va="center")
    ax.set_title("Out-of-sample growth of $100 — monthly rebalanced, 5bp costs")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    _save(fig, "01_walk_forward_growth")

    # 2. Underwater (drawdown) chart
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for name, r in ((BENCH, res.bench), (CLIENT, res.port)):
        dd = (1 + r).cumprod() / (1 + r).cumprod().cummax() - 1
        ax.plot(dd.index, dd, color=COLORS[name], lw=1.4, label=name)
    ax.fill_between(dd.index, dd, 0, color=COLORS[CLIENT], alpha=0.12)
    _pct(ax)
    ax.set_title("Drawdown from previous peak")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "02_drawdown")

    # 3. VaR backtest
    fc = res.var_forecasts
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(fc.index, fc["return"], width=1.5, color="#c3c2b7", label="Daily P&L")
    ax.plot(fc.index, -fc["Historical"], color=COLORS[CLIENT], lw=1.2, label="99% Historical VaR")
    ax.plot(fc.index, -fc["Filtered HS"], color=COLORS["Max Diversification"], lw=1.0,
            label="99% Filtered HS VaR")
    hits = fc[fc["return"] < -fc["Historical"]]
    ax.scatter(hits.index, hits["return"], s=14, color=CRITICAL, zorder=3,
               edgecolor=SURFACE, linewidth=0.8, label=f"Exceptions ({len(hits)})")
    _pct(ax, decimals=1)
    ax.set_title("VaR backtest — out-of-sample forecasts vs realised P&L")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    _save(fig, "03_var_backtest")

    # 4. Efficient frontier
    fig, ax = plt.subplots(figsize=(8, 5.4))
    fr = res.frontier
    ax.plot(fr["volatility"], fr["return"], color=INK2, lw=2, label="Efficient frontier")
    vols = np.sqrt(np.diag(res.returns.cov().values) * 252)
    mus = res.returns.mean().values * 252
    ax.scatter(vols, mus, s=22, color=MUTED, zorder=2)
    for t, v, m in zip(res.returns.columns, vols, mus):
        ax.annotate(t, (v, m), xytext=(4, 2), textcoords="offset points", fontsize=8, color=MUTED)
    for name, p in res.point_portfolios.items():
        ax.scatter(p["volatility"], p["return"], s=80, color=COLORS[name], zorder=3,
                   edgecolor=SURFACE, linewidth=2, label=name)
    _pct(ax); _pct(ax, "x")
    ax.set_xlabel("Annualised volatility"); ax.set_ylabel("Annualised return")
    ax.set_title("Efficient frontier (Ledoit-Wolf covariance, 40% cap)")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "04_efficient_frontier")

    # 5. Risk vs capital
    rc = res.risk_contrib.sort_values("% of Risk")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(rc))
    ax.barh(y - 0.2, rc["Weight"], height=0.38, color="#86b6ef", label="Capital weight")
    ax.barh(y + 0.2, rc["% of Risk"], height=0.38, color=COLORS[CLIENT], label="Share of risk")
    ax.set_yticks(y, rc.index)
    _pct(ax, "x")
    ax.set_title("Capital allocation vs risk contribution")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "05_risk_contribution")

    # 6. Stress tests
    hs = res.hist_stress.sort_values("Portfolio Return")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(hs))
    ax.barh(y + 0.2, hs["Portfolio Return"], height=0.38, color=COLORS[CLIENT], label=CLIENT)
    ax.barh(y - 0.2, hs["Benchmark Return"], height=0.38, color=COLORS[BENCH], label=BENCH)
    ax.set_yticks(y, hs.index)
    ax.axvline(0, color="#c3c2b7", lw=1)
    _pct(ax, "x")
    ax.set_title("Historical stress scenarios — cumulative return over window")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "06_stress_tests")

    # 7. Rolling stock/bond correlation (regime change)
    corr = res.returns["IVV"].rolling(126).corr(res.returns["TLT"]).dropna()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(corr.index, corr, color=COLORS[CLIENT], lw=1.4)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.fill_between(corr.index, corr, 0, where=corr > 0, color=NEG, alpha=0.18,
                    label="Positive (bonds stop hedging equities)")
    ax.fill_between(corr.index, corr, 0, where=corr <= 0, color=POS, alpha=0.12,
                    label="Negative (bonds hedge equities)")
    ax.set_title("6-month rolling correlation: S&P 500 (IVV) vs Long Treasuries (TLT)")
    ax.legend(fontsize=8, loc="lower left")
    _save(fig, "07_stock_bond_correlation")

    # 8. Rolling factor betas
    if res.rolling_betas is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        palette = list(COLORS.values())
        for i, col in enumerate(res.rolling_betas.columns):
            ax.plot(res.rolling_betas.index, res.rolling_betas[col], color=palette[i], lw=1.3,
                    label=col)
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_title("Rolling 1-year Fama-French factor exposures")
        ax.legend(ncol=3, fontsize=8, loc="upper left")
        _save(fig, "08_rolling_factor_betas")

    # 9. Correlation heatmap (diverging blue <-> red, neutral gray midpoint)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", NEUTRAL,
                                                     "#f0a3a2", "#b8302f"])
    c = res.returns.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(c, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=45)
    ax.set_yticks(range(len(c)), c.index)
    ax.grid(False)
    for i in range(len(c)):
        for j in range(len(c)):
            ax.text(j, i, f"{c.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(c.iloc[i, j]) > 0.6 else INK)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Daily return correlation matrix")
    _save(fig, "09_correlation_matrix")


# =========================================================================== HTML report
def _fmt_table(df: pd.DataFrame, pct_cols=(), money_cols=(), dec_cols=(), pct_rows=(),
               index_name="") -> str:
    def fmt(v, kind):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "—"
            if kind == "pct":
                return f"{v:.2%}"
            if kind == "money":
                return f"${v:,.0f}"
            return f"{v:.2f}" if kind == "dec" else f"{v:,.3f}"
        return html.escape(str(v.date()) if isinstance(v, pd.Timestamp) else str(v))

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [index_name, *df.columns])
    body = []
    for idx, row in df.iterrows():
        cells = [f"<td class='rowhead'>{html.escape(str(idx))}</td>"]
        for c, v in row.items():
            kind = ("pct" if (c in pct_cols or idx in pct_rows) else
                    "money" if c in money_cols else "dec" if c in dec_cols else "num")
            cls = " class='neg'" if isinstance(v, (float, np.floating)) and v < 0 and \
                kind in ("pct", "money") else ""
            cells.append(f"<td{cls}>{fmt(v, kind)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def _plot_div(fig: go.Figure, first: bool = False) -> str:
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK2, size=12),
        margin=dict(l=50, r=20, t=30, b=40), hovermode="x unified",
        legend=dict(orientation="h", y=-0.15),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor="#c3c2b7")
    fig.update_yaxes(gridcolor=GRID, linecolor="#c3c2b7")
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False,
                       config={"displaylogo": False, "responsive": True})


def _backtest_takeaway(bt: pd.DataFrame) -> str:
    h, e, f = bt.loc["Historical"], bt.loc["EWMA"], bt.loc["Filtered HS"]
    return (f"Historical VaR gets the <b>count</b> right ({h['Exceptions']} vs {h['Expected']} "
            f"expected, Kupiec p={h['Kupiec p-value']:.2f}) but its breaches <b>cluster</b> in crises "
            f"(independence p={h['Independence p-value']:.3f}) because a 2-year window reacts slowly. "
            f"EWMA reacts fast but assumes normal tails, so it breaches far too often "
            f"({e['Exceptions']} exceptions). Filtered Historical Simulation combines both ideas. "
            f"It has {f['Exceptions']} exceptions (Kupiec p={f['Kupiec p-value']:.2f}) and "
            f"reduces clustering, but clustering is still statistically significant "
            f"(p={f['Independence p-value']:.3f}). Crises like 2008 and March 2020 produce "
            f"back-to-back breaches that no 1-day model fully anticipates, which is exactly "
            f"why VaR is paired with the stress tests below. (Parameters were left at the "
            f"RiskMetrics standard, λ=0.94, rather than tuned to pass the test.)")


def build_html_report(res: Results) -> str:
    s = res.summary
    p, b = s[CLIENT], s[BENCH]
    var99 = res.var_table.loc["99.0%"]
    hist_bt = res.var_backtest.loc["Historical"]
    worst = res.hist_stress["Portfolio Return"].idxmin()
    start = max(v["returns"].index[0] for v in res.walk_forward.values())

    # --- growth chart
    g = go.Figure()
    for name, out in res.walk_forward.items():
        nav = 100 * (1 + out["returns"].loc[start:]).cumprod()
        g.add_trace(go.Scatter(x=nav.index, y=nav, name=name, line=dict(
            color=COLORS[name], width=3 if name == CLIENT else 1.6),
            hovertemplate="%{y:.1f}"))
    g.update_layout(height=420, yaxis_title="Value of $100")

    # --- drawdown
    d = go.Figure()
    for name, r in ((CLIENT, res.port), (BENCH, res.bench)):
        w = (1 + r).cumprod()
        d.add_trace(go.Scatter(x=w.index, y=w / w.cummax() - 1, name=name,
                               line=dict(color=COLORS[name], width=1.6), hovertemplate="%{y:.1%}",
                               fill="tozeroy" if name == CLIENT else None,
                               fillcolor="rgba(42,120,214,0.12)"))
    d.update_layout(height=300, yaxis_tickformat=".0%")

    # --- VaR backtest
    fc = res.var_forecasts
    v = go.Figure()
    v.add_trace(go.Bar(x=fc.index, y=fc["return"], name="Daily P&L", marker_color="#c3c2b7",
                       hovertemplate="%{y:.2%}"))
    v.add_trace(go.Scatter(x=fc.index, y=-fc["Historical"], name="99% Historical VaR",
                           line=dict(color=COLORS[CLIENT], width=1.4), hovertemplate="%{y:.2%}"))
    v.add_trace(go.Scatter(x=fc.index, y=-fc["Filtered HS"], name="99% Filtered HS VaR",
                           line=dict(color=COLORS["Max Diversification"], width=1.2),
                           hovertemplate="%{y:.2%}"))
    hits = fc[fc["return"] < -fc["Historical"]]
    v.add_trace(go.Scatter(x=hits.index, y=hits["return"], mode="markers", name="Exception",
                           marker=dict(color=CRITICAL, size=8, line=dict(color=SURFACE, width=1.5)),
                           hovertemplate="%{y:.2%}"))
    v.update_layout(height=380, yaxis_tickformat=".1%", bargap=0)

    # --- MC distribution
    mc = go.Figure(go.Histogram(x=res.mc_pnl, nbinsx=150, marker_color="#86b6ef",
                                hovertemplate="%{x:.2%}: %{y} sims<extra></extra>"))
    mc.add_vline(x=-var99["Monte Carlo (t) VaR"], line_color=CRITICAL, line_dash="dash",
                 annotation_text="99% VaR", annotation_position="top left")
    mc.update_layout(height=300, xaxis_tickformat=".1%", hovermode="closest", showlegend=False)

    # --- frontier
    f = go.Figure()
    f.add_trace(go.Scatter(x=res.frontier["volatility"], y=res.frontier["return"],
                           name="Efficient frontier", line=dict(color=INK2, width=2.5),
                           hovertemplate="vol %{x:.1%}<br>ret %{y:.1%}<extra></extra>"))
    for name, pt in res.point_portfolios.items():
        tip = "<br>".join(f"{t}: {w:.0%}" for t, w in pt["weights"].items() if w > 0.005)
        f.add_trace(go.Scatter(x=[pt["volatility"]], y=[pt["return"]], mode="markers", name=name,
                               marker=dict(size=13, color=COLORS[name],
                                           line=dict(color=SURFACE, width=2)),
                               hovertemplate=f"<b>{name}</b><br>vol %{{x:.1%}}<br>ret %{{y:.1%}}"
                                             f"<br>{tip}<extra></extra>"))
    f.update_layout(height=420, xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                    xaxis_title="Volatility", yaxis_title="Return", hovermode="closest")

    # --- risk contribution
    rc = res.risk_contrib.sort_values("% of Risk")
    r = go.Figure()
    r.add_trace(go.Bar(y=rc.index, x=rc["Weight"], orientation="h", name="Capital weight",
                       marker_color="#86b6ef", hovertemplate="%{x:.1%}"))
    r.add_trace(go.Bar(y=rc.index, x=rc["% of Risk"], orientation="h", name="Share of risk",
                       marker_color=COLORS[CLIENT], hovertemplate="%{x:.1%}"))
    r.update_layout(height=380, xaxis_tickformat=".0%", barmode="group", hovermode="y unified")

    # --- stress
    hs = res.hist_stress.sort_values("Portfolio Return")
    st = go.Figure()
    st.add_trace(go.Bar(y=hs.index, x=hs["Benchmark Return"], orientation="h", name=BENCH,
                        marker_color=COLORS[BENCH], hovertemplate="%{x:.1%}"))
    st.add_trace(go.Bar(y=hs.index, x=hs["Portfolio Return"], orientation="h", name=CLIENT,
                        marker_color=COLORS[CLIENT], hovertemplate="%{x:.1%}"))
    st.update_layout(height=400, xaxis_tickformat=".0%", barmode="group", hovermode="y unified")

    # --- factor betas
    fb = ""
    if res.rolling_betas is not None:
        fbf = go.Figure()
        palette = list(COLORS.values())
        for i, col in enumerate(res.rolling_betas.columns):
            fbf.add_trace(go.Scatter(x=res.rolling_betas.index, y=res.rolling_betas[col],
                                     name=col, line=dict(color=palette[i], width=1.6),
                                     hovertemplate="%{y:.2f}"))
        fbf.update_layout(height=360)
        fb = _plot_div(fbf)

    pct_rows = ["Annualized Return", "Annualized Volatility", "Max Drawdown", "Best Day",
                "Worst Day", "% Positive Days", "Tracking Error", "Avg Monthly Turnover"]

    factor_section = "<p>Fama-French data unavailable for this run.</p>"
    if res.factor_model is not None:
        fm = res.factor_model
        factor_section = f"""
        <p>Regression of daily portfolio excess returns on the Fama-French 5 factors + Momentum
        with Newey-West (HAC) standard errors. <b>R² = {fm['r_squared']:.2%}</b> over
        {fm['n_obs']:,} days — the share of portfolio variance explained by systematic factors.</p>
        <div class="grid2">
          <div><h3>Factor loadings</h3>{_fmt_table(fm['table'], dec_cols=('Coefficient', 't-stat'),
                                                    index_name='Factor')}</div>
          <div><h3>Variance decomposition</h3>{_fmt_table(fm['risk_decomposition'].to_frame('Share of Variance'),
                                                           pct_cols=('Share of Variance',), index_name='Source')}
               <h3>Annual return attribution</h3>{_fmt_table(fm['return_attribution'].to_frame('Contribution'),
                                                           pct_cols=('Contribution',), index_name='Source')}</div>
        </div>
        <h3>Rolling 1-year exposures</h3>{fb}"""

    best_wf = res.wf_summary.loc["Sharpe Ratio"].idxmax()
    sql_blocks = "".join(
        f"<details><summary><code>{html.escape(name)}</code> — {len(df):,} rows</summary>"
        f"{_fmt_table(df.head(15).set_index(df.columns[0]), index_name=df.columns[0])}</details>"
        for name, df in res.sql.items())
    source_note = ("" if res.source != "synthetic" else
                   "<div class='warn'>⚠ Live market data was unavailable — this run uses "
                   "SYNTHETIC prices. Re-run with internet access for real results.</div>")

    kpis = [
        ("Annualised return", f"{p['Annualized Return']:.2%}", f"Benchmark {b['Annualized Return']:.2%}"),
        ("Volatility", f"{p['Annualized Volatility']:.2%}", f"Benchmark {b['Annualized Volatility']:.2%}"),
        ("Sharpe ratio", f"{p['Sharpe Ratio']:.2f}", f"Benchmark {b['Sharpe Ratio']:.2f}"),
        ("Max drawdown", f"{p['Max Drawdown']:.1%}", f"Benchmark {b['Max Drawdown']:.1%}"),
        ("1-day 99% VaR", f"${var99['Historical VaR'] * config.NOTIONAL / 1e6:.2f}M",
         f"ES ${var99['Historical ES'] * config.NOTIONAL / 1e6:.2f}M on $100M"),
        ("Worst stress", f"{res.hist_stress.loc[worst, 'Portfolio Return']:.1%}", worst),
    ]
    kpi_html = "".join(f"<div class='kpi'><div class='kl'>{k}</div><div class='kv'>{v}</div>"
                       f"<div class='ks'>{html.escape(sub)}</div></div>" for k, v, sub in kpis)

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Risk Report</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --line:#e1e0d9; --accent:#2a78d6; --neg:#c6302f; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif }}
main {{ max-width:1180px; margin:0 auto; padding:32px 16px 80px }}
header h1 {{ font-size:28px; margin:0 0 4px }} header p {{ color:var(--ink2); margin:0 }}
section {{ background:var(--surface); border:1px solid var(--line); border-radius:12px;
          padding:22px 22px 14px; margin-top:22px }}
h2 {{ font-size:19px; margin:0 0 6px }} h3 {{ font-size:14px; margin:18px 0 6px; color:var(--ink2) }}
p {{ color:var(--ink2); max-width:80ch }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-top:22px }}
.kpi {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px }}
.kl {{ font-size:12px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em }}
.kv {{ font-size:26px; font-weight:650; margin:2px 0 }} .ks {{ font-size:12px; color:var(--ink2) }}
.tbl {{ overflow-x:auto }}
table {{ border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums }}
th {{ text-align:right; color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); padding:6px 8px; white-space:nowrap }}
td {{ text-align:right; padding:5px 8px; border-bottom:1px solid var(--line); white-space:nowrap }}
th:first-child, td.rowhead {{ text-align:left; color:var(--ink) }}
td.neg {{ color:var(--neg) }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); gap:20px }}
.callout {{ border-left:3px solid var(--accent); padding:4px 12px; color:var(--ink); background:#f0f5fc; border-radius:4px }}
.warn {{ background:#fff4d6; border:1px solid #fab219; padding:10px 14px; border-radius:8px; margin-top:16px }}
details {{ margin:6px 0 }} summary {{ cursor:pointer; padding:6px 0 }}
code {{ font-family:ui-monospace,Menlo,monospace; font-size:13px }}
</style></head><body><main>
<header><h1>Multi-Asset Portfolio Risk Report</h1>
<p>$100M model client portfolio across 10 iShares ETFs · data {res.returns.index[0].date()} to
{res.returns.index[-1].date()} ({len(res.returns):,} trading days) · source: {res.source} ·
generated {date.today()}</p>{source_note}</header>

<div class="kpis">{kpi_html}</div>

<section><h2>1. Performance vs benchmark</h2>
<p>Client portfolio (monthly rebalanced) vs a 60/40 IVV/AGG benchmark. Risk-free rate from the
Fama-French daily RF series.</p>
<div class="grid2"><div>{_fmt_table(s, pct_rows=pct_rows, index_name='Metric')}</div>
<div><h3>Drawdowns</h3>{_plot_div(d, first=True)}
<p>Max drawdown {res.drawdown['max_drawdown']:.1%}: peak {res.drawdown['peak'].date()},
trough {res.drawdown['trough'].date()}, {res.drawdown['days_to_trough']} days down,
{'recovered in ' + str(res.drawdown['days_to_recover']) + ' days' if res.drawdown['days_to_recover'] else 'not yet recovered'}.</p></div></div>
<h3>Monthly returns</h3>{_fmt_table(res.calendar.iloc[::-1], pct_cols=tuple(res.calendar.columns), index_name='Year')}
</section>

<section><h2>2. Value-at-Risk & Expected Shortfall</h2>
<p>Four VaR models on the same portfolio. Fat tails show up as Cornish-Fisher and Student-t
Monte Carlo VaR exceeding the normal (parametric) figure at high confidence.</p>
{_fmt_table(res.var_table, pct_cols=tuple(res.var_table.columns), index_name='Confidence')}
<h3>Monte Carlo P&amp;L distribution (100,000 multivariate-t scenarios, 1-day)</h3>{_plot_div(mc)}
<h3>Backtest: are the VaR models any good?</h3>
<p>Every forecast uses only data available the day before ({config.VAR_WINDOW}-day window).
Exceptions are tested with Kupiec's proportion-of-failures test (right number of breaches?) and
Christoffersen's independence test (are breaches clustered?).</p>
{_fmt_table(res.var_backtest, pct_cols=('Exception Rate',), index_name='Model')}
<p class="callout">{_backtest_takeaway(res.var_backtest)}</p>
{_plot_div(v)}</section>

<section><h2>3. Where does the risk come from?</h2>
<p>Euler decomposition of portfolio volatility using a Ledoit-Wolf shrunk covariance matrix
(shrinkage intensity {res.shrinkage:.3f}). Equities are a minority of capital but most of the risk.</p>
<div class="grid2"><div>{_plot_div(r)}</div>
<div>{_fmt_table(res.risk_contrib, pct_cols=('Weight', 'Standalone Vol', 'Marginal Contribution', 'Contribution to Vol', '% of Risk'), dec_cols=('Risk / Capital Ratio',), index_name='Asset')}
<h3>Component 99% VaR ($)</h3>{_fmt_table(res.component_var.to_frame('Component VaR'), money_cols=('Component VaR',), index_name='Asset')}</div></div>
</section>

<section><h2>4. Factor exposures</h2>{factor_section}</section>

<section><h2>5. Stress testing</h2>
<p>Historical scenarios replay what each asset actually did in past crises. Hypothetical
scenarios shock a driver asset and propagate the move to all other assets via the conditional
expectation under a covariance matrix estimated on stressed markets.</p>
{_plot_div(st)}
{_fmt_table(res.hist_stress, pct_cols=('Portfolio Return', 'Benchmark Return', 'Relative'), money_cols=('P&L ($)',), index_name='Scenario')}
<h3>Hypothetical scenarios (1-month horizon)</h3>
<p>Rate scenarios use the <b>recent</b> correlation regime. Since 2022, stocks and bonds have been
positively correlated, so a rates shock now hurts equities too. Calibrating on crisis-era data
(when Treasuries rallied as stocks fell) would wrongly show a rates shock as a <i>gain</i>.</p>
{_fmt_table(res.hypo_stress, pct_cols=('Portfolio Return',), money_cols=('P&L ($)',), index_name='Scenario')}
<h3>Implied asset moves</h3>
{_fmt_table(res.hypo_moves, pct_cols=tuple(res.hypo_moves.columns), index_name='Scenario')}
</section>

<section><h2>6. Portfolio construction — out-of-sample backtest</h2>
<p>Each month-end, every strategy re-estimates weights on the trailing 3 years only (no
look-ahead) and pays 5bp per side on turnover. Best risk-adjusted strategy: <b>{best_wf}</b>.</p>
{_plot_div(g)}
{_fmt_table(res.wf_summary, pct_rows=pct_rows, index_name='Metric')}
<h3>Efficient frontier (full-sample, for reference)</h3>{_plot_div(f)}
</section>

<section><h2>7. Performance attribution (Brinson-Fachler)</h2>
<p>Active return vs the 60/40 benchmark split into <b>allocation</b> (over/under-weighting asset
classes) and <b>selection</b> (choosing different instruments inside an asset class), summed
monthly over the full history.</p>
{_fmt_table(res.brinson, pct_cols=tuple(res.brinson.columns), index_name='Asset Class')}
</section>

<section><h2>8. SQL analytics layer</h2>
<p>Results of <code>sql/analytics.sql</code> run against the SQLite warehouse: CTEs, window
functions (running peaks, rolling volatility, ranking), self-joins for correlation and data-quality
checks. Full outputs in <code>outputs/sql/</code>.</p>{sql_blocks}
</section>
</main></body></html>"""


def write_outputs(res: Results) -> None:
    out = config.OUTPUT_DIR
    (out / "sql").mkdir(parents=True, exist_ok=True)
    res.summary.to_csv(out / "performance_summary.csv")
    res.var_table.to_csv(out / "var_es_table.csv")
    res.var_backtest.to_csv(out / "var_backtest.csv")
    res.risk_contrib.to_csv(out / "risk_contribution.csv")
    res.hist_stress.to_csv(out / "stress_historical.csv")
    res.hypo_stress.join(res.hypo_moves).to_csv(out / "stress_hypothetical.csv")
    res.wf_summary.to_csv(out / "strategy_backtest_summary.csv")
    res.brinson.to_csv(out / "brinson_attribution.csv")
    res.calendar.to_csv(out / "monthly_returns.csv")
    if res.factor_model is not None:
        res.factor_model["table"].to_csv(out / "factor_regression.csv")
    for name, df in res.sql.items():
        df.to_csv(out / "sql" / f"{name}.csv", index=False)
    (out / "risk_report.html").write_text(build_html_report(res), encoding="utf-8")
    save_static_charts(res)
