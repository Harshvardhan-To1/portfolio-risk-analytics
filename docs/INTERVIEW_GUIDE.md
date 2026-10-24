# Interview guide: how to explain this project

Read this until you can explain every section **without notes**. Interviewers at
BlackRock will drill into whatever you put on your resume. A project you can defend
is worth far more than one that just looks impressive.

---

## The 30-second pitch

> "I built an end-to-end risk analytics engine for a $100M multi-asset portfolio of
> iShares ETFs, which is a small version of what Aladdin does. It pulls 18 years of market
> data into a SQL warehouse and computes VaR four different ways. It then *backtests*
> those VaR models statistically, decomposes risk by asset and by Fama-French factor,
> stress-tests the portfolio against seven historical crises, and compares five
> portfolio-construction strategies out of sample. Everything is in a Streamlit dashboard
> and an HTML risk report."

## The 2-minute walkthrough (follow the data)

1. **Data**: 10 iShares ETFs (equities, bonds, credit, TIPS, gold, REITs) from 2007,
   chosen to include the 2008 crisis. Prices come from Yahoo Finance, with Stooq as a
   fallback. Fama-French factors come from Ken French's library.
2. **Warehouse**: The data loads into SQLite as fact tables (`fact_price`,
   `fact_return`, `fact_factor`, `fact_portfolio_nav`) plus a dimension table
   (`dim_asset`). Analysis queries live in `sql/analytics.sql`, and there are
   data-quality checks before anything is published.
3. **Risk**: VaR and ES (historical, parametric, Cornish-Fisher, Monte Carlo), then a
   backtest with Kupiec and Christoffersen tests.
4. **Decomposition**: Which assets and which factors drive the risk.
5. **Stress tests**: Historical replays plus hypothetical "what if" shocks.
6. **Construction**: Five optimizers in a walk-forward backtest with transaction costs.
7. **Attribution**: Brinson-Fachler, to see *why* the portfolio beat or lagged the benchmark.

---

## Concepts you MUST be able to explain

### Value-at-Risk (VaR)
"The loss we don't expect to exceed on X% of days." A 1-day 99% VaR of $1.2M
means that on 99 out of 100 days we expect to lose less than $1.2M.
- **Historical**: Sort past returns and take the 1st percentile. It makes no
  distribution assumption, but it's only as good as the history window.
- **Parametric (normal)**: mean − 2.33 × std. It's fast but *underestimates* tail risk
  because real returns have fat tails.
- **Cornish-Fisher**: Adjusts the normal z-score for skewness and kurtosis.
- **Monte Carlo (Student-t)**: Simulates 100k correlated scenarios from a fat-tailed
  distribution.

**Weakness of VaR**: It says nothing about *how bad* things get beyond the threshold,
and it isn't sub-additive.

### Expected Shortfall (ES / CVaR)
The average loss *on the days that exceed VaR*. Basel III/FRTB moved banks from VaR to
ES (97.5%) because ES captures tail severity and is a coherent risk measure.

### VaR backtesting (the most impressive part, so know it cold)
- **Exception**: A day where the actual loss exceeds that day's VaR forecast. At 99%
  you expect about 1% of days.
- **Kupiec POF test**: Is the *number* of exceptions consistent with 1%? It uses a
  likelihood-ratio test with chi-squared(1). A p-value < 0.05 means the model is
  rejected.
- **Christoffersen independence test**: Are exceptions *clustered*? If yesterday's
  breach makes today's more likely, the model reacts too slowly to volatility changes.
- **Basel traffic light**: Exceptions in the last 250 days at 99%. 0–4 is green,
  5–9 is yellow, and 10+ is red.
- **What the backtest found**:
  - **Parametric (normal)** had 73 exceptions vs 44 expected and was rejected. Fat tails
    make normal VaR too low.
  - **Historical** had 49, so the count is fine (p = 0.44), but the breaches were
    *clustered* in crises and it failed the independence test. A 2-year window reacts
    too slowly.
  - **EWMA** had 96. It reacts fast but still assumes normal tails, which makes things
    worse.
  - **Filtered Historical Simulation** had 55 and passes Kupiec (p = 0.10). It
    standardizes past returns by EWMA vol, takes the empirical quantile, then rescales
    by today's vol, so it gets fat tails *and* responsiveness. Clustering is reduced but
    still significant.
  - **Lesson**: No 1-day model anticipates crash clusters like March 2020. That's why
    risk teams pair VaR with stress tests. I also deliberately did *not* tune λ to pass
    the test, because that would be overfitting.
- **No look-ahead**: Each day's VaR uses only data up to the day before
  (`.shift(1)`), and there's a unit test that proves it.

### Risk contribution (Euler decomposition)
Portfolio vol σ = √(wᵀΣw). The contribution of asset i is wᵢ·(Σw)ᵢ/σ, and the
contributions add up exactly to σ.
**Talking point**: "Equities are about 45% of capital but drive the vast majority of
risk. A '60/40' portfolio is really a ~90/10 portfolio *in risk terms*. That's the
motivation for risk parity."

### Ledoit-Wolf shrinkage
The sample covariance matrix is noisy, and optimizers exploit that noise. Shrinkage
blends the sample covariance with a structured target (scaled identity), using an
optimal intensity that is calculated from the data rather than guessed.

### Portfolio optimizers
- **Min variance**: The lowest-risk mix, which needs no return forecasts.
- **Max Sharpe**: The best return per unit of risk, but it's very sensitive to expected
  returns. The backtest shrinks return estimates toward the average for that reason.
- **Risk parity (ERC)**: Every asset contributes equal risk. It's solved with a convex
  log-barrier formulation.
- **Max diversification**: Maximizes the ratio of weighted average vol to portfolio vol.
- **Walk-forward**: At each month-end the weights use only the trailing 3 years, and
  turnover costs 5bp per side. This is an *out-of-sample* test, unlike a frontier fit
  to the whole history.

### Fama-French factor regression
The portfolio's excess return is regressed on Market, Size (SMB), Value (HML),
Profitability (RMW), Investment (CMA) and Momentum.
- **Betas** = exposures. **Alpha** = return unexplained by the factors.
- **R²** = the share of variance from systematic factors. Here it's **86%**, and nearly
  all of that is the *market* factor (beta ≈ 0.50). Size, value and the other factors
  add almost nothing, and alpha is statistically zero (t = 0.17). The remaining 14% is
  mostly bond risk (duration, credit), which equity factors can't capture. *Good
  follow-up*: "I'd add rate and credit factors next."
- **Newey-West (HAC) standard errors**: Daily returns have autocorrelation and
  heteroskedasticity, which makes regular OLS t-stats overconfident.

### Stress testing
- **Historical**: "If 2008 happened again tomorrow, what would this portfolio lose?"
  It replays each asset's actual cumulative return over the crisis window.
- **Hypothetical (conditional)**: You shock one asset, for example S&P −25%. The other
  assets move by their conditional expectation E[r_other | r_driver = shock] =
  Σ_od Σ_dd⁻¹ · shock. The covariance is estimated on *stressed* periods because
  correlations rise in crises.
- **2022 insight**: Stocks and bonds fell *together* (see the rolling correlation
  chart). Diversified portfolios lost more than history suggested because the
  stock-bond hedge broke down.

### Brinson-Fachler attribution
Active return vs the benchmark splits into three parts:
- **Allocation** = (w_p − w_b) × (R_b,class − R_b,total): Did we overweight the right
  asset classes?
- **Selection** = w_b × (R_p,class − R_b,class): Did we pick better instruments inside
  each class?
- **Interaction** = the cross term.

### SQL features used (data analyst roles test this)
- CTEs (`WITH ...`), joins with `USING`, `GROUP BY` aggregation
- Window functions: `RANK() OVER (PARTITION BY ...)`, `ROW_NUMBER()`, running max
  with `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`, and a 63-day rolling
  volatility with a framed `WINDOW` clause
- Compounding returns in SQL: `EXP(SUM(LN(1 + r))) − 1`
- Pearson correlation built from raw aggregates
- Data-quality checks: missing rows, extreme moves, stale prices, duplicates

---

## Likely interview questions and strong answers

**Q: Why not just use parametric VaR?**
Returns have fat tails, with excess kurtosis well above 0 for this portfolio. The
normal model underestimates 99%+ losses, and Cornish-Fisher and Student-t Monte Carlo
correct for that.

**Q: Your VaR model failed a backtest. Is that bad?**
No. It's the point of backtesting. It showed that exceptions cluster in crises, so the
model reacts too slowly. I added Filtered Historical Simulation, which fixed the
exception count. A model that *never* fails a backtest is usually too conservative.

**Q: Your client portfolio underperformed 60/40. Isn't that a bad result?**
It's an honest result. From 2010 to 2026, US large caps beat almost everything, so a
portfolio that diversifies into international stocks, EM, REITs and gold lagged. The
Brinson attribution shows exactly where (equity *selection*, meaning ex-US exposure).
It had a smaller drawdown in 2008. Diversification is insurance against a regime you
don't get to choose in advance.

**Q: Why did a rates shock show a gain at first?**
I first estimated the scenario covariance on crisis periods, when Treasuries rallied
while stocks fell. Under those correlations, "bonds down" implies "stocks up". Since
2022, stocks and bonds have been positively correlated, so I let each scenario choose
its correlation regime. The rates shock now shows a −5.5% loss. It's a good example of
model assumptions mattering more than the math.

**Q: How did you avoid look-ahead bias?**
VaR forecasts use `.shift(1)`, and walk-forward weights only see the trailing window.
Unit tests check both, for example by changing the final day's return and confirming
that the forecast doesn't change.

**Q: What would you do with more time or data?**
- Add rate and credit-spread factors (duration, DV01, OAS) for the bond sleeve.
- Use a GARCH or filtered historical simulation VaR.
- Use liquidity-adjusted VaR, since you can't sell $100M of HYG in a day at mid.
- Model expected returns with Black-Litterman instead of historical means.
- Schedule the pipeline (Airflow or cron) and publish to a BI tool (Tableau or Power BI).

**Q: Why ETFs?**
They're real, liquid instruments with long histories that span every major asset
class. They're also iShares, BlackRock's own products, so the analysis maps directly to
what the firm manages.

**Q: How is this relevant to a data analyst role?**
It's a full data pipeline: ingestion with fallbacks, a modelled SQL warehouse,
data-quality checks, analysis, statistical validation, and communication through a
dashboard and report. That's the day-to-day of the role.

---

## Headline numbers to memorize
| | Client | 60/40 |
|---|---|---|
| Annualized return (2007–2026) | 6.50% | 7.79% |
| Volatility | 10.8% | 11.7% |
| Sharpe | 0.50 | 0.57 |
| Max drawdown | −33.9% | −35.8% |
| 1-day 99% VaR / ES on $100M | $1.94M / $2.94M | |
| GFC loss | −25.4% | −27.1% |
| 2022 rate-shock loss | −21.6% | −20.4% |

*(These change slightly every time you re-run with fresh data. Re-check them before the interview.)*

## Before the interview, actually do these
1. Run `python run_pipeline.py` and open `outputs/risk_report.html`. Know the headline
   numbers (return, vol, Sharpe, max drawdown, VaR, worst stress scenario).
2. Run `streamlit run app.py`, change weights, and watch the risk numbers move.
3. Read `riskengine/var.py` and `sql/analytics.sql` line by line.
4. Change one thing yourself (add a new stress scenario in `config.py` or a new SQL
   query) so the code is truly yours.
