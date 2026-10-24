-- Analytics queries run against the SQLite warehouse (data/risk.db).
-- Each query is preceded by a name marker comment; results are exported to outputs/sql/*.csv.
-- LN / EXP / SQRT are registered as UDFs by riskengine/db.py.


-- name: asset_risk_return_summary
-- Annualized return & volatility per asset, computed entirely in SQL.
WITH stats AS (
    SELECT  r.ticker,
            COUNT(*)                      AS n_days,
            SUM(LN(1 + r.ret))            AS log_growth,
            AVG(r.ret)                    AS mean_ret,
            AVG(r.ret * r.ret)            AS mean_sq,
            MIN(r.ret)                    AS worst_day,
            MAX(r.ret)                    AS best_day
    FROM fact_return r
    GROUP BY r.ticker
)
SELECT  a.ticker, a.name, a.asset_class, a.sub_class,
        ROUND(EXP(log_growth * 252.0 / n_days) - 1, 4)                         AS ann_return,
        ROUND(SQRT((mean_sq - mean_ret * mean_ret) * n_days / (n_days - 1)) * SQRT(252), 4) AS ann_vol,
        ROUND((EXP(log_growth * 252.0 / n_days) - 1)
              / (SQRT((mean_sq - mean_ret * mean_ret) * n_days / (n_days - 1)) * SQRT(252)), 3) AS return_per_risk,
        ROUND(worst_day, 4) AS worst_day,
        ROUND(best_day, 4)  AS best_day
FROM stats s JOIN dim_asset a USING (ticker)
ORDER BY ann_return DESC;


-- name: monthly_asset_returns
-- Compounded monthly returns per asset (long format, ready for pivoting in BI tools).
SELECT  ticker,
        SUBSTR(date, 1, 7)                      AS month,
        ROUND(EXP(SUM(LN(1 + ret))) - 1, 6)     AS monthly_return
FROM fact_return
GROUP BY ticker, SUBSTR(date, 1, 7)
ORDER BY month, ticker;


-- name: strategy_annual_league_table
-- Calendar-year return of every strategy, ranked within each year. Restricted to the
-- window where every strategy has history (walk-forward strategies need a 3y warm-up).
WITH common AS (
    SELECT MAX(first_date) AS start_date
    FROM (SELECT MIN(date) AS first_date FROM fact_portfolio_nav GROUP BY strategy)
),
yearly AS (
    SELECT  strategy,
            SUBSTR(date, 1, 4)                  AS year,
            EXP(SUM(LN(1 + ret))) - 1           AS annual_return
    FROM fact_portfolio_nav, common
    WHERE date > SUBSTR(start_date, 1, 4) || '-12-31'   -- full calendar years only
    GROUP BY strategy, SUBSTR(date, 1, 4)
)
SELECT  year, strategy,
        ROUND(annual_return, 4)                                         AS annual_return,
        RANK() OVER (PARTITION BY year ORDER BY annual_return DESC)     AS rank_in_year
FROM yearly
ORDER BY year, rank_in_year;


-- name: drawdown_by_strategy
-- Running peak with a window function, then the deepest drawdown per strategy,
-- over the common window so every strategy is measured on the same period.
WITH common AS (
    SELECT MAX(first_date) AS start_date
    FROM (SELECT MIN(date) AS first_date FROM fact_portfolio_nav GROUP BY strategy)
),
peaks AS (
    SELECT  strategy, date, nav,
            MAX(nav) OVER (PARTITION BY strategy ORDER BY date
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_peak
    FROM fact_portfolio_nav, common
    WHERE date >= start_date
),
dd AS (
    SELECT strategy, date, nav / running_peak - 1 AS drawdown FROM peaks
),
ranked AS (
    SELECT  strategy, date, drawdown,
            ROW_NUMBER() OVER (PARTITION BY strategy ORDER BY drawdown) AS rn
    FROM dd
)
SELECT strategy, date AS trough_date, ROUND(drawdown, 4) AS max_drawdown
FROM ranked WHERE rn = 1
ORDER BY max_drawdown DESC;


-- name: worst_portfolio_days
-- Ten worst days for the client portfolio alongside the benchmark move that day.
WITH p AS (SELECT date, ret FROM fact_portfolio_nav WHERE strategy = 'Client Portfolio'),
     b AS (SELECT date, ret FROM fact_portfolio_nav WHERE strategy = 'Benchmark 60/40')
SELECT  p.date,
        ROUND(p.ret, 4)             AS portfolio_return,
        ROUND(b.ret, 4)             AS benchmark_return,
        ROUND(p.ret - b.ret, 4)     AS relative,
        RANK() OVER (ORDER BY p.ret) AS severity_rank
FROM p JOIN b USING (date)
ORDER BY p.ret
LIMIT 10;


-- name: rolling_quarterly_volatility
-- 63-day rolling annualized volatility of the client portfolio via a framed window.
WITH w AS (
    SELECT  date,
            AVG(ret)       OVER win AS m1,
            AVG(ret * ret) OVER win AS m2,
            COUNT(*)       OVER win AS n
    FROM fact_portfolio_nav
    WHERE strategy = 'Client Portfolio'
    WINDOW win AS (ORDER BY date ROWS BETWEEN 62 PRECEDING AND CURRENT ROW)
)
SELECT date, ROUND(SQRT((m2 - m1 * m1) * n / (n - 1)) * SQRT(252), 4) AS rolling_vol_63d
FROM w WHERE n = 63
ORDER BY date;


-- name: equity_bond_correlation_by_year
-- Pearson correlation of S&P 500 (IVV) vs Treasuries (TLT) by year, from raw sums.
-- Tracks the 2022 regime break when the stock-bond hedge stopped working.
WITH pairs AS (
    SELECT  SUBSTR(e.date, 1, 4) AS year, e.ret AS x, b.ret AS y
    FROM fact_return e
    JOIN fact_return b ON b.date = e.date AND b.ticker = 'TLT'
    WHERE e.ticker = 'IVV'
)
SELECT  year,
        COUNT(*) AS n_days,
        ROUND((AVG(x * y) - AVG(x) * AVG(y))
              / (SQRT(AVG(x * x) - AVG(x) * AVG(x)) * SQRT(AVG(y * y) - AVG(y) * AVG(y))), 3)
              AS correlation
FROM pairs
GROUP BY year
ORDER BY year;


-- name: asset_class_annual_returns
-- Equal-weighted calendar-year return of each asset class.
WITH asset_year AS (
    SELECT  a.asset_class, r.ticker, SUBSTR(r.date, 1, 4) AS year,
            EXP(SUM(LN(1 + r.ret))) - 1 AS yr_ret
    FROM fact_return r JOIN dim_asset a USING (ticker)
    GROUP BY a.asset_class, r.ticker, SUBSTR(r.date, 1, 4)
)
SELECT asset_class, year, ROUND(AVG(yr_ret), 4) AS avg_return, COUNT(*) AS n_assets
FROM asset_year
GROUP BY asset_class, year
ORDER BY year, asset_class;


-- name: data_quality_checks
-- Completeness and sanity checks a data team would run before publishing risk numbers.
SELECT 'missing_returns'  AS check_name,
       (SELECT COUNT(DISTINCT date) FROM fact_return) * (SELECT COUNT(*) FROM dim_asset)
       - (SELECT COUNT(*) FROM fact_return)                          AS issues
UNION ALL
SELECT 'extreme_moves_gt_25pct', COUNT(*) FROM fact_return WHERE ABS(ret) > 0.25
UNION ALL
SELECT 'non_positive_prices',    COUNT(*) FROM fact_price  WHERE adj_close <= 0
UNION ALL
SELECT 'duplicate_price_rows',   COUNT(*) - COUNT(DISTINCT date || ticker) FROM fact_price
UNION ALL
SELECT 'stale_prices_5d_flat',   COUNT(*) FROM (
    SELECT ticker, date,
           SUM(CASE WHEN ret = 0 THEN 1 ELSE 0 END)
               OVER (PARTITION BY ticker ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS zeros
    FROM fact_return) WHERE zeros = 5;
