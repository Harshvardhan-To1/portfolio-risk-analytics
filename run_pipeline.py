"""Run the full analytics pipeline and write the report.

    python run_pipeline.py              # fetch fresh data and rebuild everything
    python run_pipeline.py --no-refresh # reuse data already in data/risk.db
"""
import argparse
import logging
import time

from riskengine import pipeline, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-refresh", action="store_true", help="reuse cached warehouse data")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()
    res = pipeline.run(refresh=not args.no_refresh)
    report.write_outputs(res)

    p = res.summary["Client Portfolio"]
    var99 = res.var_table.loc["99.0%", "Historical VaR"]
    print("\n" + "=" * 64)
    print(f" Data source: {res.source}   |   {res.returns.index[0].date()} -> "
          f"{res.returns.index[-1].date()}")
    print(f" Annualized return {p['Annualized Return']:.2%}  vol {p['Annualized Volatility']:.2%}"
          f"  Sharpe {p['Sharpe Ratio']:.2f}  MaxDD {p['Max Drawdown']:.1%}")
    print(f" 1-day 99% VaR on $100M: ${var99 * 1e8:,.0f}")
    print(res.var_backtest[["Exceptions", "Expected", "Kupiec p-value", "Verdict"]].to_string())
    print(f" Report: outputs/risk_report.html   ({time.time() - t0:.1f}s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
