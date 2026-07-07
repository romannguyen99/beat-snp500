"""Rebuild the full walk-forward backtest report from cached data."""
import pandas as pd

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.jobs.backtest_report import run_report


def main() -> int:
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    metrics = run_report(prices, membership, factors, config.BACKTEST_DIR)
    for name, m in metrics.items():
        print(f"{name:14s} CAGR {m['cagr']:7.2%}  Sharpe {m['sharpe']:5.2f}  "
              f"MaxDD {m['max_drawdown']:7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
