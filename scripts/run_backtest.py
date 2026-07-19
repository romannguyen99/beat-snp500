"""Rebuild the full walk-forward backtest report from cached data."""
import pandas as pd

from beat_snp500 import config, tracking
from beat_snp500.data.factors import load_ff5
from beat_snp500.jobs.backtest_report import run_report


def main() -> int:
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    tracker = tracking.Tracker("backtest")
    with tracker.start_run(run_name=f"backtest-{pd.Timestamp.today():%Y%m%d}"):
        tracker.log_params({"champion": config.CHAMPION,
                            "n_features": len(config.FEATURES),
                            "features": ",".join(config.FEATURES),
                            "cost_bps_one_way": config.COST_BPS_ONE_WAY,
                            "universe_size": config.UNIVERSE_SIZE})
        metrics = run_report(prices, membership, factors, config.BACKTEST_DIR)
        for name, m in metrics.items():
            tracker.log_metrics({f"{name}_cagr": m["cagr"],
                                 f"{name}_sharpe": m["sharpe"],
                                 f"{name}_max_drawdown": m["max_drawdown"]})
    for name, m in metrics.items():
        print(f"{name:14s} CAGR {m['cagr']:7.2%}  Sharpe {m['sharpe']:5.2f}  "
              f"MaxDD {m['max_drawdown']:7.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
