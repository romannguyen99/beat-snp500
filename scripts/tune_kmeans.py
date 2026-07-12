"""Grid-search the K-means picker on the development period (months <=
config.DEV_END), report ranked results, and confirm the winner ONCE on the
holdout (months > DEV_END). Spec: docs/superpowers/specs/
2026-07-10-must-buy-selection-design.md §4a. Run time: several minutes.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/tune_kmeans.py            # dev-period grid
    PYTHONPATH=src .venv/bin/python scripts/tune_kmeans.py --holdout  # winner vs current, once
"""
import argparse

import pandas as pd

from beat_snp500 import config
from beat_snp500.backtest.engine import run_backtest
from beat_snp500.backtest.metrics import perf_metrics
from beat_snp500.data.factors import load_ff5
from beat_snp500.data.prices import close_matrix
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.io_utils import atomic_write_json, read_json
from beat_snp500.models.kmeans import kmeans_picks

GRID = [{"k": k, "mom_mode": m, "select_rule": r, "threshold": z}
        for k in (3, 4, 5, 6)
        for m in ("mean_3_6_12", "12_1", "vol_scaled")
        for r in ("mean", "risk_adj")
        for z in (0.0, 0.25, 0.5)]
CURRENT = {"k": config.K_CLUSTERS, "mom_mode": config.KMEANS_MOM_MODE,
           "select_rule": config.KMEANS_SELECT_RULE,
           "threshold": config.MUST_BUY_Z_KMEANS}
REPORT = config.OUTPUTS_DIR / "tuning" / "kmeans_tuning.json"


def sharpe_for(cfg: dict, panel: pd.DataFrame, close: pd.DataFrame,
               rf_annual: float) -> float:
    picks = kmeans_picks(panel, **cfg)
    res = run_backtest(picks, close)
    if res.daily_returns.empty:
        return float("nan")
    return float(perf_metrics(res.daily_returns, rf_annual=rf_annual)["sharpe"])


def load_slices():
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    as_of = pd.Timestamp.today().normalize()
    prices = prices[prices["date"] < as_of.replace(day=1)]
    close = close_matrix(prices)
    panel = build_feature_panel(prices, membership, factors)
    dev_end = pd.Timestamp(config.DEV_END)
    dates = panel.index.get_level_values("date")
    dev = (panel[dates <= dev_end],
           close[close.index <= dev_end + pd.offsets.MonthEnd(1)],
           factors[factors.index <= dev_end])
    hold = (panel[dates > dev_end], close[close.index > dev_end],
            factors[factors.index > dev_end])
    return dev, hold


def rf_annual(factors: pd.DataFrame) -> float:
    return float((1 + factors["rf"]).prod() ** (12 / len(factors)) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="one-shot confirmation of the dev winner vs CURRENT")
    args = ap.parse_args()
    (dev_panel, dev_close, dev_f), (h_panel, h_close, h_f) = load_slices()

    if not args.holdout:
        rf = rf_annual(dev_f)
        rows = []
        for i, cfg in enumerate(GRID):
            rows.append({**cfg, "dev_sharpe": sharpe_for(cfg, dev_panel,
                                                         dev_close, rf)})
            print(f"[{i + 1}/{len(GRID)}]", rows[-1])
        rows.sort(key=lambda r: (r["dev_sharpe"] != r["dev_sharpe"],
                                 -r["dev_sharpe"]))  # NaNs last
        atomic_write_json({"current": CURRENT, "grid": rows}, REPORT)
        print("\ntop 5 by dev Sharpe:")
        for r in rows[:5]:
            print(" ", r)
        print("current config dev Sharpe:",
              sharpe_for(CURRENT, dev_panel, dev_close, rf))
        print(f"\nNext: review {REPORT}, then run --holdout ONCE.")
        return 0

    report = read_json(REPORT)
    winner = {k: report["grid"][0][k]
              for k in ("k", "mom_mode", "select_rule", "threshold")}
    rf = rf_annual(h_f)
    w = sharpe_for(winner, h_panel, h_close, rf)
    c = sharpe_for(CURRENT, h_panel, h_close, rf)
    print("holdout Sharpe — winner:", w, winner)
    print("holdout Sharpe — current:", c, CURRENT)
    print("verdict:", "ADOPT winner" if w >= c else "KEEP current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
