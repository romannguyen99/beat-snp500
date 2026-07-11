"""Sanity check #2: per-feature IC. If every raw feature is near-zero on its
own, the ceiling is the feature set/label at this horizon, not how the
features get combined (linear vs. GBM) — see sanity_check_linear.py for #1.
"""
import pandas as pd

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.models.champion import spearman_ic


def main() -> int:
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)

    as_of = pd.Timestamp.today().normalize()
    prices = prices[prices["date"] < as_of.replace(day=1)]
    panel = build_feature_panel(prices, membership, factors, top_n=config.UNIVERSE_SIZE)
    fwd = panel["fwd_return_1m"]

    rows = []
    for feat in config.FEATURES:
        ic = spearman_ic(panel[feat], fwd)
        rows.append({
            "feature": feat,
            "mean_ic": ic.mean(),
            "std_ic": ic.std(),
            "ic_ir": ic.mean() / ic.std(),
            "n_months": ic.shape[0],
        })

    report = pd.DataFrame(rows).sort_values("mean_ic", key=lambda s: s.abs(), ascending=False)
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 100):
        print(report.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
