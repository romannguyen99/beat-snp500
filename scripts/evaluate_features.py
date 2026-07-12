"""Dev/holdout gate for candidate features (spec §4b): a candidate set is
promoted into config.FEATURES only if it improves LightGBM's dev-period
walk-forward IC and does not degrade on the single holdout pass.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/evaluate_features.py                 # dev table
    PYTHONPATH=src .venv/bin/python scripts/evaluate_features.py --holdout NAME  # once
"""
import argparse

import pandas as pd

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.io_utils import atomic_write_json
from beat_snp500.models.lgbm import spearman_ic, walk_forward_scores

CANDIDATE_SETS = {
    "baseline": [],
    "+mom_vol_scaled": ["mom_vol_scaled"],
    "+resid_mom": ["resid_mom"],
    "+cluster_rel": ["return_3m_cz", "return_6m_cz", "return_12m_cz"],
    "+all": ["mom_vol_scaled", "resid_mom",
             "return_3m_cz", "return_6m_cz", "return_12m_cz"],
}
REPORT = config.OUTPUTS_DIR / "tuning" / "feature_eval.json"


def ic_stats(panel: pd.DataFrame, extra: list[str], dev: bool) -> dict:
    cols = config.BASE_FEATURES + extra
    ok = panel[cols].notna().all(axis=1)
    scores = walk_forward_scores(panel[ok], feature_cols=cols)
    ic = spearman_ic(scores, panel["fwd_return_1m"])
    cut = pd.Timestamp(config.DEV_END)
    ic = ic[ic.index <= cut] if dev else ic[ic.index > cut]
    return {"mean_ic": float(ic.mean()),
            "ic_ir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
            "n_months": int(len(ic))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", metavar="NAME",
                    help="candidate-set name for the single holdout pass")
    args = ap.parse_args()
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    as_of = pd.Timestamp.today().normalize()
    panel = build_feature_panel(prices[prices["date"] < as_of.replace(day=1)],
                                membership, factors)

    if args.holdout:
        for name in ("baseline", args.holdout):
            print(name, ic_stats(panel, CANDIDATE_SETS[name], dev=False))
        print("ADOPT only if the candidate beats baseline here too.")
        return 0

    rows = {name: ic_stats(panel, extra, dev=True)
            for name, extra in CANDIDATE_SETS.items()}
    for name, r in rows.items():
        print(f"{name:18s} mean IC {r['mean_ic']:+.4f}  IR {r['ic_ir']:+.2f}")
    atomic_write_json(rows, REPORT)
    print("\nNext: pick the best non-baseline set, run --holdout <name> ONCE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
