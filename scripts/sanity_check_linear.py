"""Sanity check: does a plain linear model beat champion's ~0 IC on the same
features/label/walk-forward setup? If Ridge also lands near zero, the ceiling
is set by the features/label at this horizon, not by LightGBM's complexity.

Mirrors champion.py's methodology exactly (same rank label, same walk-forward
window, same once-at-the-start hyperparam selection) so the resulting IC is
directly comparable to data/outputs/backtest/ic_monthly.parquet.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.models.champion import _months, _rank_label, decile_spread, spearman_ic

ALPHAS = [0.1, 1.0, 10.0, 100.0]


def _fit(panel: pd.DataFrame, months, alpha: float):
    rows = panel[panel.index.get_level_values("date").isin(months)]
    y = _rank_label(rows["fwd_return_1m"]).dropna()
    X = rows.loc[y.index, config.FEATURES]
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, random_state=config.SEED))
    return model.fit(X, y)


def _score_month(model, panel: pd.DataFrame, t) -> pd.Series:
    rows = panel.xs(t, level="date", drop_level=False)
    return pd.Series(model.predict(rows[config.FEATURES]), index=rows.index)


def select_alpha(panel: pd.DataFrame, train_window: int, val_months: int = 6) -> float:
    months = _months(panel)
    fit_months = months[: train_window - val_months]
    val = months[train_window - val_months: train_window]
    best, best_ic = ALPHAS[0], -2.0
    for alpha in ALPHAS:
        model = _fit(panel, fit_months, alpha)
        scores = pd.concat([_score_month(model, panel, t) for t in val])
        ic = spearman_ic(scores, panel["fwd_return_1m"]).mean()
        if ic > best_ic:
            best, best_ic = alpha, ic
    return best


def linear_walk_forward_scores(panel: pd.DataFrame, train_window: int, alpha: float) -> pd.Series:
    months = _months(panel)
    out = []
    for i in range(train_window, len(months)):
        model = _fit(panel, months[i - train_window: i], alpha)
        out.append(_score_month(model, panel, months[i]))
    return pd.concat(out).sort_index()


def main() -> int:
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)

    as_of = pd.Timestamp.today().normalize()
    prices = prices[prices["date"] < as_of.replace(day=1)]
    panel = build_feature_panel(prices, membership, factors, top_n=config.UNIVERSE_SIZE)

    train_window = config.TRAIN_WINDOW_MONTHS
    alpha = select_alpha(panel, train_window=train_window)
    scores = linear_walk_forward_scores(panel, train_window=train_window, alpha=alpha)
    ic = spearman_ic(scores, panel["fwd_return_1m"])
    spread = decile_spread(scores, panel["fwd_return_1m"])

    champ_ic = pd.read_parquet(config.BACKTEST_DIR / "ic_monthly.parquet").set_index("date")["ic"]

    print(f"selected ridge alpha: {alpha}")
    print(f"n months scored: {ic.shape[0]}")
    print()
    print(f"{'':14s}{'mean IC':>10s}{'std IC':>10s}{'IC IR':>10s}{'mean spread':>14s}")
    print(f"{'ridge':14s}{ic.mean():10.4f}{ic.std():10.4f}{ic.mean()/ic.std():10.4f}"
          f"{spread.mean():14.4f}")
    print(f"{'champion':14s}{champ_ic.mean():10.4f}{champ_ic.std():10.4f}"
          f"{champ_ic.mean()/champ_ic.std():10.4f}")

    common = ic.index.intersection(champ_ic.index)
    beat_frac = (ic.loc[common] > champ_ic.loc[common]).mean()
    print()
    print(f"ridge IC > champion IC in {beat_frac:.0%} of the {len(common)} overlapping months")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
