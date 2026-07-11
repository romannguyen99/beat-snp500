from pathlib import Path

import lightgbm as lgb
import pandas as pd

from beat_snp500 import config
from beat_snp500.portfolio.weights import conviction_weights

LGB_PARAMS = dict(objective="regression", n_estimators=200, learning_rate=0.05,
                  num_leaves=31, min_child_samples=20, subsample=0.8,
                  colsample_bytree=0.8, random_state=config.SEED, verbosity=-1)

CANDIDATE_PARAMS = [
    {**LGB_PARAMS, "learning_rate": 0.05, "num_leaves": 31},
    {**LGB_PARAMS, "learning_rate": 0.10, "num_leaves": 15},
    {**LGB_PARAMS, "learning_rate": 0.03, "num_leaves": 63},
]


def _rank_label(fwd: pd.Series) -> pd.Series:
    return fwd.groupby(level="date").rank(pct=True)


def _months(panel: pd.DataFrame) -> pd.DatetimeIndex:
    return panel.index.get_level_values("date").unique().sort_values()


def _fit(panel: pd.DataFrame, months, params) -> lgb.LGBMRegressor:
    rows = panel[panel.index.get_level_values("date").isin(months)]
    y = _rank_label(rows["fwd_return_1m"]).dropna()
    X = rows.loc[y.index, config.FEATURES]
    return lgb.LGBMRegressor(**params).fit(X, y)


def _score_month(model, panel: pd.DataFrame, t) -> pd.Series:
    rows = panel.xs(t, level="date", drop_level=False)
    return pd.Series(model.predict(rows[config.FEATURES]), index=rows.index)


def spearman_ic(scores: pd.Series, fwd: pd.Series) -> pd.Series:
    df = pd.concat({"s": scores, "f": fwd}, axis=1).dropna()
    return df.groupby(level="date").apply(
        lambda g: g["s"].corr(g["f"], method="spearman"))


def decile_spread(scores: pd.Series, fwd: pd.Series) -> pd.Series:
    df = pd.concat({"s": scores, "f": fwd}, axis=1).dropna()

    def per(g: pd.DataFrame) -> float:
        q = pd.qcut(g["s"].rank(method="first"), 10, labels=False)
        return g["f"][q == 9].mean() - g["f"][q == 0].mean()

    return df.groupby(level="date").apply(per)


def select_params(panel: pd.DataFrame,
                  train_window: int = config.TRAIN_WINDOW_MONTHS,
                  val_months: int = 6) -> dict:
    months = _months(panel)
    fit_months = months[: train_window - val_months]
    val = months[train_window - val_months: train_window]
    best, best_ic = CANDIDATE_PARAMS[0], -2.0
    for params in CANDIDATE_PARAMS:
        model = _fit(panel, fit_months, params)
        scores = pd.concat([_score_month(model, panel, t) for t in val])
        ic = spearman_ic(scores, panel["fwd_return_1m"]).mean()
        if ic > best_ic:
            best, best_ic = params, ic
    return best


def walk_forward_scores(panel: pd.DataFrame,
                        train_window: int = config.TRAIN_WINDOW_MONTHS,
                        params: dict | None = None) -> pd.Series:
    months = _months(panel)
    if params is None:
        params = select_params(panel, train_window=train_window)
    out = []
    for i in range(train_window, len(months)):
        model = _fit(panel, months[i - train_window: i], params)
        out.append(_score_month(model, panel, months[i]))
    return pd.concat(out).sort_index() if out else pd.Series(dtype=float)


def lgbm_must_buys(scores_month: pd.Series,
                   threshold: float = config.MUST_BUY_Z_LGBM,
                   min_picks: int = config.MIN_PICKS,
                   max_picks: int = config.MAX_PICKS) -> dict[str, float]:
    """{ticker: cross-sectional score z} clearing the conviction bar; {} = hold.

    A z-threshold, not a raw predicted-percentile bar: regression predictions
    compress toward the centre when signal is weak, so a raw 0.9 cut would
    select ~0 names most months. threshold must stay > 0 (weights need
    strictly positive signals).
    """
    s = scores_month.dropna()
    sd = s.std(ddof=0)
    if len(s) < min_picks or sd == 0:
        return {}
    z = (s - s.mean()) / sd
    must = z[z >= threshold].sort_values(ascending=False)
    if len(must) < min_picks:
        return {}
    return must.head(max_picks).to_dict()


def lgbm_picks(scores: pd.Series) -> dict:
    out = {}
    for t, s in scores.groupby(level="date"):
        signals = lgbm_must_buys(s.droplevel("date"))
        if signals:
            out[t] = conviction_weights(signals)
    return out


def train_lgbm(labeled_panel: pd.DataFrame,
               train_window: int = config.TRAIN_WINDOW_MONTHS,
               params: dict | None = None):
    params = params or LGB_PARAMS
    months = _months(labeled_panel)[-train_window:]
    model = _fit(labeled_panel, months, params)
    # trailing validation: refit without the last 6 months, measure IC on them
    if len(months) >= 12:
        val = months[-6:]
        val_model = _fit(labeled_panel, months[:-6], params)
        scores = pd.concat([_score_month(val_model, labeled_panel, t) for t in val])
        val_ic = float(spearman_ic(scores, labeled_panel["fwd_return_1m"]).mean())
    else:
        val_ic = float("nan")
    return model, val_ic


def save_model(model: lgb.LGBMRegressor, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(path))


def load_model(path: Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))
