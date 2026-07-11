import pandas as pd

from beat_snp500 import config
from beat_snp500.models.lgbm import (lgbm_must_buys, lgbm_picks, load_model,
                                      save_model, spearman_ic, train_lgbm,
                                      walk_forward_scores)


def test_lgbm_must_buys_thresholds_on_zscore():
    # 15 mediocre names + 5 clear winners: only the winners clear z >= 1
    s = pd.Series({f"L{i:02d}": 0.0 for i in range(15)}
                  | {f"H{i:02d}": 10.0 for i in range(5)})
    must = lgbm_must_buys(s)
    assert set(must) == {f"H{i:02d}" for i in range(5)}
    assert all(v >= config.MUST_BUY_Z_LGBM for v in must.values())


def test_lgbm_must_buys_holds_when_conviction_is_thin():
    # uniform scores 0..19: only ~4 names reach z >= 1 -> below MIN_PICKS -> hold
    s = pd.Series({f"T{i:02d}": float(i) for i in range(20)})
    assert lgbm_must_buys(s) == {}


def test_lgbm_must_buys_degenerate_scores_hold():
    s = pd.Series({f"T{i:02d}": 1.0 for i in range(20)})  # zero dispersion
    assert lgbm_must_buys(s) == {}


def test_walk_forward_learns_planted_signal(make_panel):
    panel = make_panel(n_months=40, n_tickers=40)
    scores = walk_forward_scores(panel, train_window=12)
    assert len(scores.index.get_level_values("date").unique()) == 40 - 12
    assert spearman_ic(scores, panel["fwd_return_1m"]).mean() > 0.8


def test_scores_only_after_first_window(make_panel):
    panel = make_panel(n_months=20)
    scores = walk_forward_scores(panel, train_window=12)
    first_scored = scores.index.get_level_values("date").min()
    all_months = panel.index.get_level_values("date").unique().sort_values()
    assert first_scored == all_months[12]


def test_lgbm_picks_weights_valid(make_panel):
    panel = make_panel(n_months=20, n_tickers=30)
    scores = walk_forward_scores(panel, train_window=12)
    picks = lgbm_picks(scores)
    assert picks  # planted signal: at least some active months
    for w in picks.values():
        assert config.MIN_PICKS <= len(w) <= config.MAX_PICKS
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert max(w.values()) <= config.WEIGHT_CAP + 1e-9


def test_train_save_load_roundtrip(make_panel, tmp_path):
    panel = make_panel(n_months=30)
    model, val_ic = train_lgbm(panel.dropna(subset=["fwd_return_1m"]),
                               train_window=12)
    assert val_ic > 0.5
    p = tmp_path / "lgbm.txt"
    save_model(model, p)
    booster = load_model(p)
    month = panel.xs(panel.index.get_level_values("date").max(), level="date")
    assert len(booster.predict(month[config.FEATURES])) == len(month)
