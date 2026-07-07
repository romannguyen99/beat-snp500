from beat_snp500.models.champion import (
    champion_picks, decile_spread, load_model, save_model, spearman_ic,
    train_champion, walk_forward_scores,
)


def test_walk_forward_learns_planted_signal(make_panel):
    panel = make_panel(n_months=40, n_tickers=40)
    scores = walk_forward_scores(panel, train_window=12)
    months = scores.index.get_level_values("date").unique()
    assert len(months) == 40 - 12
    ic = spearman_ic(scores, panel["fwd_return_1m"])
    assert ic.mean() > 0.8  # fwd return is a clean function of return_12m


def test_scores_only_after_first_window(make_panel):
    panel = make_panel(n_months=20)
    scores = walk_forward_scores(panel, train_window=12)
    first_scored = scores.index.get_level_values("date").min()
    all_months = panel.index.get_level_values("date").unique().sort_values()
    assert first_scored == all_months[12]


def test_champion_picks_top10(make_panel):
    panel = make_panel(n_months=20, n_tickers=30)
    scores = walk_forward_scores(panel, train_window=12)
    picks = champion_picks(scores)
    t = sorted(picks)[-1]
    assert len(picks[t]) == 10
    top_by_feature = (panel.xs(t, level="date")["return_12m"].nlargest(12).index)
    assert len(set(picks[t]) & set(top_by_feature)) >= 8


def test_decile_spread_positive(make_panel):
    panel = make_panel(n_months=30, n_tickers=40)
    scores = walk_forward_scores(panel, train_window=12)
    ds = decile_spread(scores, panel["fwd_return_1m"])
    assert ds.mean() > 0


def test_train_save_load_roundtrip(make_panel, tmp_path):
    panel = make_panel(n_months=30)
    labeled = panel.dropna(subset=["fwd_return_1m"])
    model, val_ic = train_champion(labeled, train_window=12)
    assert val_ic > 0.5
    p = tmp_path / "champ.txt"
    save_model(model, p)
    booster = load_model(p)
    month = panel.xs(panel.index.get_level_values("date").max(), level="date")
    from beat_snp500 import config
    preds = booster.predict(month[config.FEATURES])
    assert len(preds) == len(month)
