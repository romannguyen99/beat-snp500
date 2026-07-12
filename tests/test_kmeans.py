import numpy as np
import pandas as pd
import pytest

from beat_snp500 import config
from beat_snp500.models.kmeans import (cluster_month, kmeans_must_buys,
                                        kmeans_picks)


def make_month(n=40, n_hot=10, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n)]
    df = pd.DataFrame(rng.normal(0, 0.3, (n, len(config.FEATURES))),
                      index=pd.Index(tickers, name="ticker"), columns=config.FEATURES)
    hot = tickers[:n_hot]
    for c in ["return_3m", "return_6m", "return_12m"]:
        df.loc[hot, c] += 5.0  # unmistakable momentum cluster
    return df, hot


def test_cluster_month_shapes():
    month, _ = make_month()
    Xz, labels = cluster_month(month)
    assert list(Xz.index) == list(labels.index)
    assert labels.nunique() == config.K_CLUSTERS


def test_must_buys_finds_momentum_cluster():
    month, hot = make_month()
    must = kmeans_must_buys(month)
    assert config.MIN_PICKS <= len(must) <= config.MAX_PICKS
    assert set(must) <= set(hot)
    assert all(v > config.MUST_BUY_Z_KMEANS for v in must.values())


def test_must_buys_capped_at_max_picks():
    month, hot = make_month(n=60, n_hot=20)
    must = kmeans_must_buys(month)
    assert len(must) <= config.MAX_PICKS
    assert set(must) <= set(hot)


def test_must_buys_deterministic():
    month, _ = make_month()
    assert kmeans_must_buys(month) == kmeans_must_buys(month)


def test_too_few_stocks_holds():
    month, _ = make_month(n=3, n_hot=1)
    assert kmeans_must_buys(month) == {}


def test_small_momentum_cluster_holds():
    # the guard from improve_v1: a tiny trending cluster must never become
    # a concentrated bet
    month, _ = make_month(n=40, n_hot=3)
    assert kmeans_must_buys(month) == {}


def test_kmeans_picks_weights_valid():
    month, _ = make_month()
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.concat({d: month for d in dates}, names=["date"])
    picks = kmeans_picks(panel)
    assert set(picks) == set(dates)
    for w in picks.values():
        assert config.MIN_PICKS <= len(w) <= config.MAX_PICKS
        assert sum(w.values()) == 1.0 or abs(sum(w.values()) - 1.0) < 1e-9
        assert max(w.values()) <= config.WEIGHT_CAP + 1e-9


def test_mom_modes_and_select_rules_run():
    month, hot = make_month()
    for mom_mode in ("mean_3_6_12", "12_1", "vol_scaled"):
        for select_rule in ("mean", "risk_adj"):
            must = kmeans_must_buys(month, mom_mode=mom_mode,
                                    select_rule=select_rule)
            assert isinstance(must, dict)  # may be {} (hold) but never crash


def test_unknown_mode_raises():
    month, _ = make_month()
    with pytest.raises(ValueError):
        kmeans_must_buys(month, mom_mode="nope")
    with pytest.raises(ValueError):
        kmeans_must_buys(month, select_rule="nope")
