import numpy as np
import pandas as pd

from beat_snp500 import config
from beat_snp500.models.challenger import challenger_picks, kmeans_top10


def make_month(n=40, n_hot=10, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n)]
    df = pd.DataFrame(rng.normal(0, 0.3, (n, len(config.FEATURES))),
                      index=pd.Index(tickers, name="ticker"), columns=config.FEATURES)
    hot = tickers[:n_hot]
    for c in ["return_3m", "return_6m", "return_12m"]:
        df.loc[hot, c] += 5.0  # unmistakable momentum cluster
    return df, hot


def test_kmeans_top10_finds_momentum_cluster():
    month, hot = make_month()
    picks = kmeans_top10(month)
    assert len(picks) == 10
    assert set(picks) == set(hot)


def test_kmeans_deterministic():
    month, _ = make_month()
    assert kmeans_top10(month) == kmeans_top10(month)


def test_too_few_stocks_returns_empty():
    month, _ = make_month(n=3, n_hot=1)
    assert kmeans_top10(month) == []


def test_challenger_picks_shapes():
    month, hot = make_month()
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.concat({d: month for d in dates}, names=["date"])
    picks = challenger_picks(panel)
    assert set(picks) == set(dates)
    for w in picks.values():
        assert len(w) == 10
        assert sum(w.values()) == 1.0 or abs(sum(w.values()) - 1.0) < 1e-9
