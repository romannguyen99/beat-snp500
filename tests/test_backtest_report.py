import pandas as pd
import pytest

from beat_snp500.io_utils import read_json
from beat_snp500.jobs.backtest_report import run_report, survivorship_stats


def test_run_report_writes_artifacts(make_prices, make_factors, make_membership, tmp_path):
    # enough tickers that a k=4 momentum cluster is reliably >= MIN_PICKS;
    # kmeans_must_buys's min-cluster-size guard skips months where it isn't
    # (see test_kmeans.py::test_small_momentum_cluster_holds)
    tickers = tuple(f"S{i:02d}" for i in range(60)) + ("SPY",)
    prices = make_prices(tickers=tickers)
    membership = make_membership(tickers=tuple(t for t in tickers if t != "SPY"))
    metrics = run_report(prices, membership, make_factors(), tmp_path,
                         top_n=60, train_window=12, n_draws=20)
    for f in ["equity_curves.parquet", "metrics.json", "ic_monthly.parquet",
              "bootstrap.parquet", "bootstrap_summary.json", "survivorship.parquet",
              "picks.json"]:
        assert (tmp_path / f).exists(), f
    m = read_json(tmp_path / "metrics.json")
    for series in ["kmeans", "champion", "spy"]:
        assert "cagr" in m[series]
    assert "champion_cagr_percentile" in read_json(tmp_path / "bootstrap_summary.json")
    assert "survivorship" in m
    surv = pd.read_parquet(tmp_path / "survivorship.parquet")
    assert (surv["missing_frac"] == 0).all()

    picks_data = read_json(tmp_path / "picks.json")
    assert set(picks_data) == {"kmeans", "champion", "kmeans_ms", "champion_ms"}
    assert picks_data["kmeans"], "kmeans emitted no months at all"
    for weights in picks_data["kmeans"].values():
        assert 5 <= len(weights) <= 10
        assert sum(weights.values()) == pytest.approx(1.0)
        assert max(weights.values()) <= 0.20 + 1e-9


def test_survivorship_stats_flags_missing_member():
    membership = pd.DataFrame({
        "date": [pd.Timestamp("2020-01-31")] * 3,
        "ticker": ["AAA", "BBB", "CCC"],
    })
    close = pd.DataFrame(
        {"AAA": [1.0, 2.0], "BBB": [float("nan"), float("nan")]},
        index=pd.to_datetime(["2020-01-01", "2020-01-31"]))
    stats = survivorship_stats(membership, close)
    row = stats[stats["date"] == pd.Timestamp("2020-01-31")].iloc[0]
    assert row["n_members"] == 3
    assert row["n_with_data"] == 1  # AAA has data; BBB all-NaN; CCC absent entirely
    assert row["missing_frac"] == pytest.approx(2 / 3)
