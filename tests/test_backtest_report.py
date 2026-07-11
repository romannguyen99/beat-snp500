import pandas as pd
import pytest

from beat_snp500.io_utils import read_json
from beat_snp500.jobs.backtest_report import run_report, survivorship_stats


def test_run_report_writes_artifacts(make_prices, make_factors, make_membership, tmp_path):
    # enough tickers that a k=4 momentum cluster is reliably >= N_PICKS=10;
    # kmeans_top10's min-cluster-size guard skips months where it isn't
    # (see test_challenger.py::test_small_momentum_cluster_returns_empty)
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
    for series in ["champion", "challenger", "spy"]:
        assert "cagr" in m[series]
    assert "champion_cagr_percentile" in read_json(tmp_path / "bootstrap_summary.json")
    assert "survivorship" in m
    surv = pd.read_parquet(tmp_path / "survivorship.parquet")
    # membership tickers are a subset of price tickers in this fixture, so
    # every member has price history and missing_frac should be exactly 0.
    assert (surv["missing_frac"] == 0).all()

    picks_data = read_json(tmp_path / "picks.json")
    assert set(picks_data) == {"champion", "challenger", "champion_ms", "challenger_ms"}
    ic_months = pd.read_parquet(tmp_path / "ic_monthly.parquet")
    # champion picks cover every walk-forward-scored month; ic_monthly drops
    # the trailing month once its forward return isn't realized yet, so picks
    # can have at most one more month than ic (the still-in-flight last one).
    assert len(picks_data["champion"]) - len(ic_months) in (0, 1)


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
