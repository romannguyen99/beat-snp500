import pandas as pd
import pytest

from beat_snp500.backtest.engine import run_backtest


def toy_close():
    idx = pd.to_datetime(["2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"])
    return pd.DataFrame(
        {"A": [100.0, 105.0, 110.0, 110.0], "B": [100.0, 100.0, 100.0, 100.0]},
        index=idx,
    )


def test_single_month_hand_computed():
    picks = {pd.Timestamp("2024-01-31"): {"A": 0.5, "B": 0.5}}
    res = run_backtest(picks, toy_close(), cost_bps=10.0)
    r = res.daily_returns
    # day1: value 0.5*1.05+0.5*1.0 = 1.025 ; minus 10bps * turnover(=1.0)
    assert r.loc["2024-02-01"] == pytest.approx(0.025 - 0.001)
    # day2: 1.05/1.025 - 1
    assert r.loc["2024-02-02"] == pytest.approx(1.05 / 1.025 - 1)
    assert res.turnover.iloc[0] == pytest.approx(1.0)


def test_second_rebalance_turnover_uses_drifted_weights():
    picks = {
        pd.Timestamp("2024-01-31"): {"A": 0.5, "B": 0.5},
        pd.Timestamp("2024-02-29"): {"A": 0.5, "B": 0.5},
    }
    res = run_backtest(picks, toy_close(), cost_bps=10.0)
    # drifted end-of-Feb weights: A = .5*1.1/1.05 ≈ .52381, B ≈ .47619
    assert res.turnover.iloc[1] == pytest.approx(2 * (0.5 * 1.1 / 1.05 - 0.5), abs=1e-9)


def test_missing_ticker_dropped_and_weights_renormalized():
    picks = {pd.Timestamp("2024-01-31"): {"A": 0.5, "ZZZ": 0.5}}
    res = run_backtest(picks, toy_close(), cost_bps=0.0)
    assert res.daily_returns.loc["2024-02-01"] == pytest.approx(0.05)


def test_empty_picks_gives_empty_result():
    res = run_backtest({}, toy_close())
    assert res.daily_returns.empty


def test_skipped_periods_are_recorded():
    picks = {
        pd.Timestamp("2023-12-29"): {"ZZZ": 1.0},   # no such ticker -> skipped
        pd.Timestamp("2024-01-31"): {"A": 0.5, "B": 0.5},
    }
    res = run_backtest(picks, toy_close(), cost_bps=0.0)
    assert res.skipped == [pd.Timestamp("2023-12-29")]
    assert not res.daily_returns.empty
