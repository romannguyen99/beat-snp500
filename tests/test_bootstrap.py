import numpy as np
import pandas as pd
import pytest

from beat_snp500.backtest.bootstrap import monthly_holding_returns, random_portfolio_bootstrap


def test_monthly_holding_returns_alignment():
    idx = pd.bdate_range("2024-01-01", "2024-03-29")
    close = pd.DataFrame({"A": np.linspace(100, 130, len(idx))}, index=idx)
    hr = monthly_holding_returns(close, pd.to_datetime(["2024-01-31", "2024-02-29"]))
    mclose = close.groupby(close.index + pd.offsets.MonthEnd(0)).last()["A"]
    assert hr.loc["2024-01-31", "A"] == pytest.approx(mclose.iloc[1] / mclose.iloc[0] - 1)


def test_identical_stocks_collapse_band():
    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    tickers = [f"T{i}" for i in range(30)]
    hr = pd.DataFrame(0.01, index=dates, columns=tickers)
    uni = {t: tickers for t in dates}
    out = random_portfolio_bootstrap(uni, hr, n_draws=50, n_picks=10, cost_bps=0.0)
    band = out["band"]
    assert np.allclose(band["p05"], band["p95"])
    assert band["p50"].iloc[-1] == pytest.approx(1.01**12)
    assert len(out["cagr"]) == 50


def test_costs_lower_the_band():
    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    tickers = [f"T{i}" for i in range(30)]
    hr = pd.DataFrame(0.01, index=dates, columns=tickers)
    uni = {t: tickers for t in dates}
    free = random_portfolio_bootstrap(uni, hr, n_draws=10, cost_bps=0.0)
    costly = random_portfolio_bootstrap(uni, hr, n_draws=10, cost_bps=10.0)
    assert costly["band"]["p50"].iloc[-1] < free["band"]["p50"].iloc[-1]


def test_per_month_pick_counts():
    dates = pd.date_range("2024-01-31", periods=1, freq="ME")
    tickers = [f"T{i}" for i in range(30)]
    # 15 winners (+10%), 15 losers (0%). A 1-name draw lands on +10% in ~half
    # of draws, so p95 == 1.10 iff the per-month count of 1 is respected; the
    # default 10-name draw's p95 sits far below 1.10 (needs all 10 winners).
    hr = pd.DataFrame(0.0, index=dates, columns=tickers)
    hr.iloc[0, :15] = 0.10
    uni = {t: tickers for t in dates}
    out = random_portfolio_bootstrap(uni, hr, n_draws=200,
                                     n_picks={dates[0]: 1}, cost_bps=0.0)
    assert out["band"]["p95"].iloc[0] == pytest.approx(1.10)
