import numpy as np
import pandas as pd
import pytest

from beat_snp500.backtest.metrics import perf_metrics, yearly_returns


def test_constant_return_metrics():
    r = pd.Series(0.001, index=pd.bdate_range("2020-01-01", periods=252))
    m = perf_metrics(r)
    assert m["total_return"] == pytest.approx(1.001**252 - 1)
    assert m["cagr"] == pytest.approx(1.001**252 - 1)
    assert m["max_drawdown"] == 0.0
    assert m["ann_vol"] == pytest.approx(0.0)


def test_max_drawdown():
    r = pd.Series([0.10, -0.50], index=pd.bdate_range("2020-01-01", periods=2))
    m = perf_metrics(r)
    assert m["max_drawdown"] == pytest.approx(-0.50)


def test_sharpe_uses_risk_free():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 504),
                  index=pd.bdate_range("2020-01-01", periods=504))
    assert perf_metrics(r, rf_annual=0.05)["sharpe"] < perf_metrics(r, rf_annual=0.0)["sharpe"]


def test_yearly_returns():
    idx = pd.to_datetime(["2020-06-30", "2020-12-31", "2021-06-30"])
    y = yearly_returns(pd.Series([0.10, 0.10, -0.10], index=idx))
    assert y.loc[2020] == pytest.approx(1.1 * 1.1 - 1)
    assert y.loc[2021] == pytest.approx(-0.10)
