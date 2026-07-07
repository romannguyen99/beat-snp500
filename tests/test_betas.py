import numpy as np
import pandas as pd
import pytest

from beat_snp500.features.betas import rolling_ff5_betas

BETA_COLS = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]


def make_inputs(n=40, seed=3):
    dates = pd.date_range("2015-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    factors = pd.DataFrame(rng.normal(0, 0.02, (n, 5)), index=dates,
                           columns=["mkt_rf", "smb", "hml", "rmw", "cma"])
    factors["rf"] = 0.0
    factors.index.name = "date"
    ret = 2.0 * factors["mkt_rf"] + 0.5 * factors["smb"]  # exact linear model
    idx = pd.MultiIndex.from_product([dates, ["AAA"]], names=["date", "ticker"])
    return pd.Series(ret.values, index=idx, name="return_1m"), factors


def test_recovers_true_betas():
    ret, factors = make_inputs()
    b = rolling_ff5_betas(ret, factors, window=24)
    last = b.xs("AAA", level="ticker").dropna().iloc[-1]
    assert last["beta_mkt"] == pytest.approx(2.0, abs=1e-6)
    assert last["beta_smb"] == pytest.approx(0.5, abs=1e-6)
    assert last["beta_hml"] == pytest.approx(0.0, abs=1e-6)


def test_betas_are_lagged_one_month():
    ret, factors = make_inputs()
    b = rolling_ff5_betas(ret, factors, window=24).xs("AAA", level="ticker")
    # first non-NaN row must appear one month AFTER the window is first full
    first_valid = b["beta_mkt"].first_valid_index()
    assert first_valid == b.index[24]  # window fills at index 23, lag moves it to 24


def test_short_history_returns_nan():
    ret, factors = make_inputs(n=10)
    b = rolling_ff5_betas(ret, factors, window=24)
    assert b[BETA_COLS].isna().all().all()
