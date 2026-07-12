import pandas as pd
import pytest

from beat_snp500.features.advanced import (cluster_relative_momentum,
                                            residual_momentum,
                                            vol_scaled_momentum)

BETA_COLS = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]


def _zero_factors(panel):
    f = pd.DataFrame(0.0, index=panel.index.get_level_values("date").unique(),
                     columns=["mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    f.index.name = "date"
    return f


def test_vol_scaled_momentum(make_panel):
    panel = make_panel(n_months=3, n_tickers=4)
    out = vol_scaled_momentum(panel)
    assert out.name == "mom_vol_scaled"
    row = panel.iloc[5]
    assert out.iloc[5] == pytest.approx(row["return_12m"] / row["gk_vol"])


def test_residual_momentum_zero_betas_reduces_to_return_momentum(make_panel):
    panel = make_panel(n_months=16, n_tickers=4)
    panel[BETA_COLS] = 0.0
    out = residual_momentum(panel, _zero_factors(panel))
    s = panel.xs("T00", level="ticker")["return_1m"]
    roll = s.shift(1).rolling(11)      # months t-11..t-1, skipping month t
    t = s.index[-1]
    assert out.loc[(t, "T00")] == pytest.approx((roll.sum() / roll.std()).loc[t])


def test_residual_momentum_no_lookahead(make_panel):
    panel = make_panel(n_months=16, n_tickers=4)
    panel[BETA_COLS] = 0.0
    factors = _zero_factors(panel)
    base = residual_momentum(panel, factors)
    bumped_panel = panel.copy()
    last = panel.index.get_level_values("date").max()
    bumped_panel.loc[bumped_panel.index.get_level_values("date") == last,
                     "return_1m"] = 9.9
    bumped = residual_momentum(bumped_panel, factors)
    keep = base.index.get_level_values("date") < last
    pd.testing.assert_series_equal(base[keep], bumped[keep])


def test_cluster_relative_momentum_zero_mean_within_month(make_panel):
    panel = make_panel(n_months=2, n_tickers=30)
    out = cluster_relative_momentum(panel)
    assert list(out.columns) == ["return_3m_cz", "return_6m_cz", "return_12m_cz"]
    assert set(out.index.get_level_values("date").unique()) == \
        set(panel.index.get_level_values("date").unique())
    # each cluster is zero-mean, so each month's aggregate mean is ~0
    assert (out.groupby(level="date").mean().abs() < 1e-9).all().all()
