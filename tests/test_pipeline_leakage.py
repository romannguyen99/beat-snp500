import pandas as pd

from beat_snp500 import config
from beat_snp500.features.pipeline import build_feature_panel


def test_no_lookahead_in_features(make_prices, make_factors, make_membership):
    prices = make_prices()
    factors = make_factors()
    membership = make_membership()

    full = build_feature_panel(prices, membership, factors, top_n=8)
    dates = full.index.get_level_values("date").unique().sort_values()
    t = dates[-6]  # a month with plenty of history and plenty of future

    truncated = build_feature_panel(
        prices[prices["date"] <= t], membership, factors[factors.index <= t], top_n=8
    )

    full_t = full.xs(t, level="date")[config.FEATURES].sort_index()
    trunc_t = truncated.xs(t, level="date")[config.FEATURES].sort_index()
    pd.testing.assert_frame_equal(full_t, trunc_t)


def test_panel_has_expected_columns(make_prices, make_factors, make_membership):
    panel = build_feature_panel(make_prices(), make_membership(), make_factors(), top_n=8)
    for col in config.FEATURES + ["close", "fwd_return_1m"]:
        assert col in panel.columns
    assert panel[config.FEATURES].notna().all().all()
