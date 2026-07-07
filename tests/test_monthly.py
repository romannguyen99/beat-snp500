import numpy as np
import pandas as pd
import pytest

from beat_snp500.features.monthly import (
    add_forward_return, add_momentum, apply_membership, daily_indicators,
    liquidity_filter, monthly_panel,
)


def build_monthly(make_prices):
    return monthly_panel(daily_indicators(make_prices()))


def test_monthly_panel_shape(make_prices):
    m = build_monthly(make_prices)
    assert m.index.names == ["date", "ticker"]
    dates = m.index.get_level_values("date")
    assert (dates == dates + pd.offsets.MonthEnd(0)).all()  # all month-ends
    for col in ["close", "dollar_volume", "gk_vol", "rsi", "atr_norm", "bb_width", "macd_hist"]:
        assert col in m.columns


def test_momentum_matches_close_ratio(make_prices):
    m = add_momentum(build_monthly(make_prices))
    close = m["close"].unstack("ticker")
    t = close.index[20]
    expected = close.loc[t, "AAA"] / close.iloc[17]["AAA"] - 1
    got = m.loc[(t, "AAA"), "return_3m"]
    # equal unless clipped by the (tiny-sample) winsor bounds
    lo = close.pct_change(3).loc[t].quantile(0.005)
    hi = close.pct_change(3).loc[t].quantile(0.995)
    assert got == pytest.approx(float(np.clip(expected, lo, hi)))


def test_winsorization_is_cross_sectional(make_prices):
    prices = make_prices()
    # inject a crazy outlier in one ticker's close for one month
    mask = (prices["ticker"] == "AAA") & (prices["date"] == "2015-06-30")
    prices.loc[mask, "close"] *= 50
    m = add_momentum(monthly_panel(daily_indicators(prices)))
    t = pd.Timestamp("2015-06-30")
    r = m["return_1m"].xs(t, level="date")
    assert r["AAA"] == r.max()  # clipped to that month's upper quantile
    assert r["AAA"] <= m["return_1m"].xs(t, level="date").quantile(0.995) + 1e-12


def test_winsorization_clips_negative_outliers(make_prices):
    prices = make_prices()
    mask = (prices["ticker"] == "AAA") & (prices["date"] == "2015-06-30")
    prices.loc[mask, "close"] *= 0.02  # -98% crash for one month-end
    m = add_momentum(monthly_panel(daily_indicators(prices)))
    t = pd.Timestamp("2015-06-30")
    r = m["return_1m"].xs(t, level="date")
    assert r["AAA"] == r.min()
    raw = monthly_panel(daily_indicators(prices))["close"].unstack("ticker").pct_change(1, fill_method=None)
    assert r["AAA"] > raw.loc[t, "AAA"]  # actually clipped upward from the raw crash return


def test_forward_return_is_next_month(make_prices):
    m = add_forward_return(build_monthly(make_prices))
    close = m["close"].unstack("ticker")
    t, t1 = close.index[10], close.index[11]
    expected = close.loc[t1, "BBB"] / close.loc[t, "BBB"] - 1
    assert m.loc[(t, "BBB"), "fwd_return_1m"] == pytest.approx(expected)
    assert np.isnan(m.loc[(close.index[-1], "BBB"), "fwd_return_1m"])


def test_apply_membership(make_prices, make_membership):
    m = build_monthly(make_prices)
    mem = make_membership(tickers=("AAA", "BBB"))
    out = apply_membership(m, mem)
    assert set(out.index.get_level_values("ticker")) == {"AAA", "BBB"}


def test_liquidity_filter_keeps_top_n(make_prices):
    m = build_monthly(make_prices)
    out = liquidity_filter(m, top_n=3, window=12)
    counts = out.groupby(level="date").size()
    assert (counts <= 3).all()
    assert counts.index.min() >= m.index.get_level_values("date").unique()[11]
