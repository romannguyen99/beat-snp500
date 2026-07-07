import numpy as np
import pandas as pd
import pytest

from beat_snp500 import config


@pytest.fixture
def make_prices():
    def _make(tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"),
              start="2013-01-01", end="2020-12-31", seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range(start, end)
        frames = []
        for i, t in enumerate(tickers):
            rets = rng.normal(0.0003 + 0.0001 * i, 0.02, len(dates))
            close = 50.0 * np.exp(np.cumsum(rets))
            open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, len(dates))))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, len(dates))))
            volume = rng.integers(100_000, 5_000_000, len(dates)).astype(float)
            frames.append(pd.DataFrame({
                "date": dates, "ticker": t, "open": open_, "high": high,
                "low": low, "close": close, "volume": volume,
            }))
        return pd.concat(frames, ignore_index=True)
    return _make


@pytest.fixture
def make_factors():
    def _make(start="2013-01-31", end="2021-01-31", seed=1):
        dates = pd.date_range(start, end, freq="ME")
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(
            rng.normal(0.003, 0.02, (len(dates), 5)),
            index=dates, columns=["mkt_rf", "smb", "hml", "rmw", "cma"],
        )
        df["rf"] = 0.0001
        df.index.name = "date"
        return df
    return _make


@pytest.fixture
def make_membership():
    def _make(tickers=("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"),
              start="2013-01-31", end="2021-01-31"):
        dates = pd.date_range(start, end, freq="ME")
        idx = pd.MultiIndex.from_product([dates, list(tickers)], names=["date", "ticker"])
        return idx.to_frame(index=False)
    return _make


@pytest.fixture
def make_panel():
    def _make(n_months=50, n_tickers=40, seed=0, signal_coef=0.1, noise=0.001):
        dates = pd.date_range("2015-01-31", periods=n_months, freq="ME")
        tickers = [f"T{i:02d}" for i in range(n_tickers)]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        rng = np.random.default_rng(seed)
        df = pd.DataFrame(rng.normal(size=(len(idx), len(config.FEATURES))),
                          index=idx, columns=config.FEATURES)
        df["close"] = 100.0
        df["fwd_return_1m"] = signal_coef * df["return_12m"] + rng.normal(0, noise, len(idx))
        return df
    return _make
