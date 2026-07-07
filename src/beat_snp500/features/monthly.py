import numpy as np
import pandas as pd

from beat_snp500 import config
from beat_snp500.features.technical import (
    atr_norm, bb_width, garman_klass_vol, macd_hist, rsi,
)

IND_COLS = ["gk_vol", "rsi", "atr_norm", "bb_width", "macd_hist"]


def daily_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    def per(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        # A synthetic OHLC row that violates normal price invariants (e.g. an
        # injected outlier close far outside that day's high/low) can drive
        # the Garman-Klass estimator negative before the sqrt, which numpy
        # flags as "invalid value encountered in sqrt". The result is
        # correctly NaN either way; errstate just silences the benign warning
        # for that edge case without masking anything upstream.
        with np.errstate(invalid="ignore"):
            g["gk_vol"] = garman_klass_vol(g["open"], g["high"], g["low"], g["close"]).values
        g["rsi"] = rsi(g["close"]).values
        g["atr_norm"] = atr_norm(g["high"], g["low"], g["close"]).values
        g["bb_width"] = bb_width(g["close"]).values
        g["macd_hist"] = macd_hist(g["close"]).values
        g["dollar_volume"] = (g["close"] * g["volume"]).values
        return g

    # pandas 2.3 emits a DeprecationWarning from groupby(...).apply when the
    # applied function receives the grouping column back. Iterating the
    # groupby and concatenating avoids that path entirely while keeping the
    # documented schema (long frame, indicator + dollar_volume columns added,
    # ticker column intact) unchanged.
    parts = [per(g) for _, g in prices.groupby("ticker", group_keys=False)]
    return pd.concat(parts, ignore_index=True)


def monthly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    # Group by (calendar month-end, ticker) directly instead of
    # pd.Grouper(freq="ME"): on this pandas/numpy build, Grouper's resample
    # binning path hits a "generic unit for NumPy timedelta" DeprecationWarning
    # internal to pandas. Adding a MonthEnd offset to the date column produces
    # an identical grouping (verified against the Grouper output) without
    # going through that code path.
    month_end = daily["date"] + pd.offsets.MonthEnd(0)
    g = daily.groupby([month_end, "ticker"])
    out = g.agg(
        close=("close", "last"),
        dollar_volume=("dollar_volume", "mean"),
        **{c: (c, "last") for c in IND_COLS},
    )
    out.index.names = ["date", "ticker"]
    return out.sort_index()


def add_momentum(monthly: pd.DataFrame, lags=config.MOMENTUM_LAGS,
                 winsor_pct: float = config.WINSOR_PCT) -> pd.DataFrame:
    monthly = monthly.copy()
    close = monthly["close"].unstack("ticker")
    for lag in lags:
        r = close.pct_change(lag, fill_method=None).stack()
        # Clip to actual observed order statistics rather than linearly
        # interpolated values: with a small monthly cross-section, linear
        # interpolation blends the outlier itself into the bound, so clipping
        # barely moves an extreme value. Interpolation is tail-appropriate —
        # "higher" for the lower bound and "lower" for the upper bound — so
        # each tail's extreme point is always pulled inward to its neighbor
        # (all-"lower" would make the 0.005 bound the group minimum for any
        # n < 200, leaving negative outliers unclipped).
        lo = r.groupby(level="date").transform(lambda s: s.quantile(winsor_pct, interpolation="higher"))
        hi = r.groupby(level="date").transform(lambda s: s.quantile(1 - winsor_pct, interpolation="lower"))
        monthly[f"return_{lag}m"] = r.clip(lo, hi)
    return monthly


def add_forward_return(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    close = monthly["close"].unstack("ticker")
    monthly["fwd_return_1m"] = close.pct_change(fill_method=None).shift(-1).stack(future_stack=True)
    return monthly


def apply_membership(monthly: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    mem_idx = pd.MultiIndex.from_frame(membership[["date", "ticker"]])
    return monthly.loc[monthly.index.isin(mem_idx)]


def liquidity_filter(monthly: pd.DataFrame, top_n: int = config.UNIVERSE_SIZE,
                     window: int = 12) -> pd.DataFrame:
    dv = monthly["dollar_volume"].unstack("ticker")
    avg = dv.rolling(window, min_periods=window).mean()
    rank = avg.rank(axis=1, ascending=False)
    keep = rank.stack()
    keep = keep[keep <= top_n].index
    return monthly.loc[monthly.index.isin(keep)]
