import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from beat_snp500 import config
from beat_snp500.io_utils import atomic_write_parquet

PRICE_COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def _longify(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=PRICE_COLS)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw.stack(level=0, future_stack=True).rename_axis(["date", "ticker"]).reset_index()
    else:
        df = raw.rename_axis("date").reset_index()
        df["ticker"] = tickers[0]
    df.columns = [str(c).lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[[c for c in PRICE_COLS if c in df.columns]]
    return df.dropna(subset=["close"]).reset_index(drop=True)


def download_prices(tickers, start, end, retries: int = 3, backoff: float = 5.0) -> pd.DataFrame:
    tickers = sorted(set(tickers))
    last_err = None
    for attempt in range(retries):
        try:
            raw = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True,
                              progress=False, group_by="ticker", threads=True)
            return _longify(raw, tickers)
        except Exception as e:  # network hiccups, rate limits
            last_err = e
            time.sleep(backoff * 2**attempt)
    raise last_err


def update_price_cache(tickers, cache_path: Path, start=config.HISTORY_START, end=None,
                       downloader=download_prices, full_refresh: bool = False) -> pd.DataFrame:
    cache_path = Path(cache_path)
    end = end or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
    tickers = sorted(set(tickers))

    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=PRICE_COLS)
    if full_refresh or cache.empty:
        new = downloader(tickers, pd.Timestamp(start), end)
        if not cache.empty:
            # replace refreshed tickers wholesale; keep history of tickers not in this download
            cache = cache[~cache["ticker"].isin(set(new["ticker"].unique()))]
    else:
        fetch_start = pd.Timestamp(cache["date"].max()) - pd.DateOffset(days=5)
        new = downloader(tickers, fetch_start, end)

        # Adjusted-price basis-shift detection: a split/dividend event can
        # retroactively re-adjust a ticker's entire historical close series.
        # Compare the freshly fetched tail against the cached overlap; any
        # ticker whose basis moved gets a full-history refetch this run so
        # its cache never mixes two adjustment bases.
        overlap = cache.merge(new, on=["date", "ticker"], suffixes=("_old", "_new"))
        shifted = sorted(overlap.loc[
            (overlap["close_new"] / overlap["close_old"] - 1).abs() > 1e-3, "ticker"
        ].unique()) if not overlap.empty else []
        if shifted:
            reshifted = downloader(shifted, pd.Timestamp(start), end)
            cache = cache[~cache["ticker"].isin(set(shifted))]
            new = pd.concat([new, reshifted], ignore_index=True)

        missing = sorted(set(tickers) - set(cache["ticker"].unique()) - set(shifted))
        if missing:
            backfill = downloader(missing, pd.Timestamp(start), end)
            new = pd.concat([new, backfill], ignore_index=True)

    frames = [f for f in (cache, new) if not f.empty]
    out = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PRICE_COLS))
    out = (out.drop_duplicates(subset=["date", "ticker"], keep="last")
           .sort_values(["ticker", "date"])
           .reset_index(drop=True))
    atomic_write_parquet(out, cache_path)
    return out


def close_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pivot(index="date", columns="ticker", values="close").sort_index()
