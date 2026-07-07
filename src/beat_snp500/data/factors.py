from pathlib import Path

import pandas as pd

from beat_snp500.io_utils import atomic_write_parquet

RENAME = {"Mkt-RF": "mkt_rf", "SMB": "smb", "HML": "hml",
          "RMW": "rmw", "CMA": "cma", "RF": "rf"}


def _default_fetcher(start):
    import pandas_datareader.data as web
    return web.DataReader("F-F_Research_Data_5_Factors_2x3", "famafrench", start=start)[0]


def load_ff5(cache_path: Path, start: str = "2008-01-01", fetcher=None) -> pd.DataFrame:
    cache_path = Path(cache_path)
    fetch = fetcher if fetcher is not None else (lambda: _default_fetcher(start))
    try:
        raw = fetch()
    except Exception:
        if cache_path.exists():
            return pd.read_parquet(cache_path).set_index("date")
        raise
    df = raw.rename(columns=RENAME)[list(RENAME.values())].div(100.0)
    idx = raw.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp(how="end").normalize()
    df.index = pd.DatetimeIndex(idx) + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    atomic_write_parquet(df.reset_index(), cache_path)
    return df
