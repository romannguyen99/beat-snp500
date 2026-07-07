from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from beat_snp500 import config
from beat_snp500.io_utils import atomic_write_parquet

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_ticker(t: str) -> str:
    return t.strip().upper().replace(".", "-")


def fetch_wikipedia_tables(timeout: int = 30):
    resp = requests.get(WIKI_URL, headers={"User-Agent": "beat-snp500/0.1"}, timeout=timeout)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return tables[0], tables[1]


def _clean(cell) -> str | None:
    if isinstance(cell, str) and cell.strip():
        return normalize_ticker(cell)
    return None


def parse_changes(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(p) for p in c).strip().lower() for c in df.columns]
    else:
        df.columns = [str(c).strip().lower() for c in df.columns]
    # Wikipedia has used both "Date" and "Effective Date" as the header text for
    # this column over time; match on substring rather than prefix so either survives.
    date_col = next(c for c in df.columns if "date" in c)
    added_col = next(c for c in df.columns if c.startswith("added") and "ticker" in c)
    removed_col = next(c for c in df.columns if c.startswith("removed") and "ticker" in c)
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "added": df[added_col].map(_clean),
        "removed": df[removed_col].map(_clean),
    })
    out = out.dropna(subset=["date"]).sort_values("date", ascending=False)
    return out.reset_index(drop=True)


def build_membership(current: list[str], changes: pd.DataFrame, start, end) -> pd.DataFrame:
    month_ends = pd.date_range(start, pd.Timestamp(end) + pd.offsets.MonthEnd(0), freq="ME")
    ch = changes.sort_values("date", ascending=False).reset_index(drop=True)
    members = {normalize_ticker(t) for t in current}
    rows = []
    i = 0
    for me in reversed(month_ends):
        # changes dated exactly on a month-end count as already effective for that month
        while i < len(ch) and ch.loc[i, "date"] > me:
            if ch.loc[i, "added"]:
                members.discard(ch.loc[i, "added"])
            if ch.loc[i, "removed"]:
                members.add(ch.loc[i, "removed"])
            i += 1
        rows.extend((me, t) for t in sorted(members))
    out = pd.DataFrame(rows, columns=["date", "ticker"])
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


def refresh_membership(path: Path, start=config.HISTORY_START) -> tuple[pd.DataFrame, bool]:
    try:
        current_df, changes_raw = fetch_wikipedia_tables()
        symbol_col = "Symbol" if "Symbol" in current_df.columns else current_df.columns[0]
        current = current_df[symbol_col].astype(str).tolist()
        changes = parse_changes(changes_raw)
    except Exception:
        if Path(path).exists():
            return pd.read_parquet(path), False
        raise
    mem = build_membership(current, changes, start, pd.Timestamp.today())
    atomic_write_parquet(mem, path)
    return mem, True
