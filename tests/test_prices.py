import pandas as pd

from beat_snp500.data.prices import PRICE_COLS, _longify, close_matrix, update_price_cache

COLS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def fake_downloader_factory(log):
    def fake(tickers, start, end, **kw):
        log.append((sorted(tickers), pd.Timestamp(start)))
        dates = pd.bdate_range(start, end)
        rows = [
            {"date": d, "ticker": t, "open": 1.0, "high": 1.0,
             "low": 1.0, "close": 2.0, "volume": 10.0}
            for d in dates for t in tickers
        ]
        return pd.DataFrame(rows, columns=COLS)
    return fake


def test_initial_fill_then_incremental(tmp_path):
    log = []
    dl = fake_downloader_factory(log)
    cache = tmp_path / "prices.parquet"
    df1 = update_price_cache(["AAA", "BBB"], cache, start="2024-01-01",
                             end="2024-01-10", downloader=dl)
    assert set(df1["ticker"]) == {"AAA", "BBB"}
    n1 = len(df1)

    df2 = update_price_cache(["AAA", "BBB"], cache, start="2024-01-01",
                             end="2024-01-20", downloader=dl)
    assert len(df2) > n1
    assert not df2.duplicated(["date", "ticker"]).any()
    # incremental fetch must not restart from the beginning
    assert log[-1][1] > pd.Timestamp("2024-01-01")


def test_new_ticker_gets_full_backfill(tmp_path):
    log = []
    dl = fake_downloader_factory(log)
    cache = tmp_path / "prices.parquet"
    update_price_cache(["AAA"], cache, start="2024-01-01", end="2024-01-10", downloader=dl)
    df = update_price_cache(["AAA", "NEW"], cache, start="2024-01-01",
                            end="2024-01-10", downloader=dl)
    new_rows = df[df["ticker"] == "NEW"]
    assert new_rows["date"].min() == pd.Timestamp("2024-01-01")


def test_full_refresh_refetches_from_start(tmp_path):
    log = []
    dl = fake_downloader_factory(log)
    cache = tmp_path / "prices.parquet"
    update_price_cache(["AAA"], cache, start="2024-01-01", end="2024-01-10", downloader=dl)
    update_price_cache(["AAA"], cache, start="2024-01-01", end="2024-01-20",
                       downloader=dl, full_refresh=True)
    assert log[-1][1] == pd.Timestamp("2024-01-01")


def test_close_matrix_shape(tmp_path):
    dl = fake_downloader_factory([])
    cache = tmp_path / "prices.parquet"
    df = update_price_cache(["AAA", "BBB"], cache, start="2024-01-01",
                            end="2024-01-10", downloader=dl)
    m = close_matrix(df)
    assert list(m.columns) == ["AAA", "BBB"]
    assert (m == 2.0).all().all()


def test_longify_multi_ticker_multiindex():
    idx = pd.date_range("2024-01-01", periods=2, tz="America/New_York")
    cols = pd.MultiIndex.from_product([["AAA", "BBB"], ["Open", "High", "Low", "Close", "Volume"]])
    raw = pd.DataFrame(1.0, index=idx, columns=cols)
    out = _longify(raw, ["AAA", "BBB"])
    assert list(out.columns) == PRICE_COLS
    assert len(out) == 4
    assert out["date"].dt.tz is None
    assert set(out["ticker"]) == {"AAA", "BBB"}


def test_longify_single_ticker_flat_columns():
    idx = pd.date_range("2024-01-01", periods=3)
    raw = pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 2.0, "Volume": 5.0}, index=idx)
    out = _longify(raw, ["AAA"])
    assert (out["ticker"] == "AAA").all()
    assert list(out.columns) == PRICE_COLS
    assert len(out) == 3


def test_full_refresh_preserves_unrequested_tickers(tmp_path):
    dl = fake_downloader_factory([])
    cache = tmp_path / "prices.parquet"
    update_price_cache(["AAA", "OLD"], cache, start="2024-01-01", end="2024-01-10", downloader=dl)
    df = update_price_cache(["AAA"], cache, start="2024-01-01", end="2024-01-19", downloader=dl, full_refresh=True)
    assert "OLD" in set(df["ticker"])
    assert df[df["ticker"] == "AAA"]["date"].max() > df[df["ticker"] == "OLD"]["date"].max()


def test_incremental_update_detects_basis_shift(tmp_path):
    # AAA undergoes a simulated 2:1 split between the two runs: every download
    # after the first returns AAA's close halved (the new adjusted basis),
    # while BBB's basis never changes.
    calls = []

    def dl(tickers, start, end, **kw):
        calls.append((sorted(tickers), pd.Timestamp(start)))
        dates = pd.bdate_range(start, end)
        rows = [
            {"date": d, "ticker": t,
             "open": (1.0 if (t == "AAA" and len(calls) > 1) else 2.0),
             "high": (1.0 if (t == "AAA" and len(calls) > 1) else 2.0),
             "low": (1.0 if (t == "AAA" and len(calls) > 1) else 2.0),
             "close": (1.0 if (t == "AAA" and len(calls) > 1) else 2.0),
             "volume": 10.0}
            for d in dates for t in tickers
        ]
        return pd.DataFrame(rows, columns=COLS)

    cache = tmp_path / "prices.parquet"
    update_price_cache(["AAA", "BBB"], cache, start="2024-01-01", end="2024-01-10", downloader=dl)
    df = update_price_cache(["AAA", "BBB"], cache, start="2024-01-01", end="2024-01-20", downloader=dl)

    aaa = df[df["ticker"] == "AAA"]["close"]
    bbb = df[df["ticker"] == "BBB"]["close"]
    assert (aaa == 1.0).all(), "AAA's whole cached history must be refetched onto the new basis"
    assert (bbb == 2.0).all(), "BBB's history must be untouched by AAA's basis shift"
