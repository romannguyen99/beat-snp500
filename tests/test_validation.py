import pandas as pd

from beat_snp500.data.validation import validate_prices


def make_prices(last_date, tickers=("AAA", "BBB")):
    dates = pd.bdate_range("2024-01-01", last_date)
    return pd.DataFrame(
        [{"date": d, "ticker": t, "close": 1.0} for d in dates for t in tickers]
    )


def test_clean_data_passes():
    r = validate_prices(make_prices("2024-06-14"), ["AAA", "BBB"],
                        as_of=pd.Timestamp("2024-06-14"))
    assert r.ok and r.issues == []
    assert r.stats["n_tickers"] == 2


def test_stale_data_fails():
    r = validate_prices(make_prices("2024-05-01"), ["AAA", "BBB"],
                        as_of=pd.Timestamp("2024-06-14"))
    assert not r.ok and any("stale" in i for i in r.issues)


def test_too_many_missing_tickers_fails():
    r = validate_prices(make_prices("2024-06-14"), ["AAA", "BBB", "CCC", "DDD"],
                        as_of=pd.Timestamp("2024-06-14"))
    assert not r.ok and any("missing" in i for i in r.issues)


def test_nan_close_fails():
    df = make_prices("2024-06-14")
    df.loc[df.index[: len(df) // 4], "close"] = float("nan")
    r = validate_prices(df, ["AAA", "BBB"], as_of=pd.Timestamp("2024-06-14"))
    assert not r.ok and any("nan" in i.lower() for i in r.issues)


def test_junk_ticker_does_not_mask_staleness():
    # Real members (AAA, BBB) are stale, but a junk/delisted ticker (ZZZ)
    # keeps printing fresh dates. The gate must key off the real members'
    # clock, not the spoofable global max date.
    df = make_prices("2024-05-01", tickers=("AAA", "BBB"))
    junk = pd.DataFrame(
        [{"date": d, "ticker": "ZZZ", "close": 1.0}
         for d in pd.bdate_range("2024-01-01", "2024-06-14")]
    )
    df = pd.concat([df, junk], ignore_index=True)
    r = validate_prices(df, ["AAA", "BBB"], as_of=pd.Timestamp("2024-06-14"))
    assert not r.ok and any("stale" in i for i in r.issues)
    assert r.stats["global_max_date"] == "2024-06-14"


def test_implausible_return_in_window_fails():
    df = make_prices("2024-06-14", tickers=("AAA", "BBB"))
    dates = sorted(df["date"].unique())
    recent_date = dates[-3]  # inside the last-10-business-day window
    df.loc[(df["ticker"] == "AAA") & (df["date"] == recent_date), "close"] = 10.0
    r = validate_prices(df, ["AAA", "BBB"], as_of=pd.Timestamp("2024-06-14"))
    assert not r.ok
    assert any("implausible return" in i for i in r.issues)


def test_implausible_return_outside_window_does_not_fail():
    df = make_prices("2024-06-14", tickers=("AAA", "BBB"))
    dates = sorted(df["date"].unique())
    old_date = dates[20]  # well before the recent window
    df.loc[(df["ticker"] == "AAA") & (df["date"] == old_date), "close"] = 10.0
    r = validate_prices(df, ["AAA", "BBB"], as_of=pd.Timestamp("2024-06-14"))
    assert r.ok
    assert not any("implausible return" in i for i in r.issues)


def test_empty_prices_fails_cleanly():
    empty = pd.DataFrame(columns=["date", "ticker", "close"])
    r = validate_prices(empty, ["AAA"], as_of=pd.Timestamp("2024-06-14"))
    assert not r.ok
    assert any("no price rows" in i for i in r.issues)
