import pandas as pd
import pytest

from beat_snp500.data.membership import (
    build_membership, normalize_ticker, parse_changes, refresh_membership,
)


def test_normalize_ticker():
    assert normalize_ticker("BRK.B") == "BRK-B"
    assert normalize_ticker(" bf.b ") == "BF-B"


def test_parse_changes_flattens_wikipedia_multiindex():
    cols = pd.MultiIndex.from_tuples([
        ("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    raw = pd.DataFrame(
        [["June 15, 2020", "CCC", "C Corp", "DDD", "D Corp", "cap change"],
         ["March 2, 2020", "", "", "EEE", "E Corp", "acquired"]],
        columns=cols,
    )
    ch = parse_changes(raw)
    assert list(ch.columns) == ["date", "added", "removed"]
    assert ch.loc[0, "date"] == pd.Timestamp("2020-06-15")
    assert ch.loc[0, "added"] == "CCC" and ch.loc[0, "removed"] == "DDD"
    assert ch.loc[1, "added"] is None and ch.loc[1, "removed"] == "EEE"


def test_parse_changes_handles_effective_date_header():
    cols = pd.MultiIndex.from_tuples([
        ("Effective Date", "Effective Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    raw = pd.DataFrame([["June 15, 2020", "CCC", "C Corp", "DDD", "D Corp", "x"]], columns=cols)
    ch = parse_changes(raw)
    assert ch.loc[0, "date"] == pd.Timestamp("2020-06-15")
    assert ch.loc[0, "added"] == "CCC"


def test_build_membership_walks_changes_backward():
    changes = pd.DataFrame({
        "date": [pd.Timestamp("2020-06-15")],
        "added": ["CCC"], "removed": ["DDD"],
    })
    mem = build_membership(["AAA", "BBB", "CCC"], changes, "2020-04-01", "2020-07-31")
    may = set(mem[mem["date"] == "2020-05-31"]["ticker"])
    june = set(mem[mem["date"] == "2020-06-30"]["ticker"])
    assert may == {"AAA", "BBB", "DDD"}
    assert june == {"AAA", "BBB", "CCC"}


def test_refresh_membership_falls_back_to_cache(tmp_path, monkeypatch):
    import beat_snp500.data.membership as m
    path = tmp_path / "membership.parquet"
    cached = pd.DataFrame({"date": [pd.Timestamp("2020-05-31")], "ticker": ["AAA"]})
    cached.to_parquet(path, index=False)

    def boom():
        raise ConnectionError("offline")

    monkeypatch.setattr(m, "fetch_wikipedia_tables", boom)
    mem, fresh = refresh_membership(path)
    assert not fresh
    assert mem["ticker"].tolist() == ["AAA"]

    path.unlink()
    with pytest.raises(ConnectionError):
        refresh_membership(path)


def test_refresh_membership_propagates_build_errors(tmp_path, monkeypatch):
    import beat_snp500.data.membership as m
    cols = pd.MultiIndex.from_tuples([
        ("Date", "Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    raw = pd.DataFrame([["June 15, 2020", "CCC", "C Corp", "DDD", "D Corp", "x"]], columns=cols)
    monkeypatch.setattr(m, "fetch_wikipedia_tables",
                        lambda: (pd.DataFrame({"Symbol": ["AAA"]}), raw))
    def boom(*a, **k):
        raise RuntimeError("bug in build_membership")
    monkeypatch.setattr(m, "build_membership", boom)
    path = tmp_path / "membership.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2020-05-31")], "ticker": ["AAA"]}).to_parquet(path, index=False)
    with pytest.raises(RuntimeError, match="bug in build_membership"):
        m.refresh_membership(path)
