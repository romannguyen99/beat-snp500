import pandas as pd
import pytest

from beat_snp500.data.factors import load_ff5


def fake_fetcher():
    idx = pd.period_range("2020-01", "2020-03", freq="M")
    return pd.DataFrame(
        {"Mkt-RF": [1.0, -2.0, 3.0], "SMB": [0.5] * 3, "HML": [0.1] * 3,
         "RMW": [0.2] * 3, "CMA": [0.3] * 3, "RF": [0.01] * 3},
        index=idx,
    )


def test_load_ff5_normalizes(tmp_path):
    df = load_ff5(tmp_path / "ff5.parquet", fetcher=fake_fetcher)
    assert list(df.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert df.index[0] == pd.Timestamp("2020-01-31")
    assert df.index.name == "date"
    assert df["mkt_rf"].iloc[0] == pytest.approx(0.01)


def test_load_ff5_falls_back_to_cache(tmp_path):
    path = tmp_path / "ff5.parquet"
    load_ff5(path, fetcher=fake_fetcher)

    def boom():
        raise ConnectionError("offline")

    df = load_ff5(path, fetcher=boom)
    assert len(df) == 3

    path.unlink()
    with pytest.raises(ConnectionError):
        load_ff5(path, fetcher=boom)


def test_load_ff5_propagates_transform_errors(tmp_path):
    path = tmp_path / "ff5.parquet"
    load_ff5(path, fetcher=fake_fetcher)  # seed a valid cache

    def bad_schema():
        idx = pd.period_range("2020-01", "2020-02", freq="M")
        return pd.DataFrame({"WRONG": [1.0, 2.0]}, index=idx)

    with pytest.raises(KeyError):
        load_ff5(path, fetcher=bad_schema)
