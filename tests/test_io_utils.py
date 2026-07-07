import pandas as pd

from beat_snp500.io_utils import atomic_write_json, atomic_write_parquet, read_json


def test_parquet_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    p = tmp_path / "sub" / "f.parquet"
    atomic_write_parquet(df, p)
    assert p.exists() and not p.with_suffix(".parquet.tmp").exists()
    pd.testing.assert_frame_equal(pd.read_parquet(p), df)


def test_json_roundtrip(tmp_path):
    p = tmp_path / "sub" / "f.json"
    atomic_write_json({"k": [1, 2]}, p)
    assert read_json(p) == {"k": [1, 2]}
