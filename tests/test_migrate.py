import importlib.util
from pathlib import Path

import pandas as pd

from beat_snp500.io_utils import atomic_write_json, read_json


def _mod():
    spec = importlib.util.spec_from_file_location(
        "migrate_model_names", Path("scripts/migrate_model_names.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_registry_migrates_idempotently(tmp_path):
    mig = _mod()
    reg = tmp_path / "registry.json"
    atomic_write_json([{"type": "champion"}, {"type": "lgbm"}], reg)
    assert mig.migrate_registry(reg) is True
    assert [e["type"] for e in read_json(reg)] == ["lgbm", "lgbm"]
    assert mig.migrate_registry(reg) is False


def test_output_files_migrate_idempotently(tmp_path):
    mig = _mod()
    (tmp_path / "holdings_champion.json").write_text("{}")
    (tmp_path / "leaderboard_challenger.json").write_text("{}")
    moved = mig.migrate_output_files(tmp_path)
    assert sorted(moved) == ["holdings_lgbm.json", "leaderboard_kmeans.json"]
    assert (tmp_path / "holdings_lgbm.json").exists()
    assert not (tmp_path / "holdings_champion.json").exists()
    assert mig.migrate_output_files(tmp_path) == []


def test_live_track_migrates_idempotently(tmp_path):
    mig = _mod()
    p = tmp_path / "live_track.parquet"
    pd.DataFrame({"date": [pd.Timestamp("2026-07-01")] * 2,
                  "model": ["champion", "challenger"],
                  "ret": [0.0, 0.0], "spy_ret": [0.0, 0.0]}).to_parquet(p)
    assert mig.migrate_live_track(p) is True
    assert sorted(pd.read_parquet(p)["model"]) == ["kmeans", "lgbm"]
    assert mig.migrate_live_track(p) is False
