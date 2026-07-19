import importlib.util
from pathlib import Path

from mlflow import MlflowClient

from beat_snp500 import tracking
from beat_snp500.io_utils import atomic_write_json


def _mod():
    spec = importlib.util.spec_from_file_location(
        "migrate_registry_to_mlflow",
        Path("scripts/migrate_registry_to_mlflow.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_migrates_entries_chronologically(tmp_path):
    reg = tmp_path / "registry.json"
    # Newer created_at deliberately first in the file: migrate() must sort
    # by created_at, not trust file order.
    atomic_write_json([
        {"model_id": "champion_202606", "type": "lgbm",
         "trained_through": "2026-05-31", "train_window_months": 36,
         "ic_mean": 0.014, "created_at": "2026-07-07",
         "artifact": "models/champion_202606.txt"},
        {"model_id": "champion_202605", "type": "lgbm",
         "trained_through": "2026-04-30", "train_window_months": 36,
         "ic_mean": 0.001, "created_at": "2026-06-01",
         "artifact": "models/champion_202605.txt"},
    ], reg)
    t = tracking.Tracker("production",
                         tracking_uri=(tmp_path / "mlruns").as_uri(),
                         registry_uri=f"sqlite:///{tmp_path}/reg.db")
    assert _mod().migrate(reg, t) == 2
    # @current lands on the entry with the newest created_at.
    assert t.current_model_artifact() == "models/champion_202606.txt"
    client = MlflowClient(tracking_uri=t.tracking_uri,
                          registry_uri=t.registry_uri)
    mvs = client.search_model_versions("name='lgbm'")
    assert len(mvs) == 2
    for mv in mvs:
        assert mv.tags["type"] == "lgbm"
