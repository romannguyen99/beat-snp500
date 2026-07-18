import importlib.util
from pathlib import Path

from beat_snp500 import tracking


def _mod():
    spec = importlib.util.spec_from_file_location(
        "promote_model", Path("scripts/promote_model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_promote_rolls_back_alias(tmp_path):
    t = tracking.Tracker("production",
                         tracking_uri=(tmp_path / "mlruns").as_uri(),
                         registry_uri=f"sqlite:///{tmp_path}/reg.db")
    t.register_model_version(artifact="models/a.txt", run_id=None, tags={})
    t.register_model_version(artifact="models/b.txt", run_id=None, tags={})
    assert _mod().promote(1, t) == "models/a.txt"
    assert t.current_model_artifact() == "models/a.txt"
