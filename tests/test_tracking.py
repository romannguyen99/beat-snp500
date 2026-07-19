import sys
import types

import pytest

from beat_snp500 import tracking


def _uris(tmp_path):
    return {"tracking_uri": (tmp_path / "mlruns").as_uri(),
            "registry_uri": f"sqlite:///{tmp_path}/reg.db"}


def test_logs_roundtrip_to_file_store(tmp_path):
    t = tracking.Tracker("tuning", **_uris(tmp_path))
    with t.start_run(run_name="r1") as run:
        t.log_params({"k": 4})
        t.log_metrics({"dev_sharpe": 1.23, "bad": float("nan")})
        run_id = run.info.run_id
    from mlflow import MlflowClient
    got = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri()).get_run(run_id)
    assert got.data.params["k"] == "4"
    assert got.data.metrics["dev_sharpe"] == pytest.approx(1.23)
    assert "bad" not in got.data.metrics  # NaN metrics are dropped


def _broken_mlflow():
    boom = types.ModuleType("mlflow")

    def _raise(*args, **kwargs):
        raise RuntimeError("store down")

    boom.set_tracking_uri = _raise
    return boom


def test_non_strict_warns_instead_of_raising(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlflow", _broken_mlflow())
    t = tracking.Tracker("production", strict=False, **_uris(tmp_path))
    with pytest.warns(UserWarning, match="mlflow tracking skipped"):
        with t.start_run(run_name="x") as run:
            assert run is None
            t.log_metrics({"m": 1.0})


def test_strict_raises(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mlflow", _broken_mlflow())
    t = tracking.Tracker("tuning", **_uris(tmp_path))
    with pytest.raises(RuntimeError, match="store down"):
        with t.start_run(run_name="x"):
            pass


def test_non_strict_log_metrics_bad_value_warns(tmp_path):
    t = tracking.Tracker("tuning", strict=False, **_uris(tmp_path))
    with t.start_run(run_name="r"):
        with pytest.warns(UserWarning, match="mlflow tracking skipped"):
            t.log_metrics({"m": None})


def test_default_uris_use_config_paths(tmp_path):
    t = tracking.Tracker("tuning")  # no URIs: defaults from patched config
    assert t.tracking_uri == (tmp_path / "mlruns").as_uri()
    with t.start_run(run_name="d"):
        t.log_metrics({"x": 1.0})
    assert (tmp_path / "mlruns").exists()


def test_register_and_resolve_current(tmp_path):
    t = tracking.Tracker("production", **_uris(tmp_path))
    assert t.current_model_artifact() is None
    v1 = t.register_model_version(artifact="models/a.txt", run_id=None,
                                  tags={"ic_mean": 0.01})
    v2 = t.register_model_version(artifact="models/b.txt", run_id=None,
                                  tags={"ic_mean": 0.02})
    assert (v1, v2) == (1, 2)
    assert t.current_model_artifact() == "models/b.txt"


def test_set_current_moves_alias(tmp_path):
    t = tracking.Tracker("production", **_uris(tmp_path))
    t.register_model_version(artifact="models/a.txt", run_id=None, tags={})
    t.register_model_version(artifact="models/b.txt", run_id=None, tags={})
    assert t.set_current(1) == "models/a.txt"
    assert t.current_model_artifact() == "models/a.txt"


def test_crashed_run_recorded_failed(tmp_path):
    t = tracking.Tracker("tuning", **_uris(tmp_path))
    with pytest.raises(ValueError, match="boom"):
        with t.start_run(run_name="crash") as run:
            run_id = run.info.run_id
            raise ValueError("boom")
    from mlflow import MlflowClient
    got = MlflowClient(tracking_uri=(tmp_path / "mlruns").as_uri()).get_run(run_id)
    assert got.info.status == "FAILED"


def test_current_model_artifact_distinguishes_empty_from_broken(tmp_path):
    t = tracking.Tracker("production", **_uris(tmp_path))
    assert t.current_model_artifact() is None  # empty registry: quiet None
    # MLflow 3.14 creates missing sqlite parent dirs itself, so a merely
    # absent directory is NOT broken. A file squatting on the parent path
    # makes it truly unopenable (makedirs raises FileExistsError).
    (tmp_path / "blocker").write_text("")
    broken_uri = f"sqlite:///{tmp_path}/blocker/reg.db"
    strict = tracking.Tracker("production",
                              tracking_uri=(tmp_path / "mlruns").as_uri(),
                              registry_uri=broken_uri)
    with pytest.raises(Exception):
        strict.current_model_artifact()
    soft = tracking.Tracker("production", strict=False,
                            tracking_uri=(tmp_path / "mlruns").as_uri(),
                            registry_uri=broken_uri)
    with pytest.warns(UserWarning, match="mlflow tracking skipped"):
        assert soft.current_model_artifact() is None
