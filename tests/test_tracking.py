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
