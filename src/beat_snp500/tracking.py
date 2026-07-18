"""Git-native MLflow wiring (spec: docs/superpowers/specs/
2026-07-18-mlflow-tracking-design.md).

Tracking runs live in the repo's mlruns/ file store; the model registry
lives in models/mlflow_registry.db (SQLite via a separate registry URI).
Both are committed to git, so a clone reproduces full history with no
server. Job code talks to this module, never to MLflow directly: the daily
pipeline constructs Tracker(strict=False) so telemetry can warn but never
block publishing picks.
"""
import os
import warnings
from contextlib import contextmanager

from beat_snp500 import config

# MLflow 3.x gates its maintenance-mode file store behind this flag; the
# git-native design (spec §1) depends on the file store. setdefault so an
# environment can still override.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

REGISTERED_MODEL = "lgbm"
CURRENT_ALIAS = "current"


def _default_tracking_uri() -> str:
    config.MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MLRUNS_DIR.as_uri()


def _default_registry_uri() -> str:
    config.MLFLOW_REGISTRY_DB.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{config.MLFLOW_REGISTRY_DB}"


class Tracker:
    """One experiment's logging handle."""

    def __init__(self, experiment: str, strict: bool = True,
                 tracking_uri: str | None = None,
                 registry_uri: str | None = None):
        self.experiment = experiment
        self.strict = strict
        self.tracking_uri = tracking_uri or _default_tracking_uri()
        self.registry_uri = registry_uri or _default_registry_uri()

    def _mlflow(self):
        import mlflow
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        return mlflow

    def _guarded(self, fn):
        try:
            return fn()
        except Exception as exc:
            if self.strict:
                raise
            warnings.warn(f"mlflow tracking skipped: {exc}", stacklevel=3)
            return None

    @contextmanager
    def start_run(self, run_name=None, nested=False, run_id=None):
        def _enter():
            mlflow = self._mlflow()
            mlflow.set_experiment(self.experiment)
            return mlflow.start_run(run_name=run_name, nested=nested,
                                    run_id=run_id)
        run = self._guarded(_enter)
        try:
            yield run
        finally:
            if run is not None:
                self._guarded(lambda: self._mlflow().end_run())

    def log_params(self, params: dict) -> None:
        self._guarded(lambda: self._mlflow().log_params(params))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items() if v == v}
        self._guarded(lambda: self._mlflow().log_metrics(clean, step=step))

    def set_tags(self, tags: dict) -> None:
        self._guarded(lambda: self._mlflow().set_tags(tags))
