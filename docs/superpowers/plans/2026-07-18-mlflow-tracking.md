# MLflow Tracking + Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire MLflow experiment tracking (git-committed `mlruns/` file store) and MLflow Model Registry (git-committed SQLite) into all four pipeline entry points, replacing `models/registry.json` as the registry of record, per the approved spec `docs/superpowers/specs/2026-07-18-mlflow-tracking-design.md`.

**Architecture:** A new `src/beat_snp500/tracking.py` module owns all MLflow interaction through a `Tracker` class (per-experiment handle; `strict=False` downgrades every MLflow failure to a warning so the daily job can never be blocked by telemetry). Tracking uses the merge-friendly file store at `mlruns/`; the registry uses SQLite at `models/mlflow_registry.db` via a separate registry URI, written only at monthly retrain. Only `lgbm` is registered (booster stays in `models/`, referenced by repo-relative path); `kmeans` stays config-defined and `config.CHAMPION` remains the role pointer.

**Tech Stack:** Python 3.11+, mlflow-skinny + SQLAlchemy + Alembic (CI/runtime), full mlflow in `[dev]` for the local UI, pytest, existing pandas/LightGBM stack.

## Global Constraints

- `pandas>=2.2,<3.0` — pinned; do not upgrade (codebase targets 2.x semantics).
- MLflow pinned `>=2.15,<4` (skinny at runtime, full in `[dev]`); Task 1's spike confirms skinny+SQLite registry works before anything else builds on it.
- Scripts must be run as `PYTHONPATH=src .venv/bin/python scripts/<name>.py` — the sandbox re-stamps UF_HIDDEN on the editable-install `.pth` and Python 3.13 skips hidden `.pth` files. pytest is immune (`pythonpath = ["src"]` in pyproject).
- Test bar: full suite green with **zero warnings** (currently 109 passed).
- Commits: authored by Roman only — **no Co-Authored-By trailers**, message style `feat:`/`fix:`/`docs:`/`data:` as in git log.
- Model binaries never enter `mlruns/`; registry writes happen only at monthly retrain or manual migration/promotion.
- `mlruns/` and `models/mlflow_registry.db` are committed to git; `mlruns/.trash/` is gitignored.

---

### Task 1: Dependencies + spike (skinny + SQLite registry)

**Files:**
- Modify: `pyproject.toml:5-22`

**Interfaces:**
- Produces: importable `mlflow` (skinny) with working SQLite registry backend — every later task assumes this.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml`, change the two dependency blocks:

```toml
dependencies = [
    "pandas>=2.2,<3.0",
    "numpy>=1.26",
    "yfinance>=0.2.40",
    "pandas-datareader>=0.10",
    "statsmodels>=0.14",
    "scikit-learn>=1.4",
    "lightgbm>=4.3",
    "PyPortfolioOpt>=1.5.5",
    "pyarrow>=15",
    "requests>=2.31",
    "lxml>=5.1",
    "plotly>=5.20",
    "streamlit>=1.33",
    "mlflow-skinny>=2.15,<4",
    "sqlalchemy>=2",
    "alembic>=1.13",
]

[project.optional-dependencies]
dev = ["pytest>=8", "pyyaml>=6", "mlflow>=2.15,<4"]
```

- [ ] **Step 2: Install**

Run: `.venv/bin/pip install -e ".[dev]"`
Expected: installs mlflow (full, dev extra supersedes skinny in this venv), sqlalchemy, alembic without dependency conflicts. If the resolver complains about pandas/pyarrow pins, adjust the mlflow upper bound downward and note it in the commit message.

- [ ] **Step 3: Spike — prove SQLite registry + file tracking work together**

Run:

```bash
PYTHONPATH=src .venv/bin/python - <<'EOF'
import tempfile, pathlib
from mlflow import MlflowClient
d = pathlib.Path(tempfile.mkdtemp())
c = MlflowClient(tracking_uri=(d / "mlruns").as_uri(),
                 registry_uri=f"sqlite:///{d}/reg.db")
c.create_registered_model("lgbm")
mv = c.create_model_version("lgbm", source="models/x.txt", run_id=None)
c.set_registered_model_alias("lgbm", "current", mv.version)
got = c.get_model_version_by_alias("lgbm", "current")
assert got.source == "models/x.txt", got.source
print("SPIKE OK: version", got.version, "source", got.source)
EOF
```

Expected: `SPIKE OK: version 1 source models/x.txt`. If this fails under mlflow-skinny in a clean env later (CI), the documented fallback is moving full `mlflow` into core dependencies — record whichever holds.

- [ ] **Step 4: Verify existing suite still green**

Run: `.venv/bin/pytest -q`
Expected: `109 passed`, zero warnings.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add mlflow-skinny + sqlalchemy deps for tracking/registry"
```

---

### Task 2: config paths + `tracking.Tracker` (logging half)

**Files:**
- Modify: `src/beat_snp500/config.py:3-12`
- Create: `src/beat_snp500/tracking.py`
- Modify: `tests/conftest.py` (append autouse fixture)
- Test: `tests/test_tracking.py`

**Interfaces:**
- Produces: `config.MLRUNS_DIR: Path`, `config.MLFLOW_REGISTRY_DB: Path`; `tracking.Tracker(experiment: str, strict: bool = True, tracking_uri: str | None = None, registry_uri: str | None = None)` with context manager `start_run(run_name=None, nested=False, run_id=None)` (yields the mlflow ActiveRun or `None` on guarded failure) and methods `log_params(dict)`, `log_metrics(dict, step=None)` (drops NaN values), `set_tags(dict)`. Constants `tracking.REGISTERED_MODEL = "lgbm"`, `tracking.CURRENT_ALIAS = "current"`.

- [ ] **Step 1: Add store paths to config**

In `src/beat_snp500/config.py`, after `REGISTRY_JSON = MODELS_DIR / "registry.json"` add:

```python
MLRUNS_DIR = ROOT / "mlruns"
MLFLOW_REGISTRY_DB = MODELS_DIR / "mlflow_registry.db"
```

- [ ] **Step 2: Add autouse fixture so no test ever writes the real stores**

Append to `tests/conftest.py`:

```python
@pytest.fixture(autouse=True)
def _mlflow_stores_in_tmp(tmp_path, monkeypatch):
    """Never let a test write the repo's real mlruns/ or registry DB."""
    monkeypatch.setattr(config, "MLRUNS_DIR", tmp_path / "mlruns")
    monkeypatch.setattr(config, "MLFLOW_REGISTRY_DB",
                        tmp_path / "mlflow_registry.db")
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_tracking.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tracking.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'beat_snp500.tracking'`.

- [ ] **Step 5: Implement `tracking.py` (logging half)**

Create `src/beat_snp500/tracking.py`:

```python
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tracking.py -v`
Expected: 3 passed.

- [ ] **Step 7: Full suite**

Run: `.venv/bin/pytest -q`
Expected: `112 passed`, zero warnings.

- [ ] **Step 8: Commit**

```bash
git add src/beat_snp500/config.py src/beat_snp500/tracking.py tests/test_tracking.py tests/conftest.py
git commit -m "feat: add git-native mlflow Tracker (file-store tracking, guarded failures)"
```

---

### Task 3: Tracker registry operations

**Files:**
- Modify: `src/beat_snp500/tracking.py` (append methods to `Tracker`)
- Test: `tests/test_tracking.py` (append)

**Interfaces:**
- Consumes: `Tracker` from Task 2.
- Produces: `Tracker.register_model_version(*, artifact: str, run_id: str | None, tags: dict) -> int | None` (creates the `lgbm` registered model if missing, adds a version whose `source` is the repo-relative artifact ref, moves `@current` to it, returns the version number); `Tracker.current_model_artifact() -> str | None` (the `@current` version's source, or `None` if nothing registered); `Tracker.set_current(version: int) -> str | None` (moves `@current` to an explicit version — manual promotion/rollback — returns that version's source).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tracking.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_tracking.py::test_register_and_resolve_current -v`
Expected: FAIL with `AttributeError: 'Tracker' object has no attribute 'current_model_artifact'`.

- [ ] **Step 3: Implement registry methods**

Append to the `Tracker` class in `src/beat_snp500/tracking.py`:

```python
    def _client(self):
        from mlflow import MlflowClient
        return MlflowClient(tracking_uri=self.tracking_uri,
                            registry_uri=self.registry_uri)

    def register_model_version(self, *, artifact: str, run_id: str | None,
                               tags: dict) -> int | None:
        def _register():
            from mlflow.exceptions import MlflowException
            client = self._client()
            try:
                client.create_registered_model(REGISTERED_MODEL)
            except MlflowException:
                pass  # already exists
            mv = client.create_model_version(
                REGISTERED_MODEL, source=artifact, run_id=run_id,
                tags={k: str(v) for k, v in tags.items()})
            client.set_registered_model_alias(REGISTERED_MODEL,
                                              CURRENT_ALIAS, mv.version)
            return int(mv.version)
        return self._guarded(_register)

    def current_model_artifact(self) -> str | None:
        try:
            mv = self._client().get_model_version_by_alias(REGISTERED_MODEL,
                                                           CURRENT_ALIAS)
            return mv.source
        except Exception:
            return None  # empty/absent registry: serve no lgbm rather than crash

    def set_current(self, version: int) -> str | None:
        def _set():
            client = self._client()
            client.set_registered_model_alias(REGISTERED_MODEL, CURRENT_ALIAS,
                                              version)
            return client.get_model_version_by_alias(REGISTERED_MODEL,
                                                     CURRENT_ALIAS).source
        return self._guarded(_set)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tracking.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/tracking.py tests/test_tracking.py
git commit -m "feat: mlflow registry ops - register lgbm versions, @current alias resolve"
```

---

### Task 4: Registry tooling scripts (migration + manual promotion)

**Files:**
- Create: `scripts/migrate_registry_to_mlflow.py`, `scripts/promote_model.py`
- Test: `tests/test_migrate_registry.py`, `tests/test_promote_model.py`

**Interfaces:**
- Consumes: `Tracker.register_model_version` / `Tracker.set_current` (Task 3); `models/registry.json` (2 historical entries, both `type: "lgbm"`).
- Produces: populated `models/mlflow_registry.db` committed to git, `@current` → newest historical entry; `scripts/promote_model.py <version>` for manual promotion/rollback (spec §1/§5). `registry.json` is NOT deleted here (daily.py still reads it until Task 5; deletion is Task 6).

- [ ] **Step 1: Write the failing test**

Create `tests/test_migrate_registry.py` (script import mirrors `tests/test_migrate.py`):

```python
import importlib.util
from pathlib import Path

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
    atomic_write_json([
        {"model_id": "champion_202605", "type": "lgbm",
         "trained_through": "2026-04-30", "train_window_months": 36,
         "ic_mean": 0.001, "created_at": "2026-06-01",
         "artifact": "models/champion_202605.txt"},
        {"model_id": "champion_202606", "type": "lgbm",
         "trained_through": "2026-05-31", "train_window_months": 36,
         "ic_mean": 0.014, "created_at": "2026-07-07",
         "artifact": "models/champion_202606.txt"},
    ], reg)
    t = tracking.Tracker("production",
                         tracking_uri=(tmp_path / "mlruns").as_uri(),
                         registry_uri=f"sqlite:///{tmp_path}/reg.db")
    assert _mod().migrate(reg, t) == 2
    assert t.current_model_artifact() == "models/champion_202606.txt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_migrate_registry.py -v`
Expected: FAIL (script file does not exist).

- [ ] **Step 3: Implement the script**

Create `scripts/migrate_registry_to_mlflow.py`:

```python
"""One-shot import of models/registry.json into the MLflow model registry
(spec §3): entries become versions in chronological order, original fields
preserved as tags, @current ends on the newest. registry.json itself is
removed in a follow-up commit once jobs/daily.py no longer reads it.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/migrate_registry_to_mlflow.py
"""
from beat_snp500 import config, tracking
from beat_snp500.io_utils import read_json


def migrate(registry_json, tracker: tracking.Tracker) -> int:
    entries = read_json(registry_json)
    for e in entries:
        tracker.register_model_version(
            artifact=e["artifact"], run_id=None,
            tags={"model_id": e["model_id"],
                  "trained_through": e["trained_through"],
                  "train_window_months": e["train_window_months"],
                  "ic_mean": e["ic_mean"], "created_at": e["created_at"],
                  "migrated_from": "models/registry.json"})
    return len(entries)


if __name__ == "__main__":
    n = migrate(config.REGISTRY_JSON, tracking.Tracker("production"))
    print(f"migrated {n} entries; @current -> version {n}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_migrate_registry.py -v`
Expected: 1 passed.

- [ ] **Step 5: Run the migration for real**

Run: `PYTHONPATH=src .venv/bin/python scripts/migrate_registry_to_mlflow.py`
Expected: `migrated 2 entries; @current -> version 2`, and `models/mlflow_registry.db` now exists. Verify:

```bash
PYTHONPATH=src .venv/bin/python -c "
from beat_snp500 import tracking
print(tracking.Tracker('production').current_model_artifact())"
```

Expected: `models/champion_202606.txt`.

- [ ] **Step 6: Write the failing promotion test**

Create `tests/test_promote_model.py`:

```python
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
```

Run: `.venv/bin/pytest tests/test_promote_model.py -v`
Expected: FAIL (script file does not exist).

- [ ] **Step 7: Implement the promotion script**

Create `scripts/promote_model.py`:

```python
"""Move the lgbm @current alias to an explicit registered version (manual
promotion or rollback). Two-writer rule (spec §5): CI owns the registry —
run `git pull` before this, and push promptly after.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/promote_model.py 2
"""
import argparse

from beat_snp500 import tracking


def promote(version: int, tracker: tracking.Tracker) -> str | None:
    return tracker.set_current(version)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("version", type=int, help="registered lgbm version number")
    args = ap.parse_args()
    source = promote(args.version, tracking.Tracker("production"))
    print(f"@current -> version {args.version} ({source})")
```

Run: `.venv/bin/pytest tests/test_promote_model.py -v`
Expected: 1 passed.

- [ ] **Step 8: Commit (scripts + tests + real DB)**

```bash
git add scripts/migrate_registry_to_mlflow.py scripts/promote_model.py tests/test_migrate_registry.py tests/test_promote_model.py models/mlflow_registry.db
git commit -m "feat: migrate registry.json into mlflow registry, add manual promote/rollback script"
```

---

### Task 5: Rewire `jobs/daily.py` (registry lookup, production runs, heartbeat)

**Files:**
- Modify: `src/beat_snp500/jobs/daily.py:13-17,108-137,139-178`
- Test: `tests/test_daily.py:1-14,131-147`

**Interfaces:**
- Consumes: `tracking.Tracker` (Tasks 2–3).
- Produces: `monthly_rebalance(panel_completed, models_dir, out_dir, as_of, train_window=config.TRAIN_WINDOW_MONTHS, tracker: tracking.Tracker | None = None) -> None` — **`registry_path` parameter removed**; `run()` resolves the booster via `Tracker.current_model_artifact()` and logs a heartbeat run. Task 6 relies on `daily.py` having no `models.registry` import.

- [ ] **Step 1: Update the failing test first**

In `tests/test_daily.py`, replace the imports block (lines 1–14) with:

```python
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import pytest

from beat_snp500 import config, tracking
from beat_snp500.io_utils import read_json
from beat_snp500.jobs.daily import (
    artifact_ref, build_leaderboards, is_first_weekday, monthly_rebalance,
    resolve_artifact, update_live_track,
)
from beat_snp500.models.lgbm import train_lgbm
```

and replace `test_monthly_rebalance_writes_artifacts` (lines 131–147) with:

```python
def test_monthly_rebalance_registers_and_writes_artifacts(make_panel, tmp_path):
    models_dir, out_dir = tmp_path / "models", tmp_path / "out"
    tracker = tracking.Tracker(
        "production",
        tracking_uri=(tmp_path / "mlruns").as_uri(),
        registry_uri=f"sqlite:///{tmp_path}/reg.db")
    monthly_rebalance(make_panel(n_months=30), models_dir, out_dir,
                      pd.Timestamp("2026-07-01"), train_window=12,
                      tracker=tracker)
    ref = tracker.current_model_artifact()
    assert ref is not None and ref.endswith(".txt")
    assert resolve_artifact(ref).exists()
    for name in ("lgbm", "kmeans"):
        p = out_dir / f"holdings_{name}.json"
        if p.exists():  # a hold month legitimately writes nothing
            w = read_json(p)["weights"]
            assert 5 <= len(w) <= 10
            assert sum(w.values()) == pytest.approx(1.0)
            assert max(w.values()) <= 0.20 + 1e-9
    # the planted-signal fixture must produce at least one active model
    assert any((out_dir / f"holdings_{n}.json").exists() for n in ("lgbm", "kmeans"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_daily.py -v`
Expected: `test_monthly_rebalance_registers_and_writes_artifacts` FAILS with `TypeError` (unexpected keyword `tracker` / missing positional `registry_path`).

- [ ] **Step 3: Rewire `daily.py`**

In `src/beat_snp500/jobs/daily.py`:

(a) Replace the import `from beat_snp500.models.registry import append_entry, latest_model` with:

```python
from beat_snp500 import tracking
```

(b) Replace `monthly_rebalance` (lines 108–137) with:

```python
def monthly_rebalance(panel_completed: pd.DataFrame, models_dir, out_dir,
                      as_of, train_window: int = config.TRAIN_WINDOW_MONTHS,
                      tracker: tracking.Tracker | None = None) -> None:
    if tracker is None:
        tracker = tracking.Tracker("production", strict=False)
    labeled = panel_completed.dropna(subset=["fwd_return_1m"])
    model, val_ic = train_lgbm(labeled, train_window=train_window)
    latest = panel_completed.index.get_level_values("date").max()
    model_id = f"lgbm_{latest:%Y%m}"
    artifact = Path(models_dir) / f"{model_id}.txt"
    save_model(model, artifact)

    month = panel_completed.xs(latest, level="date")
    scores = pd.Series(model.predict(month[config.FEATURES]), index=month.index)
    selections = {
        "lgbm": lgbm_must_buys(scores),
        "kmeans": kmeans_must_buys(month),
    }
    for name, sig in selections.items():
        if not sig:
            continue  # hold: previous holdings_{name}.json stays in force
        atomic_write_json(
            {"signal_date": str(latest.date()),
             "generated_at": str(pd.Timestamp(as_of).date()),
             "weights": conviction_weights(sig)},
            Path(out_dir) / f"holdings_{name}.json")

    # registered AFTER holdings are written: a registry failure must never
    # block publishing; @current then keeps serving the prior booster until
    # the next successful rebalance
    trained_through = str(labeled.index.get_level_values("date").max().date())
    run_id = None
    with tracker.start_run(run_name=f"rebalance-{latest:%Y%m}") as run:
        tracker.log_params({"model_id": model_id,
                            "train_window_months": train_window,
                            "n_features": len(config.FEATURES),
                            "trained_through": trained_through})
        tracker.log_metrics({"val_ic": val_ic})
        run_id = run.info.run_id if run is not None else None
    tracker.register_model_version(
        artifact=artifact_ref(artifact), run_id=run_id,
        tags={"model_id": model_id, "trained_through": trained_through,
              "train_window_months": train_window, "ic_mean": val_ic,
              "created_at": str(pd.Timestamp(as_of).date())})
```

(c) In `run()`: insert `tracker = tracking.Tracker("production", strict=False)` as the first line after the `rebalance = ...` line. Replace the validation-failure branch with:

```python
    if not report.ok:
        with tracker.start_run(run_name=f"daily-{as_of.date()}"):
            tracker.log_params({"as_of": str(as_of.date()),
                                "rebalance": str(bool(rebalance))})
            tracker.log_metrics({"validation_ok": 0})
        print(f"validation failed: {report.issues}", file=sys.stderr)
        return 1
```

Replace the booster lookup (lines 160–161) with:

```python
    ref = tracker.current_model_artifact()
    booster = load_model(resolve_artifact(ref)) if ref else None
```

After the `update_live_track(...)` call and before the `if rebalance:` block, insert the heartbeat:

```python
    latest_month = panel.index.get_level_values("date").max()
    n_scored = len(panel.xs(latest_month, level="date"))
    with tracker.start_run(run_name=f"daily-{as_of.date()}"):
        tracker.log_params({"as_of": str(as_of.date()),
                            "rebalance": str(bool(rebalance))})
        tracker.log_metrics({"validation_ok": 1, "n_scored": n_scored})
```

Finally, change the `monthly_rebalance` call to:

```python
        monthly_rebalance(panel_completed, config.MODELS_DIR, config.OUTPUTS_DIR,
                          as_of, tracker=tracker)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_daily.py tests/test_tracking.py -v`
Expected: all pass (the autouse conftest fixture keeps default-constructed Trackers inside tmp_path).

- [ ] **Step 5: Full suite**

Run: `.venv/bin/pytest -q`
Expected: `114 passed`, zero warnings (`test_registry.py` still passes — untouched until Task 6).

- [ ] **Step 6: Commit**

```bash
git add src/beat_snp500/jobs/daily.py tests/test_daily.py
git commit -m "feat: daily job logs to mlflow and resolves booster via @current alias"
```

---

### Task 6: Retire the old registry

**Files:**
- Delete: `src/beat_snp500/models/registry.py`, `tests/test_registry.py`, `models/registry.json`
- Modify: `src/beat_snp500/config.py:12`, `scripts/migrate_model_names.py:62`, `scripts/migrate_registry_to_mlflow.py`

**Interfaces:**
- Consumes: Task 5 (daily.py no longer imports `models.registry`).
- Produces: `config.REGISTRY_JSON` removed; MLflow SQLite is the sole registry of record.

- [ ] **Step 1: Confirm nothing else uses the old registry**

Run: `grep -rn "REGISTRY_JSON\|models.registry\|registry.json" --include="*.py" src scripts app tests`
Expected: hits only in `config.py` (definition), `scripts/migrate_model_names.py:62`, `scripts/migrate_registry_to_mlflow.py` (docstring + `__main__`), `tests/test_migrate.py` and `tests/test_migrate_registry.py` (tmp-path fixture filenames only — harmless), and the files being deleted. Any other hit must be fixed before proceeding.

- [ ] **Step 2: Patch the two one-shot scripts**

In `scripts/migrate_model_names.py` line 62, replace `config.REGISTRY_JSON` with `config.MODELS_DIR / "registry.json"`.
In `scripts/migrate_registry_to_mlflow.py` `__main__`, replace `config.REGISTRY_JSON` with `config.MODELS_DIR / "registry.json"` (the script is one-shot history; it must not pin a live config constant).

- [ ] **Step 3: Delete the old registry**

```bash
git rm src/beat_snp500/models/registry.py tests/test_registry.py models/registry.json
```

In `src/beat_snp500/config.py`, delete the line `REGISTRY_JSON = MODELS_DIR / "registry.json"`.

- [ ] **Step 4: Full suite**

Run: `.venv/bin/pytest -q`
Expected: `111 passed` (3 registry tests gone), zero warnings.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat: retire registry.json - mlflow registry is the record of production models"
```

---

### Task 7: Instrument `run_backtest.py`

**Files:**
- Modify: `scripts/run_backtest.py`

**Interfaces:**
- Consumes: `tracking.Tracker` (Task 2); `run_report(prices, membership, factors, out_dir) -> dict[str, dict]` (existing).
- Produces: one `backtest` run per invocation with per-model CAGR/Sharpe/MaxDD metrics.

- [ ] **Step 1: Rewrite `main()`**

Replace the body of `main()` in `scripts/run_backtest.py` with:

```python
def main() -> int:
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    tracker = tracking.Tracker("backtest")
    with tracker.start_run(run_name=f"backtest-{pd.Timestamp.today():%Y%m%d}"):
        tracker.log_params({"champion": config.CHAMPION,
                            "n_features": len(config.FEATURES),
                            "features": ",".join(config.FEATURES),
                            "cost_bps_one_way": config.COST_BPS_ONE_WAY,
                            "universe_size": config.UNIVERSE_SIZE})
        metrics = run_report(prices, membership, factors, config.BACKTEST_DIR)
        for name, m in metrics.items():
            tracker.log_metrics({f"{name}_cagr": m["cagr"],
                                 f"{name}_sharpe": m["sharpe"],
                                 f"{name}_max_drawdown": m["max_drawdown"]})
    for name, m in metrics.items():
        print(f"{name:14s} CAGR {m['cagr']:7.2%}  Sharpe {m['sharpe']:5.2f}  "
              f"MaxDD {m['max_drawdown']:7.2%}")
    return 0
```

and add `from beat_snp500 import config, tracking` (replacing the bare `from beat_snp500 import config`).

- [ ] **Step 2: Verify with a real run** (several minutes; requires local `data/prices.parquet`)

Run: `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py`
Expected: same metric printout as before (kmeans ≈ 30.0% CAGR / 1.00 Sharpe; lgbm ≈ 8.95% / 0.41; spy ≈ 14.7% / 0.82), plus a new run directory under `mlruns/`. Verify:

```bash
PYTHONPATH=src .venv/bin/python -c "
import mlflow
from beat_snp500 import tracking
mlflow.set_tracking_uri(tracking.Tracker('backtest').tracking_uri)
runs = mlflow.search_runs(experiment_names=['backtest'])
print(runs[['run_id', 'metrics.kmeans_sharpe']].to_string())"
```

Expected: one row, `metrics.kmeans_sharpe` ≈ 1.00.

- [ ] **Step 3: Commit (code + the tracked run)**

```bash
git add scripts/run_backtest.py mlruns
git commit -m "feat: track backtest runs in mlflow"
```

---

### Task 8: Instrument `tune_kmeans.py` (parent/child runs + persist holdout)

**Files:**
- Modify: `scripts/tune_kmeans.py`

**Interfaces:**
- Consumes: `tracking.Tracker` (Task 2); existing `REPORT = config.OUTPUTS_DIR / "tuning" / "kmeans_tuning.json"`.
- Produces: dev mode logs one parent run + one nested child per grid config and writes `mlflow_parent_run_id` into the REPORT JSON; `--holdout` attaches a nested `holdout` run to that parent and (fixing the round-2 follow-up) persists holdout results into the REPORT JSON under a `"holdout"` key.

- [ ] **Step 1: Instrument dev mode**

In `main()`'s non-holdout branch, replace from `rf = rf_annual(dev_f)` through the `atomic_write_json(...)` call with:

```python
        rf = rf_annual(dev_f)
        tracker = tracking.Tracker("tuning")
        rows = []
        with tracker.start_run(
                run_name=f"kmeans-grid-{pd.Timestamp.today():%Y%m%d}") as parent:
            for i, cfg in enumerate(GRID):
                rows.append({**cfg, "dev_sharpe": sharpe_for(cfg, dev_panel,
                                                             dev_close, rf)})
                print(f"[{i + 1}/{len(GRID)}]", rows[-1])
                with tracker.start_run(run_name=f"cfg-{i:02d}", nested=True):
                    tracker.log_params(cfg)
                    tracker.log_metrics({"dev_sharpe": rows[-1]["dev_sharpe"]})
            current_dev = sharpe_for(CURRENT, dev_panel, dev_close, rf)
            tracker.log_metrics({"current_dev_sharpe": current_dev})
            parent_id = parent.info.run_id if parent is not None else None
        rows.sort(key=lambda r: (r["dev_sharpe"] != r["dev_sharpe"],
                                 -r["dev_sharpe"]))  # NaNs last
        atomic_write_json({"current": CURRENT, "grid": rows,
                           "mlflow_parent_run_id": parent_id}, REPORT)
```

(keep the following `print` lines, but change `print("current config dev Sharpe:", ...)` to reuse `current_dev` instead of recomputing).

Add `from beat_snp500 import config, tracking` (replacing `from beat_snp500 import config`).

- [ ] **Step 2: Instrument holdout mode + persist results**

Replace the holdout branch (from `report = read_json(REPORT)` to `return 0`) with:

```python
    report = read_json(REPORT)
    winner = {k: report["grid"][0][k]
              for k in ("k", "mom_mode", "select_rule", "threshold")}
    rf = rf_annual(h_f)
    w = sharpe_for(winner, h_panel, h_close, rf)
    c = sharpe_for(CURRENT, h_panel, h_close, rf)
    verdict = "ADOPT" if w >= c else "KEEP"
    tracker = tracking.Tracker("tuning")
    parent_id = report.get("mlflow_parent_run_id")
    with tracker.start_run(run_id=parent_id) if parent_id else \
            tracker.start_run(run_name="kmeans-holdout"):
        with tracker.start_run(run_name="holdout", nested=True):
            tracker.log_params({f"winner_{k}": v for k, v in winner.items()})
            tracker.log_metrics({"winner_holdout_sharpe": w,
                                 "current_holdout_sharpe": c})
            tracker.set_tags({"verdict": verdict})
    report["holdout"] = {"winner": winner, "winner_sharpe": w,
                         "current_sharpe": c, "verdict": verdict}
    atomic_write_json(report, REPORT)
    print("holdout Sharpe — winner:", w, winner)
    print("holdout Sharpe — current:", c, CURRENT)
    print("verdict:", f"{verdict} {'winner' if verdict == 'ADOPT' else 'current'}")
    return 0
```

- [ ] **Step 3: Verify the script still parses and wires up**

Run: `PYTHONPATH=src .venv/bin/python -c "import ast; ast.parse(open('scripts/tune_kmeans.py').read()); print('OK')"` then `PYTHONPATH=src .venv/bin/python scripts/tune_kmeans.py --help`
Expected: `OK`, then the argparse help text. (The full grid takes several minutes and reruns the round-2 experiment; it is exercised for real in the next tuning round, and its MLflow plumbing is the same `Tracker` covered by tests.)

- [ ] **Step 4: Commit**

```bash
git add scripts/tune_kmeans.py
git commit -m "feat: track kmeans tuning grid + holdout in mlflow, persist holdout to report"
```

---

### Task 9: Instrument `evaluate_features.py` (+ persist holdout, new report schema)

**Files:**
- Modify: `scripts/evaluate_features.py`

**Interfaces:**
- Consumes: `tracking.Tracker` (Task 2); existing `REPORT = config.OUTPUTS_DIR / "tuning" / "feature_eval.json"`.
- Produces: report schema becomes `{"dev": {name: stats}, "holdout": {...}}` (only humans and the config.py comment reference this file); one `feature-gate` run per candidate set per phase, ADOPT/KEEP as a tag.

- [ ] **Step 1: Rewrite `main()`'s two branches**

Add `from beat_snp500 import config, tracking` (replacing `from beat_snp500 import config`). Replace the holdout branch with:

```python
    if args.holdout:
        tracker = tracking.Tracker("feature-gate")
        stats = {}
        for name in ("baseline", args.holdout):
            stats[name] = ic_stats(panel, CANDIDATE_SETS[name], dev=False)
            print(name, stats[name])
        verdict = ("ADOPT" if stats[args.holdout]["mean_ic"]
                   >= stats["baseline"]["mean_ic"] else "KEEP")
        with tracker.start_run(run_name=f"holdout-{args.holdout}"):
            tracker.log_params({"candidate": args.holdout, "phase": "holdout",
                                "extra_cols": ",".join(CANDIDATE_SETS[args.holdout])})
            tracker.log_metrics({
                "candidate_mean_ic": stats[args.holdout]["mean_ic"],
                "candidate_ic_ir": stats[args.holdout]["ic_ir"],
                "baseline_mean_ic": stats["baseline"]["mean_ic"],
                "baseline_ic_ir": stats["baseline"]["ic_ir"]})
            tracker.set_tags({"verdict": verdict})
        report = read_json(REPORT) if REPORT.exists() else {}
        report["holdout"] = {"candidate": args.holdout, **stats,
                             "verdict": verdict}
        atomic_write_json(report, REPORT)
        print(f"verdict: {verdict} (candidate vs baseline on holdout mean IC)")
        return 0
```

and the dev branch with:

```python
    tracker = tracking.Tracker("feature-gate")
    rows = {}
    for name, extra in CANDIDATE_SETS.items():
        rows[name] = ic_stats(panel, extra, dev=True)
        with tracker.start_run(run_name=f"dev-{name}"):
            tracker.log_params({"candidate": name, "phase": "dev",
                                "extra_cols": ",".join(extra) or "-"})
            tracker.log_metrics(rows[name])
    for name, r in rows.items():
        print(f"{name:18s} mean IC {r['mean_ic']:+.4f}  IR {r['ic_ir']:+.2f}")
    atomic_write_json({"dev": rows}, REPORT)
    print("\nNext: pick the best non-baseline set, run --holdout <name> ONCE.")
    return 0
```

Add `read_json` to the `beat_snp500.io_utils` import line.

- [ ] **Step 2: Verify parse + help**

Run: `PYTHONPATH=src .venv/bin/python scripts/evaluate_features.py --help`
Expected: argparse help, no import errors.

- [ ] **Step 3: Update the config.py comment that references the old schema**

In `src/beat_snp500/config.py`, the `mom_vol_scaled` comment says `see data/outputs/tuning/feature_eval.json`; append ` (schema: {"dev": ..., "holdout": ...} since 2026-07)` to that comment line.

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_features.py src/beat_snp500/config.py
git commit -m "feat: track feature-gate dev/holdout runs in mlflow, persist holdout verdict"
```

---

### Task 10: CI workflow + .gitignore

**Files:**
- Modify: `.github/workflows/daily.yml:45`, `.gitignore`
- Test: `tests/test_workflows.py:15-25`

**Interfaces:**
- Consumes: nothing new; Produces: daily CI commits `mlruns/` alongside `data/` and `models/`.

- [ ] **Step 1: Write the failing test**

In `tests/test_workflows.py::test_daily_workflow_schedule_and_permissions`, after the `assert "run_daily.py" in steps_str ...` line, add:

```python
    assert "git add data models mlruns" in steps_str
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_workflows.py -v`
Expected: FAIL on the new assertion.

- [ ] **Step 3: Update workflow + gitignore**

In `.github/workflows/daily.yml` change `git add data models` to `git add data models mlruns`.
In `.gitignore` append:

```
mlruns/.trash/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_workflows.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/daily.yml .gitignore tests/test_workflows.py
git commit -m "feat: commit mlruns from daily CI, ignore mlflow trash"
```

---

### Task 11: Junior instruction doc, chapter 1

**Files:**
- Create: `docs/learning/01-experiment-tracking-mlflow.md`
- Modify: `README.md` (add a "Learning the MLOps stack" section linking the chapter)

**Interfaces:** none (documentation).

- [ ] **Step 1: Write the chapter**

Create `docs/learning/01-experiment-tracking-mlflow.md` covering, in this order (write full prose for each — this outline is the required content, not the finished text; keep the repo's plain, honest voice):

1. **The problem before the tool.** This repo's real history: results lived in `improvement/improve_v1.md`/`improve_v2.md` prose, tuning output in ad-hoc JSONs, and the round-2 K-means holdout results were only ever printed to a terminal — never persisted anywhere. Experiment tracking exists to make "what did we run, with what config, and what did it score" a queryable record instead of an anecdote.
2. **Concepts, mapped to this repo.** Run (one execution of `run_backtest.py`), experiment (the four folders: `backtest`, `tuning`, `feature-gate`, `production`), params vs metrics vs tags (config in, numbers out, decisions annotated — e.g. the feature gate's ADOPT/KEEP verdict tag), nested runs (the 72-config tuning grid under one parent), registered model / version / alias (`lgbm` versions; `@current` is what the daily job serves).
3. **Why our wiring is unusual — and when you'd do it differently.** We commit `mlruns/` (text file store) and a SQLite registry to git: zero servers, fully reproducible from a clone, right-sized for one daily CI job plus one laptop. At company scale you'd run a tracking server + object-store artifacts + Postgres instead — name the trade-offs (concurrent writers, access control, big artifacts). Include the two-writer rule verbatim: *CI owns the registry; humans pull before promoting.*
4. **Failure policy.** Why the daily job uses `Tracker(strict=False)`: telemetry must never block publishing picks; offline experiments fail loud. Point at `test_non_strict_warns_instead_of_raising`.
5. **How to use it.** Exact commands: `git pull`, `MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri "file://$PWD/mlruns"` (the env flag is needed because MLflow 3.x gates the file store; our `tracking.py` sets it automatically for pipeline code, but the standalone UI command needs it explicitly), then open http://127.0.0.1:5000; how to compare runs in the `feature-gate` experiment; how to see which model `@current` points at with the `Tracker('production').current_model_artifact()` one-liner.
6. **Exercises.** (a) Find the tuning config that won dev but failed holdout (round-2 story — the data is in `data/outputs/tuning/kmeans_tuning.json` and, for new rounds, in the `tuning` experiment). (b) After the next monthly rebalance lands, find its `rebalance-YYYYMM` run and check `val_ic`. (c) Explain why `kmeans` has no registered model even though it is the champion.

- [ ] **Step 2: Link from README**

Add to `README.md`, after the existing project-description sections, a short section:

```markdown
## Learning the MLOps stack

Guided chapters for junior data scientists, written as each piece was built:

- [01 — Experiment tracking with MLflow](docs/learning/01-experiment-tracking-mlflow.md)
```

- [ ] **Step 3: Commit**

```bash
git add docs/learning/01-experiment-tracking-mlflow.md README.md
git commit -m "docs: junior learning chapter 1 - experiment tracking with mlflow"
```

---

### Task 12: End-to-end verification + push

**Files:** none new (verification + generated `mlruns/` data).

- [ ] **Step 1: Full suite, zero warnings**

Run: `.venv/bin/pytest -q`
Expected: `115 passed` (109 baseline + 4 tracking + 1 migrate-registry + 1 workflow assertion inside existing test, − 3 removed registry tests + 1 renamed daily test ≈ 112–115; the exact count printed here is the new baseline — record it), **0 warnings**.

- [ ] **Step 2: Real daily run (non-rebalance day)**

Run: `PYTHONPATH=src .venv/bin/python scripts/run_daily.py --as-of 2026-07-18`
Expected: exit 0 (2026-07-18 is not the first weekday of July → no rebalance, no registration). Then verify the heartbeat landed:

```bash
PYTHONPATH=src .venv/bin/python -c "
import mlflow
from beat_snp500 import tracking
mlflow.set_tracking_uri(tracking.Tracker('production').tracking_uri)
runs = mlflow.search_runs(experiment_names=['production'])
print(runs[['run_id', 'tags.mlflow.runName', 'metrics.validation_ok']].to_string())"
```

Expected: a `daily-2026-07-18` row with `validation_ok = 1.0`.

- [ ] **Step 3: Optional manual check — the UI**

Run: `MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri "file://$PWD/mlruns"` and open http://127.0.0.1:5000 — confirm the `backtest` and `production` experiments render. Ctrl-C when done.

- [ ] **Step 4: Commit generated data + push**

```bash
git add mlruns data models
git commit -m "data: mlflow tracking data from verification runs"
git push
```

- [ ] **Step 5: Watch the first post-merge daily Actions run**

After tonight's `daily` workflow (22:30 UTC), confirm on GitHub: the run is green, its commit includes `mlruns/` changes, and no `mlflow` import errors appear in the log (this is the first CI execution with `mlflow-skinny` — the Task 1 spike's fallback applies if it fails).

---

## Self-review notes

- **Spec coverage:** §1 → Tasks 2–3 (stores) + Task 4 (manual promotion script); §2 table → Tasks 5, 7, 8, 9; §2 error policy → Task 2 (guard) + Task 5 (strict=False); §3 → Tasks 3–6; §4 → Tasks 1, 10; §5 → Task 4 (promote script embeds the pull-first rule) + doc rule in Task 11; §6 → Task 11; §7 → Tasks 2–5; §8 risks → Tasks 1, 12.
- **Type consistency:** `Tracker` signature identical across Tasks 2/3/5/7/8/9; `register_model_version`/`current_model_artifact` names match Tasks 3, 4, 5, 12.
- The exact final test count in Task 12 depends on whether Task 5 renames or adds tests — record the observed number as the new baseline rather than forcing a predicted one.
