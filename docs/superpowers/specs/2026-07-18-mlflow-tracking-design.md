# MLflow Experiment Tracking + Model Registry — Design

**Status:** Approved by Roman 2026-07-18.
**Date:** 2026-07-18
**Sub-project:** 1 of 4 in the round-3 MLOps scale-up (MLflow → Evidently → orchestrator → feature-expansion program).

## Context and goal

beat-snp500 is live: a daily GitHub Actions pipeline retrains monthly, publishes picks, and commits data/model artifacts to git; experiments (backtests, tuning, feature gates) run on Roman's laptop and are recorded in ad-hoc JSONs and markdown ledgers. Round 3's goal is **best engineering for real use** (maximum model precision for a real financial assistant), not portfolio showcase.

This sub-project adds MLflow so that every experiment and production run is tracked, comparable, and linked to the model it produced — the rails the later feature-expansion program will run on. Decision already made: **Approach A — full MLflow**, i.e. MLflow's Model Registry becomes the registry of record, replacing `models/registry.json`. Infrastructure constraint: **laptop + GitHub only** — no always-on server; everything must be reproducible from a clone.

The design keeps A workable by exploiting the fact that MLflow allows **separate backends for tracking and registry**: tracking stays on the merge-friendly text file store, and only the registry lives in SQLite — which is written monthly, not daily.

## §1 Stores and layout

- **Tracking:** MLflow file store at `mlruns/` in the repo root, committed to git. Runs are UUID-keyed directories of small text files — laptop and CI writers can never collide; merges are trivial.
- **Registry:** SQLite at `models/mlflow_registry.db`, committed to git, selected via `mlflow.set_registry_uri()`. Written only when a model is registered (monthly retrain in CI) or promoted (manual script, pull-first rule).
- **Helper module:** new `src/beat_snp500/tracking.py` pins both URIs (repo-relative, derived from `config.ROOT`) and provides a `start_run(experiment, ...)` wrapper so entry points share one boilerplate-free interface.
- **Experiments:** `backtest`, `tuning`, `feature-gate`, `production`.

## §2 What gets logged

| Entry point | Experiment | Logs |
|---|---|---|
| `scripts/run_backtest.py` | `backtest` | config params (champion, features, cost bps, universe); CAGR / Sharpe / MaxDD per model + SPY benchmark; small report artifacts |
| `scripts/tune_kmeans.py` | `tuning` | parent run + 72 nested child runs (one per grid config) with dev Sharpe; `--holdout` pass logs into the same parent — fixes the round-2 follow-up that holdout results were never persisted |
| `scripts/evaluate_features.py` | `feature-gate` | one run per candidate set: dev/holdout mean IC, IC IR, n_months; ADOPT/KEEP decision as a tag |
| `jobs/daily.py` — `monthly_rebalance` | `production` | train window, feature list, validation IC; registers the new booster (see §3) |
| `jobs/daily.py` — ordinary daily run | `production` | lightweight heartbeat: as-of date, names scored, rebalance yes/no |

**Error policy:** in the daily job, tracking failures **warn and continue** — telemetry must never block publishing picks. Offline scripts fail loud.

## §3 Registry semantics

- Only **lgbm** produces a serialized artifact, so the registry manages `lgbm` model versions. The booster file stays in `models/` (referenced by repo-relative path, not duplicated into `mlruns/`); the current production version is resolved via a **`@current` alias** instead of `registry.latest_model()`. Concretely: `jobs/daily.py` (the only `latest_model()` consumer) switches to an `MlflowClient` alias lookup wrapped in `tracking.py`, so job code never talks to MLflow APIs directly.
- A **migration script** imports historical `registry.json` entries as back-dated versions (original timestamps/ICs preserved as tags), after which `models/registry.py` and `registry.json` are retired — git history preserves them.
- **kmeans** has no artifact — it is refit monthly from config — so champion/challenger designation **stays in `config.CHAMPION`**. Registering a config-only "model" would be ceremony without value. This asymmetry is a deliberate teaching point for the junior doc.

## §4 Dependencies and CI

- Core deps gain `mlflow-skinny` (logging client, no UI/server) plus `sqlalchemy`/`alembic` for the SQLite registry backend. Full `mlflow` goes in `[dev]` for the local UI.
- `.github/workflows/daily.yml` changes one line: `git add data models mlruns`.
- **First implementation task is a spike** verifying that skinny + SQLite registry works and measuring CI install cost — this combination is the design's main assumption.

## §5 Two-writer discipline

- File-store tracking cannot conflict (UUID run dirs).
- The SQLite registry has exactly two writers: **CI** (monthly, automated) and **Roman** (rare manual promotions, only after `git pull`). Rule documented in the junior doc: *CI owns the registry; humans pull before promoting.*
- The daily CI `git push` racing Roman's pushes is a pre-existing property of the repo's git-as-datastore pattern; no new mechanism.

## §6 Junior instruction doc

New `docs/learning/01-experiment-tracking-mlflow.md`, written as part of this sub-project:

1. The problem before the tool — this repo's real history (markdown ledgers, the round-2 holdout results that never got persisted) as the motivating example.
2. Core concepts: run, experiment, artifact, registered model, version, alias.
3. How this repo's git-native wiring differs from a company-scale setup (tracking server + artifact store + database) and why we chose it.
4. How to use it: `mlflow ui` after a pull, comparing feature-gate runs.
5. Exercises, e.g. "find the tuning config that won dev but failed holdout."

Chapters 02 (Evidently), 03 (orchestrator), 04 (feature program) follow with later sub-projects.

## §7 Testing

File-store MLflow needs no server, so tests stay hermetic against `tmp_path` stores:

- unit tests for `tracking.py` (URIs point into the repo; run context manager works offline);
- register-and-resolve round-trip for `monthly_rebalance` against a temp SQLite registry;
- a raising MLflow client does **not** fail the daily run (error-policy test);
- migration-script tests on `registry.json` fixtures.

Bar: current 109-test suite stays green, zero warnings.

## §8 Risks

1. **skinny + SQLite footprint** — spike first (see §4).
2. **MLflow ↔ `pandas<3` pin compatibility** — pin the MLflow version explicitly after the spike.
3. **`mlruns/` growth** — ~250 small text runs/year; negligible. Model binaries never enter `mlruns/`.

## Out of scope (later sub-projects)

- Evidently data/prediction drift monitoring (sub-project 2) — will consume the `production` experiment's tracked reference data.
- Orchestrator selection and migration (sub-project 3).
- New features / conditional features / new data sources (sub-project 4).
- AutoGluon: rejected for production; at most a one-off offline upper-bound probe, run through the existing champion/challenger gate.
