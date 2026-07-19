# 01 — Experiment Tracking with MLflow

This is chapter 1 of a small series written for a junior data scientist
joining this repo, one chapter per piece of the MLOps stack as it actually
gets built. This chapter covers experiment tracking and the model registry
(MLflow 3.14.0, wired up as `src/beat_snp500/tracking.py`). Later chapters —
drift monitoring, an orchestrator, feature expansion — describe what's
actually there once each lands, not a hypothetical.

## 1. The problem before the tool

Before this sub-project, "what did we run, with what config, and what did it
score" lived in three disconnected places, none of them queryable:

- **Prose ledgers.** `improvement/improve_v1.md` and `improvement/improve_v2.md`
  are markdown write-ups where results get typed up by hand after the fact —
  a table pasted in, a paragraph of interpretation around it. Good for a
  human reading top to bottom once; useless for "show me every run where
  `k=3`."
- **Ad-hoc JSONs.** `scripts/tune_kmeans.py` and `scripts/evaluate_features.py`
  wrote (and still write) their own report files —
  `data/outputs/tuning/kmeans_tuning.json`, `data/outputs/tuning/feature_eval.json`.
  Each script invented its own schema, so comparing across scripts, or across
  runs of the same script from different days, meant opening files by hand.
- **The terminal, and nothing else.** The clearest example is round 2's
  K-means holdout check. `tune_kmeans.py --holdout` printed:

  ```
  holdout Sharpe — winner: 1.0039488457250687 {'k': 3, 'mom_mode': 'vol_scaled', 'select_rule': 'risk_adj', 'threshold': 0.5}
  holdout Sharpe — current: 1.17908295728248 {'k': 4, 'mom_mode': 'mean_3_6_12', 'select_rule': 'mean', 'threshold': 0.0}
  verdict: KEEP current
  ```

  That output was never written to a file by the script itself. It survives
  today only because someone copy-pasted it, verbatim, into
  `improvement/improve_v2.md` §3. If nobody had pasted it, the fact that the
  grid's "winner" actually lost on holdout — the single most important
  result of that round's tuning work — would be gone. Nothing links that
  paragraph back to the 72-row grid that produced it, and nothing would have
  stopped someone from re-running the grid, getting a different top row by
  chance, and never noticing the earlier holdout check disagreed.

Experiment tracking exists to close that gap: every run — its config, its
numbers, and the decision it fed — becomes a structured record you can query
later, instead of an anecdote that only exists if someone happened to write
it down.

## 2. Concepts, mapped to this repo

MLflow has a handful of core concepts. Here's what each one is, concretely,
in this codebase.

- **Run** — one execution that gets logged. `scripts/run_backtest.py` opens
  exactly one run per invocation: `tracker.start_run(run_name=f"backtest-{date:%Y%m%d}")`.
  A run carries whatever params, metrics, and tags you log inside its
  `with` block.
- **Experiment** — the bucket a run belongs to. This repo uses four:
  `backtest`, `tuning`, `feature-gate`, `production` — one per pipeline
  stage. Each becomes its own numbered folder under `mlruns/` the first time
  `mlflow.set_experiment(name)` runs for that name (`Tracker.start_run` calls
  it for you). Right now only `backtest` has a folder committed
  (`mlruns/832580556589764010/`), because it's the only stage that's been
  run for real since the switchover — `tuning`, `feature-gate`, and
  `production` will each grow their own folder the next time their script
  runs.
- **Params vs. metrics vs. tags** — three different kinds of thing you log,
  and this repo uses all three deliberately:
  - *params* = config going in — `run_backtest.py` logs `champion`,
    `n_features`, `features`, `cost_bps_one_way`, `universe_size` via
    `tracker.log_params({...})`.
  - *metrics* = numbers coming out — CAGR, Sharpe, and max drawdown per
    model, via `tracker.log_metrics({...})`. `log_metrics` also silently
    drops any `NaN` value before it reaches MLflow (see
    `tests/test_tracking.py::test_logs_roundtrip_to_file_store`), because a
    `NaN` in a metric field is noise, not a number.
  - *tags* = a decision annotated onto a run — `evaluate_features.py`'s
    holdout run and `tune_kmeans.py`'s holdout run both call
    `tracker.set_tags({"verdict": "ADOPT"})` or `{"verdict": "KEEP"}`, so the
    call this run fed into is attached to the run itself, not floating in a
    separate document.
- **Nested runs** — a run inside another run, for grid searches.
  `tune_kmeans.py`'s dev pass opens one parent run
  (`kmeans-grid-YYYYMMDD`), then 72 nested child runs — one per grid config,
  `nested=True` — each logging that config's params and its dev Sharpe. The
  later `--holdout` pass reopens that exact parent by the run ID it saved in
  `kmeans_tuning.json` (`report["mlflow_parent_run_id"]`) and adds one more
  nested `holdout` run underneath it. So the whole 72-config grid and the
  single holdout check that judges it live under one parent — exactly the
  link that was missing in §1's motivating example.
- **Registered model / version / alias** — `lgbm` is this repo's one
  registered model (`tracking.REGISTERED_MODEL = "lgbm"`). Every monthly
  retrain (`jobs/daily.py::monthly_rebalance`) creates a new version pointing
  at that month's booster file. `@current` (`tracking.CURRENT_ALIAS`) is an
  alias — a name that always points at exactly one version — and it's what
  `jobs/daily.py` resolves via `Tracker('production').current_model_artifact()`
  to decide which booster to load and score with today.

## 3. Why our wiring is unusual — and when you'd do it differently

Tracking runs live in `mlruns/` — MLflow's plain file store, one directory
per run, small text files for params/metrics/tags. The registry lives in
`models/mlflow_registry.db` — SQLite, a separate URI from tracking. **Both
are committed to git.** There is no MLflow server anywhere; the infra is a
laptop and GitHub Actions.

That's a constraint working backwards to a design, not a default. It's
right-sized for what this repo actually has: one daily CI job plus one
laptop, at most two writers, never truly concurrent — CI runs on a
schedule, Roman runs things by hand. `mlruns/`'s run directories are
UUID-keyed (`d0631881a32a4f19b6ca2a798483b000` for the one `backtest` run
committed so far), so writers can never collide there even if they do
overlap — merges are just new files appearing. The whole thing is
reproducible from a fresh `git clone`: no server to stand up, no
credentials to distribute.

At company scale you would not do this — you'd run a tracking server
(MLflow's own, or Databricks/SageMaker's managed version) backed by
object-store artifacts (S3/GCS) and a real database (Postgres) for the
registry. Three concrete reasons why:

- **Concurrent writers.** SQLite locks the whole database file per write.
  Fine for two writers who are rarely both touching it at once; it would
  start throwing "database is locked" errors under a real team pushing
  registrations from multiple pipelines at the same time. Postgres handles
  that properly.
- **Access control.** A tracking server can put real auth in front of "who
  is allowed to promote a model to production." Here, that control is
  entirely social/procedural: git branch protection, PR review, and the rule
  below — not something the tool enforces.
- **Big artifacts.** This repo's LightGBM boosters are small text files, so
  putting the registry's `source` pointer at a path in `models/` and letting
  git track it is fine. It would not be fine for a large neural net
  checkpoint — object storage exists precisely so multi-gigabyte artifacts
  don't go through git's diff/merge machinery.

Because there are exactly two writers to the registry — CI (monthly,
automated) and Roman (rare manual promotions) — the design spec
(`docs/superpowers/specs/2026-07-18-mlflow-tracking-design.md` §5) states
the rule that keeps them from clobbering each other in one line:

> **CI owns the registry; humans pull before promoting.**

`scripts/promote_model.py`'s docstring carries the same rule in its own
words: *"Two-writer rule (spec §5): CI owns the registry — run `git pull`
before this, and push promptly after."* Concretely:
`.github/workflows/daily.yml`'s last step commits `data models mlruns` after
every weekday pipeline run — the daily heartbeat writes a new `mlruns/` run
each day — while `mlflow_registry.db`'s contents change only on monthly
retrains, since `register_model_version` is called only from
`monthly_rebalance`. If you run
`scripts/promote_model.py` without pulling first, you are promoting against
a stale local copy of the registry and your next push can silently
overwrite whatever CI wrote in the meantime. `git pull` first, always.

## 4. Failure policy

`Tracker` wraps every MLflow call in a guard (`_guarded` in
`src/beat_snp500/tracking.py`): if the call raises, a **strict** tracker
re-raises; a **non-strict** tracker calls `warnings.warn(...)` and returns
`None` instead.

`jobs/daily.py` always constructs `Tracker("production", strict=False)` —
both for the ordinary daily heartbeat run and for
`monthly_rebalance`. The reasoning is a priority call: telemetry must never
block publishing picks. If the MLflow store is unreachable or corrupted,
losing that day's tracking record is a real but recoverable loss; failing
the whole daily job over it — and not writing that day's holdings and
leaderboards — would be worse. Notice too that `monthly_rebalance` registers
the new booster *after* holdings are already written to disk, for the same
reason: a registry failure must never stop the picks from publishing, and if
registration does fail, `@current` just keeps pointing at last month's
booster until the next successful rebalance.

Offline experiment scripts — `run_backtest.py`, `tune_kmeans.py`,
`evaluate_features.py` — use `Tracker`'s default, `strict=True`. There's no
"publish now" deadline riding on them, so a broken store should stop the
script and surface the error immediately, rather than silently losing a
result the way §1's terminal-only holdout output nearly was.

The behavior is pinned down by
`tests/test_tracking.py::test_non_strict_warns_instead_of_raising`: it
monkeypatches in a broken `mlflow` module, builds a `Tracker(strict=False)`,
and asserts the run's context manager yields `None` (rather than raising)
while emitting a `UserWarning` matching `"mlflow tracking skipped"`.
`test_strict_raises` is the mirror case — same broken module, default
`strict=True`, and the original exception propagates.

## 5. How to use it

Everything below assumes you've cloned the repo and set up the venv per the
README's Quick start.

**Browse runs in the UI.** Pull first (both `mlruns/` and the registry live
in git, so a stale checkout means a stale UI), then start the standalone
MLflow UI pointed at the repo's file store:

```bash
git pull
MLFLOW_ALLOW_FILE_STORE=true .venv/bin/mlflow ui --backend-store-uri "file://$PWD/mlruns"
```

Open http://127.0.0.1:5000. The `MLFLOW_ALLOW_FILE_STORE=true` env var is
required here: MLflow 3.x gates the file store behind that flag because it's
maintenance-mode upstream, and this repo's git-native design depends on it.
Pipeline code doesn't need you to set it — `tracking.py` does
`os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")` at import time —
but the `mlflow ui` command is a separate process that never imports
`tracking.py`, so on the command line you set it yourself.

**Compare runs in `feature-gate`.** In the left rail, select the
`feature-gate` experiment. You'll see `dev-baseline`, `dev-+mom_vol_scaled`,
etc. from the dev pass, and `holdout-<name>` runs from the one-shot holdout
checks. Tick two or more runs' checkboxes and click **Compare** to see their
`mean_ic` / `ic_ir` metrics side by side; open a `holdout-*` run's page and
check its **Tags** panel for `verdict` (`ADOPT` or `KEEP`) to see the
decision that run produced.

**Check what `@current` points at.** This is a one-liner, no UI needed:

```bash
PYTHONPATH=src .venv/bin/python -c "from beat_snp500 import tracking; print(tracking.Tracker('production').current_model_artifact())"
```

That resolves the `lgbm` registered model's `@current` alias and prints the
repo-relative artifact path it's pointing at (e.g. `models/champion_202606.txt`
as of this writing) — the exact file `jobs/daily.py` loads to score today's
picks.

## 6. Exercises

**(a) Find the tuning config that won dev but failed holdout.** This is
round 2's story, and the pre-MLflow record of it is in
`data/outputs/tuning/kmeans_tuning.json` (that file predates this
sub-project — its `"holdout"` and `"mlflow_parent_run_id"` fields are still
`null`; it's the last tuning run this repo did the old way). Open it: the
top row of `"grid"` is `{k: 3, mom_mode: vol_scaled, select_rule: risk_adj,
threshold: 0.5}` with `dev_sharpe` 0.8754 — the best dev-period Sharpe of
all 72 configs, clearly ahead of the then-current config's 0.7463. But per
`improvement/improve_v2.md` §3, that same config's *holdout* Sharpe was
1.00, against 1.18 for the config already in production — worse, not
better, on data the search never saw. Verdict: **KEEP**. Any *new* tuning
round you run will land in the `tuning` MLflow experiment instead of a bare
JSON, with the parent/child/holdout run structure from §2 — go find it
there once you've run one.

**(b) After the next monthly rebalance lands, check its `val_ic`.** Open
the MLflow UI (§5), select the `production` experiment, and find the run
named `rebalance-YYYYMM` for the month that just closed. Its `val_ic` metric
is the validation-period Spearman IC `train_lgbm` produced for that month's
retrain — the same number `monthly_rebalance` logs right before registering
the new booster version.

**(c) Why does `kmeans` have no registered model, even though it's the
champion (`config.CHAMPION == "kmeans"`)?** Because the registry manages
*versions of a serialized artifact*, and `kmeans` doesn't produce one. It's
refit from scratch every month straight out of `config.py`
(`K_CLUSTERS`, `KMEANS_MOM_MODE`, `KMEANS_SELECT_RULE`,
`MUST_BUY_Z_KMEANS`) — there's no booster file, no `source` path, nothing
for a model version to point at. `lgbm`, by contrast, trains into an
actual file (`models/lgbm_YYYYMM.txt`) each month, so it's the one side of
the champion/challenger pair the registry has anything to version. Champion
status itself lives in the plain string `config.CHAMPION`, not in the
registry at all — registering a config-only "model" would be ceremony with
nothing behind it.
