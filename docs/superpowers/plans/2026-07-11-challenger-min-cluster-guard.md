# Challenger Min-Cluster-Size Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the challenger (K-Means momentum cluster) so it never concentrates the whole portfolio into fewer than 10 names, regenerate the backtest artifacts, and land the investigation write-up — per the spec in `improvement/improve_v1.md`.

**Architecture:** `kmeans_top10()` gains a two-line guard: if the momentum cluster has fewer than `n_picks` members, return `[]` (mirroring the guard `champion_picks` already has). The backtest engine (`src/beat_snp500/backtest/engine.py`) only rebalances on dates present in the picks dict, so a skipped month automatically holds the prior month's drifted positions — no engine change needed. Two synthetic-data tests downstream need larger universes so the guard doesn't (correctly) skip their months.

**Tech Stack:** Python 3.13, pandas 2.x, scikit-learn (KMeans), pytest, Streamlit dashboard.

## Global Constraints

- **pandas is pinned `<3.0`** — the codebase targets pandas 2.x semantics. Do not bump it.
- **Running scripts requires `PYTHONPATH=src`** — this sandbox re-stamps macOS `UF_HIDDEN` on the venv's editable-install `.pth`, and Python 3.13 skips hidden `.pth` files. Always run scripts as `PYTHONPATH=src .venv/bin/python scripts/<name>.py`. pytest is immune (`pythonpath = ["src"]` in `pyproject.toml`) — run it as `.venv/bin/python -m pytest`.
- **Commits must be authored solely by Roman Nguyen.** NEVER add `Co-Authored-By`, `Generated with`, or any other trailer. This is a portfolio repo.
- **Never commit `data/prices.parquet`** (~96 MB price cache). It is gitignored; leave it that way.
- **Config values:** `config.N_PICKS = 10`, `config.K_CLUSTERS = 4`, `config.SEED` fixes all randomness — backtest artifacts are deterministic and reproducible.

## Current State (read before executing)

A validated prototype of Tasks 2–3 and the copy change in Task 4 **already exists uncommitted in the working tree** (suite green, backtest already rerun). Nothing is committed yet.

- If executing **on the current working tree**: for each code step, confirm the tree already matches the code shown (it should, byte-for-byte), run the verification commands, and make the commits. "Verify the test fails" steps will NOT fail here — that is expected; they describe a clean checkout of `main`.
- If executing **from a clean checkout of `main`**: follow every step literally.
- Task 4's Methodology-tab bullet and Task 1's cleanup are **new work** not yet in the tree.

## Out of Scope (explicitly deferred by the spec)

- Richer features for the champion (e.g. company fundamentals) — "a larger follow-up project rather than a quick fix."
- Acting on the within-cluster momentum finding (IC ≈ 0.037 conditional vs ≈ 0.012 unconditional) — "an interesting thread for future improvement that hasn't been acted on yet."

---

### Task 1: Delete accidental Finder duplicate files

The working tree contains 19 untracked files named `<name> 2.py` — macOS Finder copy artifacts. Every one is byte-identical to its original (verified 2026-07-11). They are untracked, so deletion needs no commit; this task just keeps them out of later `git add` calls.

**Files:**
- Delete: `src/beat_snp500/backtest/__init__ 2.py`, `src/beat_snp500/backtest/bootstrap 2.py`, `src/beat_snp500/backtest/engine 2.py`
- Delete: `src/beat_snp500/data/__init__ 2.py`, `src/beat_snp500/data/prices 2.py`
- Delete: `src/beat_snp500/features/__init__ 2.py`, `src/beat_snp500/features/betas 2.py`, `src/beat_snp500/features/monthly 2.py`
- Delete: `src/beat_snp500/jobs/backtest_report 2.py`
- Delete: `src/beat_snp500/models/champion 2.py`, `src/beat_snp500/models/registry 2.py`
- Delete: `src/beat_snp500/portfolio/__init__ 2.py`
- Delete: `tests/test_app_smoke 2.py`, `tests/test_bootstrap 2.py`, `tests/test_membership 2.py`, `tests/test_pipeline_leakage 2.py`, `tests/test_prices 2.py`, `tests/test_registry 2.py`, `tests/test_technical 2.py`

**Interfaces:**
- Consumes: nothing.
- Produces: a clean tree for later tasks' `git add` calls.

- [ ] **Step 1: Verify every duplicate is identical to its original, then delete**

Do NOT delete any file that differs — stop and report instead.

```bash
cd /Users/romannguyen/Documents/Projects/beat-snp500
find . -name "* 2.py" | while read f; do
  orig="${f/ 2.py/.py}"
  if diff -q "$f" "$orig" >/dev/null 2>&1; then
    rm "$f" && echo "deleted: $f"
  else
    echo "DIFFERS — NOT DELETED: $f"
  fi
done
```

Expected: 19 `deleted:` lines, zero `DIFFERS` lines.

- [ ] **Step 2: Confirm the suite still passes and no `* 2.py` files remain**

Run: `find . -name "* 2.py" | wc -l && .venv/bin/python -m pytest -q`
Expected: `0`, then `88 passed` (87 on a clean `main`, before Task 2 adds a test; note the duplicate test files had been inflating collection to 118 before this task). No commit — the files were untracked.

---

### Task 2: Min-cluster-size guard in `kmeans_top10` (TDD)

The bug: `kmeans_top10` picks the momentum cluster's top `n_picks` names, but never checks the cluster HAS `n_picks` members. In 52 of 197 backtest months the cluster had fewer — sometimes one stock at 100 % weight (all-TIE 2010–2012, all-NFLX Jan 2013). The fix returns `[]` for those months; `challenger_picks` already drops empty-pick months, and the engine holds prior positions through months absent from the picks dict.

**Files:**
- Modify: `src/beat_snp500/models/challenger.py:25-28`
- Test: `tests/test_challenger.py`
- Modify (downstream fixtures): `tests/test_daily.py:44-49`, `tests/test_backtest_report.py:8-16`

**Interfaces:**
- Consumes: `kmeans_top10(month_df: pd.DataFrame, n_picks: int = config.N_PICKS, k: int = config.K_CLUSTERS, seed: int = config.SEED) -> list[str]` — existing signature, unchanged.
- Produces: same signature; new contract: returns `[]` when the momentum cluster has fewer than `n_picks` members (previously returned the short list). Task 3's backtest rerun depends on this contract.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_challenger.py` (after `test_too_few_stocks_returns_empty`, which guards the *universe* being too small — this new test is the case where the universe is fine but the *cluster* is small):

```python
def test_small_momentum_cluster_returns_empty():
    # universe is plenty large, but the momentum cluster itself has fewer
    # members than n_picks -- shouldn't silently hand back a concentrated
    # sub-10-name (or single-stock) "portfolio"
    month, hot = make_month(n=40, n_hot=3)
    assert kmeans_top10(month) == []
```

(`make_month(n, n_hot)` is the existing helper at the top of this file: `n` random-feature tickers, of which `n_hot` get +5.0 on all momentum columns, forming an unmistakable momentum cluster of size `n_hot`.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_challenger.py::test_small_momentum_cluster_returns_empty -v`
Expected on clean `main`: FAIL — `kmeans_top10` returns the 3 hot tickers instead of `[]`.

- [ ] **Step 3: Implement the guard**

In `src/beat_snp500/models/challenger.py`, `kmeans_top10` currently ends:

```python
    # the momentum cluster is identified by its behaviour, never by label index
    best_cluster = composite.groupby(labels).mean().idxmax()
    members = composite[labels == best_cluster].sort_values(ascending=False)
    return members.head(n_picks).index.tolist()
```

Insert the guard so it ends:

```python
    # the momentum cluster is identified by its behaviour, never by label index
    best_cluster = composite.groupby(labels).mean().idxmax()
    members = composite[labels == best_cluster].sort_values(ascending=False)
    if len(members) < n_picks:
        return []
    return members.head(n_picks).index.tolist()
```

- [ ] **Step 4: Run the new test, then the full suite**

Run: `.venv/bin/python -m pytest tests/test_challenger.py -v`
Expected: all pass, including `test_small_momentum_cluster_returns_empty`.

Run: `.venv/bin/python -m pytest -q`
Expected on clean `main`: exactly two failures, both because their synthetic universes are too small to ever produce a ≥10-member momentum cluster under the new guard:
- `tests/test_daily.py::test_build_leaderboards_without_model` — `leaderboard_challenger.json` never written, assertion fails.
- `tests/test_backtest_report.py::test_run_report_writes_artifacts` — 15-ticker universe yields ~4-member clusters, so the challenger never picks; report artifacts fail.

- [ ] **Step 5: Enlarge the two downstream test fixtures**

In `tests/test_daily.py`, change `test_build_leaderboards_without_model` to:

```python
def test_build_leaderboards_without_model(make_panel, tmp_path):
    # n_tickers large enough that the momentum cluster k-means finds is
    # reliably >= N_PICKS, otherwise kmeans_top10's min-cluster-size guard
    # (see test_challenger.py) skips the month and no board gets written
    build_leaderboards(make_panel(n_months=5, n_tickers=200), None, tmp_path,
                       pd.Timestamp("2026-07-02"))
    assert not (tmp_path / "leaderboard_champion.json").exists()
    assert (tmp_path / "leaderboard_challenger.json").exists()
```

(`make_panel` is the existing conftest fixture; it already takes `n_tickers`, default 40.)

In `tests/test_backtest_report.py`, change the first lines of `test_run_report_writes_artifacts` from:

```python
    tickers = tuple(f"S{i:02d}" for i in range(15)) + ("SPY",)
    prices = make_prices(tickers=tickers)
    membership = make_membership(tickers=tuple(t for t in tickers if t != "SPY"))
    metrics = run_report(prices, membership, make_factors(), tmp_path,
                         top_n=15, train_window=12, n_draws=20)
```

to:

```python
    # enough tickers that a k=4 momentum cluster is reliably >= N_PICKS=10;
    # kmeans_top10's min-cluster-size guard skips months where it isn't
    # (see test_challenger.py::test_small_momentum_cluster_returns_empty)
    tickers = tuple(f"S{i:02d}" for i in range(60)) + ("SPY",)
    prices = make_prices(tickers=tickers)
    membership = make_membership(tickers=tuple(t for t in tickers if t != "SPY"))
    metrics = run_report(prices, membership, make_factors(), tmp_path,
                         top_n=60, train_window=12, n_draws=20)
```

The rest of both tests is unchanged.

- [ ] **Step 6: Run the full suite to verify it passes**

Run: `.venv/bin/python -m pytest -q`
Expected: `88 passed`, no warnings.

- [ ] **Step 7: Commit**

```bash
git add src/beat_snp500/models/challenger.py tests/test_challenger.py tests/test_daily.py tests/test_backtest_report.py
git commit -m "fix(challenger): skip months where the momentum cluster has fewer than 10 names

kmeans_top10 had no minimum-cluster-size guard, so in 52/197 backtest
months it returned a sub-10-name list -- sometimes a single stock at
100% weight. Return [] instead; the engine then holds the prior month's
positions, mirroring the guard champion_picks already had."
```

(No co-author trailer — see Global Constraints.)

---

### Task 3: Regenerate the backtest artifacts

Rerun the full 16-year walk-forward backtest so the persisted artifacts reflect the fixed challenger. Champion numbers are untouched by this change; challenger improves on every metric (this is the spec's results table).

**Files:**
- Modify (regenerated): `data/outputs/backtest/equity_curves.parquet`, `data/outputs/backtest/metrics.json`, `data/outputs/backtest/picks.json`
- Unchanged by rerun (champion-only / seeded): `ic_monthly.parquet`, `bootstrap.parquet`, `bootstrap_summary.json`, `survivorship.parquet`

**Interfaces:**
- Consumes: the Task 2 contract (`kmeans_top10` returns `[]` for thin clusters); local caches `data/prices.parquet`, `data/membership.parquet`, `data/ff5_factors.parquet` (present locally; `prices.parquet` is gitignored and rebuilds from yfinance in ~15 min on a cold machine).
- Produces: refreshed artifacts the Streamlit dashboard reads.

- [ ] **Step 1: Rerun the backtest report**

Run: `PYTHONPATH=src .venv/bin/python scripts/run_backtest.py`
(Takes several minutes — it retrains LightGBM monthly over ~160 walk-forward months.)

Expected output (exact values; everything is seeded):

```
champion       CAGR   7.92%  Sharpe  0.39  MaxDD -49.83%
challenger     CAGR  25.57%  Sharpe  0.92  MaxDD -37.92%
champion_ms    CAGR  10.32%  Sharpe  0.47  MaxDD -45.41%
challenger_ms  CAGR  24.68%  Sharpe  0.89  MaxDD -41.33%
spy            CAGR  14.70%  Sharpe  0.82  MaxDD -33.72%
```

This is the spec's table: challenger goes from 21.9 % CAGR / 0.76 Sharpe / −57 % maxDD to **25.6 % / 0.92 / −37.9 %**, now beating SPY (14.7 % / 0.82 / −33.7 %) on both CAGR and Sharpe.

- [ ] **Step 2: Sanity-check the changed artifacts**

Run: `git diff --stat data/outputs/backtest/ && .venv/bin/python -c "import json; m=json.load(open('data/outputs/backtest/metrics.json'))['challenger']; print(round(m['cagr'],4), round(m['sharpe'],2), round(m['max_drawdown'],4))"`
Expected: exactly `equity_curves.parquet`, `metrics.json`, `picks.json` modified, and `0.2557 0.92 -0.3792`. `picks.json` shrinks by ~590 lines — the challenger no longer emits pick lists for the 52 guard-skipped months.

- [ ] **Step 3: Commit**

```bash
git add data/outputs/backtest/equity_curves.parquet data/outputs/backtest/metrics.json data/outputs/backtest/picks.json
git commit -m "data: regenerate backtest artifacts after challenger min-cluster fix

challenger: 21.9% CAGR / 0.76 Sharpe / -57% maxDD ->
            25.6% CAGR / 0.92 Sharpe / -37.9% maxDD"
```

---

### Task 4: Dashboard copy and Methodology accuracy

Two edits to `app/streamlit_app.py`: the retitled header (already prototyped in the tree), and a Methodology-tab update so the Challenger description documents the new guard (new work — not yet in the tree).

**Files:**
- Modify: `app/streamlit_app.py:58-60` (title/caption), `app/streamlit_app.py:175-176` (Methodology bullet)
- Test: `tests/test_app_smoke.py` (existing smoke test, no new tests)

**Interfaces:**
- Consumes: nothing from other tasks (copy only).
- Produces: nothing consumed downstream.

- [ ] **Step 1: Retitle the dashboard**

In `app/streamlit_app.py`, replace:

```python
st.title("beat-snp500")
st.caption("Educational quant research project — NOT investment advice. "
           "Champion: LightGBM walk-forward ranking. Challenger: K-Means momentum cluster.")
```

with:

```python
st.title("Can you beat S&P 500?")
st.caption("This is an educational quant research project, NOT investment advice. "
           "The model uses LightGBM for walk-forward ranking and a challenger as K-Means momentum cluster.")
```

- [ ] **Step 2: Document the guard in the Methodology tab**

In the same file's Methodology markdown block, replace the Challenger bullet:

```markdown
- **Challenger:** monthly K-Means (k=4); the momentum cluster is identified by centroid
  behaviour, and its top 10 stocks by composite momentum are selected.
```

with:

```markdown
- **Challenger:** monthly K-Means (k=4); the momentum cluster is identified by centroid
  behaviour, and its top 10 stocks by composite momentum are selected. If the momentum
  cluster has fewer than 10 members that month, no trade is made and the prior month's
  holdings are kept (no forced concentration into a handful of names).
```

- [ ] **Step 3: Run the app smoke test**

Run: `.venv/bin/python -m pytest tests/test_app_smoke.py -q`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add app/streamlit_app.py
git commit -m "docs(app): document challenger min-cluster guard; retitle dashboard"
```

---

### Task 5: Land the investigation write-up and sanity-check scripts

The spec itself (`improvement/improve_v1.md`) and the three scripts that produced its evidence belong in the repo: they are the audit trail for "overfitting ruled out" (Step 1), "features carry no monthly signal" (Step 2), and "the bug was found by decomposing the cluster edge" (Step 3).

**Files:**
- Commit (already written, currently untracked): `improvement/improve_v1.md`, `scripts/sanity_check_linear.py`, `scripts/sanity_check_feature_ic.py`, `scripts/sanity_check_cluster_momentum.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed downstream; documentation only.

- [ ] **Step 1: Verify the scripts still import cleanly**

They are one-off research scripts (each needs `data/prices.parquet` and several minutes to actually run — do NOT run them here), so just compile-check:

```bash
.venv/bin/python -m py_compile scripts/sanity_check_linear.py scripts/sanity_check_feature_ic.py scripts/sanity_check_cluster_momentum.py && echo OK
```

Expected: `OK`.

- [ ] **Step 2: Commit**

```bash
git add improvement/improve_v1.md scripts/sanity_check_linear.py scripts/sanity_check_feature_ic.py scripts/sanity_check_cluster_momentum.py
git commit -m "docs: add challenger investigation write-up and sanity-check scripts

- sanity_check_linear: Ridge on same features matches LightGBM's ~-0.013
  OOS IC (49% win rate) -- rules out overfitting
- sanity_check_feature_ic: every raw feature |IC| < 0.016 -- the feature
  set carries ~no monthly cross-sectional signal on this universe
- sanity_check_cluster_momentum: decomposes the challenger's edge and
  surfaced the sub-10-name concentration bug fixed in the prior commits"
```

- [ ] **Step 3: Final verification**

Run: `git status --short && .venv/bin/python -m pytest -q`
Expected: clean tree (nothing modified or untracked except local data caches), `88 passed`.
