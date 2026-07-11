# Must-Buy Selection & Champion Re-Rating Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed top-10 stock picking with variable-count "must-buy" selection (5–10 names, conviction-weighted, 20% cap), re-rate K-means as champion via rename + role pointer, and run a tuning/feature improvement round gated by a dev/holdout protocol.

**Architecture:** Two monthly stock-picking models (`models/kmeans.py`, `models/lgbm.py`) emit `{date: {ticker: weight}}` pick dicts consumed unchanged by the backtest engine, the daily job, and the dashboard. Selection = signal threshold (momentum z > 0 for kmeans, score z ≥ 1 for lgbm) with a hold-below-5 floor and keep-top-10 cap; weights = signal-proportional with 20% cap. `config.CHAMPION` is the single role pointer. Tuning and new features are validated on months ≤ 2019-12 with one holdout confirmation on 2020+.

**Tech Stack:** Python 3.13 (`.venv`), pandas 2.x (<3), scikit-learn KMeans, LightGBM, pytest, Streamlit + Plotly.

**Spec:** `docs/superpowers/specs/2026-07-10-must-buy-selection-design.md`

## Global Constraints

- Run all tests with `.venv/bin/python -m pytest` (pytest `pythonpath=["src"]` handles imports; the editable-install `.pth` is unreliable on this machine — never rely on `pip install -e` imports outside pytest).
- pandas must stay `>=2.2,<3.0`; do not bump dependencies.
- Git commits: conventional prefixes matching repo history (`feat:`, `fix:`, `docs:`, `data:`, `test:`, `refactor:`). **Never add a `Co-Authored-By` trailer — commits must show only Roman as author.**
- Config values, verbatim from the spec: `MIN_PICKS = 5`, `MAX_PICKS = 10`, `WEIGHT_CAP = 0.20`, `MUST_BUY_Z_KMEANS = 0.0`, `MUST_BUY_Z_LGBM = 1.0`, `CHAMPION = "kmeans"`, `DEV_END = "2019-12-31"`.
- Series names everywhere: `kmeans`, `lgbm`, `kmeans_ms`, `lgbm_ms`, `spy`. The words champion/challenger are *roles*, resolved via `config.CHAMPION`, never hardcoded identities.
- Hold semantics: a model that can't field ≥ `MIN_PICKS` must-buys emits **no entry** for that month (the backtest engine and live holdings already interpret a missing month as "keep previous portfolio").
- Every task must leave the full suite green: `.venv/bin/python -m pytest tests/ -q` → all pass. Old `challenger.py`/`champion.py` coexist with new modules until Task 6 removes them.
- Data files under `data/` and `models/` are committed artifacts in this repo (see `data:` commits in history) — commit them when a task changes them.

## File Map

| File | Fate | Responsibility |
|---|---|---|
| `src/beat_snp500/config.py` | modify | new constants; `FEATURES` split into `BASE_FEATURES` + promotable `FEATURES` |
| `src/beat_snp500/portfolio/weights.py` | modify | add `conviction_weights` |
| `src/beat_snp500/models/kmeans.py` | create (T2) | clustering + must-buy selection + picks |
| `src/beat_snp500/models/lgbm.py` | create (T3) | walk-forward scoring + must-buy selection + picks (evolved copy of champion.py) |
| `src/beat_snp500/models/challenger.py`, `champion.py` | delete (T6) | superseded |
| `src/beat_snp500/backtest/bootstrap.py` | modify (T4) | per-month draw counts |
| `src/beat_snp500/jobs/backtest_report.py` | modify (T5) | new series names, champion-role bootstrap |
| `src/beat_snp500/jobs/daily.py` | modify (T6) | PM leaderboards/holdings, `lgbm` registry type |
| `scripts/migrate_model_names.py` | create (T7) | one-time idempotent rename of stored artifacts |
| `app/streamlit_app.py` | modify (T8) | PM portfolio view, champion badge, new names |
| `src/beat_snp500/features/advanced.py` | create (T9) | resid/vol-scaled/cluster-relative momentum |
| `src/beat_snp500/features/pipeline.py` | modify (T9) | wire advanced features |
| `scripts/tune_kmeans.py` | create (T10) | dev/holdout grid search |
| `scripts/evaluate_features.py` | create (T11) | dev/holdout feature IC gate |
| `improvement/improve_v2.md` | create (T12) | results write-up |

---

### Task 1: Config constants + `conviction_weights`

**Files:**
- Modify: `src/beat_snp500/config.py`
- Modify: `src/beat_snp500/portfolio/weights.py`
- Test: `tests/test_weights.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `config.BASE_FEATURES: list[str]` (the current 14 features), `config.FEATURES: list[str]` (same list for now; promotable), `config.MIN_PICKS=5`, `config.MAX_PICKS=10`, `config.WEIGHT_CAP=0.20`, `config.MUST_BUY_Z_KMEANS=0.0`, `config.MUST_BUY_Z_LGBM=1.0`, `config.CHAMPION="kmeans"`, `config.DEV_END="2019-12-31"`; `conviction_weights(signals: dict[str, float], cap: float = config.WEIGHT_CAP) -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_weights.py`:

```python
from beat_snp500.portfolio.weights import conviction_weights


def test_conviction_weights_proportional_and_sums_to_one():
    w = conviction_weights({"A": 2.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["A"] == pytest.approx(2 * w["B"])


def test_conviction_weights_caps_and_redistributes():
    w = conviction_weights({"A": 100.0, "B": 1.0, "C": 1.0, "D": 1.0,
                            "E": 1.0, "F": 1.0})
    assert w["A"] == pytest.approx(0.20)
    assert sum(w.values()) == pytest.approx(1.0)
    assert all(v <= 0.20 + 1e-9 for v in w.values())


def test_conviction_weights_floor_case_forces_equal_20pct():
    # 5 names, extreme conviction spread: cap forces exactly 5 x 20%
    w = conviction_weights({"A": 100.0, "B": 50.0, "C": 1.0, "D": 1.0, "E": 1.0})
    assert all(v == pytest.approx(0.20) for v in w.values())


def test_conviction_weights_rejects_nonpositive_signals():
    with pytest.raises(ValueError):
        conviction_weights({"A": 1.0, "B": 0.0, "C": 1.0, "D": 1.0, "E": 1.0})


def test_conviction_weights_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        conviction_weights({"A": 1.0, "B": 1.0}, cap=0.20)  # 2 * 0.2 < 1


def test_conviction_weights_empty_returns_empty():
    assert conviction_weights({}) == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_weights.py -q`
Expected: ImportError — `conviction_weights` not defined.

- [ ] **Step 3: Implement.** In `src/beat_snp500/config.py`, rename the existing `FEATURES = [...]` list to `BASE_FEATURES = [...]` and add below it (delete nothing else; keep `N_PICKS` for now — it's removed in Task 6):

```python
# model inputs; validated extras get appended here (spec §4b), while
# BASE_FEATURES stays the clustering space for K-means
FEATURES = list(BASE_FEATURES)

MIN_PICKS = 5          # fewer must-buys than this -> hold previous portfolio
MAX_PICKS = 10         # more than this -> keep the highest-signal names
WEIGHT_CAP = 0.20      # per-stock cap; 5 * 0.20 = 1.0 keeps the floor fully invested
MUST_BUY_Z_KMEANS = 0.0
MUST_BUY_Z_LGBM = 1.0
CHAMPION = "kmeans"    # role pointer (re-rated 2026-07); the other model is challenger
DEV_END = "2019-12-31"  # tuning/feature selection uses months <= this; one holdout pass after
```

In `src/beat_snp500/portfolio/weights.py` add (plus `from beat_snp500 import config` at the top):

```python
def conviction_weights(signals: dict[str, float],
                       cap: float = config.WEIGHT_CAP) -> dict[str, float]:
    """Signal-proportional weights with a per-stock cap (water-filling).

    Selection thresholds are >= 0 so signals must be strictly positive.
    Feasibility (len * cap >= 1) is guaranteed by MIN_PICKS * WEIGHT_CAP = 1;
    with that guard the redistribution loop always terminates with sum == 1.
    """
    if not signals:
        return {}
    w = pd.Series(signals, dtype=float)
    if (w <= 0).any():
        raise ValueError("conviction_weights requires strictly positive signals")
    if len(w) * cap < 1.0 - 1e-12:
        raise ValueError(f"cap {cap} infeasible for {len(w)} names")
    w = w / w.sum()
    for _ in range(len(w)):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = w < cap - 1e-12
        w[under] += excess * w[under] / float(w[under].sum())
    return {t: float(v) for t, v in w.items()}
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_weights.py tests/ -q`
Expected: all pass (renaming `FEATURES`→`BASE_FEATURES` while re-exporting `FEATURES` keeps every existing consumer working).

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/config.py src/beat_snp500/portfolio/weights.py tests/test_weights.py
git commit -m "feat(portfolio): conviction-weighted allocation with 20% cap + selection config"
```

---

### Task 2: K-means must-buy model (`models/kmeans.py`)

**Files:**
- Create: `src/beat_snp500/models/kmeans.py`
- Test: `tests/test_kmeans.py` (new; `test_challenger.py` stays until Task 6)

**Interfaces:**
- Consumes: `conviction_weights` (Task 1), `config.BASE_FEATURES`, `config.MIN_PICKS/MAX_PICKS/MUST_BUY_Z_KMEANS/K_CLUSTERS/SEED`.
- Produces: `cluster_month(month_df, k=config.K_CLUSTERS, seed=config.SEED) -> tuple[pd.DataFrame, pd.Series]` (standardized features `Xz` + integer cluster `labels`, both indexed by ticker; `(empty, empty)` if fewer rows than k); `kmeans_must_buys(month_df, ...) -> dict[str, float]` ({ticker: momentum z}, `{}` = hold); `kmeans_picks(panel) -> dict[pd.Timestamp, dict[str, float]]` (conviction weights). Task 9 reuses `cluster_month`; Task 10 adds `mom_mode`/`select_rule` params.

- [ ] **Step 1: Write the failing tests** — create `tests/test_kmeans.py`:

```python
import numpy as np
import pandas as pd

from beat_snp500 import config
from beat_snp500.models.kmeans import (cluster_month, kmeans_must_buys,
                                        kmeans_picks)


def make_month(n=40, n_hot=10, seed=0):
    rng = np.random.default_rng(seed)
    tickers = [f"T{i:02d}" for i in range(n)]
    df = pd.DataFrame(rng.normal(0, 0.3, (n, len(config.FEATURES))),
                      index=pd.Index(tickers, name="ticker"), columns=config.FEATURES)
    hot = tickers[:n_hot]
    for c in ["return_3m", "return_6m", "return_12m"]:
        df.loc[hot, c] += 5.0  # unmistakable momentum cluster
    return df, hot


def test_cluster_month_shapes():
    month, _ = make_month()
    Xz, labels = cluster_month(month)
    assert list(Xz.index) == list(labels.index)
    assert labels.nunique() == config.K_CLUSTERS


def test_must_buys_finds_momentum_cluster():
    month, hot = make_month()
    must = kmeans_must_buys(month)
    assert config.MIN_PICKS <= len(must) <= config.MAX_PICKS
    assert set(must) <= set(hot)
    assert all(v > config.MUST_BUY_Z_KMEANS for v in must.values())


def test_must_buys_capped_at_max_picks():
    month, hot = make_month(n=60, n_hot=20)
    must = kmeans_must_buys(month)
    assert len(must) <= config.MAX_PICKS
    assert set(must) <= set(hot)


def test_must_buys_deterministic():
    month, _ = make_month()
    assert kmeans_must_buys(month) == kmeans_must_buys(month)


def test_too_few_stocks_holds():
    month, _ = make_month(n=3, n_hot=1)
    assert kmeans_must_buys(month) == {}


def test_small_momentum_cluster_holds():
    # the guard from improve_v1: a tiny trending cluster must never become
    # a concentrated bet
    month, _ = make_month(n=40, n_hot=3)
    assert kmeans_must_buys(month) == {}


def test_kmeans_picks_weights_valid():
    month, _ = make_month()
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    panel = pd.concat({d: month for d in dates}, names=["date"])
    picks = kmeans_picks(panel)
    assert set(picks) == set(dates)
    for w in picks.values():
        assert config.MIN_PICKS <= len(w) <= config.MAX_PICKS
        assert sum(w.values()) == 1.0 or abs(sum(w.values()) - 1.0) < 1e-9
        assert max(w.values()) <= config.WEIGHT_CAP + 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_kmeans.py -q`
Expected: ModuleNotFoundError — `beat_snp500.models.kmeans` does not exist.

- [ ] **Step 3: Implement** — create `src/beat_snp500/models/kmeans.py`:

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from beat_snp500 import config
from beat_snp500.portfolio.weights import conviction_weights

MOM_COLS = ["return_3m", "return_6m", "return_12m"]


def cluster_month(month_df: pd.DataFrame, k: int = config.K_CLUSTERS,
                  seed: int = config.SEED) -> tuple[pd.DataFrame, pd.Series]:
    """Standardize BASE_FEATURES and K-means-label one month's cross-section."""
    X = month_df[config.BASE_FEATURES].dropna()
    if len(X) < k:
        return pd.DataFrame(), pd.Series(dtype=int)
    Xz = pd.DataFrame(StandardScaler().fit_transform(X.values),
                      index=X.index, columns=X.columns)
    labels = pd.Series(
        KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Xz.values),
        index=X.index,
    )
    return Xz, labels


def kmeans_must_buys(month_df: pd.DataFrame, k: int = config.K_CLUSTERS,
                     threshold: float = config.MUST_BUY_Z_KMEANS,
                     min_picks: int = config.MIN_PICKS,
                     max_picks: int = config.MAX_PICKS,
                     seed: int = config.SEED) -> dict[str, float]:
    """{ticker: momentum z-score} for the must-buy set; {} means hold."""
    Xz, labels = cluster_month(month_df, k=k, seed=seed)
    if Xz.empty:
        return {}
    composite = Xz[MOM_COLS].mean(axis=1)
    # the momentum cluster is identified by its behaviour, never by label index
    best_cluster = composite.groupby(labels).mean().idxmax()
    members = composite[labels == best_cluster]
    must = members[members > threshold].sort_values(ascending=False)
    if len(must) < min_picks:
        return {}
    return must.head(max_picks).to_dict()


def kmeans_picks(panel: pd.DataFrame, **kwargs) -> dict:
    out = {}
    for t, month_df in panel.groupby(level="date"):
        signals = kmeans_must_buys(month_df.droplevel("date"), **kwargs)
        if signals:
            out[t] = conviction_weights(signals)
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_kmeans.py tests/ -q`
Expected: all pass. (If `test_must_buys_capped_at_max_picks` is flaky on the seeded k-means split, raise `make_month(n=80, n_hot=20)` — the assertion logic stays identical.)

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/models/kmeans.py tests/test_kmeans.py
git commit -m "feat(kmeans): variable-count must-buy selection with conviction weights"
```

---

### Task 3: LightGBM must-buy model (`models/lgbm.py`)

**Files:**
- Create: `src/beat_snp500/models/lgbm.py` (evolved copy of `models/champion.py`; champion.py is untouched until Task 6)
- Test: `tests/test_lgbm.py`

**Interfaces:**
- Consumes: `conviction_weights` (Task 1), `config.MUST_BUY_Z_LGBM/MIN_PICKS/MAX_PICKS`.
- Produces: everything `champion.py` exported, renamed where noted — `LGB_PARAMS`, `CANDIDATE_PARAMS`, `spearman_ic(scores, fwd)`, `decile_spread(scores, fwd)`, `select_params(panel, train_window, val_months)`, `walk_forward_scores(panel, train_window, params)`, `train_lgbm(labeled_panel, train_window, params) -> (model, val_ic)` (was `train_champion`), `save_model(model, path)`, `load_model(path)`, plus new `lgbm_must_buys(scores_month: pd.Series, threshold=config.MUST_BUY_Z_LGBM, min_picks=config.MIN_PICKS, max_picks=config.MAX_PICKS) -> dict[str, float]` and `lgbm_picks(scores: pd.Series) -> dict[pd.Timestamp, dict[str, float]]` (replaces `champion_picks`).

- [ ] **Step 1: Write the failing tests** — create `tests/test_lgbm.py`:

```python
import pandas as pd

from beat_snp500 import config
from beat_snp500.models.lgbm import (lgbm_must_buys, lgbm_picks, load_model,
                                      save_model, spearman_ic, train_lgbm,
                                      walk_forward_scores)


def test_lgbm_must_buys_thresholds_on_zscore():
    # 15 mediocre names + 5 clear winners: only the winners clear z >= 1
    s = pd.Series({f"L{i:02d}": 0.0 for i in range(15)}
                  | {f"H{i:02d}": 10.0 for i in range(5)})
    must = lgbm_must_buys(s)
    assert set(must) == {f"H{i:02d}" for i in range(5)}
    assert all(v >= config.MUST_BUY_Z_LGBM for v in must.values())


def test_lgbm_must_buys_holds_when_conviction_is_thin():
    # uniform scores 0..19: only ~4 names reach z >= 1 -> below MIN_PICKS -> hold
    s = pd.Series({f"T{i:02d}": float(i) for i in range(20)})
    assert lgbm_must_buys(s) == {}


def test_lgbm_must_buys_degenerate_scores_hold():
    s = pd.Series({f"T{i:02d}": 1.0 for i in range(20)})  # zero dispersion
    assert lgbm_must_buys(s) == {}


def test_walk_forward_learns_planted_signal(make_panel):
    panel = make_panel(n_months=40, n_tickers=40)
    scores = walk_forward_scores(panel, train_window=12)
    assert len(scores.index.get_level_values("date").unique()) == 40 - 12
    assert spearman_ic(scores, panel["fwd_return_1m"]).mean() > 0.8


def test_lgbm_picks_weights_valid(make_panel):
    panel = make_panel(n_months=20, n_tickers=30)
    scores = walk_forward_scores(panel, train_window=12)
    picks = lgbm_picks(scores)
    assert picks  # planted signal: at least some active months
    for w in picks.values():
        assert config.MIN_PICKS <= len(w) <= config.MAX_PICKS
        assert abs(sum(w.values()) - 1.0) < 1e-9
        assert max(w.values()) <= config.WEIGHT_CAP + 1e-9


def test_train_save_load_roundtrip(make_panel, tmp_path):
    panel = make_panel(n_months=30)
    model, val_ic = train_lgbm(panel.dropna(subset=["fwd_return_1m"]),
                               train_window=12)
    assert val_ic > 0.5
    p = tmp_path / "lgbm.txt"
    save_model(model, p)
    booster = load_model(p)
    month = panel.xs(panel.index.get_level_values("date").max(), level="date")
    assert len(booster.predict(month[config.FEATURES])) == len(month)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_lgbm.py -q`
Expected: ModuleNotFoundError — `beat_snp500.models.lgbm`.

- [ ] **Step 3: Implement.** Copy `src/beat_snp500/models/champion.py` to `src/beat_snp500/models/lgbm.py` (`cp`, not `git mv` — champion.py must keep working until Task 6), then in the new file:

1. Add the import: `from beat_snp500.portfolio.weights import conviction_weights`.
2. Rename `train_champion` → `train_lgbm` (body unchanged).
3. Delete `champion_picks` and add:

```python
def lgbm_must_buys(scores_month: pd.Series,
                   threshold: float = config.MUST_BUY_Z_LGBM,
                   min_picks: int = config.MIN_PICKS,
                   max_picks: int = config.MAX_PICKS) -> dict[str, float]:
    """{ticker: cross-sectional score z} clearing the conviction bar; {} = hold.

    A z-threshold, not a raw predicted-percentile bar: regression predictions
    compress toward the centre when signal is weak, so a raw 0.9 cut would
    select ~0 names most months. threshold must stay > 0 (weights need
    strictly positive signals).
    """
    s = scores_month.dropna()
    sd = s.std(ddof=0)
    if len(s) < min_picks or sd == 0:
        return {}
    z = (s - s.mean()) / sd
    must = z[z >= threshold].sort_values(ascending=False)
    if len(must) < min_picks:
        return {}
    return must.head(max_picks).to_dict()


def lgbm_picks(scores: pd.Series) -> dict:
    out = {}
    for t, s in scores.groupby(level="date"):
        signals = lgbm_must_buys(s.droplevel("date"))
        if signals:
            out[t] = conviction_weights(signals)
    return out
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_lgbm.py tests/ -q`
Expected: all pass, including the untouched `test_champion.py`. (If the seeded `test_lgbm_picks_weights_valid` yields zero active months, raise `n_tickers` to 60 — expected active-month rate is high since ~16% of 30 names clear 1σ.)

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/models/lgbm.py tests/test_lgbm.py
git commit -m "feat(lgbm): z-threshold must-buy selection replacing fixed top-10"
```

---

### Task 4: Count-matched bootstrap

**Files:**
- Modify: `src/beat_snp500/backtest/bootstrap.py`
- Test: `tests/test_bootstrap.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `random_portfolio_bootstrap(universe, holding_rets, n_draws=1000, n_picks: int | dict = 10, cost_bps=config.COST_BPS_ONE_WAY, seed=config.SEED) -> dict` — when `n_picks` is a dict keyed by month timestamp, each month draws that month's count.

- [ ] **Step 1: Write the failing test** — append to `tests/test_bootstrap.py`:

```python
def test_per_month_pick_counts():
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    tickers = [f"T{i}" for i in range(30)]
    # month 1: winners return 10%, rest 0%; a 5-name draw can average up to
    # 10% while a 10-name draw of the same names caps at 5% -- the band
    # widths differ iff the per-month count is respected
    hr = pd.DataFrame(0.0, index=dates, columns=tickers)
    hr.iloc[0, :5] = 0.10
    uni = {t: tickers for t in dates}
    counts = {dates[0]: 5, dates[1]: 10}
    out = random_portfolio_bootstrap(uni, hr, n_draws=500, n_picks=counts,
                                     cost_bps=0.0)
    # p95 of month-1 equity exceeds anything a 10-name portfolio could reach
    assert out["band"]["p95"].iloc[0] > 1.05 + 1e-9
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_bootstrap.py -q`
Expected: FAIL — `n_picks` dict is not accepted / TypeError in `min()`.

- [ ] **Step 3: Implement.** In `random_portfolio_bootstrap`, change the signature default `n_picks: int = config.N_PICKS` to `n_picks: int | dict = 10`, and inside the date loop replace `k = min(n_picks, len(rets_t))` with:

```python
        want = n_picks[t] if isinstance(n_picks, dict) else n_picks
        k = min(want, len(rets_t))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_bootstrap.py tests/ -q
`
Expected: all pass (existing tests use the int form, unchanged behavior).

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/backtest/bootstrap.py tests/test_bootstrap.py
git commit -m "feat(bootstrap): count-matched random portfolios for variable pick counts"
```

---

### Task 5: Backtest report on the new models

**Files:**
- Modify: `src/beat_snp500/jobs/backtest_report.py`
- Test: `tests/test_backtest_report.py`

**Interfaces:**
- Consumes: `kmeans_picks` (T2), `lgbm_picks`/`walk_forward_scores`/`spearman_ic`/`decile_spread` (T3), dict-form `n_picks` (T4), `config.CHAMPION`.
- Produces: `metrics.json` keyed `kmeans/lgbm/kmeans_ms/lgbm_ms/spy` (+ `yearly`, `turnover`, `survivorship`); `picks.json` keyed the same four; `bootstrap_summary.json` still carries `champion_cagr_percentile` (champion resolved via `config.CHAMPION`).

- [ ] **Step 1: Update the test.** In `tests/test_backtest_report.py::test_run_report_writes_artifacts` replace the assertions after `metrics = run_report(...)` from the metrics-series check to the end of the function with:

```python
    for f in ["equity_curves.parquet", "metrics.json", "ic_monthly.parquet",
              "bootstrap.parquet", "bootstrap_summary.json", "survivorship.parquet",
              "picks.json"]:
        assert (tmp_path / f).exists(), f
    m = read_json(tmp_path / "metrics.json")
    for series in ["kmeans", "lgbm", "spy"]:
        assert "cagr" in m[series]
    assert "champion_cagr_percentile" in read_json(tmp_path / "bootstrap_summary.json")
    assert "survivorship" in m
    surv = pd.read_parquet(tmp_path / "survivorship.parquet")
    assert (surv["missing_frac"] == 0).all()

    picks_data = read_json(tmp_path / "picks.json")
    assert set(picks_data) == {"kmeans", "lgbm", "kmeans_ms", "lgbm_ms"}
    for name in ("kmeans", "lgbm"):
        assert picks_data[name], f"{name} emitted no months at all"
        for weights in picks_data[name].values():
            assert 5 <= len(weights) <= 10
            assert sum(weights.values()) == pytest.approx(1.0)
            assert max(weights.values()) <= 0.20 + 1e-9
```

(The old `ic_months` length-vs-picks relation no longer holds: lgbm may hold some months, so picks months are a subset of scored months.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_backtest_report.py -q`
Expected: FAIL — metrics.json still keyed champion/challenger.

- [ ] **Step 3: Implement.** In `src/beat_snp500/jobs/backtest_report.py`:

Replace the model imports:

```python
from beat_snp500.models.kmeans import kmeans_picks
from beat_snp500.models.lgbm import (decile_spread, lgbm_picks, spearman_ic,
                                      walk_forward_scores)
```

Replace the picks block and `start` line:

```python
    picks = {
        "lgbm": lgbm_picks(scores),
        "kmeans": kmeans_picks(panel),
    }
    picks["lgbm_ms"] = _with_max_sharpe(picks["lgbm"], close)
    picks["kmeans_ms"] = _with_max_sharpe(picks["kmeans"], close)

    results = {name: run_backtest(p, close) for name, p in picks.items()}
    # common reporting window starts where walk-forward lgbm can first trade
    start = results["lgbm"].daily_returns.index.min()
```

Replace the bootstrap block (`champ_months` through `boot_summary`):

```python
    champ = config.CHAMPION
    # clip to the common reporting window so the random band matches the chart
    champ_months = [t for t in sorted(picks[champ])
                    if t >= start - pd.offsets.MonthEnd(1)]
    universe = {t: panel.xs(t, level="date").index.tolist() for t in champ_months}
    counts = {t: len(picks[champ][t]) for t in champ_months}
    boot = random_portfolio_bootstrap(
        universe, monthly_holding_returns(close, champ_months),
        n_draws=n_draws, n_picks=counts)
    champ_cagr = metrics[champ]["cagr"]
    boot_summary = {
        "cagr_p05": float(np.quantile(boot["cagr"], 0.05)),
        "cagr_p50": float(np.quantile(boot["cagr"], 0.50)),
        "cagr_p95": float(np.quantile(boot["cagr"], 0.95)),
        "champion_cagr_percentile": float((boot["cagr"] < champ_cagr).mean()),
    }
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_backtest_report.py tests/ -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/jobs/backtest_report.py tests/test_backtest_report.py
git commit -m "feat(report): backtest on must-buy portfolios; kmeans/lgbm series; champion-role bootstrap"
```

---

### Task 6: Daily job on the new models; delete the old modules

**Files:**
- Modify: `src/beat_snp500/jobs/daily.py`, `src/beat_snp500/config.py` (remove `N_PICKS`), `scripts/sanity_check_cluster_momentum.py`, `scripts/sanity_check_feature_ic.py`, `scripts/sanity_check_linear.py` (import fixes only)
- Delete: `src/beat_snp500/models/challenger.py`, `src/beat_snp500/models/champion.py`, `tests/test_challenger.py`, `tests/test_champion.py`
- Test: `tests/test_daily.py`

**Interfaces:**
- Consumes: `kmeans_must_buys` (T2), `lgbm_must_buys`/`train_lgbm`/`save_model`/`load_model` (T3), `conviction_weights` (T1).
- Produces: `data/outputs/leaderboard_kmeans.json` / `leaderboard_lgbm.json` with payload `{"as_of", "signal_month", "status": "active"|"hold", "picks": [{"ticker", "weight", "score", "features"}]}` (picks sorted by weight desc; empty on hold); `holdings_kmeans.json` / `holdings_lgbm.json` (schema unchanged: `signal_date`, `generated_at`, `weights`); registry entries `type: "lgbm"`, `model_id: f"lgbm_{YYYYMM}"`. `build_leaderboards(panel, booster, out_dir, as_of)` loses its `n_picks` parameter.

- [ ] **Step 1: Update the tests.** In `tests/test_daily.py`:

Replace the imports of `train_champion` with `from beat_snp500.models.lgbm import train_lgbm` (and use it in `test_build_leaderboards`). Then replace the three affected tests:

```python
def test_build_leaderboards(make_panel, tmp_path):
    panel = make_panel(n_months=30)
    model, _ = train_lgbm(panel.dropna(subset=["fwd_return_1m"]), train_window=12)
    build_leaderboards(panel, model.booster_, tmp_path, pd.Timestamp("2026-07-02"))
    board = read_json(tmp_path / "leaderboard_lgbm.json")
    assert board["as_of"] == "2026-07-02"
    assert board["status"] in ("active", "hold")
    if board["status"] == "active":
        assert 5 <= len(board["picks"]) <= 10
        weights = [p["weight"] for p in board["picks"]]
        assert weights == sorted(weights, reverse=True)
        assert sum(weights) == pytest.approx(1.0)
        assert set(board["picks"][0]["features"]) == set(config.FEATURES)
    assert (tmp_path / "leaderboard_kmeans.json").exists()


def test_build_leaderboards_without_model(make_panel, tmp_path):
    build_leaderboards(make_panel(n_months=5, n_tickers=200), None, tmp_path,
                       pd.Timestamp("2026-07-02"))
    assert not (tmp_path / "leaderboard_lgbm.json").exists()
    assert (tmp_path / "leaderboard_kmeans.json").exists()


def test_monthly_rebalance_writes_artifacts(make_panel, tmp_path):
    models_dir, out_dir = tmp_path / "models", tmp_path / "out"
    reg = models_dir / "registry.json"
    monthly_rebalance(make_panel(n_months=30), models_dir, out_dir, reg,
                      pd.Timestamp("2026-07-01"), train_window=12)
    entries = load_registry(reg)
    assert len(entries) == 1 and entries[0]["type"] == "lgbm"
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

Also update the two `holdings = {"champion": ...}` literals in the live-track tests to `{"lgbm": ...}` (the function is name-agnostic; this just keeps fixtures honest).

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_daily.py -q`
Expected: FAIL — old leaderboard/holdings names and registry type.

- [ ] **Step 3: Implement.** In `src/beat_snp500/jobs/daily.py`:

Replace model imports:

```python
from beat_snp500.models.kmeans import kmeans_must_buys
from beat_snp500.models.lgbm import (lgbm_must_buys, load_model, save_model,
                                      train_lgbm)
from beat_snp500.portfolio.weights import conviction_weights
```

(drop the `equal_weights` import). Replace `build_leaderboards` entirely:

```python
def build_leaderboards(panel: pd.DataFrame, booster, out_dir, as_of) -> dict:
    latest = panel.index.get_level_values("date").max()
    month = panel.xs(latest, level="date")
    feats = month[config.FEATURES]
    signals = {}
    if booster is not None:
        scores = pd.Series(booster.predict(feats), index=feats.index)
        signals["lgbm"] = lgbm_must_buys(scores)
    signals["kmeans"] = kmeans_must_buys(month)
    boards = {}
    for model, sig in signals.items():
        weights = conviction_weights(sig) if sig else {}
        payload = {
            "as_of": str(pd.Timestamp(as_of).date()),
            "signal_month": str(latest.date()),
            "status": "active" if weights else "hold",
            "picks": [
                {"ticker": t, "weight": float(weights[t]), "score": float(sig[t]),
                 "features": {k: float(feats.loc[t, k]) for k in config.FEATURES}}
                for t in sorted(weights, key=weights.get, reverse=True)
            ],
        }
        atomic_write_json(payload, Path(out_dir) / f"leaderboard_{model}.json")
        boards[model] = weights
    return boards
```

In `monthly_rebalance`, replace from `model, val_ic = ...` through the end with:

```python
    model, val_ic = train_lgbm(labeled, train_window=train_window)
    latest = panel_completed.index.get_level_values("date").max()
    model_id = f"lgbm_{latest:%Y%m}"
    artifact = Path(models_dir) / f"{model_id}.txt"
    save_model(model, artifact)
    append_entry(registry_path, {
        "model_id": model_id, "type": "lgbm",
        "trained_through": str(labeled.index.get_level_values("date").max().date()),
        "train_window_months": train_window, "ic_mean": val_ic,
        "created_at": str(pd.Timestamp(as_of).date()), "artifact": artifact_ref(artifact),
    })

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
```

In `run()`: `latest_model(config.REGISTRY_JSON, "lgbm")` and `for name in ["lgbm", "kmeans"]:`.

Then: delete `src/beat_snp500/models/challenger.py`, `src/beat_snp500/models/champion.py`, `tests/test_challenger.py`, `tests/test_champion.py`. Remove `N_PICKS = 10` from `config.py`. Fix sanity-script imports: in `scripts/sanity_check_cluster_momentum.py` change `from beat_snp500.models.challenger import MOM_COLS` to `from beat_snp500.models.kmeans import MOM_COLS`; run `grep -n "models.champion\|models.challenger" scripts/*.py` and point any remaining imports at `beat_snp500.models.lgbm` (`spearman_ic`, `decile_spread`, etc. kept their names).

- [ ] **Step 4: Run tests + grep for stragglers**

Run: `.venv/bin/python -m pytest tests/ -q` — all pass.
Run: `grep -rn "N_PICKS\|models.champion\|models.challenger\|champion_picks\|challenger_picks\|kmeans_top10\|train_champion" src/ scripts/ tests/` — no hits.

- [ ] **Step 5: Commit**

```bash
git add -A src/ scripts/ tests/
git commit -m "refactor(models): retire champion/challenger identities for kmeans/lgbm must-buy models"
```

---

### Task 7: Migration script for stored artifacts

**Files:**
- Create: `scripts/migrate_model_names.py`
- Test: `tests/test_migrate.py`
- Modify (by running it): `models/registry.json`, `data/outputs/holdings_*.json`, `data/outputs/leaderboard_*.json`, `data/outputs/live_track.parquet`

**Interfaces:**
- Consumes: `io_utils.read_json/atomic_write_json/atomic_write_parquet`, `config` paths.
- Produces: `RENAME = {"champion": "lgbm", "challenger": "kmeans"}`; `migrate_registry(path) -> bool`, `migrate_output_files(out_dir) -> list[str]`, `migrate_live_track(path) -> bool` — all idempotent (second run returns False/[]). Registry `model_id`/`artifact` filenames stay as historical labels; only `type` is remapped.

- [ ] **Step 1: Write the failing tests** — create `tests/test_migrate.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_migrate.py -q`
Expected: FAIL — script file missing.

- [ ] **Step 3: Implement** — create `scripts/migrate_model_names.py`:

```python
"""One-time, idempotent rename of stored model identities.

champion -> lgbm    (what it is: a LightGBM walk-forward ranker)
challenger -> kmeans (what it is: a K-means momentum-cluster picker)

The champion/challenger *roles* now live in config.CHAMPION. Registry
model_id and artifact filenames are historical labels and stay untouched.
"""
import sys
from pathlib import Path

import pandas as pd

from beat_snp500 import config
from beat_snp500.io_utils import atomic_write_json, atomic_write_parquet, read_json

RENAME = {"champion": "lgbm", "challenger": "kmeans"}


def migrate_registry(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    reg = read_json(path)
    changed = False
    for entry in reg:
        if entry["type"] in RENAME:
            entry["type"] = RENAME[entry["type"]]
            changed = True
    if changed:
        atomic_write_json(reg, path)
    return changed


def migrate_output_files(out_dir: Path) -> list[str]:
    moved = []
    for prefix in ("holdings", "leaderboard"):
        for old, new in RENAME.items():
            src = Path(out_dir) / f"{prefix}_{old}.json"
            dst = Path(out_dir) / f"{prefix}_{new}.json"
            if src.exists() and not dst.exists():
                src.rename(dst)
                moved.append(dst.name)
    return moved


def migrate_live_track(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    track = pd.read_parquet(path)
    if not track["model"].isin(RENAME).any():
        return False
    track["model"] = track["model"].replace(RENAME)
    atomic_write_parquet(track, path)
    return True


def main() -> int:
    print("registry changed:", migrate_registry(config.REGISTRY_JSON))
    print("outputs renamed:", migrate_output_files(config.OUTPUTS_DIR))
    print("live_track changed:",
          migrate_live_track(config.OUTPUTS_DIR / "live_track.parquet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests, then run the migration for real**

Run: `.venv/bin/python -m pytest tests/test_migrate.py -q` — pass.
Run: `.venv/bin/python scripts/migrate_model_names.py`
Expected output: `registry changed: True`, `outputs renamed: ['holdings_lgbm.json', 'leaderboard_lgbm.json', 'holdings_kmeans.json', 'leaderboard_kmeans.json']` (order may vary), `live_track changed: True`.
Verify: `.venv/bin/python -c "import json; print({e['type'] for e in json.load(open('models/registry.json'))})"` → `{'lgbm'}`.

- [ ] **Step 5: Commit (code + migrated data)**

```bash
git add scripts/migrate_model_names.py tests/test_migrate.py models/registry.json data/outputs/
git commit -m "data: migrate stored artifacts to kmeans/lgbm identities"
```

---

### Task 8: Dashboard PM view + docs copy

**Files:**
- Modify: `app/streamlit_app.py`, `README.md`
- Test: `tests/test_app_smoke.py` (unchanged — just must stay green)

**Interfaces:**
- Consumes: `config.CHAMPION`, leaderboard payload from Task 6, series names from Task 5, migrated files from Task 7.
- Produces: no code interfaces; UI only.

- [ ] **Step 1: Rewire identities.** In `app/streamlit_app.py`:

Add after the existing imports (the app deliberately avoids the editable install, so put `src` on the path explicitly):

```python
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beat_snp500 import config  # noqa: E402  (needs ROOT on sys.path first)
```

(remove the later duplicate `ROOT = ...` line). Re-key `COLORS` — same palette slots, new names — and add labels/roles:

```python
COLORS = {
    "kmeans": "#2a78d6",     # categorical slot 1 (blue)
    "lgbm": "#1baf7a",       # categorical slot 2 (aqua)
    "kmeans_ms": "#eda100",  # categorical slot 3 (yellow) - kmeans, max-Sharpe weighted
    "lgbm_ms": "#008300",    # categorical slot 4 (green) - lgbm, max-Sharpe weighted
    "spy": "#898781",        # muted ink (benchmark reference, not a categorical identity)
}
LABELS = {"kmeans": "K-Means momentum cluster", "lgbm": "LightGBM ranker"}
CHAMPION = config.CHAMPION
CHALLENGER = "lgbm" if CHAMPION == "kmeans" else "kmeans"
```

Update the palette comment block's series names in place (champion/challenger → kmeans/lgbm); the rationale is unchanged. Update the title caption:

```python
st.caption("This is an educational quant research project, NOT investment advice. "
           f"Champion: {LABELS[CHAMPION]}. Challenger: {LABELS[CHALLENGER]}. "
           "Each buys only its must-buy names (5-10 per month), "
           "conviction-weighted with a 20% per-stock cap.")
```

- [ ] **Step 2: PM portfolio tab.** Rename the first tab to `"Today's Portfolios"` and replace the `tab_today` block:

```python
with tab_today:
    for model in [CHAMPION, CHALLENGER]:
        data = load_json(OUT / f"leaderboard_{model}.json")
        role = "🏆 Champion" if model == CHAMPION else "Challenger"
        st.subheader(f"{LABELS[model]} — {role}")
        if not data:
            st.info("No leaderboard yet — the daily pipeline has not produced one.")
            continue
        st.caption(f"As of {data['as_of']} · signal month {data['signal_month']}"
                   + (f" · {len(data['picks'])} must-buy names"
                      if data.get("picks") else ""))
        if data.get("status") == "hold" or not data.get("picks"):
            st.info("Holding previous portfolio — fewer than 5 names cleared "
                    "the must-buy bar this month.")
            continue
        rows = pd.DataFrame(
            [{"ticker": p["ticker"], "weight": f"{p['weight']:.1%}",
              "score": round(p["score"], 3), **p["features"]}
             for p in data["picks"]])
        st.dataframe(rows, hide_index=True, use_container_width=True)
```

- [ ] **Step 3: Backtest tab tweaks.** In the `tab_bt` block: band trace name → `"random count-matched (5–95%)"`; the turnover metric → `metrics.get("turnover", {}).get(CHAMPION)` labelled `"Champion turnover/mo"`; the bootstrap caption → `f"Champion CAGR sits at the {boot['champion_cagr_percentile']:.0%} percentile of 1,000 random count-matched portfolios (random p50 CAGR {boot['cagr_p50']:.1%})."`; the IC bar colors → `COLORS["lgbm"]` and `COLORS["lgbm_ms"]` (both are lgbm diagnostics).

- [ ] **Step 4: Methodology copy.** In the `tab_method` markdown, replace the Champion/Challenger bullets and the random-benchmark line:

```markdown
- **Champion — K-Means momentum cluster:** monthly K-Means (k=4) on the feature
  cross-section; the momentum cluster is identified by centroid behaviour, and its
  members with above-universe-average composite momentum (z > 0) become must-buys —
  minimum 5 names (else hold previous portfolio), maximum 10, weighted by conviction
  with a 20% per-stock cap.
- **Challenger — LightGBM ranker:** LightGBM regressor on cross-sectional return ranks,
  walk-forward (rolling 36-month window, retrained monthly); must-buys are names scoring
  ≥ 1 standard deviation above the monthly cross-section, same 5/10 floor-cap and
  conviction weighting.
- **Benchmarks:** SPY and 1,000 random portfolios drawn from the same point-in-time
  universe, count-matched to the champion's actual monthly holdings.
```

Update `README.md`'s Champion/Challenger bullets (lines ~10-12) to the same effect. Also fix the stale hint in `app/streamlit_app.py`'s `tab_bt` info message if present (`run_backtest.py` reference is fine).

- [ ] **Step 5: Verify + commit**

Run: `.venv/bin/python -m pytest tests/test_app_smoke.py tests/ -q` — all pass.
Run: `.venv/bin/streamlit run app/streamlit_app.py --server.headless true` briefly (Ctrl-C after it serves) to confirm no import/runtime error with the real migrated data files.

```bash
git add app/streamlit_app.py README.md
git commit -m "feat(app): portfolio-manager view with % allocations and champion badge"
```

---

### Task 9: Advanced features (residual / vol-scaled / cluster-relative momentum)

**Files:**
- Create: `src/beat_snp500/features/advanced.py`
- Modify: `src/beat_snp500/features/pipeline.py`, `src/beat_snp500/models/lgbm.py` (feature_cols threading)
- Test: `tests/test_advanced_features.py`

**Interfaces:**
- Consumes: `cluster_month` (T2), `BETA_COLS`/`FACTOR_COLS` from `features/betas.py`, conftest `make_panel`.
- Produces: `vol_scaled_momentum(monthly) -> pd.Series` named `mom_vol_scaled`; `residual_momentum(monthly, factors, window=12) -> pd.Series` named `resid_mom` (sum/std of FF5 residual returns over months t-11..t-1, skipping month t); `cluster_relative_momentum(panel) -> pd.DataFrame` with columns `return_3m_cz`, `return_6m_cz`, `return_12m_cz` (within-cluster z per month). `walk_forward_scores(..., feature_cols: list[str] | None = None)` (also `_fit`, `_score_month`, `select_params`) — `None` means `config.FEATURES`. Candidate columns are **not** added to `config.FEATURES` here; promotion happens in Task 11 only if validated.

- [ ] **Step 1: Write the failing tests** — create `tests/test_advanced_features.py`:

```python
import pandas as pd
import pytest

from beat_snp500.features.advanced import (cluster_relative_momentum,
                                            residual_momentum,
                                            vol_scaled_momentum)

BETA_COLS = ["beta_mkt", "beta_smb", "beta_hml", "beta_rmw", "beta_cma"]


def _zero_factors(panel):
    f = pd.DataFrame(0.0, index=panel.index.get_level_values("date").unique(),
                     columns=["mkt_rf", "smb", "hml", "rmw", "cma", "rf"])
    f.index.name = "date"
    return f


def test_vol_scaled_momentum(make_panel):
    panel = make_panel(n_months=3, n_tickers=4)
    out = vol_scaled_momentum(panel)
    assert out.name == "mom_vol_scaled"
    row = panel.iloc[5]
    assert out.iloc[5] == pytest.approx(row["return_12m"] / row["gk_vol"])


def test_residual_momentum_zero_betas_reduces_to_return_momentum(make_panel):
    panel = make_panel(n_months=16, n_tickers=4)
    panel[BETA_COLS] = 0.0
    out = residual_momentum(panel, _zero_factors(panel))
    s = panel.xs("T00", level="ticker")["return_1m"]
    roll = s.shift(1).rolling(11)      # months t-11..t-1, skipping month t
    t = s.index[-1]
    assert out.loc[(t, "T00")] == pytest.approx((roll.sum() / roll.std()).loc[t])


def test_residual_momentum_no_lookahead(make_panel):
    panel = make_panel(n_months=16, n_tickers=4)
    panel[BETA_COLS] = 0.0
    factors = _zero_factors(panel)
    base = residual_momentum(panel, factors)
    bumped_panel = panel.copy()
    last = panel.index.get_level_values("date").max()
    bumped_panel.loc[bumped_panel.index.get_level_values("date") == last,
                     "return_1m"] = 9.9
    bumped = residual_momentum(bumped_panel, factors)
    keep = base.index.get_level_values("date") < last
    pd.testing.assert_series_equal(base[keep], bumped[keep])


def test_cluster_relative_momentum_zero_mean_within_month(make_panel):
    panel = make_panel(n_months=2, n_tickers=30)
    out = cluster_relative_momentum(panel)
    assert list(out.columns) == ["return_3m_cz", "return_6m_cz", "return_12m_cz"]
    assert set(out.index.get_level_values("date").unique()) == \
        set(panel.index.get_level_values("date").unique())
    # each cluster is zero-mean, so each month's aggregate mean is ~0
    assert (out.groupby(level="date").mean().abs() < 1e-9).all().all()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_advanced_features.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `src/beat_snp500/features/advanced.py`:

```python
import numpy as np
import pandas as pd

from beat_snp500.features.betas import BETA_COLS, FACTOR_COLS
from beat_snp500.models.kmeans import cluster_month

CLUSTER_REL_COLS = ["return_3m_cz", "return_6m_cz", "return_12m_cz"]


def vol_scaled_momentum(monthly: pd.DataFrame) -> pd.Series:
    return (monthly["return_12m"]
            / monthly["gk_vol"].replace(0.0, np.nan)).rename("mom_vol_scaled")


def residual_momentum(monthly: pd.DataFrame, factors: pd.DataFrame,
                      window: int = 12) -> pd.Series:
    """Blitz/Huij/Martens residual momentum: FF5 residual returns summed over
    months t-(window-1)..t-1 (skip month t, the reversal month), scaled by
    their standard deviation. Uses the lagged rolling betas already in the
    panel; strictly backward-looking."""
    df = monthly[["return_1m", *BETA_COLS]].join(
        factors[FACTOR_COLS + ["rf"]], on="date")
    explained = sum(df[b] * df[f] for b, f in zip(BETA_COLS, FACTOR_COLS))
    resid = df["return_1m"] - df["rf"] - explained

    def per(s: pd.Series) -> pd.Series:
        roll = s.shift(1).rolling(window - 1)
        return roll.sum() / roll.std()

    return (resid.groupby(level="ticker", group_keys=False).apply(per)
            .sort_index().rename("resid_mom"))


def cluster_relative_momentum(panel: pd.DataFrame) -> pd.DataFrame:
    """Momentum z-scored against K-means cluster peers, per month — the
    within-cluster ranking effect documented in improvement/improve_v1.md."""
    def z(g: pd.Series) -> pd.Series:
        sd = g.std(ddof=0)
        return (g - g.mean()) / sd if sd else g * 0.0

    parts = []
    for t, month_df in panel.groupby(level="date"):
        month = month_df.droplevel("date")
        Xz, labels = cluster_month(month)
        if Xz.empty:
            continue
        rel = (Xz[["return_3m", "return_6m", "return_12m"]]
               .groupby(labels).transform(z))
        rel.columns = CLUSTER_REL_COLS
        rel.index = pd.MultiIndex.from_product(
            [[t], rel.index], names=["date", "ticker"])
        parts.append(rel)
    return (pd.concat(parts).sort_index() if parts
            else pd.DataFrame(columns=CLUSTER_REL_COLS))
```

In `src/beat_snp500/features/pipeline.py`, wire them in — time-series features before the membership/liquidity filters (they need unbroken per-ticker history), the cross-sectional one after (clusters must reflect the tradeable universe):

```python
from beat_snp500.features.advanced import (cluster_relative_momentum,
                                            residual_momentum,
                                            vol_scaled_momentum)
```

and change the tail of `build_feature_panel` to:

```python
    monthly[BETA_COLS] = (monthly[BETA_COLS]
                          .groupby(level="ticker")
                          .ffill(limit=config.BETA_FFILL_LIMIT))
    monthly["mom_vol_scaled"] = vol_scaled_momentum(monthly)
    monthly["resid_mom"] = residual_momentum(monthly, factors)
    monthly = apply_membership(monthly, membership)
    monthly = liquidity_filter(monthly, top_n=top_n)
    base_ok = monthly[config.BASE_FEATURES].notna().all(axis=1)
    monthly = monthly.join(cluster_relative_momentum(monthly[base_ok]))
    return monthly[monthly[config.FEATURES].notna().all(axis=1)]
```

In `src/beat_snp500/models/lgbm.py`, thread `feature_cols` (default `None` → `config.FEATURES`) through `_fit`, `_score_month`, `select_params`, and `walk_forward_scores` (add the parameter, replace each `config.FEATURES` usage with `cols = feature_cols or config.FEATURES`, and pass it down at each call site).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_advanced_features.py tests/ -q`
Expected: all pass — including `test_pipeline_leakage.py` (new columns are additive; `config.FEATURES` is unchanged so the final dropna filter behaves as before).

- [ ] **Step 5: Commit**

```bash
git add src/beat_snp500/features/advanced.py src/beat_snp500/features/pipeline.py src/beat_snp500/models/lgbm.py tests/test_advanced_features.py
git commit -m "feat(features): residual, vol-scaled, and cluster-relative momentum candidates"
```

---

### Task 10: K-means tuning with dev/holdout protocol

**Files:**
- Modify: `src/beat_snp500/models/kmeans.py` (parameterize), `src/beat_snp500/config.py` (add `KMEANS_MOM_MODE`, `KMEANS_SELECT_RULE`)
- Create: `scripts/tune_kmeans.py`
- Test: `tests/test_kmeans.py`

**Interfaces:**
- Consumes: cached `data/*.parquet`, `build_feature_panel`, `run_backtest`, `perf_metrics`, `config.DEV_END`.
- Produces: `kmeans_must_buys(..., mom_mode: str = config.KMEANS_MOM_MODE, select_rule: str = config.KMEANS_SELECT_RULE)` with `mom_mode ∈ {"mean_3_6_12", "12_1", "vol_scaled"}`, `select_rule ∈ {"mean", "risk_adj"}`; report `data/outputs/tuning/kmeans_tuning.json`. Adoption = editing the four config values only.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_kmeans.py`:

```python
import pytest


def test_mom_modes_and_select_rules_run():
    month, hot = make_month()
    for mom_mode in ("mean_3_6_12", "12_1", "vol_scaled"):
        for select_rule in ("mean", "risk_adj"):
            must = kmeans_must_buys(month, mom_mode=mom_mode,
                                    select_rule=select_rule)
            assert isinstance(must, dict)  # may be {} (hold) but never crash


def test_unknown_mode_raises():
    month, _ = make_month()
    with pytest.raises(ValueError):
        kmeans_must_buys(month, mom_mode="nope")
    with pytest.raises(ValueError):
        kmeans_must_buys(month, select_rule="nope")
```

(`make_month` lacks `gk_vol`/`return_1m` variation? No — it fills *all* `config.FEATURES` columns with noise, so both exist.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_kmeans.py -q`
Expected: FAIL — unexpected keyword `mom_mode`.

- [ ] **Step 3: Implement.** In `config.py` add next to `MUST_BUY_Z_KMEANS`:

```python
KMEANS_MOM_MODE = "mean_3_6_12"   # {"mean_3_6_12", "12_1", "vol_scaled"}
KMEANS_SELECT_RULE = "mean"       # {"mean", "risk_adj"}
```

In `models/kmeans.py` add `import numpy as np` and the two helpers, then use them in `kmeans_must_buys`:

```python
def _composite(Xz: pd.DataFrame, month_df: pd.DataFrame,
               mom_mode: str) -> pd.Series:
    if mom_mode == "mean_3_6_12":
        return Xz[MOM_COLS].mean(axis=1)
    if mom_mode == "12_1":       # classic momentum skipping the reversal month
        raw = (1 + month_df["return_12m"]) / (1 + month_df["return_1m"]) - 1
    elif mom_mode == "vol_scaled":
        raw = month_df["return_12m"] / month_df["gk_vol"].replace(0.0, np.nan)
    else:
        raise ValueError(f"unknown mom_mode: {mom_mode}")
    raw = raw.loc[Xz.index]
    sd = raw.std(ddof=0)
    return ((raw - raw.mean()) / sd).fillna(0.0) if sd else raw * 0.0


def _best_cluster(composite: pd.Series, labels: pd.Series, select_rule: str):
    g = composite.groupby(labels)
    if select_rule == "mean":
        return g.mean().idxmax()
    if select_rule == "risk_adj":
        return (g.mean() / g.std(ddof=0).replace(0.0, np.nan)).idxmax()
    raise ValueError(f"unknown select_rule: {select_rule}")
```

`kmeans_must_buys` gains `mom_mode: str = config.KMEANS_MOM_MODE, select_rule: str = config.KMEANS_SELECT_RULE` and its body swaps the composite/best-cluster lines for `composite = _composite(Xz, month_df.loc[Xz.index], mom_mode)` and `best_cluster = _best_cluster(composite, labels, select_rule)`.

Create `scripts/tune_kmeans.py`:

```python
"""Grid-search the K-means picker on the development period (months <=
config.DEV_END), report ranked results, and confirm the winner ONCE on the
holdout (months > DEV_END). Spec: docs/superpowers/specs/
2026-07-10-must-buy-selection-design.md §4a. Run time: several minutes.

Usage:
    .venv/bin/python scripts/tune_kmeans.py            # dev-period grid
    .venv/bin/python scripts/tune_kmeans.py --holdout  # winner vs current, once
"""
import argparse

import pandas as pd

from beat_snp500 import config
from beat_snp500.backtest.engine import run_backtest
from beat_snp500.backtest.metrics import perf_metrics
from beat_snp500.data.factors import load_ff5
from beat_snp500.data.prices import close_matrix
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.io_utils import atomic_write_json, read_json
from beat_snp500.models.kmeans import kmeans_picks

GRID = [{"k": k, "mom_mode": m, "select_rule": r, "threshold": z}
        for k in (3, 4, 5, 6)
        for m in ("mean_3_6_12", "12_1", "vol_scaled")
        for r in ("mean", "risk_adj")
        for z in (0.0, 0.25, 0.5)]
CURRENT = {"k": config.K_CLUSTERS, "mom_mode": config.KMEANS_MOM_MODE,
           "select_rule": config.KMEANS_SELECT_RULE,
           "threshold": config.MUST_BUY_Z_KMEANS}
REPORT = config.OUTPUTS_DIR / "tuning" / "kmeans_tuning.json"


def sharpe_for(cfg: dict, panel: pd.DataFrame, close: pd.DataFrame,
               rf_annual: float) -> float:
    picks = kmeans_picks(panel, **cfg)
    res = run_backtest(picks, close)
    if res.daily_returns.empty:
        return float("nan")
    return float(perf_metrics(res.daily_returns, rf_annual=rf_annual)["sharpe"])


def load_slices():
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    as_of = pd.Timestamp.today().normalize()
    prices = prices[prices["date"] < as_of.replace(day=1)]
    close = close_matrix(prices)
    panel = build_feature_panel(prices, membership, factors)
    dev_end = pd.Timestamp(config.DEV_END)
    dates = panel.index.get_level_values("date")
    dev = (panel[dates <= dev_end],
           close[close.index <= dev_end + pd.offsets.MonthEnd(1)],
           factors[factors.index <= dev_end])
    hold = (panel[dates > dev_end], close[close.index > dev_end],
            factors[factors.index > dev_end])
    return dev, hold


def rf_annual(factors: pd.DataFrame) -> float:
    return float((1 + factors["rf"]).prod() ** (12 / len(factors)) - 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true",
                    help="one-shot confirmation of the dev winner vs CURRENT")
    args = ap.parse_args()
    (dev_panel, dev_close, dev_f), (h_panel, h_close, h_f) = load_slices()

    if not args.holdout:
        rf = rf_annual(dev_f)
        rows = []
        for i, cfg in enumerate(GRID):
            rows.append({**cfg, "dev_sharpe": sharpe_for(cfg, dev_panel,
                                                         dev_close, rf)})
            print(f"[{i + 1}/{len(GRID)}]", rows[-1])
        rows.sort(key=lambda r: (r["dev_sharpe"] != r["dev_sharpe"],
                                 -r["dev_sharpe"]))  # NaNs last
        atomic_write_json({"current": CURRENT, "grid": rows}, REPORT)
        print("\ntop 5 by dev Sharpe:")
        for r in rows[:5]:
            print(" ", r)
        print("current config dev Sharpe:",
              sharpe_for(CURRENT, dev_panel, dev_close, rf))
        print(f"\nNext: review {REPORT}, then run --holdout ONCE.")
        return 0

    report = read_json(REPORT)
    winner = {k: report["grid"][0][k]
              for k in ("k", "mom_mode", "select_rule", "threshold")}
    rf = rf_annual(h_f)
    w = sharpe_for(winner, h_panel, h_close, rf)
    c = sharpe_for(CURRENT, h_panel, h_close, rf)
    print("holdout Sharpe — winner:", w, winner)
    print("holdout Sharpe — current:", c, CURRENT)
    print("verdict:", "ADOPT winner" if w >= c else "KEEP current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests, then the tuning protocol**

Run: `.venv/bin/python -m pytest tests/test_kmeans.py tests/ -q` — all pass.
Run: `.venv/bin/python scripts/tune_kmeans.py` (several minutes; writes the report).
Run: `.venv/bin/python scripts/tune_kmeans.py --holdout` — exactly once.
If verdict is **ADOPT**: set `K_CLUSTERS`, `KMEANS_MOM_MODE`, `KMEANS_SELECT_RULE`, `MUST_BUY_Z_KMEANS` in `config.py` to the winner's values and rerun `.venv/bin/python -m pytest tests/ -q` (seeded k-means tests may need their fixture sizes nudged if k changed — keep assertions identical). If **KEEP**: change nothing.

- [ ] **Step 5: Commit (code, report, and any adopted config)**

```bash
git add src/beat_snp500/models/kmeans.py src/beat_snp500/config.py scripts/tune_kmeans.py tests/test_kmeans.py data/outputs/tuning/
git commit -m "feat(kmeans): tunable momentum/cluster-selection modes + dev/holdout tuning report"
```

---

### Task 11: Feature evaluation gate

**Files:**
- Create: `scripts/evaluate_features.py`
- Modify (only on ADOPT): `src/beat_snp500/config.py` (`FEATURES += promoted columns`)

**Interfaces:**
- Consumes: `walk_forward_scores(..., feature_cols=...)` (T9), `spearman_ic` (T3), `config.BASE_FEATURES/DEV_END`.
- Produces: `data/outputs/tuning/feature_eval.json`; possibly a longer `config.FEATURES`.

- [ ] **Step 1: Create** `scripts/evaluate_features.py`:

```python
"""Dev/holdout gate for candidate features (spec §4b): a candidate set is
promoted into config.FEATURES only if it improves LightGBM's dev-period
walk-forward IC and does not degrade on the single holdout pass.

Usage:
    .venv/bin/python scripts/evaluate_features.py                 # dev table
    .venv/bin/python scripts/evaluate_features.py --holdout NAME  # once
"""
import argparse

import pandas as pd

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.io_utils import atomic_write_json, read_json
from beat_snp500.models.lgbm import spearman_ic, walk_forward_scores

CANDIDATE_SETS = {
    "baseline": [],
    "+mom_vol_scaled": ["mom_vol_scaled"],
    "+resid_mom": ["resid_mom"],
    "+cluster_rel": ["return_3m_cz", "return_6m_cz", "return_12m_cz"],
    "+all": ["mom_vol_scaled", "resid_mom",
             "return_3m_cz", "return_6m_cz", "return_12m_cz"],
}
REPORT = config.OUTPUTS_DIR / "tuning" / "feature_eval.json"


def ic_stats(panel: pd.DataFrame, extra: list[str], dev: bool) -> dict:
    cols = config.BASE_FEATURES + extra
    ok = panel[cols].notna().all(axis=1)
    scores = walk_forward_scores(panel[ok], feature_cols=cols)
    ic = spearman_ic(scores, panel["fwd_return_1m"])
    cut = pd.Timestamp(config.DEV_END)
    ic = ic[ic.index <= cut] if dev else ic[ic.index > cut]
    return {"mean_ic": float(ic.mean()),
            "ic_ir": float(ic.mean() / ic.std()) if ic.std() else float("nan"),
            "n_months": int(len(ic))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", metavar="NAME",
                    help="candidate-set name for the single holdout pass")
    args = ap.parse_args()
    prices = pd.read_parquet(config.PRICES_PARQUET)
    membership = pd.read_parquet(config.MEMBERSHIP_PARQUET)
    factors = load_ff5(config.FACTORS_PARQUET)
    as_of = pd.Timestamp.today().normalize()
    panel = build_feature_panel(prices[prices["date"] < as_of.replace(day=1)],
                                membership, factors)

    if args.holdout:
        for name in ("baseline", args.holdout):
            print(name, ic_stats(panel, CANDIDATE_SETS[name], dev=False))
        print("ADOPT only if the candidate beats baseline here too.")
        return 0

    rows = {name: ic_stats(panel, extra, dev=True)
            for name, extra in CANDIDATE_SETS.items()}
    for name, r in rows.items():
        print(f"{name:18s} mean IC {r['mean_ic']:+.4f}  IR {r['ic_ir']:+.2f}")
    atomic_write_json(rows, REPORT)
    print(f"\nNext: pick the best non-baseline set, run --holdout <name> ONCE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the dev table**

Run: `.venv/bin/python scripts/evaluate_features.py` (walk-forward × 5 sets — allow ~10–20 minutes).

- [ ] **Step 3: One holdout pass.** If a candidate set clearly beats baseline mean IC on dev (a lift that survives eyeballing the IR, not a fourth-decimal tie), run `--holdout <name>` once. **ADOPT** (append its columns to `config.FEATURES` in `config.py`, with a comment citing the report) only if it also beats baseline on holdout; otherwise **KEEP** and record that.

- [ ] **Step 4: Re-run the suite** (promotion changes `config.FEATURES`, which widens the panel dropna filter and lgbm inputs):

Run: `.venv/bin/python -m pytest tests/ -q` — all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate_features.py data/outputs/tuning/ src/beat_snp500/config.py
git commit -m "feat(features): dev/holdout feature-evaluation gate + verdicts"
```

---

### Task 12: Regenerate artifacts + results write-up + final verification

**Files:**
- Modify (by running scripts): `data/outputs/backtest/*`
- Create: `improvement/improve_v2.md`

- [ ] **Step 1: Regenerate the backtest**

Run: `.venv/bin/python scripts/run_backtest.py`
Expected: prints CAGR/Sharpe/MaxDD lines for `kmeans`, `lgbm`, `kmeans_ms`, `lgbm_ms`, `spy` — no champion/challenger keys anywhere.

- [ ] **Step 2: Write `improvement/improve_v2.md`** in the same plain-language style as `improve_v1.md`, with these sections, every number copied from the artifacts just produced (no estimates):

1. **What changed** — must-buy selection (5–10 names, hold below 5), conviction weights with 20% cap, K-means re-rated champion (`config.CHAMPION`), models renamed to what they are.
2. **New scoreboard** — table of yearly-average return / Sharpe / max drawdown for kmeans, lgbm, SPY from `data/outputs/backtest/metrics.json`, side by side with the fixed-top-10 numbers from the `improve_v1.md` table for context.
3. **Tuning verdict** — winning config + dev/holdout Sharpe from `data/outputs/tuning/kmeans_tuning.json` and the `--holdout` output captured in Task 10; state ADOPT/KEEP and why.
4. **Feature verdict** — dev/holdout IC table from `data/outputs/tuning/feature_eval.json`; state which (if any) features were promoted.
5. **Honest caveats** — the threshold/guardrail values were chosen by design, not fitted; variable counts change turnover; results still carry the survivorship caveat from v1.

- [ ] **Step 3: Full verification**

Run: `.venv/bin/python -m pytest tests/ -q` — all pass.
Run: `grep -rn "champion\|challenger" src/ scripts/ app/ --include="*.py" | grep -v "config.CHAMPION\|CHAMPION\|Challenger\b"` — remaining hits must be role-pointer usages or prose, never identity keys.
Run: `.venv/bin/streamlit run app/streamlit_app.py --server.headless true` — loads with the regenerated data (Ctrl-C after confirming).

- [ ] **Step 4: Commit**

```bash
git add data/outputs/ improvement/improve_v2.md
git commit -m "data: regenerate backtest under must-buy selection; add round-2 write-up"
```

---

## Self-Review Notes (already applied)

- Spec §1 → T2/T3; §2 → T1 (+bootstrap fairness T4); §3 → T5/T6/T7/T8; §4a → T10; §4b → T9/T11; §5 → T8/T12; hygiene item is a verified no-op (duplicates only in `.venv`).
- `N_PICKS` removal deferred to T6 (bootstrap keeps a literal default 10; daily/models drop it there).
- Old and new model modules coexist T2–T5 so every commit stays green; T6 deletes the old pair.
- Contingencies for seeded-randomness flakes are stated inside the affected steps (fixture sizes only, assertions never weaken).
