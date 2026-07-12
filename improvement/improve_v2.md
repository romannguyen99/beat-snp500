# Round 2: variable-count "must-buy" selection (and a champion re-rating)

Round 1 fixed a bug that made the simple clustering model occasionally bet the
whole portfolio on one or two stocks. That fix alone made it beat the S&P
500 on both return and risk-adjusted return for the first time. Round 2 asks
a follow-up question: both models were still hard-coded to buy exactly 10
stocks every month, split evenly. Is "always exactly 10, split evenly" the
right rule, or is it just the rule we happened to start with?

This round replaces that fixed rule with a **must-buy** rule — buy however
many stocks currently look attractive, within sensible bounds, and size each
position by how attractive it looks rather than splitting evenly. Along the
way we also renamed the two models to what they actually are, re-tuned one
of them properly (dev set / untouched holdout set, never peeking), and
tested one new candidate feature the same disciplined way.

## 1. What changed

- **Must-buy selection instead of a fixed top-10.** Each month, a model now
  buys whichever stocks clear its signal threshold — **as few as 5 or as
  many as 10** names, not always exactly 10. If fewer than 5 names clear the
  bar, the program doesn't force a trade: it **holds whatever it already
  owned** and tries again next month. This is the same "don't force a bad
  bet" idea from Round 1's bug fix, now applied as a general floor for both
  models rather than a one-off patch. (`config.MIN_PICKS = 5`,
  `config.MAX_PICKS = 10`.) In practice, over K-means's full 197-month
  history in this backtest (2010-02 through 2026-06, `picks.json`), it held
  its prior portfolio instead of trading in 32 of them (~16%), and picked
  anywhere from 5 to 10 names (average 9.6) the rest of the time. LightGBM's
  own scored history is shorter — its 36-month training window has to fill
  first — covering 161 months from 2013-02 through 2026-06. Over that same
  161-month window shared by both models, K-means held in 16 months (~10%),
  while LightGBM held in **zero**: its ≥1-standard-deviation threshold
  cleared exactly 10 names in every single one of those 161 months. The
  must-buy floor never actually bound for LightGBM in this backtest — in
  practice it stayed a top-10 portfolio with conviction weights, not the
  variable-count strategy this section describes for K-means.
- **Conviction weighting with a 20% cap, instead of an equal split.** Money
  is now allocated in proportion to how strong each stock's signal is
  (`portfolio/weights.conviction_weights`), but **no single stock can exceed
  20% of the portfolio** (`config.WEIGHT_CAP = 0.20`) — if the proportional
  weight would push a name over the cap, the excess is redistributed to the
  other names. The 20% cap and the 5-name floor are chosen together
  deliberately: 5 names × 20% = 100%, so the portfolio can always be fully
  invested even in the smallest allowed must-buy set.
- **K-means re-rated champion; both models renamed to what they are.** The
  original names — "Champion" for the LightGBM model and "Challenger" for
  the K-means model — described an assumption (the fancier ML model should
  win), not a measurement. Round 1 showed that assumption was backwards. This
  round finishes the cleanup: the models are now identified everywhere in
  code and data by what they literally are, `kmeans` and `lgbm`, and
  "champion" / "challenger" survive only as **role labels** pointing at
  whichever series is currently winning (`config.CHAMPION = "kmeans"`) —
  re-assignable, not permanent titles baked into either algorithm's identity.

## 2. New scoreboard

All numbers below for this round come straight from
`data/outputs/backtest/metrics.json`, produced by
`PYTHONPATH=src .venv/bin/python scripts/run_backtest.py` over the same
2013–2026 history used throughout this project. "Yearly average return" is
the annualized (CAGR) figure, the same measure `improve_v1.md` used under
that label.

| Stage | Yearly average return | Return per unit of risk (Sharpe) | Worst peak-to-valley loss (MaxDD) |
|---|---|---|---|
| K-means (then "Challenger"), fixed-top-10, **before** the Round-1 bug fix | 21.9% | 0.76 | −57% |
| K-means (then "Challenger"), fixed-top-10, **after** the Round-1 bug fix | 25.6% | 0.92 | −38% |
| **kmeans**, must-buy selection (this round) | **30.00%** | **1.00** | **−38.77%** |
| **lgbm**, must-buy selection (this round) | 8.95% | 0.41 | −51.75% |
| S&P 500 (SPY) — unchanged benchmark | 14.70% (v1 reported 14.7%) | 0.82 | −33.72% (v1 reported −34%) |

*(LightGBM's fixed-top-10 numbers from before this round aren't in this
table because `improve_v1.md` never tabulated them — that report's table
tracked only the K-means bug fix. `improve_v1.md`'s prose describes the
LightGBM side as "still underperforming" over the same period, without a
matching number, so it isn't repeated here as a number we didn't actually
measure in that form.)*

K-means, now the champion, improved again on every measure after moving to
must-buy selection: return per unit of risk went from 0.92 to 1.00, and
return climbed from 25.6% to 30.00%, while the worst drawdown stayed close
to flat (−38% to −38.77%). LightGBM, the challenger, still trails both
K-means and the S&P 500 on a risk-adjusted basis (Sharpe 0.41 vs. 0.82 for
SPY) — must-buy selection didn't fix the underlying problem identified in
Round 1 (its 14 features carry very little one-month-ahead signal for this
group of stocks).

For reference, `data/outputs/backtest/metrics.json` also reports
`kmeans_ms` (26.71% / 0.93 / −40.45%) and `lgbm_ms` (11.37% / 0.51 /
−50.22%) — the same monthly must-buy picks, but sized with a mean-variance
"max Sharpe" optimizer instead of conviction weighting, as a sanity check
that the results aren't an artifact of the weighting scheme. They tell a
similar story (K-means still clearly ahead of LightGBM) and aren't the
project's primary weighting scheme, so they're noted here rather than
carried through the rest of this write-up.

**Is 30% CAGR just luck from picking any 9-or-10 random large-cap names?**
The backtest also draws 1,000 random, count-matched portfolios from the same
point-in-time universe each month (`data/outputs/backtest/bootstrap_summary.json`).
Their simulated CAGRs range from 5.57% (5th percentile) to 14.07% (95th
percentile), with a median of 9.75%. K-means's actual 30.00% CAGR sits at
the **100th percentile** of that distribution — every one of the 1,000
random draws did worse. That doesn't prove the signal will keep working, but
it does rule out "this is just what happens if you own any handful of large,
liquid S&P 500 names."

## 3. Tuning verdict: KEEP the current K-means config

Task 8 ran a 72-config grid search over K-means's settings — cluster count
`k ∈ {3,4,5,6}` × momentum definition `mom_mode ∈ {mean_3_6_12, 12_1,
vol_scaled}` × cluster-selection rule `select_rule ∈ {mean, risk_adj}` ×
must-buy z-threshold `∈ {0.0, 0.25, 0.5}` — scored on the **dev** slice
only (months ≤ 2019-12-31), per `data/outputs/tuning/kmeans_tuning.json`.

The dev-period winner was `{k: 3, mom_mode: vol_scaled, select_rule:
risk_adj, threshold: 0.5}` with a dev Sharpe of **0.8754**, clearly ahead of
the then-current config `{k: 4, mom_mode: mean_3_6_12, select_rule: mean,
threshold: 0.0}`, whose dev Sharpe was **0.7463** (both numbers straight
from `kmeans_tuning.json`'s 72-row grid).

Per the project's protocol, a dev-set winner only gets adopted if it also
beats the current config on a **single, untouched holdout pass** (months
after 2019-12-31) — checked exactly once, no second looks. That holdout run
(`.superpowers/sdd/task-8-report.md`, verbatim):

```
holdout Sharpe — winner: 1.0039488457250687 {'k': 3, 'mom_mode': 'vol_scaled', 'select_rule': 'risk_adj', 'threshold': 0.5}
holdout Sharpe — current: 1.17908295728248 {'k': 4, 'mom_mode': 'mean_3_6_12', 'select_rule': 'mean', 'threshold': 0.0}
verdict: KEEP current
```

The grid's apparent winner (holdout Sharpe 1.00) actually did **worse** on
new data than the config we already had (holdout Sharpe 1.18). **Verdict:
KEEP** — no config values changed
(`K_CLUSTERS=4`, `KMEANS_MOM_MODE="mean_3_6_12"`, `KMEANS_SELECT_RULE="mean"`,
`MUST_BUY_Z_KMEANS=0.0` are exactly what they were before Task 8).

This is worth stating plainly: **KEEP is the protocol working, not a wasted
search.** The dev grid's best row looked good specifically because it fit
that stretch of history well — a classic overfitting trap. Checking it
against data the search never saw is what caught that before it could ship,
and it's exactly the discipline that would otherwise have let a
worse-out-of-sample config slip into production on the strength of a
dev-only number.

## 4. Feature verdict: ADOPT `mom_vol_scaled`

Task 9 tested three candidate features — a volatility-scaled momentum
measure, a residualized (market-neutralized) momentum measure, and a
cluster-relative momentum measure — by adding each to the existing 14-feature
set and measuring monthly rank information coefficient (IC: how well a
feature's cross-sectional ranking correlates with next month's actual
return ranking) on the dev slice. From
`data/outputs/tuning/feature_eval.json`:

| Feature set | Dev mean IC | Dev IC information ratio | Months |
|---|---|---|---|
| baseline (14 features) | −0.0031 | −0.03 | 83 |
| + `mom_vol_scaled` | **+0.0015** | **+0.02** | 83 |
| + `resid_mom` | +0.0014 | +0.01 | 72 |
| + `cluster_rel` | −0.0044 | −0.05 | 83 |
| + all three | −0.0042 | −0.05 | 72 |

`mom_vol_scaled` was the only candidate whose dev IC flipped from negative
to positive, so it was the one candidate carried into the single holdout
check (`/tmp/feature_eval_holdout.log`, months after 2019-12-31, run once):

```
baseline {'mean_ic': -0.02404853328334938, 'ic_ir': -0.18261756018333947, 'n_months': 77}
+mom_vol_scaled {'mean_ic': -0.012719681247712677, 'ic_ir': -0.1007375903932201, 'n_months': 77}
```

**Both the baseline and the `mom_vol_scaled` holdout IC are negative** —
this feature is not a positive-IC signal on its own, and it would be
dishonest to claim otherwise. What it did do is **beat the baseline on both
the dev slice and the untouched holdout slice** (dev: −0.0031 → +0.0015;
holdout: −0.0240 → −0.0127) — a smaller negative is still an improvement in
relative rank quality, and it cleared the bar on data the feature search
never saw. Per protocol, that's an ADOPT: `mom_vol_scaled` was appended to
`config.FEATURES` (commit `62b3c45`). `resid_mom` and `cluster_rel` were not
promoted — `resid_mom`'s dev IC was close behind `mom_vol_scaled`'s but
didn't clear it, `cluster_rel`'s and the combined-three dev ICs were
negative, and per the same one-look-only holdout discipline, only the
strongest dev candidate was spent on a holdout check.

## 5. Honest caveats

- **The 5–10 floor/cap and the 20% weight cap were chosen by design, not
  fitted.** `MIN_PICKS=5`, `MAX_PICKS=10`, and `WEIGHT_CAP=0.20` come from a
  simple, stated rule (5 × 20% = 100%, so the smallest allowed portfolio can
  still be fully invested) rather than from a grid search over those
  numbers. Unlike the K-means tuning knobs in §3, they were never run
  through the dev/holdout protocol, so treat them as reasonable guardrails,
  not as optimized parameters.
- **Variable pick counts change how much the portfolio trades.** Because the
  number of names (and which names) can change every month, monthly
  turnover is less predictable than the old fixed-10 rule. Average monthly
  turnover this round was 95.70% of the portfolio for `kmeans` and 171.69%
  for `lgbm` (`data/outputs/backtest/metrics.json`, `turnover`), and every
  trading-cost dollar in these results reflects that swinging, not a fixed
  amount deducted once a month.
- **The results still carry the same survivorship caveat as v1.** The
  backtest's point-in-time S&P 500 universe includes some delisted or
  acquired members with no surviving Yahoo Finance price history, so they're
  silently absent from the tradable universe in the months they'd otherwise
  have appeared. Measured directly from this run's data
  (`data/outputs/backtest/metrics.json`, `survivorship`): an average of
  13.10% of point-in-time members are missing price history in a given
  month, peaking at 23.48% in the worst month. This is the same residual
  bias the Round 1 report and the dashboard's Methodology tab describe; it
  wasn't addressed by this round's changes and doesn't disappear because the
  headline numbers improved.
- None of this is investment advice — it's a backtest over one historical
  path, with real (if disclosed) data gaps, run against a benchmark that
  itself moved a great deal over the same period.
