import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from beat_snp500 import config
from beat_snp500.data.factors import load_ff5
from beat_snp500.data.membership import refresh_membership
from beat_snp500.data.prices import close_matrix, update_price_cache
from beat_snp500.data.validation import validate_prices
from beat_snp500.features.pipeline import build_feature_panel
from beat_snp500.io_utils import atomic_write_json, atomic_write_parquet, read_json
from beat_snp500.models.champion import load_model, save_model, train_champion
from beat_snp500.models.kmeans import kmeans_must_buys
from beat_snp500.models.registry import append_entry, latest_model
from beat_snp500.portfolio.weights import conviction_weights, equal_weights

TRACK_COLS = ["date", "model", "ret", "spy_ret"]


def artifact_ref(path) -> str:
    """Repo-relative artifact reference so registry.json is portable across checkouts."""
    p = Path(path)
    return str(p.relative_to(config.ROOT)) if p.is_relative_to(config.ROOT) else str(p)


def resolve_artifact(ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else config.ROOT / p


def is_first_weekday(d: pd.Timestamp) -> bool:
    first = d.normalize().replace(day=1)
    while first.weekday() >= 5:
        first += pd.DateOffset(days=1)
    return d.normalize() == first


def build_leaderboards(panel: pd.DataFrame, booster, out_dir, as_of,
                       n_picks: int = config.N_PICKS) -> dict:
    latest = panel.index.get_level_values("date").max()
    month = panel.xs(latest, level="date")
    feats = month[config.FEATURES]
    boards = {}
    if booster is not None:
        scores = pd.Series(booster.predict(feats), index=feats.index)
        ranked = scores.nlargest(n_picks)
        atomic_write_json({
            "as_of": str(pd.Timestamp(as_of).date()),
            "signal_month": str(latest.date()),
            "picks": [
                {"ticker": t, "score": float(v),
                 "features": {k: float(feats.loc[t, k]) for k in config.FEATURES}}
                for t, v in ranked.items()
            ],
        }, Path(out_dir) / "leaderboard_champion.json")
        boards["champion"] = ranked
    sig = kmeans_must_buys(month)
    weights = conviction_weights(sig) if sig else {}
    atomic_write_json({
        "as_of": str(pd.Timestamp(as_of).date()),
        "signal_month": str(latest.date()),
        "status": "active" if weights else "hold",
        "picks": [
            {"ticker": t, "weight": float(weights[t]), "score": float(sig[t]),
             "features": {k: float(feats.loc[t, k]) for k in config.FEATURES}}
            for t in sorted(weights, key=weights.get, reverse=True)
        ],
    }, Path(out_dir) / "leaderboard_kmeans.json")
    boards["kmeans"] = weights
    return boards


def update_live_track(close: pd.DataFrame, holdings: dict, track_path, as_of) -> pd.DataFrame:
    track_path = Path(track_path)
    existing = (pd.read_parquet(track_path) if track_path.exists()
                else pd.DataFrame(columns=TRACK_COLS))
    px = close.loc[close.index <= pd.Timestamp(as_of)]
    if "SPY" not in px.columns:
        # Fail closed: without the benchmark there is no trusted trading clock
        # and no spy_ret; skip today rather than append junk-anchored rows.
        return existing
    # Anchor every row to the benchmark's own trading clock: a junk/delisted
    # ticker can keep printing stray quotes on days SPY (and every real
    # ticker) has none, e.g. a market holiday, which would otherwise survive
    # as a phantom row and corrupt the day-over-day return comparison.
    px = px.loc[px["SPY"].notna()]
    if len(px) < 2 or not holdings:
        return existing
    today = px.index[-1]
    rets = px.iloc[-1] / px.iloc[-2] - 1
    spy_ret = float(rets.get("SPY", float("nan")))
    rows = []
    for model, h in holdings.items():
        w = pd.Series(h["weights"], dtype=float)
        avail = [t for t in w.index if t in rets.index and pd.notna(rets[t])]
        if not avail:
            continue
        wa = w[avail] / w[avail].sum()
        rows.append({"date": today, "model": model,
                     "ret": float((rets[avail] * wa).sum()), "spy_ret": spy_ret})
    frames = [f for f in (existing, pd.DataFrame(rows)) if not f.empty]
    out = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=TRACK_COLS))
    out = (out.drop_duplicates(["date", "model"], keep="last")
           .sort_values(["date", "model"]).reset_index(drop=True))
    atomic_write_parquet(out, track_path)
    return out


def monthly_rebalance(panel_completed: pd.DataFrame, models_dir, out_dir, registry_path,
                      as_of, train_window: int = config.TRAIN_WINDOW_MONTHS) -> None:
    labeled = panel_completed.dropna(subset=["fwd_return_1m"])
    model, val_ic = train_champion(labeled, train_window=train_window)
    latest = panel_completed.index.get_level_values("date").max()
    model_id = f"champion_{latest:%Y%m}"
    artifact = Path(models_dir) / f"{model_id}.txt"
    save_model(model, artifact)
    append_entry(registry_path, {
        "model_id": model_id, "type": "champion",
        "trained_through": str(labeled.index.get_level_values("date").max().date()),
        "train_window_months": train_window, "ic_mean": val_ic,
        "created_at": str(pd.Timestamp(as_of).date()), "artifact": artifact_ref(artifact),
    })

    month = panel_completed.xs(latest, level="date")
    scores = pd.Series(model.predict(month[config.FEATURES]), index=month.index)
    champ = scores.nlargest(config.N_PICKS).index.tolist()
    if len(champ) == config.N_PICKS:
        atomic_write_json(
            {"signal_date": str(latest.date()),
             "generated_at": str(pd.Timestamp(as_of).date()),
             "weights": equal_weights(champ)},
            Path(out_dir) / "holdings_champion.json")
    sig = kmeans_must_buys(month)
    if sig:  # hold months keep the previous holdings_kmeans.json in force
        atomic_write_json(
            {"signal_date": str(latest.date()),
             "generated_at": str(pd.Timestamp(as_of).date()),
             "weights": conviction_weights(sig)},
            Path(out_dir) / "holdings_kmeans.json")


def run(as_of=None, force_rebalance: bool = False) -> int:
    as_of = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.today()).normalize()
    rebalance = force_rebalance or is_first_weekday(as_of)

    mem, fresh = refresh_membership(config.MEMBERSHIP_PARQUET)
    atomic_write_json({"fresh": bool(fresh), "as_of": str(as_of.date())},
                      config.OUTPUTS_DIR / "membership_status.json")

    tickers = sorted(set(mem["ticker"]).union(config.EXTRA_TICKERS))
    prices = update_price_cache(tickers, config.PRICES_PARQUET, full_refresh=rebalance)

    current = mem[mem["date"] == mem["date"].max()]["ticker"]
    report = validate_prices(prices, current, as_of)
    atomic_write_json(asdict(report), config.OUTPUTS_DIR / "validation.json")
    if not report.ok:
        print(f"validation failed: {report.issues}", file=sys.stderr)
        return 1

    factors = load_ff5(config.FACTORS_PARQUET)
    panel = build_feature_panel(prices, mem, factors)

    entry = latest_model(config.REGISTRY_JSON, "champion")
    booster = load_model(resolve_artifact(entry["artifact"])) if entry else None
    build_leaderboards(panel, booster, config.OUTPUTS_DIR, as_of)

    holdings = {}
    for name in ["champion", "kmeans"]:
        p = config.OUTPUTS_DIR / f"holdings_{name}.json"
        if p.exists():
            holdings[name] = read_json(p)
    update_live_track(close_matrix(prices), holdings,
                      config.OUTPUTS_DIR / "live_track.parquet", as_of)

    if rebalance:
        month_start = as_of.replace(day=1)
        panel_completed = build_feature_panel(
            prices[prices["date"] < month_start], mem, factors)
        monthly_rebalance(panel_completed, config.MODELS_DIR, config.OUTPUTS_DIR,
                          config.REGISTRY_JSON, as_of)
    return 0
