from pathlib import Path

import pandas as pd
import pytest

from beat_snp500 import config
from beat_snp500.io_utils import read_json
from beat_snp500.jobs.daily import (
    artifact_ref, build_leaderboards, is_first_weekday, monthly_rebalance,
    resolve_artifact, update_live_track,
)
from beat_snp500.models.lgbm import train_lgbm
from beat_snp500.models.registry import load_registry


def test_is_first_weekday():
    assert is_first_weekday(pd.Timestamp("2026-06-01"))      # Monday June 1
    assert not is_first_weekday(pd.Timestamp("2026-06-02"))
    assert is_first_weekday(pd.Timestamp("2026-08-03"))      # Aug 1-2 are weekend
    assert not is_first_weekday(pd.Timestamp("2026-08-01"))


def test_artifact_ref_roundtrip():
    inside = config.MODELS_DIR / "champion_x.txt"
    ref = artifact_ref(inside)
    assert not Path(ref).is_absolute()
    assert resolve_artifact(ref) == inside
    outside = Path("/somewhere/else/m.txt")
    assert resolve_artifact(artifact_ref(outside)) == outside


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


def test_update_live_track_idempotent(tmp_path):
    idx = pd.to_datetime(["2026-06-30", "2026-07-01"])
    close = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 90.0],
                          "SPY": [100.0, 101.0]}, index=idx)
    holdings = {"lgbm": {"weights": {"A": 0.5, "B": 0.5}}}
    p = tmp_path / "track.parquet"
    update_live_track(close, holdings, p, pd.Timestamp("2026-07-01"))
    out = update_live_track(close, holdings, p, pd.Timestamp("2026-07-01"))
    assert len(out) == 1
    assert out.iloc[0]["ret"] == pytest.approx(0.0)
    assert out.iloc[0]["spy_ret"] == pytest.approx(0.01)


def test_update_live_track_clips_to_spy_clock(tmp_path):
    # SPY has no valid bar on 2026-07-02 (e.g. a holiday); a delisted/junk
    # ticker keeps printing stray quotes past SPY's clock. The live track
    # must anchor "today" to SPY's last valid observation, not the junk row.
    idx = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02"])
    close = pd.DataFrame({
        "A": [100.0, 110.0, 115.0],
        "SPY": [100.0, 101.0, float("nan")],
        "JUNK": [100.0, 100.0, 105.0],
    }, index=idx)
    holdings = {"lgbm": {"weights": {"A": 1.0}}}
    p = tmp_path / "track.parquet"
    out = update_live_track(close, holdings, p, pd.Timestamp("2026-07-02"))
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2026-07-01")
    assert out.iloc[0]["ret"] == pytest.approx(0.10)
    assert out.iloc[0]["spy_ret"] == pytest.approx(0.01)


def test_update_live_track_skips_interior_holiday_junk_row(tmp_path):
    # A market holiday (e.g. July 3rd observed) sits between two real trading
    # days; every legitimate ticker (including SPY) is NaN that day, but a
    # delisted/junk ticker still prints a stray quote, so the day survives as
    # a row in the close matrix. The "yesterday" comparison must skip that
    # phantom row and use the last real trading day, not just positionally
    # look one row back.
    idx = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-07-02",
                          "2026-07-03", "2026-07-06"])
    close = pd.DataFrame({
        "A": [100.0, 105.0, 110.0, float("nan"), 121.0],
        "SPY": [100.0, 101.0, 102.0, float("nan"), 105.0],
        "JUNK": [100.0, 100.0, 100.0, 101.0, 102.0],
    }, index=idx)
    holdings = {"lgbm": {"weights": {"A": 1.0}}}
    p = tmp_path / "track.parquet"
    out = update_live_track(close, holdings, p, pd.Timestamp("2026-07-06"))
    assert len(out) == 1
    assert out.iloc[0]["date"] == pd.Timestamp("2026-07-06")
    assert out.iloc[0]["ret"] == pytest.approx(121.0 / 110.0 - 1)
    assert out.iloc[0]["spy_ret"] == pytest.approx(105.0 / 102.0 - 1)


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
