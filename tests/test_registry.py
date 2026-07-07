import pytest

from beat_snp500.models.registry import append_entry, latest_model, load_registry


def entry(i):
    return {"model_id": f"champion_{i}", "type": "champion",
            "trained_through": f"2026-0{i}-30", "train_window_months": 36,
            "ic_mean": 0.05, "created_at": "2026-07-01", "artifact": f"models/c{i}.txt"}


def test_append_and_load(tmp_path):
    p = tmp_path / "registry.json"
    assert load_registry(p) == []
    append_entry(p, entry(1))
    append_entry(p, entry(2))
    assert [e["model_id"] for e in load_registry(p)] == ["champion_1", "champion_2"]


def test_latest_model_by_type(tmp_path):
    p = tmp_path / "registry.json"
    append_entry(p, entry(1))
    append_entry(p, entry(2))
    assert latest_model(p, "champion")["model_id"] == "champion_2"
    assert latest_model(p, "challenger") is None


def test_missing_keys_rejected(tmp_path):
    with pytest.raises(ValueError):
        append_entry(tmp_path / "r.json", {"model_id": "x"})
