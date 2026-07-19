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
    entries = sorted(read_json(registry_json), key=lambda e: e["created_at"])
    for e in entries:
        tracker.register_model_version(
            artifact=e["artifact"], run_id=None,
            tags={"model_id": e["model_id"], "type": e["type"],
                  "trained_through": e["trained_through"],
                  "train_window_months": e["train_window_months"],
                  "ic_mean": e["ic_mean"], "created_at": e["created_at"],
                  "migrated_from": "models/registry.json"})
    return len(entries)


def main() -> int:
    n = migrate(config.MODELS_DIR / "registry.json", tracking.Tracker("production"))
    print(f"migrated {n} entries; @current -> version {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
