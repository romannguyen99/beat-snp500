from pathlib import Path

from beat_snp500.io_utils import atomic_write_json, read_json

REQUIRED = {"model_id", "type", "trained_through", "train_window_months",
            "ic_mean", "created_at", "artifact"}


def load_registry(path: Path) -> list[dict]:
    path = Path(path)
    return read_json(path) if path.exists() else []


def append_entry(path: Path, entry: dict) -> None:
    missing = REQUIRED - entry.keys()
    if missing:
        raise ValueError(f"registry entry missing keys: {sorted(missing)}")
    reg = load_registry(path)
    reg.append(entry)
    atomic_write_json(reg, path)


def latest_model(path: Path, type_: str) -> dict | None:
    entries = [e for e in load_registry(path) if e["type"] == type_]
    return entries[-1] if entries else None
