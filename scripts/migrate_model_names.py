"""One-time, idempotent rename of stored model identities.

champion -> lgbm    (what it is: a LightGBM walk-forward ranker)
challenger -> kmeans (what it is: a K-means momentum-cluster picker)

The champion/challenger *roles* now live in config.CHAMPION. Registry
model_id and artifact filenames are historical labels and stay untouched.

Run: PYTHONPATH=src .venv/bin/python scripts/migrate_model_names.py
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
    print("registry changed:", migrate_registry(config.MODELS_DIR / "registry.json"))
    print("outputs renamed:", migrate_output_files(config.OUTPUTS_DIR))
    print("live_track changed:",
          migrate_live_track(config.OUTPUTS_DIR / "live_track.parquet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
