"""Move the lgbm @current alias to an explicit registered version (manual
promotion or rollback). Two-writer rule (spec §5): CI owns the registry —
run `git pull` before this, and push promptly after.

Usage:
    PYTHONPATH=src .venv/bin/python scripts/promote_model.py 2
"""
import argparse

from beat_snp500 import tracking


def promote(version: int, tracker: tracking.Tracker) -> str | None:
    return tracker.set_current(version)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("version", type=int, help="registered lgbm version number")
    args = ap.parse_args()
    source = promote(args.version, tracking.Tracker("production"))
    print(f"@current -> version {args.version} ({source})")
