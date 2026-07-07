"""Daily pipeline entrypoint (used by the GitHub Actions cron)."""
import argparse

from beat_snp500.jobs.daily import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--force-rebalance", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(as_of=args.as_of, force_rebalance=args.force_rebalance))
