from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class ValidationReport:
    ok: bool
    issues: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def validate_prices(prices: pd.DataFrame, expected_tickers, as_of: pd.Timestamp,
                    max_stale_days: int = 5, max_missing_frac: float = 0.05,
                    max_nan_frac: float = 0.02, max_abs_return: float = 1.5,
                    return_window_days: int = 10) -> ValidationReport:
    if prices.empty:
        return ValidationReport(ok=False, issues=["no price rows at all"],
                                stats={"n_tickers": 0, "last_date": None})
    issues = []
    global_max_date = prices["date"].max()
    want = set(expected_tickers)
    member_last = prices[prices["ticker"].isin(want)].groupby("ticker")["date"].max()
    # staleness is keyed off expected members' own clock so a junk/delisted
    # ticker printing fresh (bogus) dates can't spoof the gate.
    last = member_last.median() if (want and not member_last.empty) else global_max_date
    stale_days = int(np.busday_count(last.date(), as_of.date()))
    if stale_days > max_stale_days:
        issues.append(f"stale: last bar {last.date()} is {stale_days} business days before {as_of.date()}")

    have = set(prices["ticker"].unique())
    missing = sorted(want - have)
    if want and len(missing) / len(want) > max_missing_frac:
        issues.append(f"missing {len(missing)}/{len(want)} expected tickers")

    nan_frac = float(prices["close"].isna().mean())
    if nan_frac > max_nan_frac:
        issues.append(f"NaN close fraction {nan_frac:.3f} exceeds {max_nan_frac}")

    # price-plausibility screen: flag implausible close-to-close moves among
    # expected members in the most recent window only (old, already-adjusted
    # splits or one-off data errors outside the window don't retrigger it).
    member_prices = prices[prices["ticker"].isin(want)] if want else prices.iloc[0:0]
    if not member_prices.empty:
        member_prices = member_prices.sort_values(["ticker", "date"]).copy()
        member_prices["_ret"] = member_prices.groupby("ticker")["close"].pct_change(fill_method=None)
        recent_dates = sorted(member_prices["date"].unique())[-return_window_days:]
        recent = member_prices[member_prices["date"].isin(recent_dates)]
        bad = recent[recent["_ret"].abs() > max_abs_return]
        for _, row in bad.head(5).iterrows():
            issues.append(f"implausible return: {row['ticker']} {row['_ret']:+.0%} "
                          f"on {pd.Timestamp(row['date']).date()}")

    stats = {"last_date": str(last.date()), "global_max_date": str(global_max_date.date()),
             "n_tickers": len(have), "n_missing": len(missing), "missing_sample": missing[:20],
             "nan_frac_close": nan_frac, "stale_business_days": stale_days}
    return ValidationReport(ok=not issues, issues=issues, stats=stats)
