import json
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from beat_snp500 import config  # noqa: E402  (needs ROOT on sys.path first)

OUT = ROOT / "data" / "outputs"
BT = OUT / "backtest"

# Palette per the dataviz skill's reference instance (references/palette.md).
# kmeans / lgbm / kmeans_ms / lgbm_ms are four *distinct*
# strategy variants that appear together on the backtest equity-curve chart,
# so each takes its own fixed-order categorical slot (blue, aqua, yellow,
# green — slots 1-4) rather than tinting one hue per model family; a 4-hue
# check of exactly this set passes CVD separation with worst adjacent
# Delta E 24.2 (validate_palette.js). Aqua and yellow land below 3:1 contrast
# on a light surface, which the skill flags as a WARN requiring "relief" —
# both the legend and the per-model metrics table below the chart carry the
# same values, so color is never the only way to read a series. SPY is a
# read-only external benchmark, not a strategy the app produces, so it is
# rendered in muted ink with a dashed line (identity via line style, not a
# 5th competing hue) rather than spent from the categorical palette.
COLORS = {
    "kmeans": "#2a78d6",     # categorical slot 1 (blue)
    "lgbm": "#1baf7a",       # categorical slot 2 (aqua)
    "kmeans_ms": "#eda100",  # categorical slot 3 (yellow) - kmeans, max-Sharpe weighted
    "lgbm_ms": "#008300",    # categorical slot 4 (green) - lgbm, max-Sharpe weighted
    "spy": "#898781",        # muted ink (benchmark reference, not a categorical identity)
}
LABELS = {"kmeans": "K-Means momentum cluster", "lgbm": "LightGBM ranker"}
CHAMPION = config.CHAMPION
CHALLENGER = "lgbm" if CHAMPION == "kmeans" else "kmeans"
BAND_FILL = "rgba(137,135,129,0.15)"  # muted ink wash for the random-portfolio band
GRIDLINE = "#e1e0d9"


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_parquet(p: Path):
    return pd.read_parquet(p) if p.exists() else None


def _style_axes(fig, yaxis_title=None):
    fig.update_layout(
        template="plotly_white",
        margin=dict(t=30),
        hovermode="x unified",
        legend=dict(title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRIDLINE)
    fig.update_yaxes(title=yaxis_title, showgrid=True, gridcolor=GRIDLINE,
                      gridwidth=1, zeroline=False)
    return fig


st.set_page_config(page_title="beat-snp500", page_icon="📈", layout="wide")
st.title("Can you beat S&P 500?")
st.caption("This is an educational quant research project, NOT investment advice. "
           f"Champion: {LABELS[CHAMPION]}. Challenger: {LABELS[CHALLENGER]}. "
           "Each buys only its must-buy names (5-10 per month), "
           "conviction-weighted with a 20% per-stock cap.")

tab_today, tab_live, tab_bt, tab_method = st.tabs(
    ["Today's Portfolios", "Live Performance", "Backtest Report", "Methodology & Limitations"])

with tab_today:
    for model in [CHAMPION, CHALLENGER]:
        data = load_json(OUT / f"leaderboard_{model}.json")
        role = "🏆 Champion" if model == CHAMPION else "Challenger"
        st.subheader(f"{LABELS[model]} — {role}")
        if not data:
            st.info("No leaderboard yet — the daily pipeline has not produced one.")
            continue
        st.caption(f"As of {data['as_of']} · signal month {data['signal_month']}"
                   + (f" · {len(data['picks'])} must-buy names"
                      if data.get("picks") else ""))
        if data.get("status") == "hold" or not data.get("picks"):
            st.info("Holding previous portfolio — fewer than 5 names cleared "
                    "the must-buy bar this month.")
            continue
        rows = pd.DataFrame(
            [{"ticker": p["ticker"],
              "weight": f"{p['weight']:.1%}" if "weight" in p else "—",
              "score": round(p["score"], 3), **p["features"]}
             for p in data["picks"]])
        st.dataframe(rows, hide_index=True, use_container_width=True)

with tab_live:
    track = load_parquet(OUT / "live_track.parquet")
    if track is None or track.empty:
        st.info("Live paper-tracking starts after the first daily pipeline run.")
    else:
        fig = go.Figure()
        for model, g in track.groupby("model"):
            eq = (1 + g.set_index("date")["ret"]).cumprod()
            fig.add_trace(go.Scatter(x=eq.index, y=eq, name=model,
                                     line=dict(color=COLORS.get(model), width=2)))
        spy = track.drop_duplicates("date").set_index("date")["spy_ret"]
        fig.add_trace(go.Scatter(x=spy.index, y=(1 + spy).cumprod(), name="SPY",
                                 line=dict(color=COLORS["spy"], width=2, dash="dash")))
        _style_axes(fig, yaxis_title="Growth of $1")
        st.plotly_chart(fig, use_container_width=True)

with tab_bt:
    curves = load_parquet(BT / "equity_curves.parquet")
    metrics = load_json(BT / "metrics.json")
    band = load_parquet(BT / "bootstrap.parquet")
    ic = load_parquet(BT / "ic_monthly.parquet")
    boot = load_json(BT / "bootstrap_summary.json")
    if curves is None or metrics is None:
        st.info("No backtest artifacts yet — run scripts/run_backtest.py.")
    else:
        fig = go.Figure()
        if band is not None:
            fig.add_trace(go.Scatter(x=band["date"], y=band["p95"], line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=band["date"], y=band["p05"], fill="tonexty",
                                     fillcolor=BAND_FILL, line=dict(width=0),
                                     name="random count-matched (5–95%)"))
        for series, g in curves.groupby("series"):
            line = dict(color=COLORS.get(series), width=2)
            if series == "spy":
                line["dash"] = "dash"
            fig.add_trace(go.Scatter(x=g["date"], y=g["equity"], name=series, line=line))
        _style_axes(fig, yaxis_title="Growth of $1")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Sub-period concentration matters — per-month picks are persisted "
                  "in picks.json; see the per-year table.")

        table = pd.DataFrame({k: v for k, v in metrics.items()
                              if k not in ("yearly", "turnover", "survivorship")}).T
        st.dataframe(table.style.format("{:.3f}"), use_container_width=True)

        if ic is not None and not ic.empty:
            mean_ic = float(ic["ic"].mean())
            ic_std = float(ic["ic"].std())
            ic_ir = mean_ic / ic_std if ic_std else float("nan")
            mean_spread = float(ic["decile_spread"].mean())
            champ_turnover = metrics.get("turnover", {}).get(CHAMPION)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean IC", f"{mean_ic:.3f}")
            c2.metric("IC IR", f"{ic_ir:.2f}")
            c3.metric("Mean decile spread", f"{mean_spread:.2%}")
            c4.metric("Champion turnover/mo",
                      f"{champ_turnover:.1%}" if champ_turnover is not None else "n/a")

        if boot:
            st.caption(
                f"Champion CAGR sits at the {boot['champion_cagr_percentile']:.0%} "
                f"percentile of 1,000 random count-matched portfolios "
                f"(random p50 CAGR {boot['cagr_p50']:.1%}).")
        if ic is not None:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=ic["date"], y=ic["ic"], name="Monthly rank IC",
                                  marker_color=COLORS["lgbm"]))
            fig2.add_trace(go.Bar(x=ic["date"], y=ic["decile_spread"], name="Decile spread",
                                  marker_color=COLORS["lgbm_ms"]))
            fig2.update_layout(barmode="group")
            _style_axes(fig2, yaxis_title="IC / decile spread")
            st.plotly_chart(fig2, use_container_width=True)
        if metrics.get("yearly"):
            st.dataframe(pd.DataFrame(metrics["yearly"]).style.format("{:.1%}"),
                         use_container_width=True)

with tab_method:
    validation = load_json(OUT / "validation.json")
    mem_status = load_json(OUT / "membership_status.json")
    bt_metrics = load_json(BT / "metrics.json")
    if validation:
        st.caption(f"Last data validation: {'OK' if validation['ok'] else 'FAILED'} "
                   f"({validation['stats'].get('last_date', '?')})")
    if mem_status and not mem_status.get("fresh", True):
        st.warning("Membership table is a cached copy — the Wikipedia fetch last failed.")
    st.markdown("""
## Methodology

- **Universe:** point-in-time S&P 500 membership reconstructed from Wikipedia's
  historical changes table; top 150 by rolling 12-month dollar volume.
- **Features (monthly, data ≤ t only):** 1/3/6/12-month momentum (winsorized
  cross-sectionally each month), Garman-Klass volatility, RSI, normalized ATR,
  Bollinger width, MACD histogram, rolling 24-month Fama-French-5 betas lagged one month.
- **Champion — K-Means momentum cluster:** monthly K-Means (k=4) on the feature
  cross-section; the momentum cluster is identified by centroid behaviour, and its
  members with above-universe-average composite momentum (z > 0) become must-buys —
  minimum 5 names (else hold previous portfolio), maximum 10, weighted by conviction
  with a 20% per-stock cap.
- **Challenger — LightGBM ranker:** LightGBM regressor on cross-sectional return ranks,
  walk-forward (rolling 36-month window, retrained monthly); must-buys are names scoring
  ≥ 1 standard deviation above the monthly cross-section, same 5/10 floor-cap and
  conviction weighting.
- **Backtest:** signal at month-end close, buy-and-hold with weight drift for one month,
  10 bps one-way cost on actual turnover. Simple returns throughout.
- **Benchmarks:** SPY and 1,000 random portfolios drawn from the same point-in-time
  universe, count-matched to the champion's actual monthly holdings.

## Limitations (read this)

- **Residual survivorship bias:** members delisted without Yahoo Finance history are
  missing from the backtest universe. Membership is point-in-time, price coverage isn't.
""")
    surv = (bt_metrics or {}).get("survivorship")
    if surv:
        st.caption(
            f"Measured: mean {surv['mean_missing_frac']:.1%} / max "
            f"{surv['max_missing_frac']:.1%} of point-in-time members missing price "
            f"history in a given month (see data/outputs/backtest/survivorship.parquet).")
    st.markdown("""
- **Execution assumptions:** month-end-close fills, no slippage beyond the 10 bps cost,
  no market impact, dividends assumed reinvested via adjusted prices.
- **Ticker-symbol drift:** Wikipedia's change log tracks tickers, not entities; renames
  and mergers can misalign a small share of months.
- **Fama-French publication lag:** the latest 1–3 months reuse each stock's last known
  betas (carried forward, never interpolated from the future).
- **Sub-period concentration:** the challenger leaned on a narrow subset of names
  during 2026 H1; the yearly table on the Backtest tab and the persisted
  `picks.json` (per-month holdings for all four strategies) exist so any
  sub-period can be audited directly rather than inferred from aggregate metrics.
- **Risk-free rate:** Sharpe ratios use a single whole-period average FF risk-free rate,
  not a time-varying one.
- **Live vs. backtest weight drift:** live tracking re-anchors target weights daily,
  while the backtest holds drifted weights for the full month (small semantic difference).
- **This is research, not advice.** Past (simulated) performance does not predict
  future results.
""")
