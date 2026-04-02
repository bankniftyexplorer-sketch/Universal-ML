"""
daily_backtest_engine.py — TOON v5.1 Daily Backtest Engine
==========================================================
Simulates daily-frequency trades using the trained daily model.
Reuses generic backtest infrastructure from backtest_engine.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from backtest_engine import calculate_metrics, generate_report, run_backtest
from julia_bridge import holographic_feature_engine_daily
from universal_ml_engine import (
    _compute_atr14,
    inject_thermodynamic_basis,
    migrate_legacy_artifacts,
    resolve_artifact_path,
)

warnings.filterwarnings("ignore")

BARRIER_HORIZON_BARS_DAILY = 10
LIVE_CONFIDENCE_THRESHOLD_DAILY = 0.56
EOD_GATE_HOUR_DAILY = 24


def _pick_primary_1d(tf_maps: dict) -> pd.DataFrame | None:
    fut_1d = tf_maps["FUT"].get("1D")
    if fut_1d is not None and not fut_1d.empty:
        return fut_1d.sort_values("time").reset_index(drop=True).copy()
    spot_1d = tf_maps["SPOT"].get("1D")
    if spot_1d is not None and not spot_1d.empty:
        return spot_1d.sort_values("time").reset_index(drop=True).copy()
    return None


def _get_tf(tf_maps: dict, label: str) -> pd.DataFrame | None:
    fut_df = tf_maps["FUT"].get(label)
    if fut_df is not None and not fut_df.empty:
        return fut_df.sort_values("time").reset_index(drop=True).copy()
    spot_df = tf_maps["SPOT"].get(label)
    if spot_df is not None and not spot_df.empty:
        return spot_df.sort_values("time").reset_index(drop=True).copy()
    return None


def _print_tf_span(label: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    print(
        f"  {label} bars : {len(df):>7}  ({df['time'].min().date()} → {df['time'].max().date()})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Backtest Engine")
    parser.add_argument("--outdir", type=str, default="/home/km/Universal-ML/")
    parser.add_argument("--symbol", type=str, required=True)
    args = parser.parse_args()

    data_dir = args.outdir
    symbol = args.symbol.upper()
    symbol_dir = os.path.join(data_dir, symbol)
    file_prefix = symbol.lower().replace(" ", "_")
    artifact_paths_1d = migrate_legacy_artifacts(symbol_dir, file_prefix, "1D")

    sys.path.append(os.path.join(data_dir, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("[!] FATAL: Cannot locate inference_bridge.py.")
        return

    print("=" * 60)
    print("  DAILY BACKTEST ENGINE — TOON v5.1")
    print("=" * 60)

    bridge = InferenceBridge(db_path=os.path.join(data_dir, "data_vault", "ohlcv.db"))
    tf_maps = {
        "FUT": bridge.fetch_holographic_stack(symbol, "FUT"),
        "SPOT": bridge.fetch_holographic_stack(symbol, "SPOT"),
    }

    df_1d = _pick_primary_1d(tf_maps)
    if df_1d is None or df_1d.empty:
        print(f"[!] FATAL: No 1D data for {symbol}")
        return

    df_1w = _get_tf(tf_maps, "1W")
    df_1m = _get_tf(tf_maps, "1M")
    df_3m = _get_tf(tf_maps, "3M")
    df_6m = _get_tf(tf_maps, "6M")
    df_12m = _get_tf(tf_maps, "12M")

    _print_tf_span("1D", df_1d)
    _print_tf_span("1W", df_1w)
    _print_tf_span("1M", df_1m)
    _print_tf_span("3M", df_3m)
    _print_tf_span("6M", df_6m)
    _print_tf_span("12M", df_12m)

    model_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "model")
    feat_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "features")
    trade_plan_path = resolve_artifact_path(
        symbol_dir, file_prefix, "1D", "trade_plan_models"
    )

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print(
            f"[!] Could not find {model_path} or {feat_path}. Run daily_ml_engine.py first."
        )
        return

    model = joblib.load(model_path)
    with open(feat_path, "r") as handle:
        feature_cols = [line.strip() for line in handle if line.strip()]
    trade_plan_models = (
        joblib.load(trade_plan_path) if os.path.exists(trade_plan_path) else {}
    )

    print("\n  [=] Reconstructing daily feature space over historical data...")

    spot_1d = tf_maps["SPOT"].get("1D")
    if spot_1d is not None and not spot_1d.empty:
        df_1d = inject_thermodynamic_basis(
            df_1d, spot_1d.sort_values("time").reset_index(drop=True)
        )
    else:
        for col in ["basis_pct", "basis_z_score", "basis_vel_5", "basis_vel_10"]:
            df_1d[col] = 0.0

    df_1d["session_time_pos"] = 0.0
    df_1d["eod_basis_momentum"] = 0.0

    df_1d_labelled = _compute_atr14(df_1d.copy())
    df_full = holographic_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_3m)

    from daily_ml_engine import add_daily_confluence, inject_macro_regime

    df_full = inject_macro_regime(df_full, df_6m, "6m")
    df_full = inject_macro_regime(df_full, df_12m, "12m")
    df_full = add_daily_confluence(df_full)

    all_needed = list(
        dict.fromkeys(
            feature_cols
            + [
                "time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "atr14",
                "basis_z_score",
            ]
        )
    )
    available = [col for col in all_needed if col in df_full.columns]
    df_backtest = df_full[available].copy()

    for col in feature_cols:
        if col not in df_backtest.columns:
            df_backtest[col] = 0.0
        df_backtest[col] = (
            df_backtest[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    df_backtest = df_backtest.reset_index(drop=True)

    print(f"  [=] Total Daily Bars: {len(df_backtest)}")

    oos_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "oos_proba")
    if os.path.exists(oos_path):
        oos_proba_map = joblib.load(oos_path)
        prob_array = np.array(
            [
                float(oos_proba_map[pd.Timestamp(ts)])
                if pd.Timestamp(ts) in oos_proba_map
                else np.nan
                for ts in df_backtest["time"]
            ]
        )
        print(
            f"  [=] OOS proba loaded. {np.isfinite(prob_array).sum()} honest OOS bars."
        )
    else:
        print("  [!] No OOS map. Using in-sample (unreliable).")
        prob_array = model.predict(df_backtest[feature_cols])

    prob_array_1d = np.full(len(df_backtest), np.nan)

    import backtest_engine

    original_eod = getattr(backtest_engine, "EOD_GATE_HOUR", 14)
    try:
        backtest_engine.EOD_GATE_HOUR = EOD_GATE_HOUR_DAILY
        results = run_backtest(
            df_backtest,
            prob_array,
            prob_array_1d,
            feature_cols,
            trade_plan_models=trade_plan_models,
            initial_capital=10000.0,
            risk_pct=0.02,
            conf_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
            max_hold_bars=BARRIER_HORIZON_BARS_DAILY,
        )
    finally:
        backtest_engine.EOD_GATE_HOUR = original_eod

    if results is None:
        return

    metrics = calculate_metrics(
        results["trades"],
        results.get("equity_curve"),
        results.get("time_curve"),
    )

    print("\n" + "=" * 60)
    print("  DAILY BACKTEST RESULTS")
    print("=" * 60)
    print(f"  Final Equity   : ${results['final_equity']:,.2f}")
    print(f"  Total Trades   : {metrics.get('total_trades', 0)}")
    print(f"  Win Rate       : {metrics.get('win_rate', 0) * 100:.1f}%")
    print(f"  Profit Factor  : {metrics.get('profit_factor', 0):.3f}")
    print(f"  Sharpe Ratio   : {metrics.get('sharpe', 0):.3f}")
    print(f"  Max Drawdown   : {results['max_drawdown'] * 100:.2f}%")
    print("=" * 60)

    report_path = artifact_paths_1d["backtest_report"]
    generate_report(results, metrics, f"{symbol} (DAILY)", report_path)
    print(f"\n  [✓] Report saved to {report_path}")


if __name__ == "__main__":
    main()
