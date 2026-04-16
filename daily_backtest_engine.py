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
from daily_ml_engine import LIVE_CONFIDENCE_THRESHOLD_DAILY
from julia_bridge import (
    holographic_feature_engine_daily,
    kalman_structural_engine_daily,
    narrative_context_engine_daily,
    rv_feature_engine_daily,
    smc_feature_engine_daily,
)
from universal_ml_engine import EnsembleModel  # noqa: F401
from universal_ml_engine import (
    _compute_atr14,
    apply_calibrator_to_prob_array,
    build_report_data_lines,
    build_timeframe_selection,
    describe_policy_artifact,
    describe_selected_frame,
    get_artifact_paths,
    inject_thermodynamic_basis,
    prepare_symbol_artifact_context,
    resolve_artifact_path,
    select_primary_timeframe,
)
from sleeve_registry import load_sleeve_registry_entry

warnings.filterwarnings("ignore")

BARRIER_HORIZON_BARS_DAILY = 10
EOD_GATE_HOUR_DAILY = 24


def _pick_primary_1d(tf_maps: dict) -> pd.DataFrame | None:
    return select_primary_timeframe(tf_maps, "1D")


def _get_tf(tf_maps: dict, label: str) -> pd.DataFrame | None:
    return select_primary_timeframe(tf_maps, label)


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

    data_dir = os.path.abspath(args.outdir)
    requested_symbol = args.symbol.upper()
    artifact_ctx = prepare_symbol_artifact_context(
        data_dir,
        requested_symbol,
        asset_class="SPOT",
        timeframes=("1D",),
    )
    symbol = str(artifact_ctx["symbol"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    file_prefix = str(artifact_ctx["file_prefix"])
    artifact_paths_1d = get_artifact_paths(symbol_dir, file_prefix, "1D")

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
        "SPOT": bridge.fetch_holographic_stack(
            str(artifact_ctx["identity"].market_data_symbol),
            "SPOT",
            include_realized_vol=True,
        ),
    }

    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1D", "1W", "1M", "3M", "6M", "12M")
    )
    df_1d = primary_frames["1D"]
    if df_1d is None or df_1d.empty:
        print(f"[!] FATAL: No 1D primary data for {symbol}")
        return

    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]
    df_3m = primary_frames["3M"]
    df_6m = primary_frames["6M"]
    df_12m = primary_frames["12M"]

    print(f"  1D primary lane : {describe_selected_frame(df_1d)}")
    _print_tf_span("1D", df_1d)
    if df_1w is not None and not df_1w.empty:
        print(f"  1W primary lane : {describe_selected_frame(df_1w)}")
    _print_tf_span("1W", df_1w)
    if df_1m is not None and not df_1m.empty:
        print(f"  1M primary lane : {describe_selected_frame(df_1m)}")
    _print_tf_span("1M", df_1m)
    if df_3m is not None and not df_3m.empty:
        print(f"  3M primary lane : {describe_selected_frame(df_3m)}")
    _print_tf_span("3M", df_3m)
    if df_6m is not None and not df_6m.empty:
        print(f"  6M primary lane : {describe_selected_frame(df_6m)}")
    _print_tf_span("6M", df_6m)
    if df_12m is not None and not df_12m.empty:
        print(f"  12M primary lane : {describe_selected_frame(df_12m)}")
    _print_tf_span("12M", df_12m)

    model_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "model")
    feat_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "features")
    calibrator_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "calibrator")
    trade_plan_path = resolve_artifact_path(
        symbol_dir, file_prefix, "1D", "trade_plan_models"
    )
    exit_surface_path = resolve_artifact_path(
        symbol_dir, file_prefix, "1D", "exit_surface"
    )
    policy_path = resolve_artifact_path(
        symbol_dir, file_prefix, "1D", "policy_artifact"
    )

    registry_entry = load_sleeve_registry_entry(symbol, "1D")
    if registry_entry is not None and not bool(registry_entry.get("enabled")):
        print(
            f"  [Registry] Sleeve {symbol}_1D disabled. "
            f"Reason: {registry_entry.get('reason', 'admission failed')}"
        )
        return

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print(
            f"[!] Could not find {model_path} or {feat_path}. Run daily_ml_engine.py first."
        )
        return

    model = joblib.load(model_path)
    calibrator = (
        joblib.load(calibrator_path) if os.path.exists(calibrator_path) else None
    )
    with open(feat_path, "r") as handle:
        feature_cols = [line.strip() for line in handle if line.strip()]
    trade_plan_models = (
        joblib.load(trade_plan_path) if os.path.exists(trade_plan_path) else {}
    )
    exit_surface_artifact = (
        joblib.load(exit_surface_path) if os.path.exists(exit_surface_path) else None
    )
    policy_artifact = joblib.load(policy_path) if os.path.exists(policy_path) else None
    selected_variant = (
        str(registry_entry.get("selected_variant"))
        if registry_entry is not None and registry_entry.get("selected_variant")
        else None
    )
    if selected_variant == "base":
        policy_artifact = None
        print(f"  [Registry] Using base variant for {symbol}_1D.")
    elif selected_variant == "policy":
        if policy_artifact is None:
            print(
                f"  [Registry] Policy variant selected for {symbol}_1D but artifact is missing."
            )
            return
        print(f"  [Registry] Using policy variant for {symbol}_1D.")
    if policy_artifact is not None:
        print(
            f"  [=] Opportunity head loaded. {describe_policy_artifact(policy_artifact)}"
        )
    if exit_surface_artifact is not None:
        print(
            f"  [=] Exit surface loaded. "
            f"{exit_surface_artifact.get('artifact_kind', 'exit_surface')} "
            f"[lane={exit_surface_artifact.get('lane', 'UNK')}]"
        )

    print("\n  [=] Reconstructing daily feature space over historical data...")

    df_1d = inject_thermodynamic_basis(
        df_1d,
        reference_frames["1D"],
        logger=print,
    )

    df_1d["session_time_pos"] = 0.0
    df_1d["eod_basis_momentum"] = 0.0

    rv_df = rv_feature_engine_daily(df_1d, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        df_1d[col] = rv_df[col].values

    df_1d_labelled = _compute_atr14(df_1d.copy())
    df_full = holographic_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_3m)

    smc_df = smc_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    kf_df = kalman_structural_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in kf_df.columns:
        df_full[col] = kf_df[col].values
    nc_df = narrative_context_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in nc_df.columns:
        df_full[col] = nc_df[col].values

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
        prob_array = apply_calibrator_to_prob_array(prob_array, calibrator)
        print(
            f"  [=] OOS proba loaded. {np.isfinite(prob_array).sum()} honest OOS bars."
        )
        if calibrator is not None:
            print("  [=] Applied saved 1D calibrator to OOS probabilities.")
    else:
        print("  [!] No OOS map. Using in-sample (unreliable).")
        prob_array = model.predict(df_backtest[feature_cols])
        prob_array = apply_calibrator_to_prob_array(prob_array, calibrator)

    prob_array_1d = np.full(len(df_backtest), np.nan)

    results = run_backtest(
        df_backtest,
        prob_array,
        prob_array_1d,
        feature_cols,
        trade_plan_models=trade_plan_models,
        exit_surface_artifact=exit_surface_artifact,
        policy_artifact=policy_artifact,
        initial_capital=10000.0,
        risk_pct=0.02,
        conf_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
        max_hold_bars=BARRIER_HORIZON_BARS_DAILY,
        eod_gate_hour=EOD_GATE_HOUR_DAILY,
        lane="1D",
    )

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
    print(f"  Policy Filter  : {results.get('policy_blocks', 0)} bars bypassed")
    print(f"  Max Drawdown   : {results['max_drawdown'] * 100:.2f}%")
    print("=" * 60)

    report_path = artifact_paths_1d["backtest_report"]
    generate_report(
        results,
        metrics,
        f"{symbol} (DAILY)",
        report_path,
        data_update_lines=build_report_data_lines({"1D": df_1d}),
    )
    print(f"\n  [✓] Report saved to {report_path}")


if __name__ == "__main__":
    main()
