"""
TOON v4.2 — Live Daily Inference Engine
======================================================================
Ultra-fast forward-pass execution script.
Reconstructs holographic geometry for the most recent 150 bars only.
Zero training. Zero backtesting. Pure live signal generation.
"""

import os
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

from universal_ml_engine import (
    prepare_intraday_thermodynamics,
    _compute_atr14,
    build_timeframe_selection,
    describe_selected_frame,
    merge_higher_tf,
    migrate_legacy_artifacts,
    fib_structural_basis,
    NON_FEATURE_COLS_SET,
    predict_next_bar,
    predict_trade_plan,
    resolve_artifact_path,
    LIVE_CONFIDENCE_THRESHOLD,
)
from julia_bridge import (
    holographic_feature_engine_fast as holographic_feature_engine,
    smc_feature_engine_fast,
)
from shadow_brain import ShadowBrain

warnings.filterwarnings("ignore")

EOD_GATE_HOUR = 14  # IST. Set to 24 for 24/7 crypto.


def main():
    parser = argparse.ArgumentParser(description="Live Daily Inference Engine")
    parser.add_argument("--outdir", type=str, default="/home/km/Universal-ML/")
    parser.add_argument("--symbol", type=str, required=True, help="Target Base Symbol")
    args = parser.parse_args()

    PROJECT_ROOT = args.outdir
    SYMBOL = args.symbol.upper()
    SYMBOL_DIR = os.path.join(PROJECT_ROOT, SYMBOL)
    file_prefix = SYMBOL.lower().replace(" ", "_")
    migrate_legacy_artifacts(SYMBOL_DIR, file_prefix, "1H", logger=None)

    sys.path.append(os.path.join(PROJECT_ROOT, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
        from vault_engine import DataVault
    except ImportError:
        print("  [!] FATAL: Cannot locate data_vault modules.")
        return

    bridge = InferenceBridge(
        db_path=os.path.join(PROJECT_ROOT, "data_vault", "ohlcv.db")
    )

    tf_maps = {
        "SPOT": bridge.fetch_holographic_stack(SYMBOL, "SPOT"),
    }

    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1H", "1D", "1W", "1M")
    )
    df_1h = primary_frames["1H"]
    df_1d = primary_frames["1D"]
    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]

    if df_1h is None or df_1h.empty:
        print(f"  [!] FATAL: Execution engine requires usable 1H primary data for {SYMBOL}.")
        return

    print(f"  [Live] 1H primary lane: {describe_selected_frame(df_1h)}")

    # Load Models from the isolated Symbol Directory
    mod_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "model")
    feat_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "features")
    tp_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "trade_plan_models")

    if not os.path.exists(mod_path) or not os.path.exists(feat_path):
        print(f"  [!] FATAL: Pre-trained model missing for {SYMBOL} in {SYMBOL_DIR}.")
        return

    ultimate_model = joblib.load(mod_path)
    with open(feat_path, "r") as f:
        feature_cols_to_use = [line.strip() for line in f.readlines() if line.strip()]

    trade_plan_models = joblib.load(tp_path) if os.path.exists(tp_path) else {}

    # ── 3. BUILD SHARED INTRADAY THERMODYNAMIC STATE ──────────────────────
    try:
        df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
            df_1h=df_1h,
            df_1d=df_1d,
            df_1w=df_1w,
            df_1m=df_1m,
            reference_1h=reference_frames["1H"],
            reference_1d=reference_frames["1D"],
            reference_1w=reference_frames["1W"],
            symbol=SYMBOL,
            logger=print,
        )
    except ValueError:
        return

    df_1h = fib_structural_basis(
        df_1h,
        htf_frames={"1D": df_1d, "1W": df_1w, "1M": df_1m},
        pairs=[("1D", "a"), ("1W", "b"), ("1M", "c")],
    )

    df_1h_labelled = _compute_atr14(df_1h.copy())

    # ── 5. THE SURGICAL TRUNCATION (CONTEXT ISOLATION) ────────────────────
    # Truncate the execution array to the last 150 bars to save computation.
    # The rolling Z-Score (20) and ATR (14) are already calculated securely.
    df_1h_tail = df_1h_labelled.tail(150).reset_index(drop=True)

    # ── 6. GEOMETRIC RECONSTRUCTION ───────────────────────────────────────
    df_full = holographic_feature_engine(
        df_1h_tail,
        df_1d=df_1d,
        df_1w=df_1w,
        df_1m=df_1m,
    )
    smc_df = smc_feature_engine_fast(df_1h_tail, df_1d, df_1w, df_1m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)

    NON_FEATURE_COLS = set(NON_FEATURE_COLS_SET)

    all_holo_cols = [c for c in df_full.columns if c not in NON_FEATURE_COLS]
    state_cols = ["time", "close", "atr14"]
    for b_col in ["basis_pct", "basis_z_score", "basis_vel_5", "basis_vel_10"]:
        if b_col in df_full.columns:
            state_cols.append(b_col)

    df_model_ready = df_full[all_holo_cols + state_cols].copy()
    for col in all_holo_cols:
        df_model_ready[col] = (
            df_model_ready[col].map(lambda x: np.nan if np.isinf(x) else x).fillna(0)
        )
    df_model_ready = df_model_ready.dropna(subset=all_holo_cols).reset_index(drop=True)

    # ── 7. THE TACTICAL FORECAST ──────────────────────────────────────────
    last_row = df_model_ready.iloc[-1]

    # Check the Shock Gate
    if "basis_z_score" in last_row and abs(float(last_row["basis_z_score"])) > 2.5:
        print("\n" + "=" * 70)
        print(f"  {SYMBOL.upper()} FORECAST (LIVE INFERENCE)")
        print("=" * 70)
        print(f"  Bar time    : {last_row.get('time', 'N/A')}")
        print("  [!] THERMODYNAMIC SHOCK DETECTED (|Z| > 2.5). KILL SWITCH ENGAGED.")
        print("      No directional trade plan will be generated.")
        print("======================================================================")
        return

    # Check the EOD Gate
    if "time" in last_row:
        next_time = pd.to_datetime(last_row["time"]) + pd.Timedelta(hours=1)
        if next_time.hour >= EOD_GATE_HOUR:
            print("\n" + "=" * 70)
            print(f"  {SYMBOL.upper()} FORECAST (LIVE INFERENCE)")
            print("=" * 70)
            print(f"  Bar time    : {last_row.get('time', 'N/A')}")
            print("  [!] LATE-SESSION SIGNAL DETECTED (Execution Hour >= 14).")
            print("      EOD Over-night Risk Gate engaged. Trade aborted.")
            print(
                "======================================================================"
            )
            return

    pred = predict_next_bar(
        ultimate_model,
        feature_cols_to_use,
        last_row,
        confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
    )
    close_price = float(last_row["close"])
    atr = float(last_row["atr14"]) if "atr14" in last_row else 150.0
    trade_plan = predict_trade_plan(
        trade_plan_models, feature_cols_to_use, last_row.copy(), pred["direction"], atr
    )

    # Initialize and Train Shadow Brain with Strict Symbol Isolation
    shadow = ShadowBrain(
        base_symbol=SYMBOL, db_path=os.path.join(PROJECT_ROOT, "data_vault", "ohlcv.db")
    )
    shadow.train(days=45)

    is_vetoed = False
    if pred["direction"] in {"UP", "DOWN"} and pred["signal_strength"] != "NO_TRADE":
        features_dict = last_row[feature_cols_to_use].to_dict()
        is_vetoed = shadow.predict_veto(features_dict)

    if is_vetoed:
        trade_plan["note"] = f"{trade_plan['note']} VETOED_BY_SHADOW".strip()
        pred["direction"] = "VETO"
        pred["signal_strength"] = "NO_TRADE"
    elif pred["direction"] in {"UP", "DOWN"} and pred["signal_strength"] != "NO_TRADE":
        # Log to Performance Ledger
        try:
            vault = DataVault(
                db_path=os.path.join(PROJECT_ROOT, "data_vault", "ohlcv.db")
            )
            vault.log_trade_result(
                {
                    "timestamp": str(last_row.get("time", pd.Timestamp.utcnow())),
                    "base_symbol": SYMBOL,
                    "direction": pred["direction"],
                    "conf_score": pred["confidence"],
                    "entry_price": close_price,
                    "exit_price": None,
                    "pnl_r": None,
                    "win_loss_target": None,
                }
            )
            print("  [VAULT] Tactical execution queued into Performance Ledger.")
        except Exception as e:
            print(f"  [VAULT] Failed to log trade: {e}")

    print("\n" + "=" * 70)
    print(f"  {SYMBOL.upper()} FORECAST (LIVE INFERENCE)")
    print("=" * 70)
    if "time" in last_row:
        print(f"  Bar time    : {last_row['time']}")
    print(f"  Direction   : {pred['direction']}")
    print(
        f"  Confidence  : {pred['confidence']:.1%} (Regressor Score: {pred['raw_score']:.3f})"
    )
    print(f"  Signal      : {pred['signal_strength']}")
    print("----------------------------------------------------------------------")
    print(f"  Entry Price : {close_price:,.2f} (Current Close)")

    trail_str = "N/A"
    filter_note = trade_plan["note"]
    if pred["direction"] in {"UP", "DOWN"} and np.isfinite(trade_plan["sl"]):
        sl_str = (
            f"{trade_plan['sl']:,.2f}  (ML stop {trade_plan['stop_atr']:.2f}x ATR14)"
        )
        tp1_str = (
            f"{trade_plan['tp1']:,.2f}  (ML TP1 {trade_plan['tp1_atr']:.2f}x ATR14)"
        )
        tp2_str = (
            f"{trade_plan['tp2']:,.2f}  (ML TP2 {trade_plan['tp2_atr']:.2f}x ATR14)"
        )
        trail_str = f"{trade_plan['trail_r']:.2f}R trailing stop after TP1"
    else:
        sl_str, tp1_str, tp2_str = "N/A", "N/A", "N/A"

    if pred["signal_strength"] == "NO_TRADE":
        filter_note = f"{filter_note} Filtered: conf below {LIVE_CONFIDENCE_THRESHOLD:.2f}".strip()

    if pred["direction"] in {"UP", "DOWN"}:
        print(f"  Stop Loss   : {sl_str}")
        print(f"  Target 1    : {tp1_str}")
        print(f"  Target 2    : {tp2_str}")
        print(f"  Trail Stop  : {trail_str}")
        if filter_note:
            print(f"  [{filter_note}]")
    else:
        print("  [No clear directional edge, targets N/A]")

    print("======================================================================")


if __name__ == "__main__":
    main()
