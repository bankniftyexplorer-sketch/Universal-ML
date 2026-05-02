"""
live_volatility_inference.py — VOL Live Forecast Publisher
==========================================================
Loads saved VOL model heads, rebuilds the latest feature frame, and writes
a machine-readable next-bar volatility/range forecast for dashboard use.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import joblib
import pandas as pd

from daily_volatility_engine import (
    VOL_MODEL_ARTIFACT_KEYS,
    _predict_har_rv_model,
    build_daily_volatility_feature_frame,
    build_vol_forecast_payload,
    combine_vol_forecasts,
    fetch_vol_timeframe_context,
    predict_volatility_heads,
    prepare_vol_inference_frame,
    print_vol_forecast,
    save_vol_forecast_json,
)
from universal_ml_engine import (
    describe_selected_frame,
    get_artifact_paths,
    prepare_symbol_artifact_context,
    resolve_artifact_path,
)

warnings.filterwarnings("ignore")


def _print_tf_span(label: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    print(
        f"  {label} bars : {len(df):>7}  "
        f"({df['time'].min().date()} → {df['time'].max().date()})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="VOL Live Forecast Publisher")
    parser.add_argument("--outdir", type=str, default="/home/km/Universal-ML/")
    parser.add_argument("--symbol", type=str, required=True)
    args = parser.parse_args()

    project_root = os.path.abspath(args.outdir)
    requested_symbol = args.symbol.upper()
    artifact_ctx = prepare_symbol_artifact_context(
        project_root,
        requested_symbol,
        asset_class="SPOT",
        timeframes=("VOL",),
        logger=None,
    )
    symbol = str(artifact_ctx["symbol"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    file_prefix = str(artifact_ctx["file_prefix"])
    artifact_paths_vol = get_artifact_paths(symbol_dir, file_prefix, "VOL")

    model_paths = {
        target: resolve_artifact_path(symbol_dir, file_prefix, "VOL", artifact_key)
        for target, artifact_key in VOL_MODEL_ARTIFACT_KEYS.items()
    }
    feat_path = resolve_artifact_path(symbol_dir, file_prefix, "VOL", "features")
    calibrator_path = resolve_artifact_path(
        symbol_dir, file_prefix, "VOL", "calibrators"
    )
    conformal_path = resolve_artifact_path(symbol_dir, file_prefix, "VOL", "conformal")

    if (
        any(not os.path.exists(path) for path in model_paths.values())
        or not os.path.exists(feat_path)
        or not os.path.exists(conformal_path)
    ):
        print(
            f"  [!] FATAL: Missing VOL artifacts for {symbol}. "
            "Run daily_volatility_engine.py first."
        )
        raise SystemExit(1)

    with open(feat_path, encoding="utf-8") as handle:
        feature_cols = [line.strip() for line in handle if line.strip()]
    models = {target: joblib.load(path) for target, path in model_paths.items()}
    calibrators = (
        joblib.load(calibrator_path) if os.path.exists(calibrator_path) else {}
    )
    conformal_artifact = joblib.load(conformal_path)

    sys.path.append(os.path.join(project_root, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("  [!] FATAL: Cannot locate inference_bridge.py in data_vault directory.")
        raise SystemExit(1)

    print("=" * 70)
    print(f"  VOL LIVE FORECAST PUBLISHER FOR: {symbol}")
    print("=" * 70)

    bridge = InferenceBridge(
        db_path=os.path.join(project_root, "data_vault", "ohlcv.db")
    )
    tf_ctx = fetch_vol_timeframe_context(
        bridge,
        str(artifact_ctx["identity"].market_data_symbol),
    )
    primary_frames = tf_ctx["primary_frames"]
    reference_frames = tf_ctx["reference_frames"]

    df_1d = primary_frames["1D"]
    if df_1d is None or df_1d.empty:
        print(f"  [!] FATAL: No usable 1D primary data found for {symbol}.")
        raise SystemExit(1)

    df_1h = tf_ctx["df_1h"]
    df_1h_raw = tf_ctx["df_1h_raw"]
    df_1h_status = tf_ctx["df_1h_status"]
    df_vix = tf_ctx["df_vix"]
    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]
    df_3m = primary_frames["3M"]
    df_6m = primary_frames["6M"]
    df_12m = primary_frames["12M"]

    print(f"  1D primary lane : {describe_selected_frame(df_1d)}")
    _print_tf_span("1D", df_1d)
    if df_1h_raw is not None and not df_1h_raw.empty:
        if df_1h is not None and not df_1h.empty:
            print(f"  1H intraday ref : {describe_selected_frame(df_1h)}")
            _print_tf_span("1H", df_1h)
        else:
            print(
                f"  1H intraday ref : {describe_selected_frame(df_1h_raw)} "
                f"[OPTIONAL {df_1h_status or 'UNKNOWN'} -> ignored]"
            )
            _print_tf_span("1H", df_1h_raw)
    if df_vix is not None and not df_vix.empty:
        print(f"  VIX companion    : {describe_selected_frame(df_vix)}")
        _print_tf_span("VIX", df_vix)
    else:
        print("  VIX companion    : [OPTIONAL unavailable -> ignored]")

    df_feature_frame = build_daily_volatility_feature_frame(
        df_1d,
        reference_1d=reference_frames["1D"],
        df_1h=df_1h,
        df_vix=df_vix,
        df_1w=df_1w,
        df_1m=df_1m,
        df_3m=df_3m,
        df_6m=df_6m,
        df_12m=df_12m,
        logger=print,
    )
    df_infer = prepare_vol_inference_frame(df_feature_frame, feature_cols)
    last_row = df_infer.iloc[-1]
    ml_forecasts = predict_volatility_heads(models, feature_cols, last_row)
    har_models = conformal_artifact.get("har_models", {})
    har_forecasts = {
        target: float(_predict_har_rv_model(har_models[target], last_row))
        for target in ml_forecasts
        if target in har_models
    }
    forecasts = combine_vol_forecasts(
        ml_forecasts,
        har_forecasts,
        conformal_artifact,
        calibrators=calibrators,
        targets=list(ml_forecasts),
    )

    if df_1h is not None and not df_1h.empty:
        reference_price = float(df_1h["close"].iloc[-1])
        reference_source = "latest_1h_close"
    else:
        reference_price = float(last_row["close"])
        reference_source = "latest_close"

    payload = build_vol_forecast_payload(
        symbol=symbol,
        row=last_row,
        forecasts=forecasts,
        intraday_1h_used=df_1h is not None and not df_1h.empty,
        vix_available=df_vix is not None and not df_vix.empty,
        reference_price=reference_price,
        reference_price_source=reference_source,
        raw_forecasts=ml_forecasts,
        har_forecasts=har_forecasts,
        conformal_artifact=conformal_artifact,
    )
    save_vol_forecast_json(payload, artifact_paths_vol["live_forecast"])
    print(f"  VOL live forecast saved to '{artifact_paths_vol['live_forecast']}'")
    print_vol_forecast(payload, title=f"{symbol} VOL LIVE FORECAST")

    # Auto-generate the visual PNG dashboard
    try:
        from plot_vol_forecast import plot_live_forecast

        plot_live_forecast(symbol, project_root)
    except Exception as e:
        print(f"  [!] Failed to generate VOL dashboard image: {e}")


if __name__ == "__main__":
    main()
