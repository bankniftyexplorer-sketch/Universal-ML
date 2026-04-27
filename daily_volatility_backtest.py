"""
daily_volatility_backtest.py — VOL Daily Volatility Replay
==========================================================
Replays saved VOL artifacts against reconstructed historical data and
reports forecast accuracy plus excursion-band coverage.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daily_volatility_engine import (
    VOL_MIN_TRAIN_BARS,
    VOL_PURGE_GAP,
    VOL_TARGET_LABELS,
    VOL_TARGETS,
    VOL_TEST_SIZE_RATIO,
    _har_rv_baseline,
    _regression_metrics,
    _vol_split_points,
    build_daily_volatility_feature_frame,
    fetch_vol_timeframe_context,
    prepare_vol_model_ready,
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


def replay_vol_forecasts(
    df: pd.DataFrame,
    models: dict[str, object],
    feature_cols: list[str],
    oos_map: dict | None,
) -> pd.DataFrame:
    """Reconstruct predictions from saved OOS map + apply models to recent bars."""
    replay = df[["time", *VOL_TARGETS]].copy()
    model_preds = {
        target: np.asarray(models[target].predict(df[feature_cols]), dtype=float)
        for target in VOL_TARGETS
    }
    saved_map = {pd.Timestamp(ts): payload for ts, payload in (oos_map or {}).items()}
    is_oos = np.zeros(len(df), dtype=bool)

    for target in VOL_TARGETS:
        replay[f"pred_{target}"] = model_preds[target]

    for idx, ts in enumerate(df["time"]):
        payload = saved_map.get(pd.Timestamp(ts))
        if not isinstance(payload, dict):
            continue
        is_oos[idx] = True
        for target in VOL_TARGETS:
            if target in payload and payload[target] is not None:
                replay.at[idx, f"pred_{target}"] = float(payload[target])

    replay["is_oos"] = is_oos
    return replay


def _build_har_logvol_oos_series(df: pd.DataFrame) -> np.ndarray:
    har_series = np.full(len(df), np.nan, dtype=float)
    split_points = _vol_split_points(
        len(df),
        min_train_bars=VOL_MIN_TRAIN_BARS,
        test_size_ratio=VOL_TEST_SIZE_RATIO,
    )
    for train_end, test_end in split_points:
        purged_train_end = max(0, train_end - VOL_PURGE_GAP)
        har_preds = _har_rv_baseline(
            df.iloc[:purged_train_end],
            df.iloc[train_end:test_end],
            "next_yz_logvol",
        )
        har_series[train_end:test_end] = har_preds
    return har_series


def coverage_report(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    quantile_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Compute pinball loss and empirical coverage at each quantile."""
    levels = quantile_levels or [0.10, 0.25, 0.50, 0.75, 0.90]
    rows: list[dict[str, float]] = []

    for level in levels:
        row: dict[str, float] = {"quantile": float(level)}
        for target, prefix in [
            ("next_up_exc", "up"),
            ("next_dn_exc", "dn"),
        ]:
            pred = predictions[target].to_numpy(dtype=float)
            actual = actuals[target].to_numpy(dtype=float)
            mask = np.isfinite(pred) & np.isfinite(actual) & (pred > 0.0)
            if mask.sum() < 20:
                row[f"{prefix}_scale"] = float("nan")
                row[f"{prefix}_coverage"] = float("nan")
                row[f"{prefix}_pinball"] = float("nan")
                continue

            ratios = np.clip(actual[mask] / np.maximum(pred[mask], 1e-9), 0.0, 10.0)
            scale = float(np.nanquantile(ratios, level))
            q_pred = pred[mask] * scale
            coverage = float(np.mean(actual[mask] <= q_pred))
            diff = actual[mask] - q_pred
            pinball = float(np.mean(np.maximum(level * diff, (level - 1.0) * diff)))

            row[f"{prefix}_scale"] = scale
            row[f"{prefix}_coverage"] = coverage
            row[f"{prefix}_pinball"] = pinball

        rows.append(row)

    return pd.DataFrame(rows)


def save_vol_backtest_report(
    replay_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    save_path: str,
    *,
    symbol: str,
) -> None:
    report_df = replay_df[replay_df["is_oos"]].copy()
    if report_df.empty:
        report_df = replay_df.copy()
    if report_df.empty:
        print("  [VOL Backtest] No rows available for report.")
        return

    report_df = report_df.sort_values("time").reset_index(drop=True)
    times = pd.to_datetime(report_df["time"])
    logvol_err = report_df["pred_next_yz_logvol"].to_numpy(dtype=float) - report_df[
        "next_yz_logvol"
    ].to_numpy(dtype=float)
    sigma = float(np.nanstd(logvol_err))
    if not np.isfinite(sigma):
        sigma = 0.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{symbol} VOL Backtest Replay ({len(report_df)} bars shown)",
        fontsize=14,
        y=0.98,
    )

    ax = axes[0, 0]
    pred_logvol = report_df["pred_next_yz_logvol"].to_numpy(dtype=float)
    ax.plot(
        times, report_df["next_yz_logvol"], color="black", linewidth=1.2, label="Actual"
    )
    ax.plot(times, pred_logvol, color="#1f77b4", linewidth=1.1, label="Forecast")
    ax.fill_between(
        times,
        pred_logvol - sigma,
        pred_logvol + sigma,
        color="#1f77b4",
        alpha=0.18,
        label="±1σ",
    )
    ax.set_title("Log-Vol Forecast vs Actual")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    ax.plot(
        times, report_df["next_log_range"], color="black", linewidth=1.2, label="Actual"
    )
    ax.plot(
        times,
        report_df["pred_next_log_range"],
        color="#2ca02c",
        linewidth=1.1,
        label="Forecast",
    )
    ax.set_title("Range Forecast vs Actual")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    ax.axis("off")
    if coverage_df.empty:
        ax.text(0.5, 0.5, "No coverage data", ha="center", va="center")
    else:
        table_rows = [
            [
                f"{row['quantile']:.0%}",
                f"{row['up_coverage']:.1%}",
                f"{row['dn_coverage']:.1%}",
                f"{row['up_pinball']:.4f}",
                f"{row['dn_pinball']:.4f}",
            ]
            for _, row in coverage_df.iterrows()
        ]
        table = ax.table(
            cellText=table_rows,
            colLabels=["Q", "Up Cov", "Dn Cov", "Up PB", "Dn PB"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
        ax.set_title("Excursion Band Coverage (Empirical vs Nominal)")

    ax = axes[1, 1]
    ml_err = np.abs(
        report_df["pred_next_yz_logvol"].to_numpy(dtype=float)
        - report_df["next_yz_logvol"].to_numpy(dtype=float)
    )
    ml_cum_mae = np.cumsum(ml_err) / np.arange(1, len(ml_err) + 1)
    ax.plot(times, ml_cum_mae, color="#1f77b4", linewidth=1.2, label="ML")

    har_pred = report_df["har_next_yz_logvol"].to_numpy(dtype=float)
    har_mask = np.isfinite(har_pred)
    if har_mask.any():
        har_err = np.abs(
            har_pred[har_mask]
            - report_df["next_yz_logvol"].to_numpy(dtype=float)[har_mask]
        )
        har_cum_mae = np.cumsum(har_err) / np.arange(1, len(har_err) + 1)
        ax.plot(
            times[har_mask], har_cum_mae, color="#d62728", linewidth=1.1, label="HAR-RV"
        )
    ax.set_title("Cumulative MAE Over Time")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Volatility/Range Backtest")
    parser.add_argument("--outdir", type=str, default="/home/km/Universal-ML/")
    parser.add_argument("--symbol", type=str, required=True)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.outdir)
    requested_symbol = args.symbol.upper()
    artifact_ctx = prepare_symbol_artifact_context(
        data_dir,
        requested_symbol,
        asset_class="SPOT",
        timeframes=("VOL",),
    )
    symbol = str(artifact_ctx["symbol"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    file_prefix = str(artifact_ctx["file_prefix"])
    artifact_paths_vol = get_artifact_paths(symbol_dir, file_prefix, "VOL")

    sys.path.append(os.path.join(data_dir, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("[!] FATAL: Cannot locate inference_bridge.py.")
        raise SystemExit(1)

    print("=" * 60)
    print("  VOL DAILY BACKTEST ENGINE")
    print("=" * 60)

    model_paths = {
        "next_yz_logvol": resolve_artifact_path(
            symbol_dir, file_prefix, "VOL", "model_logvol"
        ),
        "next_log_range": resolve_artifact_path(
            symbol_dir, file_prefix, "VOL", "model_range"
        ),
        "next_up_exc": resolve_artifact_path(
            symbol_dir, file_prefix, "VOL", "model_up_exc"
        ),
        "next_dn_exc": resolve_artifact_path(
            symbol_dir, file_prefix, "VOL", "model_dn_exc"
        ),
    }
    feat_path = resolve_artifact_path(symbol_dir, file_prefix, "VOL", "features")
    oos_path = resolve_artifact_path(symbol_dir, file_prefix, "VOL", "oos_forecasts")

    missing_models = [path for path in model_paths.values() if not os.path.exists(path)]
    if missing_models or not os.path.exists(feat_path):
        print("  [!] Missing VOL artifacts. Run daily_volatility_engine.py first.")
        raise SystemExit(1)

    models = {target: joblib.load(path) for target, path in model_paths.items()}
    with open(feat_path, encoding="utf-8") as handle:
        feature_cols = [line.strip() for line in handle if line.strip()]
    oos_map = joblib.load(oos_path) if os.path.exists(oos_path) else {}

    bridge = InferenceBridge(db_path=os.path.join(data_dir, "data_vault", "ohlcv.db"))
    tf_ctx = fetch_vol_timeframe_context(
        bridge,
        str(artifact_ctx["identity"].market_data_symbol),
    )
    primary_frames = tf_ctx["primary_frames"]
    reference_frames = tf_ctx["reference_frames"]
    df_1d = primary_frames["1D"]
    if df_1d is None or df_1d.empty:
        print(f"[!] FATAL: No 1D primary data for {symbol}")
        raise SystemExit(1)

    df_1h = tf_ctx["df_1h"]
    df_1h_raw = tf_ctx["df_1h_raw"]
    df_1h_status = tf_ctx["df_1h_status"]
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

    print("\n  [=] Reconstructing VOL feature space over historical data...")
    df_feature_frame = build_daily_volatility_feature_frame(
        df_1d,
        reference_1d=reference_frames["1D"],
        df_1h=df_1h,
        df_1w=df_1w,
        df_1m=df_1m,
        df_3m=df_3m,
        df_6m=df_6m,
        df_12m=df_12m,
        logger=print,
    )
    df_backtest, _ = prepare_vol_model_ready(df_feature_frame)

    for col in feature_cols:
        if col not in df_backtest.columns:
            df_backtest[col] = 0.0
        df_backtest[col] = (
            df_backtest[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    df_backtest = df_backtest.reset_index(drop=True)

    print(f"  [=] Total Daily Bars: {len(df_backtest)}")
    replay_df = replay_vol_forecasts(df_backtest, models, feature_cols, oos_map)
    replay_df["har_next_yz_logvol"] = _build_har_logvol_oos_series(df_backtest)

    oos_count = int(replay_df["is_oos"].sum())
    print(f"  [=] Saved OOS forecasts reused on {oos_count} bars.")

    report_scope = replay_df[replay_df["is_oos"]].copy()
    if report_scope.empty:
        report_scope = replay_df.copy()

    predictions = report_scope[["pred_next_up_exc", "pred_next_dn_exc"]].rename(
        columns={
            "pred_next_up_exc": "next_up_exc",
            "pred_next_dn_exc": "next_dn_exc",
        }
    )
    actuals = report_scope[["next_up_exc", "next_dn_exc"]]
    coverage_df = coverage_report(predictions, actuals)

    print("\n" + "=" * 60)
    print("  VOL DAILY REPLAY METRICS")
    print("=" * 60)
    for target in VOL_TARGETS:
        mae, rmse = _regression_metrics(
            report_scope[target],
            report_scope[f"pred_{target}"].to_numpy(dtype=float),
        )
        print(f"  {VOL_TARGET_LABELS[target]:<18}: MAE {mae:.4f} | RMSE {rmse:.4f}")

    logvol_mask = np.isfinite(report_scope["har_next_yz_logvol"].to_numpy(dtype=float))
    if logvol_mask.any():
        har_mae, har_rmse = _regression_metrics(
            report_scope.loc[logvol_mask, "next_yz_logvol"],
            report_scope.loc[logvol_mask, "har_next_yz_logvol"].to_numpy(dtype=float),
        )
        print(f"  HAR-RV Log-Vol Baseline: MAE {har_mae:.4f} | RMSE {har_rmse:.4f}")
    print(f"  Report Scope Bars      : {len(report_scope)}")
    print("=" * 60)

    save_vol_backtest_report(
        replay_df,
        coverage_df,
        artifact_paths_vol["backtest_report"],
        symbol=symbol,
    )
    print(f"\n  [✓] Report saved to {artifact_paths_vol['backtest_report']}")


if __name__ == "__main__":
    main()
