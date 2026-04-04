#!/usr/bin/env python3
"""
Accuracy guardrail for saved 1H/1D artifacts.

Purpose:
  - capture a local baseline of saved-artifact accuracy metrics
  - compare the current repo state against that baseline before trusting
    retrains, refactors, or runtime changes

This script does NOT retrain models. It rebuilds the model-ready frames from the
database, replays the saved OOS probability maps against the current target
series, and scores them with the same split-based contract used by the training
reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from daily_ml_engine import (
    BARRIER_ATR_MULT_DAILY,
    BARRIER_HORIZON_BARS_DAILY,
    MIN_TRAIN_BARS_DAILY,
    NON_FEATURE_COLS_DAILY,
    PURGE_GAP_DAILY,
    TEST_SIZE_RATIO_DAILY,
    _get_tf,
    _pick_primary_1d,
    add_daily_confluence,
    compute_macro_regime,
    holographic_feature_engine_daily,
)
from julia_bridge import (
    add_target_fast,
    holographic_feature_engine_fast,
    smc_feature_engine_daily,
    smc_feature_engine_fast,
)
from universal_ml_engine import (
    BARRIER_ATR_MULT,
    BARRIER_HORIZON_BARS,
    LIVE_CONFIDENCE_THRESHOLD,
    NON_FEATURE_COLS_SET,
    TRADE_PLAN_LABEL_COLS,
    _compute_atr14,
    fib_structural_basis,
    inject_thermodynamic_basis,
    merge_higher_tf,
    migrate_legacy_artifacts,
    prepare_intraday_thermodynamics,
)
from inference_bridge import InferenceBridge

DEFAULT_BASELINE_DIR = ".accuracy_baselines"
HASH_KEYS = (
    "model",
    "features",
    "oos_proba",
    "trade_plan_models",
    "ml_report",
    "backtest_report",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _lane_list(selected: str) -> list[str]:
    return ["1H", "1D"] if selected == "all" else [selected]


def _baseline_path(base_dir: str, symbol: str) -> Path:
    return Path(base_dir) / f"{symbol.upper()}.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_feature_cols(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def _sanitize_metrics(value: float) -> float | None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return float(value)


def _build_split_points(
    n_rows: int,
    *,
    min_train_bars: int,
    test_size_ratio: float,
    n_splits: int,
) -> list[tuple[int, int]]:
    split_points: list[tuple[int, int]] = []
    current_train_end = min_train_bars
    while current_train_end < n_rows:
        test_window_size = max(int(n_rows * test_size_ratio), 100)
        test_end = min(current_train_end + test_window_size, n_rows)
        if test_end <= current_train_end:
            break
        if (test_end - current_train_end) < 50:
            break
        split_points.append((current_train_end, test_end))
        current_train_end = test_end
        if len(split_points) >= n_splits:
            break
    return split_points


def _artifact_snapshot(paths: dict[str, str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key in HASH_KEYS:
        path = paths.get(key)
        if path is None:
            continue
        exists = os.path.exists(path)
        out[key] = {
            "path": path,
            "exists": exists,
            "size_bytes": os.path.getsize(path) if exists else None,
            "mtime_utc": (
                datetime.fromtimestamp(os.path.getmtime(path), timezone.utc)
                .replace(microsecond=0)
                .isoformat()
                if exists
                else None
            ),
            "sha256": _sha256(path) if exists else None,
        }
    return out


def _score_saved_oos(
    df: pd.DataFrame,
    feature_cols: list[str],
    oos_proba_map: dict[Any, float],
    *,
    min_train_bars: int,
    test_size_ratio: float,
    n_splits: int,
) -> dict[str, Any]:
    score_df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    split_points = _build_split_points(
        len(score_df),
        min_train_bars=min_train_bars,
        test_size_ratio=test_size_ratio,
        n_splits=n_splits,
    )
    if not split_points:
        raise ValueError("Could not determine valid split points for saved-artifact scoring.")

    oos_map = {pd.Timestamp(ts): float(prob) for ts, prob in oos_proba_map.items()}
    results: list[dict[str, Any]] = []
    total_missing = 0

    for idx, (train_end, test_end) in enumerate(split_points, start=1):
        y_test = score_df["target"].iloc[train_end:test_end]
        test_times = score_df["time"].iloc[train_end:test_end]
        preds = np.array(
            [oos_map.get(pd.Timestamp(ts), np.nan) for ts in test_times], dtype=float
        )
        valid_mask = np.isfinite(preds)
        missing = int((~valid_mask).sum())
        total_missing += missing
        if not valid_mask.any():
            raise ValueError(f"Split {idx} has zero valid OOS predictions.")

        true_binary = (y_test.to_numpy(dtype=float) > 0.5).astype(int)
        pred_binary = (preds > 0.5).astype(int)
        acc = float((true_binary[valid_mask] == pred_binary[valid_mask]).mean())
        high_conf_mask = valid_mask & (
            (preds >= LIVE_CONFIDENCE_THRESHOLD)
            | (preds <= (1.0 - LIVE_CONFIDENCE_THRESHOLD))
        )
        conf_acc = (
            float((pred_binary[high_conf_mask] == true_binary[high_conf_mask]).mean())
            if high_conf_mask.any()
            else float("nan")
        )
        results.append(
            {
                "split": idx,
                "train_bars": train_end,
                "test_bars": int(test_end - train_end),
                "accuracy": acc,
                "acc_high_conf": conf_acc,
                "high_conf_pct": float(high_conf_mask.sum() / len(preds)),
                "baseline_up": float(true_binary.mean()),
                "missing_oos_bars": missing,
            }
        )

    return {
        "split_count": len(results),
        "oos_prediction_bars": int(sum(r["test_bars"] for r in results) - total_missing),
        "total_validation_bars": int(sum(r["test_bars"] for r in results)),
        "missing_oos_bars": total_missing,
        "oos_coverage": float(
            1.0 - (total_missing / max(sum(r["test_bars"] for r in results), 1))
        ),
        "overall_accuracy": float(np.mean([r["accuracy"] for r in results])),
        "overall_high_conf_accuracy": _sanitize_metrics(
            float(np.nanmean([r["acc_high_conf"] for r in results]))
        ),
        "average_high_conf_pct": float(np.mean([r["high_conf_pct"] for r in results])),
        "overall_baseline_up": float(np.mean([r["baseline_up"] for r in results])),
        "edge_over_baseline": float(
            np.mean([r["accuracy"] for r in results])
            - np.mean([r["baseline_up"] for r in results])
        ),
        "splits": [
            {
                **r,
                "acc_high_conf": _sanitize_metrics(r["acc_high_conf"]),
            }
            for r in results
        ],
    }


def _inject_macro_regime(
    df_full: pd.DataFrame, df_macro: pd.DataFrame | None, label: str
) -> pd.DataFrame:
    if df_macro is None or df_macro.empty:
        for col in (
            f"{label}_secular_trend",
            f"{label}_range_pos",
            f"{label}_momentum_phase",
            f"{label}_vol_regime",
            f"{label}_vol_conviction",
        ):
            df_full[col] = 0.0
        return df_full

    macro = compute_macro_regime(df_macro, label)
    return pd.merge_asof(
        df_full.sort_values("time"),
        macro.sort_values("time"),
        on="time",
        direction="backward",
    )


def _load_tf_maps(data_dir: str, symbol: str) -> dict[str, dict[str, pd.DataFrame]]:
    bridge = InferenceBridge(db_path=os.path.join(data_dir, "data_vault", "ohlcv.db"))
    return {
        "FUT": bridge.fetch_holographic_stack(symbol, "FUT"),
        "SPOT": bridge.fetch_holographic_stack(symbol, "SPOT"),
    }


def _build_1h_model_ready(data_dir: str, symbol: str) -> pd.DataFrame:
    tf_maps = _load_tf_maps(data_dir, symbol)
    df_1h = tf_maps["FUT"].get("1H")
    df_1d = tf_maps["FUT"].get("1D")
    df_1w = tf_maps["FUT"].get("1W")
    df_1m = tf_maps["FUT"].get("1M")
    if df_1h is None or df_1h.empty:
        raise ValueError(f"No 1H FUT data available for {symbol}.")

    df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
        df_1h=df_1h,
        df_1d=df_1d,
        df_1w=df_1w,
        df_1m=df_1m,
        spot_1h=tf_maps["SPOT"].get("1H"),
        spot_1d=tf_maps["SPOT"].get("1D"),
        spot_1w=tf_maps["SPOT"].get("1W"),
        symbol=symbol,
        logger=None,
    )
    df_1h = fib_structural_basis(
        df_1h,
        htf_frames={"1D": df_1d, "1W": df_1w, "1M": df_1m},
        pairs=[("1D", "a"), ("1W", "b"), ("1M", "c")],
    )
    df_1h_labelled = _compute_atr14(df_1h.copy())
    df_full = holographic_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    smc_df = smc_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)
    df_full = add_target_fast(
        df_full,
        atr_mult=BARRIER_ATR_MULT,
        horizon=BARRIER_HORIZON_BARS,
        atr_col="atr14",
        drop_unresolved=False,
    )
    non_feature_cols = set(NON_FEATURE_COLS_SET)
    all_holo_cols = [c for c in df_full.columns if c not in non_feature_cols]
    state_cols = ["target", "time", "close", "atr14"]
    for col in ["basis_pct", "basis_z_score", "basis_vel_5", "basis_vel_10"]:
        if col in df_full.columns:
            state_cols.append(col)
    df_model_ready = df_full[
        all_holo_cols
        + state_cols
        + [c for c in TRADE_PLAN_LABEL_COLS if c in df_full.columns]
    ].copy()
    for col in all_holo_cols:
        df_model_ready[col] = (
            df_model_ready[col].map(lambda x: np.nan if np.isinf(x) else x).fillna(0)
        )
    return df_model_ready.dropna(subset=all_holo_cols).reset_index(drop=True)


def _build_1d_model_ready(data_dir: str, symbol: str) -> pd.DataFrame:
    tf_maps = _load_tf_maps(data_dir, symbol)
    df_1d = _pick_primary_1d(tf_maps)
    if df_1d is None or df_1d.empty:
        raise ValueError(f"No usable 1D data found for {symbol}.")
    df_1w = _get_tf(tf_maps, "1W")
    df_1m = _get_tf(tf_maps, "1M")
    df_3m = _get_tf(tf_maps, "3M")
    df_6m = _get_tf(tf_maps, "6M")
    df_12m = _get_tf(tf_maps, "12M")

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
    df_1d = fib_structural_basis(
        df_1d,
        htf_frames={"1W": df_1w, "1M": df_1m, "3M": df_3m},
        pairs=[("1W", "a"), ("1M", "b"), ("3M", "c")],
    )
    df_1d_labelled = _compute_atr14(df_1d.copy())
    df_full = holographic_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_3m)
    smc_df = smc_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    df_full = _inject_macro_regime(df_full, df_6m, "6m")
    df_full = _inject_macro_regime(df_full, df_12m, "12m")
    df_full = add_daily_confluence(df_full)
    df_full = add_target_fast(
        df_full,
        atr_mult=BARRIER_ATR_MULT_DAILY,
        horizon=BARRIER_HORIZON_BARS_DAILY,
        atr_col="atr14",
        drop_unresolved=False,
    )
    all_holo_cols = [c for c in df_full.columns if c not in NON_FEATURE_COLS_DAILY]
    state_cols = ["target", "time", "close", "atr14"]
    for col in (
        "basis_pct",
        "basis_z_score",
        "basis_vel_5",
        "basis_vel_10",
        "session_time_pos",
        "eod_basis_momentum",
    ):
        if col in df_full.columns:
            state_cols.append(col)
    df_model_ready = df_full[
        all_holo_cols
        + state_cols
        + [c for c in TRADE_PLAN_LABEL_COLS if c in df_full.columns]
    ].copy()
    for col in all_holo_cols:
        df_model_ready[col] = (
            df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    return df_model_ready.dropna(subset=all_holo_cols).reset_index(drop=True)


def _capture_lane_snapshot(data_dir: str, symbol: str, lane: str) -> dict[str, Any]:
    symbol_dir = os.path.join(data_dir, symbol)
    file_prefix = symbol.lower().replace(" ", "_")
    artifact_paths = migrate_legacy_artifacts(symbol_dir, file_prefix, lane, logger=None)

    features_path = artifact_paths["features"]
    oos_path = artifact_paths["oos_proba"]
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file for {lane}: {features_path}")
    if not os.path.exists(oos_path):
        raise FileNotFoundError(f"Missing OOS probability map for {lane}: {oos_path}")

    feature_cols = _read_feature_cols(features_path)
    oos_map = joblib.load(oos_path)
    if lane == "1H":
        df_model_ready = _build_1h_model_ready(data_dir, symbol)
        metrics = _score_saved_oos(
            df_model_ready,
            feature_cols,
            oos_map,
            min_train_bars=2500,
            test_size_ratio=0.15,
            n_splits=10,
        )
    else:
        df_model_ready = _build_1d_model_ready(data_dir, symbol)
        metrics = _score_saved_oos(
            df_model_ready,
            feature_cols,
            oos_map,
            min_train_bars=MIN_TRAIN_BARS_DAILY,
            test_size_ratio=TEST_SIZE_RATIO_DAILY,
            n_splits=10,
        )

    metrics["feature_count"] = len(feature_cols)
    metrics["frame_rows"] = len(df_model_ready)
    metrics["target_up_mean"] = float(df_model_ready["target"].mean())
    metrics["purge_gap"] = PURGE_GAP_DAILY if lane == "1D" else 24

    return {
        "lane": lane,
        "captured_at_utc": _utc_now(),
        "artifacts": _artifact_snapshot(artifact_paths),
        "metrics": metrics,
    }


def _print_lane_metrics(symbol: str, lane: str, metrics: dict[str, Any]) -> None:
    print(
        f"[{symbol} {lane}] acc={metrics['overall_accuracy']:.3f} "
        f"hc_acc={metrics['overall_high_conf_accuracy'] if metrics['overall_high_conf_accuracy'] is not None else 'NA'} "
        f"hc_pct={metrics['average_high_conf_pct']:.1%} "
        f"baseline={metrics['overall_baseline_up']:.3f} "
        f"edge={metrics['edge_over_baseline']:+.3f} "
        f"coverage={metrics['oos_coverage']:.1%} "
        f"features={metrics['feature_count']} "
        f"oos_bars={metrics['oos_prediction_bars']}"
    )


def capture_baseline(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline) if args.baseline else _baseline_path(args.base_dir, args.symbol)
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "schema_version": 1,
        "symbol": args.symbol.upper(),
        "outdir": os.path.abspath(args.outdir),
        "captured_at_utc": _utc_now(),
        "lanes": {},
    }
    if baseline_path.exists():
        try:
            existing_snapshot = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_snapshot = {}
        existing_lanes = existing_snapshot.get("lanes", {})
        if isinstance(existing_lanes, dict):
            snapshot["lanes"].update(existing_lanes)

    for lane in _lane_list(args.lane):
        print(f"[capture] rebuilding {lane} guardrail snapshot for {args.symbol.upper()}...")
        lane_snapshot = _capture_lane_snapshot(args.outdir, args.symbol.upper(), lane)
        snapshot["lanes"][lane] = lane_snapshot
        _print_lane_metrics(args.symbol.upper(), lane, lane_snapshot["metrics"])

    baseline_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"[capture] baseline written to {baseline_path}")
    return 0


def _metric_regressed(current: float | None, baseline: float | None, tol: float) -> bool:
    if baseline is None or current is None:
        return False
    return float(current) + tol < float(baseline)


def _artifact_identity_status(
    baseline_artifacts: dict[str, dict[str, Any]],
    current_artifacts: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    changed: list[str] = []
    for key in HASH_KEYS:
        if baseline_artifacts.get(key, {}).get("sha256") != current_artifacts.get(
            key, {}
        ).get("sha256"):
            changed.append(key)

    if changed:
        return "DIFFERENT RUN", changed
    return "SAME RUN", changed


def compare_baseline(args: argparse.Namespace) -> int:
    baseline_path = Path(args.baseline) if args.baseline else _baseline_path(args.base_dir, args.symbol)
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    symbol = args.symbol.upper()
    failures: list[str] = []

    for lane in _lane_list(args.lane):
        if lane not in baseline.get("lanes", {}):
            failures.append(f"{lane}: missing lane in baseline file")
            continue
        print(f"[compare] rebuilding {lane} guardrail snapshot for {symbol}...")
        current = _capture_lane_snapshot(args.outdir, symbol, lane)
        current_metrics = current["metrics"]
        base_metrics = baseline["lanes"][lane]["metrics"]
        base_artifacts = baseline["lanes"][lane]["artifacts"]
        current_artifacts = current["artifacts"]
        _print_lane_metrics(symbol, lane, current_metrics)
        run_status, changed_artifacts = _artifact_identity_status(
            base_artifacts,
            current_artifacts,
        )
        if changed_artifacts:
            print(
                f"[compare] {lane} artifact identity: {run_status} "
                f"(changed: {', '.join(changed_artifacts)})"
            )
        else:
            print(f"[compare] {lane} artifact identity: {run_status}")

        if current_metrics["oos_coverage"] + args.metric_tolerance < base_metrics["oos_coverage"]:
            failures.append(
                f"{lane}: OOS coverage regressed "
                f"{current_metrics['oos_coverage']:.3f} < {base_metrics['oos_coverage']:.3f}"
            )
        if _metric_regressed(
            current_metrics["overall_accuracy"],
            base_metrics["overall_accuracy"],
            args.metric_tolerance,
        ):
            failures.append(
                f"{lane}: overall_accuracy regressed "
                f"{current_metrics['overall_accuracy']:.3f} < {base_metrics['overall_accuracy']:.3f}"
            )
        if _metric_regressed(
            current_metrics["overall_high_conf_accuracy"],
            base_metrics["overall_high_conf_accuracy"],
            args.metric_tolerance,
        ):
            failures.append(
                f"{lane}: overall_high_conf_accuracy regressed "
                f"{current_metrics['overall_high_conf_accuracy']:.3f} < "
                f"{base_metrics['overall_high_conf_accuracy']:.3f}"
            )
        if _metric_regressed(
            current_metrics["edge_over_baseline"],
            base_metrics["edge_over_baseline"],
            args.metric_tolerance,
        ):
            failures.append(
                f"{lane}: edge_over_baseline regressed "
                f"{current_metrics['edge_over_baseline']:+.3f} < "
                f"{base_metrics['edge_over_baseline']:+.3f}"
            )

        for key, artifact in base_artifacts.items():
            current_artifact = current_artifacts.get(key, {})
            if artifact.get("exists") and artifact.get("sha256") != current_artifact.get("sha256"):
                print(
                    f"[compare] note: {lane} artifact changed for {key} "
                    f"({artifact.get('sha256', '')[:12]} -> {current_artifact.get('sha256', '')[:12]})"
                )

    if failures:
        print("[compare] FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[compare] PASS")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Saved-artifact accuracy guardrail")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Capture a local baseline snapshot")
    capture.add_argument("--symbol", required=True, help="Target symbol, e.g. NIFTY")
    capture.add_argument(
        "--outdir",
        default="/home/km/Universal-ML/",
        help="Project root / artifact root",
    )
    capture.add_argument(
        "--lane",
        choices=("1H", "1D", "all"),
        default="all",
        help="Lane to capture",
    )
    capture.add_argument(
        "--base-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Directory for local baseline files",
    )
    capture.add_argument(
        "--baseline",
        default=None,
        help="Optional explicit baseline output path",
    )
    capture.set_defaults(func=capture_baseline)

    compare = sub.add_parser("compare", help="Compare current state to a baseline")
    compare.add_argument("--symbol", required=True, help="Target symbol, e.g. NIFTY")
    compare.add_argument(
        "--outdir",
        default="/home/km/Universal-ML/",
        help="Project root / artifact root",
    )
    compare.add_argument(
        "--lane",
        choices=("1H", "1D", "all"),
        default="all",
        help="Lane to compare",
    )
    compare.add_argument(
        "--base-dir",
        default=DEFAULT_BASELINE_DIR,
        help="Directory for local baseline files",
    )
    compare.add_argument(
        "--baseline",
        default=None,
        help="Optional explicit baseline path",
    )
    compare.add_argument(
        "--metric-tolerance",
        type=float,
        default=1e-9,
        help="Allowed negative drift before compare fails",
    )
    compare.set_defaults(func=compare_baseline)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
