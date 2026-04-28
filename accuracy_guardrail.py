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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from daily_ml_engine import (
    BARRIER_ATR_MULT_DAILY,
    BARRIER_HORIZON_BARS_DAILY,
    LIVE_CONFIDENCE_THRESHOLD_DAILY,
    MIN_TRAIN_BARS_DAILY,
    NON_FEATURE_COLS_DAILY,
    PURGE_GAP_DAILY,
    TEST_SIZE_RATIO_DAILY,
    add_daily_confluence,
    compute_macro_regime,
    holographic_feature_engine_daily,
)
from daily_volatility_engine import (
    VOL_ALL_TARGETS,
    VOL_MIN_TRAIN_BARS,
    VOL_N_SPLITS,
    VOL_PURGE_GAP,
    VOL_QLIKE_TARGETS,
    VOL_TARGET_LABELS,
    _qlike_loss,
    _regression_metrics,
    build_daily_volatility_feature_frame,
    build_har_oos_series,
    enrich_vol_oos_frame,
    fetch_vol_timeframe_context,
    prepare_vol_model_ready,
)
from inference_bridge import InferenceBridge
from instrument_registry import resolve_instrument_identity
from julia_bridge import (
    add_target_fast,
    holographic_feature_engine_fast,
    kalman_structural_engine_daily,
    kalman_structural_engine_fast,
    narrative_context_engine_daily,
    narrative_context_engine_fast,
    rv_feature_engine_daily,
    rv_feature_engine_fast,
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
    apply_calibrator_to_prob_array,
    build_timeframe_selection,
    get_artifact_paths,
    inject_thermodynamic_basis,
    merge_higher_tf,
    prepare_intraday_thermodynamics,
    prepare_symbol_artifact_context,
)

DEFAULT_BASELINE_DIR = ".accuracy_baselines"
HASH_KEYS = (
    "model",
    "features",
    "oos_proba",
    "oos_forecasts",
    "calibrator",
    "calibrators",
    "conformal",
    "trade_plan_models",
    "model_logvol",
    "model_range",
    "model_up_exc",
    "model_dn_exc",
    "model_5d_logvol",
    "model_5d_range",
    "model_5d_up_exc",
    "model_5d_dn_exc",
    "exit_surface",
    "policy_artifact",
    "live_forecast",
    "ml_report",
    "backtest_report",
)


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _lane_list(selected: str) -> list[str]:
    return ["1H", "1D", "VOL"] if selected == "all" else [selected]


def _baseline_path(base_dir: str, symbol: str) -> Path:
    canonical_symbol = resolve_instrument_identity(symbol).canonical_symbol
    return Path(base_dir) / f"{canonical_symbol}.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_feature_cols(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
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
                datetime.fromtimestamp(os.path.getmtime(path), UTC)
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
    calibrator=None,
    confidence_threshold: float,
    min_train_bars: int,
    test_size_ratio: float,
    n_splits: int,
) -> dict[str, Any]:
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    score_df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    split_points = _build_split_points(
        len(score_df),
        min_train_bars=min_train_bars,
        test_size_ratio=test_size_ratio,
        n_splits=n_splits,
    )
    if not split_points:
        raise ValueError(
            "Could not determine valid split points for saved-artifact scoring."
        )

    oos_map = {pd.Timestamp(ts): float(prob) for ts, prob in oos_proba_map.items()}
    results: list[dict[str, Any]] = []
    total_missing = 0

    for idx, (train_end, test_end) in enumerate(split_points, start=1):
        y_test = score_df["target"].iloc[train_end:test_end]
        test_times = score_df["time"].iloc[train_end:test_end]
        preds = np.array(
            [oos_map.get(pd.Timestamp(ts), np.nan) for ts in test_times], dtype=float
        )
        preds = apply_calibrator_to_prob_array(preds, calibrator)
        valid_mask = np.isfinite(preds)
        missing = int((~valid_mask).sum())
        total_missing += missing
        if not valid_mask.any():
            raise ValueError(f"Split {idx} has zero valid OOS predictions.")

        true_binary = (y_test.to_numpy(dtype=float) > 0.5).astype(int)
        pred_binary = (preds > 0.5).astype(int)
        acc = float((true_binary[valid_mask] == pred_binary[valid_mask]).mean())
        high_conf_mask = valid_mask & (
            (preds >= confidence_threshold) | (preds <= (1.0 - confidence_threshold))
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
        "oos_prediction_bars": int(
            sum(r["test_bars"] for r in results) - total_missing
        ),
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
        "SPOT": bridge.fetch_holographic_stack(
            symbol,
            "SPOT",
            include_realized_vol=True,
        ),
    }


def _build_1h_model_ready(data_dir: str, symbol: str) -> pd.DataFrame:
    tf_maps = _load_tf_maps(data_dir, symbol)
    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1H", "1D", "1W", "1M")
    )
    df_1h = primary_frames["1H"]
    df_1d = primary_frames["1D"]
    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]
    if df_1h is None or df_1h.empty:
        raise ValueError(f"No 1H primary data available for {symbol}.")

    df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
        df_1h=df_1h,
        df_1d=df_1d,
        df_1w=df_1w,
        df_1m=df_1m,
        reference_1h=reference_frames["1H"],
        reference_1d=reference_frames["1D"],
        reference_1w=reference_frames["1W"],
        symbol=symbol,
        logger=None,
    )
    df_1h_labelled = _compute_atr14(df_1h.copy())
    df_full = holographic_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    smc_df = smc_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    kf_df = kalman_structural_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in kf_df.columns:
        df_full[col] = kf_df[col].values
    rv_df = rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
    nc_df = narrative_context_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in nc_df.columns:
        df_full[col] = nc_df[col].values
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
    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1D", "1W", "1M", "3M", "6M", "12M")
    )
    df_1d = primary_frames["1D"]
    if df_1d is None or df_1d.empty:
        raise ValueError(f"No usable 1D primary data found for {symbol}.")
    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]
    df_3m = primary_frames["3M"]
    df_6m = primary_frames["6M"]
    df_12m = primary_frames["12M"]

    df_1d = inject_thermodynamic_basis(
        df_1d,
        reference_frames["1D"],
        logger=None,
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


def _build_vol_model_ready(data_dir: str, symbol: str) -> pd.DataFrame:
    bridge = InferenceBridge(db_path=os.path.join(data_dir, "data_vault", "ohlcv.db"))
    tf_ctx = fetch_vol_timeframe_context(bridge, symbol)
    primary_frames = tf_ctx["primary_frames"]
    reference_frames = tf_ctx["reference_frames"]
    df_1d = primary_frames["1D"]
    if df_1d is None or df_1d.empty:
        raise ValueError(f"No usable 1D primary data found for {symbol}.")

    df_feature_frame = build_daily_volatility_feature_frame(
        df_1d,
        reference_1d=reference_frames["1D"],
        df_1h=tf_ctx["df_1h"],
        df_vix=tf_ctx["df_vix"],
        df_1w=primary_frames["1W"],
        df_1m=primary_frames["1M"],
        df_3m=primary_frames["3M"],
        df_6m=primary_frames["6M"],
        df_12m=primary_frames["12M"],
        logger=None,
    )
    df_model_ready, _ = prepare_vol_model_ready(df_feature_frame)
    return df_model_ready


def _metric_worsened(current: float | None, baseline: float | None, tol: float) -> bool:
    if baseline is None or current is None:
        return False
    return float(current) - tol > float(baseline)


def _mean_metric(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmean(arr))


def _winkler_score(
    actual: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    level: float,
) -> float:
    alpha = 1.0 - float(level)
    winkler = hi - lo
    below = actual < lo
    above = actual > hi
    if below.any():
        winkler[below] += (2.0 / alpha) * (lo[below] - actual[below])
    if above.any():
        winkler[above] += (2.0 / alpha) * (actual[above] - hi[above])
    return float(np.mean(winkler))


def _score_saved_vol_oos(
    df: pd.DataFrame,
    oos_forecast_map: dict[Any, Any],
    *,
    conformal_artifact: dict[str, Any] | None = None,
    calibrators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_targets = [target for target in VOL_ALL_TARGETS if target in df.columns]
    if not active_targets:
        raise ValueError("No VOL targets available in reconstructed model-ready frame.")

    replay_df = df[["time", *active_targets]].copy()
    for target in active_targets:
        replay_df[f"pred_{target}"] = np.nan

    saved_map = {
        pd.Timestamp(ts): payload
        for ts, payload in (oos_forecast_map or {}).items()
        if isinstance(payload, dict)
    }
    is_oos = np.zeros(len(replay_df), dtype=bool)
    for idx, ts in enumerate(replay_df["time"]):
        payload = saved_map.get(pd.Timestamp(ts))
        if not isinstance(payload, dict):
            continue
        is_oos[idx] = True
        for target in active_targets:
            value = payload.get(target)
            if value is not None:
                replay_df.at[idx, f"pred_{target}"] = float(value)
    replay_df["is_oos"] = is_oos

    har_models = (
        conformal_artifact.get("har_models", {})
        if isinstance(conformal_artifact, dict)
        else {}
    )
    for target in active_targets:
        replay_df[f"har_{target}"] = build_har_oos_series(
            df,
            target,
            n_splits=VOL_N_SPLITS,
            min_train_bars=VOL_MIN_TRAIN_BARS,
            purge_gap=VOL_PURGE_GAP,
            har_model=har_models.get(target),
        )

    replay_df = enrich_vol_oos_frame(
        replay_df,
        conformal_artifact,
        calibrators=calibrators,
        targets=active_targets,
    )
    report_scope = replay_df[replay_df["is_oos"]].copy()
    if report_scope.empty:
        raise ValueError("Saved VOL OOS forecast map produced zero matched bars.")

    target_metrics: dict[str, dict[str, Any]] = {}
    mae_values: list[float] = []
    rmse_values: list[float] = []
    edge_values: list[float] = []
    qlike_values: list[float] = []
    qlike_edge_values: list[float] = []
    picp_abs_errors: list[float] = []

    for target in active_targets:
        pred_col = (
            f"final_pred_{target}"
            if f"final_pred_{target}" in report_scope.columns
            else f"pred_{target}"
        )
        actual = report_scope[target].to_numpy(dtype=float)
        pred = report_scope[pred_col].to_numpy(dtype=float)
        mask = np.isfinite(actual) & np.isfinite(pred)
        sample_size = int(mask.sum())

        mae, rmse = _regression_metrics(actual[mask], pred[mask])
        bias = (
            float(np.mean(pred[mask] - actual[mask])) if sample_size else float("nan")
        )
        har_pred = report_scope.get(
            f"har_{target}",
            pd.Series(np.nan, index=report_scope.index),
        ).to_numpy(dtype=float)
        har_mask = mask & np.isfinite(har_pred)
        har_mae, har_rmse = _regression_metrics(actual[har_mask], har_pred[har_mask])
        edge_vs_har = (
            float(1.0 - (mae / har_mae))
            if np.isfinite(mae) and np.isfinite(har_mae) and har_mae > 0.0
            else float("nan")
        )

        target_row: dict[str, Any] = {
            "sample_size": sample_size,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "har_mae": har_mae,
            "har_rmse": har_rmse,
            "edge_vs_har": edge_vs_har,
        }

        if np.isfinite(mae):
            mae_values.append(mae)
        if np.isfinite(rmse):
            rmse_values.append(rmse)
        if np.isfinite(edge_vs_har):
            edge_values.append(edge_vs_har)

        if target in VOL_QLIKE_TARGETS:
            qlike = _qlike_loss(actual[mask], pred[mask])
            har_qlike = _qlike_loss(actual[har_mask], har_pred[har_mask])
            qlike_edge_vs_har = (
                float(1.0 - (qlike / har_qlike))
                if np.isfinite(qlike) and np.isfinite(har_qlike) and har_qlike > 0.0
                else float("nan")
            )
            target_row["qlike"] = qlike
            target_row["har_qlike"] = har_qlike
            target_row["qlike_edge_vs_har"] = qlike_edge_vs_har
            if np.isfinite(qlike):
                qlike_values.append(qlike)
            if np.isfinite(qlike_edge_vs_har):
                qlike_edge_values.append(qlike_edge_vs_har)

        lo_col = f"interval_lo_90_{target}"
        hi_col = f"interval_hi_90_{target}"
        if lo_col in report_scope.columns and hi_col in report_scope.columns:
            lo = report_scope[lo_col].to_numpy(dtype=float)
            hi = report_scope[hi_col].to_numpy(dtype=float)
            interval_mask = mask & np.isfinite(lo) & np.isfinite(hi)
            if interval_mask.any():
                picp = float(
                    np.mean(
                        (actual[interval_mask] >= lo[interval_mask])
                        & (actual[interval_mask] <= hi[interval_mask])
                    )
                )
                mpiw = float(np.mean(hi[interval_mask] - lo[interval_mask]))
                winkler = _winkler_score(
                    actual[interval_mask],
                    lo[interval_mask].copy(),
                    hi[interval_mask].copy(),
                    0.90,
                )
                picp_abs_error = abs(picp - 0.90)
                target_row["picp_90"] = picp
                target_row["mpiw_90"] = mpiw
                target_row["winkler_90"] = winkler
                target_row["picp_90_abs_error"] = picp_abs_error
                picp_abs_errors.append(picp_abs_error)

        target_metrics[target] = target_row

    saved_oos_bars = len(saved_map)
    matched_bars = int(report_scope.shape[0])
    return {
        "split_count": VOL_N_SPLITS,
        "oos_prediction_bars": matched_bars,
        "total_validation_bars": saved_oos_bars,
        "missing_oos_bars": max(saved_oos_bars - matched_bars, 0),
        "oos_coverage": float(matched_bars / max(saved_oos_bars, 1)),
        "overall_mae": _mean_metric(mae_values),
        "overall_rmse": _mean_metric(rmse_values),
        "overall_edge_vs_har": _mean_metric(edge_values),
        "overall_qlike": _mean_metric(qlike_values),
        "overall_qlike_edge_vs_har": _mean_metric(qlike_edge_values),
        "overall_picp_90_abs_error": _mean_metric(picp_abs_errors),
        "target_metrics": target_metrics,
    }


def _capture_lane_snapshot(data_dir: str, symbol: str, lane: str) -> dict[str, Any]:
    artifact_ctx = prepare_symbol_artifact_context(
        data_dir,
        symbol,
        asset_class="SPOT",
        timeframes=(lane,),
        logger=None,
    )
    symbol = str(artifact_ctx["symbol"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    file_prefix = str(artifact_ctx["file_prefix"])
    artifact_paths = get_artifact_paths(symbol_dir, file_prefix, lane)

    features_path = artifact_paths["features"]
    oos_key = "oos_forecasts" if lane == "VOL" else "oos_proba"
    oos_path = artifact_paths[oos_key]
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing features file for {lane}: {features_path}")
    if not os.path.exists(oos_path):
        raise FileNotFoundError(f"Missing OOS artifact for {lane}: {oos_path}")

    feature_cols = _read_feature_cols(features_path)
    oos_map = joblib.load(oos_path)
    if lane == "1H":
        calibrator_path = artifact_paths.get("calibrator")
        calibrator = (
            joblib.load(calibrator_path)
            if calibrator_path and os.path.exists(calibrator_path)
            else None
        )
        df_model_ready = _build_1h_model_ready(data_dir, symbol)
        metrics = _score_saved_oos(
            df_model_ready,
            feature_cols,
            oos_map,
            calibrator=calibrator,
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
            min_train_bars=2500,
            test_size_ratio=0.15,
            n_splits=10,
        )
    elif lane == "1D":
        calibrator_path = artifact_paths.get("calibrator")
        calibrator = (
            joblib.load(calibrator_path)
            if calibrator_path and os.path.exists(calibrator_path)
            else None
        )
        df_model_ready = _build_1d_model_ready(data_dir, symbol)
        metrics = _score_saved_oos(
            df_model_ready,
            feature_cols,
            oos_map,
            calibrator=calibrator,
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
            min_train_bars=MIN_TRAIN_BARS_DAILY,
            test_size_ratio=TEST_SIZE_RATIO_DAILY,
            n_splits=10,
        )
    else:
        calibrators_path = artifact_paths.get("calibrators")
        calibrators = (
            joblib.load(calibrators_path)
            if calibrators_path and os.path.exists(calibrators_path)
            else {}
        )
        conformal_path = artifact_paths.get("conformal")
        conformal_artifact = (
            joblib.load(conformal_path)
            if conformal_path and os.path.exists(conformal_path)
            else None
        )
        df_model_ready = _build_vol_model_ready(data_dir, symbol)
        metrics = _score_saved_vol_oos(
            df_model_ready,
            oos_map,
            conformal_artifact=conformal_artifact,
            calibrators=calibrators,
        )
        metrics["missing_feature_count"] = int(
            sum(1 for col in feature_cols if col not in df_model_ready.columns)
        )

    metrics["feature_count"] = len(feature_cols)
    metrics["frame_rows"] = len(df_model_ready)
    if lane != "VOL":
        metrics["target_up_mean"] = float(df_model_ready["target"].mean())
    metrics["purge_gap"] = (
        VOL_PURGE_GAP if lane == "VOL" else PURGE_GAP_DAILY if lane == "1D" else 24
    )

    return {
        "lane": lane,
        "captured_at_utc": _utc_now(),
        "artifacts": _artifact_snapshot(artifact_paths),
        "metrics": metrics,
    }


def _print_lane_metrics(symbol: str, lane: str, metrics: dict[str, Any]) -> None:
    if lane == "VOL":
        qlike_text = (
            f"{metrics['overall_qlike']:.4f}"
            if np.isfinite(float(metrics.get("overall_qlike", np.nan)))
            else "NA"
        )
        edge_text = (
            f"{metrics['overall_edge_vs_har']:+.1%}"
            if np.isfinite(float(metrics.get("overall_edge_vs_har", np.nan)))
            else "NA"
        )
        print(
            f"[{symbol} VOL] mae={metrics['overall_mae']:.4f} "
            f"rmse={metrics['overall_rmse']:.4f} "
            f"qlike={qlike_text} "
            f"edge={edge_text} "
            f"coverage={metrics['oos_coverage']:.1%} "
            f"features={metrics['feature_count']} "
            f"oos_bars={metrics['oos_prediction_bars']}/{metrics['total_validation_bars']} "
            f"missing_features={metrics.get('missing_feature_count', 0)}"
        )
        for target in VOL_ALL_TARGETS:
            target_metrics = metrics.get("target_metrics", {}).get(target)
            if not isinstance(target_metrics, dict):
                continue
            summary = (
                f"  [{VOL_TARGET_LABELS.get(target, target)}] "
                f"mae={target_metrics.get('mae', float('nan')):.4f} "
                f"rmse={target_metrics.get('rmse', float('nan')):.4f}"
            )
            if "qlike" in target_metrics:
                summary += f" qlike={target_metrics['qlike']:.4f}"
            if np.isfinite(float(target_metrics.get("edge_vs_har", np.nan))):
                summary += f" edge={target_metrics['edge_vs_har']:+.1%}"
            if np.isfinite(float(target_metrics.get("picp_90", np.nan))):
                summary += f" picp90={target_metrics['picp_90']:.1%}"
            print(summary)
        return

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
    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else _baseline_path(args.base_dir, args.symbol)
    )
    baseline_path.parent.mkdir(parents=True, exist_ok=True)

    snapshot = {
        "schema_version": 2,
        "symbol": resolve_instrument_identity(args.symbol).canonical_symbol,
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
        print(
            f"[capture] rebuilding {lane} guardrail snapshot for {snapshot['symbol']}..."
        )
        lane_snapshot = _capture_lane_snapshot(args.outdir, snapshot["symbol"], lane)
        snapshot["lanes"][lane] = lane_snapshot
        _print_lane_metrics(snapshot["symbol"], lane, lane_snapshot["metrics"])

    baseline_path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"[capture] baseline written to {baseline_path}")
    return 0


def _metric_regressed(
    current: float | None, baseline: float | None, tol: float
) -> bool:
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
    baseline_path = (
        Path(args.baseline)
        if args.baseline
        else _baseline_path(args.base_dir, args.symbol)
    )
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    symbol = resolve_instrument_identity(args.symbol).canonical_symbol
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

        if lane == "VOL":
            if (
                current_metrics["oos_coverage"] + args.metric_tolerance
                < base_metrics["oos_coverage"]
            ):
                failures.append(
                    f"{lane}: OOS coverage regressed "
                    f"{current_metrics['oos_coverage']:.3f} < {base_metrics['oos_coverage']:.3f}"
                )
            if current_metrics.get("missing_feature_count", 0) > base_metrics.get(
                "missing_feature_count", 0
            ):
                failures.append(
                    f"{lane}: missing_feature_count increased "
                    f"{current_metrics.get('missing_feature_count', 0)} > "
                    f"{base_metrics.get('missing_feature_count', 0)}"
                )
            for metric_key in (
                "overall_mae",
                "overall_rmse",
                "overall_qlike",
                "overall_picp_90_abs_error",
            ):
                if _metric_worsened(
                    current_metrics.get(metric_key),
                    base_metrics.get(metric_key),
                    args.metric_tolerance,
                ):
                    failures.append(
                        f"{lane}: {metric_key} regressed "
                        f"{current_metrics.get(metric_key):.4f} > "
                        f"{base_metrics.get(metric_key):.4f}"
                    )
            for metric_key in (
                "overall_edge_vs_har",
                "overall_qlike_edge_vs_har",
            ):
                if _metric_regressed(
                    current_metrics.get(metric_key),
                    base_metrics.get(metric_key),
                    args.metric_tolerance,
                ):
                    failures.append(
                        f"{lane}: {metric_key} regressed "
                        f"{current_metrics.get(metric_key):+.4f} < "
                        f"{base_metrics.get(metric_key):+.4f}"
                    )

            base_target_metrics = base_metrics.get("target_metrics", {})
            current_target_metrics = current_metrics.get("target_metrics", {})
            for target, base_target in base_target_metrics.items():
                current_target = current_target_metrics.get(target)
                if not isinstance(current_target, dict):
                    failures.append(f"{lane}: missing target metrics for {target}")
                    continue
                for metric_key in ("mae", "rmse", "qlike", "picp_90_abs_error"):
                    if metric_key not in base_target:
                        continue
                    if _metric_worsened(
                        current_target.get(metric_key),
                        base_target.get(metric_key),
                        args.metric_tolerance,
                    ):
                        failures.append(
                            f"{lane}: {target} {metric_key} regressed "
                            f"{current_target.get(metric_key):.4f} > "
                            f"{base_target.get(metric_key):.4f}"
                        )
                for metric_key in ("edge_vs_har", "qlike_edge_vs_har"):
                    if metric_key not in base_target:
                        continue
                    if _metric_regressed(
                        current_target.get(metric_key),
                        base_target.get(metric_key),
                        args.metric_tolerance,
                    ):
                        failures.append(
                            f"{lane}: {target} {metric_key} regressed "
                            f"{current_target.get(metric_key):+.4f} < "
                            f"{base_target.get(metric_key):+.4f}"
                        )

            for key, artifact in base_artifacts.items():
                current_artifact = current_artifacts.get(key, {})
                if artifact.get("exists") and artifact.get(
                    "sha256"
                ) != current_artifact.get("sha256"):
                    print(
                        f"[compare] note: {lane} artifact changed for {key} "
                        f"({artifact.get('sha256', '')[:12]} -> {current_artifact.get('sha256', '')[:12]})"
                    )
            continue

        if (
            current_metrics["oos_coverage"] + args.metric_tolerance
            < base_metrics["oos_coverage"]
        ):
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
            if artifact.get("exists") and artifact.get(
                "sha256"
            ) != current_artifact.get("sha256"):
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
        choices=("1H", "1D", "VOL", "all"),
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
        choices=("1H", "1D", "VOL", "all"),
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
