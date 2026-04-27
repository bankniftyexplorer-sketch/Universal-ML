"""
daily_volatility_engine.py — Daily Volatility/Range Forecasting Lane
=====================================================================
Predicts next-day volatility level and directional excursion bands.
Separate lane from daily_ml_engine.py (direction model).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from daily_ml_engine import (
    NON_FEATURE_COLS_DAILY,
    add_daily_confluence,
    inject_macro_regime,
)
from holographic_engine import correlation_filter, phase1_ranking
from julia_bridge import (
    holographic_feature_engine_daily,
    intraday_rv_summary_daily,
    kalman_structural_engine_daily,
    narrative_context_engine_daily,
    rv_feature_engine_daily,
    smc_feature_engine_daily,
    vol_target_engine_daily,
)
from universal_ml_engine import (
    MODEL_N_JOBS,
    EnsembleModel,
    _compute_atr14,
    _load_hyperparams_config,
    build_timeframe_selection,
    describe_selected_frame,
    get_artifact_paths,
    inject_thermodynamic_basis,
    prepare_symbol_artifact_context,
    select_primary_timeframe,
)

warnings.filterwarnings("ignore")

VOL_TARGETS = ["next_yz_logvol", "next_log_range", "next_up_exc", "next_dn_exc"]
VOL_MIN_TRAIN_BARS = 500
VOL_TEST_SIZE_RATIO = 0.12
VOL_PURGE_GAP = 1
VOL_FEAT_BUDGET = 40
VOL_N_SPLITS = 8
VOL_OPTIONAL_FAIL_TIMEFRAMES = ("1H",)

VOL_TARGET_LABELS = {
    "next_yz_logvol": "YZ Log-Vol",
    "next_log_range": "Log-Range",
    "next_up_exc": "Up-Excursion",
    "next_dn_exc": "Down-Excursion",
}
VOL_TARGET_SHORT = {
    "next_yz_logvol": "Vol",
    "next_log_range": "Range",
    "next_up_exc": "Up",
    "next_dn_exc": "Dn",
}
VOL_MODEL_ARTIFACT_KEYS = {
    "next_yz_logvol": "model_logvol",
    "next_log_range": "model_range",
    "next_up_exc": "model_up_exc",
    "next_dn_exc": "model_dn_exc",
}
NON_FEATURE_COLS_VOL = set(NON_FEATURE_COLS_DAILY) | set(VOL_TARGETS)


def _print_tf_span(label: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    print(
        f"  {label} bars : {len(df):>7}  "
        f"({df['time'].min().date()} → {df['time'].max().date()})"
    )


def _quality_status_from_frame(df: pd.DataFrame | None) -> str | None:
    if df is None or df.empty:
        return None
    quality = (
        df.attrs.get("data_quality")
        if isinstance(getattr(df, "attrs", None), dict)
        else None
    )
    if not isinstance(quality, dict):
        return None
    status = quality.get("status")
    if status is None:
        status = quality.get("quality_status")
    if status is None:
        return None
    return str(status).strip().upper() or None


def _vol_split_points(
    n: int,
    n_splits: int = VOL_N_SPLITS,
    min_train_bars: int = VOL_MIN_TRAIN_BARS,
    test_size_ratio: float = VOL_TEST_SIZE_RATIO,
) -> list[tuple[int, int]]:
    split_points: list[tuple[int, int]] = []
    current_train_end = min_train_bars
    while current_train_end < n:
        test_window_size = max(int(n * test_size_ratio), 100)
        test_end = min(current_train_end + test_window_size, n)
        if test_end <= current_train_end:
            break
        if (test_end - current_train_end) < 50:
            break
        split_points.append((current_train_end, test_end))
        current_train_end = test_end
        if len(split_points) >= n_splits:
            break
    return split_points


def _build_lgbm_ensemble_vol() -> list[lgb.LGBMRegressor]:
    specs = _load_hyperparams_config()["classifier_ensemble"]
    if not isinstance(specs, list) or not specs:
        raise ValueError("`classifier_ensemble` must be a non-empty list.")

    models: list[lgb.LGBMRegressor] = []
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"`classifier_ensemble[{idx - 1}]` must be an object.")
        params = dict(spec)
        params.update(
            {
                "objective": "mae",
                "metric": "mae",
                "n_jobs": MODEL_N_JOBS,
                "verbose": -1,
            }
        )
        models.append(lgb.LGBMRegressor(**params))
    return models


def _fit_lgbm_with_inner_validation_vol(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    purge_gap: int = VOL_PURGE_GAP,
) -> tuple[EnsembleModel, pd.DataFrame, pd.Series]:
    inner_val_size = max(100, int(len(X_train_full) * 0.15))
    inner_val_size = min(inner_val_size, len(X_train_full) - 1)
    if inner_val_size <= 0:
        raise ValueError("Not enough training rows for inner validation.")

    purged_end = -(inner_val_size + purge_gap)
    if abs(purged_end) >= len(X_train_full):
        purged_end = 0

    X_train_inner = (
        X_train_full.iloc[:purged_end]
        if purged_end != 0
        else X_train_full.iloc[:-inner_val_size]
    )
    y_train_inner = (
        y_train_full.iloc[:purged_end]
        if purged_end != 0
        else y_train_full.iloc[:-inner_val_size]
    )
    X_val_inner = X_train_full.iloc[-inner_val_size:]
    y_val_inner = y_train_full.iloc[-inner_val_size:]

    fitted_models: list[lgb.LGBMRegressor] = []
    for model in _build_lgbm_ensemble_vol():
        model.fit(
            X_train_inner,
            y_train_inner,
            eval_set=[(X_val_inner, y_val_inner)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=60, verbose=False),
                lgb.log_evaluation(period=10000),
            ],
        )
        fitted_models.append(model)
    return EnsembleModel(fitted_models), X_val_inner, y_val_inner


def _regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return float("nan"), float("nan")
    err = pred[mask] - true[mask]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    return mae, rmse


def _nanmean_metric(rows: list[dict], key: str) -> float:
    values = np.array([row.get(key, np.nan) for row in rows], dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return float("nan")
    return float(np.nanmean(values))


def _single_row_feature_frame(
    row: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    data = {
        col: float(row[col]) if col in row.index and pd.notna(row[col]) else 0.0
        for col in feature_cols
    }
    return pd.DataFrame([data], columns=feature_cols)


def predict_volatility_heads(
    models: dict[str, EnsembleModel],
    feature_cols: list[str],
    row: pd.Series,
) -> dict[str, float]:
    X_row = _single_row_feature_frame(row, feature_cols)
    return {target: float(models[target].predict(X_row)[0]) for target in VOL_TARGETS}


def build_vol_forecast_payload(
    *,
    symbol: str,
    row: pd.Series,
    forecasts: dict[str, float],
    intraday_1h_used: bool,
    reference_price: float | None = None,
    reference_price_source: str = "latest_close",
) -> dict[str, object]:
    latest_close = float(row["close"])
    ref_price = latest_close if reference_price is None else float(reference_price)
    up_mult = float(np.exp(forecasts["next_up_exc"]))
    dn_mult = float(np.exp(forecasts["next_dn_exc"]))
    payload = {
        "symbol": symbol,
        "lane": "VOL",
        "horizon": "next_daily_bar",
        "basis_bar_time": str(pd.Timestamp(row["time"])),
        "latest_close": latest_close,
        "reference_price": ref_price,
        "reference_price_source": reference_price_source,
        "intraday_1h_used": bool(intraday_1h_used),
        "yz_logvol": float(forecasts["next_yz_logvol"]),
        "annualized_vol_pct": float(np.expm1(forecasts["next_yz_logvol"]) * 100.0),
        "log_range": float(forecasts["next_log_range"]),
        "range_pct": float(np.expm1(forecasts["next_log_range"]) * 100.0),
        "up_exc": float(forecasts["next_up_exc"]),
        "up_bps": float(np.expm1(forecasts["next_up_exc"]) * 10000.0),
        "dn_exc": float(forecasts["next_dn_exc"]),
        "dn_bps": float(np.expm1(forecasts["next_dn_exc"]) * 10000.0),
        "projected_high": float(ref_price * up_mult),
        "projected_low": float(ref_price / dn_mult),
    }
    return payload


def save_vol_forecast_json(payload: dict[str, object], save_path: str) -> None:
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def print_vol_forecast(
    payload: dict[str, object],
    *,
    title: str = "Next-Day Volatility Forecast",
) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(f"  Basis Bar Time:    {payload['basis_bar_time']}")
    print(
        f"  YZ Log-Vol:        {payload['yz_logvol']:.4f}  "
        f"(annualized: {payload['annualized_vol_pct']:.1f}%)"
    )
    print(
        f"  Log-Range:         {payload['log_range']:.4f}  "
        f"(range: {payload['range_pct']:.2f}%)"
    )
    print(
        f"  Up-Excursion:      {payload['up_exc']:.4f}  "
        f"({payload['up_bps']:.0f} bps from open)"
    )
    print(
        f"  Down-Excursion:    {payload['dn_exc']:.4f}  "
        f"({payload['dn_bps']:.0f} bps from open)"
    )
    print(
        f"  Ref Price ({payload['reference_price_source']}): "
        f"{payload['reference_price']:,.2f}"
    )
    print(f"  Projected High:    {payload['projected_high']:,.2f}")
    print(f"  Projected Low:     {payload['projected_low']:,.2f}")
    if not payload.get("intraday_1h_used", False):
        print(
            "  [1H optional reference unavailable or failed quality; intra_* features zeroed]"
        )
    print("======================================================================")


def _har_rv_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
) -> np.ndarray:
    """Ridge-regularized HAR-RV on daily/weekly/monthly variance components."""
    har_cols = ["har_rv_d", "har_rv_w", "har_rv_m"]
    X_train = train_df[har_cols].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    train_mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)

    fallback_value = float(np.nanmean(y_train[train_mask])) if train_mask.any() else 0.0
    predictions = np.full(len(test_df), fallback_value, dtype=float)
    if train_mask.sum() < 25:
        return predictions

    X_test = test_df[har_cols].to_numpy(dtype=float)
    test_mask = np.all(np.isfinite(X_test), axis=1)
    if not test_mask.any():
        return predictions

    model = Ridge(alpha=1e-4)
    model.fit(X_train[train_mask], y_train[train_mask])
    predictions[test_mask] = model.predict(X_test[test_mask])
    return predictions


def fetch_vol_timeframe_context(
    bridge,
    market_data_symbol: str,
) -> dict[str, object]:
    tf_maps = {
        "SPOT": bridge.fetch_holographic_stack(
            market_data_symbol,
            "SPOT",
            include_realized_vol=True,
            allow_fail_timeframes=VOL_OPTIONAL_FAIL_TIMEFRAMES,
        ),
    }
    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1D", "1W", "1M", "3M", "6M", "12M")
    )
    df_1h_raw = select_primary_timeframe(tf_maps, "1H")
    df_1h_status = _quality_status_from_frame(df_1h_raw)
    df_1h = (
        None
        if df_1h_status == "FAIL" or df_1h_raw is None or df_1h_raw.empty
        else df_1h_raw
    )
    return {
        "tf_maps": tf_maps,
        "primary_frames": primary_frames,
        "reference_frames": reference_frames,
        "df_1h": df_1h,
        "df_1h_raw": df_1h_raw,
        "df_1h_status": df_1h_status,
    }


def build_daily_volatility_feature_frame(
    df_1d: pd.DataFrame,
    *,
    reference_1d: pd.DataFrame | None = None,
    df_1h: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_3m: pd.DataFrame | None = None,
    df_6m: pd.DataFrame | None = None,
    df_12m: pd.DataFrame | None = None,
    logger=print,
) -> pd.DataFrame:
    log = logger if logger is not None else (lambda *_args, **_kwargs: None)

    log("\n  [TOON VOL] Injecting Universal Basis Mechanics...")
    base_df = inject_thermodynamic_basis(df_1d, reference_1d, logger=logger)
    base_df["session_time_pos"] = 0.0
    base_df["eod_basis_momentum"] = 0.0

    log("  [TOON VOL] Building RV Surface (Julia RV engine)...")
    rv_df = rv_feature_engine_daily(base_df, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        base_df[col] = rv_df[col].values
    log(f"  [TOON VOL] RV features injected: {len(rv_df.columns)} columns")

    df_1d_labelled = _compute_atr14(base_df.copy())

    log("  [TOON VOL] Layer 1: Julia holographic extraction (1D + macro HTFs)...")
    df_full = holographic_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_3m)

    log("  [TOON VOL] Layer 1b: SMC Feature Engine...")
    smc_df = smc_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values

    log("  [TOON VOL] Layer 1c: Kalman structural feature family...")
    kf_df = kalman_structural_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in kf_df.columns:
        df_full[col] = kf_df[col].values

    log("  [TOON VOL] Layer 1d: Narrative Context Awareness...")
    nc_df = narrative_context_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in nc_df.columns:
        df_full[col] = nc_df[col].values

    log("  [TOON VOL] Layer 1e: Intraday RV summary from 1H bars...")
    if df_1h is None or df_1h.empty:
        log(
            "  [TOON VOL] 1H reference unavailable or intentionally ignored; "
            "intra_* features will be zeroed."
        )
    intra_df = intraday_rv_summary_daily(df_1d_labelled, df_1h)
    for col in intra_df.columns:
        df_full[col] = intra_df[col].values
    log(f"  [TOON VOL] Intraday RV features injected: {len(intra_df.columns)} columns")

    log("  [TOON VOL] Layer 2: Injecting macro regime overlays (6M/12M)...")
    df_full = inject_macro_regime(df_full, df_6m, "6m")
    df_full = inject_macro_regime(df_full, df_12m, "12m")

    log("  [TOON VOL] Layer 3: Adding daily confluence...")
    df_full = add_daily_confluence(df_full)
    return df_full


def add_har_rv_components(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "rv_1d_yz_log" not in df.columns:
        raise ValueError("Missing rv_1d_yz_log; cannot construct HAR-RV features.")

    har_rv_d = np.maximum(np.exp(df["rv_1d_yz_log"].to_numpy(dtype=float)) - 1.0, 0.0)
    df["har_rv_d"] = (har_rv_d * har_rv_d) / 252.0
    df["har_rv_w"] = (
        pd.Series(df["har_rv_d"]).rolling(5, min_periods=1).mean().to_numpy(dtype=float)
    )
    df["har_rv_m"] = (
        pd.Series(df["har_rv_d"])
        .rolling(22, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    return df


def prepare_vol_model_ready(
    df_full: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    df_full = vol_target_engine_daily(df_full)
    df_full = add_har_rv_components(df_full)

    candidate_feature_cols = [
        col for col in df_full.columns if col not in NON_FEATURE_COLS_VOL
    ]
    state_cols = [
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr14",
        *VOL_TARGETS,
    ]
    keep_cols = candidate_feature_cols + [c for c in state_cols if c in df_full.columns]
    df_model_ready = df_full[keep_cols].copy()

    for col in candidate_feature_cols:
        df_model_ready[col] = (
            df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    df_model_ready = df_model_ready.dropna(subset=VOL_TARGETS).reset_index(drop=True)
    return df_model_ready, candidate_feature_cols


def prepare_vol_inference_frame(
    df_feature_frame: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    df_infer = add_har_rv_components(df_feature_frame.copy())
    for col in feature_cols:
        if col not in df_infer.columns:
            df_infer[col] = 0.0
        df_infer[col] = df_infer[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df_infer.reset_index(drop=True)


def walk_forward_vol(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = VOL_N_SPLITS,
    min_train_bars: int = VOL_MIN_TRAIN_BARS,
    test_size_ratio: float = VOL_TEST_SIZE_RATIO,
    purge_gap: int = VOL_PURGE_GAP,
) -> dict:
    df = df.dropna(subset=feature_cols + VOL_TARGETS).reset_index(drop=True)
    n = len(df)

    if n < min_train_bars:
        print(
            f"  Error: Not enough data for VOL walk-forward. Need {min_train_bars}, got {n}."
        )
        return {}

    split_points = _vol_split_points(
        n,
        n_splits=n_splits,
        min_train_bars=min_train_bars,
        test_size_ratio=test_size_ratio,
    )
    if not split_points:
        print("  Error: Could not determine valid split points for VOL walk-forward.")
        return {}

    print(
        f"\n  VOL walk-forward: {len(split_points)} splits, "
        f"min_train_bars={min_train_bars}, test_ratio={test_size_ratio:.2f}, "
        f"purge_gap={purge_gap}"
    )
    print(f"  Total daily bars: {n}")

    results: list[dict[str, float | int]] = []
    oos_forecast_map: dict[pd.Timestamp, dict[str, float]] = {}
    oos_rows: list[dict[str, float | pd.Timestamp]] = []
    feature_importance_sum = np.zeros(len(feature_cols), dtype=float)

    for i, (train_end, test_end) in enumerate(split_points):
        purged_train_end = max(0, train_end - purge_gap)
        train_df = df.iloc[:purged_train_end].copy()
        test_df = df.iloc[train_end:test_end].copy()
        if len(test_df) == 0:
            continue

        X_train_full = train_df[feature_cols]
        X_test = test_df[feature_cols]

        fold_models: dict[str, EnsembleModel] = {}
        fold_preds: dict[str, np.ndarray] = {}
        fold_baselines: dict[str, np.ndarray] = {}
        fold_result: dict[str, float | int] = {
            "split": i + 1,
            "train_bars": len(X_train_full),
            "test_bars": len(X_test),
        }

        for target in VOL_TARGETS:
            model, _, _ = _fit_lgbm_with_inner_validation_vol(
                X_train_full,
                train_df[target],
                purge_gap=purge_gap,
            )
            preds = np.asarray(model.predict(X_test), dtype=float)
            har_preds = _har_rv_baseline(train_df, test_df, target)

            fold_models[target] = model
            fold_preds[target] = preds
            fold_baselines[target] = har_preds

            mae, rmse = _regression_metrics(test_df[target], preds)
            har_mae, har_rmse = _regression_metrics(test_df[target], har_preds)
            edge_vs_har = (
                float(1.0 - (mae / har_mae))
                if np.isfinite(mae) and np.isfinite(har_mae) and har_mae > 0.0
                else float("nan")
            )
            fold_result[f"{target}_mae"] = mae
            fold_result[f"{target}_rmse"] = rmse
            fold_result[f"{target}_har_mae"] = har_mae
            fold_result[f"{target}_har_rmse"] = har_rmse
            fold_result[f"{target}_edge_vs_har"] = edge_vs_har

        feature_importance_sum += fold_models["next_yz_logvol"].feature_importances_

        pieces = []
        for target in VOL_TARGETS:
            pieces.append(
                f"{VOL_TARGET_SHORT[target]} {fold_result[f'{target}_mae']:.4f}"
                f" vs HAR {fold_result[f'{target}_har_mae']:.4f}"
                f" ({fold_result[f'{target}_edge_vs_har']:+.1%})"
            )
        print(
            f"  Split {i + 1:>2} | Train:{len(X_train_full):>5} "
            f"Test:{len(X_test):>5} | {' | '.join(pieces)}"
        )

        for row_idx, ts in enumerate(test_df["time"]):
            ts_key = pd.Timestamp(ts)
            row: dict[str, float | pd.Timestamp] = {"time": ts_key}
            row_map = oos_forecast_map.setdefault(ts_key, {})
            for target in VOL_TARGETS:
                pred_val = float(fold_preds[target][row_idx])
                row_map[target] = pred_val
                row[f"pred_{target}"] = pred_val
                row[f"actual_{target}"] = float(test_df[target].iloc[row_idx])
                row[f"har_{target}"] = float(fold_baselines[target][row_idx])
            oos_rows.append(row)

        results.append(fold_result)

    final_models = {
        target: _fit_lgbm_with_inner_validation_vol(
            df[feature_cols],
            df[target],
            purge_gap=purge_gap,
        )[0]
        for target in VOL_TARGETS
    }
    split_count = max(len(results), 1)
    oos_frame = (
        pd.DataFrame(oos_rows).sort_values("time").drop_duplicates("time", keep="last")
    )
    overall_metrics = {
        target: {
            "mae": _nanmean_metric(results, f"{target}_mae"),
            "rmse": _nanmean_metric(results, f"{target}_rmse"),
            "har_mae": _nanmean_metric(results, f"{target}_har_mae"),
            "har_rmse": _nanmean_metric(results, f"{target}_har_rmse"),
            "edge_vs_har": _nanmean_metric(results, f"{target}_edge_vs_har"),
        }
        for target in VOL_TARGETS
    }

    return {
        "splits": results,
        "feature_importance": dict(
            zip(feature_cols, feature_importance_sum / split_count)
        ),
        "feature_cols": feature_cols,
        "df": df,
        "final_models": final_models,
        "oos_forecast_map": oos_forecast_map,
        "oos_frame": oos_frame,
        "overall_metrics": overall_metrics,
    }


def fold_consensus_feature_selection_vol(
    df: pd.DataFrame,
    candidate_feature_cols: list[str],
    n_splits: int = VOL_N_SPLITS,
    min_train_bars: int = VOL_MIN_TRAIN_BARS,
    test_size_ratio: float = VOL_TEST_SIZE_RATIO,
    purge_gap: int = VOL_PURGE_GAP,
    consensus_threshold: float = 0.50,
    target_col: str = "next_yz_logvol",
) -> tuple[list[str], dict]:
    df = df.dropna(
        subset=[c for c in candidate_feature_cols if c in df.columns] + [target_col]
    ).reset_index(drop=True)
    n = len(df)

    if n < min_train_bars:
        print(f"  [Nested VOL] Not enough data. Need {min_train_bars}, got {n}.")
        return [], {}

    split_points = _vol_split_points(
        n,
        n_splits=n_splits,
        min_train_bars=min_train_bars,
        test_size_ratio=test_size_ratio,
    )
    if not split_points:
        print("  [Nested VOL] No valid split points.")
        return [], {}

    valid_candidates = [c for c in candidate_feature_cols if c in df.columns]
    print(
        f"\n  [Nested VOL] {len(split_points)} folds, {len(valid_candidates)} candidates"
    )

    feature_votes: dict[str, int] = {}
    fold_results: list[dict[str, float | int]] = []

    for i, (train_end, test_end) in enumerate(split_points):
        purged_train_end = max(0, train_end - purge_gap)
        train_df = df.iloc[:purged_train_end]
        test_df = df.iloc[train_end:test_end]

        if len(test_df) == 0 or len(train_df) < 200:
            continue

        try:
            fold_features = correlation_filter(train_df, valid_candidates)
            fold_features = phase1_ranking(
                train_df, fold_features, target_col=target_col
            )
            fold_features = fold_features[:VOL_FEAT_BUDGET]
        except Exception as exc:
            print(f"  [Nested VOL] Fold {i + 1} feature selection failed: {exc}")
            continue

        if not fold_features:
            continue

        for feat in fold_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1

        X_train = train_df[fold_features]
        y_train = train_df[target_col]
        try:
            model, _, _ = _fit_lgbm_with_inner_validation_vol(
                X_train,
                y_train,
                purge_gap=purge_gap,
            )
        except Exception as exc:
            print(f"  [Nested VOL] Fold {i + 1} training failed: {exc}")
            continue

        preds = np.asarray(model.predict(test_df[fold_features]), dtype=float)
        har_preds = _har_rv_baseline(train_df, test_df, target_col)
        mae, rmse = _regression_metrics(test_df[target_col], preds)
        har_mae, har_rmse = _regression_metrics(test_df[target_col], har_preds)
        edge_vs_har = (
            float(1.0 - (mae / har_mae))
            if np.isfinite(mae) and np.isfinite(har_mae) and har_mae > 0.0
            else float("nan")
        )

        print(
            f"  [Nested VOL] Fold {i + 1:>2} | Train:{len(X_train):>5} "
            f"Test:{len(test_df):>5} | MAE:{mae:.4f} RMSE:{rmse:.4f} "
            f"HAR:{har_mae:.4f} Edge:{edge_vs_har:+.1%} | Feats:{len(fold_features)}"
        )
        fold_results.append(
            {
                "split": i + 1,
                "train_bars": len(X_train),
                "test_bars": len(test_df),
                "mae": mae,
                "rmse": rmse,
                "har_mae": har_mae,
                "har_rmse": har_rmse,
                "edge_vs_har": edge_vs_har,
            }
        )

    min_votes = max(1, int(len(split_points) * consensus_threshold))
    consensus_features = [
        feat
        for feat, votes in sorted(feature_votes.items(), key=lambda item: -item[1])
        if votes >= min_votes
    ][:VOL_FEAT_BUDGET]

    print(
        f"\n  [Nested VOL] Consensus: {len(consensus_features)} features "
        f"(voted in >={min_votes}/{len(split_points)} folds)"
    )
    if not consensus_features:
        print("  [Nested VOL] FATAL: No features survived consensus vote.")
        return [], {}

    print("  [Nested VOL] Final walk-forward with consensus features...")
    final_wf = walk_forward_vol(
        df,
        consensus_features,
        n_splits=n_splits,
        min_train_bars=min_train_bars,
        test_size_ratio=test_size_ratio,
        purge_gap=purge_gap,
    )
    if final_wf:
        final_wf["nested_fold_results"] = fold_results
        final_wf["feature_votes"] = feature_votes
    return consensus_features, final_wf


def save_vol_report(
    wf_results: dict,
    save_path: str,
    *,
    symbol: str,
) -> None:
    oos_frame = wf_results.get("oos_frame")
    if oos_frame is None or len(oos_frame) == 0:
        print("  [VOL Report] No OOS frame available to plot.")
        return

    times = pd.to_datetime(oos_frame["time"])
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"{symbol} Daily Volatility — Walk-Forward OOS "
        f"({len(wf_results['splits'])} folds, {len(oos_frame)} bars)",
        fontsize=14,
        y=0.98,
    )

    logvol_metrics = wf_results["overall_metrics"]["next_yz_logvol"]
    range_metrics = wf_results["overall_metrics"]["next_log_range"]
    up_metrics = wf_results["overall_metrics"]["next_up_exc"]
    dn_metrics = wf_results["overall_metrics"]["next_dn_exc"]

    ax = axes[0, 0]
    ax.plot(
        times,
        oos_frame["actual_next_yz_logvol"],
        color="black",
        linewidth=1.2,
        label="Actual",
    )
    ax.plot(
        times,
        oos_frame["pred_next_yz_logvol"],
        color="#1f77b4",
        linewidth=1.1,
        label="ML",
    )
    ax.plot(
        times,
        oos_frame["har_next_yz_logvol"],
        color="#d62728",
        linewidth=1.0,
        alpha=0.9,
        label="HAR-RV",
    )
    ax.set_title("Next-Day Log-Vol")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.text(
        0.02,
        0.96,
        f"MAE {logvol_metrics['mae']:.4f}\nEdge vs HAR {logvol_metrics['edge_vs_har']:+.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    ax = axes[0, 1]
    ax.plot(
        times,
        oos_frame["actual_next_log_range"],
        color="black",
        linewidth=1.2,
        label="Actual",
    )
    ax.plot(
        times,
        oos_frame["pred_next_log_range"],
        color="#2ca02c",
        linewidth=1.1,
        label="ML",
    )
    ax.set_title("Next-Day Log-Range")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    ax.text(
        0.02,
        0.96,
        f"MAE {range_metrics['mae']:.4f}\nEdge vs HAR {range_metrics['edge_vs_har']:+.1%}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    for ax, target, metrics, color in [
        (axes[1, 0], "next_up_exc", up_metrics, "#ff7f0e"),
        (axes[1, 1], "next_dn_exc", dn_metrics, "#9467bd"),
    ]:
        actual = oos_frame[f"actual_{target}"].to_numpy(dtype=float)
        pred = oos_frame[f"pred_{target}"].to_numpy(dtype=float)
        mask = np.isfinite(actual) & np.isfinite(pred)
        if mask.any():
            lo = float(np.nanmin(np.concatenate([actual[mask], pred[mask]])))
            hi = float(np.nanmax(np.concatenate([actual[mask], pred[mask]])))
            if hi <= lo:
                hi = lo + 1e-6
            ax.scatter(actual[mask], pred[mask], s=18, alpha=0.5, color=color)
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
        ax.set_title(VOL_TARGET_LABELS[target])
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.2)
        ax.text(
            0.02,
            0.96,
            f"MAE {metrics['mae']:.4f}\nEdge vs HAR {metrics['edge_vs_har']:+.1%}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily Volatility/Range Forecaster")
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
    file_prefix = str(artifact_ctx["file_prefix"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    artifact_paths_vol = get_artifact_paths(symbol_dir, file_prefix, "VOL")

    print("=" * 70)
    if requested_symbol != symbol:
        print(
            f"  INITIATING DAILY VOLATILITY UPLINK FOR: {symbol} "
            f"[requested {requested_symbol}]"
        )
    else:
        print(f"  INITIATING DAILY VOLATILITY UPLINK FOR: {symbol}")
    print("=" * 70)

    sys.path.append(os.path.join(data_dir, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("  [!] FATAL: Cannot locate inference_bridge.py in data_vault directory.")
        raise SystemExit(1)

    bridge = InferenceBridge(db_path=os.path.join(data_dir, "data_vault", "ohlcv.db"))
    tf_ctx = fetch_vol_timeframe_context(
        bridge,
        str(artifact_ctx["identity"].market_data_symbol),
    )
    tf_maps = tf_ctx["tf_maps"]
    primary_frames = tf_ctx["primary_frames"]
    reference_frames = tf_ctx["reference_frames"]
    df_1d = primary_frames["1D"]
    if df_1d is None or df_1d.empty:
        print(f"  [!] FATAL: No usable 1D primary data found for {symbol}.")
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
    df_model_ready, candidate_feature_cols = prepare_vol_model_ready(df_feature_frame)

    print(f"  [TOON VOL] Bars after cleaning            : {len(df_model_ready)}")
    print(
        f"  [TOON VOL] Features before selection      : {len(candidate_feature_cols)}"
    )
    for target in VOL_TARGETS:
        print(
            f"  [TOON VOL] {VOL_TARGET_LABELS[target]:<18}: "
            f"{df_model_ready[target].notna().sum():>5} rows"
        )

    if len(df_model_ready) < VOL_MIN_TRAIN_BARS + 100:
        print(f"\n  Error: Not enough {symbol} daily data for VOL walk-forward.")
        raise SystemExit(1)

    feature_cols, wf_results = fold_consensus_feature_selection_vol(
        df_model_ready,
        candidate_feature_cols,
        n_splits=VOL_N_SPLITS,
        min_train_bars=VOL_MIN_TRAIN_BARS,
        test_size_ratio=VOL_TEST_SIZE_RATIO,
        purge_gap=VOL_PURGE_GAP,
        target_col="next_yz_logvol",
    )
    if not feature_cols or not wf_results:
        print("  [TOON VOL] Nested feature selection failed. Aborting.")
        raise SystemExit(1)

    print(f"\n  [TOON VOL] Consensus feature count: {len(feature_cols)}")
    print("\n" + "=" * 70)
    print("  SUMMARY — TOON DAILY VOLATILITY WALK-FORWARD")
    print("=" * 70)
    for target in VOL_TARGETS:
        metrics = wf_results["overall_metrics"][target]
        print(
            f"  {VOL_TARGET_LABELS[target]:<18}: "
            f"MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f} | "
            f"HAR MAE {metrics['har_mae']:.4f} | Edge {metrics['edge_vs_har']:+.1%}"
        )
    print(f"  Number of splits performed: {len(wf_results['splits'])}")
    print(f"  Total OOS bars           : {len(wf_results['oos_frame'])}")

    with open(artifact_paths_vol["features"], "w", encoding="utf-8") as handle:
        for col in feature_cols:
            handle.write(f"{col}\n")

    for target, artifact_key in VOL_MODEL_ARTIFACT_KEYS.items():
        joblib.dump(
            wf_results["final_models"][target], artifact_paths_vol[artifact_key]
        )
    joblib.dump(wf_results["oos_forecast_map"], artifact_paths_vol["oos_forecasts"])
    save_vol_report(wf_results, artifact_paths_vol["ml_report"], symbol=symbol)

    print(f"\n  VOL feature list saved to '{artifact_paths_vol['features']}'")
    for target, artifact_key in VOL_MODEL_ARTIFACT_KEYS.items():
        print(
            f"  {VOL_TARGET_LABELS[target]:<18} model saved to "
            f"'{artifact_paths_vol[artifact_key]}'"
        )
    print(
        f"  VOL OOS forecast map saved to '{artifact_paths_vol['oos_forecasts']}' "
        f"({len(wf_results['oos_forecast_map'])} bars)"
    )
    print(f"  VOL report saved to '{artifact_paths_vol['ml_report']}'")

    df_infer = prepare_vol_inference_frame(df_feature_frame, feature_cols)
    last_row = df_infer.iloc[-1]
    latest_forecasts = predict_volatility_heads(
        wf_results["final_models"],
        feature_cols,
        last_row,
    )
    forecast_payload = build_vol_forecast_payload(
        symbol=symbol,
        row=last_row,
        forecasts=latest_forecasts,
        intraday_1h_used=df_1h is not None and not df_1h.empty,
    )
    save_vol_forecast_json(forecast_payload, artifact_paths_vol["live_forecast"])
    print(f"  VOL live forecast saved to '{artifact_paths_vol['live_forecast']}'")
    print_vol_forecast(forecast_payload)

    fi_sorted = sorted(
        wf_results["feature_importance"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    print("\n  Top 15 most predictive VOL features:")
    for i, (feature_name, feature_value) in enumerate(fi_sorted[:15], start=1):
        print(f"    {i:>2}. {feature_name:<42} {feature_value:.2f}")


if __name__ == "__main__":
    main()
