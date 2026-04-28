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
VOL_TARGETS_5D = [
    "next5d_yz_logvol",
    "next5d_log_range",
    "next5d_up_exc",
    "next5d_dn_exc",
]
VOL_ALL_TARGETS = VOL_TARGETS + VOL_TARGETS_5D
VOL_MIN_TRAIN_BARS = 500
VOL_TEST_SIZE_RATIO = 0.12
VOL_PURGE_GAP = 1
VOL_FEAT_BUDGET = 40
VOL_N_SPLITS = 8
VOL_OPTIONAL_FAIL_TIMEFRAMES = ("1H",)
VOL_CONFORMAL_LEVELS = (0.80, 0.90, 0.95)
VOL_HAR_FEATURE_COLS = ["har_rv_d", "har_rv_w", "har_rv_m"]
VOL_QLIKE_TARGETS = {
    "next_yz_logvol",
    "next_log_range",
    "next5d_yz_logvol",
    "next5d_log_range",
}

VOL_TARGET_LABELS = {
    "next_yz_logvol": "YZ Log-Vol",
    "next_log_range": "Log-Range",
    "next_up_exc": "Up-Excursion",
    "next_dn_exc": "Down-Excursion",
    "next5d_yz_logvol": "5D YZ Log-Vol",
    "next5d_log_range": "5D Log-Range",
    "next5d_up_exc": "5D Up-Excursion",
    "next5d_dn_exc": "5D Down-Excursion",
}
VOL_TARGET_SHORT = {
    "next_yz_logvol": "Vol",
    "next_log_range": "Range",
    "next_up_exc": "Up",
    "next_dn_exc": "Dn",
    "next5d_yz_logvol": "5DVol",
    "next5d_log_range": "5DRng",
    "next5d_up_exc": "5DUp",
    "next5d_dn_exc": "5DDn",
}
VOL_MODEL_ARTIFACT_KEYS = {
    "next_yz_logvol": "model_logvol",
    "next_log_range": "model_range",
    "next_up_exc": "model_up_exc",
    "next_dn_exc": "model_dn_exc",
    "next5d_yz_logvol": "model_5d_logvol",
    "next5d_log_range": "model_5d_range",
    "next5d_up_exc": "model_5d_up_exc",
    "next5d_dn_exc": "model_5d_dn_exc",
}
NON_FEATURE_COLS_VOL = set(NON_FEATURE_COLS_DAILY) | set(VOL_ALL_TARGETS)


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


def _is_qlike_target(target_col: str) -> bool:
    return target_col in VOL_QLIKE_TARGETS


def _conformal_level_key(level: float) -> str:
    return str(int(round(level * 100.0)))


def _qlike_loss(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return float("nan")
    residual = np.clip(true[mask] - pred[mask], -10.0, 10.0)
    return float(np.mean(np.exp(residual) - residual - 1.0))


def _qlike_objective(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.clip(
        np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float), -10.0, 10.0
    )
    exp_r = np.exp(residual)
    grad = 1.0 - exp_r
    hess = np.maximum(exp_r, 0.01)
    return grad, hess


def _qlike_eval_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[str, float, bool]:
    return "qlike", _qlike_loss(y_true, y_pred), False


def _register_pickle_objectives() -> None:
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        main_module._qlike_objective = _qlike_objective
        main_module._qlike_eval_metric = _qlike_eval_metric


sys.modules.setdefault("daily_volatility_engine", sys.modules[__name__])
_register_pickle_objectives()


def _build_lgbm_ensemble_vol(target_col: str) -> list[lgb.LGBMRegressor]:
    specs = _load_hyperparams_config()["classifier_ensemble"]
    if not isinstance(specs, list) or not specs:
        raise ValueError("`classifier_ensemble` must be a non-empty list.")

    objective = _qlike_objective if _is_qlike_target(target_col) else "mae"
    metric = "None" if _is_qlike_target(target_col) else "mae"
    models: list[lgb.LGBMRegressor] = []
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ValueError(f"`classifier_ensemble[{idx - 1}]` must be an object.")
        params = dict(spec)
        params.update(
            {
                "objective": objective,
                "metric": metric,
                "n_jobs": MODEL_N_JOBS,
                "verbose": -1,
            }
        )
        models.append(lgb.LGBMRegressor(**params))
    return models


def _fit_lgbm_with_inner_validation_vol(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
    target_col: str,
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
    eval_metric = _qlike_eval_metric if _is_qlike_target(target_col) else "mae"
    for model in _build_lgbm_ensemble_vol(target_col):
        model.fit(
            X_train_inner,
            y_train_inner,
            eval_set=[(X_val_inner, y_val_inner)],
            eval_metric=eval_metric,
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
    targets: list[str] | None = None,
) -> dict[str, float]:
    X_row = _single_row_feature_frame(row, feature_cols)
    active_targets = [
        target for target in (targets or list(models)) if target in models
    ]
    return {
        target: float(models[target].predict(X_row)[0]) for target in active_targets
    }


def _fit_har_rv_model(
    train_df: pd.DataFrame,
    target_col: str,
) -> dict[str, object]:
    X_train = train_df[VOL_HAR_FEATURE_COLS].to_numpy(dtype=float)
    y_train = train_df[target_col].to_numpy(dtype=float)
    train_mask = np.isfinite(y_train) & np.all(np.isfinite(X_train), axis=1)

    fallback_value = float(np.nanmean(y_train[train_mask])) if train_mask.any() else 0.0
    model_payload: dict[str, object] = {
        "feature_cols": list(VOL_HAR_FEATURE_COLS),
        "coef": [0.0] * len(VOL_HAR_FEATURE_COLS),
        "intercept": fallback_value,
        "fallback": fallback_value,
        "trained": False,
    }
    if train_mask.sum() < 25:
        return model_payload

    model = Ridge(alpha=1e-4)
    model.fit(X_train[train_mask], y_train[train_mask])
    model_payload.update(
        {
            "coef": [float(v) for v in model.coef_.ravel()],
            "intercept": float(model.intercept_),
            "trained": True,
        }
    )
    return model_payload


def _predict_har_rv_model(
    har_model: dict[str, object],
    data: pd.DataFrame | pd.Series,
) -> np.ndarray | float:
    feature_cols = list(har_model.get("feature_cols", VOL_HAR_FEATURE_COLS))
    fallback = float(har_model.get("fallback", 0.0))
    intercept = float(har_model.get("intercept", fallback))
    coef = np.asarray(
        har_model.get("coef", [0.0] * len(feature_cols)),
        dtype=float,
    )
    trained = bool(har_model.get("trained", False))

    if isinstance(data, pd.Series):
        X_row = np.array(
            [float(data.get(col, np.nan)) for col in feature_cols],
            dtype=float,
        )
        if not trained or not np.all(np.isfinite(X_row)):
            return fallback
        return float(intercept + X_row @ coef)

    X = data[feature_cols].to_numpy(dtype=float)
    preds = np.full(len(data), fallback, dtype=float)
    mask = np.all(np.isfinite(X), axis=1)
    if trained and mask.any():
        preds[mask] = intercept + X[mask] @ coef
    return preds


def fit_har_rv_models(
    df: pd.DataFrame,
    targets: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    active_targets = [
        target for target in (targets or VOL_ALL_TARGETS) if target in df.columns
    ]
    return {target: _fit_har_rv_model(df, target) for target in active_targets}


def build_har_oos_series(
    df: pd.DataFrame,
    target_col: str,
    *,
    n_splits: int = VOL_N_SPLITS,
    min_train_bars: int = VOL_MIN_TRAIN_BARS,
    test_size_ratio: float = VOL_TEST_SIZE_RATIO,
    purge_gap: int = VOL_PURGE_GAP,
    har_model: dict[str, object] | None = None,
) -> np.ndarray:
    har_series = np.full(len(df), np.nan, dtype=float)
    split_points = _vol_split_points(
        len(df),
        n_splits=n_splits,
        min_train_bars=min_train_bars,
        test_size_ratio=test_size_ratio,
    )
    for train_end, test_end in split_points:
        purged_train_end = max(0, train_end - purge_gap)
        har_preds = _har_rv_baseline(
            df.iloc[:purged_train_end],
            df.iloc[train_end:test_end],
            target_col,
        )
        har_series[train_end:test_end] = har_preds

    if har_model is not None:
        fill_mask = ~np.isfinite(har_series)
        if fill_mask.any():
            har_series[fill_mask] = np.asarray(
                _predict_har_rv_model(har_model, df.loc[fill_mask]),
                dtype=float,
            )
    return har_series


def _compute_bates_granger_weights(
    y_true: pd.Series | np.ndarray,
    ml_pred: pd.Series | np.ndarray,
    har_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float)
    ml = np.asarray(ml_pred, dtype=float)
    har = np.asarray(har_pred, dtype=float)
    mask = np.isfinite(true) & np.isfinite(ml) & np.isfinite(har)
    if mask.sum() < 25:
        return {"ml": 1.0, "har": 0.0}

    e_ml = true[mask] - ml[mask]
    e_har = true[mask] - har[mask]
    var_ml = float(np.mean(e_ml * e_ml))
    var_har = float(np.mean(e_har * e_har))
    cov_mh = float(np.mean(e_ml * e_har))
    denom = var_ml + var_har - (2.0 * cov_mh)
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return {
            "ml": 1.0 if var_ml <= var_har else 0.0,
            "har": 0.0 if var_ml <= var_har else 1.0,
        }

    w_ml = float(np.clip((var_har - cov_mh) / denom, 0.0, 1.0))
    return {"ml": w_ml, "har": float(1.0 - w_ml)}


def _combine_forecast_arrays(
    ml_pred: pd.Series | np.ndarray,
    har_pred: pd.Series | np.ndarray,
    weights: dict[str, float] | None,
) -> np.ndarray:
    ml = np.asarray(ml_pred, dtype=float)
    har = np.asarray(har_pred, dtype=float)
    combined = ml.copy()
    if not weights:
        return combined
    w_ml = float(weights.get("ml", 1.0))
    w_har = float(weights.get("har", 0.0))
    mask = np.isfinite(ml) & np.isfinite(har)
    if mask.any():
        combined[mask] = (w_ml * ml[mask]) + (w_har * har[mask])
    return combined


def build_vol_conformal_artifact(
    oos_frame: pd.DataFrame,
    har_models: dict[str, dict[str, object]],
    targets: list[str] | None = None,
) -> dict[str, object]:
    target_list = [
        target
        for target in (targets or VOL_ALL_TARGETS)
        if f"actual_{target}" in oos_frame.columns
        and f"pred_{target}" in oos_frame.columns
    ]
    artifact: dict[str, object] = {
        "version": "VOL_v5.4",
        "levels": [_conformal_level_key(level) for level in VOL_CONFORMAL_LEVELS],
        "oos_mean_logvol": float("nan"),
        "oos_std_logvol": float("nan"),
        "har_models": har_models,
        "targets": {},
    }

    logvol_actual = np.asarray(
        oos_frame.get("actual_next_yz_logvol", pd.Series(dtype=float)),
        dtype=float,
    )
    logvol_mask = np.isfinite(logvol_actual)
    if logvol_mask.any():
        artifact["oos_mean_logvol"] = float(np.mean(logvol_actual[logvol_mask]))
        artifact["oos_std_logvol"] = float(np.std(logvol_actual[logvol_mask], ddof=0))

    for target in target_list:
        actual = oos_frame[f"actual_{target}"].to_numpy(dtype=float)
        ml_pred = oos_frame[f"pred_{target}"].to_numpy(dtype=float)
        har_pred = oos_frame[f"har_{target}"].to_numpy(dtype=float)
        weights = _compute_bates_granger_weights(actual, ml_pred, har_pred)
        combined = _combine_forecast_arrays(ml_pred, har_pred, weights)
        mask = np.isfinite(actual) & np.isfinite(combined)
        abs_residuals = np.abs(actual[mask] - combined[mask])
        widths = {
            _conformal_level_key(level): float(np.nanquantile(abs_residuals, level))
            if abs_residuals.size
            else float("nan")
            for level in VOL_CONFORMAL_LEVELS
        }
        artifact["targets"][target] = {
            "weights": weights,
            "widths": widths,
            "sample_size": int(mask.sum()),
        }
    return artifact


def combine_vol_forecasts(
    ml_forecasts: dict[str, float],
    har_forecasts: dict[str, float] | None,
    conformal_artifact: dict[str, object] | None,
    *,
    targets: list[str] | None = None,
) -> dict[str, float]:
    target_list = targets or list(ml_forecasts)
    target_meta = (
        conformal_artifact.get("targets", {})
        if isinstance(conformal_artifact, dict)
        else {}
    )
    har_forecasts = har_forecasts or {}

    combined: dict[str, float] = {}
    for target in target_list:
        ml_value = float(ml_forecasts[target])
        har_value = float(har_forecasts.get(target, np.nan))
        weights = target_meta.get(target, {}).get("weights", {"ml": 1.0, "har": 0.0})
        combined[target] = float(
            _combine_forecast_arrays(
                np.array([ml_value], dtype=float),
                np.array([har_value], dtype=float),
                weights,
            )[0]
        )
    return combined


def enrich_vol_oos_frame(
    oos_frame: pd.DataFrame,
    conformal_artifact: dict[str, object] | None,
    *,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    enriched = oos_frame.copy()
    target_list = [
        target
        for target in (targets or VOL_ALL_TARGETS)
        if f"pred_{target}" in enriched.columns
    ]
    target_meta = (
        conformal_artifact.get("targets", {})
        if isinstance(conformal_artifact, dict)
        else {}
    )

    for target in target_list:
        weights = target_meta.get(target, {}).get("weights", {"ml": 1.0, "har": 0.0})
        har_vals = enriched.get(
            f"har_{target}", pd.Series(np.nan, index=enriched.index)
        )
        final_pred = _combine_forecast_arrays(
            enriched[f"pred_{target}"].to_numpy(dtype=float),
            np.asarray(har_vals, dtype=float),
            weights,
        )
        enriched[f"final_pred_{target}"] = final_pred

        widths = target_meta.get(target, {}).get("widths", {})
        for level in VOL_CONFORMAL_LEVELS:
            level_key = _conformal_level_key(level)
            width = float(widths.get(level_key, np.nan))
            enriched[f"interval_lo_{level_key}_{target}"] = final_pred - width
            enriched[f"interval_hi_{level_key}_{target}"] = final_pred + width
    return enriched


def compute_vol_forecast_confidence(
    row: pd.Series,
    intraday_1h_used: bool,
    conformal_artifact: dict[str, object] | None,
    *,
    predicted_logvol: float | None = None,
) -> tuple[float, dict[str, float]]:
    artifact = conformal_artifact if isinstance(conformal_artifact, dict) else {}
    f_avail = 1.0 if intraday_1h_used else 0.80

    vov = float(row.get("rv_1d_vov", 0.5))
    f_regime = float(np.clip(1.0 - max(0.0, vov - 1.0) / 3.0, 0.4, 1.0))

    current_logvol = (
        float(predicted_logvol)
        if predicted_logvol is not None and np.isfinite(predicted_logvol)
        else float(row.get("rv_1d_yz_log", 0.0))
    )
    oos_mean = float(artifact.get("oos_mean_logvol", current_logvol))
    oos_std = max(float(artifact.get("oos_std_logvol", 1.0)), 1e-6)
    z_score = abs(current_logvol - oos_mean) / oos_std
    f_extremity = float(np.clip(1.0 - max(0.0, z_score - 1.5) / 3.0, 0.5, 1.0))

    confidence = float(min(f_avail, f_regime, f_extremity))
    return confidence, {
        "feature_availability": float(f_avail),
        "regime_stability": float(f_regime),
        "prediction_extremity": float(f_extremity),
        "prediction_extremity_z": float(z_score),
    }


def _build_target_intervals(
    forecasts: dict[str, float],
    conformal_artifact: dict[str, object] | None,
) -> dict[str, dict[str, dict[str, float]]]:
    target_meta = (
        conformal_artifact.get("targets", {})
        if isinstance(conformal_artifact, dict)
        else {}
    )
    intervals: dict[str, dict[str, dict[str, float]]] = {}
    for target, forecast in forecasts.items():
        widths = target_meta.get(target, {}).get("widths", {})
        target_intervals: dict[str, dict[str, float]] = {}
        for level in VOL_CONFORMAL_LEVELS:
            level_key = _conformal_level_key(level)
            width = float(widths.get(level_key, np.nan))
            if np.isfinite(width):
                target_intervals[level_key] = {
                    "low": float(forecast - width),
                    "high": float(forecast + width),
                    "width": float(width),
                }
        if target_intervals:
            intervals[target] = target_intervals
    return intervals


def build_vol_forecast_payload(
    *,
    symbol: str,
    row: pd.Series,
    forecasts: dict[str, float],
    intraday_1h_used: bool,
    reference_price: float | None = None,
    reference_price_source: str = "latest_close",
    raw_forecasts: dict[str, float] | None = None,
    har_forecasts: dict[str, float] | None = None,
    conformal_artifact: dict[str, object] | None = None,
) -> dict[str, object]:
    latest_close = float(row["close"])
    ref_price = latest_close if reference_price is None else float(reference_price)
    up_mult = float(np.exp(forecasts["next_up_exc"]))
    dn_mult = float(np.exp(forecasts["next_dn_exc"]))
    up_mult_5d = float(np.exp(forecasts["next5d_up_exc"]))
    dn_mult_5d = float(np.exp(forecasts["next5d_dn_exc"]))
    intervals = _build_target_intervals(forecasts, conformal_artifact)
    confidence, confidence_factors = compute_vol_forecast_confidence(
        row,
        intraday_1h_used,
        conformal_artifact,
        predicted_logvol=forecasts.get("next_yz_logvol"),
    )
    sigma_1d = float(np.expm1(forecasts["next_yz_logvol"]))
    sigma_5d = float(np.expm1(forecasts["next5d_yz_logvol"]))

    projected_high_90 = None
    projected_low_90 = None
    projected_high_5d_90 = None
    projected_low_5d_90 = None
    if "next_up_exc" in intervals and "90" in intervals["next_up_exc"]:
        projected_high_90 = float(
            ref_price * np.exp(intervals["next_up_exc"]["90"]["high"])
        )
    if "next_dn_exc" in intervals and "90" in intervals["next_dn_exc"]:
        projected_low_90 = float(
            ref_price / np.exp(intervals["next_dn_exc"]["90"]["high"])
        )
    if "next5d_up_exc" in intervals and "90" in intervals["next5d_up_exc"]:
        projected_high_5d_90 = float(
            ref_price * np.exp(intervals["next5d_up_exc"]["90"]["high"])
        )
    if "next5d_dn_exc" in intervals and "90" in intervals["next5d_dn_exc"]:
        projected_low_5d_90 = float(
            ref_price / np.exp(intervals["next5d_dn_exc"]["90"]["high"])
        )

    payload = {
        "symbol": symbol,
        "lane": "VOL",
        "horizon": "next_daily_bar",
        "basis_bar_time": str(pd.Timestamp(row["time"])),
        "latest_close": latest_close,
        "reference_price": ref_price,
        "reference_price_source": reference_price_source,
        "intraday_1h_used": bool(intraday_1h_used),
        "forecast_confidence": confidence,
        "confidence_factors": confidence_factors,
        "yz_logvol": float(forecasts["next_yz_logvol"]),
        "annualized_vol_pct": float(sigma_1d * 100.0),
        "log_range": float(forecasts["next_log_range"]),
        "range_pct": float(np.expm1(forecasts["next_log_range"]) * 100.0),
        "up_exc": float(forecasts["next_up_exc"]),
        "up_bps": float(np.expm1(forecasts["next_up_exc"]) * 10000.0),
        "dn_exc": float(forecasts["next_dn_exc"]),
        "dn_bps": float(np.expm1(forecasts["next_dn_exc"]) * 10000.0),
        "projected_high": float(ref_price * up_mult),
        "projected_low": float(ref_price / dn_mult),
        "yz_logvol_5d": float(forecasts["next5d_yz_logvol"]),
        "annualized_vol_pct_5d": float(sigma_5d * 100.0),
        "log_range_5d": float(forecasts["next5d_log_range"]),
        "range_pct_5d": float(np.expm1(forecasts["next5d_log_range"]) * 100.0),
        "up_exc_5d": float(forecasts["next5d_up_exc"]),
        "up_bps_5d": float(np.expm1(forecasts["next5d_up_exc"]) * 10000.0),
        "dn_exc_5d": float(forecasts["next5d_dn_exc"]),
        "dn_bps_5d": float(np.expm1(forecasts["next5d_dn_exc"]) * 10000.0),
        "projected_high_5d": float(ref_price * up_mult_5d),
        "projected_low_5d": float(ref_price / dn_mult_5d),
        "vol_5d_1d_ratio": float(sigma_5d / max(sigma_1d, 1e-6)),
        "intervals": intervals,
        "projected_high_90": projected_high_90,
        "projected_low_90": projected_low_90,
        "projected_high_5d_90": projected_high_5d_90,
        "projected_low_5d_90": projected_low_5d_90,
        "combination_weights": {
            target: dict(
                conformal_artifact.get("targets", {}).get(target, {}).get("weights", {})
            )
            for target in forecasts
        }
        if isinstance(conformal_artifact, dict)
        else {},
    }
    if raw_forecasts is not None:
        payload["ml_forecasts"] = {k: float(v) for k, v in raw_forecasts.items()}
    if har_forecasts is not None:
        payload["har_forecasts"] = {k: float(v) for k, v in har_forecasts.items()}
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
        f"  Confidence:        {payload.get('forecast_confidence', float('nan')):.2f}"
    )
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
    if (
        payload.get("projected_high_90") is not None
        and payload.get("projected_low_90") is not None
    ):
        print(
            f"  90% High/Low:      {payload['projected_high_90']:,.2f} / "
            f"{payload['projected_low_90']:,.2f}"
        )
    print(
        f"  5D Annualized Vol: {payload['annualized_vol_pct_5d']:.1f}%  "
        f"(ratio: {payload['vol_5d_1d_ratio']:.2f}x)"
    )
    print(
        f"  5D Projected H/L:  {payload['projected_high_5d']:,.2f} / "
        f"{payload['projected_low_5d']:,.2f}"
    )
    if (
        payload.get("projected_high_5d_90") is not None
        and payload.get("projected_low_5d_90") is not None
    ):
        print(
            f"  5D 90% High/Low:   {payload['projected_high_5d_90']:,.2f} / "
            f"{payload['projected_low_5d_90']:,.2f}"
        )
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
    har_model = _fit_har_rv_model(train_df, target_col)
    return np.asarray(_predict_har_rv_model(har_model, test_df), dtype=float)


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
        *VOL_ALL_TARGETS,
    ]
    keep_cols = candidate_feature_cols + [c for c in state_cols if c in df_full.columns]
    df_model_ready = df_full[keep_cols].copy()

    for col in candidate_feature_cols:
        df_model_ready[col] = (
            df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )

    active_targets = [
        target for target in VOL_ALL_TARGETS if target in df_model_ready.columns
    ]
    df_model_ready = df_model_ready.dropna(subset=active_targets).reset_index(drop=True)
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
    active_targets = [target for target in VOL_ALL_TARGETS if target in df.columns]
    df = df.dropna(subset=feature_cols + active_targets).reset_index(drop=True)
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

        for target in active_targets:
            model, _, _ = _fit_lgbm_with_inner_validation_vol(
                X_train_full,
                train_df[target],
                target_col=target,
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
            qlike = (
                _qlike_loss(test_df[target], preds)
                if _is_qlike_target(target)
                else float("nan")
            )
            har_qlike = (
                _qlike_loss(test_df[target], har_preds)
                if _is_qlike_target(target)
                else float("nan")
            )
            fold_result[f"{target}_mae"] = mae
            fold_result[f"{target}_rmse"] = rmse
            fold_result[f"{target}_har_mae"] = har_mae
            fold_result[f"{target}_har_rmse"] = har_rmse
            fold_result[f"{target}_edge_vs_har"] = edge_vs_har
            fold_result[f"{target}_qlike"] = qlike
            fold_result[f"{target}_har_qlike"] = har_qlike

        feature_importance_sum += fold_models["next_yz_logvol"].feature_importances_

        pieces_1d: list[str] = []
        for target in VOL_TARGETS:
            if target not in active_targets:
                continue
            piece = (
                f"{VOL_TARGET_SHORT[target]} {fold_result[f'{target}_mae']:.4f}"
                f" vs HAR {fold_result[f'{target}_har_mae']:.4f}"
                f" ({fold_result[f'{target}_edge_vs_har']:+.1%})"
            )
            if _is_qlike_target(target) and np.isfinite(fold_result[f"{target}_qlike"]):
                piece += f" QL {fold_result[f'{target}_qlike']:.4f}"
            pieces_1d.append(piece)

        pieces_5d: list[str] = []
        for target in VOL_TARGETS_5D:
            if target not in active_targets:
                continue
            piece = (
                f"{VOL_TARGET_SHORT[target]} {fold_result[f'{target}_mae']:.4f}"
                f" vs HAR {fold_result[f'{target}_har_mae']:.4f}"
                f" ({fold_result[f'{target}_edge_vs_har']:+.1%})"
            )
            if _is_qlike_target(target) and np.isfinite(fold_result[f"{target}_qlike"]):
                piece += f" QL {fold_result[f'{target}_qlike']:.4f}"
            pieces_5d.append(piece)
        print(
            f"  Split {i + 1:>2} | Train:{len(X_train_full):>5} "
            f"Test:{len(X_test):>5} | {' | '.join(pieces_1d)}"
        )
        if pieces_5d:
            print(f"            {' | '.join(pieces_5d)}")

        for row_idx, ts in enumerate(test_df["time"]):
            ts_key = pd.Timestamp(ts)
            row: dict[str, float | pd.Timestamp] = {"time": ts_key}
            row_map = oos_forecast_map.setdefault(ts_key, {})
            for target in active_targets:
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
            target_col=target,
            purge_gap=purge_gap,
        )[0]
        for target in active_targets
    }
    final_har_models = fit_har_rv_models(df, active_targets)
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
            "qlike": _nanmean_metric(results, f"{target}_qlike"),
            "har_qlike": _nanmean_metric(results, f"{target}_har_qlike"),
        }
        for target in active_targets
    }

    return {
        "targets": active_targets,
        "splits": results,
        "feature_importance": dict(
            zip(feature_cols, feature_importance_sum / split_count)
        ),
        "feature_cols": feature_cols,
        "df": df,
        "final_models": final_models,
        "final_har_models": final_har_models,
        "oos_forecast_map": oos_forecast_map,
        "oos_frame": oos_frame,
        "overall_metrics": overall_metrics,
    }


def _multi_target_phase1_ranking(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    ranking_targets: list[str],
) -> list[str]:
    rank_positions: dict[str, list[int]] = {}
    rank_counts: dict[str, int] = {}
    for target in ranking_targets:
        ranked = phase1_ranking(train_df, feature_cols, target_col=target)
        for pos, feat in enumerate(ranked):
            rank_positions.setdefault(feat, []).append(pos)
            rank_counts[feat] = rank_counts.get(feat, 0) + 1

    if not rank_positions:
        return feature_cols[:VOL_FEAT_BUDGET]

    fused = sorted(
        rank_positions,
        key=lambda feat: (
            float(np.mean(rank_positions[feat])),
            -rank_counts.get(feat, 0),
            feat,
        ),
    )
    return fused[:VOL_FEAT_BUDGET]


def fold_consensus_feature_selection_vol(
    df: pd.DataFrame,
    candidate_feature_cols: list[str],
    n_splits: int = VOL_N_SPLITS,
    min_train_bars: int = VOL_MIN_TRAIN_BARS,
    test_size_ratio: float = VOL_TEST_SIZE_RATIO,
    purge_gap: int = VOL_PURGE_GAP,
    consensus_threshold: float = 0.50,
    ranking_targets: list[str] | None = None,
) -> tuple[list[str], dict]:
    active_ranking_targets = [
        target for target in (ranking_targets or VOL_TARGETS) if target in df.columns
    ]
    df = df.dropna(
        subset=[c for c in candidate_feature_cols if c in df.columns]
        + active_ranking_targets
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
            fold_features = _multi_target_phase1_ranking(
                train_df,
                fold_features,
                active_ranking_targets,
            )
        except Exception as exc:
            print(f"  [Nested VOL] Fold {i + 1} feature selection failed: {exc}")
            continue

        if not fold_features:
            continue

        for feat in fold_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1

        X_train = train_df[fold_features]
        y_train = train_df["next_yz_logvol"]
        try:
            model, _, _ = _fit_lgbm_with_inner_validation_vol(
                X_train,
                y_train,
                target_col="next_yz_logvol",
                purge_gap=purge_gap,
            )
        except Exception as exc:
            print(f"  [Nested VOL] Fold {i + 1} training failed: {exc}")
            continue

        preds = np.asarray(model.predict(test_df[fold_features]), dtype=float)
        har_preds = _har_rv_baseline(train_df, test_df, "next_yz_logvol")
        mae, rmse = _regression_metrics(test_df["next_yz_logvol"], preds)
        har_mae, har_rmse = _regression_metrics(test_df["next_yz_logvol"], har_preds)
        edge_vs_har = (
            float(1.0 - (mae / har_mae))
            if np.isfinite(mae) and np.isfinite(har_mae) and har_mae > 0.0
            else float("nan")
        )
        qlike = _qlike_loss(test_df["next_yz_logvol"], preds)
        har_qlike = _qlike_loss(test_df["next_yz_logvol"], har_preds)

        print(
            f"  [Nested VOL] Fold {i + 1:>2} | Train:{len(X_train):>5} "
            f"Test:{len(test_df):>5} | MAE:{mae:.4f} RMSE:{rmse:.4f} "
            f"HAR:{har_mae:.4f} Edge:{edge_vs_har:+.1%} "
            f"QL:{qlike:.4f}/{har_qlike:.4f} | Feats:{len(fold_features)}"
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
                "qlike": qlike,
                "har_qlike": har_qlike,
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
    conformal_artifact: dict[str, object] | None = None,
) -> None:
    oos_frame = wf_results.get("oos_frame")
    if oos_frame is None or len(oos_frame) == 0:
        print("  [VOL Report] No OOS frame available to plot.")
        return

    oos_frame = enrich_vol_oos_frame(
        oos_frame,
        conformal_artifact,
        targets=wf_results.get("targets"),
    )
    plot_targets = [
        target
        for target in wf_results.get("targets", VOL_ALL_TARGETS)
        if f"actual_{target}" in oos_frame.columns
    ]
    if not plot_targets:
        print("  [VOL Report] No active targets available to plot.")
        return

    times = pd.to_datetime(oos_frame["time"])
    ncols = 2
    nrows = int(np.ceil(len(plot_targets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.2 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    fig.suptitle(
        f"{symbol} Daily Volatility — Walk-Forward OOS "
        f"({len(wf_results['splits'])} folds, {len(oos_frame)} bars)",
        fontsize=14,
        y=0.98,
    )

    line_colors = {
        "next_yz_logvol": "#1f77b4",
        "next_log_range": "#2ca02c",
        "next_up_exc": "#ff7f0e",
        "next_dn_exc": "#9467bd",
        "next5d_yz_logvol": "#17becf",
        "next5d_log_range": "#8c564b",
        "next5d_up_exc": "#e377c2",
        "next5d_dn_exc": "#7f7f7f",
    }

    for ax, target in zip(axes_arr, plot_targets):
        actual = oos_frame[f"actual_{target}"].to_numpy(dtype=float)
        pred_col = (
            f"final_pred_{target}"
            if f"final_pred_{target}" in oos_frame.columns
            else f"pred_{target}"
        )
        pred = oos_frame[pred_col].to_numpy(dtype=float)
        har_col = f"har_{target}"
        mask = np.isfinite(actual) & np.isfinite(pred)
        if target.endswith("up_exc") or target.endswith("dn_exc"):
            if mask.any():
                lo = float(np.nanmin(np.concatenate([actual[mask], pred[mask]])))
                hi = float(np.nanmax(np.concatenate([actual[mask], pred[mask]])))
                if hi <= lo:
                    hi = lo + 1e-6
                ax.scatter(
                    actual[mask],
                    pred[mask],
                    s=18,
                    alpha=0.5,
                    color=line_colors.get(target, "#1f77b4"),
                )
                ax.plot(
                    [lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0
                )
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
        else:
            ax.plot(times, actual, color="black", linewidth=1.2, label="Actual")
            ax.plot(
                times,
                pred,
                color=line_colors.get(target, "#1f77b4"),
                linewidth=1.1,
                label="Served",
            )
            if har_col in oos_frame.columns:
                ax.plot(
                    times,
                    oos_frame[har_col].to_numpy(dtype=float),
                    color="#d62728",
                    linewidth=0.9,
                    alpha=0.85,
                    label="HAR-RV",
                )
            lo_90 = f"interval_lo_90_{target}"
            hi_90 = f"interval_hi_90_{target}"
            if lo_90 in oos_frame.columns and hi_90 in oos_frame.columns:
                ax.fill_between(
                    times,
                    oos_frame[lo_90].to_numpy(dtype=float),
                    oos_frame[hi_90].to_numpy(dtype=float),
                    color=line_colors.get(target, "#1f77b4"),
                    alpha=0.15,
                    label="90% conformal",
                )
            ax.legend(loc="upper right")
        ax.set_title(VOL_TARGET_LABELS[target])
        ax.grid(alpha=0.2)

        mae, rmse = _regression_metrics(actual, pred)
        har_mae, _ = (
            _regression_metrics(actual, oos_frame[har_col].to_numpy(dtype=float))
            if har_col in oos_frame.columns
            else (float("nan"), float("nan"))
        )
        edge_vs_har = (
            float(1.0 - (mae / har_mae))
            if np.isfinite(mae) and np.isfinite(har_mae) and har_mae > 0.0
            else float("nan")
        )
        metrics_lines = [f"MAE {mae:.4f}", f"RMSE {rmse:.4f}"]
        if _is_qlike_target(target):
            metrics_lines.append(f"QLIKE {_qlike_loss(actual, pred):.4f}")
        if np.isfinite(edge_vs_har):
            metrics_lines.append(f"Edge {edge_vs_har:+.1%}")
        ax.text(
            0.02,
            0.96,
            "\n".join(metrics_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )

    for ax in axes_arr[len(plot_targets) :]:
        ax.axis("off")

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
    for target in VOL_ALL_TARGETS:
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
        ranking_targets=VOL_TARGETS,
    )
    if not feature_cols or not wf_results:
        print("  [TOON VOL] Nested feature selection failed. Aborting.")
        raise SystemExit(1)

    conformal_artifact = build_vol_conformal_artifact(
        wf_results["oos_frame"],
        wf_results["final_har_models"],
        targets=wf_results["targets"],
    )

    print(f"\n  [TOON VOL] Consensus feature count: {len(feature_cols)}")
    print("\n" + "=" * 70)
    print("  SUMMARY — TOON DAILY VOLATILITY WALK-FORWARD")
    print("=" * 70)
    for target in wf_results["targets"]:
        metrics = wf_results["overall_metrics"][target]
        summary = (
            f"  {VOL_TARGET_LABELS[target]:<18}: "
            f"MAE {metrics['mae']:.4f} | RMSE {metrics['rmse']:.4f} | "
            f"HAR MAE {metrics['har_mae']:.4f} | Edge {metrics['edge_vs_har']:+.1%}"
        )
        if _is_qlike_target(target):
            summary += (
                f" | QLIKE {metrics['qlike']:.4f} (HAR {metrics['har_qlike']:.4f})"
            )
        print(summary)
    print(f"  Number of splits performed: {len(wf_results['splits'])}")
    print(f"  Total OOS bars           : {len(wf_results['oos_frame'])}")

    with open(artifact_paths_vol["features"], "w", encoding="utf-8") as handle:
        for col in feature_cols:
            handle.write(f"{col}\n")

    for target, artifact_key in VOL_MODEL_ARTIFACT_KEYS.items():
        if target not in wf_results["final_models"]:
            continue
        joblib.dump(
            wf_results["final_models"][target], artifact_paths_vol[artifact_key]
        )
    joblib.dump(wf_results["oos_forecast_map"], artifact_paths_vol["oos_forecasts"])
    joblib.dump(conformal_artifact, artifact_paths_vol["conformal"])
    save_vol_report(
        wf_results,
        artifact_paths_vol["ml_report"],
        symbol=symbol,
        conformal_artifact=conformal_artifact,
    )

    print(f"\n  VOL feature list saved to '{artifact_paths_vol['features']}'")
    for target, artifact_key in VOL_MODEL_ARTIFACT_KEYS.items():
        if target not in wf_results["final_models"]:
            continue
        print(
            f"  {VOL_TARGET_LABELS[target]:<18} model saved to "
            f"'{artifact_paths_vol[artifact_key]}'"
        )
    print(
        f"  VOL OOS forecast map saved to '{artifact_paths_vol['oos_forecasts']}' "
        f"({len(wf_results['oos_forecast_map'])} bars)"
    )
    print(f"  VOL conformal artifact saved to '{artifact_paths_vol['conformal']}'")
    print(f"  VOL report saved to '{artifact_paths_vol['ml_report']}'")

    df_infer = prepare_vol_inference_frame(df_feature_frame, feature_cols)
    last_row = df_infer.iloc[-1]
    latest_ml_forecasts = predict_volatility_heads(
        wf_results["final_models"],
        feature_cols,
        last_row,
        targets=wf_results["targets"],
    )
    latest_har_forecasts = {
        target: float(
            _predict_har_rv_model(wf_results["final_har_models"][target], last_row)
        )
        for target in wf_results["targets"]
    }
    latest_forecasts = combine_vol_forecasts(
        latest_ml_forecasts,
        latest_har_forecasts,
        conformal_artifact,
        targets=wf_results["targets"],
    )
    if df_1h is not None and not df_1h.empty:
        reference_price = float(df_1h["close"].iloc[-1])
        reference_source = "latest_1h_close"
    else:
        reference_price = float(last_row["close"])
        reference_source = "latest_close"
    forecast_payload = build_vol_forecast_payload(
        symbol=symbol,
        row=last_row,
        forecasts=latest_forecasts,
        intraday_1h_used=df_1h is not None and not df_1h.empty,
        reference_price=reference_price,
        reference_price_source=reference_source,
        raw_forecasts=latest_ml_forecasts,
        har_forecasts=latest_har_forecasts,
        conformal_artifact=conformal_artifact,
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
