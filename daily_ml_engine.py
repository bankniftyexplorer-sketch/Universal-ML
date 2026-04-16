"""
daily_ml_engine.py — TOON v5.1 Daily Prediction Pipeline
=========================================================
Daily training path built around:
  1. Julia holographic extraction on 1D / 1W / 1M / 3M
  2. Python macro regime overlays on 6M / 12M
  3. Daily confluence and walk-forward validation

The existing 1H pipeline remains untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from julia_bridge import (
    add_target_fast,
    holographic_feature_engine_daily,
    kalman_structural_engine_daily,
    narrative_context_engine_daily,
    rv_feature_engine_daily,
    smc_feature_engine_daily,
)
from universal_ml_engine import (
    MODEL_N_JOBS,
    POLICY_EOD_GATE_HOUR_1D,
    TRADE_PLAN_LABEL_COLS,
    EnsembleModel,
    _compute_atr14,
    _load_hyperparams_config,
    build_prob_array_from_oos_map,
    build_report_data_lines,
    build_timeframe_selection,
    calibrate_oos_probabilities,
    describe_selected_frame,
    finalize_forecast_context,
    get_artifact_paths,
    inject_thermodynamic_basis,
    predict_next_bar,
    predict_trade_plan,
    prepare_symbol_artifact_context,
    save_report,
    select_primary_timeframe,
    train_exit_surface_artifact,
    train_policy_artifact,
    train_trade_plan_models,
)

warnings.filterwarnings("ignore")

BARRIER_HORIZON_BARS_DAILY = 10
BARRIER_ATR_MULT_DAILY = 1.25
PURGE_GAP_DAILY = 5
FINAL_FEAT_BUDGET = 40  # Max features in final model
MIN_TRAIN_BARS_DAILY = 500
TEST_SIZE_RATIO_DAILY = 0.15
LIVE_CONFIDENCE_THRESHOLD_DAILY = 0.72

NON_FEATURE_COLS_DAILY = {
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr14",
    "realized_volatility",
    "basis_pts",
    "basis_pct",
    "basis_z_score",
    "basis_vel_5",
    "basis_vel_10",
    "session_time_pos",
    "eod_basis_momentum",
    "is_synthetic_vol",
    "target",
    "next_ret_pct",
    "bars_to_target",
    "entry_price_next_bar",
    "target_distance",
    "long_path_r",
    "short_path_r",
    "target_edge_r",
    "best_path_r",
    "long_mfe_atr",
    "long_mae_atr",
    "short_mfe_atr",
    "short_mae_atr",
}


def compute_macro_regime(
    df: pd.DataFrame, label: str, lookback: int = 3
) -> pd.DataFrame:
    """
    Extract 5 compact regime features from sparse macro OHLCV (6M/12M).
    Returns one row per macro bar with the source `time` plus 5 bounded features.
    """
    df = df.sort_values("time").reset_index(drop=True)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    vols = np.nan_to_num(df["volume"].to_numpy(dtype=float), nan=0.0)
    n = len(df)

    rows: list[dict[str, float | pd.Timestamp]] = []
    for i in range(n):
        row: dict[str, float | pd.Timestamp] = {"time": df["time"].iloc[i]}
        if i < lookback - 1:
            row[f"{label}_secular_trend"] = 0.0
            row[f"{label}_range_pos"] = 0.5
            row[f"{label}_momentum_phase"] = 0.0
            row[f"{label}_vol_regime"] = 0.0
            row[f"{label}_vol_conviction"] = 0.0
        else:
            s = slice(i - lookback + 1, i + 1)
            wc = closes[s]
            wh = highs[s]
            wl = lows[s]
            wo = opens[s]
            wv = vols[s]
            rng = float(np.max(wh) - np.min(wl))

            row[f"{label}_secular_trend"] = float(
                np.clip((wc[-1] - wc[0]) / (rng + 1e-9), -1.0, 1.0)
            )
            row[f"{label}_range_pos"] = float(
                np.clip((wc[-1] - np.min(wl)) / (rng + 1e-9), 0.0, 1.0)
            )

            recent = float(wc[-1] - wc[-2])
            prior = float(wc[-2] - wc[-3]) if lookback >= 3 else 0.0
            row[f"{label}_momentum_phase"] = float(
                np.clip((recent - prior) / (abs(recent) + abs(prior) + 1e-9), -1.0, 1.0)
            )

            bodies = np.abs(wc - wo)
            if len(bodies) >= 2 and bodies[-2] > 1e-9:
                vol_regime = (bodies[-1] - bodies[-2]) / (bodies[-2] + 1e-9)
            else:
                vol_regime = 0.0
            row[f"{label}_vol_regime"] = float(np.clip(vol_regime, -1.0, 1.0))

            trend_sign = float(np.sign(wc[-1] - wc[0]))
            vol_mean = float(np.mean(wv)) if len(wv) > 0 else 0.0
            vol_delta = float(wv[-1] - np.mean(wv[:-1])) if len(wv) > 1 else 0.0
            row[f"{label}_vol_conviction"] = float(
                np.clip(trend_sign * vol_delta / (vol_mean + 1e-9), -1.0, 1.0)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def inject_macro_regime(
    df_daily: pd.DataFrame,
    df_macro: pd.DataFrame | None,
    label: str,
) -> pd.DataFrame:
    """Merge macro regime onto daily bars via backward ASOF (look-ahead safe)."""
    feats = [
        f"{label}_secular_trend",
        f"{label}_range_pos",
        f"{label}_momentum_phase",
        f"{label}_vol_regime",
        f"{label}_vol_conviction",
    ]

    if df_macro is None or df_macro.empty or len(df_macro) < 3:
        for feat in feats:
            df_daily[feat] = 0.5 if feat.endswith("_range_pos") else 0.0
        return df_daily

    regime_df = compute_macro_regime(df_macro, label)

    # Macro features become known only after the macro candle closes.
    t = regime_df["time"]
    t_next = t.shift(-1)
    gap = t.diff().dropna().median() if len(t) > 1 else pd.Timedelta(days=180)
    regime_df["time"] = t_next.where(t_next.notna(), t + gap)

    df_daily = df_daily.sort_values("time").reset_index(drop=True)
    regime_df = regime_df.sort_values("time")
    merged = pd.merge_asof(df_daily, regime_df, on="time", direction="backward")
    for feat in feats:
        if feat in merged.columns:
            fill_value = 0.5 if feat.endswith("_range_pos") else 0.0
            merged[feat] = merged[feat].fillna(fill_value)
    return merged


def add_daily_confluence(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-TF confluence including regime alignment."""

    def g(col: str, default: float = 0.0) -> np.ndarray:
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
        return np.full(len(df), default, dtype=float)

    daily_bias = np.sign(g("1d_w13_skel_smc_phase"))
    macro_bias = np.sign(g("6m_secular_trend") + g("12m_secular_trend"))
    df["mtf_conf_regime_alignment"] = daily_bias * macro_bias
    return df


def _build_lgbm_ensemble_daily() -> list[lgb.LGBMRegressor]:
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
                "n_jobs": MODEL_N_JOBS,
                "verbose": -1,
                "objective": "regression",
                "metric": "mae",
            }
        )
        models.append(lgb.LGBMRegressor(**params))
    return models


def _fit_lgbm_with_inner_validation_daily(
    X_train_full: pd.DataFrame,
    y_train_full: pd.Series,
) -> tuple[EnsembleModel, pd.DataFrame, pd.Series]:
    inner_val_size = max(100, int(len(X_train_full) * 0.15))
    inner_val_size = min(inner_val_size, len(X_train_full) - 1)
    if inner_val_size <= 0:
        raise ValueError("Not enough training rows for inner validation.")

    purged_end = -(inner_val_size + PURGE_GAP_DAILY)
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
    for model in _build_lgbm_ensemble_daily():
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


def walk_forward_daily(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 10,
    min_train_bars: int = MIN_TRAIN_BARS_DAILY,
    test_size_ratio: float = TEST_SIZE_RATIO_DAILY,
    purge_gap: int = PURGE_GAP_DAILY,
) -> dict:
    """Daily walk-forward with daily-specific purge and minimum history."""
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    n = len(df)

    if n < min_train_bars:
        print(
            f"  Error: Not enough data for daily walk-forward. Need {min_train_bars}, got {n}."
        )
        return {}

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

    if not split_points:
        print("  Error: Could not determine valid split points for daily walk-forward.")
        return {}

    print(
        f"\n  Daily walk-forward: {len(split_points)} splits, "
        f"min_train_bars={min_train_bars}, test_ratio={test_size_ratio:.2f}, purge_gap={purge_gap}"
    )
    print(f"  Total daily bars: {n}")

    results = []
    feature_importance_sum = np.zeros(len(feature_cols))
    oos_proba_map: dict = {}

    for i, (train_end, test_end) in enumerate(split_points):
        purged_train_end = max(0, train_end - purge_gap)
        X_train_full = df[feature_cols].iloc[:purged_train_end]
        y_train_full = df["target"].iloc[:purged_train_end]
        X_test = df[feature_cols].iloc[train_end:test_end]
        y_test = df["target"].iloc[train_end:test_end]

        if len(X_test) == 0:
            continue

        model, _, _ = _fit_lgbm_with_inner_validation_daily(X_train_full, y_train_full)
        preds = model.predict(X_test)

        true_binary = (y_test > 0.5).astype(int)
        pred_binary = (preds > 0.5).astype(int)
        acc = float((true_binary == pred_binary).mean())

        high_conf_mask = (preds >= LIVE_CONFIDENCE_THRESHOLD_DAILY) | (
            preds <= (1.0 - LIVE_CONFIDENCE_THRESHOLD_DAILY)
        )
        if high_conf_mask.sum() > 0:
            conf_acc = float(
                (pred_binary[high_conf_mask] == true_binary[high_conf_mask]).mean()
            )
        else:
            conf_acc = float("nan")

        fraction_conf = float(high_conf_mask.sum() / len(preds))
        baseline = float(true_binary.mean())

        print(
            f"  Split {i + 1:>2} | Train: {len(X_train_full):>5} | Test: {len(X_test):>5} | "
            f"Acc:{acc:.3f} | ConfAcc(>={LIVE_CONFIDENCE_THRESHOLD_DAILY:.2f}):{conf_acc:.3f} | "
            f"ConfBars: {fraction_conf * 100:>4.1f}% | Baseline(UP):{baseline:.3f}"
        )

        results.append(
            {
                "split": i + 1,
                "train_bars": len(X_train_full),
                "test_bars": len(X_test),
                "accuracy": acc,
                "acc_high_conf": conf_acc,
                "high_conf_pct": fraction_conf,
                "baseline_up": baseline,
            }
        )
        feature_importance_sum += model.feature_importances_

        if "time" in df.columns:
            for ts, prob in zip(df["time"].iloc[train_end:test_end], preds):
                oos_proba_map[pd.Timestamp(ts)] = float(prob)

    final_model, _, _ = _fit_lgbm_with_inner_validation_daily(
        df[feature_cols], df["target"]
    )
    split_count = max(len(split_points), 1)
    return {
        "splits": results,
        "feature_importance": dict(
            zip(feature_cols, feature_importance_sum / split_count)
        ),
        "final_model": final_model,
        "feature_cols": feature_cols,
        "df": df,
        "overall_accuracy": np.mean([r["accuracy"] for r in results]),
        "overall_baseline_up": np.mean([r["baseline_up"] for r in results]),
        "oos_proba_map": oos_proba_map,
    }


def fold_consensus_feature_selection_daily(
    df: pd.DataFrame,
    candidate_feature_cols: list,
    n_splits: int = 10,
    min_train_bars: int = MIN_TRAIN_BARS_DAILY,
    test_size_ratio: float = TEST_SIZE_RATIO_DAILY,
    purge_gap: int = PURGE_GAP_DAILY,
    consensus_threshold: float = 0.50,
    target_col: str = "target",
) -> tuple[list, dict]:
    from holographic_engine import correlation_filter, phase1_ranking

    df = df.dropna(
        subset=[c for c in candidate_feature_cols if c in df.columns] + [target_col]
    ).reset_index(drop=True)
    n = len(df)

    if n < min_train_bars:
        print(f"  [Nested Daily] Not enough data. Need {min_train_bars}, got {n}.")
        return [], {}

    split_points: list[tuple[int, int]] = []
    current_train_end = min_train_bars
    while current_train_end < n:
        test_window_size = max(int(n * test_size_ratio), 100)
        test_end = min(current_train_end + test_window_size, n)
        if test_end <= current_train_end or (test_end - current_train_end) < 50:
            break
        split_points.append((current_train_end, test_end))
        current_train_end = test_end
        if len(split_points) >= n_splits:
            break

    if not split_points:
        print("  [Nested Daily] No valid split points.")
        return [], {}

    valid_candidates = [c for c in candidate_feature_cols if c in df.columns]
    print(
        f"\n  [Nested Daily] {len(split_points)} folds, {len(valid_candidates)} candidates"
    )

    feature_votes: dict[str, int] = {}
    fold_results = []

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
            fold_features = fold_features[:FINAL_FEAT_BUDGET]
        except Exception as exc:
            print(f"  [Nested Daily] Fold {i + 1} feature selection failed: {exc}")
            continue

        if not fold_features:
            continue

        for feat in fold_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1

        X_train = train_df[fold_features]
        y_train = train_df[target_col]
        try:
            model, _, _ = _fit_lgbm_with_inner_validation_daily(X_train, y_train)
        except Exception as exc:
            print(f"  [Nested Daily] Fold {i + 1} training failed: {exc}")
            continue

        X_test = test_df[fold_features]
        y_test = test_df[target_col]
        preds = model.predict(X_test)
        true_binary = (y_test > 0.5).astype(int)
        pred_binary = (preds > 0.5).astype(int)
        acc = float((true_binary == pred_binary).mean())
        high_conf_mask = (preds >= LIVE_CONFIDENCE_THRESHOLD_DAILY) | (
            preds <= (1.0 - LIVE_CONFIDENCE_THRESHOLD_DAILY)
        )
        conf_acc = (
            float((pred_binary[high_conf_mask] == true_binary[high_conf_mask]).mean())
            if high_conf_mask.sum() > 0
            else float("nan")
        )

        print(
            f"  [Nested Daily] Fold {i + 1:>2} | Train:{len(X_train):>5} "
            f"Test:{len(X_test):>5} | Acc:{acc:.3f} ConfAcc:{conf_acc:.3f} | "
            f"Feats:{len(fold_features)}"
        )
        fold_results.append(
            {
                "split": i + 1,
                "train_bars": len(X_train),
                "test_bars": len(X_test),
                "accuracy": acc,
                "acc_high_conf": conf_acc,
                "high_conf_pct": float(high_conf_mask.sum() / len(preds)),
                "baseline_up": float(true_binary.mean()),
            }
        )

    min_votes = max(1, int(len(split_points) * consensus_threshold))
    consensus_features = [
        feat
        for feat, votes in sorted(feature_votes.items(), key=lambda item: -item[1])
        if votes >= min_votes
    ][:FINAL_FEAT_BUDGET]

    print(
        f"\n  [Nested Daily] Consensus: {len(consensus_features)} features "
        f"(voted in >={min_votes}/{len(split_points)} folds)"
    )
    if not consensus_features:
        print("  [Nested Daily] FATAL: No features survived consensus vote.")
        return [], {}

    print("  [Nested Daily] Final walk-forward with consensus features...")
    final_wf = walk_forward_daily(
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
    parser = argparse.ArgumentParser(description="Daily ML Direction Predictor")
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/km/Universal-ML/",
        help="Output directory for models",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Target symbol (e.g., NIFTY, BTCUSDT, AAPL, ^GDAXI)",
    )
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
    file_prefix = str(artifact_ctx["file_prefix"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    artifact_paths_1d = get_artifact_paths(symbol_dir, file_prefix, "1D")

    print("=" * 70)
    if requested_symbol != symbol:
        print(
            f"  INITIATING DAILY DATABASE UPLINK FOR: {symbol} "
            f"[requested {requested_symbol}]"
        )
    else:
        print(f"  INITIATING DAILY DATABASE UPLINK FOR: {symbol}")
    print("=" * 70)

    sys.path.append(os.path.join(data_dir, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("  [!] FATAL: Cannot locate inference_bridge.py in data_vault directory.")
        raise SystemExit(1)

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
        print(f"  [!] FATAL: No usable 1D primary data found for {symbol}.")
        raise SystemExit(1)

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

    print(
        "\n  [TOON DAILY] Injecting Universal Basis Mechanics (Primary-to-Reference)..."
    )
    df_1d = inject_thermodynamic_basis(
        df_1d,
        reference_frames["1D"],
        logger=print,
    )

    # Daily bars do not carry intraday session meaning; keep inert placeholders.
    df_1d["session_time_pos"] = 0.0
    df_1d["eod_basis_momentum"] = 0.0

    print("  [TOON DAILY] Building RV Surface (Julia RV engine)...")
    rv_df = rv_feature_engine_daily(df_1d, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        df_1d[col] = rv_df[col].values
    print(f"  [TOON DAILY] RV features injected: {len(rv_df.columns)} columns")

    print("  [TOON DAILY] Layer 1: Julia holographic extraction (1D + 1W/1M/3M)...")
    df_1d_labelled = _compute_atr14(df_1d.copy())
    df_full = holographic_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_3m)

    # ── SMC institutional intent engine (42 features, 1D lane) ────────────
    print("  [TOON DAILY] Layer 1b: SMC Feature Engine (42 institutional signals)...")
    smc_df = smc_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    print(f"  [TOON DAILY] SMC features injected: {len(smc_df.columns)} columns")

    print("  [TOON DAILY] Layer 1c: Kalman structural feature family...")
    kf_df = kalman_structural_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in kf_df.columns:
        df_full[col] = kf_df[col].values
    print(
        f"  [TOON DAILY] Kalman structural features injected: {len(kf_df.columns)} columns"
    )

    print(
        "  [TOON DAILY] Layer 1d: Narrative Context Awareness (23 context signals)..."
    )
    nc_df = narrative_context_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m)
    for col in nc_df.columns:
        df_full[col] = nc_df[col].values
    print(f"  [TOON DAILY] Narrative context injected: {len(nc_df.columns)} columns")

    print("  [TOON DAILY] Layer 2: Injecting macro regime overlays (6M/12M)...")
    df_full = inject_macro_regime(df_full, df_6m, "6m")
    df_full = inject_macro_regime(df_full, df_12m, "12m")

    print("  [TOON DAILY] Layer 3: Adding daily confluence...")
    df_full = add_daily_confluence(df_full)

    print("  [TOON DAILY] Labelling daily targets via trade simulation...")
    df_full = add_target_fast(
        df_full,
        atr_mult=BARRIER_ATR_MULT_DAILY,
        horizon=BARRIER_HORIZON_BARS_DAILY,
        atr_col="atr14",
        drop_unresolved=False,
    )

    all_holo_cols = [c for c in df_full.columns if c not in NON_FEATURE_COLS_DAILY]
    # Preserve raw OHLC state for downstream trade-plan / policy replay.
    state_cols = ["target", "time", "open", "high", "low", "close", "atr14"]
    for col in [
        "basis_pct",
        "basis_z_score",
        "basis_vel_5",
        "basis_vel_10",
        "session_time_pos",
        "eod_basis_momentum",
    ]:
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
    df_model_ready = df_model_ready.dropna(subset=all_holo_cols).reset_index(drop=True)

    print(f"  [TOON DAILY] Bars after labelling & cleaning : {len(df_model_ready)}")
    print(f"  [TOON DAILY] Features before selection       : {len(all_holo_cols)}")
    print(
        f"  [TOON DAILY] Target (UP=1) distribution      : {df_model_ready['target'].mean():.1%}"
    )

    if len(df_model_ready) < MIN_TRAIN_BARS_DAILY + 100:
        print(f"\n  Error: Not enough {symbol} daily data for walk-forward validation.")
        raise SystemExit(1)

    feature_cols, wf_results = fold_consensus_feature_selection_daily(
        df_model_ready,
        all_holo_cols,
        n_splits=10,
        min_train_bars=MIN_TRAIN_BARS_DAILY,
        test_size_ratio=TEST_SIZE_RATIO_DAILY,
    )

    if not feature_cols or not wf_results:
        print("  [TOON DAILY] Nested feature selection failed. Aborting.")
        raise SystemExit(1)

    print(f"\n  [TOON DAILY v6.0] Consensus feature count: {len(feature_cols)}")
    print("\n" + "=" * 70)
    print("  Walk-Forward Validation — TOON v5.1 Daily Holographic Model")
    print("=" * 70)
    if not wf_results:
        print(
            "\nDaily walk-forward validation did not produce results. Please check errors above."
        )
        raise SystemExit(1)

    splits_df = pd.DataFrame(wf_results["splits"])
    print("\n" + "=" * 70)
    print("  SUMMARY — TOON v5.1 DAILY HOLOGRAPHIC WALK-FORWARD")
    print("=" * 70)
    print(
        f"  Overall Accuracy (All Signals)      : {wf_results['overall_accuracy']:.3f}"
    )
    print(
        f"  Overall High-Conf Accuracy (>={LIVE_CONFIDENCE_THRESHOLD_DAILY:.2f}) : "
        f"{splits_df['acc_high_conf'].mean():.3f}"
    )
    print(
        f"  Average High-Confidence Bar %       : {splits_df['high_conf_pct'].mean():.1%}"
    )
    print(
        f"  Average Always-UP Baseline          : {wf_results['overall_baseline_up']:.3f}"
    )
    print(
        f"  Edge over Baseline                  : "
        f"{wf_results['overall_accuracy'] - wf_results['overall_baseline_up']:+.3f}"
    )
    print(f"  Number of splits performed          : {len(splits_df)}")
    print(f"  Total bars used for validation      : {splits_df['test_bars'].sum()}")

    final_feats = wf_results["feature_cols"]
    dna_n = sum(1 for c in final_feats if "_bar" in c)
    gram_n = sum(1 for c in final_feats if "_gram_" in c)
    fft_n = sum(1 for c in final_feats if "_fft_" in c)
    skel_n = sum(1 for c in final_feats if "_skel_" in c)
    conf_n = sum(1 for c in final_feats if c.startswith("mtf_conf"))
    print("\n  Feature layer breakdown in final daily model:")
    print(
        f"    DNA:  {dna_n:3d}  |  Grammar: {gram_n:3d}  |  Spectral: {fft_n:3d}"
        f"  |  Skeleton: {skel_n:3d}  |  Confluence: {conf_n:3d}"
    )

    model_path = artifact_paths_1d["model"]
    feat_path = artifact_paths_1d["features"]
    oos_path = artifact_paths_1d["oos_proba"]
    trade_plan_path = artifact_paths_1d["trade_plan_models"]

    joblib.dump(wf_results["final_model"], model_path)
    with open(feat_path, "w") as handle:
        for col in wf_results["feature_cols"]:
            handle.write(f"{col}\n")
    joblib.dump(wf_results["oos_proba_map"], oos_path)
    calibrator = calibrate_oos_probabilities(
        wf_results["oos_proba_map"], df_model_ready
    )
    if calibrator is not None:
        cal_path = artifact_paths_1d["calibrator"]
        joblib.dump(calibrator, cal_path)
        print(f"  1D calibrator saved to '{cal_path}'")

    tp_train_end = len(df_model_ready) - int(
        len(df_model_ready) * TEST_SIZE_RATIO_DAILY
    )
    trade_plan_models = train_trade_plan_models(
        df_model_ready.iloc[:tp_train_end],
        wf_results["feature_cols"],
    )
    joblib.dump(trade_plan_models, trade_plan_path)
    prob_array_full = build_prob_array_from_oos_map(
        df_model_ready["time"],
        wf_results["oos_proba_map"],
        calibrator=calibrator,
    )
    exit_surface_artifact = train_exit_surface_artifact(
        df_model_ready,
        prob_array_full,
        lane="1D",
        start_idx=tp_train_end,
        max_hold_bars=BARRIER_HORIZON_BARS_DAILY,
        eod_gate_hour=POLICY_EOD_GATE_HOUR_1D,
    )
    if exit_surface_artifact is not None:
        exit_surface_path = artifact_paths_1d["exit_surface"]
        joblib.dump(exit_surface_artifact, exit_surface_path)
        print(
            f"  1D exit surface saved to '{exit_surface_path}' "
            f"({exit_surface_artifact['metadata']['candidate_rows']} candidates)"
        )
    else:
        print(
            "  [Exit Surface] Skipped 1D artifact: insufficient honest candidate trades."
        )
    policy_artifact = train_policy_artifact(
        df_model_ready,
        wf_results["feature_cols"],
        prob_array_full,
        trade_plan_models,
        exit_surface_artifact=exit_surface_artifact,
        lane="1D",
        confidence_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
        start_idx=tp_train_end,
        max_hold_bars=BARRIER_HORIZON_BARS_DAILY,
        eod_gate_hour=POLICY_EOD_GATE_HOUR_1D,
    )
    if policy_artifact is not None:
        policy_path = artifact_paths_1d["policy_artifact"]
        joblib.dump(policy_artifact, policy_path)
        print(
            f"  1D policy artifact saved to '{policy_path}' "
            f"({policy_artifact['metadata']['candidate_rows']} candidates)"
        )
    else:
        print(
            "  [Policy] Skipped 1D policy artifact: insufficient honest candidate trades."
        )

    print(f"\n  1D model saved to '{model_path}'")
    print(f"  1D feature list saved to '{feat_path}'")
    print(
        f"  1D OOS proba map saved to '{oos_path}' ({len(wf_results['oos_proba_map'])} bars)"
    )
    print(
        f"  1D trade-plan models saved to '{trade_plan_path}' ({len(trade_plan_models)} models)"
    )

    last_row = df_model_ready.iloc[-1]
    pred = predict_next_bar(
        wf_results["final_model"],
        wf_results["feature_cols"],
        last_row,
        confidence_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
        calibrator=calibrator,
        policy_artifact=policy_artifact,
        policy_lane="1D",
    )
    primary_asset_label = str(df_1d.attrs.get("selected_asset_class", "PRIMARY"))
    close_price = float(last_row["close"])
    atr_value = float(df_full["atr14"].iloc[-1]) if "atr14" in df_full.columns else 1.0
    trade_plan = predict_trade_plan(
        trade_plan_models,
        wf_results["feature_cols"],
        last_row.copy(),
        pred["direction"],
        atr_value,
        exit_surface_artifact=exit_surface_artifact,
        proba_up=float(pred.get("calibrated_score", pred.get("raw_score", np.nan))),
        lane="1D",
    )

    trail_str = "N/A"
    pred, filter_note = finalize_forecast_context(
        pred,
        timeframe="1D",
        bar_time=last_row.get("time"),
        confidence_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
        base_note=trade_plan["note"],
    )
    if pred.get("policy_filtered"):
        filter_note = f"{filter_note} POLICY_FILTERED".strip()
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
        sl_str = "N/A"
        tp1_str = "N/A"
        tp2_str = "N/A"

    print("\n" + "=" * 70)
    print(f"  {symbol} DAILY FORECAST (TOON v5.1)")
    print("=" * 70)
    print(f"  Bar time    : {last_row['time']}")
    print(f"  Direction   : {pred['direction']}")
    print(
        f"  Confidence  : {pred['confidence']:.1%} "
        f"(Raw: {pred['raw_score']:.3f} | Cal: {pred['calibrated_score']:.3f})"
    )
    if np.isfinite(pred.get("policy_score", np.nan)):
        print(
            f"  Policy      : {pred['policy_score']:.3f} "
            f"(Risk x{pred.get('policy_risk_mult', 1.0):.2f})"
        )
    print(f"  Signal      : {pred['signal_strength']}")
    print("----------------------------------------------------------------------")
    print(f"  Entry Price : {close_price:,.2f} (Current {primary_asset_label} Close)")
    print(f"  Stop Loss   : {sl_str}")
    print(f"  Target 1    : {tp1_str}")
    print(f"  Target 2    : {tp2_str}")
    print(f"  Trail Stop  : {trail_str}")
    if filter_note:
        print(f"  [{filter_note}]")
    print("======================================================================")

    pred_info = {
        "symbol": symbol,
        "forecast_label": "DAILY FORECAST",
        "time": str(last_row["time"]),
        "dir": pred["direction"],
        "conf": f"{pred['confidence'] * 100:.1f}%",
        "signal": pred["signal_strength"],
        "entry": f"{close_price:,.2f} (Current {primary_asset_label} Close)",
        "sl": sl_str,
        "tp1": tp1_str,
        "tp2": tp2_str,
        "trail": trail_str,
        "note": filter_note,
    }
    report_path = artifact_paths_1d["ml_report"]
    save_report(
        wf_results,
        report_path,
        pred_info=pred_info,
        symbol=symbol,
        data_update_lines=build_report_data_lines({"1D": df_1d}),
    )

    fi_sorted = sorted(
        wf_results["feature_importance"].items(), key=lambda x: x[1], reverse=True
    )
    print("\n  Top 15 most predictive daily features:")
    for i, (feature_name, feature_value) in enumerate(fi_sorted[:15]):
        print(f"    {i + 1}. {feature_name:<42} {feature_value:.2f}")


if __name__ == "__main__":
    main()
