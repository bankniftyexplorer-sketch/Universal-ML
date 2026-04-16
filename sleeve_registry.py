"""
Sleeve Admission Layer v1
=========================

Builds a symbol/lane registry from saved artifacts and honest replay metrics.
The registry decides whether a sleeve is tradable and whether `base` or
`policy` execution should be used.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from daily_ml_engine import (
    LIVE_CONFIDENCE_THRESHOLD_DAILY,
    add_daily_confluence,
    inject_macro_regime,
)
from inference_bridge import InferenceBridge
from instrument_registry import resolve_instrument_identity
from julia_bridge import (
    holographic_feature_engine_daily,
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
    LIVE_CONFIDENCE_THRESHOLD,
    _compute_atr14,
    build_prob_array_from_oos_map,
    build_timeframe_selection,
    inject_thermodynamic_basis,
    merge_higher_tf,
    prepare_symbol_artifact_context,
    prepare_intraday_thermodynamics,
    resolve_artifact_path,
    train_policy_artifact,
)

DEFAULT_REGISTRY_PATH = Path(__file__).with_name("portfolio_sleeve_registry.json")
MIN_PROFIT_FACTOR = 1.25
MIN_SHARPE = 0.30
MAX_DRAWDOWN = 0.20
MIN_TRADES = 8


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_sleeve_registry(registry_path: str | None = None) -> dict[str, Any]:
    path = Path(registry_path) if registry_path else DEFAULT_REGISTRY_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_sleeve_registry_entry(
    symbol: str,
    lane: str,
    registry_path: str | None = None,
) -> dict[str, Any] | None:
    registry = load_sleeve_registry(registry_path)
    sleeves = registry.get("sleeves", {})
    canonical_symbol = resolve_instrument_identity(symbol).canonical_symbol
    return sleeves.get(f"{canonical_symbol}_{lane.upper()}") or sleeves.get(
        f"{symbol.upper()}_{lane.upper()}"
    )


def _safe_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _build_intraday_conflict_array(
    df_backtest: pd.DataFrame,
    *,
    symbol_dir: str,
    file_prefix: str,
) -> np.ndarray:
    oos_1d_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "oos_proba")
    cal_1d_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "calibrator")
    if not os.path.exists(oos_1d_path):
        return np.full(len(df_backtest), np.nan, dtype=float)

    oos_map = joblib.load(oos_1d_path)
    calibrator = joblib.load(cal_1d_path) if os.path.exists(cal_1d_path) else None
    mapped = {
        pd.Timestamp(ts).date(): float(prob)
        for ts, prob in oos_map.items()
    }
    probs = np.array(
        [mapped.get(pd.Timestamp(ts).date(), np.nan) for ts in df_backtest["time"]],
        dtype=float,
    )
    return build_prob_array_from_oos_map(
        df_backtest["time"],
        {pd.Timestamp(ts): float(val) for ts, val in zip(df_backtest["time"], probs)},
        calibrator=calibrator,
    )


def _build_1h_bundle(
    bridge: InferenceBridge,
    outdir: str,
    symbol: str,
    *,
    refresh_policy_artifact: bool = True,
    persist_policy_artifact: bool = True,
) -> dict[str, Any]:
    artifact_ctx = prepare_symbol_artifact_context(
        outdir,
        symbol,
        asset_class="SPOT",
        timeframes=("1H", "1D"),
        logger=None,
    )
    symbol = str(artifact_ctx["symbol"])
    file_prefix = str(artifact_ctx["file_prefix"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    tf_maps = {
        "SPOT": bridge.fetch_holographic_stack(
            str(artifact_ctx["identity"].market_data_symbol),
            "SPOT",
        )
    }
    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1H", "1D", "1W", "1M")
    )
    df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
        df_1h=primary_frames["1H"],
        df_1d=primary_frames["1D"],
        df_1w=primary_frames["1W"],
        df_1m=primary_frames["1M"],
        reference_1h=reference_frames["1H"],
        reference_1d=reference_frames["1D"],
        reference_1w=reference_frames["1W"],
        symbol=symbol,
        logger=None,
    )

    df_1h_labelled = _compute_atr14(df_1h.copy())
    df_full = holographic_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for feat_df in (
        smc_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m),
        kalman_structural_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m),
        rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m),
        narrative_context_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m),
    ):
        for col in feat_df.columns:
            df_full[col] = feat_df[col].values
    df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)

    features_path = resolve_artifact_path(symbol_dir, file_prefix, "1H", "features")
    with open(features_path, "r", encoding="utf-8") as handle:
        feature_cols = [line.strip() for line in handle if line.strip()]

    required_cols = list(
        dict.fromkeys(
            feature_cols
            + ["time", "open", "high", "low", "close", "volume", "atr14", "basis_z_score"]
        )
    )
    df_backtest = df_full[[col for col in required_cols if col in df_full.columns]].copy()
    df_backtest = df_backtest.reset_index(drop=True)
    for col in feature_cols:
        if col not in df_backtest.columns:
            df_backtest[col] = 0.0
        df_backtest[col] = df_backtest[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    oos_path = resolve_artifact_path(symbol_dir, file_prefix, "1H", "oos_proba")
    cal_path = resolve_artifact_path(symbol_dir, file_prefix, "1H", "calibrator")
    tp_path = resolve_artifact_path(symbol_dir, file_prefix, "1H", "trade_plan_models")
    exit_surface_path = resolve_artifact_path(symbol_dir, file_prefix, "1H", "exit_surface")
    policy_path = resolve_artifact_path(symbol_dir, file_prefix, "1H", "policy_artifact")

    oos_map = joblib.load(oos_path)
    calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
    prob_array = build_prob_array_from_oos_map(
        df_backtest["time"], oos_map, calibrator=calibrator
    )
    trade_plan_models = joblib.load(tp_path) if os.path.exists(tp_path) else {}
    exit_surface_artifact = (
        joblib.load(exit_surface_path) if os.path.exists(exit_surface_path) else None
    )
    prob_array_1d = _build_intraday_conflict_array(
        df_backtest,
        symbol_dir=symbol_dir,
        file_prefix=file_prefix,
    )

    policy_artifact = joblib.load(policy_path) if os.path.exists(policy_path) else None
    if refresh_policy_artifact:
        policy_artifact = train_policy_artifact(
            df_backtest,
            feature_cols,
            prob_array,
            trade_plan_models,
            exit_surface_artifact=exit_surface_artifact,
            lane="1H",
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
            start_idx=len(df_backtest) - int(len(df_backtest) * 0.15),
        )
    if policy_artifact is not None and persist_policy_artifact:
        joblib.dump(policy_artifact, policy_path)

    return {
        "lane": "1H",
        "symbol_dir": symbol_dir,
        "feature_cols": feature_cols,
        "df_backtest": df_backtest,
        "prob_array": prob_array,
        "prob_array_1d": prob_array_1d,
        "trade_plan_models": trade_plan_models,
        "exit_surface_artifact": exit_surface_artifact,
        "policy_artifact": policy_artifact,
        "conf_threshold": LIVE_CONFIDENCE_THRESHOLD,
        "max_hold_bars": 24,
        "eod_gate_hour": 14,
    }


def _build_1d_bundle(
    bridge: InferenceBridge,
    outdir: str,
    symbol: str,
    *,
    refresh_policy_artifact: bool = True,
    persist_policy_artifact: bool = True,
) -> dict[str, Any]:
    artifact_ctx = prepare_symbol_artifact_context(
        outdir,
        symbol,
        asset_class="SPOT",
        timeframes=("1D",),
        logger=None,
    )
    symbol = str(artifact_ctx["symbol"])
    file_prefix = str(artifact_ctx["file_prefix"])
    symbol_dir = str(artifact_ctx["symbol_dir"])
    tf_maps = {
        "SPOT": bridge.fetch_holographic_stack(
            str(artifact_ctx["identity"].market_data_symbol),
            "SPOT",
            include_realized_vol=True,
        )
    }
    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1D", "1W", "1M", "3M", "6M", "12M")
    )
    df_1d = inject_thermodynamic_basis(
        primary_frames["1D"],
        reference_frames["1D"],
        logger=None,
    )
    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]
    df_3m = primary_frames["3M"]
    df_6m = primary_frames["6M"]
    df_12m = primary_frames["12M"]

    df_1d["session_time_pos"] = 0.0
    df_1d["eod_basis_momentum"] = 0.0
    rv_df = rv_feature_engine_daily(df_1d, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        df_1d[col] = rv_df[col].values

    df_1d_labelled = _compute_atr14(df_1d.copy())
    df_full = holographic_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_3m)
    for feat_df in (
        smc_feature_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m),
        kalman_structural_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m),
        narrative_context_engine_daily(df_1d_labelled, df_1w, df_1m, df_6m),
    ):
        for col in feat_df.columns:
            df_full[col] = feat_df[col].values
    df_full = inject_macro_regime(df_full, df_6m, "6m")
    df_full = inject_macro_regime(df_full, df_12m, "12m")
    df_full = add_daily_confluence(df_full)

    features_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "features")
    with open(features_path, "r", encoding="utf-8") as handle:
        feature_cols = [line.strip() for line in handle if line.strip()]

    required_cols = list(
        dict.fromkeys(
            feature_cols
            + ["time", "open", "high", "low", "close", "volume", "atr14", "basis_z_score"]
        )
    )
    df_backtest = df_full[[col for col in required_cols if col in df_full.columns]].copy()
    df_backtest = df_backtest.reset_index(drop=True)
    for col in feature_cols:
        if col not in df_backtest.columns:
            df_backtest[col] = 0.0
        df_backtest[col] = df_backtest[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    oos_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "oos_proba")
    cal_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "calibrator")
    tp_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "trade_plan_models")
    exit_surface_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "exit_surface")
    policy_path = resolve_artifact_path(symbol_dir, file_prefix, "1D", "policy_artifact")

    oos_map = joblib.load(oos_path)
    calibrator = joblib.load(cal_path) if os.path.exists(cal_path) else None
    prob_array = build_prob_array_from_oos_map(
        df_backtest["time"], oos_map, calibrator=calibrator
    )
    trade_plan_models = joblib.load(tp_path) if os.path.exists(tp_path) else {}
    exit_surface_artifact = (
        joblib.load(exit_surface_path) if os.path.exists(exit_surface_path) else None
    )

    policy_artifact = joblib.load(policy_path) if os.path.exists(policy_path) else None
    if refresh_policy_artifact:
        policy_artifact = train_policy_artifact(
            df_backtest,
            feature_cols,
            prob_array,
            trade_plan_models,
            exit_surface_artifact=exit_surface_artifact,
            lane="1D",
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
            start_idx=len(df_backtest) - int(len(df_backtest) * 0.15),
            max_hold_bars=10,
            eod_gate_hour=24,
        )
    if policy_artifact is not None and persist_policy_artifact:
        joblib.dump(policy_artifact, policy_path)

    return {
        "lane": "1D",
        "symbol_dir": symbol_dir,
        "feature_cols": feature_cols,
        "df_backtest": df_backtest,
        "prob_array": prob_array,
        "prob_array_1d": np.full(len(df_backtest), np.nan, dtype=float),
        "trade_plan_models": trade_plan_models,
        "exit_surface_artifact": exit_surface_artifact,
        "policy_artifact": policy_artifact,
        "conf_threshold": LIVE_CONFIDENCE_THRESHOLD_DAILY,
        "max_hold_bars": 10,
        "eod_gate_hour": 24,
    }


def _evaluate_variant(bundle: dict[str, Any], *, variant: str) -> dict[str, Any] | None:
    from backtest_engine import calculate_metrics, run_backtest

    use_policy = variant == "policy"
    if use_policy and bundle["policy_artifact"] is None:
        return None

    results = run_backtest(
        bundle["df_backtest"],
        bundle["prob_array"],
        bundle["prob_array_1d"],
        bundle["feature_cols"],
        trade_plan_models=bundle["trade_plan_models"],
        exit_surface_artifact=bundle.get("exit_surface_artifact"),
        policy_artifact=bundle["policy_artifact"] if use_policy else None,
        conf_threshold=bundle["conf_threshold"],
        max_hold_bars=bundle["max_hold_bars"],
        eod_gate_hour=bundle["eod_gate_hour"],
        lane=bundle["lane"],
    )
    metrics = calculate_metrics(
        results["trades"],
        results.get("equity_curve"),
        results.get("time_curve"),
    )
    return {
        "variant": variant,
        "enabled": False,
        "sharpe": _safe_float(metrics.get("sharpe")),
        "profit_factor": _safe_float(metrics.get("profit_factor")),
        "max_drawdown": _safe_float(results.get("max_drawdown")),
        "total_trades": int(metrics.get("total_trades", 0)),
        "final_equity": _safe_float(results.get("final_equity")),
        "generated_at_utc": _utc_now(),
    }


def _passes_admission(variant_metrics: dict[str, Any] | None) -> bool:
    if not variant_metrics:
        return False
    sharpe = _safe_float(variant_metrics.get("sharpe"))
    profit_factor = _safe_float(variant_metrics.get("profit_factor"))
    max_drawdown = _safe_float(variant_metrics.get("max_drawdown"))
    total_trades = int(variant_metrics.get("total_trades", 0))
    if sharpe is None or profit_factor is None or max_drawdown is None:
        return False
    return (
        profit_factor >= MIN_PROFIT_FACTOR
        and sharpe >= MIN_SHARPE
        and max_drawdown <= MAX_DRAWDOWN
        and total_trades >= MIN_TRADES
    )


def _select_variant(base_metrics: dict[str, Any] | None, policy_metrics: dict[str, Any] | None) -> tuple[str | None, dict[str, Any] | None, str]:
    candidates: list[dict[str, Any]] = []
    for metrics in (base_metrics, policy_metrics):
        if _passes_admission(metrics):
            candidates.append(metrics)

    if not candidates:
        return None, None, "No variant met admission criteria."

    winner = max(
        candidates,
        key=lambda item: (
            _safe_float(item.get("sharpe")) or float("-inf"),
            _safe_float(item.get("final_equity")) or float("-inf"),
        ),
    )
    return str(winner["variant"]), winner, f"Selected {winner['variant']} by Sharpe/equity."


def _build_entry(
    symbol: str,
    lane: str,
    base_metrics: dict[str, Any] | None,
    policy_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    identity = resolve_instrument_identity(symbol)
    symbol = identity.canonical_symbol
    sleeve_id = f"{symbol}_{lane}"
    selected_variant, selected_metrics, reason = _select_variant(base_metrics, policy_metrics)
    enabled = selected_variant is not None and selected_metrics is not None
    selected_metrics = selected_metrics or {}
    return {
        "sleeve_id": sleeve_id,
        "symbol": symbol,
        "canonical_symbol": identity.canonical_symbol,
        "display_symbol": identity.display_symbol,
        "alias_set": list(identity.alias_set),
        "lane": lane,
        "variant": selected_variant,
        "selected_variant": selected_variant,
        "enabled": enabled,
        "sharpe": _safe_float(selected_metrics.get("sharpe")),
        "profit_factor": _safe_float(selected_metrics.get("profit_factor")),
        "max_drawdown": _safe_float(selected_metrics.get("max_drawdown")),
        "total_trades": int(selected_metrics.get("total_trades", 0)),
        "final_equity": _safe_float(selected_metrics.get("final_equity")),
        "generated_at_utc": _utc_now(),
        "reason": reason,
        "variants": {
            "base": base_metrics,
            "policy": policy_metrics,
        },
    }


def build_registry_for_symbols(outdir: str, symbols: list[str]) -> dict[str, Any]:
    bridge = InferenceBridge(db_path=os.path.join(outdir, "data_vault", "ohlcv.db"))
    sleeves: dict[str, Any] = {}

    for raw_symbol in symbols:
        symbol = resolve_instrument_identity(raw_symbol, outdir=outdir).canonical_symbol
        try:
            intraday_bundle = _build_1h_bundle(bridge, outdir, symbol)
            base_metrics = _evaluate_variant(intraday_bundle, variant="base")
            policy_metrics = _evaluate_variant(intraday_bundle, variant="policy")
            entry = _build_entry(symbol, "1H", base_metrics, policy_metrics)
            sleeves[entry["sleeve_id"]] = entry
            print(
                f"[1H] {symbol}: enabled={entry['enabled']} variant={entry['selected_variant']}",
                flush=True,
            )
        except Exception as exc:
            sleeves[f"{symbol}_1H"] = {
                "sleeve_id": f"{symbol}_1H",
                "symbol": symbol,
                "lane": "1H",
                "variant": None,
                "selected_variant": None,
                "enabled": False,
                "sharpe": None,
                "profit_factor": None,
                "max_drawdown": None,
                "total_trades": 0,
                "final_equity": None,
                "generated_at_utc": _utc_now(),
                "reason": f"Replay failed: {exc}",
                "variants": {"base": None, "policy": None},
            }
            print(f"[1H] {symbol}: ERROR {exc}", flush=True)

        try:
            daily_bundle = _build_1d_bundle(bridge, outdir, symbol)
            base_metrics = _evaluate_variant(daily_bundle, variant="base")
            policy_metrics = _evaluate_variant(daily_bundle, variant="policy")
            entry = _build_entry(symbol, "1D", base_metrics, policy_metrics)
            sleeves[entry["sleeve_id"]] = entry
            print(
                f"[1D] {symbol}: enabled={entry['enabled']} variant={entry['selected_variant']}",
                flush=True,
            )
        except Exception as exc:
            sleeves[f"{symbol}_1D"] = {
                "sleeve_id": f"{symbol}_1D",
                "symbol": symbol,
                "lane": "1D",
                "variant": None,
                "selected_variant": None,
                "enabled": False,
                "sharpe": None,
                "profit_factor": None,
                "max_drawdown": None,
                "total_trades": 0,
                "final_equity": None,
                "generated_at_utc": _utc_now(),
                "reason": f"Replay failed: {exc}",
                "variants": {"base": None, "policy": None},
            }
            print(f"[1D] {symbol}: ERROR {exc}", flush=True)

    return {
        "generated_at_utc": _utc_now(),
        "criteria": {
            "min_profit_factor": MIN_PROFIT_FACTOR,
            "min_sharpe": MIN_SHARPE,
            "max_drawdown": MAX_DRAWDOWN,
            "min_trades": MIN_TRADES,
        },
        "sleeves": sleeves,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portfolio sleeve registry")
    parser.add_argument("--outdir", default="/home/km/Universal-ML/")
    parser.add_argument("--symbols", nargs="+", required=True)
    parser.add_argument("--write-path", default=str(DEFAULT_REGISTRY_PATH))
    args = parser.parse_args()

    registry = build_registry_for_symbols(os.path.abspath(args.outdir), args.symbols)
    write_path = Path(args.write_path)
    write_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"[registry] wrote {write_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
