"""
verify_julia_backtest_hurst.py
==============================
Parity checks for the selective Julia migration on the backtest path.

Covered slices:
- rolling Hurst-style rough-volatility series
- per-bar backtest gate-state precompute
- synthetic end-to-end backtest parity
- real artifact-backed intraday and daily backtest parity
- warmed runtime benchmark for the migrated intraday slice
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from daily_backtest_engine import (
    BARRIER_HORIZON_BARS_DAILY,
    EOD_GATE_HOUR_DAILY,
    LIVE_CONFIDENCE_THRESHOLD_DAILY,
    _get_tf,
    _pick_primary_1d,
)
from backtest_engine import (
    _compute_hurst_array_python,
    _compute_atr14,
    _precompute_backtest_bar_state_python,
    fib_structural_basis,
    holographic_feature_engine,
    merge_higher_tf,
    migrate_legacy_artifacts,
    prepare_intraday_thermodynamics,
    resolve_artifact_path,
    run_backtest,
)
from inference_bridge import InferenceBridge
from julia_bridge import (
    compute_backtest_bar_state_fast,
    compute_hurst_fast,
    holographic_feature_engine_daily,
)
from universal_ml_engine import inject_thermodynamic_basis


def _assert_close(name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
    if not np.allclose(lhs, rhs, atol=1e-12, rtol=0.0, equal_nan=True):
        diff_idx = np.where(
            ~np.isclose(lhs, rhs, atol=1e-12, rtol=0.0, equal_nan=True)
        )[0]
        preview = diff_idx[:5].tolist()
        raise AssertionError(f"{name} mismatch at indices {preview}")


def _assert_equal(name: str, lhs: np.ndarray, rhs: np.ndarray) -> None:
    if not np.array_equal(lhs, rhs):
        diff_idx = np.where(lhs != rhs)[0]
        preview = diff_idx[:5].tolist()
        raise AssertionError(f"{name} mismatch at indices {preview}")


def _iter_symbol_dirs(project_dir: Path) -> list[Path]:
    return sorted(
        (path for path in project_dir.iterdir() if path.is_dir()),
        key=lambda path: (path.name != "NIFTY", path.name),
    )


def _assert_backtest_match(
    name: str, py_result: dict, jl_result: dict, bars: int | None = None
) -> None:
    scalar_keys = [
        "final_equity",
        "max_drawdown",
        "conflict_blocks",
        "volatility_blocks",
        "shock_blocks",
        "no_prediction_bars",
        "prediction_bars",
    ]
    for key in scalar_keys:
        lhs = py_result[key]
        rhs = jl_result[key]
        if isinstance(lhs, float):
            if not np.isclose(lhs, rhs, atol=1e-9, rtol=0.0, equal_nan=True):
                raise AssertionError(f"{name} scalar mismatch for {key}: {lhs} vs {rhs}")
        elif lhs != rhs:
            raise AssertionError(f"{name} scalar mismatch for {key}: {lhs} vs {rhs}")

    _assert_close(
        f"{name}.equity_curve",
        np.asarray(py_result["equity_curve"], dtype=float),
        np.asarray(jl_result["equity_curve"], dtype=float),
    )
    _assert_equal(
        f"{name}.time_curve",
        np.asarray(py_result["time_curve"], dtype="datetime64[ns]"),
        np.asarray(jl_result["time_curve"], dtype="datetime64[ns]"),
    )

    if len(py_result["trades"]) != len(jl_result["trades"]):
        raise AssertionError(f"{name} trade count mismatch")

    trade_keys = [
        "type",
        "entry_time",
        "exit_time",
        "exit_reason",
        "tp1_hit",
        "tp2_hit",
    ]
    trade_float_keys = [
        "entry_price",
        "exit_price",
        "initial_risk",
        "confidence",
        "pnl",
        "stop_atr",
        "tp1_atr",
        "tp2_atr",
    ]
    for idx, (py_trade, jl_trade) in enumerate(
        zip(py_result["trades"], jl_result["trades"])
    ):
        for key in trade_keys:
            if py_trade[key] != jl_trade[key]:
                raise AssertionError(f"{name} trade {idx} mismatch for {key}")
        for key in trade_float_keys:
            if not np.isclose(
                float(py_trade[key]),
                float(jl_trade[key]),
                atol=1e-9,
                rtol=0.0,
                equal_nan=True,
            ):
                raise AssertionError(f"{name} trade {idx} float mismatch for {key}")

    bars_msg = f" across {bars} bars" if bars is not None else ""
    print(
        f"[ok] {name}: matched {len(py_result['trades'])} trades and full equity curve{bars_msg}"
    )


def _benchmark_backtest_runtime(
    name: str, run_kwargs: dict, repeats: int = 5
) -> None:
    run_backtest(use_julia_bar_state=False, **run_kwargs)
    run_backtest(use_julia_bar_state=True, **run_kwargs)

    py_times: list[float] = []
    jl_times: list[float] = []
    for idx in range(repeats):
        order = [False, True] if idx % 2 == 0 else [True, False]
        for use_julia in order:
            start = time.perf_counter()
            run_backtest(use_julia_bar_state=use_julia, **run_kwargs)
            elapsed = time.perf_counter() - start
            if use_julia:
                jl_times.append(elapsed)
            else:
                py_times.append(elapsed)

    py_median = float(np.median(py_times))
    jl_median = float(np.median(jl_times))
    speedup = py_median / jl_median if jl_median > 0 else np.inf
    print(
        f"[ok] {name}: python median {py_median:.6f}s vs julia median {jl_median:.6f}s ({speedup:.2f}x)"
    )


def _check_hurst_case(name: str, closes: np.ndarray, window_h: int) -> None:
    py_result = _compute_hurst_array_python(closes, window_h=window_h)
    jl_result = compute_hurst_fast(closes, window_h=window_h)
    _assert_close(f"{name}.hurst", py_result, jl_result)
    print(f"[ok] {name}: Hurst parity on {len(closes)} closes")


def _check_bar_state_case(
    name: str,
    closes: np.ndarray,
    prob: np.ndarray,
    atr: np.ndarray,
    zscore: np.ndarray,
    next_hour: np.ndarray,
    **kwargs,
) -> None:
    py_state = _precompute_backtest_bar_state_python(
        closes, prob, atr, zscore, next_hour, **kwargs
    )
    jl_state = compute_backtest_bar_state_fast(
        closes, prob, atr, zscore, next_hour, **kwargs
    )
    _assert_close(f"{name}.hurst", py_state["hurst"], jl_state["hurst"])
    _assert_close(f"{name}.confidence", py_state["confidence"], jl_state["confidence"])
    _assert_equal(
        f"{name}.direction_long", py_state["direction_long"], jl_state["direction_long"]
    )
    _assert_equal(f"{name}.skip_code", py_state["skip_code"], jl_state["skip_code"])
    print(f"[ok] {name}: bar-state parity on {len(closes)} bars")


def _check_end_to_end_backtest() -> None:
    rng = np.random.default_rng(123)
    n = 260
    times = pd.date_range("2024-01-01 09:00:00", periods=n, freq="h")

    close = 100.0 + np.cumsum(rng.normal(0.0, 0.8, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.6, n)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.6, n)
    atr14 = rng.uniform(0.6, 1.8, n)
    basis_z = rng.normal(0.0, 0.8, n)
    prob = np.clip(0.5 + rng.normal(0.0, 0.12, n), 0.01, 0.99)

    prob[15] = np.nan
    prob[40] = 0.53
    atr14[65] = 0.0
    basis_z[90] = 3.4

    df = pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.uniform(1000.0, 5000.0, n),
            "atr14": atr14,
            "basis_z_score": basis_z,
        }
    )

    py_result = run_backtest(
        df,
        prob,
        np.full(n, np.nan),
        [],
        trade_plan_models={},
        use_julia_bar_state=False,
    )
    jl_result = run_backtest(
        df,
        prob,
        np.full(n, np.nan),
        [],
        trade_plan_models={},
        use_julia_bar_state=True,
    )
    _assert_backtest_match("end_to_end_backtest", py_result, jl_result, bars=n)


def _find_real_intraday_artifact_cases(project_dir: Path) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for symbol_dir in _iter_symbol_dirs(project_dir):
        symbol = symbol_dir.name.upper()
        file_prefix = symbol_dir.name.lower().replace(" ", "_")
        feat_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1H", "features")
        oos_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1H", "oos_proba")
        oos_1d_path = resolve_artifact_path(
            str(symbol_dir), file_prefix, "1D", "oos_proba"
        )
        if all(os.path.exists(path) for path in (feat_path, oos_path, oos_1d_path)):
            cases.append((symbol, file_prefix))
    return cases


def _check_real_artifact_backtest() -> None:
    project_dir = Path(__file__).resolve().parent
    db_path = project_dir / "data_vault" / "ohlcv.db"
    if not db_path.exists():
        print("[skip] real_backtest: data_vault/ohlcv.db not present")
        return

    real_cases = _find_real_intraday_artifact_cases(project_dir)
    if not real_cases:
        print("[skip] real_backtest: no complete 1H/1D artifact case found")
        return

    for symbol, file_prefix in real_cases:
        symbol_dir = project_dir / symbol
        migrate_legacy_artifacts(str(symbol_dir), file_prefix, "1H")
        migrate_legacy_artifacts(str(symbol_dir), file_prefix, "1D", logger=None)

        bridge = InferenceBridge(db_path=str(db_path))
        tf_maps = {
            "FUT": bridge.fetch_holographic_stack(symbol, "FUT"),
            "SPOT": bridge.fetch_holographic_stack(symbol, "SPOT"),
        }
        df_1h = tf_maps["FUT"].get("1H")
        df_1d = tf_maps["FUT"].get("1D")
        df_1w = tf_maps["FUT"].get("1W")
        df_1m = tf_maps["FUT"].get("1M")
        if any(df is None for df in (df_1h, df_1d, df_1w, df_1m)):
            print(f"[skip] real_backtest[{symbol}]: missing 1H/1D/1W/1M FUT data")
            continue

        feat_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1H", "features")
        trade_plan_path = resolve_artifact_path(
            str(symbol_dir), file_prefix, "1H", "trade_plan_models"
        )
        oos_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1H", "oos_proba")
        oos_1d_path = resolve_artifact_path(
            str(symbol_dir), file_prefix, "1D", "oos_proba"
        )

        with open(feat_path, "r") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        trade_plan_models = (
            joblib.load(trade_plan_path) if os.path.exists(trade_plan_path) else {}
        )

        df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
            df_1h=df_1h,
            df_1d=df_1d,
            df_1w=df_1w,
            df_1m=df_1m,
            spot_1h=tf_maps["SPOT"].get("1H"),
            spot_1d=tf_maps["SPOT"].get("1D"),
            spot_1w=tf_maps["SPOT"].get("1W"),
            symbol=symbol,
            logger=lambda *_args, **_kwargs: None,
        )
        df_1h = fib_structural_basis(
            df_1h,
            htf_frames={"1D": df_1d, "1W": df_1w, "1M": df_1m},
            pairs=[("1D", "a"), ("1W", "b"), ("1M", "c")],
        )
        df_1h_labelled = _compute_atr14(df_1h.copy())
        df_full = holographic_feature_engine(
            df_1h_labelled,
            df_1d=df_1d,
            df_1w=df_1w,
            df_1m=df_1m,
        )
        df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)

        all_needed_cols = list(
            set(
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
        available = [col for col in all_needed_cols if col in df_full.columns]
        df_backtest = df_full[available].copy()
        for col in feature_cols:
            if col in df_backtest.columns:
                df_backtest[col] = (
                    df_backtest[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                )
        df_backtest = df_backtest.reset_index(drop=True)

        oos_proba_map = joblib.load(oos_path)
        prob_array = np.array(
            [
                float(oos_proba_map[pd.Timestamp(t)])
                if pd.Timestamp(t) in oos_proba_map
                else np.nan
                for t in df_backtest["time"]
            ]
        )
        oos_proba_map_1d = joblib.load(oos_1d_path)
        date_to_prob_1d = {
            pd.Timestamp(ts).date(): float(prob) for ts, prob in oos_proba_map_1d.items()
        }
        prob_array_1d = np.array(
            [
                date_to_prob_1d.get(pd.Timestamp(t).date(), np.nan)
                for t in df_backtest["time"]
            ]
        )

        py_result = run_backtest(
            df_backtest,
            prob_array,
            prob_array_1d,
            feature_cols,
            trade_plan_models=trade_plan_models,
            use_julia_bar_state=False,
        )
        jl_result = run_backtest(
            df_backtest,
            prob_array,
            prob_array_1d,
            feature_cols,
            trade_plan_models=trade_plan_models,
            use_julia_bar_state=True,
        )
        _assert_backtest_match(
            f"real_backtest[{symbol}]",
            py_result,
            jl_result,
            bars=len(df_backtest),
        )
        _benchmark_backtest_runtime(
            f"real_backtest[{symbol}].runtime",
            {
                "df": df_backtest,
                "prob_array": prob_array,
                "prob_array_1d": prob_array_1d,
                "feature_cols": feature_cols,
                "trade_plan_models": trade_plan_models,
                "initial_capital": 10000.0,
                "risk_pct": 0.02,
            },
        )


def _find_real_daily_artifact_cases(project_dir: Path) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for symbol_dir in _iter_symbol_dirs(project_dir):
        symbol = symbol_dir.name.upper()
        file_prefix = symbol_dir.name.lower().replace(" ", "_")
        feat_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1D", "features")
        oos_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1D", "oos_proba")
        if all(os.path.exists(path) for path in (feat_path, oos_path)):
            cases.append((symbol, file_prefix))
    return cases


def _check_real_daily_backtest() -> None:
    from daily_ml_engine import add_daily_confluence, inject_macro_regime

    project_dir = Path(__file__).resolve().parent
    db_path = project_dir / "data_vault" / "ohlcv.db"
    if not db_path.exists():
        print("[skip] real_daily_backtest: data_vault/ohlcv.db not present")
        return

    real_cases = _find_real_daily_artifact_cases(project_dir)
    if not real_cases:
        print("[skip] real_daily_backtest: no complete 1D artifact case found")
        return

    for symbol, file_prefix in real_cases:
        symbol_dir = project_dir / symbol
        migrate_legacy_artifacts(str(symbol_dir), file_prefix, "1D", logger=None)

        bridge = InferenceBridge(db_path=str(db_path))
        tf_maps = {
            "FUT": bridge.fetch_holographic_stack(symbol, "FUT"),
            "SPOT": bridge.fetch_holographic_stack(symbol, "SPOT"),
        }

        df_1d = _pick_primary_1d(tf_maps)
        if df_1d is None or df_1d.empty:
            print(f"[skip] real_daily_backtest[{symbol}]: no 1D data")
            continue

        df_1w = _get_tf(tf_maps, "1W")
        df_1m = _get_tf(tf_maps, "1M")
        df_3m = _get_tf(tf_maps, "3M")
        df_6m = _get_tf(tf_maps, "6M")
        df_12m = _get_tf(tf_maps, "12M")
        if any(df is None or df.empty for df in (df_1w, df_1m, df_3m)):
            print(f"[skip] real_daily_backtest[{symbol}]: missing 1W/1M/3M data")
            continue

        feat_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1D", "features")
        trade_plan_path = resolve_artifact_path(
            str(symbol_dir), file_prefix, "1D", "trade_plan_models"
        )
        oos_path = resolve_artifact_path(str(symbol_dir), file_prefix, "1D", "oos_proba")

        with open(feat_path, "r") as handle:
            feature_cols = [line.strip() for line in handle if line.strip()]
        trade_plan_models = (
            joblib.load(trade_plan_path) if os.path.exists(trade_plan_path) else {}
        )

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
        df_full = inject_macro_regime(df_full, df_6m, "6m")
        df_full = inject_macro_regime(df_full, df_12m, "12m")
        df_full = add_daily_confluence(df_full)

        all_needed_cols = list(
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
        available = [col for col in all_needed_cols if col in df_full.columns]
        df_backtest = df_full[available].copy()
        for col in feature_cols:
            if col not in df_backtest.columns:
                df_backtest[col] = 0.0
            df_backtest[col] = (
                df_backtest[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            )
        df_backtest = df_backtest.reset_index(drop=True)

        oos_proba_map = joblib.load(oos_path)
        prob_array = np.array(
            [
                float(oos_proba_map[pd.Timestamp(ts)])
                if pd.Timestamp(ts) in oos_proba_map
                else np.nan
                for ts in df_backtest["time"]
            ]
        )
        prob_array_1d = np.full(len(df_backtest), np.nan)

        py_result = run_backtest(
            df_backtest,
            prob_array,
            prob_array_1d,
            feature_cols,
            trade_plan_models=trade_plan_models,
            initial_capital=10000.0,
            risk_pct=0.02,
            conf_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
            max_hold_bars=BARRIER_HORIZON_BARS_DAILY,
            eod_gate_hour=EOD_GATE_HOUR_DAILY,
            use_julia_bar_state=False,
        )
        jl_result = run_backtest(
            df_backtest,
            prob_array,
            prob_array_1d,
            feature_cols,
            trade_plan_models=trade_plan_models,
            initial_capital=10000.0,
            risk_pct=0.02,
            conf_threshold=LIVE_CONFIDENCE_THRESHOLD_DAILY,
            max_hold_bars=BARRIER_HORIZON_BARS_DAILY,
            eod_gate_hour=EOD_GATE_HOUR_DAILY,
            use_julia_bar_state=True,
        )
        _assert_backtest_match(
            f"real_daily_backtest[{symbol}]",
            py_result,
            jl_result,
            bars=len(df_backtest),
        )


def main() -> None:
    rng = np.random.default_rng(42)

    hurst_cases = [
        ("short_series", np.array([100.0, 100.5, 100.25, 100.75], dtype=float), 100),
        ("trend_series", np.linspace(100.0, 150.0, 240, dtype=float), 100),
        (
            "noisy_walk",
            100.0 + np.cumsum(rng.normal(0.0, 1.25, 512)).astype(float),
            100,
        ),
        (
            "series_with_nan",
            np.where(
                np.arange(320) == 175,
                np.nan,
                100.0 + np.cumsum(rng.normal(0.0, 0.8, 320)),
            ),
            100,
        ),
    ]
    for name, closes, window_h in hurst_cases:
        _check_hurst_case(name, closes.astype(float), window_h)

    n = 320
    closes = 100.0 + np.cumsum(rng.normal(0.0, 0.9, n))
    prob = np.clip(0.5 + rng.normal(0.0, 0.1, n), 0.01, 0.99)
    atr = rng.uniform(0.5, 1.5, n)
    zscore = rng.normal(0.0, 0.7, n)
    next_hour = np.tile(np.array([10, 11, 12, 13, 14, 15], dtype=np.int64), n // 6 + 1)[
        :n
    ]

    prob[12] = np.nan
    prob[33] = 0.54
    atr[48] = 0.0
    zscore[77] = 3.1

    _check_bar_state_case(
        "gate_state_default",
        closes.astype(float),
        prob.astype(float),
        atr.astype(float),
        zscore.astype(float),
        next_hour.astype(np.int64),
        window_h=100,
        default_hurst=0.5,
        conf_threshold=0.56,
        shock_z_abs=2.5,
        min_hurst=0.45,
        eod_gate_hour=14,
    )
    _check_bar_state_case(
        "gate_state_strict_hurst",
        closes.astype(float),
        prob.astype(float),
        atr.astype(float),
        zscore.astype(float),
        next_hour.astype(np.int64),
        window_h=100,
        default_hurst=0.5,
        conf_threshold=0.56,
        shock_z_abs=2.5,
        min_hurst=0.70,
        eod_gate_hour=14,
    )

    _check_end_to_end_backtest()
    _check_real_artifact_backtest()
    _check_real_daily_backtest()
    print("[ok] Julia backtest migration parity checks passed")


if __name__ == "__main__":
    main()
