import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import universal_ml_engine as ume
from universal_ml_engine import (
    merge_higher_tf,
    add_target,
    train_trade_plan_models,
    walk_forward,
    predict_next_bar,
    _compute_atr14,
    fib_structural_basis,
    migrate_legacy_artifacts,
    resolve_artifact_path,
    MODEL_N_JOBS,
    LIVE_CONFIDENCE_THRESHOLD,
    BARRIER_HORIZON_BARS,
)
from julia_bridge import holographic_feature_engine_fast as holographic_feature_engine
from holographic_engine import feature_selection_pipeline
from backtest_engine import run_backtest, calculate_metrics

try:
    from universal_ml_engine import NON_FEATURE_COLS_SET
except ImportError:
    NON_FEATURE_COLS_SET = None


if NON_FEATURE_COLS_SET is not None:
    NON_FEATURE_COLS = set(NON_FEATURE_COLS_SET)
else:
    NON_FEATURE_COLS = {
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr14",
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
        "d_open",
        "d_high",
        "d_low",
        "d_close",
        "d_volume",
        "w_open",
        "w_high",
        "w_low",
        "w_close",
        "w_volume",
        "m_open",
        "m_high",
        "m_low",
        "m_close",
        "m_volume",
    }


STRATEGY_ZOO = {
    "strategy_tight": {"atr_mult": 1.00, "min_edge_r": 0.20},
    "strategy_standard": {"atr_mult": 1.25, "min_edge_r": 0.25},
    "strategy_wide": {"atr_mult": 1.75, "min_edge_r": 0.30},
}

EXECUTION_CONF_GRID = [0.60, 0.65, 0.70, 0.75, 0.80]
BACKTEST_INITIAL_CAPITAL = 10000.0
BACKTEST_RISK_PCT = 0.02
MIN_META_TRADES = 25


def _build_prob_array(df_strategy: pd.DataFrame, oos_proba_map: dict) -> np.ndarray:
    if (
        not isinstance(df_strategy, pd.DataFrame)
        or df_strategy.empty
        or "time" not in df_strategy.columns
    ):
        return np.full(0, np.nan, dtype=float)
    return np.array(
        [
            float(oos_proba_map.get(pd.Timestamp(t), np.nan))
            for t in df_strategy["time"]
        ],
        dtype=float,
    )


def _is_valid_meta_candidate(candidate: dict) -> bool:
    return (
        isinstance(candidate, dict)
        and int(candidate.get("total_trades", 0)) >= MIN_META_TRADES
        and np.isfinite(candidate.get("final_equity", np.nan))
        and np.isfinite(candidate.get("total_return_pct", np.nan))
    )


def _candidate_rank(candidate: dict) -> tuple[float, float, float]:
    final_equity = float(candidate.get("final_equity", -np.inf))
    sharpe = float(candidate.get("sharpe", -np.inf))
    max_drawdown = float(candidate.get("max_drawdown", np.inf))
    return final_equity, sharpe, -max_drawdown


def run_strategy_zoo(
    df_model_ready: pd.DataFrame,
    *,
    walk_forward_fn=walk_forward,
    symbol: str = "",
) -> dict:
    print(
        f"  [Meta] Strategy zoo starting for {symbol or 'UNKNOWN'} with MODEL_N_JOBS={MODEL_N_JOBS}"
    )
    zoo_results: dict[str, dict] = {}

    original_atr_mult = ume.BARRIER_ATR_MULT

    for key, strategy in STRATEGY_ZOO.items():
        atr_mult = float(strategy["atr_mult"])
        min_edge_r = float(strategy["min_edge_r"])
        print(
            f"  [Meta] [Zoo] Running {key} - ATR_MULT={atr_mult:.2f} MIN_EDGE_R={min_edge_r:.2f} ..."
        )

        df_copy = df_model_ready.copy(deep=True)

        try:
            ume.BARRIER_ATR_MULT = atr_mult  # Global set for documentation; Julia path receives atr_mult explicitly.

            df_copy = add_target(
                df_copy,
                atr_mult=atr_mult,
                horizon=BARRIER_HORIZON_BARS,
                atr_col="atr14",
                drop_unresolved=True,
            )
        finally:
            ume.BARRIER_ATR_MULT = original_atr_mult

        df_copy = df_copy.dropna(subset=["target"]).copy()
        if df_copy.empty:
            print(f"  [Meta] [Zoo] {key} produced no resolved labels.")
            zoo_results[key] = {"oos_proba_map": {}, "feature_cols": [], "df": df_copy}
            continue

        df_copy["target"] = df_copy["target"].astype(int)
        all_holo_cols = [c for c in df_copy.columns if c not in NON_FEATURE_COLS]

        if not all_holo_cols:
            print(f"  [Meta] [Zoo] {key} has no usable feature columns.")
            zoo_results[key] = {"oos_proba_map": {}, "feature_cols": [], "df": df_copy}
            continue

        for col in all_holo_cols:
            df_copy[col] = df_copy[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        try:
            feature_cols, _ = feature_selection_pipeline(
                df_copy.copy(),
                all_holo_cols,
                walk_forward_fn=walk_forward_fn,
                target_col="target",
                min_train_bars=2500,
                test_size_ratio=0.15,
                n_splits=10,
            )
        except Exception as exc:
            print(f"  [Meta] [Zoo] Feature selection failed for {key}: {exc}")
            zoo_results[key] = {"oos_proba_map": {}, "feature_cols": [], "df": df_copy}
            continue

        if not feature_cols:
            print(f"  [Meta] [Zoo] {key} ended with zero selected features.")
            zoo_results[key] = {"oos_proba_map": {}, "feature_cols": [], "df": df_copy}
            continue

        try:
            wf_result = walk_forward_fn(
                df_copy.copy(),
                feature_cols,
                n_splits=10,
                min_train_bars=2500,
                test_size_ratio=0.15,
                purge_gap=24,
            )
        except Exception as exc:
            print(f"  [Meta] [Zoo] Walk-forward failed for {key}: {exc}")
            wf_result = {}

        if not wf_result:
            wf_result = {
                "oos_proba_map": {},
                "feature_cols": feature_cols,
                "df": df_copy,
            }
        else:
            strategy_df = wf_result.get("df")
            trade_plan_models = {}
            if isinstance(strategy_df, pd.DataFrame) and not strategy_df.empty:
                tp_holdout = max(int(len(strategy_df) * 0.15), 1)
                tp_train_end = max(len(strategy_df) - tp_holdout, 1)
                try:
                    trade_plan_models = train_trade_plan_models(
                        strategy_df.iloc[:tp_train_end].copy(),
                        feature_cols,
                    )
                except Exception as exc:
                    print(
                        f"  [Meta] [Zoo] Trade-plan model training failed for {key}: {exc}"
                    )
            wf_result["trade_plan_models"] = trade_plan_models

        zoo_results[key] = wf_result

    return zoo_results


def meta_select_strategy(
    zoo_results: dict,
    df_full: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[str, dict]:
    del df_full, risk_free_rate

    metrics: dict[str, dict] = {}
    candidates: list[tuple[str, dict]] = []

    for key in STRATEGY_ZOO:
        wf_result = zoo_results.get(key, {})
        strategy_df = wf_result.get("df")
        feature_cols = wf_result.get("feature_cols", [])
        oos_proba_map = wf_result.get("oos_proba_map") or {}
        trade_plan_models = wf_result.get("trade_plan_models", {})

        if (
            not isinstance(strategy_df, pd.DataFrame)
            or strategy_df.empty
            or not feature_cols
            or not oos_proba_map
        ):
            metrics[key] = {
                "total_return_pct": np.nan,
                "final_equity": np.nan,
                "sharpe": np.nan,
                "sortino": np.nan,
                "win_rate": np.nan,
                "max_drawdown": np.nan,
                "profit_factor": np.nan,
                "expectancy": np.nan,
                "total_trades": 0,
                "oos_bars": 0,
                "conf_threshold": np.nan,
            }
            continue

        prob_array = _build_prob_array(strategy_df, oos_proba_map)
        oos_bars = int(np.isfinite(prob_array).sum())
        if oos_bars == 0:
            metrics[key] = {
                "total_return_pct": np.nan,
                "final_equity": np.nan,
                "sharpe": np.nan,
                "sortino": np.nan,
                "win_rate": np.nan,
                "max_drawdown": np.nan,
                "profit_factor": np.nan,
                "expectancy": np.nan,
                "total_trades": 0,
                "oos_bars": 0,
                "conf_threshold": np.nan,
            }
            continue

        best_exec = None
        nan_1d = np.full(len(strategy_df), np.nan)
        for conf_threshold in EXECUTION_CONF_GRID:
            bt_results = run_backtest(
                strategy_df,
                prob_array,
                nan_1d,
                feature_cols,
                trade_plan_models=trade_plan_models,
                initial_capital=BACKTEST_INITIAL_CAPITAL,
                risk_pct=BACKTEST_RISK_PCT,
                conf_threshold=conf_threshold,
            )
            bt_metrics = calculate_metrics(
                bt_results["trades"],
                bt_results.get("equity_curve"),
                bt_results.get("time_curve"),
            )
            final_equity = float(bt_results.get("final_equity", np.nan))
            total_return_pct = (
                ((final_equity / BACKTEST_INITIAL_CAPITAL) - 1.0) * 100.0
                if np.isfinite(final_equity)
                else np.nan
            )
            max_drawdown = float(bt_results.get("max_drawdown", np.nan))
            sharpe = float(bt_metrics.get("sharpe", np.nan))
            sortino = float(bt_metrics.get("sortino", np.nan))
            profit_factor = float(bt_metrics.get("profit_factor", np.nan))
            expectancy = float(bt_metrics.get("expectancy", np.nan))
            win_rate = float(bt_metrics.get("win_rate", np.nan))
            total_trades = int(bt_metrics.get("total_trades", 0))
            calmar = (
                (total_return_pct / 100.0) / (max_drawdown + 1e-9)
                if np.isfinite(total_return_pct) and np.isfinite(max_drawdown)
                else np.nan
            )
            candidate = {
                "total_return_pct": total_return_pct,
                "final_equity": final_equity,
                "sharpe": sharpe,
                "sortino": sortino if np.isfinite(sortino) else np.nan,
                "calmar": calmar if np.isfinite(calmar) else np.nan,
                "win_rate": win_rate if np.isfinite(win_rate) else np.nan,
                "max_drawdown": max_drawdown if np.isfinite(max_drawdown) else np.nan,
                "profit_factor": profit_factor
                if np.isfinite(profit_factor)
                else np.nan,
                "expectancy": expectancy if np.isfinite(expectancy) else np.nan,
                "total_trades": total_trades,
                "oos_bars": oos_bars,
                "conf_threshold": conf_threshold,
            }
            if best_exec is None or _candidate_rank(candidate) > _candidate_rank(
                best_exec
            ):
                best_exec = candidate

        metrics[key] = best_exec

        if _is_valid_meta_candidate(best_exec):
            candidates.append((key, best_exec))

    if not candidates:
        winner_key = "strategy_standard"
        selection_reason = "Fallback: insufficient valid OOS trades."
    else:
        candidates.sort(
            key=lambda item: (
                item[1]["final_equity"],
                item[1]["sharpe"] if np.isfinite(item[1]["sharpe"]) else -np.inf,
                -(
                    item[1]["max_drawdown"]
                    if np.isfinite(item[1]["max_drawdown"])
                    else np.inf
                ),
            ),
            reverse=True,
        )
        winner_key, winner_metrics = candidates[0]
        if len(candidates) > 1:
            runner_metrics = candidates[1][1]
            equity_gap_pct = abs(
                winner_metrics["total_return_pct"] - runner_metrics["total_return_pct"]
            )
            if equity_gap_pct < 0.5:
                winner_key, winner_metrics = max(
                    [candidates[0], candidates[1]],
                    key=lambda item: (
                        item[1]["sharpe"]
                        if np.isfinite(item[1]["sharpe"])
                        else -np.inf,
                        -(
                            item[1]["max_drawdown"]
                            if np.isfinite(item[1]["max_drawdown"])
                            else np.inf
                        ),
                    ),
                )
                selection_reason = (
                    f"Tiebreak by Sharpe after near-equal OOS return; "
                    f"{winner_key} won with return {winner_metrics['total_return_pct']:.2f}%."
                )
            else:
                selection_reason = (
                    f"Highest OOS backtest return ({winner_metrics['total_return_pct']:.2f}%) "
                    f"with {winner_metrics['total_trades']} trades."
                )
        else:
            selection_reason = (
                f"Highest OOS backtest return ({winner_metrics['total_return_pct']:.2f}%) "
                f"with {winner_metrics['total_trades']} trades."
            )

    metrics["winner"] = winner_key
    metrics["selection_reason"] = selection_reason
    return winner_key, metrics


def verdict_output(
    winner_key: str,
    metrics: dict,
    zoo_results: dict,
    symbol: str,
    timestamp: str,
) -> dict:
    winning_wf = zoo_results.get(winner_key, {})
    model = winning_wf.get("final_model")
    feature_cols = winning_wf.get("feature_cols", [])
    winning_df = winning_wf.get("df")
    last_row = (
        winning_df.iloc[-1].copy()
        if isinstance(winning_df, pd.DataFrame) and not winning_df.empty
        else pd.Series(dtype=float)
    )

    if model is not None and feature_cols and not last_row.empty:
        pred = predict_next_bar(
            model, feature_cols, last_row, LIVE_CONFIDENCE_THRESHOLD
        )
    else:
        pred = {
            "direction": "ERROR",
            "prob_up": np.nan,
            "prob_down": np.nan,
            "confidence": np.nan,
            "signal_strength": "ERROR",
            "message": "Winning strategy did not produce a usable final model.",
        }

    def _fmt_float(value: float) -> str:
        if value is None or not np.isfinite(value):
            return " n/a  "
        return f"{value:6.2f}"

    def _fmt_pct(value: float) -> str:
        if value is None or not np.isfinite(value):
            return "  n/a   "
        return f"{value * 100:6.1f}%"

    def _fmt_int(value: int) -> str:
        if value is None:
            return "   n/a   "
        try:
            return f"{int(value):8d}"
        except (TypeError, ValueError):
            return "   n/a   "

    def _fmt_drawdown(value: float) -> str:
        if value is None or not np.isfinite(value):
            return "  n/a   "
        return f"{value * 100:6.1f}%"

    confidence = pred.get("confidence")
    confidence_text = (
        f"{float(confidence) * 100:.1f}%"
        if confidence is not None and np.isfinite(confidence)
        else "n/a"
    )

    print("======================================================================")
    print(f"  TOON META-STRATEGY VERDICT — {symbol.upper()}")
    print(f"  Evaluated : {timestamp}")
    print("======================================================================")
    print(f"  WINNER      : {winner_key}")
    print(f"  Reason      : {metrics.get('selection_reason', 'n/a')}")
    print("")
    print("  Strategy Scorecard:")
    print("  ┌─────────────────────┬────────┬────────┬──────────┬──────────┬────────┐")
    print("  │ Strategy            │ Return │ Sharpe │ Max DD   │ Trades   │ Conf   │")
    print("  ├─────────────────────┼────────┼────────┼──────────┼──────────┼────────┤")
    for strategy_key in STRATEGY_ZOO:
        row = metrics.get(strategy_key, {})
        print(
            f"  │ {strategy_key:<19} │ {_fmt_pct(row.get('total_return_pct', np.nan) / 100.0)} │ "
            f"{_fmt_float(row.get('sharpe', np.nan))} │ {_fmt_drawdown(row.get('max_drawdown', np.nan))} │ "
            f"{_fmt_int(row.get('total_trades', 0))} │ {_fmt_float(row.get('conf_threshold', np.nan))} │"
        )
    print("  └─────────────────────┴────────┴────────┴──────────┴──────────┴────────┘")
    print("")
    print(f"  LIVE SIGNAL  (Winning Strategy: {winner_key})")
    print(f"  Direction  : {pred.get('direction', 'ERROR')}")
    print(f"  Confidence : {confidence_text}")
    print(f"  Signal     : {pred.get('signal_strength', 'ERROR')}")
    print(
        f"  Exec Conf  : {_fmt_float(metrics.get(winner_key, {}).get('conf_threshold', np.nan)).strip()}"
    )
    print("======================================================================")

    return {
        "winner": winner_key,
        "metrics": metrics,
        "signal": pred,
        "timestamp": timestamp,
        "symbol": symbol,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TOON meta-strategy selector")
    parser.add_argument(
        "--outdir", default="/home/km/Universal-ML/", help="Project output directory"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Target Base Symbol (e.g., BANKNIFTY, BTC)",
    )
    args = parser.parse_args()

    outdir = os.path.abspath(args.outdir)
    SYMBOL = args.symbol.upper()
    SYMBOL_DIR = os.path.join(outdir, SYMBOL)
    os.makedirs(SYMBOL_DIR, exist_ok=True)

    import sys

    sys.path.insert(0, os.path.join(outdir, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("  [Meta] FATAL: Cannot locate inference_bridge.py.")
        raise SystemExit(1)

    bridge = InferenceBridge(db_path=os.path.join(outdir, "data_vault", "ohlcv.db"))
    tf_raw = {
        "FUT": bridge.fetch_holographic_stack(SYMBOL, "FUT"),
        "SPOT": bridge.fetch_holographic_stack(SYMBOL, "SPOT"),
    }

    df_1h = tf_raw["FUT"].get("1H")
    df_1d = tf_raw["FUT"].get("1D")
    df_1w = tf_raw["FUT"].get("1W")
    df_1m = tf_raw["FUT"].get("1M")

    missing = [
        tf
        for tf, df in [("1H", df_1h), ("1D", df_1d), ("1W", df_1w), ("1M", df_1m)]
        if df is None or df.empty
    ]
    if missing:
        print(f"  [Meta] Missing required FUT timeframes: {', '.join(missing)}")
        raise SystemExit(1)

    symbol = SYMBOL
    file_prefix = symbol.lower().replace(" ", "_")
    migrate_legacy_artifacts(SYMBOL_DIR, file_prefix, "1H", logger=None)
    migrate_legacy_artifacts(SYMBOL_DIR, file_prefix, "1D", logger=None)
    baseline_model_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "model")
    baseline_oos_path = resolve_artifact_path(
        SYMBOL_DIR, file_prefix, "1H", "oos_proba"
    )
    if os.path.exists(baseline_model_path):
        print(f"  [Meta] Found baseline model: {baseline_model_path}")
    if os.path.exists(baseline_oos_path):
        print(f"  [Meta] Found baseline OOS probabilities: {baseline_oos_path}")

    print(f"  [Meta] Building feature frame for {symbol} ...")
    df_1h = df_1h.copy()
    df_1d = df_1d.copy()
    df_1w = df_1w.copy()
    df_1m = df_1m.copy()

    from universal_ml_engine import prepare_intraday_thermodynamics

    try:
        df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
            df_1h=df_1h,
            df_1d=df_1d,
            df_1w=df_1w,
            df_1m=df_1m,
            spot_1h=tf_raw["SPOT"].get("1H"),
            spot_1d=tf_raw["SPOT"].get("1D"),
            spot_1w=tf_raw["SPOT"].get("1W"),
            symbol=SYMBOL,
            logger=print,
        )
    except ValueError:
        raise SystemExit(1)

    df_1h = fib_structural_basis(
        df_1h,
        htf_frames={"1D": df_1d, "1W": df_1w, "1M": df_1m},
        pairs=[("1D", "a"), ("1W", "b"), ("1M", "c")],
    )

    df_1h_labelled = _compute_atr14(df_1h.copy())
    df_full = holographic_feature_engine(
        df_1h_labelled.copy(),
        df_1d=df_1d.copy(),
        df_1w=df_1w.copy(),
        df_1m=df_1m.copy(),
    )
    df_full = merge_higher_tf(df_full.copy(), df_1d.copy(), df_1w.copy(), df_1m.copy())
    if "atr14" not in df_full.columns:
        df_full = _compute_atr14(df_full.copy())

    zoo_results = run_strategy_zoo(df_full.copy(), symbol=SYMBOL)
    winner_key, metrics = meta_select_strategy(zoo_results, df_full.copy())

    from datetime import datetime

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict = verdict_output(winner_key, metrics, zoo_results, symbol, ts)

    report_path = os.path.join(SYMBOL_DIR, f"{symbol.lower()}_meta_report.png")
    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
    ax_top = fig.add_subplot(2, 1, 1)
    ax_bottom = fig.add_subplot(2, 1, 2)

    for ax in (ax_top, ax_bottom):
        ax.set_facecolor("#0d0d0d")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    winner_signal = verdict["signal"]
    accent = "#00ff88" if winner_signal.get("direction") == "UP" else "#ff6b6b"
    conf_text = (
        f"{float(winner_signal['confidence']) * 100:.1f}%"
        if np.isfinite(winner_signal.get("confidence", np.nan))
        else "n/a"
    )
    top_text = (
        f"==============================================================\n"
        f"  TOON META-STRATEGY REPORT : {symbol.upper()}\n"
        f"==============================================================\n"
        f"  Evaluated   : {ts}\n"
        f"  Winner      : {winner_key}\n"
        f"  Reason      : {metrics.get('selection_reason', 'n/a')}\n"
        f"--------------------------------------------------------------\n"
        f"  OOS Return  : {metrics.get(winner_key, {}).get('total_return_pct', np.nan):.2f}%\n"
        f"  Live Signal : {winner_signal.get('direction', 'ERROR')}\n"
        f"  Confidence  : {conf_text}\n"
        f"  Exec Conf   : {metrics.get(winner_key, {}).get('conf_threshold', np.nan):.2f}\n"
        f"  Strength    : {winner_signal.get('signal_strength', 'ERROR')}\n"
        f"=============================================================="
    )
    ax_top.axis("off")
    ax_top.text(
        0.5,
        0.5,
        top_text,
        color=accent,
        fontsize=16,
        fontfamily="monospace",
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="#1a1a1a", edgecolor=accent, pad=2.0, boxstyle="round"),
    )

    strategies = list(STRATEGY_ZOO.keys())
    return_vals = [
        metrics.get(key, {}).get("total_return_pct", np.nan) for key in strategies
    ]
    win_rates = [
        metrics.get(key, {}).get("win_rate", np.nan) * 100.0 for key in strategies
    ]
    bar_colors = [accent if key == winner_key else "#00d4ff" for key in strategies]
    x = np.arange(len(strategies))
    bars = ax_bottom.bar(x, return_vals, color=bar_colors, alpha=0.85, width=0.55)
    ax_bottom.set_title(
        "OOS Trading Return by Meta Strategy",
        color="white",
        fontsize=16,
        fontweight="bold",
    )
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(strategies, color="#e0e0e0")
    ax_bottom.set_ylabel("Return %", color="#e0e0e0")
    ax_bottom.grid(axis="y", alpha=0.18, color="#666666")

    for rect, key, return_val, win_rate in zip(
        bars, strategies, return_vals, win_rates
    ):
        dd = metrics.get(key, {}).get("max_drawdown", np.nan)
        total_trades = metrics.get(key, {}).get("total_trades", 0)
        conf_threshold = metrics.get(key, {}).get("conf_threshold", np.nan)
        label = (
            f"R {return_val:.2f}%\n"
            f"S {metrics.get(key, {}).get('sharpe', np.nan):.2f}\n"
            f"WR {win_rate:.1f}%\n"
            f"DD {dd * 100:.1f}%\n"
            f"T {int(total_trades)} | C {conf_threshold:.2f}"
        )
        y = rect.get_height() if np.isfinite(rect.get_height()) else 0.0
        va = "bottom" if y >= 0 else "top"
        y_offset = 0.5 if y >= 0 else -0.5
        ax_bottom.text(
            rect.get_x() + rect.get_width() / 2.0,
            y + y_offset,
            label,
            ha="center",
            va=va,
            color="#e0e0e0",
            fontsize=10,
            fontfamily="monospace",
        )

    ax_bottom.axhline(0.0, color="#888888", linewidth=1.0)
    fig.suptitle(
        f"{symbol.upper()} Meta Strategy Verdict",
        color="white",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        report_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight"
    )
    plt.close(fig)
    verdict["report_path"] = report_path
    print(f"  [Meta] Report saved → {report_path}")

    verdict_path = os.path.join(SYMBOL_DIR, f"{symbol.lower()}_meta_verdict.pkl")
    joblib.dump(verdict, verdict_path)
    print(f"  [Meta] Verdict saved → {verdict_path}")
