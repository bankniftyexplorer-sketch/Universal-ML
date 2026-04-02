import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress lightgbm warnings locally if needed
import warnings

warnings.filterwarnings("ignore")

from universal_ml_engine import (
    merge_higher_tf,
    _compute_atr14,
    BARRIER_HORIZON_BARS,
    LIVE_CONFIDENCE_THRESHOLD,
    EXEC_FEE_PCT,
    simulate_trade_path_from_arrays,
    predict_trade_plan,
    prepare_intraday_thermodynamics,
    resolve_artifact_path,
    migrate_legacy_artifacts,
)
from julia_bridge import holographic_feature_engine_fast as holographic_feature_engine

EOD_GATE_HOUR = 14  # IST. Set to 24 for 24/7 crypto.


def format_currency(val):
    return f"${val:,.2f}"


def run_backtest(
    df,
    prob_array,
    prob_array_1d,
    feature_cols,
    trade_plan_models=None,
    initial_capital=10000.0,
    risk_pct=0.02,
    conf_threshold=LIVE_CONFIDENCE_THRESHOLD,
    fixed_risk=True,
    slippage_bps=0.0003,
    max_hold_bars=BARRIER_HORIZON_BARS,
):
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    conflict_blocks = 0
    volatility_blocks = 0
    no_prediction_bars = 0
    trades = []
    equity_curve = []
    time_curve = []

    close_arr = df["close"].values
    open_arr = df["open"].values
    high_arr = df["high"].values
    low_arr = df["low"].values
    time_arr = df["time"].values
    atr_arr = df["atr14"].values

    # Secure Thermodynamic State Array (Fallback to 0.0 if not injected)
    if "basis_z_score" in df.columns:
        z_arr = df["basis_z_score"].values
    else:
        z_arr = np.zeros(len(df))

    shock_blocks = 0

    # [TOON vX.0 Alpha] Phase 4: Thermodynamic Rough Volatility (Hurst Exponent)
    hurst_arr = np.full(len(close_arr), 0.5)
    window_h = 100
    for j in range(window_h, len(close_arr)):
        slice_c = close_arr[j - window_h : j]
        diffs = np.diff(slice_c)
        if len(diffs) > 0:
            S = np.std(diffs)
            if S > 0:
                y = diffs - np.mean(diffs)
                Z = np.cumsum(y)
                R = np.max(Z) - np.min(Z)
                if R > 0:
                    hurst_arr[j] = np.clip(np.log(R / S) / np.log(window_h), 0.0, 1.0)

    i = 0
    while i < len(df) - 1:
        current_time = pd.to_datetime(time_arr[i])
        current_atr = atr_arr[i]
        equity_curve.append(equity)
        time_curve.append(current_time)

        proba_up = prob_array[i]
        proba_up_1d = float("nan")  # no separate 1D model in TOON v4.0

        if not np.isfinite(proba_up):
            no_prediction_bars += 1
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        # Continuous Regressor Confidence
        confidence = max(proba_up, 1.0 - proba_up)
        current_hurst = hurst_arr[i]

        # 0. EOD Session Block (Neutralizes negative EOD gap-against expectancy)
        # Gate on the ENTRY bar (i+1), not the signal bar (i), to close the
        # 13:00-signal → 14:00-execution temporal leak.
        next_time = pd.to_datetime(time_arr[i + 1])
        if next_time.hour >= EOD_GATE_HOUR:
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        if confidence < conf_threshold:
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        if not np.isfinite(current_atr) or current_atr <= 0:
            i += 1
            continue

        # 1. The Thermodynamic Shock Gate (Blocks Structural Liquidation/Euphoria Extremes)
        current_z = z_arr[i]
        if np.isfinite(current_z) and abs(current_z) > 2.5:
            shock_blocks += 1
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        # 2. The Thermodynamic Hurst Gate (Blocks Anti-persistent Noise)
        if not np.isfinite(current_hurst) or current_hurst < 0.45:
            volatility_blocks += 1
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        direction = "LONG" if proba_up > 0.5 else "SHORT"

        risk_dollar = (initial_capital if fixed_risk else equity) * risk_pct
        trade_plan = predict_trade_plan(
            trade_plan_models or {},
            feature_cols,
            df.iloc[i].copy(),
            "UP" if direction == "LONG" else "DOWN",
            float(current_atr),
        )

        # [TOON v4.2] Regime-Resistant Asymmetry Matrix
        # Capitalizes on the 56% ML Win Rate by harvesting risk at 1:1.
        stop_dist = float(current_atr * 2.0)  # Catastrophic stop (2 ATR)
        tp1_dist = float(
            current_atr * 2.0
        )  # Realize 50% at 2 ATR (1:1 R/R to lock in the edge)
        tp2_dist = float(current_atr * 4.0)  # Realize 25% at 4 ATR
        trail_dist = float(
            current_atr * 1.0
        )  # Aggressive trailing to prevent runner bleed

        if np.isfinite(trade_plan.get("stop_atr", np.nan)):
            stop_dist = float(current_atr * trade_plan["stop_atr"])
            tp1_dist = float(current_atr * trade_plan["tp1_atr"])
            tp2_dist = float(current_atr * trade_plan["tp2_atr"])
            trail_dist = float(current_atr * trade_plan["trail_r"])

        trade_path = simulate_trade_path_from_arrays(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            time_arr,
            i + 1,
            direction,
            stop_dist,
            tp1_dist=tp1_dist,
            tp2_dist=tp2_dist,
            trail_dist=trail_dist,
            horizon=max_hold_bars,
            fee_pct=EXEC_FEE_PCT,
            slippage_bps=slippage_bps,
        )

        if not np.isfinite(trade_path["total_r"]):
            i += 1
            continue

        pnl = trade_path["total_r"] * risk_dollar
        equity += pnl
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        trades.append(
            {
                "type": direction,
                "entry_price": trade_path["entry_price"],
                "exit_price": trade_path["exit_price"],
                "entry_time": time_arr[i + 1],
                "exit_time": trade_path["exit_time"],
                "exit_reason": trade_path["exit_reason"],
                "initial_risk": risk_dollar,
                "confidence": confidence,
                "pnl": pnl,
                "status": "CLOSED",
                "tp1_hit": trade_path["tp1_hit"],
                "tp2_hit": trade_path["tp2_hit"],
                "stop_atr": (stop_dist / (current_atr + 1e-9)),
                "tp1_atr": (tp1_dist / (current_atr + 1e-9)),
                "tp2_atr": (tp2_dist / (current_atr + 1e-9)),
            }
        )

        for k in range(i + 1, min(trade_path["exit_idx"], len(df) - 1) + 1):
            time_curve.append(pd.to_datetime(time_arr[k]))
            equity_curve.append(equity)
        i = max(i + 1, trade_path["exit_idx"] + 1)

    results = {
        "final_equity": equity,
        "max_drawdown": max_drawdown,
        "trades": trades,
        "equity_curve": equity_curve,
        "time_curve": time_curve,
        "conflict_blocks": conflict_blocks,
        "volatility_blocks": volatility_blocks,
        "shock_blocks": shock_blocks,
        "no_prediction_bars": no_prediction_bars,
        "prediction_bars": int(np.isfinite(prob_array).sum()),
    }
    return results


def calculate_metrics(trades, equity_curve=None, time_curve=None):
    if not trades:
        return {}

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]

    gross_profit = sum([t["pnl"] for t in wins])
    gross_loss = abs(sum([t["pnl"] for t in losses]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # E6: True Time-Series Risk-Adjusted Metrics
    if equity_curve is not None and time_curve is not None and len(equity_curve) > 1:
        eq_series = pd.Series(equity_curve, index=pd.to_datetime(time_curve))
        # Resample to daily frequency, forward-fill inactive days, calculate daily % return
        daily_returns = eq_series.resample("D").last().ffill().pct_change().dropna()

        sharpe = (
            (daily_returns.mean() / (daily_returns.std() + 1e-9)) * np.sqrt(252)
            if len(daily_returns) > 1
            else 0.0
        )
        downside_returns = daily_returns[daily_returns < 0]
        sortino = (
            (daily_returns.mean() / (downside_returns.std() + 1e-9)) * np.sqrt(252)
            if len(downside_returns) > 0
            else float("inf")
        )
    else:
        # Fallback to unannualized trade-based expectancy ratio if curves are missing
        r_multiples = [t["pnl"] / (t.get("initial_risk", 1) + 1e-9) for t in trades]
        sharpe = (
            np.mean(r_multiples) / (np.std(r_multiples) + 1e-9)
            if len(r_multiples) > 1
            else 0.0
        )
        downside_returns = [r for r in r_multiples if r < 0]
        sortino = (
            np.mean(r_multiples) / (np.std(downside_returns) + 1e-9)
            if downside_returns
            else float("inf")
        )

    # Calculate True Time Under Water (anchored to first executed trade,
    # ignoring the pre-trade ML warmup cold-start period)
    true_tuw_days = 0
    if trades and equity_curve is not None and time_curve is not None:
        first_trade_time = pd.to_datetime(trades[0]["entry_time"])
        post_warmup_mask = [pd.to_datetime(t) >= first_trade_time for t in time_curve]
        if any(post_warmup_mask):
            active_times = [t for i, t in enumerate(time_curve) if post_warmup_mask[i]]
            active_equity = [
                e for i, e in enumerate(equity_curve) if post_warmup_mask[i]
            ]
            peak = active_equity[0]
            peak_time = active_times[0]
            max_drought = pd.Timedelta(days=0)
            for t, e in zip(active_times, active_equity):
                if e > peak:
                    peak = e
                    peak_time = t
                else:
                    drought = pd.to_datetime(t) - pd.to_datetime(peak_time)
                    if drought > max_drought:
                        max_drought = drought
            true_tuw_days = max_drought.days

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for t in trades:
        if t["pnl"] <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    return {
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_consec_loss": max_consec_loss,
        "true_tuw_days": true_tuw_days,
    }


def generate_report(results, metrics, symbol, save_path):
    fig = plt.figure(figsize=(14, 10), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, height_ratios=[2, 1])

    dark_bg_color = "#0d0d0d"
    light_text_color = "#e0e0e0"
    axis_line_color = "#444444"
    accent_color_1 = "#00d4ff"
    accent_color_2 = "#00ff88"
    accent_color_3 = "#ff6b6b"

    # -- Eq Curve panel --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(dark_bg_color)
    ax1.tick_params(colors=light_text_color)
    for spine in ax1.spines.values():
        spine.set_color(axis_line_color)

    ax1.plot(results["time_curve"], results["equity_curve"], color=accent_color_1, lw=2)
    ax1.fill_between(
        results["time_curve"],
        results["equity_curve"],
        min(results["equity_curve"]) * 0.99,
        color=accent_color_1,
        alpha=0.1,
    )
    ax1.set_title(
        f"{symbol.upper()} Portfolio Backtest Equity Curve",
        color="white",
        fontsize=16,
        fontweight="bold",
    )
    ax1.set_ylabel("Equity ($)", color=light_text_color, fontsize=12)
    ax1.grid(color=axis_line_color, linestyle="--", alpha=0.5)

    # -- Stats panel (Text) --
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")

    m = metrics
    stats_text = (
        f"=========================================================\n"
        f"  BACKTEST ENGINE METRICS : {symbol.upper()}\n"
        f"=========================================================\n"
        f"  Final Equity   : {format_currency(results['final_equity'])}\n"
        f"  Total Return   : {((results['final_equity'] / 10000) - 1) * 100:.2f}%\n"
        f"  Max Drawdown   : {results['max_drawdown'] * 100:.2f}%\n"
        f"---------------------------------------------------------\n"
        f"  Total Trades   : {m.get('total_trades', 0)}\n"
        f"  Win Rate       : {m.get('win_rate', 0) * 100:.1f}%\n"
        f"  Profit Factor  : {m.get('profit_factor', 0):.3f}\n"
        f"  Sharpe Ratio   : {m.get('sharpe', 0):.3f}\n"
        f"  Expectancy     : {format_currency(m.get('expectancy', 0))} per trade\n"
        f"  Macro Shocks   : {results.get('shock_blocks', 0)} aborted (|Z| > 2.5)\n"
        f"  Vol Gated      : {results.get('volatility_blocks', 0)} aborted\n"
        f"  Gross Profit   : {format_currency(m.get('gross_profit', 0))}\n"
        f"  Gross Loss     : {format_currency(-m.get('gross_loss', 0))}\n"
        f"=========================================================\n"
    )

    ax2.text(
        0.5,
        0.5,
        stats_text,
        color=accent_color_2,
        fontsize=14,
        fontfamily="monospace",
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(
            facecolor="#1a1a1a", edgecolor=axis_line_color, pad=2.0, boxstyle="round"
        ),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/km/Universal-ML/",
        help="Project root directory",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Target Base Symbol (e.g., BANKNIFTY, BTC)",
    )

    args = parser.parse_args()

    DATA_DIR = args.outdir
    SYMBOL = args.symbol.upper()
    SYMBOL_DIR = os.path.join(DATA_DIR, SYMBOL)
    file_prefix = SYMBOL.lower().replace(" ", "_")
    artifact_paths_1h = migrate_legacy_artifacts(SYMBOL_DIR, file_prefix, "1H")
    migrate_legacy_artifacts(SYMBOL_DIR, file_prefix, "1D", logger=None)

    import sys

    sys.path.append(os.path.join(DATA_DIR, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("[!] FATAL: Cannot locate inference_bridge.py.")
        return

    print("=" * 60)
    print("  INITIALIZING BACKTEST ENGINE")
    print("=" * 60)

    bridge = InferenceBridge(db_path=os.path.join(DATA_DIR, "data_vault", "ohlcv.db"))
    tf_maps = {
        "FUT": bridge.fetch_holographic_stack(SYMBOL, "FUT"),
        "SPOT": bridge.fetch_holographic_stack(SYMBOL, "SPOT"),
    }

    df_1h = tf_maps["FUT"].get("1H")
    df_1d = tf_maps["FUT"].get("1D")
    df_1w = tf_maps["FUT"].get("1W")
    df_1m = tf_maps["FUT"].get("1M")

    if df_1h is None or df_1d is None or df_1w is None or df_1m is None:
        print(f"[!] Missing 1H/1D/1W/1M FUT data for {SYMBOL} in database.")
        return

    model_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "model")
    feat_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "features")
    trade_plan_path = resolve_artifact_path(
        SYMBOL_DIR, file_prefix, "1H", "trade_plan_models"
    )
    # NOTE: 1D conflict gating via separate model is not implemented.

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print(f"[!] Could not find {model_path} or {feat_path}.")
        return

    print(f"\n  [=] Loading Model: {os.path.basename(model_path)}")
    model = joblib.load(model_path)

    with open(feat_path, "r") as f:
        feature_cols = [line.strip() for line in f.readlines() if line.strip()]

    trade_plan_models = (
        joblib.load(trade_plan_path) if os.path.exists(trade_plan_path) else {}
    )
    if trade_plan_models:
        print(
            f"  [=] Trade-plan models loaded. {len(trade_plan_models)} ML exit models available."
        )
    else:
        print(
            "  [!] WARNING: No ML trade-plan models found. Falling back to static ATR exits."
        )

    print("  [=] Reconstructing holographic feature space over historical data...")
    try:
        df_1h, df_1d, df_1w, df_1m = prepare_intraday_thermodynamics(
            df_1h=df_1h,
            df_1d=df_1d,
            df_1w=df_1w,
            df_1m=df_1m,
            spot_1h=tf_maps["SPOT"].get("1H"),
            spot_1d=tf_maps["SPOT"].get("1D"),
            spot_1w=tf_maps["SPOT"].get("1W"),
            symbol=SYMBOL,
            logger=print,
        )
    except ValueError:
        return

    # Step 1: compute atr14 labelling scaffold (used for volatility gate only,
    #         not as a model input)
    df_1h_labelled = _compute_atr14(df_1h.copy())

    # Step 2: run holographic engine — same call as in main()
    df_full = holographic_feature_engine(
        df_1h_labelled,
        df_1d=df_1d,
        df_1w=df_1w,
        df_1m=df_1m,
    )

    # Step 3: ASOF-merge for temporal alignment
    df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)

    # Build model-ready frame; keep atr14 as side-channel for vol-gate
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
    available = [c for c in all_needed_cols if c in df_full.columns]
    df_model_ready = df_full[available].copy()
    for col in feature_cols:
        if col in df_model_ready.columns:
            df_model_ready[col] = (
                df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0)
            )
    df_model_ready = df_model_ready.reset_index(drop=True)

    print(f"  [=] Total Bars for Simulation: {len(df_model_ready)}")

    # ── UME-2 FIX: Load OOS probability map for honest backtesting ──
    # Align OOS probabilities onto the full historical bar timeline.
    oos_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1H", "oos_proba")
    df_backtest = df_model_ready
    if os.path.exists(oos_path):
        oos_proba_map = joblib.load(oos_path)
        prob_array = np.array(
            [
                float(oos_proba_map[pd.Timestamp(t)])
                if pd.Timestamp(t) in oos_proba_map
                else np.nan
                for t in df_backtest["time"]
            ]
        )
        print(
            f"  [=] OOS proba map loaded. {np.isfinite(prob_array).sum()} honest OOS prediction bars aligned to the full timeline."
        )
    else:
        print(
            "  [!] WARNING: No OOS proba map found. Backtest uses in-sample predictions."
        )
        print("      Re-run universal_ml_engine.py to generate a clean OOS map.")
        X = df_backtest[feature_cols]
        prob_array = model.predict_proba(X)[:, 1]

    # ── E3 FIX: Use genuine OOS 1D probabilities for conflict gating ──
    # Load the saved 1D OOS proba map. Only bars with OOS 1D predictions use
    # the genuine probability. Bars without OOS data remain NaN and do not gate.
    oos_1d_path = resolve_artifact_path(SYMBOL_DIR, file_prefix, "1D", "oos_proba")
    if os.path.exists(oos_1d_path):
        oos_proba_map_1d = joblib.load(oos_1d_path)
        date_to_prob_1d = {}
        for ts, prob in oos_proba_map_1d.items():
            date_to_prob_1d[pd.Timestamp(ts).date()] = float(prob)
        prob_array_1d = np.array(
            [
                date_to_prob_1d.get(pd.Timestamp(t).date(), np.nan)
                for t in df_backtest["time"]
            ]
        )
        print(
            f"  [=] 1D OOS proba map loaded. {len(date_to_prob_1d)} honest OOS days available for gating."
        )
    else:
        print("  [!] WARNING: No 1D OOS proba map found. 1D filtering disabled.")
        prob_array_1d = np.full(len(df_backtest), np.nan)

    print("  [=] Executing Bar-by-Bar Portfolio Walkthrough...")
    results = run_backtest(
        df_backtest,
        prob_array,
        prob_array_1d,
        feature_cols,
        trade_plan_models=trade_plan_models,
        initial_capital=10000.0,
        risk_pct=0.02,
        conf_threshold=LIVE_CONFIDENCE_THRESHOLD,
    )

    if results is None:
        return

    metrics = calculate_metrics(
        results["trades"], results.get("equity_curve"), results.get("time_curve")
    )

    print("\n" + "=" * 60)
    print("  PORTFOLIO SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Final Equity   : {format_currency(results['final_equity'])}")
    print(f"  Prediction Bars: {results.get('prediction_bars', 0)}")
    print(f"  Total Trades   : {metrics.get('total_trades', 0)}")
    print(f"  Win Rate       : {metrics.get('win_rate', 0) * 100:.1f}%")
    print(f"  Profit Factor  : {metrics.get('profit_factor', 0):.3f}")
    print(f"  Sharpe Ratio   : {metrics.get('sharpe', 0):.3f}")
    print(f"  Sortino Ratio  : {metrics.get('sortino', 0):.3f}")
    print(f"  Max Consec Loss: {metrics.get('max_consec_loss', 0)}")
    print(f"  True TUW       : {metrics.get('true_tuw_days', 0)} Days")
    print(f"  Conflict Gated : {results.get('conflict_blocks', 0)} blocks bypassed")
    print(
        f"  Shock Gate     : {results.get('shock_blocks', 0)} bars bypassed (|Z| > 2.5)"
    )
    print(f"  Volatility Gate: {results.get('volatility_blocks', 0)} bars bypassed")
    print(f"  Max Drawdown   : {results['max_drawdown'] * 100:.2f}%")
    print("=" * 60)

    report_path = artifact_paths_1h["backtest_report"]
    generate_report(results, metrics, SYMBOL, report_path)
    print(f"\n  [✓] Report visually packaged and saved to {report_path}")

    # -- Seed the Performance Ledger
    try:
        import sys

        sys.path.append(os.path.join(DATA_DIR, "data_vault"))
        from vault_engine import DataVault

        vault = DataVault(db_path=os.path.join(DATA_DIR, "data_vault", "ohlcv.db"))
        vault_trades = []
        for t in results["trades"]:
            v_t = {
                "timestamp": str(t["entry_time"]),
                "base_symbol": SYMBOL,
                "direction": "UP" if t["type"] == "LONG" else "DOWN",
                "conf_score": t["confidence"],
                "entry_price": t["entry_price"],
                "exit_price": t["exit_price"],
                "pnl_r": t["pnl"] / (t["initial_risk"] + 1e-9),
                "win_loss_target": 1 if t["pnl"] > 0 else 0,
            }
            vault_trades.append(v_t)

        vault.log_bulk_trades(vault_trades)
        print(
            f"  [VAULT] Injected {len(vault_trades)} historic trades into Performance Ledger to neutralize Shadow Brain cold-start."
        )
    except Exception as e:
        print(f"  [VAULT] Failed to seed historic trades: {e}")


if __name__ == "__main__":
    main()
