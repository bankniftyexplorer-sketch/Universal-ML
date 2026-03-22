import os
import argparse
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# Suppress lightgbm warnings locally if needed
import warnings
warnings.filterwarnings('ignore')

from universal_ml_engine import (
    parse_tv_log, add_features_single, add_calendar_features,
    merge_higher_tf, BARRIER_ATR_MULT, BARRIER_HORIZON_BARS,
    LIVE_CONFIDENCE_THRESHOLD, VOL_GATE_LOOKBACK,
    EXEC_FEE_PCT, simulate_trade_path_from_arrays
)

def format_currency(val):
    return f"${val:,.2f}"

def run_backtest(df, prob_array, prob_array_1d, initial_capital=10000.0, risk_pct=0.02,
                 conf_threshold=LIVE_CONFIDENCE_THRESHOLD, fixed_risk=True,
                 slippage_bps=0.0003, max_hold_bars=BARRIER_HORIZON_BARS):
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    conflict_blocks = 0
    volatility_blocks = 0
    no_prediction_bars = 0
    trades = []
    equity_curve = []
    time_curve = []

    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    time_arr = df['time'].values
    atr_arr = df['atr14'].values
    atr_ma50_arr = pd.Series(atr_arr).rolling(VOL_GATE_LOOKBACK).mean().to_numpy()

    i = 0
    while i < len(df) - 1:
        current_time = pd.to_datetime(time_arr[i])
        current_atr = atr_arr[i]
        equity_curve.append(equity)
        time_curve.append(current_time)

        proba_up = prob_array[i]
        proba_up_1d = prob_array_1d[i]
        if not np.isfinite(proba_up):
            no_prediction_bars += 1
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        confidence = max(proba_up, 1.0 - proba_up)
        atr_ma50 = atr_ma50_arr[i]
        if confidence < conf_threshold:
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue
        if not np.isfinite(current_atr) or current_atr <= 0:
            i += 1
            continue
        if not np.isfinite(atr_ma50) or current_atr <= atr_ma50:
            volatility_blocks += 1
            peak_equity = max(peak_equity, equity)
            max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
            i += 1
            continue

        direction = 'LONG' if proba_up > 0.5 else 'SHORT'
        if np.isfinite(proba_up_1d):
            direction_1d = 'LONG' if proba_up_1d > 0.5 else 'SHORT'
            if direction != direction_1d:
                conflict_blocks += 1
                peak_equity = max(peak_equity, equity)
                max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
                i += 1
                continue

        risk_dollar = (initial_capital if fixed_risk else equity) * risk_pct
        trade_path = simulate_trade_path_from_arrays(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            time_arr,
            i + 1,
            direction,
            float(current_atr * BARRIER_ATR_MULT),
            horizon=max_hold_bars,
            fee_pct=EXEC_FEE_PCT,
            slippage_bps=slippage_bps
        )
        if not np.isfinite(trade_path['total_r']):
            i += 1
            continue

        pnl = trade_path['total_r'] * risk_dollar
        equity += pnl
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, (peak_equity - equity) / peak_equity)
        trades.append({
            'type': direction,
            'entry_price': trade_path['entry_price'],
            'exit_price': trade_path['exit_price'],
            'entry_time': time_arr[i + 1],
            'exit_time': trade_path['exit_time'],
            'exit_reason': trade_path['exit_reason'],
            'initial_risk': risk_dollar,
            'confidence': confidence,
            'pnl': pnl,
            'status': 'CLOSED',
            'tp1_hit': trade_path['tp1_hit'],
            'tp2_hit': trade_path['tp2_hit'],
        })

        for k in range(i + 1, min(trade_path['exit_idx'], len(df) - 1) + 1):
            time_curve.append(pd.to_datetime(time_arr[k]))
            equity_curve.append(equity)
        i = max(i + 1, trade_path['exit_idx'] + 1)

    results = {
        'final_equity': equity,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'equity_curve': equity_curve,
        'time_curve': time_curve,
        'conflict_blocks': conflict_blocks,
        'volatility_blocks': volatility_blocks,
        'no_prediction_bars': no_prediction_bars,
        'prediction_bars': int(np.isfinite(prob_array).sum())
    }
    return results

def calculate_metrics(trades):
    if not trades:
        return {}
    
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    
    gross_profit = sum([t['pnl'] for t in wins])
    gross_loss = abs(sum([t['pnl'] for t in losses]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    win_rate = len(wins) / len(trades)
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
    
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

    # E6: Risk-adjusted metrics
    r_multiples = [t['pnl'] / (t.get('initial_risk', 1) + 1e-9) for t in trades]
    sharpe = (np.mean(r_multiples) / (np.std(r_multiples) + 1e-9)) * np.sqrt(252) if len(r_multiples) > 1 else 0.0
    downside_returns = [r for r in r_multiples if r < 0]
    sortino = (np.mean(r_multiples) / (np.std(downside_returns) + 1e-9)) * np.sqrt(252) if downside_returns else float('inf')

    # Max consecutive losses
    max_consec_loss = 0
    current_streak = 0
    for t in trades:
        if t['pnl'] <= 0:
            current_streak += 1
            max_consec_loss = max(max_consec_loss, current_streak)
        else:
            current_streak = 0

    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_consec_loss': max_consec_loss
    }

def generate_report(results, metrics, symbol, save_path):
    fig = plt.figure(figsize=(14, 10), facecolor='#0d0d0d')
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, height_ratios=[2, 1])

    dark_bg_color = '#0d0d0d'
    light_text_color = '#e0e0e0'
    axis_line_color = '#444444'
    accent_color_1 = '#00d4ff'
    accent_color_2 = '#00ff88'
    accent_color_3 = '#ff6b6b'
    
    # -- Eq Curve panel --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(dark_bg_color)
    ax1.tick_params(colors=light_text_color)
    for spine in ax1.spines.values(): spine.set_color(axis_line_color)
    
    ax1.plot(results['time_curve'], results['equity_curve'], color=accent_color_1, lw=2)
    ax1.fill_between(results['time_curve'], results['equity_curve'], min(results['equity_curve'])*0.99, color=accent_color_1, alpha=0.1)
    ax1.set_title(f'{symbol.upper()} Portfolio Backtest Equity Curve', color='white', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Equity ($)', color=light_text_color, fontsize=12)
    ax1.grid(color=axis_line_color, linestyle='--', alpha=0.5)

    # -- Stats panel (Text) --
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    
    m = metrics
    stats_text = (
        f"=========================================================\n"
        f"  BACKTEST ENGINE METRICS : {symbol.upper()}\n"
        f"=========================================================\n"
        f"  Final Equity   : {format_currency(results['final_equity'])}\n"
        f"  Total Return   : {((results['final_equity']/10000)-1)*100:.2f}%\n"
        f"  Max Drawdown   : {results['max_drawdown']*100:.2f}%\n"
        f"---------------------------------------------------------\n"
        f"  Total Trades   : {m.get('total_trades', 0)}\n"
        f"  Win Rate       : {m.get('win_rate', 0)*100:.1f}%\n"
        f"  Profit Factor  : {m.get('profit_factor', 0):.3f}\n"
        f"  Sharpe Ratio   : {m.get('sharpe', 0):.3f}\n"
        f"  Sortino Ratio  : {m.get('sortino', 0):.3f}\n"
        f"  Max Consec Loss: {m.get('max_consec_loss', 0)}\n"
        f"  Expectancy     : {format_currency(m.get('expectancy', 0))} per trade\n"
        f"  Conflict Gated : {results.get('conflict_blocks', 0)} blocks bypassed\n"
        f"  Gross Profit   : {format_currency(m.get('gross_profit', 0))}\n"
        f"  Gross Loss     : {format_currency(-m.get('gross_loss', 0))}\n"
        f"=========================================================\n"
    )
    
    ax2.text(0.5, 0.5, stats_text, color=accent_color_2, fontsize=14, 
             fontfamily='monospace', fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='#1a1a1a', edgecolor=axis_line_color, pad=2.0, boxstyle='round'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='/home/km/BankniftyML/', help='Directory with models and csv')
    args = parser.parse_args()
    
    DATA_DIR = args.outdir
    CSV_DIR = os.path.join(DATA_DIR, 'csv_data')
    if not os.path.exists(CSV_DIR):
        print(f"[!] No csv_data folder in {DATA_DIR}")
        return

    # Find valid models manually or infer symbol
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if len(csv_files) < 4:
        print("[!] Need 4 CSV files (1H, 1D, 1W, 1M) to reconstruct features.")
        return

    df_1h, df_1d, df_1w, df_1m = None, None, None, None
    SYMBOL = "UNKNOWN"

    print("=" * 60)
    print("  INITIALIZING BACKTEST ENGINE")
    print("=" * 60)

    for ffile in csv_files:
        path = os.path.join(CSV_DIR, ffile)
        df, sym, tf = parse_tv_log(path)
        tf = str(tf).upper()
        
        if sym and sym != "UNKNOWN":
            SYMBOL = sym.replace('!', '')

        if tf in ['60', '1H']:
            df_1h = df
            print(f"  [+] Loaded 1H: {ffile}")
        elif tf in ['1D', 'D']:
            df_1d = df
            print(f"  [+] Loaded 1D: {ffile}")
        elif tf in ['1W', 'W']:
            df_1w = df
            print(f"  [+] Loaded 1W: {ffile}")
        elif tf in ['1M', 'M']:
            df_1m = df
            print(f"  [+] Loaded 1M: {ffile}")

    if df_1h is None or df_1d is None or df_1w is None or df_1m is None:
        print("[!] Missing either 1H, 1D, 1W, or 1M file.")
        return

    file_prefix = SYMBOL.lower().replace(' ', '_')
    model_path = os.path.join(DATA_DIR, f'{file_prefix}_ultimate_model.pkl')
    feat_path = os.path.join(DATA_DIR, f'{file_prefix}_ultimate_features.txt')
    
    model_1d_path = os.path.join(DATA_DIR, f'{file_prefix}_ultimate_model_1d.pkl')
    feat_1d_path = os.path.join(DATA_DIR, f'{file_prefix}_ultimate_features_1d.txt')

    if not os.path.exists(model_path) or not os.path.exists(feat_path):
        print(f"[!] Could not find {model_path} or {feat_path}.")
        return

    if not os.path.exists(model_1d_path) or not os.path.exists(feat_1d_path):
        print(f"[!] Could not find 1D models in {model_1d_path}")
        return

    print(f"\n  [=] Loading Model: {file_prefix}_ultimate_model.pkl")
    model = joblib.load(model_path)
    
    print(f"  [=] Loading Secondary 1D Model: {file_prefix}_ultimate_model_1d.pkl")
    model_1d = joblib.load(model_1d_path)
    
    with open(feat_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        
    with open(feat_1d_path, 'r') as f:
        feature_cols_1d = [line.strip() for line in f.readlines() if line.strip()]

    print("  [=] Reconstructing feature space over historical data...")
    df_full = add_features_single(df_1h.copy(), prefix='', compute_vwap=True)  # UME-3 FIX
    df_full = add_calendar_features(df_full)
    df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)
    
    extra_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if 'atr14' not in feature_cols:
        extra_cols.append('atr14')

    # Rebuild the full feature timeline. Keep every bar so open positions are
    # evaluated on actual consecutive market bars, not only on predicted rows.
    all_needed_cols = list(set(feature_cols + extra_cols))
    df_model_ready = df_full[all_needed_cols].copy()
    for col in feature_cols:
        df_model_ready[col] = df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    df_model_ready = df_model_ready.reset_index(drop=True)
    
    print(f"  [=] Total Bars for Simulation: {len(df_model_ready)}")

    # ── UME-2 FIX: Load OOS probability map for honest backtesting ──
    # Align OOS probabilities onto the full historical bar timeline.
    oos_path = os.path.join(DATA_DIR, f'{file_prefix}_oos_proba.pkl')
    df_backtest = df_model_ready
    if os.path.exists(oos_path):
        oos_proba_map = joblib.load(oos_path)
        prob_array = np.array([
            float(oos_proba_map[pd.Timestamp(t)]) if pd.Timestamp(t) in oos_proba_map else np.nan
            for t in df_backtest['time']
        ])
        print(f"  [=] OOS proba map loaded. {np.isfinite(prob_array).sum()} honest OOS prediction bars aligned to the full timeline.")
    else:
        print("  [!] WARNING: No OOS proba map found. Backtest uses in-sample predictions.")
        print("      Re-run universal_ml_engine.py to generate a clean OOS map.")
        X = df_backtest[feature_cols]
        prob_array = model.predict_proba(X)[:, 1]

    # ── E3 FIX: Use genuine OOS 1D probabilities for conflict gating ──
    # Load the saved 1D OOS proba map. Only bars with OOS 1D predictions use
    # the genuine probability. Bars without OOS data remain NaN and do not gate.
    oos_1d_path = os.path.join(DATA_DIR, f'{file_prefix}_oos_proba_1d.pkl')
    if os.path.exists(oos_1d_path):
        oos_proba_map_1d = joblib.load(oos_1d_path)
        date_to_prob_1d = {}
        for ts, prob in oos_proba_map_1d.items():
            date_to_prob_1d[pd.Timestamp(ts).date()] = float(prob)
        prob_array_1d = np.array([
            date_to_prob_1d.get(pd.Timestamp(t).date(), np.nan)
            for t in df_backtest['time']
        ])
        print(f"  [=] 1D OOS proba map loaded. {len(date_to_prob_1d)} honest OOS days available for gating.")
    else:
        # Fallback: compute on actual daily bars (in-sample — warn loudly)
        print("  [!] WARNING: No 1D OOS proba map found. Using in-sample 1D predictions.")
        print("      Re-run universal_ml_engine.py to generate clean 1D OOS map.")
        df_1d_for_filter = add_features_single(df_1d.copy(), prefix='d_')
        for col in feature_cols_1d:
            if col in df_1d_for_filter.columns:
                df_1d_for_filter[col] = df_1d_for_filter[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        valid_1d_mask = df_1d_for_filter[feature_cols_1d].notna().all(axis=1)
        df_1d_valid = df_1d_for_filter[valid_1d_mask].reset_index(drop=True)
        daily_proba = model_1d.predict_proba(df_1d_valid[feature_cols_1d])[:, 1]
        date_to_prob_1d = {
            pd.Timestamp(row['time']).date(): float(prob)
            for row, prob in zip(df_1d_valid.to_dict('records'), daily_proba)
        }
        prob_array_1d = np.array([
            date_to_prob_1d.get(pd.Timestamp(t).date(), np.nan)
            for t in df_backtest['time']
        ])
        print(f"  [=] 1D filter computed on {len(df_1d_valid)} actual daily bars (IN-SAMPLE FALLBACK).")

    print("  [=] Executing Bar-by-Bar Portfolio Walkthrough...")
    results = run_backtest(
        df_backtest,
        prob_array,
        prob_array_1d,
        initial_capital=10000.0,
        risk_pct=0.02,
        conf_threshold=LIVE_CONFIDENCE_THRESHOLD
    )
    
    if results is None: 
        return

    metrics = calculate_metrics(results['trades'])

    print("\n" + "=" * 60)
    print("  PORTFOLIO SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Final Equity   : {format_currency(results['final_equity'])}")
    print(f"  Prediction Bars: {results.get('prediction_bars', 0)}")
    print(f"  Total Trades   : {metrics.get('total_trades', 0)}")
    print(f"  Win Rate       : {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor  : {metrics.get('profit_factor', 0):.3f}")
    print(f"  Sharpe Ratio   : {metrics.get('sharpe', 0):.3f}")
    print(f"  Sortino Ratio  : {metrics.get('sortino', 0):.3f}")
    print(f"  Max Consec Loss: {metrics.get('max_consec_loss', 0)}")
    print(f"  Conflict Gated : {results.get('conflict_blocks', 0)} blocks bypassed")
    print(f"  Volatility Gate: {results.get('volatility_blocks', 0)} bars bypassed")
    print(f"  Max Drawdown   : {results['max_drawdown']*100:.2f}%")
    print("=" * 60)

    report_path = os.path.join(DATA_DIR, f'{file_prefix}_backtest_report.png')
    generate_report(results, metrics, SYMBOL, report_path)
    print(f"\n  [✓] Report visually packaged and saved to {report_path}")

if __name__ == '__main__':
    main()
