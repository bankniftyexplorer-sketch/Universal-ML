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
    merge_higher_tf, add_target
)

def format_currency(val):
    return f"${val:,.2f}"

def _eval_pos(position, next_open, next_high, next_low, next_close, next_time, fee_pct=0.0005):
    trade_closed = False
    p_type = position['type']
    p_sl = position['sl']
    p_tp1 = position['tp1']
    
    if p_type == 'LONG':
        if next_low <= p_sl:
            exit_price = min(next_open, p_sl)
            # BE-1 FIX: after TP1, only half-size remains — fee must reflect that
            remaining_size = position['size'] * 0.5 if position['tp1_hit'] else position['size']
            fee_exit = exit_price * remaining_size * fee_pct
            pnl = (exit_price - position['entry_price']) * remaining_size
            position['pnl'] += (pnl - fee_exit)
            trade_closed = True
            position['exit_price'] = exit_price
            position['exit_time'] = next_time
            position['exit_reason'] = 'SL Hit'
            
        elif not position['tp1_hit'] and next_high >= p_tp1:
            position['tp1_hit'] = True
            exit_price = max(next_open, p_tp1)
            fee_exit = exit_price * (position['size'] * 0.5) * fee_pct
            pnl = (exit_price - position['entry_price']) * (position['size'] * 0.5)
            position['pnl'] += (pnl - fee_exit)
            position['sl'] = position['entry_price']
            
        if position['tp1_hit'] and not trade_closed:
            new_sl = next_close - position['trail_dist']
            if new_sl > position['sl']:
                position['sl'] = new_sl
                
    elif p_type == 'SHORT':
        if next_high >= p_sl:
            exit_price = max(next_open, p_sl)
            # BE-1 FIX: after TP1, only half-size remains — fee must reflect that
            remaining_size = position['size'] * 0.5 if position['tp1_hit'] else position['size']
            fee_exit = exit_price * remaining_size * fee_pct
            pnl = (position['entry_price'] - exit_price) * remaining_size
            position['pnl'] += (pnl - fee_exit)
            trade_closed = True
            position['exit_price'] = exit_price
            position['exit_time'] = next_time
            position['exit_reason'] = 'SL Hit'
            
        elif not position['tp1_hit'] and next_low <= p_tp1:
            position['tp1_hit'] = True
            exit_price = min(next_open, p_tp1)
            fee_exit = exit_price * (position['size'] * 0.5) * fee_pct
            pnl = (position['entry_price'] - exit_price) * (position['size'] * 0.5)
            position['pnl'] += (pnl - fee_exit)
            position['sl'] = position['entry_price']
            
        if position['tp1_hit'] and not trade_closed:
            new_sl = next_close + position['trail_dist']
            if new_sl < position['sl']:
                position['sl'] = new_sl
                
    return trade_closed

def run_backtest(df, prob_array, prob_array_1d, initial_capital=10000.0, risk_pct=0.02, conf_threshold=0.60):
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
    conflict_blocks = 0
    fee_pct = 0.0005 # 0.05% execution fee

    position = None 
    trades = []
    equity_curve = []
    time_curve = []

    close_arr = df['close'].values
    open_arr = df['open'].values
    high_arr = df['high'].values
    low_arr = df['low'].values
    time_arr = df['time'].values
    atr_arr = df['atr14'].values

    for i in range(len(df) - 1):
        current_time = pd.to_datetime(time_arr[i])
        current_atr = atr_arr[i]
        
        proba_up = prob_array[i]
        proba_down = 1.0 - proba_up
        proba_up_1d = prob_array_1d[i]
        proba_down_1d = 1.0 - proba_up_1d
        
        next_open = open_arr[i+1]
        next_high = high_arr[i+1]
        next_low = low_arr[i+1]
        next_close = close_arr[i+1]
        next_time = time_arr[i+1]

        # 1. Manage Existing
        if position is not None:
            if _eval_pos(position, next_open, next_high, next_low, next_close, next_time, fee_pct):
                equity += position['pnl']
                trades.append(position)
                position = None

        # 2. Open New
        if position is None:
            confidence = max(proba_up, proba_down)
            if confidence >= conf_threshold:
                direction = 'LONG' if proba_up > 0.5 else 'SHORT'
                direction_1d = 'LONG' if proba_up_1d > 0.5 else 'SHORT'
                
                if direction != direction_1d:
                    conflict_blocks += 1
                    continue
                
                risk_m = float(confidence * 2.0)
                sl_dist = float(current_atr * risk_m)
                
                if sl_dist <= 0: continue
                
                entry_price = float(next_open)
                risk_dollar = equity * risk_pct
                size = risk_dollar / (sl_dist + 1e-9)
                fee_entry = entry_price * size * fee_pct
                
                position = {
                    'type': direction,
                    'entry_price': entry_price,
                    'size': size,
                    'sl': entry_price - sl_dist if direction == 'LONG' else entry_price + sl_dist,
                    'tp1': entry_price + sl_dist if direction == 'LONG' else entry_price - sl_dist,
                    'tp1_hit': False,
                    'trail_dist': sl_dist,
                    'entry_time': next_time,
                    'initial_risk': risk_dollar,
                    'pnl': -fee_entry,
                    'confidence': confidence,
                    'status': 'OPEN'
                }
                # BE-2 FIX: do NOT evaluate the position on the bar it was just entered.
                # The first evaluation happens on the NEXT bar (next loop iteration).
                # Immediate self-evaluation creates intra-bar lookahead bias.

        # M2M equity tracking
        m2m_equity = equity
        if position is not None:
            if position['type'] == 'LONG':
                unrealized = (next_close - position['entry_price']) * position['size']
                if position['tp1_hit']:
                    unrealized = (next_close - position['entry_price']) * (position['size'] * 0.5)
            else:
                unrealized = (position['entry_price'] - next_close) * position['size']
                if position['tp1_hit']:
                    unrealized = (position['entry_price'] - next_close) * (position['size'] * 0.5)
            m2m_equity += (position['pnl'] + unrealized)
        
        peak_equity = max(peak_equity, m2m_equity)
        max_drawdown = max(max_drawdown, (peak_equity - m2m_equity) / peak_equity)
        equity_curve.append(m2m_equity)
        time_curve.append(current_time)

    # Clean up open trades at end
    if position is not None:
        last_close = close_arr[-1]
        fee_exit = last_close * position['size'] * 0.5 * fee_pct if position['tp1_hit'] else last_close * position['size'] * fee_pct
        if position['type'] == 'LONG':
            pnl = (last_close - position['entry_price']) * (position['size'] * 0.5 if position['tp1_hit'] else position['size'])
        else:
            pnl = (position['entry_price'] - last_close) * (position['size'] * 0.5 if position['tp1_hit'] else position['size'])
        
        position['pnl'] += (pnl - fee_exit)
        equity += position['pnl']
        trades.append(position)

    results = {
        'final_equity': equity,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'equity_curve': equity_curve,
        'time_curve': time_curve,
        'conflict_blocks': conflict_blocks
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
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
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
    
    # Need target for walk-forward comparison context, though backtest technically only needs features and OHLC
    df_full = add_target(df_full, ahead=1)

    extra_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if 'atr14' not in feature_cols:
        extra_cols.append('atr14')

    # Clean data identical to training
    all_needed_cols = list(set(feature_cols + extra_cols + feature_cols_1d))
    df_model_ready = df_full[all_needed_cols].copy()
    for col in set(feature_cols + feature_cols_1d):
        df_model_ready[col] = df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    df_model_ready = df_model_ready.dropna(subset=list(set(feature_cols + feature_cols_1d))).reset_index(drop=True)
    
    print(f"  [=] Total Valid Bars for Simulation: {len(df_model_ready)}")

    # ── UME-2 FIX: Load OOS probability map for honest backtesting ──
    # When available, only bars with genuine OOS predictions are backtested.
    # Falls back to in-sample batch prediction with a loud warning if map is absent.
    oos_path = os.path.join(DATA_DIR, f'{file_prefix}_oos_proba.pkl')
    if os.path.exists(oos_path):
        oos_proba_map = joblib.load(oos_path)
        oos_mask = df_model_ready['time'].apply(
            lambda t: pd.Timestamp(t) in oos_proba_map
        )
        df_backtest = df_model_ready[oos_mask].reset_index(drop=True)
        prob_array = np.array([
            oos_proba_map[pd.Timestamp(t)] for t in df_backtest['time']
        ])
        print(f"  [=] OOS proba map loaded. Backtest restricted to {len(df_backtest)} honest OOS bars.")
    else:
        print("  [!] WARNING: No OOS proba map found. Backtest uses in-sample predictions.")
        print("      Re-run universal_ml_engine.py to generate a clean OOS map.")
        df_backtest = df_model_ready
        X = df_backtest[feature_cols]
        prob_array = model.predict_proba(X)[:, 1]

    # ── BE-3 FIX: Compute 1D probabilities on ACTUAL daily bars, not on duplicated 1H rows ──
    # The 1D model was trained on daily-bar sequences; applying it to merged 1H rows
    # introduces distribution mismatch (same feature values for 23 consecutive hours).
    # Fix: run predict on the actual 1D dataframe, then map back by calendar date.
    df_1d_for_filter = add_features_single(df_1d.copy(), prefix='d_')  # compute_vwap=False (correct default)
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
    # Map each 1H bar to its calendar date's 1D probability (neutral 0.5 if date missing)
    prob_array_1d = np.array([
        date_to_prob_1d.get(pd.Timestamp(t).date(), 0.5)
        for t in df_backtest['time']
    ])
    print(f"  [=] 1D filter computed on {len(df_1d_valid)} actual daily bars.")

    print("  [=] Executing Bar-by-Bar Portfolio Walkthrough...")
    results = run_backtest(df_backtest, prob_array, prob_array_1d, initial_capital=10000.0, risk_pct=0.02, conf_threshold=0.60)
    
    if results is None: 
        return

    metrics = calculate_metrics(results['trades'])

    print("\n" + "=" * 60)
    print("  PORTFOLIO SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Final Equity   : {format_currency(results['final_equity'])}")
    print(f"  Total Trades   : {metrics.get('total_trades', 0)}")
    print(f"  Win Rate       : {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  Profit Factor  : {metrics.get('profit_factor', 0):.3f}")
    print(f"  Conflict Gated : {results.get('conflict_blocks', 0)} blocks bypassed")
    print(f"  Max Drawdown   : {results['max_drawdown']*100:.2f}%")
    print("=" * 60)

    report_path = os.path.join(DATA_DIR, f'{file_prefix}_backtest_report.png')
    generate_report(results, metrics, SYMBOL, report_path)
    print(f"\n  [✓] Report visually packaged and saved to {report_path}")

if __name__ == '__main__':
    main()
