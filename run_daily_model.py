import os
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from universal_ml_engine import parse_tv_log, add_features_single, add_calendar_features, walk_forward

def calculate_metrics(trades):
    if not trades: return {}
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    gross_profit = sum([t['pnl'] for t in wins])
    gross_loss = abs(sum([t['pnl'] for t in losses]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    wr = len(wins) / len(trades)
    return {'total': len(trades), 'wr': wr, 'pf': pf}

def run_backtest(df, prob_array, initial_capital=10000.0, risk_pct=0.02, conf_threshold=0.60):
    equity = initial_capital
    peak_equity = equity
    max_drawdown = 0.0
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
        m2m_equity = equity
        current_time = time_arr[i]
        current_close = close_arr[i]
        current_atr = atr_arr[i]
        proba_up = prob_array[i]
        proba_down = 1.0 - proba_up
        next_open = open_arr[i+1]
        next_high = high_arr[i+1]
        next_low = low_arr[i+1]
        next_close = close_arr[i+1]

        if position is not None:
            trade_closed = False
            p_type = position['type']
            p_sl = position['sl']
            p_tp1 = position['tp1']
            
            if p_type == 'LONG':
                if next_low <= p_sl:
                    exit_price = min(next_open, p_sl)
                    pnl = (exit_price - position['entry_price']) * position['size'] * (0.5 if position['tp1_hit'] else 1.0)
                    position['pnl'] += pnl
                    trade_closed = True
                elif not position['tp1_hit'] and next_high >= p_tp1:
                    position['tp1_hit'] = True
                    exit_price = max(next_open, p_tp1)
                    pnl = (exit_price - position['entry_price']) * position['size'] * 0.5
                    position['pnl'] += pnl  # RDM-3 FIX: += not = to preserve any prior accumulated pnl
                    position['sl'] = position['entry_price']

                if position['tp1_hit'] and not trade_closed:
                    new_sl = next_close - position['trail_dist']
                    if new_sl > position['sl']: position['sl'] = new_sl
            else:
                if next_high >= p_sl:
                    exit_price = max(next_open, p_sl)
                    pnl = (position['entry_price'] - exit_price) * position['size'] * (0.5 if position['tp1_hit'] else 1.0)
                    position['pnl'] += pnl
                    trade_closed = True
                elif not position['tp1_hit'] and next_low <= p_tp1:
                    position['tp1_hit'] = True
                    exit_price = min(next_open, p_tp1)
                    pnl = (position['entry_price'] - exit_price) * position['size'] * 0.5
                    position['pnl'] += pnl  # RDM-3 FIX: += not = to preserve any prior accumulated pnl
                    position['sl'] = position['entry_price']
                
                if position['tp1_hit'] and not trade_closed:
                    new_sl = next_close + position['trail_dist']
                    if new_sl < position['sl']: position['sl'] = new_sl

            if trade_closed:
                equity += position['pnl']
                trades.append(position)
                position = None

        if position is None:
            confidence = max(proba_up, proba_down)
            if confidence >= conf_threshold:
                direction = 'LONG' if proba_up > 0.5 else 'SHORT'
                risk_m = float(confidence * 2.0)
                sl_dist = float(current_atr * risk_m)
                if sl_dist > 0:
                    entry_price = float(next_open)
                    risk_dollar = equity * risk_pct
                    size = risk_dollar / (sl_dist + 1e-9)
                    
                    sl = entry_price - sl_dist if direction == 'LONG' else entry_price + sl_dist
                    tp1 = entry_price + sl_dist if direction == 'LONG' else entry_price - sl_dist

                    position = {
                        'type': direction, 'entry_price': entry_price, 'size': size,
                        'sl': sl, 'tp1': tp1, 'tp1_hit': False, 'trail_dist': sl_dist,
                        'pnl': 0.0, 'status': 'OPEN'
                    }

        # M2M Tracking
        if position is not None:
            if position['type'] == 'LONG':
                unrealized = (next_close - position['entry_price']) * position['size'] * (0.5 if position['tp1_hit'] else 1.0)
            else:
                unrealized = (position['entry_price'] - next_close) * position['size'] * (0.5 if position['tp1_hit'] else 1.0)
            if position['tp1_hit']: m2m_equity += position['pnl']
            m2m_equity += unrealized
        
        peak_equity = max(peak_equity, m2m_equity)
        max_drawdown = max(max_drawdown, (peak_equity - m2m_equity) / peak_equity)
        equity_curve.append(m2m_equity)
        time_curve.append(current_time)

    return {'equity': equity, 'dd': max_drawdown, 'trades': trades, 'eq_c': equity_curve, 't_c': time_curve}

def main():
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, default='/home/km/BankniftyML/')
    args = parser.parse_args()
    DATA_DIR = args.outdir
    CSV_DIR  = os.path.join(DATA_DIR, 'csv_data')

    # RDM-1 FIX: Auto-scan for data files instead of hard-coded XRP paths
    print("Loading 1D and 1W Data (auto-scan)...")
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    df_1d, df_1w = None, None
    SYMBOL = 'UNKNOWN'
    for ffile in csv_files:
        df, sym, tf = parse_tv_log(os.path.join(CSV_DIR, ffile))
        tf = str(tf).upper()
        if sym and sym != 'UNKNOWN':
            SYMBOL = sym.replace('!', '')
        if tf in ['1D', 'D']:
            df_1d = df
            print(f"  [+] Loaded 1D: {ffile}")
        elif tf in ['1W', 'W']:
            df_1w = df
            print(f"  [+] Loaded 1W: {ffile}")

    if df_1d is None or df_1w is None:
        print("[!] Could not find both 1D and 1W CSV files. Aborting.")
        return

    # 1D features
    df_full = add_features_single(df_1d.copy(), prefix='')
    df_full = add_calendar_features(df_full)

    # Merge 1W
    df_1w_feat = add_features_single(df_1w.copy(), prefix='w_')
    w_cols = [c for c in df_1w_feat.columns if c.startswith('w_')] + ['time', 'close', 'high', 'low']
    df_1w_feat = df_1w_feat[w_cols].copy()
    df_1w_feat.rename(columns={'close': 'w_close', 'high': 'w_high', 'low': 'w_low'}, inplace=True)
    df_full = df_full.sort_values('time')
    df_1w_feat = df_1w_feat.sort_values('time')
    merged = pd.merge_asof(df_full, df_1w_feat, on='time', direction='backward', tolerance=pd.Timedelta('7 days'))
    merged['pos_in_weekly_range'] = (merged['close'] - merged['w_low']) / (merged['w_high'] - merged['w_low']).replace(0, np.nan)
    merged['rel_strength_w'] = (merged['close'] - merged['w_close']) / (merged['w_close'] + 1e-9)

    # Target: next daily bar direction
    merged['target'] = (merged['close'].shift(-1) - merged['close'] > 0).astype(int)
    merged = merged.dropna(subset=['target'])

    exclude_cols = {'time', 'open', 'high', 'low', 'close', 'volume', 'target',
                    'w_open', 'w_high', 'w_low', 'w_close', 'w_volume'}
    feature_cols = [c for c in merged.columns if c not in exclude_cols]
    for col in feature_cols:
        merged[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    merged = merged.dropna(subset=['target'] + feature_cols).reset_index(drop=True)
    print(f"Total 1D Bars for Simulation: {len(merged)}")

    # RDM-2 FIX: Use proper walk_forward() for genuinely clean OOS probabilities.
    # The old TimeSeriesSplit approach leaked future labels into early-stopping.
    MIN_TRAIN_BARS_1D = max(200, int(len(merged) * 0.4))
    wf = walk_forward(merged, feature_cols, n_splits=5,
                      min_train_bars=MIN_TRAIN_BARS_1D, test_size_ratio=0.15)
    if not wf:
        print("Walk-forward produced no results. Aborting.")
        return

    # Build backtest arrays from honest OOS proba map
    oos_proba_map = wf['oos_proba_map']
    oos_mask = merged['time'].apply(lambda t: pd.Timestamp(t) in oos_proba_map)
    backtest_df = merged[oos_mask].reset_index(drop=True)
    backtest_probs = np.array([oos_proba_map[pd.Timestamp(t)] for t in backtest_df['time']])

    print(f"Running 1D Portfolio Walkthrough on {len(backtest_df)} honest OOS days...")
    res = run_backtest(backtest_df, backtest_probs)

    if not res['trades']:
        print("No trades taken.")
        return

    m = calculate_metrics(res['trades'])

    fig = plt.figure(figsize=(14, 10), facecolor='#0d0d0d')
    gs = gridspec.GridSpec(2, 1, figure=fig, hspace=0.3, height_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#0d0d0d')
    ax1.tick_params(colors='#e0e0e0')
    ax1.plot(res['t_c'], res['eq_c'], color='#00d4ff', lw=2)
    ax1.fill_between(res['t_c'], res['eq_c'], min(res['eq_c'])*0.99, color='#00d4ff', alpha=0.1)
    ax1.set_title(f'{SYMBOL.upper()} 1D-MODEL Backtest Equity Curve (Honest OOS)', color='white', fontsize=16, fontweight='bold')

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')
    stats_text = (
        f"=========================================================\n"
        f"  BACKTEST 1D ENGINE METRICS : {SYMBOL.upper()}\n"
        f"=========================================================\n"
        f"  Final Equity   : ${res['equity']:,.2f}\n"
        f"  Total Trades   : {m['total']}\n"
        f"  Win Rate       : {m['wr']*100:.1f}%\n"
        f"  Profit Factor  : {m['pf']:.3f}\n"
        f"  Max Drawdown   : {res['dd']*100:.2f}%\n"
        f"=========================================================\n"
    )
    ax2.text(0.5, 0.5, stats_text, color='#00ff88', fontsize=14,
             fontfamily='monospace', fontweight='bold', ha='center', va='center',
             bbox=dict(facecolor='#1a1a1a', edgecolor='#444444', pad=2.0, boxstyle='round'))

    plt.tight_layout()
    rep_path = os.path.join(DATA_DIR, f'{SYMBOL.lower()}_1D_backtest_report.png')
    plt.savefig(rep_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Done. Saved to {rep_path}")

if __name__ == '__main__':
    main()
