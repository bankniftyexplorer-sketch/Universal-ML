"""
Universal ML Direction Predictor — TOON v4.0 (Pure Holographic Engine)
Geometry-only features: scale-invariant candle shape coordinates.
No classical indicators. 4 extraction layers. Multi-timeframe pyramid.
Walk-forward validated | i7-4770 / 16 GB RAM / CPU only
"""

import re
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
import argparse
import os
import subprocess
import sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from holographic_engine import (
    holographic_feature_engine,
    feature_selection_pipeline,
    HOLO_WINDOWS_1H, HOLO_WINDOWS_1D, HOLO_WINDOWS_1W, HOLO_WINDOWS_1M,
    FINAL_FEAT_BUDGET,
)

warnings.filterwarnings('ignore')

# Execution/training alignment constants
BARRIER_ATR_MULT = 1.25
BARRIER_HORIZON_BARS = 24
LIVE_CONFIDENCE_THRESHOLD = 0.70
VOL_GATE_LOOKBACK = 50
TP1_R_MULT = 1.0
TP2_R_MULT = 2.0
TP1_FRACTION = 0.50
TP2_FRACTION = 0.25
RUNNER_FRACTION = 0.25
TRAIL_R_MULT = 1.0
MIN_LABEL_EDGE_R = 0.25
MIN_LABEL_BEST_R = 0.15
EXEC_FEE_PCT = 0.0005
EXEC_SLIPPAGE_BPS = 0.0003
TIME_COL_CANDIDATES = ('Date', 'date', 'Datetime', 'datetime', 'Timestamp', 'timestamp', 'Time', 'time')
MESSAGE_COL_CANDIDATES = ('Message', 'message')
MODEL_N_JOBS = 4
TRADE_PLAN_LABEL_COLS = (
    'long_mfe_atr', 'long_mae_atr',
    'short_mfe_atr', 'short_mae_atr'
)

# ─────────────────────────────────────────────
# 1. PARSER  (handles Indian comma-formatted numbers)
# ─────────────────────────────────────────────

def _find_first_column(raw: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in raw.columns:
            return name
    lowered = {str(col).strip().lower(): col for col in raw.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def _infer_symbol_timeframe_from_filename(path: str) -> tuple[str, str]:
    stem = os.path.splitext(os.path.basename(path))[0]
    upper_stem = stem.upper()
    tf_patterns = [('1M', r'(^|[_\- ])1M($|[_\- ])'),
                   ('1W', r'(^|[_\- ])1W($|[_\- ])'),
                   ('1D', r'(^|[_\- ])1D($|[_\- ])'),
                   ('60', r'(^|[_\- ])1H($|[_\- ])|(^|[_\- ])60($|[_\- ])')]
    tf_guess = ''
    for tf_val, pattern in tf_patterns:
        if re.search(pattern, upper_stem):
            tf_guess = tf_val
            upper_stem = re.sub(pattern, '_', upper_stem)
            break
    symbol_guess = re.sub(r'[_\- ]+', '_', upper_stem).strip('_')
    return symbol_guess, tf_guess


def _timeframe_to_timedelta(tf_str: str) -> pd.Timedelta | pd.DateOffset:
    tf_key = str(tf_str).strip().upper()
    if tf_key in {'60', '1H', 'H'}:
        return pd.Timedelta(hours=1)
    if tf_key in {'1D', 'D'}:
        return pd.Timedelta(days=1)
    if tf_key in {'1W', 'W'}:
        return pd.Timedelta(weeks=1)
    if tf_key in {'1M', 'M'}:
        return pd.DateOffset(months=1)
    return pd.Timedelta(0)


def _drop_timezone_preserve_wall_clock(parsed: pd.Series) -> pd.Series:
    try:
        return parsed.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return parsed


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors='coerce', utc=False)
    parsed = _drop_timezone_preserve_wall_clock(parsed)
    if parsed.notna().mean() >= 0.7:
        return parsed

    numeric = pd.to_numeric(text.str.replace(',', '', regex=False), errors='coerce')
    if numeric.notna().mean() < 0.7:
        return parsed

    median_abs = numeric.dropna().abs().median()
    if median_abs >= 1e17:
        unit = 'ns'
    elif median_abs >= 1e14:
        unit = 'us'
    elif median_abs >= 1e11:
        unit = 'ms'
    else:
        unit = 's'
    return pd.to_datetime(numeric, unit=unit, errors='coerce')


def _extract_symbol_and_timeframe(message_series: pd.Series, path: str) -> tuple[str, str]:
    symbol_guess, tf_guess = _infer_symbol_timeframe_from_filename(path)
    symbol_str = symbol_guess
    tf_str = tf_guess
    try:
        symbol_extracted = message_series.str.extract(r'SYMBOL:\s*([^,]+)', expand=False).dropna()
        if not symbol_extracted.empty:
            symbol_str = str(symbol_extracted.iloc[-1]).strip()

        tf_extracted = message_series.str.extract(r'TIME FRAME:\s*([A-Za-z0-9_]+)', expand=False).dropna()
        if not tf_extracted.empty:
            tf_str = str(tf_extracted.iloc[-1]).strip()
    except Exception:
        pass
    return symbol_str, tf_str


def _detect_time_column(raw: pd.DataFrame, message_series: pd.Series, tf_str: str) -> pd.Series:
    for col in TIME_COL_CANDIDATES:
        if col in raw.columns:
            parsed = _parse_datetime_series(raw[col])
            if parsed.notna().any():
                return parsed

    close_time = message_series.str.extract(r'CLOSE TIME:\s*([\d,]+)', expand=False)
    close_numeric = pd.to_numeric(close_time.str.replace(',', '', regex=False), errors='coerce')
    if close_numeric.notna().any():
        close_dt = pd.to_datetime(close_numeric, unit='ms', errors='coerce')
        return close_dt - _timeframe_to_timedelta(tf_str)

    return pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns]')

def parse_tv_log(path: str) -> tuple:
    """Parse TradingView log CSV and automatically detect Symbol and Timeframe."""
    # Use robust separator, handle bad lines
    raw = pd.read_csv(path, on_bad_lines='skip')
    
    if raw.empty:
        return pd.DataFrame(), None, None

    message_col = _find_first_column(raw, MESSAGE_COL_CANDIDATES)
    if message_col is None:
        return pd.DataFrame(), None, None

    message_series = raw[message_col].astype(str)
    symbol_str, tf_str = _extract_symbol_and_timeframe(message_series, path)

    df = pd.DataFrame()
    df['time'] = _detect_time_column(raw, message_series, tf_str)

    # Vectorized regex extraction directly from the message string.
    # Handles numbers with commas by replacing them post-extraction.
    for col, key in [('open', 'OPEN'), ('high', 'HIGH'), ('low', 'LOW'), ('close', 'CLOSE'), ('volume', 'VOLUME')]:
        extracted = message_series.str.extract(rf'{key}:\s*([\d,\.]+)', expand=False)
        df[col] = pd.to_numeric(extracted.str.replace(',', ''), errors='coerce')

    df = df.dropna().sort_values('time').reset_index(drop=True)
    return df, symbol_str, tf_str


# ─────────────────────────────────────────────
# 2. ATR14 — labelling scaffold only
#    Used by add_target() to set barrier distances.
#    NEVER passed as a model input feature.
# ─────────────────────────────────────────────

def _compute_atr14(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute atr14 on a dataframe that has 'high', 'low', 'close' columns.
    Adds a single column 'atr14'. Used exclusively by add_target().
    The model never sees this column as an input feature.
    """
    c = df['close']
    h = df['high']
    l = df['low']
    pc = c.shift(1).fillna(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    df['atr14'] = tr.rolling(14).mean().fillna(tr)
    return df




def merge_higher_tf(df_1h: pd.DataFrame,
                    df_1d: pd.DataFrame,
                    df_1w: pd.DataFrame,
                    df_1m: pd.DataFrame) -> pd.DataFrame:
    """
    Merge raw OHLCV from daily, weekly, and monthly bars into 1H dataframe.
    No classical indicators are computed here — the holographic engine handles
    all feature extraction separately.
    The look-ahead shift (+1 period) is preserved verbatim.
    """
    # ── LOOK-AHEAD FIX: Shift higher-TF timestamps forward by one full period ──
    # Only COMPLETED (closed) bars are available at any given 1H bar.
    df_1d_feat = df_1d[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_1w_feat = df_1w[['time', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_1m_feat = df_1m[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

    df_1d_feat.columns = ['time', 'd_open', 'd_high', 'd_low', 'd_close', 'd_volume']
    df_1w_feat.columns = ['time', 'w_open', 'w_high', 'w_low', 'w_close', 'w_volume']
    df_1m_feat.columns = ['time', 'm_open', 'm_high', 'm_low', 'm_close', 'm_volume']

    df_1d_feat['time'] = df_1d_feat['time'] + pd.Timedelta(days=1)
    df_1w_feat['time'] = df_1w_feat['time'] + pd.Timedelta(weeks=1)
    m_times = df_1m_feat['time']
    m_next  = m_times.shift(-1)
    m_gap   = m_times.diff().dropna().median() if len(m_times) > 1 else pd.DateOffset(months=1)
    df_1m_feat['time'] = m_next.where(m_next.notna(), m_times + m_gap)

    # Sort for merge_asof
    df_1h     = df_1h.sort_values('time').reset_index(drop=True)
    df_1d_feat = df_1d_feat.sort_values('time')
    df_1w_feat = df_1w_feat.sort_values('time')
    df_1m_feat = df_1m_feat.sort_values('time')

    merged = pd.merge_asof(df_1h, df_1d_feat, on='time', direction='backward',
                           tolerance=pd.Timedelta('2 days'))
    merged = pd.merge_asof(merged, df_1w_feat, on='time', direction='backward',
                           tolerance=pd.Timedelta('14 days'))
    merged = pd.merge_asof(merged, df_1m_feat, on='time', direction='backward',
                           tolerance=pd.Timedelta('62 days'))

    return merged



    return merged


# ─────────────────────────────────────────────
# 3. TARGET
# ─────────────────────────────────────────────

def _lift_stop(stop: float, candidate: float, is_long: bool) -> float:
    return max(stop, candidate) if is_long else min(stop, candidate)


def _favorable_touch(price: float, level: float, is_long: bool) -> bool:
    return price >= level if is_long else price <= level


def _adverse_touch(price: float, level: float, is_long: bool) -> bool:
    return price <= level if is_long else price >= level


def _realized_r(entry_price: float, exit_price: float, risk_dist: float, fraction: float, is_long: bool) -> float:
    gross = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    return (gross / (risk_dist + 1e-9)) * fraction


def _fee_r(price: float, risk_dist: float, fraction: float, fee_pct: float) -> float:
    return (price * fee_pct / (risk_dist + 1e-9)) * fraction


def simulate_trade_path_from_arrays(opens: np.ndarray,
                                    highs: np.ndarray,
                                    lows: np.ndarray,
                                    closes: np.ndarray,
                                    times: np.ndarray | None,
                                    entry_idx: int,
                                    direction: str,
                                    risk_dist: float,
                                    tp1_dist: float | None = None,
                                    tp2_dist: float | None = None,
                                    trail_dist: float | None = None,
                                    horizon: int = BARRIER_HORIZON_BARS,
                                    fee_pct: float = EXEC_FEE_PCT,
                                    slippage_bps: float = EXEC_SLIPPAGE_BPS) -> dict:
    """Simulate the exact trade plan used by both training labels and backtest.

    Execution model:
    - Entry on next bar open with slippage.
    - 50% off at TP1 = 1R.
    - Move stop to breakeven after TP1.
    - 25% off at TP2 = 2R.
    - Raise stop to TP1 after TP2.
    - Trail the final 25% using a 1R stop behind close.
    - Force-close any remainder at horizon expiry.
    - Same-bar ambiguities are resolved conservatively against the trade.
    """
    n = len(opens)
    if entry_idx >= n or risk_dist <= 0 or not np.isfinite(risk_dist):
        return {'total_r': np.nan, 'exit_idx': entry_idx, 'exit_time': None, 'exit_reason': 'INVALID'}

    is_long = direction.upper() == 'LONG'
    raw_entry = float(opens[entry_idx])
    if not np.isfinite(raw_entry):
        return {'total_r': np.nan, 'exit_idx': entry_idx, 'exit_time': None, 'exit_reason': 'INVALID_ENTRY'}

    entry_price = raw_entry * (1 + slippage_bps) if is_long else raw_entry * (1 - slippage_bps)
    tp1_dist = risk_dist * TP1_R_MULT if tp1_dist is None else tp1_dist
    tp2_dist = risk_dist * TP2_R_MULT if tp2_dist is None else tp2_dist
    trail_dist = risk_dist * TRAIL_R_MULT if trail_dist is None else trail_dist
    stop = entry_price - risk_dist if is_long else entry_price + risk_dist
    tp1 = entry_price + tp1_dist if is_long else entry_price - tp1_dist
    tp2 = entry_price + tp2_dist if is_long else entry_price - tp2_dist
    remaining = {'tp1': TP1_FRACTION, 'tp2': TP2_FRACTION, 'runner': RUNNER_FRACTION}
    total_r = -_fee_r(entry_price, risk_dist, 1.0, fee_pct)
    tp1_hit = False
    tp2_hit = False
    exit_reason = 'TIME_EXIT'
    exit_idx = min(n - 1, entry_idx + max(horizon - 1, 0))
    last_fill_price = entry_price

    def remaining_total() -> float:
        return float(remaining['tp1'] + remaining['tp2'] + remaining['runner'])

    def fill_fraction(key: str, fill_price: float) -> None:
        nonlocal total_r, last_fill_price
        fraction = remaining[key]
        if fraction <= 0:
            return
        total_r += _realized_r(entry_price, fill_price, risk_dist, fraction, is_long)
        total_r -= _fee_r(fill_price, risk_dist, fraction, fee_pct)
        remaining[key] = 0.0
        last_fill_price = fill_price

    def fill_all_remaining(fill_price: float) -> None:
        nonlocal total_r, last_fill_price
        leftover = remaining_total()
        if leftover <= 0:
            return
        total_r += _realized_r(entry_price, fill_price, risk_dist, leftover, is_long)
        total_r -= _fee_r(fill_price, risk_dist, leftover, fee_pct)
        remaining['tp1'] = 0.0
        remaining['tp2'] = 0.0
        remaining['runner'] = 0.0
        last_fill_price = fill_price

    last_idx = min(n - 1, entry_idx + max(horizon - 1, 0))
    for j in range(entry_idx, last_idx + 1):
        bar_open = float(opens[j])
        bar_high = float(highs[j])
        bar_low = float(lows[j])
        bar_close = float(closes[j])
        if not np.isfinite(bar_open) or not np.isfinite(bar_high) or not np.isfinite(bar_low) or not np.isfinite(bar_close):
            exit_reason = 'BAD_BAR'
            exit_idx = j
            break

        if _adverse_touch(bar_open, stop, is_long):
            fill_all_remaining(bar_open)
            exit_reason = 'SL_GAP'
            exit_idx = j
            break

        if remaining['tp1'] > 0 and _favorable_touch(bar_open, tp1, is_long):
            fill_fraction('tp1', tp1)
            tp1_hit = True
            stop = _lift_stop(stop, entry_price, is_long)

        if remaining['tp2'] > 0 and _favorable_touch(bar_open, tp2, is_long):
            fill_fraction('tp2', tp2)
            tp2_hit = True
            stop = _lift_stop(stop, tp1, is_long)

        if remaining_total() <= 0:
            exit_reason = 'TARGETS_GAP_FILLED'
            exit_idx = j
            break

        bar_best = bar_high if is_long else bar_low
        bar_worst = bar_low if is_long else bar_high
        favorable_same_bar = (
            (remaining['tp1'] > 0 and _favorable_touch(bar_best, tp1, is_long)) or
            (remaining['tp2'] > 0 and _favorable_touch(bar_best, tp2, is_long))
        )
        adverse_same_bar = _adverse_touch(bar_worst, stop, is_long)

        if adverse_same_bar and favorable_same_bar:
            fill_all_remaining(stop)
            exit_reason = 'AMBIGUOUS_BAR_SL'
            exit_idx = j
            break
        if adverse_same_bar:
            fill_all_remaining(stop)
            exit_reason = 'SL_HIT'
            exit_idx = j
            break

        if remaining['tp1'] > 0 and _favorable_touch(bar_best, tp1, is_long):
            fill_fraction('tp1', tp1)
            tp1_hit = True
            stop = _lift_stop(stop, entry_price, is_long)
            if remaining_total() <= 0:
                exit_reason = 'TP1_FILLED'
                exit_idx = j
                break
            if _adverse_touch(bar_worst, stop, is_long):
                fill_all_remaining(stop)
                exit_reason = 'POST_TP1_STOP'
                exit_idx = j
                break

        if remaining['tp2'] > 0 and _favorable_touch(bar_best, tp2, is_long):
            fill_fraction('tp2', tp2)
            tp2_hit = True
            stop = _lift_stop(stop, tp1, is_long)
            if remaining_total() <= 0:
                exit_reason = 'TP2_FILLED'
                exit_idx = j
                break
            if _adverse_touch(bar_worst, stop, is_long):
                fill_all_remaining(stop)
                exit_reason = 'POST_TP2_STOP'
                exit_idx = j
                break

        if remaining['runner'] > 0 and tp1_hit:
            trail_base = tp1 if tp2_hit else entry_price
            trail_candidate = bar_close - trail_dist if is_long else bar_close + trail_dist
            stop = _lift_stop(stop, trail_base, is_long)
            stop = _lift_stop(stop, trail_candidate, is_long)

    else:
        fill_all_remaining(float(closes[last_idx]))
        exit_reason = 'TIME_EXIT'
        exit_idx = last_idx

    if remaining_total() > 0 and exit_reason in {'BAD_BAR'}:
        fill_all_remaining(float(closes[min(exit_idx, n - 1)]))

    exit_time = None if times is None else times[min(exit_idx, n - 1)]
    return {
        'entry_idx': entry_idx,
        'entry_price': entry_price,
        'exit_price': float(last_fill_price),
        'exit_idx': min(exit_idx, n - 1),
        'exit_time': exit_time,
        'exit_reason': exit_reason,
        'total_r': float(total_r),
        'tp1_hit': tp1_hit,
        'tp2_hit': tp2_hit,
        'final_stop': float(stop)
    }


def add_target(df: pd.DataFrame,
               atr_mult: float = BARRIER_ATR_MULT,
               horizon: int = BARRIER_HORIZON_BARS,
               atr_col: str = 'atr14',
               drop_unresolved: bool = True) -> pd.DataFrame:
    """
    Target: whichever direction would have produced the better executed trade.

    The long and short paths are both simulated using the exact trading plan
    (TP1, TP2, breakeven move, trailing stop, fees/slippage, and max-hold).
    Training keeps only decisive bars where one side has a meaningful edge.
    """
    df = df.copy()
    n = len(df)
    opens = df['open'].to_numpy(dtype=float)
    highs = df['high'].to_numpy(dtype=float)
    lows = df['low'].to_numpy(dtype=float)
    closes = df['close'].to_numpy(dtype=float)
    times = df['time'].to_numpy() if 'time' in df.columns else None
    atrs = df[atr_col].to_numpy(dtype=float)

    target = np.full(n, np.nan, dtype=float)
    next_ret_pct = np.full(n, np.nan, dtype=float)
    bars_to_target = np.full(n, np.nan, dtype=float)
    entry_prices = np.full(n, np.nan, dtype=float)
    target_distances = np.full(n, np.nan, dtype=float)
    long_path_r = np.full(n, np.nan, dtype=float)
    short_path_r = np.full(n, np.nan, dtype=float)
    target_edge_r = np.full(n, np.nan, dtype=float)
    best_path_r = np.full(n, np.nan, dtype=float)
    long_mfe_atr = np.full(n, np.nan, dtype=float)
    long_mae_atr = np.full(n, np.nan, dtype=float)
    short_mfe_atr = np.full(n, np.nan, dtype=float)
    short_mae_atr = np.full(n, np.nan, dtype=float)

    last_start_idx = max(-1, n - horizon - 1)
    for i in range(last_start_idx + 1):
        entry_idx = i + 1
        dist = atrs[i] * atr_mult
        if not np.isfinite(dist) or dist <= 0:
            continue

        long_trade = simulate_trade_path_from_arrays(
            opens, highs, lows, closes, times, entry_idx, 'LONG', dist, horizon=horizon
        )
        short_trade = simulate_trade_path_from_arrays(
            opens, highs, lows, closes, times, entry_idx, 'SHORT', dist, horizon=horizon
        )

        long_r = long_trade['total_r']
        short_r = short_trade['total_r']
        if not np.isfinite(long_r) or not np.isfinite(short_r):
            continue

        long_path_r[i] = long_r
        short_path_r[i] = short_r
        best_path_r[i] = max(long_r, short_r)
        target_edge_r[i] = abs(long_r - short_r)
        entry_prices[i] = long_trade['entry_price']
        target_distances[i] = dist
        horizon_end = min(n, entry_idx + horizon)
        if horizon_end > entry_idx and atrs[i] > 0:
            window_high = np.nanmax(highs[entry_idx:horizon_end])
            window_low = np.nanmin(lows[entry_idx:horizon_end])
            entry_price = entry_prices[i]
            long_mfe_atr[i] = max(0.0, (window_high - entry_price) / (atrs[i] + 1e-9))
            long_mae_atr[i] = max(0.0, (entry_price - window_low) / (atrs[i] + 1e-9))
            short_mfe_atr[i] = max(0.0, (entry_price - window_low) / (atrs[i] + 1e-9))
            short_mae_atr[i] = max(0.0, (window_high - entry_price) / (atrs[i] + 1e-9))

        if long_r >= short_r + MIN_LABEL_EDGE_R and long_r >= MIN_LABEL_BEST_R:
            target[i] = 1.0
            next_ret_pct[i] = (long_r * dist / (entry_prices[i] + 1e-9)) * 100.0
            bars_to_target[i] = long_trade['exit_idx'] - i
        elif short_r >= long_r + MIN_LABEL_EDGE_R and short_r >= MIN_LABEL_BEST_R:
            target[i] = 0.0
            next_ret_pct[i] = -(short_r * dist / (entry_prices[i] + 1e-9)) * 100.0
            bars_to_target[i] = short_trade['exit_idx'] - i

    df['target'] = target
    df['next_ret_pct'] = next_ret_pct
    df['bars_to_target'] = bars_to_target
    df['entry_price_next_bar'] = entry_prices
    df['target_distance'] = target_distances
    df['long_path_r'] = long_path_r
    df['short_path_r'] = short_path_r
    df['target_edge_r'] = target_edge_r
    df['best_path_r'] = best_path_r
    df['long_mfe_atr'] = long_mfe_atr
    df['long_mae_atr'] = long_mae_atr
    df['short_mfe_atr'] = short_mfe_atr
    df['short_mae_atr'] = short_mae_atr

    if drop_unresolved:
        df = df.dropna(subset=['target'])
        df['target'] = df['target'].astype(int)
    else:
        df['target'] = df['target'].astype('Int64')
    return df


# ─────────────────────────────────────────────
# 4. WALK-FORWARD VALIDATION (Optimized for CPU)
# ─────────────────────────────────────────────

def _build_lgbm_classifier() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=31,
        max_depth=5,
        min_child_samples=40,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.15,
        reg_lambda=0.15,
        random_state=42,
        n_jobs=MODEL_N_JOBS,
        verbose=-1,
        objective='binary',
        metric='logloss'
    )


def _build_lgbm_regressor(alpha: float) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=40,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_alpha=0.10,
        reg_lambda=0.10,
        random_state=42,
        n_jobs=MODEL_N_JOBS,
        verbose=-1,
        objective='quantile',
        alpha=alpha
    )


def _fit_lgbm_with_inner_validation(X_train_full: pd.DataFrame,
                                    y_train_full: pd.Series) -> tuple[lgb.LGBMClassifier, pd.DataFrame, pd.Series]:
    inner_val_size = max(100, int(len(X_train_full) * 0.15))
    inner_val_size = min(inner_val_size, len(X_train_full) - 1)
    if inner_val_size <= 0:
        raise ValueError("Not enough training rows for inner validation.")

    X_train_inner = X_train_full.iloc[:-inner_val_size]
    y_train_inner = y_train_full.iloc[:-inner_val_size]
    X_val_inner   = X_train_full.iloc[-inner_val_size:]
    y_val_inner   = y_train_full.iloc[-inner_val_size:]

    model = _build_lgbm_classifier()
    model.fit(
        X_train_inner, y_train_inner,
        eval_set=[(X_val_inner, y_val_inner)],
        eval_metric='logloss',
        callbacks=[
            lgb.early_stopping(stopping_rounds=60, verbose=False),
            lgb.log_evaluation(period=10000)
        ]
    )
    return model, X_val_inner, y_val_inner


def train_trade_plan_models(df: pd.DataFrame, feature_cols: list) -> dict:
    specs = {
        'up_stop_atr':   (1, 'long_mae_atr', 0.80),
        'up_tp1_atr':    (1, 'long_mfe_atr', 0.50),
        'up_tp2_atr':    (1, 'long_mfe_atr', 0.80),
        'down_stop_atr': (0, 'short_mae_atr', 0.80),
        'down_tp1_atr':  (0, 'short_mfe_atr', 0.50),
        'down_tp2_atr':  (0, 'short_mfe_atr', 0.80),
    }
    models = {}
    for key, (target_value, label_col, alpha) in specs.items():
        train_df = df[df['target'] == target_value].dropna(subset=feature_cols + [label_col]).copy()
        if len(train_df) < 300:
            continue
        model = _build_lgbm_regressor(alpha)
        model.fit(train_df[feature_cols], train_df[label_col])
        models[key] = model
    return models


def predict_trade_plan(plan_models: dict,
                       feature_cols: list,
                       latest_row: pd.Series,
                       direction: str,
                       atr_value: float) -> dict:
    fallback_r = BARRIER_ATR_MULT
    if direction not in {'UP', 'DOWN'} or atr_value <= 0:
        return {
            'sl': np.nan, 'tp1': np.nan, 'tp2': np.nan,
            'stop_atr': np.nan, 'tp1_atr': np.nan, 'tp2_atr': np.nan,
            'trail_r': np.nan, 'note': 'No directional trade plan available.'
        }

    prefix = 'up' if direction == 'UP' else 'down'
    required = [f'{prefix}_stop_atr', f'{prefix}_tp1_atr', f'{prefix}_tp2_atr']
    missing_cols = [col for col in feature_cols if col not in latest_row.index]
    if missing_cols:
        for col in missing_cols:
            latest_row[col] = 0.0
    X = latest_row[feature_cols].values.reshape(1, -1)

    if all(key in plan_models for key in required):
        stop_atr = float(np.clip(plan_models[required[0]].predict(X)[0], 0.35, 4.00))
        tp1_atr = float(np.clip(plan_models[required[1]].predict(X)[0], 0.25, 6.00))
        tp2_atr = float(np.clip(plan_models[required[2]].predict(X)[0], 0.50, 8.00))
        tp1_atr = max(tp1_atr, stop_atr * 0.80)
        tp2_atr = max(tp2_atr, tp1_atr + 0.25)
        trail_r = float(np.clip(stop_atr, 0.50, 3.00))
        note = 'ML-derived levels from quantile excursion models.'
    else:
        stop_atr = fallback_r
        tp1_atr = fallback_r
        tp2_atr = fallback_r * TP2_R_MULT
        trail_r = TRAIL_R_MULT
        note = 'Fallback ATR plan: excursion models unavailable.'

    close_price = float(latest_row['close'])
    if direction == 'UP':
        sl = close_price - (atr_value * stop_atr)
        tp1 = close_price + (atr_value * tp1_atr)
        tp2 = close_price + (atr_value * tp2_atr)
    else:
        sl = close_price + (atr_value * stop_atr)
        tp1 = close_price - (atr_value * tp1_atr)
        tp2 = close_price - (atr_value * tp2_atr)

    return {
        'sl': sl,
        'tp1': tp1,
        'tp2': tp2,
        'stop_atr': stop_atr,
        'tp1_atr': tp1_atr,
        'tp2_atr': tp2_atr,
        'trail_r': trail_r,
        'note': note
    }


def walk_forward(df: pd.DataFrame,
                 feature_cols: list,
                 n_splits: int = 10,
                 min_train_bars: int = 2000,
                 test_size_ratio: float = 0.15,
                 purge_gap: int = 24
                 ) -> dict:
    """
    Pure walk-forward: train on past, test on strictly future window.
    Early-stopping uses an inner validation slice carved from the training
    window only — the test fold is NEVER visible during training.
    OOS probabilities are accumulated in oos_proba_map for honest backtesting.
    """
    df = df.dropna(subset=feature_cols + ['target']).reset_index(drop=True)
    n = len(df)
    
    # Dynamic window size calculation based on ratio
    if n < min_train_bars:
        print(f"  Error: Not enough data for walk-forward. Need {min_train_bars}, got {n}.")
        return {}

    # Calculate split points
    split_points = []
    current_train_end = min_train_bars
    while current_train_end < n:
        test_window_size = max(int(n * test_size_ratio), 100)
        test_end = min(current_train_end + test_window_size, n)
        
        if test_end <= current_train_end:
            break
        
        # DEGENERATE SPLIT GUARD: skip splits with fewer than 50 test bars
        if (test_end - current_train_end) < 50:
            break
        
        split_points.append((current_train_end, test_end))
        current_train_end = test_end
        
        if len(split_points) >= n_splits:
            break

    if not split_points:
        print("  Error: Could not determine valid split points for walk-forward.")
        return {}

    print(f"\n  Walk-forward: {len(split_points)} splits, "
          f"min_train_bars={min_train_bars}, test_ratio={test_size_ratio:.2f}")
    print(f"  Total bars: {n}")

    feature_cols = list(feature_cols)

    results = []
    feature_importance_sum = np.zeros(len(feature_cols))
    all_test_preds = []
    all_test_trues = []
    # UME-2 FIX: Genuine OOS probability map keyed by pd.Timestamp.
    # The backtest engine MUST use these — never batch-predict on this model.
    oos_proba_map: dict = {}

    for i, (train_end, test_end) in enumerate(split_points):
        # PURGE GAP: skip `purge_gap` bars between train and test to break autocorrelation
        purged_train_end = max(0, train_end - purge_gap)
        X_train_full = df[feature_cols].iloc[:purged_train_end]
        y_train_full = df['target'].iloc[:purged_train_end]
        X_test       = df[feature_cols].iloc[train_end:test_end]
        y_test       = df['target'].iloc[train_end:test_end]
        
        if len(X_test) == 0:
            continue

        # ── UME-1 FIX: Inner validation from TRAINING window only ──
        # Test fold is never seen by eval_set — no lookahead pollution.
        model, X_val_inner, y_val_inner = _fit_lgbm_with_inner_validation(X_train_full, y_train_full)

        # M4 FIX: Isotonic calibration on inner validation set
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.frozen import FrozenEstimator
        calibrated_model = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
        calibrated_model.fit(X_val_inner, y_val_inner)

        preds = calibrated_model.predict(X_test)
        proba = calibrated_model.predict_proba(X_test)[:, 1]
        acc   = accuracy_score(y_test, preds)

        # ── UME-2 FIX: Record every OOS prediction by its bar's timestamp ──
        if 'time' in df.columns:
            test_times = df['time'].iloc[train_end:test_end].values
            for ts, prob in zip(test_times, proba):
                oos_proba_map[pd.Timestamp(ts)] = float(prob)

        # High-confidence subset analysis
        conf_mask = np.maximum(proba, 1.0 - proba) >= LIVE_CONFIDENCE_THRESHOLD
        
        acc_conf  = np.nan
        high_conf_bars_count = conf_mask.sum()
        if high_conf_bars_count > 10:
            acc_conf  = accuracy_score(y_test[conf_mask], preds[conf_mask])
        
        results.append({
            'split': i + 1,
            'train_bars': train_end,
            'test_bars': len(X_test),
            'accuracy': acc,
            'acc_high_conf': acc_conf,
            'high_conf_pct': conf_mask.mean() if high_conf_bars_count > 0 else 0.0,
            'baseline_up': y_test.mean(),
            'baseline_down': 1 - y_test.mean(),
            'proba_mean': proba.mean(),
            'proba_std': proba.std(),
            'high_conf_bars': high_conf_bars_count
        })
        
        feature_importance_sum += model.feature_importances_
        all_test_preds.extend(preds)
        all_test_trues.extend(y_test)
        
        conf_str = f"{acc_conf:.3f}" if not np.isnan(acc_conf) else " n/a"
        print(f"  Split {i+1:2d} | Train:{train_end:>6} | Test:{len(X_test):>5} | "
              f"Acc:{acc:.3f} | ConfAcc(>={LIVE_CONFIDENCE_THRESHOLD:.2f}):{conf_str} | "
              f"ConfBars:{conf_mask.mean()*100:5.1f}% | Baseline(UP):{y_test.mean():.3f}")

    # ── UME-2 FIX: Final model for LIVE INFERENCE ONLY ──
    # Trained on all history so live predictions use the fullest possible signal.
    # Any backtest MUST use oos_proba_map — never batch-predict from this model on historical data.
    final_model = _build_lgbm_classifier()
    final_model.fit(df[feature_cols], df['target']) # Train on all available history

    # Overall accuracy calculation
    overall_acc = accuracy_score(all_test_trues, all_test_preds)
    overall_baseline_up = np.mean(all_test_trues)

    return {
        'splits': results,
        'feature_importance': dict(zip(feature_cols, feature_importance_sum / len(split_points))),
        'final_model': final_model,
        'feature_cols': feature_cols,
        'df': df,
        'overall_accuracy': overall_acc,
        'overall_baseline_up': overall_baseline_up,
        'oos_proba_map': oos_proba_map  # UME-2: ONLY valid source for honest backtesting
    }


# ─────────────────────────────────────────────
# 5. REPORT CHART (Visual Enhancements)
# ─────────────────────────────────────────────

def save_report(wf_results: dict, save_path: str, pred_info: dict = None, symbol: str = 'UNKNOWN'):
    if not wf_results or 'splits' not in wf_results or not wf_results['splits']:
        print("  No walk-forward results to plot.")
        return
        
    splits = pd.DataFrame(wf_results['splits'])
    fi     = wf_results['feature_importance']
    
    # Add overall metrics if available
    if 'overall_accuracy' in wf_results:
        splits_summary = splits.mean(numeric_only=True)
        splits_summary['accuracy'] = wf_results['overall_accuracy']
        splits_summary['acc_high_conf'] = splits['acc_high_conf'].mean() # Avg of high_conf acc across splits
        splits_summary['high_conf_pct'] = splits['high_conf_pct'].mean()
        splits_summary['baseline_up'] = wf_results['overall_baseline_up']
    else:
        splits_summary = splits.mean(numeric_only=True)

    fig = plt.figure(figsize=(14, 16), facecolor='#0d0d0d')
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4, height_ratios=[1, 1.5, 1.5]) 

    ax1 = fig.add_subplot(gs[0, 0]) # Top row: TRADER TEXT FORECAST
    ax2 = fig.add_subplot(gs[1, 0]) # Middle row: CODER ACCURACY
    ax3 = fig.add_subplot(gs[2, 0]) # Bottom row: CODER FEATURES (Top 15)

    # Color theme
    dark_bg_color = '#0d0d0d'
    light_text_color = '#e0e0e0'
    axis_line_color = '#444444'
    accent_color_1 = '#00d4ff' # Cyan
    accent_color_2 = '#00ff88' # Green
    accent_color_3 = '#ff6b6b' # Red
    accent_color_5 = '#7c4dff' # Purple

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(dark_bg_color)
        ax.tick_params(colors=light_text_color, labelsize=9)
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_color(axis_line_color)

    # — 1. TRADER FORECAST (TEXT PANEL) —
    ax1.axis('off') # Hide axes for text panel
    if pred_info:
        bg_color = accent_color_2 if pred_info['dir'] == 'UP' else accent_color_3
        
        # Get dynamic symbol from pred_info or default to BANKNIFTY
        symbol_display = pred_info.get('symbol', 'BANKNIFTY').upper()
        note_line = f"  Filter Note : {pred_info['note']}\n" if pred_info.get('note') else ""
        
        report_text = (
            f"======================================================================\n"
            f"  {symbol_display} ULTIMATE FORECAST (EXECUTION-ALIGNED)\n"
            f"======================================================================\n"
            f"  Bar time    : {pred_info['time']}\n"
            f"  Direction   : {pred_info['dir']}\n"
            f"  Confidence  : {pred_info['conf']}\n"
            f"  Signal      : {pred_info['signal']}\n"
            f"----------------------------------------------------------------------\n"
            f"  Entry Price : {pred_info['entry']}\n"
            f"  Stop Loss   : {pred_info['sl']}\n"
            f"  Target 1    : {pred_info['tp1']}\n"
            f"  Target 2    : {pred_info['tp2']}\n"
            f"  Trail Stop  : {pred_info['trail']}\n"
            f"{note_line}"
            f"======================================================================"
        )
        ax1.text(0.5, 0.5, report_text, color=bg_color, fontsize=16, 
                 fontfamily='monospace', fontweight='bold', ha='center', va='center',
                 bbox=dict(facecolor='#1a1a1a', edgecolor=bg_color, pad=2.0, boxstyle='round'))

    # — 2. CODER ACCURACY PLOT —
    x = splits['split']
    ax2.plot(x, splits['accuracy'],     'o-', color=accent_color_1, lw=2, label='Split Accuracy')
    ax2.plot(x, splits['acc_high_conf'],'s--',color=accent_color_2, lw=2, label=f'High-Confidence Accuracy (>={LIVE_CONFIDENCE_THRESHOLD:.2f})')
    ax2.axhline(splits_summary['baseline_up'], color=accent_color_3, ls=':', lw=1.5, label=f"Always-UP baseline ({splits_summary['baseline_up']:.2f})")
    ax2.axhline(0.5, color='#777777', ls=':', lw=1)
    
    # Overall accuracy line
    ax2.axhline(splits_summary['accuracy'], color=accent_color_1, ls='--', lw=1, label=f'Overall Acc ({splits_summary["accuracy"]:.3f})')
    ax2.fill_between(x, 0.5, splits['accuracy'], alpha=0.1, color=accent_color_1)
    
    ax2.set_title('Walk-Forward Accuracy (Out-of-Sample)', color=light_text_color, fontsize=14, fontweight='bold')
    ax2.set_xlabel('Split (Timeline)', color=light_text_color)
    ax2.set_ylabel('Accuracy', color=light_text_color)
    ax2.set_ylim(0.4, 0.85) 
    ax2.legend(facecolor=dark_bg_color, edgecolor=axis_line_color, labelcolor=light_text_color, fontsize=10, loc='lower right')

    # — 3. CODER TOP 15 FEATURES —
    fi_filtered = {k: v for k, v in fi.items() if v > 0}
    top15 = sorted(fi_filtered.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if top15:
        fnames, fvals = zip(*top15)
        fnames = [str(n).replace('_', ' ').upper() for n in fnames]
        
        bars = ax3.barh(range(len(fnames)), fvals, color=accent_color_5, alpha=0.85)
        ax3.set_yticks(range(len(fnames)))
        ax3.set_yticklabels(fnames, color='#cccccc', fontsize=10)
        ax3.set_title('Top 15 Most Predictive Features', color=light_text_color, fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.set_xlabel('Importance Score', color=light_text_color)
    else:
        ax3.text(0.5, 0.5, "No significant features", color=light_text_color, ha='center', va='center')

    fig.suptitle(f'{(pred_info or {}).get("symbol", symbol).upper()} — ML Performance Report', color='white', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"\n  Report saved → {save_path}")


# ─────────────────────────────────────────────
# 6. PREDICTION FUNCTION (live use)
# ─────────────────────────────────────────────

def predict_next_bar(model, feature_cols: list, latest_row: pd.Series, confidence_threshold: float = LIVE_CONFIDENCE_THRESHOLD) -> dict:
    """
    Pass in the latest fully-formed 1H row (with all features computed).
    Returns direction prediction + confidence.
    """
    if model is None or not feature_cols or latest_row.empty:
        return {
            'direction': 'ERROR',
            'prob_up': np.nan,
            'prob_down': np.nan,
            'confidence': np.nan,
            'signal_strength': 'ERROR',
            'message': 'Model or data not properly loaded.'
        }

    # Ensure all feature_cols are present in latest_row, fill missing with 0 if necessary (though usually they should be computed)
    missing_cols = [col for col in feature_cols if col not in latest_row.index]
    if missing_cols:
        print(f"  Warning: Missing columns in latest_row, filling with 0: {missing_cols}")
        for col in missing_cols:
            latest_row[col] = 0.0

    X = latest_row[feature_cols].values.reshape(1, -1)
    
    try:
        proba = model.predict_proba(X)[0][1] # Probability of class 1 (UP)
    except Exception as e:
        return {
            'direction': 'ERROR',
            'prob_up': np.nan,
            'prob_down': np.nan,
            'confidence': np.nan,
            'signal_strength': 'ERROR',
            'message': f'Error during prediction: {e}'
        }
        
    direction = 'UP' if proba > 0.5 else 'DOWN'
    prob_up = round(proba, 4)
    prob_down = round(1 - proba, 4)
    confidence = max(prob_up, prob_down)
    
    volatility_ok = bool(latest_row.get('atr_expanding', 1))
    if confidence < confidence_threshold or not volatility_ok:
        signal = 'NO_TRADE'
    else:
        signal = 'STRONG'
    
    return {
        'direction': direction,
        'prob_up': prob_up,
        'prob_down': prob_down,
        'confidence': confidence,
        'signal_strength': signal,
        'message': 'Prediction successful.'
    }


# ─────────────────────────────────────────────
# 7. MAIN EXECUTION (Optimized for performance)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Universal ML Direction Predictor")
    parser.add_argument('--outdir', type=str, default='/home/km/Universal-ML/', help="Output directory for models and reports")

    args = parser.parse_args()

    DATA_DIR = args.outdir
    CSV_DIR = os.path.join(DATA_DIR, 'csv_data')
    os.makedirs(CSV_DIR, exist_ok=True)
    
    print("=" * 70)
    print("  AUTO-SCANNING CSV FOLDER (csv_data/) FOR SYMBOL AND TIMEFRAMES")
    print("=" * 70)
    
    csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print(f"  [!] No CSV files found in {CSV_DIR}")
        print("      Please drop your 3 CSV files (1H, 1D, 1W) into this folder, no specific naming required.")
        exit()

    df_1h, df_1d, df_1w, df_1m = None, None, None, None
    SYMBOL = "UNKNOWN"

    for ffile in csv_files:
        path = os.path.join(CSV_DIR, ffile)
        df, sym, tf = parse_tv_log(path)
        tf = str(tf).upper()
        
        if sym and sym != "UNKNOWN":
            SYMBOL = sym.replace('!', '') # Output clean symbol without !

        if tf in ['60', '1H']:
            df_1h = df
            print(f"  [+] Detected 1H File: {ffile}")
        elif tf in ['1D', 'D']:
            df_1d = df
            print(f"  [+] Detected 1D File: {ffile}")
        elif tf in ['1W', 'W']:
            df_1w = df
            print(f"  [+] Detected 1W File: {ffile}")
        elif tf in ['1M', 'M']:
            df_1m = df
            print(f"  [+] Detected 1M File: {ffile}")

    if df_1h is None or df_1d is None or df_1w is None or df_1m is None:
        print(f"\n  [!] Error: Could not find all 4 timeframes (60, 1D, 1W, 1M) from inside you CSVs.")
        print(f"      Make sure the actual text in the CSVs say 'TIME FRAME:60', 'TIME FRAME:1D', 'TIME FRAME:1W', and 'TIME FRAME:1M'")
        exit()

    file_prefix = SYMBOL.lower().replace(' ', '_')

    print(f"  1H bars : {len(df_1h):>7}  ({df_1h['time'].min().date()} → {df_1h['time'].max().date()})")
    print(f"  1D bars : {len(df_1d):>7}  ({df_1d['time'].min().date()} → {df_1d['time'].max().date()})")
    print(f"  1W bars : {len(df_1w):>7}  ({df_1w['time'].min().date()} → {df_1w['time'].max().date()})")
    print(f"  1M bars : {len(df_1m):>7}  ({df_1m['time'].min().date()} → {df_1m['time'].max().date()})")
    
    print("\n  [TOON v4.0] Building Holographic Feature Engine...")
    print("  Philosophy: Pure geometry. No classical indicators.")

    # ── Step 1: Compute atr14 on 1H ONLY for labelling scaffold ──────────
    # atr14 is needed by add_target() to set barrier distances.
    # It is NEVER passed as a model input feature.
    df_1h_labelled = _compute_atr14(df_1h.copy())

    # ── Step 2: Run holographic engine on all 4 timeframes ────────────────
    # Each timeframe processed independently. Look-ahead rule enforced inside.
    df_full = holographic_feature_engine(
        df_1h_labelled,
        df_1d=df_1d,
        df_1w=df_1w,
        df_1m=df_1m,
    )

    # ── Step 3: ASOF-merge higher-TF timestamps for temporal alignment ────
    # merge_higher_tf now only brings in raw OHLCV columns with the look-ahead
    # shift. The holographic engine has already extracted all shape features.
    print("  [TOON v4.0] Merging higher-TF timestamp anchors...")
    df_full = merge_higher_tf(df_full, df_1d, df_1w, df_1m)

    # ── Step 4: Add target using atr14 (labelling scaffold) ───────────────
    print("  [TOON v4.0] Labelling targets via trade simulation...")
    df_full = add_target(
        df_full,
        atr_mult=BARRIER_ATR_MULT,
        horizon=BARRIER_HORIZON_BARS,
        atr_col='atr14',
        drop_unresolved=True,
    )

    # ── Step 5: Identify holographic feature columns ───────────────────────
    # Exclude: raw OHLCV, time, labelling scaffold, target and trade-plan cols.
    # atr14 is excluded here — it was labelling scaffolding only.
    NON_FEATURE_COLS = {
        'time', 'open', 'high', 'low', 'close', 'volume',
        'atr14',                                           # labelling only
        'target', 'next_ret_pct', 'bars_to_target',
        'entry_price_next_bar', 'target_distance',
        'long_path_r', 'short_path_r', 'target_edge_r', 'best_path_r',
        'long_mfe_atr', 'long_mae_atr', 'short_mfe_atr', 'short_mae_atr',
        # raw higher-TF OHLCV from merge (not holographic)
        'd_open', 'd_high', 'd_low', 'd_close', 'd_volume',
        'w_open', 'w_high', 'w_low', 'w_close', 'w_volume',
        'm_open', 'm_high', 'm_low', 'm_close', 'm_volume',
    }
    all_holo_cols = [c for c in df_full.columns if c not in NON_FEATURE_COLS]

    # Sanitize: replace inf/nan
    df_model_ready = df_full[all_holo_cols + ['target', 'time', 'close']
                             + [c for c in TRADE_PLAN_LABEL_COLS if c in df_full.columns]].copy()
    for col in all_holo_cols:
        df_model_ready[col] = df_model_ready[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    df_model_ready = df_model_ready.dropna(subset=['target'] + all_holo_cols).reset_index(drop=True)

    print(f"  [TOON v4.0] Bars after labelling & cleaning : {len(df_model_ready)}")
    print(f"  [TOON v4.0] Holographic features before selection : {len(all_holo_cols)}")
    print(f"  [TOON v4.0] Target (UP=1) distribution : {df_model_ready['target'].mean():.1%}")

    # ── Step 6: Feature selection pipeline (1800 → 40) ────────────────────
    MIN_TRAIN_BARS  = 2500
    TEST_SIZE_RATIO = 0.15

    n_available = len(df_model_ready)
    if n_available < MIN_TRAIN_BARS + 100:
        print(f"\n  Error: Not enough {SYMBOL} data for walk-forward validation.")
        exit()

    feature_cols, sel_meta = feature_selection_pipeline(
        df_model_ready,
        all_holo_cols,
        walk_forward_fn=walk_forward,
        target_col='target',
        min_train_bars=MIN_TRAIN_BARS,
        test_size_ratio=TEST_SIZE_RATIO,
        n_splits=10,
    )

    print(f"\n  [TOON v4.0] Final feature count into walk_forward : {len(feature_cols)}")

    # ── Step 7: Final walk-forward validation ─────────────────────────────
    print("\n" + "=" * 70)
    print("  Walk-Forward Validation — TOON v4.0 Holographic Model")
    print("=" * 70)

    wf_results = walk_forward(
        df_model_ready,
        feature_cols,
        n_splits=10,
        min_train_bars=MIN_TRAIN_BARS,
        test_size_ratio=TEST_SIZE_RATIO,
    )

    if wf_results:
        splits_df = pd.DataFrame(wf_results['splits'])
        print("\n" + "=" * 70)
        print("  SUMMARY — TOON v4.0 HOLOGRAPHIC WALK-FORWARD")
        print("=" * 70)
        print(f"  Overall Accuracy (All Signals)      : {wf_results['overall_accuracy']:.3f}")
        print(f"  Overall High-Conf Accuracy (>={LIVE_CONFIDENCE_THRESHOLD:.2f}) : {splits_df['acc_high_conf'].mean():.3f}")
        print(f"  Average High-Confidence Bar %       : {splits_df['high_conf_pct'].mean():.1%}")
        print(f"  Average Always-UP Baseline          : {wf_results['overall_baseline_up']:.3f}")
        print(f"  Edge over Baseline                  : {wf_results['overall_accuracy'] - wf_results['overall_baseline_up']:+.3f}")
        print(f"  Number of splits performed          : {len(splits_df)}")
        print(f"  Total bars used for validation      : {splits_df['test_bars'].sum()}")

        # Layer breakdown of final features
        final_feats = wf_results['feature_cols']
        dna_n   = sum(1 for c in final_feats if '_bar' in c)
        gram_n  = sum(1 for c in final_feats if '_gram_' in c)
        fft_n   = sum(1 for c in final_feats if '_fft_' in c)
        skel_n  = sum(1 for c in final_feats if '_skel_' in c)
        conf_n  = sum(1 for c in final_feats if c.startswith('mtf_conf'))
        print(f"\n  Feature layer breakdown in final model:")
        print(f"    DNA:  {dna_n:3d}  |  Grammar: {gram_n:3d}  |  Spectral: {fft_n:3d}"
              f"  |  Skeleton: {skel_n:3d}  |  Confluence: {conf_n:3d}")
        if fft_n == 0:
            print("  NOTE: Zero FFT features in final model. Spectral layer is "
                  "provisional — not removed. Re-evaluate after first live run.")

        # Save final model and feature list
        import joblib
        mod_path  = os.path.join(DATA_DIR, f'{file_prefix}_ultimate_model.pkl')
        feat_path = os.path.join(DATA_DIR, f'{file_prefix}_ultimate_features.txt')
        joblib.dump(wf_results['final_model'], mod_path)
        with open(feat_path, 'w') as f:
            for col in wf_results['feature_cols']:
                f.write(f"{col}\n")
        print(f"\n  Final model saved to '{mod_path}'")

        oos_path = os.path.join(DATA_DIR, f'{file_prefix}_oos_proba.pkl')
        joblib.dump(wf_results['oos_proba_map'], oos_path)
        print(f"  OOS proba map saved to '{oos_path}' ({len(wf_results['oos_proba_map'])} bars)")

        model             = wf_results['final_model']
        feature_cols_to_use = wf_results['feature_cols']
        # Train trade plan models ONLY on pre-OOS data (same window as final fold).
        # Training on the full dataset leaks future excursion outcomes into SL/TP sizing.
        _tp_train_end = len(df_model_ready) - int(len(df_model_ready) * TEST_SIZE_RATIO)
        trade_plan_models = train_trade_plan_models(
            df_model_ready.iloc[:_tp_train_end], feature_cols_to_use
        )
        trade_plan_path   = os.path.join(DATA_DIR, f'{file_prefix}_trade_plan_models.pkl')
        joblib.dump(trade_plan_models, trade_plan_path)
        print(f"  Trade-plan models saved to '{trade_plan_path}' ({len(trade_plan_models)} models)")

        last_row   = df_model_ready.iloc[-1]
        pred       = predict_next_bar(model, feature_cols_to_use, last_row,
                                      confidence_threshold=LIVE_CONFIDENCE_THRESHOLD)
        close_price = float(last_row['close'])
        # atr14 excluded from model features; use last known value for trade plan display
        atr = float(df_full['atr14'].iloc[-1]) if 'atr14' in df_full.columns else 150.0
        trade_plan  = predict_trade_plan(trade_plan_models, feature_cols_to_use,
                                         last_row.copy(), pred['direction'], atr)

        print("\n" + "=" * 70)
        print(f"  {SYMBOL.upper()} FORECAST (TOON v4.0 — PURE GEOMETRY)")
        print("=" * 70)
        if 'time' in last_row:
            print(f"  Bar time    : {last_row['time']}")
        print(f"  Direction   : {pred['direction']}")
        print(f"  Confidence  : {pred['confidence']*100:.1f}%")
        print(f"  Signal      : {pred['signal_strength']}")
        print("----------------------------------------------------------------------")
        print(f"  Entry Price : {close_price:,.2f} (Current Close)")

        trail_str   = "N/A"
        filter_note = trade_plan['note']
        if pred['direction'] in {"UP", "DOWN"} and np.isfinite(trade_plan['sl']):
            sl_str    = f"{trade_plan['sl']:,.2f}  (ML stop {trade_plan['stop_atr']:.2f}x ATR14)"
            tp1_str   = f"{trade_plan['tp1']:,.2f}  (ML TP1 {trade_plan['tp1_atr']:.2f}x ATR14)"
            tp2_str   = f"{trade_plan['tp2']:,.2f}  (ML TP2 {trade_plan['tp2_atr']:.2f}x ATR14)"
            trail_str = f"{trade_plan['trail_r']:.2f}R trailing stop after TP1"
        else:
            sl_str, tp1_str, tp2_str = "N/A", "N/A", "N/A"

        if pred['signal_strength'] == 'NO_TRADE':
            filter_note = (f"{filter_note} Filtered: confidence below "
                           f"{LIVE_CONFIDENCE_THRESHOLD:.2f}").strip()
        if pred['direction'] in {"UP", "DOWN"}:
            print(f"  Stop Loss   : {sl_str}")
            print(f"  Target 1    : {tp1_str}")
            print(f"  Target 2    : {tp2_str}")
            print(f"  Trail Stop  : {trail_str}")
            if filter_note:
                print(f"  [{filter_note}]")
        else:
            sl_str = tp1_str = tp2_str = trail_str = "N/A"
            print("  [No clear directional edge, targets N/A]")

        print("======================================================================")

        pred_info = {
            'symbol': SYMBOL,
            'time':   str(last_row.get('time', 'N/A')),
            'dir':    pred['direction'],
            'conf':   f"{pred['confidence']*100:.1f}%",
            'signal': pred['signal_strength'],
            'entry':  f"{close_price:,.2f} (Current Close)",
            'sl': sl_str, 'tp1': tp1_str, 'tp2': tp2_str,
            'trail': trail_str, 'note': filter_note,
        }

        report_file = os.path.join(DATA_DIR, f'{file_prefix}_ml_report_ultimate.png')
        save_report(wf_results, report_file, pred_info=pred_info)

        fi_sorted = sorted(wf_results['feature_importance'].items(), key=lambda x: x[1], reverse=True)
        print("\n  Top 15 most predictive holographic features:")
        for i, (fname, fval) in enumerate(fi_sorted[:15]):
            print(f"    {i+1}. {fname:<42} {fval:.2f}")

        backtest_script = os.path.join(os.path.dirname(__file__), 'backtest_engine.py')
        if os.path.exists(backtest_script):
            print("\n  Launching aligned backtest report generation...")
            completed = subprocess.run(
                [sys.executable, backtest_script, '--outdir', DATA_DIR],
                check=False
            )
            if completed.returncode != 0:
                print(f"  [!] backtest_engine.py exited with code {completed.returncode}")
        else:
            print(f"  [!] Missing backtest engine at {backtest_script}")

    else:
        print("\nWalk-forward validation did not produce results. Please check errors above.")

    print("\nScript finished.")


