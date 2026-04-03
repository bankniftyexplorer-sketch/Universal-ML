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
import matplotlib
import argparse
import os
import subprocess
import sys

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from julia_bridge import (
    holographic_feature_engine_fast as holographic_feature_engine,
    add_target_fast as add_target,
)
from holographic_engine import (
    feature_selection_pipeline,
)
from numba import njit

warnings.filterwarnings("ignore")

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
# ─────────────────────────────────────────────
# FIBONACCI STRUCTURAL BASIS CONSTANTS
# ─────────────────────────────────────────────
FIB_RATIOS = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
FIB_ATR_PROMINENCE = 0.5
_FIB_RAW_COLS = frozenset(
    f"fib_{slot}_{field}"
    for slot in ("a", "b", "c")
    for field in (
        "close_pos",
        "zone",
        "wick_rej_bull",
        "wick_rej_bear",
        "body_acc",
        "ext_pct",
    )
)
EXEC_FEE_PCT = 0.0005
EXEC_SLIPPAGE_BPS = 0.0003
TIME_COL_CANDIDATES = (
    "Date",
    "date",
    "Datetime",
    "datetime",
    "Timestamp",
    "timestamp",
    "Time",
    "time",
)
MESSAGE_COL_CANDIDATES = ("Message", "message")
MODEL_N_JOBS = 4
TRADE_PLAN_LABEL_COLS = (
    "long_mfe_atr",
    "long_mae_atr",
    "short_mfe_atr",
    "short_mae_atr",
)
NON_FEATURE_COLS_SET = frozenset(
    {
        "time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "atr14",
        "basis_pct",
        "basis_z_score",
        "basis_vel_5",
        "basis_vel_10",
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
) | _FIB_RAW_COLS

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
    tf_patterns = [
        ("1M", r"(^|[_\- ])1M($|[_\- ])"),
        ("1W", r"(^|[_\- ])1W($|[_\- ])"),
        ("1D", r"(^|[_\- ])1D($|[_\- ])"),
        ("60", r"(^|[_\- ])1H($|[_\- ])|(^|[_\- ])60($|[_\- ])"),
    ]
    tf_guess = ""
    for tf_val, pattern in tf_patterns:
        if re.search(pattern, upper_stem):
            tf_guess = tf_val
            upper_stem = re.sub(pattern, "_", upper_stem)
            break
    symbol_guess = re.sub(r"[_\- ]+", "_", upper_stem).strip("_")
    return symbol_guess, tf_guess


def _timeframe_to_timedelta(tf_str: str) -> pd.Timedelta | pd.DateOffset:
    tf_key = str(tf_str).strip().upper()
    if tf_key in {"60", "1H", "H"}:
        return pd.Timedelta(hours=1)
    if tf_key in {"1D", "D"}:
        return pd.Timedelta(days=1)
    if tf_key in {"1W", "W"}:
        return pd.Timedelta(weeks=1)
    if tf_key in {"1M", "M"}:
        return pd.DateOffset(months=1)
    return pd.Timedelta(0)


def _drop_timezone_preserve_wall_clock(parsed: pd.Series) -> pd.Series:
    try:
        return parsed.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return parsed


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce", utc=False)
    parsed = _drop_timezone_preserve_wall_clock(parsed)
    if parsed.notna().mean() >= 0.7:
        return parsed

    numeric = pd.to_numeric(text.str.replace(",", "", regex=False), errors="coerce")
    if numeric.notna().mean() < 0.7:
        return parsed

    median_abs = numeric.dropna().abs().median()
    if median_abs >= 1e17:
        unit = "ns"
    elif median_abs >= 1e14:
        unit = "us"
    elif median_abs >= 1e11:
        unit = "ms"
    else:
        unit = "s"
    return pd.to_datetime(numeric, unit=unit, errors="coerce")


def _extract_symbol_and_timeframe(
    message_series: pd.Series, path: str
) -> tuple[str, str]:
    symbol_guess, tf_guess = _infer_symbol_timeframe_from_filename(path)
    symbol_str = symbol_guess
    tf_str = tf_guess
    try:
        symbol_extracted = message_series.str.extract(
            r"SYMBOL:\s*([^,]+)", expand=False
        ).dropna()
        if not symbol_extracted.empty:
            symbol_str = str(symbol_extracted.iloc[-1]).strip()

        tf_extracted = message_series.str.extract(
            r"TIME FRAME:\s*([A-Za-z0-9_]+)", expand=False
        ).dropna()
        if not tf_extracted.empty:
            tf_str = str(tf_extracted.iloc[-1]).strip()
    except Exception:
        pass
    return symbol_str, tf_str


def _detect_time_column(
    raw: pd.DataFrame, message_series: pd.Series, tf_str: str
) -> pd.Series:
    for col in TIME_COL_CANDIDATES:
        if col in raw.columns:
            parsed = _parse_datetime_series(raw[col])
            if parsed.notna().any():
                return parsed

    close_time = message_series.str.extract(r"CLOSE TIME:\s*([\d,]+)", expand=False)
    close_numeric = pd.to_numeric(
        close_time.str.replace(",", "", regex=False), errors="coerce"
    )
    if close_numeric.notna().any():
        close_dt = pd.to_datetime(close_numeric, unit="ms", errors="coerce")
        return close_dt - _timeframe_to_timedelta(tf_str)

    return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")


def parse_tv_log(path: str) -> tuple:
    """Parse TradingView log CSV and automatically detect Symbol and Timeframe."""
    # Use robust separator, handle bad lines
    raw = pd.read_csv(path, on_bad_lines="skip")

    if raw.empty:
        return pd.DataFrame(), None, None

    message_col = _find_first_column(raw, MESSAGE_COL_CANDIDATES)
    if message_col is None:
        return pd.DataFrame(), None, None

    message_series = raw[message_col].astype(str)
    symbol_str, tf_str = _extract_symbol_and_timeframe(message_series, path)

    df = pd.DataFrame()
    df["time"] = _detect_time_column(raw, message_series, tf_str)

    # Vectorized regex extraction directly from the message string.
    # Handles numbers with commas by replacing them post-extraction.
    for col, key in [
        ("open", "OPEN"),
        ("high", "HIGH"),
        ("low", "LOW"),
        ("close", "CLOSE"),
        ("volume", "VOLUME"),
    ]:
        extracted = message_series.str.extract(rf"{key}:\s*([\d,\.]+)", expand=False)
        df[col] = pd.to_numeric(extracted.str.replace(",", ""), errors="coerce")

    df = df.dropna().sort_values("time").reset_index(drop=True)
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
    c = df["close"]
    h = df["high"]
    l = df["low"]
    pc = c.shift(1).fillna(c)
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    df["atr14"] = tr.rolling(14).mean().fillna(tr)
    return df


def merge_higher_tf(
    df_1h: pd.DataFrame, df_1d: pd.DataFrame, df_1w: pd.DataFrame, df_1m: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge raw OHLCV from daily, weekly, and monthly bars into 1H dataframe.
    No classical indicators are computed here — the holographic engine handles
    all feature extraction separately.
    The look-ahead shift (+1 period) is preserved verbatim.
    """
    # ── LOOK-AHEAD FIX: Shift higher-TF timestamps forward by one full period ──
    # Only COMPLETED (closed) bars are available at any given 1H bar.
    df_1d_feat = df_1d[["time", "open", "high", "low", "close", "volume"]].copy()
    df_1w_feat = df_1w[["time", "open", "high", "low", "close", "volume"]].copy()
    df_1m_feat = df_1m[["time", "open", "high", "low", "close", "volume"]].copy()

    df_1d_feat.columns = ["time", "d_open", "d_high", "d_low", "d_close", "d_volume"]
    df_1w_feat.columns = ["time", "w_open", "w_high", "w_low", "w_close", "w_volume"]
    df_1m_feat.columns = ["time", "m_open", "m_high", "m_low", "m_close", "m_volume"]

    df_1d_feat["time"] = df_1d_feat["time"] + pd.Timedelta(days=1)
    df_1w_feat["time"] = df_1w_feat["time"] + pd.Timedelta(weeks=1)
    m_times = df_1m_feat["time"]
    m_next = m_times.shift(-1)
    m_gap = (
        m_times.diff().dropna().median()
        if len(m_times) > 1
        else pd.DateOffset(months=1)
    )
    df_1m_feat["time"] = m_next.where(m_next.notna(), m_times + m_gap)

    # Sort for merge_asof
    df_1h = df_1h.sort_values("time").reset_index(drop=True)
    df_1d_feat = df_1d_feat.sort_values("time")
    df_1w_feat = df_1w_feat.sort_values("time")
    df_1m_feat = df_1m_feat.sort_values("time")

    merged = pd.merge_asof(
        df_1h,
        df_1d_feat,
        on="time",
        direction="backward",
        tolerance=pd.Timedelta("2 days"),
    )
    merged = pd.merge_asof(
        merged,
        df_1w_feat,
        on="time",
        direction="backward",
        tolerance=pd.Timedelta("14 days"),
    )
    merged = pd.merge_asof(
        merged,
        df_1m_feat,
        on="time",
        direction="backward",
        tolerance=pd.Timedelta("62 days"),
    )
    merged = merged.ffill()

    return merged


# ─────────────────────────────────────────────
# 3. TARGET
# ─────────────────────────────────────────────


@njit(nopython=True)
def _lift_stop(stop: float, candidate: float, is_long: bool) -> float:
    return max(stop, candidate) if is_long else min(stop, candidate)


@njit(nopython=True)
def _favorable_touch(price: float, level: float, is_long: bool) -> bool:
    return price >= level if is_long else price <= level


@njit(nopython=True)
def _adverse_touch(price: float, level: float, is_long: bool) -> bool:
    return price <= level if is_long else price >= level


@njit(nopython=True)
def _realized_r(
    entry_price: float,
    exit_price: float,
    risk_dist: float,
    fraction: float,
    is_long: bool,
) -> float:
    gross = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    return (gross / (risk_dist + 1e-9)) * fraction


@njit(nopython=True)
def _fee_r(price: float, risk_dist: float, fraction: float, fee_pct: float) -> float:
    return (price * fee_pct / (risk_dist + 1e-9)) * fraction


@njit(nopython=True)
def simulate_trade_path_from_arrays_jit(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    entry_idx: int,
    is_long: bool,
    risk_dist: float,
    tp1_dist: float,
    tp2_dist: float,
    trail_dist: float,
    horizon: int,
    fee_pct: float,
    slippage_bps: float,
    tp1_frac: float,
    tp2_frac: float,
    runner_frac: float,
):
    n = len(opens)
    if entry_idx >= n or risk_dist <= 0 or not np.isfinite(risk_dist):
        return (np.nan, entry_idx, 0, 0.0, 0.0, False, False, 0.0)

    raw_entry = float(opens[entry_idx])
    if not np.isfinite(raw_entry):
        return (np.nan, entry_idx, 1, 0.0, 0.0, False, False, 0.0)

    entry_price = (
        raw_entry * (1 + slippage_bps) if is_long else raw_entry * (1 - slippage_bps)
    )
    stop = entry_price - risk_dist if is_long else entry_price + risk_dist
    tp1 = entry_price + tp1_dist if is_long else entry_price - tp1_dist
    tp2 = entry_price + tp2_dist if is_long else entry_price - tp2_dist

    rem_tp1 = tp1_frac
    rem_tp2 = tp2_frac
    rem_runner = runner_frac

    total_r = -_fee_r(entry_price, risk_dist, 1.0, fee_pct)
    tp1_hit = False
    tp2_hit = False
    exit_reason = 2
    exit_idx = min(n - 1, entry_idx + max(horizon - 1, 0))
    last_fill_price = entry_price

    last_idx = exit_idx
    for j in range(entry_idx, last_idx + 1):
        bar_open = float(opens[j])
        bar_high = float(highs[j])
        bar_low = float(lows[j])
        bar_close = float(closes[j])
        if (
            not np.isfinite(bar_open)
            or not np.isfinite(bar_high)
            or not np.isfinite(bar_low)
            or not np.isfinite(bar_close)
        ):
            exit_reason = 3
            exit_idx = j
            break

        if _adverse_touch(bar_open, stop, is_long):
            leftover = rem_tp1 + rem_tp2 + rem_runner
            if leftover > 0:
                total_r += _realized_r(
                    entry_price, bar_open, risk_dist, leftover, is_long
                )
                total_r -= _fee_r(bar_open, risk_dist, leftover, fee_pct)
                rem_tp1 = 0.0
                rem_tp2 = 0.0
                rem_runner = 0.0
                last_fill_price = bar_open
            exit_reason = 4
            exit_idx = j
            break

        if rem_tp1 > 0 and _favorable_touch(bar_open, tp1, is_long):
            total_r += _realized_r(entry_price, tp1, risk_dist, rem_tp1, is_long)
            total_r -= _fee_r(tp1, risk_dist, rem_tp1, fee_pct)
            rem_tp1 = 0.0
            last_fill_price = tp1
            tp1_hit = True
            stop = _lift_stop(stop, entry_price, is_long)

        if rem_tp2 > 0 and _favorable_touch(bar_open, tp2, is_long):
            total_r += _realized_r(entry_price, tp2, risk_dist, rem_tp2, is_long)
            total_r -= _fee_r(tp2, risk_dist, rem_tp2, fee_pct)
            rem_tp2 = 0.0
            last_fill_price = tp2
            tp2_hit = True
            stop = _lift_stop(stop, tp1, is_long)

        rem_total = rem_tp1 + rem_tp2 + rem_runner
        if rem_total <= 0:
            exit_reason = 5
            exit_idx = j
            break

        bar_best = bar_high if is_long else bar_low
        bar_worst = bar_low if is_long else bar_high
        fav_same_bar = (rem_tp1 > 0 and _favorable_touch(bar_best, tp1, is_long)) or (
            rem_tp2 > 0 and _favorable_touch(bar_best, tp2, is_long)
        )
        adv_same_bar = _adverse_touch(bar_worst, stop, is_long)

        if adv_same_bar and fav_same_bar:
            leftover = rem_tp1 + rem_tp2 + rem_runner
            if leftover > 0:
                total_r += _realized_r(entry_price, stop, risk_dist, leftover, is_long)
                total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                rem_tp1 = 0.0
                rem_tp2 = 0.0
                rem_runner = 0.0
                last_fill_price = stop
            exit_reason = 6
            exit_idx = j
            break

        if adv_same_bar:
            leftover = rem_tp1 + rem_tp2 + rem_runner
            if leftover > 0:
                total_r += _realized_r(entry_price, stop, risk_dist, leftover, is_long)
                total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                rem_tp1 = 0.0
                rem_tp2 = 0.0
                rem_runner = 0.0
                last_fill_price = stop
            exit_reason = 7
            exit_idx = j
            break

        if rem_tp1 > 0 and _favorable_touch(bar_best, tp1, is_long):
            total_r += _realized_r(entry_price, tp1, risk_dist, rem_tp1, is_long)
            total_r -= _fee_r(tp1, risk_dist, rem_tp1, fee_pct)
            rem_tp1 = 0.0
            last_fill_price = tp1
            tp1_hit = True
            stop = _lift_stop(stop, entry_price, is_long)
            rem_total = rem_tp1 + rem_tp2 + rem_runner
            if rem_total <= 0:
                exit_reason = 8
                exit_idx = j
                break
            if _adverse_touch(bar_worst, stop, is_long):
                leftover = rem_tp1 + rem_tp2 + rem_runner
                if leftover > 0:
                    total_r += _realized_r(
                        entry_price, stop, risk_dist, leftover, is_long
                    )
                    total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                    rem_tp1 = 0.0
                    rem_tp2 = 0.0
                    rem_runner = 0.0
                    last_fill_price = stop
                exit_reason = 9
                exit_idx = j
                break

        if rem_tp2 > 0 and _favorable_touch(bar_best, tp2, is_long):
            total_r += _realized_r(entry_price, tp2, risk_dist, rem_tp2, is_long)
            total_r -= _fee_r(tp2, risk_dist, rem_tp2, fee_pct)
            rem_tp2 = 0.0
            last_fill_price = tp2
            tp2_hit = True
            stop = _lift_stop(stop, tp1, is_long)
            rem_total = rem_tp1 + rem_tp2 + rem_runner
            if rem_total <= 0:
                exit_reason = 10
                exit_idx = j
                break
            if _adverse_touch(bar_worst, stop, is_long):
                leftover = rem_tp1 + rem_tp2 + rem_runner
                if leftover > 0:
                    total_r += _realized_r(
                        entry_price, stop, risk_dist, leftover, is_long
                    )
                    total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                    rem_tp1 = 0.0
                    rem_tp2 = 0.0
                    rem_runner = 0.0
                    last_fill_price = stop
                exit_reason = 11
                exit_idx = j
                break

        if rem_runner > 0 and trail_dist > 0:
            trail_candidate = (
                bar_close - trail_dist if is_long else bar_close + trail_dist
            )
            stop = _lift_stop(stop, trail_candidate, is_long)
            if tp1_hit:
                trail_base = tp1 if tp2_hit else entry_price
                stop = _lift_stop(stop, trail_base, is_long)

    else:
        leftover = rem_tp1 + rem_tp2 + rem_runner
        if leftover > 0:
            fill_price = float(closes[last_idx])
            total_r += _realized_r(
                entry_price, fill_price, risk_dist, leftover, is_long
            )
            total_r -= _fee_r(fill_price, risk_dist, leftover, fee_pct)
            rem_tp1 = 0.0
            rem_tp2 = 0.0
            rem_runner = 0.0
            last_fill_price = fill_price
        exit_reason = 2
        exit_idx = last_idx

    rem_total_final = rem_tp1 + rem_tp2 + rem_runner
    if rem_total_final > 0 and exit_reason == 3:
        idx_to_fill = min(exit_idx, n - 1)
        fill_price = float(closes[idx_to_fill])
        total_r += _realized_r(
            entry_price, fill_price, risk_dist, rem_total_final, is_long
        )
        total_r -= _fee_r(fill_price, risk_dist, rem_total_final, fee_pct)
        last_fill_price = fill_price

    return (
        float(total_r),
        int(exit_idx),
        int(exit_reason),
        float(entry_price),
        float(last_fill_price),
        bool(tp1_hit),
        bool(tp2_hit),
        float(stop),
    )


def simulate_trade_path_from_arrays(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    times,
    entry_idx: int,
    direction: str,
    risk_dist: float,
    tp1_dist=None,
    tp2_dist=None,
    trail_dist=None,
    horizon: int = BARRIER_HORIZON_BARS,
    fee_pct: float = EXEC_FEE_PCT,
    slippage_bps: float = EXEC_SLIPPAGE_BPS,
) -> dict:
    is_long = direction.upper() == "LONG"
    tp1_d = risk_dist * TP1_R_MULT if tp1_dist is None else tp1_dist
    tp2_d = risk_dist * TP2_R_MULT if tp2_dist is None else tp2_dist
    trail_d = risk_dist * TRAIL_R_MULT if trail_dist is None else trail_dist

    ret = simulate_trade_path_from_arrays_jit(
        opens,
        highs,
        lows,
        closes,
        entry_idx,
        is_long,
        risk_dist,
        tp1_d,
        tp2_d,
        trail_d,
        horizon,
        fee_pct,
        slippage_bps,
        TP1_FRACTION,
        TP2_FRACTION,
        RUNNER_FRACTION,
    )

    total_r, exin, ereas, epr, xpr, tp1h, tp2h, fst = ret

    reasons = {
        0: "INVALID",
        1: "INVALID_ENTRY",
        2: "TIME_EXIT",
        3: "BAD_BAR",
        4: "SL_GAP",
        5: "TARGETS_GAP_FILLED",
        6: "AMBIGUOUS_BAR_SL",
        7: "SL_HIT",
        8: "TP1_FILLED",
        9: "POST_TP1_STOP",
        10: "TP2_FILLED",
        11: "POST_TP2_STOP",
    }

    exit_time = None if times is None else times[min(exin, len(opens) - 1)]
    return {
        "entry_idx": entry_idx,
        "entry_price": epr,
        "exit_price": xpr,
        "exit_idx": min(exin, len(opens) - 1),
        "exit_time": exit_time,
        "exit_reason": reasons.get(ereas, "UNKNOWN"),
        "total_r": total_r,
        "tp1_hit": tp1h,
        "tp2_hit": tp2h,
        "final_stop": fst,
    }


@njit(nopython=True)
def _add_target_loop_jit(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    atr_mult: float,
    horizon: int,
    tp1_r_mult: float,
    tp2_r_mult: float,
    trail_r_mult: float,
    fee_pct: float,
    slippage_bps: float,
    tp1_frac: float,
    tp2_frac: float,
    runner_frac: float,
):
    n = len(opens)
    last_start_idx = max(-1, n - horizon - 1)

    target = np.full(n, np.nan, dtype=np.float64)
    next_ret_pct = np.full(n, np.nan, dtype=np.float64)
    bars_to_target = np.full(n, np.nan, dtype=np.float64)
    entry_prices = np.full(n, np.nan, dtype=np.float64)
    target_distances = np.full(n, np.nan, dtype=np.float64)
    long_path_r = np.full(n, np.nan, dtype=np.float64)
    short_path_r = np.full(n, np.nan, dtype=np.float64)
    target_edge_r = np.full(n, np.nan, dtype=np.float64)
    best_path_r = np.full(n, np.nan, dtype=np.float64)
    long_mfe_atr = np.full(n, np.nan, dtype=np.float64)
    long_mae_atr = np.full(n, np.nan, dtype=np.float64)
    short_mfe_atr = np.full(n, np.nan, dtype=np.float64)
    short_mae_atr = np.full(n, np.nan, dtype=np.float64)

    for i in range(last_start_idx + 1):
        entry_idx = i + 1
        dist = atrs[i] * atr_mult
        if not np.isfinite(dist) or dist <= 0:
            continue

        long_trade = simulate_trade_path_from_arrays_jit(
            opens,
            highs,
            lows,
            closes,
            entry_idx,
            True,
            dist,
            dist * tp1_r_mult,
            dist * tp2_r_mult,
            dist * trail_r_mult,
            horizon,
            fee_pct,
            slippage_bps,
            tp1_frac,
            tp2_frac,
            runner_frac,
        )
        short_trade = simulate_trade_path_from_arrays_jit(
            opens,
            highs,
            lows,
            closes,
            entry_idx,
            False,
            dist,
            dist * tp1_r_mult,
            dist * tp2_r_mult,
            dist * trail_r_mult,
            horizon,
            fee_pct,
            slippage_bps,
            tp1_frac,
            tp2_frac,
            runner_frac,
        )

        long_r = long_trade[0]
        short_r = short_trade[0]
        long_exit_idx = long_trade[1]
        short_exit_idx = short_trade[1]
        long_entry_price = long_trade[3]

        if not np.isfinite(long_r) or not np.isfinite(short_r):
            continue

        long_path_r[i] = long_r
        short_path_r[i] = short_r
        best_path_r[i] = max(long_r, short_r)
        target_edge_r[i] = abs(long_r - short_r)
        entry_prices[i] = long_entry_price
        target_distances[i] = dist

        horizon_end = min(n, entry_idx + horizon)
        if horizon_end > entry_idx and atrs[i] > 0:
            entry_price = float(long_entry_price)
            curr_atr = float(atrs[i])

            # LONG MAE/MFE JIT natively
            sl_dist_l = 2.0 * curr_atr
            sl_price_l = entry_price - sl_dist_l

            peak_high_l = entry_price
            peak_low_l = entry_price
            hit_l = False
            for j in range(entry_idx, horizon_end):
                val_high = float(highs[j])
                val_low = float(lows[j])

                if val_low <= sl_price_l:
                    if val_high > peak_high_l:
                        peak_high_l = val_high
                    peak_low_l = sl_price_l
                    hit_l = True
                    break
                else:
                    if val_high > peak_high_l:
                        peak_high_l = val_high
                    if val_low < peak_low_l:
                        peak_low_l = val_low

            long_mfe_atr[i] = max(0.0, (peak_high_l - entry_price) / (curr_atr + 1e-9))
            long_mae_atr[i] = max(0.0, (entry_price - peak_low_l) / (curr_atr + 1e-9))

            # SHORT MAE/MFE JIT natively
            sl_dist_s = 2.0 * curr_atr
            sl_price_s = entry_price + sl_dist_s

            peak_low_s = entry_price
            peak_high_s = entry_price
            hit_s = False
            for j in range(entry_idx, horizon_end):
                val_high = float(highs[j])
                val_low = float(lows[j])

                if val_high >= sl_price_s:
                    if val_low < peak_low_s:
                        peak_low_s = val_low
                    peak_high_s = sl_price_s
                    hit_s = True
                    break
                else:
                    if val_low < peak_low_s:
                        peak_low_s = val_low
                    if val_high > peak_high_s:
                        peak_high_s = val_high

            short_mfe_atr[i] = max(0.0, (entry_price - peak_low_s) / (curr_atr + 1e-9))
            short_mae_atr[i] = max(0.0, (peak_high_s - entry_price) / (curr_atr + 1e-9))

        if horizon_end > entry_idx and atrs[i] > 0:
            mfe_l = long_mfe_atr[i]
            mae_l = long_mae_atr[i]
            raw_long = mfe_l / (mfe_l + mae_l + 1e-9)
            vel_l = 1.0 - ((long_exit_idx - i) / horizon)
            long_kinscore = raw_long * max(0.01, vel_l)

            mfe_s = short_mfe_atr[i]
            mae_s = short_mae_atr[i]
            raw_short = mfe_s / (mfe_s + mae_s + 1e-9)
            vel_s = 1.0 - ((short_exit_idx - i) / horizon)
            short_kinscore = raw_short * max(0.01, vel_s)

            if long_kinscore > short_kinscore and long_kinscore > 0.15:
                target[i] = 0.5 + (long_kinscore / 2.0)
                next_ret_pct[i] = (long_r * dist / (entry_prices[i] + 1e-9)) * 100.0
                bars_to_target[i] = long_exit_idx - i
            elif short_kinscore > long_kinscore and short_kinscore > 0.15:
                target[i] = 0.5 - (short_kinscore / 2.0)
                next_ret_pct[i] = -(short_r * dist / (entry_prices[i] + 1e-9)) * 100.0
                bars_to_target[i] = short_exit_idx - i
            else:
                target[i] = 0.5
                next_ret_pct[i] = 0.0
                bars_to_target[i] = horizon
        else:
            target[i] = 0.5
            next_ret_pct[i] = 0.0
            bars_to_target[i] = horizon

    return (
        target,
        next_ret_pct,
        bars_to_target,
        entry_prices,
        target_distances,
        long_path_r,
        short_path_r,
        target_edge_r,
        best_path_r,
        long_mfe_atr,
        long_mae_atr,
        short_mfe_atr,
        short_mae_atr,
    )


def _legacy_add_target_DEPRECATED(
    df: pd.DataFrame,
    atr_mult: float = BARRIER_ATR_MULT,
    horizon: int = BARRIER_HORIZON_BARS,
    atr_col: str = "atr14",
    drop_unresolved: bool = True,
) -> pd.DataFrame:
    """
    DEPRECATED: Replaced by ToonMath.jl. DO NOT use as fallback.
    """
    df = df.copy()
    n = len(df)
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    times = df["time"].to_numpy() if "time" in df.columns else None
    atrs = df[atr_col].to_numpy(dtype=float)

    (
        target,
        next_ret_pct,
        bars_to_target,
        entry_prices,
        target_distances,
        long_path_r,
        short_path_r,
        target_edge_r,
        best_path_r,
        long_mfe_atr,
        long_mae_atr,
        short_mfe_atr,
        short_mae_atr,
    ) = _add_target_loop_jit(
        opens,
        highs,
        lows,
        closes,
        atrs,
        atr_mult,
        horizon,
        TP1_R_MULT,
        TP2_R_MULT,
        TRAIL_R_MULT,
        EXEC_FEE_PCT,
        EXEC_SLIPPAGE_BPS,
        TP1_FRACTION,
        TP2_FRACTION,
        RUNNER_FRACTION,
    )

    df["target"] = target
    df["next_ret_pct"] = next_ret_pct
    df["bars_to_target"] = bars_to_target
    df["entry_price_next_bar"] = entry_prices
    df["target_distance"] = target_distances
    df["long_path_r"] = long_path_r
    df["short_path_r"] = short_path_r
    df["target_edge_r"] = target_edge_r
    df["best_path_r"] = best_path_r
    df["long_mfe_atr"] = long_mfe_atr
    df["long_mae_atr"] = long_mae_atr
    df["short_mfe_atr"] = short_mfe_atr
    df["short_mae_atr"] = short_mae_atr

    if drop_unresolved:
        df = df[df["target"] != 0.5].dropna(subset=["target"]).copy()
    else:
        df["target"] = df["target"].fillna(0.5)
    return df


# ─────────────────────────────────────────────
# 4. WALK-FORWARD VALIDATION (Optimized for CPU)
# ─────────────────────────────────────────────


def _build_lgbm_model() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
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
        objective="regression",
        metric="mae",
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
        objective="quantile",
        alpha=alpha,
    )


def _fit_lgbm_with_inner_validation(
    X_train_full: pd.DataFrame, y_train_full: pd.Series
) -> tuple[lgb.LGBMRegressor, pd.DataFrame, pd.Series]:
    inner_val_size = max(100, int(len(X_train_full) * 0.15))
    inner_val_size = min(inner_val_size, len(X_train_full) - 1)
    if inner_val_size <= 0:
        raise ValueError("Not enough training rows for inner validation.")

    purge_gap = 24  # must match BARRIER_HORIZON_BARS
    purged_end = -(inner_val_size + purge_gap)
    if abs(purged_end) >= len(X_train_full):
        purged_end = (
            0  # fallback: not enough data, use all for training (no purge possible)
        )
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

    model = _build_lgbm_model()
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
    return model, X_val_inner, y_val_inner


def train_trade_plan_models(df: pd.DataFrame, feature_cols: list) -> dict:
    specs = {
        "up_stop_atr": (
            1,
            "long_mae_atr",
            0.40,
        ),  # Tightened: 40th percentile (Pro-Trader SL)
        "up_tp1_atr": (1, "long_mfe_atr", 0.50),
        "up_tp2_atr": (1, "long_mfe_atr", 0.80),
        "down_stop_atr": (0, "short_mae_atr", 0.40),  # Tightened
        "down_tp1_atr": (0, "short_mfe_atr", 0.50),
        "down_tp2_atr": (0, "short_mfe_atr", 0.80),
    }
    models = {}
    for key, (target_value, label_col, alpha) in specs.items():
        train_df = (
            df[df["target"] == target_value]
            .dropna(subset=feature_cols + [label_col])
            .copy()
        )
        if len(train_df) < 300:
            continue
        model = _build_lgbm_regressor(alpha)
        model.fit(train_df[feature_cols], train_df[label_col])
        models[key] = model
    return models


def predict_trade_plan(
    plan_models: dict,
    feature_cols: list,
    latest_row: pd.Series,
    direction: str,
    atr_value: float,
) -> dict:
    fallback_r = BARRIER_ATR_MULT
    if direction not in {"UP", "DOWN"} or atr_value <= 0:
        return {
            "sl": np.nan,
            "tp1": np.nan,
            "tp2": np.nan,
            "stop_atr": np.nan,
            "tp1_atr": np.nan,
            "tp2_atr": np.nan,
            "trail_r": np.nan,
            "note": "No directional trade plan available.",
        }

    prefix = "up" if direction == "UP" else "down"
    required = [f"{prefix}_stop_atr", f"{prefix}_tp1_atr", f"{prefix}_tp2_atr"]
    missing_cols = [col for col in feature_cols if col not in latest_row.index]
    if missing_cols:
        for col in missing_cols:
            latest_row[col] = 0.0
    X = latest_row[feature_cols].values.reshape(1, -1)

    if all(key in plan_models for key in required):
        # STRUCTURAL LIMITS: Never allow Sl beyond 1.25x ATR or TP beyond 5.00x ATR
        stop_atr = float(np.clip(plan_models[required[0]].predict(X)[0], 0.35, 1.25))
        tp1_atr = float(np.clip(plan_models[required[1]].predict(X)[0], 0.25, 3.00))
        tp2_atr = float(np.clip(plan_models[required[2]].predict(X)[0], 0.50, 5.00))

        # Ensure Reward/Risk logic: TP1 should be at least 1x Stop
        tp1_atr = max(tp1_atr, stop_atr * 1.00)
        tp2_atr = max(tp2_atr, tp1_atr + 0.25)
        trail_r = float(np.clip(stop_atr, 0.40, 1.50))
        note = "ML-derived levels (Tightened for Professional Risk)."
    else:
        stop_atr = fallback_r
        tp1_atr = fallback_r
        tp2_atr = fallback_r * TP2_R_MULT
        trail_r = TRAIL_R_MULT
        note = "Fallback ATR plan: excursion models unavailable."

    close_price = float(latest_row["close"])
    if direction == "UP":
        sl = close_price - (atr_value * stop_atr)
        tp1 = close_price + (atr_value * tp1_atr)
        tp2 = close_price + (atr_value * tp2_atr)
    else:
        sl = close_price + (atr_value * stop_atr)
        tp1 = close_price - (atr_value * tp1_atr)
        tp2 = close_price - (atr_value * tp2_atr)

    return {
        "sl": sl,
        "tp1": tp1,
        "tp2": tp2,
        "stop_atr": stop_atr,
        "tp1_atr": tp1_atr,
        "tp2_atr": tp2_atr,
        "trail_r": trail_r,
        "note": note,
    }


def walk_forward(
    df: pd.DataFrame,
    feature_cols: list,
    n_splits: int = 10,
    min_train_bars: int = 2000,
    test_size_ratio: float = 0.15,
    purge_gap: int = 24,
) -> dict:
    """
    Pure walk-forward: train on past, test on strictly future window.
    Early-stopping uses an inner validation slice carved from the training
    window only — the test fold is NEVER visible during training.
    OOS probabilities are accumulated in oos_proba_map for honest backtesting.
    """
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    n = len(df)

    # Dynamic window size calculation based on ratio
    if n < min_train_bars:
        print(
            f"  Error: Not enough data for walk-forward. Need {min_train_bars}, got {n}."
        )
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

    print(
        f"\n  Walk-forward: {len(split_points)} splits, "
        f"min_train_bars={min_train_bars}, test_ratio={test_size_ratio:.2f}"
    )
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
        y_train_full = df["target"].iloc[:purged_train_end]
        X_test = df[feature_cols].iloc[train_end:test_end]
        y_test = df["target"].iloc[train_end:test_end]

        if len(X_test) == 0:
            continue

        # [TOON vX.0] True Early Stopping + Thread Safety
        model, _, _ = _fit_lgbm_with_inner_validation(X_train_full, y_train_full)

        preds = model.predict(X_test)

        # In a regression context over [0.0, 1.0], direction is >0.5 or <0.5
        # The true binary direction for evaluation
        true_binary = (y_test > 0.5).astype(int)
        pred_binary = (preds > 0.5).astype(int)
        acc = (true_binary == pred_binary).mean()

        # High confidence predictions via continuous score
        # e.g. >0.80 (Strong LONG), <0.20 (Strong SHORT)
        high_conf_mask = (preds >= LIVE_CONFIDENCE_THRESHOLD) | (
            preds <= (1.0 - LIVE_CONFIDENCE_THRESHOLD)
        )
        if high_conf_mask.sum() > 0:
            hc_preds_bin = pred_binary[high_conf_mask]
            hc_true_bin = true_binary[high_conf_mask]
            conf_acc = (hc_preds_bin == hc_true_bin).mean()
        else:
            conf_acc = float("nan")

        fraction_conf = high_conf_mask.sum() / len(preds)
        baseline = true_binary.mean()

        print(
            f"  Split {i + 1:>2} | Train: {len(X_train_full):>5} | Test: {len(X_test):>5} | "
            f"Acc:{acc:.3f} | ConfAcc(>={LIVE_CONFIDENCE_THRESHOLD:.2f}):{conf_acc:.3f} | "
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
            test_times = df["time"].iloc[train_end:test_end]
            for ts, prob in zip(test_times, preds):
                # Ensure OOS map aligns perfectly with backtest IST localized times
                oos_proba_map[pd.Timestamp(ts)] = float(prob)

    # Train ultimate model on all data
    final_model, _, _ = _fit_lgbm_with_inner_validation(df[feature_cols], df["target"])

    return {
        "splits": results,
        "feature_importance": dict(
            zip(feature_cols, feature_importance_sum / len(split_points))
        ),
        "final_model": final_model,
        "feature_cols": feature_cols,
        "df": df,
        "overall_accuracy": np.mean([r["accuracy"] for r in results]),
        "overall_baseline_up": np.mean([r["baseline_up"] for r in results]),
        "oos_proba_map": oos_proba_map,
    }


# ─────────────────────────────────────────────
# 5. REPORT CHART (Visual Enhancements)
# ─────────────────────────────────────────────


def save_report(
    wf_results: dict, save_path: str, pred_info: dict = None, symbol: str = "UNKNOWN"
):
    if not wf_results or "splits" not in wf_results or not wf_results["splits"]:
        print("  No walk-forward results to plot.")
        return

    splits = pd.DataFrame(wf_results["splits"])
    fi = wf_results["feature_importance"]

    # Add overall metrics if available
    if "overall_accuracy" in wf_results:
        splits_summary = splits.mean(numeric_only=True)
        splits_summary["accuracy"] = wf_results["overall_accuracy"]
        splits_summary["acc_high_conf"] = splits[
            "acc_high_conf"
        ].mean()  # Avg of high_conf acc across splits
        splits_summary["high_conf_pct"] = splits["high_conf_pct"].mean()
        splits_summary["baseline_up"] = wf_results["overall_baseline_up"]
    else:
        splits_summary = splits.mean(numeric_only=True)

    fig = plt.figure(figsize=(14, 16), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.4, height_ratios=[1, 1.5, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])  # Top row: TRADER TEXT FORECAST
    ax2 = fig.add_subplot(gs[1, 0])  # Middle row: CODER ACCURACY
    ax3 = fig.add_subplot(gs[2, 0])  # Bottom row: CODER FEATURES (Top 15)

    # Color theme
    dark_bg_color = "#0d0d0d"
    light_text_color = "#e0e0e0"
    axis_line_color = "#444444"
    accent_color_1 = "#00d4ff"  # Cyan
    accent_color_2 = "#00ff88"  # Green
    accent_color_3 = "#ff6b6b"  # Red
    accent_color_5 = "#7c4dff"  # Purple

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(dark_bg_color)
        ax.tick_params(colors=light_text_color, labelsize=9)
        for spine in ["bottom", "top", "left", "right"]:
            ax.spines[spine].set_color(axis_line_color)

    # — 1. TRADER FORECAST (TEXT PANEL) —
    ax1.axis("off")  # Hide axes for text panel
    if pred_info:
        bg_color = accent_color_2 if pred_info["dir"] == "UP" else accent_color_3

        # Get dynamic symbol from pred_info or default to BANKNIFTY
        symbol_display = pred_info.get("symbol", "BANKNIFTY").upper()
        note_line = (
            f"  Filter Note : {pred_info['note']}\n" if pred_info.get("note") else ""
        )

        report_text = (
            f"======================================================================\n"
            f"  {symbol_display} 1H FORECAST (EXECUTION-ALIGNED)\n"
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
        ax1.text(
            0.5,
            0.5,
            report_text,
            color=bg_color,
            fontsize=16,
            fontfamily="monospace",
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(
                facecolor="#1a1a1a", edgecolor=bg_color, pad=2.0, boxstyle="round"
            ),
        )

    # — 2. CODER ACCURACY PLOT —
    x = splits["split"]
    ax2.plot(
        x, splits["accuracy"], "o-", color=accent_color_1, lw=2, label="Split Accuracy"
    )
    ax2.plot(
        x,
        splits["acc_high_conf"],
        "s--",
        color=accent_color_2,
        lw=2,
        label=f"High-Confidence Accuracy (>={LIVE_CONFIDENCE_THRESHOLD:.2f})",
    )
    ax2.axhline(
        splits_summary["baseline_up"],
        color=accent_color_3,
        ls=":",
        lw=1.5,
        label=f"Always-UP baseline ({splits_summary['baseline_up']:.2f})",
    )
    ax2.axhline(0.5, color="#777777", ls=":", lw=1)

    # Overall accuracy line
    ax2.axhline(
        splits_summary["accuracy"],
        color=accent_color_1,
        ls="--",
        lw=1,
        label=f"Overall Acc ({splits_summary['accuracy']:.3f})",
    )
    ax2.fill_between(x, 0.5, splits["accuracy"], alpha=0.1, color=accent_color_1)

    ax2.set_title(
        "Walk-Forward Accuracy (Out-of-Sample)",
        color=light_text_color,
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_xlabel("Split (Timeline)", color=light_text_color)
    ax2.set_ylabel("Accuracy", color=light_text_color)
    ax2.set_ylim(0.4, 0.85)
    ax2.legend(
        facecolor=dark_bg_color,
        edgecolor=axis_line_color,
        labelcolor=light_text_color,
        fontsize=10,
        loc="lower right",
    )

    # — 3. CODER TOP 15 FEATURES —
    fi_filtered = {k: v for k, v in fi.items() if v > 0}
    top15 = sorted(fi_filtered.items(), key=lambda x: x[1], reverse=True)[:15]

    if top15:
        fnames, fvals = zip(*top15)
        fnames = [str(n).replace("_", " ").upper() for n in fnames]

        bars = ax3.barh(range(len(fnames)), fvals, color=accent_color_5, alpha=0.85)
        ax3.set_yticks(range(len(fnames)))
        ax3.set_yticklabels(fnames, color="#cccccc", fontsize=10)
        ax3.set_title(
            "Top 15 Most Predictive Features",
            color=light_text_color,
            fontsize=14,
            fontweight="bold",
        )
        ax3.invert_yaxis()
        ax3.set_xlabel("Importance Score", color=light_text_color)
    else:
        ax3.text(
            0.5,
            0.5,
            "No significant features",
            color=light_text_color,
            ha="center",
            va="center",
        )

    fig.suptitle(
        f"{(pred_info or {}).get('symbol', symbol).upper()} — ML Performance Report",
        color="white",
        fontsize=18,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\n  Report saved → {save_path}")


# ─────────────────────────────────────────────
# 6. PREDICTION FUNCTION (live use)
# ─────────────────────────────────────────────


def predict_next_bar(
    model: lgb.LGBMRegressor,
    feature_cols: list,
    row: pd.Series,
    confidence_threshold: float = LIVE_CONFIDENCE_THRESHOLD,
) -> dict:
    X_inf = row[feature_cols].to_frame().T.astype(float)
    proba_up = float(np.clip(model.predict(X_inf)[0], 0.0, 1.0))

    direction = "UP" if proba_up > 0.5 else "DOWN"
    # Confidence in Regressor: distance from 0.5 mapped to 0-100% equivalent
    confidence = max(proba_up, 1.0 - proba_up)

    if confidence >= confidence_threshold:
        signal = "STRONG"
    else:
        signal = "WEAK"

    return {
        "direction": direction,
        "confidence": confidence,
        "signal_strength": signal,
        "raw_score": proba_up,
    }


# ─────────────────────────────────────────────
# 7. MAIN EXECUTION (Optimized for performance)
# ─────────────────────────────────────────────


def inject_thermodynamic_basis(
    df_fut: pd.DataFrame, df_idx: pd.DataFrame
) -> pd.DataFrame:
    """
    Universally aligns Spot/Index to Futures and computes thermodynamic basis mechanics.
    Strict backward ASOF merge prevents forward-peeking.
    """
    if df_idx is None or df_idx.empty:
        print("  [Basis] Warning: No Spot/Index provided. Basis mechanics disabled.")
        return df_fut.copy()

    # Ensure temporal sorting
    fut = df_fut.sort_values("time").reset_index(drop=True)
    idx = (
        df_idx[["time", "close"]]
        .rename(columns={"close": "spot_close"})
        .sort_values("time")
    )

    # Align Spot to Future (Strict backward mapping)
    merged = pd.merge_asof(fut, idx, on="time", direction="backward")

    # Core Thermodynamics
    merged["basis_pts"] = merged["close"] - merged["spot_close"]
    merged["basis_pct"] = merged["basis_pts"] / (merged["spot_close"] + 1e-9)

    # Cross-Sectional Extremes (Z-Score)
    basis_pct = merged["basis_pct"]
    basis_ema20 = basis_pct.ewm(span=20, adjust=False).mean()
    basis_dev = (basis_pct - basis_ema20).abs().ewm(span=20, adjust=False).mean()
    merged["basis_z_score"] = (basis_pct - basis_ema20) / (basis_dev + 1e-9)

    # Temporal Physics (Velocity replaces DTE for universal Perp/Dated compatibility)
    merged["basis_vel_5"] = merged["basis_pct"].diff(5)
    merged["basis_vel_10"] = merged["basis_pct"].diff(10)

    # Cleanup
    merged = merged.drop(columns=["spot_close"]).fillna(0.0)
    return merged


def fib_structural_basis(
    df_ltf: pd.DataFrame,
    htf_frames: dict[str, pd.DataFrame | None],
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """
    Inject per-bar Fibonacci structural features onto the primary timeframe.

    For each (HTF, slot) pair, the latest qualifying HTF swing high and swing
    low are detected in real price space, merged backward onto the LTF bars,
    and converted into six structural interaction columns.
    """
    result = df_ltf.sort_values("time").reset_index(drop=True).copy()
    fib_ratios = np.array(FIB_RATIOS, dtype=np.float64)

    def _write_defaults(prefix: str) -> None:
        result[f"{prefix}_close_pos"] = 0.5
        result[f"{prefix}_zone"] = 3.0
        result[f"{prefix}_wick_rej_bull"] = 0.0
        result[f"{prefix}_wick_rej_bear"] = 0.0
        result[f"{prefix}_body_acc"] = 0.0
        result[f"{prefix}_ext_pct"] = 0.0

    for htf_key, slot in pairs:
        prefix = f"fib_{slot}"
        df_htf = htf_frames.get(htf_key)
        if df_htf is None or df_htf.empty or len(df_htf) < 10:
            _write_defaults(prefix)
            continue

        htf = df_htf.sort_values("time").reset_index(drop=True).copy()
        prev_close = htf["close"].shift(1).fillna(htf["close"])
        true_range = np.maximum(
            htf["high"] - htf["low"],
            np.maximum(
                np.abs(htf["high"] - prev_close),
                np.abs(htf["low"] - prev_close),
            ),
        )
        atr14 = true_range.rolling(14).mean().fillna(true_range).to_numpy(dtype=float)
        highs = htf["high"].to_numpy(dtype=float)
        lows = htf["low"].to_numpy(dtype=float)
        n_htf = len(htf)

        swing_highs: list[tuple[int, float]] = []
        swing_lows: list[tuple[int, float]] = []
        for j in range(1, n_htf - 1):
            if (
                highs[j] > highs[j - 1]
                and highs[j] > highs[j + 1]
                and (highs[j] - max(highs[j - 1], highs[j + 1]))
                >= FIB_ATR_PROMINENCE * atr14[j]
            ):
                swing_highs.append((j, highs[j]))
            if (
                lows[j] < lows[j - 1]
                and lows[j] < lows[j + 1]
                and (min(lows[j - 1], lows[j + 1]) - lows[j])
                >= FIB_ATR_PROMINENCE * atr14[j]
            ):
                swing_lows.append((j, lows[j]))

        fib_swing_low = np.full(n_htf, np.nan, dtype=float)
        fib_swing_high = np.full(n_htf, np.nan, dtype=float)

        if swing_highs and swing_lows:
            sh_idx = np.array([idx for idx, _ in swing_highs], dtype=int)
            sh_price = np.array([price for _, price in swing_highs], dtype=float)
            sl_idx = np.array([idx for idx, _ in swing_lows], dtype=int)
            sl_price = np.array([price for _, price in swing_lows], dtype=float)
            bars = np.arange(n_htf, dtype=int)

            sh_pos = np.searchsorted(sh_idx, bars, side="left") - 1
            sl_pos = np.searchsorted(sl_idx, bars, side="left") - 1

            valid_h = sh_pos >= 0
            valid_l = sl_pos >= 0
            valid = valid_h & valid_l
            if np.any(valid):
                last_high = np.full(n_htf, np.nan, dtype=float)
                last_low = np.full(n_htf, np.nan, dtype=float)
                last_high[valid_h] = sh_price[sh_pos[valid_h]]
                last_low[valid_l] = sl_price[sl_pos[valid_l]]
                non_degenerate = valid & (last_high > last_low)
                fib_swing_low[non_degenerate] = last_low[non_degenerate]
                fib_swing_high[non_degenerate] = last_high[non_degenerate]

        merged = pd.merge_asof(
            result[["time", "open", "high", "low", "close"]].sort_values("time"),
            pd.DataFrame(
                {
                    "time": htf["time"],
                    "fib_swing_low": fib_swing_low,
                    "fib_swing_high": fib_swing_high,
                }
            ).sort_values("time"),
            on="time",
            direction="backward",
        ).reset_index(drop=True)

        close_vals = merged["close"].to_numpy(dtype=float)
        fib_low = merged["fib_swing_low"].to_numpy(dtype=float)
        fib_high = merged["fib_swing_high"].to_numpy(dtype=float)
        fib_low = np.where(np.isnan(fib_low), close_vals * 0.97, fib_low)
        fib_high = np.where(np.isnan(fib_high), close_vals * 1.03, fib_high)
        fib_range = np.where((fib_high - fib_low) < 1e-9, 1e-9, fib_high - fib_low)

        ltf_open = merged["open"].to_numpy(dtype=float)
        ltf_high = merged["high"].to_numpy(dtype=float)
        ltf_low = merged["low"].to_numpy(dtype=float)
        ltf_close = merged["close"].to_numpy(dtype=float)

        close_pos = (ltf_close - fib_low) / fib_range
        zone = np.searchsorted(fib_ratios, close_pos, side="left").astype(float)
        zone = np.clip(zone, 0.0, float(len(fib_ratios)))

        level_prices = fib_low[:, None] + fib_range[:, None] * fib_ratios[None, :]
        nearest_idx = np.argmin(np.abs(level_prices - ltf_close[:, None]), axis=1)
        nearest_level = level_prices[np.arange(len(ltf_close)), nearest_idx]

        wick_rej_bull = ((ltf_low < nearest_level) & (ltf_close >= nearest_level)).astype(float)
        wick_rej_bear = ((ltf_high > nearest_level) & (ltf_close <= nearest_level)).astype(float)
        body_top = np.maximum(ltf_open, ltf_close)
        body_bot = np.minimum(ltf_open, ltf_close)
        body_acc = ((body_bot >= nearest_level) | (body_top <= nearest_level)).astype(float)
        ext_pct = np.maximum(0.0, close_pos - 1.0)

        result[f"{prefix}_close_pos"] = close_pos
        result[f"{prefix}_zone"] = zone
        result[f"{prefix}_wick_rej_bull"] = wick_rej_bull
        result[f"{prefix}_wick_rej_bear"] = wick_rej_bear
        result[f"{prefix}_body_acc"] = body_acc
        result[f"{prefix}_ext_pct"] = ext_pct

    return result


def analyze_reference_coverage(
    df_primary: pd.DataFrame, df_reference: pd.DataFrame
) -> dict:
    """
    Audits whether every primary timestamp has an exact reference timestamp.

    This is stricter than the ASOF basis merge and is intended for the 1H
    derivative execution layer, where reusing a stale prior SPOT bar would hide
    a real market-data gap.
    """
    primary_times = pd.to_datetime(df_primary["time"])
    reference_times = pd.to_datetime(df_reference["time"])

    if primary_times.dt.tz is not None:
        primary_times = primary_times.dt.tz_convert("UTC").dt.tz_localize(None)
    if reference_times.dt.tz is not None:
        reference_times = reference_times.dt.tz_convert("UTC").dt.tz_localize(None)

    primary_set = set(primary_times)
    reference_set = set(reference_times)
    missing_reference = sorted(primary_set - reference_set)
    extra_reference = sorted(reference_set - primary_set)

    return {
        "missing_reference_count": len(missing_reference),
        "extra_reference_count": len(extra_reference),
        "missing_reference_sample": missing_reference[:5],
        "extra_reference_sample": extra_reference[:5],
    }


def encode_session_time_vectors(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session-position and end-of-day basis momentum columns used by the
    intraday holographic bridge.
    """
    encoded = df_1h.copy()
    if "time" not in encoded.columns or encoded.empty:
        encoded["session_time_pos"] = 0.0
        encoded["eod_basis_momentum"] = 0.0
        return encoded

    time_dt = pd.to_datetime(encoded["time"])
    minutes_from_midnight = time_dt.dt.hour * 60 + time_dt.dt.minute
    min_val = minutes_from_midnight.min()
    max_val = minutes_from_midnight.max()

    encoded["session_time_pos"] = (minutes_from_midnight - min_val) / (
        max_val - min_val + 1e-9
    )
    if "basis_pct" in encoded.columns:
        encoded["eod_basis_momentum"] = (
            encoded["basis_pct"].diff(3) * encoded["session_time_pos"]
        )
    else:
        encoded["eod_basis_momentum"] = 0.0
    return encoded


def prepare_intraday_thermodynamics(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    spot_1h: pd.DataFrame | None = None,
    spot_1d: pd.DataFrame | None = None,
    spot_1w: pd.DataFrame | None = None,
    symbol: str | None = None,
    logger=print,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Centralizes the intraday thermodynamic contract used by training, backtest,
    live inference, and meta-strategy selection.

    Rules:
      - 1H FUT bars are the primary execution timeline.
      - Every 1H FUT bar must have an exact 1H SPOT match if SPOT is provided.
      - Extra SPOT-only bars are informational and ignored.
      - 1D/1W basis layers keep the existing backward-ASOF behavior.
      - Session vectors are always encoded in exactly one place.
    """
    df_1h = df_1h.sort_values("time").reset_index(drop=True).copy()
    df_1d = (
        df_1d.sort_values("time").reset_index(drop=True).copy()
        if df_1d is not None
        else None
    )
    df_1w = (
        df_1w.sort_values("time").reset_index(drop=True).copy()
        if df_1w is not None
        else None
    )
    df_1m = (
        df_1m.sort_values("time").reset_index(drop=True).copy()
        if df_1m is not None
        else None
    )
    spot_1h = (
        spot_1h.sort_values("time").reset_index(drop=True).copy()
        if spot_1h is not None and not spot_1h.empty
        else None
    )
    spot_1d = (
        spot_1d.sort_values("time").reset_index(drop=True).copy()
        if spot_1d is not None and not spot_1d.empty
        else None
    )
    spot_1w = (
        spot_1w.sort_values("time").reset_index(drop=True).copy()
        if spot_1w is not None and not spot_1w.empty
        else None
    )

    label = symbol or "TARGET"
    if spot_1h is not None:
        coverage = analyze_reference_coverage(df_1h, spot_1h)
        if coverage["missing_reference_count"] > 0:
            if logger is not None:
                logger("\n" + "=" * 70)
                logger("  [!] FATAL ERROR: REFERENCE COVERAGE VIOLATION")
                logger("=" * 70)
                logger(
                    f"  SPOT support is missing for {coverage['missing_reference_count']} tradable 1H FUT bars in {label}."
                )
                logger(
                    f"  Sample missing timestamps: {coverage['missing_reference_sample']}"
                )
                logger(
                    "  Thermodynamic state cannot be trusted on the execution timeline."
                )
                logger("  System locked to protect capital.")
            raise ValueError(f"Missing 1H SPOT coverage for {label}.")

        if logger is not None:
            extra_count = coverage["extra_reference_count"]
            if extra_count > 0:
                logger(
                    f"  [+] 1H coverage verified: every FUT bar has SPOT support. Ignoring {extra_count} SPOT-only bars outside the tradable FUT timeline."
                )
            else:
                logger(
                    "  [+] 1H coverage verified: exact SPOT <-> FUT alignment on the tradable timeline."
                )
    elif logger is not None:
        logger(
            f"  [!] WARNING: No SPOT 1H data found for {label}. Thermodynamics will be zeroed."
        )

    df_1h = inject_thermodynamic_basis(df_1h, spot_1h)
    if df_1d is not None and spot_1d is not None:
        df_1d = inject_thermodynamic_basis(df_1d, spot_1d)
    if df_1w is not None and spot_1w is not None:
        df_1w = inject_thermodynamic_basis(df_1w, spot_1w)

    df_1h = encode_session_time_vectors(df_1h)
    return df_1h, df_1d, df_1w, df_1m


ARTIFACT_NAME_SCHEME = {
    "1H": {
        "model": "{prefix}_1H_model.pkl",
        "features": "{prefix}_1H_features.txt",
        "oos_proba": "{prefix}_1H_oos_proba.pkl",
        "trade_plan_models": "{prefix}_1H_trade_plan_models.pkl",
        "ml_report": "{prefix}_1H_ml_report.png",
        "backtest_report": "{prefix}_1H_backtest_report.png",
    },
    "1D": {
        "model": "{prefix}_1D_model.pkl",
        "features": "{prefix}_1D_features.txt",
        "oos_proba": "{prefix}_1D_oos_proba.pkl",
        "trade_plan_models": "{prefix}_1D_trade_plan_models.pkl",
        "ml_report": "{prefix}_1D_ml_report.png",
        "backtest_report": "{prefix}_1D_backtest_report.png",
    },
}


LEGACY_ARTIFACT_NAME_SCHEME = {
    "1H": {
        "model": "{prefix}_ultimate_model.pkl",
        "features": "{prefix}_ultimate_features.txt",
        "oos_proba": "{prefix}_oos_proba.pkl",
        "trade_plan_models": "{prefix}_trade_plan_models.pkl",
        "ml_report": "{prefix}_ml_report_ultimate.png",
        "backtest_report": "{prefix}_backtest_report.png",
    },
    "1D": {
        "model": "{prefix}_daily_model.pkl",
        "features": "{prefix}_daily_features.txt",
        "oos_proba": "{prefix}_daily_oos_proba.pkl",
        "trade_plan_models": "{prefix}_daily_trade_plan_models.pkl",
        "ml_report": "{prefix}_daily_ml_report.png",
        "backtest_report": "{prefix}_daily_backtest_report.png",
    },
}


def get_artifact_paths(
    symbol_dir: str, file_prefix: str, timeframe: str, legacy: bool = False
) -> dict[str, str]:
    scheme = LEGACY_ARTIFACT_NAME_SCHEME if legacy else ARTIFACT_NAME_SCHEME
    tf_key = timeframe.upper()
    if tf_key not in scheme:
        raise ValueError(f"Unsupported artifact timeframe: {timeframe}")
    return {
        key: os.path.join(symbol_dir, template.format(prefix=file_prefix))
        for key, template in scheme[tf_key].items()
    }


def resolve_artifact_path(
    symbol_dir: str, file_prefix: str, timeframe: str, artifact_key: str
) -> str:
    current_paths = get_artifact_paths(symbol_dir, file_prefix, timeframe)
    if os.path.exists(current_paths[artifact_key]):
        return current_paths[artifact_key]

    legacy_paths = get_artifact_paths(symbol_dir, file_prefix, timeframe, legacy=True)
    if os.path.exists(legacy_paths[artifact_key]):
        return legacy_paths[artifact_key]

    return current_paths[artifact_key]


def migrate_legacy_artifacts(
    symbol_dir: str, file_prefix: str, timeframe: str, logger=print
) -> dict[str, str]:
    current_paths = get_artifact_paths(symbol_dir, file_prefix, timeframe)
    legacy_paths = get_artifact_paths(symbol_dir, file_prefix, timeframe, legacy=True)

    for artifact_key, current_path in current_paths.items():
        legacy_path = legacy_paths[artifact_key]
        if current_path == legacy_path:
            continue
        if os.path.exists(legacy_path) and not os.path.exists(current_path):
            os.replace(legacy_path, current_path)
            if logger is not None:
                logger(
                    f"  [Artifacts] Migrated {os.path.basename(legacy_path)} -> "
                    f"{os.path.basename(current_path)}"
                )

    return current_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal ML Direction Predictor")
    parser.add_argument(
        "--outdir",
        type=str,
        default="/home/km/Universal-ML/",
        help="Output directory for models and reports",
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
    file_prefix = SYMBOL.lower().replace(" ", "_")

    # CRITICAL: Create Symbol-Specific Artifact Vault
    SYMBOL_DIR = os.path.join(DATA_DIR, SYMBOL)
    os.makedirs(SYMBOL_DIR, exist_ok=True)
    artifact_paths_1h = migrate_legacy_artifacts(SYMBOL_DIR, file_prefix, "1H")

    print("=" * 70)
    print(f"  INITIATING DATABASE UPLINK FOR: {SYMBOL}")
    print("=" * 70)

    # Connect to Bridge
    import sys

    sys.path.append(os.path.join(DATA_DIR, "data_vault"))
    try:
        from inference_bridge import InferenceBridge
    except ImportError:
        print("  [!] FATAL: Cannot locate inference_bridge.py in data_vault directory.")
        exit()

    bridge = InferenceBridge(db_path=os.path.join(DATA_DIR, "data_vault", "ohlcv.db"))

    # Fetch Holographic Stacks
    tf_maps = {
        "FUT": bridge.fetch_holographic_stack(SYMBOL, "FUT"),
        "SPOT": bridge.fetch_holographic_stack(SYMBOL, "SPOT"),
    }

    # Enforce Derivative-First Architecture
    df_1h = tf_maps["FUT"].get("1H")
    df_1d = tf_maps["FUT"].get("1D")
    df_1w = tf_maps["FUT"].get("1W")
    df_1m = tf_maps["FUT"].get("1M")

    if df_1h is None or df_1h.empty:
        print(
            f"\n  [!] FATAL: No 1H FUT data for {SYMBOL} in database. Derivative execution layer missing."
        )
        exit()

    print(
        f"  1H bars : {len(df_1h):>7}  ({df_1h['time'].min().date()} → {df_1h['time'].max().date()})"
    )
    if df_1d is not None and not df_1d.empty:
        print(
            f"  1D bars : {len(df_1d):>7}  ({df_1d['time'].min().date()} → {df_1d['time'].max().date()})"
        )
    if df_1w is not None and not df_1w.empty:
        print(
            f"  1W bars : {len(df_1w):>7}  ({df_1w['time'].min().date()} → {df_1w['time'].max().date()})"
        )
    if df_1m is not None and not df_1m.empty:
        print(
            f"  1M bars : {len(df_1m):>7}  ({df_1m['time'].min().date()} → {df_1m['time'].max().date()})"
        )

    print("\n  [TOON v4.0] Building Holographic Feature Engine...")
    print("  Philosophy: Pure geometry. No classical indicators.")

    # ── Step 0: Build the shared intraday thermodynamic state ─────────────
    print("  [TOON] Preparing Intraday Thermodynamic State...")
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
        exit()

    print("  [TOON] Preparing Fibonacci Structural Basis (1D→a, 1W→b, 1M→c)...")
    df_1h = fib_structural_basis(
        df_1h,
        htf_frames={"1D": df_1d, "1W": df_1w, "1M": df_1m},
        pairs=[("1D", "a"), ("1W", "b"), ("1M", "c")],
    )

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
        atr_col="atr14",
        drop_unresolved=False,
    )

    # ── Step 5: Identify holographic feature columns ───────────────────────
    # Exclude: raw OHLCV, time, labelling scaffold, target and trade-plan cols.
    # atr14 is excluded here — it was labelling scaffolding only.
    NON_FEATURE_COLS = set(NON_FEATURE_COLS_SET)
    all_holo_cols = [c for c in df_full.columns if c not in NON_FEATURE_COLS]

    # Sanitize: replace inf/nan
    state_cols = ["target", "time", "close", "atr14"]
    for b_col in ["basis_pct", "basis_z_score", "basis_vel_5", "basis_vel_10"]:
        if b_col in df_full.columns:
            state_cols.append(b_col)

    df_model_ready = df_full[
        all_holo_cols
        + state_cols
        + [c for c in TRADE_PLAN_LABEL_COLS if c in df_full.columns]
    ].copy()
    for col in all_holo_cols:
        df_model_ready[col] = (
            df_model_ready[col].map(lambda x: np.nan if np.isinf(x) else x).fillna(0)
        )
    df_model_ready = df_model_ready.dropna(subset=all_holo_cols).reset_index(drop=True)

    print(f"  [TOON v4.0] Bars after labelling & cleaning : {len(df_model_ready)}")
    print(f"  [TOON v4.0] Holographic features before selection : {len(all_holo_cols)}")
    print(
        f"  [TOON v4.0] Target (UP=1) distribution : {df_model_ready['target'].mean():.1%}"
    )

    # ── Step 6: Feature selection pipeline (1800 → 40) ────────────────────
    MIN_TRAIN_BARS = 2500
    TEST_SIZE_RATIO = 0.15

    n_available = len(df_model_ready)
    if n_available < MIN_TRAIN_BARS + 100:
        print(f"\n  Error: Not enough {SYMBOL} data for walk-forward validation.")
        exit()

    feature_cols, sel_meta = feature_selection_pipeline(
        df_model_ready,
        all_holo_cols,
        walk_forward_fn=walk_forward,
        target_col="target",
        min_train_bars=MIN_TRAIN_BARS,
        test_size_ratio=TEST_SIZE_RATIO,
        n_splits=10,
    )

    print(
        f"\n  [TOON v4.0] Final feature count into walk_forward : {len(feature_cols)}"
    )

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
        splits_df = pd.DataFrame(wf_results["splits"])
        print("\n" + "=" * 70)
        print("  SUMMARY — TOON v4.0 HOLOGRAPHIC WALK-FORWARD")
        print("=" * 70)
        print(
            f"  Overall Accuracy (All Signals)      : {wf_results['overall_accuracy']:.3f}"
        )
        print(
            f"  Overall High-Conf Accuracy (>={LIVE_CONFIDENCE_THRESHOLD:.2f}) : {splits_df['acc_high_conf'].mean():.3f}"
        )
        print(
            f"  Average High-Confidence Bar %       : {splits_df['high_conf_pct'].mean():.1%}"
        )
        print(
            f"  Average Always-UP Baseline          : {wf_results['overall_baseline_up']:.3f}"
        )
        print(
            f"  Edge over Baseline                  : {wf_results['overall_accuracy'] - wf_results['overall_baseline_up']:+.3f}"
        )
        print(f"  Number of splits performed          : {len(splits_df)}")
        print(f"  Total bars used for validation      : {splits_df['test_bars'].sum()}")

        # Layer breakdown of final features
        final_feats = wf_results["feature_cols"]
        dna_n = sum(1 for c in final_feats if "_bar" in c)
        gram_n = sum(1 for c in final_feats if "_gram_" in c)
        fft_n = sum(1 for c in final_feats if "_fft_" in c)
        skel_n = sum(1 for c in final_feats if "_skel_" in c)
        conf_n = sum(1 for c in final_feats if c.startswith("mtf_conf"))
        print("\n  Feature layer breakdown in final model:")
        print(
            f"    DNA:  {dna_n:3d}  |  Grammar: {gram_n:3d}  |  Spectral: {fft_n:3d}"
            f"  |  Skeleton: {skel_n:3d}  |  Confluence: {conf_n:3d}"
        )
        if fft_n == 0:
            print(
                "  NOTE: Zero FFT features in final model. Spectral layer is "
                "provisional — not removed. Re-evaluate after first live run."
            )

        # Save final model and feature list
        import joblib

        mod_path = artifact_paths_1h["model"]
        feat_path = artifact_paths_1h["features"]
        joblib.dump(wf_results["final_model"], mod_path)
        with open(feat_path, "w") as f:
            for col in wf_results["feature_cols"]:
                f.write(f"{col}\n")
        print(f"\n  Final model saved to '{mod_path}'")

        oos_path = artifact_paths_1h["oos_proba"]
        joblib.dump(wf_results["oos_proba_map"], oos_path)
        print(
            f"  OOS proba map saved to '{oos_path}' ({len(wf_results['oos_proba_map'])} bars)"
        )

        model = wf_results["final_model"]
        feature_cols_to_use = wf_results["feature_cols"]
        # Train trade plan models ONLY on pre-OOS data (same window as final fold).
        # Training on the full dataset leaks future excursion outcomes into SL/TP sizing.
        _tp_train_end = len(df_model_ready) - int(len(df_model_ready) * TEST_SIZE_RATIO)
        trade_plan_models = train_trade_plan_models(
            df_model_ready.iloc[:_tp_train_end], feature_cols_to_use
        )
        trade_plan_path = artifact_paths_1h["trade_plan_models"]
        joblib.dump(trade_plan_models, trade_plan_path)
        print(
            f"  Trade-plan models saved to '{trade_plan_path}' ({len(trade_plan_models)} models)"
        )

        last_row = df_model_ready.iloc[-1]
        pred = predict_next_bar(
            model,
            feature_cols_to_use,
            last_row,
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
        )
        close_price = float(last_row["close"])
        # atr14 excluded from model features; use last known value for trade plan display
        atr = float(df_full["atr14"].iloc[-1]) if "atr14" in df_full.columns else 150.0
        trade_plan = predict_trade_plan(
            trade_plan_models,
            feature_cols_to_use,
            last_row.copy(),
            pred["direction"],
            atr,
        )

        print("\n" + "=" * 70)
        print(f"  {SYMBOL.upper()} 1H FORECAST (TOON v4.0 — PURE GEOMETRY)")
        print("=" * 70)
        if "time" in last_row:
            print(f"  Bar time    : {last_row['time']}")
        print(f"  Direction   : {pred['direction']}")
        print(
            f"  Confidence  : {pred['confidence']:.1%} (Regressor Score: {pred['raw_score']:.3f})"
        )
        print(f"  Signal      : {pred['signal_strength']}")
        print("----------------------------------------------------------------------")
        print(f"  Entry Price : {close_price:,.2f} (Current Close)")

        trail_str = "N/A"
        filter_note = trade_plan["note"]
        if pred["direction"] in {"UP", "DOWN"} and np.isfinite(trade_plan["sl"]):
            sl_str = f"{trade_plan['sl']:,.2f}  (ML stop {trade_plan['stop_atr']:.2f}x ATR14)"
            tp1_str = (
                f"{trade_plan['tp1']:,.2f}  (ML TP1 {trade_plan['tp1_atr']:.2f}x ATR14)"
            )
            tp2_str = (
                f"{trade_plan['tp2']:,.2f}  (ML TP2 {trade_plan['tp2_atr']:.2f}x ATR14)"
            )
            trail_str = f"{trade_plan['trail_r']:.2f}R trailing stop after TP1"
        else:
            sl_str, tp1_str, tp2_str = "N/A", "N/A", "N/A"

        if pred["signal_strength"] == "NO_TRADE":
            filter_note = (
                f"{filter_note} Filtered: confidence below "
                f"{LIVE_CONFIDENCE_THRESHOLD:.2f}"
            ).strip()
        if pred["direction"] in {"UP", "DOWN"}:
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
            "symbol": SYMBOL,
            "time": str(last_row.get("time", "N/A")),
            "dir": pred["direction"],
            "conf": f"{pred['confidence'] * 100:.1f}%",
            "signal": pred["signal_strength"],
            "entry": f"{close_price:,.2f} (Current Close)",
            "sl": sl_str,
            "tp1": tp1_str,
            "tp2": tp2_str,
            "trail": trail_str,
            "note": filter_note,
        }

        report_file = artifact_paths_1h["ml_report"]
        save_report(wf_results, report_file, pred_info=pred_info)

        fi_sorted = sorted(
            wf_results["feature_importance"].items(), key=lambda x: x[1], reverse=True
        )
        print("\n  Top 15 most predictive holographic features:")
        for i, (fname, fval) in enumerate(fi_sorted[:15]):
            print(f"    {i + 1}. {fname:<42} {fval:.2f}")

        backtest_script = os.path.join(os.path.dirname(__file__), "backtest_engine.py")
        if os.path.exists(backtest_script):
            print("\n  Launching aligned backtest report generation...")
            completed = subprocess.run(
                [
                    sys.executable,
                    backtest_script,
                    "--outdir",
                    DATA_DIR,
                    "--symbol",
                    SYMBOL,
                ],
                check=False,
            )
            if completed.returncode != 0:
                print(
                    f"  [!] backtest_engine.py exited with code {completed.returncode}"
                )
        else:
            print(f"  [!] Missing backtest engine at {backtest_script}")

    else:
        print(
            "\nWalk-forward validation did not produce results. Please check errors above."
        )

    print("\nScript finished.")
