"""
Universal ML Direction Predictor — TOON v4.0 (Pure Holographic Engine)
Geometry-only features: scale-invariant candle shape coordinates.
No classical indicators. 4 extraction layers. Multi-timeframe pyramid.
Walk-forward validated | i7-4770 / 16 GB RAM / CPU only
"""

import argparse
import json
import os
import re
import subprocess
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from numba import njit

from data_vault.symbol_identity import extract_symbol_payload, parse_symbol_payload
from instrument_registry import InstrumentIdentity, resolve_instrument_identity
from julia_bridge import (
    add_target_fast as add_target,
)
from julia_bridge import (
    compute_backtest_bar_state_fast,
    kalman_structural_engine_fast,
    narrative_context_engine_fast,
    rv_feature_engine_fast,
    smc_feature_engine_fast,
)
from julia_bridge import (
    holographic_feature_engine_fast as holographic_feature_engine,
)

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    sys.modules.setdefault("universal_ml_engine", sys.modules[__name__])

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
FINAL_FEAT_BUDGET = 40  # Max features in final model
HYPERPARAMS_PATH = Path(__file__).with_name("hyperparams.json")
# Regime gating thresholds (tunable — calibrate per-instrument if needed)
REGIME_VOV_PENALTY_THRESHOLD = 2.0
REGIME_VOV_PENALTY_VALUE = 0.03
REGIME_KF_FLAT_THRESHOLD = 0.1
REGIME_KF_FLAT_PENALTY_VALUE = 0.02
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
        "basis_pts",
        "basis_pct",
        "basis_z_score",
        "basis_vel_5",
        "basis_vel_10",
        "session_time_pos",
        "eod_basis_momentum",
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
        "realized_volatility",
    }
)
HTF_ASOF_TOLERANCE = {
    "1D": pd.Timedelta("2 days"),
    "1W": pd.Timedelta("14 days"),
    "1M": pd.Timedelta("62 days"),
    "3M": pd.Timedelta("184 days"),
    "6M": pd.Timedelta("370 days"),
    "12M": pd.Timedelta("740 days"),
}
_DEFAULT_TF_GAPS = {
    "1M": pd.Timedelta(days=31),
    "3M": pd.Timedelta(days=93),
    "6M": pd.Timedelta(days=186),
    "12M": pd.Timedelta(days=372),
}
FORECAST_STALE_THRESHOLDS = {
    "1H": pd.Timedelta(hours=36),
    "1D": pd.Timedelta(days=4),
}
PRIMARY_ASSET_CLASS = "SPOT"
POLICY_COLLECTION_THRESHOLD = 0.55
POLICY_MIN_SAMPLES = 120
POLICY_MIN_CLASS_SAMPLES = 20
POLICY_VALIDATION_RATIO = 0.25
POLICY_THRESHOLD_GRID = tuple(np.round(np.arange(0.50, 0.86, 0.05), 2))
POLICY_ARTIFACT_SCHEMA_VERSION = 2
POLICY_ARTIFACT_KIND = "opportunity_head_v2"
EXIT_SURFACE_ARTIFACT_SCHEMA_VERSION = 1
EXIT_SURFACE_ARTIFACT_KIND = "exit_surface_v1"
EXIT_SURFACE_MIN_SAMPLES = 80
EXIT_SURFACE_CONF_QUANTILES = (0.50, 0.80)
EXIT_SURFACE_TEMPLATE_CATALOG = {
    "1H": (
        {
            "id": "h1_compact",
            "stop_atr": 0.75,
            "tp1_atr": 1.00,
            "tp2_atr": 1.80,
            "trail_atr": 0.60,
        },
        {
            "id": "h1_balanced",
            "stop_atr": 0.75,
            "tp1_atr": 1.25,
            "tp2_atr": 2.25,
            "trail_atr": 0.75,
        },
        {
            "id": "h1_wide_runner",
            "stop_atr": 1.00,
            "tp1_atr": 1.25,
            "tp2_atr": 2.50,
            "trail_atr": 0.80,
        },
        {
            "id": "h1_trend",
            "stop_atr": 1.00,
            "tp1_atr": 1.50,
            "tp2_atr": 3.00,
            "trail_atr": 1.00,
        },
        {
            "id": "h1_defensive",
            "stop_atr": 1.25,
            "tp1_atr": 1.50,
            "tp2_atr": 2.50,
            "trail_atr": 1.00,
        },
        {
            "id": "h1_swing",
            "stop_atr": 1.25,
            "tp1_atr": 2.00,
            "tp2_atr": 3.50,
            "trail_atr": 1.25,
        },
    ),
    "1D": (
        {
            "id": "d1_compact",
            "stop_atr": 1.00,
            "tp1_atr": 1.50,
            "tp2_atr": 3.00,
            "trail_atr": 1.00,
        },
        {
            "id": "d1_balanced",
            "stop_atr": 1.25,
            "tp1_atr": 2.00,
            "tp2_atr": 3.50,
            "trail_atr": 1.25,
        },
        {
            "id": "d1_trend",
            "stop_atr": 1.50,
            "tp1_atr": 2.00,
            "tp2_atr": 4.00,
            "trail_atr": 1.50,
        },
        {
            "id": "d1_wide_runner",
            "stop_atr": 1.50,
            "tp1_atr": 2.50,
            "tp2_atr": 4.50,
            "trail_atr": 1.25,
        },
        {
            "id": "d1_defensive",
            "stop_atr": 2.00,
            "tp1_atr": 2.50,
            "tp2_atr": 5.00,
            "trail_atr": 1.50,
        },
    ),
}
POLICY_RISK_MULTS = (
    (0.12, 1.25),
    (0.05, 1.00),
    (0.00, 0.65),
)
POLICY_EOD_GATE_HOUR_1H = 14
POLICY_EOD_GATE_HOUR_1D = 24
SKIP_OK = 0
SKIP_NO_PREDICTION = 1
SKIP_EOD_GATE = 2
SKIP_LOW_CONFIDENCE = 3
SKIP_INVALID_ATR = 4
SKIP_SHOCK = 5
SKIP_LOW_HURST = 6


class EnsembleModel:
    """Drop-in LightGBM ensemble wrapper used by saved artifacts."""

    def __init__(self, models: list, weights: list[float] | None = None):
        self.models = models
        self.weights = (
            [w / sum(weights) for w in weights]
            if weights
            else [1.0 / len(models)] * len(models)
        )

    def predict(self, X):
        preds = np.stack([model.predict(X) for model in self.models], axis=0)
        return (preds * np.array(self.weights, dtype=float).reshape(-1, 1)).sum(axis=0)

    @property
    def feature_importances_(self):
        return np.stack(
            [model.feature_importances_ for model in self.models], axis=0
        ).mean(axis=0)


EnsembleModel.__module__ = "universal_ml_engine"


def _sorted_timeframe_copy(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    result = df.sort_values("time").reset_index(drop=True).copy()
    result.attrs.update(dict(getattr(df, "attrs", {})))
    return result


def _annotate_timeframe(df: pd.DataFrame | None, *, label: str) -> pd.DataFrame | None:
    result = _sorted_timeframe_copy(df)
    if result is None:
        return None
    result.attrs["selected_timeframe"] = label
    result.attrs["selected_asset_class"] = PRIMARY_ASSET_CLASS
    return result


def describe_selected_frame(df: pd.DataFrame | None) -> str:
    if df is None or df.empty:
        return "MISSING"
    asset_class = str(df.attrs.get("selected_asset_class", "UNKNOWN"))
    source_exchange = df.attrs.get("resolved_source_exchange")
    source_symbol = df.attrs.get("resolved_source_symbol")
    if source_exchange and source_symbol:
        return f"{asset_class} via {source_exchange}:{source_symbol}"
    return asset_class


def fetch_spot_timeframes(
    bridge,
    symbol: str,
    labels: tuple[str, ...],
    *,
    include_realized_vol: bool = False,
) -> dict[str, pd.DataFrame | None]:
    tf_map = bridge.fetch_holographic_stack(
        symbol,
        PRIMARY_ASSET_CLASS,
        include_realized_vol=include_realized_vol,
    )
    return build_spot_timeframe_selection(tf_map, labels)


def build_spot_timeframe_selection(
    tf_map: dict[str, pd.DataFrame],
    labels: tuple[str, ...],
) -> dict[str, pd.DataFrame | None]:
    frames: dict[str, pd.DataFrame | None] = {}
    for label in labels:
        frames[label] = _annotate_timeframe(tf_map.get(label), label=label)
    return frames


def _extract_primary_tf_map(
    tf_maps: dict[str, dict[str, pd.DataFrame]] | dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    if not tf_maps:
        return {}
    if "SPOT" in tf_maps:
        spot_map = tf_maps.get("SPOT")
        return spot_map if isinstance(spot_map, dict) else {}
    return tf_maps if isinstance(tf_maps, dict) else {}


def select_primary_timeframe(
    tf_maps: dict[str, dict[str, pd.DataFrame]] | dict[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame | None:
    return _annotate_timeframe(_extract_primary_tf_map(tf_maps).get(label), label=label)


def build_timeframe_selection(
    tf_maps: dict[str, dict[str, pd.DataFrame]] | dict[str, pd.DataFrame],
    labels: tuple[str, ...],
) -> tuple[dict[str, pd.DataFrame | None], dict[str, pd.DataFrame | None]]:
    primary_frames = build_spot_timeframe_selection(
        _extract_primary_tf_map(tf_maps), labels
    )
    reference_frames = {label: None for label in labels}
    return primary_frames, reference_frames


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
        symbol_payloads = message_series.apply(extract_symbol_payload).dropna()
        if not symbol_payloads.empty:
            symbol_payload = str(symbol_payloads.iloc[-1]).strip()
            try:
                symbol_str = parse_symbol_payload(symbol_payload).pair_symbol
            except ValueError:
                symbol_str = symbol_payload

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
    low_series = df["low"]
    pc = c.shift(1).fillna(c)
    tr = np.maximum(
        h - low_series,
        np.maximum(np.abs(h - pc), np.abs(low_series - pc)),
    )
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


# ─────────────────────────────────────────────
# 4. WALK-FORWARD VALIDATION (Optimized for CPU)
# ─────────────────────────────────────────────


def _build_lgbm_ensemble() -> list[lgb.LGBMRegressor]:
    """Three complementary LightGBM configurations."""
    specs = _load_hyperparams_config()["classifier_ensemble"]
    if not isinstance(specs, list) or not specs:
        raise ValueError(
            f"`classifier_ensemble` must be a non-empty list in {HYPERPARAMS_PATH}."
        )

    models: list[lgb.LGBMRegressor] = []
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, dict):
            raise ValueError(
                f"`classifier_ensemble[{idx - 1}]` must be an object in {HYPERPARAMS_PATH}."
            )
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


def _build_lgbm_regressor(alpha: float) -> lgb.LGBMRegressor:
    spec = _load_hyperparams_config()["trade_plan_regressor"]
    if not isinstance(spec, dict):
        raise ValueError(
            f"`trade_plan_regressor` must be an object in {HYPERPARAMS_PATH}."
        )

    params = dict(spec)
    params.update(
        {
            "n_jobs": MODEL_N_JOBS,
            "verbose": -1,
            "objective": "quantile",
            "alpha": alpha,
        }
    )
    return lgb.LGBMRegressor(**params)


@lru_cache(maxsize=1)
def _load_hyperparams_config() -> dict:
    try:
        with HYPERPARAMS_PATH.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Required hyperparameter config not found: {HYPERPARAMS_PATH}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse hyperparameter config {HYPERPARAMS_PATH}: {exc}"
        ) from exc

    if not isinstance(config, dict):
        raise ValueError(
            f"Hyperparameter config root must be a JSON object: {HYPERPARAMS_PATH}"
        )
    if "classifier_ensemble" not in config:
        raise KeyError(
            f"Missing `classifier_ensemble` in hyperparameter config: {HYPERPARAMS_PATH}"
        )
    if "trade_plan_regressor" not in config:
        raise KeyError(
            f"Missing `trade_plan_regressor` in hyperparameter config: {HYPERPARAMS_PATH}"
        )
    return config


def _fit_lgbm_with_inner_validation(
    X_train_full: pd.DataFrame, y_train_full: pd.Series
) -> tuple[EnsembleModel, pd.DataFrame, pd.Series]:
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

    fitted_models: list[lgb.LGBMRegressor] = []
    for model in _build_lgbm_ensemble():
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


def _safe_numeric(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def build_prob_array_from_oos_map(
    times: pd.Series | np.ndarray | list,
    oos_proba_map: dict | None,
    calibrator=None,
) -> np.ndarray:
    mapped = {
        pd.Timestamp(ts): float(prob) for ts, prob in (oos_proba_map or {}).items()
    }
    probs = np.array(
        [mapped.get(pd.Timestamp(ts), np.nan) for ts in times],
        dtype=float,
    )
    return apply_calibrator_to_prob_array(probs, calibrator)


def _policy_term_columns(lane: str) -> tuple[str, str]:
    lane_key = str(lane).upper()
    if lane_key == "1D":
        return "rv_term_1w_1d", "rv_term_1m_1d"
    return "rv_term_1d_1h", "rv_term_1w_1h"


def _build_policy_feature_record(
    row: pd.Series,
    *,
    proba_up: float,
    direction: str,
    lane: str,
    confidence_threshold: float,
) -> dict[str, float]:
    lane_key = str(lane).upper()
    rv_prefix = f"rv_{lane_key.lower()}"
    term_fast_col, term_slow_col = _policy_term_columns(lane_key)
    direction_sign = 1.0 if direction == "UP" else -1.0
    base_confidence = max(proba_up, 1.0 - proba_up)
    base_edge = abs(proba_up - 0.5) * 2.0
    kf_regime = _safe_numeric(row.get("kf_regime"))
    nc_cross_sum = _safe_numeric(row.get("nc_cross_tf_sum"))
    nc_regime_accel = _safe_numeric(row.get("nc_regime_accel"))
    basis_z = _safe_numeric(row.get("basis_z_score"))
    return {
        "policy_prob_up": float(np.clip(proba_up, 0.0, 1.0)),
        "policy_prob_down": float(np.clip(1.0 - proba_up, 0.0, 1.0)),
        "policy_base_confidence": base_confidence,
        "policy_base_edge": base_edge,
        "policy_gate_margin": base_confidence - float(confidence_threshold),
        "policy_direction_sign": direction_sign,
        "policy_basis_z": basis_z,
        "policy_basis_z_abs": abs(basis_z),
        "policy_kf_regime": kf_regime,
        "policy_abs_kf_regime": abs(kf_regime),
        "policy_signed_kf_regime": direction_sign * kf_regime,
        "policy_kf_bar_delta": _safe_numeric(row.get("kf_bar_delta")),
        "policy_kf_swing_accum": _safe_numeric(row.get("kf_swing_accum")),
        "policy_kf_net_swing_delta": _safe_numeric(row.get("kf_net_swing_delta")),
        "policy_nc_regime_streak": _safe_numeric(row.get("nc_regime_streak")),
        "policy_nc_regime_accel": nc_regime_accel,
        "policy_signed_nc_regime_accel": direction_sign * nc_regime_accel,
        "policy_nc_cum_disp": _safe_numeric(row.get("nc_cum_disp_since_flip")),
        "policy_nc_max_dd": _safe_numeric(row.get("nc_max_dd_since_flip")),
        "policy_nc_swing_count": _safe_numeric(row.get("nc_swing_count_since_flip")),
        "policy_nc_fib_age": _safe_numeric(row.get("nc_fib_range_age")),
        "policy_nc_fib_size": _safe_numeric(row.get("nc_fib_range_size")),
        "policy_nc_cross_sum": nc_cross_sum,
        "policy_nc_cross_abs": abs(
            _safe_numeric(row.get("nc_cross_tf_abs"), abs(nc_cross_sum))
        ),
        "policy_signed_nc_cross_sum": direction_sign * nc_cross_sum,
        "policy_primary_rv_z": _safe_numeric(row.get(f"{rv_prefix}_yz_z")),
        "policy_primary_rv_vov": _safe_numeric(row.get(f"{rv_prefix}_vov")),
        "policy_primary_rv_jump_ratio": _safe_numeric(
            row.get(f"{rv_prefix}_jump_ratio")
        ),
        "policy_primary_rv_range_eff": _safe_numeric(row.get(f"{rv_prefix}_range_eff")),
        "policy_term_fast": _safe_numeric(row.get(term_fast_col)),
        "policy_term_slow": _safe_numeric(row.get(term_slow_col)),
    }


def _fit_policy_ensemble(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    *,
    purge_gap: int,
) -> EnsembleModel:
    inner_val_size = max(30, int(len(X_full) * 0.20))
    inner_val_size = min(inner_val_size, len(X_full) - 1)
    if inner_val_size <= 0:
        raise ValueError("Not enough rows for policy inner validation.")

    purged_end = -(inner_val_size + int(max(purge_gap, 0)))
    if abs(purged_end) >= len(X_full):
        purged_end = 0

    X_train_inner = (
        X_full.iloc[:purged_end] if purged_end != 0 else X_full.iloc[:-inner_val_size]
    )
    y_train_inner = (
        y_full.iloc[:purged_end] if purged_end != 0 else y_full.iloc[:-inner_val_size]
    )
    X_val_inner = X_full.iloc[-inner_val_size:]
    y_val_inner = y_full.iloc[-inner_val_size:]

    fitted_models: list[lgb.LGBMRegressor] = []
    for model in _build_lgbm_ensemble():
        model.fit(
            X_train_inner,
            y_train_inner,
            eval_set=[(X_val_inner, y_val_inner)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=10000),
            ],
        )
        fitted_models.append(model)
    return EnsembleModel(fitted_models)


def _score_policy_threshold(
    candidate_df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    equity = 10000.0
    peak_equity = equity
    max_drawdown = 0.0
    total_r = 0.0
    total_trades = 0
    i = 0
    while i < len(candidate_df):
        if not np.isfinite(scores[i]) or scores[i] < threshold:
            i += 1
            continue
        trade_r = _safe_numeric(candidate_df["policy_total_r"].iat[i])
        total_r += trade_r
        equity += trade_r * (10000.0 * 0.02)
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(
            max_drawdown, (peak_equity - equity) / max(peak_equity, 1e-9)
        )
        total_trades += 1
        exit_idx = int(candidate_df["policy_exit_index"].iat[i])
        i += 1
        while (
            i < len(candidate_df)
            and int(candidate_df["policy_source_index"].iat[i]) <= exit_idx
        ):
            i += 1
    avg_r = total_r / total_trades if total_trades > 0 else float("-inf")
    return {
        "threshold": float(threshold),
        "final_equity": float(equity),
        "max_drawdown": float(max_drawdown),
        "total_trades": int(total_trades),
        "avg_r": float(avg_r),
    }


def _resolve_policy_risk_mult(policy_artifact: dict, score: float) -> float:
    for band in policy_artifact.get("risk_bands", []):
        min_score = _safe_numeric(band.get("min_score"), 1.1)
        if score >= min_score:
            return _safe_numeric(band.get("risk_mult"), 1.0)
    return 1.0


def resolve_policy_decision_threshold(policy_artifact: dict | None) -> float:
    if not policy_artifact:
        return float("nan")
    raw_value = policy_artifact.get(
        "decision_threshold",
        policy_artifact.get("deploy_threshold"),
    )
    return float(np.clip(_safe_numeric(raw_value, 0.5), 0.0, 1.0))


def describe_policy_artifact(policy_artifact: dict | None) -> str:
    if not policy_artifact or "model" not in policy_artifact:
        return "policy artifact unavailable"
    schema_version = int(policy_artifact.get("schema_version", 1))
    artifact_kind = str(policy_artifact.get("artifact_kind", "policy_artifact_v1"))
    lane = str(policy_artifact.get("lane", "UNK")).upper()
    threshold = resolve_policy_decision_threshold(policy_artifact)
    if np.isfinite(threshold):
        return (
            f"{artifact_kind} v{schema_version} [lane={lane} threshold={threshold:.2f}]"
        )
    return f"{artifact_kind} v{schema_version} [lane={lane}]"


def score_policy_artifact(
    policy_artifact: dict | None,
    row: pd.Series,
    *,
    proba_up: float,
    direction: str,
    lane: str,
    confidence_threshold: float,
) -> dict[str, float | bool]:
    if not policy_artifact or "model" not in policy_artifact:
        return {
            "score": np.nan,
            "allow_trade": True,
            "risk_mult": 1.0,
            "threshold": np.nan,
        }

    lane_key = str(lane).upper()
    artifact_lane = str(policy_artifact.get("lane", lane_key)).upper()
    if artifact_lane and artifact_lane != lane_key:
        return {
            "score": np.nan,
            "allow_trade": True,
            "risk_mult": 1.0,
            "threshold": np.nan,
        }

    feature_cols = list(policy_artifact.get("feature_cols", []))
    if not feature_cols:
        return {
            "score": np.nan,
            "allow_trade": True,
            "risk_mult": 1.0,
            "threshold": np.nan,
        }

    feature_record = _build_policy_feature_record(
        row,
        proba_up=proba_up,
        direction=direction,
        lane=lane,
        confidence_threshold=confidence_threshold,
    )
    X = pd.DataFrame(
        [{col: _safe_numeric(feature_record.get(col)) for col in feature_cols}]
    )
    raw_score = _safe_numeric(policy_artifact["model"].predict(X)[0], 0.0)
    score = float(np.clip(raw_score, 0.0, 1.0))
    threshold = resolve_policy_decision_threshold(policy_artifact)
    allow_trade = score >= threshold
    risk_mult = (
        0.0 if not allow_trade else _resolve_policy_risk_mult(policy_artifact, score)
    )
    return {
        "score": score,
        "allow_trade": bool(allow_trade),
        "risk_mult": float(risk_mult),
        "threshold": threshold,
    }


def apply_policy_artifact_to_prediction(
    pred: dict,
    policy_artifact: dict | None,
    row: pd.Series,
    *,
    lane: str,
    confidence_threshold: float,
) -> dict:
    updated = dict(pred)
    verdict = score_policy_artifact(
        policy_artifact,
        row,
        proba_up=float(updated.get("calibrated_score", updated.get("raw_score", 0.5))),
        direction=str(updated.get("direction", "UP")),
        lane=lane,
        confidence_threshold=confidence_threshold,
    )
    updated["policy_score"] = verdict["score"]
    updated["policy_threshold"] = verdict["threshold"]
    updated["policy_risk_mult"] = verdict["risk_mult"]
    updated["policy_allowed"] = verdict["allow_trade"]
    updated["policy_filtered"] = False
    updated["policy_artifact_kind"] = (
        str(policy_artifact.get("artifact_kind", "policy_artifact_v1"))
        if policy_artifact
        else "none"
    )
    updated["policy_schema_version"] = (
        int(policy_artifact.get("schema_version", 1)) if policy_artifact else 0
    )
    if updated.get("signal_strength") != "NO_TRADE" and not verdict["allow_trade"]:
        updated["signal_strength"] = "NO_TRADE"
        updated["policy_filtered"] = True
    return updated


def train_policy_artifact(
    df: pd.DataFrame,
    feature_cols: list[str],
    prob_array: np.ndarray,
    trade_plan_models: dict,
    exit_surface_artifact: dict | None = None,
    *,
    lane: str,
    confidence_threshold: float,
    start_idx: int = 0,
    max_hold_bars: int | None = None,
    eod_gate_hour: int | None = None,
) -> dict | None:
    if df is None or df.empty:
        return None

    lane_key = str(lane).upper()
    source_df = df.reset_index(drop=True).copy()
    probs = np.asarray(prob_array, dtype=float)
    if len(source_df) != len(probs):
        raise ValueError("Policy training frame and probability array length mismatch.")

    slice_start = int(max(start_idx, 0))
    if slice_start >= len(source_df) - 2:
        return None
    if slice_start > 0:
        source_df = source_df.iloc[slice_start:].reset_index(drop=True)
        probs = probs[slice_start:]

    if max_hold_bars is None:
        max_hold_bars = BARRIER_HORIZON_BARS if lane_key == "1H" else 10
    if eod_gate_hour is None:
        eod_gate_hour = (
            POLICY_EOD_GATE_HOUR_1H if lane_key == "1H" else POLICY_EOD_GATE_HOUR_1D
        )

    close_arr = source_df["close"].to_numpy(dtype=float)
    open_arr = source_df["open"].to_numpy(dtype=float)
    high_arr = source_df["high"].to_numpy(dtype=float)
    low_arr = source_df["low"].to_numpy(dtype=float)
    atr_arr = source_df["atr14"].to_numpy(dtype=float)
    time_arr = source_df["time"].to_numpy()
    time_dt = pd.to_datetime(time_arr)
    z_arr = (
        source_df["basis_z_score"].to_numpy(dtype=float)
        if "basis_z_score" in source_df.columns
        else np.zeros(len(source_df), dtype=float)
    )

    hour_arr = np.asarray(time_dt.hour, dtype=np.int64)
    next_hour_arr = np.full(len(hour_arr), 24, dtype=np.int64)
    if len(hour_arr) > 1:
        next_hour_arr[:-1] = hour_arr[1:]

    bar_state = compute_backtest_bar_state_fast(
        close_arr,
        probs,
        atr_arr,
        z_arr,
        next_hour_arr,
        window_h=100,
        default_hurst=0.5,
        conf_threshold=POLICY_COLLECTION_THRESHOLD,
        shock_z_abs=2.5,
        min_hurst=0.45,
        eod_gate_hour=eod_gate_hour,
    )
    skip_code_arr = np.asarray(bar_state["skip_code"], dtype=np.int64)

    rows: list[dict[str, float]] = []
    for i in range(len(source_df) - 1):
        skip_code = int(skip_code_arr[i])
        if skip_code in {
            SKIP_NO_PREDICTION,
            SKIP_EOD_GATE,
            SKIP_LOW_CONFIDENCE,
            SKIP_INVALID_ATR,
            SKIP_SHOCK,
            SKIP_LOW_HURST,
        }:
            continue

        curr_atr = atr_arr[i]
        if not np.isfinite(curr_atr) or curr_atr <= 0:
            continue

        proba_up = float(np.clip(probs[i], 0.0, 1.0))
        direction = "UP" if proba_up > 0.5 else "DOWN"
        row = source_df.iloc[i].copy()
        trade_plan = predict_trade_plan(
            trade_plan_models,
            feature_cols,
            row.copy(),
            direction,
            float(curr_atr),
            exit_surface_artifact=exit_surface_artifact,
            proba_up=proba_up,
            lane=lane_key,
        )
        stop_dist = float(
            curr_atr * _safe_numeric(trade_plan.get("stop_atr"), BARRIER_ATR_MULT)
        )
        tp1_dist = float(
            curr_atr * _safe_numeric(trade_plan.get("tp1_atr"), TP1_R_MULT)
        )
        tp2_dist = float(
            curr_atr * _safe_numeric(trade_plan.get("tp2_atr"), TP2_R_MULT)
        )
        trail_dist = float(
            curr_atr * _safe_numeric(trade_plan.get("trail_r"), TRAIL_R_MULT)
        )

        trade_path = simulate_trade_path_from_arrays(
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            time_arr,
            i + 1,
            "LONG" if direction == "UP" else "SHORT",
            stop_dist,
            tp1_dist=tp1_dist,
            tp2_dist=tp2_dist,
            trail_dist=trail_dist,
            horizon=max_hold_bars,
            fee_pct=EXEC_FEE_PCT,
            slippage_bps=EXEC_SLIPPAGE_BPS,
        )
        total_r = _safe_numeric(trade_path.get("total_r"), np.nan)
        if not np.isfinite(total_r):
            continue

        feature_record = _build_policy_feature_record(
            row,
            proba_up=proba_up,
            direction=direction,
            lane=lane_key,
            confidence_threshold=confidence_threshold,
        )
        feature_record.update(
            {
                "policy_label": 1.0 if total_r > 0.0 else 0.0,
                "policy_total_r": total_r,
                "policy_source_index": float(i),
                "policy_exit_index": float(trade_path.get("exit_idx", i)),
            }
        )
        rows.append(feature_record)

    candidate_df = pd.DataFrame(rows)
    if len(candidate_df) < POLICY_MIN_SAMPLES:
        return None

    positive_count = int(candidate_df["policy_label"].sum())
    negative_count = int(len(candidate_df) - positive_count)
    if (
        positive_count < POLICY_MIN_CLASS_SAMPLES
        or negative_count < POLICY_MIN_CLASS_SAMPLES
    ):
        return None

    policy_feature_cols = [
        col
        for col in candidate_df.columns
        if col.startswith("policy_")
        and col
        not in {
            "policy_label",
            "policy_total_r",
            "policy_source_index",
            "policy_exit_index",
        }
    ]

    val_size = max(40, int(len(candidate_df) * POLICY_VALIDATION_RATIO))
    val_size = min(val_size, len(candidate_df) - 60)
    if val_size < 20:
        return None

    train_df = candidate_df.iloc[:-val_size].copy()
    val_df = candidate_df.iloc[-val_size:].copy()
    if len(train_df) < 60:
        return None

    fitted_models: list[lgb.LGBMRegressor] = []
    X_train = train_df[policy_feature_cols]
    y_train = train_df["policy_label"]
    X_val = val_df[policy_feature_cols]
    y_val = val_df["policy_label"]
    for model in _build_lgbm_ensemble():
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="mae",
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=10000),
            ],
        )
        fitted_models.append(model)
    provisional_model = EnsembleModel(fitted_models)
    val_scores = np.clip(provisional_model.predict(X_val), 0.0, 1.0)

    finite_val_scores = val_scores[np.isfinite(val_scores)]
    quantile_thresholds = []
    if len(finite_val_scores) > 0:
        quantile_thresholds = np.quantile(
            finite_val_scores,
            [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.92],
        ).tolist()
    threshold_grid = sorted(
        {
            *[float(x) for x in POLICY_THRESHOLD_GRID],
            *[float(x) for x in quantile_thresholds],
        }
    )

    min_eval_trades = max(5, min(20, len(val_df) // 12))
    all_evals: list[dict[str, float]] = []
    best_eval: dict[str, float] | None = None
    for threshold in threshold_grid:
        eval_metrics = _score_policy_threshold(val_df, val_scores, float(threshold))
        all_evals.append(eval_metrics)
        if eval_metrics["total_trades"] < min_eval_trades:
            continue
        if best_eval is None:
            best_eval = eval_metrics
            continue
        current_rank = (
            eval_metrics["final_equity"],
            eval_metrics["avg_r"],
            -eval_metrics["max_drawdown"],
            eval_metrics["total_trades"],
        )
        best_rank = (
            best_eval["final_equity"],
            best_eval["avg_r"],
            -best_eval["max_drawdown"],
            best_eval["total_trades"],
        )
        if current_rank > best_rank:
            best_eval = eval_metrics

    if best_eval is None:
        fallback_candidates = [
            item for item in all_evals if int(item["total_trades"]) > 0
        ]
        if fallback_candidates:
            best_eval = max(
                fallback_candidates,
                key=lambda item: (
                    item["final_equity"],
                    item["avg_r"],
                    -item["max_drawdown"],
                    item["total_trades"],
                ),
            )

    deploy_threshold = float(best_eval["threshold"]) if best_eval is not None else 0.50
    risk_bands = [
        {
            "min_score": float(min(1.0, deploy_threshold + uplift)),
            "risk_mult": float(mult),
        }
        for uplift, mult in POLICY_RISK_MULTS
    ]
    risk_bands.sort(key=lambda item: item["min_score"], reverse=True)

    final_model = _fit_policy_ensemble(
        candidate_df[policy_feature_cols],
        candidate_df["policy_label"],
        purge_gap=max_hold_bars,
    )

    return {
        "schema_version": POLICY_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": POLICY_ARTIFACT_KIND,
        "model": final_model,
        "feature_cols": policy_feature_cols,
        "decision_threshold": deploy_threshold,
        "deploy_threshold": deploy_threshold,
        "risk_bands": risk_bands,
        "lane": lane_key,
        "collection_threshold": float(POLICY_COLLECTION_THRESHOLD),
        "metadata": {
            "threshold_scope": "symbol_lane_local",
            "base_confidence_threshold": float(confidence_threshold),
            "source_rows": int(len(source_df)),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "candidate_rows": int(len(candidate_df)),
            "candidate_rate": float(len(candidate_df) / max(len(source_df), 1)),
            "positive_rate": float(candidate_df["policy_label"].mean()),
            "validation_avg_r": (
                float(best_eval["avg_r"]) if best_eval is not None else None
            ),
            "validation_final_equity": (
                float(best_eval["final_equity"]) if best_eval is not None else None
            ),
            "validation_max_drawdown": (
                float(best_eval["max_drawdown"]) if best_eval is not None else None
            ),
            "validation_active_rate": (
                float(best_eval["total_trades"] / max(len(val_df), 1))
                if best_eval is not None
                else None
            ),
            "validation_trades": (
                int(best_eval["total_trades"]) if best_eval is not None else 0
            ),
        },
    }


def train_trade_plan_models(df: pd.DataFrame, feature_cols: list) -> dict:
    specs = {
        "up_stop_atr": (
            1,
            "long_mae_atr",
            0.40,
        ),  # Tightened: 40th percentile (Pro-Trader SL)
        "up_tp1_atr": (1, "long_mfe_atr", 0.55),
        "up_tp2_atr": (1, "long_mfe_atr", 0.75),
        "down_stop_atr": (0, "short_mae_atr", 0.40),
        "down_tp1_atr": (0, "short_mfe_atr", 0.55),
        "down_tp2_atr": (0, "short_mfe_atr", 0.75),
    }
    models = {}
    for key, (target_value, label_col, alpha) in specs.items():
        directional_mask = (
            df["target"] > 0.5 if target_value == 1 else df["target"] < 0.5
        )
        train_df = df[directional_mask].dropna(subset=feature_cols + [label_col]).copy()
        min_samples = max(100, min(300, int(len(df) * 0.05)))
        if len(train_df) < min_samples:
            print(
                f"    [Trade Plan] {key}: {len(train_df)} samples < {min_samples}. Skipping."
            )
            continue

        inner_val_size = max(30, int(len(train_df) * 0.15))
        purge_gap = BARRIER_HORIZON_BARS
        purged_end = -(inner_val_size + purge_gap)
        if abs(purged_end) >= len(train_df):
            purged_end = 0

        if purged_end != 0:
            X_tr = train_df[feature_cols].iloc[:purged_end]
            y_tr = train_df[label_col].iloc[:purged_end]
        else:
            X_tr = train_df[feature_cols].iloc[:-inner_val_size]
            y_tr = train_df[label_col].iloc[:-inner_val_size]
        X_val = train_df[feature_cols].iloc[-inner_val_size:]
        y_val = train_df[label_col].iloc[-inner_val_size:]

        model = _build_lgbm_regressor(alpha)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            callbacks=[
                lgb.early_stopping(stopping_rounds=40, verbose=False),
                lgb.log_evaluation(period=10000),
            ],
        )
        models[key] = model
    return models


def _exit_surface_bucket(confidence: float, bucket_edges: list[float]) -> int:
    bucket = 0
    for edge in bucket_edges:
        if confidence < edge:
            return bucket
        bucket += 1
    return bucket


def _select_exit_surface_template(
    exit_surface_artifact: dict | None,
    *,
    direction: str,
    proba_up: float | None,
    lane: str,
) -> dict | None:
    if not exit_surface_artifact:
        return None
    lane_key = str(lane).upper()
    artifact_lane = str(exit_surface_artifact.get("lane", lane_key)).upper()
    if artifact_lane != lane_key:
        return None
    selection_map = exit_surface_artifact.get("selection_map", {})
    if not isinstance(selection_map, dict) or not selection_map:
        return None
    confidence = (
        max(float(proba_up), 1.0 - float(proba_up))
        if proba_up is not None and np.isfinite(proba_up)
        else 0.5
    )
    bucket_edges = [
        float(edge)
        for edge in exit_surface_artifact.get("confidence_bucket_edges", [])
        if np.isfinite(edge)
    ]
    bucket = _exit_surface_bucket(confidence, bucket_edges)
    key = f"{direction}|{bucket}"
    entry = (
        selection_map.get(key)
        or selection_map.get(f"{direction}|default")
        or selection_map.get("ANY|default")
    )
    if not isinstance(entry, dict):
        return None
    template = entry.get("template")
    return dict(template) if isinstance(template, dict) else None


def train_exit_surface_artifact(
    df: pd.DataFrame,
    prob_array: np.ndarray,
    *,
    lane: str,
    start_idx: int = 0,
    max_hold_bars: int | None = None,
    eod_gate_hour: int | None = None,
) -> dict | None:
    if df is None or df.empty:
        return None

    lane_key = str(lane).upper()
    source_df = df.reset_index(drop=True).copy()
    probs = np.asarray(prob_array, dtype=float)
    if len(source_df) != len(probs):
        raise ValueError("Exit-surface frame and probability array length mismatch.")

    slice_start = int(max(start_idx, 0))
    if slice_start >= len(source_df) - 2:
        return None
    if slice_start > 0:
        source_df = source_df.iloc[slice_start:].reset_index(drop=True)
        probs = probs[slice_start:]

    if max_hold_bars is None:
        max_hold_bars = BARRIER_HORIZON_BARS if lane_key == "1H" else 10
    if eod_gate_hour is None:
        eod_gate_hour = (
            POLICY_EOD_GATE_HOUR_1H if lane_key == "1H" else POLICY_EOD_GATE_HOUR_1D
        )

    catalog = [dict(item) for item in EXIT_SURFACE_TEMPLATE_CATALOG.get(lane_key, ())]
    if not catalog:
        return None

    close_arr = source_df["close"].to_numpy(dtype=float)
    open_arr = source_df["open"].to_numpy(dtype=float)
    high_arr = source_df["high"].to_numpy(dtype=float)
    low_arr = source_df["low"].to_numpy(dtype=float)
    atr_arr = source_df["atr14"].to_numpy(dtype=float)
    time_arr = source_df["time"].to_numpy()
    time_dt = pd.to_datetime(time_arr)
    z_arr = (
        source_df["basis_z_score"].to_numpy(dtype=float)
        if "basis_z_score" in source_df.columns
        else np.zeros(len(source_df), dtype=float)
    )

    hour_arr = np.asarray(time_dt.hour, dtype=np.int64)
    next_hour_arr = np.full(len(hour_arr), 24, dtype=np.int64)
    if len(hour_arr) > 1:
        next_hour_arr[:-1] = hour_arr[1:]

    bar_state = compute_backtest_bar_state_fast(
        close_arr,
        probs,
        atr_arr,
        z_arr,
        next_hour_arr,
        window_h=100,
        default_hurst=0.5,
        conf_threshold=POLICY_COLLECTION_THRESHOLD,
        shock_z_abs=2.5,
        min_hurst=0.45,
        eod_gate_hour=eod_gate_hour,
    )
    skip_code_arr = np.asarray(bar_state["skip_code"], dtype=np.int64)

    candidates: list[dict[str, float | int | str]] = []
    for i in range(len(source_df) - 1):
        skip_code = int(skip_code_arr[i])
        if skip_code in {
            SKIP_NO_PREDICTION,
            SKIP_EOD_GATE,
            SKIP_LOW_CONFIDENCE,
            SKIP_INVALID_ATR,
            SKIP_SHOCK,
            SKIP_LOW_HURST,
        }:
            continue

        curr_atr = float(atr_arr[i])
        if not np.isfinite(curr_atr) or curr_atr <= 0:
            continue

        proba_up = float(np.clip(probs[i], 0.0, 1.0))
        confidence = max(proba_up, 1.0 - proba_up)
        candidates.append(
            {
                "index": i,
                "proba_up": proba_up,
                "confidence": confidence,
                "direction": "UP" if proba_up > 0.5 else "DOWN",
                "atr": curr_atr,
            }
        )

    if len(candidates) < EXIT_SURFACE_MIN_SAMPLES:
        return None

    confidence_values = np.asarray(
        [item["confidence"] for item in candidates], dtype=float
    )
    bucket_edges = sorted(
        {
            float(edge)
            for edge in np.quantile(confidence_values, EXIT_SURFACE_CONF_QUANTILES)
            if np.isfinite(edge)
        }
    )

    aggregate: dict[tuple[str, str], dict[str, float]] = {}

    def _record(group_key: str, template: dict, total_r: float) -> None:
        stats = aggregate.setdefault(
            (group_key, str(template["id"])),
            {"sum_r": 0.0, "wins": 0.0, "samples": 0.0},
        )
        stats["sum_r"] += float(total_r)
        stats["wins"] += 1.0 if total_r > 0.0 else 0.0
        stats["samples"] += 1.0

    for item in candidates:
        idx = int(item["index"])
        curr_atr = float(item["atr"])
        direction = str(item["direction"])
        bucket = _exit_surface_bucket(float(item["confidence"]), bucket_edges)
        group_key = f"{direction}|{bucket}"
        default_key = f"{direction}|default"
        for template in catalog:
            trade_path = simulate_trade_path_from_arrays(
                open_arr,
                high_arr,
                low_arr,
                close_arr,
                time_arr,
                idx + 1,
                "LONG" if direction == "UP" else "SHORT",
                float(curr_atr * float(template["stop_atr"])),
                tp1_dist=float(curr_atr * float(template["tp1_atr"])),
                tp2_dist=float(curr_atr * float(template["tp2_atr"])),
                trail_dist=float(curr_atr * float(template["trail_atr"])),
                horizon=max_hold_bars,
                fee_pct=EXEC_FEE_PCT,
                slippage_bps=EXEC_SLIPPAGE_BPS,
            )
            total_r = _safe_numeric(trade_path.get("total_r"), np.nan)
            if not np.isfinite(total_r):
                continue
            _record(group_key, template, total_r)
            _record(default_key, template, total_r)

    selection_map: dict[str, dict[str, object]] = {}
    min_group_samples = max(8, min(24, len(candidates) // 10))
    for group_key in sorted({group for group, _template_id in aggregate.keys()}):
        template_rows = []
        for template in catalog:
            stats = aggregate.get((group_key, str(template["id"])))
            if not stats or stats["samples"] < min_group_samples:
                continue
            avg_r = float(stats["sum_r"] / max(stats["samples"], 1.0))
            win_rate = float(stats["wins"] / max(stats["samples"], 1.0))
            template_rows.append(
                (
                    avg_r,
                    win_rate,
                    int(stats["samples"]),
                    dict(template),
                )
            )
        if not template_rows:
            continue
        template_rows.sort(reverse=True)
        avg_r, win_rate, samples, template = template_rows[0]
        selection_map[group_key] = {
            "template_id": str(template["id"]),
            "template": template,
            "samples": samples,
            "avg_r": avg_r,
            "win_rate": win_rate,
        }

    if not selection_map:
        return None

    return {
        "schema_version": EXIT_SURFACE_ARTIFACT_SCHEMA_VERSION,
        "artifact_kind": EXIT_SURFACE_ARTIFACT_KIND,
        "lane": lane_key,
        "template_catalog": catalog,
        "confidence_bucket_edges": bucket_edges,
        "selection_map": selection_map,
        "metadata": {
            "source_rows": int(len(source_df)),
            "candidate_rows": int(len(candidates)),
            "candidate_rate": float(len(candidates) / max(len(source_df), 1)),
            "bucket_count": int(len(bucket_edges) + 1),
            "template_count": int(len(catalog)),
            "selection_keys": int(len(selection_map)),
        },
    }


def predict_trade_plan(
    plan_models: dict,
    feature_cols: list,
    latest_row: pd.Series,
    direction: str,
    atr_value: float,
    *,
    exit_surface_artifact: dict | None = None,
    proba_up: float | None = None,
    lane: str = "1H",
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

    template = _select_exit_surface_template(
        exit_surface_artifact,
        direction=direction,
        proba_up=proba_up,
        lane=lane,
    )
    if template is not None:
        stop_atr = float(
            np.clip(_safe_numeric(template.get("stop_atr"), fallback_r), 0.35, 2.50)
        )
        tp1_atr = float(
            np.clip(_safe_numeric(template.get("tp1_atr"), stop_atr), 0.50, 8.00)
        )
        tp2_atr = float(
            np.clip(_safe_numeric(template.get("tp2_atr"), tp1_atr), 1.00, 12.00)
        )
        trail_r = float(
            np.clip(_safe_numeric(template.get("trail_atr"), stop_atr), 0.30, 4.00)
        )
        note = f"Exit surface template: {template.get('id', 'unknown')}."
    elif all(key in plan_models for key in required):
        stop_atr = float(np.clip(plan_models[required[0]].predict(X)[0], 0.35, 1.50))
        tp1_atr = float(np.clip(plan_models[required[1]].predict(X)[0], 0.50, 4.00))
        tp2_atr = float(np.clip(plan_models[required[2]].predict(X)[0], 1.00, 7.00))

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


def fold_consensus_feature_selection(
    df: pd.DataFrame,
    candidate_feature_cols: list,
    n_splits: int = 10,
    min_train_bars: int = 2000,
    test_size_ratio: float = 0.15,
    purge_gap: int = 24,
    consensus_threshold: float = 0.50,
    target_col: str = "target",
) -> tuple[list, dict]:
    """
    Nested feature selection: select features inside each walk-forward fold
    using only train data. Keep features appearing in the consensus set.
    """
    from holographic_engine import correlation_filter, phase1_ranking

    df = df.dropna(
        subset=[c for c in candidate_feature_cols if c in df.columns] + [target_col]
    ).reset_index(drop=True)
    n = len(df)

    if n < min_train_bars:
        print(f"  [Nested] Not enough data. Need {min_train_bars}, got {n}.")
        return [], {}

    split_points = []
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
        print("  [Nested] No valid split points.")
        return [], {}

    valid_candidates = [c for c in candidate_feature_cols if c in df.columns]
    print(f"\n  [Nested] {len(split_points)} folds, {len(valid_candidates)} candidates")

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
            print(f"  [Nested] Fold {i + 1} feature selection failed: {exc}")
            continue

        if not fold_features:
            continue

        for feat in fold_features:
            feature_votes[feat] = feature_votes.get(feat, 0) + 1

        X_train = train_df[fold_features]
        y_train = train_df[target_col]
        try:
            model, _, _ = _fit_lgbm_with_inner_validation(X_train, y_train)
        except Exception as exc:
            print(f"  [Nested] Fold {i + 1} training failed: {exc}")
            continue

        X_test = test_df[fold_features]
        y_test = test_df[target_col]
        preds = model.predict(X_test)
        true_binary = (y_test > 0.5).astype(int)
        pred_binary = (preds > 0.5).astype(int)
        acc = float((true_binary == pred_binary).mean())
        high_conf_mask = (preds >= LIVE_CONFIDENCE_THRESHOLD) | (
            preds <= (1.0 - LIVE_CONFIDENCE_THRESHOLD)
        )
        conf_acc = (
            float((pred_binary[high_conf_mask] == true_binary[high_conf_mask]).mean())
            if high_conf_mask.sum() > 0
            else float("nan")
        )

        print(
            f"  [Nested] Fold {i + 1:>2} | Train:{len(X_train):>5} Test:{len(X_test):>5} | "
            f"Acc:{acc:.3f} ConfAcc:{conf_acc:.3f} | Feats:{len(fold_features)}"
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
        f"\n  [Nested] Consensus: {len(consensus_features)} features "
        f"(voted in >={min_votes}/{len(split_points)} folds)"
    )

    if not consensus_features:
        print("  [Nested] FATAL: No features survived consensus vote.")
        return [], {}

    print("  [Nested] Final walk-forward with consensus features...")
    final_wf = walk_forward(
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


def calibrate_oos_probabilities(
    oos_proba_map: dict,
    df: pd.DataFrame,
    target_col: str = "target",
) -> IsotonicRegression | None:
    """Fit isotonic regression on OOS predictions vs actual outcomes."""
    if not oos_proba_map or "time" not in df.columns:
        return None

    matched_raw: list[float] = []
    matched_true: list[float] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["time"])
        if ts in oos_proba_map:
            matched_raw.append(float(oos_proba_map[ts]))
            matched_true.append(1.0 if row[target_col] > 0.5 else 0.0)

    if len(matched_raw) < 100:
        print(f"  [Calibration] Only {len(matched_raw)} OOS bars — skipping.")
        return None

    raw = np.array(matched_raw, dtype=float)
    true = np.array(matched_true, dtype=float)
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(raw, true)

    cal_preds = calibrator.predict(raw)
    raw_brier = float(np.mean((raw - true) ** 2))
    cal_brier = float(np.mean((cal_preds - true) ** 2))
    print(
        f"  [Calibration] Brier: raw={raw_brier:.4f} → calibrated={cal_brier:.4f} "
        f"({len(matched_raw)} OOS bars)"
    )
    return calibrator


def apply_calibrator_to_prob_array(
    prob_array: np.ndarray,
    calibrator=None,
) -> np.ndarray:
    """Apply a saved calibrator to finite probabilities while preserving NaNs."""
    result = np.asarray(prob_array, dtype=float).copy()
    if calibrator is None:
        return result

    finite_mask = np.isfinite(result)
    if not finite_mask.any():
        return result

    result[finite_mask] = np.clip(
        calibrator.predict(result[finite_mask]),
        0.0,
        1.0,
    )
    return result


# ─────────────────────────────────────────────
# 5. REPORT CHART (Visual Enhancements)
# ─────────────────────────────────────────────


def _coerce_report_timestamp(
    value: object,
    *,
    target_tz=None,
) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    if target_tz is not None:
        try:
            ts = ts.tz_convert(target_tz)
        except (TypeError, ValueError):
            pass
    return ts


def _format_report_timestamp(
    value: object,
    *,
    target_tz=None,
) -> str:
    ts = _coerce_report_timestamp(value, target_tz=target_tz)
    if ts is None:
        return "N/A"
    return ts.strftime("%Y-%m-%d %H:%M %Z")


def build_report_data_lines(frame_map: dict[str, pd.DataFrame | None]) -> list[str]:
    lines: list[str] = []
    for label, df in frame_map.items():
        if df is None or df.empty or "time" not in df.columns:
            continue
        last_bar = _coerce_report_timestamp(df["time"].iloc[-1])
        target_tz = last_bar.tzinfo if last_bar is not None else None
        quality = (
            df.attrs.get("data_quality", {})
            if isinstance(getattr(df, "attrs", None), dict)
            else {}
        )
        status = "N/A"
        synced_at = None
        if isinstance(quality, dict):
            status = str(
                quality.get("quality_status") or quality.get("status") or "N/A"
            ).upper()
            synced_at = quality.get("synced_at")
        lines.append(
            f"DB {str(label).upper()}: "
            f"last bar {_format_report_timestamp(last_bar, target_tz=target_tz)} | "
            f"last sync {_format_report_timestamp(synced_at, target_tz=target_tz)} | "
            f"quality {status}"
        )
    return lines


def save_report(
    wf_results: dict,
    save_path: str,
    pred_info: dict = None,
    symbol: str = "UNKNOWN",
    data_update_lines: list[str] | None = None,
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

        symbol_display = pred_info.get("symbol", symbol).upper()
        forecast_label = pred_info.get(
            "forecast_label", "1H FORECAST (EXECUTION-ALIGNED)"
        )
        note_line = (
            f"  Filter Note : {pred_info['note']}\n" if pred_info.get("note") else ""
        )

        report_text = (
            f"======================================================================\n"
            f"  {symbol_display} {forecast_label}\n"
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

        ax3.barh(range(len(fnames)), fvals, color=accent_color_5, alpha=0.85)
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
        y=0.985,
    )
    if data_update_lines:
        fig.text(
            0.5,
            0.955,
            "\n".join(data_update_lines),
            color="#b0bec5",
            fontsize=9,
            fontfamily="monospace",
            ha="center",
            va="top",
        )
    top_rect = 0.93 if data_update_lines else 0.96
    plt.tight_layout(rect=[0, 0, 1, top_rect])
    plt.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"\n  Report saved → {save_path}")


# ─────────────────────────────────────────────
# 6. PREDICTION FUNCTION (live use)
# ─────────────────────────────────────────────


def predict_next_bar(
    model,
    feature_cols: list,
    row: pd.Series,
    confidence_threshold: float = LIVE_CONFIDENCE_THRESHOLD,
    calibrator=None,
    policy_artifact: dict | None = None,
    policy_lane: str = "1H",
) -> dict:
    X_inf = row[feature_cols].to_frame().T.astype(float)
    raw_score = float(np.clip(model.predict(X_inf)[0], 0.0, 1.0))
    if calibrator is not None:
        proba_up = float(np.clip(calibrator.predict([raw_score])[0], 0.0, 1.0))
    else:
        proba_up = raw_score

    direction = "UP" if proba_up > 0.5 else "DOWN"
    base_confidence = max(proba_up, 1.0 - proba_up)

    regime_penalty = 0.0
    if "rv_1h_vov" in row.index and np.isfinite(row["rv_1h_vov"]):
        if abs(float(row["rv_1h_vov"])) > REGIME_VOV_PENALTY_THRESHOLD:
            regime_penalty += REGIME_VOV_PENALTY_VALUE
    if "kf_regime" in row.index and np.isfinite(row["kf_regime"]):
        if abs(float(row["kf_regime"])) < REGIME_KF_FLAT_THRESHOLD:
            regime_penalty += REGIME_KF_FLAT_PENALTY_VALUE

    confidence = max(0.50, base_confidence - regime_penalty)

    if confidence >= confidence_threshold:
        signal = "STRONG"
    else:
        signal = "NO_TRADE"

    pred = {
        "direction": direction,
        "confidence": confidence,
        "signal_strength": signal,
        "raw_score": raw_score,
        "calibrated_score": proba_up,
        "regime_penalty": regime_penalty,
    }
    return apply_policy_artifact_to_prediction(
        pred,
        policy_artifact,
        row,
        lane=policy_lane,
        confidence_threshold=confidence_threshold,
    )


def _append_filter_note(existing_note: str, extra_note: str) -> str:
    left = str(existing_note or "").strip()
    right = str(extra_note or "").strip()
    if not right:
        return left
    if not left:
        return right
    return f"{left} {right}".strip()


def _format_age_delta(delta: pd.Timedelta) -> str:
    total_seconds = max(int(delta.total_seconds()), 0)
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("0m")
    return " ".join(parts[:2])


def forecast_staleness_note(
    bar_time,
    timeframe: str,
    *,
    now: pd.Timestamp | None = None,
) -> str:
    if bar_time is None or pd.isna(bar_time):
        return ""

    bar_ts = pd.Timestamp(bar_time)
    if bar_ts.tzinfo is None:
        bar_ts = bar_ts.tz_localize("Asia/Kolkata")
    current_ts = now if now is not None else pd.Timestamp.now(tz=bar_ts.tz)
    if current_ts.tzinfo is None:
        current_ts = current_ts.tz_localize(bar_ts.tz)
    else:
        current_ts = current_ts.tz_convert(bar_ts.tz)

    threshold = FORECAST_STALE_THRESHOLDS.get(str(timeframe).upper())
    if threshold is None:
        return ""

    age = current_ts - bar_ts
    if age <= threshold:
        return ""
    return (
        f"Stale data: latest {str(timeframe).upper()} bar is "
        f"{_format_age_delta(age)} old."
    )


def finalize_forecast_context(
    pred: dict,
    *,
    timeframe: str,
    bar_time,
    confidence_threshold: float,
    base_note: str = "",
) -> tuple[dict, str]:
    result = dict(pred)
    filter_note = str(base_note or "").strip()

    if result.get("signal_strength") == "NO_TRADE":
        filter_note = _append_filter_note(
            filter_note,
            f"Filtered: confidence below {confidence_threshold:.2f}.",
        )

    stale_note = forecast_staleness_note(bar_time, timeframe)
    if stale_note:
        result["signal_strength"] = "NO_TRADE"
        filter_note = _append_filter_note(filter_note, stale_note)

    return result, filter_note


def encode_session_time_vectors(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Adds session-position placeholders for the intraday holographic bridge.
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
    encoded["eod_basis_momentum"] = 0.0
    return encoded


def _zero_basis_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in (
        "basis_pts",
        "basis_pct",
        "basis_z_score",
        "basis_vel_5",
        "basis_vel_10",
    ):
        result[col] = 0.0
    return result


def inject_thermodynamic_basis(
    df_primary: pd.DataFrame,
    df_reference: pd.DataFrame | None = None,
    *,
    logger=None,
) -> pd.DataFrame:
    """
    Preserve the legacy helper surface while enforcing the SPOT-only contract.

    Basis-derived columns remain in the feature contract, but in SPOT-only mode
    they are neutralized to zero so saved-artifact consumers can rebuild frames
    without requiring a futures/reference lane.
    """
    primary = _sorted_timeframe_copy(df_primary)
    if primary is None:
        return df_primary.copy()

    if logger is not None and df_reference is not None and not df_reference.empty:
        logger(
            "  [Thermo] Reference basis lane ignored in SPOT-only mode; zeroing basis."
        )
    return _zero_basis_columns(primary)


def analyze_reference_coverage(
    df_primary: pd.DataFrame,
    df_reference: pd.DataFrame | None,
) -> dict[str, int | list[pd.Timestamp]]:
    """
    Compatibility helper retained for callers that still import it.

    SPOT-only runtime paths no longer require a reference lane, so a missing
    reference frame is treated as a neutral, non-fatal state.
    """
    if df_reference is None or df_reference.empty:
        return {
            "missing_reference_count": 0,
            "extra_reference_count": 0,
            "missing_reference_sample": [],
            "extra_reference_sample": [],
        }

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


def prepare_intraday_thermodynamics(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    reference_1h: pd.DataFrame | None = None,
    reference_1d: pd.DataFrame | None = None,
    reference_1w: pd.DataFrame | None = None,
    spot_1h: pd.DataFrame | None = None,
    spot_1d: pd.DataFrame | None = None,
    spot_1w: pd.DataFrame | None = None,
    symbol: str | None = None,
    logger=print,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Centralize the shared 1H preprocessing contract for all runtime consumers.

    The active project contract is SPOT-only, so any legacy reference inputs
    are accepted for compatibility but ignored.
    """
    if df_1h is None or df_1h.empty:
        label = symbol or "TARGET"
        raise ValueError(f"No usable 1H primary data for {label}.")

    reference_inputs = (
        reference_1h,
        reference_1d,
        reference_1w,
        spot_1h,
        spot_1d,
        spot_1w,
    )
    if logger is not None and any(
        df is not None and not df.empty for df in reference_inputs
    ):
        logger(
            "  [Thermo] SPOT-only mode active; skipping legacy reference-lane mechanics."
        )

    primary_1h = inject_thermodynamic_basis(df_1h, None, logger=None)
    primary_1h = encode_session_time_vectors(primary_1h)
    primary_1d = (
        inject_thermodynamic_basis(df_1d, None, logger=None)
        if df_1d is not None and not df_1d.empty
        else _sorted_timeframe_copy(df_1d)
    )
    primary_1w = (
        inject_thermodynamic_basis(df_1w, None, logger=None)
        if df_1w is not None and not df_1w.empty
        else _sorted_timeframe_copy(df_1w)
    )
    primary_1m = _sorted_timeframe_copy(df_1m)
    return primary_1h, primary_1d, primary_1w, primary_1m


ARTIFACT_NAME_SCHEME = {
    "1H": {
        "model": "{prefix}_1H_model.pkl",
        "features": "{prefix}_1H_features.txt",
        "oos_proba": "{prefix}_1H_oos_proba.pkl",
        "calibrator": "{prefix}_1H_calibrator.pkl",
        "trade_plan_models": "{prefix}_1H_trade_plan_models.pkl",
        "exit_surface": "{prefix}_1H_exit_surface.pkl",
        "policy_artifact": "{prefix}_1H_policy_artifact.pkl",
        "ml_report": "{prefix}_1H_ml_report.png",
        "backtest_report": "{prefix}_1H_backtest_report.png",
    },
    "1D": {
        "model": "{prefix}_1D_model.pkl",
        "features": "{prefix}_1D_features.txt",
        "oos_proba": "{prefix}_1D_oos_proba.pkl",
        "calibrator": "{prefix}_1D_calibrator.pkl",
        "trade_plan_models": "{prefix}_1D_trade_plan_models.pkl",
        "exit_surface": "{prefix}_1D_exit_surface.pkl",
        "policy_artifact": "{prefix}_1D_policy_artifact.pkl",
        "ml_report": "{prefix}_1D_ml_report.png",
        "backtest_report": "{prefix}_1D_backtest_report.png",
    },
}


LEGACY_ARTIFACT_NAME_SCHEME = {
    "1H": {
        "model": "{prefix}_ultimate_model.pkl",
        "features": "{prefix}_ultimate_features.txt",
        "oos_proba": "{prefix}_oos_proba.pkl",
        "calibrator": "{prefix}_1H_calibrator.pkl",
        "trade_plan_models": "{prefix}_trade_plan_models.pkl",
        "exit_surface": "{prefix}_1H_exit_surface.pkl",
        "policy_artifact": "{prefix}_1H_policy_artifact.pkl",
        "ml_report": "{prefix}_ml_report_ultimate.png",
        "backtest_report": "{prefix}_backtest_report.png",
    },
    "1D": {
        "model": "{prefix}_daily_model.pkl",
        "features": "{prefix}_daily_features.txt",
        "oos_proba": "{prefix}_daily_oos_proba.pkl",
        "calibrator": "{prefix}_1D_calibrator.pkl",
        "trade_plan_models": "{prefix}_daily_trade_plan_models.pkl",
        "exit_surface": "{prefix}_1D_exit_surface.pkl",
        "policy_artifact": "{prefix}_1D_policy_artifact.pkl",
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


def _alias_artifact_locations(
    identity: InstrumentIdentity,
    outdir: str,
    timeframe: str,
) -> list[tuple[str, dict[str, str]]]:
    locations: list[tuple[str, dict[str, str]]] = []
    for alias_symbol in identity.alias_set:
        if alias_symbol == identity.canonical_symbol:
            continue
        alias_dir = os.path.join(outdir, alias_symbol)
        if not os.path.isdir(alias_dir):
            continue
        alias_prefix = alias_symbol.lower().replace(" ", "_")
        locations.append(
            (alias_symbol, get_artifact_paths(alias_dir, alias_prefix, timeframe))
        )
        locations.append(
            (
                alias_symbol,
                get_artifact_paths(alias_dir, alias_prefix, timeframe, legacy=True),
            )
        )
    return locations


def migrate_alias_artifacts(
    outdir: str,
    identity: InstrumentIdentity,
    timeframe: str,
    logger=print,
) -> dict[str, str]:
    symbol_dir = identity.symbol_dir(outdir)
    os.makedirs(symbol_dir, exist_ok=True)
    current_paths = get_artifact_paths(symbol_dir, identity.artifact_prefix, timeframe)
    moved_any = False

    for alias_symbol, alias_paths in _alias_artifact_locations(
        identity, outdir, timeframe
    ):
        for artifact_key, alias_path in alias_paths.items():
            target_path = current_paths[artifact_key]
            if not os.path.exists(alias_path) or os.path.exists(target_path):
                continue
            os.replace(alias_path, target_path)
            moved_any = True
            if logger is not None:
                logger(
                    f"  [Artifacts] Canonicalized {timeframe} {artifact_key} "
                    f"from {alias_symbol} -> {identity.canonical_symbol}"
                )

    if moved_any:
        for alias_symbol in identity.alias_set:
            if alias_symbol == identity.canonical_symbol:
                continue
            alias_dir = os.path.join(outdir, alias_symbol)
            if os.path.isdir(alias_dir) and not os.listdir(alias_dir):
                os.rmdir(alias_dir)
                if logger is not None:
                    logger(
                        f"  [Artifacts] Removed empty alias artifact directory {alias_symbol}"
                    )

    return migrate_legacy_artifacts(
        symbol_dir,
        identity.artifact_prefix,
        timeframe,
        logger=logger,
    )


def prepare_symbol_artifact_context(
    outdir: str,
    symbol: str,
    *,
    asset_class: str = "SPOT",
    timeframes: tuple[str, ...] = ("1H", "1D"),
    logger=print,
) -> dict[str, str | InstrumentIdentity]:
    identity = resolve_instrument_identity(
        symbol,
        outdir=outdir,
        asset_class=asset_class,
    )
    symbol_dir = identity.symbol_dir(outdir)
    os.makedirs(symbol_dir, exist_ok=True)
    for timeframe in timeframes:
        migrate_alias_artifacts(outdir, identity, timeframe, logger=logger)
    return {
        "identity": identity,
        "symbol": identity.display_symbol,
        "requested_symbol": identity.request_symbol,
        "canonical_symbol": identity.canonical_symbol,
        "symbol_dir": symbol_dir,
        "file_prefix": identity.artifact_prefix,
    }


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
        help="Target symbol (e.g., NIFTY, BTCUSDT, AAPL, ^GDAXI)",
    )

    args = parser.parse_args()

    DATA_DIR = os.path.abspath(args.outdir)
    requested_symbol = args.symbol.upper()
    artifact_ctx = prepare_symbol_artifact_context(
        DATA_DIR,
        requested_symbol,
        asset_class=PRIMARY_ASSET_CLASS,
        timeframes=("1H",),
    )
    SYMBOL = str(artifact_ctx["symbol"])
    SYMBOL_DIR = str(artifact_ctx["symbol_dir"])
    file_prefix = str(artifact_ctx["file_prefix"])
    artifact_paths_1h = get_artifact_paths(SYMBOL_DIR, file_prefix, "1H")

    print("=" * 70)
    if requested_symbol != SYMBOL:
        print(
            f"  INITIATING DATABASE UPLINK FOR: {SYMBOL} [requested {requested_symbol}]"
        )
    else:
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
        "SPOT": bridge.fetch_holographic_stack(
            str(artifact_ctx["identity"].market_data_symbol),
            PRIMARY_ASSET_CLASS,
        ),
    }

    primary_frames, reference_frames = build_timeframe_selection(
        tf_maps, ("1H", "1D", "1W", "1M")
    )
    df_1h = primary_frames["1H"]
    df_1d = primary_frames["1D"]
    df_1w = primary_frames["1W"]
    df_1m = primary_frames["1M"]

    if df_1h is None or df_1h.empty:
        print(
            f"\n  [!] FATAL: No usable 1H primary data for {SYMBOL} in database. SPOT-first execution layer unavailable."
        )
        exit()

    print(f"  1H primary lane : {describe_selected_frame(df_1h)}")
    print(
        f"  1H bars : {len(df_1h):>7}  ({df_1h['time'].min().date()} → {df_1h['time'].max().date()})"
    )
    if df_1d is not None and not df_1d.empty:
        print(f"  1D primary lane : {describe_selected_frame(df_1d)}")
        print(
            f"  1D bars : {len(df_1d):>7}  ({df_1d['time'].min().date()} → {df_1d['time'].max().date()})"
        )
    if df_1w is not None and not df_1w.empty:
        print(f"  1W primary lane : {describe_selected_frame(df_1w)}")
        print(
            f"  1W bars : {len(df_1w):>7}  ({df_1w['time'].min().date()} → {df_1w['time'].max().date()})"
        )
    if df_1m is not None and not df_1m.empty:
        print(f"  1M primary lane : {describe_selected_frame(df_1m)}")
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
            reference_1h=reference_frames["1H"],
            reference_1d=reference_frames["1D"],
            reference_1w=reference_frames["1W"],
            symbol=SYMBOL,
            logger=print,
        )
    except ValueError:
        exit()

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

    # ── Step 2b: SMC institutional intent engine (42 features) ────────────
    print("  [TOON v5.2] Building SMC Feature Engine (42 institutional signals)...")
    smc_df = smc_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in smc_df.columns:
        df_full[col] = smc_df[col].values
    print(f"  [TOON v5.2] SMC features injected: {len(smc_df.columns)} columns")

    print("  [TOON v5.3] Building Kalman Structural Feature Family...")
    kf_df = kalman_structural_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in kf_df.columns:
        df_full[col] = kf_df[col].values
    print(
        f"  [TOON v5.3] Kalman structural features injected: {len(kf_df.columns)} columns"
    )

    print("  [TOON v5.4] Building Realized Volatility Surface (Julia RV engine)...")
    rv_df = rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
    print(f"  [TOON v5.4] RV features injected: {len(rv_df.columns)} columns")

    print("  [TOON v5.5] Building Narrative Context Awareness (23 context signals)...")
    nc_df = narrative_context_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in nc_df.columns:
        df_full[col] = nc_df[col].values
    print(f"  [TOON v5.5] Narrative context injected: {len(nc_df.columns)} columns")

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
    # Preserve raw OHLC state for downstream trade-plan / policy replay.
    state_cols = ["target", "time", "open", "high", "low", "close", "atr14"]
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

    # ── Step 6+7: Nested feature selection + walk-forward ─────────────────
    MIN_TRAIN_BARS = 2500
    TEST_SIZE_RATIO = 0.15

    n_available = len(df_model_ready)
    if n_available < MIN_TRAIN_BARS + 100:
        print(f"\n  Error: Not enough {SYMBOL} data for walk-forward validation.")
        exit()

    feature_cols, wf_results = fold_consensus_feature_selection(
        df_model_ready,
        all_holo_cols,
        n_splits=10,
        min_train_bars=MIN_TRAIN_BARS,
        test_size_ratio=TEST_SIZE_RATIO,
    )

    if not feature_cols or not wf_results:
        print("  [TOON] Nested feature selection failed. Aborting.")
        exit()

    print(f"\n  [TOON v6.0] Consensus feature count: {len(feature_cols)}")

    print("\n" + "=" * 70)
    print("  Walk-Forward Validation — TOON v4.0 Holographic Model")
    print("=" * 70)

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
        calibrator = calibrate_oos_probabilities(
            wf_results["oos_proba_map"], df_model_ready
        )
        if calibrator is not None:
            cal_path = artifact_paths_1h["calibrator"]
            joblib.dump(calibrator, cal_path)
            print(f"  Calibrator saved → '{cal_path}'")

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
        exit_surface_artifact = train_exit_surface_artifact(
            df_model_ready,
            prob_array_full := build_prob_array_from_oos_map(
                df_model_ready["time"],
                wf_results["oos_proba_map"],
                calibrator=calibrator,
            ),
            lane="1H",
            start_idx=_tp_train_end,
            max_hold_bars=BARRIER_HORIZON_BARS,
            eod_gate_hour=POLICY_EOD_GATE_HOUR_1H,
        )
        exit_surface_path = artifact_paths_1h["exit_surface"]
        if exit_surface_artifact is not None:
            joblib.dump(exit_surface_artifact, exit_surface_path)
            print(
                f"  Exit surface saved to '{exit_surface_path}' "
                f"({exit_surface_artifact['metadata']['candidate_rows']} candidates)"
            )
        else:
            print(
                "  [Exit Surface] Skipped 1H artifact: insufficient honest candidate trades."
            )
        policy_artifact = train_policy_artifact(
            df_model_ready,
            feature_cols_to_use,
            prob_array_full,
            trade_plan_models,
            exit_surface_artifact=exit_surface_artifact,
            lane="1H",
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
            start_idx=_tp_train_end,
            max_hold_bars=BARRIER_HORIZON_BARS,
            eod_gate_hour=POLICY_EOD_GATE_HOUR_1H,
        )
        if policy_artifact is not None:
            policy_path = artifact_paths_1h["policy_artifact"]
            joblib.dump(policy_artifact, policy_path)
            print(
                f"  Policy artifact saved to '{policy_path}' "
                f"({policy_artifact['metadata']['candidate_rows']} candidates)"
            )
        else:
            print(
                "  [Policy] Skipped 1H policy artifact: insufficient honest candidate trades."
            )

        last_row = df_model_ready.iloc[-1]
        pred = predict_next_bar(
            model,
            feature_cols_to_use,
            last_row,
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
            calibrator=calibrator,
            policy_artifact=policy_artifact,
            policy_lane="1H",
        )
        primary_asset_label = str(df_1h.attrs.get("selected_asset_class", "PRIMARY"))
        close_price = float(last_row["close"])
        # atr14 excluded from model features; use last known value for trade plan display
        atr = float(df_full["atr14"].iloc[-1]) if "atr14" in df_full.columns else 150.0
        trade_plan = predict_trade_plan(
            trade_plan_models,
            feature_cols_to_use,
            last_row.copy(),
            pred["direction"],
            atr,
            exit_surface_artifact=exit_surface_artifact,
            proba_up=float(pred.get("calibrated_score", pred.get("raw_score", np.nan))),
            lane="1H",
        )

        print("\n" + "=" * 70)
        print(f"  {SYMBOL.upper()} 1H FORECAST (TOON v4.0 — PURE GEOMETRY)")
        print("=" * 70)
        if "time" in last_row:
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
        print(
            f"  Entry Price : {close_price:,.2f} (Current {primary_asset_label} Close)"
        )

        trail_str = "N/A"
        pred, filter_note = finalize_forecast_context(
            pred,
            timeframe="1H",
            bar_time=last_row.get("time"),
            confidence_threshold=LIVE_CONFIDENCE_THRESHOLD,
            base_note=trade_plan["note"],
        )
        if pred.get("policy_filtered"):
            filter_note = f"{filter_note} POLICY_FILTERED".strip()
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
            "entry": f"{close_price:,.2f} (Current {primary_asset_label} Close)",
            "sl": sl_str,
            "tp1": tp1_str,
            "tp2": tp2_str,
            "trail": trail_str,
            "note": filter_note,
        }

        report_file = artifact_paths_1h["ml_report"]
        save_report(
            wf_results,
            report_file,
            pred_info=pred_info,
            data_update_lines=build_report_data_lines({"1H": df_1h, "1D": df_1d}),
        )

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
