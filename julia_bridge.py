"""
julia_bridge.py — TOON v5.0 Python→Julia Zero-Copy Bridge
==========================================================
Provides drop-in replacements for:
  - holographic_feature_engine()  →  holographic_feature_engine_fast()
  - add_target()                  →  add_target_fast()

Uses juliacall to drive ToonMath.jl. NumPy arrays passed as PyArray — no copy.

Usage:
    from julia_bridge import holographic_feature_engine_fast, add_target_fast

    df_with_features = holographic_feature_engine_fast(df_1h, df_1d, df_1w, df_1m)
    df_labelled      = add_target_fast(df_with_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# One-time Julia / ToonMath initialisation
# ─────────────────────────────────────────────────────────────


def _init_julia():
    """
    Load juliacall and include ToonMath.jl exactly once per process.
    The returned `jl` object is cached as a module-level singleton.

    Current production safety depends on the training and inference flows
    calling the holographic bridge once per dataset before the Python-only
    feature-selection stage. `feature_selection_pipeline()` operates on the
    precomputed DataFrame and never re-enters Julia. If a future
    multiprocessing worker pool is introduced, each worker will initialize its
    own Julia runtime and must be capacity-planned accordingly.
    """
    import pathlib

    global _jl, _ToonMath
    try:
        return _jl, _ToonMath
    except NameError:
        pass

    from juliacall import Main as jl  # type: ignore
    # Cache scope is process-local; forked workers do not share this singleton.

    # Locate ToonMath.jl next to this bridge file
    _bridge_dir = pathlib.Path(__file__).parent.resolve()
    _toon_path = _bridge_dir / "ToonMath.jl"
    if not _toon_path.exists():
        raise FileNotFoundError(f"ToonMath.jl not found at {_toon_path}")

    jl.seval(f'include("{_toon_path.as_posix()}")')
    jl.seval("using .ToonMath")

    _jl = jl
    _ToonMath = jl.ToonMath
    return _jl, _ToonMath


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────


def _to_f64(arr: np.ndarray) -> np.ndarray:
    """Ensure contiguous float64 array (zero-copy if already correct dtype+order)."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _to_i64(arr: np.ndarray) -> np.ndarray:
    """Ensure contiguous int64 array."""
    return np.ascontiguousarray(arr, dtype=np.int64)


_HTF_FIXED_CLOSE_SHIFTS = {
    "1D": pd.Timedelta(days=1),
    "1W": pd.Timedelta(weeks=1),
}
_HTF_CALENDAR_CLOSE_SHIFTS = {
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "12M": pd.DateOffset(months=12),
}


def _shift_htf_times_to_close(
    series: pd.Series,
    timeframe_label: str | None,
) -> pd.Series:
    """
    Shift HTF origin timestamps to the moment the bar becomes fully observable.

    The vault stores higher-timeframe bars at period origin. The 1H/1D lanes
    must only see a higher-timeframe bar after that bar closes, so all bridge
    alignments use close-availability timestamps rather than origin timestamps.
    """
    times = pd.to_datetime(series).reset_index(drop=True)
    label = str(timeframe_label).strip().upper() if timeframe_label is not None else ""

    fixed_shift = _HTF_FIXED_CLOSE_SHIFTS.get(label)
    if fixed_shift is not None:
        return times + fixed_shift

    calendar_shift = _HTF_CALENDAR_CLOSE_SHIFTS.get(label)
    if calendar_shift is not None:
        next_times = times.shift(-1)
        fallback = times + calendar_shift
        return next_times.where(next_times.notna(), fallback)

    return times


def _htf_close_times_ns(src: pd.DataFrame, timeframe_label: str | None) -> np.ndarray:
    return _times_to_ns(_shift_htf_times_to_close(src["time"], timeframe_label))


def _compute_atr14_array(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
) -> np.ndarray:
    """Compute ATR14 from raw numpy arrays (for HTF SMC projection).

    Uses Wilder smoothing, identical to universal_ml_engine._compute_atr14
    but operates on numpy arrays rather than DataFrames.
    """
    n = len(highs)
    atr = np.full(n, np.nan, dtype=np.float64)
    if n < period + 1:
        return np.where(np.isfinite(atr), atr, 1.0)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1]),
        ),
    )
    atr[period] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i + 1] = (atr[i] * (period - 1) + tr[i]) / period
    first_valid = np.argmax(np.isfinite(atr))
    if np.isfinite(atr[first_valid]):
        atr[:first_valid] = atr[first_valid]
    else:
        atr[:] = 1.0
    return atr


def _compute_contract_atr14_array(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> np.ndarray:
    """
    Match universal_ml_engine._compute_atr14() exactly for the Kalman contract.

    This uses the repo's canonical rolling-mean ATR14 with early bars filled by
    the raw true range, which differs from the bridge's legacy Wilder helper.
    """
    prev_close = np.empty_like(closes, dtype=np.float64)
    prev_close[0] = closes[0]
    prev_close[1:] = closes[:-1]
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - prev_close),
            np.abs(lows - prev_close),
        ),
    )
    tr_series = pd.Series(tr)
    atr = tr_series.rolling(14).mean().fillna(tr_series).to_numpy(dtype=np.float64)
    return _to_f64(atr)


def _times_to_ns(series: pd.Series) -> np.ndarray:
    """
    Convert a pandas datetime Series to int64 Unix nanoseconds (sorted ascending).
    Handles both tz-aware and tz-naive series; strips timezone after conversion.
    """
    dt = pd.to_datetime(series)
    if dt.dt.tz is not None:
        dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    ns: np.ndarray = dt.values.astype("datetime64[ns]").view(np.int64)
    return _to_i64(ns)


def _extract_ohlcv(df: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Return (o, h, l, c, v) as contiguous float64 arrays from a OHLCV DataFrame."""
    return (
        _to_f64(df["open"].to_numpy()),
        _to_f64(df["high"].to_numpy()),
        _to_f64(df["low"].to_numpy()),
        _to_f64(df["close"].to_numpy()),
        _to_f64(df["volume"].to_numpy()),
    )


_THERMO_COLS = (
    "basis_pct",
    "basis_z_score",
    "basis_vel_5",
    "basis_vel_10",
    "session_time_pos",
    "eod_basis_momentum",
)
_THERMO_NT_FIELDS = (
    "basis_pct",
    "basis_z",
    "basis_v5",
    "basis_v10",
    "session_pos",
    "eod_momentum",
)


def _build_thermo(df: pd.DataFrame, n: int):
    """
    Build thermo NamedTuple for Julia if all base thermodynamic columns exist.
    Returns None otherwise (Julia side checks for `nothing`).
    """
    if not all(c in df.columns for c in _THERMO_COLS):
        return None

    jl, _ = _init_julia()
    field_sig = ", ".join(_THERMO_NT_FIELDS)
    thermo_namedtuple = jl.seval(f"({field_sig}) -> (; {field_sig})")

    return thermo_namedtuple(
        _to_f64(df["basis_pct"].to_numpy()[:n]),
        _to_f64(df["basis_z_score"].to_numpy()[:n]),
        _to_f64(df["basis_vel_5"].to_numpy()[:n]),
        _to_f64(df["basis_vel_10"].to_numpy()[:n]),
        _to_f64(df["session_time_pos"].to_numpy()[:n]),
        _to_f64(df["eod_basis_momentum"].to_numpy()[:n]),
    )


def _jl_dict_to_df(jl_dict, index: pd.Index) -> pd.DataFrame:
    """
    Convert the Julia Dict{String,Vector{Float64}} returned by
    ToonMath.compute_holographic_features into a pandas DataFrame.
    All NaN entries are preserved as-is (no fillna here — caller decides).
    """
    py_dict: dict[str, np.ndarray] = {}
    for k in jl_dict:
        key: str = str(k)
        vec = np.array(jl_dict[k], dtype=np.float64)
        py_dict[key] = vec
    return pd.DataFrame(py_dict, index=index)


def _prepare_kalman_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    return df.sort_values("time").reset_index(drop=True).copy()


def _extract_kalman_inputs(
    df: pd.DataFrame,
    *,
    timeframe_label: str | None = None,
    align_to_close: bool = False,
) -> dict[str, np.ndarray]:
    src = _prepare_kalman_frame(df)
    if src is None:
        raise ValueError("Kalman frame is missing or empty.")

    opens = _to_f64(src["open"].to_numpy())
    highs = _to_f64(src["high"].to_numpy())
    lows = _to_f64(src["low"].to_numpy())
    closes = _to_f64(src["close"].to_numpy())
    volumes = _to_f64(np.nan_to_num(src["volume"].to_numpy(dtype=float), nan=0.0))

    if "atr14" in src.columns:
        atr = _to_f64(src["atr14"].to_numpy())
        invalid = ~np.isfinite(atr) | (atr <= 0.0)
        if np.any(invalid):
            fallback = _compute_contract_atr14_array(highs, lows, closes)
            atr = atr.copy()
            atr[invalid] = fallback[invalid]
    else:
        atr = _compute_contract_atr14_array(highs, lows, closes)

    invalid = ~np.isfinite(atr) | (atr <= 0.0)
    if np.any(invalid):
        atr = atr.copy()
        atr[invalid] = 1.0

    time_ns = (
        _htf_close_times_ns(src, timeframe_label)
        if align_to_close
        else _times_to_ns(src["time"])
    )

    return {
        "time": time_ns,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "atr": atr,
    }


def _materialize_kalman_state(state) -> dict[str, np.ndarray]:
    return {
        "kalman": np.array(state.kalman, dtype=np.float64),
        "regime": np.array(state.regime, dtype=np.float64),
        "bar_delta": np.array(state.bar_delta, dtype=np.float64),
        "swing_accum": np.array(state.swing_accum, dtype=np.float64),
        "net_swing_delta": np.array(state.net_swing_delta, dtype=np.float64),
        "swing_type": np.array(state.swing_type, dtype=np.int64),
        "swing_price": np.array(state.swing_price, dtype=np.float64),
        "swing_bar": np.array(state.swing_bar, dtype=np.int64),
        "confirm_bar": np.array(state.confirm_bar, dtype=np.int64),
        "segment_direction": np.array(state.segment_direction, dtype=np.int64),
        "segment_start_confirm_bar": np.array(
            state.segment_start_confirm_bar,
            dtype=np.int64,
        ),
        "segment_end_confirm_bar": np.array(
            state.segment_end_confirm_bar,
            dtype=np.int64,
        ),
        "segment_bar_count": np.array(state.segment_bar_count, dtype=np.int64),
        "segment_raw_delta": np.array(state.segment_raw_delta, dtype=np.float64),
        "segment_high": np.array(state.segment_high, dtype=np.float64),
        "segment_low": np.array(state.segment_low, dtype=np.float64),
        "segment_price_range": np.array(state.segment_price_range, dtype=np.float64),
        "segment_atr_ref": np.array(state.segment_atr_ref, dtype=np.float64),
        "bull_swing_delta": np.array(state.bull_swing_delta, dtype=np.float64),
        "prev_bull_swing_delta": np.array(
            state.prev_bull_swing_delta,
            dtype=np.float64,
        ),
        "bull_swing_delta_div": np.array(
            state.bull_swing_delta_div,
            dtype=np.float64,
        ),
        "bear_swing_delta": np.array(state.bear_swing_delta, dtype=np.float64),
        "prev_bear_swing_delta": np.array(
            state.prev_bear_swing_delta,
            dtype=np.float64,
        ),
        "bear_swing_delta_div": np.array(
            state.bear_swing_delta_div,
            dtype=np.float64,
        ),
        "bull_swing_delta_per_bar": np.array(
            state.bull_swing_delta_per_bar,
            dtype=np.float64,
        ),
        "prev_bull_swing_delta_per_bar": np.array(
            state.prev_bull_swing_delta_per_bar,
            dtype=np.float64,
        ),
        "bull_swing_delta_per_bar_div": np.array(
            state.bull_swing_delta_per_bar_div,
            dtype=np.float64,
        ),
        "bear_swing_delta_per_bar": np.array(
            state.bear_swing_delta_per_bar,
            dtype=np.float64,
        ),
        "prev_bear_swing_delta_per_bar": np.array(
            state.prev_bear_swing_delta_per_bar,
            dtype=np.float64,
        ),
        "bear_swing_delta_per_bar_div": np.array(
            state.bear_swing_delta_per_bar_div,
            dtype=np.float64,
        ),
        "bull_swing_delta_per_range": np.array(
            state.bull_swing_delta_per_range,
            dtype=np.float64,
        ),
        "prev_bull_swing_delta_per_range": np.array(
            state.prev_bull_swing_delta_per_range,
            dtype=np.float64,
        ),
        "bull_swing_delta_per_range_div": np.array(
            state.bull_swing_delta_per_range_div,
            dtype=np.float64,
        ),
        "bear_swing_delta_per_range": np.array(
            state.bear_swing_delta_per_range,
            dtype=np.float64,
        ),
        "prev_bear_swing_delta_per_range": np.array(
            state.prev_bear_swing_delta_per_range,
            dtype=np.float64,
        ),
        "bear_swing_delta_per_range_div": np.array(
            state.bear_swing_delta_per_range_div,
            dtype=np.float64,
        ),
        "bull_swing_delta_per_atr_range": np.array(
            state.bull_swing_delta_per_atr_range,
            dtype=np.float64,
        ),
        "prev_bull_swing_delta_per_atr_range": np.array(
            state.prev_bull_swing_delta_per_atr_range,
            dtype=np.float64,
        ),
        "bull_swing_delta_per_atr_range_div": np.array(
            state.bull_swing_delta_per_atr_range_div,
            dtype=np.float64,
        ),
        "bear_swing_delta_per_atr_range": np.array(
            state.bear_swing_delta_per_atr_range,
            dtype=np.float64,
        ),
        "prev_bear_swing_delta_per_atr_range": np.array(
            state.prev_bear_swing_delta_per_atr_range,
            dtype=np.float64,
        ),
        "bear_swing_delta_per_atr_range_div": np.array(
            state.bear_swing_delta_per_atr_range_div,
            dtype=np.float64,
        ),
        "bull_swing_delta_eff": np.array(
            state.bull_swing_delta_eff,
            dtype=np.float64,
        ),
        "prev_bull_swing_delta_eff": np.array(
            state.prev_bull_swing_delta_eff,
            dtype=np.float64,
        ),
        "bull_swing_delta_eff_div": np.array(
            state.bull_swing_delta_eff_div,
            dtype=np.float64,
        ),
        "bear_swing_delta_eff": np.array(
            state.bear_swing_delta_eff,
            dtype=np.float64,
        ),
        "prev_bear_swing_delta_eff": np.array(
            state.prev_bear_swing_delta_eff,
            dtype=np.float64,
        ),
        "bear_swing_delta_eff_div": np.array(
            state.bear_swing_delta_eff_div,
            dtype=np.float64,
        ),
    }


def _split_swings_by_type(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    swing_type = state["swing_type"]
    swing_price = state["swing_price"]
    confirm_bar = state["confirm_bar"]
    high_mask = swing_type == 1
    low_mask = swing_type == -1
    return {
        "high_price": _to_f64(swing_price[high_mask]),
        "high_confirm": _to_i64(confirm_bar[high_mask]),
        "low_price": _to_f64(swing_price[low_mask]),
        "low_confirm": _to_i64(confirm_bar[low_mask]),
    }


def _map_htf_series_to_primary(
    primary_times_ns: np.ndarray,
    htf_times_ns: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    mapped = np.zeros(len(primary_times_ns), dtype=np.float64)
    if len(htf_times_ns) == 0 or len(values) == 0:
        return mapped
    idx_map = np.searchsorted(htf_times_ns, primary_times_ns, side="right") - 1
    valid = (idx_map >= 0) & (idx_map < len(values))
    if np.any(valid):
        mapped[valid] = values[idx_map[valid]]
    return mapped


_KALMAN_BASE_STATE_FIELDS = (
    "bar_delta",
    "regime",
    "swing_accum",
    "net_swing_delta",
)
_KALMAN_DIRECTIONAL_STATE_FIELDS = (
    "bull_swing_delta",
    "prev_bull_swing_delta",
    "bull_swing_delta_div",
    "bear_swing_delta",
    "prev_bear_swing_delta",
    "bear_swing_delta_div",
    "bull_swing_delta_per_bar",
    "prev_bull_swing_delta_per_bar",
    "bull_swing_delta_per_bar_div",
    "bear_swing_delta_per_bar",
    "prev_bear_swing_delta_per_bar",
    "bear_swing_delta_per_bar_div",
    "bull_swing_delta_per_range",
    "prev_bull_swing_delta_per_range",
    "bull_swing_delta_per_range_div",
    "bear_swing_delta_per_range",
    "prev_bear_swing_delta_per_range",
    "bear_swing_delta_per_range_div",
    "bull_swing_delta_per_atr_range",
    "prev_bull_swing_delta_per_atr_range",
    "bull_swing_delta_per_atr_range_div",
    "bear_swing_delta_per_atr_range",
    "prev_bear_swing_delta_per_atr_range",
    "bear_swing_delta_per_atr_range_div",
    "bull_swing_delta_eff",
    "prev_bull_swing_delta_eff",
    "bull_swing_delta_eff_div",
    "bear_swing_delta_eff",
    "prev_bear_swing_delta_eff",
    "bear_swing_delta_eff_div",
)
_KALMAN_STATE_FIELDS = _KALMAN_BASE_STATE_FIELDS + _KALMAN_DIRECTIONAL_STATE_FIELDS
_KALMAN_PIPELINES = {
    "1H": {
        "pairs": (("a", "1M"), ("b", "1W"), ("c", "1D")),
        "mapped": ("1M", "1W", "1D"),
    },
    "1D": {
        "pairs": (("a", "6M"), ("b", "1M"), ("c", "1W")),
        "mapped": ("6M", "1M", "1W"),
    },
}

_NARRATIVE_CONTEXT_FIELDS = (
    "regime_streak",
    "regime_accel",
    "cum_disp_since_flip",
    "max_dd_since_flip",
    "swing_count_since_flip",
    "fib_range_age",
    "fib_range_size",
)

_NARRATIVE_PIPELINES = {
    "1H": {"mapped": ("1D", "1W")},
    "1D": {"mapped": ("1W", "1M")},
}


def _kalman_structural_engine(
    df_primary: pd.DataFrame,
    htf_frames: dict[str, pd.DataFrame | None],
    *,
    primary_tf: str,
) -> pd.DataFrame:
    """
    Compute the full Kalman structural feature family for the requested lane.
    """
    config = _KALMAN_PIPELINES.get(primary_tf.upper())
    if config is None:
        raise ValueError(f"Unsupported Kalman structural primary TF: {primary_tf}")

    primary = _prepare_kalman_frame(df_primary)
    if primary is None:
        return pd.DataFrame(index=df_primary.index)

    _, TM = _init_julia()
    primary_inputs = _extract_kalman_inputs(primary)
    primary_state = _materialize_kalman_state(
        TM.compute_kalman_tf_state(
            primary_inputs["open"],
            primary_inputs["high"],
            primary_inputs["low"],
            primary_inputs["close"],
            primary_inputs["volume"],
            primary_inputs["atr"],
        )
    )

    result = pd.DataFrame(index=primary.index)
    for field in _KALMAN_STATE_FIELDS:
        result[f"kf_{field}"] = primary_state[field]

    htf_states: dict[str, dict[str, np.ndarray] | None] = {}
    for label in config["mapped"]:
        frame = _prepare_kalman_frame(htf_frames.get(label))
        if frame is None:
            htf_states[label] = None
            continue
        inputs = _extract_kalman_inputs(
            frame,
            timeframe_label=label,
            align_to_close=True,
        )
        state = _materialize_kalman_state(
            TM.compute_kalman_tf_state(
                inputs["open"],
                inputs["high"],
                inputs["low"],
                inputs["close"],
                inputs["volume"],
                inputs["atr"],
            )
        )
        inputs.update(state)
        htf_states[label] = inputs

        prefix = label.lower()
        for field in _KALMAN_STATE_FIELDS:
            result[f"kf_{prefix}_{field}"] = _map_htf_series_to_primary(
                primary_inputs["time"],
                inputs["time"],
                inputs[field],
            )

    empty_f64 = np.empty(0, dtype=np.float64)
    empty_i64 = np.empty(0, dtype=np.int64)

    for slot, label in config["pairs"]:
        htf_state = htf_states.get(label)
        if htf_state is None:
            fib_jl = TM.htf_fib_observation(
                primary_inputs["open"],
                primary_inputs["high"],
                primary_inputs["low"],
                primary_inputs["close"],
                primary_inputs["atr"],
                primary_state["bar_delta"],
                primary_inputs["time"],
                empty_f64,
                empty_i64,
                empty_f64,
                empty_i64,
                empty_i64,
                slot,
            )
        else:
            split_swings = _split_swings_by_type(htf_state)
            fib_jl = TM.htf_fib_observation(
                primary_inputs["open"],
                primary_inputs["high"],
                primary_inputs["low"],
                primary_inputs["close"],
                primary_inputs["atr"],
                primary_state["bar_delta"],
                primary_inputs["time"],
                split_swings["high_price"],
                split_swings["high_confirm"],
                split_swings["low_price"],
                split_swings["low_confirm"],
                htf_state["time"],
                slot,
            )

        fib_df = _jl_dict_to_df(fib_jl, primary.index)
        for col in fib_df.columns:
            result[col] = fib_df[col].values

    ordered_cols = [
        "kf_bar_delta",
        "kf_regime",
        "kf_swing_accum",
        "kf_net_swing_delta",
    ]
    ordered_cols.extend(f"kf_{field}" for field in _KALMAN_DIRECTIONAL_STATE_FIELDS)
    for slot, _ in config["pairs"]:
        for ratio_idx in range(1, 8):
            for obs in ("o", "h", "l", "c", "delta"):
                ordered_cols.append(f"kf_{slot}_{ratio_idx}_{obs}")
    for label in config["mapped"]:
        prefix = label.lower()
        for field in _KALMAN_STATE_FIELDS:
            ordered_cols.append(f"kf_{prefix}_{field}")

    return result.reindex(columns=ordered_cols, fill_value=0.0)


def _narrative_context_engine(
    df_primary: pd.DataFrame,
    htf_frames: dict[str, pd.DataFrame | None],
    *,
    primary_tf: str,
) -> pd.DataFrame:
    """
    Compute narrative context features for the primary TF and map HTF contexts.

    Returns a DataFrame with:
      - 7 primary columns:  nc_{field}
      - 7 per HTF columns:  nc_{htf_label}_{field}
      - 2 cross-TF columns: nc_cross_tf_sum, nc_cross_tf_abs
    """
    config = _NARRATIVE_PIPELINES.get(primary_tf.upper())
    if config is None:
        return pd.DataFrame(index=df_primary.index)

    primary = _prepare_kalman_frame(df_primary)
    if primary is None:
        return pd.DataFrame(index=df_primary.index)

    _, TM = _init_julia()
    primary_inputs = _extract_kalman_inputs(primary)

    primary_state = _materialize_kalman_state(
        TM.compute_kalman_tf_state(
            primary_inputs["open"],
            primary_inputs["high"],
            primary_inputs["low"],
            primary_inputs["close"],
            primary_inputs["volume"],
            primary_inputs["atr"],
        )
    )

    split = _split_swings_by_type(primary_state)

    nc_jl = TM.compute_narrative_context(
        primary_inputs["close"],
        primary_inputs["atr"],
        primary_state["regime"],
        _to_i64(primary_state["confirm_bar"]),
        split["high_price"],
        split["high_confirm"],
        split["low_price"],
        split["low_confirm"],
    )
    nc_dict = {str(k): np.array(nc_jl[k], dtype=np.float64) for k in nc_jl}

    result = pd.DataFrame(index=primary.index)
    for field in _NARRATIVE_CONTEXT_FIELDS:
        result[f"nc_{field}"] = nc_dict.get(field, np.zeros(len(primary)))

    regime_signs = [np.sign(np.asarray(primary_state["regime"], dtype=np.float64))]

    for label in config["mapped"]:
        frame = _prepare_kalman_frame(htf_frames.get(label))
        if frame is None or len(frame) < 3:
            for field in _NARRATIVE_CONTEXT_FIELDS:
                result[f"nc_{label.lower()}_{field}"] = 0.0
            continue

        htf_inputs = _extract_kalman_inputs(
            frame,
            timeframe_label=label,
            align_to_close=True,
        )
        htf_state = _materialize_kalman_state(
            TM.compute_kalman_tf_state(
                htf_inputs["open"],
                htf_inputs["high"],
                htf_inputs["low"],
                htf_inputs["close"],
                htf_inputs["volume"],
                htf_inputs["atr"],
            )
        )

        htf_split = _split_swings_by_type(htf_state)
        htf_nc_jl = TM.compute_narrative_context(
            htf_inputs["close"],
            htf_inputs["atr"],
            htf_state["regime"],
            _to_i64(htf_state["confirm_bar"]),
            htf_split["high_price"],
            htf_split["high_confirm"],
            htf_split["low_price"],
            htf_split["low_confirm"],
        )
        htf_nc = {str(k): np.array(htf_nc_jl[k], dtype=np.float64) for k in htf_nc_jl}

        prefix = label.lower()
        for field in _NARRATIVE_CONTEXT_FIELDS:
            vals = htf_nc.get(field, np.zeros(len(frame)))
            result[f"nc_{prefix}_{field}"] = _map_htf_series_to_primary(
                primary_inputs["time"],
                htf_inputs["time"],
                vals,
            )

        htf_regime = np.asarray(htf_state["regime"], dtype=np.float64)
        mapped_sign = _map_htf_series_to_primary(
            primary_inputs["time"],
            htf_inputs["time"],
            np.sign(htf_regime),
        )
        regime_signs.append(mapped_sign)

    alignment = np.sum(regime_signs, axis=0)
    result["nc_cross_tf_sum"] = alignment
    result["nc_cross_tf_abs"] = np.abs(alignment)

    ordered_cols = [f"nc_{field}" for field in _NARRATIVE_CONTEXT_FIELDS]
    for label in config["mapped"]:
        prefix = label.lower()
        ordered_cols.extend(
            f"nc_{prefix}_{field}" for field in _NARRATIVE_CONTEXT_FIELDS
        )
    ordered_cols.extend(("nc_cross_tf_sum", "nc_cross_tf_abs"))

    return (
        result.reindex(columns=ordered_cols, fill_value=0.0)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def narrative_context_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """1H-lane narrative context: primary + 1D + 1W mapped."""
    return _narrative_context_engine(
        df_1h,
        {"1D": df_1d, "1W": df_1w, "1M": df_1m},
        primary_tf="1H",
    )


def narrative_context_engine_daily(
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_6m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """1D-lane narrative context: primary + 1W + 1M mapped."""
    return _narrative_context_engine(
        df_1d,
        {"1W": df_1w, "1M": df_1m, "6M": df_6m},
        primary_tf="1D",
    )


def compute_hurst_fast(
    close_arr: np.ndarray | pd.Series,
    window_h: int = 100,
    default_value: float = 0.5,
) -> np.ndarray:
    """
    Compute the backtest Hurst-style volatility filter in Julia.

    This is a parity-first bridge for the existing Python logic in
    `backtest_engine.run_backtest()`, not a new signal definition.
    """
    _, TM = _init_julia()
    closes = _to_f64(np.asarray(close_arr))
    result = TM.compute_hurst_series(
        closes,
        window_h=int(window_h),
        default_value=float(default_value),
    )
    return np.array(result, dtype=np.float64)


def compute_backtest_bar_state_fast(
    close_arr: np.ndarray | pd.Series,
    prob_arr: np.ndarray | pd.Series,
    atr_arr: np.ndarray | pd.Series,
    z_arr: np.ndarray | pd.Series,
    next_hour_arr: np.ndarray | pd.Series,
    window_h: int = 100,
    default_hurst: float = 0.5,
    conf_threshold: float = 0.56,
    shock_z_abs: float = 2.5,
    min_hurst: float = 0.45,
    eod_gate_hour: int = 14,
) -> dict[str, np.ndarray]:
    """
    Precompute the backtest bar-state arrays in Julia.

    This bundles the current Python gate logic into a parity-first helper so
    the Python loop can focus on trade-plan prediction and execution.
    """
    _, TM = _init_julia()
    result = TM.compute_backtest_bar_state(
        _to_f64(np.asarray(close_arr)),
        _to_f64(np.asarray(prob_arr)),
        _to_f64(np.asarray(atr_arr)),
        _to_f64(np.asarray(z_arr)),
        _to_i64(np.asarray(next_hour_arr)),
        window_h=int(window_h),
        default_hurst=float(default_hurst),
        conf_threshold=float(conf_threshold),
        shock_z_abs=float(shock_z_abs),
        min_hurst=float(min_hurst),
        eod_gate_hour=int(eod_gate_hour),
    )
    return {
        "hurst": np.array(result.hurst, dtype=np.float64),
        "confidence": np.array(result.confidence, dtype=np.float64),
        "direction_long": np.array(result.direction_long, dtype=bool),
        "skip_code": np.array(result.skip_code, dtype=np.int64),
    }


# ─────────────────────────────────────────────────────────────
# PUBLIC API: holographic_feature_engine_fast
# ─────────────────────────────────────────────────────────────


def holographic_feature_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Drop-in replacement for holographic_engine.holographic_feature_engine().

    Extracts raw arrays from pandas, converts time to int64 nanoseconds,
    dispatches to ToonMath.compute_holographic_features (zero-copy PyArray),
    and reconstructs the returned Dict into a pandas DataFrame merged
    onto df_1h.

    Parameters
    ----------
    df_1h : pd.DataFrame  — primary 1H OHLCV frame (must have 'time' column)
    df_1d : pd.DataFrame | None
    df_1w : pd.DataFrame | None
    df_1m : pd.DataFrame | None

    Returns
    -------
    pd.DataFrame  — df_1h with all holographic feature columns appended.
    """
    jl, TM = _init_julia()

    # Primary 1H arrays
    base_times_ns = _times_to_ns(df_1h["time"])
    o_1h, h_1h, l_1h, c_1h, v_1h = _extract_ohlcv(df_1h)
    thermo_1h = _build_thermo(df_1h, len(df_1h))

    # Higher-TF arrays — pass empty sentinel vectors if not provided
    def _htf(df: pd.DataFrame | None, timeframe_label: str):
        if df is None or len(df) < 3:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f
        src = df.sort_values("time").reset_index(drop=True)
        t = _htf_close_times_ns(src, timeframe_label)
        o, h, lo, c, v = _extract_ohlcv(src)
        return t, o, h, lo, c, v

    t_1d, o_1d, h_1d, l_1d, c_1d, v_1d = _htf(df_1d, "1D")
    t_1w, o_1w, h_1w, l_1w, c_1w, v_1w = _htf(df_1w, "1W")
    t_1m, o_1m, h_1m, l_1m, c_1m, v_1m = _htf(df_1m, "1M")

    # Dispatch to Julia — arrays cross the bridge as PyArray (zero-copy)
    jl_result = TM.compute_holographic_features(
        base_times_ns,
        o_1h,
        h_1h,
        l_1h,
        c_1h,
        v_1h,
        t_1d,
        o_1d,
        h_1d,
        l_1d,
        c_1d,
        v_1d,
        t_1w,
        o_1w,
        h_1w,
        l_1w,
        c_1w,
        v_1w,
        t_1m,
        o_1m,
        h_1m,
        l_1m,
        c_1m,
        v_1m,
        thermo_1h=thermo_1h if thermo_1h is not None else jl.nothing,
    )

    feat_df = _jl_dict_to_df(jl_result, df_1h.index)

    # Attach new columns only (do not overwrite existing OHLCV/time)
    new_cols = [c for c in feat_df.columns if c not in df_1h.columns]
    return pd.concat([df_1h, feat_df[new_cols]], axis=1)


def holographic_feature_engine_daily(
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_3m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Daily holographic engine. Primary=1D, higher=1W/1M/3M."""
    jl, TM = _init_julia()

    base_times_ns = _times_to_ns(df_1d["time"])
    o_1d, h_1d, l_1d, c_1d, v_1d = _extract_ohlcv(df_1d)
    thermo_1d = _build_thermo(df_1d, len(df_1d))

    def _htf(df: pd.DataFrame | None, timeframe_label: str):
        if df is None or len(df) < 3:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f
        src = df.sort_values("time").reset_index(drop=True)
        return (_htf_close_times_ns(src, timeframe_label), *_extract_ohlcv(src))

    t_1w, o_1w, h_1w, l_1w, c_1w, v_1w = _htf(df_1w, "1W")
    t_1m, o_1m, h_1m, l_1m, c_1m, v_1m = _htf(df_1m, "1M")
    t_3m, o_3m, h_3m, l_3m, c_3m, v_3m = _htf(df_3m, "3M")

    jl_result = TM.compute_holographic_features_daily(
        base_times_ns,
        o_1d,
        h_1d,
        l_1d,
        c_1d,
        v_1d,
        t_1w,
        o_1w,
        h_1w,
        l_1w,
        c_1w,
        v_1w,
        t_1m,
        o_1m,
        h_1m,
        l_1m,
        c_1m,
        v_1m,
        t_3m,
        o_3m,
        h_3m,
        l_3m,
        c_3m,
        v_3m,
        thermo_1d=thermo_1d if thermo_1d is not None else jl.nothing,
    )

    feat_df = _jl_dict_to_df(jl_result, df_1d.index)
    new_cols = [c for c in feat_df.columns if c not in df_1d.columns]
    return pd.concat([df_1d, feat_df[new_cols]], axis=1)


# ─────────────────────────────────────────────────────────────
# PUBLIC API: Kalman structural feature family
# ─────────────────────────────────────────────────────────────


def kalman_structural_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute the 1H-lane Kalman structural feature family.

    Output columns include:
    - primary: base Kalman state plus direction-aware completed swing families
    - fib observation: `kf_a_*`, `kf_b_*`, `kf_c_*` across 7 fib ratios
    - mapped HTF state: `kf_1m_*`, `kf_1w_*`, `kf_1d_*`
    """
    return _kalman_structural_engine(
        df_1h,
        {
            "1D": df_1d,
            "1W": df_1w,
            "1M": df_1m,
        },
        primary_tf="1H",
    )


def kalman_structural_engine_daily(
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_6m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute the 1D-lane Kalman structural feature family.

    Output columns include:
    - primary: base Kalman state plus direction-aware completed swing families
    - fib observation: `kf_a_*`, `kf_b_*`, `kf_c_*` across 7 fib ratios
    - mapped HTF state: `kf_6m_*`, `kf_1m_*`, `kf_1w_*`
    """
    return _kalman_structural_engine(
        df_1d,
        {
            "1W": df_1w,
            "1M": df_1m,
            "6M": df_6m,
        },
        primary_tf="1D",
    )


# ─────────────────────────────────────────────────────────────
# PUBLIC API: SMC institutional intent features
# ─────────────────────────────────────────────────────────────


def smc_feature_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute 47 SMC institutional intent features for the 1H lane.

    Layer 1: 34 primary features via ToonMath.compute_smc_features on 1H
    Layer 2: 6 HTF projection features (structure_trend + amd_phase from 1D/1W/1M)
    Layer 3: 7 confluence features (cross-TF interactions)

    Parameters
    ----------
    df_1h : pd.DataFrame — must contain OHLCV + atr14 columns
    df_1d, df_1w, df_1m : higher-TF DataFrames (optional)

    Returns
    -------
    pd.DataFrame — 47 columns indexed same as df_1h, all column names prefixed 'smc_'
    """
    jl, TM = _init_julia()

    o = _to_f64(df_1h["open"].to_numpy())
    h = _to_f64(df_1h["high"].to_numpy())
    lo = _to_f64(df_1h["low"].to_numpy())
    c = _to_f64(df_1h["close"].to_numpy())
    v = _to_f64(df_1h["volume"].to_numpy())
    a = _to_f64(df_1h["atr14"].to_numpy())

    jl_result = TM.compute_smc_features(
        o,
        h,
        lo,
        c,
        v,
        a,
        structure_window=24,
        fvg_max_age=24,
        ob_max_age=24,
        warmup=32,
    )
    smc_cols = {
        f"smc_{str(k)}": np.array(jl_result[k], dtype=np.float64) for k in jl_result
    }
    result = pd.DataFrame(smc_cols, index=df_1h.index)

    htf_pairs = [
        (df_1d, "htf1", "1D"),
        (df_1w, "htf2", "1W"),
        (df_1m, "htf3", "1M"),
    ]
    base_times_ns = _times_to_ns(df_1h["time"])

    for htf_df, htf_label, timeframe_label in htf_pairs:
        if htf_df is None or len(htf_df) < 3:
            result[f"smc_{htf_label}_structure_trend_score"] = 0.0
            result[f"smc_{htf_label}_amd_phase"] = 0.0
            continue

        src = htf_df.sort_values("time").reset_index(drop=True)
        ho = _to_f64(src["open"].to_numpy())
        hh = _to_f64(src["high"].to_numpy())
        hl = _to_f64(src["low"].to_numpy())
        hc = _to_f64(src["close"].to_numpy())
        hv = _to_f64(src["volume"].to_numpy())
        ha = _to_f64(_compute_atr14_array(hh, hl, hc))
        htf_times_ns = _htf_close_times_ns(src, timeframe_label)

        htf_jl = TM.compute_smc_htf_features(ho, hh, hl, hc, hv, ha)

        for feat_name in ["structure_trend_score", "amd_phase"]:
            htf_vals = np.array(htf_jl[feat_name], dtype=np.float64)
            result[f"smc_{htf_label}_{feat_name}"] = _map_htf_series_to_primary(
                base_times_ns,
                htf_times_ns,
                htf_vals,
            )

    result["smc_sweep_disp_sync"] = (
        (result["smc_sweep_bull_mag"] > 0) | (result["smc_sweep_bear_mag"] > 0)
    ).astype(float) * result["smc_disp_confirmed"]
    result["smc_ob_fvg_confluence"] = np.where(
        result["smc_fvg_count_active"] > 0,
        result["smc_ob_quality_score"] * (1.0 - result["smc_fvg_fill_pct"]),
        0.0,
    )
    result["smc_phase_weighted_trend"] = (
        result["smc_amd_phase"] * result["smc_structure_trend_score"]
    )
    result["smc_full_entry_signal"] = result["smc_mss_confirmed"] * np.where(
        result["smc_pd_zone"] < 0.5,
        1.0 - result["smc_pd_zone"],
        result["smc_pd_zone"],
    )
    result["smc_pd_ob_confluence"] = np.where(
        result["smc_ob_nearest_dist"] < 3.0,
        np.abs(result["smc_pd_zone"] - 0.5) * result["smc_ob_quality_score"],
        0.0,
    )
    result["smc_htf_trend_alignment"] = np.sign(
        result["smc_structure_trend_score"]
    ) * np.sign(
        result.get(
            "smc_htf1_structure_trend_score",
            result.get("smc_htf1_structure_trend", 0.0),
        )
    )
    result["smc_htf_phase_cascade"] = result["smc_amd_phase"] * result.get(
        "smc_htf1_amd_phase", 0.0
    )

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def smc_feature_engine_daily(
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_6m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute 47 SMC institutional intent features for the 1D lane.

    Same architecture as smc_feature_engine_fast but:
    - Primary TF = 1D
    - HTF1 = 1W (5x), HTF2 = 1M (4x), HTF3 = 6M (6x)

    Parameters
    ----------
    df_1d : pd.DataFrame — must contain OHLCV + atr14 columns
    df_1w, df_1m, df_6m : higher-TF DataFrames (optional)

    Returns
    -------
    pd.DataFrame — 47 columns, all prefixed 'smc_'
    """
    jl, TM = _init_julia()

    o = _to_f64(df_1d["open"].to_numpy())
    h = _to_f64(df_1d["high"].to_numpy())
    lo = _to_f64(df_1d["low"].to_numpy())
    c = _to_f64(df_1d["close"].to_numpy())
    v = _to_f64(df_1d["volume"].to_numpy())
    a = _to_f64(df_1d["atr14"].to_numpy())

    jl_result = TM.compute_smc_features(
        o,
        h,
        lo,
        c,
        v,
        a,
        structure_window=40,
        fvg_max_age=60,
        ob_max_age=60,
        warmup=50,
    )
    smc_cols = {
        f"smc_{str(k)}": np.array(jl_result[k], dtype=np.float64) for k in jl_result
    }
    result = pd.DataFrame(smc_cols, index=df_1d.index)

    htf_pairs = [
        (df_1w, "htf1", "1W"),
        (df_1m, "htf2", "1M"),
        (df_6m, "htf3", "6M"),
    ]
    base_times_ns = _times_to_ns(df_1d["time"])

    for htf_df, htf_label, timeframe_label in htf_pairs:
        if htf_df is None or len(htf_df) < 3:
            result[f"smc_{htf_label}_structure_trend_score"] = 0.0
            result[f"smc_{htf_label}_amd_phase"] = 0.0
            continue

        src = htf_df.sort_values("time").reset_index(drop=True)
        ho = _to_f64(src["open"].to_numpy())
        hh = _to_f64(src["high"].to_numpy())
        hl = _to_f64(src["low"].to_numpy())
        hc = _to_f64(src["close"].to_numpy())
        hv = _to_f64(src["volume"].to_numpy())
        ha = _to_f64(_compute_atr14_array(hh, hl, hc))
        htf_times_ns = _htf_close_times_ns(src, timeframe_label)

        htf_jl = TM.compute_smc_htf_features(ho, hh, hl, hc, hv, ha)

        for feat_name in ["structure_trend_score", "amd_phase"]:
            htf_vals = np.array(htf_jl[feat_name], dtype=np.float64)
            result[f"smc_{htf_label}_{feat_name}"] = _map_htf_series_to_primary(
                base_times_ns,
                htf_times_ns,
                htf_vals,
            )

    result["smc_sweep_disp_sync"] = (
        (result["smc_sweep_bull_mag"] > 0) | (result["smc_sweep_bear_mag"] > 0)
    ).astype(float) * result["smc_disp_confirmed"]
    result["smc_ob_fvg_confluence"] = np.where(
        result["smc_fvg_count_active"] > 0,
        result["smc_ob_quality_score"] * (1.0 - result["smc_fvg_fill_pct"]),
        0.0,
    )
    result["smc_phase_weighted_trend"] = (
        result["smc_amd_phase"] * result["smc_structure_trend_score"]
    )
    result["smc_full_entry_signal"] = result["smc_mss_confirmed"] * np.where(
        result["smc_pd_zone"] < 0.5,
        1.0 - result["smc_pd_zone"],
        result["smc_pd_zone"],
    )
    result["smc_pd_ob_confluence"] = np.where(
        result["smc_ob_nearest_dist"] < 3.0,
        np.abs(result["smc_pd_zone"] - 0.5) * result["smc_ob_quality_score"],
        0.0,
    )
    result["smc_htf_trend_alignment"] = np.sign(
        result["smc_structure_trend_score"]
    ) * np.sign(
        result.get(
            "smc_htf1_structure_trend_score",
            result.get("smc_htf1_structure_trend", 0.0),
        )
    )
    result["smc_htf_phase_cascade"] = result["smc_amd_phase"] * result.get(
        "smc_htf1_amd_phase", 0.0
    )

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─────────────────────────────────────────────────────────────
# PUBLIC API: add_target_fast
# ─────────────────────────────────────────────────────────────


def add_target_fast(
    df: pd.DataFrame,
    atr_mult: float = 1.25,
    horizon: int = 24,
    atr_col: str = "atr14",
    drop_unresolved: bool = True,
) -> pd.DataFrame:
    """
    Drop-in replacement for universal_ml_engine.add_target().

    Extracts raw OHLCV + ATR arrays, passes them to ToonMath.add_target_loop,
    and assigns the 13 returned Float64 vectors back to the DataFrame columns.

    The Julia function returns a NamedTuple; each field is accessed by name
    and converted to an ndarray to avoid a second copy.

    Parameters
    ----------
    df              : pd.DataFrame — must contain open/high/low/close and `atr_col`
    atr_mult        : float        — ATR barrier multiplier (default 1.25)
    horizon         : int          — max bar horizon (default 24)
    atr_col         : str          — column name for ATR14 (default 'atr14')
    drop_unresolved : bool         — drop rows where target == 0.5 (default True)

    Returns
    -------
    pd.DataFrame with 13 new label columns.
    """
    jl, TM = _init_julia()

    df = df.copy()

    opens = _to_f64(df["open"].to_numpy())
    highs = _to_f64(df["high"].to_numpy())
    lows = _to_f64(df["low"].to_numpy())
    closes = _to_f64(df["close"].to_numpy())
    atrs = _to_f64(df[atr_col].to_numpy())

    # Dispatch — all arrays are zero-copy PyArray on Julia side
    result = TM.add_target_loop(
        opens,
        highs,
        lows,
        closes,
        atrs,
        atr_mult=float(atr_mult),
        horizon=int(horizon),
        tp1_r_mult=1.0,
        tp2_r_mult=2.0,
        trail_r_mult=1.0,
        fee_pct=0.0005,
        slippage_bps=0.0003,
        tp1_frac=0.50,
        tp2_frac=0.25,
        runner_frac=0.25,
    )

    # Unpack NamedTuple fields — np.array() copies from Julia heap once,
    # then pandas assigns in-place (no further copies).
    def _col(name: str) -> np.ndarray:
        return np.array(getattr(result, name), dtype=np.float64)

    df["target"] = _col("target")
    df["next_ret_pct"] = _col("next_ret_pct")
    df["bars_to_target"] = _col("bars_to_target")
    df["entry_price_next_bar"] = _col("entry_prices")
    df["target_distance"] = _col("target_distances")
    df["long_path_r"] = _col("long_path_r")
    df["short_path_r"] = _col("short_path_r")
    df["target_edge_r"] = _col("target_edge_r")
    df["best_path_r"] = _col("best_path_r")
    df["long_mfe_atr"] = _col("long_mfe_atr")
    df["long_mae_atr"] = _col("long_mae_atr")
    df["short_mfe_atr"] = _col("short_mfe_atr")
    df["short_mae_atr"] = _col("short_mae_atr")

    if drop_unresolved:
        df = df[df["target"] != 0.5].dropna(subset=["target"]).copy()
    else:
        df["target"] = df["target"].fillna(0.5)

    return df


# ─────────────────────────────────────────────────────────────
# PUBLIC API: daily volatility targets + intraday RV summary
# ─────────────────────────────────────────────────────────────

_INTRADAY_RV_SUMMARY_COLS = [
    "intra_rv",
    "intra_rr",
    "intra_up_sv",
    "intra_dn_sv",
    "intra_asym",
    "intra_jump",
    "intra_vov",
    "intra_half_imb",
    "intra_trend_eff",
]


def vol_target_engine_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 forward volatility target columns to a daily DataFrame.

    Calls ToonMath.compute_vol_targets and ToonMath.compute_vol_targets_5d.
    Returns the same DataFrame with 8 new columns added in-place.
    """
    _, TM = _init_julia()

    df = df.copy()
    result_1d = TM.compute_vol_targets(
        _to_f64(df["open"].to_numpy()),
        _to_f64(df["high"].to_numpy()),
        _to_f64(df["low"].to_numpy()),
        _to_f64(df["close"].to_numpy()),
    )
    result_5d = TM.compute_vol_targets_5d(
        _to_f64(df["open"].to_numpy()),
        _to_f64(df["high"].to_numpy()),
        _to_f64(df["low"].to_numpy()),
        _to_f64(df["close"].to_numpy()),
    )

    def _col(result_obj, name: str) -> np.ndarray:
        return np.array(getattr(result_obj, name), dtype=np.float64)

    df["next_yz_logvol"] = _col(result_1d, "next_yz_logvol")
    df["next_log_range"] = _col(result_1d, "next_log_range")
    df["next_up_exc"] = _col(result_1d, "next_up_excursion")
    df["next_dn_exc"] = _col(result_1d, "next_dn_excursion")
    df["next5d_yz_logvol"] = _col(result_5d, "next5d_yz_logvol")
    df["next5d_log_range"] = _col(result_5d, "next5d_log_range")
    df["next5d_up_exc"] = _col(result_5d, "next5d_up_excursion")
    df["next5d_dn_exc"] = _col(result_5d, "next5d_dn_excursion")
    return df


def _zero_intraday_rv_summary(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=index, columns=_INTRADAY_RV_SUMMARY_COLS)


def _infer_session_minutes_from_1h(df_1h: pd.DataFrame) -> int:
    if df_1h.empty or "time" not in df_1h.columns:
        return 390

    dates = pd.to_datetime(df_1h["time"]).dt.normalize()
    counts = dates.value_counts()
    if counts.empty:
        return 390

    minutes = int(np.median(counts.to_numpy(dtype=float)) * 60.0)
    return int(np.clip(minutes, 60, 1440))


def intraday_rv_summary_daily(
    df_1d: pd.DataFrame,
    df_1h: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate 1H bars into per-day volatility summaries.

    Calls ToonMath.compute_intraday_rv_summary.
    Returns DataFrame with 9 intra_* columns, indexed like df_1d.
    """
    if df_1d.empty:
        return _zero_intraday_rv_summary(df_1d.index)

    if df_1h is None or len(df_1h) < 10:
        return _zero_intraday_rv_summary(df_1d.index)

    _, TM = _init_julia()
    src_1h = df_1h.sort_values("time").reset_index(drop=True)
    result = TM.compute_intraday_rv_summary(
        _to_f64(src_1h["open"].to_numpy()),
        _to_f64(src_1h["high"].to_numpy()),
        _to_f64(src_1h["low"].to_numpy()),
        _to_f64(src_1h["close"].to_numpy()),
        _times_to_ns(src_1h["time"]),
        _times_to_ns(df_1d["time"]),
        int(_infer_session_minutes_from_1h(src_1h)),
    )

    summary = pd.DataFrame(
        {
            col: np.array(result[col], dtype=np.float64)
            for col in _INTRADAY_RV_SUMMARY_COLS
        },
        index=df_1d.index,
    )
    return summary.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─────────────────────────────────────────────────────────────
# REALIZED VOLATILITY V2 — Julia-Native Multi-Estimator Surface
# ─────────────────────────────────────────────────────────────

_RV_FEAT_KEYS = [
    "yz_log",
    "yz_z",
    "yz_pctrank",
    "yz_trend",
    "yz_shock",
    "vov",
    "pk_log",
    "range_eff",
    "jump_ratio",
    "rv_asym",
    "rv_skew",
    "rs_plus",
    "rs_minus",
    "rs_leverage",
]


def _rv_julia_to_columns(
    jl_result,
    prefix: str,
) -> dict[str, np.ndarray]:
    return {
        f"rv_{prefix}_{str(k)}": np.array(jl_result[k], dtype=np.float64)
        for k in _RV_FEAT_KEYS
    }


def _rv_htf_layer(
    result: pd.DataFrame,
    base_times_ns: np.ndarray,
    primary_log_col: str,
    htf_df: pd.DataFrame | None,
    htf_label: str,
    timeframe_label: str,
    htf_ann: float,
    TM,
) -> pd.DataFrame:
    htf_feat_cols = [f"rv_{htf_label}_{k}" for k in _RV_FEAT_KEYS]
    term_col = f"rv_term_{htf_label}_{primary_log_col.split('_')[1]}"

    if htf_df is None or len(htf_df) < 3:
        for col in htf_feat_cols + [term_col]:
            result[col] = 0.0
        return result

    src = htf_df.sort_values("time").reset_index(drop=True)
    htf_jl = TM.compute_rv_features(
        _to_f64(src["open"].to_numpy()),
        _to_f64(src["high"].to_numpy()),
        _to_f64(src["low"].to_numpy()),
        _to_f64(src["close"].to_numpy()),
        annualization=htf_ann,
    )

    htf_times_ns = _htf_close_times_ns(src, timeframe_label)

    for feat_key in _RV_FEAT_KEYS:
        htf_vals = np.array(htf_jl[feat_key], dtype=np.float64)
        result[f"rv_{htf_label}_{feat_key}"] = _map_htf_series_to_primary(
            base_times_ns,
            htf_times_ns,
            htf_vals,
        )

    result[term_col] = (
        (result[f"rv_{htf_label}_yz_log"] - result[primary_log_col])
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(-8.0, 8.0)
    )
    return result


def rv_feature_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Julia RV engine for 1H lane. Returns DataFrame with rv_* columns.

    Primary: 14 features.
    Per HTF:  14 features + 1 term structure = 15.
    Total with all 3 HTFs: 14 + 3*15 = 59 columns.
    """
    _, TM = _init_julia()

    jl_result = TM.compute_rv_features(
        _to_f64(df_1h["open"].to_numpy()),
        _to_f64(df_1h["high"].to_numpy()),
        _to_f64(df_1h["low"].to_numpy()),
        _to_f64(df_1h["close"].to_numpy()),
        annualization=1512.0,
    )
    result = pd.DataFrame(_rv_julia_to_columns(jl_result, "1h"), index=df_1h.index)

    base_times_ns = _times_to_ns(df_1h["time"])
    primary_log_col = "rv_1h_yz_log"

    htf_layers = [
        (df_1d, "1d", "1D", 252.0),
        (df_1w, "1w", "1W", 52.0),
        (df_1m, "1m", "1M", 12.0),
    ]
    if any(htf_df is not None for htf_df, _, _, _ in htf_layers):
        for htf_df, htf_label, timeframe_label, htf_ann in htf_layers:
            result = _rv_htf_layer(
                result,
                base_times_ns,
                primary_log_col,
                htf_df,
                htf_label,
                timeframe_label,
                htf_ann,
                TM,
            )

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def rv_feature_engine_daily(
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_3m: pd.DataFrame | None = None,
    df_6m: pd.DataFrame | None = None,
    df_12m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Julia RV engine for 1D lane. Returns DataFrame with rv_* columns.

    Primary: 14 features.
    Per HTF:  14 features + 1 term structure = 15.
    Total with all 5 HTFs: 14 + 5*15 = 89 columns.
    """
    _, TM = _init_julia()

    jl_result = TM.compute_rv_features(
        _to_f64(df_1d["open"].to_numpy()),
        _to_f64(df_1d["high"].to_numpy()),
        _to_f64(df_1d["low"].to_numpy()),
        _to_f64(df_1d["close"].to_numpy()),
        annualization=252.0,
    )
    result = pd.DataFrame(_rv_julia_to_columns(jl_result, "1d"), index=df_1d.index)

    base_times_ns = _times_to_ns(df_1d["time"])
    primary_log_col = "rv_1d_yz_log"

    htf_layers = [
        (df_1w, "1w", "1W", 52.0),
        (df_1m, "1m", "1M", 12.0),
        (df_3m, "3m", "3M", 4.0),
        (df_6m, "6m", "6M", 2.0),
        (df_12m, "12m", "12M", 1.0),
    ]
    if any(htf_df is not None for htf_df, _, _, _ in htf_layers):
        for htf_df, htf_label, timeframe_label, htf_ann in htf_layers:
            result = _rv_htf_layer(
                result,
                base_times_ns,
                primary_log_col,
                htf_df,
                htf_label,
                timeframe_label,
                htf_ann,
                TM,
            )

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ─────────────────────────────────────────────────────────────
# VIX IMPLIED VOLATILITY FEATURE ENGINE
# ─────────────────────────────────────────────────────────────

_VIX_FEATURE_COLS = [
    "vix_level",
    "vix_z",
    "vix_pctrank",
    "vix_regime",
    "vix_trend",
    "vix_shock",
    "vix_accel",
    "vix_5d_change",
    "vix_21d_change",
    "vix_rv_spread",
    "vix_rv_ratio",
    "vix_rv_spread_z",
    "vix_rv_spread_trend",
    "vix_distance_from_mean",
    "vix_halflife_signal",
    "vix_contango_proxy",
    "vix_floor_distance",
]

_VIX_INTERACTION_COLS = [
    "vix_x_rv_asym",
    "vix_x_jump",
    "vix_x_vov",
]
_VIX_REGIME_THRESHOLDS_DEFAULT = (15.0, 25.0, 35.0)
_VIX_REGIME_THRESHOLDS_BY_SYMBOL = {
    "INDIA_VIX": (12.0, 18.0, 25.0),
}


def _zero_vix_features(index: pd.Index) -> pd.DataFrame:
    """Return a zero-filled VIX feature DataFrame (graceful fallback)."""
    return pd.DataFrame(
        {col: np.zeros(len(index), dtype=np.float64) for col in _VIX_FEATURE_COLS},
        index=index,
    )


def _resolve_vix_regime_thresholds(
    df_vix: pd.DataFrame | None,
) -> tuple[float, float, float]:
    if df_vix is None:
        return _VIX_REGIME_THRESHOLDS_DEFAULT
    attrs = df_vix.attrs if isinstance(getattr(df_vix, "attrs", None), dict) else {}
    vix_symbol = (
        str(
            attrs.get("vix_companion_symbol")
            or attrs.get("resolved_source_symbol")
            or attrs.get("pair_symbol")
            or ""
        )
        .strip()
        .upper()
    )
    return _VIX_REGIME_THRESHOLDS_BY_SYMBOL.get(
        vix_symbol,
        _VIX_REGIME_THRESHOLDS_DEFAULT,
    )


def vix_feature_engine_daily(
    df_1d: pd.DataFrame,
    df_vix: pd.DataFrame | None,
    rv_yz_log_col: str = "rv_1d_yz_log",
) -> pd.DataFrame:
    """Compute VIX implied-volatility features aligned to daily bars.

    Parameters
    ----------
    df_1d : DataFrame with 'time' column and rv_yz_log_col (realized vol log).
    df_vix : VIX 1D DataFrame with 'time' and 'close' columns, or None.
    rv_yz_log_col : column name in df_1d for the realized vol log series.

    Returns
    -------
    DataFrame with 17 vix_* columns, indexed like df_1d.
    """
    if df_1d.empty:
        return _zero_vix_features(df_1d.index)

    if df_vix is None or df_vix.empty or len(df_vix) < 5:
        return _zero_vix_features(df_1d.index)

    _, TM = _init_julia()

    # Align VIX to daily bars via merge_asof
    df_daily = df_1d[["time"]].copy().reset_index(drop=True)
    df_daily["_idx"] = df_1d.index

    vix_sorted = (
        df_vix[["time", "close"]].copy().sort_values("time").reset_index(drop=True)
    )
    vix_sorted.rename(columns={"close": "vix_close"}, inplace=True)

    merged = pd.merge_asof(
        df_daily.sort_values("time"),
        vix_sorted,
        on="time",
        direction="backward",
    )
    merged = merged.sort_values("_idx").set_index("_idx")
    merged.index = df_1d.index

    vix_closes = merged["vix_close"].fillna(0.0).to_numpy(dtype=np.float64)

    if rv_yz_log_col in df_1d.columns:
        rv_log = df_1d[rv_yz_log_col].fillna(0.0).to_numpy(dtype=np.float64)
    else:
        rv_log = np.zeros(len(df_1d), dtype=np.float64)

    regime_low, regime_mid, regime_high = _resolve_vix_regime_thresholds(df_vix)
    jl_result = TM.compute_vix_features(
        _to_f64(vix_closes),
        _to_f64(rv_log),
        regime_low=float(regime_low),
        regime_mid=float(regime_mid),
        regime_high=float(regime_high),
    )

    result = pd.DataFrame(
        {col: np.array(jl_result[col], dtype=np.float64) for col in _VIX_FEATURE_COLS},
        index=df_1d.index,
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def vix_interaction_features(
    df_full: pd.DataFrame,
) -> pd.DataFrame:
    """Compute VIX × RV interaction features (3 columns).

    Requires vix_level, rv_1d_rv_asym, rv_1d_jump_ratio, rv_1d_vov
    to already exist in df_full.
    """
    n = len(df_full)
    zeros = pd.Series(np.zeros(n), index=df_full.index)

    vix_level = df_full.get("vix_level", zeros)
    rv_asym = df_full.get("rv_1d_rv_asym", zeros)
    jump = df_full.get("rv_1d_jump_ratio", zeros)
    vov = df_full.get("rv_1d_vov", zeros)

    return pd.DataFrame(
        {
            "vix_x_rv_asym": (vix_level * rv_asym).clip(-10.0, 10.0),
            "vix_x_jump": (vix_level * jump).clip(-10.0, 10.0),
            "vix_x_vov": (vix_level * vov).clip(-10.0, 10.0),
        },
        index=df_full.index,
    ).fillna(0.0)
