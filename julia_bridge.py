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
    """
    import pathlib

    global _jl, _ToonMath
    try:
        return _jl, _ToonMath
    except NameError:
        pass

    from juliacall import Main as jl  # type: ignore

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
_FIB_SLOTS = ("a", "b", "c")
_FIB_FIELDS = (
    "close_pos",
    "zone",
    "wick_rej_bull",
    "wick_rej_bear",
    "body_acc",
    "ext_pct",
)
_FIB_ALL_COLS = tuple(
    f"fib_{slot}_{field}" for slot in _FIB_SLOTS for field in _FIB_FIELDS
)
_THERMO_NT_FIELDS = (
    "basis_pct",
    "basis_z",
    "basis_v5",
    "basis_v10",
    "session_pos",
    "eod_momentum",
    *_FIB_ALL_COLS,
)


def _build_thermo(df: pd.DataFrame, n: int):
    """
    Build thermo NamedTuple for Julia if all base thermodynamic columns exist.
    Returns None otherwise (Julia side checks for `nothing`).

    Fib columns are included when present and zero-filled when absent so the
    Julia DNA layer can keep a stable field contract across training and
    inference paths.
    """
    if not all(c in df.columns for c in _THERMO_COLS):
        return None

    jl, _ = _init_julia()
    field_sig = ", ".join(_THERMO_NT_FIELDS)
    thermo_namedtuple = jl.seval(f"({field_sig}) -> (; {field_sig})")

    def _fib(col: str) -> np.ndarray:
        if col in df.columns:
            return _to_f64(df[col].to_numpy()[:n])
        return np.zeros(n, dtype=np.float64)

    return thermo_namedtuple(
        _to_f64(df["basis_pct"].to_numpy()[:n]),
        _to_f64(df["basis_z_score"].to_numpy()[:n]),
        _to_f64(df["basis_vel_5"].to_numpy()[:n]),
        _to_f64(df["basis_vel_10"].to_numpy()[:n]),
        _to_f64(df["session_time_pos"].to_numpy()[:n]),
        _to_f64(df["eod_basis_momentum"].to_numpy()[:n]),
        *[_fib(col) for col in _FIB_ALL_COLS],
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
    def _htf(df: pd.DataFrame | None):
        if df is None or len(df) < 3:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f
        src = df.sort_values("time").reset_index(drop=True)
        t = _times_to_ns(src["time"])
        o, h, lo, c, v = _extract_ohlcv(src)
        return t, o, h, lo, c, v

    t_1d, o_1d, h_1d, l_1d, c_1d, v_1d = _htf(df_1d)
    t_1w, o_1w, h_1w, l_1w, c_1w, v_1w = _htf(df_1w)
    t_1m, o_1m, h_1m, l_1m, c_1m, v_1m = _htf(df_1m)

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

    def _htf(df: pd.DataFrame | None):
        if df is None or len(df) < 3:
            empty_f = np.empty(0, dtype=np.float64)
            empty_i = np.empty(0, dtype=np.int64)
            return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f
        src = df.sort_values("time").reset_index(drop=True)
        return (_times_to_ns(src["time"]), *_extract_ohlcv(src))

    t_1w, o_1w, h_1w, l_1w, c_1w, v_1w = _htf(df_1w)
    t_1m, o_1m, h_1m, l_1m, c_1m, v_1m = _htf(df_1m)
    t_3m, o_3m, h_3m, l_3m, c_3m, v_3m = _htf(df_3m)

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
# PUBLIC API: SMC institutional intent features
# ─────────────────────────────────────────────────────────────


def smc_feature_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute 42 SMC institutional intent features for the 1H lane.

    Layer 1: 30 primary features via ToonMath.compute_smc_features on 1H
    Layer 2: 6 HTF projection features (structure_trend + amd_phase from 1D/1W/1M)
    Layer 3: 6 confluence features (cross-TF interactions)

    Parameters
    ----------
    df_1h : pd.DataFrame — must contain OHLCV + atr14 columns
    df_1d, df_1w, df_1m : higher-TF DataFrames (optional)

    Returns
    -------
    pd.DataFrame — 42 columns indexed same as df_1h, all column names prefixed 'smc_'
    """
    jl, TM = _init_julia()

    o = _to_f64(df_1h["open"].to_numpy())
    h = _to_f64(df_1h["high"].to_numpy())
    lo = _to_f64(df_1h["low"].to_numpy())
    c = _to_f64(df_1h["close"].to_numpy())
    v = _to_f64(df_1h["volume"].to_numpy())
    a = _to_f64(df_1h["atr14"].to_numpy())

    jl_result = TM.compute_smc_features(o, h, lo, c, v, a)
    smc_cols = {
        f"smc_{str(k)}": np.array(jl_result[k], dtype=np.float64) for k in jl_result
    }
    result = pd.DataFrame(smc_cols, index=df_1h.index)

    htf_pairs = [
        (df_1d, "htf1"),
        (df_1w, "htf2"),
        (df_1m, "htf3"),
    ]
    base_times_ns = _times_to_ns(df_1h["time"])

    for htf_df, htf_label in htf_pairs:
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

        htf_jl = TM.compute_smc_htf_features(ho, hh, hl, hc, hv, ha)

        htf_times_ns = _times_to_ns(src["time"])
        idx_map = np.searchsorted(htf_times_ns, base_times_ns, side="right") - 1
        idx_map = np.clip(idx_map, 0, len(src) - 1)

        for feat_name in ["structure_trend_score", "amd_phase"]:
            htf_vals = np.array(htf_jl[feat_name], dtype=np.float64)
            result[f"smc_{htf_label}_{feat_name}"] = htf_vals[idx_map]

    result["smc_sweep_disp_sync"] = (
        (
            (result["smc_sweep_bull_mag"] > 0)
            | (result["smc_sweep_bear_mag"] > 0)
        ).astype(float)
        * result["smc_disp_confirmed"]
    )
    result["smc_ob_fvg_confluence"] = np.where(
        result["smc_fvg_count_active"] > 0,
        result["smc_ob_quality_score"] * (1.0 - result["smc_fvg_fill_pct"]),
        0.0,
    )
    result["smc_phase_weighted_trend"] = (
        result["smc_amd_phase"] * result["smc_structure_trend_score"]
    )
    result["smc_full_entry_signal"] = (
        result["smc_indu_confirmed"] * result["smc_mss_confirmed"]
    )
    result["smc_htf_trend_alignment"] = np.sign(
        result["smc_structure_trend_score"]
    ) * np.sign(
        result.get(
            "smc_htf1_structure_trend_score",
            result.get("smc_htf1_structure_trend", 0.0),
        )
    )
    result["smc_htf_phase_cascade"] = (
        result["smc_amd_phase"] * result.get("smc_htf1_amd_phase", 0.0)
    )

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def smc_feature_engine_daily(
    df_1d: pd.DataFrame,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
    df_6m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute 42 SMC institutional intent features for the 1D lane.

    Same architecture as smc_feature_engine_fast but:
    - Primary TF = 1D
    - HTF1 = 1W (5x), HTF2 = 1M (4x), HTF3 = 6M (6x)

    Parameters
    ----------
    df_1d : pd.DataFrame — must contain OHLCV + atr14 columns
    df_1w, df_1m, df_6m : higher-TF DataFrames (optional)

    Returns
    -------
    pd.DataFrame — 42 columns, all prefixed 'smc_'
    """
    jl, TM = _init_julia()

    o = _to_f64(df_1d["open"].to_numpy())
    h = _to_f64(df_1d["high"].to_numpy())
    lo = _to_f64(df_1d["low"].to_numpy())
    c = _to_f64(df_1d["close"].to_numpy())
    v = _to_f64(df_1d["volume"].to_numpy())
    a = _to_f64(df_1d["atr14"].to_numpy())

    jl_result = TM.compute_smc_features(o, h, lo, c, v, a)
    smc_cols = {
        f"smc_{str(k)}": np.array(jl_result[k], dtype=np.float64) for k in jl_result
    }
    result = pd.DataFrame(smc_cols, index=df_1d.index)

    htf_pairs = [
        (df_1w, "htf1"),
        (df_1m, "htf2"),
        (df_6m, "htf3"),
    ]
    base_times_ns = _times_to_ns(df_1d["time"])

    for htf_df, htf_label in htf_pairs:
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

        htf_jl = TM.compute_smc_htf_features(ho, hh, hl, hc, hv, ha)

        htf_times_ns = _times_to_ns(src["time"])
        idx_map = np.searchsorted(htf_times_ns, base_times_ns, side="right") - 1
        idx_map = np.clip(idx_map, 0, len(src) - 1)

        for feat_name in ["structure_trend_score", "amd_phase"]:
            htf_vals = np.array(htf_jl[feat_name], dtype=np.float64)
            result[f"smc_{htf_label}_{feat_name}"] = htf_vals[idx_map]

    result["smc_sweep_disp_sync"] = (
        (
            (result["smc_sweep_bull_mag"] > 0)
            | (result["smc_sweep_bear_mag"] > 0)
        ).astype(float)
        * result["smc_disp_confirmed"]
    )
    result["smc_ob_fvg_confluence"] = np.where(
        result["smc_fvg_count_active"] > 0,
        result["smc_ob_quality_score"] * (1.0 - result["smc_fvg_fill_pct"]),
        0.0,
    )
    result["smc_phase_weighted_trend"] = (
        result["smc_amd_phase"] * result["smc_structure_trend_score"]
    )
    result["smc_full_entry_signal"] = (
        result["smc_indu_confirmed"] * result["smc_mss_confirmed"]
    )
    result["smc_htf_trend_alignment"] = np.sign(
        result["smc_structure_trend_score"]
    ) * np.sign(
        result.get(
            "smc_htf1_structure_trend_score",
            result.get("smc_htf1_structure_trend", 0.0),
        )
    )
    result["smc_htf_phase_cascade"] = (
        result["smc_amd_phase"] * result.get("smc_htf1_amd_phase", 0.0)
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
