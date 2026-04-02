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
    _toon_path  = _bridge_dir / "ToonMath.jl"
    if not _toon_path.exists():
        raise FileNotFoundError(f"ToonMath.jl not found at {_toon_path}")

    jl.seval(f'include("{_toon_path.as_posix()}")')
    jl.seval("using .ToonMath")

    _jl       = jl
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
    "basis_pct", "basis_z_score", "basis_vel_5", "basis_vel_10",
    "session_time_pos", "eod_basis_momentum",
)

def _build_thermo(df: pd.DataFrame, n: int):
    """
    Build thermo NamedTuple for Julia if all thermodynamic columns exist.
    Returns None otherwise (Julia side checks for `nothing`).
    """
    if not all(c in df.columns for c in _THERMO_COLS):
        return None

    jl, _ = _init_julia()
    thermo_namedtuple = jl.seval(
        "(basis_pct, basis_z, basis_v5, basis_v10, session_pos, eod_momentum) -> "
        "(; basis_pct, basis_z, basis_v5, basis_v10, session_pos, eod_momentum)"
    )
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
            empty_f  = np.empty(0, dtype=np.float64)
            empty_i  = np.empty(0, dtype=np.int64)
            return empty_i, empty_f, empty_f, empty_f, empty_f, empty_f
        src = df.sort_values("time").reset_index(drop=True)
        t = _times_to_ns(src["time"])
        o, h, l, c, v = _extract_ohlcv(src)
        return t, o, h, l, c, v

    t_1d, o_1d, h_1d, l_1d, c_1d, v_1d = _htf(df_1d)
    t_1w, o_1w, h_1w, l_1w, c_1w, v_1w = _htf(df_1w)
    t_1m, o_1m, h_1m, l_1m, c_1m, v_1m = _htf(df_1m)

    # Dispatch to Julia — arrays cross the bridge as PyArray (zero-copy)
    jl_result = TM.compute_holographic_features(
        base_times_ns,
        o_1h, h_1h, l_1h, c_1h, v_1h,
        t_1d, o_1d, h_1d, l_1d, c_1d, v_1d,
        t_1w, o_1w, h_1w, l_1w, c_1w, v_1w,
        t_1m, o_1m, h_1m, l_1m, c_1m, v_1m,
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
        o_1d, h_1d, l_1d, c_1d, v_1d,
        t_1w, o_1w, h_1w, l_1w, c_1w, v_1w,
        t_1m, o_1m, h_1m, l_1m, c_1m, v_1m,
        t_3m, o_3m, h_3m, l_3m, c_3m, v_3m,
        thermo_1d=thermo_1d if thermo_1d is not None else jl.nothing,
    )

    feat_df = _jl_dict_to_df(jl_result, df_1d.index)
    new_cols = [c for c in feat_df.columns if c not in df_1d.columns]
    return pd.concat([df_1d, feat_df[new_cols]], axis=1)


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

    opens  = _to_f64(df["open"].to_numpy())
    highs  = _to_f64(df["high"].to_numpy())
    lows   = _to_f64(df["low"].to_numpy())
    closes = _to_f64(df["close"].to_numpy())
    atrs   = _to_f64(df[atr_col].to_numpy())

    # Dispatch — all arrays are zero-copy PyArray on Julia side
    result = TM.add_target_loop(
        opens, highs, lows, closes, atrs,
        atr_mult     = float(atr_mult),
        horizon      = int(horizon),
        tp1_r_mult   = 1.0,
        tp2_r_mult   = 2.0,
        trail_r_mult = 1.0,
        fee_pct      = 0.0005,
        slippage_bps = 0.0003,
        tp1_frac     = 0.50,
        tp2_frac     = 0.25,
        runner_frac  = 0.25,
    )

    # Unpack NamedTuple fields — np.array() copies from Julia heap once,
    # then pandas assigns in-place (no further copies).
    def _col(name: str) -> np.ndarray:
        return np.array(getattr(result, name), dtype=np.float64)

    df["target"]               = _col("target")
    df["next_ret_pct"]         = _col("next_ret_pct")
    df["bars_to_target"]       = _col("bars_to_target")
    df["entry_price_next_bar"] = _col("entry_prices")
    df["target_distance"]      = _col("target_distances")
    df["long_path_r"]          = _col("long_path_r")
    df["short_path_r"]         = _col("short_path_r")
    df["target_edge_r"]        = _col("target_edge_r")
    df["best_path_r"]          = _col("best_path_r")
    df["long_mfe_atr"]         = _col("long_mfe_atr")
    df["long_mae_atr"]         = _col("long_mae_atr")
    df["short_mfe_atr"]        = _col("short_mfe_atr")
    df["short_mae_atr"]        = _col("short_mae_atr")

    if drop_unresolved:
        df = df[df["target"] != 0.5].dropna(subset=["target"]).copy()
    else:
        df["target"] = df["target"].fillna(0.5)

    return df
