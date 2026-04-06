# CODEX 5.4 — Realized Volatility V2: Julia-Native Multi-Estimator Pipeline

## IDENTITY
You are modifying `/home/km/Universal-ML/`. Execute every step exactly. No improvisation.

## CONSTRAINT RULES
- Preserve all existing function signatures that are NOT being deleted.
- `_to_f64`, `_times_to_ns`, `_init_julia` already exist in `julia_bridge.py` — reuse them.
- ALL new feature column names start with `rv_`.
- All `for col in rv_df.columns: df[col] = rv_df[col].values` injection lines must exactly match the SMC/Kalman pattern already in the codebase.

---

## STEP 1: ADD `compute_rv_features` TO `ToonMath.jl`

INSERT BEFORE `end  # module ToonMath` (line 2862 of `ToonMath.jl`):

```julia
# ─────────────────────────────────────────────────────────────
# REALIZED VOLATILITY V2 — Institutional Multi-Estimator Surface
# ─────────────────────────────────────────────────────────────

"""
    compute_rv_features(opens, highs, lows, closes; ...) -> Dict{String, Vector{Float64}}

Single-pass Yang-Zhang + Parkinson + jump/asymmetry/skewness volatility surface.
Returns 11 features per timeframe:

Estimators (2):
  yz_log     — log1p(Yang-Zhang annualized RV)
  pk_log     — log1p(Parkinson annualized RV)

Regime (2):
  yz_z       — z-score vs z_window history
  yz_pctrank — percentile rank [0,1] over z_window

Dynamics (3):
  yz_trend   — short MA / long MA ratio - 1
  yz_shock   — 1-bar surprise (rv/prev_rv - 1)
  vov        — vol-of-vol: std(rv,21)/mean(rv,21)

Structure (2):
  range_eff  — Parkinson/Yang-Zhang ratio (microstructure efficiency)
  jump_ratio — (simple_var - BPV) / simple_var  (jump vs diffusion)

Asymmetry (2):
  rv_asym    — down_vol / up_vol ratio (leverage effect, Black 1976)
  rv_skew    — realized skewness of returns (3rd moment, crash/melt-up)
"""
function compute_rv_features(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64};
    rv_window::Int = 21,
    z_window::Int = 63,
    short_window::Int = 5,
    long_window::Int = 21,
    annualization::Float64 = 252.0,
)::Dict{String, Vector{Float64}}
    n = length(closes)
    eps = 1e-9

    # ── Output vectors (11 features) ──
    yz_log_out     = fill(0.0, n)
    yz_z_out       = fill(0.0, n)
    yz_pctrank_out = fill(0.5, n)
    yz_trend_out   = fill(0.0, n)
    yz_shock_out   = fill(0.0, n)
    vov_out        = fill(0.0, n)
    pk_log_out     = fill(0.0, n)
    range_eff_out  = fill(1.0, n)
    jump_ratio_out = fill(0.0, n)
    rv_asym_out    = fill(1.0, n)
    rv_skew_out    = fill(0.0, n)

    # ── Intermediate: per-bar components ──
    log_co  = fill(0.0, n)   # overnight: log(open_t / close_{t-1})
    log_oc  = fill(0.0, n)   # intraday: log(close_t / open_t)
    log_ret = fill(0.0, n)   # close-to-close log return
    rs_comp = fill(0.0, n)   # Rogers-Satchell component
    pk_comp = fill(0.0, n)   # Parkinson component: log(H/L)^2

    for i in 1:n
        o_i = opens[i]; h_i = highs[i]; l_i = lows[i]; c_i = closes[i]
        if o_i <= 0.0 || h_i <= 0.0 || l_i <= 0.0 || c_i <= 0.0; continue; end
        log_oc[i] = log(c_i / o_i)
        if i > 1 && closes[i - 1] > 0.0
            log_co[i] = log(o_i / closes[i - 1])
            log_ret[i] = log(c_i / closes[i - 1])
        end
        log_hc = log(h_i / c_i); log_ho = log(h_i / o_i)
        log_lc = log(l_i / c_i); log_lo = log(l_i / o_i)
        rs_comp[i] = log_hc * log_ho + log_lc * log_lo
        pk_comp[i] = log(h_i / l_i)^2
    end

    # ── Rolling estimators ──
    yz_rv = fill(0.0, n)
    pk_rv = fill(0.0, n)
    pk_factor = 1.0 / (4.0 * log(2.0))
    k = 0.34 / (1.34 + (rv_window + 1) / (rv_window - 1))

    for i in rv_window:n
        s = (i - rv_window + 1):i
        co_start = max(i - rv_window + 2, 2)
        co_slice = @view log_co[co_start:i]
        if length(co_slice) < 2; continue; end

        # ── Yang-Zhang 3-component ──
        mu_o = 0.0; @inbounds for v in co_slice; mu_o += v; end; mu_o /= length(co_slice)
        sigma_o2 = 0.0; @inbounds for v in co_slice; sigma_o2 += (v - mu_o)^2; end; sigma_o2 /= length(co_slice)

        oc_slice = @view log_oc[s]
        mu_c = 0.0; @inbounds for v in oc_slice; mu_c += v; end; mu_c /= rv_window
        sigma_c2 = 0.0; @inbounds for v in oc_slice; sigma_c2 += (v - mu_c)^2; end; sigma_c2 /= rv_window

        sigma_rs2 = 0.0; @inbounds for v in @view rs_comp[s]; sigma_rs2 += v; end; sigma_rs2 /= rv_window

        yz_var = sigma_o2 + k * sigma_rs2 + (1.0 - k) * sigma_c2
        if isfinite(yz_var) && yz_var > 0.0; yz_rv[i] = sqrt(yz_var * annualization); end

        # ── Parkinson ──
        pk_sum = 0.0; @inbounds for v in @view pk_comp[s]; pk_sum += v; end
        pk_var = pk_sum / rv_window * pk_factor * annualization
        if pk_var > 0.0; pk_rv[i] = sqrt(pk_var); end

        # ── Jump ratio: (simple_var - BPV) / simple_var ──
        ret_slice = @view log_ret[s]
        mu_ret = 0.0; @inbounds for v in ret_slice; mu_ret += v; end; mu_ret /= rv_window
        simple_var = 0.0; @inbounds for v in ret_slice; simple_var += (v - mu_ret)^2; end
        simple_var /= rv_window

        # BPV: sum of |r_t| * |r_{t-1}| * (pi/2)
        bpv = 0.0
        bpv_count = 0
        for j in (i - rv_window + 2):i
            bpv += abs(log_ret[j]) * abs(log_ret[j - 1])
            bpv_count += 1
        end
        if bpv_count > 0
            bpv = bpv / bpv_count * (pi / 2.0)
        end
        if simple_var > eps
            jump_ratio_out[i] = clamp((simple_var - bpv) / simple_var, 0.0, 1.0)
        end

        # ── Asymmetric volatility: down_var / up_var ──
        down_var = 0.0; down_count = 0
        up_var = 0.0; up_count = 0
        for j in s
            r = log_ret[j]
            if r < 0.0
                down_var += r * r
                down_count += 1
            elseif r > 0.0
                up_var += r * r
                up_count += 1
            end
        end
        if down_count > 0; down_var /= down_count; end
        if up_count > 0; up_var /= up_count; end
        if up_var > eps
            rv_asym_out[i] = clamp(down_var / up_var, 0.1, 10.0)
        end

        # ── Realized skewness: mean(r³) / mean(r²)^(3/2) ──
        m2 = 0.0; m3 = 0.0
        for j in s
            r = log_ret[j]
            r2 = r * r
            m2 += r2
            m3 += r2 * r
        end
        m2 /= rv_window; m3 /= rv_window
        denom = m2^1.5
        if denom > eps
            rv_skew_out[i] = clamp(m3 / denom, -5.0, 5.0)
        end
    end

    # ── Backfill warmup ──
    first_valid = 0
    for i in rv_window:n
        if yz_rv[i] > 0.0; first_valid = i; break; end
    end
    if first_valid > 0
        for i in 1:(first_valid - 1)
            yz_rv[i] = yz_rv[first_valid]
            pk_rv[i] = pk_rv[first_valid]
            jump_ratio_out[i] = jump_ratio_out[first_valid]
            rv_asym_out[i] = rv_asym_out[first_valid]
            rv_skew_out[i] = rv_skew_out[first_valid]
        end
    end

    # ── Cumulative sum for O(1) rolling means ──
    yz_cumsum = fill(0.0, n + 1)
    for i in 1:n; yz_cumsum[i + 1] = yz_cumsum[i] + yz_rv[i]; end

    for i in 1:n
        rv_i = yz_rv[i]

        # Level
        yz_log_out[i] = log1p(rv_i)
        pk_log_out[i] = log1p(pk_rv[i])

        # Range efficiency
        if rv_i > eps; range_eff_out[i] = clamp(pk_rv[i] / rv_i, 0.2, 5.0); end

        # Shock
        if i > 1 && yz_rv[i - 1] > eps; yz_shock_out[i] = clamp((rv_i / yz_rv[i - 1]) - 1.0, -5.0, 5.0); end

        # Z-score + Percentile rank
        if i >= z_window
            z_start = i - z_window + 1
            z_mean = (yz_cumsum[i + 1] - yz_cumsum[z_start]) / z_window
            z_var = 0.0
            @inbounds for j in z_start:i; z_var += (yz_rv[j] - z_mean)^2; end
            z_std = sqrt(z_var / z_window)
            if z_std > eps; yz_z_out[i] = clamp((rv_i - z_mean) / z_std, -8.0, 8.0); end
            rank_count = 0
            @inbounds for j in z_start:i; if yz_rv[j] <= rv_i; rank_count += 1; end; end
            yz_pctrank_out[i] = Float64(rank_count) / Float64(z_window)
        end

        # Trend (short/long MA)
        if i >= long_window
            short_start = max(1, i - short_window + 1)
            long_start = i - long_window + 1
            short_mean = (yz_cumsum[i + 1] - yz_cumsum[short_start]) / (i - short_start + 1)
            long_mean = (yz_cumsum[i + 1] - yz_cumsum[long_start]) / long_window
            if long_mean > eps; yz_trend_out[i] = clamp((short_mean / long_mean) - 1.0, -5.0, 5.0); end
        end

        # Vol-of-vol
        if i >= rv_window
            vov_start = i - rv_window + 1
            vov_mean = (yz_cumsum[i + 1] - yz_cumsum[vov_start]) / rv_window
            if vov_mean > eps
                vov_var = 0.0
                @inbounds for j in vov_start:i; vov_var += (yz_rv[j] - vov_mean)^2; end
                vov_out[i] = clamp(sqrt(vov_var / rv_window) / vov_mean, 0.0, 5.0)
            end
        end
    end

    return Dict{String, Vector{Float64}}(
        "yz_log"     => yz_log_out,
        "yz_z"       => yz_z_out,
        "yz_pctrank" => yz_pctrank_out,
        "yz_trend"   => yz_trend_out,
        "yz_shock"   => yz_shock_out,
        "vov"        => vov_out,
        "pk_log"     => pk_log_out,
        "range_eff"  => range_eff_out,
        "jump_ratio" => jump_ratio_out,
        "rv_asym"    => rv_asym_out,
        "rv_skew"    => rv_skew_out,
    )
end
```

---

## STEP 2: ADD BRIDGE FUNCTIONS TO `julia_bridge.py`

APPEND at end of file (after line 1201):

```python
# ─────────────────────────────────────────────────────────────
# REALIZED VOLATILITY V2 — Julia-Native Multi-Estimator Surface
# ─────────────────────────────────────────────────────────────

_RV_FEAT_KEYS = [
    "yz_log", "yz_z", "yz_pctrank", "yz_trend",
    "yz_shock", "vov", "pk_log", "range_eff",
    "jump_ratio", "rv_asym", "rv_skew",
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

    htf_times_ns = _times_to_ns(src["time"])
    idx_map = np.searchsorted(htf_times_ns, base_times_ns, side="right") - 1
    idx_map = np.clip(idx_map, 0, len(src) - 1)

    for feat_key in _RV_FEAT_KEYS:
        htf_vals = np.array(htf_jl[feat_key], dtype=np.float64)
        result[f"rv_{htf_label}_{feat_key}"] = htf_vals[idx_map]

    result[term_col] = (
        result[f"rv_{htf_label}_yz_log"] - result[primary_log_col]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)
    return result


def rv_feature_engine_fast(
    df_1h: pd.DataFrame,
    df_1d: pd.DataFrame | None = None,
    df_1w: pd.DataFrame | None = None,
    df_1m: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Julia RV engine for 1H lane. Returns DataFrame with rv_* columns.
    
    Primary: 11 features.
    Per HTF:  11 features + 1 term structure = 12.
    Total with all 3 HTFs: 11 + 3*12 = 47 columns.
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

    for htf_df, htf_label, htf_ann in [
        (df_1d, "1d", 252.0),
        (df_1w, "1w", 52.0),
        (df_1m, "1m", 12.0),
    ]:
        result = _rv_htf_layer(
            result, base_times_ns, primary_log_col,
            htf_df, htf_label, htf_ann, TM,
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
    
    Primary: 11 features.
    Per HTF:  11 features + 1 term structure = 12.
    Total with all 5 HTFs: 11 + 5*12 = 71 columns.
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

    for htf_df, htf_label, htf_ann in [
        (df_1w, "1w", 52.0),
        (df_1m, "1m", 12.0),
        (df_3m, "3m", 4.0),
        (df_6m, "6m", 2.0),
        (df_12m, "12m", 1.0),
    ]:
        result = _rv_htf_layer(
            result, base_times_ns, primary_log_col,
            htf_df, htf_label, htf_ann, TM,
        )

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

---

## STEP 3: MODIFY `universal_ml_engine.py`

### 3a. ADD import (line 26, inside existing `from julia_bridge import (` block)

Add `rv_feature_engine_fast,` after `smc_feature_engine_fast,` (line 26):

```python
from julia_bridge import (
    holographic_feature_engine_fast as holographic_feature_engine,
    kalman_structural_engine_fast,
    rv_feature_engine_fast,
    smc_feature_engine_fast,
    add_target_fast as add_target,
)
```

### 3b. DELETE old Python RV code

DELETE these lines entirely:

- Lines 117-120 (`REALIZED_VOL_SHORT_WINDOW` through `REALIZED_VOL_EPS`)
- Lines 465-601 (functions `_sanitize_realized_vol_series`, `_rv_feature_columns`, `_build_realized_vol_feature_block`, `_shift_closed_bar_time`, `inject_realized_volatility_features`)

### 3c. ADD RV injection to 1H main path

After Kalman injection (after the line `print(f"  [TOON v5.3] Kalman structural features injected: {len(kf_df.columns)} columns")`), ADD:

```python
    print("  [TOON v5.4] Building Realized Volatility Surface (Julia RV engine)...")
    rv_df = rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
    print(f"  [TOON v5.4] RV features injected: {len(rv_df.columns)} columns")
```

---

## STEP 4: MODIFY `daily_ml_engine.py`

### 4a. UPDATE import block (lines 25-30)

Replace the `from julia_bridge import` block:

```python
from julia_bridge import (
    add_target_fast,
    holographic_feature_engine_daily,
    kalman_structural_engine_daily,
    rv_feature_engine_daily,
    smc_feature_engine_daily,
)
```

### 4b. UPDATE import from universal_ml_engine (lines 31-47)

REMOVE `inject_realized_volatility_features,` from the import list.

### 4c. REPLACE RV injection (lines 484-494)

Replace:
```python
    df_1d = inject_realized_volatility_features(
        df_1d,
        "1D",
        htf_frames={
            "1W": df_1w,
            "1M": df_1m,
            "3M": df_3m,
            "6M": df_6m,
            "12M": df_12m,
        },
    )
```

With:
```python
    print("  [TOON DAILY] Building RV Surface (Julia RV engine)...")
    rv_df = rv_feature_engine_daily(df_1d, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        df_1d[col] = rv_df[col].values
    print(f"  [TOON DAILY] RV features injected: {len(rv_df.columns)} columns")
```

---

## STEP 5: MODIFY `daily_backtest_engine.py`

### 5a. UPDATE import block (lines 20-24)

Add `rv_feature_engine_daily,` to the `from julia_bridge import` block:

```python
from julia_bridge import (
    holographic_feature_engine_daily,
    kalman_structural_engine_daily,
    rv_feature_engine_daily,
    smc_feature_engine_daily,
)
```

### 5b. UPDATE import from universal_ml_engine (lines 25-34)

REMOVE `inject_realized_volatility_features,` from the import list.

### 5c. REPLACE RV injection (lines 153-163)

Replace:
```python
    df_1d = inject_realized_volatility_features(
        df_1d,
        "1D",
        htf_frames={
            "1W": df_1w,
            "1M": df_1m,
            "3M": df_3m,
            "6M": df_6m,
            "12M": df_12m,
        },
    )
```

With:
```python
    rv_df = rv_feature_engine_daily(df_1d, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        df_1d[col] = rv_df[col].values
```

---

## STEP 6: MODIFY `accuracy_guardrail.py`

### 6a. UPDATE import from julia_bridge (lines 42-49)

Add `rv_feature_engine_daily,` and `rv_feature_engine_fast,`:

```python
from julia_bridge import (
    add_target_fast,
    holographic_feature_engine_fast,
    kalman_structural_engine_daily,
    kalman_structural_engine_fast,
    rv_feature_engine_daily,
    rv_feature_engine_fast,
    smc_feature_engine_daily,
    smc_feature_engine_fast,
)
```

### 6b. UPDATE import from universal_ml_engine (lines 50-63)

REMOVE `inject_realized_volatility_features,` from the import list.

### 6c. ADD RV to `_build_1h_model_ready` function

After the Kalman injection block (after line 306: `df_full[col] = kf_df[col].values`), ADD:

```python
    rv_df = rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
```

### 6d. REPLACE RV in `_build_1d_model_ready` function (lines 355-365)

Replace:
```python
    df_1d = inject_realized_volatility_features(
        df_1d,
        "1D",
        htf_frames={
            "1W": df_1w,
            "1M": df_1m,
            "3M": df_3m,
            "6M": df_6m,
            "12M": df_12m,
        },
    )
```

With:
```python
    rv_df = rv_feature_engine_daily(df_1d, df_1w, df_1m, df_3m, df_6m, df_12m)
    for col in rv_df.columns:
        df_1d[col] = rv_df[col].values
```

---

## STEP 7: MODIFY `backtest_engine.py`

### 7a. ADD import (line 29-34)

Add `rv_feature_engine_fast,` to the `from julia_bridge import` block:

```python
from julia_bridge import (
    compute_backtest_bar_state_fast,
    holographic_feature_engine_fast as holographic_feature_engine,
    kalman_structural_engine_fast,
    rv_feature_engine_fast,
    smc_feature_engine_fast,
)
```

### 7b. ADD RV injection

After the Kalman injection block (after line 654: `df_full[col] = kf_df[col].values`), ADD:

```python
    rv_df = rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
```

---

## STEP 8: MODIFY `live_inference.py`

### 8a. ADD import (line 31-35)

Add `rv_feature_engine_fast,` to the `from julia_bridge import` block:

```python
from julia_bridge import (
    holographic_feature_engine_fast as holographic_feature_engine,
    kalman_structural_engine_fast,
    rv_feature_engine_fast,
    smc_feature_engine_fast,
)
```

### 8b. ADD RV injection

After the SMC injection block (after line 137: `df_full[col] = smc_df[col].values`), ADD:

```python
    rv_df = rv_feature_engine_fast(df_1h_tail, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
```

---

## STEP 9: MODIFY `meta_strategy_selector.py`

### 9a. ADD import (line 28-32)

Add `rv_feature_engine_fast,` to the `from julia_bridge import` block:

```python
from julia_bridge import (
    holographic_feature_engine_fast as holographic_feature_engine,
    kalman_structural_engine_fast,
    rv_feature_engine_fast,
    smc_feature_engine_fast,
)
```

### 9b. ADD RV injection

After the Kalman injection block (after line 625: `df_full[col] = kf_df[col].values`), ADD:

```python
    rv_df = rv_feature_engine_fast(df_1h_labelled, df_1d, df_1w, df_1m)
    for col in rv_df.columns:
        df_full[col] = rv_df[col].values
```

---

## STEP 10: UPGRADE `data_vault/yfinance_vault.py`

### 10a. EXTEND annualization constants (line 22)

Replace:
```python
REALIZED_VOL_ANNUALIZATION = {"1H": 1512.0, "1D": 252.0, "1W": 52.0}
```

With:
```python
REALIZED_VOL_ANNUALIZATION = {"1H": 1512.0, "1D": 252.0, "1W": 52.0, "1M": 12.0, "3M": 4.0, "6M": 2.0, "12M": 1.0}
```

### 10b. REPLACE `_compute_realized_volatility` method (lines 1167-1191)

Replace the entire method body with:

```python
    def _compute_realized_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["realized_volatility"] = np.nan

        for timeframe, ann_factor in REALIZED_VOL_ANNUALIZATION.items():
            tf_mask = result["timeframe"] == timeframe
            if not tf_mask.any():
                continue

            tf_df = result.loc[tf_mask].sort_values("timestamp").copy()
            o = tf_df["open"].to_numpy(dtype=float)
            h = tf_df["high"].to_numpy(dtype=float)
            lo = tf_df["low"].to_numpy(dtype=float)
            c = tf_df["close"].to_numpy(dtype=float)
            n_bars = len(c)
            window = 21

            rv_yz = np.full(n_bars, np.nan)

            with np.errstate(divide="ignore", invalid="ignore"):
                log_oc = np.log(np.maximum(c, 1e-9) / np.maximum(o, 1e-9))
                log_co = np.zeros(n_bars)
                log_co[1:] = np.log(np.maximum(o[1:], 1e-9) / np.maximum(c[:-1], 1e-9))
                log_hc = np.log(np.maximum(h, 1e-9) / np.maximum(c, 1e-9))
                log_ho = np.log(np.maximum(h, 1e-9) / np.maximum(o, 1e-9))
                log_lc = np.log(np.maximum(lo, 1e-9) / np.maximum(c, 1e-9))
                log_lo = np.log(np.maximum(lo, 1e-9) / np.maximum(o, 1e-9))
                rs = log_hc * log_ho + log_lc * log_lo

            k = 0.34 / (1.34 + (window + 1) / (window - 1))

            for i in range(window, n_bars):
                s = slice(i - window + 1, i + 1)
                co_s = log_co[max(i - window + 2, 1):i + 1]
                if len(co_s) < 2:
                    continue
                mu_o = co_s.mean()
                sigma_o2 = float(np.mean((co_s - mu_o) ** 2))
                oc_s = log_oc[s]
                mu_c = oc_s.mean()
                sigma_c2 = float(np.mean((oc_s - mu_c) ** 2))
                sigma_rs2 = float(rs[s].mean())
                yz_var = sigma_o2 + k * sigma_rs2 + (1.0 - k) * sigma_c2
                if np.isfinite(yz_var) and yz_var > 0:
                    rv_yz[i] = np.sqrt(yz_var * ann_factor)

            rv_series = pd.Series(rv_yz, index=tf_df.index)
            rv_series = rv_series.bfill().fillna(0.0)
            result.loc[tf_df.index, "realized_volatility"] = rv_series.to_numpy()

        return self._copy_attrs(df, result)
```

---

## STEP 11: DELETE DEAD CODE

DELETE the file `/home/km/Universal-ML/realized_vol.py`.

---

## VERIFICATION

### Syntax check:
```bash
julia -e 'include("ToonMath.jl"); println("Julia OK: ", length(methods(ToonMath.compute_rv_features)))'
uv run python -m py_compile julia_bridge.py universal_ml_engine.py daily_ml_engine.py daily_backtest_engine.py backtest_engine.py live_inference.py meta_strategy_selector.py accuracy_guardrail.py data_vault/yfinance_vault.py
```

### Smoke test:
```bash
uv run python -c "
from julia_bridge import rv_feature_engine_fast, rv_feature_engine_daily
import pandas as pd, numpy as np
n = 500
np.random.seed(42)
px = np.random.randn(n).cumsum() + 100
df = pd.DataFrame({
    'time': pd.date_range('2024-01-01', periods=n, freq='h'),
    'open': px + np.random.randn(n) * 0.2,
    'high': px + abs(np.random.randn(n)),
    'low': px - abs(np.random.randn(n)),
    'close': px + np.random.randn(n) * 0.3,
    'volume': np.abs(np.random.randn(n)) * 1e6,
})
r = rv_feature_engine_fast(df)
assert r.shape[1] == 11, f'Primary: {r.shape[1]}, expected 11'
r2 = rv_feature_engine_fast(df, df, df, df)
assert r2.shape[1] == 47, f'1H full: {r2.shape[1]}, expected 47'
r3 = rv_feature_engine_daily(df, df, df, df, df, df)
assert r3.shape[1] == 71, f'1D full: {r3.shape[1]}, expected 71'
assert r2['rv_1h_yz_pctrank'].between(0, 1).all()
assert r2['rv_1h_range_eff'].between(0.2, 5.0).all()
assert (r2['rv_1h_vov'] >= 0).all()
assert r2['rv_1h_jump_ratio'].between(0, 1).all()
assert r2['rv_1h_rv_asym'].between(0.1, 10.0).all()
assert r2['rv_1h_rv_skew'].between(-5, 5).all()
print('ALL PASS')
"
```

### Dead code check:
```bash
test ! -f realized_vol.py && echo 'PASS' || echo 'FAIL'
```

---

## FEATURE COUNT FINAL

| Lane | Primary | HTFs | Per HTF (11 + 1 term) | Total |
|------|---------|------|-----------------------|-------|
| 1H   | 11      | 3    | 12                    | **47** |
| 1D   | 11      | 5    | 12                    | **71** |

### 11 Features per timeframe

| # | Key | Category | What it measures | Why institutional |
|---|-----|----------|------------------|-------------------|
| 1 | `yz_log` | Level | log(Yang-Zhang annual RV) | Handles overnight gaps, 3-component |
| 2 | `yz_z` | Regime | Z-score vs 63-bar history | Regime detection |
| 3 | `yz_pctrank` | Regime | Bounded [0,1] rank | Tree-friendly regime |
| 4 | `yz_trend` | Dynamics | Short/long MA ratio | Vol momentum |
| 5 | `yz_shock` | Dynamics | 1-bar surprise | Event detection |
| 6 | `vov` | Dynamics | Vol-of-vol (2nd moment) | Regime instability |
| 7 | `pk_log` | Structure | log(Parkinson annual RV) | Range efficiency |
| 8 | `range_eff` | Structure | Parkinson/YZ ratio | Market microstructure |
| 9 | `jump_ratio` | Structure | (Var - BPV)/Var | Jump vs diffusion decomposition |
| 10 | `rv_asym` | Asymmetry | Down-vol/Up-vol | Leverage effect (Black 1976) |
| 11 | `rv_skew` | Asymmetry | Realized skewness | Crash/melt-up tail asymmetry |

---

## FEATURE ENGINEERING FLOW — HOW `rv_*` ENTERS THE MODEL

**NO CODE CHANGES NEEDED for this section. This documents the existing pipeline behavior.**

### Auto-discovery mechanism

The feature selection pipeline in `universal_ml_engine.py` (line 1952) uses:

```python
all_holo_cols = [c for c in df_full.columns if c not in NON_FEATURE_COLS]
```

`NON_FEATURE_COLS_SET` (lines 70-116) contains raw scaffold columns: `time`, `open`, `high`, `low`, `close`, `volume`, `atr14`, `target`, raw OHLCV prefixed `d_*/w_*/m_*`, and `realized_volatility`.

**`rv_*` prefixed columns are NOT in `NON_FEATURE_COLS_SET`.**

Therefore all 47 (1H) or 71 (1D) `rv_*` columns automatically become feature selection candidates. No manual registration needed.

### Pipeline stages the rv_* features pass through

```
rv_feature_engine_fast/daily()
  → [47 or 71 rv_* columns injected into df_full]
    → NON_FEATURE_COLS filter (line 1952): rv_* columns PASS (not excluded)
      → inf/NaN sanitization (line 1965-1968): fill 0.0
        → feature_selection_pipeline() (line 1986):
          Stage 1: Correlation filter (remove >0.95 correlated pairs)
            → Expect: rv_1h_yz_log and rv_1d_yz_log may correlate — one survives
          Stage 2: Probe ranking (rank against random noise features)
            → Only features beating random probes survive
          Stage 3: Walk-forward validation (10-fold)
            → Final ~40 features out of ~2000 candidates
```

### Expected survivor dynamics

From the 47/71 candidates, feature selection typically retains **5-15 RV features**:

| Feature | Survival likelihood | Reason |
|---|---|---|
| `rv_1h_yz_log` | HIGH | Primary vol level — uncorrelated with geometry features |
| `rv_1h_rv_asym` | HIGH | Unique signal — no other feature captures up/down vol split |
| `rv_1h_jump_ratio` | HIGH | Unique signal — jump vs diffusion decomposition |
| `rv_1h_yz_pctrank` | MEDIUM | Tree-friendly bounded regime — but may correlate with `yz_z` |
| `rv_1h_vov` | MEDIUM | 2nd moment is distinct but noisier than level features |
| `rv_1h_rv_skew` | MEDIUM | Tail info — partially captured by shock feature |
| `rv_1h_yz_z` | LOW-MEDIUM | May lose to `yz_pctrank` in correlation filter |
| `rv_1h_yz_trend` | LOW | Trend captured by other features (Kalman vel, etc.) |
| `rv_1h_yz_shock` | LOW | 1-bar noise — high variance, often pruned |
| HTF term structure | MEDIUM | `rv_term_1d_1h` often survives — cross-TF vol slope is unique |

### Key exclusion: raw `realized_volatility` column

The raw Yang-Zhang value stored in the database (`realized_volatility` column) is ALREADY in `NON_FEATURE_COLS_SET` (line 114). It is never a model feature. Only the engineered `rv_*` features can enter the model.

### Cross-lane consistency

| Component | 1H lane | 1D lane |
|---|---|---|
| Function | `rv_feature_engine_fast` | `rv_feature_engine_daily` |
| Primary features | 11 (`rv_1h_*`) | 11 (`rv_1d_*`) |
| HTF features | 36 (`rv_1d_*`, `rv_1w_*`, `rv_1m_*` + terms) | 60 (`rv_1w_*` through `rv_12m_*` + terms) |
| Feature discovery | Via `all_holo_cols` (line 1952) | Via `all_holo_cols` in daily_ml_engine.py (same pattern) |
| Selection pipeline | `feature_selection_pipeline()` | Same pipeline via walk-forward |
| Saved artifact | `{symbol}_1H_features.txt` | `{symbol}_1D_features.txt` |
| Consumer files | backtest, live_inference, meta_strategy_selector | daily_backtest, accuracy_guardrail |

### What Codex MUST NOT do

1. **DO NOT add `rv_*` to `NON_FEATURE_COLS_SET`** — that would exclude them from the model.
2. **DO NOT manually register features** — the pipeline auto-discovers any column not in the exclusion set.
3. **DO NOT change the injection order** — RV must inject AFTER holographic/SMC/Kalman but BEFORE `merge_higher_tf` (which handles raw OHLCV merge, not engineered features).
4. **DO NOT duplicate RV in `merge_higher_tf`** — the RV bridge handles its own HTF searchsorted alignment internally.
