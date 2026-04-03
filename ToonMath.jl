"""
ToonMath.jl — TOON v5.0 Julia Microservice
===========================================
Bare-metal Julia port of:
  - holographic_engine.py  (_bbox, _dna, _grammar, _spectral, _skeleton, _confluence, _process_tf)
  - universal_ml_engine.py  (simulate_trade_path_from_arrays_jit, _add_target_loop_jit)

Constraints enforced:
  R1  1-based Julia indexing; all Python entry_idx offsets converted (+1).
  R2  Time alignment via searchsortedlast on Int64 nanosecond timestamps.
  R3  @views + @inbounds throughout; no heap allocation inside 1:N loops.
  R4  Spectral = rolling autocorrelation via Statistics.cor — no FFTW.
  R5  All PyArray inputs explicitly typed to prevent Any-typed LLVM paths.
"""

module ToonMath

using PythonCall
using Statistics: cor, median

# ─────────────────────────────────────────────────────────────
# CONSTANTS (mirror Python)
# ─────────────────────────────────────────────────────────────

const HOLO_WINDOWS_1H = (3, 5, 8, 13, 21)
const HOLO_WINDOWS_1D = (3, 5, 8, 13, 21)
const HOLO_WINDOWS_1W = (3, 5, 8)
const HOLO_WINDOWS_1M = (3, 5)
const HOLO_WINDOWS_3M = (3, 5)
const SKEL_PROMINENCE  = 0.05
const BARRIER_ATR_MULT      = 1.25
const BARRIER_HORIZON_BARS  = 24
const TP1_R_MULT            = 1.0
const TP2_R_MULT            = 2.0
const TP1_FRACTION          = 0.50
const TP2_FRACTION          = 0.25
const RUNNER_FRACTION       = 0.25
const TRAIL_R_MULT          = 1.0
const EXEC_FEE_PCT          = 0.0005
const EXEC_SLIPPAGE_BPS     = 0.0003


# ─────────────────────────────────────────────────────────────
# INTERNAL PRIMITIVE HELPERS
# ─────────────────────────────────────────────────────────────

@inline _lift_stop(stop::Float64, candidate::Float64, is_long::Bool)::Float64 =
    is_long ? max(stop, candidate) : min(stop, candidate)

@inline _favorable_touch(price::Float64, level::Float64, is_long::Bool)::Bool =
    is_long ? price >= level : price <= level

@inline _adverse_touch(price::Float64, level::Float64, is_long::Bool)::Bool =
    is_long ? price <= level : price >= level

@inline function _realized_r(entry_price::Float64, exit_price::Float64,
                              risk_dist::Float64, fraction::Float64,
                              is_long::Bool)::Float64
    gross = is_long ? (exit_price - entry_price) : (entry_price - exit_price)
    return (gross / (risk_dist + 1e-9)) * fraction
end

@inline function _fee_r(price::Float64, risk_dist::Float64,
                        fraction::Float64, fee_pct::Float64)::Float64
    return (price * fee_pct / (risk_dist + 1e-9)) * fraction
end


# ─────────────────────────────────────────────────────────────
# BOUNDING BOX NORMALISER
# ─────────────────────────────────────────────────────────────

"""
    _bbox(o, h, l, c, v, start, stop) -> named tuple of slices

Normalise the window [start, stop) (1-based, half-open) of OHLCV arrays.
Returns a NamedTuple of Float64 vectors derived from that slice.
No heap allocation for the scalars; slice views are @views.
"""
function _bbox(
    src_o::Vector{Float64}, src_h::Vector{Float64},
    src_l::Vector{Float64}, src_c::Vector{Float64},
    src_v::Vector{Float64},
    start::Int, stop::Int,   # Julia 1-based, stop is exclusive upper bound
)
    n = stop - start  # window length
    @assert n > 0 "bbox window must be positive"

    @views begin
        o_w = src_o[start:stop-1]
        h_w = src_h[start:stop-1]
        l_w = src_l[start:stop-1]
        c_w = src_c[start:stop-1]
        v_w = src_v[start:stop-1]
    end

    hi = maximum(h_w)
    lo = minimum(l_w)
    rng = hi - lo

    if rng < 1e-9
        med = median(c_w)
        rng = max(med * 0.00001, 1e-9)
    end

    inv_rng = 1.0 / (rng + 1e-9)

    norm_o = clamp.((o_w .- lo) .* inv_rng, 0.0, 1.0)
    norm_h = clamp.((h_w .- lo) .* inv_rng, 0.0, 1.0)
    norm_l = clamp.((l_w .- lo) .* inv_rng, 0.0, 1.0)
    norm_c = clamp.((c_w .- lo) .* inv_rng, 0.0, 1.0)

    vols = @. ifelse(isnan(v_w), 0.0, v_w)
    total_vol = sum(vols)
    vol_frac = total_vol > 1e-9 ? clamp.(vols ./ total_vol, 0.0, 1.0) :
                                   fill(1.0 / n, n)

    prev_close = Vector{Float64}(undef, n)
    prev_close[1] = c_w[1]
    @inbounds for k in 2:n
        prev_close[k] = c_w[k-1]
    end
    price_move = @. (2.0 * c_w) - prev_close - o_w

    norm_bias = clamp.(
        ((((c_w .- lo) .- (hi .- c_w)) ./ (rng + 1e-9)) .+ 1.0) ./ 2.0,
        0.0, 1.0
    )

    buy_vol  = sum(ifelse(pm > 0.0, vv, 0.0) for (pm, vv) in zip(price_move, vols))
    sell_vol = sum(ifelse(pm < 0.0, vv, 0.0) for (pm, vv) in zip(price_move, vols))
    vol_sum  = sum(vols)
    volume_bias = clamp((buy_vol - sell_vol) / (vol_sum + 1e-9), -1.0, 1.0)

    return (
        norm_o = norm_o,
        norm_h = norm_h,
        norm_l = norm_l,
        norm_c = norm_c,
        vol_frac  = vol_frac,
        norm_bias = norm_bias,
        volume_bias = volume_bias,
    )
end


# ─────────────────────────────────────────────────────────────
# LAYER 1 — DNA STRAND
# ─────────────────────────────────────────────────────────────

"""
    _dna!(out, b, tf, w, thermo)

Write DNA features for the window into `out` Dict.
`thermo` is a NamedTuple of optional thermodynamic arrays (may be nothing).
Bar indices are 1-based; key names use 1-based bar numbers (identical to Python's k+1).
"""
function _dna!(out::Dict{String,Float64}, b, tf::String, w::Int, thermo)
    nc  = b.norm_c
    nb  = b.norm_bias
    vb  = b.volume_bias
    n   = length(nc)

    for k in 1:n
        p = string(tf, "_w", w, "_bar", k)

        out[string(p, "_close")]       = nc[k]
        out[string(p, "_norm_bias")]   = nb[k]
        out[string(p, "_volume_bias")] = vb

        if thermo !== nothing
            out[string(p, "_basis_pct")] = thermo.basis_pct[k]
            out[string(p, "_basis_z")] = thermo.basis_z[k]
            out[string(p, "_basis_v5")] = thermo.basis_v5[k]
            out[string(p, "_basis_v10")] = thermo.basis_v10[k]
            out[string(p, "_session_pos")] = thermo.session_pos[k]
            out[string(p, "_eod_momentum")] = thermo.eod_momentum[k]
            out[string(p, "_fib_a_close_pos")] = thermo.fib_a_close_pos[k]
            out[string(p, "_fib_a_zone")] = thermo.fib_a_zone[k]
            out[string(p, "_fib_a_wick_rej_bull")] = thermo.fib_a_wick_rej_bull[k]
            out[string(p, "_fib_a_wick_rej_bear")] = thermo.fib_a_wick_rej_bear[k]
            out[string(p, "_fib_a_body_acc")] = thermo.fib_a_body_acc[k]
            out[string(p, "_fib_a_ext_pct")] = thermo.fib_a_ext_pct[k]
            out[string(p, "_fib_b_close_pos")] = thermo.fib_b_close_pos[k]
            out[string(p, "_fib_b_zone")] = thermo.fib_b_zone[k]
            out[string(p, "_fib_b_wick_rej_bull")] = thermo.fib_b_wick_rej_bull[k]
            out[string(p, "_fib_b_wick_rej_bear")] = thermo.fib_b_wick_rej_bear[k]
            out[string(p, "_fib_b_body_acc")] = thermo.fib_b_body_acc[k]
            out[string(p, "_fib_b_ext_pct")] = thermo.fib_b_ext_pct[k]
            out[string(p, "_fib_c_close_pos")] = thermo.fib_c_close_pos[k]
            out[string(p, "_fib_c_zone")] = thermo.fib_c_zone[k]
            out[string(p, "_fib_c_wick_rej_bull")] = thermo.fib_c_wick_rej_bull[k]
            out[string(p, "_fib_c_wick_rej_bear")] = thermo.fib_c_wick_rej_bear[k]
            out[string(p, "_fib_c_body_acc")] = thermo.fib_c_body_acc[k]
            out[string(p, "_fib_c_ext_pct")] = thermo.fib_c_ext_pct[k]
        else
            out[string(p, "_basis_pct")] = 0.0
            out[string(p, "_basis_z")] = 0.0
            out[string(p, "_basis_v5")] = 0.0
            out[string(p, "_basis_v10")] = 0.0
            out[string(p, "_session_pos")] = 0.0
            out[string(p, "_eod_momentum")] = 0.0
            out[string(p, "_fib_a_close_pos")] = 0.0
            out[string(p, "_fib_a_zone")] = 0.0
            out[string(p, "_fib_a_wick_rej_bull")] = 0.0
            out[string(p, "_fib_a_wick_rej_bear")] = 0.0
            out[string(p, "_fib_a_body_acc")] = 0.0
            out[string(p, "_fib_a_ext_pct")] = 0.0
            out[string(p, "_fib_b_close_pos")] = 0.0
            out[string(p, "_fib_b_zone")] = 0.0
            out[string(p, "_fib_b_wick_rej_bull")] = 0.0
            out[string(p, "_fib_b_wick_rej_bear")] = 0.0
            out[string(p, "_fib_b_body_acc")] = 0.0
            out[string(p, "_fib_b_ext_pct")] = 0.0
            out[string(p, "_fib_c_close_pos")] = 0.0
            out[string(p, "_fib_c_zone")] = 0.0
            out[string(p, "_fib_c_wick_rej_bull")] = 0.0
            out[string(p, "_fib_c_wick_rej_bear")] = 0.0
            out[string(p, "_fib_c_body_acc")] = 0.0
            out[string(p, "_fib_c_ext_pct")] = 0.0
        end
    end
end


# ─────────────────────────────────────────────────────────────
# LAYER 2 — GRAMMAR
# ─────────────────────────────────────────────────────────────

"""
    _grammar!(out, b, tf, w)

Lag-matrix encoding. Mirrors Python: lag 0 = most recent bar (-1 in Python = n in Julia).
"""
function _grammar!(out::Dict{String,Float64}, b, tf::String, w::Int)
    w < 3 && return
    nc = b.norm_c
    nb = b.norm_bias
    vb = b.volume_bias
    n  = length(nc)

    max_lag = min(w, 21) - 1   # lags 0..max_lag  (same as Python range(min(w,21)))
    for lag in 0:max_lag
        idx = n - lag          # Julia 1-based equivalent of Python's (n-1) - lag
        if idx >= 1
            out[string(tf, "_w", w, "_gram_nb_lag", lag)] = nb[idx]
            out[string(tf, "_w", w, "_gram_vb_lag", lag)] = vb
        else
            out[string(tf, "_w", w, "_gram_nb_lag", lag)] = 0.0
            out[string(tf, "_w", w, "_gram_vb_lag", lag)] = 0.0
        end
    end
end


# ─────────────────────────────────────────────────────────────
# LAYER 3 — SPECTRAL  (rolling autocorrelation — NO FFT)
# ─────────────────────────────────────────────────────────────

"""
    _spectral!(out, b, tf, w)

Rolling autocorrelation at Fibonacci lags [1,2,3,5,8] using Statistics.cor.
Only active for w >= 10. NaN/Inf guarded → 0.0.
"""
function _spectral!(out::Dict{String,Float64}, b, tf::String, w::Int)
    w < 10 && return
    sig = b.norm_c
    len = length(sig)

    for lag in (1, 2, 3, 5, 8)
        key = string(tf, "_w", w, "_fft_nc_ac", lag)
        if len > lag
            # sig[lag+1:end] vs sig[1:end-lag]  ≡  Python sig[lag:] vs sig[:-lag]
            @views a = sig[lag+1:end]
            @views b_ = sig[1:end-lag]
            r = try
                c = cor(a, b_)
                isfinite(c) ? c : 0.0
            catch
                0.0
            end
            out[key] = r
        else
            out[key] = 0.0
        end
    end
end


# ─────────────────────────────────────────────────────────────
# LAYER 4 — SKELETON  (w >= 5)
# ─────────────────────────────────────────────────────────────

"""
    _skeleton!(out, b, tf, w)

SMC-aware skeleton: peaks, troughs, BOS/CHoCH detection, liquidity grab,
order-block validity. Mirrors Python _skeleton exactly including prominence filter.
"""
function _skeleton!(out::Dict{String,Float64}, b, tf::String, w::Int)
    w < 5 && return
    nc = b.norm_c
    vf = b.vol_frac
    nb = b.norm_bias
    n  = length(nc)
    prom = SKEL_PROMINENCE

    peaks   = Int[]
    troughs = Int[]

    @inbounds for k in 2:n-1
        left, mid, right = nc[k-1], nc[k], nc[k+1]
        if mid > left && mid > right && (mid - max(left, right)) >= prom
            push!(peaks, k)
        elseif mid < left && mid < right && (min(left, right) - mid) >= prom
            push!(troughs, k)
        end
    end

    # Only use peaks/troughs that are NOT the last bar (Python: k < n-1 → Julia: k < n)
    pk_vals = [nc[k] for k in peaks   if k < n]
    tr_vals = [nc[k] for k in troughs if k < n]

    last_swing_high     = isempty(pk_vals) ? 1.0 : pk_vals[end]
    last_protected_low  = isempty(tr_vals) ? 0.0 : tr_vals[end]

    nc_last = nc[end]
    nb_last = nb[end]

    skel_smc_phase = 0.0
    if nc_last > last_swing_high
        pk_slope = length(pk_vals) >= 2 ? pk_vals[end] - pk_vals[1] : 0.0
        skel_smc_phase = pk_slope >= 0.0 ? 1.0 : 2.0
    elseif nc_last < last_protected_low
        tr_slope = length(tr_vals) >= 2 ? tr_vals[end] - tr_vals[1] : 0.0
        skel_smc_phase = tr_slope <= 0.0 ? -1.0 : -2.0
    end

    skel_liquidity_grab = (nc_last < last_protected_low && nb_last > 0.7) ? 1.0 : 0.0

    vol_at_last_peak = isempty(peaks) ? 0.0 : vf[peaks[end]]
    skel_ob_validity = vol_at_last_peak * abs(skel_smc_phase)

    p = string(tf, "_w", w, "_skel")
    out[string(p, "_smc_phase")]       = skel_smc_phase
    out[string(p, "_liquidity_grab")]  = skel_liquidity_grab
    out[string(p, "_ob_validity")]     = skel_ob_validity
    out[string(p, "_last_swing_high")] = last_swing_high
    out[string(p, "_last_swing_low")]  = last_protected_low
end


# ─────────────────────────────────────────────────────────────
# CONFLUENCE (cross-TF)
# ─────────────────────────────────────────────────────────────

function _confluence!(out::Dict{String,Float64}, row_feats::Dict{String,Float64})
    g(k) = get(row_feats, k, 0.0)

    phase_1h = g("1h_w13_skel_smc_phase")
    phase_1d = g("1d_w13_skel_smc_phase")
    phase_1w = g("1w_w8_skel_smc_phase")

    mtf_conf_smc_sync =
        (phase_1h > 0.0 && phase_1d > 0.0 && phase_1w > 0.0) ?  1.0 :
        (phase_1h < 0.0 && phase_1d < 0.0 && phase_1w < 0.0) ? -1.0 : 0.0

    h1_close = g("1h_w13_bar13_close")
    d1_high  = get(row_feats, "1d_w13_skel_last_swing_high", 1.0)
    d1_low   = get(row_feats, "1d_w13_skel_last_swing_low",  0.0)
    pd_zone  = (h1_close - d1_low) / (d1_high - d1_low + 1e-9)

    out["mtf_conf_smc_sync"] = mtf_conf_smc_sync
    out["mtf_conf_pd_zone"]  = pd_zone
end


# ─────────────────────────────────────────────────────────────
# PROCESS ONE TIMEFRAME
# ─────────────────────────────────────────────────────────────

"""
    _process_tf!(all_rows, base_times_ns, src_o, src_h, src_l, src_c, src_v,
                 src_times_ns, tf_label, windows, is_primary, thermo)

For each base bar i, find the available window, compute bbox+layers,
mutate all_rows[i] in place.

- base_times_ns / src_times_ns : Vector{Int64} of Unix nanoseconds
- is_primary : when true, uses direct index (avail = i-1 in 0-based → i in 1-based)
- thermo     : NamedTuple of optional sliceable thermodynamic arrays, or nothing
"""
function _process_tf!(
    all_rows::Vector{Dict{String,Float64}},
    base_times_ns::Vector{Int64},
    src_o::Vector{Float64}, src_h::Vector{Float64},
    src_l::Vector{Float64}, src_c::Vector{Float64},
    src_v::Vector{Float64},
    src_times_ns::Vector{Int64},
    tf_label::String,
    windows::NTuple,
    is_primary::Bool,
    thermo,
)
    n_base  = length(all_rows)
    n_src   = length(src_o)

    for i in 1:n_base
        # In Python: avail = i (0-based) for primary  →  i bars 0…i-1 closed before bar i
        # In Julia 1-based: avail = i  means bars 1…i are closed; window end = avail (exclusive upper bound is avail+1)
        avail::Int = if is_primary
            i   # bars 1..i are available (bar i+1 is the "current" bar)
        else
            t = base_times_ns[i]
            # searchsortedlast returns last index where src_times_ns[k] <= t
            pos = searchsortedlast(src_times_ns, t)
            max(0, pos)
        end

        feat = all_rows[i]

        for w in windows
            start_jl = avail - w + 1   # 1-based start of window
            stop_jl  = avail           # 1-based inclusive end = exclusive upper is avail+1
            # In _bbox call we pass start=start_jl, stop=stop_jl+1 (exclusive)
            if start_jl < 1 || stop_jl < 1 || stop_jl - start_jl + 1 < w
                continue   # not enough history
            end

            b = _bbox(src_o, src_h, src_l, src_c, src_v, start_jl, stop_jl + 1)

            # Build thermo slice for this window if available
            local thermo_slice
            if thermo !== nothing
                slice_range = start_jl:stop_jl
                thermo_slice = (
                    basis_pct = thermo.basis_pct[slice_range],
                    basis_z = thermo.basis_z[slice_range],
                    basis_v5 = thermo.basis_v5[slice_range],
                    basis_v10 = thermo.basis_v10[slice_range],
                    session_pos = thermo.session_pos[slice_range],
                    eod_momentum = thermo.eod_momentum[slice_range],
                    fib_a_close_pos = thermo.fib_a_close_pos[slice_range],
                    fib_a_zone = thermo.fib_a_zone[slice_range],
                    fib_a_wick_rej_bull = thermo.fib_a_wick_rej_bull[slice_range],
                    fib_a_wick_rej_bear = thermo.fib_a_wick_rej_bear[slice_range],
                    fib_a_body_acc = thermo.fib_a_body_acc[slice_range],
                    fib_a_ext_pct = thermo.fib_a_ext_pct[slice_range],
                    fib_b_close_pos = thermo.fib_b_close_pos[slice_range],
                    fib_b_zone = thermo.fib_b_zone[slice_range],
                    fib_b_wick_rej_bull = thermo.fib_b_wick_rej_bull[slice_range],
                    fib_b_wick_rej_bear = thermo.fib_b_wick_rej_bear[slice_range],
                    fib_b_body_acc = thermo.fib_b_body_acc[slice_range],
                    fib_b_ext_pct = thermo.fib_b_ext_pct[slice_range],
                    fib_c_close_pos = thermo.fib_c_close_pos[slice_range],
                    fib_c_zone = thermo.fib_c_zone[slice_range],
                    fib_c_wick_rej_bull = thermo.fib_c_wick_rej_bull[slice_range],
                    fib_c_wick_rej_bear = thermo.fib_c_wick_rej_bear[slice_range],
                    fib_c_body_acc = thermo.fib_c_body_acc[slice_range],
                    fib_c_ext_pct = thermo.fib_c_ext_pct[slice_range],
                )
            else
                thermo_slice = nothing
            end

            _dna!(feat, b, tf_label, w, thermo_slice)
            w >= 3  && _grammar!(feat, b, tf_label, w)
            w >= 8  && _spectral!(feat, b, tf_label, w)
            w >= 5  && _skeleton!(feat, b, tf_label, w)
        end
    end
end


# ─────────────────────────────────────────────────────────────
# PUBLIC API: compute_holographic_features
# ─────────────────────────────────────────────────────────────

"""
    compute_holographic_features(
        base_times_ns,
        o_1h, h_1h, l_1h, c_1h, v_1h,
        t_1d, o_1d, h_1d, l_1d, c_1d, v_1d,
        t_1w, o_1w, h_1w, l_1w, c_1w, v_1w,
        t_1m, o_1m, h_1m, l_1m, c_1m, v_1m;
        thermo_1h = nothing,
    ) -> Dict{String, Vector{Float64}}

All time arrays must be sorted ascending Int64 Unix nanoseconds.
Returns a Dict mapping feature name → Float64 column vector of length N_base.
Python side must reconstruct a DataFrame from this dict.
NaN is used as the missing-data sentinel (matches Python behaviour).
"""
function compute_holographic_features(
    base_times_ns::AbstractVector{Int64},
    o_1h::AbstractVector{Float64}, h_1h::AbstractVector{Float64},
    l_1h::AbstractVector{Float64}, c_1h::AbstractVector{Float64}, v_1h::AbstractVector{Float64},
    t_1d::AbstractVector{Int64},
    o_1d::AbstractVector{Float64}, h_1d::AbstractVector{Float64},
    l_1d::AbstractVector{Float64}, c_1d::AbstractVector{Float64}, v_1d::AbstractVector{Float64},
    t_1w::AbstractVector{Int64},
    o_1w::AbstractVector{Float64}, h_1w::AbstractVector{Float64},
    l_1w::AbstractVector{Float64}, c_1w::AbstractVector{Float64}, v_1w::AbstractVector{Float64},
    t_1m::AbstractVector{Int64},
    o_1m::AbstractVector{Float64}, h_1m::AbstractVector{Float64},
    l_1m::AbstractVector{Float64}, c_1m::AbstractVector{Float64}, v_1m::AbstractVector{Float64};
    thermo_1h = nothing,
)::Dict{String, Vector{Float64}}
    return compute_holographic_features(
        collect(Int64, base_times_ns),
        collect(Float64, o_1h), collect(Float64, h_1h),
        collect(Float64, l_1h), collect(Float64, c_1h), collect(Float64, v_1h),
        collect(Int64, t_1d),
        collect(Float64, o_1d), collect(Float64, h_1d),
        collect(Float64, l_1d), collect(Float64, c_1d), collect(Float64, v_1d),
        collect(Int64, t_1w),
        collect(Float64, o_1w), collect(Float64, h_1w),
        collect(Float64, l_1w), collect(Float64, c_1w), collect(Float64, v_1w),
        collect(Int64, t_1m),
        collect(Float64, o_1m), collect(Float64, h_1m),
        collect(Float64, l_1m), collect(Float64, c_1m), collect(Float64, v_1m);
        thermo_1h = thermo_1h,
    )
end

function compute_holographic_features(
    base_times_ns::Vector{Int64},
    o_1h::Vector{Float64}, h_1h::Vector{Float64},
    l_1h::Vector{Float64}, c_1h::Vector{Float64}, v_1h::Vector{Float64},
    t_1d::Vector{Int64},
    o_1d::Vector{Float64}, h_1d::Vector{Float64},
    l_1d::Vector{Float64}, c_1d::Vector{Float64}, v_1d::Vector{Float64},
    t_1w::Vector{Int64},
    o_1w::Vector{Float64}, h_1w::Vector{Float64},
    l_1w::Vector{Float64}, c_1w::Vector{Float64}, v_1w::Vector{Float64},
    t_1m::Vector{Int64},
    o_1m::Vector{Float64}, h_1m::Vector{Float64},
    l_1m::Vector{Float64}, c_1m::Vector{Float64}, v_1m::Vector{Float64};
    thermo_1h = nothing,
)::Dict{String, Vector{Float64}}

    n_base = length(base_times_ns)
    all_rows = [Dict{String,Float64}() for _ in 1:n_base]

    # 1H — primary (no timestamp alignment needed)
    _process_tf!(
        all_rows, base_times_ns,
        o_1h, h_1h, l_1h, c_1h, v_1h,
        Int64[],          # src_times not used for primary
        "1h", HOLO_WINDOWS_1H, true, thermo_1h,
    )

    # 1D
    if length(o_1d) >= 3
        _process_tf!(
            all_rows, base_times_ns,
            o_1d, h_1d, l_1d, c_1d, v_1d,
            t_1d, "1d", HOLO_WINDOWS_1D, false, nothing,
        )
    end

    # 1W
    if length(o_1w) >= 3
        _process_tf!(
            all_rows, base_times_ns,
            o_1w, h_1w, l_1w, c_1w, v_1w,
            t_1w, "1w", HOLO_WINDOWS_1W, false, nothing,
        )
    end

    # 1M
    if length(o_1m) >= 3
        _process_tf!(
            all_rows, base_times_ns,
            o_1m, h_1m, l_1m, c_1m, v_1m,
            t_1m, "1m", HOLO_WINDOWS_1M, false, nothing,
        )
    end

    # Confluence pass (reads from merged row dict, writes 2 keys)
    for i in 1:n_base
        _confluence!(all_rows[i], all_rows[i])
    end

    # Collect all unique feature keys first
    all_keys = Set{String}()
    for row in all_rows
        union!(all_keys, keys(row))
    end

    # Build output columns; NaN for missing entries
    result = Dict{String, Vector{Float64}}()
    for k in all_keys
        col = Vector{Float64}(undef, n_base)
        for i in 1:n_base
            col[i] = get(all_rows[i], k, NaN)
        end
        result[k] = col
    end

    return result
end


# ─────────────────────────────────────────────────────────────
# TRADE SIMULATION KERNEL
# ─────────────────────────────────────────────────────────────

"""
    _simulate_trade(opens, highs, lows, closes, entry_idx_jl, is_long,
                    risk_dist, tp1_dist, tp2_dist, trail_dist,
                    horizon, fee_pct, slippage_bps,
                    tp1_frac, tp2_frac, runner_frac)
        -> (total_r, exit_idx_jl, exit_reason, entry_price, last_fill_price, tp1_hit, tp2_hit, final_stop)

entry_idx_jl is already 1-based (caller converts from Python 0-based by +1).
Returns NaN tuple on invalid input.
"""
function _simulate_trade(
    opens::Vector{Float64}, highs::Vector{Float64},
    lows::Vector{Float64},  closes::Vector{Float64},
    entry_idx_jl::Int, is_long::Bool,
    risk_dist::Float64, tp1_dist::Float64, tp2_dist::Float64, trail_dist::Float64,
    horizon::Int, fee_pct::Float64, slippage_bps::Float64,
    tp1_frac::Float64, tp2_frac::Float64, runner_frac::Float64,
)::Tuple{Float64,Int,Int,Float64,Float64,Bool,Bool,Float64}

    n = length(opens)
    _nan_ret(idx) = (NaN, idx, 0, NaN, NaN, false, false, NaN)

    if entry_idx_jl > n || risk_dist <= 0.0 || !isfinite(risk_dist)
        return _nan_ret(entry_idx_jl)
    end

    raw_entry = opens[entry_idx_jl]
    if !isfinite(raw_entry)
        return _nan_ret(entry_idx_jl)
    end

    entry_price = is_long ? raw_entry * (1.0 + slippage_bps) :
                            raw_entry * (1.0 - slippage_bps)
    stop  = is_long ? entry_price - risk_dist : entry_price + risk_dist
    tp1   = is_long ? entry_price + tp1_dist  : entry_price - tp1_dist
    tp2   = is_long ? entry_price + tp2_dist  : entry_price - tp2_dist

    rem_tp1    = tp1_frac
    rem_tp2    = tp2_frac
    rem_runner = runner_frac

    total_r = -_fee_r(entry_price, risk_dist, 1.0, fee_pct)
    tp1_hit  = false
    tp2_hit  = false
    exit_reason = 2
    last_idx    = min(n, entry_idx_jl + max(horizon - 1, 0))
    exit_idx    = last_idx
    last_fill_price = entry_price

    broke = false
    @inbounds for j in entry_idx_jl:last_idx
        bar_open  = opens[j]
        bar_high  = highs[j]
        bar_low   = lows[j]
        bar_close = closes[j]

        if !isfinite(bar_open) || !isfinite(bar_high) || !isfinite(bar_low) || !isfinite(bar_close)
            exit_reason = 3; exit_idx = j; broke = true; break
        end

        # Gap open adverse
        if _adverse_touch(bar_open, stop, is_long)
            leftover = rem_tp1 + rem_tp2 + rem_runner
            if leftover > 0.0
                total_r += _realized_r(entry_price, bar_open, risk_dist, leftover, is_long)
                total_r -= _fee_r(bar_open, risk_dist, leftover, fee_pct)
                rem_tp1 = 0.0; rem_tp2 = 0.0; rem_runner = 0.0
                last_fill_price = bar_open
            end
            exit_reason = 4; exit_idx = j; broke = true; break
        end

        # Gap open TP1 fill
        if rem_tp1 > 0.0 && _favorable_touch(bar_open, tp1, is_long)
            total_r += _realized_r(entry_price, tp1, risk_dist, rem_tp1, is_long)
            total_r -= _fee_r(tp1, risk_dist, rem_tp1, fee_pct)
            rem_tp1 = 0.0; last_fill_price = tp1; tp1_hit = true
            stop = _lift_stop(stop, entry_price, is_long)
        end

        # Gap open TP2 fill
        if rem_tp2 > 0.0 && _favorable_touch(bar_open, tp2, is_long)
            total_r += _realized_r(entry_price, tp2, risk_dist, rem_tp2, is_long)
            total_r -= _fee_r(tp2, risk_dist, rem_tp2, fee_pct)
            rem_tp2 = 0.0; last_fill_price = tp2; tp2_hit = true
            stop = _lift_stop(stop, tp1, is_long)
        end

        rem_total = rem_tp1 + rem_tp2 + rem_runner
        if rem_total <= 0.0
            exit_reason = 5; exit_idx = j; broke = true; break
        end

        bar_best  = is_long ? bar_high : bar_low
        bar_worst = is_long ? bar_low  : bar_high
        fav_same_bar = (rem_tp1 > 0.0 && _favorable_touch(bar_best, tp1, is_long)) ||
                       (rem_tp2 > 0.0 && _favorable_touch(bar_best, tp2, is_long))
        adv_same_bar = _adverse_touch(bar_worst, stop, is_long)

        # Ambiguous bar: adverse wins
        if adv_same_bar && fav_same_bar
            leftover = rem_tp1 + rem_tp2 + rem_runner
            if leftover > 0.0
                total_r += _realized_r(entry_price, stop, risk_dist, leftover, is_long)
                total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                rem_tp1 = 0.0; rem_tp2 = 0.0; rem_runner = 0.0
                last_fill_price = stop
            end
            exit_reason = 6; exit_idx = j; broke = true; break
        end

        if adv_same_bar
            leftover = rem_tp1 + rem_tp2 + rem_runner
            if leftover > 0.0
                total_r += _realized_r(entry_price, stop, risk_dist, leftover, is_long)
                total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                rem_tp1 = 0.0; rem_tp2 = 0.0; rem_runner = 0.0
                last_fill_price = stop
            end
            exit_reason = 7; exit_idx = j; broke = true; break
        end

        # Intra-bar TP1
        if rem_tp1 > 0.0 && _favorable_touch(bar_best, tp1, is_long)
            total_r += _realized_r(entry_price, tp1, risk_dist, rem_tp1, is_long)
            total_r -= _fee_r(tp1, risk_dist, rem_tp1, fee_pct)
            rem_tp1 = 0.0; last_fill_price = tp1; tp1_hit = true
            stop = _lift_stop(stop, entry_price, is_long)
            rem_total = rem_tp1 + rem_tp2 + rem_runner
            if rem_total <= 0.0
                exit_reason = 8; exit_idx = j; broke = true; break
            end
            if _adverse_touch(bar_worst, stop, is_long)
                leftover = rem_tp1 + rem_tp2 + rem_runner
                if leftover > 0.0
                    total_r += _realized_r(entry_price, stop, risk_dist, leftover, is_long)
                    total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                    rem_tp1 = 0.0; rem_tp2 = 0.0; rem_runner = 0.0
                    last_fill_price = stop
                end
                exit_reason = 9; exit_idx = j; broke = true; break
            end
        end

        # Intra-bar TP2
        if rem_tp2 > 0.0 && _favorable_touch(bar_best, tp2, is_long)
            total_r += _realized_r(entry_price, tp2, risk_dist, rem_tp2, is_long)
            total_r -= _fee_r(tp2, risk_dist, rem_tp2, fee_pct)
            rem_tp2 = 0.0; last_fill_price = tp2; tp2_hit = true
            stop = _lift_stop(stop, tp1, is_long)
            rem_total = rem_tp1 + rem_tp2 + rem_runner
            if rem_total <= 0.0
                exit_reason = 10; exit_idx = j; broke = true; break
            end
            if _adverse_touch(bar_worst, stop, is_long)
                leftover = rem_tp1 + rem_tp2 + rem_runner
                if leftover > 0.0
                    total_r += _realized_r(entry_price, stop, risk_dist, leftover, is_long)
                    total_r -= _fee_r(stop, risk_dist, leftover, fee_pct)
                    rem_tp1 = 0.0; rem_tp2 = 0.0; rem_runner = 0.0
                    last_fill_price = stop
                end
                exit_reason = 11; exit_idx = j; broke = true; break
            end
        end

        # Trailing stop update
        if rem_runner > 0.0 && trail_dist > 0.0
            trail_candidate = is_long ? bar_close - trail_dist : bar_close + trail_dist
            stop = _lift_stop(stop, trail_candidate, is_long)
            if tp1_hit
                trail_base = tp2_hit ? tp1 : entry_price
                stop = _lift_stop(stop, trail_base, is_long)
            end
        end
    end  # bar loop

    # Time-exit: close at last bar
    if !broke
        leftover = rem_tp1 + rem_tp2 + rem_runner
        if leftover > 0.0
            fill_price = closes[last_idx]
            total_r += _realized_r(entry_price, fill_price, risk_dist, leftover, is_long)
            total_r -= _fee_r(fill_price, risk_dist, leftover, fee_pct)
            last_fill_price = fill_price
        end
        exit_reason = 2
        exit_idx    = last_idx
    end

    # Cleanup for BAD_BAR exit
    rem_total_final = rem_tp1 + rem_tp2 + rem_runner
    if rem_total_final > 0.0 && exit_reason == 3
        idx_to_fill = min(exit_idx, n)
        fill_price  = closes[idx_to_fill]
        total_r += _realized_r(entry_price, fill_price, risk_dist, rem_total_final, is_long)
        total_r -= _fee_r(fill_price, risk_dist, rem_total_final, fee_pct)
        last_fill_price = fill_price
    end

    return (total_r, exit_idx, exit_reason, entry_price, last_fill_price, tp1_hit, tp2_hit, stop)
end


# ─────────────────────────────────────────────────────────────
# PUBLIC API: add_target_loop
# ─────────────────────────────────────────────────────────────

"""
    compute_hurst_series(closes; window_h=100, default_value=0.5)

Compute the rolling Hurst-style rough-volatility series used by
`backtest_engine.run_backtest()`.

This mirrors the current Python implementation exactly:

- the first `window_h` slots stay at `default_value`
- each later slot uses the prior `window_h` closes
- the result is clipped into `[0.0, 1.0]`
"""
function compute_hurst_series(
    closes::AbstractVector{Float64};
    window_h::Int = 100,
    default_value::Float64 = 0.5,
)::Vector{Float64}
    src = closes isa Vector{Float64} ? closes : collect(Float64, closes)
    n = length(src)
    hurst = fill(default_value, n)

    if window_h < 2 || n <= window_h
        return hurst
    end

    diffs = Vector{Float64}(undef, window_h - 1)
    log_window = log(Float64(window_h))

    @inbounds for out_idx in (window_h + 1):n
        start_idx = out_idx - window_h

        diff_sum = 0.0
        for k in 1:(window_h - 1)
            val = src[start_idx + k] - src[start_idx + k - 1]
            diffs[k] = val
            diff_sum += val
        end

        diff_mean = diff_sum / Float64(window_h - 1)

        var_sum = 0.0
        for k in 1:(window_h - 1)
            centered = diffs[k] - diff_mean
            diffs[k] = centered
            var_sum += centered * centered
        end

        s = sqrt(var_sum / Float64(window_h - 1))
        if s > 0.0
            z = 0.0
            z_min = 0.0
            z_max = 0.0

            for k in 1:(window_h - 1)
                z += diffs[k]
                z_min = min(z_min, z)
                z_max = max(z_max, z)
            end

            r = z_max - z_min
            if r > 0.0
                hurst[out_idx] = clamp(log(r / s) / log_window, 0.0, 1.0)
            end
        end
    end

    return hurst
end

"""
    compute_backtest_bar_state(closes, probas, atrs, zscores, next_hours;
                               window_h=100, default_hurst=0.5,
                               conf_threshold=0.56, shock_z_abs=2.5,
                               min_hurst=0.45, eod_gate_hour=14)

Precompute the per-bar backtest state used by `backtest_engine.run_backtest()`.

Skip-code contract:
- `0`: tradable bar
- `1`: no prediction
- `2`: end-of-day gate
- `3`: low confidence
- `4`: invalid ATR
- `5`: thermodynamic shock
- `6`: low Hurst / anti-persistent noise
"""
function compute_backtest_bar_state(
    closes::AbstractVector{Float64},
    probas::AbstractVector{Float64},
    atrs::AbstractVector{Float64},
    zscores::AbstractVector{Float64},
    next_hours::AbstractVector{Int64};
    window_h::Int = 100,
    default_hurst::Float64 = 0.5,
    conf_threshold::Float64 = 0.56,
    shock_z_abs::Float64 = 2.5,
    min_hurst::Float64 = 0.45,
    eod_gate_hour::Int = 14,
)
    to_f64_vec(v) = v isa Vector{Float64} ? v : collect(Float64, v)
    to_i64_vec(v) = v isa Vector{Int64} ? v : collect(Int64, v)

    src_closes = to_f64_vec(closes)
    src_probas = to_f64_vec(probas)
    src_atrs = to_f64_vec(atrs)
    src_zscores = to_f64_vec(zscores)
    src_next_hours = to_i64_vec(next_hours)

    n = length(src_closes)
    if length(src_probas) != n || length(src_atrs) != n || length(src_zscores) != n || length(src_next_hours) != n
        throw(ArgumentError("compute_backtest_bar_state inputs must have identical lengths"))
    end

    hurst = compute_hurst_series(
        src_closes;
        window_h = window_h,
        default_value = default_hurst,
    )
    confidence = fill(NaN, n)
    direction_long = fill(false, n)
    skip_code = fill(Int64(0), n)

    @inbounds for i in 1:n
        proba = src_probas[i]
        if !isfinite(proba)
            skip_code[i] = 1
            continue
        end

        conf = max(proba, 1.0 - proba)
        confidence[i] = conf
        direction_long[i] = proba > 0.5

        if src_next_hours[i] >= eod_gate_hour
            skip_code[i] = 2
        elseif conf < conf_threshold
            skip_code[i] = 3
        elseif !isfinite(src_atrs[i]) || src_atrs[i] <= 0.0
            skip_code[i] = 4
        elseif isfinite(src_zscores[i]) && abs(src_zscores[i]) > shock_z_abs
            skip_code[i] = 5
        elseif !isfinite(hurst[i]) || hurst[i] < min_hurst
            skip_code[i] = 6
        end
    end

    return (
        hurst = hurst,
        confidence = confidence,
        direction_long = direction_long,
        skip_code = skip_code,
    )
end

"""
    add_target_loop(opens, highs, lows, closes, atrs;
                    atr_mult, horizon, tp1_r_mult, tp2_r_mult, trail_r_mult,
                    fee_pct, slippage_bps, tp1_frac, tp2_frac, runner_frac)

Exact Julia port of Python's _add_target_loop_jit.
Returns a NamedTuple of 13 Float64 vectors.

Index convention:
  Python: for i in range(last_start_idx+1)  →  entry_idx = i+1  (0-based)
  Julia:  for i in 1:last_start_jl          →  entry_idx_jl = i+1 (1-based)
"""
function add_target_loop(
    opens::AbstractVector{Float64}, highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},  closes::AbstractVector{Float64},
    atrs::AbstractVector{Float64};
    atr_mult      ::Float64 = BARRIER_ATR_MULT,
    horizon       ::Int     = BARRIER_HORIZON_BARS,
    tp1_r_mult    ::Float64 = TP1_R_MULT,
    tp2_r_mult    ::Float64 = TP2_R_MULT,
    trail_r_mult  ::Float64 = TRAIL_R_MULT,
    fee_pct       ::Float64 = EXEC_FEE_PCT,
    slippage_bps  ::Float64 = EXEC_SLIPPAGE_BPS,
    tp1_frac      ::Float64 = TP1_FRACTION,
    tp2_frac      ::Float64 = TP2_FRACTION,
    runner_frac   ::Float64 = RUNNER_FRACTION,
)
    to_f64_vec(v) = v isa Vector{Float64} ? v : collect(Float64, v)
    opens = to_f64_vec(opens)
    highs = to_f64_vec(highs)
    lows = to_f64_vec(lows)
    closes = to_f64_vec(closes)
    atrs = to_f64_vec(atrs)

    n = length(opens)
    # Python: last_start_idx = max(-1, n - horizon - 1)
    # so loop is range(last_start_idx+1) = 0..last_start_idx
    # Julia equivalent: 1..last_start_jl where last_start_jl = max(0, n - horizon - 1) + 1 - 1
    #                                                         = max(0, n - horizon - 1)
    last_start_jl = max(0, n - horizon - 1)  # zero means nothing to iterate

    target           = fill(NaN, n)
    next_ret_pct     = fill(NaN, n)
    bars_to_target   = fill(NaN, n)
    entry_prices     = fill(NaN, n)
    target_distances = fill(NaN, n)
    long_path_r      = fill(NaN, n)
    short_path_r     = fill(NaN, n)
    target_edge_r    = fill(NaN, n)
    best_path_r      = fill(NaN, n)
    long_mfe_atr     = fill(NaN, n)
    long_mae_atr     = fill(NaN, n)
    short_mfe_atr    = fill(NaN, n)
    short_mae_atr    = fill(NaN, n)

    @inbounds for i in 1:last_start_jl
        # Python: entry_idx = i + 1 (0-based)  →  Julia: entry_idx_jl = i + 1 (1-based)
        entry_idx_jl = i + 1
        dist = atrs[i] * atr_mult
        (!isfinite(dist) || dist <= 0.0) && continue

        tp1_dist  = dist * tp1_r_mult
        tp2_dist  = dist * tp2_r_mult
        trail_dist_val = dist * trail_r_mult

        long_trade = _simulate_trade(
            opens, highs, lows, closes,
            entry_idx_jl, true, dist, tp1_dist, tp2_dist, trail_dist_val,
            horizon, fee_pct, slippage_bps, tp1_frac, tp2_frac, runner_frac,
        )
        short_trade = _simulate_trade(
            opens, highs, lows, closes,
            entry_idx_jl, false, dist, tp1_dist, tp2_dist, trail_dist_val,
            horizon, fee_pct, slippage_bps, tp1_frac, tp2_frac, runner_frac,
        )

        long_r  = long_trade[1];  short_r       = short_trade[1]
        long_eidx = long_trade[2]; short_eidx   = short_trade[2]
        long_entry_price = long_trade[4]

        (!isfinite(long_r) || !isfinite(short_r)) && continue

        long_path_r[i]      = long_r
        short_path_r[i]     = short_r
        best_path_r[i]      = max(long_r, short_r)
        target_edge_r[i]    = abs(long_r - short_r)
        entry_prices[i]     = long_entry_price
        target_distances[i] = dist

        horizon_end = min(n, entry_idx_jl + horizon)   # exclusive upper (matches Python min(n, entry_idx+horizon))
        if horizon_end > entry_idx_jl && atrs[i] > 0.0
            entry_price = long_entry_price
            curr_atr    = atrs[i]

            ## LONG MFE/MAE
            sl_dist_l  = 2.0 * curr_atr
            sl_price_l = entry_price - sl_dist_l
            peak_high_l = entry_price
            peak_low_l  = entry_price

            for j in entry_idx_jl:horizon_end-1
                val_high = highs[j]; val_low = lows[j]
                if val_low <= sl_price_l
                    peak_high_l = max(peak_high_l, val_high)
                    peak_low_l  = sl_price_l
                    break
                else
                    peak_high_l = max(peak_high_l, val_high)
                    peak_low_l  = min(peak_low_l, val_low)
                end
            end
            long_mfe_atr[i] = max(0.0, (peak_high_l - entry_price) / (curr_atr + 1e-9))
            long_mae_atr[i] = max(0.0, (entry_price - peak_low_l)  / (curr_atr + 1e-9))

            ## SHORT MFE/MAE
            sl_dist_s  = 2.0 * curr_atr
            sl_price_s = entry_price + sl_dist_s
            peak_low_s  = entry_price
            peak_high_s = entry_price

            for j in entry_idx_jl:horizon_end-1
                val_high = highs[j]; val_low = lows[j]
                if val_high >= sl_price_s
                    peak_low_s  = min(peak_low_s, val_low)
                    peak_high_s = sl_price_s
                    break
                else
                    peak_low_s  = min(peak_low_s, val_low)
                    peak_high_s = max(peak_high_s, val_high)
                end
            end
            short_mfe_atr[i] = max(0.0, (entry_price - peak_low_s) / (curr_atr + 1e-9))
            short_mae_atr[i] = max(0.0, (peak_high_s - entry_price) / (curr_atr + 1e-9))
        end

        # Kinetic score and target assignment (mirrors Python exactly)
        if horizon_end > entry_idx_jl && atrs[i] > 0.0
            mfe_l = long_mfe_atr[i]; mae_l = long_mae_atr[i]
            raw_long = mfe_l / (mfe_l + mae_l + 1e-9)
            vel_l    = 1.0 - ((long_eidx - i) / horizon)
            long_kinscore = raw_long * max(0.01, vel_l)

            mfe_s = short_mfe_atr[i]; mae_s = short_mae_atr[i]
            raw_short = mfe_s / (mfe_s + mae_s + 1e-9)
            vel_s     = 1.0 - ((short_eidx - i) / horizon)
            short_kinscore = raw_short * max(0.01, vel_s)

            if long_kinscore > short_kinscore && long_kinscore > 0.15
                target[i]         = 0.5 + (long_kinscore / 2.0)
                next_ret_pct[i]   = (long_r * dist / (entry_prices[i] + 1e-9)) * 100.0
                bars_to_target[i] = Float64(long_eidx - i)
            elseif short_kinscore > long_kinscore && short_kinscore > 0.15
                target[i]         = 0.5 - (short_kinscore / 2.0)
                next_ret_pct[i]   = -(short_r * dist / (entry_prices[i] + 1e-9)) * 100.0
                bars_to_target[i] = Float64(short_eidx - i)
            else
                target[i]         = 0.5
                next_ret_pct[i]   = 0.0
                bars_to_target[i] = Float64(horizon)
            end
        else
            target[i]         = 0.5
            next_ret_pct[i]   = 0.0
            bars_to_target[i] = Float64(horizon)
        end
    end

    return (
        target           = target,
        next_ret_pct     = next_ret_pct,
        bars_to_target   = bars_to_target,
        entry_prices     = entry_prices,
        target_distances = target_distances,
        long_path_r      = long_path_r,
        short_path_r     = short_path_r,
        target_edge_r    = target_edge_r,
        best_path_r      = best_path_r,
        long_mfe_atr     = long_mfe_atr,
        long_mae_atr     = long_mae_atr,
        short_mfe_atr    = short_mfe_atr,
        short_mae_atr    = short_mae_atr,
    )
end

function _confluence_daily!(out::Dict{String,Float64}, row_feats::Dict{String,Float64})
    g(k) = get(row_feats, k, 0.0)
    phase_1d = g("1d_w13_skel_smc_phase")
    phase_1w = g("1w_w8_skel_smc_phase")
    phase_1m = g("1m_w5_skel_smc_phase")
    mtf = (phase_1d > 0.0 && phase_1w > 0.0 && phase_1m > 0.0) ?  1.0 :
          (phase_1d < 0.0 && phase_1w < 0.0 && phase_1m < 0.0) ? -1.0 : 0.0
    d1c = g("1d_w13_bar13_close")
    w1h = get(row_feats, "1w_w8_skel_last_swing_high", 1.0)
    w1l = get(row_feats, "1w_w8_skel_last_swing_low",  0.0)
    out["mtf_conf_smc_sync"] = mtf
    out["mtf_conf_pd_zone"]  = (d1c - w1l) / (w1h - w1l + 1e-9)
end

function compute_holographic_features_daily(
    base_times_ns::AbstractVector{Int64},
    o_1d::AbstractVector{Float64}, h_1d::AbstractVector{Float64},
    l_1d::AbstractVector{Float64}, c_1d::AbstractVector{Float64}, v_1d::AbstractVector{Float64},
    t_1w::AbstractVector{Int64},
    o_1w::AbstractVector{Float64}, h_1w::AbstractVector{Float64},
    l_1w::AbstractVector{Float64}, c_1w::AbstractVector{Float64}, v_1w::AbstractVector{Float64},
    t_1m::AbstractVector{Int64},
    o_1m::AbstractVector{Float64}, h_1m::AbstractVector{Float64},
    l_1m::AbstractVector{Float64}, c_1m::AbstractVector{Float64}, v_1m::AbstractVector{Float64},
    t_3m::AbstractVector{Int64},
    o_3m::AbstractVector{Float64}, h_3m::AbstractVector{Float64},
    l_3m::AbstractVector{Float64}, c_3m::AbstractVector{Float64}, v_3m::AbstractVector{Float64};
    thermo_1d = nothing,
)::Dict{String, Vector{Float64}}
    return compute_holographic_features_daily(
        collect(Int64, base_times_ns),
        collect(Float64, o_1d), collect(Float64, h_1d),
        collect(Float64, l_1d), collect(Float64, c_1d), collect(Float64, v_1d),
        collect(Int64, t_1w),
        collect(Float64, o_1w), collect(Float64, h_1w),
        collect(Float64, l_1w), collect(Float64, c_1w), collect(Float64, v_1w),
        collect(Int64, t_1m),
        collect(Float64, o_1m), collect(Float64, h_1m),
        collect(Float64, l_1m), collect(Float64, c_1m), collect(Float64, v_1m),
        collect(Int64, t_3m),
        collect(Float64, o_3m), collect(Float64, h_3m),
        collect(Float64, l_3m), collect(Float64, c_3m), collect(Float64, v_3m);
        thermo_1d = thermo_1d,
    )
end

function compute_holographic_features_daily(
    base_times_ns::Vector{Int64},
    o_1d::Vector{Float64}, h_1d::Vector{Float64},
    l_1d::Vector{Float64}, c_1d::Vector{Float64}, v_1d::Vector{Float64},
    t_1w::Vector{Int64},
    o_1w::Vector{Float64}, h_1w::Vector{Float64},
    l_1w::Vector{Float64}, c_1w::Vector{Float64}, v_1w::Vector{Float64},
    t_1m::Vector{Int64},
    o_1m::Vector{Float64}, h_1m::Vector{Float64},
    l_1m::Vector{Float64}, c_1m::Vector{Float64}, v_1m::Vector{Float64},
    t_3m::Vector{Int64},
    o_3m::Vector{Float64}, h_3m::Vector{Float64},
    l_3m::Vector{Float64}, c_3m::Vector{Float64}, v_3m::Vector{Float64};
    thermo_1d = nothing,
)::Dict{String, Vector{Float64}}

    n_base = length(base_times_ns)
    all_rows = [Dict{String,Float64}() for _ in 1:n_base]

    # 1D — primary
    _process_tf!(all_rows, base_times_ns,
                 o_1d, h_1d, l_1d, c_1d, v_1d,
                 Int64[], "1d", HOLO_WINDOWS_1D, true, thermo_1d)

    # 1W
    if length(o_1w) >= 3
        _process_tf!(all_rows, base_times_ns,
                     o_1w, h_1w, l_1w, c_1w, v_1w,
                     t_1w, "1w", HOLO_WINDOWS_1W, false, nothing)
    end

    # 1M
    if length(o_1m) >= 3
        _process_tf!(all_rows, base_times_ns,
                     o_1m, h_1m, l_1m, c_1m, v_1m,
                     t_1m, "1m", HOLO_WINDOWS_1M, false, nothing)
    end

    # 3M
    if length(o_3m) >= 3
        _process_tf!(all_rows, base_times_ns,
                     o_3m, h_3m, l_3m, c_3m, v_3m,
                     t_3m, "3m", HOLO_WINDOWS_3M, false, nothing)
    end

    for i in 1:n_base
        _confluence_daily!(all_rows[i], all_rows[i])
    end

    all_keys = Set{String}()
    for row in all_rows
        union!(all_keys, keys(row))
    end
    result = Dict{String, Vector{Float64}}()
    for k in all_keys
        col = Vector{Float64}(undef, n_base)
        for i in 1:n_base
            col[i] = get(all_rows[i], k, NaN)
        end
        result[k] = col
    end
    return result
end

end  # module ToonMath
