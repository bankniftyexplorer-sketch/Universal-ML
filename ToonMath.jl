"""
ToonMath.jl — TOON v5.0 Julia Microservice
===========================================
Bare-metal Julia port of:
  - holographic_engine.py  (_bbox, _dna, _grammar, _spectral, _skeleton, _confluence, _process_tf)
  - legacy universal_ml_engine.py kernels  (simulate_trade_path_from_arrays_jit, removed bulk-label target loop)

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
# ratio_idx mapping is a public production contract:
# 1 => 0.000, 2 => 0.236, 3 => 0.382, 4 => 0.500, 5 => 0.618, 6 => 0.786, 7 => 1.000
const KALMAN_FIB_RATIOS     = (0.000, 0.236, 0.382, 0.500, 0.618, 0.786, 1.000)
const KALMAN_SWING_EPS      = 1e-9


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
# KALMAN STRUCTURAL FEATURE FAMILY
# ─────────────────────────────────────────────────────────────

@inline function _safe_ohlc4(o::Float64, h::Float64, l::Float64, c::Float64)::Float64
    prod = o * h * l * c
    if isfinite(prod) && prod > 0.0
        return prod ^ 0.25
    end
    return isfinite(c) ? c : 0.0
end

@inline function _safe_oc(o::Float64, c::Float64)::Float64
    prod = o * c
    if isfinite(prod) && prod > 0.0
        return sqrt(prod)
    end
    return isfinite(o) && isfinite(c) ? (o + c) * 0.5 : 0.0
end

function kalman_filter(
    ohlc4::AbstractVector{Float64},
    closes::AbstractVector{Float64};
    R::Float64 = 1.0,
    Q::Float64 = 0.001,
)
    return kalman_filter(
        collect(Float64, ohlc4),
        collect(Float64, closes);
        R = R,
        Q = Q,
    )
end

function kalman_filter(
    ohlc4::Vector{Float64},
    closes::Vector{Float64};
    R::Float64 = 1.0,
    Q::Float64 = 0.001,
)::NamedTuple{(:kalman, :regime), Tuple{Vector{Float64}, Vector{Float64}}}
    n = length(ohlc4)
    if length(closes) != n
        throw(ArgumentError("kalman_filter inputs must have identical lengths"))
    end

    kalman = zeros(Float64, n)
    regime = zeros(Float64, n)
    n == 0 && return (kalman = kalman, regime = regime)

    x = isfinite(ohlc4[1]) ? ohlc4[1] : (isfinite(closes[1]) ? closes[1] : 0.0)
    P = 1.0

    @inbounds for i in 1:n
        meas = isfinite(ohlc4[i]) ? ohlc4[i] : x
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x = x + K * (meas - x)
        P = (1.0 - K) * P_pred
        kalman[i] = x
        close_i = isfinite(closes[i]) ? closes[i] : x
        regime[i] = close_i - x
    end

    return (kalman = kalman, regime = regime)
end

function kalman_swing_detect(
    opens::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    closes::AbstractVector{Float64},
    oc::AbstractVector{Float64},
    kalman::AbstractVector{Float64},
)
    return kalman_swing_detect(
        collect(Float64, opens),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, closes),
        collect(Float64, oc),
        collect(Float64, kalman),
    )
end

function kalman_swing_detect(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    oc::Vector{Float64},
    kalman::Vector{Float64},
)::NamedTuple{
    (:swing_type, :swing_price, :swing_bar, :confirm_bar),
    Tuple{Vector{Int64}, Vector{Float64}, Vector{Int64}, Vector{Int64}},
}
    n = length(opens)
    if length(highs) != n || length(lows) != n || length(closes) != n || length(oc) != n || length(kalman) != n
        throw(ArgumentError("kalman_swing_detect inputs must have identical lengths"))
    end

    max_events = max(2 * max(n - 1, 0), 0)
    swing_type = Vector{Int64}(undef, max_events)
    swing_price = Vector{Float64}(undef, max_events)
    swing_bar = Vector{Int64}(undef, max_events)
    confirm_bar = Vector{Int64}(undef, max_events)
    count = 0

    @inbounds for i in 2:n
        k = kalman[i]
        body_high_cross = closes[i] < k && opens[i] > k
        wick_high_test = oc[i] > k && lows[i] <= k
        body_low_cross = closes[i] > k && opens[i] < k
        wick_low_test = oc[i] < k && highs[i] >= k

        if body_high_cross || wick_high_test
            prev_high = highs[i - 1]
            curr_high = highs[i]
            count += 1
            swing_type[count] = 1
            if curr_high >= prev_high
                swing_price[count] = curr_high
                swing_bar[count] = Int64(i)
            else
                swing_price[count] = prev_high
                swing_bar[count] = Int64(i - 1)
            end
            confirm_bar[count] = Int64(i)
        end

        if body_low_cross || wick_low_test
            prev_low = lows[i - 1]
            curr_low = lows[i]
            count += 1
            swing_type[count] = -1
            if curr_low <= prev_low
                swing_price[count] = curr_low
                swing_bar[count] = Int64(i)
            else
                swing_price[count] = prev_low
                swing_bar[count] = Int64(i - 1)
            end
            confirm_bar[count] = Int64(i)
        end
    end

    resize!(swing_type, count)
    resize!(swing_price, count)
    resize!(swing_bar, count)
    resize!(confirm_bar, count)

    return (
        swing_type = swing_type,
        swing_price = swing_price,
        swing_bar = swing_bar,
        confirm_bar = confirm_bar,
    )
end

function candle_geometry_delta(
    opens::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    closes::AbstractVector{Float64},
    volumes::AbstractVector{Float64},
)
    return candle_geometry_delta(
        collect(Float64, opens),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, closes),
        collect(Float64, volumes),
    )
end

function candle_geometry_delta(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    volumes::Vector{Float64},
)::Vector{Float64}
    n = length(opens)
    if length(highs) != n || length(lows) != n || length(closes) != n || length(volumes) != n
        throw(ArgumentError("candle_geometry_delta inputs must have identical lengths"))
    end

    delta = zeros(Float64, n)
    n < 2 && return delta

    @inbounds for i in 2:n
        prev_c = isfinite(closes[i - 1]) ? closes[i - 1] : closes[i]
        tw = max(0.0, max(highs[i], prev_c) - max(opens[i], closes[i]))
        bw = max(0.0, min(opens[i], closes[i]) - min(lows[i], prev_c))
        body = abs(opens[i] - closes[i])

        tw_sq = tw * tw
        bw_sq = bw * bw
        body_sq = body * body
        total_sq = tw_sq + bw_sq + body_sq
        special_move = bw + (closes[i] - opens[i]) - tw

        if total_sq < 1e-18
            delta[i] = 0.0
            continue
        end

        cond = special_move >= 0.0
        buy_share = bw_sq + (cond ? body_sq : 0.0)
        sell_share = tw_sq + (cond ? 0.0 : body_sq)
        ret_u = buy_share / total_sq
        ret_d = sell_share / total_sq
        vol_i = isfinite(volumes[i]) ? volumes[i] : 0.0
        delta[i] = cond ? vol_i * ret_u : -vol_i * ret_d
    end

    return delta
end

function swing_delta_accumulation(
    bar_delta::AbstractVector{Float64},
    swing_confirm_bars::AbstractVector{Int64},
)
    return swing_delta_accumulation(
        collect(Float64, bar_delta),
        collect(Int64, swing_confirm_bars),
    )
end

function swing_delta_accumulation(
    bar_delta::Vector{Float64},
    swing_confirm_bars::Vector{Int64},
)::NamedTuple{(:swing_accum, :net_swing_delta), Tuple{Vector{Float64}, Vector{Float64}}}
    n = length(bar_delta)
    swing_accum = zeros(Float64, n)
    net_swing_delta = zeros(Float64, n)

    segment_sum = 0.0
    swing_ptr = 1
    n_swings = length(swing_confirm_bars)
    completed_count = 0
    completed_prev = 0.0
    completed_curr = 0.0
    current_net = 0.0

    @inbounds for i in 1:n
        while swing_ptr <= n_swings && swing_confirm_bars[swing_ptr] < i
            swing_ptr += 1
        end

        while swing_ptr <= n_swings && swing_confirm_bars[swing_ptr] == i
            completed = segment_sum
            completed_count += 1
            if completed_count == 1
                completed_curr = completed
                current_net = 0.0
            else
                completed_prev = completed_curr
                completed_curr = completed
                current_net = completed_curr - completed_prev
            end
            segment_sum = 0.0
            swing_ptr += 1
        end

        delta_i = isfinite(bar_delta[i]) ? bar_delta[i] : 0.0
        segment_sum += delta_i
        swing_accum[i] = segment_sum
        net_swing_delta[i] = current_net
    end

    return (swing_accum = swing_accum, net_swing_delta = net_swing_delta)
end

function completed_swing_state(
    bar_delta::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    atrs::AbstractVector{Float64},
    swing_type::AbstractVector{Int64},
    swing_price::AbstractVector{Float64},
    swing_confirm_bars::AbstractVector{Int64},
)
    return completed_swing_state(
        collect(Float64, bar_delta),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, atrs),
        collect(Int64, swing_type),
        collect(Float64, swing_price),
        collect(Int64, swing_confirm_bars),
    )
end

function completed_swing_state(
    bar_delta::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    atrs::Vector{Float64},
    swing_type::Vector{Int64},
    swing_price::Vector{Float64},
    swing_confirm_bars::Vector{Int64},
)
    n = length(bar_delta)
    if length(highs) != n || length(lows) != n || length(atrs) != n
        throw(ArgumentError("completed_swing_state price and ATR inputs must have identical lengths"))
    end

    n_swings = length(swing_type)
    if length(swing_price) != n_swings || length(swing_confirm_bars) != n_swings
        throw(ArgumentError("completed_swing_state swing inputs must have identical lengths"))
    end

    bull_swing_delta = zeros(Float64, n)
    prev_bull_swing_delta = zeros(Float64, n)
    bull_swing_delta_div = zeros(Float64, n)
    bear_swing_delta = zeros(Float64, n)
    prev_bear_swing_delta = zeros(Float64, n)
    bear_swing_delta_div = zeros(Float64, n)

    bull_swing_delta_per_bar = zeros(Float64, n)
    prev_bull_swing_delta_per_bar = zeros(Float64, n)
    bull_swing_delta_per_bar_div = zeros(Float64, n)
    bear_swing_delta_per_bar = zeros(Float64, n)
    prev_bear_swing_delta_per_bar = zeros(Float64, n)
    bear_swing_delta_per_bar_div = zeros(Float64, n)

    bull_swing_delta_per_range = zeros(Float64, n)
    prev_bull_swing_delta_per_range = zeros(Float64, n)
    bull_swing_delta_per_range_div = zeros(Float64, n)
    bear_swing_delta_per_range = zeros(Float64, n)
    prev_bear_swing_delta_per_range = zeros(Float64, n)
    bear_swing_delta_per_range_div = zeros(Float64, n)

    bull_swing_delta_per_atr_range = zeros(Float64, n)
    prev_bull_swing_delta_per_atr_range = zeros(Float64, n)
    bull_swing_delta_per_atr_range_div = zeros(Float64, n)
    bear_swing_delta_per_atr_range = zeros(Float64, n)
    prev_bear_swing_delta_per_atr_range = zeros(Float64, n)
    bear_swing_delta_per_atr_range_div = zeros(Float64, n)

    bull_swing_delta_eff = zeros(Float64, n)
    prev_bull_swing_delta_eff = zeros(Float64, n)
    bull_swing_delta_eff_div = zeros(Float64, n)
    bear_swing_delta_eff = zeros(Float64, n)
    prev_bear_swing_delta_eff = zeros(Float64, n)
    bear_swing_delta_eff_div = zeros(Float64, n)

    segment_direction = zeros(Int64, n_swings)
    segment_start_confirm_bar = zeros(Int64, n_swings)
    segment_end_confirm_bar = zeros(Int64, n_swings)
    segment_bar_count = zeros(Int64, n_swings)
    segment_raw_delta = zeros(Float64, n_swings)
    segment_high = zeros(Float64, n_swings)
    segment_low = zeros(Float64, n_swings)
    segment_price_range = zeros(Float64, n_swings)
    segment_atr_ref = zeros(Float64, n_swings)

    bull_current = zeros(Float64, 5)
    bull_previous = zeros(Float64, 5)
    bear_current = zeros(Float64, 5)
    bear_previous = zeros(Float64, 5)
    has_bull_current = false
    has_bull_previous = false
    has_bear_current = false
    has_bear_previous = false

    segment_sum = 0.0
    swing_ptr = 1
    last_confirm_bar = 1

    @inbounds for i in 1:n
        while swing_ptr <= n_swings && swing_confirm_bars[swing_ptr] < i
            swing_ptr += 1
        end

        while swing_ptr <= n_swings && swing_confirm_bars[swing_ptr] == i
            confirm_idx = clamp(swing_confirm_bars[swing_ptr], 1, n)
            start_confirm_idx = clamp(last_confirm_bar, 1, n)
            segment_last_bar = confirm_idx - 1
            bar_count = max(confirm_idx - start_confirm_idx, 0)

            raw_delta = segment_sum
            fallback_atr = atrs[confirm_idx]
            atr_ref = isfinite(fallback_atr) && fallback_atr > KALMAN_SWING_EPS ? fallback_atr : 1.0
            anchor_price = isfinite(swing_price[swing_ptr]) ? swing_price[swing_ptr] : 0.0
            hi = anchor_price
            lo = anchor_price

            if bar_count > 0
                hi = maximum(@view highs[start_confirm_idx:segment_last_bar])
                lo = minimum(@view lows[start_confirm_idx:segment_last_bar])
                atr_ref = median(@view atrs[start_confirm_idx:segment_last_bar])
                if !isfinite(atr_ref) || atr_ref <= KALMAN_SWING_EPS
                    atr_ref = isfinite(fallback_atr) && fallback_atr > KALMAN_SWING_EPS ? fallback_atr : 1.0
                end
            end

            if !isfinite(hi)
                hi = anchor_price
            end
            if !isfinite(lo)
                lo = anchor_price
            end

            price_range = max(hi - lo, 0.0)
            bar_count_f = Float64(bar_count)
            delta_per_bar = raw_delta / max(bar_count_f, 1.0)
            delta_per_range = raw_delta / max(price_range, KALMAN_SWING_EPS)
            atr_scaled_range = price_range / max(atr_ref, KALMAN_SWING_EPS)
            delta_per_atr_range = raw_delta / max(atr_scaled_range, KALMAN_SWING_EPS)
            delta_efficiency = raw_delta / max(bar_count_f * atr_ref, KALMAN_SWING_EPS)

            segment_direction[swing_ptr] = swing_type[swing_ptr] >= 0 ? 1 : -1
            segment_start_confirm_bar[swing_ptr] = Int64(start_confirm_idx)
            segment_end_confirm_bar[swing_ptr] = Int64(confirm_idx)
            segment_bar_count[swing_ptr] = Int64(bar_count)
            segment_raw_delta[swing_ptr] = raw_delta
            segment_high[swing_ptr] = hi
            segment_low[swing_ptr] = lo
            segment_price_range[swing_ptr] = price_range
            segment_atr_ref[swing_ptr] = atr_ref

            if segment_direction[swing_ptr] == 1
                if has_bull_current
                    bull_previous[1] = bull_current[1]
                    bull_previous[2] = bull_current[2]
                    bull_previous[3] = bull_current[3]
                    bull_previous[4] = bull_current[4]
                    bull_previous[5] = bull_current[5]
                    has_bull_previous = true
                end
                bull_current[1] = raw_delta
                bull_current[2] = delta_per_bar
                bull_current[3] = delta_per_range
                bull_current[4] = delta_per_atr_range
                bull_current[5] = delta_efficiency
                has_bull_current = true
            else
                if has_bear_current
                    bear_previous[1] = bear_current[1]
                    bear_previous[2] = bear_current[2]
                    bear_previous[3] = bear_current[3]
                    bear_previous[4] = bear_current[4]
                    bear_previous[5] = bear_current[5]
                    has_bear_previous = true
                end
                bear_current[1] = raw_delta
                bear_current[2] = delta_per_bar
                bear_current[3] = delta_per_range
                bear_current[4] = delta_per_atr_range
                bear_current[5] = delta_efficiency
                has_bear_current = true
            end

            segment_sum = 0.0
            last_confirm_bar = confirm_idx
            swing_ptr += 1
        end

        bull_swing_delta[i] = bull_current[1]
        prev_bull_swing_delta[i] = has_bull_previous ? bull_previous[1] : 0.0
        bull_swing_delta_div[i] = has_bull_previous ? (bull_current[1] - bull_previous[1]) : 0.0
        bear_swing_delta[i] = bear_current[1]
        prev_bear_swing_delta[i] = has_bear_previous ? bear_previous[1] : 0.0
        bear_swing_delta_div[i] = has_bear_previous ? (bear_current[1] - bear_previous[1]) : 0.0

        bull_swing_delta_per_bar[i] = bull_current[2]
        prev_bull_swing_delta_per_bar[i] = has_bull_previous ? bull_previous[2] : 0.0
        bull_swing_delta_per_bar_div[i] = has_bull_previous ? (bull_current[2] - bull_previous[2]) : 0.0
        bear_swing_delta_per_bar[i] = bear_current[2]
        prev_bear_swing_delta_per_bar[i] = has_bear_previous ? bear_previous[2] : 0.0
        bear_swing_delta_per_bar_div[i] = has_bear_previous ? (bear_current[2] - bear_previous[2]) : 0.0

        bull_swing_delta_per_range[i] = bull_current[3]
        prev_bull_swing_delta_per_range[i] = has_bull_previous ? bull_previous[3] : 0.0
        bull_swing_delta_per_range_div[i] = has_bull_previous ? (bull_current[3] - bull_previous[3]) : 0.0
        bear_swing_delta_per_range[i] = bear_current[3]
        prev_bear_swing_delta_per_range[i] = has_bear_previous ? bear_previous[3] : 0.0
        bear_swing_delta_per_range_div[i] = has_bear_previous ? (bear_current[3] - bear_previous[3]) : 0.0

        bull_swing_delta_per_atr_range[i] = bull_current[4]
        prev_bull_swing_delta_per_atr_range[i] = has_bull_previous ? bull_previous[4] : 0.0
        bull_swing_delta_per_atr_range_div[i] = has_bull_previous ? (bull_current[4] - bull_previous[4]) : 0.0
        bear_swing_delta_per_atr_range[i] = bear_current[4]
        prev_bear_swing_delta_per_atr_range[i] = has_bear_previous ? bear_previous[4] : 0.0
        bear_swing_delta_per_atr_range_div[i] = has_bear_previous ? (bear_current[4] - bear_previous[4]) : 0.0

        bull_swing_delta_eff[i] = bull_current[5]
        prev_bull_swing_delta_eff[i] = has_bull_previous ? bull_previous[5] : 0.0
        bull_swing_delta_eff_div[i] = has_bull_previous ? (bull_current[5] - bull_previous[5]) : 0.0
        bear_swing_delta_eff[i] = bear_current[5]
        prev_bear_swing_delta_eff[i] = has_bear_previous ? bear_previous[5] : 0.0
        bear_swing_delta_eff_div[i] = has_bear_previous ? (bear_current[5] - bear_previous[5]) : 0.0

        delta_i = isfinite(bar_delta[i]) ? bar_delta[i] : 0.0
        segment_sum += delta_i
    end

    return (
        segment_direction = segment_direction,
        segment_start_confirm_bar = segment_start_confirm_bar,
        segment_end_confirm_bar = segment_end_confirm_bar,
        segment_bar_count = segment_bar_count,
        segment_raw_delta = segment_raw_delta,
        segment_high = segment_high,
        segment_low = segment_low,
        segment_price_range = segment_price_range,
        segment_atr_ref = segment_atr_ref,
        bull_swing_delta = bull_swing_delta,
        prev_bull_swing_delta = prev_bull_swing_delta,
        bull_swing_delta_div = bull_swing_delta_div,
        bear_swing_delta = bear_swing_delta,
        prev_bear_swing_delta = prev_bear_swing_delta,
        bear_swing_delta_div = bear_swing_delta_div,
        bull_swing_delta_per_bar = bull_swing_delta_per_bar,
        prev_bull_swing_delta_per_bar = prev_bull_swing_delta_per_bar,
        bull_swing_delta_per_bar_div = bull_swing_delta_per_bar_div,
        bear_swing_delta_per_bar = bear_swing_delta_per_bar,
        prev_bear_swing_delta_per_bar = prev_bear_swing_delta_per_bar,
        bear_swing_delta_per_bar_div = bear_swing_delta_per_bar_div,
        bull_swing_delta_per_range = bull_swing_delta_per_range,
        prev_bull_swing_delta_per_range = prev_bull_swing_delta_per_range,
        bull_swing_delta_per_range_div = bull_swing_delta_per_range_div,
        bear_swing_delta_per_range = bear_swing_delta_per_range,
        prev_bear_swing_delta_per_range = prev_bear_swing_delta_per_range,
        bear_swing_delta_per_range_div = bear_swing_delta_per_range_div,
        bull_swing_delta_per_atr_range = bull_swing_delta_per_atr_range,
        prev_bull_swing_delta_per_atr_range = prev_bull_swing_delta_per_atr_range,
        bull_swing_delta_per_atr_range_div = bull_swing_delta_per_atr_range_div,
        bear_swing_delta_per_atr_range = bear_swing_delta_per_atr_range,
        prev_bear_swing_delta_per_atr_range = prev_bear_swing_delta_per_atr_range,
        bear_swing_delta_per_atr_range_div = bear_swing_delta_per_atr_range_div,
        bull_swing_delta_eff = bull_swing_delta_eff,
        prev_bull_swing_delta_eff = prev_bull_swing_delta_eff,
        bull_swing_delta_eff_div = bull_swing_delta_eff_div,
        bear_swing_delta_eff = bear_swing_delta_eff,
        prev_bear_swing_delta_eff = prev_bear_swing_delta_eff,
        bear_swing_delta_eff_div = bear_swing_delta_eff_div,
    )
end

function _init_kf_pair_features(pair_label::String, n::Int)::Dict{String, Vector{Float64}}
    result = Dict{String, Vector{Float64}}()
    for ratio_idx in 1:length(KALMAN_FIB_RATIOS)
        result[string("kf_", pair_label, "_", ratio_idx, "_o")] = zeros(Float64, n)
        result[string("kf_", pair_label, "_", ratio_idx, "_h")] = zeros(Float64, n)
        result[string("kf_", pair_label, "_", ratio_idx, "_l")] = zeros(Float64, n)
        result[string("kf_", pair_label, "_", ratio_idx, "_c")] = zeros(Float64, n)
        result[string("kf_", pair_label, "_", ratio_idx, "_delta")] = zeros(Float64, n)
    end
    return result
end

function htf_fib_observation(
    ltf_opens::AbstractVector{Float64},
    ltf_highs::AbstractVector{Float64},
    ltf_lows::AbstractVector{Float64},
    ltf_closes::AbstractVector{Float64},
    ltf_atrs::AbstractVector{Float64},
    ltf_bar_delta::AbstractVector{Float64},
    ltf_times_ns::AbstractVector{Int64},
    htf_swing_highs_price::AbstractVector{Float64},
    htf_swing_highs_confirm::AbstractVector{Int64},
    htf_swing_lows_price::AbstractVector{Float64},
    htf_swing_lows_confirm::AbstractVector{Int64},
    htf_times_ns::AbstractVector{Int64},
    pair_label::AbstractString,
)
    return htf_fib_observation(
        collect(Float64, ltf_opens),
        collect(Float64, ltf_highs),
        collect(Float64, ltf_lows),
        collect(Float64, ltf_closes),
        collect(Float64, ltf_atrs),
        collect(Float64, ltf_bar_delta),
        collect(Int64, ltf_times_ns),
        collect(Float64, htf_swing_highs_price),
        collect(Int64, htf_swing_highs_confirm),
        collect(Float64, htf_swing_lows_price),
        collect(Int64, htf_swing_lows_confirm),
        collect(Int64, htf_times_ns),
        String(pair_label),
    )
end

function htf_fib_observation(
    ltf_opens::Vector{Float64},
    ltf_highs::Vector{Float64},
    ltf_lows::Vector{Float64},
    ltf_closes::Vector{Float64},
    ltf_atrs::Vector{Float64},
    ltf_bar_delta::Vector{Float64},
    ltf_times_ns::Vector{Int64},
    htf_swing_highs_price::Vector{Float64},
    htf_swing_highs_confirm::Vector{Int64},
    htf_swing_lows_price::Vector{Float64},
    htf_swing_lows_confirm::Vector{Int64},
    htf_times_ns::Vector{Int64},
    pair_label::String,
)::Dict{String, Vector{Float64}}
    n = length(ltf_opens)
    if length(ltf_highs) != n || length(ltf_lows) != n || length(ltf_closes) != n || length(ltf_atrs) != n || length(ltf_bar_delta) != n || length(ltf_times_ns) != n
        throw(ArgumentError("htf_fib_observation LTF inputs must have identical lengths"))
    end
    if length(htf_swing_highs_price) != length(htf_swing_highs_confirm) || length(htf_swing_lows_price) != length(htf_swing_lows_confirm)
        throw(ArgumentError("htf_fib_observation swing price/confirm inputs must have identical lengths"))
    end

    result = _init_kf_pair_features(pair_label, n)
    open_cols = [result[string("kf_", pair_label, "_", ratio_idx, "_o")] for ratio_idx in 1:length(KALMAN_FIB_RATIOS)]
    high_cols = [result[string("kf_", pair_label, "_", ratio_idx, "_h")] for ratio_idx in 1:length(KALMAN_FIB_RATIOS)]
    low_cols = [result[string("kf_", pair_label, "_", ratio_idx, "_l")] for ratio_idx in 1:length(KALMAN_FIB_RATIOS)]
    close_cols = [result[string("kf_", pair_label, "_", ratio_idx, "_c")] for ratio_idx in 1:length(KALMAN_FIB_RATIOS)]
    delta_cols = [result[string("kf_", pair_label, "_", ratio_idx, "_delta")] for ratio_idx in 1:length(KALMAN_FIB_RATIOS)]

    n_high = length(htf_swing_highs_confirm)
    n_low = length(htf_swing_lows_confirm)
    if n_high == 0 || n_low == 0 || isempty(htf_times_ns)
        return result
    end

    high_confirm_times = Vector{Int64}(undef, n_high)
    low_confirm_times = Vector{Int64}(undef, n_low)

    @inbounds for j in 1:n_high
        idx = clamp(htf_swing_highs_confirm[j], 1, length(htf_times_ns))
        high_confirm_times[j] = htf_times_ns[idx]
    end
    @inbounds for j in 1:n_low
        idx = clamp(htf_swing_lows_confirm[j], 1, length(htf_times_ns))
        low_confirm_times[j] = htf_times_ns[idx]
    end

    @inbounds for i in 1:n
        high_pos = searchsortedlast(high_confirm_times, ltf_times_ns[i])
        low_pos = searchsortedlast(low_confirm_times, ltf_times_ns[i])
        if high_pos == 0 || low_pos == 0
            continue
        end

        swing_high = htf_swing_highs_price[high_pos]
        swing_low = htf_swing_lows_price[low_pos]
        if !isfinite(swing_high) || !isfinite(swing_low) || swing_high <= swing_low
            continue
        end

        atr_i = ltf_atrs[i]
        atr_scale = isfinite(atr_i) && abs(atr_i) > 1e-9 ? atr_i : 1.0
        bar_delta_i = isfinite(ltf_bar_delta[i]) ? ltf_bar_delta[i] : 0.0
        fib_range = swing_high - swing_low

        for (ratio_idx, ratio) in enumerate(KALMAN_FIB_RATIOS)
            level = swing_low + ratio * fib_range
            open_cols[ratio_idx][i] = (ltf_opens[i] - level) / atr_scale
            high_cols[ratio_idx][i] = (ltf_highs[i] - level) / atr_scale
            low_cols[ratio_idx][i] = (ltf_lows[i] - level) / atr_scale
            close_cols[ratio_idx][i] = (ltf_closes[i] - level) / atr_scale
            delta_cols[ratio_idx][i] = bar_delta_i
        end
    end

    return result
end

function compute_kalman_tf_state(
    opens::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    closes::AbstractVector{Float64},
    volumes::AbstractVector{Float64},
    atrs::AbstractVector{Float64},
)
    return compute_kalman_tf_state(
        collect(Float64, opens),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, closes),
        collect(Float64, volumes),
        collect(Float64, atrs),
    )
end

function compute_kalman_tf_state(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    volumes::Vector{Float64},
    atrs::Vector{Float64},
)
    n = length(opens)
    if length(highs) != n || length(lows) != n || length(closes) != n || length(volumes) != n || length(atrs) != n
        throw(ArgumentError("compute_kalman_tf_state inputs must have identical lengths"))
    end

    ohlc4 = Vector{Float64}(undef, n)
    oc = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        ohlc4[i] = _safe_ohlc4(opens[i], highs[i], lows[i], closes[i])
        oc[i] = _safe_oc(opens[i], closes[i])
    end

    kalman_state = kalman_filter(ohlc4, closes)
    bar_delta = candle_geometry_delta(opens, highs, lows, closes, volumes)
    swings = kalman_swing_detect(opens, highs, lows, closes, oc, kalman_state.kalman)
    swing_delta = swing_delta_accumulation(bar_delta, swings.confirm_bar)
    completed_swings = completed_swing_state(
        bar_delta,
        highs,
        lows,
        atrs,
        swings.swing_type,
        swings.swing_price,
        swings.confirm_bar,
    )

    return (
        kalman = kalman_state.kalman,
        regime = kalman_state.regime,
        bar_delta = bar_delta,
        swing_accum = swing_delta.swing_accum,
        net_swing_delta = swing_delta.net_swing_delta,
        swing_type = swings.swing_type,
        swing_price = swings.swing_price,
        swing_bar = swings.swing_bar,
        confirm_bar = swings.confirm_bar,
        segment_direction = completed_swings.segment_direction,
        segment_start_confirm_bar = completed_swings.segment_start_confirm_bar,
        segment_end_confirm_bar = completed_swings.segment_end_confirm_bar,
        segment_bar_count = completed_swings.segment_bar_count,
        segment_raw_delta = completed_swings.segment_raw_delta,
        segment_high = completed_swings.segment_high,
        segment_low = completed_swings.segment_low,
        segment_price_range = completed_swings.segment_price_range,
        segment_atr_ref = completed_swings.segment_atr_ref,
        bull_swing_delta = completed_swings.bull_swing_delta,
        prev_bull_swing_delta = completed_swings.prev_bull_swing_delta,
        bull_swing_delta_div = completed_swings.bull_swing_delta_div,
        bear_swing_delta = completed_swings.bear_swing_delta,
        prev_bear_swing_delta = completed_swings.prev_bear_swing_delta,
        bear_swing_delta_div = completed_swings.bear_swing_delta_div,
        bull_swing_delta_per_bar = completed_swings.bull_swing_delta_per_bar,
        prev_bull_swing_delta_per_bar = completed_swings.prev_bull_swing_delta_per_bar,
        bull_swing_delta_per_bar_div = completed_swings.bull_swing_delta_per_bar_div,
        bear_swing_delta_per_bar = completed_swings.bear_swing_delta_per_bar,
        prev_bear_swing_delta_per_bar = completed_swings.prev_bear_swing_delta_per_bar,
        bear_swing_delta_per_bar_div = completed_swings.bear_swing_delta_per_bar_div,
        bull_swing_delta_per_range = completed_swings.bull_swing_delta_per_range,
        prev_bull_swing_delta_per_range = completed_swings.prev_bull_swing_delta_per_range,
        bull_swing_delta_per_range_div = completed_swings.bull_swing_delta_per_range_div,
        bear_swing_delta_per_range = completed_swings.bear_swing_delta_per_range,
        prev_bear_swing_delta_per_range = completed_swings.prev_bear_swing_delta_per_range,
        bear_swing_delta_per_range_div = completed_swings.bear_swing_delta_per_range_div,
        bull_swing_delta_per_atr_range = completed_swings.bull_swing_delta_per_atr_range,
        prev_bull_swing_delta_per_atr_range = completed_swings.prev_bull_swing_delta_per_atr_range,
        bull_swing_delta_per_atr_range_div = completed_swings.bull_swing_delta_per_atr_range_div,
        bear_swing_delta_per_atr_range = completed_swings.bear_swing_delta_per_atr_range,
        prev_bear_swing_delta_per_atr_range = completed_swings.prev_bear_swing_delta_per_atr_range,
        bear_swing_delta_per_atr_range_div = completed_swings.bear_swing_delta_per_atr_range_div,
        bull_swing_delta_eff = completed_swings.bull_swing_delta_eff,
        prev_bull_swing_delta_eff = completed_swings.prev_bull_swing_delta_eff,
        bull_swing_delta_eff_div = completed_swings.bull_swing_delta_eff_div,
        bear_swing_delta_eff = completed_swings.bear_swing_delta_eff,
        prev_bear_swing_delta_eff = completed_swings.prev_bear_swing_delta_eff,
        bear_swing_delta_eff_div = completed_swings.bear_swing_delta_eff_div,
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
        else
            out[string(p, "_basis_pct")] = 0.0
            out[string(p, "_basis_z")] = 0.0
            out[string(p, "_basis_v5")] = 0.0
            out[string(p, "_basis_v10")] = 0.0
            out[string(p, "_session_pos")] = 0.0
            out[string(p, "_eod_momentum")] = 0.0
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

Exact Julia port of the removed Python _add_target_loop_jit bulk-label kernel.
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

            ## LONG MFE/MAE (DECOUPLED)
            sl_dist_l  = 2.0 * curr_atr
            sl_price_l = entry_price - sl_dist_l
            
            # Kinematic trackers (Capped)
            peak_high_l = entry_price
            peak_low_l  = entry_price
            kin_alive_l = true
            
            # True Limit trackers (Uncapped)
            uncap_peak_high_l = entry_price
            uncap_peak_low_l  = entry_price

            for j in entry_idx_jl:horizon_end-1
                val_high = highs[j]; val_low = lows[j]
                
                # Uncapped Tracking (For ML Trade Plan)
                uncap_peak_high_l = max(uncap_peak_high_l, val_high)
                uncap_peak_low_l  = min(uncap_peak_low_l, val_low)
                
                # Kinematic Tracking (For Classifier)
                if kin_alive_l
                    if val_low <= sl_price_l
                        peak_high_l = max(peak_high_l, val_high)
                        peak_low_l  = sl_price_l
                        kin_alive_l = false
                    else
                        peak_high_l = max(peak_high_l, val_high)
                        peak_low_l  = min(peak_low_l, val_low)
                    end
                end
            end
            
            kin_mfe_l = max(0.0, (peak_high_l - entry_price) / (curr_atr + 1e-9))
            kin_mae_l = max(0.0, (entry_price - peak_low_l)  / (curr_atr + 1e-9))
            
            # Exported True Limits
            long_mfe_atr[i] = max(0.0, (uncap_peak_high_l - entry_price) / (curr_atr + 1e-9))
            long_mae_atr[i] = max(0.0, (entry_price - uncap_peak_low_l)  / (curr_atr + 1e-9))

            ## SHORT MFE/MAE (DECOUPLED)
            sl_dist_s  = 2.0 * curr_atr
            sl_price_s = entry_price + sl_dist_s
            
            # Kinematic trackers (Capped)
            peak_low_s  = entry_price
            peak_high_s = entry_price
            kin_alive_s = true
            
            # True Limit trackers (Uncapped)
            uncap_peak_low_s  = entry_price
            uncap_peak_high_s = entry_price

            for j in entry_idx_jl:horizon_end-1
                val_high = highs[j]; val_low = lows[j]
                
                # Uncapped Tracking (For ML Trade Plan)
                uncap_peak_low_s  = min(uncap_peak_low_s, val_low)
                uncap_peak_high_s = max(uncap_peak_high_s, val_high)
                
                # Kinematic Tracking (For Classifier)
                if kin_alive_s
                    if val_high >= sl_price_s
                        peak_low_s  = min(peak_low_s, val_low)
                        peak_high_s = sl_price_s
                        kin_alive_s = false
                    else
                        peak_low_s  = min(peak_low_s, val_low)
                        peak_high_s = max(peak_high_s, val_high)
                    end
                end
            end
            
            kin_mfe_s = max(0.0, (entry_price - peak_low_s) / (curr_atr + 1e-9))
            kin_mae_s = max(0.0, (peak_high_s - entry_price) / (curr_atr + 1e-9))
            
            # Exported True Limits
            short_mfe_atr[i] = max(0.0, (entry_price - uncap_peak_low_s) / (curr_atr + 1e-9))
            short_mae_atr[i] = max(0.0, (uncap_peak_high_s - entry_price) / (curr_atr + 1e-9))
        
            # Kinetic score and target assignment
            raw_long = kin_mfe_l / (kin_mfe_l + kin_mae_l + 1e-9)
            vel_l    = 1.0 - ((long_eidx - i) / horizon)
            long_kinscore = raw_long * max(0.01, vel_l)

            raw_short = kin_mfe_s / (kin_mfe_s + kin_mae_s + 1e-9)
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


# ─────────────────────────────────────────────────────────────
# SMC TOKEN 1 / TOKEN 2 FEATURES
# ─────────────────────────────────────────────────────────────

mutable struct FVGEntry
    bar_idx::Int
    top::Float64
    bottom::Float64
    direction::Float64
    max_fill_pct::Float64
end

mutable struct OBEntry
    bar_idx::Int
    ob_high::Float64
    ob_low::Float64
    midpoint::Float64
    direction::Float64
    quality::Float64
    mitigated::Bool
end

struct BreakerEntry
    bar_idx::Int
    zone_high::Float64
    zone_low::Float64
    midpoint::Float64
    direction::Float64
    strength::Float64
end

@inline function _smc_input_length(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    volumes::Vector{Float64},
    atrs::Vector{Float64},
)::Int
    n = length(opens)
    if length(highs) != n || length(lows) != n || length(closes) != n ||
       length(volumes) != n || length(atrs) != n
        throw(ArgumentError("compute_smc_features inputs must have identical lengths"))
    end
    return n
end

function _compute_smc_core(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    volumes::Vector{Float64},
    atrs::Vector{Float64};
    swing_lookback::Int = 10,
    structure_window::Int = 40,
    fvg_max_age::Int = 50,
    ob_max_age::Int = 50,
    ob_decay_rate::Float64 = 0.02,
    disp_body_threshold::Float64 = 1.5,
    amd_accum_window::Int = 20,
    indu_threshold_atr::Float64 = 0.5,
    warmup::Int = 50,
    include_liquidity::Bool = true,
    include_ob::Bool = true,
    include_fvg::Bool = true,
    include_indu::Bool = true,
)
    _ = swing_lookback
    _ = amd_accum_window

    n = _smc_input_length(opens, highs, lows, closes, volumes, atrs)

    nearest_ssl_dist = fill(99.0, n)
    nearest_bsl_dist = fill(99.0, n)
    ob_nearest_dist = fill(99.0, n)
    fvg_nearest_dist = fill(99.0, n)

    sweep_bull_mag = fill(0.0, n)
    sweep_bear_mag = fill(0.0, n)
    pool_density = fill(0.0, n)
    ob_quality_score = fill(0.0, n)
    ob_direction = fill(0.0, n)
    ob_age_bars = fill(0.0, n)
    fvg_direction = fill(0.0, n)
    fvg_fill_pct = fill(0.5, n)
    fvg_count_active = fill(0.0, n)
    fvg_imbalance_ratio = fill(0.0, n)
    disp_magnitude = fill(0.0, n)
    disp_body_ratio = fill(0.0, n)
    disp_volume_ratio = fill(0.0, n)
    disp_confirmed = fill(0.0, n)
    structure_trend_score = fill(0.0, n)
    choch_bull_signal = fill(0.0, n)
    choch_bear_signal = fill(0.0, n)
    mss_confirmed = fill(0.0, n)
    amd_phase = fill(0.0, n)
    amd_duration = fill(0.0, n)
    amd_range_pct = fill(0.0, n)
    amd_volume_profile = fill(0.5, n)
    indu_confirmed = fill(0.0, n)
    indu_direction = fill(0.0, n)
    indu_magnitude = fill(0.0, n)
    indu_age_bars = fill(0.0, n)
    pd_zone = fill(0.5, n)
    breaker_direction = fill(0.0, n)
    breaker_dist = fill(99.0, n)
    breaker_strength = fill(0.0, n)

    swing_high_indices = Int[]
    swing_high_prices = Float64[]
    swing_low_indices = Int[]
    swing_low_prices = Float64[]

    structure_events = Float64[]
    last_confirmed_swing_high = NaN
    last_confirmed_swing_low = NaN

    active_fvgs = FVGEntry[]
    active_obs = OBEntry[]
    active_breakers = BreakerEntry[]

    amd_current_state = 0
    amd_state_start_bar = 1
    amd_state_high = -Inf
    amd_state_low = Inf
    amd_state_vol_sum = 0.0
    amd_state_bar_count = 0

    last_indu_bar = 0
    last_indu_dir = 0.0
    last_indu_mag = 0.0

    @inbounds for i in 2:n
        curr_atr = atrs[i]
        if !isfinite(curr_atr) || curr_atr <= 0.0
            continue
        end
        prev_atr = (isfinite(atrs[i - 1]) && atrs[i - 1] > 0.0) ? atrs[i - 1] : curr_atr
        ready = i > warmup

        if i >= 6
            k = i - 3
            swing_atr = (isfinite(atrs[k]) && atrs[k] > 0.0) ? atrs[k] : prev_atr
            prominence_threshold = 0.3 * swing_atr

            is_sh = highs[k] > highs[k - 1] && highs[k] > highs[k - 2] &&
                    highs[k] > highs[k + 1] && highs[k] > highs[k + 2]
            if is_sh
                left_prom = min(highs[k] - highs[k - 1], highs[k] - highs[k - 2])
                right_prom = min(highs[k] - highs[k + 1], highs[k] - highs[k + 2])
                if min(left_prom, right_prom) >= prominence_threshold
                    push!(swing_high_indices, k)
                    push!(swing_high_prices, highs[k])

                    if isfinite(last_confirmed_swing_high)
                        if highs[k] > last_confirmed_swing_high
                            push!(structure_events, 1.0)
                        else
                            push!(structure_events, -1.0)
                        end
                    end
                    last_confirmed_swing_high = highs[k]
                end
            end

            is_sl = lows[k] < lows[k - 1] && lows[k] < lows[k - 2] &&
                    lows[k] < lows[k + 1] && lows[k] < lows[k + 2]
            if is_sl
                left_prom = min(lows[k - 1] - lows[k], lows[k - 2] - lows[k])
                right_prom = min(lows[k + 1] - lows[k], lows[k + 2] - lows[k])
                if min(left_prom, right_prom) >= prominence_threshold
                    push!(swing_low_indices, k)
                    push!(swing_low_prices, lows[k])

                    if isfinite(last_confirmed_swing_low)
                        if lows[k] < last_confirmed_swing_low
                            push!(structure_events, -1.0)
                        else
                            push!(structure_events, 1.0)
                        end
                    end
                    last_confirmed_swing_low = lows[k]
                end
            end
        end

        if ready
            body_i = abs(closes[i - 1] - opens[i - 1])
            range_i = highs[i - 1] - lows[i - 1]

            body_sum = 0.0
            vol_sum = 0.0
            count = 0
            for j in max(1, i - 21):(i - 2)
                body_sum += abs(closes[j] - opens[j])
                vol_sum += volumes[j]
                count += 1
            end
            avg_body = count > 0 ? body_sum / count : body_i
            avg_vol = count > 0 ? vol_sum / count : volumes[i - 1]

            disp_magnitude[i] = range_i / curr_atr
            disp_body_ratio[i] = body_i / (avg_body + 1e-9)
            disp_volume_ratio[i] = volumes[i - 1] / (avg_vol + 1e-9)
            disp_confirmed[i] = (disp_body_ratio[i] > disp_body_threshold &&
                                 disp_magnitude[i] > 1.5) ? 1.0 : 0.0
        end

        if include_liquidity && ready
            if !isempty(swing_low_prices)
                nearest_sl_price = swing_low_prices[end]
                nearest_sl_dist_val = abs(closes[i - 1] - nearest_sl_price) / curr_atr
                nearest_ssl_dist[i] = min(nearest_sl_dist_val, 99.0)

                sweep_depth = nearest_sl_price - lows[i - 1]
                if sweep_depth > 0.0
                    sweep_bull_mag[i] = sweep_depth / curr_atr
                end
            end

            if !isempty(swing_high_prices)
                nearest_sh_price = swing_high_prices[end]
                nearest_bsl_dist_val = abs(nearest_sh_price - closes[i - 1]) / curr_atr
                nearest_bsl_dist[i] = min(nearest_bsl_dist_val, 99.0)

                sweep_depth = highs[i - 1] - nearest_sh_price
                if sweep_depth > 0.0
                    sweep_bear_mag[i] = sweep_depth / curr_atr
                end
            end

            density = 0.0
            radius = 5.0 * curr_atr
            c_prev = closes[i - 1]
            for idx in length(swing_high_indices):-1:1
                if i - swing_high_indices[idx] > structure_window
                    break
                end
                if abs(swing_high_prices[idx] - c_prev) <= radius
                    density += 1.0
                end
            end
            for idx in length(swing_low_indices):-1:1
                if i - swing_low_indices[idx] > structure_window
                    break
                end
                if abs(swing_low_prices[idx] - c_prev) <= radius
                    density += 1.0
                end
            end
            pool_density[i] = density
        end

        if include_ob && ready && disp_confirmed[i] == 1.0 && i >= 3
            k = i - 2
            if k >= 1
                dir_k = closes[k] >= opens[k] ? 1.0 : -1.0
                dir_k1 = closes[i - 1] >= opens[i - 1] ? 1.0 : -1.0
                if dir_k != dir_k1
                    body_ratio_k = abs(closes[k] - opens[k]) / (curr_atr + 1e-9)
                    quality = disp_body_ratio[i] * body_ratio_k
                    ob_h = max(opens[k], closes[k])
                    ob_l = min(opens[k], closes[k])
                    push!(
                        active_obs,
                        OBEntry(k, ob_h, ob_l, (ob_h + ob_l) / 2.0, -dir_k, quality, false),
                    )
                end
            end
        end

        if include_ob && ready
            j_ob = 1
            while j_ob <= length(active_obs)
                ob = active_obs[j_ob]
                age = i - 1 - ob.bar_idx

                if age > ob_max_age
                    deleteat!(active_obs, j_ob)
                    continue
                end

                if !ob.mitigated
                    if ob.direction > 0.0
                        if lows[i - 1] < ob.ob_low
                            ob.mitigated = true
                            push!(
                                active_breakers,
                                BreakerEntry(
                                    ob.bar_idx,
                                    ob.ob_high,
                                    ob.ob_low,
                                    ob.midpoint,
                                    -ob.direction,
                                    ob.quality * 0.8,
                                ),
                            )
                        end
                    else
                        if highs[i - 1] > ob.ob_high
                            ob.mitigated = true
                            push!(
                                active_breakers,
                                BreakerEntry(
                                    ob.bar_idx,
                                    ob.ob_high,
                                    ob.ob_low,
                                    ob.midpoint,
                                    -ob.direction,
                                    ob.quality * 0.8,
                                ),
                            )
                        end
                    end
                end

                if ob.mitigated && age > 5
                    deleteat!(active_obs, j_ob)
                    continue
                end

                j_ob += 1
            end

            best_dist = 99.0
            best_ob_local = nothing
            for ob in active_obs
                if ob.mitigated
                    continue
                end
                age = i - 1 - ob.bar_idx
                dist = abs(closes[i - 1] - ob.midpoint) / curr_atr
                if dist < best_dist
                    best_dist = dist
                    best_ob_local = ob
                end
            end
            if best_ob_local !== nothing
                age = i - 1 - best_ob_local.bar_idx
                ob_quality_score[i] = best_ob_local.quality * exp(-ob_decay_rate * age)
                ob_direction[i] = best_ob_local.direction
                ob_nearest_dist[i] = min(best_dist, 99.0)
                ob_age_bars[i] = Float64(age)
            end

            filter!(b -> (i - 1 - b.bar_idx) <= ob_max_age, active_breakers)

            if !isempty(active_breakers)
                best_breaker_dist = 99.0
                best_brk = active_breakers[1]
                for brk in active_breakers
                    dist = abs(closes[i - 1] - brk.midpoint) / curr_atr
                    if dist < best_breaker_dist
                        best_breaker_dist = dist
                        best_brk = brk
                    end
                end
                age = i - 1 - best_brk.bar_idx
                breaker_direction[i] = best_brk.direction
                breaker_dist[i] = min(best_breaker_dist, 99.0)
                breaker_strength[i] = best_brk.strength * exp(-ob_decay_rate * age)
            end
        end

        if include_fvg
            if i >= 4
                if lows[i - 1] > highs[i - 3]
                    push!(active_fvgs, FVGEntry(i - 2, lows[i - 1], highs[i - 3], 1.0, 0.0))
                end
                if highs[i - 1] < lows[i - 3]
                    push!(active_fvgs, FVGEntry(i - 2, lows[i - 3], highs[i - 1], -1.0, 0.0))
                end
            end

            j = 1
            while j <= length(active_fvgs)
                fvg = active_fvgs[j]
                age = i - fvg.bar_idx
                if age > fvg_max_age
                    deleteat!(active_fvgs, j)
                    continue
                end

                gap_size = fvg.top - fvg.bottom
                if gap_size > 0.0
                    fill_this_bar = 0.0
                    if fvg.direction > 0.0
                        if lows[i - 1] <= fvg.top
                            fill_this_bar = (fvg.top - max(lows[i - 1], fvg.bottom)) / gap_size
                        end
                    else
                        if highs[i - 1] >= fvg.bottom
                            fill_this_bar = (min(highs[i - 1], fvg.top) - fvg.bottom) / gap_size
                        end
                    end
                    fvg.max_fill_pct = max(fvg.max_fill_pct, fill_this_bar)
                    if fvg.max_fill_pct >= 0.95
                        deleteat!(active_fvgs, j)
                        continue
                    end
                end
                j += 1
            end

            if ready && !isempty(active_fvgs)
                nearest_dist = 99.0
                nearest_idx = 0
                bull_count = 0
                bear_count = 0

                for (idx, fvg) in enumerate(active_fvgs)
                    gap_mid = (fvg.top + fvg.bottom) / 2.0
                    dist = abs(closes[i - 1] - gap_mid) / curr_atr
                    if dist < nearest_dist
                        nearest_dist = dist
                        nearest_idx = idx
                    end
                    if fvg.direction > 0.0
                        bull_count += 1
                    else
                        bear_count += 1
                    end
                end

                fvg_nearest_dist[i] = min(nearest_dist, 99.0)
                fvg_count_active[i] = Float64(length(active_fvgs))
                total_fvg = bull_count + bear_count
                fvg_imbalance_ratio[i] = Float64(bull_count - bear_count) / Float64(total_fvg + 1)

                if nearest_idx > 0
                    nfvg = active_fvgs[nearest_idx]
                    fvg_direction[i] = nfvg.direction
                    fvg_fill_pct[i] = clamp(nfvg.max_fill_pct, 0.0, 1.0)
                end
            end
        end

        if ready
            n_events = length(structure_events)
            window_start = max(1, n_events - structure_window + 1)

            bullish_count = 0.0
            bearish_count = 0.0
            for ev_idx in window_start:n_events
                ev = structure_events[ev_idx]
                if ev > 0.0
                    bullish_count += 1.0
                elseif ev < 0.0
                    bearish_count += 1.0
                end
            end
            total = bullish_count + bearish_count
            if total > 0.0
                structure_trend_score[i] = clamp((bullish_count - bearish_count) / total, -1.0, 1.0)
            end

            if !isempty(structure_events) && structure_events[end] > 0.0
                if n_events >= 3
                    prev_bear = 0
                    for check_idx in (n_events - 1):-1:max(1, n_events - 5)
                        if structure_events[check_idx] < 0.0
                            prev_bear += 1
                        else
                            break
                        end
                    end
                    if prev_bear >= 2
                        choch_bull_signal[i] = 1.0
                    end
                end
            end

            if !isempty(structure_events) && structure_events[end] < 0.0
                if n_events >= 3
                    prev_bull = 0
                    for check_idx in (n_events - 1):-1:max(1, n_events - 5)
                        if structure_events[check_idx] > 0.0
                            prev_bull += 1
                        else
                            break
                        end
                    end
                    if prev_bull >= 2
                        choch_bear_signal[i] = 1.0
                    end
                end
            end

            if isfinite(last_confirmed_swing_high) && isfinite(last_confirmed_swing_low)
                dealing_range = last_confirmed_swing_high - last_confirmed_swing_low
                if dealing_range > 0.0
                    pd_zone[i] = clamp(
                        (closes[i - 1] - last_confirmed_swing_low) / dealing_range,
                        0.0,
                        1.0,
                    )
                end
            end

            mss_score = 0.0
            has_choch = choch_bull_signal[i] == 1.0 || choch_bear_signal[i] == 1.0
            if has_choch
                mss_score += 0.4
            end
            if disp_magnitude[i] > 1.0
                mss_score += clamp(0.3 * (disp_magnitude[i] - 1.0) / 2.0, 0.0, 0.3)
            end
            if has_choch
                if choch_bull_signal[i] == 1.0 && isfinite(last_confirmed_swing_high)
                    if closes[i - 1] > last_confirmed_swing_high
                        mss_score += 0.3
                    end
                elseif choch_bear_signal[i] == 1.0 && isfinite(last_confirmed_swing_low)
                    if closes[i - 1] < last_confirmed_swing_low
                        mss_score += 0.3
                    end
                end
            end
            mss_confirmed[i] = clamp(mss_score, 0.0, 1.0)
        end

        if ready
            amd_state_bar_count += 1
            amd_state_high = max(amd_state_high, highs[i - 1])
            amd_state_low = min(amd_state_low, lows[i - 1])
            amd_state_vol_sum += volumes[i - 1]

            if amd_current_state == 0
                amd_current_state = 1
                amd_state_start_bar = i
                amd_state_high = highs[i - 1]
                amd_state_low = lows[i - 1]
                amd_state_vol_sum = volumes[i - 1]
                amd_state_bar_count = 1
            end

            transitioned = false

            if amd_current_state == 1
                if sweep_bull_mag[i] > 0.5 || sweep_bear_mag[i] > 0.5
                    amd_current_state = 2
                    transitioned = true
                elseif amd_state_bar_count > 30
                    amd_state_start_bar = i
                    amd_state_high = highs[i - 1]
                    amd_state_low = lows[i - 1]
                    amd_state_vol_sum = volumes[i - 1]
                    amd_state_bar_count = 1
                end
            elseif amd_current_state == 2
                if disp_confirmed[i] == 1.0
                    amd_current_state = 3
                    transitioned = true
                end
            elseif amd_current_state == 3
                state_range = amd_state_high - amd_state_low
                avg_state_vol = amd_state_vol_sum / max(amd_state_bar_count, 1)

                vol_sum_50 = 0.0
                vol_count_50 = 0
                for j in max(1, i - 50):(i - 1)
                    vol_sum_50 += volumes[j]
                    vol_count_50 += 1
                end
                avg_vol_50 = vol_count_50 > 0 ? vol_sum_50 / vol_count_50 : avg_state_vol

                if amd_state_bar_count >= 10 &&
                   state_range < 0.5 * curr_atr &&
                   avg_state_vol < 0.7 * avg_vol_50
                    amd_current_state = 1
                    transitioned = true
                end
            end

            if transitioned
                amd_state_start_bar = i
                amd_state_high = highs[i - 1]
                amd_state_low = lows[i - 1]
                amd_state_vol_sum = volumes[i - 1]
                amd_state_bar_count = 1
            end

            amd_phase[i] = amd_current_state == 1 ? 0.33 :
                           amd_current_state == 2 ? 0.67 :
                           amd_current_state == 3 ? 1.0 : 0.0

            amd_duration[i] = clamp(Float64(amd_state_bar_count) / 50.0, 0.0, 1.0)
            amd_range_pct[i] = (amd_state_high - amd_state_low) / curr_atr

            avg_state_vol = amd_state_vol_sum / max(amd_state_bar_count, 1)
            vol_sum_50 = 0.0
            vol_count_50 = 0
            for j in max(1, i - 50):(i - 1)
                vol_sum_50 += volumes[j]
                vol_count_50 += 1
            end
            avg_vol_50 = vol_count_50 > 0 ? vol_sum_50 / vol_count_50 : avg_state_vol
            amd_volume_profile[i] = avg_state_vol / (avg_vol_50 + 1e-9)
        end

        if include_indu && i > warmup + 1
            if !isempty(swing_low_prices)
                nearest_sl = swing_low_prices[end]
                break_depth = nearest_sl - lows[i - 2]
                if break_depth > 0.0 && break_depth < indu_threshold_atr * curr_atr
                    if closes[i - 1] > nearest_sl
                        indu_confirmed[i] = 1.0
                        indu_direction[i] = 1.0
                        indu_magnitude[i] = break_depth / curr_atr
                        last_indu_bar = i
                        last_indu_dir = 1.0
                        last_indu_mag = indu_magnitude[i]
                    end
                end
            end

            if !isempty(swing_high_prices)
                nearest_sh = swing_high_prices[end]
                break_depth = highs[i - 2] - nearest_sh
                if break_depth > 0.0 && break_depth < indu_threshold_atr * curr_atr
                    if closes[i - 1] < nearest_sh
                        indu_confirmed[i] = 1.0
                        indu_direction[i] = -1.0
                        indu_magnitude[i] = break_depth / curr_atr
                        last_indu_bar = i
                        last_indu_dir = -1.0
                        last_indu_mag = indu_magnitude[i]
                    end
                end
            end

            if last_indu_bar > 0 && i > last_indu_bar
                indu_age_bars[i] = clamp(Float64(i - last_indu_bar) / 20.0, 0.0, 1.0)
                if indu_confirmed[i] == 0.0
                    indu_direction[i] = last_indu_dir
                    indu_magnitude[i] = last_indu_mag * exp(-0.05 * (i - last_indu_bar))
                end
            end
        end
    end

    return (
        sweep_bull_mag = sweep_bull_mag,
        sweep_bear_mag = sweep_bear_mag,
        nearest_ssl_dist = nearest_ssl_dist,
        nearest_bsl_dist = nearest_bsl_dist,
        pool_density = pool_density,
        ob_quality_score = ob_quality_score,
        ob_direction = ob_direction,
        ob_nearest_dist = ob_nearest_dist,
        ob_age_bars = ob_age_bars,
        fvg_nearest_dist = fvg_nearest_dist,
        fvg_direction = fvg_direction,
        fvg_fill_pct = fvg_fill_pct,
        fvg_count_active = fvg_count_active,
        fvg_imbalance_ratio = fvg_imbalance_ratio,
        disp_magnitude = disp_magnitude,
        disp_body_ratio = disp_body_ratio,
        disp_volume_ratio = disp_volume_ratio,
        disp_confirmed = disp_confirmed,
        structure_trend_score = structure_trend_score,
        choch_bull_signal = choch_bull_signal,
        choch_bear_signal = choch_bear_signal,
        mss_confirmed = mss_confirmed,
        amd_phase = amd_phase,
        amd_duration = amd_duration,
        amd_range_pct = amd_range_pct,
        amd_volume_profile = amd_volume_profile,
        indu_confirmed = indu_confirmed,
        indu_direction = indu_direction,
        indu_magnitude = indu_magnitude,
        indu_age_bars = indu_age_bars,
        pd_zone = pd_zone,
        breaker_direction = breaker_direction,
        breaker_dist = breaker_dist,
        breaker_strength = breaker_strength,
    )
end

function compute_smc_features(
    opens::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    closes::AbstractVector{Float64},
    volumes::AbstractVector{Float64},
    atrs::AbstractVector{Float64};
    swing_lookback::Int = 10,
    structure_window::Int = 40,
    fvg_max_age::Int = 50,
    ob_max_age::Int = 50,
    ob_decay_rate::Float64 = 0.02,
    disp_body_threshold::Float64 = 1.5,
    amd_accum_window::Int = 20,
    indu_threshold_atr::Float64 = 0.5,
    warmup::Int = 50,
)::Dict{String, Vector{Float64}}
    return compute_smc_features(
        collect(Float64, opens),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, closes),
        collect(Float64, volumes),
        collect(Float64, atrs);
        swing_lookback = swing_lookback,
        structure_window = structure_window,
        fvg_max_age = fvg_max_age,
        ob_max_age = ob_max_age,
        ob_decay_rate = ob_decay_rate,
        disp_body_threshold = disp_body_threshold,
        amd_accum_window = amd_accum_window,
        indu_threshold_atr = indu_threshold_atr,
        warmup = warmup,
    )
end

"""
    compute_smc_features(opens, highs, lows, closes, volumes, atrs; ...) -> Dict{String, Vector{Float64}}

Compute the 34 Token 1 / Token 2 SMC feature columns defined in the repo spec.
The returned keys intentionally omit the Python-side `smc_` prefix.
"""
function compute_smc_features(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    volumes::Vector{Float64},
    atrs::Vector{Float64};
    swing_lookback::Int = 10,
    structure_window::Int = 40,
    fvg_max_age::Int = 50,
    ob_max_age::Int = 50,
    ob_decay_rate::Float64 = 0.02,
    disp_body_threshold::Float64 = 1.5,
    amd_accum_window::Int = 20,
    indu_threshold_atr::Float64 = 0.5,
    warmup::Int = 50,
)::Dict{String, Vector{Float64}}
    cols = _compute_smc_core(
        opens,
        highs,
        lows,
        closes,
        volumes,
        atrs;
        swing_lookback = swing_lookback,
        structure_window = structure_window,
        fvg_max_age = fvg_max_age,
        ob_max_age = ob_max_age,
        ob_decay_rate = ob_decay_rate,
        disp_body_threshold = disp_body_threshold,
        amd_accum_window = amd_accum_window,
        indu_threshold_atr = indu_threshold_atr,
        warmup = warmup,
        include_liquidity = true,
        include_ob = true,
        include_fvg = true,
        include_indu = true,
    )

    return Dict{String, Vector{Float64}}(
        "sweep_bull_mag" => cols.sweep_bull_mag,
        "sweep_bear_mag" => cols.sweep_bear_mag,
        "nearest_ssl_dist" => cols.nearest_ssl_dist,
        "nearest_bsl_dist" => cols.nearest_bsl_dist,
        "pool_density" => cols.pool_density,
        "ob_quality_score" => cols.ob_quality_score,
        "ob_direction" => cols.ob_direction,
        "ob_nearest_dist" => cols.ob_nearest_dist,
        "ob_age_bars" => cols.ob_age_bars,
        "fvg_nearest_dist" => cols.fvg_nearest_dist,
        "fvg_direction" => cols.fvg_direction,
        "fvg_fill_pct" => cols.fvg_fill_pct,
        "fvg_count_active" => cols.fvg_count_active,
        "fvg_imbalance_ratio" => cols.fvg_imbalance_ratio,
        "disp_magnitude" => cols.disp_magnitude,
        "disp_body_ratio" => cols.disp_body_ratio,
        "disp_volume_ratio" => cols.disp_volume_ratio,
        "disp_confirmed" => cols.disp_confirmed,
        "structure_trend_score" => cols.structure_trend_score,
        "choch_bull_signal" => cols.choch_bull_signal,
        "choch_bear_signal" => cols.choch_bear_signal,
        "mss_confirmed" => cols.mss_confirmed,
        "amd_phase" => cols.amd_phase,
        "amd_duration" => cols.amd_duration,
        "amd_range_pct" => cols.amd_range_pct,
        "amd_volume_profile" => cols.amd_volume_profile,
        "indu_confirmed" => cols.indu_confirmed,
        "indu_direction" => cols.indu_direction,
        "indu_magnitude" => cols.indu_magnitude,
        "indu_age_bars" => cols.indu_age_bars,
        "pd_zone" => cols.pd_zone,
        "breaker_direction" => cols.breaker_direction,
        "breaker_dist" => cols.breaker_dist,
        "breaker_strength" => cols.breaker_strength,
    )
end

function compute_smc_htf_features(
    opens::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    closes::AbstractVector{Float64},
    volumes::AbstractVector{Float64},
    atrs::AbstractVector{Float64};
    swing_lookback::Int = 10,
    structure_window::Int = 40,
    fvg_max_age::Int = 50,
    ob_max_age::Int = 50,
    amd_accum_window::Int = 20,
    warmup::Int = 50,
)::Dict{String, Vector{Float64}}
    return compute_smc_htf_features(
        collect(Float64, opens),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, closes),
        collect(Float64, volumes),
        collect(Float64, atrs);
        swing_lookback = swing_lookback,
        structure_window = structure_window,
        fvg_max_age = fvg_max_age,
        ob_max_age = ob_max_age,
        amd_accum_window = amd_accum_window,
        warmup = warmup,
    )
end

"""
    compute_smc_htf_features(opens, highs, lows, closes, volumes, atrs; ...) -> Dict{String, Vector{Float64}}

Reduced higher-timeframe SMC projection that only returns structure and AMD state.
"""
function compute_smc_htf_features(
    opens::Vector{Float64},
    highs::Vector{Float64},
    lows::Vector{Float64},
    closes::Vector{Float64},
    volumes::Vector{Float64},
    atrs::Vector{Float64};
    swing_lookback::Int = 10,
    structure_window::Int = 40,
    fvg_max_age::Int = 50,
    ob_max_age::Int = 50,
    amd_accum_window::Int = 20,
    warmup::Int = 50,
)::Dict{String, Vector{Float64}}
    cols = _compute_smc_core(
        opens,
        highs,
        lows,
        closes,
        volumes,
        atrs;
        swing_lookback = swing_lookback,
        structure_window = structure_window,
        fvg_max_age = fvg_max_age,
        ob_max_age = ob_max_age,
        ob_decay_rate = 0.02,
        disp_body_threshold = 1.5,
        amd_accum_window = amd_accum_window,
        indu_threshold_atr = 0.5,
        warmup = warmup,
        include_liquidity = false,
        include_ob = false,
        include_fvg = false,
        include_indu = false,
    )

    return Dict{String, Vector{Float64}}(
        "structure_trend_score" => cols.structure_trend_score,
        "amd_phase" => cols.amd_phase,
    )
end


# ─────────────────────────────────────────────────────────────
# REALIZED VOLATILITY V2 — Institutional Multi-Estimator Surface
# ─────────────────────────────────────────────────────────────

function compute_rv_features(
    opens::AbstractVector{Float64},
    highs::AbstractVector{Float64},
    lows::AbstractVector{Float64},
    closes::AbstractVector{Float64};
    rv_window::Int = 21,
    z_window::Int = 63,
    short_window::Int = 5,
    long_window::Int = 21,
    annualization::Float64 = 252.0,
)::Dict{String, Vector{Float64}}
    return compute_rv_features(
        collect(Float64, opens),
        collect(Float64, highs),
        collect(Float64, lows),
        collect(Float64, closes);
        rv_window = rv_window,
        z_window = z_window,
        short_window = short_window,
        long_window = long_window,
        annualization = annualization,
    )
end

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

    yz_log_out = fill(0.0, n)
    yz_z_out = fill(0.0, n)
    yz_pctrank_out = fill(0.5, n)
    yz_trend_out = fill(0.0, n)
    yz_shock_out = fill(0.0, n)
    vov_out = fill(0.0, n)
    pk_log_out = fill(0.0, n)
    range_eff_out = fill(1.0, n)
    jump_ratio_out = fill(0.0, n)
    rv_asym_out = fill(1.0, n)
    rv_skew_out = fill(0.0, n)

    log_co = fill(0.0, n)
    log_oc = fill(0.0, n)
    log_ret = fill(0.0, n)
    rs_comp = fill(0.0, n)
    pk_comp = fill(0.0, n)

    for i in 1:n
        o_i = opens[i]
        h_i = highs[i]
        l_i = lows[i]
        c_i = closes[i]
        if o_i <= 0.0 || h_i <= 0.0 || l_i <= 0.0 || c_i <= 0.0
            continue
        end
        log_oc[i] = log(c_i / o_i)
        if i > 1 && closes[i - 1] > 0.0
            log_co[i] = log(o_i / closes[i - 1])
            log_ret[i] = log(c_i / closes[i - 1])
        end
        log_hc = log(h_i / c_i)
        log_ho = log(h_i / o_i)
        log_lc = log(l_i / c_i)
        log_lo = log(l_i / o_i)
        rs_comp[i] = log_hc * log_ho + log_lc * log_lo
        pk_comp[i] = log(h_i / l_i)^2
    end

    yz_rv = fill(0.0, n)
    pk_rv = fill(0.0, n)
    pk_factor = 1.0 / (4.0 * log(2.0))
    k = 0.34 / (1.34 + (rv_window + 1) / (rv_window - 1))

    for i in rv_window:n
        s = (i - rv_window + 1):i
        co_start = max(i - rv_window + 2, 2)
        co_slice = @view log_co[co_start:i]
        if length(co_slice) < 2
            continue
        end

        mu_o = 0.0
        @inbounds for v in co_slice
            mu_o += v
        end
        mu_o /= length(co_slice)
        sigma_o2 = 0.0
        @inbounds for v in co_slice
            sigma_o2 += (v - mu_o)^2
        end
        sigma_o2 /= length(co_slice)

        oc_slice = @view log_oc[s]
        mu_c = 0.0
        @inbounds for v in oc_slice
            mu_c += v
        end
        mu_c /= rv_window
        sigma_c2 = 0.0
        @inbounds for v in oc_slice
            sigma_c2 += (v - mu_c)^2
        end
        sigma_c2 /= rv_window

        sigma_rs2 = 0.0
        @inbounds for v in @view rs_comp[s]
            sigma_rs2 += v
        end
        sigma_rs2 /= rv_window

        yz_var = sigma_o2 + k * sigma_rs2 + (1.0 - k) * sigma_c2
        if isfinite(yz_var) && yz_var > 0.0
            yz_rv[i] = sqrt(yz_var * annualization)
        end

        pk_sum = 0.0
        @inbounds for v in @view pk_comp[s]
            pk_sum += v
        end
        pk_var = pk_sum / rv_window * pk_factor * annualization
        if pk_var > 0.0
            pk_rv[i] = sqrt(pk_var)
        end

        ret_slice = @view log_ret[s]
        mu_ret = 0.0
        @inbounds for v in ret_slice
            mu_ret += v
        end
        mu_ret /= rv_window
        simple_var = 0.0
        @inbounds for v in ret_slice
            simple_var += (v - mu_ret)^2
        end
        simple_var /= rv_window

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

        down_var = 0.0
        down_count = 0
        up_var = 0.0
        up_count = 0
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
        if down_count > 0
            down_var /= down_count
        end
        if up_count > 0
            up_var /= up_count
        end
        if up_var > eps
            rv_asym_out[i] = clamp(down_var / up_var, 0.1, 10.0)
        end

        m2 = 0.0
        m3 = 0.0
        for j in s
            r = log_ret[j]
            r2 = r * r
            m2 += r2
            m3 += r2 * r
        end
        m2 /= rv_window
        m3 /= rv_window
        denom = m2^1.5
        if denom > eps
            rv_skew_out[i] = clamp(m3 / denom, -5.0, 5.0)
        end
    end

    first_valid = 0
    for i in rv_window:n
        if yz_rv[i] > 0.0
            first_valid = i
            break
        end
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

    yz_cumsum = fill(0.0, n + 1)
    for i in 1:n
        yz_cumsum[i + 1] = yz_cumsum[i] + yz_rv[i]
    end

    for i in 1:n
        rv_i = yz_rv[i]

        yz_log_out[i] = log1p(rv_i)
        pk_log_out[i] = log1p(pk_rv[i])

        if rv_i > eps
            range_eff_out[i] = clamp(pk_rv[i] / rv_i, 0.2, 5.0)
        end

        if i > 1 && yz_rv[i - 1] > eps
            yz_shock_out[i] = clamp((rv_i / yz_rv[i - 1]) - 1.0, -5.0, 5.0)
        end

        if i >= z_window
            z_start = i - z_window + 1
            z_mean = (yz_cumsum[i + 1] - yz_cumsum[z_start]) / z_window
            z_var = 0.0
            @inbounds for j in z_start:i
                z_var += (yz_rv[j] - z_mean)^2
            end
            z_std = sqrt(z_var / z_window)
            if z_std > eps
                yz_z_out[i] = clamp((rv_i - z_mean) / z_std, -8.0, 8.0)
            end
            rank_count = 0
            @inbounds for j in z_start:i
                if yz_rv[j] <= rv_i
                    rank_count += 1
                end
            end
            yz_pctrank_out[i] = Float64(rank_count) / Float64(z_window)
        end

        if i >= long_window
            short_start = max(1, i - short_window + 1)
            long_start = i - long_window + 1
            short_mean = (yz_cumsum[i + 1] - yz_cumsum[short_start]) / (i - short_start + 1)
            long_mean = (yz_cumsum[i + 1] - yz_cumsum[long_start]) / long_window
            if long_mean > eps
                yz_trend_out[i] = clamp((short_mean / long_mean) - 1.0, -5.0, 5.0)
            end
        end

        if i >= rv_window
            vov_start = i - rv_window + 1
            vov_mean = (yz_cumsum[i + 1] - yz_cumsum[vov_start]) / rv_window
            if vov_mean > eps
                vov_var = 0.0
                @inbounds for j in vov_start:i
                    vov_var += (yz_rv[j] - vov_mean)^2
                end
                vov_out[i] = clamp(sqrt(vov_var / rv_window) / vov_mean, 0.0, 5.0)
            end
        end
    end

    return Dict{String, Vector{Float64}}(
        "yz_log" => yz_log_out,
        "yz_z" => yz_z_out,
        "yz_pctrank" => yz_pctrank_out,
        "yz_trend" => yz_trend_out,
        "yz_shock" => yz_shock_out,
        "vov" => vov_out,
        "pk_log" => pk_log_out,
        "range_eff" => range_eff_out,
        "jump_ratio" => jump_ratio_out,
        "rv_asym" => rv_asym_out,
        "rv_skew" => rv_skew_out,
    )
end

"""
    compute_narrative_context(
        closes, atrs, regime, swing_confirm_bars,
        swing_highs_price, swing_highs_confirm,
        swing_lows_price, swing_lows_confirm
    ) -> Dict{String, Vector{Float64}}

Narrative Context Awareness — 7 columns per timeframe:
  regime_streak           — consecutive bars of same-sign regime
  regime_accel            — MA(5) - MA(20) of regime, ATR-normalized
  cum_disp_since_flip     — cumulative ATR-normalized returns since regime flip
  max_dd_since_flip       — max drawdown from phase high, ATR-normalized
  swing_count_since_flip  — Kalman swing events since last regime flip
  fib_range_age           — bars since latest confirming swing
  fib_range_size          — current swing range / ATR

All O(n) single-pass. No heap allocation in hot loops.
"""
function compute_narrative_context(
    closes::Vector{Float64},
    atrs::Vector{Float64},
    regime::Vector{Float64},
    swing_confirm_bars::Vector{Int64},
    swing_highs_price::Vector{Float64},
    swing_highs_confirm::Vector{Int64},
    swing_lows_price::Vector{Float64},
    swing_lows_confirm::Vector{Int64},
)::Dict{String, Vector{Float64}}
    n = length(closes)
    eps = 1e-9

    streak = zeros(Float64, n)
    accel = zeros(Float64, n)
    cum_disp = zeros(Float64, n)
    max_dd = zeros(Float64, n)
    sw_count = zeros(Float64, n)
    fib_age = zeros(Float64, n)
    fib_size = zeros(Float64, n)

    # ── 1. Regime streak ──
    @inbounds for i in 2:n
        if (regime[i] >= 0.0) == (regime[i-1] >= 0.0)
            streak[i] = streak[i-1] + 1.0
        else
            streak[i] = 1.0
        end
    end

    # ── 2. Regime acceleration (MA5 - MA20 of regime / ATR) ──
    # Use cumulative sum for O(1) rolling mean
    regime_cumsum = zeros(Float64, n + 1)
    @inbounds for i in 1:n
        regime_cumsum[i + 1] = regime_cumsum[i] + regime[i]
    end
    @inbounds for i in 1:n
        s5 = max(i - 4, 1)
        s20 = max(i - 19, 1)
        ma5 = (regime_cumsum[i + 1] - regime_cumsum[s5]) / (i - s5 + 1)
        ma20 = (regime_cumsum[i + 1] - regime_cumsum[s20]) / (i - s20 + 1)
        atr_i = atrs[i]
        scale = (isfinite(atr_i) && atr_i > eps) ? atr_i : 1.0
        accel[i] = (ma5 - ma20) / scale
    end

    # ── 3. Cumulative displacement since flip + 4. Max drawdown since flip + 5. Swing count ──
    swing_set = Set{Int64}(swing_confirm_bars)
    cum_sum = 0.0
    phase_high = closes[1]
    sw_cnt = 0

    @inbounds for i in 2:n
        atr_i = atrs[i]
        scale = (isfinite(atr_i) && atr_i > eps) ? atr_i : 1.0

        # Flip detection
        if (regime[i] >= 0.0) != (regime[i-1] >= 0.0)
            cum_sum = 0.0
            phase_high = closes[i]
            sw_cnt = 0
        end

        # Cumulative displacement
        bar_ret = (closes[i] - closes[i-1]) / scale
        cum_sum += bar_ret
        cum_disp[i] = cum_sum

        # Max drawdown from phase high
        if closes[i] > phase_high
            phase_high = closes[i]
        end
        max_dd[i] = (phase_high - closes[i]) / scale

        # Swing count
        if i in swing_set
            sw_cnt += 1
        end
        sw_count[i] = Float64(sw_cnt)
    end

    # ── 6 & 7. Fib range age and size ──
    n_sh = length(swing_highs_confirm)
    n_sl = length(swing_lows_confirm)
    sh_ptr = 1
    sl_ptr = 1
    curr_sh = NaN
    curr_sl = NaN
    latest_bar = 0

    @inbounds for i in 1:n
        while sh_ptr <= n_sh && swing_highs_confirm[sh_ptr] <= i
            curr_sh = swing_highs_price[sh_ptr]
            latest_bar = max(latest_bar, Int(swing_highs_confirm[sh_ptr]))
            sh_ptr += 1
        end
        while sl_ptr <= n_sl && swing_lows_confirm[sl_ptr] <= i
            curr_sl = swing_lows_price[sl_ptr]
            latest_bar = max(latest_bar, Int(swing_lows_confirm[sl_ptr]))
            sl_ptr += 1
        end

        if isfinite(curr_sh) && isfinite(curr_sl) && curr_sh > curr_sl
            fib_age[i] = Float64(i - latest_bar)
            atr_i = atrs[i]
            scale = (isfinite(atr_i) && atr_i > eps) ? atr_i : 1.0
            fib_size[i] = (curr_sh - curr_sl) / scale
        end
    end

    return Dict{String, Vector{Float64}}(
        "regime_streak" => streak,
        "regime_accel" => accel,
        "cum_disp_since_flip" => cum_disp,
        "max_dd_since_flip" => max_dd,
        "swing_count_since_flip" => sw_count,
        "fib_range_age" => fib_age,
        "fib_range_size" => fib_size,
    )
end

# AbstractVector dispatch wrapper
function compute_narrative_context(
    closes::AbstractVector{Float64},
    atrs::AbstractVector{Float64},
    regime::AbstractVector{Float64},
    swing_confirm_bars::AbstractVector{Int64},
    swing_highs_price::AbstractVector{Float64},
    swing_highs_confirm::AbstractVector{Int64},
    swing_lows_price::AbstractVector{Float64},
    swing_lows_confirm::AbstractVector{Int64},
)::Dict{String, Vector{Float64}}
    return compute_narrative_context(
        collect(Float64, closes),
        collect(Float64, atrs),
        collect(Float64, regime),
        collect(Int64, swing_confirm_bars),
        collect(Float64, swing_highs_price),
        collect(Int64, swing_highs_confirm),
        collect(Float64, swing_lows_price),
        collect(Int64, swing_lows_confirm),
    )
end

end  # module ToonMath
