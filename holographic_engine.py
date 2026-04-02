"""
TOON v4.0 — Pure Holographic Candle Pattern Engine
===================================================
Pure geometry. No classical indicators. The model discovers
500+ candle patterns from shape, not price level.

STRICT RULES (enforced here, never waived):
  - Window always ends at bar t-1. Bar t is NEVER inside a window.
  - No FFT on windows < 8.  No Skeleton on windows < 5.
  - FFT: magnitudes only — phase angles discarded.
  - Data always chronological. Never shuffled.
  - n_jobs capped at 4.
  - atr14 used for labelling only; excluded from feature list.
  - Final feature count into walk_forward: 40.
"""

import re
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════
# LEGACY: Feature extraction replaced by ToonMath.jl via julia_bridge.py.
# Retained for: (1) reference, (2) feature_selection_pipeline host.
# For new code: from julia_bridge import holographic_feature_engine_fast
# ════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

HOLO_WINDOWS_1H = [3, 5, 8, 13, 21]
HOLO_WINDOWS_1D = [3, 5, 8, 13, 21]
HOLO_WINDOWS_1W = [3, 5, 8]
HOLO_WINDOWS_1M = [3, 5]
HOLO_WINDOWS_3M = [3, 5]

SKEL_PROMINENCE   = 0.05   # [TOON vX.0] Relaxed prominence for fine skeleton detection
CORR_THRESHOLD    = 0.80   # correlation cutoff for twin removal
PHASE1_FOLDS      = 5
PHASE1_MAX_DEPTH  = 3
PHASE1_TOP_N      = 100
FINAL_FEAT_BUDGET = 40


# ─────────────────────────────────────────────────────────────
# BOUNDING BOX NORMALISER
# ─────────────────────────────────────────────────────────────

def _bbox(opens, highs, lows, closes, volumes):
    """
    Normalise N candles to [0,1] coordinates.

    Price range: max(highs) → 1.0, min(lows) → 0.0.
    Zero-range guard: substitute 0.00001 * median(closes).
    Volume: each bar's share of window total; equal share if total==0.

    Returns dict of arrays (all length N).
    """
    n = len(closes)
    hi = highs.max()
    lo = lows.min()
    rng = hi - lo

    if rng < 1e-9:
        med = float(np.median(closes))
        rng = max(med * 0.00001, 1e-9)
        print(f"  [HOLO] Warning: near-zero range window (N={n}). "
              f"Substituting rng={rng:.6f}. Coords will cluster near 0.5.")

    norm_o = np.clip((opens  - lo) / rng, 0.0, 1.0)
    norm_h = np.clip((highs  - lo) / rng, 0.0, 1.0)
    norm_l = np.clip((lows   - lo) / rng, 0.0, 1.0)
    norm_c = np.clip((closes - lo) / rng, 0.0, 1.0)

    vols = np.nan_to_num(volumes, nan=0.0)
    total_vol = vols.sum()
    vol_frac = vols / total_vol if total_vol > 1e-9 else np.full(n, 1.0 / n)
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0] # Handle first index
    
    price_move = (2 * closes) - prev_close - opens
    
    # norm_bias replaces old body/wick logic
    norm_bias = ((((closes - lo) - (hi - closes)) / (rng + 1e-9)) + 1.0) / 2.0
    
    # volume_bias (Cumulative Delta Imbalance)
    buy_vol = np.where(price_move > 0, vols, 0)
    sell_vol = np.where(price_move < 0, vols, 0)
    volume_bias = (np.sum(buy_vol) - np.sum(sell_vol)) / (np.sum(vols) + 1e-9)

    return dict(
        norm_o=norm_o, norm_h=norm_h, norm_l=norm_l, norm_c=norm_c,
        vol_frac=np.clip(vol_frac, 0.0, 1.0),
        norm_bias=np.clip(norm_bias, 0.0, 1.0),
        volume_bias=float(np.clip(volume_bias, -1.0, 1.0)),
    )


# ─────────────────────────────────────────────────────────────
# LAYER 1 — DNA STRAND
# ─────────────────────────────────────────────────────────────

def _dna(b: dict, tf: str, w: int) -> Dict[str, float]:
    """13 numbers per candle (Geometric + Thermodynamic)."""
    n = len(b['norm_c'])
    out: Dict[str, float] = {}
    for k in range(n):
        p = f"{tf}_w{w}_bar{k+1}"
        # Expose only norm_c, norm_bias, and volume_bias per bar
        out[f"{p}_close"]      = float(b['norm_c'][k])
        out[f"{p}_norm_bias"]  = float(b['norm_bias'][k])
        out[f"{p}_volume_bias"]= float(b['volume_bias'])
        
        # Thermodynamic Expose (Defaults to 0.0 if not injected)
        out[f"{p}_basis_pct"]  = float(b.get('basis_pct', [0.0]*n)[k])
        out[f"{p}_basis_z"]    = float(b.get('basis_z_score', [0.0]*n)[k])
        out[f"{p}_basis_v5"]   = float(b.get('basis_vel_5', [0.0]*n)[k])
        out[f"{p}_basis_v10"]  = float(b.get('basis_vel_10', [0.0]*n)[k])
        
        # Session Gap Physics
        out[f"{p}_session_pos"] = float(b.get('session_time_pos', [0.0]*n)[k])
        out[f"{p}_eod_momentum"] = float(b.get('eod_basis_momentum', [0.0]*n)[k])
    return out


# ─────────────────────────────────────────────────────────────
# LAYER 2 — GRAMMAR
# ─────────────────────────────────────────────────────────────

def _grammar(b: dict, tf: str, w: int) -> Dict[str, float]:
    """[TOON vX.0] Full sequence encoding via lag matrix. LightGBM sees order natively."""
    if w < 3:
        return {}
    
    nc, vf = b['norm_c'], b['vol_frac']
    nb, vb = b['norm_bias'], b['volume_bias']
    n = len(nc)
    out = {}
    
    # Send up to the entire window size backward as distinct lag columns
    for lag in range(min(w, 21)):
        # -1 is the most recent bar, -2 is previous, etc.
        idx = (n - 1) - lag
        if idx >= 0:
            out[f"{tf}_w{w}_gram_nb_lag{lag}"] = float(nb[idx])
            out[f"{tf}_w{w}_gram_vb_lag{lag}"] = float(vb)
        else:
            out[f"{tf}_w{w}_gram_nb_lag{lag}"] = 0.0
            out[f"{tf}_w{w}_gram_vb_lag{lag}"] = 0.0
            
    return out


# ─────────────────────────────────────────────────────────────
# LAYER 3 — SPECTRAL (w >= 8 only)
# ─────────────────────────────────────────────────────────────

def _spectral(b: dict, tf: str, w: int) -> Dict[str, float]:
    """[TOON vX.0] Replace FFT with Rolling Autocorrelation at Fibonacci lags."""
    if w < 10:
        return {}
    
    out: Dict[str, float] = {}
    sig = b['norm_c']
    
    for lag in [1, 2, 3, 5, 8]:
        if len(sig) > lag:
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.corrcoef(sig[lag:], sig[:-lag])[0, 1]
            out[f"{tf}_w{w}_fft_nc_ac{lag}"] = 0.0 if not np.isfinite(r) else float(r)
        else:
            out[f"{tf}_w{w}_fft_nc_ac{lag}"] = 0.0
            
    return out


# ─────────────────────────────────────────────────────────────
# LAYER 4 — SKELETON (w >= 5 only)
# ─────────────────────────────────────────────────────────────

def _skeleton(b: dict, tf: str, w: int) -> Dict[str, float]:
    """13 structural peak/trough features with prominence filter."""
    if w < 5:
        return {}
    nc, vf = b['norm_c'], b['vol_frac']
    n = len(nc)
    prom = SKEL_PROMINENCE

    peaks, troughs = [], []
    for k in range(1, n - 1):
        left, mid, right = nc[k-1], nc[k], nc[k+1]
        if mid > left and mid > right and (mid - max(left, right)) >= prom:
            peaks.append(k)
        elif mid < left and mid < right and (min(left, right) - mid) >= prom:
            troughs.append(k)

    def _last_vol(idxs):
        return float(vf[idxs[-1]]) if idxs and 'vol_frac' in b else 0.0

    pk_vals = [nc[k] for k in peaks if k < n - 1]
    tr_vals = [nc[k] for k in troughs if k < n - 1]
    last_swing_high = pk_vals[-1] if pk_vals else 1.0
    last_protected_low = tr_vals[-1] if tr_vals else 0.0
    
    nc_last = nc[-1]
    nb_last = b['norm_bias'][-1]
    
    skel_smc_phase = 0.0
    if nc_last > last_swing_high:
        pk_slope = pk_vals[-1] - pk_vals[0] if len(pk_vals) >= 2 else 0.0
        # +1.0: Bullish BOS (Close > Last Swing High in Up-Trend)
        # +2.0: Bullish CHoCH (Close > Last Protected High in Down-Trend)
        skel_smc_phase = 1.0 if pk_slope >= 0 else 2.0
    elif nc_last < last_protected_low:
        tr_slope = tr_vals[-1] - tr_vals[0] if len(tr_vals) >= 2 else 0.0
        skel_smc_phase = -1.0 if tr_slope <= 0 else -2.0

    skel_liquidity_grab = 0.0
    if nc_last < last_protected_low and nb_last > 0.7:
        skel_liquidity_grab = 1.0
        
    vol_at_last_peak = _last_vol(peaks)
    skel_ob_validity = vol_at_last_peak * abs(skel_smc_phase)

    p = f"{tf}_w{w}_skel"
    return {
        f"{p}_smc_phase":       float(skel_smc_phase),
        f"{p}_liquidity_grab":  float(skel_liquidity_grab),
        f"{p}_ob_validity":     float(skel_ob_validity),
        f"{p}_last_swing_high": float(last_swing_high),
        f"{p}_last_swing_low":  float(last_protected_low),
    }


# ─────────────────────────────────────────────────────────────
# CONFLUENCE (10 cross-TF features)
# ─────────────────────────────────────────────────────────────

def _confluence(row_feats: Dict[str, float]) -> Dict[str, float]:
    """
    Called once per bar after all TF features exist in row_feats.
    Uses w5 skeleton drift from 1h/1d/1w and w5 grammar close_rising
    from 1h/1d, plus w8 regime from 1h/1d.
    """
    def g(k, default=0.0):
        return row_feats.get(k, default)

    phase_1h = g('1h_w13_skel_smc_phase')
    phase_1d = g('1d_w13_skel_smc_phase')
    phase_1w = g('1w_w8_skel_smc_phase')
    
    if phase_1h > 0 and phase_1d > 0 and phase_1w > 0:
        mtf_conf_smc_sync = 1.0
    elif phase_1h < 0 and phase_1d < 0 and phase_1w < 0:
        mtf_conf_smc_sync = -1.0
    else:
        mtf_conf_smc_sync = 0.0
        
    # Premium / Discount Pricing
    h1_close = g('1h_w13_bar13_close') 
    d1_high = g('1d_w13_skel_last_swing_high', 1.0)
    d1_low = g('1d_w13_skel_last_swing_low', 0.0)
    
    pd_zone = (h1_close - d1_low) / (d1_high - d1_low + 1e-9)
    
    return {
        'mtf_conf_smc_sync': float(mtf_conf_smc_sync),
        'mtf_conf_pd_zone':  float(pd_zone)
    }


# ─────────────────────────────────────────────────────────────
# MASTER HOLOGRAPHIC ENGINE
# ─────────────────────────────────────────────────────────────

def _process_tf(
    base_df: pd.DataFrame,
    src_df: pd.DataFrame,
    tf_label: str,
    windows: List[int],
    is_primary: bool,
) -> List[Dict[str, float]]:
    """
    For every row in base_df (1H bars), compute holographic features
    from src_df (same or higher TF).

    LOOK-AHEAD RULE: for bar at index i, window ends at bar t-1.
    Primary (1H): uses iloc positions directly.
    Higher TF: searchsorted on timestamps to find closed bars.
    """
    n_base = len(base_df)
    rows: List[Dict[str, float]] = [{} for _ in range(n_base)]

    src = src_df.sort_values('time').reset_index(drop=True)
    src_times = src['time'].values if not is_primary else None

    src_o = src['open'].to_numpy(dtype=float)
    src_h = src['high'].to_numpy(dtype=float)
    src_l = src['low'].to_numpy(dtype=float)
    src_c = src['close'].to_numpy(dtype=float)
    src_v = src['volume'].to_numpy(dtype=float)

    base_times = base_df['time'].values if not is_primary else None

    for i in range(n_base):
        if is_primary:
            avail = i          # bars 0…i-1 are closed before bar i
        else:
            t = base_times[i]
            avail = max(0, int(np.searchsorted(src_times, t, side='right')) - 1)

        feat: Dict[str, float] = {}
        for w in windows:
            start = avail - w
            if start < 0:
                continue
            b = _bbox(
                src_o[start:avail], src_h[start:avail],
                src_l[start:avail], src_c[start:avail],
                src_v[start:avail],
            )
            # Passthrough Thermodynamic & Session vectors
            if 'basis_pct' in src_df.columns:
                b['basis_pct'] = src_df['basis_pct'].to_numpy(dtype=float)[start:avail]
                b['basis_z_score'] = src_df['basis_z_score'].to_numpy(dtype=float)[start:avail]
                b['basis_vel_5'] = src_df['basis_vel_5'].to_numpy(dtype=float)[start:avail]
                b['basis_vel_10'] = src_df['basis_vel_10'].to_numpy(dtype=float)[start:avail]
            if 'session_time_pos' in src_df.columns:
                b['session_time_pos'] = src_df['session_time_pos'].to_numpy(dtype=float)[start:avail]
                b['eod_basis_momentum'] = src_df['eod_basis_momentum'].to_numpy(dtype=float)[start:avail]
            feat.update(_dna(b, tf_label, w))
            if w >= 3:
                feat.update(_grammar(b, tf_label, w))
            if w >= 8:
                feat.update(_spectral(b, tf_label, w))
            if w >= 5:
                feat.update(_skeleton(b, tf_label, w))
        rows[i] = feat

    return rows


def holographic_feature_engine(
    df_1h: pd.DataFrame,
    df_1d: Optional[pd.DataFrame] = None,
    df_1w: Optional[pd.DataFrame] = None,
    df_1m: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Adds all holographic features to df_1h in-place (copy returned).
    Nothing from classical indicators. Pure geometry only.
    """
    df = df_1h.copy()
    n = len(df)

    print("  [HOLO] 1H windows…")
    rows_1h = _process_tf(df, df, '1h', HOLO_WINDOWS_1H, is_primary=True)

    rows_1d = [{} for _ in range(n)]
    if df_1d is not None and len(df_1d) >= 3:
        print("  [HOLO] 1D windows…")
        rows_1d = _process_tf(df, df_1d, '1d', HOLO_WINDOWS_1D, is_primary=False)

    rows_1w = [{} for _ in range(n)]
    if df_1w is not None and len(df_1w) >= 3:
        print("  [HOLO] 1W windows…")
        rows_1w = _process_tf(df, df_1w, '1w', HOLO_WINDOWS_1W, is_primary=False)

    rows_1m = [{} for _ in range(n)]
    if df_1m is not None and len(df_1m) >= 3:
        print("  [HOLO] 1M windows…")
        rows_1m = _process_tf(df, df_1m, '1m', HOLO_WINDOWS_1M, is_primary=False)

    # Merge per-row dicts, then add confluence
    print("  [HOLO] Merging + confluence…")
    all_rows = []
    for i in range(n):
        merged: Dict[str, float] = {}
        merged.update(rows_1h[i])
        merged.update(rows_1d[i])
        merged.update(rows_1w[i])
        merged.update(rows_1m[i])
        merged.update(_confluence(merged))
        all_rows.append(merged)

    feat_df = pd.DataFrame(all_rows, index=df.index)
    n_generated = len(feat_df.columns)
    print(f"  [HOLO] Features generated before filtering: {n_generated}")

    # Attach to base
    new_cols = [c for c in feat_df.columns if c not in df.columns]
    return pd.concat([df, feat_df[new_cols]], axis=1)


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION — PHASE 1: CORRELATION FILTER
# ─────────────────────────────────────────────────────────────

def correlation_filter(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = CORR_THRESHOLD,
) -> List[str]:
    """
    Remove one member of each highly-correlated pair.
    When |r| > threshold, drop the one from the larger window.
    Returns filtered feature list.
    """
    valid = [c for c in feature_cols if c in df.columns]
    sub = df[valid].replace([np.inf, -np.inf], np.nan).fillna(0)
    corr = sub.corr(method='pearson').abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    def _win(col):
        m = re.search(r'_w(\d+)_', col)
        return int(m.group(1)) if m else 0

    drop = set()
    for col in upper.columns:
        for other in upper[col][upper[col] > threshold].index:
            if col in drop or other in drop:
                continue
            drop.add(col if _win(col) >= _win(other) else other)

    filtered = [c for c in valid if c not in drop]
    print(f"  [HOLO] Correlation filter: {len(valid)} → {len(filtered)}"
          f" (removed {len(drop)} twins, threshold={threshold})")
    return filtered


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION — PHASE 2: PROBE RANKING
# ─────────────────────────────────────────────────────────────

def phase1_ranking(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'target',
    n_folds: int = PHASE1_FOLDS,
    top_n: int = PHASE1_TOP_N,
) -> List[str]:
    """
    Shallow probe (max_depth=3) on first n_folds chronological windows.
    Returns top_n features by averaged importance.
    """
    valid = [c for c in feature_cols if c in df.columns]
    clean = df[valid + [target_col]].replace([np.inf, -np.inf], np.nan).fillna(0)
    clean = clean.dropna(subset=[target_col]).reset_index(drop=True)
    n = len(clean)

    fold_size = n // (n_folds + 1)
    if fold_size < 50 or n < 300:
        print("  [HOLO] Phase-1 ranking: too little data, skipping.")
        return valid

    cfg = dict(n_estimators=150, learning_rate=0.05, num_leaves=15,
               max_depth=PHASE1_MAX_DEPTH, min_child_samples=30,
               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
               reg_alpha=0.1, reg_lambda=0.1, random_state=42,
               n_jobs=4, verbose=-1, objective='binary')

    imp_sum = np.zeros(len(valid))
    folds_ok = 0
    for f in range(n_folds):
        te = fold_size * (f + 1)
        Xt = clean[valid].iloc[:te]
        Yt = clean[target_col].iloc[:te]
        if len(Yt.unique()) < 2:
            continue
        m = lgb.LGBMClassifier(**cfg)
        try:
            m.fit(Xt, Yt)
            imp_sum += m.feature_importances_
            folds_ok += 1
        except Exception:
            pass

    if folds_ok == 0:
        return valid

    avg = imp_sum / folds_ok
    ranked = sorted(zip(valid, avg), key=lambda x: x[1], reverse=True)
    top = [c for c, _ in ranked[:top_n]]

    fft_in_top = sum(1 for c in top if '_fft_' in c)
    if fft_in_top == 0:
        print("  [HOLO] NOTE: Zero FFT features in phase-1 top candidates. "
              "Spectral layer may be weak on this data. Kept for now — monitor.")

    print(f"  [HOLO] Phase-1 ranking: {len(valid)} → {len(top)} "
          f"candidates over {folds_ok} folds")
    return top


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION — PIPELINE (1800 → 40)
# ─────────────────────────────────────────────────────────────

def feature_selection_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    walk_forward_fn,
    target_col: str = 'target',
    min_train_bars: int = 2500,
    test_size_ratio: float = 0.15,
    n_splits: int = 10,
) -> Tuple[List[str], Dict]:
    """
    Phase 1 — correlation filter (→ ~1000)
    Phase 2 — shallow probe ranking (→ top 100)
    Phase 3 — full walk-forward on 100, keep top 40
    Returns (final_feature_cols, meta_dict)
    """
    print(f"\n  [HOLO] Feature selection starting with {len(feature_cols)} features…")

    p1 = correlation_filter(df, feature_cols)
    p2 = phase1_ranking(df, p1, target_col=target_col)

    final = p2
    if len(p2) > FINAL_FEAT_BUDGET and walk_forward_fn is not None:
        print(f"  [HOLO] Phase-3: full walk-forward on {len(p2)} candidates…")
        try:
            wf = walk_forward_fn(df, p2, n_splits=n_splits,
                                 min_train_bars=min_train_bars,
                                 test_size_ratio=test_size_ratio)
            if wf and 'feature_importance' in wf:
                ranked = sorted(wf['feature_importance'].items(),
                                key=lambda x: x[1], reverse=True)
                final = [c for c, _ in ranked[:FINAL_FEAT_BUDGET] if c in df.columns]
        except Exception as e:
            print(f"  [HOLO] Phase-3 failed ({e}). Trimming phase-2 to {FINAL_FEAT_BUDGET}.")
            final = p2[:FINAL_FEAT_BUDGET]
    else:
        final = [c for c in p2 if c in df.columns][:FINAL_FEAT_BUDGET]

    layer = {
        'dna':        sum(1 for c in final if '_bar'  in c and '_close' in c),
        'grammar':    sum(1 for c in final if '_gram_' in c),
        'spectral':   sum(1 for c in final if '_fft_'  in c),
        'skeleton':   sum(1 for c in final if '_skel_' in c),
        'confluence': sum(1 for c in final if c.startswith('mtf_conf')),
    }

    print("\n  [HOLO] ┌─ Selection Complete ─────────────────────────────────┐")
    print(f"  [HOLO] │  Generated   : {len(feature_cols)}")
    print(f"  [HOLO] │  After corr  : {len(p1)}")
    print(f"  [HOLO] │  After probe : {len(p2)}")
    print(f"  [HOLO] │  FINAL       : {len(final)}")
    print(f"  [HOLO] │  DNA:{layer['dna']}  Gram:{layer['grammar']}  "
          f"FFT:{layer['spectral']}  Skel:{layer['skeleton']}  "
          f"Conf:{layer['confluence']}")
    print("  [HOLO] └────────────────────────────────────────────────────────┘")

    if layer['spectral'] == 0:
        print("  [HOLO] NOTE: Zero FFT features survived final selection. "
              "Spectral is provisional — not removed yet.")

    return final, {'n_initial': len(feature_cols), 'n_after_corr': len(p1),
                   'n_after_probe': len(p2), 'n_final': len(final), 'layers': layer}
