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

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

HOLO_WINDOWS_1H = [3, 5, 8, 13, 21]
HOLO_WINDOWS_1D = [3, 5, 8, 13, 21]
HOLO_WINDOWS_1W = [3, 5, 8]
HOLO_WINDOWS_1M = [3, 5]

SKEL_PROMINENCE   = 0.10   # peak must stand 10% of window range above neighbours
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

    near_hi = np.maximum(norm_o, norm_c)
    near_lo = np.minimum(norm_o, norm_c)
    crange = norm_h - norm_l + 1e-9

    body_dom   = np.abs(norm_c - norm_o) / crange
    upper_wick = (norm_h - near_hi) / crange
    lower_wick = (near_lo - norm_l) / crange

    return dict(
        norm_o=norm_o, norm_h=norm_h, norm_l=norm_l, norm_c=norm_c,
        vol_frac=np.clip(vol_frac, 0.0, 1.0),
        body_dom=np.clip(body_dom, 0.0, 1.0),
        upper_wick=np.clip(upper_wick, 0.0, 1.0),
        lower_wick=np.clip(lower_wick, 0.0, 1.0),
    )


# ─────────────────────────────────────────────────────────────
# LAYER 1 — DNA STRAND
# ─────────────────────────────────────────────────────────────

def _dna(b: dict, tf: str, w: int) -> Dict[str, float]:
    """9 numbers per candle × N candles."""
    n = len(b['norm_c'])
    out: Dict[str, float] = {}
    for k in range(n):
        p = f"{tf}_w{w}_bar{k+1}"
        out[f"{p}_time_pos"]   = (k + 1) / n
        out[f"{p}_open"]       = float(b['norm_o'][k])
        out[f"{p}_high"]       = float(b['norm_h'][k])
        out[f"{p}_low"]        = float(b['norm_l'][k])
        out[f"{p}_close"]      = float(b['norm_c'][k])
        out[f"{p}_vol_frac"]   = float(b['vol_frac'][k])
        out[f"{p}_body_dom"]   = float(b['body_dom'][k])
        out[f"{p}_upper_wick"] = float(b['upper_wick'][k])
        out[f"{p}_lower_wick"] = float(b['lower_wick'][k])
    return out


# ─────────────────────────────────────────────────────────────
# LAYER 2 — GRAMMAR
# ─────────────────────────────────────────────────────────────

def _grammar(b: dict, tf: str, w: int) -> Dict[str, float]:
    """4 transition consistency numbers (requires w >= 3)."""
    if w < 3:
        return {}
    nc, nh, vf, bd = b['norm_c'], b['norm_h'], b['vol_frac'], b['body_dom']
    n = len(nc)
    cr = hr = ve = be = 0.0
    pairs = n - 1
    for k in range(1, n):
        cr += 1.0 if nc[k] > nc[k-1] else 0.0
        hr += 1.0 if nh[k] > nh[k-1] else 0.0
        ve += 1.0 if vf[k] > vf[k-1] else 0.0
        be += 1.0 if bd[k] > bd[k-1] else 0.0
    p = f"{tf}_w{w}_gram"
    return {
        f"{p}_close_rising":   cr / pairs,
        f"{p}_high_rising":    hr / pairs,
        f"{p}_vol_expanding":  ve / pairs,
        f"{p}_body_expanding": be / pairs,
    }


# ─────────────────────────────────────────────────────────────
# LAYER 3 — SPECTRAL (w >= 8 only)
# ─────────────────────────────────────────────────────────────

def _spectral(b: dict, tf: str, w: int) -> Dict[str, float]:
    """FFT magnitudes F1–F4 on close, body, vol sequences. No phase."""
    if w < 8:
        return {}
    out: Dict[str, float] = {}
    signals = {'close': b['norm_c'], 'body': b['body_dom'], 'vol': b['vol_frac']}
    hann = np.hanning(w)
    for name, sig in signals.items():
        mags = np.abs(np.fft.rfft(sig * hann))
        for fi in range(1, 5):
            v = float(mags[fi]) / w if fi < len(mags) else 0.0
            out[f"{tf}_w{w}_fft_{name}_f{fi}"] = 0.0 if not np.isfinite(v) else v
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

    def _slope(idxs):
        if len(idxs) < 2:
            return 0.0
        return float(nc[idxs[-1]] - nc[idxs[0]])

    def _first_pos(idxs):
        return float(idxs[0] / (n - 1)) if idxs else 0.5

    def _last_pos(idxs):
        return float(idxs[-1] / (n - 1)) if idxs else 0.5

    def _last_vol(idxs):
        return float(vf[idxs[-1]]) if idxs else 0.0

    pk_slope = _slope(peaks)
    tr_slope = _slope(troughs)

    if   pk_slope > 0 and tr_slope > 0:  regime =  1.0
    elif pk_slope < 0 and tr_slope < 0:  regime = -1.0
    elif pk_slope > 0 and tr_slope < 0:  regime =  0.5
    elif pk_slope < 0 and tr_slope > 0:  regime = -0.5
    else:                                 regime =  0.0

    prior_peak_vals = [nc[k] for k in peaks if k < n - 1]
    breakout = 1.0 if prior_peak_vals and nc[-1] > max(prior_peak_vals) else 0.0

    p = f"{tf}_w{w}_skel"
    return {
        f"{p}_npeaks":          float(len(peaks)),
        f"{p}_ntroughs":        float(len(troughs)),
        f"{p}_first_peak_pos":  _first_pos(peaks),
        f"{p}_last_peak_pos":   _last_pos(peaks),
        f"{p}_peaks_slope":     pk_slope,
        f"{p}_vol_at_last_peak": _last_vol(peaks),
        f"{p}_first_trough_pos": _first_pos(troughs),
        f"{p}_last_trough_pos":  _last_pos(troughs),
        f"{p}_troughs_slope":    tr_slope,
        f"{p}_vol_at_last_trough": _last_vol(troughs),
        f"{p}_regime":           regime,
        f"{p}_breakout":         breakout,
        f"{p}_drift":            float(nc[-1] - nc[0]),
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

    drift_1h = g('1h_w5_skel_drift')
    drift_1d = g('1d_w5_skel_drift')
    drift_1w = g('1w_w5_skel_drift')
    cr_1h    = g('1h_w5_gram_close_rising', 0.5)
    cr_1d    = g('1d_w5_gram_close_rising', 0.5)
    reg8_1h  = g('1h_w8_skel_regime')
    reg8_1d  = g('1d_w8_skel_regime')
    pk_1h    = g('1h_w5_skel_peaks_slope')
    pk_1d    = g('1d_w5_skel_peaks_slope')
    tr_1h    = g('1h_w5_skel_troughs_slope')
    tr_1d    = g('1d_w5_skel_troughs_slope')

    return {
        'mtf_conf_all_up':          1.0 if drift_1h > 0 and drift_1d > 0 and drift_1w > 0 else 0.0,
        'mtf_conf_all_down':        1.0 if drift_1h < 0 and drift_1d < 0 and drift_1w < 0 else 0.0,
        'mtf_conf_1h_1d_conflict':  1.0 if drift_1h * drift_1d < 0 else 0.0,
        'mtf_conf_bull_grammar':    1.0 if cr_1h > 0.6 and cr_1d > 0.6 else 0.0,
        'mtf_conf_peaks_aligned':   1.0 if np.sign(pk_1h) == np.sign(pk_1d) and pk_1h != 0 else 0.0,
        'mtf_conf_troughs_aligned': 1.0 if np.sign(tr_1h) == np.sign(tr_1d) and tr_1h != 0 else 0.0,
        'mtf_conf_compression':     1.0 if reg8_1h == -0.5 and reg8_1d == -0.5 else 0.0,
        'mtf_conf_expansion':       1.0 if reg8_1h ==  0.5 and reg8_1d ==  0.5 else 0.0,
        'mtf_conf_drift_diff':      float(drift_1h - drift_1d),
        'mtf_conf_drift_product':   float(abs(drift_1h) * abs(drift_1d)),
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
            avail = int(np.searchsorted(src_times, t, side='left'))

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

    p1 = feature_cols # correlation_filter(df, feature_cols)
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

    print(f"\n  [HOLO] ┌─ Selection Complete ─────────────────────────────────┐")
    print(f"  [HOLO] │  Generated   : {len(feature_cols)}")
    print(f"  [HOLO] │  After corr  : {len(p1)}")
    print(f"  [HOLO] │  After probe : {len(p2)}")
    print(f"  [HOLO] │  FINAL       : {len(final)}")
    print(f"  [HOLO] │  DNA:{layer['dna']}  Gram:{layer['grammar']}  "
          f"FFT:{layer['spectral']}  Skel:{layer['skeleton']}  "
          f"Conf:{layer['confluence']}")
    print(f"  [HOLO] └────────────────────────────────────────────────────────┘")

    if layer['spectral'] == 0:
        print("  [HOLO] NOTE: Zero FFT features survived final selection. "
              "Spectral is provisional — not removed yet.")

    return final, {'n_initial': len(feature_cols), 'n_after_corr': len(p1),
                   'n_after_probe': len(p2), 'n_final': len(final), 'layers': layer}
