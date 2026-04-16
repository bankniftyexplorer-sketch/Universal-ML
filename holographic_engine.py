"""
FROZEN LEGACY — DO NOT MODIFY
=============================
This module is permanently frozen.

Live exports that remain production-resident in Python:
  - feature_selection_pipeline
  - correlation_filter
  - phase1_ranking

Dead functions removed from this module because ToonMath.jl, via
julia_bridge.py, owns the live holographic feature engine:
  - holographic_feature_engine
  - _bbox
  - _dna
  - _grammar
  - _spectral
  - _skeleton
  - _confluence
  - _process_tf
"""

import re
import warnings
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CORR_THRESHOLD = 0.80
PHASE1_FOLDS = 5
PHASE1_MAX_DEPTH = 3
PHASE1_TOP_N = 100
FINAL_FEAT_BUDGET = 40


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
    corr = sub.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    def _win(col):
        m = re.search(r"_w(\d+)_", col)
        return int(m.group(1)) if m else 0

    drop = set()
    for col in upper.columns:
        for other in upper[col][upper[col] > threshold].index:
            if col in drop or other in drop:
                continue
            drop.add(col if _win(col) >= _win(other) else other)

    filtered = [c for c in valid if c not in drop]
    print(
        f"  [HOLO] Correlation filter: {len(valid)} → {len(filtered)}"
        f" (removed {len(drop)} twins, threshold={threshold})"
    )
    return filtered


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION — PHASE 2: PROBE RANKING
# ─────────────────────────────────────────────────────────────


def phase1_ranking(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target",
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

    cfg = dict(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=PHASE1_MAX_DEPTH,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=4,
        verbose=-1,
        objective="regression",
        metric="mae",
    )

    imp_sum = np.zeros(len(valid))
    folds_ok = 0
    for f in range(n_folds):
        te = fold_size * (f + 1)
        Xt = clean[valid].iloc[:te]
        Yt = clean[target_col].iloc[:te]
        if len(Yt.unique()) < 2:
            continue
        m = lgb.LGBMRegressor(**cfg)
        try:
            m.fit(Xt, Yt)
            imp_sum += m.feature_importances_
            folds_ok += 1
        except Exception as e:
            print(f"  [HOLO] Phase 1 Fold {f} failed with exception: {e}")

    if folds_ok == 0:
        return valid

    avg = imp_sum / folds_ok
    ranked = sorted(zip(valid, avg), key=lambda x: x[1], reverse=True)
    top = [c for c, _ in ranked[:top_n]]

    fft_in_top = sum(1 for c in top if "_fft_" in c)
    if fft_in_top == 0:
        print(
            "  [HOLO] NOTE: Zero FFT features in phase-1 top candidates. "
            "Spectral layer may be weak on this data. Kept for now — monitor."
        )

    print(
        f"  [HOLO] Phase-1 ranking: {len(valid)} → {len(top)} "
        f"candidates over {folds_ok} folds"
    )
    return top


# ─────────────────────────────────────────────────────────────
# FEATURE SELECTION — PIPELINE (1800 → 40)
# ─────────────────────────────────────────────────────────────


def feature_selection_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    walk_forward_fn,
    target_col: str = "target",
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
            wf = walk_forward_fn(
                df,
                p2,
                n_splits=n_splits,
                min_train_bars=min_train_bars,
                test_size_ratio=test_size_ratio,
            )
            if wf and "feature_importance" in wf:
                ranked = sorted(
                    wf["feature_importance"].items(), key=lambda x: x[1], reverse=True
                )
                final = [c for c, _ in ranked[:FINAL_FEAT_BUDGET] if c in df.columns]
        except Exception as e:
            print(
                f"  [HOLO] Phase-3 failed ({e}). Trimming phase-2 to {FINAL_FEAT_BUDGET}."
            )
            final = p2[:FINAL_FEAT_BUDGET]
    else:
        final = [c for c in p2 if c in df.columns][:FINAL_FEAT_BUDGET]

    layer = {
        "dna": sum(1 for c in final if "_bar" in c and "_close" in c),
        "grammar": sum(1 for c in final if "_gram_" in c),
        "spectral": sum(1 for c in final if "_fft_" in c),
        "skeleton": sum(1 for c in final if "_skel_" in c),
        "confluence": sum(1 for c in final if c.startswith("mtf_conf")),
    }

    print("\n  [HOLO] ┌─ Selection Complete ─────────────────────────────────┐")
    print(f"  [HOLO] │  Generated   : {len(feature_cols)}")
    print(f"  [HOLO] │  After corr  : {len(p1)}")
    print(f"  [HOLO] │  After probe : {len(p2)}")
    print(f"  [HOLO] │  FINAL       : {len(final)}")
    print(
        f"  [HOLO] │  DNA:{layer['dna']}  Gram:{layer['grammar']}  "
        f"FFT:{layer['spectral']}  Skel:{layer['skeleton']}  "
        f"Conf:{layer['confluence']}"
    )
    print("  [HOLO] └────────────────────────────────────────────────────────┘")

    if layer["spectral"] == 0:
        print(
            "  [HOLO] NOTE: Zero FFT features survived final selection. "
            "Spectral is provisional — not removed yet."
        )

    return final, {
        "n_initial": len(feature_cols),
        "n_after_corr": len(p1),
        "n_after_probe": len(p2),
        "n_final": len(final),
        "layers": layer,
    }
