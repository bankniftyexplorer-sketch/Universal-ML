# TOON v4.0 — ROOT CAUSE & ENGINEERING REPORT
## Phase-1 Feature Ranking Defect

**Date:** 2026-04-03
**Component:** `holographic_engine.py` (Phase-1 Feature Selection Probe)
**Affected Lanes:** `1H` Core Engine, `1D` Core Engine

---

### 1. SYSTEM IMPACT & OBSERVATION
The `feature_selection_pipeline()` is failing to aggressively pare down holographic feature candidates. Instead of isolating the Top 100 highest-value features before Phase-3 walk-forward scoring, the pipeline allows **thousands** of weakly predictive features to bypass the cut.

**Resulting System Degradation:**
- Severe compute bloating during Phase-3 full walk-forward loops.
- Possible inclusion of noisy/unstable features in the final `FINAL_FEAT_BUDGET=40` pool.
- Slower end-to-end retraining timelines for both 1H and 1D paths.

---

### 2. DETERMINISTIC ROOT CAUSE
The defect originates in `phase1_ranking()` (`holographic_engine.py : 440`).

1. **Label Mismatch:** The system uses `LGBMClassifier(objective='binary')` for the shallow probe. However, `drop_unresolved=False` allows continuous pseudo-labels (e.g., neutral `0.5`) into the target dataset.
2. **Hard Exception:** The presence of `0.5` labels causes LightGBM to immediately throw:
   `LightGBMError: binary objective and metric shouldn't have > 2 class`
3. **Silent Failure Chain:** The exception is completely masked by a bare `except Exception: pass` (Line 481). Because every chronological fold crashes, the aggregation variable `folds_ok` remains exactly `0`.
4. **Fallback Triggered:** When `folds_ok == 0`, the function assumes a lack of data and returns the full `valid` list of features identically without any ranking cuts (Line 483).

---

### 3. SURGICAL FIX DIRECTIVES
*Rule: Do not modify feature engineering structure; repair the bottleneck to perform its intended coarse filtration.*

**Location:** `holographic_engine.py` logic block for `phase1_ranking`
**Action:** Transition the binary probe into a continuous regression constraint.

```python
# 1. Configuration Change:
cfg = dict(
    ...,
    objective='regression'  # Replaces 'binary'
)

# 2. Model Swap:
m = lgb.LGBMRegressor(**cfg)  # Replaces LGBMClassifier
```

**Logging:** 
Remove the silent exception swallowing trap `except Exception: pass` and replace it with at least standard standard output exception tracing so future bounds violations are immediately surfaced.

---

### 4. VALIDATION ZERO-TOLERANCE GUARDRAILS
Under no circumstances should the fix be pushed to production without strictly proving stability.

**Execution Steps Prior to Finalizing Fix:**
1. **[BASELINE]** Run `accuracy_guardrail.py capture` on `1H` and `1D` lanes (already completed).
2. **[PATCH]** Apply the `LGBMRegressor` patch carefully.
3. **[RETRAIN]** Regenerate models for `1H` (`universal_ml_engine.py`) and `1D` (`run_daily_model.py`).
4. **[VERIFY]** Prove deterministic extraction count. `Phase 1` output logs *must* show the candidate pool accurately collapsing to `100`.
5. **[METRICS]** Run `accuracy_guardrail.py compare`. 
   * The new metrics must equal or marginally outperform the baseline `coverage` and High-Confidence `hc_pct`. No regressions are permitted.
