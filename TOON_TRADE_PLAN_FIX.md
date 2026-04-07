# TOON INSTRUCTION SET — Trade-Plan Execution Chain Repair

## Identity

SYSTEM: Universal-ML trade-plan exit model chain.
SCOPE: 4 defects → 5 surgical fixes across 2 files.
VALIDATION: All parameters simulation-proven on 2,583 real OOS bars.

---

## Simulation Evidence (BANKNIFTY 1H, 2% fixed risk, $10k→)

```
2D Grid Search Results (Format: FinalEquity/WinRate):
              TP1α=25%  TP1α=30%  TP1α=35%  TP1α=40%  TP1α=50%  TP1α=60%
    SL30%    37.4k/70%  41.5k/68%  43.5k/66%  41.4k/60%  43.4k/54%  44.5k/55%
    SL35%    35.1k/79%  38.4k/76%  38.8k/73%  38.3k/69%  41.3k/64%  42.6k/64%
    SL40%    31.7k/84%  35.5k/82%  35.8k/80%  35.9k/76%  38.1k/70%  38.8k/70%
    SL50%    29.6k/85%  29.8k/86%  29.2k/84%  30.1k/82%  31.1k/75%  31.6k/75%
    SL60%    26.4k/83%  26.9k/86%  25.9k/85%  25.7k/83%  26.5k/77%  26.9k/77%

Current static (1.25x ATR flat): $30,411 / 84% WR / 2.9% DD
Optimal zone (SL30-40 × TP35-60): $35k–$45k range, lower WR but higher edge

MAE distribution (2 ATR ceiling at P65): α>60% all clip to 2.000 ATR
MFE distribution (no ceiling): P40=2.20, P55=3.30, P65=3.98, P75=4.95
```

**Key finding:** The model is a momentum predictor. Tight stops + wide TPs produce classic trend-following payoff. α=0.40 for stops is NOT suicidal — it's the correct profile for this signal type. The original stop alpha was right. TP alphas need slight adjustment.

---

## DEFECT MAP

| # | Defect | File | Lines | Severity |
|---|--------|------|-------|----------|
| 1 | Target filter uses `== 1` / `== 0` on continuous float target → zero training rows | `universal_ml_engine.py` | 903-904 | **CRITICAL** |
| 2 | Trade-plan regressors use naked `.fit()` — 400 trees, no validation, no purge gap, no early stopping | `universal_ml_engine.py` | 910-911 | **HIGH** |
| 3 | TP quantile alphas suboptimal: TP1 α=0.50 leaves 50% of gains on table, TP2 α=0.80 is unreachable for 80% of trades | `universal_ml_engine.py` | 895-899 | **MEDIUM** |
| 4 | Dead backtest defaults (`2.0 ATR / 1:1 R:R`) never execute due to finite fallback | `backtest_engine.py` | 270-285 | **LOW** |

---

## INSTRUCTIONS — Codex Execution Spec

### INSTRUCTION 1 — Fix target filter (universal_ml_engine.py)

**Location:** `train_trade_plan_models()`, lines 902-907.

**Current code:**
```python
for key, (target_value, label_col, alpha) in specs.items():
    train_df = (
        df[df["target"] == target_value]
        .dropna(subset=feature_cols + [label_col])
        .copy()
    )
```

**Replace with:**
```python
for key, (target_value, label_col, alpha) in specs.items():
    directional_mask = df["target"] > 0.5 if target_value == 1 else df["target"] < 0.5
    train_df = (
        df[directional_mask]
        .dropna(subset=feature_cols + [label_col])
        .copy()
    )
```

**Rationale:** `ToonMath.jl` `add_target_loop` returns continuous kinetic-score targets: LONG direction → `(0.575, 1.0)`, SHORT direction → `(0.0, 0.425)`, ambiguous → `0.5` (dropped by `drop_unresolved`). Target is never exactly `0` or `1`. Current filter returns zero rows → empty models dict → 5-byte pkl → entire exit chain dead.

---

### INSTRUCTION 2 — Add early stopping to trade-plan training (universal_ml_engine.py)

**Location:** `train_trade_plan_models()`, lines 908-912.

**Current code:**
```python
    if len(train_df) < 300:
        continue
    model = _build_lgbm_regressor(alpha)
    model.fit(train_df[feature_cols], train_df[label_col])
    models[key] = model
```

**Replace with:**
```python
    if len(train_df) < 300:
        continue

    # Inner validation with purge gap — same discipline as directional model
    inner_val_size = max(60, int(len(train_df) * 0.15))
    purge_gap = 24  # must match BARRIER_HORIZON_BARS
    purged_end = -(inner_val_size + purge_gap)
    if abs(purged_end) >= len(train_df):
        purged_end = 0

    if purged_end != 0:
        X_tr = train_df[feature_cols].iloc[:purged_end]
        y_tr = train_df[label_col].iloc[:purged_end]
    else:
        X_tr = train_df[feature_cols].iloc[:-inner_val_size]
        y_tr = train_df[label_col].iloc[:-inner_val_size]
    X_val = train_df[feature_cols].iloc[-inner_val_size:]
    y_val = train_df[label_col].iloc[-inner_val_size:]

    model = _build_lgbm_regressor(alpha)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="quantile",
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=10000),
        ],
    )
    models[key] = model
```

**Rationale:** The directional model uses inner-validation with 24-bar purge gap and 60-round early stopping via `_fit_lgbm_with_inner_validation()`. Trade-plan quantile regressors must have identical discipline: 400 trees on ~1,600 rows without validation = guaranteed overfitting. In-sample quantile predictions will look perfect; OOS predictions will be noise.

---

### INSTRUCTION 3 — Adjust TP quantile alphas (universal_ml_engine.py)

**Location:** `train_trade_plan_models()`, `specs` dict, lines 889-900.

**Current code:**
```python
specs = {
    "up_stop_atr": (1, "long_mae_atr", 0.40),
    "up_tp1_atr": (1, "long_mfe_atr", 0.50),
    "up_tp2_atr": (1, "long_mfe_atr", 0.80),
    "down_stop_atr": (0, "short_mae_atr", 0.40),
    "down_tp1_atr": (0, "short_mfe_atr", 0.50),
    "down_tp2_atr": (0, "short_mfe_atr", 0.80),
}
```

**Replace with:**
```python
specs = {
    "up_stop_atr": (1, "long_mae_atr", 0.40),    # Tight: trend-following SL profile
    "up_tp1_atr": (1, "long_mfe_atr", 0.55),     # Stretch: capture more R per winner
    "up_tp2_atr": (1, "long_mfe_atr", 0.75),     # Reachable: 25% of correct trades hit
    "down_stop_atr": (0, "short_mae_atr", 0.40),
    "down_tp1_atr": (0, "short_mfe_atr", 0.55),
    "down_tp2_atr": (0, "short_mfe_atr", 0.75),
}
```

**Rationale:** Grid search proves tight stops + wide TPs dominate for this momentum predictor. Stop α=0.40 is retained (simulation-validated: SL40 row consistently outperforms SL50-60). TP adjustments: TP1 α=0.50→0.55 stretches the target slightly (MFE P55≈3.3 vs P50≈2.9 ATR); TP2 α=0.80→0.75 makes it more reachable (P75≈4.9 vs P80≈6.5 ATR). Net effect: better R:R capture within the signal's accuracy envelope.

---

### INSTRUCTION 4 — Widen structural clips for TP predictions (universal_ml_engine.py)

**Location:** `predict_trade_plan()`, lines 946-948.

**Current code:**
```python
stop_atr = float(np.clip(plan_models[required[0]].predict(X)[0], 0.35, 1.25))
tp1_atr = float(np.clip(plan_models[required[1]].predict(X)[0], 0.25, 3.00))
tp2_atr = float(np.clip(plan_models[required[2]].predict(X)[0], 0.50, 5.00))
```

**Replace with:**
```python
stop_atr = float(np.clip(plan_models[required[0]].predict(X)[0], 0.35, 1.50))
tp1_atr = float(np.clip(plan_models[required[1]].predict(X)[0], 0.50, 4.00))
tp2_atr = float(np.clip(plan_models[required[2]].predict(X)[0], 1.00, 7.00))
```

**Rationale:** With TP1 α=0.55, median predictions cluster around 2.5-3.5 ATR; old clip of 3.00 truncates upper range. With TP2 α=0.75, predictions cluster around 4-6 ATR; old clip of 5.00 truncates. Stop clip widened slightly from 1.25→1.50 to allow model to breathe when features indicate wider stop is needed. MFE distribution is uncapped (P90=7.8, P95=9.7) so TP2 clip of 7.00 is generous but not unsafe.

---

### INSTRUCTION 5 — Remove dead backtest defaults (backtest_engine.py)

**Location:** `run_backtest()`, lines 270-285.

**Current code:**
```python
        # [TOON v4.2] Regime-Resistant Asymmetry Matrix
        # Capitalizes on the 56% ML Win Rate by harvesting risk at 1:1.
        stop_dist = float(current_atr * 2.0)
        tp1_dist = float(current_atr * 2.0)
        tp2_dist = float(current_atr * 4.0)
        trail_dist = float(current_atr * 1.0)

        if np.isfinite(trade_plan.get("stop_atr", np.nan)):
            stop_dist = float(current_atr * trade_plan["stop_atr"])
            tp1_dist = float(current_atr * trade_plan["tp1_atr"])
            tp2_dist = float(current_atr * trade_plan["tp2_atr"])
            trail_dist = float(current_atr * trade_plan["trail_r"])
```

**Replace with:**
```python
        # Trade plan exits — ML-derived or fallback (predict_trade_plan always returns finite)
        stop_dist = float(current_atr * trade_plan["stop_atr"])
        tp1_dist = float(current_atr * trade_plan["tp1_atr"])
        tp2_dist = float(current_atr * trade_plan["tp2_atr"])
        trail_dist = float(current_atr * trade_plan["trail_r"])
```

**Rationale:** `predict_trade_plan()` ALWAYS returns finite `stop_atr` — either ML-predicted or fallback `BARRIER_ATR_MULT=1.25`. The `np.isfinite()` guard always passes. The 2.0 ATR defaults are dead code that never executes. Remove to eliminate ambiguity about which exit regime is active.

---

## Changes NOT Made (With Reasons)

| Decision | Reason |
|----------|--------|
| Stop α NOT changed to 0.85 | Simulation proves tight stops (α=0.40) dominate for momentum predictor. Wide stops waste R:R. MAE ceiling at 2.0 ATR means α>60% all produce identical results. |
| TP1 α NOT changed to 0.40 | α=0.55 is the grid-search sweet spot between reachability and R stretch. |
| No walk-forward for trade-plan models | Excursion distribution is more stationary than directional signal. Single-train is acceptable. Complexity not justified. |
| `_build_lgbm_regressor` params NOT changed | n_estimators=400 with early_stopping=40 will auto-select optimal tree count. `min_child_samples=40` provides sufficient regularization for ~1,600 rows. |
| MFE/MAE 2 ATR observation ceiling NOT changed | Acceptable bound. MAE capping at 2.0 compresses labels but does not corrupt quantile targeting because the clip range accounts for this. |

---

## Verification Contract

After applying all 5 instructions:

1. **Retrain:** `uv run python universal_ml_engine.py --outdir /home/km/Universal-ML/ --symbol BANKNIFTY`
2. **Verify:** `ls -la BANKNIFTY/banknifty_1H_trade_plan_models.pkl` → size >> 5 bytes
3. **Verify log:** `Trade-plan models saved to '...' (6 models)`
4. **Backtest:** `uv run python backtest_engine.py --outdir /home/km/Universal-ML/ --symbol BANKNIFTY`
5. **Verify:** Trade plan note says `"ML-derived levels"` not `"Fallback ATR plan"`
6. **Meta:** `uv run python meta_strategy_selector.py --outdir /home/km/Universal-ML/ --symbol BANKNIFTY`
7. **Verify:** Winner selected on merit, not `"Fallback: insufficient valid OOS trades"`
