# Universal-ML Project Map

## Why this file exists

This is the shortest accurate map of the repo.
Read this first if you want to understand the system before opening the large engine files.

## One-screen summary

- Core purpose: predict market direction from OHLCV using geometry-only "holographic" features.
- Active production lanes:
  - `1H` intraday lane
  - `1D` daily lane
- Storage:
  - market data DB: `data_vault/ohlcv.db`
  - per-symbol artifacts: `<PROJECT_ROOT>/<SYMBOL>/`
  - local accuracy baselines: `<PROJECT_ROOT>/.accuracy_baselines/<SYMBOL>.json`
- Feature math lives in Julia kernels, called from Python.
- LightGBM is the model layer.
- Backtests and live inference reuse saved artifacts instead of rebuilding everything from scratch.
- Saved-artifact accuracy can be checked without retraining via `accuracy_guardrail.py`.

## Repo layout

| Path | Role | Read when |
| --- | --- | --- |
| `data_vault/vault_engine.py` | CSV ingestion, DB schema, macro timeframe generation, trade ledger | You care about data entering the system |
| `vault_engine.py` | Root-level compatibility shim that re-exports `DataVault` | A root script imports `vault_engine` directly |
| `inference_bridge.py` | Reads DB and returns timeframe-indexed pandas stacks | You care about how scripts fetch market data |
| `universal_ml_engine.py` | Main `1H` trainer plus shared runtime helpers/constants | You care about the intraday core |
| `daily_ml_engine.py` | Main `1D` trainer | You care about the daily core |
| `backtest_engine.py` | `1H` backtest/report generator | You care about intraday replay and equity curve |
| `daily_backtest_engine.py` | `1D` backtest/report generator | You care about daily replay and equity curve |
| `live_inference.py` | Fast `1H` signal path without retraining | You care about current live signal |
| `meta_strategy_selector.py` | Strategy zoo and winner selection layer | You care about model-selection logic |
| `accuracy_guardrail.py` | Rebuilds model-ready frames, replays saved OOS maps, compares against a local baseline | You care about proving an update did or did not change saved-artifact accuracy |
| `holographic_engine.py` | Python reference feature logic and feature-selection pipeline | You care about feature families or selection |
| `julia_bridge.py` | Python-to-Julia adapter | You care about bridge contracts or array prep |
| `ToonMath.jl` | Fast feature extraction and target-generation kernels | You care about core math/performance |
| `shadow_brain.py` | Optional meta-model trained from prior trade outcomes | You care about veto/approval overlay |
| `run_daily_model.py` | Legacy script | Ignore unless debugging old behavior |
| `LAUNCH_INSTRUCTIONS.md` | Operational runbook | You want commands, not architecture |

## Active pipelines

### `1H` intraday lane

1. `data_vault/vault_engine.py` ingests TradingView CSV logs into `market_dna`.
2. `inference_bridge.py` fetches `FUT` and `SPOT` stacks by timeframe.
3. `universal_ml_engine.prepare_intraday_thermodynamics()` builds the shared intraday state:
   - `FUT 1H` is the tradable timeline
   - `SPOT 1H` is the thermodynamic reference timeline
   - missing `SPOT` for a tradable `FUT 1H` bar is fatal
   - extra `SPOT`-only bars are tolerated
4. `universal_ml_engine._compute_atr14()` adds ATR for labelling and execution logic.
5. `julia_bridge.holographic_feature_engine_fast()` calls `ToonMath.jl` to build `1H` geometry features using `1H + 1D + 1W + 1M`.
6. `universal_ml_engine.merge_higher_tf()` aligns higher-timeframe context onto the `1H` frame.
7. `julia_bridge.add_target_fast()` creates labels from forward trade simulation.
8. `holographic_engine.feature_selection_pipeline()` reduces the feature set.
9. `universal_ml_engine.walk_forward()` performs honest out-of-sample validation.
10. Final artifacts are saved under `<SYMBOL>/` using the `1H` naming scheme.
11. `backtest_engine.py`, `live_inference.py`, and `meta_strategy_selector.py` reload the saved artifacts and reuse the same prep contract.
12. `accuracy_guardrail.py` can reconstruct the same `1H` model-ready frame from the DB and score the saved OOS probability map without retraining.

### `1D` daily lane

1. `data_vault/vault_engine.py` ingests raw bars and also generates `1W`, `1M`, `3M`, `6M`, `12M` macro layers from `1D`.
2. `daily_ml_engine.py` chooses primary `1D` bars:
   - prefer `FUT 1D`
   - fallback to `SPOT 1D`
3. `universal_ml_engine.inject_thermodynamic_basis()` adds spread/basis features from `SPOT 1D` when available.
4. Daily session placeholders are inert:
   - `session_time_pos = 0`
   - `eod_basis_momentum = 0`
5. `julia_bridge.holographic_feature_engine_daily()` builds daily geometry features from `1D + 1W + 1M + 3M`.
6. `daily_ml_engine.compute_macro_regime()` adds compact `6M` and `12M` regime overlays.
7. `daily_ml_engine.add_daily_confluence()` adds daily confluence terms.
8. `julia_bridge.add_target_fast()` creates daily labels with the daily horizon settings.
9. `feature_selection_pipeline()` plus `walk_forward_daily()` validate and train the final `1D` model.
10. Final artifacts are saved under `<SYMBOL>/` using the `1D` naming scheme.
11. `daily_backtest_engine.py` reconstructs the feature space and replays the saved `1D` model.
12. `accuracy_guardrail.py` can reconstruct the same `1D` model-ready frame from the DB and score the saved OOS probability map without retraining.

## Data contracts

### Database

`data_vault/ohlcv.db` contains:

- `market_dna`
  - key fields: `base_symbol`, `asset_class`, `timeframe`, `timestamp`
  - OHLCV payload plus `is_synthetic_vol`
- `performance_ledger`
  - trade outcome log used by `shadow_brain.py`

### In-memory market stack

`InferenceBridge.fetch_holographic_stack(symbol, asset_class)` returns:

- `dict[str, pd.DataFrame]`
- keys are timeframes like `1H`, `1D`, `1W`, `1M`, `3M`, `6M`, `12M`
- each frame is normalized to:
  - `time`, `open`, `high`, `low`, `close`, `volume`, `is_synthetic_vol`

## Artifact contracts

Artifacts live in `<PROJECT_ROOT>/<SYMBOL>/`.

### `1H`

- `{symbol}_1H_model.pkl`
- `{symbol}_1H_features.txt`
- `{symbol}_1H_oos_proba.pkl`
- `{symbol}_1H_trade_plan_models.pkl`
- `{symbol}_1H_ml_report.png`
- `{symbol}_1H_backtest_report.png`

### `1D`

- `{symbol}_1D_model.pkl`
- `{symbol}_1D_features.txt`
- `{symbol}_1D_oos_proba.pkl`
- `{symbol}_1D_trade_plan_models.pkl`
- `{symbol}_1D_ml_report.png`
- `{symbol}_1D_backtest_report.png`

### Local verification baselines

- `.accuracy_baselines/{SYMBOL}.json`
- local only; intentionally not committed
- stores:
  - saved-artifact hashes and mtimes
  - split-level accuracy metrics
  - OOS coverage
  - edge over baseline
  - enough identity data for `compare` to report `SAME RUN` vs `DIFFERENT RUN`

## Module dependencies

- `data_vault/vault_engine.py` writes the database.
- root scripts may import `vault_engine.py`, which is a compatibility shim over `data_vault/vault_engine.py`.
- `inference_bridge.py` reads the database.
- `universal_ml_engine.py` and `daily_ml_engine.py` are the training roots.
- `julia_bridge.py` is the adapter into `ToonMath.jl`.
- `holographic_engine.py` owns feature selection and the Python reference feature family definitions.
- `backtest_engine.py` and `daily_backtest_engine.py` depend on saved model artifacts.
- `live_inference.py` depends on saved `1H` artifacts and the latest DB state.
- `meta_strategy_selector.py` depends on the same reconstructed feature space as the `1H` lane.
- `accuracy_guardrail.py` depends on the same DB + feature-prep contracts as the training roots, but does not retrain models.

## Fast read order for another AI

If you need:

- operating commands: read `LAUNCH_INSTRUCTIONS.md`
- saved-artifact accuracy verification: read `accuracy_guardrail.py`, then `LAUNCH_INSTRUCTIONS.md`
- architecture: read this file
- DB and ingestion: read `data_vault/vault_engine.py`, then `inference_bridge.py`
- `1H` training behavior: read `universal_ml_engine.py`
- `1D` training behavior: read `daily_ml_engine.py`
- feature math: read `julia_bridge.py`, then `ToonMath.jl`, then `holographic_engine.py`
- backtest behavior: read `backtest_engine.py` or `daily_backtest_engine.py`
- live signal behavior: read `live_inference.py`
- meta strategy behavior: read `meta_strategy_selector.py`
- legacy daily code: read `run_daily_model.py` only if required

## What can usually be ignored

- `run_daily_model.py` unless you are comparing against old behavior
- `shadow_brain.py` unless you are using the optional trade-history veto layer
- `Manifest.toml` unless you are debugging Julia package resolution
- generated symbol folders unless you need model outputs
- `.accuracy_baselines/` unless you are auditing drift against a saved baseline

## Current design truths

- The system is not a generic plugin framework; it is a focused research/production trading stack.
- `1H` and `1D` are the active canonical lanes.
- The DB is the source of truth for market history.
- Julia does the heavy numerical work; Python orchestrates data flow, model training, reporting, and file management.
- There is no formal unit-test suite; confidence comes from walk-forward validation, saved OOS probability maps, and backtest reports.
- `accuracy_guardrail.py` is the machine-facing safety layer around the saved artifacts and reports.
- `compare` can tell you two different truths:
  - `SAME RUN`: the tracked artifact hashes match the captured baseline
  - `DIFFERENT RUN`: the artifacts have changed, even if the metrics are still acceptable
