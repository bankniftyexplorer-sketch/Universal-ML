# Universal-ML Project Map

## Why this file exists

This is the shortest accurate map of the repo.
Read this first if you want to understand the system before opening the large engine files.
After any bugfix, modification, or addition that changes repo behavior or contracts, this file must be updated in the same turn.

## One-screen summary

- Core purpose: predict market direction from OHLCV using Julia-native holographic geometry, SMC institutional-intent, Kalman structural, and realized-volatility surface features.
- Active production lanes:
  - `1H` intraday lane
  - `1D` daily lane
- Storage:
  - canonical market data DB: `data_vault/ohlcv.db`
  - optional local custom Yahoo alias registry: `data_vault/custom_yahoo_instruments.json`
  - per-symbol artifacts: `<PROJECT_ROOT>/<SYMBOL>/`
  - local accuracy baselines: `<PROJECT_ROOT>/.accuracy_baselines/<SYMBOL>.json`
- Market data is now guarded by a sync-quality ledger plus a strict runtime gate at the bridge.
- The vault can also run as a resource-gated auto-sync supervisor that only refreshes when CPU load and Yahoo probe throughput clear configured thresholds.
- Feature space is now built from four Julia-driven families:
  - holographic geometry features
  - SMC institutional-intent features
  - Kalman structural features
  - realized-volatility surface features
- Both active lanes now inject Julia-native realized-volatility surfaces through `julia_bridge.py`.
- The raw `realized_volatility` series from the vault remains state only and is excluded from feature selection.
- Feature math lives in Julia kernels, called from Python bridges.
- `julia_bridge.py` is the only supported Python interface to `ToonMath.jl`, and it shifts higher-timeframe timestamps to bar-close availability before Julia dispatch or Python-side HTF projection mapping.
- `holographic_engine.py` is a frozen legacy module; only its feature-selection helpers remain live.
- LightGBM is the model layer.
- Backtests and live inference reuse saved artifacts instead of rebuilding everything from scratch.
- Saved-artifact accuracy can be checked without retraining via `accuracy_guardrail.py`.

## Repo layout

| Path | Role | Read when |
| --- | --- | --- |
| `data_vault/vault_engine.py` | Yahoo `SPOT` sync entrypoint, DB schema, incremental/full-refresh logic, resource-gated auto-sync supervisor, custom index alias registry, sync-quality ledger, macro timeframe generation, trade ledger | You care about data entering the system |
| `start_vault_autosync.sh` | Safe wrapper that starts the resource-gated auto-sync supervisor with a single-instance lock | You want the vault to stay running in the background |
| `ops/systemd/user/universal-ml-vault-autosync.service` | User-level systemd unit template for starting the vault auto-sync supervisor at login | You want the auto-sync loop to come up automatically with the system session |
| `vault_engine.py` | Root-level compatibility shim that re-exports `DataVault` | A root script imports `vault_engine` directly |
| `inference_bridge.py` | Reads DB, attaches `data_quality`, and enforces the strict runtime gate on `FAIL` quality by default | You care about how scripts fetch market data |
| `universal_ml_engine.py` | Main `1H` trainer plus shared runtime helpers/constants | You care about the intraday core |
| `daily_ml_engine.py` | Main `1D` trainer | You care about the daily core |
| `backtest_engine.py` | `1H` backtest/report generator | You care about intraday replay and equity curve |
| `daily_backtest_engine.py` | `1D` backtest/report generator | You care about daily replay and equity curve |
| `live_inference.py` | Fast `1H` signal path without retraining | You care about current live signal |
| `meta_strategy_selector.py` | Strategy zoo and winner selection layer | You care about model-selection logic |
| `accuracy_guardrail.py` | Rebuilds model-ready frames, replays saved OOS maps, compares against a local baseline | You care about proving an update did or did not change saved-artifact accuracy |
| `holographic_engine.py` | Frozen legacy module; only `feature_selection_pipeline`, `correlation_filter`, and `phase1_ranking` remain live | You care about feature selection or legacy context |
| `julia_bridge.py` | Only supported Python-to-Julia adapter for holographic, SMC, Kalman, realized-volatility, target, and backtest helper kernels; it also enforces higher-timeframe close-availability alignment | You care about bridge contracts or array prep |
| `ToonMath.jl` | Fast holographic, SMC, Kalman, and realized-volatility extraction plus target-generation and replay kernels | You care about core math/performance |
| `shadow_brain.py` | Optional meta-model trained from prior trade outcomes and used as a veto layer in live inference | You care about veto/approval overlay |
| `run_daily_model.py` | Legacy script | Ignore unless debugging old behavior |
| `LAUNCH_INSTRUCTIONS.md` | Operational runbook | You want commands, not architecture |

## Active pipelines

### `1H` intraday lane

1. `data_vault/vault_engine.py` syncs Yahoo `SPOT` history into `market_dna` for `1D` and `1H`, then forges `1W` and `1M` from `1D`.
   - `1D` uses Yahoo `max` history
   - `1H` asks Yahoo for the maximum available history, which Yahoo currently caps to roughly the last 730 days
   - default sync mode is incremental merge with overlap windows
   - `--full-refresh` forces a max-history rebuild for the requested symbol
   - `--auto-sync` keeps the vault alive and only starts a sync cycle when sampled CPU usage and Yahoo probe throughput pass the configured thresholds
   - the default wrapper now uses a `45%` CPU ceiling so the auto-sync loop can still run on the target workstation during normal activity
   - `--import-index ALIAS=YAHOO_TICKER` lets operators register and sync a new Yahoo-backed index without editing code
   - non-finite OHLC provider rows are dropped before audit/save so one broken upstream bar does not poison runtime quality
   - when a session-market `1D` bar is missing or broken but a completed `1H` session exists, the vault backfills the daily bar from hourly aggregation before writing quality
   - each run writes a `market_sync_quality` audit row per base timeframe
2. `inference_bridge.py` fetches the `SPOT` stack by timeframe.
   - `strict_gating=True` is the default
   - if any fetched timeframe carries `data_quality.status == "FAIL"` or `quality_status == "FAIL"`, the bridge raises `DataIntegrityError` and blocks the runtime
   - macro frames inherit the latest `1D` quality metadata
3. `universal_ml_engine.build_timeframe_selection()` selects the active `SPOT` primary frames used by training, backtest, live inference, and meta selection.
4. `universal_ml_engine.prepare_intraday_thermodynamics()` normalizes the shared intraday state:
   - `SPOT 1H` is the only active execution timeline
   - legacy basis columns are preserved as zeroed placeholders in SPOT-only mode
   - session-position vectors are encoded in exactly one place
5. `universal_ml_engine._compute_atr14()` adds ATR for labelling and execution logic.
6. `julia_bridge.holographic_feature_engine_fast()` calls `ToonMath.jl` to build `1H` geometry features using `1H + 1D + 1W + 1M`.
   - higher-timeframe timestamps are shifted to close-availability before Julia alignment, so intraday bars cannot see incomplete `1D/1W/1M` candles
7. `julia_bridge.smc_feature_engine_fast()` adds 47 institutional-intent features:
   - 34 primary `1H` SMC features
   - 6 higher-timeframe projection features from `1D/1W/1M`, mapped with the same close-availability contract
   - 7 confluence features
8. `julia_bridge.kalman_structural_engine_fast()` adds the `1H` Kalman structural family:
   - primary Kalman state
   - 7-ratio fib observation ladders
   - mapped higher-timeframe Kalman state from `1D/1W/1M`, aligned to completed bars only
9. `julia_bridge.rv_feature_engine_fast()` adds a 47-column `1H` realized-volatility surface:
   - 11 primary `1H` RV features
   - 12 columns each from `1D/1W/1M` (11 HTF RV features + 1 term-structure column), aligned to completed bars only
10. `universal_ml_engine.merge_higher_tf()` aligns higher-timeframe raw context onto the `1H` frame.
11. `julia_bridge.add_target_fast()` creates labels from forward trade simulation.
   - directional trade-plan regressors treat targets as continuous kinetic scores and split direction with `target > 0.5` for long and `target < 0.5` for short
12. `holographic_engine.feature_selection_pipeline()` reduces the combined feature set; it is the only live production export from that frozen legacy module.
13. `universal_ml_engine.walk_forward()` performs honest out-of-sample validation.
14. `train_trade_plan_models()` fits the six `1H` exit quantile regressors with an inner validation slice, a `24`-bar purge gap, and LightGBM early stopping before saving them under the standard trade-plan artifact.
15. Final artifacts are saved under `<SYMBOL>/` using the `1H` naming scheme.
16. `backtest_engine.py`, `live_inference.py`, and `meta_strategy_selector.py` reconstruct the same `1H + SMC + Kalman + RV` SPOT-only feature-prep contract before consuming saved artifacts.
17. `live_inference.py` can optionally pass the latest row through `shadow_brain.py` as a veto layer backed by `performance_ledger`.
18. Weak-confidence or stale latest-bar `1H` forecasts are surfaced as `NO_TRADE`, but the forecast text/report still shows the experimental SL/TP/trailing levels.
19. `accuracy_guardrail.py` can reconstruct the same `1H` model-ready frame from the DB and score the saved OOS probability map without retraining.

### `1D` daily lane

1. `data_vault/vault_engine.py` syncs Yahoo `SPOT 1D` bars and derives `1W`, `1M`, `3M`, `6M`, `12M` macro layers from them.
   - daily syncs also record quality rows covering gaps, staleness, synthetic volume, and fetch-retention state
2. `daily_ml_engine.py` uses `SPOT 1D` as the only active primary lane.
   - the same strict `InferenceBridge` gate applies before the daily lane can train or infer
3. `universal_ml_engine.inject_thermodynamic_basis()` preserves the historical basis columns as zeroed placeholders in SPOT-only mode.
4. Daily session placeholders are inert:
   - `session_time_pos = 0`
   - `eod_basis_momentum = 0`
5. The raw vault-side `realized_volatility` series remains state only and is excluded from feature selection.
6. `julia_bridge.holographic_feature_engine_daily()` builds daily geometry features from `1D + 1W + 1M + 3M`.
   - higher-timeframe timestamps are shifted to close-availability before Julia alignment, so daily bars cannot see incomplete macro candles
7. `julia_bridge.smc_feature_engine_daily()` adds 47 institutional-intent features for the daily lane:
   - 34 primary `1D` SMC features
   - 6 higher-timeframe projection features from `1W/1M/6M`, mapped with the same close-availability contract
   - 7 confluence features
8. `julia_bridge.kalman_structural_engine_daily()` adds the `1D` Kalman structural family:
   - primary Kalman state
   - 7-ratio fib observation ladders
   - mapped higher-timeframe Kalman state from `1W/1M/6M`, aligned to completed bars only
9. `julia_bridge.rv_feature_engine_daily()` adds a 71-column daily realized-volatility surface:
   - 11 primary `1D` RV features
   - 12 columns each from `1W/1M/3M/6M/12M` (11 HTF RV features + 1 term-structure column), aligned to completed bars only
10. `daily_ml_engine.compute_macro_regime()` adds compact `6M` and `12M` regime overlays.
11. `daily_ml_engine.add_daily_confluence()` adds daily confluence terms.
12. `julia_bridge.add_target_fast()` creates daily labels with the daily horizon settings.
13. `feature_selection_pipeline()` plus `walk_forward_daily()` validate and train the final `1D` model.
14. Final artifacts are saved under `<SYMBOL>/` using the `1D` naming scheme.
15. Weak-confidence or stale latest-bar `1D` forecasts are surfaced as `NO_TRADE`, but the forecast text/report still shows the experimental SL/TP/trailing levels.
16. `daily_backtest_engine.py` reconstructs the same `1D + RV + SMC + Kalman + macro-regime` feature space and replays the saved `1D` model.
17. `accuracy_guardrail.py` can reconstruct the same `1D` model-ready frame from the DB and score the saved OOS probability map without retraining.

## Data contracts

### Database

`data_vault/ohlcv.db` contains:

- `market_dna`
  - key fields: `base_symbol`, `asset_class`, `timeframe`, `timestamp`
  - OHLCV payload plus `is_synthetic_vol` and `realized_volatility`
- `market_sync_quality`
  - one row per sync run, base symbol, and timeframe
  - stores gap counts, synthetic-volume counts, staleness, fetch mode, refresh reason, and provider-retention status
  - this is the authoritative upstream truth used by the runtime gate
- `data_vault/custom_yahoo_instruments.json`
  - optional local operator registry for custom Yahoo-backed aliases such as `DAX -> ^GDAXI`
  - written by `data_vault/vault_engine.py --register-index` and `--import-index`
  - custom aliases are loaded alongside the built-in Yahoo map and can be synced with plain `--symbol <ALIAS>`
- `start_vault_autosync.sh`
  - wrapper around `data_vault/vault_engine.py --auto-sync`
  - uses a lock file in `/tmp` when `flock` is available so duplicate background loops do not stack up
- `performance_ledger`
  - trade outcome log used by `shadow_brain.py`
  - written during live execution flow when a directional plan is queued

### In-memory market stack

`InferenceBridge.fetch_holographic_stack(symbol, asset_class, strict_gating=True)` returns:

- `dict[str, pd.DataFrame]`
- keys are timeframes like `1H`, `1D`, `1W`, `1M`, `3M`, `6M`, `12M`
- each frame is normalized to:
  - `time`, `open`, `high`, `low`, `close`, `volume`, `is_synthetic_vol`
  - `realized_volatility` is also present when `include_realized_vol=True` is requested
  - `df.attrs["data_quality"]` carries the latest sync audit for that timeframe when available
  - `data_quality` exposes both `status` and `quality_status`
  - if strict gating is left enabled, any timeframe with `FAIL` aborts the fetch with `DataIntegrityError`

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
- `inference_bridge.py` reads the database, materializes sync-quality metadata, and blocks `FAIL` frames by default.
- `universal_ml_engine.py` and `daily_ml_engine.py` are the training roots.
- `julia_bridge.py` is the only supported adapter into `ToonMath.jl`; Python code should not include or reference `ToonMath.jl` directly.
- `holographic_engine.py` is a frozen legacy file whose only live exports are `feature_selection_pipeline`, `correlation_filter`, and `phase1_ranking`.
- `backtest_engine.py` and `daily_backtest_engine.py` depend on saved model artifacts.
- `live_inference.py` depends on saved `1H` artifacts, the latest DB state, and optionally `shadow_brain.py`.
- `meta_strategy_selector.py` depends on the same reconstructed `1H` feature space as the training lane.
- `accuracy_guardrail.py` depends on the same DB + feature-prep contracts as the training roots, but does not retrain models.
- Both active lanes consume Julia-native realized-volatility surfaces built from OHLC data, while the DB-level `realized_volatility` series remains supporting state.

## Fast read order for another AI

If you need:

- operating commands: read `LAUNCH_INSTRUCTIONS.md`
- saved-artifact accuracy verification: read `accuracy_guardrail.py`, then `LAUNCH_INSTRUCTIONS.md`
- architecture: read this file
- DB and ingestion: read `data_vault/vault_engine.py`, then `inference_bridge.py`
- `1H` training behavior: read `universal_ml_engine.py`
- `1D` training behavior: read `daily_ml_engine.py`
- feature math: read `julia_bridge.py`, then `ToonMath.jl`; read `holographic_engine.py` only for the frozen feature-selection helpers
- backtest behavior: read `backtest_engine.py` or `daily_backtest_engine.py`
- live signal behavior: read `live_inference.py`
- meta strategy behavior: read `meta_strategy_selector.py`
- live veto overlay: read `shadow_brain.py`
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
- Yahoo-fed `SPOT` is the active market-data contract for both lanes.
- The vault can sync curated aliases like `BTCUSDT` and raw Yahoo tickers such as `AAPL`, `^GDAXI`, and `EURUSD=X`.
- Operators can persist their own Yahoo-backed index aliases locally with `--register-index` or `--import-index` instead of editing Python maps.
- Operators can list currently available aliases with `--list-symbols` and inspect the CPU/network gate with `--resource-gate-status`.
- The DB is the source of truth for market history.
- `data_vault/ohlcv.db` is the canonical DB location; stray repo-root DB copies are not part of the intended design.
- Normal Yahoo syncs are incremental for speed; `--full-refresh` is the repair path when the ledger or history needs a hard rebuild.
- The vault can also stay resident in `--auto-sync` mode and wait for a healthy resource window before refreshing.
- `market_sync_quality` is the machine-facing health ledger for base-lane history.
- `InferenceBridge` now defaults to strict runtime gating, so `FAIL` quality is an operational firewall rather than a warning.
- Julia does the heavy numerical work; Python orchestrates data flow, model training, reporting, and file management.
- `ToonMath.jl` is the active holographic, SMC, Kalman, and realized-volatility engine.
- Both active lanes combine holographic geometry, SMC institutional-intent, Kalman structural, and realized-volatility surface features.
- The `1D` lane additionally injects `6M/12M` macro-regime overlays on top of the shared Julia feature families.
- Weak-confidence or stale latest-bar forecasts are marked `NO_TRADE`, but forecast text and PNG reports still print the experimental SL/TP/trailing levels.
- There is no formal unit-test suite; confidence comes from walk-forward validation, saved OOS probability maps, and backtest reports.
- `accuracy_guardrail.py` is the machine-facing safety layer around the saved artifacts and reports.
- `compare` can tell you two different truths:
  - `SAME RUN`: the tracked artifact hashes match the captured baseline
  - `DIFFERENT RUN`: the artifacts have changed, even if the metrics are still acceptable
