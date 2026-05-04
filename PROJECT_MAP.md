# Universal-ML Project Map

## Why this file exists

This is the shortest accurate map of the repo.
Read this first if you want to understand the system before opening the large engine files.
After any bugfix, modification, or addition that changes repo behavior or contracts, this file must be updated in the same turn.

## One-screen summary

- Core purpose: predict market direction from OHLCV using Julia-native holographic geometry, SMC institutional-intent, Kalman structural, realized-volatility surface, and narrative-context features.
- Active production lanes:
  - `1H` intraday lane
  - `1D` daily lane
  - `VOL` daily volatility/range lane
- Storage:
  - canonical market data DB: `data_vault/ohlcv.db`
  - optional local custom Yahoo alias registry: `data_vault/custom_yahoo_instruments.json`
  - per-symbol artifacts: `<PROJECT_ROOT>/<SYMBOL>/`
  - cross-symbol sleeve admission registry: `<PROJECT_ROOT>/portfolio_sleeve_registry.json`
  - local accuracy baselines: `<PROJECT_ROOT>/.accuracy_baselines/<SYMBOL>.json`
- Market data is now guarded by a sync-quality ledger plus a strict runtime gate at the bridge.
- The vault can also run as a resource-gated auto-sync supervisor that only refreshes when CPU load and Yahoo probe throughput clear configured thresholds.
- Feature space is now built from five always-on Julia-driven families, and the `VOL` lane can optionally add a sixth implied-volatility family from companion VIX data:
  - holographic geometry features
  - SMC institutional-intent features
  - Kalman structural features
  - realized-volatility surface features
  - narrative-context awareness features
  - VIX implied-volatility features (`VOL` lane optional enrichment)
- Both active lanes now inject Julia-native realized-volatility surfaces and narrative-context awareness through `julia_bridge.py`.
- The `VOL` daily volatility lane can optionally enrich itself with `1H` intraday RV summaries, but a failed `1H` quality row no longer blocks the lane; it zeros those features, stamps `intra_available=0`, and continues on the strict daily stack.
- The `VOL` lane can also optionally enrich itself with a mapped `1D` VIX companion; if the companion is missing or fails quality, it zeros those features, stamps `vix_available=0`, and continues.
- The `VOL` lane now serves both next-day and 5-day volatility term-structure heads, heterogeneous QLIKE/MAE/Huber variance ensembles plus MAE/Huber/quantile excursion ensembles, companion-aware VIX regimes, synthetic-volume-aware intraday gating, post-hoc ML+HAR combination weights, isotonic point calibrators, decay-weighted asymmetric conformal intervals, and a forecast-confidence score from one saved artifact set.
- The raw `realized_volatility` series from the vault remains state only and is excluded from feature selection.
- Feature math lives in Julia kernels, called from Python bridges.
- `julia_bridge.py` is the only supported Python interface to `ToonMath.jl`, and it shifts higher-timeframe timestamps to bar-close availability before Julia dispatch or Python-side HTF projection mapping.
- `holographic_engine.py` is a frozen legacy module; only its feature-selection helpers remain live.
- LightGBM now trains as a 3-member ensemble in both active lanes.
- OOS probability maps are isotonic-calibrated and the fitted calibrators are saved as artifacts.
- Both active lanes can also train a second-stage policy artifact that learns execution filtering and risk scaling on top of the base directional model.
- Both active lanes preserve raw OHLC state inside their model-ready training frames so trade-plan and policy replay can simulate candidate trades without rebuilding the raw bars.
- Backtests and live inference reuse saved artifacts instead of rebuilding everything from scratch.
- Saved-artifact accuracy can be checked without retraining via `accuracy_guardrail.py`, now including explicit `VOL` replay scoring from saved OOS forecast maps.

## Repo layout

| Path | Role | Read when |
| --- | --- | --- |
| `data_vault/vault_engine.py` | Yahoo `SPOT` sync entrypoint, DB schema, incremental/full-refresh logic, resource-gated auto-sync supervisor, custom index alias registry, sync-quality ledger, macro timeframe generation, trade ledger | You care about data entering the system |
| `start_vault_autosync.sh` | Safe wrapper that starts the resource-gated auto-sync supervisor with a single-instance lock | You want the vault to stay running in the background |
| `ops/systemd/user/universal-ml-vault-autosync.service` | User-level systemd unit template for starting the vault auto-sync supervisor at login | You want the auto-sync loop to come up automatically with the system session |
| `vault_engine.py` | Root-level compatibility shim that re-exports `DataVault` | A root script imports `vault_engine` directly |
| `instrument_registry.py` | Shared runtime/artifact symbol identity helper that collapses aliases like `BTCUSDT -> BTC` and `SP500 -> SPX500` into one canonical artifact contract | You care about canonical symbol naming across scripts |
| `inference_bridge.py` | Reads DB, attaches `data_quality`, enforces the strict runtime gate on `FAIL` quality by default, and can optionally fetch mapped `1D` VIX companion series for the `VOL` lane | You care about how scripts fetch market data |
| `universal_ml_engine.py` | Main `1H` trainer plus shared runtime helpers/constants, nested consensus feature selection, ensemble model wrapper, and probability calibration | You care about the intraday core |
| `daily_ml_engine.py` | Main `1D` trainer with the same nested-selection, ensemble, and calibration contract | You care about the daily core |
| `daily_volatility_engine.py` | `VOL` trainer with optional `1H` + VIX enrichment, synthetic-volume-aware intraday gating, companion-aware VIX regimes, multi-target ranking, heterogeneous variance/excursion ensembles, extended HAR-RV combination, isotonic point calibration, decay-weighted asymmetric conformal intervals, and 1D+5D heads | You care about daily vol/range forecasting |
| `backtest_engine.py` | `1H` backtest/report generator | You care about intraday replay and equity curve |
| `daily_backtest_engine.py` | `1D` backtest/report generator | You care about daily replay and equity curve |
| `daily_volatility_backtest.py` | `VOL` backtest, calibrator-aware conformal coverage reporter, and OOS replay validator | You care about vol forecast replay |
| `live_volatility_inference.py` | `VOL` live forecast publisher that writes calibrated combined 1D+5D projected levels, intervals, and confidence to JSON from saved VOL heads | You care about dashboard-ready daily vol levels without retraining |
| `live_inference.py` | Fast `1H` signal path without retraining | You care about current live signal |
| `meta_strategy_selector.py` | Strategy zoo and winner selection layer | You care about model-selection logic |
| `sleeve_registry.py` | Builds the cross-symbol sleeve admission registry from honest replay metrics and chooses `base` vs `policy` execution per symbol/lane | You want deployable sleeve selection |
| `accuracy_guardrail.py` | Rebuilds model-ready frames, replays saved `1H`/`1D` OOS probability maps and `VOL` OOS forecast maps, compares against a local baseline | You care about proving an update did or did not change saved-artifact accuracy |
| `holographic_engine.py` | Frozen legacy module; only the selection helpers (`feature_selection_pipeline`, `correlation_filter`, `phase1_ranking`) remain live | You care about feature selection or legacy context |
| `julia_bridge.py` | Only supported Python-to-Julia adapter for holographic, SMC, Kalman, realized-volatility, VIX implied-volatility, target, and backtest helper kernels; it also enforces higher-timeframe close-availability alignment | You care about bridge contracts or array prep |
| `ToonMath.jl` | Fast holographic, SMC, Kalman, realized-volatility, and VIX implied-volatility extraction plus target-generation and replay kernels | You care about core math/performance |
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
9. `julia_bridge.rv_feature_engine_fast()` adds a 59-column `1H` realized-volatility surface:
   - 14 primary `1H` RV features, now including `rs_plus`, `rs_minus`, and `rs_leverage`
   - 15 columns each from `1D/1W/1M` (14 HTF RV features + 1 term-structure column), aligned to completed bars only
10. `julia_bridge.narrative_context_engine_fast()` adds a 23-column `1H` narrative-context family:
   - 7 primary `1H` narrative-context features from Kalman regime persistence, displacement, drawdown, swing count, and fib range state
   - 7 mapped columns each from `1D` and `1W`, aligned to completed bars only
   - 2 cross-timeframe alignment columns: `nc_cross_tf_sum` and `nc_cross_tf_abs`
11. `universal_ml_engine.merge_higher_tf()` aligns higher-timeframe raw context onto the `1H` frame.
12. `julia_bridge.add_target_fast()` creates decoupled labels from forward trade simulation: Kinematic (capped) ensuring entry classifier precision, and Excursion (uncapped) ensuring exit regressor ceilings.
   - directional trade-plan regressors treat targets as continuous kinetic scores and split direction with `target > 0.5` for long and `target < 0.5` for short
13. `universal_ml_engine.fold_consensus_feature_selection()` runs nested feature selection inside each walk-forward fold using `holographic_engine.correlation_filter()` and `phase1_ranking()`, then keeps the consensus feature set.
14. `universal_ml_engine.walk_forward()` performs honest out-of-sample validation with the shared ensemble trainer.
15. `calibrate_oos_probabilities()` fits an isotonic calibrator from the saved OOS map and persists it beside the model artifact.
16. `train_trade_plan_models()` fits the six `1H` exit quantile regressors with an inner validation slice, a `24`-bar purge gap, and dynamic minimum-sample guards before saving them under the standard trade-plan artifact.
17. `train_exit_surface_artifact()` evaluates a small discrete `1H` exit-template catalog on honest post-signal candidate trades and saves a symbol-scoped exit-surface artifact that selects a tested SL / TP1 / TP2 / trailing ladder by confidence bucket and direction.
18. `train_policy_artifact()` fits a second-stage `1H` opportunity head on honest post-exit-surface candidate trades and saves a symbol-lane-local decision threshold plus risk bands as a symbol-scoped artifact.
19. Final artifacts are saved under `<SYMBOL>/` using the `1H` naming scheme, now including the calibrator, exit-surface artifact, and policy artifact.
20. `backtest_engine.py` and `live_inference.py` reconstruct the same `1H + SMC + Kalman + RV + Narrative Context` SPOT-only feature-prep contract before consuming saved artifacts, and they prefer the saved exit-surface artifact over the continuous trade-plan regressors when present.
21. `backtest_engine.py` and `meta_strategy_selector.py` now apply the saved calibrator to OOS probabilities before confidence gating or strategy ranking, so historical simulation matches the calibrated runtime path.
22. The same saved `1H` opportunity head artifact is consumed by training summaries, backtests, live inference, and sleeve replay, so execution filtering is no longer a training-only idea.
23. `sleeve_registry.py` can replay saved `1H` and `1D` artifacts, reuse the saved exit-surface artifact, train candidate policy artifacts if needed, and write a deployable `portfolio_sleeve_registry.json` that decides whether each sleeve is enabled and whether `base` or `policy` execution is allowed.
24. `backtest_engine.py` and `live_inference.py` now obey the `1H` sleeve registry before executing signals or replay, so disabled sleeves are blocked consistently.
25. `live_inference.py` applies the saved calibrator to the latest prediction, then the policy artifact when the registry allows it, and `predict_next_bar()` also applies regime-aware confidence penalties from RV vol-of-vol and Kalman flat-regime checks.
26. `live_inference.py` can optionally pass the latest row through `shadow_brain.py` as a veto layer backed by `performance_ledger`.
27. Weak-confidence or stale latest-bar `1H` forecasts are surfaced as `NO_TRADE`, but forecast text and reports still print the currently selected exit ladder.
28. `accuracy_guardrail.py` can reconstruct the same `1H` model-ready frame from the DB and score the saved OOS probability map, including the saved calibrator when present, without retraining.
29. `meta_strategy_selector.py` now reuses the saved `1H` exit-surface artifact for strategy replay and local policy training when it is present and valid, and only falls back to locally trained trade-plan regressors if that artifact is missing.

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
9. `julia_bridge.rv_feature_engine_daily()` adds an 89-column daily realized-volatility surface:
   - 14 primary `1D` RV features, now including `rs_plus`, `rs_minus`, and `rs_leverage`
   - 15 columns each from `1W/1M/3M/6M/12M` (14 HTF RV features + 1 term-structure column), aligned to completed bars only
10. `julia_bridge.narrative_context_engine_daily()` adds a 23-column daily narrative-context family:
   - 7 primary `1D` narrative-context features from Kalman regime persistence, displacement, drawdown, swing count, and fib range state
   - 7 mapped columns each from `1W` and `1M`, aligned to completed bars only
   - 2 cross-timeframe alignment columns: `nc_cross_tf_sum` and `nc_cross_tf_abs`
11. `daily_ml_engine.compute_macro_regime()` adds compact `6M` and `12M` regime overlays.
12. `daily_ml_engine.add_daily_confluence()` adds daily confluence terms.
13. `julia_bridge.add_target_fast()` creates daily labels with the daily horizon settings.
14. `fold_consensus_feature_selection_daily()` performs the same nested consensus-selection contract as the `1H` lane before `walk_forward_daily()` trains the ensemble final model.
15. `calibrate_oos_probabilities()` fits and saves a `1D` isotonic calibrator from the saved daily OOS map.
16. `train_exit_surface_artifact()` evaluates a small discrete `1D` exit-template catalog on honest post-signal candidate trades and saves a symbol-scoped daily exit-surface artifact that selects a tested SL / TP1 / TP2 / trailing ladder by confidence bucket and direction.
17. `train_policy_artifact()` fits a second-stage `1D` opportunity head on honest post-exit-surface candidate trades and saves a symbol-lane-local decision threshold plus risk bands as a symbol-scoped artifact.
18. Final artifacts are saved under `<SYMBOL>/` using the `1D` naming scheme, now including the calibrator, exit-surface artifact, and policy artifact.
19. The daily lane now uses its own calibrated confidence gate (`0.72` at present) instead of the intraday threshold because calibrated daily probabilities were too dense around the old shared gate.
20. Weak-confidence or stale latest-bar `1D` forecasts are surfaced as `NO_TRADE`, but forecast text and reports still print the currently selected exit ladder and the opportunity head can still suppress execution.
21. `daily_backtest_engine.py` reconstructs the same `1D + RV + SMC + Kalman + Narrative Context + macro-regime` feature space, applies the saved daily calibrator to the OOS map, and replays the saved `1D` model through the same exit-surface and opportunity-head artifacts when present.
22. `sleeve_registry.py` writes the daily sleeve admission verdict into the shared registry, and `daily_backtest_engine.py` now obeys that verdict before replay.
23. `accuracy_guardrail.py` can reconstruct the same `1D` model-ready frame from the DB and score the saved OOS probability map, including the saved calibrator when present, without retraining.

### `VOL` daily volatility lane

1. Same data loading as the `1D` lane via `InferenceBridge`.
2. Same feature families as the `1D` lane, plus optional VIX implied-volatility enrichment: RV surface, holographic, SMC, Kalman, narrative context, macro regime overlays, and companion-VIX features.
3. `InferenceBridge.fetch_vix_series()` can optionally fetch a mapped `1D` VIX companion with `strict_gating=False`.
   - current companion map: `NIFTY`, `BANKNIFTY`, `SENSEX`, `FINNIFTY`, `MIDCPNIFTY`, `NIFTYNXT50` -> `INDIA_VIX`; `SPX500` -> `VIX`
   - if the companion series is missing or carries `FAIL` quality, the bridge returns `None` and the lane continues
   - the fetched frame now preserves the mapped companion identity so downstream VIX regime bucketing can adapt by geography
4. `julia_bridge.intraday_rv_summary_daily()` additionally injects 9 session-aggregated `1H` volatility features when hourly data exists, passes quality, and is not dominated by synthetic volume; otherwise those columns are zeroed, `intra_available` is set to `0`, and the lane continues.
5. `julia_bridge.vix_feature_engine_daily()` aligns companion VIX closes to the daily bars and injects 17 Julia-built `vix_*` surface columns; `vix_interaction_features()` adds 3 Python-side `VIX x RV` interaction terms, `vix_available` records whether companion data was present, and the Julia `vix_regime` cutoffs are now selected from the mapped companion identity (`INDIA_VIX` vs `VIX`).
6. `julia_bridge.vol_target_engine_daily()` creates 8 forward regression targets: 4 next-day heads and 4 next-5-day heads for YZ log-vol, log-range, up-excursion, and down-excursion.
7. Consensus feature selection keeps a shared 40-feature budget, ranks candidates across all 4 next-day VOL heads before voting a fold consensus, and now validates fold feature sets on multi-target edge vs HAR instead of log-vol alone.
8. `daily_volatility_engine.py` runs fixed-size expanding walk-forward regression validation with per-fold HAR-RV baseline comparison and QLIKE reporting for the variance-style heads.
9. The lane trains 8 separate LightGBM ensemble heads that all reuse the same consensus feature set; variance-style heads cycle QLIKE, MAE, and Huber member objectives, excursion heads now cycle MAE, Huber, and quantile members, and the lane fits post-hoc Bates-Granger ML+HAR combination weights, isotonic point calibrators, and decay-weighted asymmetric conformal interval widths from the honest OOS replay.
10. Artifacts are saved under `<SYMBOL>/` with the `VOL` naming scheme, including 1D models, 5D models, the OOS forecast map, a VOL calibrator bundle, a conformal/HAR post-hoc artifact, and the latest live-forecast JSON payload.
11. `live_volatility_inference.py` rebuilds the latest VOL feature frame from saved artifacts without retraining, applies the saved HAR-combination weights, isotonic calibrators, and conformal intervals, and refreshes the dashboard-facing JSON payload with 1D and 5D projected levels plus a forecast-confidence score and asymmetrically bounded conformal `projected_peak` / `projected_bottom` price levels.
12. `daily_volatility_backtest.py` replays the saved VOL artifacts, restores honest OOS forecasts from disk, re-applies the calibrator-aware conformal/HAR post-hoc layer, and reports interval coverage metrics alongside the older excursion coverage table, now using asymmetric conformal widths when present.
13. `accuracy_guardrail.py` can now rebuild the VOL feature frame, replay the saved VOL OOS forecast map with HAR/conformal overlays, and compare per-target regression metrics against a local baseline without retraining.

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

### Shared portfolio artifact

- `portfolio_sleeve_registry.json`
- built by `sleeve_registry.py`
- stores sleeve admission criteria, replay metrics, `enabled` status, and chosen `base` vs `policy` execution variant per `SYMBOL x LANE`

### `1H`

- `{symbol}_1H_model.pkl`
- `{symbol}_1H_features.txt`
- `{symbol}_1H_oos_proba.pkl`
- `{symbol}_1H_calibrator.pkl`
- `{symbol}_1H_trade_plan_models.pkl`
- `{symbol}_1H_exit_surface.pkl`
- `{symbol}_1H_policy_artifact.pkl`
- `{symbol}_1H_ml_report.png`
- `{symbol}_1H_backtest_report.png`

### `1D`

- `{symbol}_1D_model.pkl`
- `{symbol}_1D_features.txt`
- `{symbol}_1D_oos_proba.pkl`
- `{symbol}_1D_calibrator.pkl`
- `{symbol}_1D_trade_plan_models.pkl`
- `{symbol}_1D_exit_surface.pkl`
- `{symbol}_1D_policy_artifact.pkl`
- `{symbol}_1D_ml_report.png`
- `{symbol}_1D_backtest_report.png`

### `VOL`

- `{symbol}_VOL_model_logvol.pkl`
- `{symbol}_VOL_model_range.pkl`
- `{symbol}_VOL_model_up_exc.pkl`
- `{symbol}_VOL_model_dn_exc.pkl`
- `{symbol}_VOL_model_5d_logvol.pkl`
- `{symbol}_VOL_model_5d_range.pkl`
- `{symbol}_VOL_model_5d_up_exc.pkl`
- `{symbol}_VOL_model_5d_dn_exc.pkl`
- `{symbol}_VOL_features.txt`
- `{symbol}_VOL_oos_forecasts.pkl`
- `{symbol}_VOL_calibrators.pkl`
- `{symbol}_VOL_conformal.pkl`
- `{symbol}_VOL_live_forecast.json`
- `{symbol}_VOL_report.png`
- `{symbol}_VOL_backtest_report.png`

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
- `universal_ml_engine.py` also defines the ensemble wrapper used by saved model pickles.
- `julia_bridge.py` is the only supported adapter into `ToonMath.jl`; Python code should not include or reference `ToonMath.jl` directly.
- `holographic_engine.py` is a frozen legacy file whose only live selection helpers are `feature_selection_pipeline`, `correlation_filter`, and `phase1_ranking`.
- `backtest_engine.py` and `daily_backtest_engine.py` depend on saved model artifacts.
- `sleeve_registry.py` depends on saved model artifacts plus honest replay through the backtest engines.
- `live_inference.py` depends on saved `1H` artifacts, the latest DB state, and optionally `shadow_brain.py`.
- `meta_strategy_selector.py` depends on the same reconstructed `1H` feature space as the training lane.
- `accuracy_guardrail.py` depends on the same DB + feature-prep contracts as the training roots, but does not retrain models.
- Both active lanes consume Julia-native realized-volatility surfaces and narrative-context families through `julia_bridge.py`, while the DB-level `realized_volatility` series remains supporting state.

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
- Runtime consumers now canonicalize symbol aliases into one artifact identity, so `BTCUSDT` reuses the `BTC/` artifact tree and `SP500` reuses `SPX500/`.
- Operators can persist their own Yahoo-backed index aliases locally with `--register-index` or `--import-index` instead of editing Python maps.
- Operators can list currently available aliases with `--list-symbols` and inspect the CPU/network gate with `--resource-gate-status`.
- The DB is the source of truth for market history.
- `data_vault/ohlcv.db` is the canonical DB location; stray repo-root DB copies are not part of the intended design.
- Normal Yahoo syncs are incremental for speed; `--full-refresh` is the repair path when the ledger or history needs a hard rebuild.
- The vault can also stay resident in `--auto-sync` mode and wait for a healthy resource window before refreshing.
- `market_sync_quality` is the machine-facing health ledger for base-lane history.
- `InferenceBridge` now defaults to strict runtime gating, so `FAIL` quality is an operational firewall rather than a warning.
- Julia does the heavy numerical work; Python orchestrates data flow, model training, reporting, and file management.
- `ToonMath.jl` is the active holographic, SMC, Kalman, realized-volatility, and narrative-context engine.
- Both active lanes combine holographic geometry, SMC institutional-intent, Kalman structural, realized-volatility surface, and narrative-context features.
- The `1D` lane additionally injects `6M/12M` macro-regime overlays on top of the shared Julia feature families.
- Weak-confidence or stale latest-bar forecasts are marked `NO_TRADE`, but forecast text and PNG reports still print the experimental SL/TP/trailing levels.
- There is no formal unit-test suite; confidence comes from walk-forward validation, saved OOS probability maps, and backtest reports.
- `accuracy_guardrail.py` is the machine-facing safety layer around the saved artifacts and reports.
- `execution_guardrail.py` is the machine-facing replay safety layer around the saved `base` vs `policy` execution variants for both active lanes.
- `compare` can tell you two different truths:
  - `SAME RUN`: the tracked artifact hashes match the captured baseline
  - `DIFFERENT RUN`: the artifacts have changed, even if the metrics are still acceptable
