# Universal-ML Launch Instructions

## Purpose

This project has two active model lanes:

- `1H` intraday model
- `1D` daily model

Current operational safety layer:

- `accuracy_guardrail.py` can validate saved-artifact accuracy without retraining
- `vault_engine.py` at the repo root is an import-compatibility shim, not the canonical ingestion entrypoint

The database lives at:

- `/home/km/Universal-ML/data_vault/ohlcv.db`

Model artifacts are stored per symbol, for example:

- `/home/km/Universal-ML/NIFTY/`

---

## 1. Environment Setup

From the project root:

```bash
cd /home/km/Universal-ML
uv sync --locked
uv run python -V
uv run python -c "import juliacall"
uv run ruff --version
julia --version
```

Optional interactive shell activation:

```bash
source .venv/bin/activate
```

Environment truths:

- the repo is managed with `uv`
- the active environment is `.venv`
- `pyproject.toml` and `uv.lock` are the authoritative Python environment source
- `requirements.txt` is a legacy fallback mirror, not the standard setup path
- `juliacall` is required for `julia_bridge.py`

---

## 2. How To Update The Database

Sync Yahoo `SPOT` history with:

```bash
cd /home/km/Universal-ML
uv run python data_vault/vault_engine.py --symbol NIFTY --pause-seconds 0
```

Notes:

- repeat `--symbol` to sync multiple instruments
- omit `--symbol` to sync the core watchlist first (`NIFTY`, `BANKNIFTY`, `SENSEX`, `FINNIFTY`, `MIDCPNIFTY`, `NIFTYNXT50`, `SPX500`, `BTC`), then the rest of the curated Yahoo map and any custom aliases saved locally
- `--symbol` accepts curated aliases like `BTCUSDT`, custom registered aliases, and raw Yahoo tickers like `AAPL`, `^GDAXI`, or `EURUSD=X`
- normal syncs are incremental by default for speed; use `--full-refresh` when you want a max-history rebuild from Yahoo
- `data_vault/vault_engine.py` is the canonical entrypoint
- the root-level `vault_engine.py` remains an import-compatibility shim

Register and import a new index without editing Python:

```bash
cd /home/km/Universal-ML
uv run python data_vault/vault_engine.py --import-index DAX=^GDAXI --pause-seconds 0
```

What this does:

- saves `DAX -> ^GDAXI` into `data_vault/custom_yahoo_instruments.json`
- syncs that index immediately into the canonical DB
- lets future runs use the plain alias:

```bash
uv run python data_vault/vault_engine.py --symbol DAX --pause-seconds 0
```

If you only want to save the alias for later:

```bash
uv run python data_vault/vault_engine.py --register-index DAX=^GDAXI
```

You can also skip alias registration and sync a raw Yahoo ticker directly:

```bash
uv run python data_vault/vault_engine.py --symbol ^GDAXI --pause-seconds 0
```

List the aliases you can use without opening Python:

```bash
uv run python data_vault/vault_engine.py --list-symbols
```

What this does:

- fetches Yahoo `SPOT` history for `1D` and `1H`
- merges incremental Yahoo refreshes onto the canonical rows in `data_vault/ohlcv.db`
- falls back to a full rebuild when the audit trail is missing, stale, or explicitly forced
- derives `1W`, `1M`, `3M`, `6M`, `12M` macro layers from `1D`
- records per-timeframe sync quality into `market_sync_quality`

Important:

- Only trust fully closed `1H` and `1D` bars.
- The active project contract is Yahoo-fed `SPOT` only.
- There is no CSV inbox in the current operating path.
- `1D` requests use Yahoo `period=max`; `1H` requests ask for `period=max` too, but Yahoo currently returns only the last ~730 days for hourly data.
- The sync loop is best-effort: if one Yahoo symbol returns no usable history, the run continues and reports a warning instead of aborting the whole refresh.
- Quality rows now capture gap counts, synthetic-volume counts, staleness, fetch mode, and whether the engine had to retain old bars because Yahoo returned nothing.

Automatic background refresh while the system is on:

```bash
cd /home/km/Universal-ML
uv run python data_vault/vault_engine.py \
  --auto-sync \
  --pause-seconds 0 \
  --auto-max-cpu-percent 20 \
  --auto-min-download-kbps 32 \
  --auto-check-interval-seconds 300 \
  --auto-min-sync-gap-seconds 3600
```

This mode:

- keeps the vault process alive
- samples CPU usage before each decision
- probes Yahoo throughput before each decision
- waits until `cpu <= 20%` and the download probe clears the configured KB/s floor
- only then starts the next sync cycle

Check why auto-sync is waiting:

```bash
uv run python data_vault/vault_engine.py --resource-gate-status
```

Simple start command with the default safe thresholds:

```bash
cd /home/km/Universal-ML
./start_vault_autosync.sh
```

Start it automatically when your Linux user session comes up:

```bash
mkdir -p ~/.config/systemd/user
cp /home/km/Universal-ML/ops/systemd/user/universal-ml-vault-autosync.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now universal-ml-vault-autosync.service
```

Check the service:

```bash
systemctl --user status universal-ml-vault-autosync.service
journalctl --user -u universal-ml-vault-autosync.service -f
```

---

## 3. Generate 1D Daily Model And Reports

Train the `1D` model:

```bash
cd /home/km/Universal-ML
uv run python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Run the `1D` backtest report:

```bash
uv run python daily_backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Main `1D` output files for `NIFTY`:

- `NIFTY/nifty_1D_model.pkl`
- `NIFTY/nifty_1D_features.txt`
- `NIFTY/nifty_1D_oos_proba.pkl`
- `NIFTY/nifty_1D_trade_plan_models.pkl`
- `NIFTY/nifty_1D_ml_report.png`
- `NIFTY/nifty_1D_backtest_report.png`

Notes:

- `daily_ml_engine.py` retrains the daily model and also prints the latest daily forecast.
- `daily_backtest_engine.py` rebuilds the daily equity-curve report from the saved `1D` model.

---

## 4. Generate 1H Intraday Model And Reports

Train the `1H` model:

```bash
cd /home/km/Universal-ML
uv run python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Run the standalone `1H` backtest report:

```bash
uv run python backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Main `1H` output files for `NIFTY`:

- `NIFTY/nifty_1H_model.pkl`
- `NIFTY/nifty_1H_features.txt`
- `NIFTY/nifty_1H_oos_proba.pkl`
- `NIFTY/nifty_1H_trade_plan_models.pkl`
- `NIFTY/nifty_1H_ml_report.png`
- `NIFTY/nifty_1H_backtest_report.png`

Notes:

- `universal_ml_engine.py` retrains the `1H` model, prints the latest `1H` forecast, and may also trigger the aligned backtest flow.
- `backtest_engine.py` explicitly rebuilds the `1H` backtest report from saved artifacts.

---

## 5. Get A Fast 1H Live Forecast Without Retraining

If the model is already trained and the database is up to date:

```bash
cd /home/km/Universal-ML
uv run python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Use this when:

- you already have a trained `1H` model
- you only want the latest intraday signal
- you do not want to retrain the whole model

This is the correct script for routine `1H` forecast refresh.

There is currently no separate dedicated `1D` live-inference script.
For `1D`, the active path is:

- update DB
- run `daily_ml_engine.py`

---

## 6. Optional Meta Strategy Selector

```bash
cd /home/km/Universal-ML
uv run python meta_strategy_selector.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Use this only if you want the strategy-comparison verdict layer.

---

## 6A. Accuracy Guardrail Before Trusting Any Change

Capture the current saved-artifact baseline:

```bash
cd /home/km/Universal-ML
uv run python accuracy_guardrail.py capture --symbol NIFTY --outdir /home/km/Universal-ML/
```

Later, compare the current repo state against that baseline:

```bash
uv run python accuracy_guardrail.py compare --symbol NIFTY --outdir /home/km/Universal-ML/
```

Lane-specific examples:

```bash
uv run python accuracy_guardrail.py capture --symbol NIFTY --outdir /home/km/Universal-ML/ --lane 1H
uv run python accuracy_guardrail.py compare --symbol NIFTY --outdir /home/km/Universal-ML/ --lane 1H

uv run python accuracy_guardrail.py capture --symbol NIFTY --outdir /home/km/Universal-ML/ --lane 1D
uv run python accuracy_guardrail.py compare --symbol NIFTY --outdir /home/km/Universal-ML/ --lane 1D
```

What this checks:

- rebuilds the model-ready `1H` and `1D` frames from the database
- replays the saved OOS probability maps against the real target series
- fails if accuracy, high-confidence accuracy, edge over baseline, or OOS
  coverage regresses
- compares tracked artifact hashes for:
  - model
  - features
  - OOS probability map
  - trade-plan models
  - ML report
  - backtest report

This is the safest first check before trusting refactors, optimizations, or
runtime changes.

Outputs to understand:

- `SAME RUN`:
  - the tracked artifact hashes match the captured baseline
- `DIFFERENT RUN`:
  - one or more tracked artifacts changed since the baseline was captured
- `PASS`:
  - the current state did not regress the guarded metrics versus the baseline
- `FAIL`:
  - one or more guarded metrics regressed

Where the baseline is stored:

- `.accuracy_baselines/<SYMBOL>.json`
- local only, intentionally ignored by git

Important limitation:

- the guardrail compares the current artifact + current DB state to a saved baseline
- it is a regression checker, not a historical time machine
- if the DB changed after an old PNG was generated, the guardrail may not recreate that old historical moment exactly

---

## 7. Recommended Daily Operating Routine

### If you want both `1H` and `1D` up to date

```bash
cd /home/km/Universal-ML
uv run python data_vault/vault_engine.py --symbol NIFTY --pause-seconds 0
uv run python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
uv run python daily_backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
uv run python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
uv run python backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
uv run python accuracy_guardrail.py compare --symbol NIFTY --outdir /home/km/Universal-ML/
```

### If you only want the latest `1H` signal during the day

```bash
cd /home/km/Universal-ML
uv run python data_vault/vault_engine.py --symbol NIFTY --pause-seconds 0
uv run python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

---

## 8. How Often Should You Run It?

### Short answer

- `1H` forecast: run after each fully closed `1H` candle
- `1D` forecast: run once after the daily candle is fully closed
- Full retraining: usually once per day is enough
- Do not run the full model every `20m`

### Practical recommendation

For `1H`:

- Do **not** retrain every `20m`
- Do **not** sync partial hourly candles
- Best practice is:
  - wait for the `1H` bar to close
  - sync Yahoo with `data_vault/vault_engine.py --symbol <SYMBOL>`
  - run `live_inference.py`

For `1D`:

- run once after market close when the final daily bar is fixed
- sync Yahoo daily history
- run `daily_ml_engine.py`

For retraining:

- `1H` full retrain: once per trading day is reasonable
- `1D` full retrain: once per trading day is reasonable
- backtests are for validation, not required every 20 minutes

### Best operating cadence

- During market hours:
  - update DB only when a new closed `1H` bar exists
  - run `live_inference.py`
- After market close:
  - update DB with final `1H` and `1D` bars
  - run `daily_ml_engine.py`
  - optionally run `daily_backtest_engine.py`
  - run `universal_ml_engine.py`
  - optionally run `backtest_engine.py`
  - run `accuracy_guardrail.py compare` if you want a strict saved-artifact drift check

---

## 9. Symbol Example

Replace `NIFTY` with any symbol mapped in `data_vault/yfinance_vault.py`, for example:

- `NIFTY`
- `BANKNIFTY`
- `SENSEX`
- `BTC`

Example:

```bash
python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

---

## 10. Important Safety Notes

- The `1H` model is built on closed `1H` bars, not `20m` bars.
- The active data contract is Yahoo-fed `SPOT`; runtime consumers no longer require a `FUT` lane.
- The project expects the active database to be:
  - `/home/km/Universal-ML/data_vault/ohlcv.db`
- Keep the symbol folders:
  - they contain trained models and reports
