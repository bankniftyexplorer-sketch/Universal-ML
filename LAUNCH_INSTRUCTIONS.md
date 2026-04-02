# Universal-ML Launch Instructions

## Purpose

This project has two active model lanes:

- `1H` intraday model
- `1D` daily model

The database lives at:

- `/home/km/Universal-ML/data_vault/ohlcv.db`

Model artifacts are stored per symbol, for example:

- `/home/km/Universal-ML/NIFTY/`

---

## 1. Environment Setup

From the project root:

```bash
cd /home/km/Universal-ML
source mlenv/bin/activate
```

If the environment is missing:

```bash
cd /home/km/Universal-ML
python3 -m venv mlenv
source mlenv/bin/activate
pip install -r requirements.txt
```

---

## 2. How To Update The Database

Put new TradingView CSV exports into:

- `/home/km/Universal-ML/data_vault/inbox/`

Then ingest them:

```bash
cd /home/km/Universal-ML/data_vault
source ../mlenv/bin/activate
python vault_engine.py
```

What this does:

- reads all CSV files from `data_vault/inbox/`
- parses `SPOT` vs `FUT`
- writes data into `data_vault/ohlcv.db`
- auto-builds macro layers like `1W`, `1M`, `3M`, `6M`, `12M`
- removes successfully processed CSV files from `data_vault/inbox/`

Important:

- Only ingest fully closed bars.
- Do not ingest partial `1H` candles if you want clean predictions.
- Do not ingest partial daily candles if you want clean `1D` predictions.

---

## 3. Generate 1D Daily Model And Reports

Train the `1D` model:

```bash
cd /home/km/Universal-ML
source mlenv/bin/activate
python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Run the `1D` backtest report:

```bash
python daily_backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
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
source mlenv/bin/activate
python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Run the standalone `1H` backtest report:

```bash
python backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
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
source mlenv/bin/activate
python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
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
source mlenv/bin/activate
python meta_strategy_selector.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Use this only if you want the strategy-comparison verdict layer.

---

## 7. Recommended Daily Operating Routine

### If you want both `1H` and `1D` up to date

```bash
cd /home/km/Universal-ML/data_vault
source ../mlenv/bin/activate
python vault_engine.py

cd /home/km/Universal-ML
python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python daily_backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

### If you only want the latest `1H` signal during the day

```bash
cd /home/km/Universal-ML/data_vault
source ../mlenv/bin/activate
python vault_engine.py

cd /home/km/Universal-ML
python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
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
- Do **not** ingest partial hourly candles
- Best practice is:
  - wait for the `1H` bar to close
  - export/update the new closed `1H` data
  - ingest with `vault_engine.py`
  - run `live_inference.py`

For `1D`:

- run once after market close when the final daily bar is fixed
- ingest the new daily CSV
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

---

## 9. Symbol Example

Replace `NIFTY` with any symbol already present in the database, for example:

- `NIFTY`
- `BSX`

Example:

```bash
python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

---

## 10. Important Safety Notes

- The `1H` model is built on closed `1H` bars, not `20m` bars.
- Extra `SPOT` bars are ignored, but missing `SPOT` support for tradable `FUT 1H` bars is treated as a fatal data-quality issue.
- The project expects the active database to be:
  - `/home/km/Universal-ML/data_vault/ohlcv.db`
- Keep the symbol folders:
  - they contain trained models and reports

