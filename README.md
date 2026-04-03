# Universal-ML

Universal-ML is a research and production trading system that predicts market
direction from OHLCV data using geometry-only holographic features.

The active model lanes are:

- `1H` intraday
- `1D` daily

Core numerical feature extraction and label generation run through Julia
kernels in [ToonMath.jl](ToonMath.jl), while Python
orchestrates data flow, model training, walk-forward validation, reporting, and
artifact management.

## Fast Start

```bash
cd /home/km/Universal-ML
source mlenv/bin/activate
```

Train the active daily lane:

```bash
python daily_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python daily_backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Train the active intraday lane:

```bash
python universal_ml_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
python backtest_engine.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

Fast live intraday inference without retraining:

```bash
python live_inference.py --symbol NIFTY --outdir /home/km/Universal-ML/
```

## Read Order

If you are new to the repo, read these in order:

1. [PROJECT_MAP.md](PROJECT_MAP.md)
2. [LAUNCH_INSTRUCTIONS.md](LAUNCH_INSTRUCTIONS.md)
3. [AGENTS.md](AGENTS.md)

## Architecture Summary

- `data_vault/vault_engine.py`: CSV ingestion, database writes, macro timeframe
  generation, performance ledger
- `inference_bridge.py`: database reads and timeframe stack assembly
- `universal_ml_engine.py`: canonical `1H` training pipeline
- `daily_ml_engine.py`: canonical `1D` training pipeline
- `backtest_engine.py`: `1H` historical replay and report generation
- `daily_backtest_engine.py`: `1D` historical replay and report generation
- `julia_bridge.py`: Python-to-Julia bridge
- `ToonMath.jl`: high-performance feature and target kernels
- `shadow_brain.py`: optional trade-history overlay

## Operating Model

1. Ingest closed TradingView CSV exports into `data_vault/ohlcv.db`.
2. Build timeframe-aligned market stacks from the database.
3. Extract holographic features with Julia kernels.
4. Create forward targets from trade-simulation logic.
5. Run walk-forward validation and train final LightGBM models.
6. Save model, feature list, OOS probability map, trade-plan models, and
   reports under the symbol folder.
7. Reuse saved artifacts for backtesting and live inference.

## Repository Standards

- Generated symbol artifacts are intentionally ignored by git.
- The database file is local state, not source code.
- `PROJECT_MAP.md` is the shortest accurate architecture map.
- `LAUNCH_INSTRUCTIONS.md` is the operational runbook.
- Accuracy-sensitive code changes must be validated with real walk-forward and
  backtest outputs before being trusted.

## Environment

- Python dependencies: [requirements.txt](requirements.txt)
- Julia environment: [Project.toml](Project.toml), [Manifest.toml](Manifest.toml)
- Baseline CPU-only target: Intel i7-4770 class hardware with 16 GB RAM

## Current Status

- Active canonical lanes: `1H`, `1D`
- Database is the system source of truth
- No formal unit-test suite is present
- Validation confidence comes from walk-forward splits, OOS probability maps,
  and backtest reports
