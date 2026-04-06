# Agent Guidelines for Universal-ML

## Project Overview

Universal-ML is a machine learning trading system that predicts financial instrument direction using holographic (pure geometry-based) features. It processes OHLCV data from TradingView CSV logs across multiple timeframes (1H, 1D, 1W, 1M).

## Fast AI Read

- Read `PROJECT_MAP.md` first for the shortest accurate architecture map.
- Read `LAUNCH_INSTRUCTIONS.md` for run commands and operating routine.
- Open the large engine files only after the map tells you which lane matters.

## Instruction Hierarchy

- Explicit user requests are the highest-priority contract.
- Task-specific instructions named or linked by the user in token docs, spec docs, patch notes, handover notes, or other `.md` files define the exact task contract for that turn.
- This `AGENTS.md` file is the general operating law. It governs judgment, discipline, safety, and cleanup. It does not override an explicit task specification unless that specification would clearly destroy data or contradict an explicit user constraint.
- If multiple docs apply, use the most task-specific document first, then use `PROJECT_MAP.md` and `LAUNCH_INSTRUCTIONS.md` to anchor the change in the real system.

## General Operating Doctrine

- Solve the actual problem, not the nearest visible symptom. Trace the behavior to the contract that owns it before editing.
- Work from the system map inward. Identify the active lane (`1H`, `1D`, backtest, live inference, selector, guardrail, data vault) before touching large files.
- Preserve canonical contracts across training, backtest, inference, and guardrail paths. A feature or label change is incomplete if parity is broken in downstream consumers.
- Do not improvise strategy logic, feature semantics, model thresholds, artifact names, or file contracts when the repo already defines them.
- Avoid broad refactors when a narrow fix will solve the real issue. Minimize change surface area, but do not under-fix the root cause.
- Never trade away model quality, walk-forward honesty, output accuracy, or artifact compatibility for convenience or speed unless the user explicitly approves that trade.
- Assume the target machine is `i7-4770`, `16 GB DDR3 RAM`, `M.2 NVMe`. Optimize for that hardware with memory-aware and CPU-aware engineering, but without changing results.
- Prefer deterministic, auditable behavior over cleverness. Hidden fallbacks, silent behavior drift, and speculative “helpful” shortcuts are defects unless explicitly intended.
- Validate all affected execution paths, not just the file being edited. In this repo, that often means training plus at least the matching backtest, live, selector, or guardrail path.
- After any bugfix, modification, or addition that changes architecture, lane behavior, feature contracts, data contracts, or file ownership, update `PROJECT_MAP.md` in the same turn before considering the task complete.
- Do not leave dead compatibility shims, one-off migration hacks, debug prints, placeholder code, or speculative scaffolding after the task unless the user explicitly asks to keep them.
- If a task-specific document says exactly what to insert, preserve names, signatures, keys, and contracts exactly unless the existing codebase makes literal insertion invalid. In that case, adapt only the minimum necessary while preserving the intended interface.

## Temporary Artifact Discipline

- Temporary files are allowed only when they materially help complete the task and no cleaner path exists.
- Prefer ephemeral locations outside the repo for scratch work, especially `/tmp`, over repo-local throwaway files.
- If a repo-local temporary artifact is unavoidable during debugging, validation, repro, migration, or patching, it must be clearly disposable and must be deleted before the task is considered complete.
- This deletion requirement includes, but is not limited to: scratch scripts, patch helpers, ad hoc verification runners, copied snippets, temporary notebooks, local test fixtures, alternate database copies, CSV dumps, debug logs, profiling output, `__pycache__`, `.pyc` files, and one-off markdown task files created only for the current task.
- Never leave behind duplicate or stray database files. The canonical market database is `data_vault/ohlcv.db`; do not keep alternate repo-root `.db` byproducts unless the user explicitly requests them.
- Never leave behind verification folders or patch utilities such as `verification/`, `patch_*.py`, or similar disposable helpers once the relevant task is finished.
- If a temporary artifact must survive across multiple commands in the same task, keep its scope narrow, keep its naming obvious, and remove it in the same turn after its purpose is complete.
- Do not report a task as finished until temporary byproducts have been removed.
- Before finalizing a task, perform a cleanup sweep and confirm that no disposable artifacts remain in the repo tree.

## Definition of Done

- A task is done only when the requested source change is implemented, the relevant behavior is checked to a reasonable level, and disposable byproducts created during the task are removed.
- “It works” is not enough if the repo is left dirtier, more ambiguous, or more fragile than before.
- Keep permanent files only if they are part of the maintained system, explicitly requested by the user, or necessary project documentation for future work.

## Build/Lint/Test Commands

### Environment Truth

- This repo is now managed with `uv`, not `mlenv`.
- The project environment is the repo-local `.venv` created and maintained by `uv`.
- The authoritative Python dependency sources are `pyproject.toml` and `uv.lock`.
- `.python-version` currently pins Python `3.12`.
- `requirements.txt` still exists, but it is now a legacy mirror of core packages, not the primary environment source of truth.
- `julia_bridge.py` requires a working Julia executable on `PATH` and the Python package `juliacall`.
- `juliacall` is a declared runtime dependency and is expected to be present after a standard `uv sync --locked`.
- `ruff` is declared in the project `dev` dependency group, and the repo is configured so a standard `uv sync --locked` installs it.
- A clean `uv sync --locked` should therefore be sufficient for the normal Python developer workflow in this repo.

### Environment Setup
```bash
# Create/update the uv-managed project environment from locked metadata
uv sync --locked

# Sanity checks for the actual runtime contract
uv run python -V
uv run python -c "import juliacall"
uv run ruff --version
julia --version
```

### Day-To-Day Command Execution

- Prefer `uv run <command>` for all project commands.
- Optional interactive shell activation:
```bash
source .venv/bin/activate
```
- Do not create or use `mlenv`.
- Do not create an additional virtual environment unless the task is explicitly about environment surgery.
- Do not treat `pip install -r requirements.txt` as the standard setup path for this repo.

### Running the System
```bash
# Train model and generate walk-forward validation
uv run python universal_ml_engine.py --outdir /path/to/data/

# Run backtest on trained model
uv run python backtest_engine.py --outdir /path/to/data/

# Run meta-strategy selector
uv run python meta_strategy_selector.py --outdir /path/to/data/

# Run daily model (legacy 1D/1W strategy)
uv run python run_daily_model.py --outdir /path/to/data/
```

### Testing
- **No formal test suite exists** - this is a research/production trading system
- Manual validation via backtest_engine.py output (equity curves, Sharpe, Drawdown)
- Verify OOS probability maps are properly generated and loaded
- Preferred baseline syntax check:
```bash
uv run python -m py_compile universal_ml_engine.py daily_ml_engine.py backtest_engine.py daily_backtest_engine.py live_inference.py meta_strategy_selector.py accuracy_guardrail.py julia_bridge.py
```
- Preferred saved-artifact safety check before trusting refactors:
```bash
uv run python accuracy_guardrail.py compare --symbol <SYMBOL> --outdir /home/km/Universal-ML/
```

### Linting/Type Checking
```bash
uv run ruff check .

# Auto-fix safe issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Full lint + format
uv run ruff check --fix . && uv run ruff format .
```

Important:
- If `uv run ruff ...` fails, the environment is out of sync and should be repaired with `uv sync --locked` before doing deeper work.

**Known issues (intentional)**:
- `E402`: Module imports after `warnings.filterwarnings('ignore')` — required for suppressing LightGBM verbosity
- `E701`: Multiple statements on one line (`if cond: assignment`) — stylistic pattern used throughout
- `E741`: Ambiguous variable `l` for `low` in `_compute_atr14` and trade simulation loops — rename to `lo` if fixing
- `F841`: Unused variables like `all_test_preds`, `proba_up_1d` — may be dead code or future use

## Code Style Guidelines

### Imports
- Group in order: stdlib, third-party (numpy, pandas, sklearn, lightgbm, matplotlib), local
- Use alphabetical ordering within groups
- Example:
```python
import re
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from holographic_engine import holographic_feature_engine, feature_selection_pipeline
```

### Type Hints
- Use Python 3.10+ union syntax (`str | None` not `Optional[str]`)
- Add type hints for function parameters and return values when unambiguous
- Prefer `pd.DataFrame`, `pd.Series`, `np.ndarray` over generic types
- Example:
```python
def _parse_datetime_series(series: pd.Series) -> pd.Series:
    ...

def walk_forward(df: pd.DataFrame, feature_cols: list, ...) -> dict:
    ...
```

### Naming Conventions
- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private functions**: prefix with `_` (e.g., `_compute_atr14`)
- Feature columns: descriptive with underscores (e.g., `holo_w5_skel_drift`)

### Function Organization
- Private helpers prefixed with `_` at module level
- Functions grouped by purpose with section headers:
```python
# ─────────────────────────────────────────────
# 1. PARSER
# ─────────────────────────────────────────────
```

### Docstrings
- Use triple-quoted strings at module and function level
- Include parameter/return descriptions for complex functions
- Keep brief for simple utilities

### Error Handling
- Use `try/except` blocks sparingly, catching specific exceptions
- Validate inputs early with clear error messages
- Use `np.isfinite()` for numeric validation
- Example:
```python
if not np.isfinite(dist) or dist <= 0:
    continue
```

### Data Validation
- Replace inf/-inf with nan, then fillna(0) for features
- Use `np.nan` for missing numeric values
- Validate DataFrame columns exist before accessing

### NumPy/Pandas Patterns
- Prefer `.values` or `.to_numpy()` for array operations
- Use vectorized operations over loops when possible
- Leverage `np.searchsorted` for time-based lookups
- Example:
```python
opens = df['open'].to_numpy(dtype=float)
highs = df['high'].to_numpy(dtype=float)
```

### Constants
- Define magic numbers as module-level constants
- Use descriptive names with units where applicable
- Group related constants together

### Performance Considerations
- `n_jobs` capped at 4 for LightGBM (CPU-bound system)
- Use `np.errstate` for divide/invalid warnings in tight loops
- Consider `@njit` or vectorization for hot paths

## File Structure

```
/home/km/Universal-ML/
├── universal_ml_engine.py  # Core ML training, walk-forward, prediction
├── holographic_engine.py   # Feature extraction (pure geometry)
├── backtest_engine.py     # Historical simulation engine
├── meta_strategy_selector.py # Strategy selection/verdict
├── run_daily_model.py     # Legacy 1D/1W strategy
├── csv_data/              # Input OHLCV data (user-provided)
├── pyproject.toml        # Authoritative Python project metadata
├── uv.lock               # Locked Python dependency graph for uv
├── requirements.txt      # Legacy dependency mirror, not primary env source
└── *.pkl                 # Trained models (generated)
```

## Key Design Patterns

### Walk-Forward Validation
- Train on past, test on strictly future data
- Inner validation slice for early stopping (never sees test fold)
- Purge gap of 24 bars between train/test to break autocorrelation

### Holographic Features
- Pure geometry: no classical indicators (no RSI, MACD, etc.)
- 4 extraction layers: DNA, Grammar, Spectral, Skeleton
- Multi-timeframe pyramid: 1H, 1D, 1W, 1M
- Feature budget: 40 final features from ~1800 generated

### Data Pipeline
1. Parse TradingView CSV logs (auto-detect symbol/timeframe)
2. Compute ATR14 (labelling only, never a model feature)
3. Extract holographic features
4. Merge higher-TF data via merge_asof
5. Add targets via trade simulation
6. Feature selection (correlation filter → probe ranking → walk-forward)
7. Train final model on all data

### Output Files
- `{symbol}_1H_model.pkl` - Trained 1H LightGBM model
- `{symbol}_1H_features.txt` - 1H feature column names
- `{symbol}_1H_oos_proba.pkl` - 1H out-of-sample probability map
- `{symbol}_1H_trade_plan_models.pkl` - 1H trade plan (SL/TP) models
- `{symbol}_1H_ml_report.png` - 1H forecast/model report
- `{symbol}_1H_backtest_report.png` - 1H equity curve and metrics
- `{symbol}_1D_model.pkl` - Trained 1D LightGBM model
- `{symbol}_1D_features.txt` - 1D feature column names
- `{symbol}_1D_oos_proba.pkl` - 1D out-of-sample probability map
- `{symbol}_1D_trade_plan_models.pkl` - 1D trade plan (SL/TP) models
- `{symbol}_1D_ml_report.png` - 1D forecast/model report
- `{symbol}_1D_backtest_report.png` - 1D equity curve and metrics
