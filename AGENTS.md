# Agent Guidelines for Universal-ML

## Project Overview

Universal-ML is a machine learning trading system that predicts financial instrument direction using holographic (pure geometry-based) features. It processes OHLCV data from TradingView CSV logs across multiple timeframes (1H, 1D, 1W, 1M).

## Build/Lint/Test Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run in mlenv virtual environment (if using)
source mlenv/bin/activate
```

### Running the System
```bash
# Train model and generate walk-forward validation
python universal_ml_engine.py --outdir /path/to/data/

# Run backtest on trained model
python backtest_engine.py --outdir /path/to/data/

# Run meta-strategy selector
python meta_strategy_selector.py --outdir /path/to/data/

# Run daily model (legacy 1D/1W strategy)
python run_daily_model.py --outdir /path/to/data/
```

### Testing
- **No formal test suite exists** - this is a research/production trading system
- Manual validation via backtest_engine.py output (equity curves, Sharpe, Drawdown)
- Verify OOS probability maps are properly generated and loaded

### Linting/Type Checking
```bash
# Run with mlenv activated
source mlenv/bin/activate

# Lint check (26 auto-fixable, some manual)
ruff check .

# Auto-fix safe issues
ruff check --fix .

# Format code
ruff format .

# Full lint + format
ruff check --fix . && ruff format .
```

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
├── requirements.txt      # Dependencies
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
- `{symbol}_ultimate_model.pkl` - Trained LightGBM model
- `{symbol}_ultimate_features.txt` - Feature column names
- `{symbol}_oos_proba.pkl` - Out-of-sample probability map
- `{symbol}_trade_plan_models.pkl` - Trade plan (SL/TP) models
- `{symbol}_backtest_report.png` - Equity curve and metrics
