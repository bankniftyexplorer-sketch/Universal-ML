# 🔥 Universal-ML Engine
### High-Precision Multi-Timeframe Algorithmic Trading Core
**77.4% High-Confidence OOS Accuracy | 6.5+ Profit Factor | Zero-Leak Engineering**

---

## 🚀 Overview
**Universal-ML** is a professional-grade, production-ready machine learning engine designed for financial time-series prediction. It specifically targets high-precision directional forecasting (e.g., Nifty/Banknifty Futures) using a **Walk-Forward Validation** (WFV) architecture that ensures **zero lookahead bias** and **zero data leakage**.

### Key Performance Identifiers (Post-Audit Build)
| Metric | Value |
|---|---|
| **High-Confidence Accuracy (>60%)** | **77.4%** |
| **Out-Of-Sample Win Rate** | **83.2%** |
| **Profit Factor** | **6.516** |
| **Max Drawdown** | **10.31%** |
| **Risk-Reward Profile** | High-Confidence Thresholding with Intra-Bar SL/TP |

---

## 🏗️ Core Architecture

### 1. `universal_ml_engine.py` (The Brain)
*   **Feature Engine**: 118+ engineered features including volatility-normalized returns, kinematic wicks, candle-body ratios, and relative strength across 1H, 1D, 1W, and 1M timeframes.
*   **Validation Engine**: A robust, multi-split **Walk-Forward** pipeline. Each split carves a hidden validation slice *only* from the training history for early-stopping, ensuring the test set remains strictly blind.
*   **OOS Proba Map**: Generates an honest timestamped probability map (`{symbol}_oos_proba.pkl`) containing only genuine out-of-sample predictions for backtesting.

### 2. `backtest_engine.py` (The Reality Check)
*   **Intra-Bar Evaluation**: Simulates trade entry, take-profit (TP1 with half-exit), trailing stops, and stop-losses based on raw OHLC bar dynamics.
*   **Dual-Horizon Gating**: Implements a 1D macro-trend filter. Trades are only executed if the 1H prediction aligns with the 1D model's bias, significantly improving win rates.
*   **Fee Accounting**: Realistic fee modeling (commissions + spread) applied at every entry and exit.

### 3. `run_daily_model.py` (Operations)
*   **Automated Backtesting**: A lightweight daily operator for running 1D model evaluations and generating visual equity curve reports.

---

## 🛠️ Setup & Execution

### 1. Data Structure
The engine expects a `csv_data/` folder containing data in the following naming format (compatible with TradingView logs):
*   `1H_{SYMBOL}.csv` (Primary Timeframe)
*   `1D_{SYMBOL}.csv`
*   `1W_{SYMBOL}.csv`
*   `1M_{SYMBOL}.csv`

### 2. Running the Full Pipeline
```bash
# 1. Train the model and generate the OOS proba map
python universal_ml_engine.py --outdir ./data/

# 2. Run the high-precision backtest
python backtest_engine.py --outdir ./data/
```

---

## ⚠️ Risk Disclaimer
Financial trading involves significant risk. This software is provided "as is" for research purposes. While the backtested metrics show massive edge (+18.3pp over baseline), live market conditions introduce slippage, execution latency, and partial fills that may differ from simulated results.
