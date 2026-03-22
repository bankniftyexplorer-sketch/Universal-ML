# 🛠️ Universal-ML Engine (Manual Operating Guide)
### Surgical-Precision Machine Learning for Price Series Prediction

---

## 📂 1. THE DATA PROTOCOL (Manual Feed)
This engine is designed as a **manual-feed system**. It does NOT have a built-in data downloader. You must manually provide the price data in CSV format from TradingView, MT5, or any other source.

### Setup Requirement:
1.  Create a folder named `csv_data` in the project root.
2.  **Naming is Critical**: The engine scans for files based on their prefix. You must name your files exactly as follows:
    *   `1H_ANYNAME.csv` (Primary - 1 Hour timeframe)
    *   `1D_ANYNAME.csv` (Secondary - 1 Day timeframe)
    *   `1W_ANYNAME.csv` (Secondary - 1 Week timeframe)
    *   `1M_ANYNAME.csv` (Secondary - 1 Month timeframe)

*Note: Replace `ANYNAME` with your symbol (e.g., `1H_BANKNIFTY.csv`).*

---

## ⚡ 2. QUICK START (Install & Run)

### A. Environment Setup
```bash
# 1. Open your terminal in the Universal-ML folder
# 2. Install dependencies
pip install -r requirements.txt
```

### B. The 2-Step Execution Playbook

#### STEP 1: Train & Validate (The Brain)
This script processes your `csv_data` files, engineers features, and trains the model using a leak-proof walk-forward process.
```bash
python universal_ml_engine.py
```
*   **Output**: Saves `{symbol}_ultimate_model.pkl` and a **crucial** `{symbol}_oos_proba.pkl` for the backtester.

#### STEP 2: Backtest & Verify (The Reality Check)
Once the model is trained, use this to see how it would have performed across history.
```bash
python backtest_engine.py
```
*   **Output**: Generates a detailed performance report (`{symbol}_backtest_report.png`) with equity curves and win rates.

---

## ⚙️ 3. ENGINE MECHANICS
*   **Manual CSV Feed**: You control the data. The engine handles the math.
*   **Zero Leakage**: Built with a strict walk-forward timeline. It never looks at "future" bars while training on "past" bars.
*   **Dual-Horizon**: The backtester will only enter a trade if the **Daily (1D)** trend aligns with the **Hourly (1H)** prediction.
*   **Fee Model**: 0.05% per side (adjust in `backtest_engine.py` if your exchange is different).

---

## 📦 4. REQUIREMENTS
The engine relies on these core scientific libraries:
*   `LightGBM`: High-speed gradient boosting.
*   `Pandas`: Data structure management.
*   `Scikit-Learn`: Metrics and preprocessing.
*   `Matplotlib`: Report generation.
*   `Joblib`: Model persistence.
