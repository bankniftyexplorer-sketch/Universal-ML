#!/usr/bin/env bash

# SOTA Overnight Volatility Trainer
# Decouples heavy training from the dashboard
# Register with: crontab -e
# Example: 0 2 * * * cd /home/km/Universal-ML && ./cron_trainer.sh >> /home/km/Universal-ML/cron.log 2>&1

cd "$(dirname "$0")"

for symbol in NIFTY BANKNIFTY SENSEX; do
    echo "[$(date -u)] Starting overnight pipeline for $symbol"
    uv run python data_vault/yfinance_vault.py --symbol "$symbol"
    uv run python daily_volatility_engine.py --symbol "$symbol"
    echo "[$(date -u)] Completed overnight pipeline for $symbol"
done
