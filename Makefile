# Universal-ML Makefile Task Runner
#
# A convenience wrapper for `uv run python` CLI commands.

.PHONY: sync format lint check clean train-1H train-1D test-accuracy live-1H

## Environment & Quality Control
sync:
	uv sync --locked

format:
	uv run ruff format .

lint:
	uv run ruff check .

check:
	uv run python -m py_compile universal_ml_engine.py daily_ml_engine.py backtest_engine.py daily_backtest_engine.py live_inference.py meta_strategy_selector.py accuracy_guardrail.py julia_bridge.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf verification/

## ML Training Operations (Requires populated database)
train-1H:
	uv run python universal_ml_engine.py --symbol NIFTY --outdir ./

train-1D:
	uv run python daily_ml_engine.py --symbol NIFTY --outdir ./

## Live Inference Paths
live-1H:
	uv run python live_inference.py --symbol NIFTY --outdir ./

## Institutional Guardrails
validate-pre:
	uv run python accuracy_guardrail.py capture --symbol NIFTY --outdir ./

validate-post:
	uv run python accuracy_guardrail.py compare --symbol NIFTY --outdir ./
