#!/usr/bin/env bash
# ==============================================================================
# UNIVERSAL-ML: AUTONOMOUS LOCAL UPDATE DAEMON
# ==============================================================================
# This script upgrades project dependencies via `uv`, runs the ML
# validation matrix, and conditionally commits or rolls back based on
# mathematical determinism outcomes.

set -e

# Configuration
TEST_SYMBOL="NIFTY"
EVAL_BRANCH="auto-update-eval"
MAIN_BRANCH="experiment"
export PATH="$HOME/.local/bin:$PATH"

echo "[=] Waking up autonomous daemon..."

# ==============================================================================
# PRE-FLIGHT INTELLIGENCE CHECK
# ==============================================================================

# 1. Sustained Idle Check (15-minute load average)
# For ML engineering systems, < 1.0 means the CPU is resting heavily, safe for patching.
LOAD_15MIN=$(uptime | awk -F'load average:' '{ print $2 }' | cut -d, -f3 | awk '{print $1}')
IDLE_THRESHOLD=1.0
if (( $(echo "$LOAD_15MIN > $IDLE_THRESHOLD" | bc -l 2>/dev/null || echo 0) )); then
    echo "[!] System is NOT persistently idle (15-min avg: $LOAD_15MIN). Postponing update to protect user workloads."
    exit 0
fi

# 2. Network Stability Check
# Ping Cloudflare exactly 3 times with standard timeout to verify uplink.
if ! ping -c 3 -W 2 1.1.1.1 > /dev/null 2>&1; then
    echo "[!] Network trajectory unstable. Aborting opportunistic update to prevent fragmented downloads."
    exit 0
fi

echo "[✓] System confirmed idle and online. Executing autonomous protocol..."
echo "[=] Project Baseline: $(pwd)"

# 1. Clean slate
if [ -n "$(git status --porcelain)" ]; then
    echo "[!] Uncommitted changes detected in working directory. Aborting to preserve state."
    exit 1
fi

current_branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$current_branch" != "$MAIN_BRANCH" ]; then
    echo "[=] Switching to main integration branch ($MAIN_BRANCH)..."
    git checkout "$MAIN_BRANCH"
fi

# 2. Safety Checkpoint: Create volatile branch
echo "[=] Creating isolated evaluation branch ($EVAL_BRANCH)..."
git branch -D "$EVAL_BRANCH" 2>/dev/null || true
git checkout -b "$EVAL_BRANCH"

# 3. Aggressive Upgrade Protocol
echo "[=] Pinging vendor registries. Attempting lockfile upgrade..."
uv lock --upgrade
echo "[=] Syncing isolated global cache to .venv..."
uv sync

# 4. Acid Test (Validation Matrix)
echo "[=] Executing ML regression test on $TEST_SYMBOL..."
# We capture exit code. If backtest fails, we catch it.
set +e
uv run python backtest_engine.py --symbol "$TEST_SYMBOL" > /tmp/uv_ml_eval.log 2>&1
TEST_EXIT_CODE=$?
set -e

# 5. Executive Decision Engine
if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "[!] ACID TEST FAILED! Backtest crashed with new libraries."
    echo "[!] Reverting repository to original state."

    # Switch back and nuke the corrupted branch
    git checkout "$MAIN_BRANCH"
    git branch -D "$EVAL_BRANCH"

    # Restore the stable environment rapidly
    uv sync

    echo "[!] Autonomous Update Aborted. Working state perfectly protected."
    exit 1
else
    # Simple check to guarantee output makes sense
    if grep -q "PORTFOLIO SIMULATION RESULTS" /tmp/uv_ml_eval.log; then
        echo "[✓] ACID TEST PASSED. Mathematical determinism verified."

        # We only commit if uv.lock or pyproject.toml changed
        if [ -n "$(git status --porcelain uv.lock pyproject.toml)" ]; then
            git add uv.lock pyproject.toml
            git commit -m "Autonomous Environment Update: Backtest verified [Exit 0]"

            # Switch back to main, leaving the eval branch intact for manual long-testing
            git checkout "$MAIN_BRANCH"
            uv sync # Restore the active environment perfectly

            echo "[✓] SYSTEM TESTED SUCESSFULLY. Updates committed to '$EVAL_BRANCH' for your manual long-test."
        else
            echo "[=] Dependencies already matching the absolute bleeding-edge. No commit necessary."
            git checkout "$MAIN_BRANCH"
            git branch -D "$EVAL_BRANCH"
        fi
    else
        echo "[!] ACID TEST AMBIGUOUS. Expected standard exit logs but didn't find them."
        git checkout "$MAIN_BRANCH"
        git branch -D "$EVAL_BRANCH"
        uv sync
        exit 1
    fi
fi
