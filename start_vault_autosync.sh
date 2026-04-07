#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/km/Universal-ML"
LOCK_FILE="/tmp/universal_ml_vault_autosync.lock"

cd "$REPO_ROOT"

if command -v flock >/dev/null 2>&1; then
  exec flock -n "$LOCK_FILE" \
    uv run python data_vault/vault_engine.py \
      --auto-sync \
      --pause-seconds 0 \
      --auto-max-cpu-percent 45 \
      --auto-min-download-kbps 32 \
      --auto-check-interval-seconds 300 \
      --auto-min-sync-gap-seconds 3600 \
      "$@"
fi

uv run python data_vault/vault_engine.py \
  --auto-sync \
  --pause-seconds 0 \
  --auto-max-cpu-percent 45 \
  --auto-min-download-kbps 32 \
  --auto-check-interval-seconds 300 \
  --auto-min-sync-gap-seconds 3600 \
  "$@"
