#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BATCH_SIZE="${BATCH_SIZE:-32}" \
FINETUNE_MODE=full \
  bash "${SCRIPT_DIR}/train_15_epochs.sh"
