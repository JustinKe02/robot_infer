#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCHS=15 \
BATCH_SIZE="${BATCH_SIZE:-32}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
FINETUNE_MODE="${FINETUNE_MODE:-expert}" \
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-5}" \
SAVE_CHECKPOINT=true \
LOG_FREQ="${LOG_FREQ:-100}" \
  bash "${SCRIPT_DIR}/train_pi05.sh"
