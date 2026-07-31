#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

EPOCHS=1 \
BATCH_SIZE="${BATCH_SIZE:-1}" \
NUM_WORKERS="${NUM_WORKERS:-0}" \
STEPS_OVERRIDE="${STEPS_OVERRIDE:-2}" \
SAVE_CHECKPOINT=false \
LOG_FREQ=1 \
RUN_STAMP="${RUN_STAMP}" \
RUN_NAME="${RUN_NAME:-pi05_jz_pin_smoke_${RUN_STAMP}}" \
  bash "${SCRIPT_DIR}/train_pi05.sh"

