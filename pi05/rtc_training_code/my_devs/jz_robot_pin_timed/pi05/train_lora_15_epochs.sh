#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BATCH_SIZE="${BATCH_SIZE:-32}" \
FINETUNE_MODE=lora \
LORA_R="${LORA_R:-16}" \
  bash "${SCRIPT_DIR}/train_15_epochs.sh"
