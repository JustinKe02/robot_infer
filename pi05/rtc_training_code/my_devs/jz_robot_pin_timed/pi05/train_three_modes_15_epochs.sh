#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-false}"
FULL_BATCH_SIZE="${FULL_BATCH_SIZE:-32}"
EXPERT_BATCH_SIZE="${EXPERT_BATCH_SIZE:-${ACTION_HEAD_BATCH_SIZE:-32}}"
LORA_BATCH_SIZE="${LORA_BATCH_SIZE:-32}"
LORA_R="${LORA_R:-16}"

if [[ -n "${RUN_NAME:-}" || -n "${OUTPUT_DIR:-}" ]]; then
  echo "[jz/pi05/three-modes] RUN_NAME and OUTPUT_DIR must be unset for a three-mode run" >&2
  exit 2
fi

run_training() {
  local label="$1"
  local batch_size="$2"
  local script="$3"

  echo "[jz/pi05/three-modes] START mode=${label} batch_size=${batch_size} run_stamp=${RUN_STAMP}"
  BATCH_SIZE="${batch_size}" \
  RUN_STAMP="${RUN_STAMP}" \
  DRY_RUN="${DRY_RUN}" \
  LORA_R="${LORA_R}" \
    bash "${script}"
  echo "[jz/pi05/three-modes] PASS mode=${label}"
}

run_training full "${FULL_BATCH_SIZE}" "${SCRIPT_DIR}/train_full_15_epochs.sh"
run_training expert "${EXPERT_BATCH_SIZE}" "${SCRIPT_DIR}/train_expert_15_epochs.sh"
run_training lora "${LORA_BATCH_SIZE}" "${SCRIPT_DIR}/train_lora_15_epochs.sh"

echo "[jz/pi05/three-modes] ALL PASS run_stamp=${RUN_STAMP}"
