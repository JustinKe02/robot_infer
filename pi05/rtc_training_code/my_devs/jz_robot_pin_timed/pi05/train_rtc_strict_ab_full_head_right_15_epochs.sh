#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONTRACT_MANIFEST="${SCRIPT_DIR}/rtc_strict_ab_full_head_right_15e_d5_seed1000.json"

die() {
  echo "[jz/pi05/strict-ab] ERROR: $*" >&2
  exit 2
}

normalize_bool() {
  local name="$1"
  local value="${2,,}"
  case "${value}" in
    1 | true | yes | on) echo true ;;
    0 | false | no | off | "") echo false ;;
    *) die "${name} must be true or false, got $2" ;;
  esac
}

[[ -z "${STEPS_OVERRIDE:-}" ]] || die "STEPS_OVERRIDE is forbidden by the strict A/B contract"
[[ -f "${CONTRACT_MANIFEST}" ]] || die "training contract is missing: ${CONTRACT_MANIFEST}"

PRINT_CONTRACT_ONLY="$(normalize_bool PRINT_CONTRACT_ONLY "${PRINT_CONTRACT_ONLY:-false}")"
DRY_RUN="$(normalize_bool DRY_RUN "${DRY_RUN:-false}")"
SOURCE_SMOKE_ONLY="$(normalize_bool SOURCE_SMOKE_ONLY "${SOURCE_SMOKE_ONLY:-false}")"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#GPU_IDS[@]} == 1 )) || die "CUDA_VISIBLE_DEVICES must select exactly one GPU"
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "${gpu_id}" =~ ^[0-9]+$ ]] || die "invalid GPU id in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  [[ -z "${SEEN_GPU_IDS[${gpu_id}]:-}" ]] || die "duplicate GPU id in CUDA_VISIBLE_DEVICES"
  SEEN_GPU_IDS[${gpu_id}]=1
done

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
[[ "${RUN_STAMP}" =~ ^[A-Za-z0-9._-]+$ ]] || die "RUN_STAMP contains unsafe characters"

SOURCE_DATASET_NAME="jz_robot_pin_timed_merged_100eps_20260728"
DATASET_NAME="${SOURCE_DATASET_NAME}_pi05_head_right"
TASK_PROMPT="Put the bottle on the right into the basket on the left."
RUN_VARIANT="strict_ab"
SAVE_CHECKPOINT=true
SMOKE_ARGS=()
if [[ "${SOURCE_SMOKE_ONLY}" == "true" ]]; then
  RUN_VARIANT="strict_ab_2step_smoke"
  SAVE_CHECKPOINT=false
  SMOKE_ARGS+=("STEPS_OVERRIDE=2")
fi
RUN_NAME="pi05_jz100_model16_head_right_full_b_rtc_e15_d5_seed1000_${RUN_VARIANT}_${RUN_STAMP}"
SOURCE_DATASET_ROOT="${STRICT_SOURCE_DATASET_ROOT:-${TRAIN_REPO_ROOT}/${SOURCE_DATASET_NAME}}"
DATASET_ROOT="${STRICT_DATASET_ROOT:-${TRAIN_REPO_ROOT}/${DATASET_NAME}}"
PI05_BASE_PATH="${STRICT_PI05_BASE_PATH:-${TRAIN_REPO_ROOT}/assets/modelscope/lerobot/pi05_base}"
PALIGEMMA_TOKENIZER_PATH="${STRICT_TOKENIZER_PATH:-${TRAIN_REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224}"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/${RUN_NAME}"
LOG_DIR="${SCRIPT_DIR}/logs/${RUN_NAME}"

COMMAND=(
  env
  "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "CONDA_ENV=lerobot_flex"
  "CONDA_ROOT=${STRICT_CONDA_ROOT:-/mnt/data/public/miniconda3}"
  "SOURCE_DATASET_NAME=${SOURCE_DATASET_NAME}"
  "SOURCE_DATASET_ROOT=${SOURCE_DATASET_ROOT}"
  "DATASET_NAME=${DATASET_NAME}"
  "DATASET_ROOT=${DATASET_ROOT}"
  "DATASET_REPO_ID=local/${DATASET_NAME}"
  "TASK_PROMPT=${TASK_PROMPT}"
  "PI05_BASE_PATH=${PI05_BASE_PATH}"
  "PALIGEMMA_TOKENIZER_PATH=${PALIGEMMA_TOKENIZER_PATH}"
  "EPOCHS=15"
  "BATCH_SIZE=32"
  "NUM_PROCESSES=1"
  "NUM_WORKERS=4"
  "CHECKPOINT_EVERY_EPOCHS=5"
  "SAVE_CHECKPOINT=${SAVE_CHECKPOINT}"
  "LOG_FREQ=100"
  "NORMALIZATION_MODE=QUANTILES"
  "FINETUNE_MODE=full"
  "TRAINING_MODE=rtc"
  "RTC_MAX_DELAY=5"
  "RTC_MIN_POSTFIX_STEPS=1"
  "OPTIMIZER_LR=2.5e-5"
  "SCHEDULER_DECAY_LR=2.5e-6"
  "CAMERA_MODE=head_right"
  "RUN_NAME=${RUN_NAME}"
  "OUTPUT_DIR=${OUTPUT_DIR}"
  "LOG_DIR=${LOG_DIR}"
  "DRY_RUN=${DRY_RUN}"
  "${SMOKE_ARGS[@]}"
  bash
  "${SCRIPT_DIR}/train_pi05.sh"
)

echo "[jz/pi05/strict-ab] execution=source-training-host-only local_training=forbidden"
echo "[jz/pi05/strict-ab] contract=${CONTRACT_MANIFEST}"
echo "[jz/pi05/strict-ab] dataset=${SOURCE_DATASET_NAME} view=${DATASET_NAME} cameras=head,right"
echo "[jz/pi05/strict-ab] task=${TASK_PROMPT}"
echo "[jz/pi05/strict-ab] finetune=full training=rtc epochs=15 delay=0..5 seed=1000"
echo "[jz/pi05/strict-ab] per_device_batch=32 num_processes=1 effective_batch=32"
echo "[jz/pi05/strict-ab] scheduler_config=warmup1000_decay30000 effective=warmup530_decay15900"
echo "[jz/pi05/strict-ab] source_smoke_only=${SOURCE_SMOKE_ONLY} save_checkpoint=${SAVE_CHECKPOINT}"
echo "[jz/pi05/strict-ab] base=${PI05_BASE_PATH} output=${OUTPUT_DIR}"
printf '[jz/pi05/strict-ab] command='
printf '%q ' "${COMMAND[@]}"
printf '\n'

if [[ "${PRINT_CONTRACT_ONLY}" == "true" ]]; then
  echo "[jz/pi05/strict-ab] PRINT_CONTRACT_ONLY PASS; no data, model, GPU, or trainer access occurred"
  exit 0
fi

exec "${COMMAND[@]}"
