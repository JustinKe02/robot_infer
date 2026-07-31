#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/data/public/miniconda3}"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/lerobot-train}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/accelerate}"

SOURCE_DATASET_NAME="${SOURCE_DATASET_NAME:-jz_robot_pin_timed_merged_100eps_20260728}"
SOURCE_DATASET_ROOT="${SOURCE_DATASET_ROOT:-${REPO_ROOT}/${SOURCE_DATASET_NAME}}"
CAMERA_MODE="${CAMERA_MODE:-head_right}"
CAMERA_HEAD="observation.images.camera_head"
CAMERA_LEFT="observation.images.camera_left"
CAMERA_RIGHT="observation.images.camera_right"
case "${CAMERA_MODE}" in
  head_right)
    DEFAULT_DATASET_NAME="${SOURCE_DATASET_NAME}_pi05_head_right"
    CAMERA_KEYS=("${CAMERA_HEAD}" "${CAMERA_RIGHT}")
    CAMERA_LABEL="head,right"
    ;;
  three | all | head_left_right)
    DEFAULT_DATASET_NAME="${SOURCE_DATASET_NAME}_pi05_head_left_right"
    CAMERA_KEYS=("${CAMERA_HEAD}" "${CAMERA_LEFT}" "${CAMERA_RIGHT}")
    CAMERA_LABEL="head,left,right"
    ;;
  *)
    echo "[jz/pi05/train] CAMERA_MODE must be head_right or three; got ${CAMERA_MODE}" >&2
    exit 2
    ;;
esac
CAMERA_ARGS=()
for camera_key in "${CAMERA_KEYS[@]}"; do
  CAMERA_ARGS+=("--camera-key" "${camera_key}")
done

DATASET_NAME="${DATASET_NAME:-${DEFAULT_DATASET_NAME}}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/${DATASET_NAME}}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/${DATASET_NAME}}"
TRAINING_SCHEMA="${TRAINING_SCHEMA:-${DATASET_ROOT}/meta/jz_pin_training_schema.json}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-100}"
EXPECTED_FRAMES="${EXPECTED_FRAMES:-33898}"
EXPECTED_FPS="${EXPECTED_FPS:-20}"
TASK_PROMPT="${TASK_PROMPT:-Put the bottle on the right into the basket on the left.}"

PI05_BASE_PATH="${PI05_BASE_PATH:-${REPO_ROOT}/assets/modelscope/lerobot/pi05_base}"
PALIGEMMA_TOKENIZER_PATH="${PALIGEMMA_TOKENIZER_PATH:-${REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224}"

EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
GPU_MONITOR_INTERVAL_S="${GPU_MONITOR_INTERVAL_S:-30}"
CHECKPOINT_EVERY_EPOCHS="${CHECKPOINT_EVERY_EPOCHS:-5}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
NORMALIZATION_MODE="${NORMALIZATION_MODE:-QUANTILES}"
FINETUNE_MODE="${FINETUNE_MODE:-expert_only}"
TRAINING_MODE="${TRAINING_MODE:-standard}"
RTC_MAX_DELAY="${RTC_MAX_DELAY:-10}"
RTC_MIN_POSTFIX_STEPS="${RTC_MIN_POSTFIX_STEPS:-1}"
LORA_R="${LORA_R:-16}"
DRY_RUN="${DRY_RUN:-false}"

case "${FINETUNE_MODE}" in
  expert | action_head | expert_only)
    TRAIN_VARIANT="expert"
    OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"
    SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
    FINETUNE_ARGS=("--policy.train_expert_only=true")
    ;;
  lora)
    TRAIN_VARIANT="lora"
    OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
    SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-5}"
    FINETUNE_ARGS=("--peft.method_type=LORA" "--peft.r=${LORA_R}")
    ;;
  full)
    TRAIN_VARIANT="full"
    OPTIMIZER_LR="${OPTIMIZER_LR:-2.5e-5}"
    SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
    FINETUNE_ARGS=()
    ;;
  *)
    echo "[jz/pi05/train] FINETUNE_MODE must be full, expert, or lora; got ${FINETUNE_MODE}" >&2
    exit 2
    ;;
esac

case "${TRAINING_MODE}" in
  standard)
    METHOD_VARIANT="${TRAIN_VARIANT}"
    RTC_TRAINING_ARGS=()
    ;;
  rtc)
    METHOD_VARIANT="${TRAIN_VARIANT}_rtc"
    RTC_TRAINING_ARGS=(
      "--policy.rtc_training.enabled=true"
      "--policy.rtc_training.max_delay=${RTC_MAX_DELAY}"
      "--policy.rtc_training.min_postfix_steps=${RTC_MIN_POSTFIX_STEPS}"
    )
    ;;
  *)
    echo "[jz/pi05/train] TRAINING_MODE must be standard or rtc; got ${TRAINING_MODE}" >&2
    exit 2
    ;;
esac

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-pi05_${DATASET_NAME}_${METHOD_VARIANT}_e${EPOCHS}_b${BATCH_SIZE}_${RUN_STAMP}}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs/${RUN_NAME}}"
RUNTIME_DIR="${RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"
MODEL16_STATS="${MODEL16_STATS:-${RUNTIME_DIR}/${DATASET_NAME}_model16_stats.json}"

for value_name in EPOCHS BATCH_SIZE NUM_PROCESSES GPU_MONITOR_INTERVAL_S CHECKPOINT_EVERY_EPOCHS LOG_FREQ EXPECTED_EPISODES \
  EXPECTED_FRAMES EXPECTED_FPS; do
  value="${!value_name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[jz/pi05/train] ${value_name} must be a positive integer, got ${value}" >&2
    exit 2
  fi
done
if (( NUM_PROCESSES > 1 )) && [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "[jz/pi05/train] accelerate executable is missing: ${ACCELERATE_BIN}" >&2
  exit 2
fi
if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]]; then
  echo "[jz/pi05/train] NUM_WORKERS must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${LORA_R}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[jz/pi05/train] LORA_R must be a positive integer" >&2
  exit 2
fi
if ! [[ "${RTC_MAX_DELAY}" =~ ^[0-9]+$ ]]; then
  echo "[jz/pi05/train] RTC_MAX_DELAY must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "${RTC_MIN_POSTFIX_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[jz/pi05/train] RTC_MIN_POSTFIX_STEPS must be a positive integer" >&2
  exit 2
fi
if (( RTC_MIN_POSTFIX_STEPS > 50 )); then
  echo "[jz/pi05/train] RTC_MIN_POSTFIX_STEPS cannot exceed the 50-step action chunk" >&2
  exit 2
fi
if [[ -n "${STEPS_OVERRIDE:-}" ]] && ! [[ "${STEPS_OVERRIDE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[jz/pi05/train] STEPS_OVERRIDE must be a positive integer" >&2
  exit 2
fi
if [[ "${SAVE_CHECKPOINT}" != "true" && "${SAVE_CHECKPOINT}" != "false" ]]; then
  echo "[jz/pi05/train] SAVE_CHECKPOINT must be true or false" >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "true" && "${DRY_RUN}" != "false" ]]; then
  echo "[jz/pi05/train] DRY_RUN must be true or false" >&2
  exit 2
fi
if [[ "${NORMALIZATION_MODE}" != "QUANTILES" && "${NORMALIZATION_MODE}" != "MEAN_STD" ]]; then
  echo "[jz/pi05/train] NORMALIZATION_MODE must be QUANTILES or MEAN_STD" >&2
  exit 2
fi

for required_file in \
  "${CONDA_PYTHON}" \
  "${LEROBOT_TRAIN_BIN}" \
  "${SCRIPT_DIR}/prepare_training_view.py" \
  "${SOURCE_DATASET_ROOT}/meta/info.json" \
  "${SOURCE_DATASET_ROOT}/meta/stats.json" \
  "${SOURCE_DATASET_ROOT}/meta/tasks.parquet" \
  "${SOURCE_DATASET_ROOT}/meta/jz_pin_training_schema.json" \
  "${PI05_BASE_PATH}/model.safetensors" \
  "${PALIGEMMA_TOKENIZER_PATH}/tokenizer.json"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[jz/pi05/train] required file is missing: ${required_file}" >&2
    exit 2
  fi
done

"${CONDA_PYTHON}" "${SCRIPT_DIR}/prepare_training_view.py" \
  --source-root "${SOURCE_DATASET_ROOT}" \
  --output-root "${DATASET_ROOT}" \
  --task "${TASK_PROMPT}" \
  "${CAMERA_ARGS[@]}"

for required_file in "${DATASET_ROOT}/meta/info.json" "${TRAINING_SCHEMA}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[jz/pi05/train] derived dataset file is missing: ${required_file}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "[jz/pi05/train] output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

read -r TOTAL_EPISODES TOTAL_FRAMES FPS RAW_ACTION_DIM RAW_STATE_DIM < <(
  "${CONDA_PYTHON}" - "${DATASET_ROOT}/meta/info.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    info = json.load(stream)
features = info["features"]
print(
    info["total_episodes"],
    info["total_frames"],
    info["fps"],
    features["action"]["shape"][0],
    features["observation.state"]["shape"][0],
)
PY
)
if [[ "${TOTAL_EPISODES}" != "${EXPECTED_EPISODES}" || "${TOTAL_FRAMES}" != "${EXPECTED_FRAMES}" \
  || "${FPS}" != "${EXPECTED_FPS}" ]]; then
  echo "[jz/pi05/train] expected ${EXPECTED_EPISODES}eps/${EXPECTED_FRAMES}frames/${EXPECTED_FPS}fps, got ${TOTAL_EPISODES}/${TOTAL_FRAMES}/${FPS}" >&2
  exit 2
fi
if [[ "${RAW_ACTION_DIM}" != "18" || "${RAW_STATE_DIM}" != "18" ]]; then
  echo "[jz/pi05/train] expected raw18 action/state, got ${RAW_ACTION_DIM}/${RAW_STATE_DIM}" >&2
  exit 2
fi

EFFECTIVE_BATCH_SIZE=$(( BATCH_SIZE * NUM_PROCESSES ))
STEPS_PER_EPOCH=$(( (TOTAL_FRAMES + EFFECTIVE_BATCH_SIZE - 1) / EFFECTIVE_BATCH_SIZE ))
if [[ -n "${STEPS_OVERRIDE:-}" ]]; then
  STEPS="${STEPS_OVERRIDE}"
else
  STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
fi
SAVE_FREQ=$(( STEPS_PER_EPOCH * CHECKPOINT_EVERY_EPOCHS ))
if (( SAVE_FREQ > STEPS )); then
  SAVE_FREQ="${STEPS}"
fi

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}/google" "${RUNTIME_DIR}/cache/huggingface" \
  "${RUNTIME_DIR}/cache/torch" "$(dirname "${OUTPUT_DIR}")"
ln -sfn "${PALIGEMMA_TOKENIZER_PATH}" "${RUNTIME_DIR}/google/paligemma-3b-pt-224"

export HF_HOME="${RUNTIME_DIR}/cache/huggingface"
export TORCH_HOME="${RUNTIME_DIR}/cache/torch"
export XDG_CACHE_HOME="${RUNTIME_DIR}/cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

BASE_PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHONPATH="${BASE_PYTHONPATH}" "${CONDA_PYTHON}" "${SCRIPT_DIR}/compute_model16_stats.py" \
  --dataset-root "${DATASET_ROOT}" \
  --schema "${TRAINING_SCHEMA}" \
  --output "${MODEL16_STATS}"

PYTHONPATH="${BASE_PYTHONPATH}" "${CONDA_PYTHON}" "${SCRIPT_DIR}/preflight.py" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --schema "${TRAINING_SCHEMA}" \
  --stats "${MODEL16_STATS}" \
  --pi05-base "${PI05_BASE_PATH}" \
  --tokenizer "${PALIGEMMA_TOKENIZER_PATH}" \
  --expected-episodes "${EXPECTED_EPISODES}" \
  --expected-frames "${EXPECTED_FRAMES}" \
  --expected-fps "${EXPECTED_FPS}" \
  --expected-task "${TASK_PROMPT}" \
  "${CAMERA_ARGS[@]}"

NORMALIZATION_MAPPING="{\"ACTION\":\"${NORMALIZATION_MODE}\",\"STATE\":\"${NORMALIZATION_MODE}\",\"VISUAL\":\"IDENTITY\"}"
TRAIN_COMMAND=(
  "${LEROBOT_TRAIN_BIN}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.video_backend=pyav"
  "--dataset.image_transforms.enable=false"
  "--policy.type=pi05"
  "--policy.pretrained_path=${PI05_BASE_PATH}"
  "--policy.compile_model=false"
  "--policy.gradient_checkpointing=true"
  "--policy.dtype=bfloat16"
  "--policy.device=cuda"
  "--policy.normalization_mapping=${NORMALIZATION_MAPPING}"
  "--policy.optimizer_lr=${OPTIMIZER_LR}"
  "--policy.scheduler_decay_lr=${SCHEDULER_DECAY_LR}"
  "--policy.push_to_hub=false"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_NAME}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--save_checkpoint=${SAVE_CHECKPOINT}"
  "--save_freq=${SAVE_FREQ}"
  "--log_freq=${LOG_FREQ}"
  "--eval_freq=0"
  "--num_workers=${NUM_WORKERS}"
  "--seed=1000"
  "--wandb.enable=false"
  "${FINETUNE_ARGS[@]}"
  "${RTC_TRAINING_ARGS[@]}"
)
if (( NUM_PROCESSES > 1 )); then
  TRAIN_COMMAND=(
    "${ACCELERATE_BIN}"
    launch
    --multi_gpu
    --num_machines=1
    "--num_processes=${NUM_PROCESSES}"
    "${TRAIN_COMMAND[@]}"
  )
fi

{
  echo "[jz/pi05/train] conda_env=${CONDA_ENV} python=${CONDA_PYTHON}"
  echo "[jz/pi05/train] source_dataset=${SOURCE_DATASET_ROOT}"
  echo "[jz/pi05/train] dataset=${DATASET_ROOT} episodes=${TOTAL_EPISODES} frames=${TOTAL_FRAMES} fps=${FPS}"
  echo "[jz/pi05/train] task=${TASK_PROMPT}"
  echo "[jz/pi05/train] data_policy=all_100_episodes episode_93_included=true old_200_mixed=false"
  echo "[jz/pi05/train] boundary=raw18->model16 camera_mode=${CAMERA_MODE} \
cameras=${CAMERA_LABEL} normalization=${NORMALIZATION_MODE}"
  echo "[jz/pi05/train] finetune_mode=${TRAIN_VARIANT} lora_r=${LORA_R} lr=${OPTIMIZER_LR} decay_lr=${SCHEDULER_DECAY_LR}"
  echo "[jz/pi05/train] training_mode=${TRAINING_MODE} rtc_max_delay=${RTC_MAX_DELAY}" \
    "rtc_min_postfix_steps=${RTC_MIN_POSTFIX_STEPS}"
  echo "[jz/pi05/train] pretrained=${PI05_BASE_PATH} tokenizer=${PALIGEMMA_TOKENIZER_PATH}"
  echo "[jz/pi05/train] per_device_batch=${BATCH_SIZE} num_processes=${NUM_PROCESSES} effective_batch=${EFFECTIVE_BATCH_SIZE} steps_per_epoch=${STEPS_PER_EPOCH} epochs=${EPOCHS} steps=${STEPS}"
  echo "[jz/pi05/train] output=${OUTPUT_DIR} save_checkpoint=${SAVE_CHECKPOINT}"
  printf '[jz/pi05/train] command='
  printf '%q ' "${TRAIN_COMMAND[@]}"
  printf '\n'
} | tee "${LOG_DIR}/launch.log"

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[jz/pi05/train] DRY_RUN PASS; training command was not started" | tee "${LOG_DIR}/DRY_RUN_SUCCESS"
  exit 0
fi

GPU_MONITOR_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  (
    while true; do
      date '+%F %T'
      nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits
      sleep "${GPU_MONITOR_INTERVAL_S}"
    done
  ) > "${LOG_DIR}/gpu_usage.log" 2>&1 &
  GPU_MONITOR_PID=$!
fi
cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${RUNTIME_DIR}"
PYTHONPATH="${SCRIPT_DIR}/cli_hook:${SCRIPT_DIR}:${BASE_PYTHONPATH}" \
JZ_PI05_ENABLE_TRAIN_HOOK=1 \
JZ_PI05_TRAINING_SCHEMA="${TRAINING_SCHEMA}" \
JZ_PI05_MODEL16_STATS="${MODEL16_STATS}" \
  "${TRAIN_COMMAND[@]}" 2>&1 | tee "${LOG_DIR}/train_terminal.log"

mkdir -p "${OUTPUT_DIR}"
printf 'status=PASS\nrun_name=%s\noutput_dir=%s\nsteps=%s\nper_device_batch_size=%s\nnum_processes=%s\neffective_batch_size=%s\ntraining_mode=%s\nrtc_max_delay=%s\nrtc_min_postfix_steps=%s\n' \
  "${RUN_NAME}" "${OUTPUT_DIR}" "${STEPS}" "${BATCH_SIZE}" "${NUM_PROCESSES}" \
  "${EFFECTIVE_BATCH_SIZE}" "${TRAINING_MODE}" "${RTC_MAX_DELAY}" \
  "${RTC_MIN_POSTFIX_STEPS}" > "${LOG_DIR}/SUCCESS"
cp "${LOG_DIR}/SUCCESS" "${OUTPUT_DIR}/TRAINING_SUCCESS"
echo "[jz/pi05/train] PASS output=${OUTPUT_DIR} log=${LOG_DIR}/train_terminal.log"
