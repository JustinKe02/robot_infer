#!/usr/bin/env bash

# Shared, side-effect-limited launcher helpers for the JZ PI0.5 RTC runtime.
# This file is sourced by the entry scripts; do not execute it directly.

RTC_INFER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_DIR="$(cd "${RTC_INFER_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RTC_INFER_DIR}/../../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_ROOT="${CONDA_ROOT:-/home/cqy/miniconda3}"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"

# This is the 15-epoch run started for the curated 42-episode PI0.5 dataset.
# Override POLICY_PATH when selecting another completed pretrained_model directory.
PI05_15E_RUN_NAME="${PI05_15E_RUN_NAME:-pi05_jz_robot_pin_timed_curated_42eps_20260713_e15_b8_20260714_202432}"
DEFAULT_POLICY_PATH="${PI05_DIR}/outputs/${PI05_15E_RUN_NAME}/checkpoints/last/pretrained_model"
POLICY_PATH="${POLICY_PATH:-${DEFAULT_POLICY_PATH}}"

# Keep the original model asset read-only. The server passes this absolute path
# as a processor override; clients do not need the tokenizer at all.
TOKENIZER_PATH="${TOKENIZER_PATH:-/data/cqy_workspace/flexible_lerobot/assets/modelscope/google/paligemma-3b-pt-224}"

RUNTIME_DIR="${RUNTIME_DIR:-${RTC_INFER_DIR}/runtime}"
CACHE_DIR="${CACHE_DIR:-${RUNTIME_DIR}/cache}"
LOG_ROOT="${LOG_ROOT:-${RTC_INFER_DIR}/logs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RTC_INFER_DIR}/outputs}"

rtc_die() {
  echo "[jz/pi05/rtc_infer] ERROR: $*" >&2
  exit 2
}

rtc_normalize_bool() {
  local name="$1"
  local value="${2:-false}"
  case "${value,,}" in
    1|true|yes|y|on)
      printf 'true\n'
      ;;
    0|false|no|n|off|'')
      printf 'false\n'
      ;;
    *)
      rtc_die "${name} must be true or false, got ${value}"
      ;;
  esac
}

rtc_require_choice() {
  local name="$1"
  local value="$2"
  shift 2
  local candidate
  for candidate in "$@"; do
    if [[ "${value}" == "${candidate}" ]]; then
      return 0
    fi
  done
  rtc_die "${name} must be one of [$*], got ${value}"
}

rtc_require_executable() {
  [[ -x "$1" ]] || rtc_die "required executable is missing: $1"
}

rtc_require_directory() {
  [[ -d "$1" ]] || rtc_die "required directory is missing: $1"
}

rtc_require_local_write_path() {
  local name="$1"
  local value="$2"
  local resolved
  resolved="$(readlink -m -- "${value}")" \
    || rtc_die "cannot resolve ${name}: ${value}"
  case "${resolved}" in
    "${RTC_INFER_DIR}"|"${RTC_INFER_DIR}"/*)
      ;;
    *)
      rtc_die "${name} must stay inside ${RTC_INFER_DIR}, got ${resolved}"
      ;;
  esac
}

rtc_require_policy() {
  rtc_require_directory "${POLICY_PATH}"
  local required
  for required in \
    config.json \
    policy_preprocessor.json \
    policy_postprocessor.json \
    train_config.json; do
    [[ -f "${POLICY_PATH}/${required}" ]] \
      || rtc_die "checkpoint is missing ${required}: ${POLICY_PATH}"
  done
  if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
    [[ -f "${POLICY_PATH}/adapter_config.json" ]] \
      || rtc_die "checkpoint lacks model.safetensors or adapter_config.json: ${POLICY_PATH}"
    [[ -f "${POLICY_PATH}/adapter_model.safetensors" ]] \
      || rtc_die "adapter checkpoint is missing adapter_model.safetensors: ${POLICY_PATH}"
  fi
}

rtc_require_tokenizer() {
  rtc_require_directory "${TOKENIZER_PATH}"
  local required
  for required in tokenizer.json tokenizer_config.json; do
    [[ -f "${TOKENIZER_PATH}/${required}" ]] \
      || rtc_die "tokenizer is missing ${required}: ${TOKENIZER_PATH}"
  done
}

rtc_prepare_runtime() {
  rtc_require_executable "${CONDA_PYTHON}"
  rtc_require_local_write_path RUNTIME_DIR "${RUNTIME_DIR}"
  rtc_require_local_write_path CACHE_DIR "${CACHE_DIR}"
  rtc_require_local_write_path LOG_ROOT "${LOG_ROOT}"
  rtc_require_local_write_path OUTPUT_ROOT "${OUTPUT_ROOT}"

  mkdir -p \
    "${CACHE_DIR}/huggingface" \
    "${CACHE_DIR}/torch" \
    "${RUNTIME_DIR}/tmp" \
    "${LOG_ROOT}/server" \
    "${LOG_ROOT}/client" \
    "${OUTPUT_ROOT}/server" \
    "${OUTPUT_ROOT}/client"

  export HF_HOME="${CACHE_DIR}/huggingface"
  export TORCH_HOME="${CACHE_DIR}/torch"
  export XDG_CACHE_HOME="${CACHE_DIR}"
  export TMPDIR="${RUNTIME_DIR}/tmp"
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export TOKENIZERS_PARALLELISM=false
  export WANDB_MODE=disabled
  export PYTHONDONTWRITEBYTECODE=1
  export PYTHONUNBUFFERED=1
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
}

rtc_require_armed_confirmation() {
  [[ "${JZ_ROBOT_PIN_ARMED:-}" == "1" ]] \
    || rtc_die "armed inference requires JZ_ROBOT_PIN_ARMED=1"
  [[ "${I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT:-}" == "1" ]] \
    || rtc_die "armed inference requires I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1"
  [[ "${JZ_POLICY_INFERENCE_ARMED:-}" == "1" ]] \
    || rtc_die "armed inference requires JZ_POLICY_INFERENCE_ARMED=1"
}

rtc_print_command() {
  printf '[jz/pi05/rtc_infer] command='
  printf '%q ' "$@"
  printf '\n'
}

rtc_run_or_print() {
  local component="$1"
  shift
  local -a command=("$@")
  local print_only
  print_only="$(rtc_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"

  rtc_print_command "${command[@]}"
  if [[ "${print_only}" == "true" ]]; then
    echo "[jz/pi05/rtc_infer] PRINT_COMMAND_ONLY=true; nothing was started"
    return 0
  fi

  local stamp log_path
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_path="${LOG_ROOT}/${component}/${component}_${stamp}.log"
  echo "[jz/pi05/rtc_infer] log=${log_path}"
  (
    cd "${RUNTIME_DIR}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
}
