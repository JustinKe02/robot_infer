#!/usr/bin/env bash

# Shared helpers for the optimized PI0.5 server. This file only defines variables and functions when sourced.

PI05_OPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TK_INFER_DIR="$(cd "${PI05_OPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PI05_OPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_flex}"
CONDA_ROOT="${CONDA_ROOT:-/home/luzhuang/miniconda3}"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"

POLICY_PATH="${PI05_OPT_POLICY_PATH:-}"
TOKENIZER_PATH="${PI05_OPT_TOKENIZER_PATH:-${REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224}"
RUNTIME_DIR="${PI05_OPT_RUNTIME_DIR:-${PI05_OPT_DIR}/run_state}"
CACHE_DIR="${PI05_OPT_CACHE_DIR:-${RUNTIME_DIR}/cache}"
LOG_ROOT="${PI05_OPT_LOG_ROOT:-${PI05_OPT_DIR}/logs}"
OUTPUT_ROOT="${PI05_OPT_OUTPUT_ROOT:-${PI05_OPT_DIR}/outputs}"
ARTIFACT_ROOT="${PI05_OPT_ARTIFACT_ROOT:-${PI05_OPT_DIR}/artifacts}"

if [[ -n "${POLICY_PATH}" ]]; then
  POLICY_PATH="$(readlink -m -- "${POLICY_PATH}")"
fi
TOKENIZER_PATH="$(readlink -m -- "${TOKENIZER_PATH}")"
RUNTIME_DIR="$(readlink -m -- "${RUNTIME_DIR}")"
CACHE_DIR="$(readlink -m -- "${CACHE_DIR}")"
LOG_ROOT="$(readlink -m -- "${LOG_ROOT}")"
OUTPUT_ROOT="$(readlink -m -- "${OUTPUT_ROOT}")"
ARTIFACT_ROOT="$(readlink -m -- "${ARTIFACT_ROOT}")"

opt_die() {
  echo "[tk_infer/pi05_optimized] ERROR: $*" >&2
  exit 2
}

opt_normalize_bool() {
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
      opt_die "${name} must be true or false, got ${value}"
      ;;
  esac
}

opt_require_executable() {
  [[ -x "$1" ]] || opt_die "required executable is missing: $1"
}

opt_require_directory() {
  [[ -d "$1" ]] || opt_die "required directory is missing: $1"
}

opt_require_local_write_path() {
  local name="$1"
  local value="$2"
  local resolved
  resolved="$(readlink -m -- "${value}")" || opt_die "cannot resolve ${name}: ${value}"
  case "${resolved}" in
    "${PI05_OPT_DIR}"|"${PI05_OPT_DIR}"/*)
      ;;
    *)
      opt_die "${name} must stay inside ${PI05_OPT_DIR}, got ${resolved}"
      ;;
  esac
}

opt_require_policy() {
  opt_require_directory "${POLICY_PATH}"
  local required
  for required in config.json policy_preprocessor.json policy_postprocessor.json train_config.json; do
    [[ -f "${POLICY_PATH}/${required}" ]] || opt_die "checkpoint is missing ${required}: ${POLICY_PATH}"
  done
  if [[ ! -f "${POLICY_PATH}/model.safetensors" ]]; then
    [[ -f "${POLICY_PATH}/adapter_config.json" ]] \
      || opt_die "checkpoint lacks model.safetensors or adapter_config.json: ${POLICY_PATH}"
    [[ -f "${POLICY_PATH}/adapter_model.safetensors" ]] \
      || opt_die "adapter checkpoint is missing adapter_model.safetensors: ${POLICY_PATH}"
  fi
}

opt_require_tokenizer() {
  opt_require_directory "${TOKENIZER_PATH}"
  local required
  for required in tokenizer.json tokenizer_config.json; do
    [[ -f "${TOKENIZER_PATH}/${required}" ]] || opt_die "tokenizer is missing ${required}: ${TOKENIZER_PATH}"
  done
}

opt_prepare_runtime() {
  opt_require_executable "${CONDA_PYTHON}"
  opt_require_local_write_path RUNTIME_DIR "${RUNTIME_DIR}"
  opt_require_local_write_path CACHE_DIR "${CACHE_DIR}"
  opt_require_local_write_path LOG_ROOT "${LOG_ROOT}"
  opt_require_local_write_path OUTPUT_ROOT "${OUTPUT_ROOT}"
  opt_require_local_write_path ARTIFACT_ROOT "${ARTIFACT_ROOT}"
  mkdir -p \
    "${CACHE_DIR}/huggingface" \
    "${CACHE_DIR}/torch" \
    "${RUNTIME_DIR}/tmp" \
    "${LOG_ROOT}/server" \
    "${OUTPUT_ROOT}/server" \
    "${ARTIFACT_ROOT}"

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

opt_print_command() {
  printf '[tk_infer/pi05_optimized] command='
  printf '%q ' "$@"
  printf '\n'
}

opt_run_or_print() {
  local -a command=("$@")
  local print_only
  print_only="$(opt_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"
  opt_print_command "${command[@]}"
  if [[ "${print_only}" == "true" ]]; then
    echo "[tk_infer/pi05_optimized] PRINT_COMMAND_ONLY=true; nothing was started"
    return 0
  fi
  local stamp log_path
  stamp="$(date +%Y%m%d_%H%M%S)"
  log_path="${LOG_ROOT}/server/server_${stamp}.log"
  echo "[tk_infer/pi05_optimized] log=${log_path}"
  (
    cd "${RUNTIME_DIR}"
    "${command[@]}"
  ) 2>&1 | tee "${log_path}"
}
