#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for argument in "$@"; do
  case "${argument}" in
    --auth-token|--auth-token=*)
      opt_die "pass the server token via JZ_PI05_OPT_SERVER_AUTH_TOKEN, not CLI arguments"
      ;;
  esac
done
if (( $# > 0 )); then
  opt_die "run_server.sh accepts configuration through PI05_OPT_*/JZ_PI05_OPT_* environment variables only"
fi

CONFIG_ONLY="$(opt_normalize_bool CONFIG_ONLY "${CONFIG_ONLY:-false}")"
CHECK_POLICY_LOAD="$(opt_normalize_bool CHECK_POLICY_LOAD "${CHECK_POLICY_LOAD:-false}")"
PRINT_ONLY="$(opt_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"
SERVER_HOST="${JZ_PI05_OPT_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${JZ_PI05_OPT_SERVER_PORT:-18088}"
POLICY_DEVICE="${PI05_OPT_DEVICE:-cuda}"
BACKEND="${PI05_OPT_BACKEND:-torch}"
TRITON_ARTIFACT_PATH="${PI05_OPT_TRITON_ARTIFACT_PATH:-}"
TRAJECTORY_PROCESSOR="${PI05_OPT_TRAJECTORY_PROCESSOR:-pass_through}"
MAX_REQUEST_BYTES="${JZ_PI05_OPT_MAX_REQUEST_BYTES:-67108864}"
METRICS_WINDOW_SIZE="${PI05_OPT_METRICS_WINDOW_SIZE:-512}"
TRACE_PATH="${PI05_OPT_TRACE_PATH:-${LOG_ROOT}/server/inference_trace.jsonl}"
TRACE_PATH="$(readlink -m -- "${TRACE_PATH}")"
TRACE_STRICT="$(opt_normalize_bool PI05_OPT_TRACE_STRICT "${PI05_OPT_TRACE_STRICT:-false}")"
TRACE_MAX_BYTES="${PI05_OPT_TRACE_MAX_BYTES:-16777216}"
TRACE_BACKUP_COUNT="${PI05_OPT_TRACE_BACKUP_COUNT:-2}"
TORCH_INFERENCE_MODE="$(opt_normalize_bool PI05_OPT_TORCH_INFERENCE_MODE "${PI05_OPT_TORCH_INFERENCE_MODE:-false}")"
TORCH_BF16_AUTOCAST="$(opt_normalize_bool PI05_OPT_TORCH_BF16_AUTOCAST "${PI05_OPT_TORCH_BF16_AUTOCAST:-false}")"
TORCH_PINNED_MEMORY="$(opt_normalize_bool PI05_OPT_TORCH_PINNED_MEMORY "${PI05_OPT_TORCH_PINNED_MEMORY:-false}")"
TORCH_NON_BLOCKING_COPIES="$(opt_normalize_bool PI05_OPT_TORCH_NON_BLOCKING_COPIES "${PI05_OPT_TORCH_NON_BLOCKING_COPIES:-false}")"
TORCH_STATIC_BUFFERS="$(opt_normalize_bool PI05_OPT_TORCH_STATIC_BUFFERS "${PI05_OPT_TORCH_STATIC_BUFFERS:-false}")"
TORCH_CUDA_GRAPH="$(opt_normalize_bool PI05_OPT_TORCH_CUDA_GRAPH "${PI05_OPT_TORCH_CUDA_GRAPH:-false}")"
TORCH_WARMUP_ITERATIONS="${PI05_OPT_TORCH_WARMUP_ITERATIONS:-0}"
TORCH_WARMUP_SEED="${PI05_OPT_TORCH_WARMUP_SEED:-12345}"
RTC_CONDITIONED_TASK="${PI05_OPT_RTC_CONDITIONED_TASK:-}"
TEMPORAL_SPEED_FACTOR="${PI05_OPT_TEMPORAL_SPEED_FACTOR:-1.0}"
TEMPORAL_MAX_JOINT_STEP_RAD="${PI05_OPT_TEMPORAL_MAX_JOINT_STEP_RAD:-0.02}"
TEMPORAL_SOLVER_TIMEOUT_S="${PI05_OPT_TEMPORAL_SOLVER_TIMEOUT_S:-0.05}"
SERVER_AUTH_TOKEN="${JZ_PI05_OPT_SERVER_AUTH_TOKEN:-}"

case "${SERVER_HOST}" in
  127.0.0.1|localhost|::1)
    ;;
  *)
    [[ -n "${SERVER_AUTH_TOKEN}" ]] || opt_die "non-loopback host requires JZ_PI05_OPT_SERVER_AUTH_TOKEN"
    ;;
esac

opt_prepare_runtime
if [[ "${PRINT_ONLY}" != "true" && "${CONFIG_ONLY}" != "true" ]]; then
  opt_require_policy
  opt_require_tokenizer
fi

COMMAND=(
  "${CONDA_PYTHON}"
  "${PI05_OPT_DIR}/run_policy_server.py"
  "--host=${SERVER_HOST}"
  "--port=${SERVER_PORT}"
  "--backend=${BACKEND}"
  "--trajectory-processor=${TRAJECTORY_PROCESSOR}"
  "--tokenizer-path=${TOKENIZER_PATH}"
  "--device=${POLICY_DEVICE}"
  "--max-request-bytes=${MAX_REQUEST_BYTES}"
  "--metrics-window-size=${METRICS_WINDOW_SIZE}"
  "--trace-path=${TRACE_PATH}"
  "--trace-strict=${TRACE_STRICT}"
  "--trace-max-bytes=${TRACE_MAX_BYTES}"
  "--trace-backup-count=${TRACE_BACKUP_COUNT}"
  "--torch-inference-mode=${TORCH_INFERENCE_MODE}"
  "--torch-bf16-autocast=${TORCH_BF16_AUTOCAST}"
  "--torch-pinned-memory=${TORCH_PINNED_MEMORY}"
  "--torch-non-blocking-copies=${TORCH_NON_BLOCKING_COPIES}"
  "--torch-static-buffers=${TORCH_STATIC_BUFFERS}"
  "--torch-cuda-graph=${TORCH_CUDA_GRAPH}"
  "--torch-warmup-iterations=${TORCH_WARMUP_ITERATIONS}"
  "--torch-warmup-seed=${TORCH_WARMUP_SEED}"
  "--temporal-speed-factor=${TEMPORAL_SPEED_FACTOR}"
  "--temporal-max-joint-step-rad=${TEMPORAL_MAX_JOINT_STEP_RAD}"
  "--temporal-solver-timeout-s=${TEMPORAL_SOLVER_TIMEOUT_S}"
)
if [[ -n "${POLICY_PATH}" ]]; then
  COMMAND+=("--policy-path=${POLICY_PATH}")
fi
if [[ -n "${TRITON_ARTIFACT_PATH}" ]]; then
  COMMAND+=("--triton-artifact-path=${TRITON_ARTIFACT_PATH}")
fi
if [[ -n "${RTC_CONDITIONED_TASK}" ]]; then
  COMMAND+=("--rtc-conditioned-task=${RTC_CONDITIONED_TASK}")
fi
if [[ "${CONFIG_ONLY}" == "true" ]]; then
  COMMAND+=(--config-only)
fi
if [[ "${CHECK_POLICY_LOAD}" == "true" ]]; then
  COMMAND+=(--check-policy-load)
fi

echo "[tk_infer/pi05_optimized] backend=${BACKEND} trajectory_processor=${TRAJECTORY_PROCESSOR}"
echo "[tk_infer/pi05_optimized] listen=http://${SERVER_HOST}:${SERVER_PORT}"
RUNTIME_PHASE=2
if [[ "${BACKEND}" == "triton" ]]; then
  RUNTIME_PHASE=3
elif [[ "${BACKEND}" == "torch_rtc_conditioned" ]]; then
  RUNTIME_PHASE=10
fi
if [[ "${TRAJECTORY_PROCESSOR}" == "paired_temporal" ]]; then
  RUNTIME_PHASE=6
fi
echo "[tk_infer/pi05_optimized] robot_adapter=absent armed_capability=false phase=${RUNTIME_PHASE}"
opt_run_or_print "${COMMAND[@]}"
