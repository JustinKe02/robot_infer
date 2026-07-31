#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

for argument in "$@"; do
  case "${argument}" in
    --auth-token|--auth-token=*)
      rtc_die "pass the server token via SERVER_AUTH_TOKEN/JZ_PI05_SERVER_AUTH_TOKEN, not CLI arguments"
      ;;
  esac
done

CONFIG_ONLY="$(rtc_normalize_bool CONFIG_ONLY "${CONFIG_ONLY:-false}")"
CHECK_POLICY_LOAD="$(rtc_normalize_bool CHECK_POLICY_LOAD "${CHECK_POLICY_LOAD:-false}")"
PRINT_ONLY="$(rtc_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8088}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
MAX_REQUEST_BYTES="${MAX_REQUEST_BYTES:-67108864}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-LINEAR}"
RTC_DEBUG="$(rtc_normalize_bool RTC_DEBUG "${RTC_DEBUG:-false}")"
REQUIRE_COMPLETE_STEP="$(rtc_normalize_bool REQUIRE_COMPLETE_STEP "${REQUIRE_COMPLETE_STEP:-true}")"
SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN:-${JZ_PI05_SERVER_AUTH_TOKEN:-}}"

case "${SERVER_HOST}" in
  127.0.0.1|localhost|::1)
    ;;
  *)
    [[ -n "${SERVER_AUTH_TOKEN}" ]] \
      || rtc_die "non-loopback SERVER_HOST requires SERVER_AUTH_TOKEN or JZ_PI05_SERVER_AUTH_TOKEN"
    ;;
esac
if [[ -n "${SERVER_AUTH_TOKEN}" ]]; then
  export JZ_PI05_SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN}"
fi

rtc_prepare_runtime
if [[ "${PRINT_ONLY}" != "true" && "${CONFIG_ONLY}" != "true" ]]; then
  rtc_require_policy
  rtc_require_tokenizer
fi

COMMAND=(
  "${CONDA_PYTHON}"
  "${PI05_TASK_DIR}/run_policy_server.py"
  "--host=${SERVER_HOST}"
  "--port=${SERVER_PORT}"
  "--policy-path=${POLICY_PATH}"
  "--tokenizer-path=${TOKENIZER_PATH}"
  "--device=${POLICY_DEVICE}"
  "--max-request-bytes=${MAX_REQUEST_BYTES}"
  "--rtc-execution-horizon=${RTC_EXECUTION_HORIZON}"
  "--rtc-max-guidance-weight=${RTC_MAX_GUIDANCE_WEIGHT}"
  "--rtc-prefix-attention-schedule=${RTC_PREFIX_ATTENTION_SCHEDULE}"
  "--require-complete-step=${REQUIRE_COMPLETE_STEP}"
)
if [[ "${RTC_DEBUG}" == "true" ]]; then
  COMMAND+=(--rtc-debug)
fi
if [[ "${CONFIG_ONLY}" == "true" ]]; then
  COMMAND+=(--config-only)
fi
if [[ "${CHECK_POLICY_LOAD}" == "true" ]]; then
  COMMAND+=(--check-policy-load)
fi

echo "[tk_infer/pi05] backend=pytorch (TensorRT is not used)"
echo "[tk_infer/pi05] conda_env=${CONDA_ENV} python=${CONDA_PYTHON}"
echo "[tk_infer/pi05] policy=${POLICY_PATH:-<required-for-server-start>}"
echo "[tk_infer/pi05] listen=http://${SERVER_HOST}:${SERVER_PORT} auth=$([[ -n "${SERVER_AUTH_TOKEN}" ]] && echo enabled || echo disabled)"
rtc_run_or_print server "${COMMAND[@]}" "$@"
