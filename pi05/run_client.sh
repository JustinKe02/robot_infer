#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODE="${MODE:-rtc}"
EXECUTION="${EXECUTION:-dry_run}"
rtc_require_choice MODE "${MODE}" single_step async_single_step rtc
rtc_require_choice EXECUTION "${EXECUTION}" dry_run armed
for argument in "$@"; do
  case "${argument}" in
    --mode|--mode=*|--execution|--execution=*)
      rtc_die "set MODE/EXECUTION as environment variables; overriding safety mode via extra CLI args is forbidden"
      ;;
    --auth-token|--auth-token=*)
      rtc_die "pass the server token via SERVER_AUTH_TOKEN/JZ_PI05_SERVER_AUTH_TOKEN, not CLI arguments"
      ;;
  esac
done

CONFIG_ONLY="$(rtc_normalize_bool CONFIG_ONLY "${CONFIG_ONLY:-false}")"
HEALTH_ONLY="$(rtc_normalize_bool HEALTH_ONLY "${HEALTH_ONLY:-false}")"
CONNECT_SMOKE="$(rtc_normalize_bool CONNECT_SMOKE "${CONNECT_SMOKE:-false}")"
INFERENCE_SMOKE="$(rtc_normalize_bool INFERENCE_SMOKE "${INFERENCE_SMOKE:-false}")"
if [[ "${EXECUTION}" == "armed" ]]; then
  rtc_require_armed_confirmation
fi

SERVER_URL="${SERVER_URL:-http://127.0.0.1:8088}"
SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN:-${JZ_PI05_SERVER_AUTH_TOKEN:-}}"
case "${SERVER_URL}" in
  http://127.0.0.1:*|http://localhost:*|http://\[::1\]:*)
    ;;
  *)
    [[ -n "${SERVER_AUTH_TOKEN}" ]] \
      || rtc_die "non-loopback SERVER_URL requires SERVER_AUTH_TOKEN or JZ_PI05_SERVER_AUTH_TOKEN"
    ;;
esac
if [[ -n "${SERVER_AUTH_TOKEN}" ]]; then
  export JZ_PI05_SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN}"
fi
ROBOT_ID="${ROBOT_ID:-jz_robot_pin_timed_pi05_rtc}"
CAMERA_PROFILE="${CAMERA_PROFILE:-three_camera}"
rtc_require_choice CAMERA_PROFILE "${CAMERA_PROFILE}" three_camera head_right
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
STATE_TIMEOUT_S="${STATE_TIMEOUT_S:-1.0}"
MAX_CAMERA_STATE_RECEIVE_SKEW_MS="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS:-250}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-300.0}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
TASK="${TASK:-jz robot pin timed vr teleoperation}"
SENSOR_FPS="${SENSOR_FPS:-20}"
CONTROL_FPS="${CONTROL_FPS:-20}"
RUN_TIME_S="${RUN_TIME_S:-10}"
QUEUE_LOW_WATERMARK="${QUEUE_LOW_WATERMARK:-30}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-50}"
FIRST_CHUNK_TIMEOUT_S="${FIRST_CHUNK_TIMEOUT_S:-60}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-120}"
EMPTY_QUEUE_STRATEGY="${EMPTY_QUEUE_STRATEGY:-stop}"
FULLY_STALE_CHUNK_LIMIT="${FULLY_STALE_CHUNK_LIMIT:-3}"
METRICS_LOG_INTERVAL_S="${METRICS_LOG_INTERVAL_S:-2}"
MAX_SENT_ACTIONS="${MAX_SENT_ACTIONS:-0}"

rtc_prepare_runtime

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
CLIENT_OUTPUT_DIR="${CLIENT_OUTPUT_DIR:-${OUTPUT_ROOT}/client/${MODE}_${EXECUTION}_${RUN_STAMP}}"
rtc_require_local_write_path CLIENT_OUTPUT_DIR "${CLIENT_OUTPUT_DIR}"
COMMAND=(
  "${CONDA_PYTHON}"
  "${PI05_TASK_DIR}/run_robot_client.py"
  "--server-url=${SERVER_URL}"
  "--mode=${MODE}"
  "--execution=${EXECUTION}"
  "--robot-id=${ROBOT_ID}"
  "--camera-profile=${CAMERA_PROFILE}"
  "--orin-ip=${ORIN_IP}"
  "--state-bind-ip=${STATE_BIND_IP}"
  "--state-port=${STATE_PORT}"
  "--state-timeout-s=${STATE_TIMEOUT_S}"
  "--max-camera-state-receive-skew-ms=${MAX_CAMERA_STATE_RECEIVE_SKEW_MS}"
  "--connect-timeout-s=${CONNECT_TIMEOUT_S}"
  "--command-port=${COMMAND_PORT}"
  "--task=${TASK}"
  "--sensor-fps=${SENSOR_FPS}"
  "--control-fps=${CONTROL_FPS}"
  "--run-time-s=${RUN_TIME_S}"
  "--queue-low-watermark=${QUEUE_LOW_WATERMARK}"
  "--max-queue-size=${MAX_QUEUE_SIZE}"
  "--first-chunk-timeout-s=${FIRST_CHUNK_TIMEOUT_S}"
  "--rtc-execution-horizon=${RTC_EXECUTION_HORIZON}"
  "--request-timeout-s=${REQUEST_TIMEOUT_S}"
  "--empty-queue-strategy=${EMPTY_QUEUE_STRATEGY}"
  "--fully-stale-chunk-limit=${FULLY_STALE_CHUNK_LIMIT}"
  "--metrics-log-interval-s=${METRICS_LOG_INTERVAL_S}"
  "--max-sent-actions=${MAX_SENT_ACTIONS}"
  "--config-only=${CONFIG_ONLY}"
  "--health-only=${HEALTH_ONLY}"
  "--connect-smoke=${CONNECT_SMOKE}"
  "--inference-smoke=${INFERENCE_SMOKE}"
  "--output-dir=${CLIENT_OUTPUT_DIR}"
)

echo "[tk_infer/pi05] backend=pytorch-client (TensorRT is not used)"
echo "[tk_infer/pi05] mode=${MODE} execution=${EXECUTION} server=${SERVER_URL} auth=$([[ -n "${SERVER_AUTH_TOKEN}" ]] && echo enabled || echo disabled)"
echo "[tk_infer/pi05] camera_profile=${CAMERA_PROFILE} state=udp://${STATE_BIND_IP}:${STATE_PORT} allowed_source=${ORIN_IP}"
echo "[tk_infer/pi05] camera_state_receive_skew_limit_ms=${MAX_CAMERA_STATE_RECEIVE_SKEW_MS}"
echo "[tk_infer/pi05] action_boundary=full_raw18 max_sent_actions=${MAX_SENT_ACTIONS}"
echo "[tk_infer/pi05] command_target=udp://${ORIN_IP}:${COMMAND_PORT}"
echo "[tk_infer/pi05] output=${CLIENT_OUTPUT_DIR}"
if [[ "${INFERENCE_SMOKE}" == "true" ]]; then
  echo "[tk_infer/pi05] inference_smoke=read-only send_action=disabled"
fi
rtc_run_or_print client "${COMMAND[@]}" "$@"
