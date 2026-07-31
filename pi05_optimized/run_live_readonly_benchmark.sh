#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

(( $# == 0 )) || opt_die "configure the live read-only benchmark through environment variables"
for name in JZ_ROBOT_PIN_ARMED I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT JZ_POLICY_INFERENCE_ARMED; do
  [[ -z "${!name:-}" ]] || opt_die "${name} must be unset for the live read-only benchmark"
done

SERVER_URL="${SERVER_URL:-http://127.0.0.1:18088}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-3}"
MEASURE_REQUESTS="${MEASURE_REQUESTS:-30}"
CONTROL_HZ="${CONTROL_HZ:-5}"
CONNECT_TIMEOUT_S="${CONNECT_TIMEOUT_S:-5}"
STATE_TIMEOUT_S="${STATE_TIMEOUT_S:-1}"
CAMERA_TIMEOUT_MS="${CAMERA_TIMEOUT_MS:-5000}"
MAX_CAMERA_STATE_RECEIVE_SKEW_MS="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS:-250}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-120}"
TASK="${TASK:-jz robot pin timed vr teleoperation}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_ROOT}/live/live_readonly_${RUN_STAMP}.json}"

opt_prepare_runtime
opt_require_local_write_path OUTPUT_JSON "${OUTPUT_JSON}"

echo "[tk_infer/pi05_optimized/live] read-only sensors + policy; outputs are recorded locally"
echo "[tk_infer/pi05_optimized/live] server=${SERVER_URL} state=udp://${STATE_BIND_IP}:${STATE_PORT} orin=${ORIN_IP}"
echo "[tk_infer/pi05_optimized/live] warmup=${WARMUP_REQUESTS} measure=${MEASURE_REQUESTS} hz=${CONTROL_HZ}"
exec "${CONDA_PYTHON}" "${SCRIPT_DIR}/tools/live_readonly_benchmark.py" \
  --server-url="${SERVER_URL}" \
  --orin-ip="${ORIN_IP}" \
  --state-bind-ip="${STATE_BIND_IP}" \
  --state-port="${STATE_PORT}" \
  --warmup-requests="${WARMUP_REQUESTS}" \
  --measure-requests="${MEASURE_REQUESTS}" \
  --control-hz="${CONTROL_HZ}" \
  --connect-timeout-s="${CONNECT_TIMEOUT_S}" \
  --state-timeout-s="${STATE_TIMEOUT_S}" \
  --camera-timeout-ms="${CAMERA_TIMEOUT_MS}" \
  --max-camera-state-receive-skew-ms="${MAX_CAMERA_STATE_RECEIVE_SKEW_MS}" \
  --request-timeout-s="${REQUEST_TIMEOUT_S}" \
  --task="${TASK}" \
  --output-json="${OUTPUT_JSON}"
