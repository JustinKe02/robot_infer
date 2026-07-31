#!/usr/bin/env bash
set -euo pipefail

YBD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/home/luzhuang/miniconda3}"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/lerobot_flex/bin/python}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:8088}"
SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN:-${JZ_PI05_SERVER_AUTH_TOKEN:-}}"
ORIN_IP="${ORIN_IP:-192.168.1.81}"
STATE_BIND_IP="${STATE_BIND_IP:-0.0.0.0}"
STATE_PORT="${STATE_PORT:-39010}"
COMMAND_PORT="${COMMAND_PORT:-39020}"
SENSOR_FPS="${SENSOR_FPS:-5}"
CONTROL_FPS="${CONTROL_FPS:-5}"
TASK="Put the bottle on the right into the basket on the left."

[[ -x "${CONDA_PYTHON}" ]] || { echo "lerobot_flex python is missing: ${CONDA_PYTHON}" >&2; exit 2; }
case "${SERVER_URL}" in
  http://127.0.0.1:*|http://localhost:*) ;;
  *) [[ -n "${SERVER_AUTH_TOKEN}" ]] || { echo "remote SERVER_URL requires SERVER_AUTH_TOKEN" >&2; exit 2; } ;;
esac
if [[ -n "${SERVER_AUTH_TOKEN}" ]]; then
  export JZ_PI05_SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN}"
fi

echo "[jz/pi05/ybd] mode=single_step execution=dry_run inference_smoke=true"
echo "[jz/pi05/ybd] send_action=disabled task=${TASK}"
exec "${CONDA_PYTHON}" "${YBD_DIR}/policy_service.py" client \
  "--server-url=${SERVER_URL}" \
  "--mode=single_step" \
  "--execution=dry_run" \
  "--inference-smoke=true" \
  "--orin-ip=${ORIN_IP}" \
  "--state-bind-ip=${STATE_BIND_IP}" \
  "--state-port=${STATE_PORT}" \
  "--command-port=${COMMAND_PORT}" \
  "--task=${TASK}" \
  "--sensor-fps=${SENSOR_FPS}" \
  "--control-fps=${CONTROL_FPS}" \
  "$@"
