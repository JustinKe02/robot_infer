#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( $# != 0 )); then
  echo "[tk_infer/pi05/smoke] pass configuration through environment variables, not CLI args" >&2
  exit 2
fi

SMOKE_SENSOR_FPS="${SMOKE_SENSOR_FPS:-5}"
SMOKE_CONTROL_FPS="${SMOKE_CONTROL_FPS:-5}"
for value_name in SMOKE_SENSOR_FPS SMOKE_CONTROL_FPS; do
  value="${!value_name}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 || value > 10 )); then
    echo "[tk_infer/pi05/smoke] ${value_name} must be an integer in 1..10, got ${value}" >&2
    exit 2
  fi
done

echo "[tk_infer/pi05/smoke] read-only live inference: health + state/cameras + one policy request"
echo "[tk_infer/pi05/smoke] robot.send_action is not called; command UDP is not used"

MODE=single_step \
EXECUTION=dry_run \
CONFIG_ONLY=false \
HEALTH_ONLY=false \
CONNECT_SMOKE=false \
INFERENCE_SMOKE=true \
SENSOR_FPS="${SMOKE_SENSOR_FPS}" \
CONTROL_FPS="${SMOKE_CONTROL_FPS}" \
RUN_TIME_S=1 \
EMPTY_QUEUE_STRATEGY=stop \
  bash "${SCRIPT_DIR}/run_client.sh"
