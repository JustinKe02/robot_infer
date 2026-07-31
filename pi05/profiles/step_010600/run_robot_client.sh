#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"

(( $# == 0 )) || rtc_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"
profile_require_confirmation

PROFILE_MODE="${PROFILE_MODE:-single_step}"
PROFILE_EXECUTION="${PROFILE_EXECUTION:-dry_run}"
rtc_require_choice PROFILE_MODE "${PROFILE_MODE}" single_step async_single_step rtc
rtc_require_choice PROFILE_EXECUTION "${PROFILE_EXECUTION}" dry_run armed

if [[ "${PROFILE_MODE}" == "single_step" ]]; then
  SENSOR_FPS="${PROFILE_SENSOR_FPS:-5}"
  CONTROL_FPS="${PROFILE_CONTROL_FPS:-5}"
  if [[ "${PROFILE_EXECUTION}" == "armed" ]]; then
    RUN_TIME_S="${PROFILE_RUN_TIME_S:-1}"
    MAX_SENT_ACTIONS=1
  else
    RUN_TIME_S="${PROFILE_RUN_TIME_S:-10}"
    MAX_SENT_ACTIONS="${MAX_SENT_ACTIONS:-0}"
  fi
else
  SENSOR_FPS="${PROFILE_SENSOR_FPS:-20}"
  CONTROL_FPS="${PROFILE_CONTROL_FPS:-20}"
  RUN_TIME_S="${PROFILE_RUN_TIME_S:-10}"
  MAX_SENT_ACTIONS="${MAX_SENT_ACTIONS:-0}"
  if [[ "${PROFILE_EXECUTION}" == "armed" ]]; then
    [[ "${JZ_PI05_SINGLE_STEP_ARMED_PASSED:-}" == "1" ]] \
      || rtc_die "armed ${PROFILE_MODE} requires JZ_PI05_SINGLE_STEP_ARMED_PASSED=1"
  fi
fi

profile_require_fps PROFILE_SENSOR_FPS "${SENSOR_FPS}"
profile_require_fps PROFILE_CONTROL_FPS "${CONTROL_FPS}"
profile_require_run_time "${RUN_TIME_S}"

export MODE="${PROFILE_MODE}"
export EXECUTION="${PROFILE_EXECUTION}"
export SENSOR_FPS
export CONTROL_FPS
export RUN_TIME_S
export MAX_SENT_ACTIONS
export EMPTY_QUEUE_STRATEGY=stop

echo "[tk_infer/pi05/profile] client=${PROFILE_LABEL} mode=${MODE} execution=${EXECUTION}"
echo "[tk_infer/pi05/profile] checkpoint_step=10600/15900 cameras=head:5555,right:5557 action_boundary=full_raw18"
echo "[tk_infer/pi05/profile] sensor_fps=${SENSOR_FPS} control_fps=${CONTROL_FPS} run_time_s=${RUN_TIME_S} max_sent_actions=${MAX_SENT_ACTIONS}"
exec bash "${PI05_TASK_DIR}/run_client.sh"
