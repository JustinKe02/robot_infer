#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"

(( $# == 0 )) || rtc_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"
profile_require_confirmation

PROFILE_MODE="${PROFILE_MODE:-single_step}"
PROFILE_EXECUTION="${PROFILE_EXECUTION:-dry_run}"
PROFILE_RTC_RUN_MODE="${PROFILE_RTC_RUN_MODE:-bounded}"
rtc_require_choice PROFILE_MODE "${PROFILE_MODE}" single_step rtc
rtc_require_choice PROFILE_EXECUTION "${PROFILE_EXECUTION}" dry_run armed
rtc_require_choice PROFILE_RTC_RUN_MODE "${PROFILE_RTC_RUN_MODE}" bounded qualification continuous

if [[ "${PROFILE_MODE}" == "single_step" ]]; then
  [[ "${PROFILE_RTC_RUN_MODE}" == "bounded" ]] \
    || rtc_die "${PROFILE_LABEL} permits PROFILE_RTC_RUN_MODE only in RTC mode"
  SENSOR_FPS="${PROFILE_SENSOR_FPS:-5}"
  CONTROL_FPS="${PROFILE_CONTROL_FPS:-5}"
  if [[ "${PROFILE_EXECUTION}" == "armed" ]]; then
    RUN_TIME_S=1
    MAX_SENT_ACTIONS=1
  elif [[ "${PROFILE_JOINT_DELTA_MODE}" == "bypass" ]]; then
    RUN_TIME_S=1
    MAX_SENT_ACTIONS=1
  else
    RUN_TIME_S="${PROFILE_RUN_TIME_S:-10}"
    MAX_SENT_ACTIONS=0
  fi
else
  SENSOR_FPS="${PROFILE_SENSOR_FPS:-20}"
  CONTROL_FPS="${PROFILE_CONTROL_FPS:-20}"
  if [[ "${PROFILE_EXECUTION}" == "armed" ]]; then
    [[ "${!PROFILE_SINGLE_STEP_PASS_ENV:-}" == "1" ]] \
      || rtc_die "armed RTC requires ${PROFILE_SINGLE_STEP_PASS_ENV}=1"
    if [[ "${PROFILE_RTC_RUN_MODE}" == "continuous" ]]; then
      [[ "${PROFILE_JOINT_DELTA_MODE}" == "bypass" ]] \
        || rtc_die "continuous RTC armed requires joint-delta bypass mode"
      [[ "${!PROFILE_CONTINUOUS_DRY_RUN_PASS_ENV:-}" == "1" ]] \
        || rtc_die "continuous RTC armed requires ${PROFILE_CONTINUOUS_DRY_RUN_PASS_ENV}=1"
      [[ "${!PROFILE_CONTINUOUS_ARMED_ENV:-}" == "1" ]] \
        || rtc_die "continuous RTC armed requires ${PROFILE_CONTINUOUS_ARMED_ENV}=1"
      [[ -z "${PROFILE_RUN_TIME_S:-}" || "${PROFILE_RUN_TIME_S}" == "0" ]] \
        || rtc_die "continuous RTC armed requires PROFILE_RUN_TIME_S unset or 0"
      [[ -z "${PROFILE_MAX_SENT_ACTIONS:-}" || "${PROFILE_MAX_SENT_ACTIONS}" == "0" ]] \
        || rtc_die "continuous RTC armed requires PROFILE_MAX_SENT_ACTIONS unset or 0"
      RUN_TIME_S=0
      MAX_SENT_ACTIONS=0
    elif [[ "${PROFILE_RTC_RUN_MODE}" == "qualification" ]]; then
      rtc_die "RTC qualification is dry-run only"
    else
      RUN_TIME_S=1
      MAX_SENT_ACTIONS="${PROFILE_MAX_SENT_ACTIONS:-10}"
      profile_require_bounded_actions "${MAX_SENT_ACTIONS}"
    fi
  elif [[ "${PROFILE_JOINT_DELTA_MODE}" == "bypass" ]]; then
    if [[ "${PROFILE_RTC_RUN_MODE}" == "qualification" ]]; then
      RUN_TIME_S=5
      MAX_SENT_ACTIONS=0
    elif [[ "${PROFILE_RTC_RUN_MODE}" == "continuous" ]]; then
      rtc_die "continuous RTC mode is armed-only; use qualification for dry-run"
    else
      RUN_TIME_S=1
      MAX_SENT_ACTIONS="${PROFILE_MAX_SENT_ACTIONS:-10}"
      profile_require_bounded_actions "${MAX_SENT_ACTIONS}"
    fi
  else
    [[ "${PROFILE_RTC_RUN_MODE}" == "bounded" ]] \
      || rtc_die "RTC qualification/continuous modes require joint-delta bypass"
    RUN_TIME_S="${PROFILE_RUN_TIME_S:-10}"
    MAX_SENT_ACTIONS=0
  fi
fi

profile_require_fps PROFILE_SENSOR_FPS "${SENSOR_FPS}"
profile_require_fps PROFILE_CONTROL_FPS "${CONTROL_FPS}"

export MODE="${PROFILE_MODE}"
export EXECUTION="${PROFILE_EXECUTION}"
export SENSOR_FPS
export CONTROL_FPS
export RUN_TIME_S
export MAX_SENT_ACTIONS
export EMPTY_QUEUE_STRATEGY=stop

echo "[tk_infer/pi05/profile] client=${PROFILE_LABEL} mode=${MODE} execution=${EXECUTION}"
echo "[tk_infer/pi05/profile] server=${SERVER_URL} checkpoint_step=null/${PROFILE_CONFIGURED_STEPS}"
echo "[tk_infer/pi05/profile] cameras=head:5555,left:5556,right:5557 action_boundary=full_raw18"
echo "[tk_infer/pi05/profile] sensor_fps=${SENSOR_FPS} control_fps=${CONTROL_FPS} run_time_s=${RUN_TIME_S} max_sent_actions=${MAX_SENT_ACTIONS}"
echo "[tk_infer/pi05/profile] joint_delta_checks=${PROFILE_JOINT_DELTA_MODE} rtc_run_mode=${PROFILE_RTC_RUN_MODE}"
exec bash "${PI05_TASK_DIR}/run_client.sh"
