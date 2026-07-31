#!/usr/bin/env bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_TASK_DIR="$(cd "${PROFILE_DIR}/../.." && pwd)"
source "${PI05_TASK_DIR}/common.sh"

PROFILE_LABEL="rtc_conditioned_step_010600_three_camera_client"
PROFILE_CONFIRM_ENV="TK_PI05_RTC_CONDITIONED_010600_CONFIRMED"
PROFILE_SINGLE_STEP_PASS_ENV="JZ_PI05_RTC_CONDITIONED_SINGLE_STEP_ARMED_PASSED"
PROFILE_JOINT_DELTA_BYPASS_ENV="JZ_PI05_RTC_CONDITIONED_JOINT_DELTA_BYPASS_CONFIRMED"
PROFILE_CONTINUOUS_DRY_RUN_PASS_ENV="JZ_PI05_RTC_CONDITIONED_CONTINUOUS_DRY_RUN_PASSED"
PROFILE_CONTINUOUS_ARMED_ENV="JZ_PI05_RTC_CONDITIONED_CONTINUOUS_ARMED_CONFIRMED"
PROFILE_SERVER_URL="http://127.0.0.1:18089"
PROFILE_POLICY_PATH="${PI05_TASK_DIR}/checkpoints/pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000_010600/pretrained_model"
PROFILE_CONFIGURED_STEPS=10600
PROFILE_CHECKPOINT_FINGERPRINT=039ef411871f75e8504b7b72ccb299c29c4cdf3a99e7bfbc241a3daae7bfaa57
PROFILE_TASK="Put the bottle on the right into the basket on the left."
PROFILE_ORIN_IP=192.168.1.81
PROFILE_STATE_BIND_IP=0.0.0.0
PROFILE_STATE_PORT=39010
PROFILE_COMMAND_PORT=39020
PROFILE_MAX_CAMERA_STATE_RECEIVE_SKEW_MS=250

if [[ -n "${SERVER_URL:-}" && "${SERVER_URL}" != "${PROFILE_SERVER_URL}" ]]; then
  rtc_die "${PROFILE_LABEL} forbids SERVER_URL=${SERVER_URL}"
fi
if [[ -n "${CAMERA_PROFILE:-}" && "${CAMERA_PROFILE}" != "three_camera" ]]; then
  rtc_die "${PROFILE_LABEL} requires CAMERA_PROFILE=three_camera"
fi
if [[ -n "${TASK:-}" && "${TASK}" != "${PROFILE_TASK}" ]]; then
  rtc_die "${PROFILE_LABEL} forbids TASK=${TASK@Q}"
fi
for locked_name in ORIN_IP STATE_BIND_IP STATE_PORT COMMAND_PORT MAX_CAMERA_STATE_RECEIVE_SKEW_MS; do
  profile_name="PROFILE_${locked_name}"
  if [[ -n "${!locked_name:-}" && "${!locked_name}" != "${!profile_name}" ]]; then
    rtc_die "${PROFILE_LABEL} forbids ${locked_name}=${!locked_name}"
  fi
done
PROFILE_JOINT_DELTA_MODE="${PROFILE_JOINT_DELTA_MODE:-enabled}"
rtc_require_choice PROFILE_JOINT_DELTA_MODE "${PROFILE_JOINT_DELTA_MODE}" enabled bypass
if [[ "${PROFILE_JOINT_DELTA_MODE}" == "bypass" ]]; then
  [[ "${!PROFILE_JOINT_DELTA_BYPASS_ENV:-}" == "1" ]] \
    || rtc_die "${PROFILE_LABEL} joint-delta bypass requires ${PROFILE_JOINT_DELTA_BYPASS_ENV}=1"
  export JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1
  export I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1
else
  if [[ "${JZ_PI05_DISABLE_JOINT_DELTA_CHECKS:-0}" == "1" ]]; then
    rtc_die "${PROFILE_LABEL} forbids JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=1 outside the bypass launcher"
  fi
  if [[ "${I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED:-0}" == "1" ]]; then
    rtc_die "${PROFILE_LABEL} forbids I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=1 outside the bypass launcher"
  fi
  if [[ "${!PROFILE_JOINT_DELTA_BYPASS_ENV:-}" == "1" ]]; then
    rtc_die "${PROFILE_LABEL} forbids ${PROFILE_JOINT_DELTA_BYPASS_ENV}=1 outside the bypass launcher"
  fi
  export JZ_PI05_DISABLE_JOINT_DELTA_CHECKS=0
  export I_UNDERSTAND_JOINT_DELTA_CHECKS_ARE_DISABLED=0
fi

export SERVER_URL="${PROFILE_SERVER_URL}"
export CAMERA_PROFILE=three_camera
export TASK="${PROFILE_TASK}"
export ORIN_IP="${PROFILE_ORIN_IP}"
export STATE_BIND_IP="${PROFILE_STATE_BIND_IP}"
export STATE_PORT="${PROFILE_STATE_PORT}"
export COMMAND_PORT="${PROFILE_COMMAND_PORT}"
export MAX_CAMERA_STATE_RECEIVE_SKEW_MS="${PROFILE_MAX_CAMERA_STATE_RECEIVE_SKEW_MS}"
export JZ_PI05_EXPECTED_CHECKPOINT_STEP=null
export JZ_PI05_EXPECTED_CONFIGURED_STEPS="${PROFILE_CONFIGURED_STEPS}"
export JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT="${PROFILE_CHECKPOINT_FINGERPRINT}"
export JZ_PI05_EXPECTED_CHECKPOINT_PATH="$(readlink -m -- "${PROFILE_POLICY_PATH}")"
export JZ_PI05_EXPECTED_COMPLETE_STEP=null

profile_require_confirmation() {
  [[ "${!PROFILE_CONFIRM_ENV:-}" == "1" ]] \
    || rtc_die "${PROFILE_LABEL} requires ${PROFILE_CONFIRM_ENV}=1"
}

profile_require_fps() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || rtc_die "${name} must be an integer in 1..20, got ${value}"
  (( value >= 1 && value <= 20 )) \
    || rtc_die "${name} must be an integer in 1..20, got ${value}"
}

profile_require_bounded_actions() {
  local value="$1"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || rtc_die "PROFILE_MAX_SENT_ACTIONS must be an integer in 1..20, got ${value}"
  (( value >= 1 && value <= 20 )) \
    || rtc_die "PROFILE_MAX_SENT_ACTIONS must be an integer in 1..20, got ${value}"
}
