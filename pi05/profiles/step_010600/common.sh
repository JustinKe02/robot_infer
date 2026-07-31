#!/usr/bin/env bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_TASK_DIR="$(cd "${PROFILE_DIR}/../.." && pwd)"
source "${PI05_TASK_DIR}/common.sh"

PROFILE_LABEL="step_010600_epoch10_head_right"
PROFILE_CONFIRM_ENV="TK_PI05_010600_INTERMEDIATE_CONFIRMED"
PROFILE_POLICY_PATH="${PI05_TASK_DIR}/checkpoints/010600/pretrained_model"
PROFILE_CHECKPOINT_STEP=10600
PROFILE_CONFIGURED_STEPS=15900
PROFILE_CHECKPOINT_FINGERPRINT=4698315f6936f9e9ef19017cfdb873588eba771fdb23595879ce2a7703b4c8dd
PROFILE_CAMERA_PROFILE=head_right
PROFILE_TASK="jz robot pin timed vr teleoperation"

if [[ -n "${POLICY_PATH:-}" ]]; then
  requested_policy="$(readlink -m -- "${POLICY_PATH}")"
  expected_policy="$(readlink -m -- "${PROFILE_POLICY_PATH}")"
  [[ "${requested_policy}" == "${expected_policy}" ]] \
    || rtc_die "${PROFILE_LABEL} forbids POLICY_PATH override: ${requested_policy}"
fi

export POLICY_PATH="${PROFILE_POLICY_PATH}"
export CAMERA_PROFILE="${PROFILE_CAMERA_PROFILE}"
export TASK="${PROFILE_TASK}"
export JZ_PI05_EXPECTED_CHECKPOINT_STEP="${PROFILE_CHECKPOINT_STEP}"
export JZ_PI05_EXPECTED_CONFIGURED_STEPS="${PROFILE_CONFIGURED_STEPS}"
export JZ_PI05_EXPECTED_CHECKPOINT_FINGERPRINT="${PROFILE_CHECKPOINT_FINGERPRINT}"
export JZ_PI05_EXPECTED_CHECKPOINT_PATH="$(readlink -m -- "${PROFILE_POLICY_PATH}")"
export JZ_PI05_EXPECTED_COMPLETE_STEP=false

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

profile_require_run_time() {
  local value="$1"
  [[ "${value}" =~ ^[0-9]+$ ]] \
    || rtc_die "PROFILE_RUN_TIME_S must be an integer in 1..300, got ${value}"
  (( value >= 1 && value <= 300 )) \
    || rtc_die "PROFILE_RUN_TIME_S must be an integer in 1..300, got ${value}"
}
