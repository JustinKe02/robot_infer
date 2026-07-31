#!/usr/bin/env bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_OPT_DIR="$(cd "${PROFILE_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PI05_OPT_DIR}/../.." && pwd)"

PROFILE_LABEL="rtc_conditioned_step_010600_three_camera"
PROFILE_POLICY_PATH="${REPO_ROOT}/tk_infer/pi05/checkpoints/pi05_jz100_model16_head_left_right_expert_b_rtc_e10_seed1000_010600/pretrained_model"
PROFILE_CONFIGURED_STEPS=10600
PROFILE_CHECKPOINT_FINGERPRINT=039ef411871f75e8504b7b72ccb299c29c4cdf3a99e7bfbc241a3daae7bfaa57
PROFILE_TASK="Put the bottle on the right into the basket on the left."
PROFILE_BACKEND="torch_rtc_conditioned"

if [[ -n "${PI05_OPT_POLICY_PATH:-}" ]]; then
  requested_policy="$(readlink -m -- "${PI05_OPT_POLICY_PATH}")"
  expected_policy="$(readlink -m -- "${PROFILE_POLICY_PATH}")"
  if [[ "${requested_policy}" != "${expected_policy}" ]]; then
    echo "[tk_infer/pi05_optimized/profile] ERROR: ${PROFILE_LABEL} forbids PI05_OPT_POLICY_PATH override: ${requested_policy}" >&2
    exit 2
  fi
fi
if [[ -n "${PI05_OPT_BACKEND:-}" && "${PI05_OPT_BACKEND}" != "${PROFILE_BACKEND}" ]]; then
  echo "[tk_infer/pi05_optimized/profile] ERROR: ${PROFILE_LABEL} forbids PI05_OPT_BACKEND=${PI05_OPT_BACKEND}" >&2
  exit 2
fi

export PI05_OPT_POLICY_PATH="${PROFILE_POLICY_PATH}"
export PI05_OPT_BACKEND="${PROFILE_BACKEND}"
export PI05_OPT_REQUIRE_COMPLETE_STEP=false
export PI05_OPT_RTC_CONDITIONED_TASK="${PROFILE_TASK}"
export JZ_PI05_OPT_SERVER_PORT="${JZ_PI05_OPT_SERVER_PORT:-18089}"
export PI05_OPT_TRACE_PATH="${PI05_OPT_TRACE_PATH:-${PI05_OPT_DIR}/logs/server/rtc_conditioned_inference_trace.jsonl}"
source "${PI05_OPT_DIR}/common.sh"
