#!/usr/bin/env bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_OPT_DIR="$(cd "${PROFILE_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${PI05_OPT_DIR}/../.." && pwd)"

PROFILE_LABEL="step_015900_epoch15_head_right_complete"
PROFILE_POLICY_PATH="${REPO_ROOT}/tk_infer/pi05/checkpoints/015900/pretrained_model"
PROFILE_CHECKPOINT_STEP=15900
PROFILE_CONFIGURED_STEPS=15900
PROFILE_CHECKPOINT_FINGERPRINT=9d6d37f6111a034209c9bdc2899423a3258cc35070cb8294194c9c594197b58a

if [[ -n "${PI05_OPT_POLICY_PATH:-}" ]]; then
  requested_policy="$(readlink -m -- "${PI05_OPT_POLICY_PATH}")"
  expected_policy="$(readlink -m -- "${PROFILE_POLICY_PATH}")"
  if [[ "${requested_policy}" != "${expected_policy}" ]]; then
    echo "[tk_infer/pi05_optimized/profile] ERROR: ${PROFILE_LABEL} forbids PI05_OPT_POLICY_PATH override: ${requested_policy}" >&2
    exit 2
  fi
fi

export PI05_OPT_POLICY_PATH="${PROFILE_POLICY_PATH}"
source "${PI05_OPT_DIR}/common.sh"
