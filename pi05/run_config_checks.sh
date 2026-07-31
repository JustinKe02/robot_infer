#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[tk_infer/pi05/check] policy server config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_server.sh"

echo "[tk_infer/pi05/check] single-step dry-run client config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_single_step_dry_run.sh"

echo "[tk_infer/pi05/check] head-right single-step dry-run client config"
CAMERA_PROFILE=head_right CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_single_step_dry_run.sh"

echo "[tk_infer/pi05/check] async single-step dry-run client config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_async_single_step_dry_run.sh"

echo "[tk_infer/pi05/check] RTC dry-run client config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_rtc_dry_run.sh"

echo "[tk_infer/pi05/check] read-only live inference smoke command"
PRINT_COMMAND_ONLY=true bash "${SCRIPT_DIR}/run_inference_smoke.sh"

echo "[tk_infer/pi05/check] fixed step-010600 profile configs"
PROFILE_ENV=(
  "TK_PI05_010600_INTERMEDIATE_CONFIRMED=1"
)
env "${PROFILE_ENV[@]}" CONFIG_ONLY=true \
  bash "${SCRIPT_DIR}/profiles/step_010600/run_policy_server.sh"
env "${PROFILE_ENV[@]}" CONFIG_ONLY=true \
  bash "${SCRIPT_DIR}/profiles/step_010600/run_single_step_dry_run.sh"
env "${PROFILE_ENV[@]}" CONFIG_ONLY=true \
  bash "${SCRIPT_DIR}/profiles/step_010600/run_async_single_step_dry_run.sh"
env "${PROFILE_ENV[@]}" CONFIG_ONLY=true \
  bash "${SCRIPT_DIR}/profiles/step_010600/run_rtc_dry_run.sh"

echo "[tk_infer/pi05/check] armed launch gates and configs (CONFIG_ONLY; no robot I/O)"
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
CONFIG_ONLY=true \
  bash "${SCRIPT_DIR}/run_single_step_armed.sh"
JZ_ROBOT_PIN_ARMED=1 \
I_UNDERSTAND_JZ_ROBOT_PIN_MOVES_ROBOT=1 \
JZ_POLICY_INFERENCE_ARMED=1 \
CONFIG_ONLY=true \
  bash "${SCRIPT_DIR}/run_rtc_armed.sh"

echo "[tk_infer/pi05/check] PASS: config-only checks completed without robot access"
