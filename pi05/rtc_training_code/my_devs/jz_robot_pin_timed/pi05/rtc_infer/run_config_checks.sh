#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[jz/pi05/rtc_infer/check] policy server config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_server.sh"

echo "[jz/pi05/rtc_infer/check] single-step dry-run client config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_single_step_dry_run.sh"

echo "[jz/pi05/rtc_infer/check] RTC dry-run client config"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_rtc_dry_run.sh"

echo "[jz/pi05/rtc_infer/check] armed launch gates and configs (CONFIG_ONLY; no robot I/O)"
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

echo "[jz/pi05/rtc_infer/check] PASS: config-only checks completed without robot access"

