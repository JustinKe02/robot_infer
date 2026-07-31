#!/usr/bin/env bash
set -euo pipefail

YBD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RTC_INFER_DIR="$(cd "${YBD_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${YBD_DIR}/../../../../.." && pwd)"

POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/my_devs/jz_robot_pin_timed/pi05/outputs/pi05_jz100_model16_head_right_expert_a_e10_seed1000/checkpoints/010600/pretrained_model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224}"
CONDA_ROOT="${CONDA_ROOT:-/mnt/data/public/miniconda3}"
CONDA_PYTHON="${CONDA_PYTHON:-${CONDA_ROOT}/envs/lerobot_flex/bin/python}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8088}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN:-${JZ_PI05_SERVER_AUTH_TOKEN:-}}"
CONFIG_ONLY="${CONFIG_ONLY:-false}"
CHECK_POLICY_LOAD="${CHECK_POLICY_LOAD:-false}"

[[ -x "${CONDA_PYTHON}" ]] || { echo "lerobot_flex python is missing: ${CONDA_PYTHON}" >&2; exit 2; }
if [[ "${SERVER_HOST}" != "127.0.0.1" && "${SERVER_HOST}" != "localhost" && -z "${SERVER_AUTH_TOKEN}" ]]; then
  echo "non-loopback SERVER_HOST requires SERVER_AUTH_TOKEN" >&2
  exit 2
fi
if [[ -n "${SERVER_AUTH_TOKEN}" ]]; then
  export JZ_PI05_SERVER_AUTH_TOKEN="${SERVER_AUTH_TOKEN}"
fi

COMMAND=(
  "${CONDA_PYTHON}"
  "${YBD_DIR}/policy_service.py"
  server
  "--host=${SERVER_HOST}"
  "--port=${SERVER_PORT}"
  "--policy-path=${POLICY_PATH}"
  "--tokenizer-path=${TOKENIZER_PATH}"
  "--device=${POLICY_DEVICE}"
  "--require-complete-step=true"
)
[[ "${CONFIG_ONLY}" == "true" ]] && COMMAND+=(--config-only)
[[ "${CHECK_POLICY_LOAD}" == "true" ]] && COMMAND+=(--check-policy-load)

echo "[jz/pi05/ybd] mode=server conda_env=lerobot_flex"
echo "[jz/pi05/ybd] policy=${POLICY_PATH}"
echo "[jz/pi05/ybd] cameras=head,right right_arm_only=true"
exec "${COMMAND[@]}" "$@"
