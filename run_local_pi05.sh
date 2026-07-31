#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${TASK_ROOT}/.." && pwd)"

export POLICY_PATH="${POLICY_PATH:-${REPO_ROOT}/outputs/pi05_jz100_model16_head_right_expert_a_e10_seed1000/checkpoints/010600/pretrained_model}"
export DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/jz_robot_pin_timed_merged_100eps_20260728}"
export DATASET_REPO_ID="${DATASET_REPO_ID:-local/jz_robot_pin_timed_merged_100eps_20260728_pi05_head_right}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/assets/modelscope/google/paligemma-3b-pt-224}"
export SAMPLE_INDICES="${SAMPLE_INDICES:-first}"
export DEVICE="${DEVICE:-cuda}"

echo "[tk_infer/pi05] preset=current-local-head-right sample_indices=${SAMPLE_INDICES}"
exec bash "${TASK_ROOT}/run_infer.sh" "$@"

