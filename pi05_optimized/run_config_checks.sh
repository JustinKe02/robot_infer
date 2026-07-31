#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

opt_require_executable "${CONDA_PYTHON}"

echo "[tk_infer/pi05_optimized] direct config-only check"
"${CONDA_PYTHON}" "${SCRIPT_DIR}/run_policy_server.py" --config-only

echo "[tk_infer/pi05_optimized] shell launcher config-only check"
CONFIG_ONLY=true bash "${SCRIPT_DIR}/run_server.sh"

echo "[tk_infer/pi05_optimized] shell launcher print-only check"
PRINT_COMMAND_ONLY=true bash "${SCRIPT_DIR}/run_server.sh"

echo "[tk_infer/pi05_optimized] Phase 2 optimized backend config-only check"
"${CONDA_PYTHON}" "${SCRIPT_DIR}/run_policy_server.py" \
  --config-only \
  --backend=torch_optimized \
  --torch-inference-mode=true \
  --torch-bf16-autocast=true

echo "[tk_infer/pi05_optimized] Phase 3 Triton config-only check"
"${CONDA_PYTHON}" "${SCRIPT_DIR}/run_policy_server.py" \
  --config-only \
  --backend=triton \
  --triton-artifact-path="${SCRIPT_DIR}/artifacts/triton/realtime_vla_b86a942"

echo "[tk_infer/pi05_optimized] Phase 6 paired temporal config-only check"
"${CONDA_PYTHON}" "${SCRIPT_DIR}/run_policy_server.py" \
  --config-only \
  --trajectory-processor=paired_temporal \
  --temporal-speed-factor=1.0 \
  --temporal-max-joint-step-rad=0.02 \
  --temporal-solver-timeout-s=0.05

echo "[tk_infer/pi05_optimized] client config-only check"
"${CONDA_PYTHON}" "${SCRIPT_DIR}/run_client.py" --config-only

echo "[tk_infer/pi05_optimized] client in-memory offline smoke"
"${CONDA_PYTHON}" "${SCRIPT_DIR}/run_client.py" --offline-smoke

echo "[tk_infer/pi05_optimized] Optimized config checks passed; no model, socket, or robot path was started."
