#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"

(( $# == 0 )) || opt_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"

echo "[tk_infer/pi05_optimized/profile] server=${PROFILE_LABEL}"
echo "[tk_infer/pi05_optimized/profile] checkpoint=${POLICY_PATH} configured_steps=10600 complete=path-unproven"
echo "[tk_infer/pi05_optimized/profile] backend=${PROFILE_BACKEND} task=${PROFILE_TASK@Q} port=${JZ_PI05_OPT_SERVER_PORT}"

print_only="$(opt_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"
config_only="$(opt_normalize_bool CONFIG_ONLY "${CONFIG_ONLY:-false}")"
if [[ "${print_only}" != "true" && "${config_only}" != "true" ]]; then
  opt_prepare_runtime
  "${CONDA_PYTHON}" "${PROFILE_DIR}/verify_checkpoint.py"
fi

exec bash "${PI05_OPT_DIR}/run_server.sh"
