#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"

(( $# == 0 )) || rtc_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"
profile_require_confirmation
export REQUIRE_COMPLETE_STEP=false

echo "[tk_infer/pi05/profile] server=${PROFILE_LABEL}"
echo "[tk_infer/pi05/profile] checkpoint=${POLICY_PATH} step=10600/15900 complete=false"
print_only="$(rtc_normalize_bool PRINT_COMMAND_ONLY "${PRINT_COMMAND_ONLY:-false}")"
config_only="$(rtc_normalize_bool CONFIG_ONLY "${CONFIG_ONLY:-false}")"
if [[ "${print_only}" != "true" && "${config_only}" != "true" ]]; then
  rtc_prepare_runtime
  "${CONDA_PYTHON}" "${PROFILE_DIR}/verify_checkpoint.py"
fi
exec bash "${PI05_TASK_DIR}/run_server.sh"
