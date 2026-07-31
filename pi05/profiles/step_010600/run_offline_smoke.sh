#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"
(( $# == 0 )) || rtc_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"
profile_require_confirmation
rtc_prepare_runtime
echo "[tk_infer/pi05/profile] offline smoke; no robot is constructed or connected"
exec "${CONDA_PYTHON}" "${PROFILE_DIR}/offline_smoke.py"
