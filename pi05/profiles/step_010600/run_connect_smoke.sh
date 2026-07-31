#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"
(( $# == 0 )) || rtc_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"
profile_require_confirmation
MODE=single_step EXECUTION=dry_run CONNECT_SMOKE=true \
  exec bash "${PI05_TASK_DIR}/run_client.sh"
