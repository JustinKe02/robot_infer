#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_MODE=rtc PROFILE_EXECUTION=dry_run PROFILE_JOINT_DELTA_MODE=bypass PROFILE_RTC_RUN_MODE=qualification \
  exec bash "${PROFILE_DIR}/run_robot_client.sh"
