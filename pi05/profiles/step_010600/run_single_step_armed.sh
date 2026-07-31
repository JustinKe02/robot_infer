#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE_MODE=single_step PROFILE_EXECUTION=armed \
  exec bash "${PROFILE_DIR}/run_robot_client.sh"
