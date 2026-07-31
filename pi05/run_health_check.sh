#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=single_step \
EXECUTION=dry_run \
HEALTH_ONLY=true \
  bash "${SCRIPT_DIR}/run_client.sh" "$@"

