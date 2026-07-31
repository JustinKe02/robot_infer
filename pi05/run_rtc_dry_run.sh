#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=rtc EXECUTION=dry_run \
  bash "${SCRIPT_DIR}/run_client.sh" "$@"

