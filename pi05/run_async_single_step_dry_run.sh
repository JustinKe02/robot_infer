#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE=async_single_step EXECUTION=dry_run \
  bash "${SCRIPT_DIR}/run_client.sh" "$@"
