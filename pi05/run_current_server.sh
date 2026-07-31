#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CURRENT_POLICY="$(realpath "${SCRIPT_DIR}/checkpoints/current/pretrained_model")"
CURRENT_STEP="$(basename "$(dirname "${CURRENT_POLICY}")")"

case "${CURRENT_STEP}" in
  010600)
    exec bash "${SCRIPT_DIR}/profiles/step_010600/run_policy_server.sh" "$@"
    ;;
  *)
    echo "[tk_infer/pi05] No audited server profile is registered for current step ${CURRENT_STEP}" >&2
    exit 2
    ;;
esac
