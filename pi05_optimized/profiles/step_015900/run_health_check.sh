#!/usr/bin/env bash
set -euo pipefail

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${PROFILE_DIR}/common.sh"

(( $# == 0 )) || opt_die "configure ${PROFILE_LABEL} through environment variables, not CLI arguments"

server_host="${JZ_PI05_OPT_SERVER_HOST:-127.0.0.1}"
server_port="${JZ_PI05_OPT_SERVER_PORT:-18088}"
exec "${CONDA_PYTHON}" "${PI05_OPT_DIR}/run_client.py" \
  --health-only \
  --server-url="http://${server_host}:${server_port}"
