#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"

set -a
source "${SCRIPT_DIR}/config/infer.env"
set +a

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" && -f "${LAST_RUN_FILE}" ]]; then
  RUN_DIR=$(cat "${LAST_RUN_FILE}")
fi
[[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is not set and ${LAST_RUN_FILE} not found." 1>&2; exit 1; }

RUN_DIR="${RUN_DIR}" bash "${REPO_ROOT}/scripts/infer/inference_cv.sh"
