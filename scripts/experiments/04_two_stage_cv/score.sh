#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${LAST_RUN_FILE:-${SCRIPT_DIR}/last_run_dir.txt}"
set -a
source "${SCRIPT_DIR}/config/score.env"
set +a
RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" && -f "${LAST_RUN_FILE}" ]]; then RUN_DIR=$(cat "${LAST_RUN_FILE}"); fi
[[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is not set and ${LAST_RUN_FILE} not found." 1>&2; exit 1; }
AGG_SCORE_REL_PATH="${AGG_SCORE_REL_PATH:-score_hungarian_cv.json}"
PER_KEY_REL_PATH="${PER_KEY_REL_PATH:-score_hungarian_cv_per_key.json}"
RUN_DIR="${RUN_DIR}" \
AGG_SCORE_OUT="${RUN_DIR}/${AGG_SCORE_REL_PATH}" \
PER_KEY_OUT="${RUN_DIR}/${PER_KEY_REL_PATH}" \
bash "${REPO_ROOT}/scripts/score/score_lung_cv.sh"
