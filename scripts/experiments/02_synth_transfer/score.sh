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
GOLD_JSONL="${GOLD_JSONL}" \
PRED_JSONL="${RUN_DIR}/${PRED_REL_PATH}" \
SCORE_OUT="${RUN_DIR}/${SCORE_REL_PATH}" \
PER_KEY_OUT="${PER_KEY_REL_PATH:+${RUN_DIR}/${PER_KEY_REL_PATH}}" \
bash "${REPO_ROOT}/scripts/score/score_lung.sh"
