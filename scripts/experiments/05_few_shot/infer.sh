#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"

set -a
source "${SCRIPT_DIR}/config/infer.env"
set +a

RUN_NAME="${RUN_NAME:-few_shot}"
RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${REPO_ROOT}/results/few_shot/${RUN_NAME}_${TS}"
fi
printf '%s\n' "${RUN_DIR}" > "${LAST_RUN_FILE}"

RUN_DIR="${RUN_DIR}" \
OUT_DIR="${RUN_DIR}/${OUT_SUBDIR}" \
OUT_JSONL="${RUN_DIR}/${OUT_SUBDIR}/${OUT_BASENAME}_${SPLIT}.jsonl" \
bash "${REPO_ROOT}/scripts/infer/inference.sh"
