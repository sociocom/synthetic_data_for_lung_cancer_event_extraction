#!/usr/bin/env bash
# Usage: MODEL_CFG=gptoss|weblab [CUDA_VISIBLE_DEVICES=...] bash infer.sh [--dry_run]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"
MODEL_CFG="${MODEL_CFG:?set MODEL_CFG=gptoss|weblab}"

set -a
source "${SCRIPT_DIR}/config/base.env"
source "${SCRIPT_DIR}/config/${MODEL_CFG}.env"
set +a

if [[ "${1:-}" == "--dry_run" ]]; then
  # 2 documents only, into a scratch run dir, to check channel parsing / JSON validity / speed
  export INPUT_JSONL_HEAD=2
  RUN_NAME="${RUN_NAME}_dryrun"
fi

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${REPO_ROOT}/results/llm_baselines/${RUN_NAME}_${TS}"
fi
printf '%s\n' "${RUN_DIR}" > "${LAST_RUN_FILE}"
mkdir -p "${RUN_DIR}"
env | grep -E '^(MODEL_ID|RUN_NAME|QUANT|REASONING_EFFORT|FINAL_CHANNEL|MAX_NEW_TOKENS|MAX_INPUT_LENGTH|N_CONTEXT|ICL_SEED|CUDA_VISIBLE_DEVICES)=' > "${RUN_DIR}/run_env.txt"

RUN_DIR="${RUN_DIR}" \
OUT_DIR="${RUN_DIR}/${OUT_SUBDIR}" \
OUT_JSONL="${RUN_DIR}/${OUT_SUBDIR}/${OUT_BASENAME}_${SPLIT}.jsonl" \
bash "${REPO_ROOT}/scripts/infer/inference.sh"
