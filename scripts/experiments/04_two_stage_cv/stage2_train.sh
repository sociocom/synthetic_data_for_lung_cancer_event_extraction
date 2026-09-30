#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${LAST_RUN_FILE:-${SCRIPT_DIR}/last_run_dir.txt}"
set -a
source "${SCRIPT_DIR}/config/stage2_train.env"
set +a
[[ -n "${INIT_ADAPTER_PATH:-}" ]] || { echo "INIT_ADAPTER_PATH is required" 1>&2; exit 1; }
[[ -d "${INIT_ADAPTER_PATH}" ]] || { echo "Stage1 adapter not found: ${INIT_ADAPTER_PATH}" 1>&2; exit 1; }

# Optional mode: keep many checkpoints and select by validation F1 later.
if [[ "${STAGE2_F1SELECT:-false}" == "true" ]]; then
  SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT_F1_SELECT:-0}"
  TOPK_BY_EVAL_LOSS="${TOPK_BY_EVAL_LOSS:-4}"
  KEEP_RECENT_N="${KEEP_RECENT_N:-1}"
  PRUNE_PER_FOLD_DURING_TRAIN="${PRUNE_PER_FOLD_DURING_TRAIN:-true}"
  EXP_NAME="${EXP_NAME_F1_SELECT:-${EXP_NAME}_f1select}"
fi

LAST_RUN_FILE="${LAST_RUN_FILE}" \
INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH}" \
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
TOPK_BY_EVAL_LOSS="${TOPK_BY_EVAL_LOSS:-4}" \
KEEP_RECENT_N="${KEEP_RECENT_N:-1}" \
PRUNE_PER_FOLD_DURING_TRAIN="${PRUNE_PER_FOLD_DURING_TRAIN:-false}" \
EXP_NAME="${EXP_NAME}" \
bash "${REPO_ROOT}/scripts/train/train_sft_cv.sh"
