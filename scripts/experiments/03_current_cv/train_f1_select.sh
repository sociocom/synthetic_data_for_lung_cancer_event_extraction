#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"

set -a
source "${SCRIPT_DIR}/config/train.env"
# Keep more checkpoints, then select by validation F1 in f1_select.sh.
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT_F1_SELECT:-0}"
TOPK_BY_EVAL_LOSS="${TOPK_BY_EVAL_LOSS:-4}"
KEEP_RECENT_N="${KEEP_RECENT_N:-1}"
PRUNE_PER_FOLD_DURING_TRAIN="${PRUNE_PER_FOLD_DURING_TRAIN:-true}"
EXP_NAME="${EXP_NAME_F1_SELECT:-${EXP_NAME}_f1select}"
set +a

LAST_RUN_FILE="${LAST_RUN_FILE}" bash "${REPO_ROOT}/scripts/train/train_sft_cv.sh"
