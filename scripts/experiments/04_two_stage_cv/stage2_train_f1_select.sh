#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Recommended defaults for validation-based F1 checkpoint selection.
# Users can still override these from environment variables at runtime.
export CV_VAL_SIZE="${CV_VAL_SIZE:-14}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
export EVAL_STEPS="${EVAL_STEPS:-5}"
export SAVE_TOTAL_LIMIT_F1_SELECT="${SAVE_TOTAL_LIMIT_F1_SELECT:-0}"
export TOPK_BY_EVAL_LOSS="${TOPK_BY_EVAL_LOSS:-6}"
export KEEP_RECENT_N="${KEEP_RECENT_N:-2}"
export PRUNE_PER_FOLD_DURING_TRAIN="${PRUNE_PER_FOLD_DURING_TRAIN:-false}"

STAGE2_F1SELECT=true bash "${SCRIPT_DIR}/stage2_train.sh"
