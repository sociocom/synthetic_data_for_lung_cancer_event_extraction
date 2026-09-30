#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-}"
PROGRESS_JSONL="${PROGRESS_JSONL:-}"
ANNOTATIONS_JSONL="${ANNOTATIONS_JSONL:-}"
GUIDELINE_PATH="${GUIDELINE_PATH:-}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"
EXP_NAME="${EXP_NAME:-exp_train}"
RUN_DIR="${RUN_DIR:-}"
LAST_RUN_FILE="${LAST_RUN_FILE:-}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-}"

SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0}"
EVAL_STEPS="${EVAL_STEPS:-20}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-5}"
MAX_LENGTH="${MAX_LENGTH:-4600}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-8}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

cd "${REPO_ROOT}"

[[ -n "${MODEL_ID}" ]] || { echo "MODEL_ID is required" 1>&2; exit 1; }
[[ -n "${PROGRESS_JSONL}" ]] || { echo "PROGRESS_JSONL is required" 1>&2; exit 1; }
[[ -n "${ANNOTATIONS_JSONL}" ]] || { echo "ANNOTATIONS_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUTPUT_BASE_DIR}" ]] || { echo "OUTPUT_BASE_DIR is required" 1>&2; exit 1; }

if [[ -z "${RUN_DIR}" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}_${TS}"
fi
mkdir -p "${RUN_DIR}"
if [[ -n "${LAST_RUN_FILE}" ]]; then
  printf '%s\n' "${RUN_DIR}" > "${LAST_RUN_FILE}"
fi

echo "Run dir: ${RUN_DIR}"

OUTPUT_DIR="${RUN_DIR}"
if [[ -n "${OUTPUT_SUBDIR}" ]]; then
  OUTPUT_DIR="${RUN_DIR}/${OUTPUT_SUBDIR}"
  mkdir -p "${OUTPUT_DIR}"
fi

EXTRA_ARGS=()
if [[ -n "${INIT_ADAPTER_PATH:-}" ]]; then
  EXTRA_ARGS+=(--init_adapter_path "${INIT_ADAPTER_PATH}")
fi

uv run python -m trainpipe.cli.train_sft train \
  --model_id "${MODEL_ID}" \
  --progress_jsonl "${PROGRESS_JSONL}" \
  --annotations_jsonl "${ANNOTATIONS_JSONL}" \
  --guideline_path "${GUIDELINE_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --val_ratio "${VAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  --eval_steps "${EVAL_STEPS}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --max_length "${MAX_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
  --max_samples "${MAX_SAMPLES}" \
  "${EXTRA_ARGS[@]}"
