#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

RUN_DIR="${RUN_DIR:-}"
[[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is required" 1>&2; exit 1; }

FOLDS_DIR="${RUN_DIR}/folds"
[[ -d "${FOLDS_DIR}" ]] || { echo "FOLDS_DIR not found: ${FOLDS_DIR}" 1>&2; exit 1; }

MODEL_ID="${MODEL_ID:-}"
SPLIT="${SPLIT:-test}"
OUT_SUBDIR="${OUT_SUBDIR:-inference}"
OUT_BASENAME="${OUT_BASENAME:-pred_test}"
ADAPTER_SUBDIR="${ADAPTER_SUBDIR:-best}"
USE_ADAPTER="${USE_ADAPTER:-true}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${INFER_CUDA_VISIBLE_DEVICES:-0}}"
export NUM_SHARDS="${NUM_SHARDS:-4}"
export MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-12000}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
export TEMPERATURE="${TEMPERATURE:-0.0}"
export TOP_P="${TOP_P:-1.0}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export MAX_EVENTS="${MAX_EVENTS:-64}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export RESUME="${RESUME:-true}"
export ID_KEY="${ID_KEY:-id}"
export INPUT_KEY="${INPUT_KEY:-progress_note}"
export PRED_KEY="${PRED_KEY:-pred_annotation}"
export RAW_PRED_KEY="${RAW_PRED_KEY:-}"

export GUIDELINE_PATH="${GUIDELINE_PATH:-}"
export PROMPT_FILE="${PROMPT_FILE:-}"

export ICL_ENABLED="${ICL_ENABLED:-false}"
export ICL_INPUT_DIR="${ICL_INPUT_DIR:-}"
export ICL_OUTPUT_DIR="${ICL_OUTPUT_DIR:-}"
export N_CONTEXT="${N_CONTEXT:-2}"
export ICL_SEED="${ICL_SEED:-42}"

[[ -n "${MODEL_ID}" ]] || { echo "MODEL_ID is required" 1>&2; exit 1; }
if [[ "${ICL_ENABLED}" == "true" ]]; then
  [[ -n "${ICL_INPUT_DIR}" ]] || { echo "ICL_INPUT_DIR is required when ICL_ENABLED=true" 1>&2; exit 1; }
  [[ -n "${ICL_OUTPUT_DIR}" ]] || { echo "ICL_OUTPUT_DIR is required when ICL_ENABLED=true" 1>&2; exit 1; }
fi

mapfile -t fold_dirs < <(find "${FOLDS_DIR}" -maxdepth 1 -type d -name 'fold=*' | sort -V)
[[ "${#fold_dirs[@]}" -gt 0 ]] || { echo "No folds found under: ${FOLDS_DIR}" 1>&2; exit 1; }

for fold_dir in "${fold_dirs[@]}"; do
  echo "Running fold: ${fold_dir}"
  RUN_DIR="${fold_dir}" \
  MODEL_ID="${MODEL_ID}" \
  INPUT_JSONL="${fold_dir}/datasets_raw/${SPLIT}.jsonl" \
  OUT_DIR="${fold_dir}/${OUT_SUBDIR}" \
  OUT_JSONL="${fold_dir}/${OUT_SUBDIR}/${OUT_BASENAME}_${SPLIT}.jsonl" \
  USE_ADAPTER="${USE_ADAPTER}" \
  ADAPTER_PATH="${fold_dir}/${ADAPTER_SUBDIR}" \
  bash "${SCRIPT_DIR}/inference.sh"
done

echo "All folds inference done. RUN_DIR=${RUN_DIR}"
