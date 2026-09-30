#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

MODEL_ID="${MODEL_ID:-}"
[[ -n "${MODEL_ID}" ]] || { echo "MODEL_ID is required" 1>&2; exit 1; }

# Create a stable, filesystem-safe tag when not explicitly provided.
default_tag="$(basename "${MODEL_ID}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_')"
MODEL_TAG="${MODEL_TAG:-${default_tag}}"

RUN_BASE_DIR="${RUN_BASE_DIR:-/path/to/results}"
TS="${TS:-$(date +"%Y%m%d_%H%M%S")}" 

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-${TRAIN_CUDA_VISIBLE_DEVICES%%,*}}"
INFER_NUM_SHARDS="${INFER_NUM_SHARDS:-4}"
EXP02_MAX_NEW_TOKENS="${EXP02_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS:-}}"
EXP04_MAX_NEW_TOKENS="${EXP04_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS:-}}"

EXP02_PREFIX="${EXP02_PREFIX:-exp02_synth_transfer}"
EXP04_PREFIX="${EXP04_PREFIX:-exp04_two_stage_cv}"

RUN02="${RUN02:-${RUN_BASE_DIR}/${EXP02_PREFIX}_${MODEL_TAG}_${TS}}"
RUN04="${RUN04:-${RUN_BASE_DIR}/${EXP04_PREFIX}_${MODEL_TAG}_${TS}}"
EXP02_LAST_RUN_FILE="${EXP02_LAST_RUN_FILE:-${RUN02}/.meta/last_run_dir.txt}"
EXP04_LAST_RUN_FILE="${EXP04_LAST_RUN_FILE:-${RUN04}/.meta/last_run_dir.txt}"

mkdir -p "$(dirname "${EXP02_LAST_RUN_FILE}")"
mkdir -p "$(dirname "${EXP04_LAST_RUN_FILE}")"

echo "[exp02][train] ${RUN02}"
LAST_RUN_FILE="${EXP02_LAST_RUN_FILE}" \
RUN_DIR="${RUN02}" MODEL_ID="${MODEL_ID}" CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" STAGE=train \
  bash "${REPO_ROOT}/scripts/experiments/02_synth_transfer/run.sh"

echo "[exp02][infer] ${RUN02}"
LAST_RUN_FILE="${EXP02_LAST_RUN_FILE}" \
RUN_DIR="${RUN02}" MODEL_ID="${MODEL_ID}" CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES}" NUM_SHARDS="${INFER_NUM_SHARDS}" STAGE=infer \
MAX_NEW_TOKENS="${EXP02_MAX_NEW_TOKENS}" \
  bash "${REPO_ROOT}/scripts/experiments/02_synth_transfer/run.sh"

echo "[exp02][score] ${RUN02}"
LAST_RUN_FILE="${EXP02_LAST_RUN_FILE}" \
RUN_DIR="${RUN02}" STAGE=score \
  bash "${REPO_ROOT}/scripts/experiments/02_synth_transfer/run.sh"

echo "[exp04][stage2] ${RUN04}"
LAST_RUN_FILE="${EXP04_LAST_RUN_FILE}" \
RUN_DIR="${RUN04}" MODEL_ID="${MODEL_ID}" INIT_ADAPTER_PATH="${RUN02}/best" CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" STAGE=stage2 \
  bash "${REPO_ROOT}/scripts/experiments/04_two_stage_cv/run.sh"

echo "[exp04][infer] ${RUN04}"
LAST_RUN_FILE="${EXP04_LAST_RUN_FILE}" \
RUN_DIR="${RUN04}" MODEL_ID="${MODEL_ID}" CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES}" NUM_SHARDS="${INFER_NUM_SHARDS}" STAGE=infer \
MAX_NEW_TOKENS="${EXP04_MAX_NEW_TOKENS}" \
  bash "${REPO_ROOT}/scripts/experiments/04_two_stage_cv/run.sh"

echo "[exp04][score] ${RUN04}"
LAST_RUN_FILE="${EXP04_LAST_RUN_FILE}" \
RUN_DIR="${RUN04}" STAGE=score \
  bash "${REPO_ROOT}/scripts/experiments/04_two_stage_cv/run.sh"

echo "Done"
echo "  model: ${MODEL_ID}"
echo "  exp02: ${RUN02}"
echo "  exp04: ${RUN04}"
