# MODEL0_ID=/path/to/models/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.5 \
# MODEL0_TAG=llama31 \
# MODEL1_ID=/path/to/models/tokyotech-llm/Qwen3-Swallow-8B-SFT-v0.2 \
# MODEL1_TAG=qwen3 \
# GPU0=0 \
# GPU1=1 \
# RUN_BASE_DIR=/path/to/results/sft \
# INFER_NUM_SHARDS=8 \
# bash run_exp02_exp04_dual_models.sh

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ONE_MODEL_SH="${SCRIPT_DIR}/run_exp02_exp04_model.sh"

MODEL0_ID="${MODEL0_ID:-}"
MODEL1_ID="${MODEL1_ID:-}"
[[ -n "${MODEL0_ID}" ]] || { echo "MODEL0_ID is required" 1>&2; exit 1; }
[[ -n "${MODEL1_ID}" ]] || { echo "MODEL1_ID is required" 1>&2; exit 1; }

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
INFER_NUM_SHARDS="${INFER_NUM_SHARDS:-4}"
RUN_BASE_DIR="${RUN_BASE_DIR:-/path/to/results}"
TS="${TS:-$(date +"%Y%m%d_%H%M%S")}" 

sanitize_tag() {
  basename "$1" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_'
}

TAG0="${MODEL0_TAG:-$(sanitize_tag "${MODEL0_ID}")}"
TAG1="${MODEL1_TAG:-$(sanitize_tag "${MODEL1_ID}")}"

run_one() {
  local model_id="$1"
  local tag="$2"
  local gpu="$3"
  local prefix="$4"

  MODEL_ID="${model_id}" \
  MODEL_TAG="${tag}" \
  RUN_BASE_DIR="${RUN_BASE_DIR}" \
  TS="${TS}" \
  TRAIN_CUDA_VISIBLE_DEVICES="${gpu}" \
  INFER_CUDA_VISIBLE_DEVICES="${gpu}" \
  INFER_NUM_SHARDS="${INFER_NUM_SHARDS}" \
  bash "${ONE_MODEL_SH}" \
    > >(sed "s/^/[${prefix}] /") \
    2> >(sed "s/^/[${prefix}] /" >&2)
}

run_one "${MODEL0_ID}" "${TAG0}" "${GPU0}" "model0" &
pid0=$!
run_one "${MODEL1_ID}" "${TAG1}" "${GPU1}" "model1" &
pid1=$!

rc=0
wait "${pid0}" || rc=1
wait "${pid1}" || rc=1

exit "${rc}"
