#!/usr/bin/env bash

# RUN_DIR=/path/to/results \
# TARGET_EPOCHS_CSV=8 \
# INFER_CUDA_VISIBLE_DEVICES=2 \
# INFER_NUM_SHARDS=2 \
# bash infer_score_captured_epochs.sh

# strict
# GOLD_JSONL=/path/to/this/repository/data/processed/simulation/simu_annotation_noae.jsonl \
# PRED_JSONL=/path/to/results \
# SCORE_OUT=/path/to/results \
# PER_KEY_OUT=/path/to/results \
# PROCEDUREDIAGNOSIS_MATCH_MODE=strict \
# bash scripts/score/score_lung.sh

# compat (pred_child_of_gold)
# GOLD_JSONL=/path/to/this/repository/data/processed/simulation/simu_annotation_noae.jsonl \
# PRED_JSONL=/path/to/results \
# SCORE_OUT=/path/to/results \
# PER_KEY_OUT=/path/to/results \
# PROCEDUREDIAGNOSIS_MATCH_MODE=pred_child_of_gold \
# bash scripts/score/score_lung.sh

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
INFER_SH="${REPO_ROOT}/scripts/infer/inference.sh"
SCORE_SH="${REPO_ROOT}/scripts/score/score_lung.sh"

# Load exp02 defaults (infer + score settings).
set -a
export REPO_ROOT
source "${SCRIPT_DIR}/config/infer.env"
source "${SCRIPT_DIR}/config/score.env"
set +a

RUN_DIR="${RUN_DIR:-}"
[[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is required." 1>&2; exit 1; }
[[ -d "${RUN_DIR}" ]] || { echo "RUN_DIR not found: ${RUN_DIR}" 1>&2; exit 1; }

TARGET_EPOCHS_CSV="${TARGET_EPOCHS_CSV:-5,8,11,14}"
INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-2}"
INFER_NUM_SHARDS="${INFER_NUM_SHARDS:-4}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
PROCEDUREDIAGNOSIS_MATCH_MODE="${PROCEDUREDIAGNOSIS_MATCH_MODE:-}"

IFS=',' read -r -a TARGET_EPOCHS <<< "${TARGET_EPOCHS_CSV}"
if [[ "${#TARGET_EPOCHS[@]}" -eq 0 ]]; then
  echo "TARGET_EPOCHS_CSV is empty." 1>&2
  exit 1
fi

PROBE_ROOT="${RUN_DIR}/epoch_probes"
[[ -d "${PROBE_ROOT}" ]] || { echo "Probe root not found: ${PROBE_ROOT}" 1>&2; exit 1; }

run_one_epoch_probe() {
  local target_epoch="$1"
  local tag
  tag=$(printf "%s" "${target_epoch}" | sed 's/[.]/p/g')

  local probe_dir="${PROBE_ROOT}/epoch_${tag}"
  local adapter_dir="${probe_dir}/adapter"
  local out_subdir="inference_transfer_ep${tag}"
  local out_dir="${RUN_DIR}/${out_subdir}"
  local out_jsonl="${out_dir}/${OUT_BASENAME}_${SPLIT}.jsonl"
  local score_out="${out_dir}/score_hungarian_transfer.json"
  local per_key_out="${out_dir}/score_hungarian_transfer_per_key.json"

  if [[ ! -f "${probe_dir}/captured.ok" ]]; then
    echo "[skip] epoch ${target_epoch}: capture not found (${probe_dir}/captured.ok)"
    return 0
  fi
  [[ -f "${adapter_dir}/adapter_config.json" ]] || { echo "[skip] epoch ${target_epoch}: adapter_config.json missing"; return 0; }
  [[ -f "${adapter_dir}/adapter_model.safetensors" ]] || { echo "[skip] epoch ${target_epoch}: adapter_model.safetensors missing"; return 0; }

  if [[ "${SKIP_IF_EXISTS}" == "true" && -f "${score_out}" && -f "${per_key_out}" ]]; then
    echo "[skip] epoch ${target_epoch}: score already exists"
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[infer] epoch ${target_epoch} -> ${out_subdir}"
  CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES}" \
  NUM_SHARDS="${INFER_NUM_SHARDS}" \
  MODEL_ID="${MODEL_ID}" \
  INPUT_JSONL="${INPUT_JSONL}" \
  GUIDELINE_PATH="${GUIDELINE_PATH}" \
  USE_ADAPTER=true \
  ADAPTER_PATH="${adapter_dir}" \
  OUT_DIR="${out_dir}" \
  OUT_JSONL="${out_jsonl}" \
  OUT_BASENAME="${OUT_BASENAME}" \
  SPLIT="${SPLIT}" \
  RESUME=true \
  RAW_PRED_KEY="${RAW_PRED_KEY:-}" \
  bash "${INFER_SH}"

  echo "[score] epoch ${target_epoch}"
  GOLD_JSONL="${GOLD_JSONL}" \
  PRED_JSONL="${out_jsonl}" \
  SCORE_OUT="${score_out}" \
  PER_KEY_OUT="${per_key_out}" \
  PROCEDUREDIAGNOSIS_MATCH_MODE="${PROCEDUREDIAGNOSIS_MATCH_MODE}" \
  bash "${SCORE_SH}"
}

echo "[infer] RUN_DIR=${RUN_DIR}"
echo "[infer] TARGET_EPOCHS=${TARGET_EPOCHS_CSV}"
echo "[infer] CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} NUM_SHARDS=${INFER_NUM_SHARDS}"

for t in "${TARGET_EPOCHS[@]}"; do
  run_one_epoch_probe "${t}"
done

echo "Done: ${RUN_DIR}"
for t in "${TARGET_EPOCHS[@]}"; do
  tag=$(printf "%s" "${t}" | sed 's/[.]/p/g')
  out="${RUN_DIR}/inference_transfer_ep${tag}/score_hungarian_transfer.json"
  if [[ -f "${out}" ]]; then
    echo "  epoch ${t}: ${out}"
  else
    echo "  epoch ${t}: no score"
  fi
done
