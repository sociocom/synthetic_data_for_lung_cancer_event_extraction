#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

RUN_DIR="${RUN_DIR:-}"
MODEL_ID="${MODEL_ID:-}"
INPUT_JSONL="${INPUT_JSONL:-}"
OUT_SUBDIR="${OUT_SUBDIR:-inference}"
OUT_BASENAME="${OUT_BASENAME:-pred_test}"
SPLIT="${SPLIT:-test}"
OUT_DIR="${OUT_DIR:-}"
OUT_JSONL="${OUT_JSONL:-}"

GUIDELINE_PATH="${GUIDELINE_PATH:-}"
PROMPT_FILE="${PROMPT_FILE:-}"

USE_ADAPTER="${USE_ADAPTER:-true}"
ADAPTER_PATH="${ADAPTER_PATH:-}"

ICL_ENABLED="${ICL_ENABLED:-false}"
ICL_INPUT_DIR="${ICL_INPUT_DIR:-}"
ICL_OUTPUT_DIR="${ICL_OUTPUT_DIR:-}"
N_CONTEXT="${N_CONTEXT:-2}"
ICL_SEED="${ICL_SEED:-42}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${INFER_CUDA_VISIBLE_DEVICES:-0}}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_INPUT_LENGTH="${MAX_INPUT_LENGTH:-12000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
MAX_EVENTS="${MAX_EVENTS:-64}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RESUME="${RESUME:-true}"

ID_KEY="${ID_KEY:-id}"
INPUT_KEY="${INPUT_KEY:-progress_note}"
PRED_KEY="${PRED_KEY:-pred_annotation}"
RAW_PRED_KEY="${RAW_PRED_KEY:-}"
QUANT="${QUANT:-nf4}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
FINAL_CHANNEL="${FINAL_CHANNEL:-false}"
GPU_HEADROOM_GB="${GPU_HEADROOM_GB:-10}"
INPUT_JSONL_HEAD="${INPUT_JSONL_HEAD:-0}"   # >0: only the first N input rows (NUM_SHARDS<=1 only)
ATTN_IMPL="${ATTN_IMPL:-}"
ATTN_CHUNK="${ATTN_CHUNK:-1024}"
export QUANT REASONING_EFFORT FINAL_CHANNEL GPU_HEADROOM_GB ATTN_IMPL ATTN_CHUNK

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES

[[ -n "${MODEL_ID}" ]] || { echo "MODEL_ID is required" 1>&2; exit 1; }
[[ -n "${INPUT_JSONL}" ]] || { echo "INPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUT_DIR}" ]] || { echo "OUT_DIR is required" 1>&2; exit 1; }
[[ -n "${OUT_JSONL}" ]] || { echo "OUT_JSONL is required" 1>&2; exit 1; }
[[ -f "${INPUT_JSONL}" ]] || { echo "Input JSONL not found: ${INPUT_JSONL}" 1>&2; exit 1; }
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ "${USE_ADAPTER}" == "true" ]]; then
  [[ -n "${ADAPTER_PATH}" ]] || { echo "ADAPTER_PATH is required when USE_ADAPTER=true" 1>&2; exit 1; }
  [[ -d "${ADAPTER_PATH}" ]] || { echo "Adapter dir not found: ${ADAPTER_PATH}" 1>&2; exit 1; }
  EXTRA_ARGS+=(--use_adapter --adapter_path "${ADAPTER_PATH}")
else
  EXTRA_ARGS+=(--no_adapter)
fi

if [[ -n "${PROMPT_FILE}" ]]; then
  EXTRA_ARGS+=(--prompt_file "${PROMPT_FILE}")
fi
if [[ -n "${GUIDELINE_PATH}" ]]; then
  EXTRA_ARGS+=(--guideline_path "${GUIDELINE_PATH}")
fi
if [[ "${ICL_ENABLED}" == "true" ]]; then
  [[ -n "${ICL_INPUT_DIR}" ]] || { echo "ICL_INPUT_DIR is required when ICL_ENABLED=true" 1>&2; exit 1; }
  [[ -n "${ICL_OUTPUT_DIR}" ]] || { echo "ICL_OUTPUT_DIR is required when ICL_ENABLED=true" 1>&2; exit 1; }
  EXTRA_ARGS+=(
    --icl_input_dir "${ICL_INPUT_DIR}"
    --icl_output_dir "${ICL_OUTPUT_DIR}"
    --n_context "${N_CONTEXT}"
    --icl_seed "${ICL_SEED}"
  )
fi
if [[ "${RESUME}" == "true" ]]; then
  EXTRA_ARGS+=(--resume)
fi

run_one() {
  local start="$1"
  local end="$2"
  local out="$3"

  uv run python -m inferpipe.cli.run_inference \
    --model_id "${MODEL_ID}" \
    --input_jsonl "${INPUT_JSONL}" \
    --output_jsonl "${out}" \
    --id_key "${ID_KEY}" \
    --input_key "${INPUT_KEY}" \
    --pred_key "${PRED_KEY}" \
    --max_input_length "${MAX_INPUT_LENGTH}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --repetition_penalty "${REPETITION_PENALTY}" \
    --max_events "${MAX_EVENTS}" \
    --batch_size "${BATCH_SIZE}" \
    --quant "${QUANT}" \
    --start "${start}" \
    --end "${end}" \
    "${EXTRA_ARGS[@]}"
}

if [[ -n "${RAW_PRED_KEY}" ]]; then
  EXTRA_ARGS+=(--raw_pred_key "${RAW_PRED_KEY}")
fi

if [[ "${NUM_SHARDS}" -le 1 ]]; then
  run_one 0 "${INPUT_JSONL_HEAD}" "${OUT_JSONL}"
  echo "Done: ${OUT_JSONL}"
  exit 0
fi

TOTAL_LINES=$(grep -cve '^[[:space:]]*$' "${INPUT_JSONL}" || true)
[[ "${TOTAL_LINES}" -gt 0 ]] || { echo "No lines in input: ${INPUT_JSONL}" 1>&2; exit 1; }
LINES_PER_SHARD=$(( (TOTAL_LINES + NUM_SHARDS - 1) / NUM_SHARDS ))

pids=()
for (( shard=0; shard<NUM_SHARDS; shard++ )); do
  start=$(( shard * LINES_PER_SHARD ))
  end=$(( (shard + 1) * LINES_PER_SHARD ))
  [[ "${start}" -lt "${TOTAL_LINES}" ]] || continue
  [[ "${end}" -le "${TOTAL_LINES}" ]] || end="${TOTAL_LINES}"

  (
    shard_out="${OUT_DIR}/${OUT_BASENAME}_${SPLIT}.shard${shard}.jsonl"
    run_one "${start}" "${end}" "${shard_out}"
  ) &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  wait "${pid}" || fail=1
done
[[ "${fail}" -eq 0 ]] || { echo "One or more shards failed." 1>&2; exit 1; }

mapfile -t shard_files < <(ls -1 "${OUT_DIR}"/"${OUT_BASENAME}"_"${SPLIT}".shard*.jsonl 2>/dev/null | sort -V)
[[ "${#shard_files[@]}" -gt 0 ]] || { echo "No shard outputs found to merge." 1>&2; exit 1; }
cat "${shard_files[@]}" > "${OUT_JSONL}"

echo "Merged: ${OUT_JSONL}"
