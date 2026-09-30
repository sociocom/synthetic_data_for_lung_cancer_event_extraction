#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-}"
SIM_PROGRESS_JSONL="${SIM_PROGRESS_JSONL:-}"
SIM_ANNOTATIONS_JSONL="${SIM_ANNOTATIONS_JSONL:-}"
SYNTH_PROGRESS_JSONL="${SYNTH_PROGRESS_JSONL:-}"
SYNTH_ANNOTATIONS_JSONL="${SYNTH_ANNOTATIONS_JSONL:-}"
GUIDELINE_PATH="${GUIDELINE_PATH:-}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"
EXP_NAME="${EXP_NAME:-exp06_mix_cv}"
RUN_DIR="${RUN_DIR:-}"
LAST_RUN_FILE="${LAST_RUN_FILE:-}"

NUM_FOLDS="${NUM_FOLDS:-5}"
CV_SPLIT_SEED="${CV_SPLIT_SEED:-42}"
CV_VAL_SIZE="${CV_VAL_SIZE:-14}"
CV_VAL_SEED_BASE="${CV_VAL_SEED_BASE:-42}"

MIX_SYNTH_RATIO="${MIX_SYNTH_RATIO:-50}"
MIX_SIM_RATIO="${MIX_SIM_RATIO:-50}"
MIX_SEED_BASE="${MIX_SEED_BASE:-42}"

SEED="${SEED:-42}"
EVAL_STEPS="${EVAL_STEPS:-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
MAX_LENGTH="${MAX_LENGTH:-4600}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"

cd "${REPO_ROOT}"

[[ -n "${MODEL_ID}" ]] || { echo "MODEL_ID is required" 1>&2; exit 1; }
[[ -n "${SIM_PROGRESS_JSONL}" ]] || { echo "SIM_PROGRESS_JSONL is required" 1>&2; exit 1; }
[[ -n "${SIM_ANNOTATIONS_JSONL}" ]] || { echo "SIM_ANNOTATIONS_JSONL is required" 1>&2; exit 1; }
[[ -n "${SYNTH_PROGRESS_JSONL}" ]] || { echo "SYNTH_PROGRESS_JSONL is required" 1>&2; exit 1; }
[[ -n "${SYNTH_ANNOTATIONS_JSONL}" ]] || { echo "SYNTH_ANNOTATIONS_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUTPUT_BASE_DIR}" ]] || { echo "OUTPUT_BASE_DIR is required" 1>&2; exit 1; }

if [[ "${NUM_FOLDS}" -lt 2 ]]; then
  echo "NUM_FOLDS must be >= 2" 1>&2
  exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}_${TS}"
fi
mkdir -p "${RUN_DIR}"

STATUS_FILE="${RUN_DIR}/run_status.json"
NOW="$(date -Iseconds)"
printf '{\n  "status": "running",\n  "started_at": "%s"\n}\n' "${NOW}" > "${STATUS_FILE}"

on_err() {
  local now
  now="$(date -Iseconds)"
  printf '{\n  "status": "failed",\n  "failed_at": "%s",\n  "failed_fold": %s\n}\n' "${now}" "${FOLD:-null}" > "${STATUS_FILE}"
}
trap on_err ERR

echo "Run dir: ${RUN_DIR}"
echo "Mix ratio (synth:sim) = ${MIX_SYNTH_RATIO}:${MIX_SIM_RATIO}"

EXTRA_ARGS=()
if [[ -n "${INIT_ADAPTER_PATH:-}" ]]; then
  EXTRA_ARGS+=(--init_adapter_path "${INIT_ADAPTER_PATH}")
fi

for ((FOLD=0; FOLD<NUM_FOLDS; FOLD++)); do
  FOLD_DIR="${RUN_DIR}/folds/fold=${FOLD}"
  DATASETS_DIR="${FOLD_DIR}/datasets_raw"
  mkdir -p "${FOLD_DIR}"

  echo "==============================="
  echo "MIX-CV fold ${FOLD}/${NUM_FOLDS}"
  echo "Output: ${FOLD_DIR}"
  echo "==============================="

  uv run python -m trainpipe.cli.prepare_mix_cv \
    --sim_progress_jsonl "${SIM_PROGRESS_JSONL}" \
    --sim_annotations_jsonl "${SIM_ANNOTATIONS_JSONL}" \
    --synth_progress_jsonl "${SYNTH_PROGRESS_JSONL}" \
    --synth_annotations_jsonl "${SYNTH_ANNOTATIONS_JSONL}" \
    --output_dir "${FOLD_DIR}" \
    --num_folds "${NUM_FOLDS}" \
    --cv_fold "${FOLD}" \
    --cv_split_seed "${CV_SPLIT_SEED}" \
    --cv_val_size "${CV_VAL_SIZE}" \
    --cv_val_seed_base "${CV_VAL_SEED_BASE}" \
    --mix_synth_ratio "${MIX_SYNTH_RATIO}" \
    --mix_sim_ratio "${MIX_SIM_RATIO}" \
    --mix_seed_base "${MIX_SEED_BASE}"

  uv run python -m trainpipe.cli.train_sft train-prepared \
    --model_id "${MODEL_ID}" \
    --train_jsonl "${DATASETS_DIR}/train_sim.jsonl" \
    --validation_jsonl "${DATASETS_DIR}/validation.jsonl" \
    --synth_pool_jsonl "${DATASETS_DIR}/synth_pool.jsonl" \
    --test_jsonl "${DATASETS_DIR}/test.jsonl" \
    --guideline_path "${GUIDELINE_PATH}" \
    --output_dir "${FOLD_DIR}" \
    --seed "${SEED}" \
    --eval_steps "${EVAL_STEPS}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_length "${MAX_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --mix_synth_ratio "${MIX_SYNTH_RATIO}" \
    --mix_sim_ratio "${MIX_SIM_RATIO}" \
    --mix_seed_base "$((MIX_SEED_BASE + FOLD * 1000))" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
    "${EXTRA_ARGS[@]}"
done

trap - ERR
NOW="$(date -Iseconds)"
printf '{\n  "status": "completed",\n  "completed_at": "%s"\n}\n' "${NOW}" > "${STATUS_FILE}"
if [[ -n "${LAST_RUN_FILE}" ]]; then
  printf '%s\n' "${RUN_DIR}" > "${LAST_RUN_FILE}"
fi

echo "Done. All folds saved under: ${RUN_DIR}/folds/"
