#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

set -a
source "${SCRIPT_DIR}/config.env"
set +a

ROOT_RUN_DIR="${ROOT_RUN_DIR:-${OUTPUT_BASE_DIR}/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
LAST_RUN_FILE="${LAST_RUN_FILE:-${SCRIPT_DIR}/last_run_dir.txt}"
STAGE="${STAGE:-all}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"

mkdir -p "${ROOT_RUN_DIR}"
printf '%s\n' "${ROOT_RUN_DIR}" > "${LAST_RUN_FILE}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"
mkdir -p "${UV_CACHE_DIR}"

run_cmd() {
  echo "+ $*"
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

make_eval_jsonl() {
  local out="$1"
  if [[ "${SKIP_IF_EXISTS}" == "true" && -f "${out}" ]]; then
    return 0
  fi
  run_cmd uv run python -m src.make_paired_dataset_jsonl \
    --progress_jsonl "${MOCK_PROGRESS_JSONL}" \
    --annotations_jsonl "${MOCK_ANNOTATIONS_JSONL}" \
    --output_jsonl "${out}"
}

make_folds_json() {
  local seed="$1"
  local out="$2"
  if [[ "${SKIP_IF_EXISTS}" == "true" && -f "${out}" ]]; then
    return 0
  fi
  run_cmd uv run python -m src.prepare_multilabel_cv_folds \
    --progress_jsonl "${MOCK_PROGRESS_JSONL}" \
    --annotations_jsonl "${MOCK_ANNOTATIONS_JSONL}" \
    --output_json "${out}" \
    --num_folds "${NUM_FOLDS}" \
    --seed "${seed}"
}

train_ft_synthetic() {
  local seed="$1"
  local run_dir="$2"
  if [[ "${SKIP_IF_EXISTS}" == "true" && -d "${run_dir}/best" ]]; then
    return 0
  fi
  run_cmd env \
    MODEL_ID="${MODEL_ID}" \
    PROGRESS_JSONL="${SYNTH_PROGRESS_JSONL}" \
    ANNOTATIONS_JSONL="${SYNTH_ANNOTATIONS_JSONL}" \
    GUIDELINE_PATH="${GUIDELINE_PATH}" \
    OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
    RUN_DIR="${run_dir}" \
    SEED="${seed}" \
    VAL_RATIO=0 \
    TEST_RATIO=0 \
    EVAL_STEPS="${SYNTH_EVAL_STEPS}" \
    NUM_TRAIN_EPOCHS="${SYNTH_NUM_TRAIN_EPOCHS}" \
    MAX_LENGTH="${SYNTH_MAX_LENGTH}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${SYNTH_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    PER_DEVICE_EVAL_BATCH_SIZE="${SYNTH_PER_DEVICE_EVAL_BATCH_SIZE}" \
    GRADIENT_ACCUMULATION_STEPS="${SYNTH_GRADIENT_ACCUMULATION_STEPS}" \
    LEARNING_RATE="${SYNTH_LEARNING_RATE}" \
    WARMUP_RATIO="${SYNTH_WARMUP_RATIO}" \
    SAVE_TOTAL_LIMIT="${SYNTH_SAVE_TOTAL_LIMIT}" \
    EARLY_STOPPING_PATIENCE=99999 \
    EARLY_STOPPING_THRESHOLD=0 \
    MAX_SAMPLES=0 \
    bash "${REPO_ROOT}/scripts/train/train_sft.sh"
}

infer_score_ft_synthetic() {
  local run_dir="$1"
  local eval_jsonl="$2"
  local pred="${run_dir}/inference/pred_test_test.jsonl"
  local score="${run_dir}/score.json"
  local per_key="${run_dir}/score_per_key.json"
  if [[ "${SKIP_IF_EXISTS}" == "true" && -f "${score}" && -f "${per_key}" ]]; then
    return 0
  fi
  run_cmd env \
    MODEL_ID="${MODEL_ID}" \
    INPUT_JSONL="${eval_jsonl}" \
    OUT_DIR="${run_dir}/inference" \
    OUT_JSONL="${pred}" \
    OUT_SUBDIR=inference \
    OUT_BASENAME=pred_test \
    SPLIT=test \
    USE_ADAPTER=true \
    ADAPTER_PATH="${run_dir}/best" \
    GUIDELINE_PATH="${GUIDELINE_PATH}" \
    NUM_SHARDS="${INFER_NUM_SHARDS}" \
    MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS}" \
    MAX_INPUT_LENGTH="${INFER_MAX_INPUT_LENGTH}" \
    BATCH_SIZE="${INFER_BATCH_SIZE}" \
    TEMPERATURE="${INFER_TEMPERATURE}" \
    RAW_PRED_KEY=raw_pred_annotation \
    bash "${REPO_ROOT}/scripts/infer/inference.sh"

  run_cmd env PROCEDUREDIAGNOSIS_MATCH_MODE="${PROCEDUREDIAGNOSIS_MATCH_MODE}" \
    bash "${REPO_ROOT}/scripts/score/score_lung.sh" \
    "${eval_jsonl}" "${pred}" "${score}" "${per_key}"
}

train_cv_condition() {
  local seed="$1"
  local run_dir="$2"
  local folds_json="$3"
  local init_adapter_path="${4:-}"
  local num_train_epochs="${5:-${MOCK_NUM_TRAIN_EPOCHS}}"
  if [[ "${SKIP_IF_EXISTS}" == "true" && -d "${run_dir}/folds/fold=0/best" ]]; then
    return 0
  fi
  run_cmd env \
    MODEL_ID="${MODEL_ID}" \
    PROGRESS_JSONL="${MOCK_PROGRESS_JSONL}" \
    ANNOTATIONS_JSONL="${MOCK_ANNOTATIONS_JSONL}" \
    GUIDELINE_PATH="${GUIDELINE_PATH}" \
    OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR}" \
    RUN_DIR="${run_dir}" \
    INIT_ADAPTER_PATH="${init_adapter_path}" \
    NUM_FOLDS="${NUM_FOLDS}" \
    CV_SPLIT_SEED="${seed}" \
    CV_FOLDS_JSON="${folds_json}" \
    CV_VAL_SIZE="${MOCK_CV_VAL_SIZE}" \
    CV_VAL_SEED_BASE="${seed}" \
    SEED="${seed}" \
    VAL_RATIO=0 \
    TEST_RATIO=0 \
    EVAL_STEPS="${MOCK_EVAL_STEPS}" \
    NUM_TRAIN_EPOCHS="${num_train_epochs}" \
    MAX_LENGTH="${MOCK_MAX_LENGTH}" \
    PER_DEVICE_TRAIN_BATCH_SIZE="${MOCK_PER_DEVICE_TRAIN_BATCH_SIZE}" \
    PER_DEVICE_EVAL_BATCH_SIZE="${MOCK_PER_DEVICE_EVAL_BATCH_SIZE}" \
    GRADIENT_ACCUMULATION_STEPS="${MOCK_GRADIENT_ACCUMULATION_STEPS}" \
    LEARNING_RATE="${MOCK_LEARNING_RATE}" \
    WARMUP_RATIO="${MOCK_WARMUP_RATIO}" \
    SAVE_TOTAL_LIMIT="${MOCK_SAVE_TOTAL_LIMIT}" \
    EARLY_STOPPING_PATIENCE=99999 \
    EARLY_STOPPING_THRESHOLD=0 \
    MAX_SAMPLES=0 \
    bash "${REPO_ROOT}/scripts/train/train_sft_cv.sh"
}

infer_score_cv_condition() {
  local run_dir="$1"
  if [[ "${SKIP_IF_EXISTS}" == "true" && -f "${run_dir}/score_hungarian_cv.json" && -f "${run_dir}/score_hungarian_cv_per_key.json" ]]; then
    return 0
  fi
  run_cmd env \
    RUN_DIR="${run_dir}" \
    MODEL_ID="${MODEL_ID}" \
    SPLIT=test \
    OUT_SUBDIR=inference \
    OUT_BASENAME=pred_test \
    USE_ADAPTER=true \
    ADAPTER_SUBDIR=best \
    GUIDELINE_PATH="${GUIDELINE_PATH}" \
    NUM_SHARDS="${INFER_NUM_SHARDS}" \
    MAX_NEW_TOKENS="${INFER_MAX_NEW_TOKENS}" \
    MAX_INPUT_LENGTH="${INFER_MAX_INPUT_LENGTH}" \
    BATCH_SIZE="${INFER_BATCH_SIZE}" \
    TEMPERATURE="${INFER_TEMPERATURE}" \
    RAW_PRED_KEY=raw_pred_annotation \
    bash "${REPO_ROOT}/scripts/infer/inference_cv.sh"

  run_cmd env \
    RUN_DIR="${run_dir}" \
    AGG_SCORE_OUT="${run_dir}/score_hungarian_cv.json" \
    PER_KEY_OUT="${run_dir}/score_hungarian_cv_per_key.json" \
    PROCEDUREDIAGNOSIS_MATCH_MODE="${PROCEDUREDIAGNOSIS_MATCH_MODE}" \
    bash "${REPO_ROOT}/scripts/score/score_lung_cv.sh"
}

aggregate_scores() {
  run_cmd uv run python -m src.analyze_multiseed_scores \
    --root_dir "${ROOT_RUN_DIR}" \
    --output_dir "${ROOT_RUN_DIR}/summary"
}

case "${STAGE}" in
  all|train|infer_score|aggregate) ;;
  *) echo "Unknown STAGE=${STAGE}. Use all|train|infer_score|aggregate" 1>&2; exit 1 ;;
esac

EVAL_JSONL="${ROOT_RUN_DIR}/mock_eval_paired.jsonl"
make_eval_jsonl "${EVAL_JSONL}"

for seed in ${SEEDS}; do
  echo "================ seed=${seed} ================"
  SEED_DIR="${ROOT_RUN_DIR}/seed=${seed}"
  mkdir -p "${SEED_DIR}"
  FOLDS_JSON="${SEED_DIR}/mock_multilabel_folds.json"
  make_folds_json "${seed}" "${FOLDS_JSON}"

  FT_SYN_DIR="${SEED_DIR}/ft_synthetic"
  FT_MOCK_DIR="${SEED_DIR}/ft_mock"
  FT_ALL_DIR="${SEED_DIR}/ft_all"

  if [[ "${STAGE}" == "all" || "${STAGE}" == "train" ]]; then
    train_ft_synthetic "${seed}" "${FT_SYN_DIR}"
    train_cv_condition "${seed}" "${FT_MOCK_DIR}" "${FOLDS_JSON}" "" "${MOCK_NUM_TRAIN_EPOCHS}"
    train_cv_condition "${seed}" "${FT_ALL_DIR}" "${FOLDS_JSON}" "${FT_SYN_DIR}/best" "${ALL_STAGE2_NUM_TRAIN_EPOCHS}"
  fi

  if [[ "${STAGE}" == "all" || "${STAGE}" == "infer_score" ]]; then
    infer_score_ft_synthetic "${FT_SYN_DIR}" "${EVAL_JSONL}"
    infer_score_cv_condition "${FT_MOCK_DIR}"
    infer_score_cv_condition "${FT_ALL_DIR}"
  fi
done

if [[ "${STAGE}" == "all" || "${STAGE}" == "aggregate" ]]; then
  aggregate_scores
fi

echo "ROOT_RUN_DIR=${ROOT_RUN_DIR}"
