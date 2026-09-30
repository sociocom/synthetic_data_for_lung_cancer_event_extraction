#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"

set -a
source "${SCRIPT_DIR}/config/stage2_train.env"
set +a

SWEEP_GRID="${SWEEP_GRID:-1e-4:5 1e-4:2 8e-5:3 5e-5:3}"
SWEEP_STAGE="${SWEEP_STAGE:-all}" # train|all
SWEEP_NAME="${SWEEP_NAME:-exp04_two_stage_cv_sweep}"
COMPAT_MATCH_MODE="${COMPAT_MATCH_MODE:-pred_child_of_gold}"

case "${SWEEP_STAGE}" in
  train|all) ;;
  *)
    echo "Unknown SWEEP_STAGE=${SWEEP_STAGE}. Use train|all" 1>&2
    exit 1
    ;;
esac

BASE_EXP_NAME="${EXP_NAME:-exp04_two_stage_cv}"
GRID_NORM="${SWEEP_GRID//,/ }"
SWEEP_TS="$(date +"%Y%m%d_%H%M%S")"
SWEEP_LOG_DIR="${OUTPUT_BASE_DIR}/${SWEEP_NAME}_${SWEEP_TS}"
SWEEP_RUNS_TSV="${SWEEP_LOG_DIR}/runs.tsv"
mkdir -p "${SWEEP_LOG_DIR}"
printf 'learning_rate\tnum_train_epochs\trun_dir\tstage\n' > "${SWEEP_RUNS_TSV}"

echo "Sweep output: ${SWEEP_LOG_DIR}"
echo "Grid (lr:epochs): ${GRID_NORM}"
echo "Stage: ${SWEEP_STAGE}"

LAST_COMPLETED_RUN_DIR=""

for spec in ${GRID_NORM}; do
  lr="${spec%%:*}"
  epochs="${spec##*:}"
  if [[ -z "${lr}" || -z "${epochs}" || "${lr}" == "${epochs}" ]]; then
    echo "Invalid spec: ${spec}. Expected <lr>:<epochs>" 1>&2
    exit 1
  fi
  if ! [[ "${epochs}" =~ ^[0-9]+$ ]]; then
    echo "Invalid epochs in spec: ${spec}" 1>&2
    exit 1
  fi

  lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  export LEARNING_RATE="${lr}"
  export NUM_TRAIN_EPOCHS="${epochs}"
  export EXP_NAME="${BASE_EXP_NAME}_lr${lr_tag}_ep${NUM_TRAIN_EPOCHS}"

  spec_last_run_file="${SWEEP_LOG_DIR}/last_run_lr${lr_tag}_ep${NUM_TRAIN_EPOCHS}.txt"
  rm -f "${spec_last_run_file}"

  echo "========================================"
  echo "SWEEP RUN lr=${LEARNING_RATE} epochs=${NUM_TRAIN_EPOCHS}"
  echo "EXP_NAME=${EXP_NAME}"
  echo "========================================"

  LAST_RUN_FILE="${spec_last_run_file}" \
  EXP_NAME="${EXP_NAME}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
  bash "${REPO_ROOT}/scripts/train/train_sft_cv.sh"

  run_dir="$(cat "${spec_last_run_file}")"
  LAST_COMPLETED_RUN_DIR="${run_dir}"

  if [[ "${SWEEP_STAGE}" == "all" ]]; then
    RUN_DIR="${run_dir}" bash "${SCRIPT_DIR}/infer.sh"

    RUN_DIR="${run_dir}" \
    AGG_SCORE_OUT="${run_dir}/score_hungarian_cv_strict.json" \
    PER_KEY_OUT="${run_dir}/score_hungarian_cv_per_key_strict.json" \
    PER_FOLD_SCORE_NAME="score_hungarian_strict.json" \
    PER_FOLD_PER_KEY_NAME="score_hungarian_per_key_strict.json" \
    PROCEDUREDIAGNOSIS_MATCH_MODE="strict" \
    bash "${REPO_ROOT}/scripts/score/score_lung_cv.sh"

    RUN_DIR="${run_dir}" \
    AGG_SCORE_OUT="${run_dir}/score_hungarian_cv_compat.json" \
    PER_KEY_OUT="${run_dir}/score_hungarian_cv_per_key_compat.json" \
    PER_FOLD_SCORE_NAME="score_hungarian_compat.json" \
    PER_FOLD_PER_KEY_NAME="score_hungarian_per_key_compat.json" \
    PROCEDUREDIAGNOSIS_MATCH_MODE="${COMPAT_MATCH_MODE}" \
    bash "${REPO_ROOT}/scripts/score/score_lung_cv.sh"
  fi

  printf '%s\t%s\t%s\t%s\n' "${LEARNING_RATE}" "${NUM_TRAIN_EPOCHS}" "${run_dir}" "${SWEEP_STAGE}" >> "${SWEEP_RUNS_TSV}"
done

if [[ -n "${LAST_COMPLETED_RUN_DIR}" ]]; then
  printf '%s\n' "${LAST_COMPLETED_RUN_DIR}" > "${LAST_RUN_FILE}"
fi

printf '%s\n' "${SWEEP_RUNS_TSV}" > "${SCRIPT_DIR}/last_sweep_runs.txt"
echo "Sweep finished. Summary: ${SWEEP_RUNS_TSV}"
