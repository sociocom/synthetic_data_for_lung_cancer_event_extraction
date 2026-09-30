#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"

set -a
source "${SCRIPT_DIR}/config/train.env"
set +a

SWEEP_SYNTH_RATIOS="${SWEEP_SYNTH_RATIOS:-100 150 200}"
SWEEP_SIM_RATIO="${SWEEP_SIM_RATIO:-50}"
SWEEP_STAGE="${SWEEP_STAGE:-all}" # train|all
SWEEP_NAME="${SWEEP_NAME:-exp06_mix_cv_sweep}"

case "${SWEEP_STAGE}" in
  train|all) ;;
  *)
    echo "Unknown SWEEP_STAGE=${SWEEP_STAGE}. Use train|all" 1>&2
    exit 1
    ;;
esac

BASE_EXP_NAME="${EXP_NAME:-exp06_mix_cv}"
RATIOS_NORM="${SWEEP_SYNTH_RATIOS//,/ }"
SWEEP_TS="$(date +"%Y%m%d_%H%M%S")"
SWEEP_LOG_DIR="${OUTPUT_BASE_DIR}/${SWEEP_NAME}_${SWEEP_TS}"
SWEEP_RUNS_TSV="${SWEEP_LOG_DIR}/runs.tsv"
mkdir -p "${SWEEP_LOG_DIR}"
printf 'synth_ratio\tsim_ratio\trun_dir\tstage\n' > "${SWEEP_RUNS_TSV}"

echo "Sweep output: ${SWEEP_LOG_DIR}"
echo "Ratios (synth:sim): ${RATIOS_NORM}:${SWEEP_SIM_RATIO}"
echo "Stage: ${SWEEP_STAGE}"

LAST_COMPLETED_RUN_DIR=""

for ratio in ${RATIOS_NORM}; do
  if ! [[ "${ratio}" =~ ^[0-9]+$ ]]; then
    echo "Invalid synth ratio: ${ratio}" 1>&2
    exit 1
  fi

  export MIX_SYNTH_RATIO="${ratio}"
  export MIX_SIM_RATIO="${SWEEP_SIM_RATIO}"
  export EXP_NAME="${BASE_EXP_NAME}_s${MIX_SYNTH_RATIO}_r${MIX_SIM_RATIO}"

  ratio_last_run_file="${SWEEP_LOG_DIR}/last_run_s${MIX_SYNTH_RATIO}_r${MIX_SIM_RATIO}.txt"
  rm -f "${ratio_last_run_file}"

  echo "========================================"
  echo "SWEEP RUN synth:sim=${MIX_SYNTH_RATIO}:${MIX_SIM_RATIO}"
  echo "EXP_NAME=${EXP_NAME}"
  echo "========================================"

  LAST_RUN_FILE="${ratio_last_run_file}" bash "${REPO_ROOT}/scripts/train/train_sft_mix_cv.sh"
  run_dir="$(cat "${ratio_last_run_file}")"
  LAST_COMPLETED_RUN_DIR="${run_dir}"

  if [[ "${SWEEP_STAGE}" == "all" ]]; then
    RUN_DIR="${run_dir}" bash "${SCRIPT_DIR}/infer.sh"
    RUN_DIR="${run_dir}" bash "${SCRIPT_DIR}/score.sh"
  fi

  printf '%s\t%s\t%s\t%s\n' "${MIX_SYNTH_RATIO}" "${MIX_SIM_RATIO}" "${run_dir}" "${SWEEP_STAGE}" >> "${SWEEP_RUNS_TSV}"
done

if [[ -n "${LAST_COMPLETED_RUN_DIR}" ]]; then
  printf '%s\n' "${LAST_COMPLETED_RUN_DIR}" > "${LAST_RUN_FILE}"
fi

printf '%s\n' "${SWEEP_RUNS_TSV}" > "${SCRIPT_DIR}/last_sweep_runs.txt"
echo "Sweep finished. Summary: ${SWEEP_RUNS_TSV}"
