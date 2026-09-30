#!/usr/bin/env bash
# Sequential production runs on 3 GPUs: gpt-oss-120b -> Weblab-MedLLM (both bf16), each followed by scoring.
# Usage: nohup bash run_all.sh > results/llm_baselines/logs/run_all.log 2>&1 &
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LOG_DIR="${REPO_ROOT}/results/llm_baselines/logs"
mkdir -p "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

for cfg in ${MODEL_CFGS:-gptoss weblab}; do
  echo "[$(date '+%F %T')] START infer ${cfg}"
  MODEL_CFG="${cfg}" bash "${SCRIPT_DIR}/infer.sh" > "${LOG_DIR}/${cfg}_infer.log" 2>&1
  rc=$?
  RUN_DIR=$(cat "${SCRIPT_DIR}/last_run_dir.txt")
  echo "[$(date '+%F %T')] END infer ${cfg} rc=${rc} run_dir=${RUN_DIR}"
  if [[ "${rc}" -eq 0 ]]; then
    RUN_DIR="${RUN_DIR}" bash "${SCRIPT_DIR}/score.sh" > "${LOG_DIR}/${cfg}_score.log" 2>&1
    echo "[$(date '+%F %T')] END score ${cfg} rc=$? -> ${RUN_DIR}/score/score_hungarian.json"
  fi
done
echo "[$(date '+%F %T')] ALL DONE"
