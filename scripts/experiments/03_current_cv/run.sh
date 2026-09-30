#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
case "${STAGE}" in
  train) bash "${SCRIPT_DIR}/train.sh" ;;
  train_f1select) bash "${SCRIPT_DIR}/train_f1_select.sh" ;;
  infer) bash "${SCRIPT_DIR}/infer.sh" ;;
  f1select) bash "${SCRIPT_DIR}/f1_select.sh" ;;
  score) bash "${SCRIPT_DIR}/score.sh" ;;
  all_f1select) bash "${SCRIPT_DIR}/train_f1_select.sh"; bash "${SCRIPT_DIR}/f1_select.sh" ;;
  all) bash "${SCRIPT_DIR}/train.sh"; bash "${SCRIPT_DIR}/infer.sh"; bash "${SCRIPT_DIR}/score.sh" ;;
  *) echo "Unknown STAGE=${STAGE}. Use train|train_f1select|infer|f1select|score|all|all_f1select" 1>&2; exit 1 ;;
esac
