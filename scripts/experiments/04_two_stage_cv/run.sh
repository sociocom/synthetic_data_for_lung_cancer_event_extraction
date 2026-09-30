#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
case "${STAGE}" in
  stage1_scratch) bash "${SCRIPT_DIR}/stage1_train.sh" ;;
  stage2) bash "${SCRIPT_DIR}/stage2_train.sh" ;;
  stage2_f1select) bash "${SCRIPT_DIR}/stage2_train_f1_select.sh" ;;
  infer) bash "${SCRIPT_DIR}/infer.sh" ;;
  f1select) bash "${SCRIPT_DIR}/f1_select.sh" ;;
  score) bash "${SCRIPT_DIR}/score.sh" ;;
  all_f1select) bash "${SCRIPT_DIR}/stage2_train_f1_select.sh"; bash "${SCRIPT_DIR}/f1_select.sh" ;;
  all) bash "${SCRIPT_DIR}/stage2_train.sh"; bash "${SCRIPT_DIR}/infer.sh"; bash "${SCRIPT_DIR}/score.sh" ;;
  *) echo "Unknown STAGE=${STAGE}. Use stage1_scratch|stage2|stage2_f1select|infer|f1select|score|all|all_f1select" 1>&2; exit 1 ;;
esac
