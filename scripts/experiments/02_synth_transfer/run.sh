#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STAGE="${STAGE:-all}"
case "${STAGE}" in
  train) bash "${SCRIPT_DIR}/train.sh" ;;
  infer) bash "${SCRIPT_DIR}/infer.sh" ;;
  score) bash "${SCRIPT_DIR}/score.sh" ;;
  all) bash "${SCRIPT_DIR}/train.sh"; bash "${SCRIPT_DIR}/infer.sh"; bash "${SCRIPT_DIR}/score.sh" ;;
  *) echo "Unknown STAGE=${STAGE}. Use train|infer|score|all" 1>&2; exit 1 ;;
esac
