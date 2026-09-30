#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"
set -a
source "${SCRIPT_DIR}/config/stage1_train.env"
set +a
LAST_RUN_FILE="${LAST_RUN_FILE}" bash "${REPO_ROOT}/scripts/train/train_sft.sh"
