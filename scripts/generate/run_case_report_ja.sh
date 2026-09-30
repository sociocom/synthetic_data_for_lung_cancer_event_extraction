#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export MAX_WORKERS="${MAX_WORKERS:-100}"
export MAX_RETRY="${MAX_RETRY:-2}"

INPUT_JSONL="${INPUT_JSONL:-}"
OUTPUT_JSONL="${OUTPUT_JSONL:-}"
PROMPT_FILE="${PROMPT_FILE:-}"

[[ -n "${INPUT_JSONL}" ]] || { echo "INPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUTPUT_JSONL}" ]] || { echo "OUTPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${PROMPT_FILE}" ]] || { echo "PROMPT_FILE is required" 1>&2; exit 1; }

uv run python -m genpipe.cli.gen_case_reports_ja \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --prompt_file "${PROMPT_FILE}"
