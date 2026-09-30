#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export MAX_WORKERS="${MAX_WORKERS:-500}"
export MAX_RETRY="${MAX_RETRY:-2}"

INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/data/processed/synthetic_data/case_reports_ja.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${REPO_ROOT}/data/processed/synthetic_data/discharge_summaries_bias.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-${REPO_ROOT}/src/prompts/gen_discharge_summaries_bias.toml}"
CONTEXT_DIR="${CONTEXT_DIR:-${REPO_ROOT}/data/raw/contexts/discharge_summary_20}"
N_CONTEXT="${N_CONTEXT:-3}"
SEED="${SEED:-42}"

[[ -n "${INPUT_JSONL}" ]] || { echo "INPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUTPUT_JSONL}" ]] || { echo "OUTPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${PROMPT_FILE}" ]] || { echo "PROMPT_FILE is required" 1>&2; exit 1; }
[[ -n "${CONTEXT_DIR}" ]] || { echo "CONTEXT_DIR is required" 1>&2; exit 1; }

uv run python -m genpipe.cli.gen_discharge_summaries \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --prompt_file "${PROMPT_FILE}" \
  --context_dir "${CONTEXT_DIR}" \
  --n_context "${N_CONTEXT}" \
  --seed "${SEED}"
