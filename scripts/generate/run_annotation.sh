#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
export MAX_WORKERS="${MAX_WORKERS:-300}"
export MAX_RETRY="${MAX_RETRY:-2}"

INPUT_JSONL="${INPUT_JSONL:-${REPO_ROOT}/data/processed/synthetic_data/progress_notes_bias.jsonl}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${REPO_ROOT}/data/processed/synthetic_data/annotations_bias_noae_v2.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-${REPO_ROOT}/src/prompts/gen_annotation_noae_v2.toml}"
ICL_INPUT_DIR="${ICL_INPUT_DIR:-${REPO_ROOT}/data/raw/contexts/simu_progress_notes}"
ICL_OUTPUT_DIR="${ICL_OUTPUT_DIR:-${REPO_ROOT}/data/raw/contexts/simu_annotation_noae}"
N_CONTEXT="${N_CONTEXT:-2}"
SEED="${SEED:-42}"

[[ -n "${INPUT_JSONL}" ]] || { echo "INPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUTPUT_JSONL}" ]] || { echo "OUTPUT_JSONL is required" 1>&2; exit 1; }
[[ -n "${PROMPT_FILE}" ]] || { echo "PROMPT_FILE is required" 1>&2; exit 1; }
[[ -n "${ICL_INPUT_DIR}" ]] || { echo "ICL_INPUT_DIR is required" 1>&2; exit 1; }
[[ -n "${ICL_OUTPUT_DIR}" ]] || { echo "ICL_OUTPUT_DIR is required" 1>&2; exit 1; }

uv run python -m genpipe.cli.gen_annotation \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL}" \
  --prompt_file "${PROMPT_FILE}" \
  --id_key "id" \
  --text_key "progress_note" \
  --icl_input_dir "${ICL_INPUT_DIR}" \
  --icl_output_dir "${ICL_OUTPUT_DIR}" \
  --n_context "${N_CONTEXT}" \
  --seed "${SEED}"
