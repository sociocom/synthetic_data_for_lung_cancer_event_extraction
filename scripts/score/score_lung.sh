#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

GOLD_JSONL="${1:-${GOLD_JSONL:-}}"
PRED_JSONL="${2:-${PRED_JSONL:-}}"
SCORE_OUT="${3:-${SCORE_OUT:-}}"
PER_KEY_OUT="${4:-${PER_KEY_OUT:-}}"
PROCEDUREDIAGNOSIS_MATCH_MODE="${5:-${PROCEDUREDIAGNOSIS_MATCH_MODE:-}}"

[[ -n "${GOLD_JSONL}" ]] || { echo "GOLD_JSONL is required" 1>&2; exit 2; }
[[ -n "${PRED_JSONL}" ]] || { echo "PRED_JSONL is required" 1>&2; exit 2; }

cd "${REPO_ROOT}"

ARGS=(--gold "${GOLD_JSONL}" --pred "${PRED_JSONL}")
if [[ -n "${SCORE_OUT}" ]]; then
  mkdir -p "$(dirname "${SCORE_OUT}")"
  ARGS+=(--save-json "${SCORE_OUT}")
fi
if [[ -n "${PER_KEY_OUT}" ]]; then
  mkdir -p "$(dirname "${PER_KEY_OUT}")"
  ARGS+=(--save-per-key-json "${PER_KEY_OUT}")
fi
if [[ -n "${PROCEDUREDIAGNOSIS_MATCH_MODE}" ]]; then
  ARGS+=(--procedurediagnosis-match-mode "${PROCEDUREDIAGNOSIS_MATCH_MODE}")
fi

uv run python -m src.score_lung "${ARGS[@]}"
