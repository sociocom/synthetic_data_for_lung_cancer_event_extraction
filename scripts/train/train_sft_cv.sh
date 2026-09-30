#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MODEL_ID="${MODEL_ID:-}"
PROGRESS_JSONL="${PROGRESS_JSONL:-}"
ANNOTATIONS_JSONL="${ANNOTATIONS_JSONL:-}"
GUIDELINE_PATH="${GUIDELINE_PATH:-}"

OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-}"
EXP_NAME="${EXP_NAME:-exp_cv}"
RUN_DIR="${RUN_DIR:-}"
LAST_RUN_FILE="${LAST_RUN_FILE:-}"

NUM_FOLDS="${NUM_FOLDS:-5}"
CV_SPLIT_SEED="${CV_SPLIT_SEED:-42}"
CV_FOLDS_JSON="${CV_FOLDS_JSON:-}"
CV_VAL_SIZE="${CV_VAL_SIZE:-7}"
CV_VAL_SEED_BASE="${CV_VAL_SEED_BASE:-42}"

SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0}"
EVAL_STEPS="${EVAL_STEPS:-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-10}"
MAX_LENGTH="${MAX_LENGTH:-4600}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-15}"
EARLY_STOPPING_THRESHOLD="${EARLY_STOPPING_THRESHOLD:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
PRUNE_PER_FOLD_DURING_TRAIN="${PRUNE_PER_FOLD_DURING_TRAIN:-false}"
TOPK_BY_EVAL_LOSS="${TOPK_BY_EVAL_LOSS:-4}"
KEEP_RECENT_N="${KEEP_RECENT_N:-1}"

cd "${REPO_ROOT}"

[[ -n "${MODEL_ID}" ]] || { echo "MODEL_ID is required" 1>&2; exit 1; }
[[ -n "${PROGRESS_JSONL}" ]] || { echo "PROGRESS_JSONL is required" 1>&2; exit 1; }
[[ -n "${ANNOTATIONS_JSONL}" ]] || { echo "ANNOTATIONS_JSONL is required" 1>&2; exit 1; }
[[ -n "${OUTPUT_BASE_DIR}" ]] || { echo "OUTPUT_BASE_DIR is required" 1>&2; exit 1; }

if [[ "${NUM_FOLDS}" -lt 2 ]]; then
  echo "NUM_FOLDS must be >= 2" 1>&2
  exit 1
fi
if ! [[ "${TOPK_BY_EVAL_LOSS}" =~ ^[0-9]+$ ]] || [[ "${TOPK_BY_EVAL_LOSS}" -lt 1 ]]; then
  echo "TOPK_BY_EVAL_LOSS must be a positive integer. got=${TOPK_BY_EVAL_LOSS}" 1>&2
  exit 1
fi
if ! [[ "${KEEP_RECENT_N}" =~ ^[0-9]+$ ]]; then
  echo "KEEP_RECENT_N must be a non-negative integer. got=${KEEP_RECENT_N}" 1>&2
  exit 1
fi

if [[ -z "${RUN_DIR}" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}_${TS}"
fi
mkdir -p "${RUN_DIR}"
if [[ -n "${LAST_RUN_FILE}" ]]; then
  printf '%s\n' "${RUN_DIR}" > "${LAST_RUN_FILE}"
fi

echo "Run dir: ${RUN_DIR}"

EXTRA_ARGS=()
if [[ -n "${INIT_ADAPTER_PATH:-}" ]]; then
  EXTRA_ARGS+=(--init_adapter_path "${INIT_ADAPTER_PATH}")
fi
if [[ -n "${CV_FOLDS_JSON}" ]]; then
  [[ -f "${CV_FOLDS_JSON}" ]] || { echo "CV_FOLDS_JSON not found: ${CV_FOLDS_JSON}" 1>&2; exit 1; }
  EXTRA_ARGS+=(--cv_folds_json "${CV_FOLDS_JSON}")
fi

prune_fold_checkpoints() {
  local fold_dir="$1"
  local topk="$2"
  local recent_n="$3"

  [[ -d "${fold_dir}" ]] || return 0

  mapfile -t keep_names < <(
    uv run python - "${fold_dir}" "${topk}" "${recent_n}" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

fold_dir = Path(sys.argv[1])
topk = int(sys.argv[2])
recent_n = int(sys.argv[3])
target = topk + recent_n

step_re = re.compile(r"^checkpoint-(\d+)$")

ckpts = []
for p in fold_dir.glob("checkpoint-*"):
    if not p.is_dir():
        continue
    m = step_re.match(p.name)
    if not m:
        continue
    ckpts.append((int(m.group(1)), p.name))
if not ckpts:
    sys.exit(0)
ckpts.sort()

latest_names = [name for _st, name in ckpts[-recent_n:]] if recent_n > 0 else []

loss_map = {}
loss_csv = fold_dir / "loss_history.csv"
if loss_csv.exists():
    with loss_csv.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for row in rd:
            step_raw = (row.get("step") or "").strip()
            val_raw = (row.get("val_loss") or "").strip()
            if not step_raw or not val_raw:
                continue
            try:
                st = int(float(step_raw))
                v = float(val_raw)
            except Exception:
                continue
            if math.isfinite(v):
                loss_map[st] = v

ranked = []
for st, name in ckpts:
    if st in loss_map:
        ranked.append((loss_map[st], -st, name))
ranked.sort()

selected = []
for _loss, _neg_step, name in ranked[:topk]:
    if name not in selected:
        selected.append(name)
for name in latest_names:
    if name not in selected:
        selected.append(name)
for _loss, _neg_step, name in ranked:
    if len(selected) >= target:
        break
    if name not in selected:
        selected.append(name)
for _st, name in reversed(ckpts):
    if len(selected) >= target:
        break
    if name not in selected:
        selected.append(name)

for name in selected:
    print(name)
PY
  )

  mapfile -t all_ckpt_dirs < <(find "${fold_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)
  if [[ "${#all_ckpt_dirs[@]}" -eq 0 ]]; then
    return 0
  fi

  local keep_csv
  keep_csv="$(for n in "${keep_names[@]}"; do printf '%s,' "${n}"; done)"

  local removed=0
  for d in "${all_ckpt_dirs[@]}"; do
    local n
    n="$(basename "${d}")"
    if [[ ",${keep_csv}" == *",${n},"* ]]; then
      continue
    fi
    rm -rf "${d}"
    removed=$((removed + 1))
  done

  echo "[fold_prune] fold_dir=${fold_dir} kept={${keep_csv%,}} removed=${removed}"
}

for ((FOLD=0; FOLD<NUM_FOLDS; FOLD++)); do
  echo "==============================="
  echo "CV fold ${FOLD}/${NUM_FOLDS}"
  echo "Output: ${RUN_DIR}/folds/fold=${FOLD}"
  echo "==============================="

  uv run python -m trainpipe.cli.train_sft train \
    --model_id "${MODEL_ID}" \
    --progress_jsonl "${PROGRESS_JSONL}" \
    --annotations_jsonl "${ANNOTATIONS_JSONL}" \
    --guideline_path "${GUIDELINE_PATH}" \
    --output_dir "${RUN_DIR}" \
    --seed "${SEED}" \
    --val_ratio "${VAL_RATIO}" \
    --test_ratio "${TEST_RATIO}" \
    --eval_steps "${EVAL_STEPS}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --max_length "${MAX_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    --learning_rate "${LEARNING_RATE}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
    --early_stopping_threshold "${EARLY_STOPPING_THRESHOLD}" \
    --max_samples "${MAX_SAMPLES}" \
    --cv_num_folds "${NUM_FOLDS}" \
    --cv_fold "${FOLD}" \
    --cv_split_seed "${CV_SPLIT_SEED}" \
    --cv_val_size "${CV_VAL_SIZE}" \
    --cv_val_seed_base "${CV_VAL_SEED_BASE}" \
    "${EXTRA_ARGS[@]}"

  if [[ "${PRUNE_PER_FOLD_DURING_TRAIN}" == "true" ]]; then
    prune_fold_checkpoints "${RUN_DIR}/folds/fold=${FOLD}" "${TOPK_BY_EVAL_LOSS}" "${KEEP_RECENT_N}"
  fi
done

echo "Done. All folds saved under: ${RUN_DIR}/folds/"
