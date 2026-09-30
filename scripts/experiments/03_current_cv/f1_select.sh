#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
LAST_RUN_FILE="${SCRIPT_DIR}/last_run_dir.txt"

set -a
source "${SCRIPT_DIR}/config/infer.env"
set +a

RUN_DIR="${RUN_DIR:-}"
if [[ -z "${RUN_DIR}" && -f "${LAST_RUN_FILE}" ]]; then
  RUN_DIR=$(cat "${LAST_RUN_FILE}")
fi
[[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is not set and ${LAST_RUN_FILE} not found." 1>&2; exit 1; }

FOLDS_DIR="${RUN_DIR}/folds"
[[ -d "${FOLDS_DIR}" ]] || { echo "FOLDS_DIR not found: ${FOLDS_DIR}" 1>&2; exit 1; }

VAL_SPLIT="${VAL_SPLIT:-validation}"
VAL_OUT_SUBDIR="${VAL_OUT_SUBDIR:-inference_validation_candidates}"
VAL_OUT_BASENAME="${VAL_OUT_BASENAME:-pred_val}"
VAL_SCORE_NAME="${VAL_SCORE_NAME:-score_hungarian_validation.json}"

TEST_SPLIT="${TEST_SPLIT:-test}"
TEST_OUT_SUBDIR="${TEST_OUT_SUBDIR:-inference_f1selected}"
TEST_OUT_BASENAME="${TEST_OUT_BASENAME:-pred_test}"
TEST_PER_FOLD_SCORE_NAME="${TEST_PER_FOLD_SCORE_NAME:-score_hungarian_f1selected.json}"

CANDIDATE_PATTERN="${CANDIDATE_PATTERN:-checkpoint-*}"
TOPK_BY_EVAL_LOSS="${TOPK_BY_EVAL_LOSS:-4}"
KEEP_RECENT_N="${KEEP_RECENT_N:-1}"
PRUNE_UNUSED_CHECKPOINTS="${PRUNE_UNUSED_CHECKPOINTS:-true}"
SELECTION_SUMMARY_OUT="${SELECTION_SUMMARY_OUT:-${RUN_DIR}/f1_selection_validation.json}"
AGG_SCORE_OUT="${AGG_SCORE_OUT:-${RUN_DIR}/score_hungarian_cv_f1selected.json}"
PER_KEY_OUT="${PER_KEY_OUT:-${RUN_DIR}/score_hungarian_cv_per_key_f1selected.json}"

if ! [[ "${TOPK_BY_EVAL_LOSS}" =~ ^[0-9]+$ ]] || [[ "${TOPK_BY_EVAL_LOSS}" -lt 1 ]]; then
  echo "TOPK_BY_EVAL_LOSS must be a positive integer. got=${TOPK_BY_EVAL_LOSS}" 1>&2
  exit 1
fi
if ! [[ "${KEEP_RECENT_N}" =~ ^[0-9]+$ ]]; then
  echo "KEEP_RECENT_N must be a non-negative integer. got=${KEEP_RECENT_N}" 1>&2
  exit 1
fi

mapfile -t fold_dirs < <(find "${FOLDS_DIR}" -maxdepth 1 -type d -name 'fold=*' | sort -V)
[[ "${#fold_dirs[@]}" -gt 0 ]] || { echo "No folds found under: ${FOLDS_DIR}" 1>&2; exit 1; }

tmp_rows="$(mktemp)"
tmp_selected="$(mktemp)"
trap 'rm -f "${tmp_rows}" "${tmp_selected}"' EXIT
declare -A candidate_names_by_fold

echo "RUN_DIR: ${RUN_DIR}"
echo "Selecting checkpoints by validation micro-F1..."

for fold_dir in "${fold_dirs[@]}"; do
  fold_name="$(basename "${fold_dir}")"
  val_gold_jsonl="${fold_dir}/datasets_raw/${VAL_SPLIT}.jsonl"
  [[ -f "${val_gold_jsonl}" ]] || { echo "[${fold_name}] validation file not found: ${val_gold_jsonl}" 1>&2; exit 1; }

  mapfile -t ckpt_dirs < <(find "${fold_dir}" -maxdepth 1 -mindepth 1 -type d -name "${CANDIDATE_PATTERN}" | sort -V)
  if [[ "${#ckpt_dirs[@]}" -eq 0 && -d "${fold_dir}/best" ]]; then
    ckpt_dirs=("${fold_dir}/best")
  fi
  [[ "${#ckpt_dirs[@]}" -gt 0 ]] || { echo "[${fold_name}] no candidate checkpoints found." 1>&2; exit 1; }

  mapfile -t selected_names < <(
    uv run python - "${fold_dir}" "${CANDIDATE_PATTERN}" "${TOPK_BY_EVAL_LOSS}" "${KEEP_RECENT_N}" <<'PY'
import csv
import math
import re
import sys
from pathlib import Path

fold_dir = Path(sys.argv[1])
pattern = sys.argv[2]
topk = int(sys.argv[3])
keep_recent = int(sys.argv[4])
target = topk + keep_recent

step_re = re.compile(r"^checkpoint-(\d+)$")

def parse_step(name: str):
    m = step_re.match(name)
    if not m:
        return None
    return int(m.group(1))

ckpt_names = []
for p in sorted(fold_dir.glob(pattern)):
    if not p.is_dir():
        continue
    st = parse_step(p.name)
    if st is None:
        continue
    ckpt_names.append((st, p.name))

if not ckpt_names:
    sys.exit(0)

ckpt_names.sort()
latest_names = [name for _st, name in ckpt_names[-keep_recent:]] if keep_recent > 0 else []

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
for st, name in ckpt_names:
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

for _st, name in reversed(ckpt_names):
    if len(selected) >= target:
        break
    if name not in selected:
        selected.append(name)

for name in selected:
    print(name)
PY
  )

  if [[ "${#selected_names[@]}" -gt 0 ]]; then
    selected_ckpt_dirs=()
    for n in "${selected_names[@]}"; do
      p="${fold_dir}/${n}"
      if [[ -d "${p}" ]]; then
        selected_ckpt_dirs+=("${p}")
      fi
    done
    if [[ "${#selected_ckpt_dirs[@]}" -gt 0 ]]; then
      ckpt_dirs=("${selected_ckpt_dirs[@]}")
    fi
  fi

  candidate_csv="$(for x in "${ckpt_dirs[@]}"; do basename "${x}"; done | paste -sd ',' -)"
  candidate_names_by_fold["${fold_name}"]="${candidate_csv}"
  echo "[${fold_name}] candidates(top${TOPK_BY_EVAL_LOSS}+latest${KEEP_RECENT_N}): ${candidate_csv}"

  best_ckpt=""
  best_f1="-1.0"
  best_step=-1

  echo "---- ${fold_name} ----"
  for ckpt_dir in "${ckpt_dirs[@]}"; do
    ckpt_name="$(basename "${ckpt_dir}")"
    step=0
    if [[ "${ckpt_name}" =~ ^checkpoint-([0-9]+)$ ]]; then
      step="${BASH_REMATCH[1]}"
    fi

    val_out_dir="${fold_dir}/${VAL_OUT_SUBDIR}/${ckpt_name}"
    val_pred_jsonl="${val_out_dir}/${VAL_OUT_BASENAME}_${VAL_SPLIT}.jsonl"
    val_score_json="${val_out_dir}/${VAL_SCORE_NAME}"

    RUN_DIR="${fold_dir}" \
    MODEL_ID="${MODEL_ID}" \
    INPUT_JSONL="${val_gold_jsonl}" \
    OUT_DIR="${val_out_dir}" \
    OUT_JSONL="${val_pred_jsonl}" \
    OUT_BASENAME="${VAL_OUT_BASENAME}" \
    SPLIT="${VAL_SPLIT}" \
    USE_ADAPTER=true \
    ADAPTER_PATH="${ckpt_dir}" \
    bash "${REPO_ROOT}/scripts/infer/inference.sh"

    GOLD_JSONL="${val_gold_jsonl}" \
    PRED_JSONL="${val_pred_jsonl}" \
    SCORE_OUT="${val_score_json}" \
    bash "${REPO_ROOT}/scripts/score/score_lung.sh"

    f1="$(uv run python - "${val_score_json}" <<'PY'
import json
import sys
path = sys.argv[1]
obj = json.load(open(path, "r", encoding="utf-8"))
print(float(obj["summary"]["micro_f1"]))
PY
)"

    printf "%s\t%s\t%s\t%s\n" "${fold_name}" "${ckpt_name}" "${f1}" "${val_score_json}" >> "${tmp_rows}"
    echo "candidate=${ckpt_name} val_micro_f1=${f1}"

    choose="$(uv run python - "${f1}" "${best_f1}" "${step}" "${best_step}" <<'PY'
import sys
cur_f1 = float(sys.argv[1])
best_f1 = float(sys.argv[2])
cur_step = int(sys.argv[3])
best_step = int(sys.argv[4])
if cur_f1 > best_f1:
    print("1")
elif cur_f1 == best_f1 and cur_step > best_step:
    print("1")
else:
    print("0")
PY
)"

    if [[ "${choose}" == "1" ]]; then
      best_ckpt="${ckpt_name}"
      best_f1="${f1}"
      best_step="${step}"
    fi
  done

  [[ -n "${best_ckpt}" ]] || { echo "[${fold_name}] failed to select best checkpoint." 1>&2; exit 1; }

  printf "%s\t%s\t%s\n" "${fold_name}" "${best_ckpt}" "${best_f1}" >> "${tmp_selected}"
  printf "%s\n" "${best_ckpt}" > "${fold_dir}/best_f1.txt"
  ln -sfn "${best_ckpt}" "${fold_dir}/best_f1"
  echo "[${fold_name}] selected=${best_ckpt} val_micro_f1=${best_f1}"

  test_input_jsonl="${fold_dir}/datasets_raw/${TEST_SPLIT}.jsonl"
  [[ -f "${test_input_jsonl}" ]] || { echo "[${fold_name}] test file not found: ${test_input_jsonl}" 1>&2; exit 1; }
  adapter_path="${fold_dir}/${best_ckpt}"
  [[ -d "${adapter_path}" ]] || { echo "[${fold_name}] selected adapter not found: ${adapter_path}" 1>&2; exit 1; }
  test_out_dir="${fold_dir}/${TEST_OUT_SUBDIR}"
  test_pred_jsonl="${test_out_dir}/${TEST_OUT_BASENAME}_${TEST_SPLIT}.jsonl"

  echo "[${fold_name}] test inference with ${best_ckpt}"
  RUN_DIR="${fold_dir}" \
  MODEL_ID="${MODEL_ID}" \
  INPUT_JSONL="${test_input_jsonl}" \
  OUT_DIR="${test_out_dir}" \
  OUT_JSONL="${test_pred_jsonl}" \
  OUT_BASENAME="${TEST_OUT_BASENAME}" \
  SPLIT="${TEST_SPLIT}" \
  USE_ADAPTER=true \
  ADAPTER_PATH="${adapter_path}" \
  bash "${REPO_ROOT}/scripts/infer/inference.sh"

  if [[ "${PRUNE_UNUSED_CHECKPOINTS}" == "true" ]]; then
    keep_csv="${candidate_names_by_fold["${fold_name}"]}"
    keep_csv="${keep_csv},${best_ckpt}"
    mapfile -t all_ckpt_dirs < <(find "${fold_dir}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)
    removed=0
    for d in "${all_ckpt_dirs[@]}"; do
      n="$(basename "${d}")"
      if [[ ",${keep_csv}," == *",${n},"* ]]; then
        continue
      fi
      rm -rf "${d}"
      removed=$((removed + 1))
    done
    echo "[${fold_name}] pruned_unused_checkpoints=${removed} keep={${keep_csv}}"
  fi
done

uv run python - "${tmp_rows}" "${tmp_selected}" "${SELECTION_SUMMARY_OUT}" <<'PY'
import json
import sys
from collections import defaultdict

rows_tsv, selected_tsv, out_json = sys.argv[1], sys.argv[2], sys.argv[3]
rows = defaultdict(list)
selected = {}

with open(rows_tsv, "r", encoding="utf-8") as f:
    for line in f:
        fold, ckpt, f1, score_json = line.rstrip("\n").split("\t")
        rows[fold].append(
            {
                "checkpoint": ckpt,
                "val_micro_f1": float(f1),
                "score_json": score_json,
            }
        )

with open(selected_tsv, "r", encoding="utf-8") as f:
    for line in f:
        fold, ckpt, f1 = line.rstrip("\n").split("\t")
        selected[fold] = {
            "checkpoint": ckpt,
            "val_micro_f1": float(f1),
        }

out = {
    "summary": {
        "num_folds": len(selected),
        "selection_metric": "validation_micro_f1",
    },
    "selected": selected,
    "candidates": rows,
}

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

echo "Saved selection summary: ${SELECTION_SUMMARY_OUT}"

echo "Scoring selected-checkpoint test outputs..."
RUN_DIR="${RUN_DIR}" \
SPLIT="${TEST_SPLIT}" \
OUT_SUBDIR="${TEST_OUT_SUBDIR}" \
OUT_BASENAME="${TEST_OUT_BASENAME}" \
PER_FOLD_SCORE_NAME="${TEST_PER_FOLD_SCORE_NAME}" \
AGG_SCORE_OUT="${AGG_SCORE_OUT}" \
PER_KEY_OUT="${PER_KEY_OUT}" \
bash "${REPO_ROOT}/scripts/score/score_lung_cv.sh"

echo "Done."
echo "  selection: ${SELECTION_SUMMARY_OUT}"
echo "  cv score : ${AGG_SCORE_OUT}"
