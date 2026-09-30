#!/usr/bin/env bash
set -euo pipefail

# CV score wrapper for lung Hungarian scorer.
#
# What it does:
# - Finds fold directories under RUN_DIR/folds/fold=*
# - Scores each fold with src.score_lung
# - Saves per-fold score JSONs (default: fold*/inference/score_hungarian.json)
# - Saves per-fold per-key JSONs (default: fold*/inference/score_hungarian_per_key.json)
# - Aggregates all folds into one global micro score
#
# Usage:
#   bash scripts/score_lung_cv.sh [RUN_DIR] [AGG_SCORE_OUT_JSON]
#
# Env overrides:
#   RUN_DIR=... SPLIT=test OUT_SUBDIR=inference OUT_BASENAME=pred_test \
#   PER_FOLD_SCORE_NAME=score_hungarian.json PER_FOLD_PER_KEY_NAME=score_hungarian_per_key.json \
#   AGG_SCORE_OUT=... PER_KEY_OUT=... \
#   bash scripts/score_lung_cv.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR="${1:-${RUN_DIR:-}}"
AGG_SCORE_OUT="${2:-${AGG_SCORE_OUT:-}}"
PER_KEY_OUT="${PER_KEY_OUT:-}"

SPLIT="${SPLIT:-test}"
OUT_SUBDIR="${OUT_SUBDIR:-inference}"
OUT_BASENAME="${OUT_BASENAME:-pred_test}"
PER_FOLD_SCORE_NAME="${PER_FOLD_SCORE_NAME:-score_hungarian.json}"
PER_FOLD_PER_KEY_NAME="${PER_FOLD_PER_KEY_NAME:-score_hungarian_per_key.json}"
PROCEDUREDIAGNOSIS_MATCH_MODE="${PROCEDUREDIAGNOSIS_MATCH_MODE:-}"

FOLDS_DIR="${RUN_DIR}/folds"
[[ -n "${RUN_DIR}" ]] || { echo "RUN_DIR is required" 1>&2; exit 1; }
[[ -n "${AGG_SCORE_OUT}" ]] || { echo "AGG_SCORE_OUT is required" 1>&2; exit 1; }
[[ -n "${PER_KEY_OUT}" ]] || { echo "PER_KEY_OUT is required" 1>&2; exit 1; }
[[ -d "${FOLDS_DIR}" ]] || { echo "FOLDS_DIR not found: ${FOLDS_DIR}" 1>&2; exit 1; }

# Make module execution location-independent.
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

mapfile -t fold_dirs < <(find "${FOLDS_DIR}" -maxdepth 1 -type d -name 'fold=*' | sort -V)
[[ "${#fold_dirs[@]}" -gt 0 ]] || { echo "No folds found under: ${FOLDS_DIR}" 1>&2; exit 1; }

echo "RUN_DIR            : ${RUN_DIR}"
echo "FOLDS              : ${#fold_dirs[@]}"
echo "SPLIT              : ${SPLIT}"
echo "OUT_SUBDIR         : ${OUT_SUBDIR}"
echo "OUT_BASENAME       : ${OUT_BASENAME}"
echo "PER_FOLD_SCORE_NAME: ${PER_FOLD_SCORE_NAME}"
echo "PER_FOLD_PER_KEY_NAME: ${PER_FOLD_PER_KEY_NAME}"
echo "PER_KEY_OUT         : ${PER_KEY_OUT}"
echo "PROCEDUREDIAGNOSIS_MATCH_MODE: ${PROCEDUREDIAGNOSIS_MATCH_MODE:-strict}"
echo

score_jsons=()
per_key_jsons=()

for fold_dir in "${fold_dirs[@]}"; do
  fold_name="$(basename "${fold_dir}")"
  gold_jsonl="${fold_dir}/datasets_raw/${SPLIT}.jsonl"
  pred_jsonl="${fold_dir}/${OUT_SUBDIR}/${OUT_BASENAME}_${SPLIT}.jsonl"
  fold_score_json="${fold_dir}/${OUT_SUBDIR}/${PER_FOLD_SCORE_NAME}"
  fold_per_key_json="${fold_dir}/${OUT_SUBDIR}/${PER_FOLD_PER_KEY_NAME}"

  [[ -f "${gold_jsonl}" ]] || { echo "[${fold_name}] gold not found: ${gold_jsonl}" 1>&2; exit 1; }
  [[ -f "${pred_jsonl}" ]] || { echo "[${fold_name}] pred not found: ${pred_jsonl}" 1>&2; exit 1; }

  echo "---- ${fold_name} ----"
  ARGS=(
    --gold "${gold_jsonl}"
    --pred "${pred_jsonl}"
    --save-json "${fold_score_json}"
    --save-per-key-json "${fold_per_key_json}"
  )
  if [[ -n "${PROCEDUREDIAGNOSIS_MATCH_MODE}" ]]; then
    ARGS+=(--procedurediagnosis-match-mode "${PROCEDUREDIAGNOSIS_MATCH_MODE}")
  fi
  uv run python -m src.score_lung "${ARGS[@]}"

  score_jsons+=("${fold_score_json}")
  per_key_jsons+=("${fold_per_key_json}")
  echo
done

tmp_list="$(mktemp)"
tmp_per_key_list="$(mktemp)"
trap 'rm -f "${tmp_list}" "${tmp_per_key_list}"' EXIT
printf "%s\n" "${score_jsons[@]}" > "${tmp_list}"
printf "%s\n" "${per_key_jsons[@]}" > "${tmp_per_key_list}"

mkdir -p "$(dirname "${AGG_SCORE_OUT}")"

uv run python - "${tmp_list}" "${tmp_per_key_list}" "${AGG_SCORE_OUT}" "${PER_KEY_OUT}" <<'PY'
import json
import sys
from pathlib import Path

score_list_path = Path(sys.argv[1])
per_key_list_path = Path(sys.argv[2])
out_path = Path(sys.argv[3])
per_key_out_path = Path(sys.argv[4])
score_paths = [Path(x.strip()) for x in score_list_path.read_text(encoding="utf-8").splitlines() if x.strip()]
per_key_paths = [Path(x.strip()) for x in per_key_list_path.read_text(encoding="utf-8").splitlines() if x.strip()]

if len(score_paths) != len(per_key_paths):
    raise ValueError(
        f"fold file count mismatch: score_paths={len(score_paths)} per_key_paths={len(per_key_paths)}"
    )

total_gold = 0
total_pred = 0
total_tp = 0
per_fold = []
by_key = {}


def _acc_key(d, key, tp=0, fp=0, fn=0):
    rec = d.setdefault(key, {"tp": 0, "fp": 0, "fn": 0})
    rec["tp"] += tp
    rec["fp"] += fp
    rec["fn"] += fn

for p in score_paths:
    obj = json.loads(p.read_text(encoding="utf-8"))
    s = obj["summary"]
    g = int(s["total_gold_items"])
    pr = int(s["total_pred_items"])
    tp = int(s["total_tp_items"])
    total_gold += g
    total_pred += pr
    total_tp += tp
    per_fold.append(
        {
            "fold_score_json": str(p),
            "num_examples": int(s["num_examples"]),
            "total_gold_items": g,
            "total_pred_items": pr,
            "total_tp_items": tp,
            "micro_precision": float(s["micro_precision"]),
            "micro_recall": float(s["micro_recall"]),
            "micro_f1": float(s["micro_f1"]),
        }
    )

for p in per_key_paths:
    obj = json.loads(p.read_text(encoding="utf-8"))
    per_key = obj.get("per_key", {})
    for key, rec in per_key.items():
        _acc_key(
            by_key,
            key,
            tp=int(rec.get("tp_items", 0)),
            fp=int(rec.get("fp_items", 0)),
            fn=int(rec.get("fn_items", 0)),
        )

prec = (total_tp / total_pred) if total_pred > 0 else 0.0
rec = (total_tp / total_gold) if total_gold > 0 else 0.0
f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

out = {
    "summary": {
        "num_folds": len(per_fold),
        "num_examples_total": sum(x["num_examples"] for x in per_fold),
        "total_gold_items": total_gold,
        "total_pred_items": total_pred,
        "total_tp_items": total_tp,
        "micro_precision": prec,
        "micro_recall": rec,
        "micro_f1": f1,
    },
    "per_fold": per_fold,
}

out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

per_key = {}
for k in sorted(by_key.keys()):
    tp = int(by_key[k]["tp"])
    fp = int(by_key[k]["fp"])
    fn = int(by_key[k]["fn"])
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1k = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    per_key[k] = {
        "precision": p,
        "recall": r,
        "f1": f1k,
        "tp_items": tp,
        "fp_items": fp,
        "fn_items": fn,
        "gold_items": tp + fn,
        "pred_items": tp + fp,
    }

per_key_out = {
    "summary": {
        "num_keys": len(per_key),
        "note": "Per-key item-level scores aggregated across all CV folds using Hungarian event alignment.",
    },
    "per_key": per_key,
}
per_key_out_path.write_text(json.dumps(per_key_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("# CV total (lung/hungarian, item-level)")
print(f"num_folds: {out['summary']['num_folds']}")
print(f"num_examples_total: {out['summary']['num_examples_total']}")
print(f"total_gold_items: {out['summary']['total_gold_items']}")
print(f"total_pred_items: {out['summary']['total_pred_items']}")
print(f"total_tp_items: {out['summary']['total_tp_items']}")
print(f"micro_precision: {out['summary']['micro_precision']:.6f}")
print(f"micro_recall: {out['summary']['micro_recall']:.6f}")
print(f"micro_f1: {out['summary']['micro_f1']:.6f}")
print(f"saved: {out_path}")
print(f"saved per-key: {per_key_out_path}")
PY
