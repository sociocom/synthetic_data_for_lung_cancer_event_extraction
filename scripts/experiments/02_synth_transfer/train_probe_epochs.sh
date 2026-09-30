#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
RUN_SH="${SCRIPT_DIR}/run.sh"
INFER_SH="${REPO_ROOT}/scripts/infer/inference.sh"
SCORE_SH="${REPO_ROOT}/scripts/score/score_lung.sh"

# Load exp02 defaults
set -a
source "${SCRIPT_DIR}/config/train.env"
source "${SCRIPT_DIR}/config/infer.env"
source "${SCRIPT_DIR}/config/score.env"
set +a

TARGET_EPOCHS_CSV="${TARGET_EPOCHS_CSV:-5,8,11,14}"
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1}"
INFER_CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES:-2}"
INFER_NUM_SHARDS="${INFER_NUM_SHARDS:-4}"
POLL_SECONDS="${POLL_SECONDS:-10}"
REMOVE_SNAPSHOT_AFTER_SCORE="${REMOVE_SNAPSHOT_AFTER_SCORE:-false}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Python interpreter not found. Set PYTHON_BIN explicitly." 1>&2
    exit 1
  fi
fi

if [[ -z "${RUN_DIR:-}" ]]; then
  TS=$(date +"%Y%m%d_%H%M%S")
  RUN_DIR="${OUTPUT_BASE_DIR}/${EXP_NAME}_probe_${TS}"
fi
mkdir -p "${RUN_DIR}"

IFS=',' read -r -a TARGET_EPOCHS <<< "${TARGET_EPOCHS_CSV}"
if [[ "${#TARGET_EPOCHS[@]}" -eq 0 ]]; then
  echo "TARGET_EPOCHS_CSV is empty." 1>&2
  exit 1
fi

for e in "${TARGET_EPOCHS[@]}"; do
  if ! [[ "${e}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Invalid target epoch: ${e}" 1>&2
    exit 1
  fi
done

PROBE_ROOT="${RUN_DIR}/epoch_probes"
mkdir -p "${PROBE_ROOT}"

epoch_reached() {
  local checkpoint_dir="$1"
  local target_epoch="$2"
  local trainer_state="${checkpoint_dir}/trainer_state.json"
  [[ -f "${trainer_state}" ]] || return 1

  "${PYTHON_BIN}" - "$trainer_state" "$target_epoch" <<'PY'
import json
import sys
from pathlib import Path

st_path = Path(sys.argv[1])
target = float(sys.argv[2])
obj = json.loads(st_path.read_text(encoding="utf-8"))
epoch = obj.get("epoch", -1)
try:
    epoch = float(epoch)
except Exception:
    epoch = -1.0
print("1" if epoch >= target else "0")
PY
}

checkpoint_epoch_value() {
  local checkpoint_dir="$1"
  local trainer_state="${checkpoint_dir}/trainer_state.json"
  [[ -f "${trainer_state}" ]] || { echo "-1"; return 0; }
  "${PYTHON_BIN}" - "$trainer_state" <<'PY'
import json
import sys
from pathlib import Path

st_path = Path(sys.argv[1])
obj = json.loads(st_path.read_text(encoding="utf-8"))
epoch = obj.get("epoch", -1)
try:
    epoch = float(epoch)
except Exception:
    epoch = -1.0
print(epoch)
PY
}

capture_snapshot_for_target() {
  local target_epoch="$1"
  local target_tag
  target_tag=$(printf "%s" "${target_epoch}" | sed 's/[.]/p/g')
  local probe_dir="${PROBE_ROOT}/epoch_${target_tag}"

  if [[ -f "${probe_dir}/captured.ok" || -f "${probe_dir}/skipped.ok" ]]; then
    return 0
  fi

  mapfile -t ckpts < <(find "${RUN_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)
  [[ "${#ckpts[@]}" -gt 0 ]] || return 0

  local chosen=""
  local chosen_epoch=""
  for cp in "${ckpts[@]}"; do
    local reached
    reached="$(epoch_reached "${cp}" "${target_epoch}" || echo 0)"
    if [[ "${reached}" == "1" ]]; then
      chosen="${cp}"
      chosen_epoch="$(checkpoint_epoch_value "${cp}")"
      break
    fi
  done

  [[ -n "${chosen}" ]] || return 0
  [[ -f "${chosen}/adapter_config.json" ]] || return 0
  [[ -f "${chosen}/adapter_model.safetensors" ]] || return 0

  mkdir -p "${probe_dir}/adapter"
  cp "${chosen}/adapter_config.json" "${probe_dir}/adapter/"
  cp "${chosen}/adapter_model.safetensors" "${probe_dir}/adapter/"
  if [[ -f "${chosen}/README.md" ]]; then
    cp "${chosen}/README.md" "${probe_dir}/adapter/"
  fi
  cat > "${probe_dir}/capture_meta.txt" <<EOF
target_epoch=${target_epoch}
source_checkpoint=${chosen}
source_epoch=${chosen_epoch}
captured_at=$(date -Iseconds)
EOF
  touch "${probe_dir}/captured.ok"
  echo "[capture] target_epoch=${target_epoch} from=$(basename "${chosen}") epoch=${chosen_epoch}"
}

run_probe_infer_score() {
  local probe_dir="$1"
  local target_tag
  target_tag="$(basename "${probe_dir}" | sed 's/^epoch_//')"
  local out_subdir="inference_transfer_ep${target_tag}"
  local out_dir="${RUN_DIR}/${out_subdir}"
  local out_jsonl="${out_dir}/${OUT_BASENAME}_${SPLIT}.jsonl"
  local score_out="${out_dir}/score_hungarian_transfer.json"
  local per_key_out="${out_dir}/score_hungarian_transfer_per_key.json"

  mkdir -p "${out_dir}"
  echo "[infer] ${out_subdir}"
  CUDA_VISIBLE_DEVICES="${INFER_CUDA_VISIBLE_DEVICES}" \
  NUM_SHARDS="${INFER_NUM_SHARDS}" \
  MODEL_ID="${MODEL_ID}" \
  INPUT_JSONL="${INPUT_JSONL}" \
  GUIDELINE_PATH="${GUIDELINE_PATH}" \
  USE_ADAPTER=true \
  ADAPTER_PATH="${probe_dir}/adapter" \
  OUT_DIR="${out_dir}" \
  OUT_JSONL="${out_jsonl}" \
  OUT_BASENAME="${OUT_BASENAME}" \
  SPLIT="${SPLIT}" \
  RESUME=true \
  RAW_PRED_KEY="${RAW_PRED_KEY:-}" \
  bash "${INFER_SH}"

  echo "[score] ${out_subdir}"
  GOLD_JSONL="${GOLD_JSONL}" \
  PRED_JSONL="${out_jsonl}" \
  SCORE_OUT="${score_out}" \
  PER_KEY_OUT="${per_key_out}" \
  bash "${SCORE_SH}"
}

all_targets_finished() {
  local pending=0
  for t in "${TARGET_EPOCHS[@]}"; do
    local tag
    tag=$(printf "%s" "${t}" | sed 's/[.]/p/g')
    if [[ ! -f "${PROBE_ROOT}/epoch_${tag}/inference.done" && ! -f "${PROBE_ROOT}/epoch_${tag}/inference.failed" && ! -f "${PROBE_ROOT}/epoch_${tag}/skipped.ok" ]]; then
      pending=$((pending + 1))
    fi
  done
  [[ "${pending}" -eq 0 ]]
}

pick_next_pending_probe() {
  mapfile -t probe_dirs < <(find "${PROBE_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'epoch_*' | sort -V)
  for d in "${probe_dirs[@]}"; do
    if [[ -f "${d}/captured.ok" && ! -f "${d}/inference.done" && ! -f "${d}/inference.failed" && ! -f "${d}/inference.running" ]]; then
      echo "${d}"
      return 0
    fi
  done
  return 1
}

echo "[train] RUN_DIR=${RUN_DIR}"
echo "[train] TARGET_EPOCHS=${TARGET_EPOCHS_CSV}"
echo "[train] TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "[infer] INFER_CUDA_VISIBLE_DEVICES=${INFER_CUDA_VISIBLE_DEVICES} NUM_SHARDS=${INFER_NUM_SHARDS}"

CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" RUN_DIR="${RUN_DIR}" STAGE=train bash "${RUN_SH}" &
TRAIN_PID=$!
echo "[train] pid=${TRAIN_PID}"

INFER_PID=""
TRAIN_DONE=0

while true; do
  for target in "${TARGET_EPOCHS[@]}"; do
    capture_snapshot_for_target "${target}"
  done

  if [[ -n "${INFER_PID}" ]]; then
    if ! kill -0 "${INFER_PID}" 2>/dev/null; then
      wait "${INFER_PID}" || true
      INFER_PID=""
    fi
  fi

  if [[ -z "${INFER_PID}" ]]; then
    if next_probe="$(pick_next_pending_probe)"; then
      touch "${next_probe}/inference.running"
      (
        set -euo pipefail
        if run_probe_infer_score "${next_probe}"; then
          rm -f "${next_probe}/inference.failed"
          touch "${next_probe}/inference.done"
          if [[ "${REMOVE_SNAPSHOT_AFTER_SCORE}" == "true" ]]; then
            rm -rf "${next_probe}/adapter"
          fi
        else
          touch "${next_probe}/inference.failed"
        fi
        rm -f "${next_probe}/inference.running"
      ) &
      INFER_PID=$!
    fi
  fi

  if [[ "${TRAIN_DONE}" -eq 0 ]]; then
    if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
      wait "${TRAIN_PID}"
      TRAIN_DONE=1
      echo "[train] finished"
    fi
  fi

  if [[ "${TRAIN_DONE}" -eq 1 ]]; then
    for t in "${TARGET_EPOCHS[@]}"; do
      tag=$(printf "%s" "${t}" | sed 's/[.]/p/g')
      dir="${PROBE_ROOT}/epoch_${tag}"
      if [[ ! -f "${dir}/captured.ok" && ! -f "${dir}/skipped.ok" ]]; then
        mkdir -p "${dir}"
        echo "target_epoch=${t}" > "${dir}/capture_meta.txt"
        echo "status=not_reached_before_train_end" >> "${dir}/capture_meta.txt"
        touch "${dir}/skipped.ok"
      fi
    done

    if [[ -z "${INFER_PID}" ]] && all_targets_finished; then
      break
    fi
  fi

  sleep "${POLL_SECONDS}"
done

echo "Done: ${RUN_DIR}"
for t in "${TARGET_EPOCHS[@]}"; do
  tag=$(printf "%s" "${t}" | sed 's/[.]/p/g')
  d="${PROBE_ROOT}/epoch_${tag}"
  if [[ -f "${d}/inference.done" ]]; then
    echo "  epoch ${t}: done -> ${RUN_DIR}/inference_transfer_ep${tag}/score_hungarian_transfer.json"
  elif [[ -f "${d}/inference.failed" ]]; then
    echo "  epoch ${t}: failed (see ${d})"
  elif [[ -f "${d}/skipped.ok" ]]; then
    echo "  epoch ${t}: skipped (checkpoint not reached)"
  else
    echo "  epoch ${t}: pending/failed (see ${d})"
  fi
done
