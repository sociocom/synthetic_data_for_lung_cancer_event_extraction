#!/usr/bin/env bash
set -euo pipefail

resolve_run_dir_or_fail() {
  local run_dir="${1:-}"
  local last_run_file="${2:-}"

  if [[ -n "${run_dir}" ]]; then
    printf '%s\n' "${run_dir}"
    return 0
  fi
  if [[ -n "${last_run_file}" && -f "${last_run_file}" ]]; then
    cat "${last_run_file}"
    return 0
  fi
  return 1
}

save_run_dir() {
  local run_dir="${1:?run_dir required}"
  local last_run_file="${2:-}"
  if [[ -n "${last_run_file}" ]]; then
    mkdir -p "$(dirname "${last_run_file}")"
    printf '%s\n' "${run_dir}" > "${last_run_file}"
  fi
}

make_timestamped_run_dir() {
  local output_base_dir="${1:?output_base_dir required}"
  local exp_name="${2:?exp_name required}"
  local ts
  ts=$(date +"%Y%m%d_%H%M%S")
  printf '%s/%s_%s\n' "${output_base_dir}" "${exp_name}" "${ts}"
}
