#!/bin/bash
# Shared helpers for datatrace sbatch scripts.

wait_for_tasks_ready() {
  local path="$1"
  local timeout="${2:-1800}"
  local interval=10
  local elapsed=0

  if [ -z "$path" ]; then
    echo "ERROR: wait_for_tasks_ready requires a directory path" >&2
    return 1
  fi

  echo "Waiting for tasks to appear in: $path"
  while [ "$elapsed" -le "$timeout" ]; do
    if [ -d "$path" ]; then
      local first_task
      first_task=$(find "$path" -mindepth 1 -maxdepth 1 -type d -print -quit 2>/dev/null)
      if [ -n "$first_task" ]; then
        echo "Detected task directory: $first_task"
        return 0
      fi
    fi
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done

  echo "ERROR: No task directories found in $path after waiting ${timeout}s" >&2
  return 1
}

ensure_tasks_ready_or_exit() {
  local tasks_path="$1"
  local timeout="${2:-1800}"

  if [ ! -e "$tasks_path" ]; then
    echo "ERROR: Trace input tasks path not found: $tasks_path" >&2
    exit 1
  fi

  if ! wait_for_tasks_ready "$tasks_path" "$timeout"; then
    exit 1
  fi
}

load_trace_chunk_env() {
  if [ "${TRACE_CHUNK_MODE:-0}" != "1" ]; then
    return
  fi

  if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "ERROR: TRACE_CHUNK_MODE enabled but SLURM_ARRAY_TASK_ID is unset." >&2
    exit 1
  fi

  local env_dir="${TRACE_CHUNK_ENV_DIR:-}"
  if [ -z "$env_dir" ]; then
    echo "ERROR: TRACE_CHUNK_ENV_DIR is not set for chunked trace job." >&2
    exit 1
  fi

  local idx="${SLURM_ARRAY_TASK_ID}"
  local padded_idx
  if ! padded_idx=$(printf "%03d" "$idx" 2>/dev/null); then
    padded_idx="$idx"
  fi

  local env_file="${env_dir}/chunk_${padded_idx}.env"
  if [ ! -f "$env_file" ]; then
    env_file="${env_dir}/chunk_${idx}.env"
  fi
  if [ ! -f "$env_file" ]; then
    echo "ERROR: Chunk env file not found for index ${idx} (expected ${env_dir}/chunk_${padded_idx}.env)." >&2
    exit 1
  fi

  echo "Loading chunk env: ${env_file} (array task ${idx}/${TRACE_CHUNK_COUNT:-?})"

  local nounset_was_on=0
  if set -o | grep -qE '^nounset[[:space:]]+on'; then
    nounset_was_on=1
    set +u
  fi
  # shellcheck disable=SC1090
  source "$env_file"
  if [ "$nounset_was_on" -eq 1 ]; then
    set -u
  fi
}
