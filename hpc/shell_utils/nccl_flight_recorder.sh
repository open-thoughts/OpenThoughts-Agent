#!/usr/bin/env bash

setup_nccl_flight_recorder() {
    local experiments_dir="$1"
    if [[ -z "$experiments_dir" ]]; then
        echo "Usage: setup_nccl_flight_recorder <experiments_dir>" >&2
        return 1
    fi

    if [[ -n "${TORCH_FR_DUMP_TEMP_FILE:-}" &&
          -n "${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-}" &&
          "$TORCH_FR_DUMP_TEMP_FILE" != "$TORCH_NCCL_DEBUG_INFO_TEMP_FILE" ]]; then
        echo "TORCH_FR_DUMP_TEMP_FILE and TORCH_NCCL_DEBUG_INFO_TEMP_FILE must use the same prefix" >&2
        return 1
    fi

    local job_id="${SLURM_JOB_ID:-manual}"
    local default_dump_dir="${experiments_dir}/nccl_fr/${job_id}"
    local dump_prefix="${TORCH_FR_DUMP_TEMP_FILE:-${TORCH_NCCL_DEBUG_INFO_TEMP_FILE:-${default_dump_dir}/nccl_fr_rank_}}"
    local dump_dir="${dump_prefix%/*}"
    if [[ "$dump_dir" == "$dump_prefix" ]]; then
        dump_dir="."
    fi
    mkdir -p "$dump_dir"

    export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-20000}"
    export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$dump_prefix"
    export TORCH_FR_DUMP_TEMP_FILE="$dump_prefix"

    echo "NCCL flight recorder: prefix=$dump_prefix buffer=$TORCH_NCCL_TRACE_BUFFER_SIZE"
}
