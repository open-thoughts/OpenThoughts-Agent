#!/bin/bash
# ==============================================================================
# Triton and TorchInductor Cache Setup for HPC
# ==============================================================================
# This script configures Triton and TorchInductor to use node-local storage
# for their compilation caches. This prevents race conditions and cache
# corruption on shared filesystems (Lustre, GPFS) where multiple workers
# may try to compile and cache the same kernels simultaneously.
#
# Usage:
#   source /path/to/hpc/shell_utils/triton_cache.sh
#
# Environment variables set:
#   TRITON_CACHE_DIR         - Node-local Triton kernel cache
#   TRITON_CACHE_MANAGER     - Triton cache manager class
#   TORCHINDUCTOR_CACHE_DIR  - Node-local TorchInductor cache
#   FLASHINFER_WORKSPACE_BASE - Node-local flashinfer JIT cache (GPFS race fix)
# ==============================================================================

setup_triton_cache() {
    local job_id="${SLURM_JOB_ID:-$$}"
    local user="${USER:-unknown}"

    # Use node-local /tmp for Triton cache to avoid shared filesystem issues
    local triton_base="/tmp/triton_cache_${user}_${job_id}"
    mkdir -p "$triton_base" 2>/dev/null || true
    export TRITON_CACHE_DIR="$triton_base"
    export TRITON_CACHE_MANAGER="triton.runtime.cache:FileCacheManager"

    # Also set TorchInductor cache to node-local storage
    local inductor_base="/tmp/torchinductor_cache_${user}_${job_id}"
    mkdir -p "$inductor_base" 2>/dev/null || true
    export TORCHINDUCTOR_CACHE_DIR="$inductor_base"

    # flashinfer JIT cache → node-local too. Default is $HOME/.cache/flashinfer on shared
    # GPFS; concurrent multi-cell vLLM launches race to JIT-build the SAME sampling.so into
    # the SAME path, and a worker that dlopen()s a half-built .so dies with
    # "symbol lookup error: ... undefined symbol: _Z..top_p_sampling_from_probs...", crashing
    # the whole engine at CUDA-graph profiling (~40min healthcheck timeout). Node-local
    # isolation eliminates the cross-job/cross-worker race. (Diagnosed 2026-06-17, Lever-2 pilot.)
    local flashinfer_base="/tmp/flashinfer_${user}_${job_id}"
    mkdir -p "$flashinfer_base" 2>/dev/null || true
    export FLASHINFER_WORKSPACE_BASE="$flashinfer_base"

    # Print status if verbose
    if [[ "${TRITON_CACHE_VERBOSE:-0}" == "1" ]]; then
        echo "[triton_cache] Triton cache: $TRITON_CACHE_DIR"
        echo "[triton_cache] TorchInductor cache: $TORCHINDUCTOR_CACHE_DIR"
        echo "[triton_cache] flashinfer workspace: $FLASHINFER_WORKSPACE_BASE"
    fi
}

# Auto-setup when sourced (can be disabled with TRITON_CACHE_MANUAL=1)
if [[ "${TRITON_CACHE_MANUAL:-0}" != "1" ]]; then
    setup_triton_cache
fi
