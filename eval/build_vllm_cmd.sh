#!/bin/bash
# ==============================================================================
# eval/build_vllm_cmd.sh — Shared vLLM command builder for eval scripts
#
# Reads EVAL_VLLM_* env vars (set by unified_eval_listener.py from
# baseline_model_configs.yaml) and builds a vLLM launch command array.
#
# Usage (source from any eval sbatch/pbs script):
#
#   source "$DCFT/eval/build_vllm_cmd.sh"
#   build_vllm_cmd "$PYTHON_BIN" "$MODEL" "$GPU_MEMORY_UTIL"
#   # Result is in VLLM_CMD array; launch with:
#   env ... "${VLLM_CMD[@]}" > logfile 2>&1 &
#
# Env vars consumed (all optional, set by listener):
#   EVAL_VLLM_TENSOR_PARALLEL_SIZE  (default: 4)
#   EVAL_VLLM_MAX_MODEL_LEN         (default: unset = model native)
#   EVAL_VLLM_SWAP_SPACE            (default: 32)
#   EVAL_VLLM_TRUST_REMOTE_CODE     (default: unset; set to "1" to enable)
#   EVAL_VLLM_TOOL_CALL_PARSER      (default: unset)
#   EVAL_VLLM_REASONING_PARSER      (default: unset)
#   EVAL_VLLM_DATA_PARALLEL_SIZE    (default: unset; vLLM v0.8+ only)
#   EVAL_VLLM_EXTRA_ARGS            (default: unset; space-separated string)
#   EVAL_VLLM_HF_OVERRIDES          (default: unset; JSON string for --hf-overrides)
# ==============================================================================

build_vllm_cmd() {
    local python_bin="${1:?Usage: build_vllm_cmd <python_bin> <model> <gpu_mem_util>}"
    local model="${2:?Missing model}"
    local gpu_mem_util="${3:-0.95}"

    # Read overrides from env (set by listener via sbatch --export)
    local tp="${EVAL_VLLM_TENSOR_PARALLEL_SIZE:-4}"
    local dp="${EVAL_VLLM_DATA_PARALLEL_SIZE:-}"
    local max_model_len="${EVAL_VLLM_MAX_MODEL_LEN:-32768}"
    local swap_space="${EVAL_VLLM_SWAP_SPACE:-32}"
    local trust_remote_code="${EVAL_VLLM_TRUST_REMOTE_CODE:-}"
    local tool_call_parser="${EVAL_VLLM_TOOL_CALL_PARSER:-}"
    local reasoning_parser="${EVAL_VLLM_REASONING_PARSER:-}"
    local extra_args="${EVAL_VLLM_EXTRA_ARGS:-}"
    local hf_overrides="${EVAL_VLLM_HF_OVERRIDES:-}"

    # Build command array
    VLLM_CMD=(
        "$python_bin" -m vllm.entrypoints.openai.api_server
        --model "$model"
        --host 0.0.0.0 --port "${VLLM_PORT:-8000}"
        --served-model-name "$model"
        --tensor-parallel-size "$tp"
        --gpu-memory-utilization "$gpu_mem_util"
        --swap-space "$swap_space"
        --disable-custom-all-reduce
    )

    if [ -n "$dp" ] && [ "$dp" -gt 1 ] 2>/dev/null; then
        VLLM_CMD+=(--data-parallel-size "$dp")
    fi

    if [ -n "$max_model_len" ]; then
        VLLM_CMD+=(--max-model-len "$max_model_len")
    fi

    if [ "$trust_remote_code" = "1" ]; then
        VLLM_CMD+=(--trust-remote-code)
    fi

    if [ -n "$tool_call_parser" ]; then
        VLLM_CMD+=(--enable-auto-tool-choice --tool-call-parser "$tool_call_parser")
    fi

    if [ -n "$reasoning_parser" ]; then
        VLLM_CMD+=(--reasoning-parser "$reasoning_parser")
    fi

    # HF model config overrides (JSON string, properly quoted)
    if [ -n "$hf_overrides" ]; then
        VLLM_CMD+=(--hf-overrides "$hf_overrides")
    fi

    # Append extra args (space-separated string)
    if [ -n "$extra_args" ]; then
        # shellcheck disable=SC2206
        VLLM_CMD+=($extra_args)
    fi

    # Log what we built
    echo "vLLM command config:"
    echo "  TP=$tp, DP=${dp:-1}, swap=$swap_space, max_model_len=${max_model_len:-auto}"
    echo "  trust_remote_code=${trust_remote_code:-no}"
    echo "  tool_call_parser=${tool_call_parser:-none}"
    echo "  reasoning_parser=${reasoning_parser:-none}"
    echo "  extra_args=${extra_args:-none}"
    echo "  Full command: ${VLLM_CMD[*]}"
}
