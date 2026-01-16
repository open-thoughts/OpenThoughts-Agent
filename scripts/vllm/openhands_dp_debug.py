#!/usr/bin/env python3
"""
vLLM Ray Serve deployment script for OpenHands with tool calling support.

Based on dp_debug.py but with enable_auto_tool_choice and tool_call_parser enabled.

All parameters can be set via environment variables:
  VLLM_MODEL_PATH          - Model path (default: QuantTrio/GLM-4.6-AWQ)
  VLLM_TENSOR_PARALLEL_SIZE - Tensor parallel size (default: 8)
  VLLM_PIPELINE_PARALLEL_SIZE - Pipeline parallel size (default: 1)
  VLLM_GPU_MEMORY_UTILIZATION - GPU memory utilization (default: 0.95)
  VLLM_MAX_MODEL_LEN       - Max model length (default: 32768)
  VLLM_MAX_NUM_SEQS        - Max concurrent sequences (default: 256)
  VLLM_MAX_NUM_BATCHED_TOKENS - Max batched tokens (default: 16384)
  VLLM_ENABLE_EXPERT_PARALLEL - Enable expert parallel (default: true)
  VLLM_KV_CACHE_DTYPE      - KV cache dtype: auto, fp8_e5m2 (default: auto)
  VLLM_ENABLE_PREFIX_CACHING - Enable prefix caching (default: false)
  VLLM_SWAP_SPACE          - Swap space in GB (default: 4)
  VLLM_BLOCK_SIZE          - Block size (default: 16)
  VLLM_ENABLE_CHUNKED_PREFILL - Enable chunked prefill (default: true)
  VLLM_ALL2ALL_BACKEND     - All2all backend: pplx, deepep_low_latency (default: pplx)
  VLLM_ENABLE_EPLB         - Enable expert load balancing (default: false)
  VLLM_EPLB_NUM_REDUNDANT_EXPERTS - EPLB redundant experts (default: 32)
  VLLM_TOOL_CALL_PARSER    - Tool call parser (default: glm45)
"""

from ray.serve.llm import LLMConfig, build_openai_app
from ray import serve
import ray
import os


def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean from environment variable."""
    val = os.environ.get(name, "").lower()
    if val in ("true", "1", "yes"):
        return True
    elif val in ("false", "0", "no"):
        return False
    return default


def get_env_float(name: str, default: float) -> float:
    """Get float from environment variable."""
    val = os.environ.get(name, "")
    if val:
        try:
            return float(val)
        except ValueError:
            pass
    return default


def get_env_int(name: str, default: int) -> int:
    """Get int from environment variable."""
    val = os.environ.get(name, "")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return default


def main():
    # Connect to Ray cluster
    ray.init(address="auto")

    # Read configuration from environment variables
    model_path = os.environ.get("VLLM_MODEL_PATH", "QuantTrio/GLM-4.6-AWQ")
    tensor_parallel_size = get_env_int("VLLM_TENSOR_PARALLEL_SIZE", 8)
    pipeline_parallel_size = get_env_int("VLLM_PIPELINE_PARALLEL_SIZE", 1)
    gpu_memory_utilization = get_env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.95)
    max_model_len = get_env_int("VLLM_MAX_MODEL_LEN", 32768)
    max_num_seqs = get_env_int("VLLM_MAX_NUM_SEQS", 256)
    max_num_batched_tokens = get_env_int("VLLM_MAX_NUM_BATCHED_TOKENS", 16384)
    enable_expert_parallel = get_env_bool("VLLM_ENABLE_EXPERT_PARALLEL", True)
    kv_cache_dtype = os.environ.get("VLLM_KV_CACHE_DTYPE", "auto")
    enable_prefix_caching = get_env_bool("VLLM_ENABLE_PREFIX_CACHING", False)
    swap_space = get_env_int("VLLM_SWAP_SPACE", 4)
    block_size = get_env_int("VLLM_BLOCK_SIZE", 16)
    enable_chunked_prefill = get_env_bool("VLLM_ENABLE_CHUNKED_PREFILL", True)
    all2all_backend = os.environ.get("VLLM_ALL2ALL_BACKEND", "pplx")
    enable_eplb = get_env_bool("VLLM_ENABLE_EPLB", False)
    eplb_num_redundant_experts = get_env_int("VLLM_EPLB_NUM_REDUNDANT_EXPERTS", 32)
    tool_call_parser = os.environ.get("VLLM_TOOL_CALL_PARSER", "glm47")
    reasoning_parser = os.environ.get("VLLM_REASONING_PARSER", "glm45")


    hf_token = os.environ.get("HF_TOKEN", "")

    # Print configuration
    print("=" * 60)
    print("vLLM Ray Serve Configuration (OpenHands)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Tensor Parallel Size: {tensor_parallel_size}")
    print(f"Pipeline Parallel Size: {pipeline_parallel_size}")
    print(f"GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"Max Model Length: {max_model_len}")
    print(f"Max Num Seqs: {max_num_seqs}")
    print(f"Max Num Batched Tokens: {max_num_batched_tokens}")
    print(f"Enable Expert Parallel: {enable_expert_parallel}")
    print(f"KV Cache Dtype: {kv_cache_dtype}")
    print(f"Enable Prefix Caching: {enable_prefix_caching}")
    print(f"Swap Space: {swap_space} GB")
    print(f"Block Size: {block_size}")
    print(f"Enable Chunked Prefill: {enable_chunked_prefill}")
    print(f"All2All Backend: {all2all_backend}")
    print(f"Enable EPLB: {enable_eplb}")
    if enable_eplb:
        print(f"EPLB Redundant Experts: {eplb_num_redundant_experts}")
    print(f"Enable Auto Tool Choice: True")
    print(f"Tool Call Parser: {tool_call_parser}")
    print(f"Reasoning Parser: {reasoning_parser}")

    print("=" * 60)

    # Build engine kwargs
    engine_kwargs = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "enable_expert_parallel": enable_expert_parallel,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enforce_eager": True,
        "trust_remote_code": True,
        "swap_space": swap_space,
        "block_size": block_size,
        "enable_chunked_prefill": enable_chunked_prefill,
        "enable_prefix_caching": enable_prefix_caching,
        # Tool calling support for OpenHands
        "enable_auto_tool_choice": True,
        "tool_call_parser": tool_call_parser,
        "reasoning_parser": reasoning_parser,
    }

    # Add KV cache dtype if not auto
    if kv_cache_dtype and kv_cache_dtype != "auto":
        engine_kwargs["kv_cache_dtype"] = kv_cache_dtype

    # Add expert parallel specific settings
    if enable_expert_parallel:
        # Set all2all backend via environment variable for vLLM
        # vLLM reads VLLM_ALL2ALL_BACKEND from env
        pass

    # Add EPLB settings if enabled
    if enable_eplb:
        engine_kwargs["enable_eplb"] = True
        engine_kwargs["eplb_num_redundant_experts"] = eplb_num_redundant_experts

    # Create placement group config for parallelism across GPUs
    # Total GPUs = tensor_parallel_size * pipeline_parallel_size
    total_gpus = tensor_parallel_size * pipeline_parallel_size
    placement_group_config = dict(
        bundles=[{"GPU": 1}] * total_gpus,
        strategy="SPREAD",
    )

    # Build LLM config
    llm_config = LLMConfig(
        placement_group_config=placement_group_config,
        model_loading_config={"model_id": "glm", "model_source": model_path},
        engine_kwargs=engine_kwargs,
        runtime_env={
            "env_vars": {
                "VLLM_LOGGING_LEVEL": "DEBUG",
                "HF_TOKEN": hf_token,
                "HUGGING_FACE_HUB_TOKEN": hf_token,
                "HF_HOME": "/tmp/hf_home",
                "TRITON_CACHE_DIR": "/tmp/triton_cache",
                "TORCH_COMPILE_CACHE_DIR": "/tmp/torch_cache",
                "VLLM_ALL2ALL_BACKEND": all2all_backend,
            }
        },
    )

    # Get HTTP port from environment (default 8000)
    http_port = get_env_int("SERVE_HTTP_PORT", 8000)

    print("Building OpenAI-compatible app...")
    dp_app = build_openai_app({"llm_configs": [llm_config]})

    print(f"Starting Ray Serve on port {http_port}...")
    # Start serve with custom HTTP options for port
    serve.start(http_options={"host": "0.0.0.0", "port": http_port})
    serve.run(dp_app, name="vllm-dp-app", blocking=True)


if __name__ == "__main__":
    main()
