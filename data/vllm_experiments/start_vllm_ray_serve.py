#!/usr/bin/env python3
"""Launch a vLLM Ray Serve deployment with optional multi-replica support.

Modified version with location="HeadOnly" to only run HTTP proxy on head node.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict

try:
    import ray
    from ray import serve
    from ray.serve.llm import LLMConfig, LLMServingArgs, build_openai_app
    from ray.serve.config import ProxyLocation
except ImportError as exc:  # pragma: no cover - defensive guard for misconfigured envs
    print(
        "[start_vllm_ray_serve] Ray Serve LLM components are unavailable. "
        "Install a Ray version that includes ray.serve.llm before using the "
        "ray_serve backend.",
        file=sys.stderr,
    )
    raise


STOP_REQUESTED = False


def _register_signal_handlers() -> None:
    """Ensure SIGTERM/SIGINT trigger a graceful shutdown."""

    def _handler(signum: int, _: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"[start_vllm_ray_serve] Received signal {signum}, shutting down...")

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _safe_cast(value: str | None, cast_type: Any) -> Any | None:
    if value is None or value == "":
        return None
    try:
        return cast_type(value)
    except Exception:
        return None


def _gather_engine_kwargs(
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> Dict[str, Any]:
    env = os.environ
    kwargs: Dict[str, Any] = {
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
    }

    int_fields = [
        ("VLLM_MAX_MODEL_LEN", "max_model_len"),
        ("VLLM_MAX_NUM_SEQS", "max_num_seqs"),
        ("VLLM_MAX_SEQ_LEN_TO_CAPTURE", "max_seq_len_to_capture"),
    ]
    float_fields = [
        ("VLLM_GPU_MEMORY_UTILIZATION", "gpu_memory_utilization"),
        ("VLLM_SWAP_SPACE", "swap_space"),
    ]

    for env_key, field_name in int_fields:
        value = _safe_cast(env.get(env_key), int)
        if value is not None:
            kwargs[field_name] = value

    for env_key, field_name in float_fields:
        value = _safe_cast(env.get(env_key), float)
        if value is not None:
            kwargs[field_name] = value

    bool_fields = [
        ("VLLM_TRUST_REMOTE_CODE", "trust_remote_code"),
        ("VLLM_USE_DEEP_GEMM", "use_deep_gemm"),
        ("VLLM_ENABLE_EXPERT_PARALLEL", "enable_expert_parallel"),
        ("VLLM_DISABLE_LOG_REQUESTS", "disable_log_requests"),
        ("VLLM_ENABLE_AUTO_TOOL_CHOICE", "enable_auto_tool_choice"),
    ]

    for env_key, field_name in bool_fields:
        if _parse_bool(env.get(env_key)):
            kwargs[field_name] = True

    str_fields = [
        ("VLLM_TOOL_CALL_PARSER", "tool_call_parser"),
        ("VLLM_REASONING_PARSER", "reasoning_parser"),
    ]
    for env_key, field_name in str_fields:
        value = env.get(env_key)
        if value:
            kwargs[field_name] = value

    overrides = env.get("VLLM_HF_OVERRIDES")
    if overrides:
        try:
            kwargs["hf_overrides"] = json.loads(overrides)
        except json.JSONDecodeError:
            print(
                "[start_vllm_ray_serve] Warning: unable to parse VLLM_HF_OVERRIDES "
                f"as JSON: {overrides}",
                file=sys.stderr,
            )

    temperature = env.get("VLLM_DEFAULT_TEMPERATURE")
    if temperature:
        value = _safe_cast(temperature, float)
        if value is not None:
            kwargs.setdefault("temperature", value)

    seed = env.get("VLLM_DEFAULT_SEED")
    if seed:
        value = _safe_cast(seed, int)
        if value is not None:
            kwargs.setdefault("seed", value)

    return kwargs


def _collect_runtime_env() -> Dict[str, Any]:
    """Forward key environment variables to Ray replica workers."""
    env_vars: Dict[str, str] = {}
    allowlisted_prefixes = (
        "VLLM_",
        "TRANSFORMERS_CACHE",
        "HF_",
        "TOKENIZERS_PARALLELISM",
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
    )
    for key, value in os.environ.items():
        if key.startswith(allowlisted_prefixes):
            env_vars[key] = value

    runtime_env: Dict[str, Any] = {}
    if env_vars:
        runtime_env["env_vars"] = env_vars
    return runtime_env


def _write_endpoint_json(
    endpoint_path: Path,
    host: str,
    port: int,
    model_name: str,
    metadata: Dict[str, Any],
    ray_address: str,
) -> None:
    payload = {
        "model_name": model_name,
        "endpoint_url": f"http://{host}:{port}",
        "ray_address": ray_address,
        "created_by": "start_vllm_ray_serve",
        "metadata": metadata,
    }
    endpoint_path.parent.mkdir(parents=True, exist_ok=True)
    endpoint_path.write_text(json.dumps(payload, indent=2))
    print(f"[start_vllm_ray_serve] ✓ Endpoint configuration written to {endpoint_path}")


def _await_serve_ready(app_name: str, timeout_s: int = 300) -> bool:
    """Poll Ray Serve until the application reports RUNNING."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = serve.status()
        app_status = status.applications.get(app_name)
        if app_status and getattr(app_status.status, "name", str(app_status.status)) == "RUNNING":
            return True
        if STOP_REQUESTED:
            return False
        time.sleep(2)
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a Ray Serve-hosted vLLM deployment.")
    parser.add_argument("--ray-address", required=True, help="Ray head address (host:port)")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP host for the Ray Serve endpoint")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port for the Ray Serve endpoint")
    parser.add_argument(
        "--endpoint-json",
        type=Path,
        required=True,
        help="Path to write the endpoint metadata JSON file",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel size for each replica (overrides VLLM_TENSOR_PARALLEL_SIZE)",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=None,
        help="Pipeline parallel size for each replica (overrides VLLM_PIPELINE_PARALLEL_SIZE)",
    )
    parser.add_argument(
        "--app-name",
        default="vllm-openai-app",
        help="Serve application name used when deploying (default: vllm-openai-app)",
    )
    parser.add_argument(
        "--proxy-location",
        default="HeadOnly",
        choices=["HeadOnly", "EveryNode", "NoServer"],
        help="Where to run HTTP proxy (default: HeadOnly)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _register_signal_handlers()

    model_path = os.environ.get("VLLM_MODEL_PATH")
    if not model_path:
        raise ValueError("VLLM_MODEL_PATH environment variable is required for ray_serve backend.")

    model_name = os.environ.get("VLLM_CUSTOM_MODEL_NAME") or Path(model_path).name

    env_tensor_parallel = _safe_cast(os.environ.get("VLLM_TENSOR_PARALLEL_SIZE"), int) or 1
    env_pipeline_parallel = _safe_cast(os.environ.get("VLLM_PIPELINE_PARALLEL_SIZE"), int) or 1

    tensor_parallel_size = args.tensor_parallel_size or env_tensor_parallel
    pipeline_parallel_size = args.pipeline_parallel_size or env_pipeline_parallel

    num_replicas = _safe_cast(os.environ.get("VLLM_NUM_REPLICAS"), int) or 1

    runtime_env = _collect_runtime_env()
    engine_kwargs = _gather_engine_kwargs(tensor_parallel_size, pipeline_parallel_size)
    # disable_log_requests is a frontend flag in vLLM; Ray Serve LLM rejects it
    engine_kwargs.pop("disable_log_requests", None)

    model_loading_config = {
        "model_id": model_name,
        "model_source": model_path,
    }

    deployment_config: Dict[str, Any] = {
        "autoscaling_config": {
            "min_replicas": num_replicas,
            "max_replicas": num_replicas,
        },
    }

    accelerator_type = os.environ.get("VLLM_ACCELERATOR_TYPE")

    llm_config = LLMConfig(
        model_loading_config=model_loading_config,
        runtime_env=runtime_env or None,
        deployment_config=deployment_config,
        engine_kwargs=engine_kwargs,
        accelerator_type=accelerator_type,
    )

    serving_args = LLMServingArgs(llm_configs=[llm_config])

    print("[start_vllm_ray_serve] Connecting to Ray cluster...")
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    # Map proxy location string to enum
    proxy_location_map = {
        "HeadOnly": ProxyLocation.HeadOnly,
        "EveryNode": ProxyLocation.EveryNode,
        "NoServer": ProxyLocation.Disabled,
    }
    proxy_location = proxy_location_map.get(args.proxy_location, ProxyLocation.HeadOnly)

    print(
        f"[start_vllm_ray_serve] Starting Ray Serve HTTP proxy on "
        f"{args.host}:{args.port} (app: {args.app_name}, location: {args.proxy_location})"
    )
    serve.start(http_options={
        "host": args.host,
        "port": args.port,
        "location": proxy_location,
    })

    llm_app = build_openai_app(serving_args)
    serve.run(llm_app, name=args.app_name)

    if not _await_serve_ready(args.app_name):
        raise RuntimeError("Timed out waiting for Ray Serve application to become healthy.")

    # For endpoint JSON, use the actual head node IP if host is 0.0.0.0
    endpoint_host = args.host
    if endpoint_host == "0.0.0.0":
        # Extract head node IP from ray address
        endpoint_host = args.ray_address.split(":")[0]

    metadata = {
        "backend": "ray_serve",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "num_replicas": num_replicas,
        "proxy_location": args.proxy_location,
    }

    _write_endpoint_json(
        endpoint_path=args.endpoint_json,
        host=endpoint_host,
        port=args.port,
        model_name=model_name,
        metadata=metadata,
        ray_address=args.ray_address,
    )

    print("[start_vllm_ray_serve] Ray Serve deployment is live. Waiting for shutdown signal.")
    try:
        while not STOP_REQUESTED:
            time.sleep(5)
    finally:
        print("[start_vllm_ray_serve] Shutting down Ray Serve deployment...")
        try:
            serve.shutdown()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface errors clearly
        print(f"[start_vllm_ray_serve] ERROR: {exc}", file=sys.stderr)
        raise
