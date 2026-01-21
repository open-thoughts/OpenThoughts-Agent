#!/usr/bin/env python3
"""Start a vLLM API server backed by a Ray cluster.

This script accepts a minimal set of required arguments and passes any additional
arguments directly through to the vLLM API server. This allows flexible configuration
without maintaining mappings between config fields and CLI args.

Usage:
    python start_vllm_ray_controller.py \
        --ray-address localhost:6379 \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 4 \
        -- \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.9 \
        --enable-chunked-prefill

Or without the explicit separator (unknown args are passed through):
    python start_vllm_ray_controller.py \
        --ray-address localhost:6379 \
        --max-num-seqs 16 \
        --gpu-memory-utilization 0.9
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import ray
except ImportError:  # pragma: no cover
    ray = None

_CLI_FLAG_PATTERN = re.compile(r"--[a-z0-9-]+")


def _release_torch_memory() -> None:
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _discover_cli_flags() -> set[str]:
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as exc:  # pragma: no cover - best effort
        print(f"[start_vllm_ray_controller] Warning: unable to inspect vLLM CLI flags: {exc}")
        return set()

    flags = set(_CLI_FLAG_PATTERN.findall(result.stdout or ""))
    flags.update(_CLI_FLAG_PATTERN.findall(result.stderr or ""))
    return flags


@functools.lru_cache(maxsize=1)
def _supported_flags() -> set[str]:
    return _discover_cli_flags()


def _flag_supported(flag: str) -> bool:
    return flag in _supported_flags()


def _append_flag(cmd: List[str], flag: str, value: str | None = None) -> None:
    if value is None:
        cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])


def build_vllm_command(args: argparse.Namespace, extra_args: List[str]) -> List[str]:
    """Build the vLLM command from parsed args and pass-through arguments."""
    env = os.environ
    model = args.model or env.get("VLLM_MODEL_PATH")
    if not model:
        raise ValueError("--model or VLLM_MODEL_PATH environment variable is required")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--distributed-executor-backend",
        "ray",
        "--trust-remote-code",
    ]

    # Data parallel size
    if args.data_parallel_size > 1:
        if _flag_supported("--data-parallel-size"):
            _append_flag(cmd, "--data-parallel-size", str(args.data_parallel_size))
        else:
            print(
                "[start_vllm_ray_controller] WARNING: --data-parallel-size not supported by vLLM version",
                file=sys.stderr,
            )

    # Pipeline parallel size
    if args.pipeline_parallel_size > 1:
        if _flag_supported("--pipeline-parallel-size"):
            _append_flag(cmd, "--pipeline-parallel-size", str(args.pipeline_parallel_size))
            print(f"[start_vllm_ray_controller] Enabling pipeline parallelism: {args.pipeline_parallel_size}")
        else:
            print(
                "[start_vllm_ray_controller] WARNING: --pipeline-parallel-size not supported by vLLM version",
                file=sys.stderr,
            )

    # Ray address
    if args.ray_address:
        if _flag_supported("--ray-address"):
            _append_flag(cmd, "--ray-address", args.ray_address)
        else:
            print(
                "[start_vllm_ray_controller] --ray-address flag unsupported; relying on RAY_ADDRESS env",
                file=sys.stderr,
            )

    # Served model name
    if args.served_model_name:
        _append_flag(cmd, "--served-model-name", args.served_model_name)

    # Pass through all extra arguments directly to vLLM
    if extra_args:
        cmd.extend(extra_args)

    return cmd


def write_endpoint_json(endpoint_json: Path, host: str, port: int, model: str, args: argparse.Namespace) -> None:
    endpoint_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": args.served_model_name or model,
        "endpoint_url": f"http://{host}:{port}",
        "ray_address": args.ray_address,
        "created_by": "start_vllm_ray_controller",
        "metadata": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": args.pipeline_parallel_size,
            "data_parallel_size": args.data_parallel_size,
        },
    }
    endpoint_json.write_text(json.dumps(payload, indent=2))
    print(f"✓ Endpoint configuration written to {endpoint_json}")


def parse_args() -> tuple[argparse.Namespace, List[str]]:
    """Parse known arguments and return remaining as pass-through for vLLM."""
    parser = argparse.ArgumentParser(
        description="Launch vLLM API server backed by Ray. Unknown arguments are passed through to vLLM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ray-address", required=True, help="Address of the Ray head (host:port)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the API server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server (default: 8000)")
    parser.add_argument("--model", type=str, help="Model path (defaults to VLLM_MODEL_PATH env var)")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size (default: 1)",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Pipeline parallel size (default: 1)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Data parallel size (default: 1)",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help="Custom model name for the API",
    )
    parser.add_argument(
        "--endpoint-json",
        type=Path,
        default=None,
        help="Optional path to write endpoint metadata JSON",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Use parse_known_args to capture any additional vLLM arguments
    args, extra_args = parser.parse_known_args()
    return args, extra_args


def main() -> None:
    args, extra_args = parse_args()
    _release_torch_memory()

    env = os.environ.copy()
    if args.verbose:
        env.setdefault("VLLM_LOG_LEVEL", "INFO")

    # NOTE: RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES and RAY_NOSET_CUDA_VISIBLE_DEVICES
    # must be set BEFORE the Ray cluster is started (in the sbatch template), not here.
    # Setting them here is too late - Ray actors have already been spawned with modified
    # CUDA_VISIBLE_DEVICES. See universal_*gen.sbatch and universal_eval.sbatch.

    print("vLLM controller environment snapshot:")
    for key in (
        "TRITON_CC",
        "LD_LIBRARY_PATH",
        "PATH",
        "HF_HOME",
        "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_ADDRESS",
    ):
        value = env.get(key, "<unset>")
        print(f"  {key}={value}")
    sys.stdout.flush()

    ray_address = env.get("RAY_ADDRESS") or args.ray_address
    if ray is not None and ray_address:
        try:
            print(f"[start_vllm_ray_controller] Inspecting Ray resources at {ray_address} before placement.")
            ray.init(address=ray_address, ignore_reinit_error=True)
            cluster_resources = ray.cluster_resources()
            nodes = ray.nodes()
            print(f"[start_vllm_ray_controller] Ray cluster resources: {cluster_resources}")
            print(
                f"[start_vllm_ray_controller] Ray nodes ({len(nodes)}): "
                f"{[node.get('NodeID') for node in nodes]}"
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[start_vllm_ray_controller] Warning: unable to query Ray cluster resources: {exc}")
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass

    cmd = build_vllm_command(args, extra_args)
    print("Launching vLLM controller:")
    print("  " + " ".join(cmd))
    if extra_args:
        print(f"  (pass-through args: {' '.join(extra_args)})")

    process = subprocess.Popen(cmd, env=env)

    def _flush_loop():
        try:
            while process.poll() is None:
                sys.stdout.flush()
                sys.stderr.flush()
                time.sleep(5)
        except Exception:
            pass

    flush_thread = threading.Thread(target=_flush_loop, daemon=True)
    flush_thread.start()

    def _shutdown(signum, frame):
        print(f"Received signal {signum}, terminating vLLM server...")
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.endpoint_json:
        model = args.model or os.environ.get("VLLM_MODEL_PATH", "unknown")
        write_endpoint_json(args.endpoint_json, args.host, args.port, model, args)

    exit_code = process.wait()
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
