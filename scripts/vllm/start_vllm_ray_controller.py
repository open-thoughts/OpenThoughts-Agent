#!/usr/bin/env python3
"""Start a vLLM API server backed by a Ray cluster."""

from __future__ import annotations

import argparse
import json
import os
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

from scripts.vllm.start_vllm_cluster import VLLMCluster


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


def _supported_flags() -> set[str]:
    return VLLMCluster._discover_cli_flags()


def _flag_supported(flag: str) -> bool:
    return flag in _supported_flags()


def _append_flag(cmd: List[str], flag: str, value: str | None = None) -> None:
    if value is None:
        cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])


def _load_extra_args_from_env() -> List[str]:
    payload = os.environ.get("VLLM_SERVER_EXTRA_ARGS_JSON")
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[start_vllm_ray_controller] Warning: unable to parse VLLM_SERVER_EXTRA_ARGS_JSON: {exc}")
        return []
    if not isinstance(data, list):
        print(
            "[start_vllm_ray_controller] Warning: VLLM_SERVER_EXTRA_ARGS_JSON must be a JSON array; ignoring value."
        )
        return []
    return [str(item) for item in data if item is not None]


def build_vllm_command(args: argparse.Namespace) -> List[str]:
    env = os.environ
    model = args.model or env.get("VLLM_MODEL_PATH")
    if not model:
        raise ValueError("VLLM_MODEL_PATH environment variable (or --model) is required")

    tensor_parallel = args.tensor_parallel_size or env.get("VLLM_TENSOR_PARALLEL_SIZE") or 1
    pipeline_parallel = args.pipeline_parallel_size or env.get("VLLM_PIPELINE_PARALLEL_SIZE") or 1
    data_parallel = args.data_parallel_size or env.get("VLLM_DATA_PARALLEL_SIZE") or 1
    custom_name = env.get("VLLM_CUSTOM_MODEL_NAME")
    hf_overrides = env.get("VLLM_HF_OVERRIDES")
    max_num_seqs = env.get("VLLM_MAX_NUM_SEQS")
    gpu_mem_util = env.get("VLLM_GPU_MEMORY_UTILIZATION")
    enable_expert_parallel = env.get("VLLM_ENABLE_EXPERT_PARALLEL") == "1"
    swap_space = env.get("VLLM_SWAP_SPACE")
    max_seq_len_to_capture = env.get("VLLM_MAX_SEQ_LEN_TO_CAPTURE")
    trust_remote_code = env.get("VLLM_TRUST_REMOTE_CODE") == "1"
    disable_log_requests = env.get("VLLM_DISABLE_LOG_REQUESTS") == "1"
    enable_auto_tool_choice = env.get("VLLM_ENABLE_AUTO_TOOL_CHOICE") == "1"
    tool_call_parser = env.get("VLLM_TOOL_CALL_PARSER")
    reasoning_parser = env.get("VLLM_REASONING_PARSER")

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
        str(tensor_parallel),
        "--distributed-executor-backend",
        "ray",
    ]
    if _flag_supported("--data-parallel-size"):
        _append_flag(cmd, "--data-parallel-size", str(data_parallel))
    else:
        print(
            "[start_vllm_ray_controller] WARNING: --data-parallel-size not supported by vLLM version",
            file=sys.stderr,
        )

    # NEW: Add pipeline parallelism support
    if pipeline_parallel > 1:
        if _flag_supported("--pipeline-parallel-size"):
            _append_flag(cmd, "--pipeline-parallel-size", str(pipeline_parallel))
            print(f"[start_vllm_ray_controller] Enabling pipeline parallelism: {pipeline_parallel}")
        else:
            print(
                f"[start_vllm_ray_controller] WARNING: --pipeline-parallel-size not supported by vLLM version",
                file=sys.stderr,
            )

    if args.ray_address:
        if _flag_supported("--ray-address"):
            _append_flag(cmd, "--ray-address", args.ray_address)
        else:
            print(
                "[start_vllm_ray_controller] --ray-address flag unsupported; relying on RAY_ADDRESS env",
                file=sys.stderr,
            )

    if custom_name:
        _append_flag(cmd, "--served-model-name", custom_name)
    if hf_overrides:
        _append_flag(cmd, "--hf-overrides", hf_overrides)
    if max_num_seqs:
        _append_flag(cmd, "--max-num-seqs", max_num_seqs)
    if gpu_mem_util:
        _append_flag(cmd, "--gpu-memory-utilization", gpu_mem_util)
    if enable_expert_parallel and _flag_supported("--enable-expert-parallel"):
        cmd.append("--enable-expert-parallel")
    if swap_space and _flag_supported("--swap-space"):
        _append_flag(cmd, "--swap-space", swap_space)
    if max_seq_len_to_capture and _flag_supported("--max-seq-len-to-capture"):
        _append_flag(cmd, "--max-seq-len-to-capture", max_seq_len_to_capture)
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    if disable_log_requests:
        if _flag_supported("--no-enable-log-requests"):
            cmd.append("--no-enable-log-requests")
        elif _flag_supported("--disable-log-requests"):
            cmd.append("--disable-log-requests")
    if enable_auto_tool_choice and _flag_supported("--enable-auto-tool-choice"):
        cmd.append("--enable-auto-tool-choice")
    if tool_call_parser and _flag_supported("--tool-call-parser"):
        _append_flag(cmd, "--tool-call-parser", tool_call_parser)
    if reasoning_parser and _flag_supported("--reasoning-parser"):
        _append_flag(cmd, "--reasoning-parser", reasoning_parser)
        if _flag_supported("--enable-reasoning"):
            cmd.append("--enable-reasoning")
        else:
            print(
                "[start_vllm_ray_controller] WARNING: --enable-reasoning not supported by current vLLM version",
                file=sys.stderr,
            )

    overrides = {
        "temperature": args.temperature,
        "seed": args.seed,
    }
    _append_flag(cmd, "--override-generation-config", json.dumps(overrides))

    max_model_len = args.max_model_len or env.get("VLLM_MAX_MODEL_LEN")
    if max_model_len:
        _append_flag(cmd, "--max-model-len", max_model_len)

    extra_cli_args = _load_extra_args_from_env()
    if extra_cli_args:
        cmd.extend(extra_cli_args)

    return cmd


def write_endpoint_json(endpoint_json: Path, host: str, port: int, model: str, extra: Dict[str, str]) -> None:
    endpoint_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": extra.get("VLLM_CUSTOM_MODEL_NAME") or model,
        "endpoint_url": f"http://{host}:{port}",
        "ray_address": extra.get("RAY_ADDRESS"),
        "created_by": "start_vllm_ray_controller",
        "metadata": {
            "tensor_parallel_size": extra.get("VLLM_TENSOR_PARALLEL_SIZE"),
            "pipeline_parallel_size": extra.get("VLLM_PIPELINE_PARALLEL_SIZE"),
            "data_parallel_size": extra.get("VLLM_DATA_PARALLEL_SIZE"),
        },
    }
    endpoint_json.write_text(json.dumps(payload, indent=2))
    print(f"✓ Endpoint configuration written to {endpoint_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch vLLM API server backed by Ray.")
    parser.add_argument("--ray-address", required=True, help="Address of the Ray head (host:port)")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface for the API server (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server (default: 8000)")
    parser.add_argument("--model", type=str, help="Model path (defaults to VLLM_MODEL_PATH env var)")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        help="Tensor parallel size (defaults to VLLM_TENSOR_PARALLEL_SIZE env var or 1)",
    )
    parser.add_argument(
        "--pipeline-parallel-size",  # NEW
        type=int,
        help="Pipeline parallel size (defaults to VLLM_PIPELINE_PARALLEL_SIZE env var or 1)",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        help="Data parallel size (defaults to VLLM_DATA_PARALLEL_SIZE env var or 1)",
    )
    parser.add_argument(
        "--endpoint-json",
        type=Path,
        default=None,
        help="Optional path to write endpoint metadata JSON",
    )
    parser.add_argument("--temperature", type=float, default=0.1, help="Default sampling temperature")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed")
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional override for --max-model-len",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _release_torch_memory()

    env = os.environ.copy()
    if args.verbose:
        env.setdefault("VLLM_LOG_LEVEL", "INFO")

    print("vLLM controller environment snapshot:")
    for key in ("TRITON_CC", "LD_LIBRARY_PATH", "PATH", "HF_HOME", "PYTHONPATH"):
        value = env.get(key, "<unset>")
        print(f"  {key}={value}")
    sys.stdout.flush()

    cmd = build_vllm_command(args)
    print("Launching vLLM controller:")
    print("  " + " ".join(cmd))

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
        write_endpoint_json(args.endpoint_json, args.host, args.port, cmd[4], env)

    exit_code = process.wait()
    if exit_code != 0:
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
