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
from typing import List

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
        # vllm-tpu's --help import path triggers libtpu/XLA bring-up which can
        # hang for minutes on a cold worker; 60s lets us fall back to the
        # assume-supported path before a TPU import deadlock.
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
    except Exception as exc:  # pragma: no cover - best effort
        print(
            f"[start_vllm_ray_controller] Warning: unable to inspect vLLM CLI flags ({exc}); "
            "falling back to assume-supported for all flags we emit.",
            flush=True,
        )
        return set()

    flags = set(_CLI_FLAG_PATTERN.findall(result.stdout or ""))
    flags.update(_CLI_FLAG_PATTERN.findall(result.stderr or ""))
    return flags


@functools.lru_cache(maxsize=1)
def _supported_flags() -> set[str]:
    # ``VLLM_SKIP_FLAG_DISCOVERY=1`` short-circuits the ``vllm --help`` probe.
    # The iris+TPU path sets this because importing vllm-tpu cold-bootstraps
    # libtpu inside a subprocess.run, which can hang for >5 min on the first
    # invocation and deadlock the parent controller with no diagnostic output.
    # The fallback "empty set" means the launcher assumes every flag it emits
    # is supported and emits it unconditionally.
    if os.environ.get("VLLM_SKIP_FLAG_DISCOVERY") == "1":
        return set()
    return _discover_cli_flags()


# Flags that the CUDA vLLM api_server accepts but the vLLM-TPU api_server
# rejects with "unrecognized arguments". When VLLM_SKIP_FLAG_DISCOVERY=1 we
# can't probe the real argparse, so we have to know these statically.
_TPU_UNSUPPORTED_API_SERVER_FLAGS = frozenset({
    "--ray-address",
    "--swap-space",
})


def _is_tpu_env() -> bool:
    """True when this process is running on a Cloud TPU host.

    Iris's TPU workers set TPU_ACCELERATOR_TYPE / TPU_TYPE. Our task image
    also sets JAX_PLATFORMS=tpu. Either signal counts.
    """
    return bool(
        os.environ.get("JAX_PLATFORMS", "").startswith("tpu")
        or os.environ.get("TPU_ACCELERATOR_TYPE")
        or os.environ.get("TPU_TYPE")
    )


def _flag_supported(flag: str) -> bool:
    # On TPU we cannot run the vllm --help probe (libtpu cold-bootstrap
    # deadlocks the subprocess). Filter out the few flags the CUDA vLLM
    # api_server accepts but the TPU one rejects, or the api_server child
    # exits immediately with "unrecognized arguments".
    if _is_tpu_env() and flag in _TPU_UNSUPPORTED_API_SERVER_FLAGS:
        return False
    if os.environ.get("VLLM_SKIP_FLAG_DISCOVERY") == "1":
        # Assume yes for everything else.
        return True
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

    # Ray address: vllm-tpu 0.20.0 dropped the --ray-address CLI flag — the
    # entrypoint only accepts RAY_ADDRESS via env var. Caller must set the
    # env var BEFORE invoking subprocess.Popen; see main() below.
    if args.ray_address:
        if os.environ.get("VLLM_SKIP_FLAG_DISCOVERY") == "1":
            # main() has already mirrored args.ray_address into env["RAY_ADDRESS"].
            pass
        elif _flag_supported("--ray-address"):
            _append_flag(cmd, "--ray-address", args.ray_address)
        else:
            print(
                "[start_vllm_ray_controller] --ray-address flag unsupported; relying on RAY_ADDRESS env",
                file=sys.stderr,
                flush=True,
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

    # Mirror --ray-address into RAY_ADDRESS env so vllm-tpu (which dropped
    # the CLI flag in 0.20.0) and modern CUDA vllm (which honors the env)
    # both pick it up. Must happen before build_vllm_command() so the env
    # we hand to subprocess.Popen has it.
    if args.ray_address and os.environ.get("VLLM_SKIP_FLAG_DISCOVERY") == "1":
        env["RAY_ADDRESS"] = args.ray_address

    # Strip extra_args that aren't accepted by vllm-tpu 0.20.0 when the
    # iris/TPU shortcut is in effect. ``--swap-space 0`` is the prime
    # offender: CUDA vLLM accepts swap_space=0 to mean "disable CPU swap"
    # but vllm-tpu doesn't model swap at all and the CLI parser refuses
    # the unknown flag.
    if os.environ.get("VLLM_SKIP_FLAG_DISCOVERY") == "1":
        _TPU_UNACCEPTED = {"--swap-space"}
        filtered: List[str] = []
        skip_next = False
        for arg in extra_args:
            if skip_next:
                skip_next = False
                continue
            if arg in _TPU_UNACCEPTED:
                skip_next = True
                print(
                    f"[start_vllm_ray_controller] Dropping TPU-incompatible flag: {arg}",
                    flush=True,
                )
                continue
            filtered.append(arg)
        extra_args = filtered

    # RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES and RAY_NOSET_CUDA_VISIBLE_DEVICES
    # must be set BEFORE the Ray cluster is started (in the sbatch template), not
    # here — setting them here is too late (Ray actors already spawned with
    # modified CUDA_VISIBLE_DEVICES).

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
    # Skip the Ray-probe sanity check when running in iris/TPU mode — every
    # ray.init / ray.shutdown roundtrip costs ~3-5s and isn't load-bearing
    # (vLLM connects to Ray itself with its own ray.init). Set
    # VLLM_SKIP_RAY_PROBE=1 (iris launcher sets it automatically) to skip.
    if (
        ray is not None
        and ray_address
        and os.environ.get("VLLM_SKIP_RAY_PROBE") != "1"
    ):
        try:
            print(f"[start_vllm_ray_controller] Inspecting Ray resources at {ray_address} before placement.", flush=True)
            ray.init(address=ray_address, ignore_reinit_error=True)
            cluster_resources = ray.cluster_resources()
            nodes = ray.nodes()
            print(f"[start_vllm_ray_controller] Ray cluster resources: {cluster_resources}", flush=True)
            print(
                f"[start_vllm_ray_controller] Ray nodes ({len(nodes)}): "
                f"{[node.get('NodeID') for node in nodes]}",
                flush=True,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            print(f"[start_vllm_ray_controller] Warning: unable to query Ray cluster resources: {exc}", flush=True)
        finally:
            try:
                ray.shutdown()
            except Exception:
                pass
    else:
        print(
            f"[start_vllm_ray_controller] Skipping Ray probe (ray_address={ray_address!r}, "
            f"VLLM_SKIP_RAY_PROBE={os.environ.get('VLLM_SKIP_RAY_PROBE')!r}).",
            flush=True,
        )

    cmd = build_vllm_command(args, extra_args)
    print("Launching vLLM controller:")
    print("  " + " ".join(cmd))
    if extra_args:
        print(f"  (pass-through args: {' '.join(extra_args)})")
    # Flush before Popen so the launch line lands in iris logs even if the
    # child segfaults during startup before Python writes a traceback.
    sys.stdout.flush()
    sys.stderr.flush()

    # Capture stderr to a pipe so quick startup failures get surfaced even
    # when the parent's stdout is buffered; tee both streams to the parent
    # in a background thread until the child exits.
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )

    def _tee_child_output():
        assert process.stdout is not None
        try:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception as exc:  # pragma: no cover
            print(f"[start_vllm_ray_controller] tee error: {exc}", file=sys.stderr, flush=True)

    tee_thread = threading.Thread(target=_tee_child_output, daemon=True)
    tee_thread.start()

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
