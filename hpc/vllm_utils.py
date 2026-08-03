"""vLLM server management for Ray clusters.

This module provides a context manager for managing vLLM server lifecycle
on Ray clusters, eliminating duplicated vLLM setup code across SBATCH scripts.

Usage:
    from hpc.ray_utils import RayCluster, RayClusterConfig
    from hpc.vllm_utils import VLLMServer, VLLMConfig

    ray_config = RayClusterConfig(num_nodes=4, gpus_per_node=4, cpus_per_node=48)

    with RayCluster.from_slurm(ray_config) as ray_cluster:
        vllm_config = VLLMConfig(
            model_path="meta-llama/Llama-3.1-70B-Instruct",
            tensor_parallel_size=4,
        )

        with VLLMServer(vllm_config, ray_cluster) as vllm:
            print(f"vLLM ready at {vllm.endpoint}")
            # ... run inference workloads
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hpc.ray_utils import RayCluster


# Fields from vllm_server/engine config that are for our system, not vLLM
# These are either internal fields, Harbor/Daytona-specific fields, or handled explicitly
_OUR_FIELDS = {
    # Internal/system fields
    "num_replicas",
    "time_limit",
    "endpoint_json_path",
    "model_path",
    # Parallelism fields — handled explicitly by VLLMServer.start() as positional args.
    # Must be excluded here to avoid double-emission (last value wins with argparse).
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "data_parallel_size",
    # Harbor/Daytona engine-specific fields (not vLLM args)
    "type",                   # engine type (e.g., "vllm_local")
    "max_output_tokens",      # Harbor config, not vLLM
    "healthcheck_interval",   # Harbor config, not vLLM
    "vllm_local",            # Daytona backend config
    "model",                  # handled separately via --model
}

# Fields that map to different vLLM CLI arg names
_FIELD_RENAMES = {
    "model_path": "model",
}

# Boolean flags (passed as --flag without value when True)
_BOOLEAN_FLAGS = {
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "enable_auto_tool_choice",
    "enable_expert_parallel",
    "trust_remote_code",
    "disable_log_requests",
    "enable_reasoning",
}

# Boolean flags whose vLLM internal default is True but that OT-Agent
# defaults to OFF unless explicitly enabled by the YAML. Setting the YAML
# key to False (or omitting it) emits `--no-<flag>`; setting it True emits
# `--<flag>`. Also, if neither form appears anywhere in the final CLI args
# (including extra_args), `--no-<flag>` is injected at the end as a
# belt-and-suspenders opt-out.
#
# 2026-05-23: enable_chunked_prefill REMOVED from this set. Original
# rationale was that mid-inference Triton JIT compilation of chunked-
# prefill kernels overran the vLLM shm_broadcast 60 s wait and killed
# the engine with EngineDeadError (see
# `vllm_v2_bugs/bug_c_iter_4_to_8_jit_progression/SUMMARY.md`). We now
# know the shm_broadcast 600s warning is purely informational
# (`feedback_shm_broadcast_warning_non_fatal` memory + vLLM source
# review). The real engine deaths were NCCL collective timeouts upstream.
# Forcing chunked-prefill OFF on MoE models like MiniMax-M2 triggers a
# different problem: vLLM warns the model "does not officially support
# disabling chunked prefill" AND blocks any config with
# max_num_batched_tokens < max_model_len (the SchedulerConfig validator
# at config/scheduler.py:259 conditions on `not enable_chunked_prefill`).
# Letting vLLM's per-model default win is the right move now.
_DEFAULT_OFF_BOOLEAN_FLAGS = {
    "enable_prefix_caching",
}

# Fields that are environment variables, not CLI args.
# When the YAML sets a value for one of these keys, we write the env var
# explicitly as "1" or "0" so the user can FORCE-disable a flag whose
# vLLM default is True (e.g. VLLM_USE_DEEP_GEMM, VLLM_USE_FLASHINFER_SAMPLER
# both default to True in vllm/envs.py). Omitting the key entirely from
# the YAML leaves the env var unset → vLLM uses its built-in default.
_ENV_VAR_FIELDS = {
    "use_deep_gemm": "VLLM_USE_DEEP_GEMM",
    "use_flashinfer_sampler": "VLLM_USE_FLASHINFER_SAMPLER",
    "use_flashinfer_moe_fp16": "VLLM_USE_FLASHINFER_MOE_FP16",
    "pynccl_pyspy_on_sigusr1": "VLLM_PYNCCL_PYSPY_ON_SIGUSR1",
    # NCCL_CUMEM_ENABLE: false → "0". Disables NCCL cuMem buffer
    # registration; candidate workaround for the cudagraph-capture illegal
    # memory access on the cross-node MoE all-to-all. Propagates to Ray DP
    # actors via vLLM's NCCL_ copy-prefix (vllm/ray/ray_env.py).
    "nccl_cumem_enable": "NCCL_CUMEM_ENABLE",
    # CUDA_LAUNCH_BLOCKING: true → "1". Serializes CUDA ops so an async
    # illegal-memory-access aborts synchronously at the offending kernel,
    # making the traceback name the exact failing line (H1 vs H3
    # discriminator for the MiniMax DP=2 capture crash). CUDA_ is NOT a
    # default vLLM copy-prefix → also set vllm_ray_extra_env_vars_to_copy
    # = 'CUDA_LAUNCH_BLOCKING' so vLLM ferries it to the cross-node DP actor.
    "cuda_launch_blocking": "CUDA_LAUNCH_BLOCKING",
}

# Numeric env var fields. Same idea as _ENV_VAR_FIELDS but the value
# is written through unchanged (e.g. an integer seconds-interval) rather
# than coerced to "1"/"0". YAML key absent → env var unset → consumer
# falls back to its own default (typically off / 0).
_NUMERIC_ENV_VAR_FIELDS = {
    "pynccl_trace_flush_interval_sec": "VLLM_PYNCCL_TRACE_FLUSH_INTERVAL_SEC",
    "pynccl_faulthandler_interval_sec": "VLLM_PYNCCL_FAULTHANDLER_INTERVAL_SEC",
    # String passthrough (despite the "numeric" name — value is str-coerced).
    # NCCL_DEBUG / NCCL_DEBUG_SUBSYS surface connection/channel setup so we can
    # tell whether it happens DURING the profile_cudagraph_memory capture.
    "nccl_debug": "NCCL_DEBUG",
    "nccl_debug_subsys": "NCCL_DEBUG_SUBSYS",
    # VLLM_RAY_EXTRA_ENV_VARS_TO_COPY: comma-separated EXACT env var names
    # vLLM copies to the cross-node Ray DP actors in addition to the
    # DEFAULT_ENV_VAR_PREFIXES. Self-copies (it is VLLM_-prefixed and read by
    # get_env_vars_to_copy). Use to ferry non-prefixed vars (CUDA_LAUNCH_BLOCKING).
    "vllm_ray_extra_env_vars_to_copy": "VLLM_RAY_EXTRA_ENV_VARS_TO_COPY",
}


def _build_vllm_cli_args(server_config: dict) -> tuple[list[str], dict[str, str]]:
    """Convert vllm_server config dict to CLI args and env vars.

    Returns:
        Tuple of (cli_args list, env_vars dict)
    """
    cli_args = []
    env_vars = {}

    for key, value in server_config.items():
        # Skip our internal fields
        if key in _OUR_FIELDS:
            continue

        # Skip None/empty values
        if value is None or value == "":
            continue

        # Handle extra_args (already a list of CLI args)
        if key == "extra_args":
            if isinstance(value, list):
                cli_args.extend(str(v) for v in value)
            continue

        # Handle env var fields. We write "1" or "0" explicitly when the
        # YAML key is present so callers can FORCE-disable env-vars that
        # default to True in vLLM (e.g. VLLM_USE_DEEP_GEMM). The earlier
        # "skip on falsy" behavior couldn't disable those defaults.
        if key in _ENV_VAR_FIELDS:
            env_vars[_ENV_VAR_FIELDS[key]] = "1" if value else "0"
            continue

        # Numeric env var fields: pass through as-is (str-coerced).
        if key in _NUMERIC_ENV_VAR_FIELDS:
            env_vars[_NUMERIC_ENV_VAR_FIELDS[key]] = str(value)
            continue

        # Rename field if needed
        arg_name = _FIELD_RENAMES.get(key, key)

        # Convert underscore to dash for CLI
        arg_name = arg_name.replace("_", "-")

        # Handle boolean flags
        if key in _BOOLEAN_FLAGS:
            if value:
                cli_args.append(f"--{arg_name}")
            elif key in _DEFAULT_OFF_BOOLEAN_FLAGS:
                # Explicit opt-out: emit `--no-<flag>` so vLLM's default
                # True is actually overridden. Without this branch, the
                # old behavior was to emit nothing on False, which let
                # vLLM's internal default win.
                cli_args.append(f"--no-{arg_name}")
            continue

        # Handle regular key-value args
        if isinstance(value, bool):
            # Non-flag booleans: pass as true/false string
            cli_args.extend([f"--{arg_name}", str(value).lower()])
        elif isinstance(value, dict):
            # Dict values need to be JSON-encoded (e.g., default_chat_template_kwargs)
            cli_args.extend([f"--{arg_name}", json.dumps(value)])
        else:
            cli_args.extend([f"--{arg_name}", str(value)])

    # For default-OFF flags whose vLLM internal default is True, inject the
    # `--no-<flag>` form when neither the affirmative nor negative form has
    # appeared yet (e.g. YAML omitted the top-level key AND extra_args did
    # not include either `--enable-X` or `--no-enable-X`). This preserves
    # opt-in via either the top-level YAML key (`enable_X: true`) or an
    # explicit `--enable-X` in `extra_args`.
    for key in _DEFAULT_OFF_BOOLEAN_FLAGS:
        arg_name = key.replace("_", "-")
        pos_flag = f"--{arg_name}"
        neg_flag = f"--no-{arg_name}"
        if pos_flag not in cli_args and neg_flag not in cli_args:
            cli_args.append(neg_flag)

    return cli_args, env_vars


@dataclass
class VLLMConfig:
    """Configuration for a vLLM server."""

    model_path: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    api_port: int = 8000
    endpoint_json_path: Optional[str] = None
    custom_model_name: Optional[str] = None

    # Health check settings
    # Default 480 × 15s = 7200s (2 hours). Long enough to cover slow
    # post-load init paths (Qwen3.5-MoE Mamba page-size calibration on
    # 4× GH200 single-node observed at 18+ min; cross-node TP weight
    # quantization on 2-node observed at ~12 min). Override via
    # `backend.healthcheck_max_attempts` in the datagen YAML if you
    # want a tighter SLA for fast-failing experiments.
    health_max_attempts: int = 480
    health_retry_delay: int = 15
    health_path: str = "v1/models"

    # Controller script path
    controller_script: str = "hpc/vllm/start_vllm_ray_controller.py"
    wait_for_endpoint_script: str = "hpc/vllm/wait_for_endpoint.py"

    # Raw vllm_server config from YAML - passed through to vLLM
    server_config: dict = field(default_factory=dict)


@dataclass
class VLLMServer:
    """Context manager for vLLM server lifecycle.

    This class handles:
    - Starting vLLM API server on Ray cluster
    - Writing endpoint JSON for clients
    - Health checking until server is ready
    - Graceful shutdown on exit
    """

    config: VLLMConfig
    ray_cluster: "RayCluster"
    log_path: Optional[Path] = None
    extra_env_vars: Optional[Dict[str, str]] = None  # Additional env vars (e.g., tiktoken)
    _process: Optional[subprocess.Popen] = None
    _log_file: Optional[object] = None

    @property
    def endpoint(self) -> str:
        """OpenAI-compatible API endpoint URL."""
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}/v1"

    @property
    def base_url(self) -> str:
        """Base URL without /v1 suffix."""
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}"

    @property
    def metrics_endpoint(self) -> str:
        """Prometheus metrics endpoint URL."""
        return f"http://{self.ray_cluster.head_ip}:{self.config.api_port}/metrics"

    def start(self) -> str:
        """Start the vLLM server.

        Returns:
            The API endpoint URL
        """
        if self._process is not None:
            print(f"vLLM server already started at {self.endpoint}")
            return self.endpoint

        # Clean up any stale endpoint JSON from a previous job to avoid IP mismatch
        if self.config.endpoint_json_path and os.path.exists(self.config.endpoint_json_path):
            print(f"Removing stale endpoint JSON: {self.config.endpoint_json_path}")
            try:
                os.remove(self.config.endpoint_json_path)
            except OSError as e:
                print(f"  Warning: could not remove stale endpoint file: {e}")

        print("=== Starting vLLM Server ===")
        print(f"  Model: {self.config.model_path}")
        print(f"  Tensor Parallel: {self.config.tensor_parallel_size}")
        print(f"  Pipeline Parallel: {self.config.pipeline_parallel_size}")
        print(f"  Data Parallel: {self.config.data_parallel_size}")
        print(f"  Host: {self.ray_cluster.head_ip}")
        print(f"  Port: {self.config.api_port}")
        print(f"  Ray Address: {self.ray_cluster.address}")
        print("============================")

        # NOTE: We intentionally do NOT call apply_numa_affinity(gpu_id=0) here.
        # The orchestrator process may already be pinned (by ray_utils.py), but the
        # vLLM API server + EngineCore should NOT inherit that restriction:
        # - The API server handles HTTP, tokenization, and scheduling across all GPUs
        # - EngineCore communicates with Ray workers via compiled DAGs (network), not
        #   shared memory, so GPU 0 locality provides no benefit
        # - Pinning to one NUMA node's CPUs (e.g., 72/288 on GH200) wastes 75% of
        #   available CPU capacity for tokenization and I/O handling
        # Instead, we reset affinity before spawning so the child gets all CPUs.

        # Open log file if path provided (line-buffered for real-time tail access)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self.log_path, "w", buffering=1)  # Line buffering
            stdout_dest = self._log_file
            stderr_dest = subprocess.STDOUT
        else:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL

        # Build command
        cmd = [
            sys.executable,
            self.config.controller_script,
            "--ray-address",
            self.ray_cluster.address,
            "--host",
            self.ray_cluster.head_ip,
            "--port",
            str(self.config.api_port),
            "--tensor-parallel-size",
            str(self.config.tensor_parallel_size),
            "--pipeline-parallel-size",
            str(self.config.pipeline_parallel_size),
            "--data-parallel-size",
            str(self.config.data_parallel_size),
        ]

        if self.config.endpoint_json_path:
            cmd.extend(["--endpoint-json", self.config.endpoint_json_path])

        if self.config.custom_model_name:
            cmd.extend(["--served-model-name", self.config.custom_model_name])

        # Build CLI args and env vars from server_config (pass-through from YAML)
        extra_env_vars: dict[str, str] = {}
        if self.config.server_config:
            extra_cli_args, extra_env_vars = _build_vllm_cli_args(self.config.server_config)
            cmd.extend(extra_cli_args)
            if extra_cli_args:
                print(f"  Extra vLLM args: {' '.join(extra_cli_args[:10])}{'...' if len(extra_cli_args) > 10 else ''}")

        # When DP>1 with the Ray backend, vLLM's create_dp_placement_groups
        # asserts the dp_master_ip is a key in Ray's known-nodes dict. Ray
        # registers the head node under its real IPv4 (set via
        # `--node-ip-address=<head_ip>` in ray_utils.py), NOT under
        # "localhost" / "127.0.0.1" — so a misset --data-parallel-address
        # fails with "AssertionError: The DP master node (ip: 127.0.0.1)
        # is missing or dead". Inject head_ip here unless the caller
        # explicitly overrode it via extra_args.
        if self.config.data_parallel_size > 1 and "--data-parallel-address" not in cmd:
            cmd.extend(["--data-parallel-address", self.ray_cluster.head_ip])

        # --data-parallel-size-local tells vLLM how many DP ranks live on
        # each node. Without it, vLLM defaults local_world_size to
        # data_parallel_size and tries to pack ALL DP ranks on the head
        # node's local GPUs, failing with:
        #   Exception: Error setting CUDA_VISIBLE_DEVICES:
        #     local range: [N*TP, (N+1)*TP)  base value: "0,1,2,3"
        # (e.g. local range [4,8) for DP rank 1 with TP=4, but the node only
        # exposes GPUs 0-3). Observed on Jupiter 491271 and Perlmutter
        # 53299838 on 2026-05-22 with the multi-node DP=4 TP=4 EP=true
        # GLM-4.7-AWQ config — the bug is launcher-side and fires on BOTH
        # the current Jupiter wheel and the older Perlmutter wheel, so it
        # is independent of the V1/V2 executor + shm_broadcast question.
        #
        # Layout: each DP rank consumes TP GPUs, so DP-ranks-per-node =
        # gpus_per_node // tensor_parallel_size, clamped to data_parallel_size.
        # This also handles the cross-node-TP case (TP > gpus_per_node): the
        # max(1, ...) clamp produces size_local=1, which still requires
        # data_parallel_size <= num_nodes for vLLM to accept the layout.
        if self.config.data_parallel_size > 1 and "--data-parallel-size-local" not in cmd:
            gpus_per_node = self.ray_cluster.config.gpus_per_node
            dp_per_node = max(1, gpus_per_node // self.config.tensor_parallel_size)
            dp_per_node = min(dp_per_node, self.config.data_parallel_size)
            cmd.extend(["--data-parallel-size-local", str(dp_per_node)])

        # --data-parallel-backend=ray is distinct from
        # --distributed-executor-backend=ray (the latter controls TP/PP
        # execution; the former controls DP rank PLACEMENT). With DP>1
        # cross-node the Ray DP backend is required; the default tries to
        # spawn worker procs locally and re-hits the CUDA_VISIBLE_DEVICES
        # error. Pair with VLLM_RAY_DP_PACK_STRATEGY=span (set in env
        # below) for the single-launch multi-node DP pattern documented at
        # https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/.
        if self.config.data_parallel_size > 1 and "--data-parallel-backend" not in cmd:
            cmd.extend(["--data-parallel-backend", "ray"])

        # Set environment
        env = os.environ.copy()
        env["VLLM_MODEL_PATH"] = self.config.model_path
        env["PYTHONUNBUFFERED"] = "1"  # Ensure real-time log output
        # Apply env vars derived from server_config YAML (e.g.
        # VLLM_USE_DEEP_GEMM, VLLM_PYNCCL_TRACE_FLUSH_INTERVAL_SEC) — these
        # were computed by _build_vllm_cli_args alongside the CLI args.
        # Previously the env_vars dict was unpacked but silently discarded.
        if extra_env_vars:
            env.update(extra_env_vars)
            print(f"  Extra vLLM env: {', '.join(f'{k}={v}' for k, v in extra_env_vars.items())}")
        # VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS controls the collective_rpc
        # deadline for execute_model + sample_tokens (multiproc_executor.py
        # lines 314, 326). Default is 300s; on cross-node TP=16 with our
        # GLM-5.1-AWQ config, the first inference-time Triton JIT compile
        # for _topk_topp_kernel / _build_prefill_chunk_metadata_kernel
        # exceeds that, tripping the watchdog → "RPC call to sample_tokens
        # timed out" → EngineDeadError → every subsequent request 500s.
        # Bumped 1800s → 7200s (2h) to also survive Qwen3.5-MoE Mamba
        # post-load page-size calibration which has been observed at
        # 18+ min of shm_broadcast wait on single-node TP=4.
        # See vllm_v2_bugs/OVERVIEW.md (Bug C) for the full trace.
        env.setdefault("VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS", "7200")
        # VLLM_RINGBUFFER_WARNING_INTERVAL controls how often shm_broadcast
        # logs the "No available shared memory broadcast block found in
        # N seconds" hint while a cross-rank wait is in flight. Default
        # 60s — produces hundreds of redundant log lines while the engine
        # is doing legitimately long work (model load, Mamba calibration,
        # cross-node weight quantization). Bump to 600s (10min) so the
        # logs stay legible without losing the warning entirely.
        env.setdefault("VLLM_RINGBUFFER_WARNING_INTERVAL", "600")
        # V2 model runner DISABLED — chosen default until vLLM patches
        # the multi-node DP path in RayExecutorV2.
        #
        # Why: RayExecutorV2 (introduced in vLLM 2026-05-13 upstream bump
        # via PR #36836) inherits from MultiprocExecutor and uses
        # shm_broadcast for inter-rank communication. Shared memory is
        # single-host by definition, so cross-node DP falls back to Gloo
        # TCP, which times out
        # (gloo/transport/tcp/unbound_buffer.cc Timed out 1800000ms) on
        # Jupiter's interconnect. Job 487456 (dp=2 MiniMax-M2.7) hit
        # this consistently even with EP disabled.
        #
        # Validation: job 490175 (dp=2 same yaml + this env=0) ran clean
        # with ZERO shm_broadcast warnings, sample_tokens timeouts, or
        # Gloo unbound_buffer timeouts. The v0.16.0 known-good branch
        # (penfever/debug-layer-split-v0.16.0, 2026-04-03) doesn't have
        # RayExecutorV2 at all — it uses ray_executor.py which routes
        # everything through Ray RPC (cross-node native).
        #
        # Throughput note: V1 (ray_executor.py) is somewhat slower per
        # request than V2 was designed to be. Single-node dp=1 paths
        # tested fine on V1 too. Revisit if a vLLM patch lands that
        # fixes RayExecutorV2's multi-node DP coordination.
        env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        # VLLM_RAY_DP_PACK_STRATEGY controls how the Ray DP backend places
        # ranks on the cluster. Three options, defined in
        # vllm/v1/engine/utils.py:create_dp_placement_groups:
        #
        #   strict — 1 DP rank per node, no oversubscription. Required for
        #            DeepEP kernels (EP ranks must co-reside on a node).
        #   fill   — greedy pack: fit as many DP ranks per node as possible.
        #   span   — a SINGLE DP rank spans multiple nodes (requires
        #            world_size = TP*PP > gpus_per_node, i.e. cross-node TP).
        #
        # If we pick the wrong one, vLLM asserts at engine init with
        #   AssertionError: World size N is smaller than the maximum number
        #     of devices per node M. Make sure to set
        #     `VLLM_RAY_DP_PACK_STRATEGY` to `strict` or `fill`
        # (observed on Jupiter 491789 + Perlmutter 53302207 when we tried
        # span with TP=4 on 4-GPU nodes — span is for the opposite case).
        #
        # Heuristic: if a single DP rank fits in one node (TP*PP <=
        # gpus_per_node), use strict. Otherwise use span. We never
        # auto-select fill — strict is the safer default for MoE (DeepEP
        # compatibility) and behaves identically to fill for the
        # single-DP-per-node case we hit in practice.
        if self.config.data_parallel_size > 1:
            tp_pp = self.config.tensor_parallel_size * self.config.pipeline_parallel_size
            if tp_pp > self.ray_cluster.config.gpus_per_node:
                env.setdefault("VLLM_RAY_DP_PACK_STRATEGY", "span")
            else:
                env.setdefault("VLLM_RAY_DP_PACK_STRATEGY", "strict")
        # Set VLLM_HOST_IP so vLLM's internal get_ip() returns the real node IP.
        # This is used for Ray placement group node constraints and NCCL communication,
        # NOT for the API server bind address (that's --host above).
        # Without this, vLLM auto-detects 0.0.0.0 on some HPC nodes, causing:
        #   "No available node types can fulfill resource request {'node:0.0.0.0': ...}"
        env["VLLM_HOST_IP"] = self.ray_cluster.head_ip
        if self.config.server_config:
            env.update(extra_env_vars)
        # Merge extra env vars from caller (e.g., TIKTOKEN_ENCODINGS_BASE for GPT-OSS)
        if self.extra_env_vars:
            env.update(self.extra_env_vars)

        # Reset CPU affinity before spawning the vLLM server so it can use all CPUs.
        # The parent may be pinned to one NUMA node's CPUs (e.g., CPUs 0-71 on GH200)
        # by apply_numa_affinity() in ray_utils.py. The child inherits this restriction,
        # but the vLLM server needs all CPUs for tokenization and scheduling.
        _saved_affinity = None
        try:
            _saved_affinity = os.sched_getaffinity(0)
            all_cpus = set(range(os.cpu_count() or 1))
            if _saved_affinity != all_cpus:
                os.sched_setaffinity(0, all_cpus)
                print(f"  Reset CPU affinity: {len(_saved_affinity)} → {len(all_cpus)} CPUs for vLLM server")
        except (OSError, AttributeError):
            pass

        # Start the server process
        self._process = subprocess.Popen(
            cmd,
            stdout=stdout_dest,
            stderr=stderr_dest,
            env=env,
        )

        # Restore the parent's original NUMA affinity (keep orchestrator pinned)
        if _saved_affinity is not None:
            try:
                os.sched_setaffinity(0, _saved_affinity)
            except (OSError, AttributeError):
                pass

        print(f"  Started vLLM controller (PID: {self._process.pid})")
        if self.log_path:
            print(f"  Log file: {self.log_path}")

        # Wait for server to be healthy
        self._wait_for_healthy()

        # Send a few pre-Harbor warmup requests to JIT-compile vLLM-native
        # Triton kernels not covered by FlashInfer autotune. Skipping this
        # would lead to first-request JIT during inference; on cross-node
        # TP=16, that JIT exceeds vLLM's 60s shm_broadcast watchdog and
        # kills the engine (see job 448672 logs). Failures here don't abort
        # the server — best-effort.
        try:
            self._warmup_serving()
        except Exception as e:
            print(f"  [warmup] non-fatal exception during warmup: {e!r}")

        print("=== vLLM Server Ready ===")
        print(f"  Endpoint: {self.endpoint}")
        print(f"  Metrics: {self.metrics_endpoint}")
        print("=========================")

        return self.endpoint

    def _warmup_serving(self) -> None:
        """Pre-JIT the vLLM-native Triton kernels that are most likely to
        deadlock the engine on first production traffic (Bug C: cross-rank
        ``shm_broadcast`` RPC timeout while one worker is stuck mid-JIT).

        Runs two phases:

          1. **Sequential, short, varied-prompt** — 8 requests, max_tokens=32,
             one at a time. Fires per-prompt-shape JIT for:
               - ``_topk_topp_kernel`` (vllm/v1/sample/ops/topk_topp_triton.py),
                 specialized per (top_k, top_p, vocab, dtype) tuple
               - ``_build_prefill_chunk_metadata_kernel`` (only relevant if
                 chunked prefill is ON; with the OT-Agent launcher's
                 default-OFF this is moot but still cheap to cover)

          2. **Concurrent, batched, longer-decode** — `concurrent_n` requests
             fired in parallel, max_tokens=512, one of them a long-ish prompt
             (~2k tokens). This drives the *batched* sampling kernel — Triton
             specializes on batch dimension, so a single-request warmup
             only JITs the batch=1 variant. Production traffic (Harbor at
             ``trace_n_concurrent>=16``) immediately hits the batch=N
             specialization, which then JITs mid-inference and stalls
             cross-rank coordination → ``EngineDeadError`` (the residual
             failure observed in iter 8 + iter 9).

        Total wall budget: ~1-3 min on cross-node TP=16, much less than the
        ~5+ min JIT-then-die cycle we hit at first real serving request.

        Failures bubble up to the caller wrapped in try/except — a partial
        warmup is acceptable; we still JIT what we can.
        """
        import urllib.request
        import urllib.error
        import json as _json
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # First, fetch the served model name (auto-generated when
        # custom_model_name is None).
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/v1/models", timeout=10
            ) as r:
                model_name = _json.loads(r.read().decode())["data"][0]["id"]
        except Exception as e:
            print(f"  [warmup] could not fetch /v1/models ({e!r}); skipping")
            return

        # ---- shared helper: drive one /v1/chat/completions request ----
        def _fire(prompt: str, max_tokens: int, label: str) -> tuple[str, float, bool, str]:
            body = _json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                # Sampling params chosen to match the most common production
                # config (Harbor terminus-2 + Qwen3/GLM-family). If your
                # serving uses different params, JIT for THOSE specializations
                # will still happen on the first real request — but the
                # most-common path is what matters here.
                "top_k": 20,
                "top_p": 0.95,
                "temperature": 0.7,
            }).encode()
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    _ = r.read()
                return (label, time.time() - t0, True, "")
            except Exception as e:
                return (label, time.time() - t0, False, repr(e))

        # ---- phase 1: sequential, short, varied-prompt ----
        # Varied prompt lengths so the prefill-chunk metadata kernel sees
        # several distinct chunk patterns. Short-ish prompts only — we
        # don't want warmup to eat real GPU time.
        seq_prompts = [
            "Hello world.",
            "Write a short Python function that adds two numbers.",
            "Once upon a time, in a faraway land, there lived a curious cat. "
            * 8,
            "Solve: x^2 + 3x - 4 = 0. Walk through it step by step.",
            "Describe a sunset in 2 sentences.",
            "What is the capital of France?",
            "Compute the integral of x dx from 0 to 1.",
            "List 10 random English words: " + " ".join(f"word{i}" for i in range(48)),
        ]
        print(f"  [warmup] phase 1: {len(seq_prompts)} sequential requests "
              f"against {model_name} (max_tokens=32, JITs single-request "
              "sampling kernel)...")
        seq_ok = 0
        for i, prompt in enumerate(seq_prompts, 1):
            label = f"seq {i}/{len(seq_prompts)}"
            _, dt, success, err = _fire(prompt, max_tokens=32, label=label)
            if success:
                seq_ok += 1
                print(f"  [warmup] {label} OK ({dt:.1f}s, {len(prompt.split())} words)")
            else:
                # Don't bail — partial warmup still helps. JIT errors here
                # are the load-bearing thing we WANT to pay during warmup.
                print(f"  [warmup] {label} FAILED: {err}")

        # ---- phase 2: concurrent, batched, longer-decode ----
        # Critical for Bug C iter 8/9 residual stall: Triton's
        # ``_topk_topp_kernel`` specializes on batch dimension. Phase 1
        # only JITs the batch=1 variant; first real Harbor traffic hits
        # batch=16 or 32 which then JITs mid-inference and stalls
        # cross-rank coordination. Drive a batch=concurrent_n forward
        # pass here so the batched variant is JIT'd before traffic.
        concurrent_n = 16
        long_prompt = (
            "You are an experienced software engineer. Consider the following "
            "context carefully and respond with a short summary at the end. "
            + ("Context line: a thoughtful description of a non-trivial "
               "engineering tradeoff. " * 80)
        )
        batch_prompts = list(seq_prompts) + [long_prompt]
        # Cycle batch_prompts up to concurrent_n entries
        batch_n_prompts = [
            batch_prompts[i % len(batch_prompts)] for i in range(concurrent_n)
        ]
        print(f"  [warmup] phase 2: {concurrent_n} concurrent requests "
              "(max_tokens=512, one ~2k-token prompt; JITs batched sampling "
              "kernel at production batch size)...")
        batch_ok = 0
        t_batch = time.time()
        with ThreadPoolExecutor(max_workers=concurrent_n) as ex:
            futures = [
                ex.submit(_fire, p, 512, f"par {i + 1}/{concurrent_n}")
                for i, p in enumerate(batch_n_prompts)
            ]
            for f in as_completed(futures):
                label, dt, success, err = f.result()
                if success:
                    batch_ok += 1
                    print(f"  [warmup] {label} OK ({dt:.1f}s)")
                else:
                    print(f"  [warmup] {label} FAILED: {err}")
        print(f"  [warmup] phase 2 wall: {time.time() - t_batch:.1f}s "
              f"({batch_ok}/{concurrent_n} succeeded)")
        print(f"  [warmup] complete (phase 1: {seq_ok}/{len(seq_prompts)}, "
              f"phase 2: {batch_ok}/{concurrent_n})")

    def stop(self) -> None:
        """Stop the vLLM server."""
        if self._process is None:
            return

        print("Stopping vLLM server...")

        # Graceful termination
        self._process.terminate()
        try:
            self._process.wait(timeout=30)
            print("  vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("  vLLM server not responding, killing...")
            self._process.kill()
            self._process.wait()

        self._process = None

        # Close log file
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def _wait_for_healthy(self) -> None:
        """Wait for the vLLM server to be healthy."""
        # First wait for endpoint JSON if configured
        if self.config.endpoint_json_path:
            self._wait_for_endpoint_json()

        # Then use the wait script if available
        script_path = Path(self.config.wait_for_endpoint_script)
        if script_path.exists():
            self._wait_with_script()
        else:
            self._wait_with_http()

    def _wait_for_endpoint_json(self, timeout: int = 600) -> None:
        """Wait for endpoint JSON file to be created."""
        if not self.config.endpoint_json_path:
            return

        print(f"  Waiting for endpoint JSON: {self.config.endpoint_json_path}")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM controller exited early (code {self._process.returncode}). "
                    f"Check logs at {self.log_path}"
                )

            if os.path.exists(self.config.endpoint_json_path):
                print(f"  Endpoint JSON found after {time.time() - start_time:.1f}s")
                return

            time.sleep(5)

        raise TimeoutError(
            f"Endpoint JSON not created at {self.config.endpoint_json_path} "
            f"after {timeout}s"
        )

    def _wait_with_script(self) -> None:
        """Wait using the wait_for_endpoint.py script."""
        cmd = [
            sys.executable,
            self.config.wait_for_endpoint_script,
            "--max-attempts",
            str(self.config.health_max_attempts),
            "--retry-delay",
            str(self.config.health_retry_delay),
            "--health-path",
            self.config.health_path,
        ]

        if self.config.endpoint_json_path:
            cmd.extend(["--endpoint-json", self.config.endpoint_json_path])
        else:
            # Direct URL mode
            cmd.extend(["--endpoint", self.base_url])

        print(f"  Running health check (max {self.config.health_max_attempts} attempts)...")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"vLLM health check failed after {self.config.health_max_attempts} attempts"
            ) from e

    def _wait_with_http(self) -> None:
        """Fallback: wait using direct HTTP health checks."""
        import urllib.request
        import urllib.error

        health_url = f"{self.base_url}/{self.config.health_path}"
        print(f"  Waiting for health endpoint: {health_url}")

        for attempt in range(1, self.config.health_max_attempts + 1):
            # Check if process died
            if self._process and self._process.poll() is not None:
                raise RuntimeError(
                    f"vLLM controller exited early (code {self._process.returncode})"
                )

            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=10) as response:
                    if response.status == 200:
                        print(f"  Health check passed on attempt {attempt}")
                        return
            except (urllib.error.URLError, urllib.error.HTTPError, OSError):
                pass

            if attempt < self.config.health_max_attempts:
                time.sleep(self.config.health_retry_delay)

        raise RuntimeError(
            f"vLLM health check failed after {self.config.health_max_attempts} attempts"
        )

    def get_endpoint_info(self) -> dict:
        """Get endpoint information as a dictionary.

        Useful for passing to client code or saving to a file.
        """
        return {
            "endpoint": self.endpoint,
            "base_url": self.base_url,
            "metrics_endpoint": self.metrics_endpoint,
            "model": self.config.model_path,
            "host": self.ray_cluster.head_ip,
            "port": self.config.api_port,
        }

    def write_endpoint_json(self, path: Optional[str] = None) -> str:
        """Write endpoint information to a JSON file.

        Args:
            path: Path to write to. If None, uses config.endpoint_json_path

        Returns:
            The path that was written to
        """
        output_path = path or self.config.endpoint_json_path
        if not output_path:
            raise ValueError("No endpoint JSON path specified")

        info = self.get_endpoint_info()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(info, f, indent=2)

        return output_path

    def __enter__(self) -> VLLMServer:
        """Context manager entry - start the server."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the server."""
        self.stop()


def create_vllm_server(
    model_path: str,
    ray_cluster: "RayCluster",
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    api_port: int = 8000,
    log_dir: Optional[str] = None,
    job_name: Optional[str] = None,
) -> VLLMServer:
    """Convenience function to create a VLLMServer.

    Args:
        model_path: HuggingFace model path
        ray_cluster: RayCluster instance to run on
        tensor_parallel_size: Number of GPUs for tensor parallelism
        pipeline_parallel_size: Number of stages for pipeline parallelism
        data_parallel_size: Number of data parallel replicas
        api_port: Port for the API server
        log_dir: Directory for log files
        job_name: Job name for log file naming

    Returns:
        A configured VLLMServer instance
    """
    config = VLLMConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        api_port=api_port,
    )

    log_path = None
    if log_dir and job_name:
        log_path = Path(log_dir) / f"{job_name}_vllm.log"
    elif log_dir:
        log_path = Path(log_dir) / "vllm_controller.log"

    return VLLMServer(config=config, ray_cluster=ray_cluster, log_path=log_path)


def run_endpoint_health_check(
    endpoint_json: Path,
    max_attempts: int,
    retry_delay: int,
    repo_root: Optional[Path] = None,
) -> None:
    """Run the vLLM endpoint health check script.

    This is a standalone function for running health checks outside of
    the VLLMServer context manager (e.g., for local runners that manage
    their own vLLM processes).

    Args:
        endpoint_json: Path to the endpoint JSON file
        max_attempts: Maximum number of health check attempts
        retry_delay: Delay in seconds between attempts
        repo_root: Repository root path (defaults to parent of hpc/)

    Raises:
        subprocess.CalledProcessError: If health check fails
    """
    if repo_root is None:
        # Default to repo root (parent of hpc/)
        repo_root = Path(__file__).resolve().parent.parent

    cmd = [
        sys.executable,
        str(repo_root / "hpc" / "vllm" / "wait_for_endpoint.py"),
        "--endpoint-json",
        str(endpoint_json),
        "--max-attempts",
        str(max_attempts),
        "--retry-delay",
        str(retry_delay),
        "--health-path",
        "v1/models",
    ]
    subprocess.run(cmd, check=True)
