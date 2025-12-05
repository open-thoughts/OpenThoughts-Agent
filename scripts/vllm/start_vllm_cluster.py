#!/usr/bin/env python3
"""
Multi-replica vLLM server launcher with HAProxy load balancer.
Creates multiple vLLM API server processes and an HAProxy reverse proxy.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import signal
import contextlib
import threading
import urllib.error
import urllib.parse
import urllib.request
import platform
from pathlib import Path
from typing import List, Optional
from datetime import datetime

# Ensure TF32 tensor cores are used for float32 matmul when available.
os.environ.setdefault("TORCH_FLOAT32_MATMUL_PRECISION", "high")

try:
    import torch
except ImportError:  # pragma: no cover - fine if torch unavailable
    torch = None


def _load_extra_args_from_env() -> List[str]:
    payload = os.environ.get("VLLM_SERVER_EXTRA_ARGS_JSON")
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[start_vllm_cluster] Warning: unable to parse VLLM_SERVER_EXTRA_ARGS_JSON: {exc}")
        return []
    if not isinstance(data, list):
        print(
            "[start_vllm_cluster] Warning: VLLM_SERVER_EXTRA_ARGS_JSON must be a JSON array; ignoring value."
        )
        return []
    return [str(item) for item in data if item is not None]


def _coerce_optional_float(value: Optional[float], name: str) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"[start_vllm_cluster] Warning: invalid {name} value {value!r}; ignoring.")
        return None


class VLLMCluster:
    _cli_flags_cache: Optional[set] = None
    _cli_flags_error_logged: bool = False

    def __init__(self, model: str, num_replicas: int, base_port: int = 8000,
                 tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1,
                 verbose: bool = False,
                 extended_context: bool = False, custom_model_name: str = None,
                 temperature: float = 0.1, seed: int = 42, enable_pinggy: bool = False,
                 pinggy_output_path: Optional[str] = None,
                 pinggy_health_interval: int = 120,
                 pinggy_health_timeout: int = 10,
                 pinggy_max_restarts: int = 3,
                 pinggy_health_url: Optional[str] = None,
                 cpu_offload_gb: Optional[float] = None,
                 kv_offloading_size: Optional[float] = None,
                 kv_offloading_backend: Optional[str] = None,
                 hf_overrides: Optional[str] = None,
                 max_num_seqs: Optional[int] = None,
                 gpu_memory_utilization: Optional[float] = None,
                 enable_expert_parallel: bool = False,
                 swap_space: Optional[int] = None,
                 max_seq_len_to_capture: Optional[int] = None,
                 max_output_tokens: Optional[int] = None,
                 max_model_len: Optional[int] = None,
                 trust_remote_code: bool = False,
                 disable_log_requests: bool = False):
        self.model = model
        self.num_replicas = num_replicas
        self.base_port = base_port
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.verbose = verbose
        self.extended_context = extended_context
        self.custom_model_name = custom_model_name
        self.temperature = temperature
        self.seed = seed
        self.enable_pinggy = enable_pinggy
        self.pinggy_output_path = pinggy_output_path or str(Path.cwd() / "vllm_endpoint.json")
        self.processes: List[subprocess.Popen] = []
        self.haproxy_process = None
        self.haproxy_config_path = None
        self.haproxy_binary = None
        self.pinggy_process = None
        self.pinggy_url: Optional[str] = None
        self.pinggy_ssh_command = os.environ.get("PINGGY_SSH_COMMAND")
        self.persistent_pinggy_url = os.environ.get("PINGGY_PERSISTENT_URL")
        self.pinggy_debugger_url = os.environ.get("PINGGY_DEBUGGER_URL")
        self.pinggy_health_interval = max(15, int(pinggy_health_interval))
        self.pinggy_health_timeout = max(3, int(pinggy_health_timeout))
        self.pinggy_max_restarts = max(1, int(pinggy_max_restarts))
        self.pinggy_health_url = pinggy_health_url or self._default_pinggy_health_url()
        self._pinggy_monitor_thread: Optional[threading.Thread] = None
        self._pinggy_monitor_stop = threading.Event()
        self._pinggy_failure_event = threading.Event()
        self._pinggy_lock = threading.Lock()
        self.pinggy_status = "initializing"
        self.pinggy_failure_reason: Optional[str] = None
        self.pinggy_last_healthcheck: Optional[float] = None
        self.pid_file = Path(__file__).parent / f"vllm_cluster_{base_port}.pids"
        self.haproxy_port = 8080
        self.healthcheck_port = 8081
        self.hf_overrides = hf_overrides
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else 32
        self.gpu_memory_utilization = gpu_memory_utilization
        self.enable_expert_parallel = enable_expert_parallel
        self.swap_space = int(round(swap_space)) if isinstance(swap_space, float) else swap_space
        self.cpu_offload_gb = _coerce_optional_float(cpu_offload_gb, "cpu_offload_gb")
        self.kv_offloading_size = _coerce_optional_float(kv_offloading_size, "kv_offloading_size")
        self.kv_offloading_backend = kv_offloading_backend
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_output_tokens = max_output_tokens
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code
        self.disable_log_requests = disable_log_requests

        # Set up logging
        level = logging.INFO if verbose else logging.WARNING
        logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self._release_torch_memory()

    def _effective_max_model_len(self) -> int:
        if self.max_model_len is not None:
            try:
                return int(self.max_model_len)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid max_model_len value %r; falling back to default.",
                    self.max_model_len,
                )
        return 131072 if self.extended_context else 32768

    def _default_pinggy_health_url(self) -> str:
        if self.pinggy_debugger_url:
            try:
                parsed = urllib.parse.urlparse(self.pinggy_debugger_url)
                if parsed.scheme and parsed.netloc:
                    sanitized = parsed._replace(
                        path=parsed.path or "/",
                        params="",
                        query="",
                        fragment="",
                    )
                    return urllib.parse.urlunparse(sanitized)
            except Exception:  # pragma: no cover - best effort only
                pass
        return "http://127.0.0.1:4300/"

    @classmethod
    def _discover_cli_flags(cls) -> set:
        if cls._cli_flags_cache is not None:
            return cls._cli_flags_cache

        flags: set[str] = set()
        try:
            result = subprocess.run(
                [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                capture_output=True,
                text=True,
                check=True,
            )
            for token in result.stdout.split():
                cleaned = token.strip("[],|")
                if cleaned.startswith("--"):
                    flags.add(cleaned)
        except Exception as exc:  # pragma: no cover - best effort guard
            if not cls._cli_flags_error_logged:
                print(f"Warning: unable to inspect vLLM CLI flags: {exc}")
                cls._cli_flags_error_logged = True
            flags = set()
        cls._cli_flags_cache = flags
        return flags

    @classmethod
    def _flag_supported(cls, flag: str) -> bool:
        return flag in cls._discover_cli_flags()

    @staticmethod
    def _release_torch_memory() -> None:
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover - best effort only
            pass
    
    def _get_cuda_visible_devices(self, replica_idx: int) -> str:
        """Calculate CUDA_VISIBLE_DEVICES for each replica based on tensor and pipeline parallel size."""
        gpus_per_replica = self.tensor_parallel_size * self.pipeline_parallel_size
        start_gpu = replica_idx * gpus_per_replica
        end_gpu = start_gpu + gpus_per_replica
        return ",".join(str(i) for i in range(start_gpu, end_gpu))
    
    def _ensure_haproxy(self) -> str:
        """Download HAProxy static binary if not present and return path to binary."""
        script_dir = Path(__file__).parent
        haproxy_path = script_dir / "haproxy"
        
        # Check if we already have the binary
        if haproxy_path.exists():
            self.logger.info(f"Using existing HAProxy binary: {haproxy_path}")
            return str(haproxy_path)
        
        # Check if haproxy is available in system PATH
        try:
            result = subprocess.run(["which", "haproxy"], capture_output=True, text=True)
            if result.returncode == 0:
                haproxy_path = result.stdout.strip()
                self.logger.info(f"Using system HAProxy: {haproxy_path}")
                return haproxy_path
        except Exception:
            pass
        
        # Try conda install first
        try:
            self.logger.info("Attempting to install HAProxy via conda...")
            result = subprocess.run(["conda", "install", "-c", "conda-forge", "haproxy", "-y"], 
                                  capture_output=True, text=True, timeout=120)
            if result.returncode == 0:
                result = subprocess.run(["which", "haproxy"], capture_output=True, text=True)
                if result.returncode == 0:
                    haproxy_path = result.stdout.strip()
                    self.logger.info(f"Successfully installed HAProxy via conda: {haproxy_path}")
                    return haproxy_path
        except Exception as e:
            self.logger.debug(f"Conda installation failed: {e}")
        
        # Download static binary from prosimcorp/haproxy-bins
        arch = platform.machine().lower()
        if arch in ['x86_64', 'amd64']:
            arch_suffix = 'x86_64'
        elif arch in ['aarch64', 'arm64']:
            arch_suffix = 'aarch64'
        else:
            raise RuntimeError(f"Unsupported architecture: {arch}")
        
        url = f"https://github.com/prosimcorp/haproxy-bins/releases/download/v0.3.11/haproxy-linux_{arch_suffix}"
        
        self.logger.info(f"Downloading HAProxy static binary for {arch_suffix}...")
        self.logger.info(f"URL: {url}")
        
        try:
            urllib.request.urlretrieve(url, haproxy_path)
            haproxy_path.chmod(0o755)
            self.logger.info(f"HAProxy downloaded successfully: {haproxy_path}")
            return str(haproxy_path)
        except Exception as e:
            self.logger.error(f"Failed to download HAProxy: {e}")
            raise RuntimeError(f"Could not download HAProxy: {e}")
    
    def _create_haproxy_config(self) -> str:
        """Generate HAProxy configuration for load balancing."""
        backend_servers = "\n".join([
            f"    server vllm{i} 127.0.0.1:{self.base_port + i} check"
            for i in range(self.num_replicas)
        ])
        
        config = f"""
global
    maxconn 2048

defaults
    mode http
    timeout connect 5000ms
    timeout client 300000ms
    timeout server 300000ms
        
frontend vllm_frontend
    bind *:{self.haproxy_port}
    default_backend vllm_backend

backend vllm_backend
    balance roundrobin
    option httpchk GET /v1/models
{backend_servers}

# Health check endpoint
frontend health_check
    bind *:{self.healthcheck_port}
    http-request return status 200 content-type text/plain string "healthy\\n"
"""
        return config
    
    def _start_vllm_server(self, replica_idx: int) -> subprocess.Popen:
        """Start a single vLLM server instance."""
        port = self.base_port + replica_idx
        cuda_devices = self._get_cuda_visible_devices(replica_idx)
        effective_max_model_len = str(self._effective_max_model_len())
        # "--compilation-config", json.dumps({"enabled": "false"})
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--host", "0.0.0.0",
            "--max-model-len", effective_max_model_len,
        ]

        # Add pipeline parallelism if specified
        if self.pipeline_parallel_size > 1:
            if self._flag_supported("--pipeline-parallel-size"):
                cmd.extend(["--pipeline-parallel-size", str(self.pipeline_parallel_size)])
            else:
                self.logger.warning("Skipping --pipeline-parallel-size; flag not supported by current vLLM version.")
        
        # Add custom model name if specified
        if self.custom_model_name:
            cmd.extend(["--served-model-name", self.custom_model_name])
        if self.hf_overrides:
            cmd.extend(["--hf-overrides", self.hf_overrides])

        if self.max_num_seqs:
            cmd.extend(["--max-num-seqs", str(self.max_num_seqs)])

        if self.gpu_memory_utilization:
            cmd.extend(["--gpu-memory-utilization", str(self.gpu_memory_utilization)])

        if self.enable_expert_parallel and self._flag_supported("--enable-expert-parallel"):
            cmd.append("--enable-expert-parallel")
        elif self.enable_expert_parallel:
            self.logger.warning("Skipping --enable-expert-parallel; flag not supported by current vLLM version.")

        if self.swap_space is not None and self._flag_supported("--swap-space"):
            swap_value = str(int(round(self.swap_space))) if isinstance(self.swap_space, float) else str(self.swap_space)
            cmd.extend(["--swap-space", swap_value])
        elif self.swap_space is not None:
            self.logger.warning("Skipping --swap-space; flag not supported by current vLLM version.")

        if self.cpu_offload_gb is not None:
            if self._flag_supported("--cpu-offload-gb"):
                cmd.extend(["--cpu-offload-gb", str(self.cpu_offload_gb)])
            else:
                self.logger.warning("Skipping --cpu-offload-gb; flag not supported by current vLLM version.")

        if self.kv_offloading_size is not None:
            if self._flag_supported("--kv-offloading-size"):
                cmd.extend(["--kv-offloading-size", str(self.kv_offloading_size)])
            else:
                self.logger.warning("Skipping --kv-offloading-size; flag not supported by current vLLM version.")

        if self.kv_offloading_backend:
            if self._flag_supported("--kv-offloading-backend"):
                cmd.extend(["--kv-offloading-backend", self.kv_offloading_backend])
            else:
                self.logger.warning("Skipping --kv-offloading-backend; flag not supported by current vLLM version.")

        if self.max_seq_len_to_capture is not None and self._flag_supported("--max-seq-len-to-capture"):
            cmd.extend(["--max-seq-len-to-capture", str(self.max_seq_len_to_capture)])
        elif self.max_seq_len_to_capture is not None:
            self.logger.warning("Skipping --max-seq-len-to-capture; flag not supported by current vLLM version.")

        enable_auto_tool_choice = os.environ.get("VLLM_ENABLE_AUTO_TOOL_CHOICE") == "1"
        tool_call_parser = os.environ.get("VLLM_TOOL_CALL_PARSER")
        reasoning_parser = os.environ.get("VLLM_REASONING_PARSER")

        if self.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.disable_log_requests:
            if self._flag_supported("--no-enable-log-requests"):
                cmd.append("--no-enable-log-requests")
            elif self._flag_supported("--disable-log-requests"):
                cmd.append("--disable-log-requests")
            else:
                self.logger.warning("Skipping disable log requests; relevant CLI flag not supported.")

        if enable_auto_tool_choice and self._flag_supported("--enable-auto-tool-choice"):
            cmd.append("--enable-auto-tool-choice")
        if tool_call_parser and self._flag_supported("--tool-call-parser"):
            cmd.extend(["--tool-call-parser", tool_call_parser])
        if reasoning_parser and self._flag_supported("--reasoning-parser"):
            cmd.extend(["--reasoning-parser", reasoning_parser])
            if self._flag_supported("--enable-reasoning"):
                cmd.append("--enable-reasoning")
            else:
                self.logger.warning(
                    "Skipping --enable-reasoning; flag not supported by current vLLM version."
                )
        # Add generation config with temperature and seed
        generation_config = {
            "temperature": self.temperature,
            "seed": self.seed
        }
        if self.max_output_tokens is not None:
            try:
                generation_config["max_tokens"] = int(self.max_output_tokens)
            except (TypeError, ValueError):
                self.logger.warning(
                    "Invalid max_output_tokens value %r; ignoring.",
                    self.max_output_tokens,
                )
                generation_config.pop("max_tokens", None)
        cmd.extend(["--override-generation-config", json.dumps(generation_config)])
        
        # Add extended context options if requested
        if self.extended_context:
            cmd.extend([
                "--rope-scaling", '{"type": "yarn", "factor": 4.0}',
                "--rope-theta", "10000"
            ])
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        
        if self.verbose:
            env["VLLM_LOG_LEVEL"] = "INFO"

        extra_cli_args = _load_extra_args_from_env()
        if extra_cli_args:
            cmd.extend(extra_cli_args)
        
        print(f"Starting vLLM server {replica_idx} on port {port} with GPUs {cuda_devices}")
        if self.verbose:
            self.logger.info(f"Starting vLLM server {replica_idx} on port {port} with GPUs {cuda_devices}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE if not self.verbose else None,
            stderr=subprocess.PIPE if not self.verbose else None
        )
        
        return process
    
    def _save_pids(self):
        """Save process IDs to file for cleanup."""
        pids = []
        for process in self.processes:
            if process and process.poll() is None:
                pids.append(str(process.pid))

        if self.haproxy_process and self.haproxy_process.poll() is None:
            pids.append(f"haproxy:{self.haproxy_process.pid}")

        if self.pinggy_process and self.pinggy_process.poll() is None:
            pids.append(f"pinggy:{self.pinggy_process.pid}")

        try:
            with open(self.pid_file, 'w') as f:
                f.write('\n'.join(pids))
            self.logger.info(f"Saved PIDs to {self.pid_file}")
        except Exception as e:
            self.logger.warning(f"Could not save PIDs: {e}")
    
    def _cleanup_from_pid_file(self):
        """Kill processes from saved PID file."""
        if not self.pid_file.exists():
            return
        
        try:
            with open(self.pid_file, 'r') as f:
                pids = f.read().strip().split('\n')
            
            for pid_line in pids:
                if not pid_line:
                    continue

                if pid_line.startswith('haproxy:') or pid_line.startswith('pinggy:'):
                    pid = pid_line.split(':')[1]
                else:
                    pid = pid_line

                try:
                    subprocess.run(['kill', '-TERM', pid], check=False)
                    self.logger.info(f"Killed process {pid}")
                except Exception:
                    pass
            
            # Remove PID file
            self.pid_file.unlink()
            self.logger.info("Cleaned up from PID file")
            
        except Exception as e:
            self.logger.warning(f"Error cleaning up from PID file: {e}")
    
    def _start_haproxy(self) -> subprocess.Popen:
        """Start HAProxy with the generated configuration."""
        # Ensure HAProxy binary is available
        self.haproxy_binary = self._ensure_haproxy()
        
        # Create temporary HAProxy config file
        config_content = self._create_haproxy_config()
        fd, self.haproxy_config_path = tempfile.mkstemp(suffix='.cfg', prefix='haproxy_vllm_')
        
        with os.fdopen(fd, 'w') as f:
            f.write(config_content)
        
        cmd = [self.haproxy_binary, "-f", self.haproxy_config_path]
        
        self.logger.info(f"Starting HAProxy load balancer on port {self.haproxy_port}")
        self.logger.info(f"HAProxy config: {self.haproxy_config_path}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if not self.verbose else None,
            stderr=subprocess.PIPE if not self.verbose else None
        )
        
        return process
    
    def _wait_for_server(self, server_idx: int, timeout_minutes: int = 10):
        """Wait for a specific vLLM server to be ready."""
        import requests
        
        port = self.base_port + server_idx
        url = f"http://localhost:{port}/v1/models"
        max_attempts = timeout_minutes * 12  # 5 second intervals
        
        print(f"Waiting for server {server_idx} (port {port}) to be ready...")
        if self.verbose:
            self.logger.info(f"Waiting for server {server_idx} (port {port}) to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f"Server {server_idx} (port {port}) is ready")
                    if self.verbose:
                        self.logger.info(f"Server {server_idx} (port {port}) is ready")
                    return
            except requests.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(5)
            else:
                self.logger.error(f"Server {server_idx} (port {port}) failed to start within {timeout_minutes} minutes")
                raise RuntimeError(f"Server {server_idx} failed to start")
    
    def _wait_for_servers(self, start_idx: int = 0):
        """Wait for vLLM servers to be ready, starting from start_idx."""
        import requests

        if start_idx == 0:
            self.logger.info("Waiting for all vLLM servers to be ready...")
        else:
            self.logger.info(f"Waiting for remaining vLLM servers ({start_idx} onwards) to be ready...")

        for i in range(start_idx, self.num_replicas):
            port = self.base_port + i
            url = f"http://localhost:{port}/v1/models"

            for attempt in range(60):  # Wait up to 5 minutes
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        self.logger.info(f"Server {i} (port {port}) is ready")
                        break
                except requests.RequestException:
                    pass

                if attempt < 59:
                    time.sleep(5)
                else:
                    self.logger.error(f"Server {i} (port {port}) failed to start within timeout")
                    raise RuntimeError(f"Server {i} failed to start")

    def _pinggy_process_alive(self) -> bool:
        with self._pinggy_lock:
            return bool(self.pinggy_process and self.pinggy_process.poll() is None)

    def _check_pinggy_health(self) -> bool:
        if not self.enable_pinggy:
            return True

        url = self.pinggy_health_url
        if not url:
            return True

        try:
            request = urllib.request.Request(url, method="HEAD")
        except TypeError:  # pragma: no cover - compatibility fallback
            request = urllib.request.Request(url)
            request.get_method = lambda: "HEAD"  # type: ignore[attr-defined]

        try:
            with contextlib.closing(
                urllib.request.urlopen(request, timeout=self.pinggy_health_timeout)
            ) as response:
                status = getattr(response, "status", None) or response.getcode()
                if 200 <= status < 400:
                    self.pinggy_last_healthcheck = time.time()
                    self.pinggy_status = "running"
                    return True
                self.logger.warning(
                    "Pinggy health check received status %s from %s", status, url
                )
        except urllib.error.HTTPError as exc:
            self.logger.warning(
                "Pinggy health check HTTP error for %s: %s", url, exc.code
            )
        except Exception as exc:  # pragma: no cover - best effort only
            self.logger.warning("Pinggy health check failed for %s: %s", url, exc)

        return False

    def _restart_pinggy_tunnel(self) -> bool:
        if not self.enable_pinggy or self._pinggy_monitor_stop.is_set():
            return False

        self.logger.info("Attempting Pinggy tunnel restart")
        with self._pinggy_lock:
            if self.pinggy_process and self.pinggy_process.poll() is None:
                self.logger.info("Terminating existing Pinggy process")
                self.pinggy_process.terminate()
                try:
                    self.pinggy_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.pinggy_process.kill()
            self.pinggy_process = None
            self.pinggy_status = "restarting"
            self.pinggy_failure_reason = None

            try:
                self.pinggy_process = self._start_pinggy_tunnel()
                # Give the tunnel a brief moment before health-checking.
                time.sleep(3)
                if self._check_pinggy_health():
                    self.pinggy_status = "running"
                    self._save_endpoint_json(force=bool(self.pinggy_url))
                    self.logger.info("Pinggy tunnel restart succeeded")
                    return True
            except Exception as exc:
                self.logger.error("Pinggy restart failed: %s", exc)
                self.pinggy_failure_reason = str(exc)

        # Restart failed; persist state for consumers.
        self.pinggy_status = "failed"
        self._save_endpoint_json(force=True)
        return False

    def _start_pinggy_monitor(self) -> None:
        if not self.enable_pinggy:
            return
        if self._pinggy_monitor_thread and self._pinggy_monitor_thread.is_alive():
            return

        self._pinggy_monitor_stop.clear()
        monitor_thread = threading.Thread(
            target=self._pinggy_monitor_loop,
            name="PinggyMonitor",
            daemon=True,
        )
        self._pinggy_monitor_thread = monitor_thread
        monitor_thread.start()

    def _pinggy_monitor_loop(self) -> None:
        consecutive_failures = 0
        # Give the tunnel a grace period after any restart before probing again.
        next_check_delay = self.pinggy_health_interval
        while not self._pinggy_monitor_stop.wait(next_check_delay):
            next_check_delay = self.pinggy_health_interval
            if not self.enable_pinggy:
                break

            failure_detected = False
            failure_reason = "unknown"

            if not self._pinggy_process_alive():
                failure_detected = True
                failure_reason = "process exited"
            elif not self._check_pinggy_health():
                failure_detected = True
                failure_reason = "health probe failed"

            if not failure_detected:
                consecutive_failures = 0
                continue

            consecutive_failures += 1
            self.logger.warning(
                "Pinggy tunnel unhealthy (%s). Attempt %d/%d",
                failure_reason,
                consecutive_failures,
                self.pinggy_max_restarts,
            )

            restart_ok = self._restart_pinggy_tunnel()
            if restart_ok:
                # Successful restart resets failure counter after confirming health.
                # Delay probing briefly to allow the tunnel to stabilize.
                next_check_delay = max(self.pinggy_health_interval, 30)
                if self._check_pinggy_health():
                    consecutive_failures = 0
                    continue

            if consecutive_failures >= self.pinggy_max_restarts:
                self._fatal_pinggy_failure(failure_reason)
                return

            # Allow a short backoff before next iteration to avoid tight loops.
            if self._pinggy_monitor_stop.wait(10):
                return

    def _fatal_pinggy_failure(self, reason: str) -> None:
        self.logger.error(
            "Pinggy tunnel failed %d times. Triggering shutdown (reason: %s).",
            self.pinggy_max_restarts,
            reason,
        )
        self.pinggy_status = "failed"
        self.pinggy_failure_reason = reason
        self._pinggy_failure_event.set()
        self._pinggy_monitor_stop.set()
        with self._pinggy_lock:
            if self.pinggy_process and self.pinggy_process.poll() is None:
                self.pinggy_process.terminate()
                try:
                    self.pinggy_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.pinggy_process.kill()
        self._save_endpoint_json(force=True)

    def _start_pinggy_tunnel(self) -> subprocess.Popen:
        """Start Pinggy tunnel to expose HAProxy load balancer."""
        haproxy_port = self.haproxy_port
        self.pinggy_status = "starting"

        if self.pinggy_ssh_command:
            self.logger.info("Starting Pinggy tunnel via SSH command")
            cmd = self.pinggy_ssh_command.format(PORT=haproxy_port)
            process = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            if self.persistent_pinggy_url:
                url = self.persistent_pinggy_url.strip()
                if not url.startswith("http"):
                    url = f"https://{url}"
                self.pinggy_url = url
                self.pinggy_status = "running"
                self.logger.info(f"Using persistent Pinggy URL: {self.pinggy_url}")
            else:
                self.logger.warning(
                    "PINGGY_PERSISTENT_URL not set; tunnel command started but public URL is unknown"
                )

            return process

        try:
            import pinggy
        except ImportError:
            self.logger.error("Pinggy package not installed. Install with: pip install pinggy")
            raise RuntimeError("Pinggy package required but not installed")

        self.logger.info("Starting Pinggy HTTP tunnel via python package...")

        cmd = [
            sys.executable, "-m", "pinggy",
            "tcp", str(haproxy_port)
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        tunnel_url = None
        for _ in range(60):  # Wait up to 60 seconds for tunnel establishment
            line = process.stdout.readline()
            if not line:
                break

            self.logger.debug(f"Pinggy output: {line.strip()}")

            if "http" in line.lower() and ("pinggy" in line.lower() or ".a.pinggy." in line.lower()):
                import re
                url_match = re.search(r'(https?://[^\s]+)', line)
                if url_match:
                    tunnel_url = url_match.group(1).strip()
                    self.logger.info(f"Pinggy tunnel established: {tunnel_url}")
                    break

            time.sleep(1)

        if not tunnel_url:
            process.kill()
            raise RuntimeError("Failed to establish Pinggy tunnel - could not extract URL")

        self.pinggy_url = tunnel_url
        self.pinggy_status = "running"
        return process

    def _save_endpoint_json(self, *, force: bool = False) -> None:
        """Save endpoint information to JSON file."""
        if not self.pinggy_url and not force:
            self.logger.warning("No Pinggy URL available, skipping JSON save")
            return

        now = datetime.utcnow().isoformat()
        endpoint_data = {
            "model_name": self.custom_model_name or self.model,
            "endpoint_url": self.pinggy_url or "",
            "local_port": self.haproxy_port,
            "base_model_path": self.model,
            "num_replicas": self.num_replicas,
            "tensor_parallel_size": self.tensor_parallel_size,
            "cpu_offload_gb": self.cpu_offload_gb,
            "kv_offloading_size": self.kv_offloading_size,
            "kv_offloading_backend": self.kv_offloading_backend,
            "temperature": self.temperature,
            "seed": self.seed,
            "extended_context": self.extended_context,
            "max_model_len": str(self._effective_max_model_len()),
            "created_at": now,
            "debugger_url": self.pinggy_debugger_url,
            "cluster_config": {
                "base_port": self.base_port,
                "haproxy_port": self.haproxy_port,
                "health_check_port": self.healthcheck_port,
                "individual_server_ports": [self.base_port + i for i in range(self.num_replicas)]
            },
            "pinggy_status": self.pinggy_status,
            "pinggy_health_url": self.pinggy_health_url,
            "pinggy_failure_reason": self.pinggy_failure_reason,
            "pinggy_last_healthcheck": (
                datetime.utcfromtimestamp(self.pinggy_last_healthcheck).isoformat()
                if self.pinggy_last_healthcheck
                else None
            ),
            "status_updated_at": now,
        }
        if self.max_output_tokens is not None:
            endpoint_data["max_output_tokens"] = str(self.max_output_tokens)

        output_path = Path(self.pinggy_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(endpoint_data, f, indent=2)

        self.logger.info(f"Endpoint information saved to: {output_path}")
        print(f"✓ Endpoint configuration saved to: {output_path}")
        print(f"✓ Model: {endpoint_data['model_name']}")
        print(f"✓ Endpoint URL: {endpoint_data['endpoint_url']}")

    def start(self):
        """Start all vLLM servers and HAProxy load balancer."""
        # Ensure no stale processes from a previous run linger.
        self._cleanup_from_pid_file()

        # Calculate total GPUs needed
        gpus_per_replica = self.tensor_parallel_size * self.pipeline_parallel_size
        total_gpus_needed = self.num_replicas * gpus_per_replica
        self.logger.info(f"Starting {self.num_replicas} replicas with {gpus_per_replica} GPUs each (TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size})")
        self.logger.info(f"Total GPUs required: {total_gpus_needed}")
        
        # Start vLLM servers
        for i in range(self.num_replicas):
            process = self._start_vllm_server(i)
            self.processes.append(process)
            
            if i == 0:
                # Make first server startup blocking to avoid compilation race conditions
                print("Waiting for first server to be fully ready before starting others...")
                if self.verbose:
                    self.logger.info("Waiting for first server to be fully ready before starting others...")
                self._wait_for_server(0)
                if self.num_replicas > 1:
                    print("First server ready, starting remaining servers...")
                    if self.verbose:
                        self.logger.info("First server ready, starting remaining servers...")
            else:
                time.sleep(2)  # Stagger startup for subsequent servers
        
        # Wait for all remaining servers to be ready
        if self.num_replicas > 1:
            self._wait_for_servers(start_idx=1)
        
        # Start HAProxy load balancer
        self.haproxy_process = self._start_haproxy()
        time.sleep(2)

        # Start Pinggy tunnel if enabled
        if self.enable_pinggy:
            try:
                self.pinggy_process = self._start_pinggy_tunnel()
                self._save_endpoint_json(force=bool(self.pinggy_url))
                self._start_pinggy_monitor()
                if self.pinggy_url:
                    print(f"✓ Pinggy tunnel established: {self.pinggy_url}")
                else:
                    print("✓ Pinggy tunnel command launched")
                if self.pinggy_debugger_url:
                    print(f"✓ Pinggy debugger available at: {self.pinggy_debugger_url}")
                if self.pinggy_health_url:
                    print(f"✓ Pinggy health monitor watching: {self.pinggy_health_url}")
                print("VLLM_CLUSTER_READY")  # Success signal for batch scripts
            except Exception as e:
                self.logger.error(f"Failed to start Pinggy tunnel: {e}")
                print(f"Warning: Pinggy tunnel failed, but cluster is still running locally")

        # Save PIDs for cleanup
        self._save_pids()

        print("=== vLLM Cluster Started Successfully ===")
        print(f"Load balancer endpoint: http://localhost:{self.haproxy_port}")
        print(f"Health check endpoint: http://localhost:{self.healthcheck_port}")
        print(f"Individual servers: http://localhost:{self.base_port}-{self.base_port + self.num_replicas - 1}")
        if self.enable_pinggy and self.pinggy_url:
            print(f"Public endpoint (Pinggy): {self.pinggy_url}")
            print(f"Endpoint config: {self.pinggy_output_path}")
            if self.pinggy_debugger_url:
                print(f"Pinggy debugger: {self.pinggy_debugger_url}")
        print(f"PID file: {self.pid_file}")
        print("Press Ctrl+C to stop all servers")

        if self.verbose:
            self.logger.info("=== vLLM Cluster Started Successfully ===")
            self.logger.info(f"Load balancer endpoint: http://localhost:{self.haproxy_port}")
            self.logger.info(f"Health check endpoint: http://localhost:{self.healthcheck_port}")
            self.logger.info(f"Individual servers: http://localhost:{self.base_port}-{self.base_port + self.num_replicas - 1}")
            if self.enable_pinggy and self.pinggy_url:
                self.logger.info(f"Public endpoint (Pinggy): {self.pinggy_url}")
                if self.pinggy_debugger_url:
                    self.logger.info(f"Pinggy debugger: {self.pinggy_debugger_url}")
            self.logger.info(f"PID file: {self.pid_file}")
            self.logger.info("Press Ctrl+C to stop all servers")
    
    def stop(self):
        """Stop all processes and clean up."""
        self.logger.info("Stopping vLLM cluster...")
        self._pinggy_monitor_stop.set()
        if (
            self._pinggy_monitor_thread
            and self._pinggy_monitor_thread.is_alive()
            and threading.current_thread() is not self._pinggy_monitor_thread
        ):
            self._pinggy_monitor_thread.join(timeout=15)
        self._pinggy_monitor_thread = None

        # First try to clean up from PID file (handles crashed processes)
        self._cleanup_from_pid_file()

        # Stop Pinggy tunnel
        if self.pinggy_process:
            self.logger.info("Stopping Pinggy tunnel")
            self.pinggy_process.terminate()
            try:
                self.pinggy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.pinggy_process.kill()

        if self.enable_pinggy and self.pinggy_status != "failed":
            self.pinggy_status = "stopped"
            self.pinggy_failure_reason = None
            self._save_endpoint_json(force=bool(self.pinggy_url))

        # Stop HAProxy
        if self.haproxy_process:
            self.haproxy_process.terminate()
            try:
                self.haproxy_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.haproxy_process.kill()

        # Stop vLLM servers
        for i, process in enumerate(self.processes):
            if process:
                self.logger.info(f"Stopping vLLM server {i}")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

        # Clean up HAProxy config file
        if self.haproxy_config_path and os.path.exists(self.haproxy_config_path):
            os.unlink(self.haproxy_config_path)
        
        # Clean up PID file if it still exists
        if self.pid_file.exists():
            self.pid_file.unlink()
        
        self.logger.info("All processes stopped")
    
    def run(self):
        """Run the cluster and handle shutdown signals."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self.start()
        
        try:
            # Keep the main process alive
            while True:
                # Check if any process has died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        self.logger.error(f"vLLM server {i} has died")
                        self.stop()
                        sys.exit(1)
                
                if self.haproxy_process and self.haproxy_process.poll() is not None:
                    self.logger.error("HAProxy has died")
                    self.stop()
                    sys.exit(1)

                if self.enable_pinggy and self._pinggy_failure_event.is_set():
                    self.logger.error("Pinggy tunnel failed irrecoverably; shutting down.")
                    self.stop()
                    sys.exit(1)
                
                time.sleep(5)
        
        except KeyboardInterrupt:
            self.stop()


def cleanup_orphaned_processes(base_port=8000):
    """Clean up orphaned vLLM processes from previous runs."""
    script_dir = Path(__file__).parent
    pid_file = script_dir / f"vllm_cluster_{base_port}.pids"
    
    print(f"Looking for PID file: {pid_file}")
    
    if not pid_file.exists():
        print("No PID file found. Trying to kill vLLM processes manually...")
        try:
            # Kill all vLLM API server processes
            subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
            print("Killed vLLM processes")
        except Exception as e:
            print(f"Error killing processes: {e}")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pids = f.read().strip().split('\n')
        
        for pid_line in pids:
            if not pid_line:
                continue
                
            if pid_line.startswith('haproxy:'):
                pid = pid_line.split(':')[1]
                process_type = "HAProxy"
            else:
                pid = pid_line
                process_type = "vLLM"
            
            try:
                subprocess.run(['kill', '-TERM', pid], check=True)
                print(f"Killed {process_type} process {pid}")
            except subprocess.CalledProcessError:
                print(f"Process {pid} already dead or not found")
            except Exception as e:
                print(f"Error killing process {pid}: {e}")
        
        # Remove PID file
        pid_file.unlink()
        print("Cleanup completed")
        
    except Exception as e:
        print(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='Start multiple vLLM servers with HAProxy load balancer')
    parser.add_argument('--model', type=str, help='Model name/path')
    parser.add_argument('--num-replicas', type=int, help='Number of server replicas')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size per replica (default: 1)')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Pipeline parallel size per replica (default: 1)')
    parser.add_argument('--base-port', type=int, default=8000, help='Base port for vLLM servers (default: 8000)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--cleanup', action='store_true', help='Clean up orphaned processes from previous runs')
    parser.add_argument('--extended-context', action='store_true', help='Enable 131k token context length with RoPE+YARN scaling (default: 32k)')
    parser.add_argument('--custom-model-name', type=str, help='Custom model name to serve (if different from actual model path)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Default temperature for sampling (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--enable-pinggy', action='store_true', help='Enable Pinggy HTTP tunnel for public access')
    parser.add_argument('--pinggy-output-path', type=str, help='Path to save Pinggy endpoint JSON (default: ./vllm_endpoint.json)')
    parser.add_argument('--pinggy-health-interval', type=int, default=120, help='Seconds between Pinggy health checks (default: 120)')
    parser.add_argument('--pinggy-health-timeout', type=int, default=10, help='Timeout for Pinggy health probe requests (default: 10)')
    parser.add_argument('--pinggy-max-restarts', type=int, default=3, help='Consecutive Pinggy restart attempts before giving up (default: 3)')
    parser.add_argument('--pinggy-health-url', type=str, help='Override Pinggy health-check URL (defaults to debugger URL or http://127.0.0.1:4300/)')
    parser.add_argument('--cpu-offload-gb', type=float, help='CPU memory (GiB) reserved for KV cache offloading')
    parser.add_argument('--kv-offloading-size', type=float, help='Size (GiB) of KV cache to offload')
    parser.add_argument('--kv-offloading-backend', type=str, help='KV cache offloading backend (e.g., native, lmcache)')
    parser.add_argument('--hf-overrides', type=str, help='JSON string of Hugging Face config overrides passed to vLLM')
    parser.add_argument('--max-num-seqs', type=int, help='Override vLLM scheduler max_num_seqs (default: 512)')
    parser.add_argument('--gpu-memory-utilization', type=float, help='Override vLLM GPU memory utilization fraction')
    parser.add_argument('--enable-expert-parallel', action='store_true', help='Enable vLLM expert parallel mode')
    parser.add_argument('--swap-space', type=int, help='Swap space in GB allocated by vLLM')
    parser.add_argument('--max-seq-len-to-capture', type=int, help='Maximum sequence length captured by vLLM for logging or inspection')
    parser.add_argument('--max-output-tokens', type=int, help='Default max tokens per completion')
    parser.add_argument('--max-model-len', type=int, help='Override vLLM max model length')
    parser.add_argument('--trust-remote-code', action='store_true', help='Allow vLLM to trust remote code from model repository')
    parser.add_argument('--disable-log-requests', action='store_true', help='Disable request logging in vLLM API server')

    args = parser.parse_args()

    if args.cleanup:
        cleanup_orphaned_processes(args.base_port)
        return

    if not args.model or not args.num_replicas:
        parser.error("--model and --num-replicas are required unless using --cleanup")

    if args.max_model_len is None:
        env_max_model_len = os.environ.get("VLLM_MAX_MODEL_LEN")
        if env_max_model_len:
            try:
                args.max_model_len = int(env_max_model_len)
            except ValueError:
                print(
                    f"Warning: VLLM_MAX_MODEL_LEN='{env_max_model_len}' is not an int; ignoring."
                )

    if args.max_output_tokens is None:
        env_max_output_tokens = os.environ.get("VLLM_MAX_OUTPUT_TOKENS")
        if env_max_output_tokens:
            try:
                args.max_output_tokens = int(env_max_output_tokens)
            except ValueError:
                print(
                    f"Warning: VLLM_MAX_OUTPUT_TOKENS='{env_max_output_tokens}' is not an int; ignoring."
                )

    cluster = VLLMCluster(
        model=args.model,
        num_replicas=args.num_replicas,
        base_port=args.base_port,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        verbose=args.verbose,
        extended_context=args.extended_context,
        custom_model_name=args.custom_model_name,
        temperature=args.temperature,
        seed=args.seed,
        enable_pinggy=args.enable_pinggy,
        pinggy_output_path=args.pinggy_output_path,
        pinggy_health_interval=args.pinggy_health_interval,
        pinggy_health_timeout=args.pinggy_health_timeout,
        pinggy_max_restarts=args.pinggy_max_restarts,
        pinggy_health_url=args.pinggy_health_url,
        cpu_offload_gb=args.cpu_offload_gb,
        kv_offloading_size=args.kv_offloading_size,
        kv_offloading_backend=args.kv_offloading_backend,
        hf_overrides=args.hf_overrides,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_expert_parallel=args.enable_expert_parallel,
        swap_space=args.swap_space,
        max_seq_len_to_capture=args.max_seq_len_to_capture,
        max_output_tokens=args.max_output_tokens,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        disable_log_requests=args.disable_log_requests,
    )

    print("Configuration:")
    print(f"  Model: {cluster.model}")
    print(f"  Custom model name: {cluster.custom_model_name if cluster.custom_model_name else 'None (use model path)'}")
    print(f"  Number of replicas: {cluster.num_replicas}")
    print(f"  Tensor parallel size: {cluster.tensor_parallel_size}")
    print(f"  Pipeline parallel size: {cluster.pipeline_parallel_size}")
    print(f"  GPUs per replica: {cluster.tensor_parallel_size * cluster.pipeline_parallel_size}")
    print(f"  Base port: {cluster.base_port}")
    print(f"  Load balancer port: {cluster.haproxy_port}")
    print(f"  Temperature: {cluster.temperature}")
    print(f"  Seed: {cluster.seed}")
    print(f"  Verbose: {cluster.verbose}")
    print(f"  Extended context: {cluster.extended_context} ({'131k tokens' if cluster.extended_context else '32k tokens'})")
    print(f"  HF overrides: {cluster.hf_overrides or 'None'}")
    print(f"  Max num seqs: {cluster.max_num_seqs}")
    if cluster.gpu_memory_utilization:
        print(f"  GPU memory utilization: {cluster.gpu_memory_utilization}")
    print(f"  Expert parallel: {cluster.enable_expert_parallel}")
    if cluster.swap_space is not None:
        print(f"  Swap space: {cluster.swap_space} GB")
    if cluster.max_seq_len_to_capture is not None:
        print(f"  Max seq len to capture: {cluster.max_seq_len_to_capture}")
    if cluster.max_output_tokens is not None:
        print(f"  Max output tokens: {cluster.max_output_tokens}")
    print(f"  Max model len: {cluster._effective_max_model_len()}")
    cpu_offload_display = (
        f"{cluster.cpu_offload_gb} GiB" if cluster.cpu_offload_gb is not None else "<none>"
    )
    kv_offload_size_display = (
        f"{cluster.kv_offloading_size} GiB" if cluster.kv_offloading_size is not None else "<none>"
    )
    print(f"  CPU offload: {cpu_offload_display}")
    print(
        f"  KV offloading: size={kv_offload_size_display} "
        f"(backend: {cluster.kv_offloading_backend or '<default>'})"
    )
    print(f"  Trust remote code: {cluster.trust_remote_code}")
    print(f"  Log requests: {'disabled' if cluster.disable_log_requests else 'enabled'}")
    print(f"  Pinggy tunnel: {cluster.enable_pinggy}")
    if cluster.enable_pinggy:
        print(f"  Pinggy output path: {cluster.pinggy_output_path}")
        print(f"  Pinggy health URL: {cluster.pinggy_health_url}")
        print(
            f"  Pinggy monitor: interval={cluster.pinggy_health_interval}s "
            f"timeout={cluster.pinggy_health_timeout}s max_restarts={cluster.pinggy_max_restarts}"
        )
        if cluster.persistent_pinggy_url:
            print(f"  Persistent Pinggy URL: {cluster.persistent_pinggy_url}")
        if cluster.pinggy_debugger_url:
            print(f"  Pinggy debugger: {cluster.pinggy_debugger_url}")

    cluster.run()


if __name__ == "__main__":
    main()
