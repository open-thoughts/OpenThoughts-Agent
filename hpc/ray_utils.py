"""Ray cluster management for SLURM-based HPC systems.

This module provides a context manager for managing Ray cluster lifecycle
within SLURM jobs, eliminating duplicated Ray setup code across SBATCH scripts.

Usage:
    from hpc.ray_utils import RayCluster, RayClusterConfig

    config = RayClusterConfig(
        num_nodes=4,
        gpus_per_node=4,
        cpus_per_node=48,
    )

    with RayCluster.from_slurm(config) as ray_cluster:
        print(f"Ray cluster ready at {ray_cluster.address}")
        print(f"Total GPUs: {ray_cluster.total_gpus}")
        # ... launch vLLM or other Ray-based workloads
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from hpc.hpc import HPC


# Memory configuration constants
# Headroom scales with node size to handle CUDA graph capture and system overhead
DEFAULT_MEMORY_HEADROOM_MB = 32768  # 32GB default headroom
MIN_MEMORY_HEADROOM_MB = 16384  # 16GB minimum
MEMORY_HEADROOM_PERCENT = 0.05  # 5% of total memory as headroom for large nodes
DEFAULT_OBJECT_STORE_MEMORY_BYTES = 40 * 1024 * 1024 * 1024  # 40GB for Ray plasma store


def compute_ray_memory_from_slurm(headroom_mb: Optional[int] = None) -> Optional[int]:
    """Compute Ray memory limit from SLURM allocation.

    Reads SLURM_MEM_PER_NODE environment variable and subtracts headroom
    to leave space for system overhead, SLURM processes, and CUDA graph capture.

    The headroom scales with node size:
    - For nodes < 512GB: uses MIN_MEMORY_HEADROOM_MB (16GB)
    - For larger nodes: uses max(DEFAULT_MEMORY_HEADROOM_MB, 3% of total)

    This prevents OOM during vLLM's CUDA graph capture phase which has
    significant memory spikes.

    Args:
        headroom_mb: Override headroom in MB (if None, auto-scales with node size)

    Returns:
        Memory in bytes for Ray's --memory flag, or None if SLURM_MEM_PER_NODE not set
    """
    slurm_mem_str = os.environ.get("SLURM_MEM_PER_NODE")
    if not slurm_mem_str:
        return None

    # SLURM_MEM_PER_NODE is in MB (e.g., "1536000" for 1.5TB)
    try:
        slurm_mem_mb = int(slurm_mem_str)
    except ValueError:
        print(f"Warning: Could not parse SLURM_MEM_PER_NODE={slurm_mem_str}", file=sys.stderr)
        return None

    # Compute headroom: either explicit override, or scale with node size
    if headroom_mb is not None:
        actual_headroom_mb = headroom_mb
    else:
        # Scale headroom with node size
        # For small nodes (<512GB): use minimum headroom
        # For large nodes: use max(default, 3% of total)
        if slurm_mem_mb < 512 * 1024:  # < 512GB
            actual_headroom_mb = MIN_MEMORY_HEADROOM_MB
        else:
            percent_headroom = int(slurm_mem_mb * MEMORY_HEADROOM_PERCENT)
            actual_headroom_mb = max(DEFAULT_MEMORY_HEADROOM_MB, percent_headroom)

    usable_mem_mb = slurm_mem_mb - actual_headroom_mb
    if usable_mem_mb <= 0:
        print(f"Warning: SLURM_MEM_PER_NODE ({slurm_mem_mb}MB) <= headroom ({actual_headroom_mb}MB)", file=sys.stderr)
        return None

    print(f"[Ray] Memory: {slurm_mem_mb}MB SLURM - {actual_headroom_mb}MB headroom = {usable_mem_mb}MB for Ray")
    return usable_mem_mb * 1024 * 1024  # Convert MB to bytes


@dataclass
class RayClusterConfig:
    """Configuration for a Ray cluster on SLURM."""

    num_nodes: int
    gpus_per_node: int
    cpus_per_node: int
    ray_port: int = 6379
    srun_export_env: str = "ALL"
    ray_env_vars: str = ""  # Space-separated KEY=value pairs for Ray workers
    wait_for_cluster_script: str = "scripts/ray/wait_for_cluster.py"
    poll_interval: int = 10
    startup_timeout: int = 600
    # Memory configuration (bytes). If None, Ray auto-detects (which can cause OOM).
    # Set explicitly to limit Ray to the SLURM allocation minus headroom.
    memory_per_node: Optional[int] = None  # Total memory Ray can use per node
    object_store_memory: Optional[int] = None  # Ray object store (plasma) size
    # Disable CPU binding for srun commands (needed for Frontier/Cray systems)
    disable_cpu_bind: bool = False
    # GPU binding mode for srun. "closest" binds CPUs based on GPU NUMA proximity,
    # "none" disables SLURM GPU-CPU binding. Default "none" avoids SLURM restricting
    # CPU affinity on complex NUMA topologies (e.g., GH200 with 36 NUMA nodes).
    # Use SKYRL_ENABLE_NUMA_AFFINITY for application-level per-GPU NUMA binding.
    gpu_bind: str = "none"
    # Enable periodic NUMA monitoring (useful for debugging GH200 unified memory allocation)
    # When enabled, logs numastat and nvidia-smi output every numa_monitor_interval seconds
    enable_numa_monitoring: bool = False
    numa_monitor_interval: int = 300  # 5 minutes
    # Enable proxychains for Ray workers (needed for JSC/Jupiter to access Daytona)
    # When True, LD_PRELOAD is preserved so Ray workers can make proxied external calls
    use_proxychains: bool = False
    # Path to proxychains4 binary for wrapped command approach (alternative to LD_PRELOAD)
    # When set, wraps ray commands with: proxychains4 -f $PROXYCHAINS_CONF_FILE ray start ...
    # This is more reliable on some systems (e.g., Jupiter ARM GH200 nodes)
    proxychains_binary: str = ""


@dataclass
class RayCluster:
    """Context manager for Ray cluster lifecycle on SLURM.

    This class handles:
    - Starting Ray head node
    - Starting Ray worker nodes
    - Waiting for cluster to be ready
    - Graceful shutdown on exit
    """

    config: RayClusterConfig
    head_ip: str
    node_list: List[str]
    _ray_pids: List[int] = field(default_factory=list)
    _ray_procs: List[subprocess.Popen] = field(default_factory=list)
    _ray_log_files: List = field(default_factory=list)  # Log file handles
    _numa_monitor_procs: List[subprocess.Popen] = field(default_factory=list)
    _numa_monitor_log_files: List = field(default_factory=list)
    _started: bool = False

    @classmethod
    def from_slurm(cls, config: RayClusterConfig) -> RayCluster:
        """Create a RayCluster from SLURM environment variables.

        This should be called inside a SLURM job where SLURM_JOB_NODELIST
        and related variables are set.
        """
        node_list = cls._get_slurm_nodes()
        head_ip = cls._get_node_ip(node_list[0], config.srun_export_env)
        return cls(config=config, head_ip=head_ip, node_list=node_list)

    @classmethod
    def from_hpc(cls, hpc: "HPC", num_nodes: int) -> RayCluster:
        """Create a RayCluster from an HPC configuration.

        Convenience method that extracts Ray-relevant settings from HPC.
        """
        # Compute Ray memory limit from SLURM allocation (prevents OOM from over-detection)
        ray_memory = compute_ray_memory_from_slurm()

        # Enable proxychains if the HPC cluster has it configured (e.g., JSC/Jupiter)
        # This allows Ray workers to make proxied external calls (e.g., Daytona API)
        # Prefer wrapped binary approach (proxychains_binary) over LD_PRELOAD (proxychains_preload)
        proxychains_binary = getattr(hpc, "proxychains_binary", "")
        use_proxychains = bool(proxychains_binary or getattr(hpc, "proxychains_preload", ""))

        # Enable NUMA monitoring if configured for this cluster (e.g., Jupiter GH200)
        # This helps debug NUMA locality issues that cause variable vLLM latency
        enable_numa_monitoring = getattr(hpc, "enable_numa_monitoring", False)

        ray_config = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=hpc.gpus_per_node,
            cpus_per_node=hpc.cpus_per_node,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
            memory_per_node=ray_memory,
            object_store_memory=DEFAULT_OBJECT_STORE_MEMORY_BYTES,
            disable_cpu_bind=getattr(hpc, "disable_cpu_bind", False),
            gpu_bind=getattr(hpc, "gpu_bind", "none"),
            use_proxychains=use_proxychains,
            proxychains_binary=proxychains_binary,
            enable_numa_monitoring=enable_numa_monitoring,
        )
        return cls.from_slurm(ray_config)

    @staticmethod
    def _get_slurm_nodes() -> List[str]:
        """Get list of node hostnames from SLURM environment."""
        nodelist = os.environ.get("SLURM_JOB_NODELIST", "")
        if not nodelist:
            raise RuntimeError(
                "SLURM_JOB_NODELIST not set. Are you running inside a SLURM job?"
            )
        result = subprocess.run(
            ["scontrol", "show", "hostnames", nodelist],
            capture_output=True,
            text=True,
            check=True,
        )
        nodes = result.stdout.strip().split("\n")
        if not nodes or nodes == [""]:
            raise RuntimeError(f"No nodes found in SLURM_JOB_NODELIST: {nodelist}")
        return nodes

    @staticmethod
    def _get_node_ip(node: str, srun_export: str) -> str:
        """Get IP address for a node using srun.

        Tries 'hostname -i' first (portable), then '--ip-address' as fallback.
        Frontier's Cray compute nodes don't support --ip-address long form.
        """
        srun_base = [
            "srun",
            f"--export={srun_export}",
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "--cpu-bind=none",  # Disable CPU binding for simple hostname lookup (fixes Frontier)
            "-w",
            node,
        ]

        # Try long form first (original behavior), fall back to short form (Frontier)
        # Unset proxychains env vars to avoid any interference with hostname lookup
        last_error = None
        for hostname_flag in ["--ip-address", "-i"]:
            try:
                result = subprocess.run(
                    srun_base + ["bash", "-c", f"unset LD_PRELOAD PROXYCHAINS_CONF_FILE 2>/dev/null; hostname {hostname_flag}"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                # hostname -i can return multiple IPs; take the first
                ip = result.stdout.strip().split()[0]
                if ip:
                    return ip
            except subprocess.CalledProcessError as e:
                last_error = e
                print(
                    f"[ray_utils] hostname {hostname_flag} failed on {node} "
                    f"(code {e.returncode}), trying next method...",
                    file=sys.stderr,
                )
                continue

        # Include details from last error in the exception
        error_msg = f"Failed to get IP address for node {node}"
        if last_error:
            error_msg += f": exit code {last_error.returncode}"
            if last_error.stderr:
                error_msg += f", stderr: {last_error.stderr.strip()}"
        raise RuntimeError(error_msg)

    @property
    def address(self) -> str:
        """Ray cluster address in the format host:port."""
        return f"{self.head_ip}:{self.config.ray_port}"

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs across all nodes."""
        return len(self.node_list) * self.config.gpus_per_node

    @property
    def total_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return len(self.node_list)

    def _cleanup_existing_ray(self) -> None:
        """Stop any existing Ray instances on allocated nodes.

        This ensures a clean slate before starting a new cluster,
        preventing conflicts with lingering processes from previous jobs.
        """
        print("Cleaning up existing Ray instances...", flush=True)
        for node in self.node_list:
            try:
                # Unset proxychains env vars so ray stop doesn't go through proxy
                subprocess.run(
                    [
                        "srun",
                        f"--export={self.config.srun_export_env}",
                        "--nodes=1",
                        "--ntasks=1",
                        "--overlap",
                        "--cpu-bind=none",  # No binding needed for cleanup
                        "-w",
                        node,
                        "bash", "-c",
                        "unset LD_PRELOAD PROXYCHAINS_CONF_FILE 2>/dev/null; ray stop --force",
                    ],
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                pass  # Ignore errors - node may not have Ray running
        # Brief pause to let processes terminate
        time.sleep(2)

    def start(self) -> None:
        """Start the Ray cluster.

        Starts the head node first, then worker nodes with a small delay
        between each to avoid overwhelming the head.
        """
        if self._started:
            print(f"Ray cluster already started at {self.address}")
            return

        # Clean up any lingering Ray instances from previous jobs
        self._cleanup_existing_ray()

        print(f"=== Starting Ray Cluster ===", flush=True)
        print(f"  Nodes: {len(self.node_list)}", flush=True)
        print(f"  GPUs per node: {self.config.gpus_per_node}", flush=True)
        print(f"  CPUs per node: {self.config.cpus_per_node}", flush=True)
        print(f"  Head node: {self.node_list[0]} ({self.head_ip})", flush=True)
        print(f"  Ray port: {self.config.ray_port}", flush=True)
        print(f"============================", flush=True)

        # Start head node
        self._start_node(self.node_list[0], is_head=True)
        print(f"  Started Ray head on {self.node_list[0]}", flush=True)

        # Wait for Ray head to register with SLURM before proceeding
        # This prevents "Socket timed out" / "Expired or invalid job" errors
        # when running subsequent srun commands too quickly after the head starts
        time.sleep(10)

        # Verify the Ray head process is still running
        if self._ray_procs and self._ray_procs[0].poll() is not None:
            log_dir = Path(os.environ.get("DCFT", ".")) / "experiments" / "logs"
            ray_log = log_dir / f"ray_head_{self.node_list[0]}.log"
            raise RuntimeError(
                f"Ray head process exited prematurely with code {self._ray_procs[0].returncode}. "
                f"Check log file: {ray_log}"
            )

        # Start worker nodes with delay
        for i, node in enumerate(self.node_list[1:], start=1):
            self._start_node(node, is_head=False)
            print(f"  Started Ray worker {i} on {node}", flush=True)
            time.sleep(3)  # Small delay between workers

        # Wait for cluster to be ready
        self._wait_for_cluster()
        self._started = True

        # Start NUMA monitoring if enabled (useful for GH200 debugging)
        self._start_numa_monitoring()

        print(f"=== Ray Cluster Ready ===", flush=True)
        print(f"  Address: {self.address}", flush=True)
        print(f"  Total GPUs: {self.total_gpus}", flush=True)
        print(f"=========================", flush=True)

    def stop(self) -> None:
        """Stop the Ray cluster.

        Sends stop commands to all nodes and waits for processes to exit.
        """
        if not self._started and not self._ray_procs:
            return

        print("Stopping Ray cluster...", flush=True)

        # Stop Ray on all nodes
        for node in self.node_list:
            try:
                # Unset proxychains env vars so ray stop doesn't go through proxy
                subprocess.run(
                    [
                        "srun",
                        f"--export={self.config.srun_export_env}",
                        "--nodes=1",
                        "--ntasks=1",
                        "--overlap",
                        "--cpu-bind=none",  # No binding needed for cleanup
                        "-w",
                        node,
                        "bash", "-c",
                        "unset LD_PRELOAD PROXYCHAINS_CONF_FILE 2>/dev/null; ray stop --force",
                    ],
                    capture_output=True,
                    timeout=30,
                )
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"  Warning: Failed to stop Ray on {node}: {e}", file=sys.stderr)

        # Wait for background processes
        for proc in self._ray_procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

        self._ray_procs.clear()
        self._ray_pids.clear()
        self._started = False

        # Close log files
        for log_file in self._ray_log_files:
            try:
                log_file.close()
            except Exception:
                pass
        self._ray_log_files.clear()

        # Stop NUMA monitoring
        self._stop_numa_monitoring()

        print("Ray cluster stopped", flush=True)

    def _start_node(self, node: str, is_head: bool) -> None:
        """Start Ray on a single node."""
        # Get IPv4 address for this node (ensures Ray uses IPv4, not hostnames that may resolve to IPv6)
        node_ip = self._get_node_ip(node, self.config.srun_export_env) if node != self.node_list[0] else self.head_ip

        if is_head:
            cmd = [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={self.head_ip}",
                f"--port={self.config.ray_port}",
                f"--num-gpus={self.config.gpus_per_node}",
                f"--num-cpus={self.config.cpus_per_node}",
                "--block",
            ]
        else:
            cmd = [
                "ray",
                "start",
                f"--address={self.address}",
                f"--node-ip-address={node_ip}",  # Force IPv4 for worker nodes too
                f"--num-gpus={self.config.gpus_per_node}",
                f"--num-cpus={self.config.cpus_per_node}",
                "--block",
            ]

        # Add memory limits to prevent Ray from detecting more memory than SLURM allocated
        if self.config.memory_per_node is not None:
            cmd.append(f"--memory={self.config.memory_per_node}")
        if self.config.object_store_memory is not None:
            cmd.append(f"--object-store-memory={self.config.object_store_memory}")

        # Build the bash command with environment variables and optional proxychains wrapper
        # Two proxychains modes are supported:
        # 1. Wrapped binary approach (preferred): proxychains4 -f <config> ray start ...
        #    More reliable on some systems (e.g., Jupiter ARM GH200 nodes)
        # 2. LD_PRELOAD approach: preserve LD_PRELOAD env var for Ray workers
        #    Requires localnet exclusions in proxychains config to not proxy Ray traffic

        if self.config.proxychains_binary:
            # Wrapped binary approach: unset LD_PRELOAD (avoid double-proxying) and wrap ray command
            # Uses $PROXYCHAINS_CONF_FILE env var (set by SSH tunnel setup script)
            unset_proxychains = "unset LD_PRELOAD 2>/dev/null; "
            ray_cmd_str = ' '.join(cmd)
            proxychains_wrap = f'{self.config.proxychains_binary} -f "$PROXYCHAINS_CONF_FILE" '
            if self.config.ray_env_vars:
                bash_cmd = f"{proxychains_wrap}{ray_cmd_str}"
            else:
                bash_cmd = f"{proxychains_wrap}{ray_cmd_str}"
        elif self.config.use_proxychains:
            # LD_PRELOAD approach: preserve proxychains env vars for external API calls
            # The proxychains config should have localnet exclusions for internal IPs
            if self.config.ray_env_vars:
                bash_cmd = f"env {self.config.ray_env_vars} {' '.join(cmd)}"
            else:
                bash_cmd = ' '.join(cmd)
        else:
            # No proxychains: unset env vars to prevent interference with Ray networking
            unset_proxychains = "unset LD_PRELOAD PROXYCHAINS_CONF_FILE 2>/dev/null; "
            if self.config.ray_env_vars:
                bash_cmd = f"{unset_proxychains}env {self.config.ray_env_vars} {' '.join(cmd)}"
            else:
                bash_cmd = f"{unset_proxychains}{' '.join(cmd)}"

        srun_cmd = [
            "srun",
            f"--export=ALL,VLLM_HOST_IP={node_ip}",
            "--nodes=1",
            "--ntasks=1",
            f"--gres=gpu:{self.config.gpus_per_node}",
            f"--gpu-bind={self.config.gpu_bind}",
            "--overlap",
        ]
        # Add --cpu-bind=none for Frontier/Cray systems
        if self.config.disable_cpu_bind:
            srun_cmd.append("--cpu-bind=none")
        srun_cmd.extend(["-w", node, "bash", "-c", bash_cmd])

        # Log Ray startup command and output for debugging
        role = "head" if is_head else "worker"
        log_dir = Path(os.environ.get("DCFT", ".")) / "experiments" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        ray_log_path = log_dir / f"ray_{role}_{node}.log"

        # Open log file for Ray output
        ray_log_file = open(ray_log_path, "w")
        ray_log_file.write(f"Ray {role} startup on {node}\n")
        ray_log_file.write(f"Command: {' '.join(srun_cmd)}\n")
        ray_log_file.write(f"Bash command: {bash_cmd}\n")
        ray_log_file.write("=" * 60 + "\n")
        ray_log_file.flush()

        print(f"  Starting Ray {role} on {node} (logging to {ray_log_path})...", flush=True)
        print(f"  Command: {' '.join(srun_cmd)}", flush=True)

        proc = subprocess.Popen(
            srun_cmd,
            stdout=ray_log_file,
            stderr=subprocess.STDOUT,
        )
        self._ray_procs.append(proc)
        self._ray_pids.append(proc.pid)
        self._ray_log_files.append(ray_log_file)  # Keep reference to close later

    def _wait_for_cluster(self) -> None:
        """Wait for the Ray cluster to be ready with expected resources.

        The wait script must run ON the head node (via srun) because ray.init()
        needs to connect to a local raylet to register as a driver.
        """
        script_path = Path(self.config.wait_for_cluster_script)

        if not script_path.exists():
            print(
                f"  Warning: {script_path} not found, using fallback wait",
                file=sys.stderr,
            )
            self._fallback_wait()
            return

        # Build the wait command
        # Unset proxychains env vars to avoid interfering with Ray communication
        unset_proxychains = "unset LD_PRELOAD PROXYCHAINS_CONF_FILE 2>/dev/null; "
        wait_cmd = unset_proxychains + " ".join([
            sys.executable,
            str(script_path),
            "--address", self.address,
            "--expected-gpus", str(self.total_gpus),
            "--expected-nodes", str(len(self.node_list)),
            "--timeout", str(self.config.startup_timeout),
            "--poll-interval", str(self.config.poll_interval),
        ])

        # Run on the head node via srun so ray.init() can connect to the local raylet
        srun_cmd = [
            "srun",
            f"--export={self.config.srun_export_env}",
            "--nodes=1",
            "--ntasks=1",
            "--overlap",
            "--cpu-bind=none",  # No binding needed for wait script
            "-w", self.node_list[0],  # Head node
            "bash", "-c", wait_cmd,
        ]

        print(f"  Waiting for cluster ({self.total_gpus} GPUs, {len(self.node_list)} nodes)...", flush=True)

        # Retry logic for transient SLURM communication errors
        # (e.g., "Socket timed out", "Expired or invalid job" when slurmctld is slow)
        max_retries = 3
        retry_delay = 5
        last_error = None

        for attempt in range(max_retries):
            try:
                # Run with stderr captured but stdout visible to user
                result = subprocess.run(
                    srun_cmd,
                    check=True,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return  # Success
            except subprocess.CalledProcessError as e:
                last_error = e
                stderr = e.stderr or ""
                # Check for transient SLURM communication errors
                if "Socket timed out" in stderr or "Unable to confirm allocation" in stderr:
                    if attempt < max_retries - 1:
                        print(
                            f"  SLURM communication error (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {retry_delay}s...",
                            flush=True,
                        )
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                # Log stderr for debugging if available
                if stderr:
                    print(f"  srun stderr: {stderr}", file=sys.stderr, flush=True)
                # Non-transient error or max retries reached
                break

        # All retries failed
        raise RuntimeError(
            f"Ray cluster failed to start within {self.config.startup_timeout}s "
            f"(last error: {last_error})"
        ) from last_error

    def _fallback_wait(self) -> None:
        """Fallback wait using ray.init() to check cluster status.

        This runs a polling loop ON the head node via srun, since ray.init()
        requires a local raylet connection.
        """
        import tempfile

        # Build Python script for polling
        poll_script = f'''import ray
import time
import sys

address = "{self.address}"
expected_gpus = {self.total_gpus}
timeout = {self.config.startup_timeout}
poll_interval = {self.config.poll_interval}

start_time = time.time()
while time.time() - start_time < timeout:
    try:
        ray.init(address=address, ignore_reinit_error=True)
        resources = ray.cluster_resources()
        gpu_count = resources.get("GPU", 0)
        num_nodes = len(ray.nodes())
        print(f"[Ray wait] nodes={{num_nodes}} GPUs={{gpu_count}}/{{expected_gpus}}", flush=True)
        if gpu_count >= expected_gpus:
            print("Cluster ready", flush=True)
            ray.shutdown()
            sys.exit(0)
        ray.shutdown()
    except Exception as e:
        print(f"[Ray wait] Connection error: {{e}}", flush=True)
    time.sleep(poll_interval)

print(f"Timeout: cluster did not reach {{expected_gpus}} GPUs within {{timeout}}s", flush=True)
sys.exit(1)
'''

        # Write script to a temp file (on shared filesystem) to avoid shell escaping issues
        # Using a file avoids the repr() escaping problems with inline -c scripts
        script_dir = os.environ.get("RAY_TMPDIR", "/tmp")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, f"ray_wait_{os.getpid()}.py")

        try:
            with open(script_path, "w") as f:
                f.write(poll_script)

            # Run on head node via srun
            # Wrap in bash to unset proxychains env vars before running Python with ray.init()
            bash_cmd = f"unset LD_PRELOAD PROXYCHAINS_CONF_FILE 2>/dev/null; {sys.executable} {script_path}"
            srun_cmd = [
                "srun",
                f"--export={self.config.srun_export_env}",
                "--nodes=1",
                "--ntasks=1",
                "--overlap",
                "--cpu-bind=none",  # No binding needed for polling script
                "-w", self.node_list[0],
                "bash", "-c", bash_cmd,
            ]

            print(f"  Waiting for cluster ({self.total_gpus} GPUs, fallback mode)...", flush=True)
            try:
                subprocess.run(srun_cmd, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Ray cluster failed to reach {self.total_gpus} GPUs "
                    f"within {self.config.startup_timeout}s"
                ) from e
        finally:
            # Clean up temp script
            if os.path.exists(script_path):
                os.remove(script_path)

    def _start_numa_monitoring(self) -> None:
        """Start background NUMA monitoring on all nodes.

        Periodically logs numastat and nvidia-smi output to help debug
        NUMA locality issues on unified memory architectures (e.g., GH200).
        """
        if not self.config.enable_numa_monitoring:
            return

        interval = self.config.numa_monitor_interval
        print(f"  Starting NUMA monitoring (interval: {interval}s)...", flush=True)

        log_dir = Path(os.environ.get("DCFT", ".")) / "experiments" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Monitoring script that runs on each node
        monitor_script = f'''
import time
import subprocess
import os
from datetime import datetime

interval = {interval}
node = os.environ.get("SLURMD_NODENAME", "unknown")

while True:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\\n{'='*60}", flush=True)
    print(f"NUMA Monitor - {{node}} - {{timestamp}}", flush=True)
    print(f"{'='*60}", flush=True)

    # GPU memory and NUMA topology
    print("\\n--- nvidia-smi ---", flush=True)
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        print(result.stdout, flush=True)
        if result.stderr:
            print(f"stderr: {{result.stderr}}", flush=True)
    except Exception as e:
        print(f"nvidia-smi error: {{e}}", flush=True)

    # NUMA memory statistics
    print("\\n--- numastat -m ---", flush=True)
    try:
        result = subprocess.run(["numastat", "-m"], capture_output=True, text=True, timeout=30)
        print(result.stdout, flush=True)
        if result.stderr:
            print(f"stderr: {{result.stderr}}", flush=True)
    except Exception as e:
        print(f"numastat error: {{e}}", flush=True)

    # CPU and memory binding info
    print("\\n--- numactl --show ---", flush=True)
    try:
        result = subprocess.run(["numactl", "--show"], capture_output=True, text=True, timeout=30)
        print(result.stdout, flush=True)
    except Exception as e:
        print(f"numactl error: {{e}}", flush=True)

    time.sleep(interval)
'''

        for node in self.node_list:
            numa_log_path = log_dir / f"numa_monitor_{node}.log"
            numa_log_file = open(numa_log_path, "w")
            numa_log_file.write(f"NUMA monitoring started on {node}\n")
            numa_log_file.write(f"Interval: {interval}s\n")
            numa_log_file.write("=" * 60 + "\n")
            numa_log_file.flush()

            # Run monitoring script in background via srun
            srun_cmd = [
                "srun",
                f"--export={self.config.srun_export_env}",
                "--nodes=1",
                "--ntasks=1",
                f"--gres=gpu:{self.config.gpus_per_node}",
                f"--gpu-bind={self.config.gpu_bind}",
                "--overlap",
                "-w", node,
                sys.executable, "-c", monitor_script,
            ]

            proc = subprocess.Popen(
                srun_cmd,
                stdout=numa_log_file,
                stderr=subprocess.STDOUT,
            )
            self._numa_monitor_procs.append(proc)
            self._numa_monitor_log_files.append(numa_log_file)
            print(f"    NUMA monitor started on {node} (log: {numa_log_path})", flush=True)

    def _stop_numa_monitoring(self) -> None:
        """Stop NUMA monitoring processes."""
        if not self._numa_monitor_procs:
            return

        print("  Stopping NUMA monitors...", flush=True)
        for proc in self._numa_monitor_procs:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass

        self._numa_monitor_procs.clear()

        for log_file in self._numa_monitor_log_files:
            try:
                log_file.close()
            except Exception:
                pass
        self._numa_monitor_log_files.clear()

    def __enter__(self) -> RayCluster:
        """Context manager entry - start the cluster."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop the cluster."""
        self.stop()


def create_ray_cluster_from_slurm(
    gpus_per_node: int,
    cpus_per_node: int,
    ray_port: int = 6379,
    srun_export_env: str = "ALL",
    ray_env_vars: str = "",
    memory_per_node: Optional[int] = None,
    object_store_memory: Optional[int] = None,
    disable_cpu_bind: bool = False,
) -> RayCluster:
    """Convenience function to create a Ray cluster from SLURM environment.

    This reads SLURM_JOB_NUM_NODES to determine the number of nodes.

    Args:
        gpus_per_node: Number of GPUs per node
        cpus_per_node: Number of CPUs per node
        ray_port: Port for Ray head node (default: 6379)
        srun_export_env: Environment export string for srun
        ray_env_vars: Space-separated KEY=value pairs for Ray workers
        memory_per_node: Memory limit per node in bytes (auto-detected from SLURM if None)
        object_store_memory: Ray object store size in bytes (default: 40GB)
        disable_cpu_bind: If True, add --cpu-bind=none to srun (needed for Frontier/Cray)

    Returns:
        A RayCluster configured from SLURM environment
    """
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))

    # Auto-detect memory from SLURM if not provided
    if memory_per_node is None:
        memory_per_node = compute_ray_memory_from_slurm()
    if object_store_memory is None:
        object_store_memory = DEFAULT_OBJECT_STORE_MEMORY_BYTES

    config = RayClusterConfig(
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        ray_port=ray_port,
        srun_export_env=srun_export_env,
        ray_env_vars=ray_env_vars,
        memory_per_node=memory_per_node,
        object_store_memory=object_store_memory,
        disable_cpu_bind=disable_cpu_bind,
    )

    return RayCluster.from_slurm(config)
