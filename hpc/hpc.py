import math
import os
import re
import socket
from typing import Dict, List
from pydantic import BaseModel, computed_field


class HPC(BaseModel):
    """Base pydantic model for HPC clusters.

    This class contains both job submission parameters (account, partition, etc.)
    and runtime configuration (modules, conda activation, env vars) needed for
    SBATCH job execution with Ray and vLLM.
    """

    name: str = ""
    hostname: str = ""
    hostname_pattern: str
    dotenv_filename: str
    account: str
    partition: str
    gpus_per_node: int
    cpus_per_node: int
    cpus_per_gpu: int | None = None
    mem_per_node: str = ""
    internet_node: bool
    gpus_type: str
    total_partition_nodes: int
    node_exclusion_list: str = ""
    qos: str = ""  # Most clusters don't use QOS; set explicitly where needed
    # GPU directive format: "--gres=gpu:{n}", "--gres=gpu:{type}:{n}", "--gpus-per-node={n}", or "" (no directive)
    # Use {n} as placeholder for GPU count, {type} for GPU type (e.g., h200, l40s)
    gpu_directive_format: str = ""
    # Default GPU type for clusters with multiple GPU types (e.g., "h200", "l40s")
    # Only used if gpu_directive_format contains {type}
    default_gpu_type: str = ""
    pretok_qos: str = ""
    pretok_cpus_per_node: int = 0  # will use all available cpus
    pretok_time_limit: str = "24:00:00"
    pretok_partition: str = ""
    pretok_gpus_per_node: int = 0  # will ask for 0 gpus
    local_mode: bool = False

    # Runtime configuration for SBATCH jobs (Ray/vLLM)
    modules: List[str] = []
    conda_activate: str = ""
    env_vars: Dict[str, str] = {}
    library_paths: Dict[str, str] = {}

    # NCCL/Networking settings (cluster-specific, used by universal templates)
    nccl_settings: Dict[str, str] = {}

    # Training launcher preference: "torchrun" or "accelerate"
    training_launcher: str = "torchrun"

    # SSH tunneling for no-internet clusters (JSC)
    needs_ssh_tunnel: bool = False

    # InfiniBand hostname suffix for MASTER_ADDR (JSC clusters use "i" suffix)
    master_addr_suffix: str = ""

    # SOCKS5 proxy configuration for no-internet clusters (JSC)
    # Alternative to SSH tunneling - uses existing proxy on login node
    proxy_host: str = ""
    proxy_port: int = 0
    proxychains_preload: str = ""
    # Path to proxychains4 binary for wrapped command approach (alternative to LD_PRELOAD)
    # Use this when LD_PRELOAD doesn't work reliably (e.g., Jupiter with ARM GH200 nodes)
    proxychains_binary: str = ""

    # Pre-run shell commands (cluster-specific setup)
    # These run at the start of the batch script before any other setup
    pre_run_commands: List[str] = []

    # CUDA path detection for complex clusters (Perlmutter)
    needs_cuda_detection: bool = False

    # Job time limits (cluster-specific)
    default_time_limit: str = "24:00:00"
    max_time_limit: str = "48:00:00"
    # Node-count-based time limits: list of (max_nodes, max_time) tuples, sorted by max_nodes ascending
    # Example: [(91, "02:00:00"), (183, "06:00:00")] means 1-91 nodes -> 2h, 92-183 nodes -> 6h
    time_limit_by_nodes: List[tuple[int, str]] = []

    # Node scaling presets for gosmall/gotrain/gofast helpers
    num_nodes_slow: int = 1
    num_nodes_default: int = 4
    num_nodes_fast: int = 8

    # Extra SBATCH directives (cluster-specific, e.g., licenses)
    extra_sbatch_directives: List[str] = []

    # GPU type to constraint mapping (e.g., Perlmutter needs --constraint for A100 variants)
    # Keys are GPU type strings (matching gpus_type or user-specified gpu_type)
    # Values are constraint strings (without #SBATCH prefix)
    # Special key "_default" is used when no gpu_type is specified
    gpu_type_constraints: Dict[str, str] = {}

    # Disable CPU binding for srun commands (needed for Frontier/Cray systems)
    # When True, adds --cpu-bind=none to srun commands for Ray startup
    disable_cpu_bind: bool = False

    # GPU binding mode for srun commands. "closest" tells SLURM to bind CPUs based
    # on GPU NUMA proximity, but can restrict affinity on complex topologies (GH200).
    # Default "none" avoids SLURM interference; use SKYRL_ENABLE_NUMA_AFFINITY for
    # application-level per-GPU NUMA binding instead.
    gpu_bind: str = "none"

    # Enable periodic NUMA monitoring for debugging unified memory allocation (GH200)
    # When True, logs numastat and nvidia-smi output every 5 minutes during Ray jobs
    enable_numa_monitoring: bool = False

    # Disable Ray's memory monitor (RAY_memory_monitor_refresh_ms=0) for all job types.
    # Needed when either: (1) GH200 unified memory makes GPU HBM visible as system RAM,
    # causing Ray to double-count GPU allocations toward the OOM threshold, or
    # (2) cgroup memory reporting is broken (returns -1), causing spurious kills.
    unified_gpu_memory: bool = False

    # Environment variables to unset after module loading (e.g., ROCR_VISIBLE_DEVICES on Frontier)
    # Modules may set these but they conflict with Ray/vLLM
    env_unsets: List[str] = []

    # Ray tmpdir base path (used for RAY_TMPDIR)
    # Default: /tmp/ray (suitable for most clusters)
    # JSC clusters use $SCRATCH/ray due to /tmp limitations on compute nodes
    ray_tmpdir_base: str = "/tmp/ray"

    def model_post_init(self, __context) -> None:
        # Derive a default CPU-per-GPU ratio when not explicitly provided.
        if not self.cpus_per_gpu:
            gpus = max(self.gpus_per_node, 1)
            if self.cpus_per_node:
                self.cpus_per_gpu = math.ceil(self.cpus_per_node / gpus)

    def get_max_time_limit(self, num_nodes: int) -> str:
        """Get the maximum allowed time limit for a given number of nodes.

        For clusters with node-count-based scheduling policies (e.g., Frontier),
        returns the max walltime for the appropriate bin. Falls back to max_time_limit
        if no node-based limits are configured.

        Args:
            num_nodes: Number of nodes requested for the job.

        Returns:
            Maximum allowed time limit string (e.g., "02:00:00").
        """
        if not self.time_limit_by_nodes:
            return self.max_time_limit

        for max_nodes, max_time in self.time_limit_by_nodes:
            if num_nodes <= max_nodes:
                return max_time

        # If num_nodes exceeds all bins, return the last (largest) bin's limit
        return self.time_limit_by_nodes[-1][1] if self.time_limit_by_nodes else self.max_time_limit

    @computed_field
    def dotenv_path(self) -> str:
        hpc_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(hpc_dir, "dotenv", self.dotenv_filename)

    # =========================================================================
    # Runtime configuration methods for SBATCH/Ray/vLLM
    # =========================================================================

    def get_module_commands(self) -> str:
        """Generate module load commands for SBATCH scripts.

        Also includes unset commands for env vars that modules set but
        conflict with Ray/vLLM (e.g., ROCR_VISIBLE_DEVICES on Frontier).
        """
        if not self.modules and not self.env_unsets:
            return ""
        lines = []
        lines.extend(f"module load {m}" for m in self.modules)
        # Unset env vars that modules set but conflict with Ray/vLLM
        for var in self.env_unsets:
            lines.append(f"unset {var}")
        return "\n".join(lines)

    def get_env_exports(self) -> str:
        """Generate environment variable exports for SBATCH scripts."""
        lines = []
        for key, value in {**self.env_vars, **self.library_paths}.items():
            lines.append(f'export {key}="{value}"')
        return "\n".join(lines)

    def get_exclude_directive(self) -> str:
        """Generate SBATCH exclude directive if nodes should be excluded."""
        if not self.node_exclusion_list:
            return ""
        return f"#SBATCH --exclude={self.node_exclusion_list}"

    def get_gpu_directive(self, gpus: int, gpu_type: str | None = None) -> str:
        """Generate SBATCH GPU directive for the given GPU count and type.

        Args:
            gpus: Number of GPUs to request.
            gpu_type: GPU type override (e.g., "h200", "l40s"). If None, uses default_gpu_type.

        Returns:
            SBATCH directive string (e.g., "#SBATCH --gres=gpu:h200:4") or empty string
            if the cluster doesn't use GPU directives (like TACC GH200 clusters).
        """
        if not self.gpu_directive_format or gpus <= 0:
            return ""
        directive = self.gpu_directive_format.replace("{n}", str(gpus))
        # Handle GPU type if format includes {type} placeholder
        if "{type}" in directive:
            resolved_type = gpu_type or self.default_gpu_type
            if not resolved_type:
                # If no type specified and format requires it, fall back to removing the type placeholder
                # This handles cases where a type is optional
                directive = directive.replace("{type}:", "").replace(":{type}", "").replace("{type}", "")
            else:
                directive = directive.replace("{type}", resolved_type)
        return f"#SBATCH {directive}"

    def get_mem_directive(self, mem: str | None = None) -> str:
        """Generate SBATCH memory directive.

        Args:
            mem: Memory string override. If None, uses cluster's mem_per_node.

        Returns:
            SBATCH directive string (e.g., "#SBATCH --mem=192G") or empty string
            if memory is not configured for this cluster.
        """
        mem_value = mem or self.mem_per_node
        if not mem_value:
            return ""
        return f"#SBATCH --mem={mem_value}"

    def get_sbatch_directives(
        self, qos: str = "", gpus: int = 0, gpu_type: str | None = None, mem: str | None = None
    ) -> str:
        """Generate cluster-specific SBATCH directives.

        Returns directives for partition, account, QoS, GPU, memory, exclusions, etc.
        Only includes directives that are actually needed for this cluster.

        Args:
            qos: Optional QoS string.
            gpus: Number of GPUs to request (0 = use cluster default or skip).
            gpu_type: GPU type override (e.g., "h200", "l40s"). Uses default if None.
            mem: Memory override (uses cluster default if None).
        """
        lines = []
        if self.partition:
            lines.append(f"#SBATCH -p {self.partition}")
        if self.account:
            lines.append(f"#SBATCH --account {self.account}")
        if qos:
            lines.append(f"#SBATCH --qos {qos}")
        gpu_directive = self.get_gpu_directive(gpus, gpu_type)
        if gpu_directive:
            lines.append(gpu_directive)
        mem_directive = self.get_mem_directive(mem)
        if mem_directive:
            lines.append(mem_directive)
        if self.node_exclusion_list:
            lines.append(f"#SBATCH --exclude={self.node_exclusion_list}")
        # Add constraint directive based on GPU type (e.g., Perlmutter A100 variants)
        if self.gpu_type_constraints:
            constraint_key = gpu_type if gpu_type else "_default"
            constraint = self.gpu_type_constraints.get(constraint_key)
            if constraint:
                lines.append(f"#SBATCH --constraint {constraint}")
        # Add any extra cluster-specific directives (e.g., licenses)
        for directive in self.extra_sbatch_directives:
            lines.append(directive)
        return "\n".join(lines)

    def get_srun_export_env(self) -> str:
        """Generate SRUN --export string with all necessary env vars."""
        env_parts = ["ALL"]
        for key, value in {**self.env_vars, **self.library_paths}.items():
            env_parts.append(f"{key}={value}")
        # Add common paths
        # env_parts.append("PATH=$PATH")
        # env_parts.append("LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}")
        # env_parts.append("PYTHONPATH=${PYTHONPATH:-}")
        # env_parts.append("HF_HOME=${HF_HOME:-}")
        return ",".join(env_parts)

    def get_ray_env_vars(self) -> str:
        """Generate space-separated env vars for Ray worker processes."""
        env_parts = []
        for key, value in {**self.env_vars, **self.library_paths}.items():
            env_parts.append(f"{key}={value}")
        env_parts.append("PATH=$PATH")
        env_parts.append("LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}")
        env_parts.append("PYTHONPATH=${PYTHONPATH:-}")
        env_parts.append("HF_HOME=${HF_HOME:-}")
        # Propagate CUDA_HOME so Triton can find ptxas on worker nodes
        # (set by 'module load CUDA/...' but not always inherited by Ray workers)
        env_parts.append("CUDA_HOME=${CUDA_HOME:-}")
        return " ".join(env_parts)

    def get_nccl_exports(self) -> str:
        """Generate export statements for NCCL/networking settings."""
        if not self.nccl_settings:
            return "# No cluster-specific NCCL settings"

        lines = ["# Cluster-specific NCCL/networking settings"]
        for key, value in self.nccl_settings.items():
            lines.append(f'export {key}="{value}"')
        return "\n".join(lines)

    def get_ray_env_exports(self, experiments_dir: str) -> str:
        """Generate Ray-specific environment defaults for SBATCH scripts.

        RAY_TMPDIR_BASE is configurable per-cluster (ray_tmpdir_base field).
        For clusters using $SCRATCH, falls back to /tmp if SCRATCH is undefined.
        """
        # Handle env var references (e.g., "$SCRATCH/ray") with fallback to /tmp
        if self.ray_tmpdir_base.startswith("$"):
            # Extract var name and path suffix (e.g., "$SCRATCH/ray" -> "SCRATCH", "/ray")
            parts = self.ray_tmpdir_base[1:].split("/", 1)
            var_name = parts[0]
            suffix = "/" + parts[1] if len(parts) > 1 else ""
            tmpdir_base_line = f'  RAY_TMPDIR_BASE="${{{var_name}:-/tmp}}{suffix}"'
        else:
            tmpdir_base_line = f'  RAY_TMPDIR_BASE="{self.ray_tmpdir_base}"'

        lines = [
            "# --- Ray defaults ---",
            'export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-900}"',
            "# Disable Ray OOM monitor: FSDP init transiently spikes CPU RAM",
            "# (e.g., 4 workers × 32B model peaks at ~249GB on 251GB nodes).",
            "# The spike settles after init; Ray's default 0.95 threshold kills",
            "# workers during this transient phase. Already disabled for GH200",
            "# (unified memory), now disabled universally.",
            'export RAY_memory_monitor_refresh_ms=0',
        ]

        lines += [
            'if [ -z "${RAY_TMPDIR:-}" ]; then',
            tmpdir_base_line,
            '  RAY_TMPDIR="${RAY_TMPDIR_BASE}/ray_${SLURM_JOB_ID:-$$}"',
            '  mkdir -p "$RAY_TMPDIR"',
            "fi",
            'export RAY_TMPDIR="${RAY_TMPDIR}"',
            'echo "[ray] RAY_TMPDIR=$RAY_TMPDIR"',
        ]
        return "\n".join(lines)

    def get_ssh_tunnel_setup(self) -> str:
        """Generate SSH tunnel setup script for no-internet clusters (JSC).

        Creates SSH tunnel from compute node to login node, then uses LD_PRELOAD
        with proxychains to route external traffic through the tunnel.

        IMPORTANT: Uses LD_PRELOAD (not CMD_PREFIX wrapper) so that Ray workers
        inherit the proxy configuration. The CMD_PREFIX approach doesn't work
        because Ray spawns child processes that don't inherit the wrapper.

        Requirements:
        - SSH_KEY environment variable must be set to path of SSH private key
        - Public key must be in ~/.ssh/authorized_keys on login node
        - proxychains-ng library must exist at cluster-specific path
        """
        if not self.needs_ssh_tunnel:
            return "# No SSH tunnel needed for this cluster"

        return r'''# ============================================================================
# SSH Tunnel + Proxychains Setup for No-Internet Clusters (JSC)
# Wrapped in a function so `return` works correctly in sbatch scripts.
# ============================================================================
_setup_proxy() {
# ============================================================================
#
# Creates SOCKS5 proxy via SSH tunnel to login node, then uses proxychains
# to route external traffic through the tunnel.
#
# Jupiter (ARM GH200): Uses wrapped binary approach (proxychains4 -f <config> cmd)
# Other JSC clusters: Uses LD_PRELOAD approach for Ray worker inheritance
# ============================================================================

# Determine login node and proxychains paths based on cluster
NODE_HOST=$(hostname -s)
PROXYCHAINS_MODE=""  # "binary" or "ldpreload"

if [[ $NODE_HOST == jrc* ]]; then
    LOGIN_NODE="jrlogin05i"
    PROXYCHAINS_LIB="/p/scratch/synthlaion/dc-agent-shared/tools/proxychains-ng-install/lib/libproxychains4.so"
    PROXYCHAINS_MODE="ldpreload"
elif [[ $NODE_HOST == jwb* ]]; then
    LOGIN_NODE="jwlogin22i"
    PROXYCHAINS_LIB="/p/scratch/synthlaion/dc-agent-shared/tools/proxychains-ng-install/lib/libproxychains4.so"
    PROXYCHAINS_MODE="ldpreload"
elif [[ $NODE_HOST == jpb* ]] || [[ $NODE_HOST == jpc* ]]; then
    LOGIN_NODE="jpbl-s01-01"
    # Jupiter uses aarch64 build - binary wrapper approach (LD_PRELOAD doesn't work reliably)
    PROXYCHAINS_BIN="/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4"
    PROXYCHAINS_MODE="binary"
elif [[ $NODE_HOST == lrdn* ]] || [[ $NODE_HOST == *.leonardo.local ]]; then
    LOGIN_NODE="login05-ext.leonardo.cineca.it"
    # Leonardo uses x86 build - binary wrapper approach
    PROXYCHAINS_BIN="/leonardo/home/userexternal/bfeuer00/proxychains/bin/proxychains4"
    PROXYCHAINS_MODE="binary"
else
    echo "[proxy] Unknown cluster for node $NODE_HOST - skipping proxy setup"
    return 0
fi

TUNNEL_PORT=7003

# Check if proxychains is available
if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
    if [ ! -x "$PROXYCHAINS_BIN" ]; then
        echo "[proxy] ✗ proxychains binary not found at $PROXYCHAINS_BIN"
        echo "[proxy] Skipping proxy setup - external connectivity will fail"
        return 0
    fi
    echo "[proxy] ✓ Found proxychains binary at $PROXYCHAINS_BIN"
else
    if [ ! -f "$PROXYCHAINS_LIB" ]; then
        echo "[proxy] ✗ proxychains library not found at $PROXYCHAINS_LIB"
        echo "[proxy] Skipping proxy setup - external connectivity will fail"
        return 0
    fi
    echo "[proxy] ✓ Found proxychains library at $PROXYCHAINS_LIB"
fi

if [ -z "${SSH_KEY:-}" ]; then
    echo "[proxy] SSH_KEY not set - skipping proxy setup"
    echo "[proxy] Set SSH_KEY in your environment to enable internet access"
else
    # Get this node's IP address for multi-node proxy access
    NODE_IP=$(nslookup $NODE_HOST | grep 'Address' | tail -n1 | awk '{print $2}')
    echo "[proxy] Setting up SSH tunnel to $LOGIN_NODE"
    echo "[proxy] SSH key: $SSH_KEY"
    echo "[proxy] Tunnel port: $TUNNEL_PORT"
    echo "[proxy] Node IP: $NODE_IP (workers will connect here)"

    # Create SSH tunnel with SOCKS5 proxy
    # -g flag allows remote hosts (worker nodes) to connect to the tunnel
    ssh -g -f -N -D ${TUNNEL_PORT} \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=1000 \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=30 \
        -o TCPKeepAlive=yes \
        -o ExitOnForwardFailure=yes \
        -o BatchMode=yes \
        -i ${SSH_KEY} \
        ${USER}@${LOGIN_NODE}

    # Give tunnel time to establish
    sleep 5

    # Verify tunnel is running
    if pgrep -f "ssh.*-D.*${TUNNEL_PORT}" > /dev/null; then
        echo "[proxy] ✓ SSH tunnel started successfully"
    else
        echo "[proxy] ✗ SSH tunnel failed to start"
        return 0
    fi

    # ============================================================================
    # Generate proxychains config
    # Key: Uses NODE_IP (not localhost) so worker nodes can access the tunnel
    # localnet entries ensure internal traffic (Ray, NCCL) bypasses proxy
    # ============================================================================
    SLURM_JOB_ID=${SLURM_JOB_ID:-"local"}
    CFG_PATH=~/.proxychains/proxychains_${SLURM_JOB_ID}.conf
    mkdir -p ~/.proxychains

    cat > "$CFG_PATH" <<PCEOF
strict_chain
quiet_mode
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
localnet 169.254.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF

    echo "[proxy] ✓ Generated proxychains config at $CFG_PATH"
    echo "[proxy]   - Internal traffic (10.x.x.x, 172.x.x.x, 169.254.x.x) → DIRECT"
    echo "[proxy]   - External traffic (internet) → PROXY via tunnel"

    # ============================================================================
    # Export proxychains configuration based on mode
    # ============================================================================
    export PROXYCHAINS_CONF_FILE="$CFG_PATH"
    export PROXYCHAINS_SOCKS5_HOST="${NODE_IP}"
    export PROXYCHAINS_SOCKS5_PORT="${TUNNEL_PORT}"

    # if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
    #     # Binary wrapper approach (Jupiter ARM GH200)
    #     # Ray workers will use: proxychains4 -f $PROXYCHAINS_CONF_FILE ray start ...
    #     export PROXYCHAINS_BINARY="$PROXYCHAINS_BIN"
    #     echo "[proxy] ✓ PROXYCHAINS_BINARY=$PROXYCHAINS_BIN"
    #     echo "[proxy] ✓ PROXYCHAINS_CONF_FILE=$CFG_PATH"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_HOST=${NODE_IP} (accessible from worker nodes)"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_PORT=${TUNNEL_PORT}"
    # else
    #     # LD_PRELOAD approach (Jureca, Juwels)
    #     # Ray workers inherit proxy via LD_PRELOAD environment variable
    #     export LD_PRELOAD="$PROXYCHAINS_LIB"
    #     echo "[proxy] ✓ LD_PRELOAD set to $PROXYCHAINS_LIB"
    #     echo "[proxy] ✓ PROXYCHAINS_CONF_FILE=$CFG_PATH"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_HOST=${NODE_IP} (accessible from worker nodes)"
    #     echo "[proxy] ✓ PROXYCHAINS_SOCKS5_PORT=${TUNNEL_PORT}"
    # fi

    # ============================================================================
    # Daytona/aiohttp timeout and retry settings
    # ============================================================================
    export DAYTONA_MAX_RETRIES=5
    export DAYTONA_RETRY_DELAY=30
    export DAYTONA_BACKOFF_FACTOR=2
    export DAYTONA_TIMEOUT=1800  # 30 minutes
    export AIOHTTP_CLIENT_TIMEOUT=900  # 15 minutes
    export AIOHTTP_CONNECTOR_TIMEOUT=900
    export AIOHTTP_SOCK_CONNECT_TIMEOUT=300
    export AIOHTTP_TOTAL_TIMEOUT=1800

    # Disable SSL verification (JSC certificate issues)
    export PYTHONHTTPSVERIFY=0
    unset SSL_CERT_FILE
    unset CURL_CA_BUNDLE
    unset REQUESTS_CA_BUNDLE
    unset SSL_CERT_DIR

    echo "[proxy] ✓ Daytona timeout settings configured"

    # Test proxy connectivity
    echo "[proxy] Testing proxy connectivity..."
    if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
        if "$PROXYCHAINS_BIN" -f "$CFG_PATH" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null; then
            echo "[proxy] ✓ Proxy connectivity test passed (huggingface.co reachable via wrapped binary)"
        else
            echo "[proxy] ⚠ Proxy connectivity test failed (may still work for Daytona)"
        fi
    else
        if curl -s --connect-timeout 10 https://huggingface.co -o /dev/null 2>/dev/null; then
            echo "[proxy] ✓ Proxy connectivity test passed (huggingface.co reachable via LD_PRELOAD)"
        else
            echo "[proxy] ⚠ Proxy connectivity test failed (may still work for Daytona)"
        fi
    fi

    # Test that tunnel is accessible from this node's IP (for worker node access)
    if nc -z ${NODE_IP} ${TUNNEL_PORT} 2>/dev/null; then
        echo "[proxy] ✓ Tunnel accessible at ${NODE_IP}:${TUNNEL_PORT} (workers can connect)"
    else
        echo "[proxy] ⚠ Tunnel not accessible at ${NODE_IP}:${TUNNEL_PORT} (workers may fail)"
    fi

    if [[ "$PROXYCHAINS_MODE" == "binary" ]]; then
        echo "[proxy] ✓ Proxy setup complete (using wrapped binary for Ray workers)"
    else
        echo "[proxy] ✓ Proxy setup complete (using LD_PRELOAD for Ray worker inheritance)"
    fi
fi
}
_setup_proxy
'''

    def get_proxy_setup(self) -> str:
        """Generate SOCKS5 proxy setup script for no-internet clusters (JSC).

        Uses an existing SOCKS5 proxy (e.g., JSC's shared proxy at 10.14.0.53:1080)
        instead of setting up an SSH tunnel. This is more reliable and doesn't
        require SSH keys.

        Sets:
        - SOCKS_PROXY_URL: For Harbor/httpx to use via ALL_PROXY
        - PROXYCHAINS_SOCKS5_HOST/PORT: For proxychains-ng
        - PROXYCHAINS_PRELOAD: LD_PRELOAD path for proxychains

        Returns:
            Bash script for proxy setup, or comment if no proxy configured.
        """
        if not self.proxy_host or not self.proxy_port:
            return "# No proxy configured for this cluster"

        script = f'''# ============================================================================
# SOCKS5 Proxy Setup for No-Internet Clusters (JSC)
# Uses existing proxy instead of SSH tunnel - more reliable
#
# KEY: Proxychains with localnet exclusions ensures:
#   - Internal traffic (Ray, NCCL) → DIRECT (no proxy)
#   - External traffic (Daytona API) → Through SOCKS5 proxy
# ============================================================================
PROXY_HOST="{self.proxy_host}"
PROXY_PORT="{self.proxy_port}"
PROXYCHAINS_CONF="/tmp/proxychains_${{SLURM_JOB_ID}}.conf"

echo "[proxy] Setting up SOCKS5 proxy at $PROXY_HOST:$PROXY_PORT"

# Test proxy connectivity
if nc -z $PROXY_HOST $PROXY_PORT 2>/dev/null; then
    echo "[proxy] ✓ Proxy reachable at $PROXY_HOST:$PROXY_PORT"
else
    echo "[proxy] ✗ WARNING: Proxy not reachable at $PROXY_HOST:$PROXY_PORT"
fi

# Generate proxychains config with localnet exclusions
cat > "$PROXYCHAINS_CONF" << PCEOF
# Proxychains config for JSC HPC - auto-generated
# Proxy ONLY external traffic (Daytona), bypass internal (Ray, NCCL)

dynamic_chain
quiet_mode

# NOTE: proxy_dns is DISABLED to allow local DNS resolution for internal hostnames
# (e.g., jpbo-021-27.jupiter.internal). External hostnames like api.daytona.io
# will still work because socks5h:// does DNS at the proxy.

tcp_read_time_out 15000
tcp_connect_time_out 8000

# CRITICAL: Exclude internal networks from proxying
# This is what keeps Ray and NCCL working!
localnet 127.0.0.0/255.0.0.0
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
localnet 169.254.0.0/255.255.0.0

[ProxyList]
socks5 $PROXY_HOST $PROXY_PORT
PCEOF

echo "[proxy] ✓ Generated proxychains config at $PROXYCHAINS_CONF"
echo "[proxy]   - Internal traffic (10.x.x.x) → DIRECT (no proxy)"
echo "[proxy]   - External traffic (internet) → PROXY"

# Save a copy to the experiments directory for debugging
EXPERIMENTS_DIR="${DCFT:-$PWD}/experiments"
if [ -d "$EXPERIMENTS_DIR" ]; then
    PERSISTENT_CONF="$EXPERIMENTS_DIR/proxychains_${SLURM_JOB_ID}.conf"
    cp "$PROXYCHAINS_CONF" "$PERSISTENT_CONF" 2>/dev/null && \
        echo "[proxy] ✓ Saved config copy to $PERSISTENT_CONF"
fi

# Export for proxychains
export PROXYCHAINS_CONF_FILE="$PROXYCHAINS_CONF"
echo "[proxy] PROXYCHAINS_CONF_FILE=$PROXYCHAINS_CONF_FILE"

# Also set SOCKS_PROXY_URL for applications that can use it directly
export SOCKS_PROXY_URL="socks5h://$PROXY_HOST:$PROXY_PORT"
echo "[proxy] SOCKS_PROXY_URL=$SOCKS_PROXY_URL"'''

        if self.proxychains_binary:
            # Wrapped binary approach (preferred for Jupiter ARM GH200 nodes)
            script += f'''

# Proxychains binary for wrapped command approach
export PROXYCHAINS_BINARY="{self.proxychains_binary}"
echo "[proxy] PROXYCHAINS_BINARY=$PROXYCHAINS_BINARY"

# Test proxychains with the config (using wrapped binary)
echo "[proxy] Testing proxychains connectivity..."
if "{self.proxychains_binary}" -f "$PROXYCHAINS_CONF" curl -s --connect-timeout 10 https://huggingface.co -o /dev/null; then
    echo "[proxy] ✓ Proxychains test passed - external connectivity works"
else
    echo "[proxy] ⚠ Proxychains test failed (may still work for applications)"
fi'''
        elif self.proxychains_preload:
            # LD_PRELOAD approach (fallback)
            script += f'''

# Proxychains library for LD_PRELOAD
export PROXYCHAINS_PRELOAD="{self.proxychains_preload}"
echo "[proxy] PROXYCHAINS_PRELOAD=$PROXYCHAINS_PRELOAD"

# Test proxychains with the config
if PROXYCHAINS_CONF_FILE="$PROXYCHAINS_CONF" LD_PRELOAD="{self.proxychains_preload}" \\
   curl -s --connect-timeout 5 https://huggingface.co -o /dev/null 2>/dev/null; then
    echo "[proxy] ✓ Proxychains test passed - external connectivity works"
else
    echo "[proxy] ⚠ Proxychains test failed (may still work for applications)"
fi'''

        script += '''

echo "[proxy] ✓ Proxy environment configured"'''

        return script

    def get_pre_run_commands(self) -> str:
        """Generate pre-run commands for cluster-specific setup.

        Returns:
            Bash commands to run at the start of the batch script.
        """
        if not self.pre_run_commands:
            return "# No cluster-specific pre-run commands"

        lines = ["# Cluster-specific pre-run commands"]
        for cmd in self.pre_run_commands:
            lines.append(cmd)
        return "\n".join(lines)


jureca = HPC(
    name="jureca",
    hostname_pattern=r"jr.*?.jureca",
    dotenv_filename="jureca.env",
    account="westai0007",  # synthlaion (24 nodes per job)
    partition="dc-hwai",  # dc-gpu
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="H100 94GB",
    total_partition_nodes=16,
    gpu_directive_format="--gres=gpu:{n}",
    # Runtime configuration for Ray/vLLM
    modules=["CUDA/12.3"],
    env_vars={
        "PYTHONFAULTHANDLER": "1",
        "WANDB_MODE": "offline",  # No internet on compute nodes
        # Disable symmetric memory allreduce — send_fd fails in Singularity containers
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    },
    # NCCL/networking settings for SFT training (InfiniBand, no internet)
    nccl_settings={
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_IB_TIMEOUT": "60",
    },
    training_launcher="accelerate",
    # JSC shared SOCKS5 proxy (more reliable than SSH tunnels)
    needs_ssh_tunnel=False,
    proxy_host="10.14.0.53",
    proxy_port=1080,
    proxychains_preload="/p/scratch/laionize/raj3/proxychains-ng/libproxychains4.so",
    # JSC-specific setup (disable core dumps to save disk space)
    pre_run_commands=["ulimit -c 0"],
    # Ray tmpdir on scratch (JSC /tmp is limited on compute nodes)
    #ray_tmpdir_base="$SCRATCH/ray",
    # Job scaling (from jureca.env)
    default_time_limit="24:00:00",
    num_nodes_default=1,
    num_nodes_fast=4,
)

jupiter = HPC(
    name="jupiter",
    # Matches login nodes like jpbl-s01-01 (jupiter booster login) and compute nodes
    hostname_pattern=r"jp(bl|cn|c)-.*",
    dotenv_filename="jupiter.env",
    account="reformo",
    partition="booster",
    gpus_per_node=4,  # 4x GH200 superchips per node
    cpus_per_node=288,  # 4 Grace CPUs × 72 cores = 288 ARM cores per node
    internet_node=False,  # Compute nodes have no internet (like other JSC clusters)
    gpus_type="GH200 96GB (H100 + Grace)",
    unified_gpu_memory=True,
    total_partition_nodes=6000,  # ~6000 booster nodes
    gpu_directive_format="--gres=gpu:{n}",
    # GCC 14 + CUDA 13 modules required for vLLM wheel builds and DeepSpeed
    # (sets CUDA_HOME=/e/software/.../CUDA/13)
    modules=["GCC/14.3.0", "nvidia-compilers/25.9-CUDA-13"],
    env_vars={
        "WANDB_MODE": "offline",  # Compute nodes have no internet
        # Force GLOO and NCCL to use IPv4 (IPv6 doesn't work on Jupiter compute nodes)
        "GLOO_USE_IPV6": "0",
        "NCCL_SOCKET_FAMILY": "AF_INET",
        # NOTE: Do NOT set GLOO_SOCKET_IFNAME=ib0 - it causes Gloo to use the IB hostname
        # which resolves to IPv6. Let Gloo auto-detect the interface.
        # GH200 NUMA affinity: bind each GPU worker to its local CPU NUMA node
        "SKYRL_ENABLE_NUMA_AFFINITY": "1",
        # Use httpx instead of aiohttp for LiteLLM HTTP transport.
        # aiohttp + uvloop has a known issue: when agent timeouts cancel in-flight
        # acompletion() calls, the abandoned coroutine's aiohttp session gets GC'd,
        # closing the socket fd while uvloop's epoll still references it → EBADF → SIGABRT.
        "DISABLE_AIOHTTP_TRANSPORT": "True",
        # Disable symmetric memory allreduce — send_fd fails in Singularity containers
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        # Disable cuDNN SDP backend in SDPA. On GH200 + torch 2.9.1 + CUDA 13,
        # cuDNN's MHA graph execution hits an edge case on long sequences:
        #   RuntimeError: Expected mha_graph->execute(...).is_good() to be true
        # This was fixed upstream (disabled by default) in pytorch/pytorch#459e2aa
        # but that commit post-dates torch 2.9.1. Flash SDP and math backends
        # still work fine. Only affects SFT (RL uses vLLM, not SDPA).
        "TORCH_CUDNN_SDPA_ENABLED": "0",
    },
    # NOTE: Do NOT use master_addr_suffix="i" - the "i" suffixed hostname is not DNS-resolvable
    # InfiniBand routing is handled by NCCL_SOCKET_IFNAME=ib0 instead
    # NCCL/networking settings for SFT training (InfiniBand NDR)
    nccl_settings={
        "NCCL_DEBUG": "INFO",
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_IB_TIMEOUT": "60",
    },
    training_launcher="accelerate",
    needs_ssh_tunnel=True,
    proxychains_binary="/e/scratch/jureap59/feuer1/proxychains-ng-aarch64/bin/proxychains4",
    # Enable NUMA monitoring for GH200 unified memory debugging
    enable_numa_monitoring=True,
    # GH200 NUMA: --gpu-bind=closest restricts CPU affinity to NUMA node 0 only
    # and overrides --cpu-bind=none. Use gpu_bind="none" + disable_cpu_bind=True
    # to let SKYRL_ENABLE_NUMA_AFFINITY handle per-GPU NUMA binding at app level.
    conda_activate="source /e/scratch/jureap59/feuer1/miniforge3/etc/profile.d/conda.sh && conda activate otagent",
    gpu_bind="none",
    disable_cpu_bind=True,
    pre_run_commands=["ulimit -c 0"],
    # Ray tmpdir on scratch (JSC /tmp is limited on compute nodes)
    #ray_tmpdir_base="$SCRATCH/ray",
    default_time_limit="12:00:00",
    max_time_limit="23:59:00",
    num_nodes_slow=1,
    num_nodes_default=4,
    num_nodes_fast=8,
    # Exclude rack 031 - nodes are on 10.128.26.x subnet which can't communicate
    # with 10.128.25.x subnet nodes (racks 026-030) causing Ray cluster failures
    # Also exclude jpbo-038-38, jpbo-065-17, jpbo-074-22, jpbo-048-41, jpbo-011-[01-48] - recurring NCCL socket/heartbeat failures
    # jpbo-091-05 added 2026-04-15: repeated SIGABRT on 32B training (hung the 32B-5ds superset job chain twice)
    # jpbo-044-0[1-5] added 2026-04-22: repeated NCCL TCPStore broken-pipe stalls on Qwen3.5-9B chain (4 consecutive failures)
    node_exclusion_list="jpbo-031-[01-48],jpbo-011-[01-48],jpbo-038-38,jpbo-004-46,jpbo-065-17,jpbo-074-22,jpbo-048-41,jpbo-091-05,jpbo-044-0[1-5]",
)

juwels = HPC(
    name="juwels",
    hostname_pattern=r"jw.*?.juwels",
    dotenv_filename="juwels.env",
    account="laionize",
    partition="booster",
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="A100 40GB",
    total_partition_nodes=936,
    node_exclusion_list="jwb[0059,0067,0069,0193,0198,0215,0266,0284,0287,0294,0359,0418,0637,0647,0829,0832,0838,0898,0907,0921,0971,1004,1023,1029,1213]",
    gpu_directive_format="--gres=gpu:{n}",
    env_vars={
        "WANDB_MODE": "offline",  # No internet on compute nodes
        # Disable symmetric memory allreduce — send_fd fails in Singularity containers
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
    },
    # NCCL/networking settings for SFT training (InfiniBand, no internet)
    nccl_settings={
        "NCCL_NET_GDR_LEVEL": "0",
        "NCCL_SOCKET_IFNAME": "ib0",
        "NCCL_IB_TIMEOUT": "60",
    },
    training_launcher="accelerate",
    # JSC shared SOCKS5 proxy (more reliable than SSH tunnels)
    needs_ssh_tunnel=False,
    proxy_host="10.14.0.53",
    proxy_port=1080,
    proxychains_preload="/p/scratch/laionize/raj3/proxychains-ng/libproxychains4.so",
    # JSC-specific setup (disable core dumps to save disk space)
    pre_run_commands=["ulimit -c 0"],
    # Ray tmpdir on scratch (JSC /tmp is limited on compute nodes but $SCRATCH path is too long)
    #ray_tmpdir_base="$SCRATCH/ray",
    # Job scaling (from juwels.env)
    default_time_limit="24:00:00",
    num_nodes_default=4,
    num_nodes_fast=8,
)

leonardo = HPC(
    name="leonardo",
    hostname_pattern=r".*?.leonardo.local",
    dotenv_filename="leonardo.env",
    account="AIFAC_5C0_290",
    partition="boost_usr_prod",
    gpus_per_node=4,
    cpus_per_node=32,
    internet_node=False,
    gpus_type="A100 64GB",
    total_partition_nodes=3456,
    env_vars={
        "WANDB_MODE": "offline",  # No internet on compute nodes
        "TORCH_NCCL_ENABLE_MONITORING": "1",  # Kill hung jobs instead of burning wall-time
        "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "600",  # 10 min before declaring hang (default 480s)
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    },
    # node_exclusion_list="lrdn[1606,2776,2425,2808,3064,3064,1953,2414,1506,1718,1779,2828,2354,3279,1370,2595,2751,2921,2368,2976,2733,2277,3136,2013,2952,1427,2682,2349,1655,1390,3151,3130,2002,2654,2101,2358,1597,2585,2900,2687,3165,3031,2798,2530,2344,1384,1420,1474,1509,1520,1556,1607,1647,1810,1927,2000,2028,2056,2120,2136,2371,2384,2444,2465,2479,2563,2598,2652,2716,2731,2746,2755,2772,2775,2792,2794,2917,2926,2927,3110,3221,3395,0666,0291,0043,1743,3299,3434,2379,2660,2711,2855,3444,3354,3111,2736,2345,0021,0037,2350,2201,2674,2642,2734,2690,3004,3091,1670,2689,3002,2362,1714,2071,1399,2940,2581,1357,3439,1569,1591,3439,1507,1531,2297,3379,3277,2912,1930,2878,2363,2984,3012,2663,2139,1457,2197]",
    gpu_directive_format="--gres=gpu:{n}",
    training_launcher="accelerate",
    pretok_qos="boost_qos_dbg",
    pretok_time_limit="00:30:00",
    pretok_partition="boost_usr_prod",
    # this version doesn't work due to RuntimeError: 0 active drivers ([]). There should only be one.
    # errors that come up during imports.... could go deeper but wasn't working immediately
    # pretok_qos="normal",
    # pretok_cpus_per_node=4,
    # pretok_time_limit="4:00:00",
    # pretok_partition="lrd_all_serial",
    # SSH tunnel + proxychains for no-internet compute nodes (like JSC clusters)
    needs_ssh_tunnel=True,
    proxychains_binary="/leonardo_work/AIFAC_5C0_290/bfeuer00/proxychains/bin/proxychains4",
    conda_activate="source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && conda activate otagent",
    # Note: PBS Pro is NOT used here — Leonardo uses SLURM
)

capella = HPC(
    name="capella",
    hostname_pattern=r"c\d",
    dotenv_filename="zih_capella.env",
    account="p_agents_finetuning",
    partition="capella",
    gpus_per_node=4,
    cpus_per_node=32,
    mem_per_node="710GB",  # need this for ZIH since they don't have a default
    internet_node=True,
    gpus_type="H100 94GB",
    total_partition_nodes=146,
    gpu_directive_format="--gpus-per-node={n}",
    # Runtime configuration for Ray/vLLM
    modules=["CUDA/12.8.0"],
    env_vars={
        "PYTHONFAULTHANDLER": "1",
        "NCCL_DEBUG": "INFO",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "CUDA_LAUNCH_BLOCKING": "0",
        "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128",
    },
    # NCCL/networking settings for SFT training (InfiniBand)
    nccl_settings={
        "NCCL_DEBUG": "INFO",
        "NCCL_PROTO": "simple",
        "NCCL_TIMEOUT": "1800",
        "NCCL_IB_TIMEOUT": "23",
        "NCCL_IB_RETRY_CNT": "13",
        "FI_EFA_FORK_SAFE": "1",
        "FI_LOG_LEVEL": "1",
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "NCCL_NET_GDR_LEVEL": "SYS",
        "NCCL_NET_GDR_READ": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    },
    training_launcher="torchrun",
    # Job scaling (from zih_capella.env)
    default_time_limit="47:59:00",
    num_nodes_slow=1,
    num_nodes_default=1,
    num_nodes_fast=4,
    # Exclude flaky nodes: c69,c76,c144 have hanging SLURM prologs; c63,c108 have IB transport errors; c77 has rendezvous errors
    node_exclusion_list="c63,c69,c76,c77,c108,c144",
    # ZIH license requirements
    extra_sbatch_directives=["#SBATCH --licenses=walrus:1,octopus:1,narwhal:1,cat:1"],
)

alpha = HPC(
    name="alpha",
    hostname_pattern=r".*?.alpha.hpc.tu-dresden.de",
    dotenv_filename="zih_capella.env",  # Alpha uses same ZIH env as Capella
    account="p_finetuning",
    partition="alpha",
    gpus_per_node=8,
    cpus_per_node=24,
    mem_per_node="768G",  # need this for ZIH since they don't have a default
    internet_node=True,
    gpus_type="A100 40GB",
    total_partition_nodes=37,
    gpu_directive_format="--gpus-per-node={n}",
    # NCCL/networking settings for SFT training (similar to Capella, ZIH cluster)
    nccl_settings={
        "NCCL_DEBUG": "INFO",
        "NCCL_PROTO": "simple",
        "NCCL_TIMEOUT": "1800",
        "NCCL_IB_TIMEOUT": "23",
        "NCCL_IB_RETRY_CNT": "13",
        "FI_EFA_FORK_SAFE": "1",
        "FI_LOG_LEVEL": "1",
        "FI_EFA_USE_DEVICE_RDMA": "1",
        "NCCL_NET_GDR_LEVEL": "SYS",
        "NCCL_NET_GDR_READ": "1",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
    },
    training_launcher="torchrun",
    # Job scaling (same as Capella, ZIH cluster)
    default_time_limit="47:59:00",
    num_nodes_slow=1,
    num_nodes_default=1,
    num_nodes_fast=4,
    # ZIH license requirements
    extra_sbatch_directives=["#SBATCH --licenses=walrus:1,octopus:1,narwhal:1,cat:1"],
)

dip = HPC(
    name="dip",
    hostname_pattern=r".*dip\.tu-dresden\.de$",
    dotenv_filename="dip.env",
    account="",
    partition="",
    gpus_per_node=0,
    cpus_per_node=16,
    internet_node=True,
    gpus_type="CPU-only",
    total_partition_nodes=1,
    local_mode=True,
)

lrz = HPC(
    name="lrz",
    hostname_pattern=r"lrz.*?",  # Placeholder pattern
    dotenv_filename="lrz.env",
    account="XXXXX",
    partition="mcml-hgx-h100-92x4",
    gpus_per_node=4,
    cpus_per_node=96,
    internet_node=True,
    gpus_type="H100 94GB",
    total_partition_nodes=30,
    gpu_directive_format="--gres=gpu:{n}",
)

vista = HPC(
    name="vista",
    hostname_pattern=r".*?.vista.tacc.utexas.edu",
    dotenv_filename="tacc.env",
    account="CCR24067",
    partition="gh",
    gpus_per_node=1,
    cpus_per_node=72,
    internet_node=True,
    gpus_type="GH200 96GB",
    unified_gpu_memory=True,
    total_partition_nodes=552,
    pretok_time_limit="4:00:00",
    pretok_partition="gh",
    node_exclusion_list="c610-021,c611-011,c640-041,c611-041,c611-122,c637-082",
    # Runtime configuration for Ray/vLLM
    modules=["gcc/15.1.0", "cuda/12.8", "tacc-apptainer"],
    conda_activate="source $SCRATCH/miniconda3/etc/profile.d/conda.sh && conda activate $SCRATCH/miniconda3/envs/vllm_sandboxes",
    env_vars={
        "HF_HOME": "/tmp/hf_home",
        "PYTHONFAULTHANDLER": "1",
        "NCCL_TIMEOUT": "1800",
        "NCCL_IB_TIMEOUT": "23",
        "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128",
    },
    library_paths={
        "TRITON_CC": "/home1/apps/gcc/15.1.0/bin/gcc",
        "LD_PRELOAD": "/home1/apps/gcc/15.1.0/lib64/libstdc++.so.6",
    },
    # NCCL/networking settings for SFT training (EFA networking)
    nccl_settings={
        "NCCL_PROTO": "simple",
        "NCCL_DEBUG": "INFO",
        "FI_EFA_FORK_SAFE": "1",
        "FI_LOG_LEVEL": "1",
        "FI_EFA_ENABLE_SHM_TRANSFER": "0",
        "FI_PROVIDER": "efa",
        "FI_EFA_TX_MIN_CREDITS": "64",
        "NCCL_TREE_THRESHOLD": "0",
        "NCCL_TIMEOUT": "1800",
        "NCCL_IB_TIMEOUT": "23",
    },
    training_launcher="torchrun",
    # Job scaling (from tacc.env)
    default_time_limit="24:00:00",
    max_time_limit="48:00:00",
    num_nodes_slow=4,
    num_nodes_default=16,
    num_nodes_fast=32,
)

lonestar = HPC(
    name="lonestar",
    hostname_pattern=r".*?.ls6.tacc.utexas.edu",
    dotenv_filename="tacc.env",
    account="CCR24067",
    partition="gpu-a100",
    gpus_per_node=3,
    cpus_per_node=128,
    internet_node=True,
    gpus_type="A100 40GB",
    total_partition_nodes=73,
    # Job scaling (from tacc.env)
    default_time_limit="24:00:00",
    max_time_limit="48:00:00",
    num_nodes_slow=4,
    num_nodes_default=16,
    num_nodes_fast=32,
)

claix = HPC(
    name="claix",
    hostname_pattern=r".*?.hpc.itc.rwth-aachen.de",
    dotenv_filename="claix.env",
    account="rwth1775",
    partition="c23g",
    gpus_per_node=4,
    cpus_per_node=96,
    internet_node=True,
    gpus_type="H100 96GB",
    total_partition_nodes=50,
    gpu_directive_format="--gres=gpu:{n}",
)

nyugreene = HPC(
    name="nyugreene",
    hostname_pattern=r"log-\d+\.hpc\.nyu\.edu",
    dotenv_filename="nyugreene.env",
    account="pr_95_tandon_advanced",
    partition="gpu",
    gpus_per_node=4,
    cpus_per_node=24,
    mem_per_node="192G",
    internet_node=True,
    gpus_type="A100/H100 80GB",
    total_partition_nodes=48,
    gpu_directive_format="--gres=gpu:{n}",
    # Job scaling (from nyugreene.env)
    default_time_limit="24:00:00",
    max_time_limit="47:59:00",
    num_nodes_slow=1,
    num_nodes_default=2,
    num_nodes_fast=4,
)

nyutorch = HPC(
    name="nyutorch",
    # hostname_pattern=r"gh\d+\.hpc\.nyu\.edu",
    hostname_pattern=r"torch-login.*\.(hpc\.nyu\.edu|hpc-infra\.svc\.cluster\.local)",
    dotenv_filename="nyutorch.env",
    account="torch_pr_40_tandon_advanced",
    partition="",
    gpus_per_node=8,
    cpus_per_node=96,  # H200 nodes have ~128 cores; request 96 to leave headroom
    mem_per_node="1800G",  # H200 nodes have ~2TB RAM; request less to leave headroom
    internet_node=True,
    gpus_type="H200 141GB / L40S 48GB",
    total_partition_nodes=48,
    gpu_directive_format="--gres=gpu:{type}:{n}",
    default_gpu_type="h200",  # Options: h200, l40s
    # GPU type to constraint mapping for SLURM scheduling
    gpu_type_constraints={
        "_default": "h200",
        "h200": "h200",
        "l40s": "l40s",
    },
    # Runtime configuration for Ray/vLLM (from legacy scripts)
    conda_activate="source $SCRATCH/miniconda3/etc/profile.d/conda.sh && conda activate dcagent312",
    env_vars={
        "PYTHONFAULTHANDLER": "1",
        "NCCL_TIMEOUT": "1800",
        "NCCL_IB_TIMEOUT": "23",
        "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128",
    },
    library_paths={
        "TRITON_CC": "/usr/bin/gcc",
    },
    # Ray memory monitor is broken on Torch — cgroup returns -1, triggering
    # spurious OOM kills during model weight loading.
    unified_gpu_memory=True,
    # Job scaling (from nyutorch.env)
    default_time_limit="24:00:00",
    max_time_limit="47:59:00",
    num_nodes_slow=1,
    num_nodes_default=2,
    num_nodes_fast=4,
)

oumi = HPC(
    name="oumi",
    hostname_pattern=r"oumi-login\d+",
    dotenv_filename="oumi.env",
    account="",
    partition="",
    gpus_per_node=8,
    cpus_per_node=192,
    mem_per_node="1024GB",
    internet_node=True,
    gpus_type="H100 80GB",
    total_partition_nodes=4,
    gpu_directive_format="--gpus-per-node={n}",
    # Job scaling (from oumi.env)
    default_time_limit="168:00:00",
    max_time_limit="168:00:00",
    num_nodes_slow=4,
    num_nodes_default=16,
    num_nodes_fast=32,
)

perlmutter = HPC(
    name="perlmutter",
    hostname_pattern=r"login\d+\.perlmutter\.nersc\.gov",
    dotenv_filename="perlmutter.env",
    account="m5091",
    partition="",
    gpus_per_node=4,
    cpus_per_node=64,
    mem_per_node="",  # Perlmutter doesn't accept explicit memory requests
    internet_node=True,
    gpus_type="A100 80GB",
    total_partition_nodes=256,
    qos="premium",
    gpu_directive_format="--gpus-per-node={n}",
    # GPU type constraints for A100 variants
    # _default (80GB): requires hbm80g constraint
    # A100 40GB: requires gpu constraint only
    gpu_type_constraints={
        "_default": '"gpu&hbm80g"',
        "A100 80GB": '"gpu&hbm80g"',
        "A100 40GB": '"gpu"',
    },
    # Modules to load (CUDA toolkit and native GCC for compilation)
    modules=["cudatoolkit/13.0", "gcc-native/13.2"],
    # Compiler environment variables for flash_attn and other CUDA builds
    env_vars={
        "CC": "gcc",
        "CXX": "g++",
        "CUDAHOSTCXX": "g++",
        # Disable addr2line for vLLM model inspection subprocess (prevents SIGSEGV hangs)
        "TORCH_DISABLE_ADDR2LINE": "1",
    },
    # Library paths for CUDA
    library_paths={
        "LIBRARY_PATH": "${CUDA_HOME}/lib64:${LIBRARY_PATH:-}",
    },
    # NCCL/networking settings for SFT training
    nccl_settings={
        "NCCL_DEBUG": "INFO",
        "NCCL_PROTO": "simple",
        "NCCL_IB_TIMEOUT": "22",
    },
    training_launcher="torchrun",
    needs_cuda_detection=True,  # Complex CUDA SDK path detection
    # Job scaling (from perlmutter.env)
    default_time_limit="168:00:00",
    max_time_limit="168:00:00",
    num_nodes_slow=1,
    num_nodes_default=4,
    num_nodes_fast=8,
)

frontier = HPC(
    name="frontier",
    hostname_pattern=r"login\d+\.frontier\.olcf\.ornl\.gov",
    dotenv_filename="frontier.env",
    account="LRN081",
    partition="batch",
    gpus_per_node=4,
    cpus_per_node=48,
    mem_per_node="",  # Frontier doesn't accept explicit memory requests; uses exclusive nodes
    internet_node=False,
    gpus_type="AMD Instinct MI250X",
    total_partition_nodes=9216,
    qos="normal",
    gpu_directive_format="--gpus-per-node={n}",
    # ROCm modules for AMD MI250X GPUs
    # See: https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html
    # Note: cray-mpich removed - not needed for vLLM/Ray and causes libmpi_cxx.so.40 errors
    # rocm/7.0.2 required for vLLM wheel compatibility
    modules=["PrgEnv-gnu/8.6.0", "gcc-native/14.2", "rocm/7.0.2", "craype-accel-amd-gfx90a"],
    env_vars={
        "ROCM_PATH": "/opt/rocm",
        "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    },
    library_paths={
        "LD_LIBRARY_PATH": "$CONDA_PREFIX/lib:$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}",
    },
    # Unset env vars that ROCm modules set but conflict with Ray
    env_unsets=["ROCR_VISIBLE_DEVICES"],
    # Frontier scheduling bins (node-count-based time limits)
    # Bin 5: 1-91 nodes -> 2 hours max
    # Bin 4: 92-183 nodes -> 6 hours max
    # Bin 3: 184+ nodes -> 12 hours max
    time_limit_by_nodes=[(91, "02:00:00"), (183, "06:00:00"), (9216, "12:00:00")],
    default_time_limit="12:00:00",
    max_time_limit="12:00:00",
    # Node scaling presets for Frontier
    num_nodes_slow=16,
    num_nodes_default=64,
    num_nodes_fast=91,  # Max nodes for Bin 5 (2h limit)
    # Frontier requires exclusive node allocation
    extra_sbatch_directives=["#SBATCH --exclusive"],
    # Frontier/Cray needs --cpu-bind=none for srun commands
    disable_cpu_bind=True,
)

polaris = HPC(
    name="polaris",
    # ALCF Polaris login nodes: polaris-login-01 through polaris-login-04
    hostname_pattern=r"polaris-login-\d+",
    dotenv_filename="polaris.env",
    account="CausalAlign",
    partition="",  # PBS uses queues, not partitions; prod queue auto-routes by node count
    gpus_per_node=4,
    cpus_per_node=64,  # 32 cores x 2 threads
    mem_per_node="512GB",
    internet_node=True,  # Via proxy (proxy.alcf.anl.gov:3128)
    gpus_type="A100 40GB",
    total_partition_nodes=560,
    gpu_directive_format="",  # PBS uses ngpus= in select chunks, not SLURM directives
    env_vars={
        # Disable addr2line for vLLM model inspection subprocess (prevents SIGSEGV hangs)
        "TORCH_DISABLE_ADDR2LINE": "1",
    },
    # Note: Polaris uses PBS Pro, not SLURM. The HPC model is used for env config,
    # eval listener, and datagen — not for sbatch submission. PBS job scripts are
    # separate (eval/polaris/*.pbs).
    default_time_limit="24:00:00",
    max_time_limit="24:00:00",
    num_nodes_slow=10,
    num_nodes_default=24,
    num_nodes_fast=56,
)

clusters = [jureca, jupiter, juwels, leonardo, capella, alpha, dip, lrz, vista, lonestar, claix, nyugreene, nyutorch, oumi, perlmutter, frontier, polaris]


def detect_hpc() -> HPC:
    """Factory function that automatically detects the HPC based on hostname"""
    hostname = socket.gethostname()
    fqdn = socket.getfqdn()
    candidate_hostnames = {hostname, fqdn}

    # Some systems (e.g., NERSC Perlmutter) expose short hostnames but also set an env var.
    nersc_host = os.environ.get("NERSC_HOST", "").strip().lower()
    if nersc_host == "perlmutter":
        candidate_hostnames.add(f"{hostname}.perlmutter.nersc.gov")

    for cluster in clusters:
        pattern = re.compile(cluster.hostname_pattern)
        for candidate in candidate_hostnames:
            if pattern.match(candidate):
                print(f"Automatically detected HPC: {cluster.name}")
                return cluster.model_copy(update={"hostname": candidate})

    raise ValueError(f"HPC not recognized for hostname {hostname}")


def set_environment(hpc_name: HPC) -> None:
    """Set environment variables for the current HPC"""
    dotenv = hpc_name.dotenv_path
    if os.path.exists(dotenv):
        with open(dotenv, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key_part, value_part = line.split("=", 1)
                key = key_part.replace("export ", "").strip()
                value = value_part.strip().strip('"').strip("'")

                os.environ[key] = os.path.expandvars(value)
        print(f"Environment variables set from {dotenv}")

        # Legacy compatibility: treat DC_AGENT as the canonical repo root when DCFT is unset.
        if "DCFT" not in os.environ and os.environ.get("DC_AGENT"):
            os.environ["DCFT"] = os.environ["DC_AGENT"]

        # Capella account is project-specific; respect DCFT_GROUP when available.
        if hpc_name.name.lower() == "capella":
            env_account = os.environ.get("DCFT_GROUP")
            if env_account:
                hpc_name.account = env_account
    else:
        print(
            f"Warning: No dotenv file found for {hpc_name.name} in {dotenv}. Skipping environment variable setup."
        )
