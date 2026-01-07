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
    train_sbatch_filename: str
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

    def model_post_init(self, __context) -> None:
        # Derive a default CPU-per-GPU ratio when not explicitly provided.
        if not self.cpus_per_gpu:
            gpus = max(self.gpus_per_node, 1)
            if self.cpus_per_node:
                self.cpus_per_gpu = math.ceil(self.cpus_per_node / gpus)

    @computed_field
    def train_sbatch_path(self) -> str:
        # All sbatch files should be in the "dcft/hpc/sbatch" directory
        hpc_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(hpc_dir, "sbatch", self.train_sbatch_filename)

    @computed_field
    def dotenv_path(self) -> str:
        hpc_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(hpc_dir, "dotenv", self.dotenv_filename)

    # =========================================================================
    # Runtime configuration methods for SBATCH/Ray/vLLM
    # =========================================================================

    def get_module_commands(self) -> str:
        """Generate module load commands for SBATCH scripts."""
        if not self.modules:
            return ""
        lines = ["set +u"]
        lines.extend(f"module load {m}" for m in self.modules)
        lines.append("set -u")
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
            lines.append(f"#SBATCH -q {qos}")
        gpu_directive = self.get_gpu_directive(gpus, gpu_type)
        if gpu_directive:
            lines.append(gpu_directive)
        mem_directive = self.get_mem_directive(mem)
        if mem_directive:
            lines.append(mem_directive)
        if self.node_exclusion_list:
            lines.append(f"#SBATCH --exclude={self.node_exclusion_list}")
        return "\n".join(lines)

    def get_srun_export_env(self) -> str:
        """Generate SRUN --export string with all necessary env vars."""
        env_parts = ["ALL"]
        for key, value in {**self.env_vars, **self.library_paths}.items():
            env_parts.append(f"{key}={value}")
        # Add common paths
        env_parts.append("PATH=$PATH")
        env_parts.append("LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}")
        env_parts.append("PYTHONPATH=${PYTHONPATH:-}")
        env_parts.append("HF_HOME=${HF_HOME:-}")
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
        return " ".join(env_parts)


jureca = HPC(
    name="jureca",
    hostname_pattern=r"jr.*?.jureca",
    train_sbatch_filename="jsc_train.sbatch",
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
    },
)

jupiter = HPC(
    name="jupiter",
    hostname_pattern=r"j.*?.jupiter.internal",
    train_sbatch_filename="jsc_train.sbatch",
    dotenv_filename="jupiter.env",
    account="jureap1",
    partition="all",
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="GH200 96GB",
    total_partition_nodes=48,
    gpu_directive_format="--gres=gpu:{n}",
)

juwels = HPC(
    name="juwels",
    hostname_pattern=r"jw.*?.juwels",
    train_sbatch_filename="jsc_train.sbatch",
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
)

leonardo = HPC(
    name="leonardo",
    hostname_pattern=r".*?.leonardo.local",
    train_sbatch_filename="leonardo_train.sbatch",
    dotenv_filename="leonardo.env",
    account="EUHPC_E03_068",
    partition="boost_usr_prod",
    gpus_per_node=4,
    cpus_per_node=32,
    internet_node=False,
    gpus_type="A100 64GB",
    total_partition_nodes=3456,
    node_exclusion_list="lrdn[1606,2776,2425,2808,3064,3064,1953,2414,1506,1718,1779,2828,2354,3279,1370,2595,2751,2921,2368,2976,2733,2277,3136,2013,2952,1427,2682,2349,1655,1390,3151,3130,2002,2654,2101,2358,1597,2585,2900,2687,3165,3031,2798,2530,2344,1384,1420,1474,1509,1520,1556,1607,1647,1810,1927,2000,2028,2056,2120,2136,2371,2384,2444,2465,2479,2563,2598,2652,2716,2731,2746,2755,2772,2775,2792,2794,2917,2926,2927,3110,3221,3395,0666,0291,0043,1743,3299,3434,2379,2660,2711,2855,3444,3354,3111,2736,2345,0021,0037,2350,2201,2674,2642,2734,2690,3004,3091,1670,2689,3002,2362,1714,2071,1399,2940,2581,1357,3439,1569,1591,3439,1507,1531,2297,3379,3277,2912,1930,2878,2363,2984,3012,2663,2139,1457,2197]",
    gpu_directive_format="--gres=gpu:{n}",
    pretok_qos="boost_qos_dbg",
    pretok_time_limit="00:30:00",
    pretok_partition="boost_usr_prod",
    # this version doesn't work due to RuntimeError: 0 active drivers ([]). There should only be one.
    # errors that come up during imports.... could go deeper but wasn't working immediately
    # pretok_qos="normal",
    # pretok_cpus_per_node=4,
    # pretok_time_limit="4:00:00",
    # pretok_partition="lrd_all_serial",
)

capella = HPC(
    name="capella",
    hostname_pattern=r"c\d",
    train_sbatch_filename="zih_capella_train.sbatch",
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
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
        "RAY_NOSET_CUDA_VISIBLE_DEVICES": "1",
    },
)

alpha = HPC(
    name="alpha",
    hostname_pattern=r".*?.alpha.hpc.tu-dresden.de",
    train_sbatch_filename="alpha_train.sbatch",
    dotenv_filename="alpha.env",
    account="p_finetuning",
    partition="alpha",
    gpus_per_node=8,
    cpus_per_node=24,
    mem_per_node="768G",  # need this for ZIH since they don't have a default
    internet_node=True,
    gpus_type="A100 40GB",
    total_partition_nodes=37,
    gpu_directive_format="--gpus-per-node={n}",
)

dip = HPC(
    name="dip",
    hostname_pattern=r".*dip\.tu-dresden\.de$",
    train_sbatch_filename="local_stub.sbatch",
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
    train_sbatch_filename="lrz_train.sbatch",
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
    train_sbatch_filename="vista_train.sbatch",
    dotenv_filename="tacc.env",
    account="CCR24067",
    partition="gh",
    gpus_per_node=1,
    cpus_per_node=72,
    internet_node=True,
    gpus_type="GH200 96GB",
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
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128",
    },
    library_paths={
        "TRITON_CC": "/home1/apps/gcc/15.1.0/bin/gcc",
        "LD_PRELOAD": "/home1/apps/gcc/15.1.0/lib64/libstdc++.so.6",
    },
)

lonestar = HPC(
    name="lonestar",
    hostname_pattern=r".*?.ls6.tacc.utexas.edu",
    train_sbatch_filename="lonestar_train.sbatch",
    dotenv_filename="tacc.env",
    account="CCR24067",
    partition="gpu-a100",
    gpus_per_node=3,
    cpus_per_node=128,
    internet_node=True,
    gpus_type="A100 40GB",
    total_partition_nodes=73,
)

claix = HPC(
    name="claix",
    hostname_pattern=r".*?.hpc.itc.rwth-aachen.de",
    train_sbatch_filename="claix_train.sbatch",
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
    train_sbatch_filename="nyugreene_train.sbatch",
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
)

nyutorch = HPC(
    name="nyutorch",
    # hostname_pattern=r"gh\d+\.hpc\.nyu\.edu",
    hostname_pattern=r"torch-login.*\.hpc\.nyu\.edu",
    train_sbatch_filename="nyutorch_train.sbatch",
    dotenv_filename="nyutorch.env",
    account="torch_pr_40_tandon_advanced",
    partition="",
    gpus_per_node=8,
    cpus_per_node=24,
    mem_per_node="192G",
    internet_node=True,
    gpus_type="H200 141GB / L40S 48GB",
    total_partition_nodes=48,
    gpu_directive_format="--gres=gpu:{type}:{n}",
    default_gpu_type="h200",  # Options: h200, l40s
    # Runtime configuration for Ray/vLLM (from legacy scripts)
    conda_activate="source $SCRATCH/miniconda3/etc/profile.d/conda.sh && conda activate dcagent312",
    env_vars={
        "PYTHONFAULTHANDLER": "1",
        "NCCL_TIMEOUT": "1800",
        "NCCL_IB_TIMEOUT": "23",
        "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.6,max_split_size_mb:128",
    },
    library_paths={
        "TRITON_CC": "/usr/bin/gcc",
    },
)

oumi = HPC(
    name="oumi",
    hostname_pattern=r"oumi-login\d+",
    train_sbatch_filename="oumi_train.sbatch",
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
)

perlmutter = HPC(
    name="perlmutter",
    hostname_pattern=r"login\d+\.perlmutter\.nersc\.gov",
    train_sbatch_filename="perlmutter_train.sbatch",
    dotenv_filename="perlmutter.env",
    account="m5091",
    partition="",
    gpus_per_node=4,
    cpus_per_node=64,
    mem_per_node="256GB",
    internet_node=True,
    gpus_type="A100 80GB",
    total_partition_nodes=256,
    qos="regular",
    gpu_directive_format="--gpus-per-node={n}",
)

frontier = HPC(
    name="frontier",
    hostname_pattern=r"login\d+\.frontier\.olcf\.ornl\.gov",
    train_sbatch_filename="frontier_train.sbatch",
    dotenv_filename="frontier.env",
    account="LRN081",
    partition="batch",
    gpus_per_node=4,
    cpus_per_node=48,
    mem_per_node="512GB",
    internet_node=False,
    gpus_type="AMD Instinct MI250X",
    total_partition_nodes=9216,
    qos="normal",
    gpu_directive_format="--gpus-per-node={n}",
)

clusters = [jureca, jupiter, juwels, leonardo, capella, alpha, dip, lrz, vista, lonestar, claix, nyugreene, nyutorch, oumi, perlmutter, frontier]


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
