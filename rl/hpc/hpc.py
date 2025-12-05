import math
import os
import re
import socket

from pydantic import BaseModel, computed_field


class HPC(BaseModel):
    """Base model for HPC clusters"""
    
    # Define all fields as class attributes with default values
    name: str = ""
    hostname: str = ""
    hostname_pattern: str = ""
    dotenv_filename: str = ""
    account: str = ""
    partition: str = ""
    gpus_per_node: int = 1
    cpus_per_node: int = 1
    cpus_per_gpu: int | None = None
    mem_per_node: str = ""
    internet_node: bool = True
    gpus_type: str = ""
    total_partition_nodes: int = 1
    train_sbatch_jinja_filename: str = ""
    node_exclusion_list: str = ""
    qos: str = "normal"
    pretok_qos: str = "normal"
    pretok_cpus_per_node: int = 0
    pretok_time_limit: str = "24:00:00"
    pretok_partition: str = ""
    pretok_gpus_per_node: int = 0

    def model_post_init(self, __context) -> None:
        if not self.cpus_per_gpu:
            gpus = max(self.gpus_per_node, 1)
            if self.cpus_per_node:
                self.cpus_per_gpu = math.ceil(self.cpus_per_node / gpus)

    @computed_field
    @property
    def train_sbatch_jinja_path(self) -> str:
        # All sbatch files should be in the "dc-agent/rl/hpc/sbatch" directory
        hpc_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(hpc_dir, "sbatch", self.train_sbatch_jinja_filename)

    @computed_field
    @property
    def dotenv_path(self) -> str:
        hpc_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(hpc_dir, "dotenv", self.dotenv_filename)


# JSC (Jülich Supercomputing Centre) configurations
jsc_jureca = HPC(
    name="jsc_jureca",
    hostname_pattern=r"jr.*?.jureca",
    train_sbatch_jinja_filename="jsc_train.j2",
    dotenv_filename="jsc.env",
    account="westai0007", # synthlaion (24 nodes per job)
    partition="dc-hwai", # dc-gpu
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="H100 94GB",
    total_partition_nodes=16,
)

jsc_jupiter = HPC(
    name="jsc_jupiter",
    hostname_pattern=r"j.*?.jupiter.internal",
    train_sbatch_jinja_filename="jsc_train.j2",
    dotenv_filename="jsc.env",
    account="jureap1",
    partition="all",
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="GH200 96GB",
    total_partition_nodes=48,
)

jsc_juwels = HPC(
    name="jsc_juwels",
    hostname_pattern=r"jw.*?.juwels",
    train_sbatch_jinja_filename="jsc_train.j2",
    dotenv_filename="jsc.env",
    account="laionize",
    partition="booster",
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="A100 40GB",
    total_partition_nodes=936,
    node_exclusion_list="jwb[0059,0067,0069,0193,0198,0215,0266,0284,0287,0294,0359,0418,0637,0647,0829,0832,0838,0898,0907,0921,0971,1004,1023,1029,1213]",
)

# TACC (Texas Advanced Computing Center) configurations
tacc_vista = HPC(
    name="tacc_vista",
    hostname_pattern=r".*?.vista.tacc.utexas.edu",
    train_sbatch_jinja_filename="tacc_train.j2",
    dotenv_filename="tacc.env",
    account="CCR24067",
    partition="gh",
    gpus_per_node=1,
    cpus_per_node=72,
    internet_node=True,
    gpus_type="GH200 96GB",
    total_partition_nodes=552,
    qos="",
    pretok_time_limit="4:00:00",
    pretok_partition="gh",
)

tacc_lonestar = HPC(
    name="tacc_lonestar",
    hostname_pattern=r".*?.ls6.tacc.utexas.edu",
    train_sbatch_jinja_filename="tacc_train.j2",
    dotenv_filename="tacc.env",
    account="CCR24067",
    partition="gpu-a100",
    gpus_per_node=3,
    cpus_per_node=128,
    internet_node=True,
    gpus_type="A100 40GB",
    total_partition_nodes=73,
    qos="",
)

# List of all available clusters
clusters = [jsc_jureca, jsc_jupiter, jsc_juwels, tacc_vista, tacc_lonestar]


def detect_hpc() -> HPC:
    """Factory function that automatically detects the HPC based on hostname"""
    hostname = socket.gethostname()
    for cluster in clusters:
        if re.compile(cluster.hostname_pattern).match(hostname):
            print(f"Automatically detected HPC: {cluster.name}")
            return cluster.model_copy(update={"hostname": hostname})

    raise ValueError(f"HPC not recognized for hostname {hostname}")


def set_environment(hpc_name: HPC) -> None:
    """Set environment variables for the current HPC"""
    dotenv = hpc_name.dotenv_path
    if os.path.exists(dotenv):
        with open(dotenv, "r") as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split("=", 1)
                    key = key.replace("export ", "")
                    os.environ[key] = os.path.expandvars(
                        value.strip().replace('"', "").replace("'", "")
                    )
        print(f"Environment variables set from {dotenv}")
    else:
        print(
            f"Warning: No dotenv file found for {hpc_name.name} in {dotenv}. Skipping environment variable setup."
        )
