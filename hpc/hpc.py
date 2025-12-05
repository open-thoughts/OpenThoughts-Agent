import math
import os
import re
import socket
from pydantic import BaseModel, computed_field


class HPC(BaseModel):
    """Base pydantic model for HPC clusters"""

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
    qos: str = "normal"
    pretok_qos: str = "normal"
    pretok_cpus_per_node: int = 0 # will use all available cpus
    pretok_time_limit: str = "24:00:00"
    pretok_partition: str = ""
    pretok_gpus_per_node: int = 0 # will ask for 0 gpus

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


jureca = HPC(
    name="jureca",
    hostname_pattern=r"jr.*?.jureca",
    train_sbatch_filename="jsc_train.sbatch",
    dotenv_filename="jureca.env",
    account="westai0007", # synthlaion (24 nodes per job)
    partition="dc-hwai", # dc-gpu
    gpus_per_node=4,
    cpus_per_node=48,
    internet_node=False,
    gpus_type="H100 94GB",
    total_partition_nodes=16,
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
    dotenv_filename="zih.env",
    account="p_finetuning",
    partition="capella",
    gpus_per_node=4,
    cpus_per_node=32,
    mem_per_node="710GB",  # need this for ZIH since they don't have a default
    internet_node=True,
    gpus_type="H100 94GB",
    total_partition_nodes=146,
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
)

nyutorch = HPC(
    name="nyutorch",
    # hostname_pattern=r"gh\d+\.hpc\.nyu\.edu",
    hostname_pattern = r"torch-login.*\.hpc\.nyu\.edu",
    train_sbatch_filename="nyutorch_train.sbatch",
    dotenv_filename="nyutorch.env",
    account="",
    partition="",
    gpus_per_node=8,
    cpus_per_node=24,
    mem_per_node="192G",
    internet_node=True,
    gpus_type="H200 141GB",
    total_partition_nodes=48,
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
)

clusters = [jureca, jupiter, juwels, leonardo, capella, alpha, lrz, vista, lonestar, claix, nyugreene, nyutorch, oumi, perlmutter]


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
    else:
        print(
            f"Warning: No dotenv file found for {hpc_name.name} in {dotenv}. Skipping environment variable setup."
        )
