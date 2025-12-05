import dataclasses
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import argparse
import os
from typing import get_origin, get_args, Union


@dataclass
class LaunchArgs:
    """Arguments for job launching"""

    # ----------------------------------
    #  Core launch arguments
    # ----------------------------------
    job_name: Optional[str] = field(
        default=None, 
        metadata={"help": "Job name. This will determine outputs (logs, checkpoints, etc.)."}
    )
    experiments_dir: Optional[str] = field(
        # TODO(Charlie): should we reuse EXPORT_PATH instead so everything is in the same place?
        default=os.getenv("EXPERIMENTS_DIR", "experiments"),
        metadata={
            "help": "Output directory for storing the generated sbatch scripts and the SLURM `.out` logs."
        },
    )
    time_limit: Optional[str] = field(
        default=None, 
        metadata={"help": "Time limit for the job"}
    )
    max_restarts: Optional[int] = field(
        default=None, 
        metadata={"help": "Maximum number of job restarts"}
    )
    num_nodes: Optional[int] = field(
        default=None, 
        metadata={"help": "Number of nodes to use"}
    )

    # Dry run
    dry_run: bool = field(
        default=False,
        metadata={
            "help": "When present, the job will not be submitted",
            "store_true": True,
        },
    )

    # --------------------------------------------
    # Watchdog and Hugging Face upload arguments
    # --------------------------------------------
    final_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name or path of the model (used for export and checkpoint paths)"}
    )
    enable_hf_upload: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable automatic upload to Hugging Face Hub after job completion",
            "store_true": True,
        },
    )
    hf_repo_name: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face repository name (default: DCAgent/{final_model_name})"}
    )
    hf_token: Optional[str] = field(
        default=os.getenv("HF_TOKEN"),
        metadata={"help": "Hugging Face token for authentication"}
    )
    watchdog_check_interval: Optional[int] = field(
        default=300,
        metadata={"help": "Check interval in seconds for job status monitoring (default: 300)"}
    )

    # --------------------------------------------------------------
    # SkyRL specific args.
    # These arguments are post-processed in `launch.py` and then passed to the SkyRL via
    # `inplace_update_skyrl_args()`. All the other SkyRL args should be passed via `-S key=value`
    # flags where we directly relay them to SkyRL.
    # --------------------------------------------------------------

    # Corresponds to `data.train_data` and `data.val_data` in SkyRL.
    # To specify multiple datasets, do:
    # --train_data <dataset1> <dataset2> ... \
    # --val_data <dataset3> <dataset4> ... \
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of training datasets, separated by spaces"},
    )
    val_data: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of validation datasets, separated by spaces"},
    )

    # The policy model (i.e. the base model to start the training from).
    # Corresponds to `trainer.policy.model.path` in SkyRL.
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model for training (policy model)"}
    )

    # The entrypoint for the SkyRL job. This is not an argument of SkyRL, but rather the script to run
    # We will do python -m <skyrl_entrypoint> <skyrl_args> in the generated sbatch script.
    skyrl_entrypoint: str = field(
        default=None,
        metadata={"help": "Entrypoint for the SkyRL job. E.g. `skyrl_train.entrypoints.main_base`"}
    )


def _add_dataclass_arguments(arg_group, dataclass_type, exclude_fields=None):
    """
    Helper function to add arguments from a dataclass to an argument group.

    Args:
        arg_group: The argument group to add arguments to
        dataclass_type: The dataclass type to extract fields from
        exclude_fields: Optional list of field names to exclude
    """
    exclude_fields = exclude_fields or []

    def _is_list_of_str(annotation) -> bool:
        """
        Returns True when the provided type annotation represents a list of strings,
        including Optional[List[str]] and similar unions (e.g. `train_data` and `val_data`)
        """
        origin = get_origin(annotation)
        # Direct list or typing.List
        if origin in (list, List):
            args = get_args(annotation)
            return len(args) == 0 or args[0] is str
        # Optional/Union types
        if origin is Union:
            return any(
                _is_list_of_str(arg) for arg in get_args(annotation) if arg is not type(None)
            )
        return False

    def str_to_bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    for field in dataclasses.fields(dataclass_type):
        if field.name in exclude_fields:
            continue

        if field.metadata.get("store_true"):
            arg_group.add_argument(
                f"--{field.name}",
                action="store_true",
                help=field.metadata.get("help"),
                default=field.default,
            )
        elif _is_list_of_str(field.type):
            # Accept one or more string values for list-of-string types
            # e.g. --train_data <dataset1> <dataset2> ...
            arg_group.add_argument(
                f"--{field.name}",
                nargs="+",
                type=str,
                help=field.metadata.get("help"),
                default=field.default,
            )
        elif field.type == bool or (field.type is Optional[bool] and field.default is not None):
            arg_group.add_argument(
                f"--{field.name}",
                type=str_to_bool,
                help=field.metadata.get("help"),
                default=field.default,
            )
        else:
            arg_group.add_argument(
                f"--{field.name}",
                type=type(field.default) if field.default is not None else str,
                help=field.metadata.get("help"),
                default=field.default,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Launch HPC jobs for dc-agent experiment")

    # Create argument groups for better organization
    launch_group = parser.add_argument_group("Launch Arguments")
    hpc_group = parser.add_argument_group("HPC Arguments")
    skyrl_group = parser.add_argument_group("SkyRL Arguments")

    # Add LaunchArgs arguments
    _add_dataclass_arguments(launch_group, LaunchArgs)
    # Add HPC arguments
    hpc_fields = [
        "name",
        "account",
        "partition",
        "gpus_per_node",
        "cpus_per_node",
        "cpus_per_gpu",
        "gpus_type",
        "total_partition_nodes",
        "qos",
    ]
    for field in hpc_fields:
        hpc_group.add_argument(
            f"--{field}",
            type=(
                str
                if field == "name"
                or field == "account"
                or field == "partition"
                or field == "gpus_type"
                or field == "qos"
                else int
            ),
            help=f"HPC {field}",
        )

    # Add SkyRL override flags: (those set with -S key=value)
    skyrl_group.add_argument(
        "-S",
        "--set",
        dest="skyrl_set",
        action="append",
        default=[],
        help="SkyRL arguments in key=value format. Repeat for multiple. Example: -S trainer.epochs=1 -S trainer.train_batch_size=64",
    )

    args = parser.parse_args()

    # Validate required arguments
    required_args = ['job_name', 'train_data', 'val_data', 'model_path', 'skyrl_entrypoint']
    for arg_name in required_args:
        if getattr(args, arg_name) is None:
            parser.error(f"the following argument is required: --{arg_name}")

    args_dict = {k: v for k, v in vars(args).items() if v is not None}

    skyrl_args: Dict[str, Any] = {}
    for item in args_dict.pop("skyrl_set", []):
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"Invalid --set entry (expected key=value): {item}")
        key, value = item.split("=", 1)
        skyrl_args[key.strip()] = value.strip()
    args_dict["skyrl_args"] = skyrl_args

    return args_dict
