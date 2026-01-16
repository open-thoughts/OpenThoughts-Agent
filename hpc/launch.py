import os
import re
import json
import yaml
import dataclasses
from typing import Any, Optional

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError

from hpc.arguments import JobType, LlamaFactoryArgs, parse_args
from hpc.cli_utils import normalize_job_type
from hpc.launch_utils import (
    _merge_dependencies,
    apply_env_overrides,
    get_job_name,
    launch_sbatch,
    sanitize_repo_component,
    resolve_job_and_paths,
    derive_default_job_name,
    update_exp_args,
)
from hpc.pretokenize_launch_utils import schedule_pretokenize, should_run_pretokenize
from hpc.sft_launch_utils import (
    apply_mca_training_template,
    build_training_parameters_link,
    configure_sft_reporting,
    construct_sft_sbatch_script,
    ensure_deepspeed_config,
    maybe_apply_cluster_specific_env_overrides,
    maybe_compute_gradient_accumulation,
    apply_data_argument_overrides,
    submit_sft_job,
)
from hpc.wandb_launch_utils import collect_wandb_metadata
from hpc.hpc import detect_hpc, set_environment
from hpc.datagen_launch_utils import (
    _prepare_datagen_configuration,
    launch_datagen_job_v2,
)
from hpc.consolidate_launch_utils import (
    launch_consolidate_job,
)
from hpc.eval_launch_utils import (
    launch_eval_job_v2,
    prepare_eval_configuration,
    remap_eval_cli_args,
)
from scripts.harbor.tasks_parquet_converter import from_parquet
from database.unified_db.utils import load_supabase_keys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _extract_agent_name(dataset_name: str) -> str:
    """Extract agent name from dataset name for run_summary.json.

    Tries in order:
    1. Extract from traces-<slug> pattern (e.g., "traces-swebench" -> "swebench")
    2. Extract repo name from HF-style path (e.g., "org/repo-name" -> "repo-name")
    3. Fall back to the full dataset name

    Returns:
        A non-empty agent name string (never None)
    """
    if not dataset_name:
        return "unknown"

    # Try traces-<slug> pattern first
    agent = sanitize_repo_component(dataset_name)
    if agent:
        return agent

    # Extract repo name from HF-style path (org/repo-name -> repo-name)
    if "/" in dataset_name:
        repo_part = dataset_name.split("/")[-1].strip()
        if repo_part:
            return repo_part

    # Fall back to full dataset name
    return dataset_name.strip() or "unknown"


def write_run_summary(exp_args, train_config):
    job_type = normalize_job_type(exp_args)
    if job_type is None or job_type not in (JobType.SFT.value, JobType.SFT_MCA.value, JobType.RL.value):
        return

    output_dir = train_config.get("output_dir") or exp_args.get("output_dir")
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    dataset_name = train_config.get("dataset") or exp_args.get("dataset")
    # Use explicit trace_agent_name if provided, otherwise derive from dataset
    agent_name = exp_args.get("trace_agent_name") or _extract_agent_name(dataset_name)

    hub_model_id = train_config.get("hub_model_id") or exp_args.get("hub_model_id")
    training_parameters_link = build_training_parameters_link(hub_model_id)

    wandb_link, training_start, training_end = collect_wandb_metadata(exp_args, train_config)

    training_type = "SFT" if job_type != JobType.RL.value else "RL"

    summary_payload = {
        "agent_name": agent_name,
        "training_start": training_start,
        "training_end": training_end,
        "created_by": exp_args.get("job_creator", "DCAgent"),
        # Use original model name (e.g., "qwen/qwen3-8B") not resolved HF snapshot path
        "base_model_name": exp_args.get("_original_model_name_or_path") or train_config.get("model_name_or_path") or exp_args.get("model_name_or_path"),
        "dataset_name": dataset_name,
        "training_type": training_type,
        "training_parameters": training_parameters_link,
        "wandb_link": wandb_link,
        "traces_location_s3": None,  # Placeholder until trace uploads record S3 locations
    }

    summary_path = os.path.join(output_dir, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Wrote run summary to {summary_path}")


@dataclasses.dataclass
class _DatasetArtifacts:
    dataset_paths: list[str]
    dataset_path: str
    model_path: str


def _load_base_train_config(train_config_path: str) -> dict:
    with open(train_config_path, "r") as f:
        return yaml.safe_load(f.read())


def _sanitize_model_component(component: str) -> str:
    comp = component.strip().rstrip("/")
    comp = os.path.basename(comp) if comp else component
    comp = re.sub(r"[^A-Za-z0-9]+", "-", comp).strip("-")
    return comp or "model"


def _maybe_include_model_in_job_name(base_config: dict, exp_args: dict) -> dict:
    model_name = base_config.get("model_name_or_path") or exp_args.get("model_name_or_path")
    if not isinstance(model_name, str) or not model_name:
        return exp_args

    model_component = _sanitize_model_component(model_name)
    current_job_name = exp_args.get("job_name", "")
    if current_job_name and model_component.lower() in current_job_name.lower():
        return exp_args

    suggested = f"{current_job_name}_{model_component}" if current_job_name else model_component
    if len(suggested) > 96:
        suggested = suggested[:96]
    print(f"Including model identifier in job name: {current_job_name} -> {suggested}")
    return update_exp_args(exp_args, {"job_name": suggested})


def _drop_deprecated_fields(exp_args: dict, base_config: dict) -> None:
    if exp_args.pop("push_to_db", None) is not None:
        print("Dropping deprecated argument 'push_to_db' from launcher inputs")
    if base_config.pop("push_to_db", None) is not None:
        print("Dropping deprecated argument 'push_to_db' from train config")


def _merge_launch_overrides(base_config: dict, exp_args: dict) -> dict:
    explicit_cli_keys = set(exp_args.get("_explicit_cli_keys", []))
    exp_args.pop("_explicit_cli_keys", None)

    llama_fields = {field.name for field in dataclasses.fields(LlamaFactoryArgs)}
    for key, value in exp_args.items():
        if key.startswith("_"):
            continue
        if key == "deepspeed" and key not in explicit_cli_keys:
            continue
        if key in base_config or key in llama_fields:
            print(f"Setting {key} to {value}")
            base_config[key] = value
    return base_config


def _extract_dataset_entries(dataset_field: Any) -> list[str]:
    if isinstance(dataset_field, str):
        return [item.strip() for item in dataset_field.split(",") if item.strip()]
    if isinstance(dataset_field, (list, tuple)):
        return [str(item).strip() for item in dataset_field if str(item).strip()]
    return []


def _materialize_dataset_and_model(
    base_config: dict,
    exp_args: dict,
    dataset_entries: list[str],
    datasets_dir: str,
) -> _DatasetArtifacts:
    if exp_args.get("job_type") == JobType.PRETOKENIZE.value:
        model_path = exp_args["model_name_or_path"]
        dataset_path = exp_args["dataset"]
        dataset_paths = [item.strip() for item in str(dataset_path).split(",") if item.strip()]
        return _DatasetArtifacts(dataset_paths, str(dataset_path), str(model_path))

    download_datasets = not exp_args.get("internet_node", False)
    dataset_paths: list[str] = []
    if dataset_entries:
        if download_datasets:
            for repo in dataset_entries:
                try:
                    local_path = snapshot_download(repo_id=repo, repo_type="dataset")
                except HFValidationError:
                    if os.path.isdir(repo):
                        local_path = os.path.abspath(repo)
                    else:
                        raise
                dataset_paths.append(local_path)
        else:
            dataset_paths = dataset_entries.copy()
    else:
        raise ValueError("No dataset specified in training configuration.")

    dataset_path = dataset_paths[0]
    if download_datasets:
        print(f"Downloaded dataset to {dataset_path}")

    if exp_args.get("job_type") == JobType.DATAGEN.value and base_config.get("datagen_mode") == "trace":
        parquet_files: list[str] = []
        for root, _, files in os.walk(dataset_path):
            for fname in files:
                if fname.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, fname))
            if parquet_files:
                break
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {dataset_path}")

        parquet_file_path = parquet_files[0]
        print(f"Found parquet file: {parquet_file_path}")

        tasks_base_dir = os.path.join(os.environ.get("DATASETS_DIR", datasets_dir), "tasks_from_parquet")
        os.makedirs(tasks_base_dir, exist_ok=True)
        dataset_name = base_config["dataset"].split("/")[-1]
        tasks_output_dir = os.path.join(tasks_base_dir, dataset_name)

        print(f"Converting parquet to tasks folder at {tasks_output_dir}")
        from_parquet(parquet_file_path, tasks_output_dir, on_exist="skip")
        dataset_path = tasks_output_dir
        print(f"Converted parquet to tasks folder: {dataset_path}")

    model_path = snapshot_download(repo_id=base_config["model_name_or_path"], repo_type="model")
    print(f"Downloaded model to {model_path}")

    return _DatasetArtifacts(dataset_paths, dataset_path, model_path)


def _configure_output_and_logging(base_config: dict, exp_args: dict, checkpoints_dir: str) -> dict:
    raw_output_dir = base_config.get("output_dir")
    if raw_output_dir and checkpoints_dir not in raw_output_dir:
        output_dir = os.path.join(checkpoints_dir, raw_output_dir)
    else:
        output_dir = os.path.join(checkpoints_dir, exp_args["job_name"])
    os.makedirs(output_dir, exist_ok=True)
    base_config["output_dir"] = output_dir

    wandb_dir = os.path.join(exp_args["experiments_dir"], "wandb", exp_args["job_name"])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    if not base_config.get("run_name"):
        base_config["run_name"] = exp_args["job_name"]
    os.environ["WANDB_NAME"] = str(base_config["run_name"]) if base_config.get("run_name") else exp_args["job_name"]
    return base_config

def _maybe_assign_tokenized_path(base_config: dict, exp_args: dict, dataset_entries: list[str]) -> None:
    if base_config.get("tokenized_path") is not None or not should_run_pretokenize(exp_args):
        return

    tokenized_dir = exp_args.get("tokenized_dir")
    tokenized_dir = os.path.expandvars(os.environ.get("TOKENIZED_DATASETS_DIR", tokenized_dir))
    model_name = "_".join(base_config["model_name_or_path"].split("/")[-2:]).replace(".", "-")

    def _slugify(entry: str) -> str:
        entry = entry.strip().rstrip("/")
        if "/" in entry:
            entry = entry.split("/")[-1]
        return entry.replace(".", "-")

    dataset_name_parts = [_slugify(entry) for entry in dataset_entries] or ["dataset"]
    dataset_name = "-".join(dataset_name_parts)
    tokenized_path = os.path.join(tokenized_dir, "_".join([dataset_name, model_name, "tokenized"]))
    base_config["tokenized_path"] = tokenized_path
    exp_args["tokenized_path"] = tokenized_path


def _write_train_config(configs_dir: str, job_name: str, base_config: dict) -> str:
    train_config_path_out = os.path.join(configs_dir, f"{job_name}_train_config.yaml")
    with open(train_config_path_out, "w") as f:
        yaml.dump(base_config, f)
    print(f"Wrote config to {train_config_path_out}")
    return train_config_path_out


def construct_config_yaml(exp_args):
    # Legacy SFT path - auto-derive job_name if not provided
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="SFT",
        derive_job_name_fn=derive_default_job_name,
    )
    configs_dir = str(job_setup.paths.configs)

    train_config_path = exp_args.get("train_config_path")
    checkpoints_dir = exp_args.get("checkpoints_dir")
    models_dir = exp_args.get("models_dir")
    datasets_dir = exp_args.get("datasets_dir")

    datasets_dir = os.path.expandvars(os.environ.get("DATASETS_DIR", datasets_dir))
    models_dir = os.path.expandvars(os.environ.get("MODELS_DIR", models_dir))
    checkpoints_dir = os.path.expandvars(
        os.environ.get("CHECKPOINTS_DIR", checkpoints_dir)
    )

    os.makedirs(checkpoints_dir, exist_ok=True)
    base_config = _load_base_train_config(train_config_path)
    exp_args = _maybe_include_model_in_job_name(base_config, exp_args)
    _drop_deprecated_fields(exp_args, base_config)
    base_config = _merge_launch_overrides(base_config, exp_args)
    base_config = ensure_deepspeed_config(base_config, exp_args)

    if base_config.get("dataset_dir") is None:
        base_config["dataset_dir"] = "ONLINE"

    dataset_entries = _extract_dataset_entries(base_config.get("dataset"))

    # Preserve original model name before HF resolution (for database registration)
    original_model_name = base_config.get("model_name_or_path")
    exp_args["_original_model_name_or_path"] = original_model_name

    artifacts = _materialize_dataset_and_model(base_config, exp_args, dataset_entries, datasets_dir)

    hub_model_id = base_config.get("hub_model_id")
    if hub_model_id is not None:
        hub_model_id = hub_model_id.replace(".", "_")
    else:
        hub_model_id = f"mlfoundations-dev/{exp_args['job_name']}"
    base_config["hub_model_id"] = hub_model_id

    if exp_args.get("job_type") == JobType.DATAGEN.value and base_config.get("datagen_mode") == "trace":
        base_config["dataset"] = artifacts.dataset_path
        base_config["dataset_dir"] = artifacts.dataset_path
    elif not exp_args["internet_node"]:
        if artifacts.dataset_paths:
            base_config["dataset"] = ",".join(artifacts.dataset_paths)

    base_config = configure_sft_reporting(base_config, exp_args, artifacts.model_path)
    base_config = _configure_output_and_logging(base_config, exp_args, checkpoints_dir)
    base_config = maybe_compute_gradient_accumulation(base_config, exp_args)
    _maybe_assign_tokenized_path(base_config, exp_args, dataset_entries)
    apply_data_argument_overrides(base_config, exp_args)

    train_config_path_out = _write_train_config(configs_dir, exp_args["job_name"], base_config)
    exp_args["output_dir"] = base_config["output_dir"]
    exp_args["dataset"] = base_config["dataset"]
    exp_args["model_name_or_path"] = base_config["model_name_or_path"]
    exp_args["hub_model_id"] = base_config.get("hub_model_id", None)
    return base_config, train_config_path_out

def submit_job(
    exp_args=None,
    dependency=None,
):
    # Legacy SFT path - auto-derive job_name if not provided
    job_setup = resolve_job_and_paths(
        exp_args or {},
        job_type_label="SFT",
        derive_job_name_fn=derive_default_job_name,
    )
    exp_args["logs_dir"] = str(job_setup.paths.logs)

    base_dependency = _merge_dependencies(exp_args.get("dependency"), dependency)
    current_dependency = base_dependency

    job_id = None
    if exp_args.get("max_restarts") is not None:
        max_restarts = int(exp_args["max_restarts"])
        if max_restarts > 0:
            for _ in range(max_restarts):
                job_id = launch_sbatch(
                    exp_args["train_sbatch_path_out"], dependency=current_dependency
                )
                job_id = job_id.split()[-1]
                current_dependency = f"afternotok:{job_id}"

    job_id = launch_sbatch(
        exp_args["train_sbatch_path_out"], current_dependency
    )
    job_id = job_id.split()[-1]
    print(f"Writing logs to {exp_args['logs_dir']}/{exp_args['job_name']}_{job_id}.out")
    return job_id

def display_args(exp_args, name):
    print()
    print("=" * 20 + f" {name} Args " + "=" * 20)
    for key, value in exp_args.items():
        print(f"{key}: {value}")
    print()

def main():
    load_supabase_keys()
    # this is where defaults are stored for experiments_dir and deepspeed
    cli_args = parse_args()

    # Apply job-type-specific argument remapping
    job_type_raw = cli_args.get("job_type", "").lower()
    if job_type_raw == "eval":
        cli_args = remap_eval_cli_args(cli_args)

    # Storing all the arguments in a dictionary that we add to in order of precedence
    exp_args = dict()

    # Add arguments to experiment from automatically detecting HPC
    hpc = detect_hpc()
    set_environment(hpc)

    # Add arguments and validate
    exp_args = update_exp_args(exp_args, hpc.model_dump())
    explicit_cli_keys = set(cli_args.get("_explicit_cli_keys", []))
    cli_args_filtered = {k: v for k, v in cli_args.items() if k != "_explicit_cli_keys"}
    exp_args = update_exp_args(exp_args, cli_args_filtered, explicit_keys=explicit_cli_keys)
    if explicit_cli_keys:
        exp_args["_explicit_cli_keys"] = list(explicit_cli_keys)

    exp_args, job_type, _ = apply_env_overrides(
        exp_args,
        cli_args_filtered,
        hpc,
        apply_mca_template_fn=apply_mca_training_template,
        apply_cluster_overrides_fn=maybe_apply_cluster_specific_env_overrides,
        prepare_datagen_fn=_prepare_datagen_configuration,
        prepare_eval_fn=prepare_eval_configuration,
    )

    # Job name
    if "job_name" not in exp_args:
        exp_args["job_name"] = get_job_name(cli_args)
    print(f"Job name: {exp_args['job_name']}")

    # Experiments directory - set default if not specified
    # This must be set early as many functions depend on it
    if not exp_args.get("experiments_dir"):
        exp_args["experiments_dir"] = os.path.join("experiments", exp_args["job_name"])

    if job_type == JobType.CONSOLIDATE.value:
        launch_consolidate_job(
            exp_args,
            hpc,
            update_exp_args_fn=update_exp_args,
            launch_sbatch_fn=launch_sbatch,
        )
        return

    # Check if this is a data generation job
    if job_type == JobType.DATAGEN.value:
        launch_datagen_job_v2(exp_args, hpc)
        return  # Skip normal training flow

    if job_type == JobType.EVAL.value:
        launch_eval_job_v2(exp_args, hpc)
        return

    if job_type == JobType.PRETOKENIZE.value:
        schedule_pretokenize(
            exp_args,
            update_exp_args_fn=update_exp_args,
            construct_config_yaml_fn=construct_config_yaml,
            construct_sbatch_script_fn=lambda args: construct_sft_sbatch_script(args, hpc),
            submit_job_fn=submit_job,
        )
        return

    if job_type in (JobType.SFT.value, JobType.SFT_MCA.value):
        submit_sft_job(
            exp_args,
            cli_args,
            hpc,
            construct_config_yaml_fn=construct_config_yaml,
            update_exp_args_fn=update_exp_args,
            write_run_summary_fn=write_run_summary,
            display_args_fn=display_args,
            submit_job_fn=submit_job,
            should_run_pretokenize_fn=should_run_pretokenize,
            schedule_pretokenize_fn=schedule_pretokenize,
        )
        return

    if job_type == JobType.RL.value:
        from hpc.rl_launch_utils import launch_rl_job
        launch_rl_job(exp_args, hpc)
        return

    # If we reach here, the job type is not implemented or invalid
    raise NotImplementedError(
        f"Job type '{job_type}' is not yet implemented or is invalid. "
        f"Supported job types: {', '.join(jt.value for jt in JobType)}"
    )

if __name__ == "__main__":
    main()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
