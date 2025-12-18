import math
import os
import re
import json
import yaml
import subprocess
import dataclasses
import shlex
import importlib.util
import shutil
import textwrap
from pathlib import Path
from typing import Any, Optional, Union

from collections import defaultdict

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
import wandb

from datasets import load_dataset

from hpc.arguments import LlamaFactoryArgs, parse_args
from hpc.launch_utils import (
    get_job_name,
    sanitize_repo_component,
    sanitize_repo_for_job,
)
from hpc.hpc import detect_hpc, set_environment
from harbor.models.environment_type import EnvironmentType
from hpc.datagen_launch_utils import (
    TraceChunkPlan,
    _build_vllm_env_vars,
    _discover_task_entries,
    _format_chunk_target_repo,
    _maybe_set_ray_cgraph_env,
    _prepare_datagen_configuration,
    _prepare_trace_chunk_plans,
    _snapshot_datagen_config,
    default_vllm_endpoint_path,
    resolve_harbor_config_path,
)
from hpc.consolidate_launch_utils import (
    launch_consolidate_job,
)
from data.generation import BaseDataGenerator
from scripts.harbor.tasks_parquet_converter import from_parquet
from database.unified_db.utils import load_supabase_keys
from scripts.harbor.job_config_utils import (
    dump_job_config,
    load_job_config,
    set_job_metadata,
    set_local_dataset,
    overwrite_agent_fields,
    update_agent_kwargs,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HARBOR_MODEL_PLACEHOLDER = "placeholder/override-at-runtime"


def _detect_gpu_required(datagen_script: str) -> bool:
    """Best-effort detection of GPU requirement for a datagen script."""

    try:
        script_path = os.path.abspath(datagen_script)
        if not os.path.exists(script_path):
            return False

        spec = importlib.util.spec_from_file_location("datagen_module", script_path)
        if spec is None or spec.loader is None:
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[attr-defined]

        generator_cls = None
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseDataGenerator)
                and obj is not BaseDataGenerator
            ):
                generator_cls = obj
                break

        if not generator_cls:
            return False

        generator = generator_cls()
        run_fn = getattr(generator, "run_task_generation", None)
        return bool(getattr(run_fn, "_gpu_required", False))
    except Exception:
        return False


def _validate_sbatch_templates(hpc_obj) -> None:
    import json
    from pathlib import Path

    req_path = Path(__file__).parent / "sbatch_data_requirements.json"
    if not req_path.exists():
        return

    data = json.loads(req_path.read_text())
    entries = data.get(hpc_obj.name.lower())
    if not entries:
        raise ValueError(
            f"No sbatch templates registered for cluster '{hpc_obj.name}'. "
            "Please add entries to hpc/sbatch_data_requirements.json."
        )

    missing = [entry["path"] for entry in entries if not Path(entry["path"]).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing sbatch templates for datagen: " + ", ".join(missing)
        )


def _inject_env_block(text: str, env_map: dict) -> str:
    exports = []
    for k, v in env_map.items():
        if v in (None, ""):
            continue
        quoted = shlex.quote(str(v))
        exports.append(f"export {k}={quoted}")
    if not exports:
        return text
    lines = text.splitlines(True)
    idx = 0
    if lines and lines[0].startswith("#!"):
        idx = 1
    while idx < len(lines) and (
        lines[idx].startswith("#SBATCH")
        or lines[idx].strip() == ""
        or lines[idx].startswith("#")
    ):
        idx += 1
    return "".join(lines[:idx] + ["\n".join(exports) + "\n"] + lines[idx:])


def _ensure_dependency_directive(text: str, dependency: Optional[str]) -> str:
    if not dependency:
        return text

    directive_prefix = "#SBATCH --dependency"
    lines = text.splitlines()
    for line in lines:
        if directive_prefix in line:
            return text

    insert_idx = 0
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("#!"):
            insert_idx = 1
            continue
        stripped = line.strip()
        if stripped.startswith("#SBATCH"):
            insert_idx = idx + 1
            continue
        if not stripped:
            insert_idx = idx + 1
            continue
        break

    dependency_line = f"#SBATCH --dependency={dependency}"
    lines.insert(insert_idx, dependency_line)
    new_text = "\n".join(lines)
    if text.endswith("\n"):
        new_text += "\n"
    return new_text


def _merge_dependencies(*deps: Optional[str]) -> Optional[str]:
    merged: list[str] = []
    for dep in deps:
        if not dep:
            continue
        dep_str = str(dep).strip()
        if not dep_str:
            continue
        merged.append(dep_str)
    if not merged:
        return None
    return ",".join(merged)


def check_exists(local_path):
    if os.path.exists(local_path):
        return True
    else:
        return False

def launch_sbatch(sbatch_script_path, dependency=None, array: str | None = None) -> str:
    extra_args: list[str] = []
    if dependency is not None:
        extra_args.append(f"--dependency={dependency}")
    if array:
        extra_args.append(f"--array={array}")
    extra_flags = " ".join(extra_args)
    sbatch_cmd = f"sbatch {extra_flags} {sbatch_script_path}".strip()

    result = subprocess.run(
        sbatch_cmd,
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stdout.strip()
        err = result.stderr.strip()
        combined = "\n".join(filter(None, [msg, err]))
        raise RuntimeError(
            f"sbatch command failed (code {result.returncode}): {sbatch_cmd}\n{combined}"
        )

    raw_output = (result.stdout or "").strip()
    job_id = raw_output.split()[::-1][0]
    print(
        f"Job {job_id} submitted"
        f"{f' with dependency {dependency}' if dependency else ''}"
        f"{f' and array {array}' if array else ''}."
    )
    return job_id

def wandb_init(kwargs):
    wandb_run_name = "_".join([str(value) for key, value in kwargs.items()])
    wandb_run_name = wandb_run_name.replace("/", "_")
    wandb_project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    wandb.init(project=wandb_project, name=wandb_run_name, config=kwargs)

def _extract_agent_name(dataset_name: str):
    return sanitize_repo_component(dataset_name)


def _build_training_parameters_link(hub_model_id: str):
    if not hub_model_id:
        return None
    hub_model_id = hub_model_id.strip("/")
    return f"https://huggingface.co/{hub_model_id}/blob/main/config.json"


def _fetch_wandb_times(entity: str, project: str, run_name: str):
    if not entity or not project or not run_name:
        return None, None
    try:
        api = wandb.Api()
    except Exception:
        return None, None

    try:
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
    except TypeError:
        runs = api.runs(f"{entity}/{project}")
    except Exception:
        return None, None

    try:
        for run in runs:
            run_display = getattr(run, "display_name", None)
            run_name_attr = getattr(run, "name", None)
            if run_display == run_name or run_name_attr == run_name:
                start = getattr(run, "created_at", None)
                end = getattr(run, "finished_at", None)
                if end is None:
                    end = getattr(run, "updated_at", None)
                start_iso = start.isoformat() if hasattr(start, "isoformat") else start
                end_iso = end.isoformat() if hasattr(end, "isoformat") else end
                return start_iso, end_iso
    except ValueError:
        # Happens when the project does not exist or is not accessible.
        return None, None
    return None, None


def _collect_wandb_metadata(exp_args, train_config):
    report_to = train_config.get("report_to", "")
    wandb_enabled = False
    if isinstance(report_to, str):
        wandb_enabled = report_to.lower() == "wandb"
    elif isinstance(report_to, (list, tuple, set)):
        wandb_enabled = any(str(item).lower() == "wandb" for item in report_to)

    if not wandb_enabled:
        return None, None, None

    project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    entity = (
        os.environ.get("WANDB_ENTITY")
        or os.environ.get("WANDB_USERNAME")
        or exp_args.get("job_creator")
    )
    run_name_value = train_config.get("run_name") or exp_args.get("job_name")
    run_name = str(run_name_value) if run_name_value else None

    wandb_link = None
    if entity and project and run_name:
        wandb_link = f"https://wandb.ai/{entity}/{project}/runs/{run_name}"

    training_start, training_end = _fetch_wandb_times(entity, project, run_name)
    return wandb_link, training_start, training_end


def write_run_summary(exp_args, train_config):
    job_type = str(exp_args.get("job_type", "train") or "train").lower()
    if job_type not in ("train", "rl"):
        return

    output_dir = train_config.get("output_dir") or exp_args.get("output_dir")
    if not output_dir:
        return

    os.makedirs(output_dir, exist_ok=True)

    dataset_name = train_config.get("dataset") or exp_args.get("dataset")
    agent_name = _extract_agent_name(dataset_name) if dataset_name else None

    hub_model_id = train_config.get("hub_model_id") or exp_args.get("hub_model_id")
    training_parameters_link = _build_training_parameters_link(hub_model_id)

    wandb_link, training_start, training_end = _collect_wandb_metadata(exp_args, train_config)

    training_type = "SFT" if job_type != "rl" else "RL"

    summary_payload = {
        "agent_name": agent_name,
        "training_start": training_start,
        "training_end": training_end,
        "created_by": exp_args.get("job_creator", "DCAgent"),
        "base_model_name": train_config.get("model_name_or_path") or exp_args.get("model_name_or_path"),
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


# Curly braces but not those within ${...}
curly_brace_pattern = r"(?<!\$)\{([^{}]*)\}"

def extract_template_keys(file_path):
    with open(file_path, "r") as f:
        file = f.read()
    return re.findall(curly_brace_pattern, file)

def fill_template(file_path, exp_args, new_file_path):
    with open(file_path, "r") as f:
        file = f.read()

    file = re.sub(curly_brace_pattern, lambda m: exp_args[m.group(1)], file)

    with open(new_file_path, "w") as f:
        f.write(file)


def _escape_template_braces(text: str) -> str:
    """
    Escape braces for str.format() except when part of bash-style ${var} blocks.
    """
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        char = text[i]
        if char == "$" and i + 1 < length and text[i + 1] == "{":
            # Preserve bash variable references (${VAR}) exactly.
            j = i + 2
            while j < length and text[j] != "}":
                j += 1
            if j < length:
                result.append(text[i : j + 1])
                i = j + 1
                continue
        if char == "{":
            result.append("{{")
        elif char == "}":
            result.append("}}")
        else:
            result.append(char)
        i += 1
    return "".join(result)

def _escape_bash_variables(text: str) -> str:
    """
    Escape `${...}` sequences so `str.format` leaves the shell variable syntax
    untouched while still allowing template placeholders **inside** the default
    portions (e.g. `$FOO/{experiments_dir}`) to be substituted.
    """
    result: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] == "$" and i + 1 < length and text[i + 1] == "{":
            start = i
            depth = 1
            j = i + 2
            while j < length and depth > 0:
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                j += 1
            # Recursively escape nested `${...}` blocks but leave template braces alone.
            inner = text[i + 2 : j - 1]
            escaped_inner = _escape_bash_variables(inner)
            result.append("${{" + escaped_inner + "}}")
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)

def _normalize_strategy_value(value):
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or normalized.lower() in {"none", "null", "false"}:
            return None
        return normalized
    return value


def _detect_distributed_strategy(exp_args) -> Optional[str]:
    if _normalize_strategy_value(exp_args.get("deepspeed")):
        return "deepspeed"
    fsdp_cfg = exp_args.get("fsdp_config")
    if isinstance(fsdp_cfg, dict) and fsdp_cfg:
        return "fsdp"
    if _normalize_strategy_value(exp_args.get("fsdp")):
        return "fsdp"
    return None


def _resolve_mixed_precision_setting(exp_args) -> str:
    if _normalize_strategy_value(exp_args.get("fp8")):
        return "fp8"
    if exp_args.get("bf16") or exp_args.get("pure_bf16"):
        return "bf16"
    if exp_args.get("fp16"):
        return "fp16"
    return "no"


def _render_fsdp_config_block(exp_args) -> str:
    fsdp_cfg = exp_args.get("fsdp_config")
    if isinstance(fsdp_cfg, dict) and fsdp_cfg:
        rendered = yaml.safe_dump(fsdp_cfg, sort_keys=False).strip()
    else:
        rendered = "\n".join(
            [
                "fsdp_version: 2",
                "fsdp_state_dict_type: SHARDED_STATE_DICT",
                "fsdp_offload_params: false",
                "fsdp_reshard_after_forward: true",
                "fsdp_cpu_ram_efficient_loading: true",
            ]
        )
    return textwrap.indent(rendered, "  ")


def _build_accelerate_config_block(exp_args) -> str:
    strategy = _detect_distributed_strategy(exp_args)
    if not strategy:
        return ""

    lines: list[str] = []
    accelerate_header = 'ACCELERATE_CONFIG_FILE="$TMP_DIR/${SLURM_JOB_ID}_accelerate_config.yaml.autogenerated"'
    if strategy == "deepspeed":
        ds_config_path = str(exp_args.get("deepspeed") or "").strip()
        lines.append(f"DEEPSPEED_CONFIG_FILE={ds_config_path}")
        lines.append(accelerate_header)
        lines.append("export ACCELERATE_CONFIG_FILE")
        lines.append('cat << EOT > "$ACCELERATE_CONFIG_FILE"')
        lines.append("# WARNING: auto-generated by launcher")
        lines.append("compute_environment: LOCAL_MACHINE")
        lines.append("deepspeed_config:")
        lines.append("  deepspeed_multinode_launcher: standard")
        lines.append("  deepspeed_config_file: $DEEPSPEED_CONFIG_FILE")
        lines.append("  zero3_init_flag: true")
        lines.append("distributed_type: DEEPSPEED")
        lines.append("fsdp_config:")
        lines.append("machine_rank: 0")
        lines.append("main_process_ip: $MASTER_ADDR")
        lines.append("main_process_port: $MASTER_PORT")
        lines.append("main_training_function: main")
        lines.append("num_machines: $SLURM_NNODES")
        lines.append("num_processes: $NUM_GPUS")
        lines.append("use_cpu: false")
        lines.append("EOT")
    elif strategy == "fsdp":
        mixed_precision = _resolve_mixed_precision_setting(exp_args)
        fsdp_config_lines = _render_fsdp_config_block(exp_args)
        lines.append(accelerate_header)
        lines.append("export ACCELERATE_CONFIG_FILE")
        lines.append('cat << EOT > "$ACCELERATE_CONFIG_FILE"')
        lines.append("# WARNING: auto-generated by launcher")
        lines.append("compute_environment: LOCAL_MACHINE")
        lines.append("distributed_type: FSDP")
        lines.append(f"mixed_precision: {mixed_precision}")
        lines.append("fsdp_config:")
        if fsdp_config_lines:
            lines.extend(fsdp_config_lines.splitlines())
        lines.append("machine_rank: 0")
        lines.append("main_process_ip: $MASTER_ADDR")
        lines.append("main_process_port: $MASTER_PORT")
        lines.append("main_training_function: main")
        lines.append("num_machines: $SLURM_NNODES")
        lines.append("num_processes: $NUM_GPUS")
        lines.append("use_cpu: false")
        lines.append("EOT")

    block = "\n".join(lines).strip()
    if not block:
        return ""
    return _escape_template_braces(block) + "\n"

def construct_sbatch_script(exp_args):
    base_script_path = exp_args["train_sbatch_path"]
    with open(base_script_path, "r") as f:
        base_script = f.read()

    kwargs = defaultdict(str, **exp_args)
    kwargs["accelerate_config_block"] = _build_accelerate_config_block(exp_args)

    # find JSON file creation with cat
    json_files_cat = re.findall(r"cat.*?<<EOT >.*?EOT", base_script, re.DOTALL)
    json_filenames = []
    for json_file in json_files_cat:
        json_file_name = re.match(
            r"cat.*?<<EOT >.*?(\S+).*?EOT", json_file, re.DOTALL
        ).group(1)
        json_filenames.append(json_file_name)

        base_script = re.sub(
            r"cat.*?<<EOT >.*?" + json_file_name.replace("$", "\\$") + r".*?EOT",
            f"cat {json_file_name}",
            base_script,
            count=1,
            flags=re.DOTALL,
        )

    # safeguard against injection of bash ${} variables
    base_script = _escape_bash_variables(base_script)

    time_limit = kwargs.get("time_limit")
    if time_limit is None:
        time_limit = "01:00:00"
        kwargs["time_limit"] = time_limit

     
    hpc = detect_hpc()
    hpc_name = hpc.name
    if hpc_name == "jureca" or hpc_name == "juwels": 
        login_node = socket.gethostname().split('.')[0] + "i"
        if "{login_node}" in base_script:
            if kwargs.get("internet_node", False):
                # check if proxychains installed
                if not shutil.which("proxychains4"):
                    raise RuntimeError("proxychains4 not found, please install it to use internet_node")
            base_script = base_script.replace("{login_node}", login_node)

    sbatch_script = base_script.format(**kwargs)
    sbatch_script = _ensure_dependency_directive(sbatch_script, exp_args.get("dependency"))

    # Ensure version checks are disabled across all training runs
    # by injecting a simple export block after SBATCH headers.
    env_block = {
        "DISABLE_VERSION_CHECK": "1",
    }
    stage_value = str(exp_args.get("stage") or "").lower()
    if exp_args.get("use_mca") and stage_value == "sft":
        env_block["USE_MCA"] = "1"
        os.environ.setdefault("USE_MCA", "1")

    sbatch_script = _inject_env_block(sbatch_script, env_block)

    for json_file, json_file_name in zip(json_files_cat, json_filenames):
        sbatch_script = sbatch_script.replace(f"cat {json_file_name}", json_file)

    sbatch_dir = os.path.join(kwargs["experiments_dir"], "sbatch_scripts")
    os.makedirs(sbatch_dir, exist_ok=True)
    sbatch_script_path = os.path.join(sbatch_dir, f"{kwargs['job_name']}.sbatch")
    with open(sbatch_script_path, "w") as f:
        f.write(sbatch_script)
        print(f"Wrote sbatch script to {sbatch_script_path}")

    return sbatch_script_path

def construct_config_yaml(exp_args):
    configs_dir = os.path.join(exp_args["experiments_dir"], "configs")
    os.makedirs(configs_dir, exist_ok=True)

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
    with open(train_config_path, "r") as f:
        base_config = f.read()

    # don't do templating for the yaml - simplification
    # base_config = base_config.format(**exp_args)

    base_config = yaml.safe_load(base_config)

    def _sanitize_component(component: str) -> str:
        import re

        comp = component.strip().rstrip('/')
        comp = os.path.basename(comp) if comp else component
        comp = re.sub(r"[^A-Za-z0-9]+", "-", comp).strip("-")
        return comp or "model"

    model_name = base_config.get("model_name_or_path") or exp_args.get("model_name_or_path")
    if isinstance(model_name, str) and model_name:
        model_component = _sanitize_component(model_name)
        current_job_name = exp_args.get("job_name", "")
        if model_component.lower() not in current_job_name.lower():
            suggested = f"{current_job_name}_{model_component}" if current_job_name else model_component
            if len(suggested) > 96:
                suggested = suggested[:96]
            print(f"Including model identifier in job name: {current_job_name} -> {suggested}")
            exp_args = update_exp_args(exp_args, {"job_name": suggested})

    if exp_args.pop("push_to_db", None) is not None:
        print("Dropping deprecated argument 'push_to_db' from launcher inputs")
    if base_config.pop("push_to_db", None) is not None:
        print("Dropping deprecated argument 'push_to_db' from train config")

    # Normalize legacy DeepSpeed config paths to submodule paths
    def _normalize_deepspeed_path(ds_val: str) -> str:
        if not isinstance(ds_val, str):
            return ds_val
        mapping = {
            "dcft/train/zero3.json": "dcft/train/llamafactory/examples/deepspeed/ds_z3_config.json",
            "dcft/train/zero3_offload.json": "dcft/train/llamafactory/examples/deepspeed/ds_z3_offload_config.json",
            "dcft/train/zero2.json": "dcft/train/llamafactory/examples/deepspeed/ds_z2_config.json",
        }
        # Exact match first
        if ds_val in mapping:
            return mapping[ds_val]
        # Fallback: replace basename if path variants appear
        for old, new in mapping.items():
            if ds_val.endswith(old) or old in ds_val:
                return ds_val.replace(old, new)
        return ds_val

    explicit_cli_keys = set(exp_args.get("_explicit_cli_keys", []))
    exp_args.pop("_explicit_cli_keys", None)

    # Update base config with experiment arguments
    for key, value in exp_args.items():
        if key.startswith("_"):
            continue
        if key == "deepspeed" and key not in explicit_cli_keys:
            continue
        if key in base_config or key in [
            field.name for field in dataclasses.fields(LlamaFactoryArgs)
        ]:
            print(f"Setting {key} to {value}")
            base_config[key] = value

    default_ds = LlamaFactoryArgs.__dataclass_fields__["deepspeed"].default
    if not base_config.get("deepspeed"):
        base_config["deepspeed"] = exp_args.get("deepspeed", default_ds) or default_ds

    # Apply DS path normalization after merges (covers YAML and CLI overrides)
    if isinstance(base_config.get("deepspeed"), str):
        normalized = _normalize_deepspeed_path(base_config["deepspeed"])
        if normalized != base_config["deepspeed"]:
            print(f"Normalized deepspeed path: {base_config['deepspeed']} -> {normalized}")
            base_config["deepspeed"] = normalized

    if base_config.get("dataset_dir") is None:
        base_config["dataset_dir"] = "ONLINE"

    dataset_field = base_config.get("dataset")
    dataset_entries: list[str] = []
    if isinstance(dataset_field, str):
        dataset_entries = [item.strip() for item in dataset_field.split(",") if item.strip()]
    elif isinstance(dataset_field, (list, tuple)):
        dataset_entries = [str(item).strip() for item in dataset_field if str(item).strip()]

    dataset_paths: list[str] = []

    if "_pretokenize" in exp_args["job_name"]:
        # Already have downloaded dataset and model, since the train yaml is already constructed (a bit hacky)
        model_path = exp_args["model_name_or_path"]
        dataset_path = exp_args["dataset"]
        dataset_paths = [item.strip() for item in str(dataset_path).split(",") if item.strip()]
    else:
        # Download Dataset and Model - MAKE SURE HF_HUB_CACHE is set!
        # Download the parquet dataset from HuggingFace
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

        parquet_download_path = dataset_paths[0]
        if download_datasets:
            print(f"Downloaded dataset to {parquet_download_path}")

        # Find the parquet file in the downloaded directory
        if exp_args.get("job_type") == "datagen" and base_config.get("datagen_mode") == "trace":
            parquet_files = []
            for root, _, files in os.walk(parquet_download_path):
                for fname in files:
                    if fname.endswith(".parquet"):
                        parquet_files.append(os.path.join(root, fname))
                if parquet_files:
                    break

            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found in {parquet_download_path}")

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
        else:
            dataset_path = parquet_download_path

        model_path = snapshot_download(repo_id=base_config["model_name_or_path"], repo_type="model")
        print(f"Downloaded model to {model_path}")

    if base_config.get("output_dir") and checkpoints_dir not in base_config.get(
        "output_dir"
    ):
        base_config["output_dir"] = os.path.join(
            checkpoints_dir, base_config["output_dir"]
        )
    else:
        base_config["output_dir"] = os.path.join(checkpoints_dir, exp_args["job_name"])
    os.makedirs(base_config["output_dir"], exist_ok=True)

    wandb_dir = os.path.join(exp_args["experiments_dir"], "wandb", exp_args["job_name"])
    os.makedirs(wandb_dir, exist_ok=True)
    os.environ["WANDB_DIR"] = wandb_dir

    # If no explicit run_name is provided, default to the launcher job_name.
    # This keeps W&B runs aligned with the job naming used for outputs.
    if not base_config.get("run_name"):
        base_config["run_name"] = exp_args["job_name"]
    # Mirror into environment so W&B SDK can also pick it up when initializing.
    os.environ["WANDB_NAME"] = str(base_config["run_name"]) if base_config.get("run_name") else exp_args["job_name"]

    hub_model_id = base_config.get("hub_model_id", None)
    if hub_model_id is not None:
        hub_model_id = hub_model_id.replace(".", "_")
    base_config["hub_model_id"] = hub_model_id

    if exp_args.get("job_type") == "datagen" and base_config.get("datagen_mode") == "trace":
        base_config["dataset"] = dataset_path
        base_config["dataset_dir"] = dataset_path
    elif not exp_args["internet_node"]:
        base_config["datasets_cache_dir"] = os.environ["HF_HUB_CACHE"]
        if dataset_paths:
            base_config["dataset"] = ",".join(dataset_paths)

    if exp_args["internet_node"]:
        base_config["report_to"] = "wandb"
        base_config["push_to_hub"] = True
    else:
        # no wandb reporting
        if "report_to" in base_config:
            del base_config["report_to"]
        base_config["push_to_hub"] = False
        # we need to pass directly the downloaded model path in the cache (there is no setting for model_cache_dirs)
        base_config["model_name_or_path"] = model_path

    num_nodes = int(exp_args.get("num_nodes"))
    num_gpus = int(exp_args.get("gpus_per_node"))

    raw_global_batch_size = exp_args.pop("global_batch_size", None)
    if raw_global_batch_size is None:
        raw_global_batch_size = base_config.pop("global_batch_size", None)
    else:
        base_config.pop("global_batch_size", None)

    global_batch_size = None
    if raw_global_batch_size is not None:
        try:
            global_batch_size = int(raw_global_batch_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected integer-like global_batch_size, got {raw_global_batch_size!r}"
            ) from exc

    if global_batch_size is not None:
        print(
            f"\nCalculated based on {num_nodes} nodes, {num_gpus} GPUs per node, and global batch size {global_batch_size}:"
        )
        total_gpu_count = num_nodes * num_gpus

        def _int_config_value(key: str, default: int = 1) -> int:
            raw_value = base_config.get(key, default)
            try:
                return int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Expected integer-like value for {key}, got {raw_value!r}"
                ) from exc

        tensor_model_parallel_size = _int_config_value("tensor_model_parallel_size", 1)
        pipeline_model_parallel_size = _int_config_value("pipeline_model_parallel_size", 1)
        expert_model_parallel_size = _int_config_value("expert_model_parallel_size", 1)

        model_parallel_world_size = (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * expert_model_parallel_size
        )
        if model_parallel_world_size <= 0:
            raise ValueError(
                f"Model parallel world size must be positive; got {model_parallel_world_size}"
            )

        if total_gpu_count % model_parallel_world_size != 0:
            print(
                f"Warning: total GPU count ({total_gpu_count}) is not divisible by model parallel size "
                f"({model_parallel_world_size}). Rounding down data parallel replicas."
            )
        data_parallel_replicas = max(total_gpu_count // model_parallel_world_size, 1)

        per_device_train_batch_size = base_config.get("per_device_train_batch_size", 1)
        try:
            per_device_train_batch_size = int(per_device_train_batch_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Expected integer-like per_device_train_batch_size, got {per_device_train_batch_size!r}"
            ) from exc
        per_device_train_batch_size = max(per_device_train_batch_size, 1)

        effective_batch_denom = per_device_train_batch_size * data_parallel_replicas
        if effective_batch_denom == 0:
            raise ValueError("Effective batch denominator resolved to zero.")

        gradient_accumulation_steps = global_batch_size // effective_batch_denom
        if gradient_accumulation_steps == 0 or (
            gradient_accumulation_steps * effective_batch_denom != global_batch_size
        ):
            raise ValueError(
                "Global batch size is not divisible by per-device batch * data-parallel replicas. "
                f"global_batch_size={global_batch_size}, per_device_train_batch_size={per_device_train_batch_size}, "
                f"data_parallel_replicas={data_parallel_replicas}"
            )

        base_config["gradient_accumulation_steps"] = gradient_accumulation_steps
        base_config["per_device_train_batch_size"] = per_device_train_batch_size
        print(f"data_parallel_replicas: {data_parallel_replicas}")
        print(f"per_device_train_batch_size: {per_device_train_batch_size}")
        print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    else:
        print("\nSkipping automatic gradient accumulation calculation because global_batch_size was not provided.")
    hub_model_id = base_config.get("hub_model_id", None)

    if hub_model_id is None:
        hub_model_id = "mlfoundations-dev/" + exp_args["job_name"]
    base_config["hub_model_id"] = hub_model_id
    tokenized_path = base_config.get("tokenized_path")

    if tokenized_path is None and exp_args.get("pretokenize"):
        tokenized_dir = exp_args.get("tokenized_dir")
        tokenized_dir = os.path.expandvars(
        os.environ.get("TOKENIZED_DATASETS_DIR", tokenized_dir)
        )
        model_name = "_".join(
            base_config["model_name_or_path"].split("/")[-2:]
        ).replace(".", "-")
        def _slugify(entry: str) -> str:
            entry = entry.strip().rstrip("/")
            if "/" in entry:
                entry = entry.split("/")[-1]
            return entry.replace(".", "-")

        dataset_name_parts = [_slugify(entry) for entry in dataset_entries] or ["dataset"]
        dataset_name = "-".join(dataset_name_parts)
        base_config["tokenized_path"] = os.path.join(
            tokenized_dir, "_".join([dataset_name, model_name, "tokenized"])
        )
        exp_args["tokenized_path"] = base_config["tokenized_path"]

    data_tags = [
        "role_tag",
        "content_tag",
        "assistant_tag",
        "user_tag",
        "messages",
        "system",
    ]

    for tag in data_tags:
        if tag in exp_args:
            tag_value = exp_args[tag]
            if tag_value is not None:
                base_config[tag] = tag_value

    train_config_path_out = os.path.join(
        configs_dir, exp_args["job_name"] + "_train_config.yaml"
    )
    with open(train_config_path_out, "w") as f:
        yaml.dump(base_config, f)
        print(f"Wrote config to {train_config_path_out}")

    exp_args["output_dir"] = base_config["output_dir"]
    exp_args["dataset"] = base_config["dataset"]
    exp_args["model_name_or_path"] = base_config["model_name_or_path"]
    exp_args["hub_model_id"] = base_config.get("hub_model_id", None)
    return base_config, train_config_path_out

def submit_job(
    exp_args=None,
    dependency=None,
):
    exp_args["logs_dir"] = os.path.join(exp_args["experiments_dir"], "logs")
    os.makedirs(exp_args["logs_dir"], exist_ok=True)

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

def update_exp_args(exp_args, args, *, explicit_keys: Optional[set[str]] = None):
    explicit_keys = set(explicit_keys or [])
    for key, value in args.items():
        if key.startswith("_"):
            continue

        has_existing = key in exp_args
        existing_value = exp_args.get(key)
        is_explicit = not explicit_keys or key in explicit_keys

        if value is None:
            if has_existing and is_explicit:
                del exp_args[key]
                print(f"Removed {key} from experiment arguments")
            continue

        if has_existing:
            if not is_explicit and value != existing_value:
                continue
            if value != existing_value:
                print(f"Overwrote {key} from {existing_value} to {value}")
        exp_args[key] = value
    return exp_args

def display_args(exp_args, name):
    print()
    print("=" * 20 + f" {name} Args " + "=" * 20)
    for key, value in exp_args.items():
        print(f"{key}: {value}")
    print()

def _build_vllm_env_vars(exp_args: dict, *, include_pinggy: bool = True) -> tuple[dict, dict]:
    """Return environment variables used to configure vLLM processes.

    Args:
        exp_args: Experiment arguments (mutated in-place with defaults).
        include_pinggy: Whether to export Pinggy-related env vars. Only the
            standalone vLLM server needs these; datagen/trace jobs derive their
            API base directly from the endpoint JSON.
    """
    env: dict[str, str] = {}
    cfg = exp_args.get("_datagen_vllm_server_config")
    if not cfg:
        return env, exp_args

    env["VLLM_MODEL_PATH"] = cfg.model_path
    env["VLLM_NUM_REPLICAS"] = str(cfg.num_replicas or 1)
    env["VLLM_TENSOR_PARALLEL_SIZE"] = str(cfg.tensor_parallel_size or 1)
    env["VLLM_PIPELINE_PARALLEL_SIZE"] = str(cfg.pipeline_parallel_size or 1)

    if cfg.hf_overrides:
        env["VLLM_HF_OVERRIDES"] = cfg.hf_overrides
    if cfg.use_deep_gemm:
        env["VLLM_USE_DEEP_GEMM"] = "1"
    if cfg.max_num_seqs is not None:
        env["VLLM_MAX_NUM_SEQS"] = str(cfg.max_num_seqs)
    if cfg.gpu_memory_utilization is not None:
        env["VLLM_GPU_MEMORY_UTILIZATION"] = str(cfg.gpu_memory_utilization)
    if cfg.enable_expert_parallel:
        env["VLLM_ENABLE_EXPERT_PARALLEL"] = "1"
    if cfg.swap_space is not None:
        env["VLLM_SWAP_SPACE"] = str(cfg.swap_space)
    if cfg.max_seq_len_to_capture is not None:
        env["VLLM_MAX_SEQ_LEN_TO_CAPTURE"] = str(cfg.max_seq_len_to_capture)
    if cfg.max_model_len is not None:
        env["VLLM_MAX_MODEL_LEN"] = str(cfg.max_model_len)
    if cfg.trust_remote_code:
        env["VLLM_TRUST_REMOTE_CODE"] = "1"
    if cfg.disable_log_requests:
        env["VLLM_DISABLE_LOG_REQUESTS"] = "1"
    if cfg.custom_model_name:
        env["VLLM_CUSTOM_MODEL_NAME"] = cfg.custom_model_name
    if cfg.enable_auto_tool_choice:
        env["VLLM_ENABLE_AUTO_TOOL_CHOICE"] = "1"
    if cfg.tool_call_parser:
        env["VLLM_TOOL_CALL_PARSER"] = cfg.tool_call_parser
    if cfg.reasoning_parser:
        env["VLLM_REASONING_PARSER"] = cfg.reasoning_parser

    if include_pinggy:
        explicit_cli_keys = set(exp_args.get("_explicit_cli_keys", []) or [])
        pinggy_fields = (
            ("VLLM_PINGGY_PERSISTENT_URL", "pinggy_persistent_url", "PINGGY_PERSISTENT_URL"),
            ("VLLM_PINGGY_SSH_COMMAND", "pinggy_ssh_command", "PINGGY_SSH_COMMAND"),
            ("VLLM_PINGGY_DEBUGGER_URL", "pinggy_debugger_url", "PINGGY_DEBUGGER_URL"),
        )
        for env_key, arg_key, fallback_env in pinggy_fields:
            candidate = exp_args.get(arg_key)
            explicit = arg_key in explicit_cli_keys
            if isinstance(candidate, str):
                candidate = candidate.strip()
            fallback_allowed = not explicit
            if candidate in (None, "", "None") and fallback_allowed:
                fallback = os.environ.get(fallback_env)
                if isinstance(fallback, str):
                    fallback = fallback.strip()
                candidate = fallback
            if candidate in (None, "", "None"):
                continue
            candidate_str = str(candidate)
            env[env_key] = candidate_str
            if exp_args.get(arg_key) != candidate:
                exp_args = update_exp_args(exp_args, {arg_key: candidate})

    max_output_tokens = (
        exp_args.get("trace_max_tokens")
        if exp_args.get("trace_max_tokens") not in (None, "", "None")
        else exp_args.get("datagen_max_tokens")
    )
    if max_output_tokens not in (None, "", "None"):
        env["VLLM_MAX_OUTPUT_TOKENS"] = str(max_output_tokens)

    endpoint_path = exp_args.get("vllm_endpoint_json_path")
    if not endpoint_path and cfg and cfg.endpoint_json_path:
        endpoint_path = cfg.endpoint_json_path
    if not endpoint_path:
        endpoint_path = os.path.join(exp_args["experiments_dir"], "vllm_endpoint.json")
        exp_args = update_exp_args(exp_args, {"vllm_endpoint_json_path": endpoint_path})
    env["VLLM_ENDPOINT_JSON_PATH"] = endpoint_path

    submit_timeout = exp_args.get("ray_cgraph_submit_timeout")
    get_timeout = exp_args.get("ray_cgraph_get_timeout")
    max_inflight = exp_args.get("ray_cgraph_max_inflight_executions")
    if submit_timeout:
        env["RAY_CGRAPH_submit_timeout"] = str(submit_timeout)
    if get_timeout:
        env["RAY_CGRAPH_get_timeout"] = str(get_timeout)
    if max_inflight:
        env["RAY_CGRAPH_max_inflight_executions"] = str(max_inflight)

    _maybe_set_ray_cgraph_env(env)

    extra_cli_args = exp_args.get("_vllm_server_extra_args")
    if extra_cli_args:
        env["VLLM_SERVER_EXTRA_ARGS_JSON"] = json.dumps(extra_cli_args)

    return env, exp_args

def pre_validation(exp_args, cli_args):

    # Add arguments to experiment from train config file
    if "train_config_path" in cli_args and os.path.exists(
        cli_args["train_config_path"]
    ):
        # with open(cli_args["train_config_path"], "r") as f:
        #     config = yaml.safe_load(f)
        #     exp_args = update_exp_args(exp_args, config)
        pass
    elif "train_config_path" in cli_args:
        raise FileNotFoundError(
            f"Train config file {cli_args['train_config_path']} does not exist."
        )

    # Fill in sbatch template
    if "train_sbatch_path" in exp_args and os.path.exists(
        exp_args["train_sbatch_path"]
    ):
        template_keys = extract_template_keys(exp_args["train_sbatch_path"])
        allowlist = {"train_config_path_out", "accelerate_config_block"}
        for key in template_keys:
            if (
                key not in exp_args
                and key not in cli_args
                and key not in allowlist
            ):
                raise ValueError(
                    f"Template key {key} not found in experiment arguments or cli arguments."
                )
    elif "train_sbatch_path" in exp_args:
        raise FileNotFoundError(
            f"Train sbatch file {exp_args['train_sbatch_path']} does not exist."
        )

    # Cluster-specific validation
    if exp_args.get("name") == "perlmutter":
        requested_gpus = exp_args.get("gpus_per_node") or cli_args.get("gpus_per_node")
        if requested_gpus:
            try:
                requested_gpus_int = int(requested_gpus)
            except (TypeError, ValueError):
                raise ValueError("Perlmutter requires 4 GPUs per node.")
            if requested_gpus_int != 4:
                raise ValueError("Perlmutter requires 4 GPUs per node.")
        exp_args["gpus_per_node"] = "4"


def schedule_eval(exp_args, train_job_id):
    eval_tasks = exp_args["eval_tasks"]
    model_name = f"DCAgent/{exp_args['job_name']}"
    
    num_nodes = exp_args.get("eval_num_nodes")
    if num_nodes is None:
        num_nodes = os.environ["NUM_NODES_DEFAULT"]
    num_shards = str(int(num_nodes) * int(os.environ["NUM_GPUS_PER_NODE"]))

    eval_time_limit = exp_args.get("eval_time_limit", "4:00:00")
    max_job_duration = str(int(eval_time_limit.split(":")[0]))

    evalchemy_path = os.environ["EVALCHEMY"]
    evalchemy_activate_env = os.environ["EVALCHEMY_ACTIVATE_ENV"]
    
    print(f"Scheduling automatic evalution following training job:")
    if eval_tasks:
        eval_cmd = f"{evalchemy_activate_env}"
        eval_cmd += f" && cd {evalchemy_path}"
        eval_cmd += f" && python eval/distributed/launch_simple.py"
        eval_cmd += f" --tasks {eval_tasks}"
        eval_cmd += f" --num_shards {num_shards}"
        eval_cmd += f" --max-job-duration {max_job_duration}"
        eval_cmd += f" --model_name {model_name}"
        eval_cmd += f" --dependency afterok:{train_job_id}"
        
        print(f"Launching evaluation with command: {eval_cmd}")
        result = subprocess.run(eval_cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            # Filter out Vista system checks using regex
            filtered_output = re.sub(r'-+\n\s+Welcome to.*\n-+\n.*?(Submitted batch job \d+)', r'\1', result.stdout, flags=re.DOTALL)
            filtered_output = re.sub(r'No reservation.*\n(-->.*\n)*', '', filtered_output)
            print(filtered_output.strip())
        else:
            print("Error launching evaluation job:")
            print(result.stderr)

def schedule_pretokenize(exp_args):
    pretok_args = exp_args.copy()
    pretok_args.pop("dependency", None)
    pretok_args = update_exp_args(pretok_args, {"job_name": f"{exp_args['job_name']}_pretokenize"})
    # this keeps world size, per-device batch, and accumulation factors aligned after shrinking to one node for pretokenization
    pretok_args = update_exp_args(pretok_args, {"num_nodes": 1})
    # pretok_args = update_exp_args(pretok_args, {"deepspeed": None, "enable_liger_kernel": False, })
    # Just use the same LF yaml for the pretokenization job
    pretok_train_config, pretok_train_config_path_out = construct_config_yaml(pretok_args)
    if exp_args.get("pretok_large"):
        # You shouldn't pretokenize on 128 nodes
        # pretok_args = update_exp_args(pretok_args, {"num_nodes": 128, "qos": "boost_qos_bprod", "time_limit": "1-00:00:00", "max_restarts": 0, "job_name": f"{exp_args['job_name']}_pretokenize"})
        if exp_args["name"] != "leonardo":
            raise ValueError("Large pretokenization is only supported on leonardo")
        pretok_args = update_exp_args(pretok_args, {
        "time_limit": "03:00:00",
        "qos": "normal",
        "max_restarts": 0,
        "node_exclusion_list": "",
        "job_name": f"{exp_args['job_name']}_pretokenize"})
    else:
        pretok_args = update_exp_args(pretok_args, {
        "partition": exp_args['pretok_partition'], 
        "qos": exp_args['pretok_qos'], 
        "time_limit": exp_args['pretok_time_limit'],
        # this I was using to test with cpu only nodes - needed to make the srun work
        # "cpus_per_node": exp_args['pretok_cpus_per_node'],
        # "gpus_per_node": exp_args['pretok_gpus_per_node'],
        "max_restarts": 0,
        })
    pretok_args = update_exp_args(pretok_args, pretok_train_config)
    pretok_args = update_exp_args(pretok_args, {"train_config_path_out": pretok_train_config_path_out})
    pretok_sbatch_path_out = construct_sbatch_script(pretok_args)
    pretok_args = update_exp_args(pretok_args, {"train_sbatch_path_out": pretok_sbatch_path_out})
    pretok_job_id = submit_job(
        exp_args=pretok_args,
        dependency=None,
    )
    return pretok_job_id

def launch_vllm_server(exp_args: dict, hpc) -> str:
    """
    Launch VLLM server job with Pinggy tunnel.

    Returns:
        Job ID of the VLLM server job
    """
    print("\n=== Launching VLLM Server ===")

    # Prepare environment variables for VLLM server
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    if not vllm_cfg:
        raise ValueError("vLLM server launch requested but no vllm_server configuration provided.")

    vllm_env_vars, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=True)

    model_path = vllm_cfg.model_path

    num_replicas = vllm_cfg.num_replicas or 1
    tensor_parallel = vllm_cfg.tensor_parallel_size or 1
    pipeline_parallel = vllm_cfg.pipeline_parallel_size or 1
    server_time_limit = vllm_cfg.time_limit or "48:00:00"

    # Prepare sbatch script path
    sbatch_dir = os.path.join(os.path.dirname(__file__), "sbatch_data")
    vllm_sbatch_template = os.path.join(sbatch_dir, f"{hpc.name}_vllm_server.sbatch")

    if not os.path.exists(vllm_sbatch_template):
        # Fallback to vista template
        vllm_sbatch_template = os.path.join(sbatch_dir, "vista_vllm_server.sbatch")

    # Create output sbatch script
    vllm_job_name = f"{exp_args['job_name']}_vllm_server"
    vllm_sbatch_output = os.path.join(
        exp_args["experiments_dir"],
        "sbatch",
        f"{vllm_job_name}.sbatch"
    )
    os.makedirs(os.path.dirname(vllm_sbatch_output), exist_ok=True)
    os.makedirs(os.path.join(exp_args["experiments_dir"], "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_args["experiments_dir"], "logs"), exist_ok=True)

    # Read template and substitute
    with open(vllm_sbatch_template) as f:
        sbatch_content = f.read()

    # Calculate GPUs needed (must account for TP * PP per replica)
    num_gpus = int(num_replicas or 1) * int(tensor_parallel or 1) * int(pipeline_parallel or 1)

    num_nodes = exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1
    gpus_per_node = exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", num_gpus)

    cpus_per_node_effective = exp_args.get("cpus_per_node")
    if cpus_per_node_effective in (None, "", "None"):
        cpus_per_node_effective = getattr(hpc, "cpus_per_node", 1)
    cpus_per_node_effective = int(cpus_per_node_effective)

    substitutions = {
        "{partition}": hpc.partition,
        "{time_limit}": server_time_limit,
        "{num_nodes}": str(num_nodes),
        "{cpus_per_node}": str(cpus_per_node_effective),
        "{num_gpus}": str(num_gpus),
        "{gpus_per_node}": str(gpus_per_node),
        "{account}": hpc.account,
        "{experiments_dir}": exp_args["experiments_dir"],
        "{job_name}": vllm_job_name,
    }

    for key, value in substitutions.items():
        sbatch_content = sbatch_content.replace(key, value)

    if not gpus_per_node:
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node {gpus_per_node}\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node 0\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node=0\n", "")

    sbatch_content = _inject_env_block(sbatch_content, vllm_env_vars)

    # Write output sbatch
    with open(vllm_sbatch_output, 'w') as f:
        f.write(sbatch_content)

    print(f"VLLM server sbatch: {vllm_sbatch_output}")
    print(f"Endpoint JSON will be saved to: {exp_args['vllm_endpoint_json_path']}")

    # Submit job
    if not exp_args.get("dry_run"):
        job_id = launch_sbatch(vllm_sbatch_output)
        print(f"✓ VLLM server job submitted: {job_id}")
        return job_id
    else:
        print("DRY RUN: Would submit VLLM server job")
        return "dry_run_vllm_job_id"


def launch_task_job(exp_args: dict, hpc, vllm_job_id: str = None) -> str:
    """
    Launch data generation job.

    Args:
        exp_args: Experiment arguments
        hpc: HPC configuration
        vllm_job_id: Optional VLLM server job ID for dependency

    Returns:
        Job ID of the datagen job
    """
    print("\n=== Launching Task Generation Job ===")

    runtime = exp_args.get("_datagen_engine_runtime")
    if runtime is None:
        runtime = _prepare_datagen_configuration(exp_args)

    # Prepare environment variables for datagen
    datagen_engine_value = exp_args.get("datagen_engine", runtime.type if runtime else "openai")
    engine = str(datagen_engine_value or "openai").lower()
    datagen_env_vars = {
        "DATAGEN_SCRIPT": exp_args["datagen_script"],
        "DATAGEN_HEALTHCHECK_INTERVAL": str(exp_args.get("datagen_healthcheck_interval", 300)),
        "DATAGEN_ENGINE": engine,
        "DATAGEN_STAGE": "tasks",
        "DATAGEN_BACKEND": str(exp_args.get("datagen_backend", "vllm")),
    }
    task_type_value = exp_args.get("task_type")
    if task_type_value:
        datagen_env_vars["DATAGEN_TASK_TYPE"] = str(task_type_value)
        print(f"Task generation task type: {task_type_value}")
    if exp_args.get("datagen_target_repo"):
        datagen_env_vars["DATAGEN_TARGET_REPO"] = exp_args["datagen_target_repo"]

    backend = str(datagen_env_vars["DATAGEN_BACKEND"]).lower()
    if backend != datagen_env_vars["DATAGEN_BACKEND"]:
        datagen_env_vars["DATAGEN_BACKEND"] = backend
    is_ray_backend = backend == "ray"

    is_cpu_only = engine == "none" and backend in ("vllm", "none")
    if is_cpu_only:
        backend = "none"
        datagen_env_vars["DATAGEN_BACKEND"] = backend
        exp_args = update_exp_args(exp_args, {"datagen_backend": backend})
    vllm_env_vars, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=False)
    datagen_env_vars.update(vllm_env_vars)

    if is_ray_backend:
        if exp_args.get("ray_cgraph_submit_timeout"):
            datagen_env_vars["RAY_CGRAPH_submit_timeout"] = str(exp_args["ray_cgraph_submit_timeout"])
        if exp_args.get("ray_cgraph_get_timeout"):
            datagen_env_vars["RAY_CGRAPH_get_timeout"] = str(exp_args["ray_cgraph_get_timeout"])
        if exp_args.get("ray_cgraph_max_inflight_executions"):
            datagen_env_vars["RAY_CGRAPH_max_inflight_executions"] = str(
                exp_args["ray_cgraph_max_inflight_executions"]
            )
        _maybe_set_ray_cgraph_env(datagen_env_vars)

    script_gpu_required = False
    script_path = exp_args.get("datagen_script")
    if script_path:
        script_gpu_required = _detect_gpu_required(script_path)

    requested_gpus = 0
    try:
        requested_gpus = int(exp_args.get("gpus_per_node") or 0)
    except (TypeError, ValueError):
        requested_gpus = 0

    if script_gpu_required and requested_gpus == 0:
        requested_gpus = int(getattr(hpc, "gpus_per_node", 0) or 0)

    datagen_gpu_required = script_gpu_required or requested_gpus > 0

    datagen_env_vars["DATAGEN_GPU_REQUIRED"] = "1" if requested_gpus > 0 else "0"
    gpus_per_node = exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", requested_gpus)
    datagen_env_vars["DATAGEN_GPUS_PER_NODE"] = str(gpus_per_node)
    datagen_env_vars["DATAGEN_RAY_PORT"] = str(exp_args.get("datagen_ray_port", 6379))
    datagen_env_vars["DATAGEN_API_PORT"] = str(exp_args.get("datagen_api_port", 8000))
    tensor_parallel_size = None
    pipeline_parallel_size = None
    backend_cfg = exp_args.get("_datagen_backend_config")
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    data_parallel_size = None
    if vllm_cfg:
        tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None)
        pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None)
        data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None and backend_cfg:
        tensor_parallel_size = getattr(backend_cfg, "tensor_parallel_size", None)
    if pipeline_parallel_size is None and backend_cfg:
        pipeline_parallel_size = getattr(backend_cfg, "pipeline_parallel_size", None)
    if data_parallel_size is None and backend_cfg:
        data_parallel_size = getattr(backend_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None:
        tensor_parallel_size = 1
    if pipeline_parallel_size is None:
        pipeline_parallel_size = 1
    if data_parallel_size is None:
        data_parallel_size = 1
    datagen_env_vars["DATAGEN_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    datagen_env_vars["DATAGEN_PIPELINE_PARALLEL_SIZE"] = str(pipeline_parallel_size)
    datagen_env_vars["DATAGEN_DATA_PARALLEL_SIZE"] = str(data_parallel_size)
    datagen_env_vars["DATAGEN_NUM_NODES"] = str(exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1)
    gcs_credentials_path = os.environ.get("GCS_CREDENTIALS_PATH")
    if gcs_credentials_path:
        datagen_env_vars["GCS_CREDENTIALS_PATH"] = gcs_credentials_path
    if exp_args.get("datagen_engine") != engine:
        exp_args = update_exp_args(exp_args, {"datagen_engine": engine})

    if exp_args.get("datagen_input_dir"):
        datagen_env_vars["DATAGEN_INPUT_DIR"] = exp_args["datagen_input_dir"]

    output_dir = exp_args.get("datagen_output_dir")
    if not output_dir:
        output_dir = os.path.join(exp_args["experiments_dir"], "outputs", "tasks")
        exp_args = update_exp_args(exp_args, {"datagen_output_dir": output_dir})
    datagen_env_vars["DATAGEN_OUTPUT_DIR"] = output_dir

    datagen_extra_args = exp_args.get("datagen_extra_args") or ""
    extra_tokens: list[str]
    if datagen_extra_args:
        try:
            extra_tokens = shlex.split(datagen_extra_args)
        except ValueError:
            extra_tokens = datagen_extra_args.split()
    else:
        extra_tokens = []

    if exp_args.get("disable_verification"):
        disable_flags = {"--disable-verification", "--disable_verification"}
        if not any(token in disable_flags for token in extra_tokens):
            extra_tokens.append("--disable-verification")

    if extra_tokens:
        datagen_extra_args = " ".join(extra_tokens)
        datagen_env_vars["DATAGEN_EXTRA_ARGS"] = datagen_extra_args
        exp_args = update_exp_args(exp_args, {"datagen_extra_args": datagen_extra_args})

    datagen_env_vars["VLLM_JOB_ID"] = (str(vllm_job_id) if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id" else "")

    sandbox_overrides = {
        "SANDBOX_CPU": exp_args.get("sandbox_cpu"),
        "SANDBOX_MEMORY_GB": exp_args.get("sandbox_memory_gb"),
        "SANDBOX_DISK_GB": exp_args.get("sandbox_disk_gb"),
        "SANDBOX_GPU": exp_args.get("sandbox_gpu"),
    }
    for env_key, value in sandbox_overrides.items():
        if value is not None:
            datagen_env_vars[env_key] = str(value)

    # Set endpoint JSON path
    endpoint_path = exp_args.get("vllm_endpoint_json_path")
    requires_endpoint = engine == "vllm_local" and (script_gpu_required or is_ray_backend)
    if requires_endpoint and not endpoint_path:
        endpoint_path = os.path.join(exp_args["experiments_dir"], "vllm_endpoint.json")
        exp_args["vllm_endpoint_json_path"] = endpoint_path

    datagen_env_vars["DATAGEN_ENDPOINT_JSON"] = endpoint_path or ""
    datagen_env_vars["DATAGEN_REQUIRE_ENDPOINT"] = "1" if requires_endpoint else "0"
    datagen_wait_flag = exp_args.get("datagen_wait_for_endpoint")
    if datagen_wait_flag is None:
        datagen_wait_flag = "1" if requires_endpoint and backend != "ray" else "0"
    else:
        datagen_wait_flag = "1" if str(datagen_wait_flag).lower() in ("1", "true", "yes") else "0"
    datagen_env_vars["DATAGEN_WAIT_FOR_ENDPOINT"] = datagen_wait_flag

    config_path = _snapshot_datagen_config(exp_args)
    datagen_env_vars["DATAGEN_CONFIG_PATH"] = config_path

    # Prepare sbatch script
    sbatch_dir = os.path.join(os.path.dirname(__file__), "sbatch_data")
    hpc_name = getattr(hpc, "name", "") or ""
    uses_vllm_cluster = is_ray_backend or requires_endpoint
    datagen_sbatch_template = os.path.join(sbatch_dir, f"{hpc_name}_datagen.sbatch")

    if hpc_name.lower() == "vista":
        datagen_sbatch_template = os.path.join(
            sbatch_dir,
            "vista_datagen.sbatch" if uses_vllm_cluster else "vista_datagen_cpu.sbatch",
        )

    if not os.path.exists(datagen_sbatch_template):
        # Fallback to vista template
        datagen_sbatch_template = os.path.join(sbatch_dir, "vista_datagen.sbatch")

    # Create output sbatch script
    datagen_job_name = f"{exp_args['job_name']}_datagen"
    datagen_sbatch_output = os.path.join(
        exp_args["experiments_dir"],
        "sbatch",
        f"{datagen_job_name}.sbatch"
    )

    # Read template and substitute
    with open(datagen_sbatch_template) as f:
        sbatch_content = f.read()

    num_nodes = exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1
    if is_cpu_only:
        num_nodes = 1
    if hpc_name.lower() == "vista" and not uses_vllm_cluster and not exp_args.get("num_nodes"):
        num_nodes = 1
    gpus_per_node_directive = gpus_per_node
    if getattr(hpc, "name", "").lower() == "vista":
        gpus_per_node_directive = None

    cpus_per_node_effective = exp_args.get("cpus_per_node")
    if cpus_per_node_effective in (None, "", "None"):
        cpus_per_node_effective = getattr(hpc, "cpus_per_node", 1)
    cpus_per_node_effective = int(cpus_per_node_effective)

    substitutions = {
        "{partition}": hpc.partition,
        "{time_limit}": exp_args["time_limit"],
        "{num_nodes}": str(num_nodes),
        "{cpus_per_node}": str(cpus_per_node_effective),
        "{account}": hpc.account,
        "{experiments_dir}": exp_args["experiments_dir"],
        "{job_name}": datagen_job_name,
        "{num_gpus}": str(requested_gpus),
        "{gpus_per_node}": "" if gpus_per_node_directive is None else str(gpus_per_node_directive),
    }

    for key, value in substitutions.items():
        sbatch_content = sbatch_content.replace(key, value)

    if datagen_env_vars.get("DATAGEN_NUM_NODES") != str(num_nodes):
        datagen_env_vars["DATAGEN_NUM_NODES"] = str(num_nodes)

    if requested_gpus == 0:
        sbatch_content = sbatch_content.replace("#SBATCH --gres=gpu:0\n", "")
    if not gpus_per_node_directive:
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node {gpus_per_node}\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node 0\n", "")
        sbatch_content = sbatch_content.replace("#SBATCH --gpus-per-node=0\n", "")

    sbatch_content = _inject_env_block(sbatch_content, datagen_env_vars)

    # Ensure output directory exists and write output sbatch
    os.makedirs(os.path.dirname(datagen_sbatch_output), exist_ok=True)
    os.makedirs(os.path.join(exp_args["experiments_dir"], "logs"), exist_ok=True)
    with open(datagen_sbatch_output, 'w') as f:
        f.write(sbatch_content)

    print(f"Task generation sbatch: {datagen_sbatch_output}")
    print(f"Generation script: {exp_args['datagen_script']}")
    target_repo_display = exp_args.get("datagen_target_repo") or "<none>"
    print(f"Target repo: {target_repo_display}")
    print(f"Datagen model: {exp_args.get('datagen_model')}")

    # Submit job with dependency if VLLM job exists
    if not exp_args.get("dry_run"):
        dependency = None
        if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id":
            # Wait until the VLLM job begins running so the endpoint is ready
            dependency = f"after:{vllm_job_id}"
            print(f"Setting dependency on VLLM job start: {vllm_job_id}")

        job_id = launch_sbatch(datagen_sbatch_output, dependency=dependency)
        print(f"✓ Task generation job submitted: {job_id}")
        return job_id
    else:
        print("DRY RUN: Would submit task generation job")
        if vllm_job_id:
            print(f"  with dependency on VLLM job: {vllm_job_id}")
        return "dry_run_task_job_id"


def launch_trace_job(
    exp_args: dict,
    hpc,
    tasks_input_path: str,
    dependency: Optional[str] = None,
) -> Union[str, list[str]]:
    """Launch trace generation job."""

    print("\n=== Launching Trace Generation Job ===")

    trace_script = exp_args.get("trace_script") or exp_args.get("datagen_script")
    if not trace_script:
        raise ValueError("Trace generation requires --trace-script or --datagen-script")

    trace_target_repo = exp_args.get("trace_target_repo")
    if not trace_target_repo:
        raise ValueError("--trace-target-repo is required when enabling trace generation")

    trace_output_dir = exp_args.get("trace_output_dir")
    if not trace_output_dir:
        trace_output_dir = os.path.join(exp_args["experiments_dir"], "outputs", "traces")
        exp_args = update_exp_args(exp_args, {"trace_output_dir": trace_output_dir})

    dry_run_flag = bool(exp_args.get("dry_run"))

    chunk_size_raw = exp_args.get("chunk_size")
    chunk_size_value: Optional[int]
    if chunk_size_raw in (None, "", "None"):
        chunk_size_value = None
    elif isinstance(chunk_size_raw, int):
        chunk_size_value = chunk_size_raw
    else:
        try:
            chunk_size_value = int(chunk_size_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("--chunk_size must be an integer value") from exc
        exp_args = update_exp_args(exp_args, {"chunk_size": chunk_size_value})

    tasks_root = Path(tasks_input_path)
    if chunk_size_value and not tasks_root.exists():
        raise ValueError(
            "Chunked trace generation is not yet supported when task generation runs in the same job. "
            f"Tasks directory not found at {tasks_root}. Generate/download tasks first (without --chunk_size) "
            "and re-run traces with chunking enabled."
        )

    task_entries = _discover_task_entries(
        tasks_root,
        create_if_missing=bool(dependency) and not dry_run_flag,
    )
    total_tasks = len(task_entries)
    if total_tasks == 0:
        # Do not fail early; the downstream trace runner waits for tasks to appear.
        # This allows submitting a trace job even when tasks are not yet materialized.
        if dependency:
            print(
                "Trace tasks directory is empty; dependency on task job will ensure "
                "tasks populate before traces start."
            )
        else:
            print(
                f"No tasks currently found under {tasks_root}; submitting trace job that will wait for tasks."
            )
    else:
        print(f"Discovered {total_tasks} tasks for trace generation at {tasks_root}")

    trace_config_spec = exp_args.get("trace_harbor_config")
    if not trace_config_spec:
        raise ValueError("Trace generation requires --trace-harbor-config.")
    trace_config_path = resolve_harbor_config_path(trace_config_spec)
    base_trace_config = load_job_config(trace_config_path)

    trace_configs_dir = Path(exp_args["experiments_dir"]) / "configs" / "trace"
    trace_configs_dir.mkdir(parents=True, exist_ok=True)

    if "trace_jobs_dir" in exp_args and exp_args["trace_jobs_dir"]:
        trace_jobs_dir_path = Path(str(exp_args["trace_jobs_dir"])).expanduser().resolve()
    else:
        experiments_dir = Path(str(exp_args["experiments_dir"])).expanduser().resolve()
        trace_jobs_dir_path = experiments_dir / "trace_jobs"
        exp_args = update_exp_args(exp_args, {"trace_jobs_dir": str(trace_jobs_dir_path)})
    trace_jobs_dir_path.mkdir(parents=True, exist_ok=True)
    trace_jobs_dir = str(trace_jobs_dir_path)

    chunk_plans: list[TraceChunkPlan] = []
    chunk_map_path: Optional[Path] = None
    if chunk_size_value:
        chunk_plans, chunk_map_path = _prepare_trace_chunk_plans(
            tasks_root=tasks_root,
            task_entries=task_entries,
            chunk_size=chunk_size_value,
            trace_jobs_dir=trace_jobs_dir,
            trace_output_dir=trace_output_dir,
            trace_target_repo=trace_target_repo,
            dry_run=dry_run_flag,
        )
        if chunk_plans:
            print(f"Prepared chunk plan for {len(chunk_plans)} trace jobs.")
        else:
            print(
                f"Chunk size {chunk_size_value} does not require splitting "
                f"(total tasks: {total_tasks})."
            )

    cfg_agent = base_trace_config.agents[0] if base_trace_config.agents else None

    trace_agent_timeout_override = exp_args.get("trace_agent_timeout_sec")
    base_agent_kwargs = exp_args.get("_datagen_extra_agent_kwargs") or {}
    if not isinstance(base_agent_kwargs, dict):
        base_agent_kwargs = dict(base_agent_kwargs)
    trace_agent_kwargs: dict[str, Any] = dict(base_agent_kwargs)
    trace_agent_kwargs_raw = exp_args.get("trace_agent_kwargs")
    cli_agent_kwargs: dict[str, Any] | None = None
    if isinstance(trace_agent_kwargs_raw, dict):
        cli_agent_kwargs = dict(trace_agent_kwargs_raw)
    elif trace_agent_kwargs_raw not in (None, "", "None"):
        try:
            parsed_kwargs = json.loads(str(trace_agent_kwargs_raw))
        except json.JSONDecodeError as exc:
            raise ValueError("--trace_agent_kwargs must be valid JSON") from exc
        if not isinstance(parsed_kwargs, dict):
            raise ValueError("--trace_agent_kwargs must be a JSON object")
        cli_agent_kwargs = parsed_kwargs
    if cli_agent_kwargs is not None:
        trace_agent_kwargs.update(cli_agent_kwargs)

    trace_n_concurrent_override = exp_args.get("trace_n_concurrent")
    trace_env_override = exp_args.get("trace_env")
    trace_model_override = exp_args.get("trace_model")
    trace_agent_name_override = exp_args.get("trace_agent_name")

    trace_model = trace_model_override or (
        cfg_agent.model_name if cfg_agent and cfg_agent.model_name else None
    )
    if trace_model and str(trace_model).strip() == HARBOR_MODEL_PLACEHOLDER:
        trace_model = ""
    raw_trace_model = trace_model

    datagen_runtime = exp_args.get("_datagen_engine_runtime")
    runtime_engine_type = ""
    runtime_model_name: str | None = None
    if datagen_runtime is not None:
        runtime_engine_type = str(getattr(datagen_runtime, "type", "") or "").lower()
        runtime_model_name = (
            datagen_runtime.engine_kwargs.get("model")
            or datagen_runtime.engine_kwargs.get("model_name")
        )

    if (
        trace_model_override is None
        and runtime_model_name
        and runtime_engine_type not in ("vllm_local", "none")
    ):
        trace_model = runtime_model_name

    if not trace_model:
        vllm_cfg = exp_args.get("_datagen_vllm_server_config")
        if vllm_cfg:
            trace_model = vllm_cfg.model_path
    trace_model = trace_model or ""

    provider_prefix: str | None = None
    if runtime_engine_type == "openai":
        provider_prefix = "openai"
    elif runtime_engine_type == "anthropic":
        provider_prefix = "anthropic"
    elif runtime_engine_type in {"gemini_openai", "google_gemini"}:
        provider_prefix = "gemini"

    def _normalize_model_for_provider(model_value: str, provider: str | None) -> str:
        if not model_value:
            return ""
        candidate = str(model_value).strip()
        if provider and candidate.lower().startswith(f"{provider.lower()}/"):
            return candidate
        for prefix in (
            "openai/",
            "anthropic/",
            "gemini/",
            "google_gemini/",
        ):
            if candidate.lower().startswith(prefix):
                candidate = candidate[len(prefix):]
                break
        if candidate.lower().startswith("models/"):
            candidate = candidate[len("models/"):]
        candidate = candidate.lstrip("/")
        if provider:
            if not candidate:
                return ""
            return f"{provider}/{candidate}"
        return candidate

    trace_model_dispatch = _normalize_model_for_provider(trace_model, provider_prefix)

    if not trace_model_dispatch and runtime_model_name:
        trace_model_dispatch = _normalize_model_for_provider(runtime_model_name, provider_prefix)

    if not trace_model_dispatch:
        raise ValueError(
            "Unable to determine trace model. Provide --trace_model or specify engine.model in the datagen config."
        )

    if trace_model_dispatch == HARBOR_MODEL_PLACEHOLDER:
        raise ValueError(
            "Trace Harbor config still references placeholder/override-at-runtime. "
            "Ensure the datagen config supplies a concrete model."
        )

    raw_model_display = raw_trace_model or runtime_model_name or trace_model_dispatch
    if trace_model_dispatch != (raw_trace_model or runtime_model_name or ""):
        print(f"Trace model selected: {trace_model_dispatch} (raw: {raw_model_display})")
    else:
        print(f"Trace model selected: {trace_model_dispatch}")

    exp_args = update_exp_args(exp_args, {"trace_model": trace_model_dispatch})

    trace_agent_name = (
        trace_agent_name_override
        or (cfg_agent.name if cfg_agent and cfg_agent.name else None)
        or "terminus-2"
    )

    trace_episodes = exp_args.get("trace_episodes") or "last"
    trace_export_filter = exp_args.get("trace_export_filter") or "none"
    trace_dataset_type = exp_args.get("trace_dataset_type") or "SFT"
    disable_verification_flag = bool(exp_args.get("disable_verification"))
    trace_export_subagents_value = exp_args.get("trace_export_subagents")
    if trace_export_subagents_value in (None, "", "None"):
        trace_export_subagents = True
    elif isinstance(trace_export_subagents_value, str):
        trace_export_subagents = trace_export_subagents_value.strip().lower() not in {"0", "false", "no"}
    else:
        trace_export_subagents = bool(trace_export_subagents_value)

    trace_engine_value = exp_args.get("trace_engine") or exp_args.get("datagen_engine") or ""
    trace_engine = str(trace_engine_value).lower()

    trace_backend_value = exp_args.get("trace_backend") or exp_args.get("datagen_backend") or "vllm"
    trace_backend = str(trace_backend_value).lower()
    cpu_only_trace = trace_backend == "vllm_local"

    if trace_backend != trace_backend_value:
        exp_args = update_exp_args(exp_args, {"trace_backend": trace_backend})

    if trace_engine == "none" and trace_backend in ("vllm", "none"):
        trace_backend = "none"
        exp_args = update_exp_args(exp_args, {"trace_backend": trace_backend})

    requires_endpoint = trace_engine == "vllm_local" and trace_backend in ("vllm", "ray")

    trace_endpoint_json: str = ""
    if requires_endpoint:
        base_endpoint_path = exp_args.get("trace_endpoint_json") or exp_args.get("vllm_endpoint_json_path")
        if base_endpoint_path:
            trace_endpoint_json = base_endpoint_path
        else:
            experiments_dir = exp_args.get("experiments_dir")
            if not experiments_dir:
                raise ValueError("experiments_dir is required to compute trace endpoint path")
            trace_endpoint_json = default_vllm_endpoint_path(
                experiments_dir,
                trace=bool(chunk_plans),
            )
    else:
        trace_endpoint_json = exp_args.get("trace_endpoint_json") or exp_args.get("vllm_endpoint_json_path") or ""

    trace_use_gpu_flag = bool(exp_args.get("trace_use_gpu"))
    if cpu_only_trace and trace_use_gpu_flag:
        print("TRACE backend 'vllm_local' is CPU-only; ignoring --trace_use_gpu request.")
        trace_use_gpu_flag = False
        exp_args = update_exp_args(exp_args, {"trace_use_gpu": False})
    if requires_endpoint and not trace_use_gpu_flag:
        trace_use_gpu_flag = True
        exp_args = update_exp_args(exp_args, {"trace_use_gpu": True})

    num_gpus = hpc.gpus_per_node if trace_use_gpu_flag else 0
    trace_memory_limit_gb = 32 if cpu_only_trace else None

    trace_n_concurrent_effective = (
        trace_n_concurrent_override
        if trace_n_concurrent_override is not None
        else base_trace_config.orchestrator.n_concurrent_trials
    )
    if trace_n_concurrent_effective is None:
        trace_n_concurrent_effective = 8
    print(f"Trace concurrency selected: {trace_n_concurrent_effective}")
    if trace_env_override:
        print(f"Trace environment override: {trace_env_override}")

    task_type_value = exp_args.get("task_type")

    trace_env_vars = {
        "TRACE_SCRIPT": trace_script,
        "TRACE_STAGE": "traces",
        "TRACE_TASKS_PATH": tasks_input_path,
        "TRACE_TARGET_REPO": trace_target_repo,
        "TRACE_MODEL": trace_model_dispatch,
        "TRACE_ENGINE": trace_engine,
        "TRACE_OUTPUT_DIR": trace_output_dir,
        "TRACE_USE_GPU": "1" if trace_use_gpu_flag else "0",
        "TRACE_EPISODES": trace_episodes,
        "TRACE_EXPORT_FILTER": trace_export_filter,
        "TRACE_DATASET_TYPE": trace_dataset_type,
        "TRACE_EXPORT_SUBAGENTS": "1" if trace_export_subagents else "0",
        "TRACE_JOBS_DIR": trace_jobs_dir,
        "TRACE_ENDPOINT_JSON": trace_endpoint_json or "",
        "TRACE_REQUIRE_ENDPOINT": "1" if requires_endpoint else "0",
        "TRACE_WAIT_FOR_ENDPOINT": "1" if requires_endpoint else "0",
        "TRACE_HEALTH_MAX_ATTEMPTS": str(
            exp_args.get("trace_health_max_attempts", getattr(BaseDataGenerator, "HEALTHCHECK_MAX_ATTEMPTS", 20))
        ),
        "TRACE_HEALTH_RETRY_DELAY": str(
            exp_args.get("trace_health_retry_delay", getattr(BaseDataGenerator, "HEALTHCHECK_RETRY_DELAY", 30))
        ),
        "TRACE_DISABLE_VERIFICATION": "1" if disable_verification_flag else "0",
        "TRACE_EVAL_ONLY": "1" if exp_args.get("trace_eval_only") else "0",
        "VLLM_JOB_ID": str(exp_args.get("vllm_job_id", "")),
        "TRACE_NUM_GPUS": str(num_gpus),
        "TRACE_BACKEND": trace_backend,
        "TRACE_AGENT_TIMEOUT_SEC": str(trace_agent_timeout_override or ""),
        "TRACE_VERIFIER_TIMEOUT_SEC": str(exp_args.get("trace_verifier_timeout_sec") or ""),
        "TRACE_TOTAL_TASKS": str(total_tasks),
        "TRACE_CHUNK_SIZE": str(chunk_size_value or ""),
        "TRACE_CHUNK_COUNT": str(len(chunk_plans) if chunk_plans else 1),
        "TRACE_CHUNK_MAP_PATH": str(chunk_map_path) if chunk_map_path else "",
        "TRACE_JOB_INDEX": "0",
        "TRACE_TARGET_REPO_BASE": trace_target_repo,
        "TRACE_HARBOR_CONFIG": "",
    }
    if task_type_value:
        trace_env_vars["TRACE_TASK_TYPE"] = str(task_type_value)
    def _materialize_trace_config(
        dataset_path: Path,
        job_suffix: str,
        jobs_dir: Path | str,
        *,
        agent_name_override: str | None,
        agent_model_override: str | None,
        agent_timeout_override: float | None,
        agent_kwargs_override: dict[str, Any] | None,
        n_concurrent_override: int | None,
        env_override: str | None,
        disable_verification: bool,
    ) -> Path:
        config = set_local_dataset(base_trace_config, Path(dataset_path))
        config = set_job_metadata(
            config,
            job_name=f"{exp_args['job_name']}{job_suffix}",
            jobs_dir=jobs_dir,
        )
        config = overwrite_agent_fields(
            config,
            name=agent_name_override,
            model_name=agent_model_override or None,
            override_timeout_sec=agent_timeout_override,
        )
        if (
            config.agents
            and getattr(config.agents[0], "model_name", None) == HARBOR_MODEL_PLACEHOLDER
        ):
            raise ValueError(
                "Trace Harbor config still references placeholder/override-at-runtime after resolution."
            )
        config = update_agent_kwargs(config, agent_kwargs_override)
        if n_concurrent_override is not None:
            config.orchestrator.n_concurrent_trials = int(n_concurrent_override)
        if env_override:
            env_str = str(env_override).strip()
            try:
                env_enum = EnvironmentType(env_str.lower())
            except ValueError:
                try:
                    env_enum = EnvironmentType[env_str.upper()]
                except KeyError as exc:
                    raise ValueError(f"Invalid trace environment override: {env_override}") from exc
            config.environment.type = env_enum
        if disable_verification:
            config.verifier.disable = True
        output_path = trace_configs_dir / f"{config.job_name}.yaml"
        if not dry_run_flag:
            dump_job_config(config, output_path)
        return output_path

    trace_max_tokens = exp_args.get("trace_max_tokens")
    if trace_max_tokens in (None, "", "None"):
        trace_max_tokens = exp_args.get("datagen_max_tokens")
    if trace_max_tokens not in (None, "", "None"):
        trace_env_vars["TRACE_MAX_TOKENS"] = str(trace_max_tokens)

    # Overwrite trace health max_attempts and delay if environment variable is specified
    if "TRACE_HEALTH_MAX_ATTEMPTS" in os.environ:
        trace_env_vars["TRACE_HEALTH_MAX_ATTEMPTS"] = os.environ["TRACE_HEALTH_MAX_ATTEMPTS"]

    if "TRACE_HEALTH_RETRY_DELAY" in os.environ:
        trace_env_vars["TRACE_HEALTH_RETRY_DELAY"] = os.environ["TRACE_HEALTH_RETRY_DELAY"]

    backend = str(trace_env_vars["TRACE_BACKEND"]).lower()
    is_trace_ray_backend = backend == "ray"
    vllm_trace_env, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=False)
    vllm_cfg = exp_args.get("_datagen_vllm_server_config")
    backend_cfg = exp_args.get("_datagen_backend_config")
    trace_env_vars.update(vllm_trace_env)

    if dry_run_flag:
        trace_engine_config_path = exp_args.get("datagen_config_path") or exp_args.get(
            "_datagen_config_original_path"
        )
    else:
        trace_engine_config_path = _snapshot_datagen_config(exp_args)
    if trace_engine_config_path:
        trace_env_vars["TRACE_ENGINE_CONFIG_PATH"] = str(trace_engine_config_path)
    if is_trace_ray_backend:
        if exp_args.get("ray_cgraph_submit_timeout"):
            trace_env_vars["RAY_CGRAPH_submit_timeout"] = str(exp_args["ray_cgraph_submit_timeout"])
        if exp_args.get("ray_cgraph_get_timeout"):
            trace_env_vars["RAY_CGRAPH_get_timeout"] = str(exp_args["ray_cgraph_get_timeout"])
        if exp_args.get("ray_cgraph_max_inflight_executions"):
            trace_env_vars["RAY_CGRAPH_max_inflight_executions"] = str(
                exp_args["ray_cgraph_max_inflight_executions"]
            )
        _maybe_set_ray_cgraph_env(trace_env_vars)
    trace_env_vars["TRACE_RAY_PORT"] = str(exp_args.get("datagen_ray_port", 6379))
    trace_env_vars["TRACE_API_PORT"] = str(exp_args.get("datagen_api_port", 8000))
    tensor_parallel_size = None
    pipeline_parallel_size = None
    data_parallel_size = None
    if vllm_cfg:
        tensor_parallel_size = getattr(vllm_cfg, "tensor_parallel_size", None)
        pipeline_parallel_size = getattr(vllm_cfg, "pipeline_parallel_size", None)
        data_parallel_size = getattr(vllm_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None and backend_cfg:
        tensor_parallel_size = getattr(backend_cfg, "tensor_parallel_size", None)
    if pipeline_parallel_size is None and backend_cfg:
        pipeline_parallel_size = getattr(backend_cfg, "pipeline_parallel_size", None)
    if data_parallel_size is None and backend_cfg:
        data_parallel_size = getattr(backend_cfg, "data_parallel_size", None)
    if tensor_parallel_size is None:
        tensor_parallel_size = 1
    if pipeline_parallel_size is None:
        pipeline_parallel_size = 1
    if data_parallel_size is None:
        data_parallel_size = 1
    trace_env_vars["TRACE_TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    trace_env_vars["TRACE_PIPELINE_PARALLEL_SIZE"] = str(pipeline_parallel_size)
    trace_env_vars["TRACE_DATA_PARALLEL_SIZE"] = str(data_parallel_size)
    trace_env_vars["TRACE_NUM_NODES"] = str(exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1)
    trace_env_vars["TRACE_GPUS_PER_NODE"] = str(exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", num_gpus))

    sandbox_overrides = {
        "SANDBOX_CPU": exp_args.get("sandbox_cpu"),
        "SANDBOX_MEMORY_GB": exp_args.get("sandbox_memory_gb"),
        "SANDBOX_DISK_GB": exp_args.get("sandbox_disk_gb"),
        "SANDBOX_GPU": exp_args.get("sandbox_gpu"),
    }
    for env_key, value in sandbox_overrides.items():
        if value is not None:
            trace_env_vars[env_key] = str(value)

    daytona_api_key = os.environ.get("DAYTONA_API_KEY")
    if daytona_api_key:
        trace_env_vars["DAYTONA_API_KEY"] = daytona_api_key
    else:
        print(
            "Warning: DAYTONA_API_KEY not found in environment; trace jobs may fail to start Daytona sandboxes."
        )

    sbatch_dir = os.path.join(os.path.dirname(__file__), "sbatch_data")
    trace_sbatch_template = os.path.join(sbatch_dir, f"{hpc.name}_datatrace.sbatch")
    if not os.path.exists(trace_sbatch_template):
        trace_sbatch_template = os.path.join(sbatch_dir, "vista_datatrace.sbatch")

    with open(trace_sbatch_template) as f:
        trace_sbatch_template_text = f.read()

    partition = getattr(hpc, "partition", "") or ""
    account = getattr(hpc, "account", "") or ""
    cpus_per_task_raw = exp_args.get("cpus_per_node") or getattr(hpc, "cpus_per_node", 1)
    cpus_per_task = int(cpus_per_task_raw)
    if cpu_only_trace and cpus_per_task > 32:
        print("TRACE backend 'vllm_local' is CPU-only; limiting CPUs per node to 32.")
        cpus_per_task = 32
        exp_args = update_exp_args(exp_args, {"cpus_per_node": cpus_per_task})
    trace_env_vars["TRACE_CPUS_PER_NODE"] = str(cpus_per_task)
    num_nodes = exp_args.get("num_nodes") or getattr(hpc, "num_nodes", 1) or 1
    gpus_per_node_trace = exp_args.get("gpus_per_node") or getattr(hpc, "gpus_per_node", num_gpus)
    if cpu_only_trace:
        gpus_per_node_trace = 0
    gpus_per_node_trace_directive = gpus_per_node_trace
    if getattr(hpc, "name", "").lower() == "vista":
        gpus_per_node_trace_directive = None

    base_substitutions = {
        "{partition}": partition,
        "{time_limit}": exp_args["time_limit"],
        "{num_nodes}": str(num_nodes),
        "{cpus_per_node}": str(cpus_per_task),
        "{num_gpus}": str(num_gpus),
        "{gpus_per_node}": "" if gpus_per_node_trace_directive is None else str(gpus_per_node_trace_directive),
        "{account}": account,
        "{experiments_dir}": exp_args["experiments_dir"],
    }

    def _render_trace_sbatch_content(
        job_name: str,
        env_vars: dict[str, str],
        *,
        memory_limit_gb: int | None = None,
        array_spec: str | None = None,
    ) -> str:
        content = trace_sbatch_template_text
        for key, value in base_substitutions.items():
            content = content.replace(key, value)
        content = content.replace("{job_name}", job_name)
        if array_spec:
            job_line = f"#SBATCH --job-name {job_name}\n"
            array_line = f"#SBATCH --array={array_spec}\n"
            if job_line in content:
                content = content.replace(job_line, job_line + array_line, 1)
            else:
                if content.startswith("#!"):
                    newline_idx = content.find("\n")
                    if newline_idx == -1:
                        newline_idx = len(content)
                    content = (
                        content[: newline_idx + 1]
                        + array_line
                        + content[newline_idx + 1 :]
                    )
                else:
                    content = array_line + content
        if num_gpus == 0:
            content = content.replace("#SBATCH --gres=gpu:0\n", "")
        if not gpus_per_node_trace_directive:
            content = content.replace("#SBATCH --gpus-per-node {gpus_per_node}\n", "")
            content = content.replace("#SBATCH --gpus-per-node 0\n", "")
            content = content.replace("#SBATCH --gpus-per-node=0\n", "")
        if not partition:
            content = content.replace("#SBATCH -p \n", "")
        if not account:
            content = content.replace("#SBATCH --account \n", "")
        if memory_limit_gb is not None:
            memory_override = f"{memory_limit_gb}GB"
            existing_mem = re.search(
                r"^#SBATCH --mem=([0-9]+)([A-Za-z]+)",
                content,
                flags=re.MULTILINE,
            )
            if existing_mem:
                unit = existing_mem.group(2)
                try:
                    current_mem_value = int(existing_mem.group(1))
                except ValueError:
                    current_mem_value = None
                if current_mem_value is not None:
                    target_mem_value = min(current_mem_value, memory_limit_gb)
                    memory_override = f"{target_mem_value}{unit}"
            if re.search(r"^#SBATCH --mem=\S+", content, flags=re.MULTILINE):
                content = re.sub(
                    r"^#SBATCH --mem=\S+",
                    f"#SBATCH --mem={memory_override}",
                    content,
                    count=1,
                    flags=re.MULTILINE,
                )
            else:
                cpu_line_pattern = rf"(#SBATCH --cpus-per-task={cpus_per_task}\s*\n)"
                if re.search(cpu_line_pattern, content):
                    content = re.sub(
                        cpu_line_pattern,
                        r"\1#SBATCH --mem=" + memory_override + "\n",
                        content,
                        count=1,
                    )
                else:
                    content = content.replace(
                        "#SBATCH --ntasks-per-node 1\n",
                        "#SBATCH --ntasks-per-node 1\n" + f"#SBATCH --mem={memory_override}\n",
                        1,
                    )
            if not re.search(r"^#SBATCH --mem=\S+", content, flags=re.MULTILINE):
                content = f"#SBATCH --mem={memory_override}\n" + content
        return _inject_env_block(content, env_vars)

    def _write_chunk_env_file(path: Path, env_map: dict[str, Any]) -> None:
        lines: list[str] = []
        for key, value in env_map.items():
            if value is None:
                continue
            lines.append(f"export {key}={shlex.quote(str(value))}")
        lines.append("")
        path.write_text("\n".join(lines))

    sbatch_output_dir = os.path.join(exp_args["experiments_dir"], "sbatch")
    logs_dir = os.path.join(exp_args["experiments_dir"], "logs")
    os.makedirs(sbatch_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if chunk_plans:
        if chunk_map_path:
            print(f"Trace chunk map saved to: {chunk_map_path}")
        chunk_array_max = exp_args.get("_chunk_array_max")
        try:
            chunk_array_max = int(chunk_array_max) if chunk_array_max is not None else None
        except (TypeError, ValueError):
            chunk_array_max = None
        if not chunk_array_max or chunk_array_max <= 0:
            chunk_array_max = 3

        chunk_env_dir = trace_configs_dir / "chunk_envs"
        if dry_run_flag:
            print(f"DRY RUN: Would create chunk env directory at {chunk_env_dir}")
        else:
            if chunk_env_dir.exists():
                shutil.rmtree(chunk_env_dir)
            chunk_env_dir.mkdir(parents=True, exist_ok=True)

        original_endpoint_json_path = exp_args.get("vllm_endpoint_json_path")
        base_trace_engine_config_path = trace_env_vars.get("TRACE_ENGINE_CONFIG_PATH")

        for plan in chunk_plans:
            chunk_overrides: dict[str, Any] = {
                "TRACE_TASKS_PATH": str(plan.tasks_path),
                "TRACE_TARGET_REPO": plan.target_repo,
                "TRACE_OUTPUT_DIR": str(plan.output_dir),
                "TRACE_JOBS_DIR": str(plan.jobs_dir),
                "TRACE_JOB_INDEX": str(plan.index),
                "TRACE_CHUNK_COUNT": str(len(chunk_plans)),
                "JOB_SUBMIT_ORDER": str(plan.index + 1),
            }
            chunk_config_path = _materialize_trace_config(
                dataset_path=plan.tasks_path,
                job_suffix=f"_trace_{plan.index:03d}",
                jobs_dir=plan.jobs_dir,
                agent_name_override=trace_agent_name,
                agent_model_override=trace_model_dispatch or None,
                agent_timeout_override=trace_agent_timeout_override,
                agent_kwargs_override=trace_agent_kwargs,
                n_concurrent_override=trace_n_concurrent_effective,
                env_override=trace_env_override,
                disable_verification=disable_verification_flag,
            )
            chunk_overrides["TRACE_HARBOR_CONFIG"] = str(chunk_config_path)
            chunk_endpoint_json = (
                default_vllm_endpoint_path(
                    exp_args["experiments_dir"],
                    trace=True,
                    chunk_index=plan.index,
                )
                if requires_endpoint
                else ""
            )
            chunk_engine_config_path = base_trace_engine_config_path
            if chunk_endpoint_json:
                update_exp_args(
                    exp_args,
                    {"vllm_endpoint_json_path": chunk_endpoint_json},
                )
                chunk_engine_config_path = _snapshot_datagen_config(
                    exp_args,
                    output_filename=f"datagen_config_trace_{plan.index:03d}.resolved.yaml",
                    update_exp_args=False,
                )
                chunk_overrides["TRACE_ENDPOINT_JSON"] = chunk_endpoint_json
                chunk_overrides["VLLM_ENDPOINT_JSON_PATH"] = chunk_endpoint_json
            if chunk_engine_config_path:
                chunk_overrides["TRACE_ENGINE_CONFIG_PATH"] = str(chunk_engine_config_path)

            env_file = chunk_env_dir / f"chunk_{plan.index:03d}.env"
            if dry_run_flag:
                print(
                    f"DRY RUN: Would write chunk env {env_file} "
                    f"(tasks: {len(plan.task_names)}, repo: {plan.target_repo})"
                )
            else:
                _write_chunk_env_file(env_file, chunk_overrides)
            print(
                f"Prepared trace chunk {plan.index}: tasks={len(plan.task_names)} "
                f"repo={plan.target_repo} env_file={env_file}"
            )

        if original_endpoint_json_path is not None:
            update_exp_args(
                exp_args,
                {"vllm_endpoint_json_path": original_endpoint_json_path},
            )
        elif "vllm_endpoint_json_path" in exp_args:
            del exp_args["vllm_endpoint_json_path"]

        if dry_run_flag:
            return [f"dry_run_trace_job_id_{plan.index:03d}" for plan in chunk_plans]

        chunk_base_env = dict(trace_env_vars)
        chunk_base_env.update(
            {
                "TRACE_CHUNK_MODE": "1",
                "TRACE_CHUNK_ENV_DIR": str(chunk_env_dir),
                "TRACE_CHUNK_COUNT": str(len(chunk_plans)),
            }
        )

        array_limit = min(chunk_array_max, len(chunk_plans))
        array_spec = f"0-{len(chunk_plans) - 1}%{array_limit}"
        chunk_job_name = f"{exp_args['job_name']}_trace_chunks"
        chunk_sbatch_output = os.path.join(sbatch_output_dir, f"{chunk_job_name}.sbatch")
        sbatch_content = _render_trace_sbatch_content(
            chunk_job_name,
            chunk_base_env,
            memory_limit_gb=trace_memory_limit_gb,
            array_spec=array_spec,
        )
        with open(chunk_sbatch_output, "w") as f:
            f.write(sbatch_content)

        print(
            f"Trace chunk job array sbatch: {chunk_sbatch_output} "
            f"(chunks: {len(chunk_plans)}, array: {array_spec})"
        )

        job_id = launch_sbatch(chunk_sbatch_output, dependency=dependency, array=array_spec)
        print(f"✓ Trace chunk array job submitted: {job_id}")
        return [job_id]

    trace_config_output = _materialize_trace_config(
        dataset_path=tasks_root,
        job_suffix="_trace",
        jobs_dir=trace_jobs_dir_path,
        agent_name_override=trace_agent_name,
        agent_model_override=trace_model_dispatch or None,
        agent_timeout_override=trace_agent_timeout_override,
        agent_kwargs_override=trace_agent_kwargs,
        n_concurrent_override=trace_n_concurrent_effective,
        env_override=trace_env_override,
        disable_verification=disable_verification_flag,
    )
    trace_env_vars["TRACE_HARBOR_CONFIG"] = str(trace_config_output)

    trace_job_name = f"{exp_args['job_name']}_trace"
    trace_sbatch_output = os.path.join(sbatch_output_dir, f"{trace_job_name}.sbatch")
    sbatch_content = _render_trace_sbatch_content(
        trace_job_name,
        trace_env_vars,
        memory_limit_gb=trace_memory_limit_gb,
    )
    with open(trace_sbatch_output, "w") as f:
        f.write(sbatch_content)

    print(f"Trace generation sbatch: {trace_sbatch_output}")
    print(f"Trace script: {trace_script}")
    print(f"Trace target repo: {trace_target_repo}")

    if dry_run_flag:
        print("DRY RUN: Would submit trace generation job")
        return "dry_run_trace_job_id"

    job_id = launch_sbatch(trace_sbatch_output, dependency=dependency)
    print(f"✓ Trace generation job submitted: {job_id}")
    return job_id


def main():
    load_supabase_keys()
    print()
    # this is where defaults are stored for experiments_dir and deepspeed
    cli_args = parse_args()
    literal_none_keys = {"datagen_engine", "trace_engine", "datagen_backend", "trace_backend"}
    for key, value in cli_args.items():
        if isinstance(value, str):
            lowered = value.lower()
            if lowered == "false":
                cli_args[key] = False
            elif lowered == "true":
                cli_args[key] = True
            elif lowered == "none":
                if key not in literal_none_keys:
                    cli_args[key] = None
                else:
                    cli_args[key] = lowered
    numeric_fields = {
        "adam_beta1": float,
        "adam_beta2": float,
        "learning_rate": float,
        "warmup_ratio": float,
        "weight_decay": float,
        "max_grad_norm": float,
        "num_train_epochs": float,
        "max_steps": int,
        "chunk_size": int,
    }
    for key, caster in numeric_fields.items():
        if key not in cli_args or cli_args[key] is None:
            continue
        value = cli_args[key]
        if isinstance(value, (int, float)):
            continue
        try:
            cli_args[key] = caster(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Expected {key} to be {caster.__name__}-like, got {value!r}") from exc

    # Storing all the arguments in a dictionary that we add to in order of precedence
    exp_args = dict()

    # Add arguments to experiment from automatically detecting HPC
    hpc = detect_hpc()
    set_environment(hpc)
    _validate_sbatch_templates(hpc)

    # Add arguments and validate
    print()
    exp_args = update_exp_args(exp_args, hpc.model_dump())
    explicit_cli_keys = set(cli_args.get("_explicit_cli_keys", []))
    cli_args_filtered = {k: v for k, v in cli_args.items() if k != "_explicit_cli_keys"}
    exp_args = update_exp_args(exp_args, cli_args_filtered, explicit_keys=explicit_cli_keys)
    if explicit_cli_keys:
        exp_args["_explicit_cli_keys"] = list(explicit_cli_keys)

    def _parse_int(value, label: str) -> Optional[int]:
        if value in (None, "", "None"):
            return None
        if isinstance(value, bool):
            raise ValueError(f"{label} must be an integer, got boolean {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be an integer, got {value!r}") from exc

    gpus_per_node_norm = _parse_int(exp_args.get("gpus_per_node"), "--gpus_per_node") or 0
    exp_args = update_exp_args(exp_args, {"gpus_per_node": gpus_per_node_norm})

    cpus_per_node_norm = _parse_int(exp_args.get("cpus_per_node"), "--cpus_per_node")
    if cpus_per_node_norm is not None:
        exp_args = update_exp_args(exp_args, {"cpus_per_node": cpus_per_node_norm})

    cpus_per_gpu_norm = _parse_int(exp_args.get("cpus_per_gpu"), "--cpus_per_gpu")
    if cpus_per_gpu_norm is not None:
        exp_args = update_exp_args(exp_args, {"cpus_per_gpu": cpus_per_gpu_norm})

    cpus_per_node_cli_norm = _parse_int(cli_args_filtered.get("cpus_per_node"), "--cpus_per_node")
    cpus_per_gpu_cli_norm = _parse_int(cli_args_filtered.get("cpus_per_gpu"), "--cpus_per_gpu")

    if cpus_per_gpu_cli_norm is not None:
        if cpus_per_node_cli_norm is not None:
            raise ValueError("Provide only one of --cpus_per_node or --cpus_per_gpu, not both.")
        if gpus_per_node_norm <= 0:
            raise ValueError("--cpus_per_gpu requires --gpus_per_node to be greater than zero.")
        cpus_per_node_norm = cpus_per_gpu_cli_norm * gpus_per_node_norm
        exp_args = update_exp_args(
            exp_args,
            {
                "cpus_per_gpu": cpus_per_gpu_cli_norm,
                "cpus_per_node": cpus_per_node_norm,
            },
        )
        cpus_per_gpu_norm = cpus_per_gpu_cli_norm
    else:
        if cpus_per_node_norm is None and cpus_per_gpu_norm is not None and gpus_per_node_norm > 0:
            cpus_per_node_norm = cpus_per_gpu_norm * gpus_per_node_norm
            exp_args = update_exp_args(exp_args, {"cpus_per_node": cpus_per_node_norm})
        elif (
            cpus_per_node_norm is not None
            and (cpus_per_gpu_norm is None or cpus_per_gpu_norm == 0)
            and gpus_per_node_norm > 0
        ):
            derived_cpus_per_gpu = max(1, math.ceil(cpus_per_node_norm / gpus_per_node_norm))
            exp_args = update_exp_args(exp_args, {"cpus_per_gpu": derived_cpus_per_gpu})
            cpus_per_gpu_norm = derived_cpus_per_gpu

    if exp_args.get("use_mca"):
        mca_template = Path(__file__).parent / "sbatch" / f"{hpc.name.lower()}_train_mca.sbatch"
        if mca_template.exists():
            exp_args = update_exp_args(
                exp_args,
                {
                    "train_sbatch_filename": mca_template.name,
                    "train_sbatch_path": str(mca_template),
                },
            )
        else:
            print(
                f"Warning: MCA sbatch template {mca_template} not found for cluster {hpc.name}; using default template."
            )

    job_creator = str(exp_args.get("job_creator", "mlfoundations-dev") or "mlfoundations-dev").strip()
    if not job_creator:
        raise ValueError("--job_creator must be a non-empty string.")
    if len(job_creator) > 96:
        raise ValueError("--job_creator must be 96 characters or fewer.")
    exp_args = update_exp_args(exp_args, {"job_creator": job_creator})

    # Provide a default time_limit before template validation
    if exp_args.get("time_limit") in (None, "",):
        default_time = os.path.expandvars(os.environ.get("DEFAULT_TIME_LIMIT", "24:00:00"))
        exp_args = update_exp_args(exp_args, {"time_limit": default_time})
        print(f"Using default time_limit: {default_time}")

    job_type = str(exp_args.get("job_type", "train") or "train").lower()
    datagen_runtime = None
    if job_type == "datagen" or exp_args.get("datagen_script"):
        datagen_runtime = _prepare_datagen_configuration(exp_args)

    # Job name
    if "job_name" not in exp_args:
        exp_args["job_name"] = get_job_name(cli_args)
    # Fallback job names for special job types
    if job_type == "datagen" and not exp_args["job_name"]:
        datagen_engine_value = exp_args.get("datagen_engine")
        trace_engine_value = exp_args.get("trace_engine")
        parts = ["datagen", str(datagen_engine_value or trace_engine_value or "engine")]

        repo_candidate = (
            exp_args.get("datagen_target_repo")
            or exp_args.get("trace_target_repo")
        )
        if exp_args.get("datagen_model"):
            parts.append(str(exp_args["datagen_model"]).split("/")[-1])
        elif repo_candidate:
            sanitized_repo = sanitize_repo_for_job(repo_candidate) or str(repo_candidate).split("/")[-1]
            if sanitized_repo:
                parts.append(sanitized_repo)
        elif exp_args.get("trace_model"):
            parts.append(str(exp_args["trace_model"]).split("/")[-1])
        exp_args["job_name"] = "_".join(parts)
    elif job_type == "consolidate" and not exp_args["job_name"]:
        repo_id = exp_args.get("consolidate_repo_id") or "consolidate"
        exp_args["job_name"] = f"{sanitize_repo_for_job(repo_id)}_consolidate"
    print(f"Job name: {exp_args['job_name']}")

    if job_type == "consolidate":
        launch_consolidate_job(
            exp_args,
            hpc,
            update_exp_args_fn=update_exp_args,
            launch_sbatch_fn=launch_sbatch,
        )
        return

    # Check if this is a data generation job
    if job_type == "datagen":
        print("\n=== DATA GENERATION MODE ===")

        task_enabled = bool(exp_args.get("enable_task_gen", True))
        trace_enabled = bool(exp_args.get("enable_trace_gen", False))

        # Keep datagen/trace engine/backend selections in sync when only one is provided
        datagen_engine_value = exp_args.get("datagen_engine")
        trace_engine_value = exp_args.get("trace_engine")
        if datagen_engine_value is None and trace_engine_value is not None:
            exp_args = update_exp_args(exp_args, {"datagen_engine": trace_engine_value})
            datagen_engine_value = trace_engine_value
        elif trace_engine_value is None and datagen_engine_value is not None:
            exp_args = update_exp_args(exp_args, {"trace_engine": datagen_engine_value})
            trace_engine_value = datagen_engine_value

        datagen_backend_value = exp_args.get("datagen_backend")
        trace_backend_value = exp_args.get("trace_backend")
        if datagen_backend_value is None and trace_backend_value is not None:
            exp_args = update_exp_args(exp_args, {"datagen_backend": trace_backend_value})
            datagen_backend_value = trace_backend_value
        elif trace_backend_value is None and datagen_backend_value is not None:
            exp_args = update_exp_args(exp_args, {"trace_backend": datagen_backend_value})
            trace_backend_value = datagen_backend_value

        if not task_enabled and not trace_enabled:
            raise ValueError("Enable at least one of task or trace generation")

        if task_enabled and not exp_args.get("datagen_script"):
            raise ValueError("--datagen-script is required for task generation")

        datagen_extra_args_value = exp_args.get("datagen_extra_args")
        no_upload_requested = False
        if isinstance(datagen_extra_args_value, str):
            lowered = datagen_extra_args_value.replace("_", "-").lower()
            no_upload_requested = "--no-upload" in lowered
        elif isinstance(datagen_extra_args_value, (list, tuple)):
            normalized = [
                str(item).replace("_", "-").lower() for item in datagen_extra_args_value
            ]
            no_upload_requested = any("--no-upload" in token for token in normalized)

        if task_enabled and not exp_args.get("datagen_target_repo") and not no_upload_requested:
            raise ValueError("--datagen-target-repo is required for task generation (omit only when --no-upload is set)")

        task_job_id = None
        vllm_job_id = None

        tasks_output_dir = exp_args.get("datagen_output_dir")

        if task_enabled:
            engine = str(exp_args.get("datagen_engine") or "openai").lower()
            backend = str(exp_args.get("datagen_backend") or "vllm").lower()
            vllm_cfg = exp_args.get("_datagen_vllm_server_config")
            if vllm_cfg and engine == "vllm_local":
                if backend == "vllm":
                    print("Mode: Local VLLM inference for task generation (standalone server)")
                    vllm_job_id = launch_vllm_server(exp_args, hpc)
                    if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id":
                        exp_args = update_exp_args(exp_args, {"vllm_job_id": vllm_job_id})
                elif backend == "ray":
                    print("Mode: Ray-backed local VLLM inference for task generation")
                    _, exp_args = _build_vllm_env_vars(exp_args, include_pinggy=False)
                else:
                    raise ValueError(f"Unsupported datagen backend: {backend}")
            else:
                print(f"Mode: {engine} task generation")

            task_job_id = launch_task_job(exp_args, hpc, vllm_job_id)
            print("\n=== Task Generation Submitted ===")
            print(f"Task Job ID: {task_job_id}")
            if vllm_job_id:
                print(f"VLLM Server Job ID: {vllm_job_id}")
            tasks_output_dir = exp_args.get("datagen_output_dir")
            if tasks_output_dir:
                print(f"Task output directory: {tasks_output_dir}")

        trace_job_id = None
        trace_requires_vllm_server = False

        if trace_enabled:
            trace_script = exp_args.get("trace_script") or exp_args.get("datagen_script")
            if not trace_script:
                raise ValueError("Trace generation requires --trace-script (or reuse --datagen-script)")

            trace_input_path = exp_args.get("trace_input_path")
            if not trace_input_path:
                trace_input_path = tasks_output_dir
            if trace_input_path:
                exp_args = update_exp_args(exp_args, {"trace_input_path": trace_input_path})
            if not trace_input_path:
                raise ValueError("Trace generation requires --trace-input-path when task generation is disabled")

            if not task_enabled and not os.path.exists(trace_input_path):
                raise FileNotFoundError(f"Trace input path not found: {trace_input_path}")

            if exp_args.get("trace_use_gpu") and getattr(hpc, "gpus_per_node", 0) == 0:
                raise ValueError("trace_use_gpu requested but HPC configuration has no GPUs available")

            trace_backend = str(exp_args.get("trace_backend") or exp_args.get("datagen_backend") or "vllm").lower()
            trace_engine = str(exp_args.get("trace_engine") or exp_args.get("datagen_engine") or "").lower()
            trace_requires_vllm_server = trace_engine == "vllm_local" and trace_backend == "vllm"

            vllm_cfg = exp_args.get("_datagen_vllm_server_config")
            if trace_requires_vllm_server and not vllm_job_id and vllm_cfg:
                print("Trace stage requires a standalone VLLM server – launching now")
                vllm_job_id = launch_vllm_server(exp_args, hpc)
                if vllm_job_id and vllm_job_id != "dry_run_vllm_job_id":
                    exp_args = update_exp_args(exp_args, {"vllm_job_id": vllm_job_id})

            dependency = None
            if task_enabled and task_job_id and task_job_id != "dry_run_task_job_id":
                dependency = f"afterany:{task_job_id}"

            trace_job_id = launch_trace_job(exp_args, hpc, trace_input_path, dependency=dependency)
            print("\n=== Trace Generation Submitted ===")
            if isinstance(trace_job_id, list):
                print(f"Trace Job IDs: {', '.join(trace_job_id)}")
            else:
                print(f"Trace Job ID: {trace_job_id}")
            print(f"Trace input path: {trace_input_path}")

        return  # Skip normal training flow

    # Pre-validation
    pre_validation(exp_args, cli_args)

    # Construct the config yaml
    print()
    train_config, train_config_path_out = construct_config_yaml(exp_args)
    exp_args = update_exp_args(exp_args, train_config)
    exp_args = update_exp_args(
        exp_args, {"train_config_path_out": train_config_path_out}
    )
    write_run_summary(exp_args, train_config)

    # Construct the sbatch script
    print()
    train_sbatch_path_out = construct_sbatch_script(exp_args)
    exp_args = update_exp_args(
        exp_args, {"train_sbatch_path_out": train_sbatch_path_out}
    )

    display_args(exp_args, "Train")
    if exp_args.get("dry_run", False):
        print(
            "DRY RUN: Job would be submitted with the above parameters, but --dry_run flag was set."
        )
    else:
        dependency = None
        if exp_args.get("pretokenize"):
            if os.path.exists(exp_args["tokenized_path"]):
                print(f"Tokenized directory {exp_args['tokenized_path']} already exists, skipping pretokenization job submission")
            else:
                pretok_job_id = schedule_pretokenize(exp_args)
                dependency = f"afterok:{pretok_job_id}"

        train_job_id = submit_job(
            exp_args=exp_args,
            dependency=dependency,
        )

        if exp_args.get("eval_tasks"):
            if exp_args.get("internet_node", False):
                print()
                schedule_eval(exp_args, train_job_id)
            else:
                print("Skipping evaluation because internet_node is False")

if __name__ == "__main__":
    main()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
