from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from hpc.arguments import LlamaFactoryArgs
from hpc.data_argument_keys import DATA_ARGUMENT_KEYS
from hpc.launch_utils import (
    resolve_job_and_paths,
    substitute_template,
    build_sbatch_directives,
    coerce_positive_int,
    parse_bool_with_default,
)


def apply_mca_training_template(
    exp_args: dict,
    hpc,
    *,
    update_exp_args_fn: Callable[[dict, dict], dict],
) -> dict:
    """Point training jobs at the MCA-specific sbatch template when requested."""

    mca_template = Path(__file__).parent / "sbatch" / f"{hpc.name.lower()}_train_mca.sbatch"
    if mca_template.exists():
        return update_exp_args_fn(
            exp_args,
            {
                "train_sbatch_filename": mca_template.name,
                "train_sbatch_path": str(mca_template),
            },
        )

    print(
        f"Warning: MCA sbatch template {mca_template} not found for cluster {hpc.name}; using default template."
    )
    return exp_args


def build_training_parameters_link(hub_model_id: Optional[str]) -> Optional[str]:
    if not hub_model_id:
        return None
    hub_model_id = hub_model_id.strip("/")
    return f"https://huggingface.co/{hub_model_id}/blob/main/config.json"


def ensure_deepspeed_config(base_config: dict, exp_args: dict) -> dict:
    """Pass through DeepSpeed config if explicitly set in YAML or CLI.

    Does NOT inject a default — if neither the YAML nor CLI specifies
    deepspeed, the job uses FSDP via accelerate instead.
    """
    # CLI override takes priority
    if exp_args.get("deepspeed"):
        base_config["deepspeed"] = exp_args["deepspeed"]
    # Otherwise, keep whatever the base YAML had (including None/absent)
    return base_config


def maybe_compute_gradient_accumulation(base_config: dict, exp_args: dict) -> dict:
    num_nodes = coerce_positive_int(exp_args.get("num_nodes"), 1)
    num_gpus = coerce_positive_int(exp_args.get("gpus_per_node"), 1)

    # Extract global_batch_size from exp_args or base_config
    raw_global_batch_size = exp_args.pop("global_batch_size", None)
    if raw_global_batch_size is None:
        raw_global_batch_size = base_config.pop("global_batch_size", None)
    else:
        base_config.pop("global_batch_size", None)

    if raw_global_batch_size is None:
        print("\nSkipping automatic gradient accumulation calculation because global_batch_size was not provided.")
        return base_config

    global_batch_size = coerce_positive_int(raw_global_batch_size, 0)
    if global_batch_size <= 0:
        raise ValueError(f"Expected positive global_batch_size, got {raw_global_batch_size!r}")

    total_gpu_count = num_nodes * num_gpus

    # Model parallelism settings
    tensor_model_parallel_size = coerce_positive_int(base_config.get("tensor_model_parallel_size"), 1)
    pipeline_model_parallel_size = coerce_positive_int(base_config.get("pipeline_model_parallel_size"), 1)
    expert_model_parallel_size = coerce_positive_int(base_config.get("expert_model_parallel_size"), 1)
    sequence_parallel_size = coerce_positive_int(base_config.get("sequence_parallel_size"), 1)

    model_parallel_world_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * expert_model_parallel_size
        * sequence_parallel_size
    )

    if total_gpu_count % model_parallel_world_size != 0:
        print(
            f"Warning: total GPU count ({total_gpu_count}) is not divisible by model parallel size "
            f"({model_parallel_world_size}). Rounding down data parallel replicas."
        )
    data_parallel_replicas = max(total_gpu_count // model_parallel_world_size, 1)

    per_device_train_batch_size = coerce_positive_int(base_config.get("per_device_train_batch_size"), 1)

    effective_batch_denom = per_device_train_batch_size * data_parallel_replicas
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
    # base_config["global_batch_size"] = global_batch_size
    print(f"\nCalculated based on {num_nodes} nodes, {num_gpus} GPUs per node, and global batch size {global_batch_size}:")
    print(f"data_parallel_replicas: {data_parallel_replicas}")
    print(f"per_device_train_batch_size: {per_device_train_batch_size}")
    print(f"gradient_accumulation_steps: {gradient_accumulation_steps}")
    return base_config


def prebuild_arrow_cache(base_config: dict, train_config_path: str = "") -> None:
    """Pre-build HF datasets arrow cache AND LlamaFactory tokenization cache.

    When training on multi-node no-internet clusters (Jupiter, Leonardo), all
    ranks race to build the arrow cache simultaneously on shared NFS, causing
    ``FileNotFoundError`` when one rank reads a partially-written ``.arrow``
    file from another, or NCCL heartbeat timeouts when fast ranks enter
    collectives while slow ranks are still tokenizing.

    This runs LlamaFactory's actual data pipeline (single process, CPU-only)
    on the login node before SLURM submission.  The resulting ``.map()`` cache
    files will be found by all compute ranks, skipping tokenization entirely.
    """
    dataset_path = base_config.get("dataset", "")
    cache_dir = base_config.get("datasets_cache_dir", "")

    if not dataset_path or not cache_dir:
        return

    # Only pre-build for local/resolved paths
    dataset_paths = [p.strip() for p in dataset_path.split(",") if p.strip()]
    local_paths = [p for p in dataset_paths if os.path.isdir(p)]

    if not local_paths:
        return

    if not train_config_path:
        print("[arrow-cache] No train_config_path provided, skipping pre-build.")
        return

    print(f"[arrow-cache] Pre-building tokenization cache via LlamaFactory pipeline...")
    os.makedirs(cache_dir, exist_ok=True)

    try:
        # Force CPU-only and single-process
        prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # Prevent distributed init
        prev_world = os.environ.get("WORLD_SIZE")
        os.environ.pop("WORLD_SIZE", None)
        os.environ.pop("RANK", None)
        os.environ.pop("LOCAL_RANK", None)

        # Run LlamaFactory's data loading with the actual training config.
        # This uses the real template, tokenizer, and preprocessing — the
        # exact same .map() calls that training will use — so the cache
        # fingerprints match and compute nodes skip tokenization.
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import os; os.environ['CUDA_VISIBLE_DEVICES']=''; "
                "from llamafactory.hparams import get_train_args; "
                "from llamafactory.data.loader import get_dataset; "
                "from transformers import AutoTokenizer; "
                f"model_args, data_args, training_args, finetuning_args, gen_args = "
                f"get_train_args(['--config', '{train_config_path}']); "
                "tokenizer = AutoTokenizer.from_pretrained("
                "  model_args.model_name_or_path, trust_remote_code=True); "
                "ds = get_dataset(model_args, data_args, training_args, "
                "  stage='sft', tokenizer=tokenizer, processor=None); "
                "print(f'Cache built: {len(ds[\"train_dataset\"])} examples')"
            ],
            capture_output=True, text=True, timeout=600,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        if result.returncode == 0:
            print(f"[arrow-cache] {result.stdout.strip()}")
        else:
            # Common failure: bf16 validation on CPU. Fall back to raw cache only.
            stderr_short = result.stderr.strip().split("\n")[-3:]
            print(f"[arrow-cache] LlamaFactory pipeline failed (non-fatal):")
            for line in stderr_short:
                print(f"[arrow-cache]   {line}")
            print("[arrow-cache] Falling back to raw dataset cache only.")

            from datasets import load_dataset
            for ds_path in local_paths:
                ds_name = os.path.basename(ds_path)[:50]
                print(f"[arrow-cache]   Loading raw: {ds_name}...")
                load_dataset(ds_path, cache_dir=cache_dir)
            print("[arrow-cache] Raw cache built.")

        # Restore env
        if prev_cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        if prev_world is not None:
            os.environ["WORLD_SIZE"] = prev_world

    except Exception as exc:
        print(f"[arrow-cache] WARNING: Pre-build failed ({exc}). "
              "Training may work but tokenization will race on compute nodes.")


def apply_data_argument_overrides(base_config: dict, exp_args: dict) -> None:
    tool_call_tag = exp_args.get("tool_call_tag")
    if tool_call_tag:
        base_config["tools"] = tool_call_tag

    for tag in DATA_ARGUMENT_KEYS:
        if tag in exp_args:
            tag_value = exp_args[tag]
            if tag_value is not None:
                base_config[tag] = tag_value


# Models that require a dedicated conda environment with transformers v5+
# and specialized kernels (e.g., flash-linear-attention for Gated DeltaNet).
_MODELS_REQUIRING_SPECIAL_ENV = {
    "qwen3.5": "sft-qwen35",
    "qwen3_5": "sft-qwen35",
}


def _get_sft_conda_activate(hpc, exp_args: dict) -> str:
    """Determine the conda activation command for SFT jobs.

    Priority:
    1. --conda_env CLI override (via resolve_conda_activate)
    2. Model-specific env from _MODELS_REQUIRING_SPECIAL_ENV
    3. Default HPC conda_activate
    """
    from hpc.launch_utils import resolve_conda_activate

    # 1. Explicit CLI override takes priority
    if exp_args.get("conda_env"):
        return resolve_conda_activate(hpc, exp_args)

    # 2. Model-specific auto-detection
    model_name = str(exp_args.get("model_name_or_path") or exp_args.get("_original_model_name_or_path") or "").lower()

    for pattern, env_name in _MODELS_REQUIRING_SPECIAL_ENV.items():
        if pattern in model_name:
            print(f"[conda] Model '{model_name}' requires special env '{env_name}'")
            if hpc.conda_activate and "conda.sh" in hpc.conda_activate:
                conda_sh = hpc.conda_activate.split("&&")[0].strip()
                return f"{conda_sh} && conda activate {env_name}"
            conda_prefix = os.environ.get("CONDA_PREFIX", "")
            if conda_prefix:
                import re
                base = re.sub(r"/envs/[^/]+$", "", conda_prefix)
                return f"source {base}/etc/profile.d/conda.sh && conda activate {env_name}"
            return f"conda activate {env_name}"

    return hpc.conda_activate or "# No conda activation configured"


def maybe_apply_cluster_specific_env_overrides(exp_args: dict, hpc) -> dict:
    """Inject cluster-specific defaults into exp_args when the user hasn't set them."""

    if hpc is None:
        return exp_args

    hpc_name = str(getattr(hpc, "name", "") or "").lower()
    explicit_cli_keys = set(exp_args.get("_explicit_cli_keys", []) or [])

    def _set_default(key: str, value):
        if key in explicit_cli_keys:
            return
        if exp_args.get(key) is None:
            exp_args[key] = value

    if hpc_name == "capella":
        _set_default("data_shared_file_system", True)

    return exp_args


def configure_sft_reporting(base_config: dict, exp_args: dict, model_path: str) -> dict:
    """Configure wandb reporting and push_to_hub for SFT training.

    Args:
        base_config: LlamaFactory training configuration dict
        exp_args: Experiment arguments from CLI
        model_path: Path to the model (used for no-internet clusters)

    Returns:
        Updated base_config with reporting settings
    """
    # Default: push only if cluster has direct internet. SSH tunnels/proxychains
    # on no-internet clusters (Jupiter, Leonardo) often can't reach HF Hub API
    # during trainer init, causing ConnectionError crashes.
    # Only enable push_to_hub on no-internet clusters if explicitly set via CLI.
    has_direct_internet = exp_args.get("internet_node", False)
    default_push = has_direct_internet
    cli_push = exp_args.get("push_to_hub")
    yaml_push = base_config.get("push_to_hub")
    if cli_push is not None:
        # Explicit CLI flag always wins
        push_to_hub = parse_bool_with_default(cli_push, default_push)
    elif has_direct_internet and yaml_push is not None:
        # On internet nodes, respect the YAML config
        push_to_hub = parse_bool_with_default(yaml_push, default_push)
    else:
        # On no-internet nodes, default to False regardless of YAML
        push_to_hub = default_push

    if exp_args.get("internet_node"):
        base_config["report_to"] = "wandb"
        base_config["push_to_hub"] = push_to_hub
    else:
        base_config.pop("report_to", None)
        base_config["push_to_hub"] = push_to_hub
        base_config["model_name_or_path"] = model_path
        # Use a dedicated arrow cache dir alongside HF_HUB_CACHE (not inside it) to avoid
        # datasets>=4.7.0 cache resolution bugs while keeping arrow caches off /tmp.
        _hf_cache = os.environ.get("HF_HUB_CACHE", "")
        base_config["datasets_cache_dir"] = os.path.join(os.path.dirname(_hf_cache), "arrow_cache") if _hf_cache else ""
    return base_config


# Templates that use LLaMA-Factory's ReasoningTemplate and need thinking preprocessing.
# Other templates (e.g. qwen3_nothink, qwen2_5, chatml) do NOT use ReasoningTemplate.
_REASONING_TEMPLATES = {"qwen3", "qwen3_5"}

# Mapping from ReasoningTemplate names to their non-thinking counterparts.
_NOTHINK_TEMPLATE_MAP = {"qwen3": "qwen3_nothink", "qwen3_5": "qwen3_5_nothink"}

# Threshold: if fewer than this fraction of assistant messages contain real
# <think> content, automatically switch to the _nothink template variant.
_THINKING_RATE_THRESHOLD = 0.5

# How many parquet rows to sample when estimating thinking rate.
_THINKING_SAMPLE_ROWS = 500


def _estimate_thinking_rate(
    dataset_paths: list[str],
    role_tag: str = "role",
    content_tag: str = "content",
    conversations_col: str = "conversations",
) -> tuple[float, int, int]:
    """Estimate the fraction of assistant messages with real <think> content.

    Samples up to ``_THINKING_SAMPLE_ROWS`` rows from the first parquet file
    in each dataset path and checks assistant messages for non-empty thinking
    blocks.

    Returns:
        (thinking_rate, n_with_thinking, n_total_assistant_msgs)
    """
    import re

    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    n_real_thinking = 0
    n_total = 0

    try:
        import pyarrow.parquet as pq
    except ImportError:
        # Can't sample without pyarrow; assume thinking to be safe
        return 1.0, 0, 0

    for ds_path in dataset_paths:
        if not os.path.isdir(ds_path):
            continue
        # Find parquet files (may be in data/ subdirectory for HF snapshots)
        parquet_files = sorted(Path(ds_path).rglob("*.parquet"))
        for pq_file in parquet_files:
            try:
                tbl = pq.read_table(
                    str(pq_file),
                    columns=[conversations_col],
                )
                rows = tbl.to_pydict().get(conversations_col, [])
                for conv in rows[:_THINKING_SAMPLE_ROWS]:
                    for msg in (conv or []):
                        if not isinstance(msg, dict):
                            continue
                        if msg.get(role_tag) != "assistant":
                            continue
                        text = msg.get(content_tag, "")
                        if not isinstance(text, str):
                            continue
                        n_total += 1
                        m = think_pattern.search(text)
                        if m and m.group(1).strip():
                            n_real_thinking += 1
            except Exception:
                continue

    rate = n_real_thinking / n_total if n_total > 0 else 0.0
    return rate, n_real_thinking, n_total


def _apply_nothink_config_overrides(base_config: dict, exp_args: dict) -> None:
    """Load a ``_nothink`` config variant and apply differing values.

    When auto-detection switches from a thinking to a nothink template, the
    corresponding ``_nothink`` YAML file may contain different hyperparameters
    (e.g. ``save_steps``).  This function derives the nothink config path from
    the original ``train_config_path``, loads it, and merges any values that
    differ from the current config — *except* keys that are managed by the
    launch pipeline (``template``, ``dataset``, ``dataset_dir``).
    """
    original_path = exp_args.get("train_config_path")
    if not original_path:
        return

    original = Path(original_path)
    # Derive nothink path: foo.yaml -> foo_nothink.yaml
    nothink_path = original.with_stem(original.stem + "_nothink")
    if not nothink_path.exists():
        return

    try:
        with open(nothink_path, "r") as f:
            nothink_config = yaml.safe_load(f.read()) or {}
    except Exception:
        return

    # Keys managed by the pipeline — never override from the nothink file.
    _SKIP_KEYS = {"template", "dataset", "dataset_dir"}
    applied: list[str] = []
    for key, value in nothink_config.items():
        if key in _SKIP_KEYS:
            continue
        if base_config.get(key) != value:
            base_config[key] = value
            applied.append(key)

    if applied:
        print(
            f"[prep_for_thinking] Applied overrides from {nothink_path.name}: "
            f"{', '.join(applied)}"
        )


def maybe_preprocess_thinking(
    base_config: dict,
    exp_args: dict,
    artifacts,
):
    """Preprocess datasets for Qwen3 ReasoningTemplate thought_words format.

    When the training template uses ReasoningTemplate (e.g. ``qwen3``), the
    thought_words ``("<think>\\n", "\\n</think>\\n\\n")`` must be present in
    assistant messages for proper loss masking.  This step normalises diverse
    input formats (``<think>content</think>``, orphaned ``</think>``, no tags,
    etc.) into the canonical format **before** LlamaFactory sees the data.

    **Auto-detection:** If the majority of assistant messages do *not* contain
    real ``<think>`` content (threshold: {threshold}%), the template is
    automatically switched to the ``_nothink`` variant (e.g.
    ``qwen3`` → ``qwen3_nothink``).  This avoids training the model on
    hundreds of thousands of empty ``<think>\\n\\n</think>`` blocks, which
    degrades reasoning ability.

    For non-ReasoningTemplate templates, a warning is emitted if the dataset
    appears to contain ``<think>`` tags, since those tags will be treated as
    plain text and the model may learn an unintended output format.

    If the template does not require preprocessing, ``artifacts`` is returned
    unchanged.
    """.format(threshold=int(_THINKING_RATE_THRESHOLD * 100))
    from hpc.arguments import JobType

    template = base_config.get("template", "")

    job_type = exp_args.get("job_type")
    if job_type and job_type not in (JobType.SFT.value, None):
        return artifacts

    agent_name = (
        exp_args.get("trace_agent_name")
        or exp_args.get("agent")
        or "terminus-2"
    )

    if template not in _REASONING_TEMPLATES:
        # Warn if the dataset likely contains thinking tags but the template
        # won't handle them with ReasoningTemplate.
        _warn_if_thinking_data_with_plain_template(template, artifacts)
        return artifacts

    if agent_name != "terminus-2":
        # TODO: support prep_for_thinking for non-terminus-2 harnesses.
        print(
            "[prep_for_thinking] Skipping preprocessing for non-terminus-2 agent "
            f"'{agent_name}'."
        )
        return artifacts

    from huggingface_hub import snapshot_download

    role_tag = exp_args.get("role_tag", "role")
    content_tag = exp_args.get("content_tag", "content")

    # Ensure datasets are local before sampling
    local_paths: list[str] = []
    for ds_path in artifacts.dataset_paths:
        if not os.path.isdir(ds_path):
            print(f"[prep_for_thinking] Downloading {ds_path} for preprocessing...")
            ds_path = snapshot_download(repo_id=ds_path, repo_type="dataset")
        local_paths.append(ds_path)

    # Auto-detect: should we use thinking or nothink template?
    thinking_rate, n_think, n_total = _estimate_thinking_rate(
        local_paths, role_tag=role_tag, content_tag=content_tag,
    )
    print(
        f"[prep_for_thinking] Thinking rate: {thinking_rate:.1%} "
        f"({n_think}/{n_total} sampled assistant messages have real <think> content)"
    )

    # Decide template based on thinking rate, but always preprocess below.
    use_nothink = False
    if thinking_rate < _THINKING_RATE_THRESHOLD:
        nothink_template = _NOTHINK_TEMPLATE_MAP.get(template)
        if nothink_template:
            print(
                f"[prep_for_thinking] Majority of data has no thinking blocks "
                f"({thinking_rate:.1%} < {_THINKING_RATE_THRESHOLD:.0%} threshold). "
                f"Switching template: {template} -> {nothink_template}"
            )
            base_config["template"] = nothink_template
            use_nothink = True
            # Try to load the _nothink config variant for any additional
            # overrides (e.g. different save_steps, hyperparams).
            _apply_nothink_config_overrides(base_config, exp_args)
        else:
            print(
                f"[prep_for_thinking] WARNING: No nothink variant for template "
                f"'{template}'. Proceeding with thinking preprocessing."
            )

    # Always preprocess: normalises format, captures free-text reasoning,
    # and prints per-dataset stats regardless of template choice.
    print(f"[prep_for_thinking] Preprocessing data for template '{base_config['template']}'")
    from scripts.datagen.prep_for_thinking import preprocess_local_dataset

    new_paths: list[str] = []
    for ds_path in local_paths:
        processed_path = preprocess_local_dataset(
            ds_path,
            role_tag=role_tag,
            content_tag=content_tag,
        )
        new_paths.append(processed_path)

    new_dataset_path = new_paths[0] if new_paths else artifacts.dataset_path
    # Force the config to use the local preprocessed paths (even on internet
    # nodes where LlamaFactory would otherwise load from HF Hub directly).
    # The cache_dir bug with datasets>=4.7.0 is fixed in LlamaFactory's
    # loader.py (skip cache_dir for local directory paths).
    base_config["dataset"] = ",".join(new_paths)
    base_config["dataset_dir"] = "ONLINE"

    # Return a new artifacts object with updated paths.  We reconstruct the
    # same dataclass the caller passed in so we don't need to import it here.
    return type(artifacts)(
        dataset_paths=new_paths,
        dataset_path=new_dataset_path,
        model_path=artifacts.model_path,
    )


def _warn_if_thinking_data_with_plain_template(template: str, artifacts) -> None:
    """Emit a warning if the dataset appears to contain <think> tags but the
    template is not a ReasoningTemplate.  This is a common misconfiguration
    that results in the model learning to produce ``<think>`` as literal text
    without proper loss masking or tokenization."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return  # best-effort; skip if pyarrow not available

    for ds_path in artifacts.dataset_paths:
        if not os.path.isdir(ds_path):
            continue
        # Quick check: read first parquet file and look for <think> in a sample
        parquet_files = sorted(Path(ds_path).rglob("*.parquet"))
        for pq_file in parquet_files[:1]:  # only check first file
            try:
                tbl = pq.read_table(str(pq_file), columns=["conversations"])
                sample = tbl.to_pydict().get("conversations", [])[:5]
                for conv in sample:
                    for msg in (conv or []):
                        content = msg.get("content", "") if isinstance(msg, dict) else ""
                        if "<think>" in content or "</think>" in content:
                            print(
                                f"\n*** WARNING: Dataset at {ds_path} contains <think> tags "
                                f"but template '{template}' is not a ReasoningTemplate. "
                                f"Thinking tokens will NOT be automatically converted to "
                                f"'{template}' format. This can cause training/inference "
                                f"incompatibilities. Consider using template 'qwen3' or "
                                f"pre-processing the dataset with:\n"
                                f"  python -m scripts.datagen.prep_for_thinking "
                                f"--source <dataset> --dry-run\n"
                            )
                            return
            except Exception:
                continue


def pre_validation_sft(cli_args: dict) -> None:
    """Validate SFT experiment configuration before job submission.

    Args:
        cli_args: Raw CLI arguments dict

    Raises:
        FileNotFoundError: If train_config_path doesn't exist
    """
    if "train_config_path" in cli_args:
        config_path = cli_args["train_config_path"]
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Train config file {config_path} does not exist.")


def submit_sft_job(
    exp_args: dict,
    cli_args: dict,
    hpc,
    *,
    construct_config_yaml_fn: Callable,
    update_exp_args_fn: Callable[[dict, dict], dict],
    write_run_summary_fn: Callable[[dict, dict], None],
    display_args_fn: Callable[[dict, str], None],
    submit_job_fn: Callable,
    should_run_pretokenize_fn: Callable,
    schedule_pretokenize_fn: Callable,
) -> Optional[str]:
    """Submit an SFT training job to SLURM.

    This function handles the complete SFT job submission flow:
    1. Pre-validation of config files
    2. Construction of LlamaFactory config YAML
    3. Generation of sbatch script
    4. Optional pretokenization job scheduling
    5. Job submission to SLURM

    Args:
        exp_args: Experiment arguments dict
        cli_args: Raw CLI arguments dict
        hpc: HPC cluster configuration object
        construct_config_yaml_fn: Function to construct the training config YAML
        update_exp_args_fn: Function to update exp_args dict
        write_run_summary_fn: Function to write run summary metadata
        display_args_fn: Function to display arguments
        submit_job_fn: Function to submit sbatch job
        should_run_pretokenize_fn: Function to check if pretokenization is needed
        schedule_pretokenize_fn: Function to schedule pretokenization job

    Returns:
        Job ID string if submitted, None if dry run
    """
    job_type = exp_args.get("job_type")

    # Pre-validation
    pre_validation_sft(cli_args)

    # Construct the config yaml
    train_config, train_config_path_out = construct_config_yaml_fn(exp_args)
    exp_args = update_exp_args_fn(exp_args, train_config)
    exp_args = update_exp_args_fn(exp_args, {"train_config_path_out": train_config_path_out})
    write_run_summary_fn(exp_args, train_config)

    # Construct the sbatch script using universal SFT template
    train_sbatch_path_out = construct_sft_sbatch_script(exp_args, hpc)
    exp_args = update_exp_args_fn(exp_args, {"train_sbatch_path_out": train_sbatch_path_out})

    display_args_fn(exp_args, "Train")

    if exp_args.get("dry_run", False):
        print("DRY RUN: Job would be submitted with the above parameters, but --dry_run flag was set.")
        return None

    dependency = None
    wants_pretokenize = should_run_pretokenize_fn(exp_args, job_type)
    if wants_pretokenize:
        tokenized_path = exp_args.get("tokenized_path", "")
        if tokenized_path and os.path.exists(tokenized_path):
            print(f"Tokenized directory {tokenized_path} already exists, skipping pretokenization job submission")
        else:
            pretok_job_id = schedule_pretokenize_fn(
                exp_args,
                update_exp_args_fn=update_exp_args_fn,
                construct_config_yaml_fn=construct_config_yaml_fn,
                construct_sbatch_script_fn=lambda args: construct_sft_sbatch_script(args, hpc),
                submit_job_fn=submit_job_fn,
            )
            dependency = f"afterok:{pretok_job_id}"

    train_job_id = submit_job_fn(exp_args=exp_args, dependency=dependency)
    return train_job_id


# =============================================================================
# Universal SFT Job Runner (Phase 2 refactoring)
# =============================================================================


@dataclass
class SFTJobConfig:
    """Configuration for an SFT training job (serialized to JSON for sbatch)."""

    job_name: str
    train_config_path: str
    experiments_dir: str
    cluster_name: str

    # Resource allocation
    num_nodes: int = 1
    gpus_per_node: int = 1
    cpus_per_node: int = 24

    # Training launcher: "torchrun" or "accelerate"
    launcher: str = "torchrun"

    # Accelerate config (if launcher == "accelerate")
    accelerate_config_path: Optional[str] = None

    # DeepSpeed config path
    deepspeed_config: Optional[str] = None

    # Networking
    master_port: int = 12802

    # SSH tunneling (JSC clusters)
    needs_ssh_tunnel: bool = False

    # CUDA path detection (Perlmutter)
    needs_cuda_detection: bool = False


class SFTJobRunner:
    """Runs SFT training jobs with proper distributed setup.

    This class encapsulates the SFT training logic that was previously
    spread across multiple cluster-specific sbatch scripts.

    Usage (from sbatch):
        python -m hpc.sft_launch_utils --config /path/to/config.json
    """

    def __init__(self, config: SFTJobConfig):
        self.config = config
        self._hpc = None

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters

            if self.config.cluster_name:
                # Find by name
                for c in clusters:
                    if c.name.lower() == self.config.cluster_name.lower():
                        self._hpc = c
                        break
                if self._hpc is None:
                    raise ValueError(f"Unknown cluster: {self.config.cluster_name}")
            else:
                self._hpc = detect_hpc()
        return self._hpc

    def run(self) -> int:
        """Execute the SFT training job.

        Returns:
            Exit code (0 for success)
        """
        print(f"=== SFTJobRunner: {self.config.job_name} ===")

        try:
            self._setup_environment()

            if self.config.launcher == "torchrun":
                exit_code = self._run_torchrun()
            else:
                exit_code = self._run_accelerate()

            if exit_code == 0:
                print(f"SFT job '{self.config.job_name}' completed successfully")
            else:
                print(f"SFT job '{self.config.job_name}' failed with code {exit_code}")

            return exit_code

        except Exception as e:
            print(f"SFT job failed with exception: {e}", file=sys.stderr)
            raise

    def _setup_environment(self):
        """Set up NCCL and other environment variables."""
        hpc = self._get_hpc()

        # Apply NCCL settings from HPC config
        for key, value in hpc.nccl_settings.items():
            os.environ[key] = value
            print(f"[env] {key}={value}")

        # Apply CUDA environment detection (Perlmutter, etc.)
        if self.config.needs_cuda_detection or hpc.needs_cuda_detection:
            from hpc.cuda_utils import setup_cuda_environment

            cuda_env = setup_cuda_environment()
            for key, value in cuda_env.items():
                os.environ[key] = value
                print(f"[cuda] {key}={value}")

    def _run_torchrun(self) -> int:
        """Launch training with torchrun."""
        # Get distributed training parameters from environment
        num_nodes = int(os.environ.get("NUM_NODES", self.config.num_nodes))
        gpus_per_node = int(os.environ.get("NUM_GPUS_PER_NODE", self.config.gpus_per_node))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", str(self.config.master_port))
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")

        cmd = [
            "torchrun",
            f"--nproc-per-node={gpus_per_node}",
            f"--nnodes={num_nodes}",
            f"--rdzv_id={slurm_job_id}",
            "--rdzv_backend=c10d",
            f"--rdzv_endpoint={master_addr}:{master_port}",
            "sft/llamafactory/src/train.py",
            self.config.train_config_path,
        ]

        print(f"Running torchrun command: {' '.join(cmd)}")
        sys.stdout.flush()

        return subprocess.call(cmd)

    def _run_accelerate(self) -> int:
        """Launch training with accelerate."""
        # Get distributed training parameters from environment
        num_nodes = int(os.environ.get("NUM_NODES", self.config.num_nodes))
        gpus_per_node = int(os.environ.get("NUM_GPUS_PER_NODE", self.config.gpus_per_node))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = os.environ.get("MASTER_PORT", str(self.config.master_port))
        # Use SLURM_NODEID for machine rank (node index within the allocation)
        # SLURM_PROCID is the global task ID which may not match node index
        slurm_nodeid = os.environ.get("SLURM_NODEID", os.environ.get("SLURM_PROCID", "0"))

        # Build accelerate config if not provided
        accelerate_config = self.config.accelerate_config_path
        if not accelerate_config:
            accelerate_config = self._generate_accelerate_config(num_nodes, gpus_per_node)

        # Debug: print multi-node configuration
        print(f"Multi-node config: num_nodes={num_nodes}, gpus_per_node={gpus_per_node}, "
              f"machine_rank={slurm_nodeid}, master_addr={master_addr}:{master_port}")
        print(f"SLURM env: SLURM_NODEID={os.environ.get('SLURM_NODEID')}, "
              f"SLURM_PROCID={os.environ.get('SLURM_PROCID')}, "
              f"SLURM_JOB_NUM_NODES={os.environ.get('SLURM_JOB_NUM_NODES')}")
        sys.stdout.flush()

        cmd = [
            "python", "-u", "-m", "accelerate.commands.launch",
            f"--rdzv_conf=rdzv_backend=c10d,rdzv_endpoint={master_addr}:{master_port}",
            f"--config_file={accelerate_config}",
            f"--main_process_ip={master_addr}",
            f"--main_process_port={master_port}",
            f"--machine_rank={slurm_nodeid}",
            f"--num_machines={num_nodes}",
            f"--num_processes={num_nodes * gpus_per_node}",
            "--tee=3",
            "sft/llamafactory/src/train.py",
            self.config.train_config_path,
        ]

        print(f"Running accelerate command: {' '.join(cmd)}")
        sys.stdout.flush()

        return subprocess.call(cmd)

    def _generate_accelerate_config(self, num_nodes: int, gpus_per_node: int) -> str:
        """Generate an accelerate config file for distributed training."""
        config_dir = Path(self.config.experiments_dir) / "accelerate_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / f"{self.config.job_name}_accelerate.yaml"

        # Basic accelerate config for multi-node training
        config = {
            "compute_environment": "LOCAL_MACHINE",
            "distributed_type": "FSDP" if self.config.deepspeed_config is None else "DEEPSPEED",
            "downcast_bf16": "no",
            "enable_cpu_affinity": False,
            "machine_rank": 0,
            "main_training_function": "main",
            "num_machines": num_nodes,
            "num_processes": num_nodes * gpus_per_node,
            "rdzv_backend": "c10d",
            "same_network": True,
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False,
        }

        if self.config.deepspeed_config:
            # When using deepspeed_config_file, do NOT set mixed_precision in accelerate config
            # All these settings must be in the DeepSpeed config file instead:
            # gradient_accumulation_steps, gradient_clipping, zero_stage, mixed_precision,
            # offload_optimizer_device, offload_param_device, zero3_save_16bit_model
            #
            # CRITICAL for multi-node: Use "standard" launcher (torch.distributed.run) instead of
            # DeepSpeed's launcher. DeepSpeed's launcher ignores accelerate's multi-node settings
            # and uses its own world discovery (which defaults to localhost only).
            config["deepspeed_config"] = {
                "deepspeed_config_file": self.config.deepspeed_config,
                "zero3_init_flag": True,
                "deepspeed_multinode_launcher": "standard",
            }
        else:
            # FSDP config - mixed_precision is set here (not in deepspeed case)
            config["mixed_precision"] = "bf16"
            config["fsdp_config"] = {
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_backward_prefetch": "BACKWARD_PRE",
                "fsdp_cpu_ram_efficient_loading": True,
                "fsdp_forward_prefetch": False,
                "fsdp_offload_params": False,
                "fsdp_sharding_strategy": "FULL_SHARD",
                "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                "fsdp_sync_module_states": True,
                "fsdp_use_orig_params": True,
            }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Generated accelerate config: {config_path}")
        return str(config_path)


def construct_sft_sbatch_script(exp_args: dict, hpc) -> str:
    """Construct SFT sbatch script using the new universal template system.

    This is a drop-in replacement for construct_sbatch_script() for SFT jobs.
    It creates the sbatch script and returns the path, letting the caller
    handle job submission (including dependencies and max_restarts).

    Args:
        exp_args: Experiment arguments dictionary
        hpc: HPC cluster configuration

    Returns:
        Path to the generated sbatch script
    """
    print("\n=== SFT MODE (Universal Launcher) ===")

    # Resolve job_name and paths (job_name already set by get_job_name() in launch.py)
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="SFT",
    )
    job_name = job_setup.job_name
    exp_paths = job_setup.paths
    experiments_subdir = str(exp_paths.root)

    # Extract config values
    train_config_path = exp_args.get("train_config_path_out")
    if not train_config_path:
        raise ValueError("SFT jobs require a train config path.")

    num_nodes = int(exp_args.get("num_nodes") or 1)
    gpus_per_node = int(exp_args.get("gpus_per_node") or hpc.gpus_per_node)
    cpus_per_node = int(exp_args.get("cpus_per_node") or hpc.cpus_per_node)

    # Build SFTJobConfig
    job_config = SFTJobConfig(
        job_name=job_name,
        train_config_path=train_config_path,
        experiments_dir=experiments_subdir,
        cluster_name=hpc.name,
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        launcher=hpc.training_launcher,
        deepspeed_config=exp_args.get("deepspeed"),
        needs_ssh_tunnel=hpc.needs_ssh_tunnel,
        needs_cuda_detection=hpc.needs_cuda_detection,
        master_port=int(exp_args.get("master_port") or 12802),
    )

    # Write config JSON
    config_dir = exp_paths.configs if hasattr(exp_paths, "configs") else exp_paths.root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{job_name}_sft_config.json"
    config_path.write_text(json.dumps(asdict(job_config), indent=2))
    print(f"Wrote SFT job config to {config_path}")

    # Load and populate universal template
    template_path = Path(__file__).parent / "sbatch_sft" / "universal_sft.sbatch"
    template_text = template_path.read_text()

    # Build cluster-specific SBATCH directives
    sbatch_directives = build_sbatch_directives(hpc, exp_args)

    # Generate CUDA setup code
    cuda_setup = ""
    if hpc.needs_cuda_detection:
        cuda_setup = """# CUDA path detection (handled by Python runner)
# Additional CUDA setup can be done in SFTJobRunner._setup_environment()"""

    srun_prefix = f"srun --nodes={num_nodes} --ntasks-per-node=1"
    # Use --nodes and --ntasks-per-node=1 to ensure one process per node for multi-node training
    # Each node then launches its own accelerate processes for local GPUs
    if hpc.needs_ssh_tunnel:
        # JSC clusters use proxychains4 for internet access
        srun_prefix += " $PROXY_CMD"
    # if os.environ.get("IMAGE"):
    #     print(f"Using Apptainer image: {os.environ['IMAGE']}")
    #     srun_prefix += f' apptainer exec --nv {os.environ["IMAGE"]}'

    # The srun child processes need the conda environment re-activated because
    # srun launches a non-interactive shell that doesn't inherit the batch step's
    # conda activation.  Prepend the conda activate so every node uses the right
    # Python (critical for sft-qwen35 which needs transformers >= 5.3.0).
    conda_activate = _get_sft_conda_activate(hpc, exp_args)
    cmd = f'{conda_activate} && python -m hpc.sft_launch_utils --config "{config_path}"'
    srun_command = f"{srun_prefix} bash -c '{cmd}'"
    substitutions = {
        "time_limit": exp_args.get("time_limit") or "24:00:00",
        "num_nodes": str(num_nodes),
        "cpus_per_node": str(cpus_per_node),
        "experiments_dir": experiments_subdir,
        "job_name": job_name,
        "sbatch_extra_directives": "\n".join(sbatch_directives),
        "module_commands": hpc.get_module_commands(),
        "conda_activate": _get_sft_conda_activate(hpc, exp_args),
        "cluster_env_file": hpc.dotenv_filename,
        "cuda_setup": cuda_setup,
        "nccl_exports": hpc.get_nccl_exports(),
        "env_exports": hpc.get_env_exports(),
        "ray_env_exports": hpc.get_ray_env_exports(experiments_subdir),
        "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
        "master_port": str(job_config.master_port),
        "master_addr_suffix": hpc.master_addr_suffix or "",
        "gpus_per_node": str(gpus_per_node),
        "config_path": str(config_path),
        "srun_command": srun_command,
        "email_address": os.environ.get("EMAIL_ADDRESS", ""),
    }

    sbatch_text = substitute_template(template_text, substitutions)

    # Write sbatch script
    sbatch_dir = exp_paths.sbatch if hasattr(exp_paths, "sbatch") else exp_paths.root / "sbatch_scripts"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_output = sbatch_dir / f"{job_name}_sft.sbatch"
    sbatch_output.write_text(sbatch_text)
    os.chmod(sbatch_output, 0o750)
    print(f"Wrote SFT sbatch script to {sbatch_output}")

    return str(sbatch_output)


def run_sft_job_main():
    """Entry point for running SFT jobs from sbatch."""
    import argparse

    parser = argparse.ArgumentParser(description="Run SFT training job")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)

    config = SFTJobConfig(**config_dict)
    runner = SFTJobRunner(config)
    sys.exit(runner.run())


if __name__ == "__main__":
    run_sft_job_main()
