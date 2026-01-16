import dataclasses
from dataclasses import dataclass, field
from typing import Optional
import argparse
import sys
from enum import Enum

class JobType(str, Enum):
    """Enumerates supported HPC launcher job categories."""

    SFT = "sft"
    SFT_MCA = "sft_mca"
    PRETOKENIZE = "pretokenize"
    DATAGEN = "datagen"
    EVAL = "eval"
    CONSOLIDATE = "consolidate"
    RL = "rl"

    @classmethod
    def choices(cls) -> list[str]:
        return [member.value for member in cls]

    @classmethod
    def default_value(cls) -> str:
        return cls.SFT.value


from hpc.cli_utils import parse_bool_flag, coerce_str_bool_none, coerce_numeric_cli_values

@dataclass
class LlamaFactoryArgs:
    """Arguments for LlamaFactory training"""

    # Model arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )

    # Method arguments
    stage: Optional[str] = field(
        default=None, metadata={"help": "Training stage: sft, rm, ppo, dpo"}
    )
    do_train: Optional[bool] = field(
        default=None, metadata={"help": "Whether to run training or not"}
    )
    finetuning_type: Optional[str] = field(
        default=None, metadata={"help": "Finetuning type: full, lora, qlora"}
    )
    deepspeed: Optional[str] = field(
        default="sft/llamafactory/examples/deepspeed/ds_z3_config.json",
        metadata={"help": "Path to deepspeed config file"},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to pack multiple sequences into one batch"},
    )
    neat_packing: Optional[bool] = field(
        default=None, metadata={"help": "Whether to use neat packing"}
    )
    enable_liger_kernel: Optional[bool] = field(
        default=None, metadata={"help": "Whether to use liger kernel"}
    )

    # Attention implementation
    attn: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Attention implementation. One of: 'eager', 'sdpa', 'fa2', 'fa3', "
                "or a HuggingFace kernel identifier (optionally prefixed with 'hf:')."
            )
        },
    )

    # Dataset arguments
    dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset identifier from huggingface.co/datasets or local dataset"
        },
    )
    dataset_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing dataset files"}
    )
    prompt_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the instruction/prompt column for Alpaca-style datasets (use '' or 'none' to disable)."
        },
    )
    query_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the optional input/query column for Alpaca-style datasets (use '' or 'none' to disable)."
        },
    )
    response_column: Optional[str] = field(
        default=None,
        metadata={"help": "Override the completion/response column for Alpaca-style datasets."},
    )
    history_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the history column for Alpaca-style datasets (use '' or 'none' to disable)."
        },
    )
    formatting: Optional[str] = field(
        default="sharegpt",
        metadata={"help": "Dataset formatting to align (e.g., 'sharegpt' or 'alpaca')"},
    )
    template: Optional[str] = field(
        default=None, metadata={"help": "Chat template to use"}
    )
    system: Optional[str] = field(
        default=None, metadata={"help": "System column in dataset"}
    )
    messages: Optional[str] = field(
        default="conversations", metadata={"help": "Message column in dataset"}
    )
    cutoff_len: Optional[int] = field(
        default=None, metadata={"help": "Maximum length of input sequences"}
    )
    overwrite_cache: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of workers for preprocessing"}
    )
    role_tag: Optional[str] = field(
        default="from", metadata={"help": "Role tag for the dataset"}
    )
    user_tag: Optional[str] = field(
        default="human", metadata={"help": "User tag for the dataset"}
    )
    content_tag: Optional[str] = field(
        default="value", metadata={"help": "Content tag for the dataset"}
    )
    assistant_tag: Optional[str] = field(
        default="gpt", metadata={"help": "Assistant tag for the dataset"}
    )
    mix_strategy: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Choose how multiple datasets are combined during training. "
                "Accepts 'concat', 'interleave_under', or 'interleave_over' to control sampling order."
            )
        },
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Provide comma-separated sampling probabilities when using an interleave mixing strategy. "
                "Ensure the list length matches the number of datasets to avoid validation errors."
            )
        },
    )
    streaming: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Enable Hugging Face streaming mode to iterate datasets without preloading them. "
                "Useful for very large corpora but restricts certain transformations like random shuffling."
            )
        },
    )
    buffer_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Define the shuffle buffer size used in streaming mode. "
                "Larger buffers improve randomness at the cost of additional host memory."
            )
        },
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit each training dataset to a fixed number of samples for debugging or curriculum staging. "
                "Applied before any shuffling or mixing so ordering semantics remain predictable."
            )
        },
    )
    val_size: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Reserve a portion of the training data for validation when no explicit eval dataset is provided. "
                "Float values denote fractions while integers are treated as absolute sample counts."
            )
        },
    )
    eval_on_each_dataset: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Emit per-dataset validation metrics instead of aggregating everything together. "
                "Helpful for diagnosing regressions when mixing heterogeneous data sources."
            )
        },
    )
    disable_shuffling: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Disable random shuffling and iterate datasets sequentially every epoch. "
                "Use this for deterministic runs or curriculum learning where order matters."
            )
        },
    )
    # Output arguments
    save_strategy: Optional[str] = field(
        default=None, metadata={"help": "The checkpoint save strategy to use"}
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory to store the model checkpoints"}
    )
    logging_steps: Optional[int] = field(
        default=None, metadata={"help": "Log metrics every X updates steps"}
    )
    plot_loss: Optional[bool] = field(
        default=None, metadata={"help": "Whether to plot losses"}
    )
    overwrite_output_dir: Optional[bool] = field(
        default=None, metadata={"help": "Whether to overwrite the output directory"}
    )

    # Training arguments
    per_device_train_batch_size: Optional[int] = field(
        default=None, metadata={"help": "Batch size per GPU for training"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of updates steps to accumulate before backward"},
    )
    # Convenience input used by the launcher to derive
    # gradient_accumulation_steps based on cluster size.
    # Not passed through to the underlying trainer as-is.
    global_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Global batch size. Launcher computes "
                "gradient_accumulation_steps = global_batch_size // (num_nodes * gpus_per_node) "
                "and sets per_device_train_batch_size = 1."
            )
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether to use gradient checkpointing to save memory"},
    )
    learning_rate: Optional[float] = field(
        default=None, metadata={"help": "The initial learning rate"}
    )
    adam_beta1: Optional[float] = field(
        default=None, metadata={"help": "Adam optimizer beta1 coefficient"}
    )
    adam_beta2: Optional[float] = field(
        default=None, metadata={"help": "Adam optimizer beta2 coefficient"}
    )
    num_train_epochs: Optional[int] = field(
        default=None, metadata={"help": "Total number of training epochs"}
    )
    lr_scheduler_type: Optional[str] = field(
        default=None, metadata={"help": "The scheduler type to use"}
    )
    warmup_ratio: Optional[float] = field(
        default=None, metadata={"help": "Linear warmup ratio"}
    )
    weight_decay: Optional[float] = field(
        default=None, metadata={"help": "Weight decay to apply"}
    )
    bf16: Optional[bool] = field(
        default=None, metadata={"help": "Whether to use bf16 mixed precision"}
    )
    ddp_timeout: Optional[int] = field(default=None, metadata={"help": "DDP timeout"})
    report_to: Optional[str] = field(default=None, metadata={"help": "Report to wandb"})
    run_name: Optional[str] = field(
        default=None, metadata={"help": "Run name for wandb"}
    )
    seed: Optional[int] = field(
        default=42, metadata={"help": "Random seed for reproducibility"}
    )
    max_grad_norm: Optional[float] = field(
        default=None, metadata={"help": "Maximum gradient norm for clipping"}
    )
    max_steps: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of training steps"}
    )
    use_unsloth_gc: Optional[bool] = field(
        default=None, metadata={"help": "Whether to use unsloth gc", "store_true": True}
    )

    # Eval arguments
    eval_strategy: Optional[str] = field(
        default=None, metadata={"help": "The evaluation strategy to use"}
    )
    push_to_hub: Optional[bool] = field(
        default=None, metadata={"help": "Whether to push to hub"}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "Repo name to push to hub (default: mlfoundations-dev/{job_name})"}
    )

    # Extra arguments that might be used depending on finetuning type
    lora_rank: Optional[int] = field(default=None, metadata={"help": "Rank of LoRA"})
    lora_alpha: Optional[float] = field(
        default=None, metadata={"help": "Alpha of LoRA"}
    )
    lora_dropout: Optional[float] = field(
        default=None, metadata={"help": "Dropout of LoRA"}
    )

@dataclass
class LaunchArgs:
    """Arguments for job launching"""

    # Core launch arguments
    job_name: Optional[str] = field(
        default=None, metadata={"help": "Job name. This will determine outputs, including HF repo."}
    )
    job_creator: str = field(
        default="DCAgent",
        metadata={"help": "Name responsible for launching the job (<=96 characters)"}
    )
    train_config_path: Optional[str] = field(
        default=None, metadata={"help": "Path to config file"}
    )
    experiments_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output for storing experiment outputs - logs, configs, sbatch scripts. "
            "Defaults to ./experiments/<job_name> when not specified."
        },
    )
    image: Optional[str] = field(
        default=None, metadata={"help": "Container image to use"}
    )
    checkpoints_dir: Optional[str] = field(
        default=None, metadata={"help": "Checkpoints directory"}
    )
    models_dir: Optional[str] = field(
        default=None, metadata={"help": "Models directory"}
    )
    datasets_dir: Optional[str] = field(
        default=None, metadata={"help": "Datasets directory"}
    )
    tokenized_dir: Optional[str] = field(
        default=None, metadata={"help": "Tokenized datasets directory"}
    )
    base_model: Optional[str] = field(
        default=None, metadata={"help": "Base model name for output directory naming"}
    )
    chat_template: Optional[str] = field(
        default=None, metadata={"help": "Chat template to use"}
    )
    time_limit: Optional[str] = field(
        default=None, metadata={"help": "Time limit for the job"}
    )
    max_restarts: Optional[int] = field(
        default=None, metadata={"help": "Maximum number of job restarts"}
    )
    dependency: Optional[str] = field(
        default=None,
        metadata={"help": "SLURM dependency expression to include with submissions (e.g., 'afterany:12345')"},
    )

    # Pretokenize
    pretokenize: bool = field(
        default=False, metadata={"help": "Whether to pretokenize", "store_true": True}
    )
    pretok_large: bool = field(
        default=False, metadata={"help": "If true, pretokenize on boost_qos_bprod 128 nodes", "store_true": True}
    )

    # Job parameters
    num_nodes: Optional[int] = field(
        default=None, metadata={"help": "Number of nodes to use"}
    )
    num_gpus: Optional[int] = field(
        default=None, metadata={"help": "Number of GPUs per node to use"}
    )

    # Dry run
    dry_run: bool = field(
        default=False,
        metadata={
            "help": "When present, the job will not be submitted",
            "store_true": True,
        },
    )

    internet_node: bool = field(
        default=False,
        metadata={
            "help": "Whether to enable internet access using proxies on the login node",
            "store_true": True,
        },
    )
    pinggy_persistent_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Persistent Pinggy hostname (e.g., xxxxx.a.pinggy.link) to reuse for tunnels",
        },
    )
    pinggy_ssh_command: Optional[str] = field(
        default=None,
        metadata={
            "help": "Custom Pinggy SSH tunnel command template (use {PORT} for HAProxy port)",
        },
    )
    pinggy_debugger_url: Optional[str] = field(
        default=None,
        metadata={
            "help": "Debugger URL exposed via Pinggy (used for health checks)",
        },
    )
    use_mca: bool = field(
        default=False,
        metadata={
            "help": "Enable Megatron Core Adapter integration (sets USE_MCA=1 for SFT jobs)",
            "store_true": True,
        },
    )


@dataclass
class DataGenArgs:
    """Arguments for data generation jobs"""

    # Job type
    job_type: Optional[str] = field(
        default=JobType.default_value(),
        metadata={
            "help": "Job type: 'sft', 'sft_mca', 'pretokenize', 'datagen', 'eval', 'consolidate', or 'rl'",
            "choices": JobType.choices(),
            "required": False,
        },
    )

    # Data generation specific
    enable_task_gen: bool = field(
        default=False,
        metadata={"help": "Whether to run task generation stage"}
    )
    enable_trace_gen: bool = field(
        default=False,
        metadata={"help": "Enable trace generation stage"}
    )
    disable_verification: bool = field(
        default=False,
        metadata={
            "help": "Disable Harbor verification during trace generation",
            "store_true": True,
        },
    )
    chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of tasks per trace chunk when splitting trace jobs"}
    )
    datagen_script: Optional[str] = field(
        default=None,
        metadata={"help": "Path to data generation script (e.g., data/gsm8k_test/generate.py)"}
    )
    datagen_target_repo: Optional[str] = field(
        default=None,
        metadata={"help": "Target HuggingFace repository for generated data"}
    )
    datagen_input_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Input directory for data generation"}
    )
    datagen_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for generated artifacts"}
    )
    task_type: Optional[str] = field(
        default=None,
        metadata={"help": "Optional task type identifier forwarded to the datagen script"}
    )
    datagen_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to YAML config describing datagen inference engine/backends"}
    )
    datagen_extra_args: Optional[str] = field(
        default="",
        metadata={"help": "Additional arguments to pass to generation script"}
    )

    # Daytona sandbox resource overrides
    sandbox_cpu: Optional[int] = field(
        default=1,
        metadata={"help": "Override Daytona sandbox vCPU allocation"}
    )
    sandbox_memory_gb: Optional[int] = field(
        default=1,
        metadata={"help": "Override Daytona sandbox memory in GB"}
    )
    sandbox_disk_gb: Optional[int] = field(
        default=3,
        metadata={"help": "Override Daytona sandbox disk in GB"}
    )
    sandbox_gpu: Optional[int] = field(
        default=None,
        metadata={"help": "Override Daytona sandbox GPU allocation"}
    )

    trace_script: Optional[str] = field(
        default=None,
        metadata={"help": "Path to trace generation script"}
    )
    tasks_input_path: Optional[str] = field(
        default=None,
        metadata={"help": "Existing task dataset path for trace generation"}
    )
    trace_target_repo: Optional[str] = field(
        default=None,
        metadata={"help": "Target HuggingFace repo for traces"}
    )
    trace_output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for generated traces"}
    )
    trace_model: Optional[str] = field(
        default=None,
        metadata={"help": "Override Harbor agent model for trace generation"}
    )
    trace_agent_name: Optional[str] = field(
        default="terminus-2",
        metadata={"help": "Agent name for trace generation and run_summary.json (default: terminus-2)"}
    )
    trace_agent_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string of additional kwargs for the trace agent (e.g. '{\"max_episodes\": 32}')."}
    )
    trace_n_concurrent: Optional[int] = field(
        default=None,
        metadata={"help": "Override Harbor orchestrator concurrency for trace generation"}
    )
    trace_env: Optional[str] = field(
        default=None,
        metadata={"help": "Override Harbor environment type for trace generation (e.g. docker, daytona)"}
    )
    trace_harbor_config: Optional[str] = field(
        default=None,
        metadata={"help": "Harbor job YAML describing trace execution"}
    )
    trace_engine: Optional[str] = field(
        default=None,
        metadata={"help": "Engine to use for trace generation (supports 'openai', 'anthropic', 'vllm_local', 'none'; defaults to datagen_engine)"}
    )
    trace_backend: Optional[str] = field(
        default=None,
        metadata={"help": "Backend to use for trace generation (e.g., 'vllm', 'ray', 'none'; defaults to datagen_backend)"}
    )
    trace_export_subagents: bool = field(
        default=True,
        metadata={"help": "Export subagent traces (e.g., context summarization) alongside main agent traces"}
    )
    trace_use_gpu: bool = field(
        default=False,
        metadata={"help": "Request GPUs for trace generation", "store_true": True}
    )
    trace_agent_timeout_sec: Optional[float] = field(
        default=None,
        metadata={"help": "Override Harbor agent timeout (seconds) for trace generation"}
    )
    trace_verifier_timeout_sec: Optional[float] = field(
        default=None,
        metadata={"help": "Override Harbor verifier timeout (seconds) for trace generation"}
    )
    harbor_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Harbor registry dataset slug such as 'terminal-bench@2.0'"}
    )
    # Upload settings (traces -> HuggingFace, result abstracts -> Supabase)
    upload_to_database: bool = field(
        default=False,
        metadata={"help": "Upload result abstracts to Supabase and traces to HuggingFace after eval", "store_true": True}
    )
    upload_username: Optional[str] = field(
        default=None,
        metadata={"help": "Username for Supabase result attribution (defaults to $UPLOAD_USERNAME or current user)"}
    )
    upload_error_mode: str = field(
        default="skip_on_error",
        metadata={"help": "Supabase upload error handling: 'skip_on_error' or 'rollback_on_error'"}
    )
    upload_hf_repo: Optional[str] = field(
        default=None,
        metadata={"help": "HuggingFace repo for traces upload (auto-derived from benchmark if not provided)"}
    )
    upload_hf_private: bool = field(
        default=False,
        metadata={"help": "Create the HuggingFace traces repo as private", "store_true": True}
    )
    upload_hf_episodes: str = field(
        default="last",
        metadata={"help": "Which episodes to include in HuggingFace traces upload: 'last' or 'all'"}
    )
    upload_forced_update: bool = field(
        default=False,
        metadata={"help": "Allow overwriting existing Supabase result records for the same job", "store_true": True}
    )

@dataclass
class ConsolidateArgs:
    consolidate_input: Optional[str] = field(
        default=None,
        metadata={"help": "Input for consolidation: either a local directory with ZeRO shards or a Hugging Face repo ID"}
    )
    consolidate_base_repo: Optional[str] = field(
        default=None,
        metadata={"help": "Base Hugging Face model repo to copy ancillary files (config, tokenizer, chat template) from"}
    )
    consolidate_output_repo: Optional[str] = field(
        default=None,
        metadata={"help": "Destination Hugging Face repo to upload merged weights"}
    )
    consolidate_workdir: Optional[str] = field(
        default=None,
        metadata={"help": "Working directory on the cluster filesystem for consolidation artifacts"}
    )
    consolidate_commit_message: Optional[str] = field(
        default="Merge ZeRO shards into safetensors",
        metadata={"help": "Commit message to use when uploading consolidated weights back to Hugging Face"}
    )


@dataclass
class RLArgs:
    """Arguments for RL (reinforcement learning) training jobs using SkyRL."""

    rl_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to RL config YAML file (e.g., terminal_bench.yaml). "
            "Can be absolute path or name of built-in config in hpc/skyrl_yaml/"
        }
    )
    skyrl_override: Optional[str] = field(
        default=None,
        metadata={
            "help": "SkyRL Hydra override (key=value). Can be specified multiple times. "
            "Example: --skyrl_override trainer.epochs=5",
            "action": "append",
        }
    )
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Model path for RL training (e.g., Qwen/Qwen2.5-7B-Instruct)"}
    )
    train_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Training dataset path(s). Use JSON list format for multiple: "
            "'[\"org/dataset1\",\"org/dataset2\"]'"
        }
    )
    val_data: Optional[str] = field(
        default=None,
        metadata={
            "help": "Validation dataset path(s). Use JSON list format for multiple: "
            "'[\"org/val-dataset\"]'"
        }
    )
    skyrl_entrypoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override SkyRL entrypoint module. "
            "Default: inferred from rl_config YAML"
        }
    )
    policy_num_nodes: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of nodes for policy (actor) workers. "
            "If not set, defaults to num_nodes (symmetric setup)"
        }
    )
    tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Tensor parallel size for vLLM inference engines. "
            "Higher values needed for larger models (70B+). Default from config or 1"
        }
    )
    ray_port: Optional[int] = field(
        default=None,
        metadata={"help": "Ray head node port (default: 6379)"}
    )
    rl_use_conda: bool = field(
        default=False,
        metadata={
            "help": "Use conda environment for RL instead of venv. "
            "Useful for clusters like Perlmutter where conda is preferred.",
            "store_true": True,
        }
    )
    rl_conda_env: Optional[str] = field(
        default="dcagent-rl",
        metadata={
            "help": "Name of conda environment to use for RL when --rl_use_conda is set. "
            "Default: dcagent-rl"
        }
    )


def _option_strings(field_name: str) -> list[str]:
    primary = f"--{field_name}"
    dashed = f"--{field_name.replace('_', '-')}"
    if dashed == primary:
        return [primary]
    return [primary, dashed]


def _add_dataclass_arguments(arg_group, dataclass_type, exclude_fields=None, *, bool_fields: set[str] | None = None):
    """
    Helper function to add arguments from a dataclass to an argument group.

    Args:
        arg_group: The argument group to add arguments to
        dataclass_type: The dataclass type to extract fields from
        exclude_fields: Optional list of field names to exclude
    """
    exclude_fields = exclude_fields or []

    for field in dataclasses.fields(dataclass_type):
        if field.name in exclude_fields:
            continue

        option_strings = _option_strings(field.name)
        help_text = field.metadata.get("help")
        choices = field.metadata.get("choices")
        required = field.metadata.get("required", False)
        action = field.metadata.get("action")

        if field.metadata.get("store_true"):
            arg_group.add_argument(
                *option_strings,
                dest=field.name,
                action="store_true",
                help=help_text,
                default=field.default,
            )
            if bool_fields is not None:
                bool_fields.add(field.name)
        elif action == "append":
            # Handle append action (e.g., --skyrl_override can be repeated)
            arg_group.add_argument(
                *option_strings,
                dest=field.name,
                action="append",
                default=[],
                help=help_text,
            )
        elif isinstance(field.default, bool):
            kwargs = {
                "dest": field.name,
                "type": parse_bool_flag,
                "help": help_text,
                "default": field.default,
            }
            if choices:
                kwargs["choices"] = choices
            if required:
                kwargs["required"] = True
            arg_group.add_argument(*option_strings, **kwargs)
            if bool_fields is not None:
                bool_fields.add(field.name)
        else:
            arg_type = type(field.default) if field.default is not None else str
            kwargs = {
                "dest": field.name,
                "type": arg_type,
                "help": help_text,
                "default": field.default,
            }
            if choices:
                kwargs["choices"] = choices
            if required:
                kwargs["required"] = True
            arg_group.add_argument(*option_strings, **kwargs)

def parse_args():
    parser = argparse.ArgumentParser(description="Launch HPC jobs for dcft experiment")

    bool_keys: set[str] = set()

    raw_argv = sys.argv[1:]
    explicit_cli_keys = set()
    i = 0
    while i < len(raw_argv):
        token = raw_argv[i]
        if token.startswith("--") and len(token) > 2:
            key = token[2:]
            if key.startswith("no-"):
                key = key[3:]
            if "=" in key:
                key = key.split("=", 1)[0]
            explicit_cli_keys.add(key.replace("-", "_"))
            if "=" not in token and (i + 1) < len(raw_argv) and not raw_argv[i + 1].startswith("--"):
                i += 1
        i += 1

    # Create argument groups for better organization
    launch_group = parser.add_argument_group("Launch Arguments")
    hpc_group = parser.add_argument_group("HPC Arguments")
    train_group = parser.add_argument_group("Training Arguments")
    datagen_group = parser.add_argument_group("Data Generation Arguments")
    consolidate_group = parser.add_argument_group("Consolidation Arguments")
    rl_group = parser.add_argument_group("RL Training Arguments")

    # Add LaunchArgs arguments
    _add_dataclass_arguments(launch_group, LaunchArgs, bool_fields=bool_keys)

    # Add --harbor_config as alias for --trace_harbor_config (more concise for eval jobs)
    launch_group.add_argument(
        "--harbor_config", "--harbor-config",
        dest="trace_harbor_config",
        help=argparse.SUPPRESS,  # Hidden alias
    )

    # Add --model as alias for --trace_model (more concise for eval jobs)
    launch_group.add_argument(
        "--model",
        dest="trace_model",
        help=argparse.SUPPRESS,  # Hidden alias
    )

    # Add DataGenArgs arguments
    _add_dataclass_arguments(datagen_group, DataGenArgs, bool_fields=bool_keys)
    _add_dataclass_arguments(consolidate_group, ConsolidateArgs, bool_fields=bool_keys)

    # Add RLArgs arguments
    _add_dataclass_arguments(rl_group, RLArgs, bool_fields=bool_keys)

    # Add HPC arguments
    # Note: HPC is a Pydantic model, not a dataclass, so we need to handle it differently
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
        "gpu_type",
    ]
    str_hpc_fields = {"name", "account", "partition", "gpus_type", "qos", "gpu_type"}
    for field in hpc_fields:
        hpc_group.add_argument(
            f"--{field}",
            type=str if field in str_hpc_fields else int,
            help=f"HPC {field}" if field != "gpu_type" else "GPU type override (e.g., h200, l40s) for clusters with multiple GPU types",
        )

    # Add LlamaFactoryArgs arguments
    _add_dataclass_arguments(train_group, LlamaFactoryArgs, bool_fields=bool_keys)

    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    args_dict["_explicit_cli_keys"] = explicit_cli_keys
    literal_none_keys = {"datagen_engine", "trace_engine", "datagen_backend", "trace_backend"}
    args_dict = coerce_str_bool_none(args_dict, literal_none_keys, bool_keys)
    args_dict = coerce_numeric_cli_values(args_dict)
    return args_dict
