# HPC Launch

## What is HPC Launch?

HPC launch is an infrastructure-aware master framework designed to allow a wide range of experiments to be launched via the command line. It is the best starting point for those who are new to OT-Agent and want to get up and running quickly.

## Which HPC Launch do I need?

There are actually two HPC launchers in OT-Agent right now, one for RL and one for SFT, datagen and model consolidation. If you are interested in RL please refer to `dc-agent/rl/hpc`; otherwise, continue here.

## What job types can I launch with HPC launch?

All jobs start with `python -m hpc.launch --job_type <mode>`. The supported modes are:
- `train` (default): launches SFT/finetuning runs driven by llama-factory configs.
- `datagen`: executes dataset generators to produce tasks, traces, or both.
- `consolidate`: merges ZeRO-sharded checkpoints into FP32 weights using the consolidate sbatch templates.

You can optionally schedule evaluations after a `train` job by supplying `--eval_tasks`; the launcher submits the eval job once training completes.

## Before you launch

Please review this checklist before you try to launch a job with HPC launch.

* Am I on a supported cluster for the job type I want to launch?
* Have I reviewed and installed necessary dependencies for the job type I want to launch?
* Have I registered all required environment variables? `dc-agent/hpc/dotenv` contains 'starter packs' but you may need to customize or extend this with your own API keys, wandb accounts, et cetera
* Have I been added to all of the OT-Agent accounts I will need access to? For training you will need Huggingface, WandB, Supabase, and Github.
* Have I checked the exact command line arguments in HPC launch for the job type I want? Not all available flags are documented here.

### Supported Cluster List

| Cluster | GPU type | Internet | Recommended job types |
| --- | --- | --- | --- |
| alpha | A100 40 GB | ✅ | train, datagen |
| capella | H100 94 GB | ✅ | train, datagen |
| claix | H100 96 GB | ✅ | train, datagen |
| jureca | H100 94 GB | ❌ | train (datagen requires cached artifacts) |
| jupiter | GH200 96 GB | ❌ | train |
| juwels | A100 40 GB | ❌ | train |
| leonardo | A100 64 GB | ❌ | train |
| lonestar | A100 40 GB | ✅ | train, datagen |
| lrz | H100 94 GB | ✅ | train, datagen |
| nyugreene | A100/H100 80 GB | ✅ | train, datagen |
| nyutorch | H200 141 GB | ✅ | train, datagen |
| oumi | H100 80 GB | ✅ | train, datagen |
| perlmutter | A100 80 GB | ✅ | train, datagen |
| vista | GH200 96 GB | ✅ | train, datagen, consolidate |

Clusters without outbound internet access (`❌`) are typically restricted to training or offline datagen runs where datasets are pre-synced and uploads are disabled.

This guide focuses on `python -m hpc.launch`, the entry point used to submit both data-generation and training workloads. The launcher coordinates sbatch templates, shared environment variables, and per-job configuration so a single command can schedule a job from a login node.

## Prerequisites
- **Submodules:** initialise `sft/llamafactory` before launching SFT jobs  
  `git submodule update --init --remote sft/llamafactory`
- **Python environment:** create a Conda/virtualenv suited to your cluster, then install launcher requirements  
  `uv pip install -r hpc_requirements.txt`
- **LLM backends:** install vLLM or other serving stacks required by your datagen jobs. Follow your cluster’s guidance for CUDA/toolkit versions.
- **Dotenv:** copy and edit `hpc/dotenv/tacc.env` (or prepare your own) so that paths such as `$DCFT`, `$DCFT_ACTIVATE_ENV`, and sbatch directories point at your environment. Do **not** store secrets in the file.
- **Secrets:** keep API keys in a private env file and export `DC_AGENT_SECRET_ENV=/path/to/secrets.env` before sourcing the cluster dotenv. The launcher and sbatch scripts will source that file automatically.
- **Harbor (for trace generation):** clone `https://github.com/laude-institute/harbor` and run `pip install -e .` inside the environment you use for datagen.

To start a session:
```bash
source /path/to/dc-agent/hpc/dotenv/tacc.env
cd "$DCFT"
$DCFT_ACTIVATE_ENV
```

## Launcher Overview
`hpc.launch` reads configuration, sbatch templates, and generator modules based on the flags you provide:
```bash
python -m hpc.launch [core flags] [job-type flags] [--dry_run]
```
Use `--dry_run` to inspect generated sbatch scripts without submitting a job.

### Common flags
- `--job_type {datagen,train}`: choose between dataset generation and SFT runs.
- `--experiments_dir`: directory where sbatch files, logs, and generated configs are written.
- `--time_limit`: wall-clock limit passed to sbatch (e.g. `24:00:00`).
- `--num_nodes`: requested node count for training jobs.
- `--dry_run`: render outputs without calling `sbatch`.

## Datagen Jobs
Datagen jobs build datasets (tasks, traces, or both). Key flags:
- `--datagen_script`: generator entry point (e.g. `data/nl2bash/generate_abstract.py`).
- `--datagen_target_repo`: Hugging Face repo for uploading generated tasks.
- `--datagen_engine`: inference backend (`openai`, `anthropic`, `vllm_local`, `none`).
- `--datagen_extra_args`: extra CLI switches forwarded to the generator.

Optional trace generation:
- `--enable_trace_gen` / `--trace_script` / `--trace_target_repo`: analogous settings for trace export.
- `--trace_engine`, `--trace_backend`, `--trace_harbor_config`: configure the inference stack and Harbor job definition used during trace collection.
- `--trace_model`, `--trace_agent_name`, `--trace_agent_kwargs`, `--trace_n_concurrent`, `--trace_env`: override fields from the Harbor YAML without editing the file.
- `--trace_input_path`: reuse an existing tasks dataset instead of regenerating tasks.
- `--chunk_size`: optionally split trace generation into parallel chunks when the task count exceeds the given size; each chunk is launched as its own SLURM job with an incremented `--trace_target_repo`.

Example (tasks + traces using a vLLM endpoint):
```bash
python -m hpc.launch \
  --job_type datagen \
  --datagen_script data/nl2bash/generate_abstract.py \
  --datagen_target_repo my-org/nl2bash-tasks \
  --datagen_engine vllm_local \
  --datagen_extra_args "--stage both --limit 200" \
  --trace_target_repo my-org/nl2bash-traces \
  --trace_harbor_config path/to/harbor_config.yaml \
  --experiments_dir "$DCFT/experiments" \
  --time_limit 12:00:00
```
Adjust `--datagen_extra_args` to pass dataset-specific switches such as sampling bounds or input paths.

## SFT Jobs
Training runs rely on llama-factory configs stored under `sft/hp_settings`. Supply the path to a YAML alongside optional overrides:
```bash
python -m hpc.launch \
  --job_type train \
  --train_config_path sft/hp_settings/paper/reasoning_medium.yaml \
  --dataset my-org/my-training-set \
  --num_nodes 8 \
  --time_limit 24:00:00 \
  --experiments_dir "$DCFT/experiments"
```
You can extend llama-factory settings directly in the YAML or pass `--train_extra_args "..."`. Run with `--dry_run` while iterating on sbatch templates.

## Customisation Tips
- **Sbatch templates:** edit or copy files under `hpc/sbatch_data` to align with your cluster’s modules, queues, and launch commands.
- **Multiple environments:** when datagen and training use different Python envs, set dedicated activation commands (e.g. `$DCFT_ACTIVATE_ENV` and `$DCFT_PRIVATE_ACTIVATE_ENV`) in your dotenv.
- **Health checks:** `BaseDataGenerator` includes initial health checks for local inference endpoints. Ensure environment variables referenced in your sbatch scripts expose any required URLs or credentials.

## Scripts

### `scripts/database/manual_db_push.py`
- Manually registers a trained model with the shared Supabase instance when an automated post-training hook was skipped or failed.
- Loads run metadata (start/finish timestamps, links) from W&B, then calls `register_trained_model` from LlamaFactory to upsert the record.
- Requires Supabase and W&B credentials exported (e.g. `source hpc/dotenv/tacc.env`) plus `TRACE_S3_PATH` if you want to attach the traces location.
- Edit the module-level constants (`HF_MODEL_ID`, `WANDB_RUN`, `DATASET_ID`, etc.) to describe the model you are registering before running.

#### Example
```bash
source hpc/dotenv/tacc.env
python scripts/database/manual_db_push.py
```

For deeper, cluster-specific procedures keep notes in experiment logs or private documentation so this README stays concise and current.
