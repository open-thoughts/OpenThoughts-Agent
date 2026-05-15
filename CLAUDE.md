# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claude Code Environment Notes

**Conda Environment**: Use the otagent Python directly (symlinks don't work in the sandbox):
```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python your_script.py
```

**Syntax Checking**: Use the IDE MCP tool `mcp__ide__getDiagnostics` for checking Python syntax errors and linting issues. Do NOT use bash commands like `python -m py_compile` or `flake8` as the bash environment may have issues with output capture.

```
# Example: Check a file for errors
mcp__ide__getDiagnostics(uri="file:///path/to/file.py")
```

## Local Companion Codebases

When making changes to Harbor or SkyRL, edit the local repos and sync via git (commit, push, then pull on the cluster). Do NOT manually patch files on remote clusters.

All three codebases (Harbor, SkyRL, OT-Agent) are installed as **editable installs** (`pip install -e .`) on all clusters. After `git pull`, the updated source is immediately active — no reinstall needed.

- **Harbor**: `/Users/benjaminfeuer/Documents/harbor` — agent framework, environment backends, terminus agent
- **SkyRL**: `/Users/benjaminfeuer/Documents/SkyRL` — RL training framework, trainer, terminal_bench generator

**Jupiter conda environments**: Use `otagent` for all OT-Agent work (job launching, scripts, uploads). Use `curator` only for curator data-generation jobs.

## Repository Overview

ot-agent is a distributed training and evaluation system for large language models on HPC clusters. It consists of four main subsystems:

1. **Data Generation**: Task and trace generation pipelines using Harbor/Daytona
2. **SFT Training (DCFT)**: Supervised fine-tuning using LLaMA-Factory
3. **RL Training**: Reinforcement learning training with SkyRL (using GRPO algorithm)
4. **Evaluation**: Terminal-bench based evaluation system

## Architecture

### Directory Structure

- **`data/`**: Data generation pipelines - each subdirectory is a named pipeline
- **`hpc/`**: DCFT SFT training launcher (uses LLaMA-Factory)
- **`train/hpc/`**: OT-Agent RL training launcher (uses SkyRL framework)
- **`eval/`**: Evaluation systems for both TACC and JSC clusters
- **`database/unified_db/`**: Supabase registry for datasets, models, and agents
- **`scripts/`**: Utility scripts for database, datagen, harbor, vllm, etc.

Both HPC launchers share similar architecture:
- `launch.py`: Main entry point for job submission
- `hpc.py`: Cluster detection and configuration (Pydantic models)
- `arguments.py`: CLI argument parsing
- `sbatch/`: SLURM job templates (Jinja2 for RL, plain for SFT)
- `dotenv/`: Environment variable files per cluster
- `scripts/common.sh`: Shared bash utilities and aliases

### Key Distinction: Internet Access

The codebase handles two types of HPC clusters:

**Internet-enabled clusters** (TACC: Vista, Lonestar; ZIH: Capella, Alpha):
- Compute nodes can directly access HuggingFace Hub
- Standard dataset/model loading works

**No-internet clusters** (JSC: Jureca, Jupiter, Juwels; Leonardo):
- Compute nodes have NO internet access
- `pre_download_dataset()` function pre-downloads datasets/models on login nodes
- Downloads stored in `HF_HUB_CACHE` before job submission
- Training uses SSH tunnels and Ray for coordination

### Supported HPC Clusters

**TACC (Texas Advanced Computing Center)**:
- Vista: GH200 96GB GPUs, 552 nodes, internet access
- Lonestar (ls6): A100 40GB GPUs, 73 nodes, internet access

**JSC (Jülich Supercomputing Centre)**:
- Jureca: H100 94GB GPUs, 16 nodes, no internet
- Jupiter: GH200 96GB GPUs, 48 nodes, no internet
- Juwels: A100 40GB GPUs, 936 nodes, no internet

**ZIH (TU Dresden)**:
- Capella: H100 94GB GPUs, 146 nodes, internet access
- Alpha: A100 40GB GPUs, 37 nodes, internet access

**Leonardo** (CINECA):
- A100 64GB GPUs, 3456 nodes, no internet

### Data Generation System

`data/` contains named pipeline directories. Two approaches are supported:

**Declarative scripts (`generate.py`)**: Self-contained Python scripts for local/one-off runs
```bash
python data/<dataset>/generate.py --optional-flags
```

**Class-based generators (`generate_abstract.py`)**: Subclass `BaseDataGenerator` for HPC runs with managed vLLM endpoints
```bash
python -m hpc.launch \
  --job_type datagen \
  --datagen_script data/<dataset>/generate_abstract.py \
  --datagen_target_repo <org/repo> \
  --datagen_extra_args "--stage both --limit 2000"
```

Key modules in `data/generation/`: `base.py` (BaseDataGenerator), `schemas.py` (GenerationRequest/Result), `engines.py` (InferenceEngine implementations for OpenAI/Anthropic/vLLM)

**Curator sharded datagen (`run_curator_datagen_sharded.sbatch`)**: Multi-node data-parallel generation using vLLM + async_datagen.py. Default: 32 nodes (one vLLM server per node). Supports auto-resume via stable shard output dirs. Uses `--account=reformo` on Jupiter (not the default `jureap59` account).
```bash
# Submit with restart chain (recommended for long datasets):
FIRST=$(sbatch data/sbatches/run_curator_datagen_sharded.sbatch \
  <model> <input_dataset> <output_repo> [limit] [save_every] | awk '{print $4}')
PREV=$FIRST; for i in $(seq 1 6); do
  PREV=$(sbatch --dependency=afterany:$PREV \
    data/sbatches/run_curator_datagen_sharded.sbatch \
    <model> <input_dataset> <output_repo> [limit] [save_every] | awk '{print $4}')
done
```
Note: The `MAX_RESTARTS` env var in the sbatch header comments is **not implemented** — you must manually create the `--dependency=afterany:` chain as shown above.

### Harbor Environment Backends

Harbor supports three environment backends for running sandbox containers:

- **`daytona`** (default): Cloud-managed containers via Daytona API
- **`docker`**: Local Docker/Podman runtime
- **`modal`**: Modal's cloud container platform

**Docker Backend Setup**:
```bash
# Auto-detect Docker/Podman (sets DOCKER_HOST automatically)
python -m data.local.run_tracegen \
    --harbor-config hpc/harbor_yaml/trace_docker_32concurrency_ctx32k.yaml \
    --tasks-input-path ./my-tasks \
    --trace-env docker

# For SLURM with Podman, source the helper first
source docker/setup_docker_runtime.sh
python -m data.local.run_tracegen --trace-env docker ...
```

Docker backend configs are in `hpc/harbor_yaml/trace_docker_*.yaml`.

**Runtime Detection** (`hpc/docker_runtime.py`):
- Auto-detects Docker vs Podman
- Sets `DOCKER_HOST` environment variable
- Supports SSH tunnels to remote Docker daemons

### Database System

`database/unified_db/`: Supabase-backed registry for datasets, models, and agents
- Auto-fills 9-12 metadata fields per entry
- Supports HuggingFace and local file registration
- Python API: `register_hf_dataset()`, `register_local_parquet()`, `register_hf_model()`, `register_agent()`

## Common Commands

### DCFT SFT Training (hpc/)

Setup:
```bash
# Initialize LLaMA-Factory submodule
git submodule update --init --remote dcft/train/llamafactory

# Install dependencies
uv pip install -r hpc_requirements.txt

# Source environment (cluster-specific)
source /PATH/TO/ot-agent/hpc/dotenv/tacc.env  # or jureca.env, etc.
cd $DCFT
$DCFT_ACTIVATE_ENV
```

Launch training:
```bash
python3 -m hpc.launch \
  --train_config_path dcft/train/hp_settings/paper/reasoning_medium.yaml \
  --time_limit 24:00:00 \
  --num_nodes 16 \
  --dataset mlfoundations-dev/your-dataset

# Dry run (preview without submitting)
python3 -m hpc.launch --dry_run [other args]
```

Helper commands (defined in `hpc/scripts/common.sh`):
```bash
gotrain <name>   # Standard (medium) hyperparameters
gosmall <name>   # Small scale training
golarge <name>   # Large scale training
gofast <name>    # More GPUs for faster training
goeval <name>    # Eval on pipeline evals
fulleval <name>  # Full reasoning evals including held-out
```

### OT-Agent RL Training (train/hpc/)

Setup:
```bash
cd train
source hpc/setup.sh  # Auto-detects cluster and loads environment
```

Launch training:
```bash
python3 -m hpc.launch \
  --job_type rl \
  --rl_config ./hpc/skyrl_yaml/jupiter/48GPU_base_32b.yaml \
  --model_path Qwen/Qwen3-32B \
  --time_limit 11:59:00 \
  --num_nodes 12 \
  --train_data '["DCAgent/dataset-name"]' \
  --max_restarts 8 \
  --experiments_dir /e/data1/datasets/playground/ot-baf

# Dry run
python3 -m hpc.launch --dry_run [other args]
```

**RL config files** (`hpc/skyrl_yaml/jupiter/`):
- `24GPU_base.yaml` — 8B model, 6 nodes (8 TP=1 inference engines)
- `48GPU_base_32b.yaml` — 32B model, 12 nodes (16 TP=2 inference engines, thinking)
- `48GPU_base_32b_nothink.yaml` — 32B nothink variant
- `_131k` variants — 131k context length

**Key RL defaults** (from YAML, override via CLI `trainer.X=Y` or `generator.X=Y`):
- `trainer.epochs`: 2, `trainer.max_steps`: 60
- `trainer.train_batch_size`: 64, `generator.n_samples_per_prompt`: 8
- `trainer.policy.optimizer_config.lr`: 5e-6
- `generator.sampling_params.temperature`: 0.7
- `trainer.strategy`: fsdp2, `trainer.algorithm.advantage_estimator`: rloo_n
- `--experiments_dir`: defaults to `experiments/` in repo; use `/e/data1/datasets/playground/ot-baf` for personal runs on Jupiter

Helper commands (same as DCFT):
```bash
gotrain <name>   # Standard training
gosmall <name>   # Small scale
golarge <name>   # Large scale
gofast <name>    # Fast training
```

### Job Monitoring (both systems)

```bash
sqme              # Show your queued jobs
status [lines]    # Show job status and recent logs
sfail [hours]     # Show failed jobs in last N hours
swin [hours]      # Show completed jobs in last N hours
soops [hours]     # Show cancelled jobs in last N hours
sinf              # Show formatted cluster information

# Tail logs
tail $DCFT/experiments/logs/<job_name>_<job_id>.out
tail $CHECKPOINTS_DIR/<job_name>/trainer_log.jsonl

# Cancel jobs
scancel <job_id>
scancel -u $USER -t PENDING  # Cancel all pending jobs
scancelall                    # Cancel all your jobs

# Cleanup
rmlogs [threshold]  # Remove old log files
```

### Database Commands

```bash
# Install database CLI
cd database/unified_db
pip install -r requirements.txt

# Setup environment
export SUPABASE_URL=your_url
export SUPABASE_ANON_KEY=your_key
export SUPABASE_SERVICE_ROLE_KEY=your_service_key

# Apply schema
psql $DATABASE_URL -f complete_schema.sql
```

## Key Implementation Details

### HPC Auto-Detection

Both launchers use `hpc.py:detect_hpc()` to automatically detect the cluster from hostname:
- Matches hostname against regex patterns for each cluster
- Returns HPC configuration object with cluster-specific settings
- If not recognized, raises ValueError

### Pre-Download System (JSC/No-Internet Clusters)

For JSC clusters, `train/hpc/launch.py:pre_download_dataset()`:
1. Runs on login node (which has internet access)
2. Uses `huggingface_hub.snapshot_download()` for datasets and models
3. Downloads to `HF_HUB_CACHE` directory
4. Compute nodes then use cached files during training

### SLURM Job Templates

Templates use Jinja2 for dynamic generation:
- `hpc/sbatch/*.sbatch`: DCFT training templates
- `train/hpc/sbatch/*.sbatch`: RL training templates
- Variables substituted from CLI arguments and HPC config

### Environment Variables

Critical environment variables (set in `dotenv/*.env`):
- `HF_TOKEN`: HuggingFace API token
- `WANDB_TOKEN`: Weights & Biases token
- `HF_HUB_CACHE`: Cache directory for HF datasets/models
- `DCFT`: Base directory for DCFT system
- `DC_AGENT_TRAIN`: Base directory for RL training system
- `CHECKPOINTS_DIR`: Output directory for checkpoints

### Batch Job Submission

To launch multiple jobs at once:
```bash
cat << 'EOF' | while read -r model; do [[ -z "$model" ]] || gotrain "$model"; done
Qwen/Qwen2.5-7B-Instruct
microsoft/phi-2
meta-llama/Llama-3-8b
EOF
```

## Testing

```bash
# Test HPC detection
cd train
python3 hpc/test_hpc.py

# Test database
cd database/unified_db/test_sft_dataset_register
python test_dataset_upload.py
python test_model_upload.py
python test_agent_upload.py

# Example terminal-bench run
python example_tbench.py
```

## Important Notes

### Dependencies

This repository does NOT have managed dependencies. Each subsystem has its own:
- Data: `data/data_requirements.txt`
- HPC: `hpc/hpc_requirements.txt`
- SFT: `dcft/train/llamafactory/pyproject.toml` (install with liger-kernel, deepspeed, hf-kernels extras)
- RL: SkyRL requirements (coming soon)

### Git Submodules

LLaMA-Factory is included as a submodule:
```bash
git submodule update --init --remote dcft/train/llamafactory
```

### Configuration Files

**SFT configs**: `dcft/train/hp_settings/` and `dcft/train/lf_configs/qwen2_5/`
- Paper configs: `paper/reasoning_{small,medium,large}.yaml`

**RL configs**: Arguments passed via CLI to `train/hpc/launch.py`
- Algorithm: GRPO (default), Strategy: FSDP2 (default), Backend: vLLM (default)

### Common Pitfalls

1. **JSC pre-download**: Always ensure datasets/models are pre-downloaded on login node before job submission
2. **Node exclusions**: Some clusters have exclusion lists for faulty nodes (see `hpc.py`)
3. **Internet access**: Know whether your cluster has internet on compute nodes. JSC compute nodes use proxychains for HF access (W&B doesn't work through proxychains)
4. **LLaMA-Factory submodule**: Must be initialized before SFT training
5. **Environment sourcing**: Must source correct `dotenv/*.env` for your cluster
6. **ShareGPT role tags**: Harbor/DCAgent datasets use `role`/`content` keys with `user`/`assistant` values. LLaMA-Factory defaults to `from`/`value` with `human`/`gpt`. Always pass explicit tags for Harbor datasets:
   ```
   --role_tag role --user_tag user --assistant_tag assistant --content_tag content
   ```
   Without these, the thinking preprocessor finds 0 assistant messages and training produces garbage.
7. **push_to_hub on JSC**: Defaults to `false` (no-internet cluster). Override with `--push_to_hub true` since proxychains provides HF Hub access on compute nodes.

## JSC Jupiter Access

**SSH**: Connect with IPv4-only (`-4` required):
```bash
ssh -i ~/.ssh/id_ed25519_jsc feuer1@login01.jupiter.fz-juelich.de -4
```

**Tmux**: Sessions persist across SSH disconnects. Key sessions:
```bash
tmux ls                    # List sessions
tmux attach -t 2           # Attach to session "2" (main work session)
```

**Pre-launch preamble** (run before launching any new job — pulls latest code):
```bash
source ~/.bashrc; source ~/secrets.env; \
cd /e/scratch/jureap59/feuer1/harbor && git stash && git pull; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL && git stash && git pull; \
conda activate otagent; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
git submodule update --init --remote sft/llamafactory; \
source hpc/dotenv/jupiter.env
```
Note: `GIT_TERMINAL_PROMPT=0` prevents interactive auth prompts from blocking the shell.

**Key paths**:
- Code: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent`
- Personal data root (`$DCFT_DATA`): `/e/data1/datasets/playground/ot-baf` ← USE THIS
- HF cache (`$HF_HUB_CACHE`, `$HF_HOME`): `/e/data1/datasets/playground/ot-baf/hf_hub`
- SFT/RL checkpoints (`$CHECKPOINTS_DIR`): `/e/data1/datasets/playground/ot-baf/checkpoints/`
- Legacy shared data (avoid for new writes): `/e/data1/datasets/playground/ot` — owned by `nezhurina1`; its xet/datasets cache subdirs were created by other users (`guha1`, etc.) with `0755` perms, causing `Permission denied` on HF Xet uploads and dataset lock files. Read-only references to existing artifacts in `/ot` are fine.
- Harbor: `/e/scratch/jureap59/feuer1/harbor`

**Job management** (SLURM):
```bash
sqme                       # Show your queued/running jobs
squeue -u feuer1           # Detailed job queue
scancel <job_id>           # Cancel a job
```

**Rsync files to local** (from Mac):
```bash
rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519_jsc -4" \
  feuer1@login01.jupiter.fz-juelich.de:/remote/path /local/path
```

**Cluster details**: GH200 96GB GPUs, 48 nodes, no internet on compute nodes. Pre-download datasets/models on login node before submitting jobs.

## NERSC Perlmutter Access

**SSH**: Uses ControlMaster multiplexing (2FA required on first connect):
```bash
ssh perlmutter    # Complete 2FA once; socket persists 8h
```

**Pre-launch preamble** (run before launching any new job):
```bash
conda activate dcagent; cd $SCRATCH/OpenThoughts-Agent; git pull; \
source hpc/dotenv/perlmutter.env; source ~/secrets.env; \
git submodule update --init --remote sft/llamafactory; \
cd $SCRATCH/SkyRL; git pull; \
cd $SCRATCH/harbor; git pull; \
cd $SCRATCH/OpenThoughts-Agent;
```

**Key paths**:
- Code: `$SCRATCH/OpenThoughts-Agent`
- SkyRL: `$SCRATCH/SkyRL`
- Harbor: `$SCRATCH/harbor`

**Cluster details**: A100 80GB GPUs, internet access on compute nodes. User: `penfever`.

## ALCF Polaris Access

**SSH**: Uses ControlMaster multiplexing (2FA required on first connect):
```bash
ssh ALCFPolaris    # Complete 2FA once; socket persists 8h
```

**Pre-launch preamble** (run before launching any new job):
```bash
source ~/.bashrc && conda activate otagent && \
cd /lus/eagle/projects/CausalAlign/penfever42/code/OpenThoughts-Agent && git pull && \
cd /lus/eagle/projects/CausalAlign/penfever42/code/harbor && git pull && \
source hpc/dotenv/polaris.env && source ~/secrets.env && \
cd /lus/eagle/projects/CausalAlign/penfever42/code/OpenThoughts-Agent
```

**Key paths**:
- Code: `/lus/eagle/projects/CausalAlign/penfever42/code/OpenThoughts-Agent`
- Harbor: `/lus/eagle/projects/CausalAlign/penfever42/code/harbor`
- Data/HF cache: `/lus/eagle/projects/CausalAlign/penfever42/data/hub`
- Experiments: `/lus/eagle/projects/CausalAlign/penfever42/experiments`

**Cluster details**: A100 40GB GPUs, 4/node, 560 nodes, PBS Pro scheduler (not SLURM). Internet via proxy (`proxy.alcf.anl.gov:3128`). User: `penfever42`.

**Important**: The OT-Agent repo is `open-thoughts/OpenThoughts-Agent` (NOT `laude-institute`). Harbor is `laude-institute/harbor`.

**Package management**: Use `uv pip install` (not bare `pip`) for all installs on Polaris.

## JSC Jupiter Access

**SSH**: `ssh Jupiter` (alias in ~/.ssh/config). User: `feuer1`, group: `jureap59`.

**Key paths**:
- Code: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent`
- Experiments: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/experiments/`
- Eval logs: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/eval/jupiter/logs/`
- Eval job files: `/e/data1/datasets/playground/ot/eval_jobs/`
- HF cache: `/e/data1/datasets/playground/ot/hf_hub`
- Checkpoints (SFT/RL): `/e/data1/datasets/playground/ot/checkpoints/`
- Conda env: `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/`
- Dotenv: `hpc/dotenv/jupiter.env`
- Wheels: `/e/data1/datasets/playground/ot-baf/wheels/`

**Non-interactive SSH note**: `$DCFT_ACTIVATE_ENV` doesn't work in non-interactive SSH. Use full paths:
```bash
ssh Jupiter '/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python ...'
ssh Jupiter '/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/pip install ...'
```

**Cluster details**: GH200 96GB GPUs (aarch64), 4/node, 48 nodes, SLURM scheduler. No internet on compute (proxy via SSH tunnel on compute, direct internet on login nodes). Login nodes have direct HF Hub access.

## NERSC Perlmutter Access

**SSH**: `ssh perlmutter` (alias for `perlmutter.nersc.gov`). User: `penfever`.

**Key paths**:
- Code: `/pscratch/sd/p/penfever/OpenThoughts-Agent`
- Experiments: `/pscratch/sd/p/penfever/OpenThoughts-Agent/experiments/`
- Trace jobs (RL): `experiments/<job_name>/<job_name>/trace_jobs/`
- Trace jobs (eval): `/pscratch/sd/p/penfever/OpenThoughts-Agent/trace_jobs/`
- Home: `/global/homes/p/penfever`
- Dotenv: `hpc/dotenv/perlmutter.env`

**Cluster details**: A100 80GB GPUs, 4/node, SLURM scheduler. Internet on compute nodes. Conda env: `dcagent`.

**One-off eval launch** (via HPC launcher, not the Leonardo listener):
```bash
python -m hpc.launch \
  --job_type eval \
  --model_path laion/<model_name> \
  --tasks_input_path DCAgent2/swebench-verified-random-100-folders \
  --trace_harbor_config hpc/harbor_yaml/eval/eval_ctx32k_non_it_2x_eval_.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_8b_vllm_serve_32k_4xA100.yaml \
  --trace-n-concurrent 48 \
  --upload_to_database \
  --daytona_api_key "$DAYTONA_BASE_API_KEY" \
  --time_limit 11:59:00 \
  --num_nodes 1 \
  --gpus_per_node 4
```

**IMPORTANT — Daytona key and datagen config for Perlmutter evals:**
- **`--daytona_api_key "$DAYTONA_BASE_API_KEY"`** — required. The default
  `DAYTONA_API_KEY` (RL org) blocks declarative builds needed for SWE-bench.
  Use `DAYTONA_BASE_API_KEY` for eval jobs. Available keys in `~/secrets.env`:
  `DAYTONA_API_KEY` (RL), `DAYTONA_B_API_KEY`, `DAYTONA_BASE_API_KEY` (evals),
  `DAYTONA_DATA_API_KEY`.
- **`--datagen_config qwen3_8b_vllm_serve_32k_4xA100.yaml`** — use the A100
  variant, NOT the GH200 one. The GH200 config has `--all2all-backend pplx`
  which crashes on A100 (PPLX library not available). A100 variants:
  `qwen3_8b_vllm_serve_32k_4xA100.yaml` (8B) and
  `qwen3_32b_vllm_serve_32k_4xA100.yaml` (32B).

Replace `--tasks_input_path` with the appropriate benchmark dataset (`DCAgent/dev_set_v2`, `DCAgent2/terminal_bench_2`, etc.).

## Eval Job Submission Defaults

When submitting eval jobs via `unified_eval_listener.py`, always use these flags unless explicitly told otherwise:
- `--require-priority-list` — only eval models in the priority file
- `--n-concurrent 64` on Jupiter (48 times out with fewer concurrent trials)
- `--n-concurrent 48` on other clusters
- `--gpu-memory-util 0.85` on Leonardo (A100 64GB OOMs at 0.90+)
- `--gpu-memory-util 0.95` on all other clusters (default)
- `--pre-download` on no-internet clusters (Jupiter, Leonardo)
- `--harbor-config hpc/harbor_yaml/eval/eval_ctx32k_non_it.yaml` for 32k context models
- `--harbor-config hpc/harbor_yaml/eval/eval_ctx131k_non_it.yaml` for 131k context models

Model lists live in `eval/lists/` (`models_32b.txt`, `models_131k.txt`, `models_8b_dsv2_remaining.txt`).

### Querying Unevaled Models

Use `scripts/database/query_unevaled_models.py` to find models not yet evaluated on a benchmark family. The script resolves benchmark families via the `duplicate_of` field in Supabase (e.g. `dev_set_v2` includes `DCAgent_dev_set_v2`, `dev_set_v2_2.0x`, `openthoughts-tblite`).

```bash
# List 8B models not yet evaluated on dev_set_v2 family
python scripts/database/query_unevaled_models.py --benchmark dev_set_v2 --size 8 --exclude test_ --exclude NO_EVAL -v

# List 8B models not yet evaluated on terminal_bench_2 family
python scripts/database/query_unevaled_models.py --benchmark terminal_bench_2 --size 8 -v

# Write to priority list file
python scripts/database/query_unevaled_models.py --benchmark dev_set_v2 --size 8 -o eval/lists/models_8b_dsv2_remaining.txt

# 32B models on terminal_bench_2
python scripts/database/query_unevaled_models.py --benchmark terminal_bench_2 --size 32 -o eval/lists/models_32b_tb2_remaining.txt
```

Requires `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` env vars.

### Launching the Eval Listener on Leonardo

The eval listener (`eval/jupiter/unified_eval_listener.py`) is shared across clusters. On Leonardo, point it to the Leonardo sbatch script and ensure the SSH tunnel cert is fresh:

**Prerequisites** (from local machine, before launching):
```bash
# Refresh cert (expires ~12h) and sync to Leonardo
step ssh certificate 'bfeuer00' --provisioner cineca-hpc ~/.ssh/leonardo_daytona --no-password --insecure
ssh-keygen -R login.leonardo.cineca.it && rsync -avz -e 'ssh -i ~/.ssh/leonardo_daytona -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
  ~/.ssh/leonardo_daytona ~/.ssh/leonardo_daytona.pub ~/.ssh/leonardo_daytona-cert.pub \
  bfeuer00@login.leonardo.cineca.it:~/.ssh/
```

**Launch** (on Leonardo login node — use tmux):
```bash
# IMPORTANT: Launch in a tmux session. The listener takes minutes per model
# (pre-download) and will be killed if the SSH session closes. nohup/disown
# are NOT reliable for this — use tmux.
tmux new-session -d -s eval_listener "\
  source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
  conda activate otagent && \
  cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent && \
  source hpc/dotenv/leonardo.env && source ~/secrets.env && \
  python eval/jupiter/unified_eval_listener.py \
    --preset tb2 \
    --sbatch-script eval/leonardo/unified_eval_harbor.sbatch \
    --require-priority-list \
    --priority-file eval/lists/models_8b_tb2_remaining.txt \
    --n-concurrent 48 \
    --gpu-memory-util 0.85 \
    --harbor-config hpc/harbor_yaml/eval/eval_ctx32k_non_it_2x_eval_.yaml \
    --pre-download \
    --once --verbose --batch-size 12 \
    2>&1 | tee eval/leonardo/logs/tb2_listener_\$(date +%Y%m%d_%H%M%S).log"

# Monitor: tmux attach -t eval_listener
# Check alive: tmux ls | grep eval_listener
```

**Available presets**: `tb2` (terminal_bench_2), `v2` (dev_set_v2), `dev` (dev_set_71_tasks), `swebench`, `bfcl`, `aider`. Use `--preset` OR `--datasets`, not both.

Key differences from Jupiter:
- `--sbatch-script eval/leonardo/unified_eval_harbor.sbatch`
- `--pre-download` — essential (no internet on compute nodes)
- `--gpu-memory-util 0.85` — Leonardo A100 64GB OOMs at 0.90+
- SSH tunnel cert must be refreshed before each session
- Must use tmux (not nohup/disown) — listener is too slow for detach hacks
- proxychains4 built at `/leonardo_work/AIFAC_5C0_290/bfeuer00/proxychains/bin/proxychains4`

### Post-Submission: Verify Eval Jobs Are Running

After submitting eval jobs, wait ~15 minutes for the first jobs to start, then tail the logs to confirm they are healthy:
```bash
# Check job status
ssh <cluster> "squeue -u $USER --format='%.18i %.50j %.8T %.10M'"

# Tail the most recent log
ssh <cluster> "tail -50 <log_dir>/$(ls -t <log_dir>/ | head -1)"
```
Things to look for:
- vLLM server started successfully (health check passed)
- SSH tunnel established (Leonardo only)
- Harbor trials are running (look for `trial` or `reward` lines)
- No OOM errors, no repeated DaytonaErrors

## Harbor Job File Organization

Harbor eval jobs use a **single unified directory** per eval run at `$EVAL_JOBS_DIR/<run_tag>/`.
Run tags follow the format `eval-<SAFE_MODEL>_<SAFE_REPO>` (model first, `eval-` prefix).

A `trace_jobs/<run_tag>` symlink in the working dir points to the unified run dir so Harbor writes there directly.

**Contents of `$EVAL_JOBS_DIR/<run_tag>/`**:
- `<task_name>__<trial_id>/agent/trajectory.json` — full agent conversation trace
- `<task_name>__<trial_id>/exception.txt` — error traceback if the trial failed
- `<task_name>__<trial_id>/verifier/` — verifier output and reward
- `result.json` — aggregate results, exception stats, metrics
- `config.json` — Harbor run configuration
- `meta.env` — model, dataset, SLURM job ID, DB job ID
- `vllm.log` — vLLM server log
- `upload.log` — DB/HF upload log
- `slurm.log` — symlink to the SLURM output log

To debug DaytonaErrors or other trial failures, read `exception.txt` in the trial directory:
```bash
cat $EVAL_JOBS_DIR/<run_tag>/<task>__<id>/exception.txt
```

**Config mismatch on auto-resume**: If Harbor fails with `FileExistsError: Job directory ... already exists and cannot be resumed with a different config`, the run dir has a `config.json` from a previous run with different settings. To fix, delete only the specific stale run dir **after confirming no useful trials exist**:
```bash
# Check if the dir has any completed trials before deleting
ls $EVAL_JOBS_DIR/<run_tag>/*/result.json 2>/dev/null | wc -l
# If zero, safe to delete
rm -rf $EVAL_JOBS_DIR/<run_tag>
```

## Parsing Eval Trial Directories

Each trial lives in `<run_tag>/<task_name>__<trial_id>/` under the trace_jobs directory. Use these techniques to extract timing, progress, and health information.

### Trial directory structure
```
<task_name>__<trial_id>/
├── config.json         # Trial config (mtime ≈ trial start time)
├── trial.log           # Harbor-level log (agent setup, env build, errors)
├── result.json         # Written on completion (has timestamps + reward)
├── exception.txt       # Traceback if trial failed
├── agent/
│   ├── trajectory.json              # Main agent trajectory (ATIF format)
│   ├── trajectory.cont-N.json       # Continuation after Nth summarization
│   └── trajectory.summarization-*   # Summarization subagent traces
├── verifier/
│   ├── reward.txt                   # Raw reward value
│   ├── detailed_scores.json         # Per-test results
│   └── test-stdout.txt              # Verifier output
└── artifacts/
    └── manifest.json                # Downloaded artifacts list
```

### Key fields in result.json
```python
{
    "started_at": "2026-03-31T09:46:42+00:00",     # ISO 8601 UTC
    "finished_at": "2026-03-31T10:22:59+00:00",    # ISO 8601 UTC
    "exception_info": {                              # null if no error
        "exception_type": "AgentTimeoutError",
        "exception_message": "..."
    },
    "verifier_result": {
        "rewards": {"reward": 1.0}                  # 0.0 or 1.0 typically
    },
    "agent_info": {"name": "terminus-2", "model_info": {"name": "..."}},
    "environment_setup": {"started_at": "...", "finished_at": "..."},
    "agent_setup":       {"started_at": "...", "finished_at": "..."},
    "agent_execution":   {"started_at": "...", "finished_at": "..."},
    "verifier":          {"started_at": "...", "finished_at": "..."}
}
```

### Computing trial stats from result.json
```python
import json, os, glob, statistics
from datetime import datetime
from collections import Counter

jobs_dir = "trace_jobs/<RUN_TAG>"
durations, rewards, exceptions = [], [], []

for trial_dir in glob.glob(os.path.join(jobs_dir, "*__*")):
    result = os.path.join(trial_dir, "result.json")
    if not os.path.exists(result): continue
    with open(result) as f:
        data = json.load(f)
    s, e = data.get("started_at"), data.get("finished_at")
    if s and e:
        d = (datetime.fromisoformat(e) - datetime.fromisoformat(s)).total_seconds()
        durations.append(d)
    vr = data.get("verifier_result") or {}
    r = (vr.get("rewards") or {}).get("reward")
    if r is not None: rewards.append(r)
    exc = (data.get("exception_info") or {}).get("exception_type")
    if exc: exceptions.append(exc)

# Throughput
completed_times = sorted([datetime.fromisoformat(json.load(open(os.path.join(d, "result.json")))["finished_at"])
    for d in glob.glob(os.path.join(jobs_dir, "*__*"))
    if os.path.exists(os.path.join(d, "result.json")) and json.load(open(os.path.join(d, "result.json"))).get("finished_at")])
if len(completed_times) >= 2:
    wall = (completed_times[-1] - completed_times[0]).total_seconds()
    rate = len(completed_times) / wall * 3600  # trials/hr
```

### Detecting stalls and anomalies
```bash
# Count completed trials
find trace_jobs/<RUN_TAG> -maxdepth 2 -name "result.json" | wc -l

# Most recent result.json (stale = possible hang)
ls -lt trace_jobs/<RUN_TAG>/*/result.json | head -1

# In-flight trials (started but no result)
for d in trace_jobs/<RUN_TAG>/*__*/; do
  [ -d "$d/agent" ] && [ ! -f "$d/result.json" ] && echo "$d"
done | wc -l

# Check vLLM health (Running: 0 = idle, no agent requests)
tail -5 experiments/<RUN_TAG>/logs/<RUN_TAG>_vllm.log

# Trials stuck on Daytona env build (no trajectory, only "Building environment")
for d in trace_jobs/<RUN_TAG>/*__*/; do
  [ ! -f "$d/agent/trajectory.json" ] && [ -f "$d/trial.log" ] && \
    grep -q "Building environment" "$d/trial.log" && echo "$(basename $d)"
done
```

### Health check thresholds
- **No result.json in 60+ minutes** with job RUNNING → stall (Harbor hung or all trials in long timeout)
- **vLLM Running: 0 reqs for 10+ minutes** → agents not generating (env build stall, auth errors, or drain)
- **All trials complete but job still RUNNING** → zombie (Harbor process didn't exit, cancel immediately)
- **Repeated "Bearer token is invalid" in job.log** → Daytona auth degradation (trials will retry but waste time)

## CINECA Leonardo Access

**SSH**: Uses ControlMaster multiplexing + step-ca certificate auth:
```bash
ssh Leonardo    # Complete 2FA once; socket persists 8h
```

**Pre-launch preamble** (run before launching any new job):
```bash
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
conda activate otagent && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor && GIT_TERMINAL_PROMPT=0 git pull && \
source hpc/dotenv/leonardo.env && source ~/secrets.env && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
```

**Key paths**:
- Code: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent`
- Harbor: `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/harbor`
- Data/HF cache: `/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub`
- Experiments: `/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments`

**Cluster details**: A100 64GB GPUs, 4/node, 3456 nodes, SLURM scheduler. No internet on compute nodes (use proxychains/SSH tunnel). User: `bfeuer00`. Account: `AIFAC_5C0_290`.

**Important**: Compilers come from conda (GCC 15.2, CUDA 13.2) — do NOT load system modules (`module load gcc cuda`), they are too old.

**Max wall time**: 24 hours (`--time 23:59:00`). The boost_usr_prod partition has a 1-day limit.

### SFT Launch on Leonardo

SFT jobs use a separate conda env from eval/datagen due to different transformers requirements.

**Available conda environments**:
- `otagent` — eval, datagen, general use (transformers 4.x)
- `sft-qwen35` — Qwen3.5 SFT (transformers 5.3.0+, torch 2.10+, deepspeed 0.18+)

**Pre-launch preamble** (for SFT):
```bash
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
conda activate sft-qwen35 && \
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent && \
GIT_TERMINAL_PROMPT=0 git pull && \
source hpc/dotenv/leonardo.env && source ~/secrets.env
```

**Launch command**:
```bash
DISABLE_VERSION_CHECK=1 python -m hpc.launch \
  --train_config_path sft/lf_configs/qwen3_5/32k_9b.yaml \
  --time_limit 23:59:00 \
  --num_nodes 4 --gpus_per_node 4 \
  --dataset DCAgent/exp_tas_optimal_combined_traces \
  --role_tag role --user_tag user --assistant_tag assistant --content_tag content \
  --hub_model_id laion/exp_tas_optimal_combined_traces-Qwen3.5-9B
```

**IMPORTANT — sbatch patching required**: The launcher does NOT auto-configure conda activation or WORKDIR for SFT jobs on Leonardo. After the launcher generates the sbatch, you MUST manually patch it before submitting:

```bash
SBATCH=experiments/<exp_dir>/sbatch/<job_name>_sft.sbatch

# 1. Add conda activation
sed -i 's|# No conda activation configured|source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh\nconda activate sft-qwen35|' $SBATCH

# 2. Fix WORKDIR (defaults to $PWD which is wrong on compute nodes)
sed -i 's|WORKDIR="$PWD"|WORKDIR="/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent"|' $SBATCH

# 3. Fix DCFT
sed -i 's|export DCFT="$WORKDIR"|export DCFT="/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent"|' $SBATCH

# 4. Fix any doubled paths ($DCFT//leonardo_work/...)
sed -i 's|\$DCFT//leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/experiments|/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/experiments|g' $SBATCH

# 5. Submit
sbatch $SBATCH
```

**SFT config files**: `sft/lf_configs/qwen3_5/` — configs for Qwen3.5-9B and 27B at 32k and 131k context. These require transformers >= 5.3.0 (Qwen3.5 uses a hybrid GDN+Attention architecture not in transformers 4.x).

**Multi-node SFT**: Leonardo uses `accelerate launch` (not `torchrun`) for multi-node SFT, matching Jupiter. This is set via `training_launcher="accelerate"` in hpc.py. The `torchrun` c10d rendezvous consistently fails on Leonardo due to TCP connectivity issues between compute nodes.

### LLaMA-Factory Patching Workflow

LLaMA-Factory lives as a **git submodule** at `sft/llamafactory/`. When making changes:

1. **Edit locally** in `/Users/benjaminfeuer/Documents/LLaMA-Factory/`
2. **Commit and push** to the LLaMA-Factory repo
3. **On the cluster**, pull the changes via:
   ```bash
   cd /path/to/OpenThoughts-Agent
   git submodule update --init --remote sft/llamafactory
   ```

Do NOT rsync or manually copy files — use the git submodule workflow. The editable pip install from `/code/LLaMA-Factory/` may not take precedence over the submodule copy if both exist.

**Known transformers v5 incompatibilities** (patched in our fork):
- `AutoModelForVision2Seq` removed → falls back to `AutoModelForImageTextToText`

## Experiment Launch Command References

- **SFT experiments**: `/Users/benjaminfeuer/Documents/notes/ot-agent/sft_experiments.md`
- **RL experiments**: `/Users/benjaminfeuer/Documents/notes/ot-agent/rl_experiments.md`

When resubmitting cancelled jobs, look up the original launch command in these files first.

## Axolotl SFT on Jupiter (GH200 aarch64, CUDA 13, torch 2.9)

Used for the `/baselines/sera` and `/baselines/coderforge` v3 reference-reproduction runs. LLaMA-Factory is our default for SFT scaling sweeps; axolotl is used when the upstream paper's data pipeline requires it (e.g. SERA's OpenAI-native `messages` layout with `tool_calls` + `train` fields, or CoderForge's pre-tokenized `input_ids`/`labels`).

### Install recipe (`sera-axolotl` conda env)

Assumes `sera-axolotl` env has torch `2.9.1+cu130` preinstalled (Jupiter-safe build).

```bash
# Clone axolotl at v0.16.1
mkdir -p /e/scratch/jureap59/feuer1/code && cd /e/scratch/jureap59/feuer1/code
git clone https://github.com/axolotl-ai-cloud/axolotl.git && cd axolotl
git checkout v0.16.1

# Install axolotl from source (--no-build-isolation so setup.py sees torch 2.9.1
# and axolotl's aarch64 filter excludes torchao/fla-core/flash-linear-attention)
conda activate sera-axolotl
uv pip install "setuptools>=64" wheel "setuptools_scm>=8" "packaging==26.0"
uv pip install -e . --no-build-isolation

# deepspeed (excluded from axolotl requirements as "extra"; compiles cpu_adam)
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
uv pip install "deepspeed>=0.18.6,<0.19.0" --no-build-isolation

# flash-attn — use mjun0812's prebuilt wheels for aarch64+cu13+torch2.9
WHL=flash_attn-2.8.3+cu130torch2.9-cp312-cp312-manylinux_2_34_aarch64.whl
wget -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/$WHL" -O /tmp/$WHL
uv pip install --no-deps /tmp/$WHL

# axolotl-ai-cloud's cut-cross-entropy fork (required even though we disable
# the plugin in config — import chains still reference it on some paths)
uv pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@63b15e6" --no-build-isolation
```

### Mandatory env patches (before first run)

1. **`axolotl/utils/callbacks/qat.py`** — guard torchao imports. `setup.py` filters torchao on aarch64 but the QAT callback unconditionally imports it at module load, breaking `from axolotl.core.builders import ...`. Patch the two imports in a `try/except ImportError:` block, assigning unreachable stub classes so `isinstance()` checks in `toggle_fake_quant` return False.

2. **`convert_axolotl_checkpoint.py`** — axolotl (with gradient checkpointing) writes state_dict keys like `model.layers.N._checkpoint_wrapped_module.<param>`. vLLM / sglang can't load these. Use the helper from SERA's repo (`sera/datagen/train/convert_axolotl_checkpoint.py`) which strips the prefix. Lives at `/e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py` on Jupiter and a local copy at `/baselines/sera/convert_axolotl_checkpoint.py`.

### sbatch env must include (on Jupiter compute nodes)

```bash
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export GCC_HOME=/e/software/default/stages/2026/software/GCCcore/14.3.0
export CC=$GCC_HOME/bin/gcc
export CXX=$GCC_HOME/bin/g++              # Triton JIT kernels need these; compute nodes don't have gcc on PATH by default
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1                   # compute has no internet
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export AXOLOTL_DO_NOT_TRACK=1
export DO_NOT_TRACK=1
export HF_HUB_DISABLE_TELEMETRY=1
```

### Config gotchas (axolotl YAML)

- **Omit `hub_model_id` / `hub_strategy`**: transformers' `init_hf_repo` runs at train start and calls `create_repo` → HF API → `OfflineModeIsEnabled` crash. Push manually after training.
- **Disable `CutCrossEntropyPlugin`**: on aarch64+torch2.9+FA2, CCE causes bf16 grad explosion (grad_norm 9.8e+11) → loss → NaN → masked as 0. Comment out the entry under `plugins:`.
- **Set `max_grad_norm: 1.0` explicitly** as belt-and-suspenders.
- **Use `zero3_bf16.json` deepspeed config** (not zero1 — OOM without CCE) AND avoid the default `zero2.json` / `zero3.json` which offload optimizer to CPU → DeepSpeedCPUAdam JIT compile → `Error: unrecognized option -march=armv9-a+...+nossbs+nopauth` on GCC 14.3. `zero3_bf16.json` keeps Adam on GPU and shards everything (params + grads + moments).

### End-to-end post-training flow

```bash
conda activate otagent   # or sera-axolotl — either works for the CLI
source /e/scratch/jureap59/feuer1/OpenThoughts-Agent/hpc/dotenv/jupiter.env
source ~/secrets.env
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent

IN=$CHECKPOINTS_DIR/<jobname>__Qwen3-8B
OUT=$CHECKPOINTS_DIR/<jobname>__Qwen3-8B-converted

# 1. Strip _checkpoint_wrapped_module prefixes (required for vLLM/sglang)
python /e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py "$IN" "$OUT"

# 2. Secret scan before upload
grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34})' "$OUT"

# 3. Upload to HF (axolotl's `hub_strategy: end` is disabled per above)
huggingface-cli upload-large-folder laion/<hub_model_id> "$OUT" --repo-type=model

# 4. Register in Supabase
python scripts/database/manual_db_push.py \
  --hf-model-id laion/<hub_model_id> \
  --base-model Qwen/Qwen3-8B \
  --dataset-name <dataset_repo>
```

## Manual Eval Upload (when auto-upload fails)

When an eval job completes but the automatic HF upload or DB sync fails (e.g., path mismatch, missing result.json), use `manual_db_eval_push.py` to manually trigger the upload:

```bash
# On the cluster (source secrets first)
source ~/secrets.env
cd /path/to/OpenThoughts-Agent

# Basic usage — auto-detects agent/model/benchmark from job metadata
python scripts/database/manual_db_eval_push.py \
    --job-dir trace_jobs/<RUN_TAG> \
    --verbose

# With explicit HuggingFace repo
python scripts/database/manual_db_eval_push.py \
    --job-dir trace_jobs/<RUN_TAG> \
    --hf-repo DCAgent2/<RUN_TAG>-traces \
    --verbose

# Skip HF upload (database only)
python scripts/database/manual_db_eval_push.py \
    --job-dir trace_jobs/<RUN_TAG> \
    --skip-hf --verbose

# Force update existing records
python scripts/database/manual_db_eval_push.py \
    --job-dir trace_jobs/<RUN_TAG> \
    --forced-update --verbose
```

**Important**: Pass the `trace_jobs/<RUN_TAG>` path (where Harbor writes trial dirs), NOT the `eval_jobs/<RUN_TAG>` path (which only has meta.env). The script auto-resolves nested directories to find trial subdirectories.

**CRITICAL — verify the model name in the DB after upload**: The script auto-detects
the model from trial `result.json` → `agent_info.model_info.name`. For vLLM-served
models, this field contains the **vLLM served model name** (a numeric ID like
`1774950145766573`), NOT the HuggingFace model name. The script may register a
bogus model entry with this numeric name instead of the real HF name.

To get the correct model name, read the eval config:
```bash
python3 -c "import json; d=json.load(open('experiments/<RUN_TAG>/configs/<RUN_TAG>_eval_config.json')); print(d['model_hf_name'])"
```

After running `manual_db_eval_push.py`, verify the model name in Supabase:
```python
# Check what model name was registered
c.table("sandbox_jobs").select("model_id").eq("id", "<JOB_ID>").execute()
c.table("models").select("name").eq("id", "<MODEL_ID>").execute()
```

If the model name is a numeric ID instead of an HF repo name, fix it:
```python
# Find the correct model by HF name
correct = c.table("models").select("id").eq("name", "laion/<real-model-name>").execute()
# Update the job
c.table("sandbox_jobs").update({"model_id": correct.data[0]["id"]}).eq("id", "<JOB_ID>").execute()
# Update trial_model_usage rows
c.table("sandbox_trial_model_usage").update({"model_id": correct.data[0]["id"]}).eq("model_id", "<BOGUS_ID>").execute()
# Delete the bogus model entry
c.table("models").delete().eq("id", "<BOGUS_ID>").execute()
```

**Required env vars**: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `HF_TOKEN` (from `secrets.env`).

## RL Job Monitoring Format

When reporting RL job progress, use this table format:

```
┌─────────────────────────┬───────┬────────┬─────────────┬───────────┬─────────────────────────────────────────┐
│           Job           │ Step  │ Reward │ Policy Loss │ Grad Norm │                  Trend                  │
├─────────────────────────┼───────┼────────┼─────────────┼───────────┼─────────────────────────────────────────┤
│ SWE-rebench 8B (shaped) │ 15/80 │ 0.619  │ -0.0040     │ 0.006     │ Checkpoint saved. Slight dip from 0.652 │
├─────────────────────────┼───────┼────────┼─────────────┼───────────┼─────────────────────────────────────────┤
│ Code-contests 8B (base) │ 26/80 │ 0.451  │ -0.0930     │ 0.021     │ Stable, gradients strong                │
└─────────────────────────┴───────┴────────┴─────────────┴───────────┴─────────────────────────────────────────┘
```

Use box-drawing characters for the table borders. Include columns: Job, Step, Reward, Policy Loss, Grad Norm, Trend. Use a separate table for new jobs that are still filling their generation buffer.

### Metrics to track per RL run (in this priority order)

The minimal status table above carries the primary signals (step, reward, policy_loss, grad_norm). When doing deeper diagnostic passes — especially if a run is showing decline, instability, or you're investigating collapse — also pull these from the wandb run / `.out` log:

**Always track when reporting (the 5 core metrics):**
- `reward/avg_raw_reward` — primary learning signal
- `reward/avg_pass_at_8` — n=8 pass rate; less noisy than raw_reward when group composition fluctuates
- `policy/policy_loss` — bounded by clipping, mostly diagnostic for "is the loss flowing"
- `policy/policy_entropy` — distribution concentration. Both *direction* and *magnitude* of changes matter. Sudden drops (slow concentration) and sudden jumps (semantic confusion) are both pre-collapse signatures.
- `policy/raw_grad_norm` — the most predictive single signal for collapse. In healthy phases (this codebase + 32k context + Qwen3-derived bases) ‖g‖ stays < 1.0; values above 1.0 for ≥2 consecutive steps have predicted collapse 2-5 steps before reward visibly degrades.

**Track the clip ratio whenever you have wandb access (we did NOT track this in older runs):**
- `policy/ppo_clip_ratio` — fraction of tokens hitting the PPO surrogate clip. With our default eps_clip=0.2 + lr=8e-6 + 1 update epoch per batch this metric is essentially always ≈0 (max observed 0.03%) because ratios stay near 1.0. Useful as a *consistency check*: if clip_ratio is non-trivially elevated (>1%), it means the LR ↔ eps_clip configuration is mismatched (eps_clip is too tight for the LR regime, OR multiple update epochs are letting ratios drift). An elevated clip_ratio with a stable training run is a reason to re-examine the trust-region config.

**Per-token log-ratio diagnostics (added to SkyRL 2026-05-06; available on runs launched from that commit onward):**
- `policy/log_ratio_abs_mean` / `policy/log_ratio_abs_p99` / `policy/log_ratio_abs_max` — distribution of |log r_t| across all response tokens in a batch. Identifies whether updates are spread across many tokens with similar magnitudes (mean rises, max stays bounded → within-batch *correlation* is amplifying ‖g‖) vs. concentrated on a few outlier tokens (max spikes far above mean → individual-token-blowup mode).
- `policy/n_tokens_dp_gt_1pct` / `n_tokens_dp_gt_10pct` / `n_tokens_dp_gt_50pct` — how many tokens moved more than 1% / 10% / 50% in probability. Big rises in `_gt_10pct` while `_gt_50pct` stays flat = correlation mode. Rises across all three together = both correlation + outliers.
- `policy/log_ratio_abs_pos00..pos90` — mean |log r_t| binned into 10 relative-position buckets (0-10%, 10-20%, ..., 90-100% of each row's response length). Useful for seeing *where in the response* the policy is concentrating updates. Useful learning often concentrates on early tokens (decision branches); degenerate patterns (think-spam, JSON-loops) on late tokens (autoregressive trap).

When a run is healthy, these per-token metrics should show: `log_ratio_abs_mean` ~ 0.005-0.02, `log_ratio_abs_max` < 0.5, `n_tokens_dp_gt_50pct` near 0, position buckets roughly equal. Any sustained departure from this profile alongside rising grad_norm is a collapse warning.

**Collapse warning rule (combine signals — single metrics are noisy):**
A run is at meaningful collapse risk when ≥2 of the following fire in the same logged step:
- `policy/raw_grad_norm` > 1.0 (or > 2× recent-window mean)
- `policy/policy_entropy` deviates from its rolling 10-step trend by > 30% (in either direction)
- `policy/log_ratio_abs_mean` rises > 2× its recent-window mean while `log_ratio_abs_max` stays bounded (within-batch correlation rising — distinct from individual-token outliers)
- Trial-level reward distribution (from `parse_skyrl_metrics.py` if available) shows pass-rate < 10% for last 100 trials

A single metric crossing a threshold is suggestive but not actionable; two-of-the-above firing simultaneously is the threshold for cancel + salvage per the RL Job Cleanup Checklist below.

## RL Job Cleanup Checklist

After an RL job terminates (early or completed), follow these steps to preserve and publish the checkpoint:

0. **Cancel pending retries**: Before anything else, cancel any queued retry jobs for the same run so they don't start while you're uploading:
   ```bash
   squeue -u $USER --format='%.18i %.80j %.8T' | grep <job_name>
   scancel <retry_job_ids>
   ```

1. **Locate the best checkpoint** (by reward) in the exports folder:
   ```bash
   # NOTE: There is an empty exports/ dir at the base level — ignore it.
   # The real HF-exportable checkpoints are in the nested subdir:
   ls -lt $EXPERIMENTS_DIR/<job_name>/<job_name>/exports/ | head -10
   ```
   Check `trainer_log.jsonl` for `reward/avg_raw_reward` at each step to identify the checkpoint with the highest reward. Upload that one, not necessarily the last one (reward can degrade in later steps).

2. **Locate the W&B run**: Check the job logs or `trainer_log.jsonl` for the wandb run URL. Format: `https://wandb.ai/dogml/OpenThoughts-Agent/runs/<run_id>`

3. **Flatten model files to upload dir root**: The HF model files **must be at the base** of the directory you upload — not nested in `policy/` or any subdirectory. Copy from the export's `policy/` subdir into a flat staging dir:
   ```bash
   UPLOAD_DIR=/e/scratch/jureap59/feuer1/upload_staging/<job_name>-<step>
   mkdir -p $UPLOAD_DIR
   cp $EXPORT_DIR/policy/* $UPLOAD_DIR/
   # Verify: safetensors, config.json, tokenizer files should all be at the root
   ls $UPLOAD_DIR/
   ```

4. **Copy the launch config**: Copy the RL YAML config used to launch the job into the model folder for reproducibility:
   ```bash
   cp hpc/skyrl_yaml/<config_used>.yaml $CHECKPOINT_DIR/rl_config.yaml
   ```

5. **Scan for secrets**: Before uploading, scan the checkpoint dir and traces for leaked API keys/tokens. HuggingFace runs [TruffleHog](https://huggingface.co/docs/hub/en/security-secrets) post-upload, but we should catch secrets *before* they hit the Hub:
   ```bash
   # If trufflehog is installed:
   trufflehog filesystem $CHECKPOINT_DIR --no-update
   # Also scan the experiment logs/traces dir:
   trufflehog filesystem $EXPERIMENTS_DIR/<job_name>/<job_name> --no-update
   # If trufflehog is not available, use grep as a fallback:
   grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34}|eyJ[a-zA-Z0-9._-]+)' $CHECKPOINT_DIR
   ```
   If any secrets are found, remove or redact them before proceeding.

6. **Upload to HuggingFace**: Use `huggingface-cli upload-large-folder` to push to `laion/<job_name>-<step>-<size>` (append the global step AND the base-model size suffix, e.g. `-20-32B` for step 20 of a 32B model, `-20-8B` for an 8B model). The size suffix is required:
   ```bash
   huggingface-cli upload-large-folder laion/<job_name>-<step>-<size> $UPLOAD_DIR
   ```
   **Naming convention:** the SkyRL trainer auto-pushes intermediates to a
   *canonical* repo `laion/<job_name>` with the wrong layout (weights nested
   under `checkpoints/step_N/` instead of root) and auto-registers it in
   Supabase. We bypass that by uploading the manually-flattened export to
   the `-<step>-<size>` repo with weights at root.

7. **Register in DB**: First, delete the trainer's auto-registered duplicate, then push the correct row.

   - **Delete the trainer's auto-registered Supabase row** (it points to the wrong-layout canonical repo):
     ```python
     # Find the row by name and delete it
     c.table("models").delete().eq("name", "laion/<job_name>").execute()
     ```
   - **Optionally also delete the trainer's auto-uploaded canonical HF repo**:
     ```python
     from huggingface_hub import HfApi
     HfApi().delete_repo("laion/<job_name>", repo_type="model")
     ```
   - **Then register the manually-uploaded `-<step>-<size>` repo** via `scripts/database/manual_db_push.py` with `--training-type RL` (the script defaults to SFT — passing RL is required, otherwise you create a second wrong-type row):
     ```bash
     # Single dataset:
     python scripts/database/manual_db_push.py \
       --hf-model-id laion/<job_name>-<step>-<size> \
       --base-model <base_model_hf> \
       --dataset-name <dataset_name> \
       --training-type RL

     # Multi-dataset (comma-separated → sets dataset_names instead of dataset_id):
     python scripts/database/manual_db_push.py \
       --hf-model-id laion/<job_name>-<step>-<size> \
       --base-model <base_model_hf> \
       --dataset-name "DCAgent/dataset-a,DCAgent/dataset-b" \
       --training-type RL
     # --wandb-run is optional (timestamps default to now if omitted; Jupiter has no W&B)
     ```

   **IMPORTANT — verify `--base-model` carefully**: The `--base-model` flag must be
   the exact HF repo name of the SFT/base model that RL was trained *from* — NOT
   a default or the most common base. The base is encoded in the job name suffix
   (e.g. `__exp_tas_optimal_comb` → `laion/exp_tas_optimal_combined_traces`,
   `__GLM-4_7-swesmith-san` → `laion/GLM-4_7-swesmith-sandboxes-with_tests-...`).
   Cross-check against the RL config YAML or the launch command in
   `notes/ot-agent/rl_experiments.md`. Getting this wrong corrupts the base_model_id
   tree used for size classification and RL bump analysis.

8. **Upload RL traces**: Upload the training traces from the job:
   ```bash
   python -m scripts.harbor.make_and_upload_trace_dataset \
     --job_dir "$EXPERIMENTS_DIR/<job_name>/<job_name>" \
     --repo_id penfever/<job_name> \
     --episodes last
   ```

9. **Parse metrics and preserve training logs**: Run the metrics parser to generate tables and plots, then upload the logs and analysis alongside the model on HF. This is especially important on Jupiter where W&B is unavailable.
   ```bash
   # Generate metrics CSV, markdown report, and reward plot
   python scripts/analysis/parse_skyrl_metrics.py \
     $EXPERIMENTS_DIR/<job_name>/logs \
     $UPLOAD_DIR/training_logs \
     --trace_jobs_dir $EXPERIMENTS_DIR/<job_name>/<job_name>/trace_jobs

   # Also copy raw logs for archival
   cp $EXPERIMENTS_DIR/<job_name>/<job_name>/trainer_log.jsonl $UPLOAD_DIR/training_logs/
   cp $EXPERIMENTS_DIR/<job_name>/logs/<job_name>_*.out $UPLOAD_DIR/training_logs/

   # Re-upload the model folder (now includes training_logs/)
   huggingface-cli upload-large-folder laion/<job_name>-<step>-<size> $UPLOAD_DIR
   ```
   This produces: `metrics.csv`, `vllm_metrics.csv`, `trial_stats.csv`, `report.md`, `reward_plot.png` in `training_logs/`.

   **WARNING**: Do NOT use `huggingface_hub.upload_folder()` Python API to add files to an existing repo without setting `delete_patterns=[]`. By default it deletes files not present in the local folder, which will clobber existing model weights. Always use `huggingface-cli upload-large-folder` (which is additive) or pass `delete_patterns=[]` explicitly.

10. **Clean up experiments dir**: Only after all above steps succeed, remove the local job directory to free disk space.

## 8B SFT Job Cleanup Checklist

After an 8B SFT job completes on a no-internet cluster (Jupiter, Leonardo), follow these steps to publish and clean up:

0. **Cancel pending retries** before anything else, so stale restarts don't start
   while you're uploading or after you've cleaned up:
   ```bash
   squeue -u $USER --format='%i %j %T' | grep <job_name> | grep PENDING | awk '{print $1}' | xargs -r scancel
   ```

1. **Remove intermediate checkpoints** before uploading to avoid uploading unnecessary cruft:
   ```bash
   rm -rf $CHECKPOINTS_DIR/<job_name>/checkpoint-*
   rm -rf $CHECKPOINTS_DIR/<job_name>/.cache
   ```

1b. **Qwen 3.5 only — copy `preprocessor_config.json` from the base model**:
   LLaMA-Factory does not emit `preprocessor_config.json` during SFT, but some
   inference engines (e.g. vLLM) require it. Without it, the model may fail to
   load or produce garbled output. Copy it from the base model before uploading:
   ```bash
   # For Qwen3.5-9B:
   cp /path/to/Qwen3.5-9B/preprocessor_config.json $CHECKPOINTS_DIR/<job_name>/
   # For Qwen3.5-27B:
   cp /path/to/Qwen3.5-27B/preprocessor_config.json $CHECKPOINTS_DIR/<job_name>/
   ```

2. **Upload model weights to HuggingFace**:

   **Naming convention:**
   - **Full final upload (training reached 100%)**: use the configured
     `--hub_model_id` from the launch command (typically `laion/<descriptive_name>`,
     no `-step` or `-size` suffix). Do NOT use the job name verbatim — the launch
     command's `--hub_model_id` is the canonical name.
   - **Partial salvage upload (uncommon — only when explicitly requested)**: use
     `laion/<job_name>-<step>-<size>` (e.g. `laion/<job_name>-2400-32B` for a
     step-2400 partial of a 32B model). Default policy is **don't upload partials**;
     relaunch and let chain restarts auto-resume from the latest checkpoint.

   ```bash
   # On the login node (has direct internet on Jupiter; use proxychains on Leonardo)
   source ~/secrets.env
   huggingface-cli upload-large-folder \
     <hub_model_id> \
     $CHECKPOINTS_DIR/<job_name> \
     --repo-type=model
   ```
   Wait for the upload to finish and verify the repo exists on HF Hub.

3. **Register in the unified DB** (W&B run is optional — Jupiter has no W&B):
   ```bash
   # Single dataset:
   python scripts/database/manual_db_push.py \
     --hf-model-id <hub_model_id> \
     --base-model <base_model_hf> \
     --dataset-name <dataset_name>

   # Multi-dataset (comma-separated → sets dataset_names instead of dataset_id):
   python scripts/database/manual_db_push.py \
     --hf-model-id <hub_model_id> \
     --base-model <base_model_hf> \
     --dataset-name "DCAgent/dataset-a,DCAgent/dataset-b"
   ```
   `manual_db_push.py` defaults to `--training-type SFT`, so no flag needed for
   SFT jobs. (RL jobs require `--training-type RL` — see RL Cleanup Checklist.)

4. **Clean up experiments dir**: Only after steps 1-3 succeed, remove the local experiment directory to free disk space:
   ```bash
   rm -rf $EXPERIMENTS_DIR/<job_name>
   ```

## 32B SFT Job Cleanup Checklist

For 32B SFT (and any SFT run using DeepSpeed ZeRO-3 sharding without
`stage3_gather_16bit_weights_on_model_save: true`), the trainer writes sharded
ZeRO-3 state into `checkpoint-N/global_stepN/` instead of consolidated
safetensors at root. You must consolidate before uploading. Most steps mirror
the 8B checklist; only the consolidate step + the manual upload location
change.

0. **Cancel pending retries** (same as 8B):
   ```bash
   squeue -u $USER --format='%i %j %T' | grep <job_name> | grep PENDING | awk '{print $1}' | xargs -r scancel
   ```

1. **Verify training reached 100%** — open `trainer_log.jsonl` and confirm
   `current_steps == total_steps`. Per `feedback_no_partial_checkpoint_uploads`,
   default policy is **don't salvage-upload partials**; relaunch and let chain
   restarts auto-resume. Only proceed to step 2 if explicitly OK'd as a partial.

2. **Run consolidate** to convert ZeRO-3 shards → fp32 state_dict → safetensors:
   ```bash
   python -m hpc.launch \
     --job_type consolidate \
     --consolidate_input $CHECKPOINTS_DIR/<job_name> \
     --consolidate_output_repo <hub_model_id> \
     --consolidate_workdir <writable_workdir>/<job_name> \
     --time_limit 02:00:00 \
     --num_nodes 1
   ```
   This produces `<workdir>/<job_name>/final_repo/` containing
   `model-NNNN-of-MMMM.safetensors` + tokenizer + config files at root.

   The consolidate job ALSO attempts an HF push at the end. **Do not rely on
   that auto-push — it has historically hit a `BrokenPipeError` at
   `api.create_commit` for very large 32B uploads** (observed for the
   tasrep-...-2400 case). Treat the consolidate job as having succeeded as
   long as `final_repo/` is fully written; do the HF upload manually in
   step 3.

3. **Manually upload from `final_repo/` to HuggingFace** (NOT from the original
   checkpoint dir — that still contains ZeRO-3 shards):

   **Naming convention** (same as 8B):
   - Full final upload: use the configured `--consolidate_output_repo` /
     `--hub_model_id` (typically `laion/<descriptive_name>`, no suffix).
   - Partial salvage (uncommon, only when explicitly OK'd): use
     `laion/<job_name>-<step>-32B`.

   ```bash
   source ~/secrets.env
   huggingface-cli upload-large-folder \
     <hub_model_id> \
     <consolidate_workdir>/<job_name>/final_repo \
     --repo-type=model
   ```

4. **Register in the unified DB** (same flow as 8B; SFT is the default
   `--training-type`):
   ```bash
   python scripts/database/manual_db_push.py \
     --hf-model-id <hub_model_id> \
     --base-model <base_model_hf> \
     --dataset-name <dataset_name>
   ```

5. **Clean up**: only after steps 2-4 succeed, remove BOTH the original
   sharded checkpoint dir AND the consolidate workdir to free disk space (32B
   sharded checkpoint is ~700GB, consolidate workdir adds ~200GB):
   ```bash
   rm -rf $CHECKPOINTS_DIR/<job_name>
   rm -rf <consolidate_workdir>/<job_name>
   ```

**Recognition heuristic — when do I need this checklist vs the 8B one?**
After training completes, check the checkpoint root:
```bash
ls $CHECKPOINTS_DIR/<job_name>/ | grep -E 'safetensors|global_step'
```
- `model-*.safetensors` at root → 8B path (or Qwen3.5 — no consolidate needed)
- `global_stepN/` + `zero_to_fp32.py` at root, no safetensors → 32B path
  (this checklist)

## NYU Torch Access

**SSH**: `ssh torch` (alias in ~/.ssh/config). User: `bf996`. Requires `-o StrictHostKeyChecking=no` (host keys rotate).

**Pre-launch preamble** (run before launching any new job):
```bash
cd ~/harbor && git pull && \
cd /scratch/bf996/SkyRL && git pull && \
cd /scratch/bf996/OpenThoughts-Agent && \
conda activate dcagent312 && \
source hpc/dotenv/nyutorch.env && source ~/secrets.env && \
git pull && git submodule update --init --remote sft/llamafactory
```

**Key paths**:
- Code: `/scratch/bf996/OpenThoughts-Agent`
- SkyRL: `/scratch/bf996/SkyRL`
- Harbor: `~/harbor`
- Conda env: `dcagent312` (Python 3.12, PyTorch 2.9+cu128, vLLM 0.16+)
- Conda python: `/scratch/bf996/miniconda3/envs/dcagent312/bin/python`

**Cluster details**: H200 141GB GPUs (8/node, 29 nodes = 232 GPUs) + L40S 48GB GPUs (4/node, 68 nodes = 272 GPUs). SLURM scheduler. Internet on compute nodes. NVIDIA driver 580.82, CUDA 13.0.

**GPU partitions** (use `--partition`):
- `h200_tandon` — primary H200 partition (up to 112 GPUs)
- `h200_tandon,h200_public` — fallback combo for H200
- `h200_public` — shared H200 (up to 24 GPUs)
- `l40s_public` — shared L40S (up to 208 GPUs)
- `l40s_courant` — Courant L40S (up to 52 GPUs)

**QOS limits** (wall time):
- `gpu48` — 2 day max, 2000 job limit
- `gpu168` — 7 day max, 50 job limit
- `gpuplus` — 7 day max, 50 job limit
- `interactive` — 6 hour max, 20 job limit

**SLURM account**: `torch_pr_40_tandon_advanced`

**Interactive session**:
```bash
srun --gres=gpu:h200:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=8 --mem=32GB --time=04:00:00 --account=torch_pr_40_tandon_advanced --pty /bin/bash
```

**Package management**: Use `uv pip install` for all installs on Torch.

**Datagen on Torch**:

Before launching datagen, you must first extract tasks from the source parquet dataset:
```bash
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet mlfoundations-dev/ling-coder-sft-sandboxes-1 \
  --output_dir $SCRATCH/tasks/ling-coder-sft-sandboxes-1 \
  --on_exist overwrite
```

Then launch the datagen job:
```bash
python3 -m hpc.launch \
  --job_type datagen \
  --trace_harbor_config "./hpc/harbor_yaml/datagen/ctx32k.yaml" \
  --datagen_config kimi_k2_5_vllm_serve_torch_h200.yaml \
  --tasks_input_path "$SCRATCH/tasks/stackexchange-tezos-sandboxes" \
  --trace_target_repo DCAgent2/Kimi-2.5-stackexchange-tezos-sandboxes-maxeps-32k \
  --time_limit 47:59:00 \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --trace-n-concurrent 20
```

Key flags: `--datagen_config` selects the vLLM serving config (model, TP, etc.), `--tasks_input_path` points to the extracted tasks dir, `--trace_target_repo` is the HF repo for output traces.

**Rsync files to local** (from Mac):
```bash
rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null" \
  bf996@login.torch.hpc.nyu.edu:/scratch/bf996/path /local/path
```

## Code Ownership (DRIs)

- Data: Etash (`EtashGuha`)
- RL: Tyler and Charlie (`tyler-griggs`, `CharlieFRuan`)
- SFT: Ben (`penfever`)
- Eval: Negin (`neginraoof`)
