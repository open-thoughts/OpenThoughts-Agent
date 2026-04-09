# Eval System — Cluster Onboarding Guide

Set up the cluster-agnostic eval system on a new HPC cluster. After these steps you can run evals with `git pull` + a listener command.

## Architecture

```
Listener (login node)
  └── submits sbatch jobs ──> SLURM scheduler
                                  │
                              Compute node
                                  ├── vLLM server (TP across GPUs)
                                  ├── Harbor orchestrator (N concurrent trials)
                                  │     └── Daytona cloud sandboxes (Docker)
                                  └── Upload: HF traces + Supabase DB
```

**Key files:**

| File | Purpose |
|------|---------|
| `eval/unified_eval_listener.py` | Polls DB, submits SLURM eval jobs |
| `eval/unified_eval_harbor.sbatch` | Single-node sbatch (starts vLLM + harbor) |
| `eval/unified_eval_harbor_dp.sbatch` | Multi-node data-parallel sbatch |
| `eval/clusters/<cluster>.yaml` | Cluster config (paths, hardware, SLURM settings) |
| `hpc/dotenv/<cluster>.env` | Env vars sourced by sbatch on compute nodes |
| `eval/configs/dcagent_eval_config.yaml` | Harbor agent/orchestrator config |
| `eval/baseline_model_configs.yaml` | Per-model vLLM overrides (TP, rope, etc.) |

---

## Step 0: Gather Cluster Info

Run on the cluster and record:

```bash
sinfo --format="%P %D %G" --noheader     # partitions, node count, GPUs
sinfo -N --format="%.30N %.5c %.10G %.10m" | head -5   # CPUs, GPUs, memory per node
uname -m                                   # x86_64 or aarch64
nvidia-smi | head -3                       # GPU type, CUDA version
```

You need: **GPU count/type per node, CPUs per node, memory per node, architecture, CUDA path, SLURM partition/account, internet on compute nodes (yes/no).**

---

## Step 1: Create Cluster Config YAML

Create `eval/clusters/<cluster_name>.yaml`. This is the single source of truth for all cluster-specific settings.

```yaml
cluster_name: mycluster

# SLURM
slurm_partition: gpu
slurm_account: ""              # empty = no --account flag
slurm_time: "24:00:00"

# Conda environments: name → prefix path
conda_envs:
  otagent: /path/to/conda/envs/otagent

# Paths (use ${USER} for multi-user support)
paths:
  project_root: /path/to/OpenThoughts-Agent
  hf_cache: /path/to/.cache/huggingface/hub
  eval_jobs_dir: /path/to/OpenThoughts-Agent/jobs
  eval_logs_dir: eval/logs
  listener_logs_dir: experiments/listener_logs
  sbatch_script: eval/unified_eval_harbor.sbatch
  dp_sbatch_script: eval/unified_eval_harbor_dp.sbatch
  harbor_src: /path/to/harbor/src
  datasets_dirs:
    - /path/to/.cache/huggingface/hub
  secrets_file: ~/secrets.env

# Proxy (for clusters where compute nodes have no internet)
proxy:
  enabled: false
  # If enabled:
  # login_node: login01
  # proxychains_bin: /path/to/proxychains4

# Hardware (used for GPU/CPU scaling and job packing)
hardware:
  gpus_per_node: 8
  cpus_per_node: 128
  mem_per_node_mb: 1600000     # from sinfo; omit to skip --mem in sbatch
  arch: x86_64                 # x86_64 or aarch64
  cuda_home: /usr/local/cuda
```

Reference: `eval/clusters/jupiter.yaml` for a working example (aarch64, proxy, 4 GPUs/node).

---

## Step 2: Create Cluster Dotenv

Create `hpc/dotenv/<cluster_name>.env`. Sourced by the sbatch on compute nodes.

```bash
export SCRATCH="/path/to/$USER"
export DCFT="$SCRATCH/OpenThoughts-Agent"
export DC_AGENT_SECRET_ENV=~/secrets.env
export HF_CACHE_DIR="$SCRATCH/.cache/huggingface"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export HF_HOME="$HF_CACHE_DIR"
export DATASETS_DIR="$HF_HUB_CACHE"
export MODELS_DIR="$HF_HUB_CACHE"
export VLLM_CACHE_ROOT="$SCRATCH/.cache/vllm"
export TRITON_CACHE_DIR="$SCRATCH/.cache/triton"
export FLASHINFER_CACHE_DIR="$SCRATCH/.cache/flashinfer"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONPATH="${DCFT}${PYTHONPATH:+:$PYTHONPATH}"
```

No secrets here — those go in `~/secrets.env`.

---

## Step 3: Secrets (never commit)

Create `~/secrets.env` on the cluster:

```bash
export DAYTONA_API_KEY="dtn_..."
export DAYTONA_TARGET=""           # empty = default region
export HF_TOKEN="hf_..."
export SUPABASE_URL="https://..."
export SUPABASE_ANON_KEY="..."
export SUPABASE_SERVICE_ROLE_KEY="..."
```

See `eval/secret.env.template` for the full template.

---

## Step 4: Conda Environment

### Required packages

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12 | |
| vLLM | ≥0.16 | `pip install vllm` (x86_64) or build from source (aarch64) |
| harbor | 0.1.45+ | `pip install -e /path/to/harbor` at commit `6fdb92e7` or later |
| hf_transfer | latest | `pip install hf_transfer` (fast HF downloads) |
| supabase | latest | For DB registration |

```bash
conda create -n otagent python=3.12 -y
conda activate otagent
pip install vllm hf_transfer
pip install -e /path/to/harbor    # clone from github.com/laude-institute/harbor
pip install -e /path/to/OpenThoughts-Agent
```

### Architecture notes

- **x86_64** (H100/H200/A100): `pip install vllm` works directly
- **aarch64** (GH200): Must build vLLM from source or use a pre-built wheel. PyPI vLLM wheels are x86_64-only. See the Jupiter setup for reference.

### Optional: second env for newer models

Models like Qwen3.5 require vLLM ≥0.17 with newer transformers. Create a second env and add it to the cluster YAML:

```yaml
conda_envs:
  otagent: /path/to/envs/otagent
  otagent2: /path/to/envs/otagent2
```

Use `--conda-env otagent2` when evaluating these models.

---

## Step 5: Pre-download Datasets

Run on the login node (which has internet):

```bash
source ~/secrets.env
python eval/snapshot_download.py DCAgent/dev_set_v2
python eval/snapshot_download.py DCAgent2/terminal_bench_2
python eval/snapshot_download.py DCAgent2/swebench-verified-random-100-folders
```

---

## Step 6: Create Directories & Verify

```bash
mkdir -p eval/logs experiments/listener_logs
```

Dry-run the listener:

```bash
source ~/secrets.env
PYTHONPATH=$PWD python eval/unified_eval_listener.py \
  --cluster-config eval/clusters/<cluster>.yaml \
  --preset v2 \
  --priority-file eval/lists/example_models.txt \
  --baseline-model-configs eval/baseline_model_configs.yaml \
  --timeout-multiplier 2.0 --tp-size 2 --enable-thinking \
  --dry-run --once --verbose
```

Check for: `[v6] Cluster config: <name>`, correct sbatch params, `[DRY RUN] Would submit`, no ERROR lines.

---

## Running Evals

### Basic launch

```bash
source ~/secrets.env
PYTHONPATH=$PWD python eval/unified_eval_listener.py \
  --cluster-config eval/clusters/<cluster>.yaml \
  --preset v2 \
  --priority-file eval/lists/<models>.txt \
  --baseline-model-configs eval/baseline_model_configs.yaml \
  --timeout-multiplier 2.0 --tp-size 2 --enable-thinking \
  --auto-snapshot --once
```

### Presets

| Preset | Dataset | Notes |
|--------|---------|-------|
| `v2` | DCAgent/dev_set_v2 | Dev eval (100 tasks) |
| `tb2` | DCAgent2/terminal_bench_2 | Terminal benchmark (89 tasks) |
| `swebench` | DCAgent2/swebench-verified-random-100-folders | SWE-bench (use `--no-auto-snapshot`) |
| `aider` | DCAgent2/aider_polyglot | Aider benchmark |
| `bfcl` | DCAgent2/bfcl-parity | BFCL benchmark |

### Key flags

| Flag | Purpose |
|------|---------|
| `--tp-size {1,2,4}` | Tensor parallelism (GPUs per vLLM replica) |
| `--dp-size {1,2,4,8}` | Data parallelism (vLLM replicas per job) |
| `--n-concurrent N` | Concurrent trials per job |
| `--timeout-multiplier F` | Scale agent timeout |
| `--force-reeval` | Bypass DB status checks |
| `--resume-only` | Only submit resume jobs |
| `--max-jobs-submitted N` | Cap active SLURM jobs |
| `--conda-env NAME` | Select conda env from cluster YAML |
| `--auto-snapshot` / `--no-auto-snapshot` | Daytona snapshot control |
| `--once` | Single iteration then exit |
| `--dry-run` | Show what would be submitted |

### Monitoring

```bash
python eval/check_progress.py                    # text output
python eval/check_progress.py --live              # rich dashboard
python eval/check_progress.py --live -b tb2       # filter by benchmark
```
