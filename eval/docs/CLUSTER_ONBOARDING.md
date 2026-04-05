# Eval System — New Cluster Onboarding Guide

This guide walks you through setting up the eval system on a new HPC cluster. After following these steps, you can run evals with a single `git pull` + listener command.

## Step 0: Gather Cluster Information

Before creating any config files, collect the following about your cluster:

### Hardware (run on the cluster)
```bash
# GPU type, count per node, total nodes
sinfo -N --format="%.30N %.6t %.5c %.10G %.10m" | head -5

# Partitions
sinfo --format="%P %D %G" --noheader

# Architecture
uname -m   # x86_64 or aarch64

# CUDA version
nvidia-smi | head -3
ls /usr/local/cuda*
```

**Record these values:**
| Field | Example | Your cluster |
|-------|---------|-------------|
| GPUs per node | 8 | |
| GPU type | H200 141GB | |
| CPUs per node | 128 | |
| Memory per node (MB) | 1612647 | |
| Architecture | x86_64 | |
| CUDA path | /usr/local/cuda-12.8 | |
| SLURM partition | main | |
| SLURM account | (empty or required) | |
| Max wall time | 24:00:00 | |
| Internet on compute? | yes / no | |

### Paths (decide these)
| Path | Purpose | Example |
|------|---------|---------|
| Project root | Where OpenThoughts-Agent is cloned | `/home/$USER/OpenThoughts-Agent` |
| Harbor source | Where harbor repo is cloned | `/home/$USER/harbor` |
| HF cache | HuggingFace hub cache for models/datasets | `/home/$USER/.cache/huggingface/hub` |
| Jobs dir | Where harbor writes eval trial output | `$PROJECT_ROOT/jobs` |
| Conda prefix | Conda environment prefix directory | `/home/$USER/miniconda3/envs/otagent` |

### Network
- **Internet on compute nodes?** If no, you need a proxy setup (proxychains, SSH tunnel). See the Jupiter cluster config for an example.
- **HuggingFace Hub access?** Test with `curl -s https://huggingface.co/api/models | head -c 100`

---

## Step 1: Create the Cluster Config YAML

Create `eval/clusters/<cluster_name>.yaml`. This is the single source of truth for all cluster-specific settings.

```yaml
# eval/clusters/mycluster.yaml
cluster_name: mycluster

# SLURM
slurm_partition: gpu
slurm_account: ""              # empty = no --account flag; set if required
slurm_time: "24:00:00"         # max wall time

# Conda environments: name → prefix directory
# The listener passes this as OTAGENT_DIR to the sbatch
conda_envs:
  otagent: /path/to/miniconda3/envs/otagent

# Paths — use ${USER} for multi-user support
paths:
  project_root: /path/to/${USER}/OpenThoughts-Agent
  hf_cache: /path/to/${USER}/.cache/huggingface/hub
  eval_jobs_dir: /path/to/${USER}/OpenThoughts-Agent/jobs
  eval_logs_dir: eval/local/mycluster/logs        # relative to project_root
  listener_logs_dir: experiments/listener_logs     # relative to project_root
  sbatch_script: eval/unified_eval_harbor.sbatch   # DON'T CHANGE (cluster-agnostic)
  dp_sbatch_script: eval/unified_eval_harbor_dp.sbatch
  harbor_src: /path/to/${USER}/harbor/src
  datasets_dirs:
    - /path/to/${USER}/.cache/huggingface/hub
  secrets_file: ~/secrets.env

# Proxy (set enabled: true for no-internet compute nodes)
proxy:
  enabled: false
  # Only needed if enabled: true
  # login_node: login01
  # proxychains_bin: /path/to/proxychains4

# Hardware (critical for job packing and resource requests)
hardware:
  gpus_per_node: 8
  cpus_per_node: 128
  mem_per_node_mb: 1612647     # from `sinfo --format="%.10m"` or `free -m`
  arch: x86_64                 # x86_64 or aarch64
  cuda_home: /usr/local/cuda-12.8
```

**Commit this file** — it contains no secrets, only paths and hardware specs.

---

## Step 2: Create the Cluster Dotenv

Create `hpc/dotenv/<cluster_name>.env`. This is sourced by the sbatch on compute nodes.

```bash
# hpc/dotenv/mycluster.env
export SCRATCH="/path/to/$USER"
export DCFT="$SCRATCH/OpenThoughts-Agent"
export DC_AGENT="$DCFT"
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
export WANDB_PROJECT="OpenThoughts-Agent"
export PYTHONPATH="${DCFT}${PYTHONPATH:+:$PYTHONPATH}"
export DCFT_CONDA="$SCRATCH/miniconda3"
export PYTORCH_CUDA_ALLOW_TF32=1
export PYTORCH_CUDNN_ALLOW_TF32=1
```

**Commit this file** — it contains no secrets (secrets come from `~/secrets.env`).

---

## Step 3: Set Up Secrets (Local Only — Never Commit)

Create `~/secrets.env` on the cluster:

```bash
export DAYTONA_API_KEY="dtn_..."
export DAYTONA_TARGET=""           # empty = default region
export HF_TOKEN="hf_..."
export SUPABASE_URL="https://..."
export SUPABASE_ANON_KEY="..."
export SUPABASE_SERVICE_ROLE_KEY="..."
```

**Never commit this file.** It's gitignored.

---

## Step 4: Set Up Conda Environment

```bash
# Create the conda env
conda create -n otagent python=3.12 -y
conda activate otagent

# Install core dependencies
pip install vllm         # or specific version for your GPU arch
pip install hf_transfer  # fast HF downloads

# Install harbor (pin to known-good commit)
cd /path/to/harbor
git checkout 6fdb92e7f5707c2b01214933f1622771784e6f67
pip install -e .

# Install the repo itself (for database utils)
cd /path/to/OpenThoughts-Agent
pip install -e .
```

### Architecture-specific notes

| GPU arch | vLLM install | Notes |
|----------|-------------|-------|
| x86_64 (H100/H200/A100) | `pip install vllm` | Standard install |
| aarch64 (GH200) | Build from source or use pre-built wheel | See Jupiter setup |

### Optional: otagent2 env (for newer models)

Some models (Qwen3.5, GLM-4.7) require newer vLLM (≥0.17). Create a second env:

```bash
conda create -n otagent2 python=3.12 -y
conda activate otagent2
pip install vllm>=0.17 hf_transfer
pip install -e /path/to/harbor
```

Add it to your cluster YAML:
```yaml
conda_envs:
  otagent: /path/to/envs/otagent
  otagent2: /path/to/envs/otagent2
```

---

## Step 5: Pre-download Datasets

Do this on the login node (which has internet). This avoids race conditions when multiple sbatch jobs try to download simultaneously.

```bash
source ~/secrets.env
python eval/snapshot_download.py DCAgent/dev_set_v2
python eval/snapshot_download.py DCAgent2/terminal_bench_2
python eval/snapshot_download.py DCAgent2/swebench-verified-random-100-folders
```

For no-internet clusters, also pre-download models you plan to eval:
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('DCAgent/a1-nl2bash')"
```

---

## Step 6: Create Log Directories

```bash
mkdir -p eval/local/<cluster_name>/logs
mkdir -p experiments/listener_logs
```

---

## Step 7: Verify Setup (Dry Run)

```bash
source ~/secrets.env
PYTHONPATH=$PWD python eval/unified_eval_listener.py \
  --cluster-config eval/clusters/<cluster_name>.yaml \
  --preset v2 \
  --priority-file eval/lists/a1_nl2bash.txt \
  --require-priority-list \
  --baseline-model-config eval/baseline_model_configs.yaml \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --enable-thinking \
  --slurm-time 24:00:00 \
  --dry-run --once --verbose
```

Check the output for:
- `[v6] Cluster config: <cluster_name>` — cluster detected
- `Sbatch params: ...` — parameters look correct
- `[DRY RUN] Would submit ...` — job would be submitted
- No `ERROR:` lines about missing paths

---

## Step 8: Run for Real

```bash
source ~/secrets.env
PYTHONPATH=$PWD python eval/unified_eval_listener.py \
  --cluster-config eval/clusters/<cluster_name>.yaml \
  --preset v2 \
  --priority-file eval/lists/<model_list>.txt \
  --require-priority-list \
  --baseline-model-config eval/baseline_model_configs.yaml \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --enable-thinking \
  --slurm-partition <partition> \
  --slurm-time <max_time> \
  --once
```

### Common flag combinations

```bash
# 8B models: TP=1, DP=4 (fast, 4 replicas)
--tp-size 1 --dp-size 4 --conda-env otagent2

# 32B models: TP=4 (one replica across 4 GPUs)
--tp-size 4

# Packed jobs (multiple models per node)
--pack-jobs --stagger-delay 1 --chain-batch-size 10

# Batch with sliding window (max 32 SLURM jobs active)
--max-jobs-submitted 32 --batch-size 32
```

### Available presets

| Preset | Dataset | auto_snapshot | Typical use |
|--------|---------|---------------|-------------|
| `v2` | DCAgent/dev_set_v2 | true | Dev eval (100 tasks) |
| `tb2` | DCAgent2/terminal_bench_2 | true | Terminal benchmark |
| `swebench` | DCAgent2/swebench-verified-random-100-folders | false | SWE-bench |
| `aider` | DCAgent2/aider_polyglot | false | Aider benchmark |
| `bfcl` | DCAgent2/bfcl-parity | false | BFCL benchmark |

---

## Monitoring

```bash
# Check job status
squeue -u $USER --format="%.10i %.40j %.8T %.12M"

# Tail job output
tail -f eval/local/<cluster>/logs/data_<JOB_ID>.out

# Tail vLLM log
tail -f eval/local/<cluster>/logs/vllm_<JOB_ID>.log

# Progress dashboard
PYTHONPATH=$PWD python eval/check_progress.py --live
```

---

## Existing Cluster Configs (Reference)

| Cluster | YAML | GPUs/node | Arch | Internet | Partition | Account |
|---------|------|-----------|------|----------|-----------|---------|
| M2 (MBZ) | `eval/clusters/m2.yaml` | 8× H200 | x86_64 | yes | main | — |
| MBZ/M1 | `eval/clusters/mbz.yaml` | 8× H200 | x86_64 | yes | main | — |
| Jupiter | `eval/clusters/jupiter.yaml` | 4× GH200 | aarch64 | **no** | booster | reformo |

---

## Troubleshooting

### "ERROR: EVAL_PROJECT_ROOT not set"
You're running the sbatch directly without the listener. Use the listener with `--cluster-config`, or set the env vars manually.

### "ERROR: OTAGENT_DIR not set"
The conda env wasn't found. Check that `conda_envs` in your cluster YAML has the correct prefix path.

### Port collision (vLLM "Port X already in use" spam)
- **Without `--pack-jobs`**: The sbatch derives a unique port from `SLURM_JOB_ID`. If another user's job is on the same node, ports may collide. The sbatch will increment until it finds a free port.
- **With `--pack-jobs`**: The listener centrally assigns ports. Check the log for `Pack: <node> (GPUs X/8, port XXXXX)`.
- **With DP > 1**: Do NOT export `VLLM_PORT` as an environment variable — vLLM's DP subprocesses all read it and try to bind the same port. The sbatch passes it via `--port` CLI flag only.

### No internet on compute nodes
Set `proxy.enabled: true` in your cluster YAML and provide `proxy.login_node` and `proxy.proxychains_bin`. The sbatch will auto-configure SSH tunnel + proxychains. Pre-download all models/datasets on the login node before submitting jobs.

### Model needs newer vLLM (Qwen3.5, GLM-4.7, etc.)
Use `--conda-env otagent2` with a second conda env that has vLLM ≥0.17.
