# Setting Up OpenThoughts-Agent Eval on a New MBZUAI Cluster

## 1. Transfer Code to the New Cluster

Since we don't want to push uncommitted changes to GitHub, use `git bundle` to create a single portable file containing the full repo + all local changes.

### On the source cluster (current MBZ H200)

```bash
cd /mnt/weka/home/richard.zhuang/OpenThoughts-Agent

# Stage everything (including untracked eval/MBZ files)
git add -A

# Create a stash commit (doesn't affect your branch)
git stash

# Create bundle of the entire repo (all branches + tags)
git bundle create /tmp/openthoughts-agent.bundle --all

# Pop stash back
git stash pop

# Also bundle the uncommitted/untracked files separately
# (git bundle only contains committed objects)
tar czf /tmp/openthoughts-uncommitted.tar.gz \
    --exclude='jobs/' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.eggs' \
    eval/MBZ/ \
    hpc/dotenv/mbz.env \
    database/unified_db/utils.py \
    .gitignore

# Transfer both files to new cluster
scp /tmp/openthoughts-agent.bundle <new-cluster>:/path/to/scratch/
scp /tmp/openthoughts-uncommitted.tar.gz <new-cluster>:/path/to/scratch/
```

**Alternative: if clusters share a filesystem (same Weka mount)**, just reference the same path — no transfer needed.

**Alternative: rsync** (if you prefer a live copy instead of git bundle):
```bash
rsync -avz --exclude='jobs/' --exclude='__pycache__' --exclude='*.pyc' \
    /mnt/weka/home/richard.zhuang/OpenThoughts-Agent/ \
    <new-cluster>:/path/to/scratch/OpenThoughts-Agent/
```

### On the destination cluster

```bash
SCRATCH="/path/to/your/scratch"  # e.g., /mnt/weka/home/<user>
cd "$SCRATCH"

# Clone from bundle
git clone /path/to/openthoughts-agent.bundle OpenThoughts-Agent
cd OpenThoughts-Agent

# Checkout working branch
git checkout penfever/working

# Apply uncommitted files on top
tar xzf /path/to/openthoughts-uncommitted.tar.gz

# Set the real remote (so future git pull works)
git remote set-url origin https://github.com/<org>/OpenThoughts-Agent.git
```

---

## 2. Install Miniconda (if not present)

```bash
SCRATCH="/path/to/your/scratch"

# Download and install
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$SCRATCH/miniconda3"
eval "$($SCRATCH/miniconda3/bin/conda shell.bash hook)"
conda init bash
```

---

## 3. Create the `otagent` Conda Environment

```bash
SCRATCH="/path/to/your/scratch"
DCFT="$SCRATCH/OpenThoughts-Agent"

eval "$(conda shell.bash hook 2>/dev/null)" || source "$SCRATCH/miniconda3/etc/profile.d/conda.sh"

# Create Python 3.12 env
conda create -n otagent python=3.12 -y
conda activate otagent

# Install uv (fast pip replacement)
pip install uv

# Install PyTorch (CUDA 12.8 — adjust for your cluster's CUDA version)
uv pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install vLLM 0.13.0 (the version used by otagent for existing model evals)
uv pip install "vllm==0.13.0"

# Install the project in editable mode (pulls core deps from pyproject.toml)
cd "$DCFT"
uv pip install -e . --no-deps

# Install eval infrastructure packages
uv pip install \
    "pydantic>=2.0.0,<3.0.0" \
    pyyaml \
    omegaconf \
    wandb \
    bs4 \
    "numpy<=2.26.0" \
    "huggingface_hub>=0.20.0,<1.0.0" \
    "datasets>=2.0.0" \
    "supabase>=2.22.3" \
    "python-dotenv>=1.0.0" \
    "google-cloud-storage" \
    h5py \
    certifi \
    rapidfuzz \
    "uv>=0.4.17" \
    socksio \
    "litellm>=1.80.0" \
    "ray[default]>=2.50.0" \
    "hydra-core>=1.3.2" \
    aiohttp-socks \
    Jinja2 \
    "transformers==4.57.3" \
    "accelerate==1.12.0"

# Install harbor (with Daytona support)
uv pip install "harbor[daytona] @ git+https://github.com/laude-institute/harbor.git@penfever/temp-override"

# Install dynamic-semaphore
uv pip install "dynamic-semaphore @ git+https://github.com/penfever/dynamic-semaphore"
```

### Verify

```bash
python -c "
import torch; print(f'torch: {torch.__version__} CUDA: {torch.version.cuda}')
import vllm; print(f'vllm: {vllm.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
import ray; print(f'ray: {ray.__version__}')
import harbor; print(f'harbor: {harbor.__version__}')
import litellm; print('litellm: OK')
import daytona; print('daytona: OK')
from database.unified_db.utils import upload_eval_results; print('unified_db: OK')
"
```

Expected output:
```
torch: 2.9.0 CUDA: 12.8
vllm: 0.13.0
transformers: 4.57.3
ray: 2.54.0
harbor: 0.1.45
litellm: OK
daytona: OK
unified_db: OK
```

---

## 4. (Optional) Create `otagent2` for Newer Models

If you need to evaluate models like Qwen3.5 that require newer vLLM:

```bash
bash eval/MBZ/setup_eval_env.sh
```

This creates `otagent2` with vLLM 0.17.1 + transformers from source.

---

## 5. Configure Secrets

```bash
# Copy the template and fill in your keys
cp eval/MBZ/secret.env.template ~/secrets.env
# Edit ~/secrets.env with real values:
#   DAYTONA_API_KEY, DAYTONA_TARGET, HF_TOKEN,
#   SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY

# For RL org (if needed):
cp eval/MBZ/secret.env.template ~/secrets_rl_org.env
# Edit with RL org keys (DAYTONA_TARGET='RL')
```

---

## 6. Adapt Cluster-Specific Paths

Files that need path updates for the new cluster:

| File | What to change |
|------|---------------|
| `eval/MBZ/unified_eval_harbor_v4.sbatch` | `SCRATCH=` path (line 92), SLURM partition/QoS (`#SBATCH -p`, `#SBATCH --qos`) |
| `eval/MBZ/unified_eval_listener_v4.py` | Default paths if launching from new cluster |
| `eval/MBZ/reupload_hf.py` | `SCRATCH=`, `DEFAULT_EVAL_JOBS_DIR`, `DEFAULT_LOG_DIR` |
| `hpc/dotenv/mbz.env` | Cluster-specific env vars |

### Key path pattern

The sbatch uses `SCRATCH` as the root for everything:
```bash
SCRATCH="/path/to/your/scratch"   # <-- CHANGE THIS
DCFT="$SCRATCH/OpenThoughts-Agent"
```

All other paths (`VLLM_CACHE_ROOT`, `HF_HUB_CACHE`, etc.) derive from `$SCRATCH`.

---

## 7. Download Datasets

```bash
conda activate otagent

# Download eval datasets to HF cache
python eval/MBZ/snapshot_download.py DCAgent/dev_set_v2
python eval/MBZ/snapshot_download.py DCAgent2/terminal_bench_2
```

---

## 8. Pre-download Model Weights

Compute nodes typically have no internet. Pre-download on login node:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('your-org/model-name', cache_dir='$SCRATCH/.cache/huggingface/hub')
"
```

---

## 9. Test a Single Job

```bash
conda activate otagent

# Manual single-job test (no listener)
sbatch eval/MBZ/unified_eval_harbor_v4.sbatch \
    your-org/model-name \
    DCAgent/dev_set_v2
```

Check logs:
```bash
tail -f experiments/logs/terminal_<SLURM_JOB_ID>.out
tail -f experiments/logs/vllm_<SLURM_JOB_ID>.log
```

---

## 10. Start the Listener

```bash
# Basic listener (scans DB for pending evals)
python eval/MBZ/unified_eval_listener_v4.py \
    --datasets DCAgent/dev_set_v2 DCAgent2/terminal_bench_2 \
    --priority-file eval/MBZ/lists/priority_models.txt \
    --secrets-file ~/secrets.env \
    --verbose

# For newer models (Qwen3.5+), add:
#   --conda-env otagent2
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `RequestsDependencyWarning: chardet` | Harmless warning, ignore |
| vLLM fails with `qwen3_5` not recognized | Use `otagent2` env (vLLM 0.17.1) |
| HF upload: XET permission denied | Ensure `HF_XET_CACHE` is set (already in v4 sbatch) |
| HF upload: subagent traces bloating dataset | Fixed in `database/unified_db/utils.py` (export_subagents forwarding) |
| Snapshot quota exceeded | Daytona org limited to 40 snapshots; request quota increase |
| CUDA version mismatch | Change `--index-url` in torch install to match your CUDA (cu118, cu121, cu124, cu128) |
