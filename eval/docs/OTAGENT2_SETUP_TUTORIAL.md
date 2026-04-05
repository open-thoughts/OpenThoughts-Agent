# otagent2 Environment Setup Tutorial

How to create the `otagent2` conda environment for evaluating newer model architectures (Qwen3.5, etc.) that require vLLM >= 0.17.1.

## Why a Separate Environment?

The existing `otagent` env (vLLM 0.13.0) cannot serve newer models like Qwen3.5-9B:

1. **vLLM 0.13.0** doesn't have `Qwen3_5ForConditionalGeneration` in its model registry
2. **Upgrading transformers alone** breaks vLLM 0.13.0 (internal `ALLOWED_LAYER_TYPES` was renamed)
3. **vLLM 0.16.0** still doesn't have Qwen3.5 — need **0.17.1** minimum
4. **transformers on PyPI** (4.57.x) doesn't recognize `qwen3_5` config — need to install from GitHub source

So we create a parallel `otagent2` env that keeps `otagent` intact for all existing models.

## Target Versions

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12 | |
| vLLM | 0.17.1 | First version with Qwen3.5 support |
| torch | 2.10.0+cu128 | Pulled by vLLM |
| transformers | 5.x (dev) | From GitHub main, not PyPI |
| ray | 2.54.0 | |
| harbor | 0.1.45 | From laude-institute fork |
| CUDA | 12.8 | Pulled by vLLM's torch |

## Setup Instructions

### Option A: Use the Setup Script (Recommended)

The script at `eval/MBZ/setup_eval_env.sh` automates everything:

```bash
cd /path/to/OpenThoughts-Agent
bash eval/MBZ/setup_eval_env.sh
```

The script accepts an optional env name argument (default: `otagent2`):
```bash
bash eval/MBZ/setup_eval_env.sh my_custom_env_name
```

**Before running**, edit these paths at the top of the script to match your cluster:
```bash
SCRATCH="/mnt/weka/home/richard.zhuang"   # Your home/scratch directory
DCFT="$SCRATCH/OpenThoughts-Agent"         # Path to this repo
CONDA_BASE="$SCRATCH/miniconda3"           # Path to miniconda installation
```

### Option B: Manual Step-by-Step

If you need to adapt for a different cluster or debug issues:

#### Step 1: Create conda env
```bash
conda create -n otagent2 python=3.12 -y
conda activate otagent2
pip install uv  # Fast dependency resolver
```

#### Step 2: Install vLLM 0.17.1
This pulls compatible torch, triton, and CUDA dependencies automatically:
```bash
uv pip install "vllm==0.17.1"
```

**CUDA note:** vLLM 0.17.1 installs torch with CUDA 12.8. If your cluster has a different CUDA version, you may need to adjust. Check `nvidia-smi` for your driver version — CUDA 12.8 requires driver >= 570.x.

#### Step 3: Install transformers from source
PyPI's transformers (4.57.x) doesn't have Qwen3.5 config. Must install from GitHub:
```bash
uv pip install "git+https://github.com/huggingface/transformers.git"
```

This gives you transformers 5.x with `Qwen3_5ForConditionalGeneration` support.

#### Step 4: Install project dependencies
```bash
cd /path/to/OpenThoughts-Agent

# Install the project itself (no-deps to avoid conflicting torch/vllm pins)
uv pip install -e . --no-deps

# Install eval infrastructure packages
uv pip install \
    "pydantic>=2.0.0,<3.0.0" \
    pyyaml \
    omegaconf \
    wandb \
    bs4 \
    "numpy<=2.26.0" \
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
    Jinja2
```

**Important:** Use `--no-deps` for the project install. The project's `pyproject.toml` pins older vllm/torch versions that would downgrade what we just installed.

#### Step 5: Install Harbor
```bash
uv pip install "harbor[daytona] @ git+https://github.com/laude-institute/harbor.git@penfever/temp-override"
```

If you have a local harbor checkout (e.g. with custom patches), use editable install instead:
```bash
uv pip install -e /path/to/harbor
```

#### Step 6: Install dynamic-semaphore
```bash
uv pip install "dynamic-semaphore @ git+https://github.com/penfever/dynamic-semaphore"
```

## Verification

Run this after setup to confirm everything works:

```bash
conda activate otagent2
python -c "
import sys; print(f'Python: {sys.version}')
import torch; print(f'torch: {torch.__version__} (CUDA: {torch.version.cuda})')
import vllm; print(f'vllm: {vllm.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')

# Critical: Qwen3.5 config must be recognized
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-9B', trust_remote_code=True)
print(f'Qwen3.5-9B config: OK (arch={cfg.architectures})')

import ray; print(f'ray: {ray.__version__}')
import harbor; print(f'harbor: {harbor.__version__}')
import litellm; print('litellm: OK')
import supabase; print('supabase: OK')
import daytona; print('daytona: OK')

from database.unified_db.utils import upload_eval_results; print('database.unified_db: OK')
from harbor.utils.traces_utils import export_traces; print('harbor traces: OK')
"
```

Expected output:
```
Python: 3.12.x
torch: 2.10.0+cu128 (CUDA: 12.8)
vllm: 0.17.1
transformers: 5.3.0.dev0
Qwen3.5-9B config: OK (arch=['Qwen3_5ForConditionalGeneration'])
ray: 2.54.0
harbor: 0.1.45
litellm: OK
supabase: OK
daytona: OK
database.unified_db: OK
harbor traces: OK
```

## GPU Smoke Test

After env setup, verify vLLM can actually serve the model on your GPUs:

```bash
# Interactive SLURM job (adjust partition/qos for your cluster)
srun -p main --gres=gpu:1 --cpus-per-task=16 --time=00:30:00 --pty bash -c '
  eval "$(conda shell.bash hook 2>/dev/null)"
  conda activate otagent2
  python -c "import torch; print(f\"GPU: {torch.cuda.get_device_name(0)}\"); print(f\"CUDA available: {torch.cuda.is_available()}\")"
  vllm serve Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 --enforce-eager --gpu-memory-utilization 0.9
'
```

On H200 (141GB), single-GPU results:
- Model memory: 17.66 GiB
- KV cache available: 104.16 GiB
- Architecture: `Qwen3_5ForConditionalGeneration`
- Max concurrency at 262K context: 12.93x

## Using otagent2 with the Eval Listener

Pass `--conda-env otagent2` to the listener:

```bash
python eval/MBZ/unified_eval_listener_v4.py \
  --preset v2 \
  --priority-file eval/MBZ/lists/my_models.txt \
  --conda-env otagent2 \
  --enable-thinking \
  --tp-size 2 \
  --secrets-file ~/secrets.env \
  --once --verbose
```

This sets `EVAL_CONDA_ENV=otagent2` → the sbatch script does `conda activate "$CONDA_ENV"` instead of the default `otagent`.

## When to Use otagent2 vs otagent

| Model | Environment |
|-------|------------|
| Qwen3-8B, Qwen3-32B, etc. | `otagent` (vLLM 0.13.0 supports these) |
| Qwen3.5-9B, Qwen3.5-* | `otagent2` (needs vLLM 0.17.1) |
| DeepSeek-R1-*, Nemotron-*, most 7-14B models | `otagent` |
| Any model with `Qwen3_5ForConditionalGeneration` arch | `otagent2` |
| Future new architectures not in vLLM 0.13.0 | `otagent2` (or newer) |

**Rule of thumb:** Use `otagent` by default. Only switch to `otagent2` if vLLM fails with "Unsupported model architecture" or if the model's `config.json` has an architecture not in vLLM 0.13.0's registry.

## Troubleshooting

### "No module named 'transformers.models.qwen3_5'"
Transformers was installed from PyPI instead of source. Fix:
```bash
uv pip install --force-reinstall "git+https://github.com/huggingface/transformers.git"
```

### vLLM crashes with ALLOWED_LAYER_TYPES error
You have a transformers version that's too new for your vLLM. Ensure vLLM is 0.17.1:
```bash
python -c "import vllm; print(vllm.__version__)"
# Should print 0.17.1
```

### CUDA version mismatch
If you get CUDA errors, check compatibility:
```bash
nvidia-smi  # Shows driver version and max CUDA
python -c "import torch; print(torch.version.cuda)"  # Should be 12.8
```
CUDA 12.8 requires NVIDIA driver >= 570.x. If your driver is older, you may need a different vLLM version compiled for your CUDA.

### torch.cuda.is_available() returns False
Ensure you're on a GPU node (not login node). Run the GPU smoke test via `srun`.

### Harbor import errors
If harbor fails to import, ensure it was installed after torch/vllm (not before):
```bash
uv pip install -e /path/to/harbor  # or from git
```
