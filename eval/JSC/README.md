# JSC Quick Start Guide

Set up environment and run SWE-bench evaluation on JSC supercomputers.

## Prerequisites

- JSC account with SSH key configured in JuDoor
- Joined projects: `ccstdl`, `synthlaion` (or `laionize`)

## Setup (One-time, ~30 minutes)

### 1. Login
```bash
ssh -i ~/.ssh/id_ed25519 <username>@jureca.fz-juelich.de -4
```

### 2. Configure Home Directory
```bash
export USER_MACHINE="${USER}_jureca"
mkdir -p /p/project1/ccstdl/${USER_MACHINE}
ln -s /p/project1/ccstdl/${USER_MACHINE} ~/${USER_MACHINE}

# Move cache/local to avoid quota
if [ -d ~/.cache ] && [ ! -L ~/.cache ]; then mv ~/.cache /p/project1/ccstdl/${USER_MACHINE}/; fi
mkdir -p /p/project1/ccstdl/${USER_MACHINE}/.cache
ln -sf /p/project1/ccstdl/${USER_MACHINE}/.cache ~/.cache

if [ -d ~/.local ] && [ ! -L ~/.local ]; then mv ~/.local /p/project1/ccstdl/${USER_MACHINE}/; fi
mkdir -p /p/project1/ccstdl/${USER_MACHINE}/.local
ln -sf /p/project1/ccstdl/${USER_MACHINE}/.local ~/.local

echo "ulimit -c 0" >> ~/.bashrc
source ~/.bashrc
```

### 3. Install Miniforge
```bash
cd ~/${USER_MACHINE} && mkdir -p tmp && cd tmp
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
export MAMBA_INSTALL="/p/project1/ccstdl/${USER}/mamba"
bash Miniforge3-*.sh -b -p ${MAMBA_INSTALL}
```

### 4. Create Python 3.11 Environment
```bash
eval "$(${MAMBA_INSTALL}/bin/conda shell.bash hook)"
export EVAL_ENV="/p/project1/ccstdl/${USER}/eval_py311"
${MAMBA_INSTALL}/bin/mamba create --prefix ${EVAL_ENV} python=3.11 -c conda-forge --override-channels -y
source ${MAMBA_INSTALL}/bin/activate ${EVAL_ENV}
```

### 5. Install Dependencies
```bash
pip install -U pip
pip install uv vllm

cd /p/project1/ccstdl/${USER}
git clone https://github.com/laude-institute/harbor.git
cd harbor
uv pip install -e . --group dev

# Verify
harbor --help
```

### 6. Add to .bashrc
```bash
cat >> ~/.bashrc <<'EOF'

# Harbor Eval
export MAMBA_INSTALL="/p/project1/ccstdl/${USER}/mamba"
export EVAL_ENV="/p/project1/ccstdl/${USER}/eval_py311"
alias activate_eval='source ${MAMBA_INSTALL}/bin/activate ${EVAL_ENV}'
EOF
```

### 7. Clone Repository
```bash
cd /p/project1/ccstdl/${USER}
git clone https://github.com/mlfoundations/OpenThoughts-Agent.git
cd OpenThoughts-Agent/eval/JSC
```

### 8. Configure Secrets
Create a `secret.env` file for API keys:

```bash
cat > secret.env <<EOF
export DAYTONA_API_KEY="your-daytona-api-key"
export HF_TOKEN="your-huggingface-token"  # Optional: for upload feature
EOF
chmod 600 secret.env
```

> **How to get API keys**:
> - `DAYTONA_API_KEY`: Contact your team lead or check internal documentation
> - `HF_TOKEN`: Create at https://huggingface.co/settings/tokens (optional, for upload only)

> **Note**: `HF_TOKEN` is required if you want to enable automatic upload of results to HuggingFace.

---

## Run Evaluation

### Single Task Test
```bash
cd /p/project1/ccstdl/${USER}/OpenThoughts-Agent/eval/JSC
mkdir -p experiments/logs

sbatch jsc_eval_harbor.sbatch \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "swebench-verified@1.0" \
  "django__django-13406" \
  4 \
  3
```

### Full Benchmark
```bash
sbatch jsc_eval_harbor.sbatch \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "swebench-verified@1.0" \
  "" \
  8 \
  1
```

### Full Benchmark with Auto-Upload
```bash
sbatch jsc_eval_harbor.sbatch \
  "Qwen/Qwen2.5-Coder-7B-Instruct" \
  "swebench-verified@1.0" \
  "" \
  8 \
  1 \
  "true"
```

> **Upload Feature**: When `ENABLE_UPLOAD=true`, results are automatically uploaded to HuggingFace (`mlfoundations-dev/` org) after evaluation completes. Requires `HF_TOKEN` in `secret.env`. Upload is skipped if there are more than 3 DaytonaErrors.

### Monitor
```bash
# Check status
squeue -u $USER

# View logs
tail -f experiments/logs/eval_harbor_<JOB_ID>.out

# View vLLM logs
tail -f experiments/logs/vllm_<JOB_ID>.log
```

---

## Common Issues

**Queue taking too long**: Use `dc-gpu` instead of `dc-gpu-devel` for longer jobs (24h limit)

**vLLM fails**: Check `experiments/logs/vllm_<JOB_ID>.log`

**Out of memory**: Reduce `--gpu-memory-utilization` in sbatch script

**Budget exhausted**: Projects have time limits (6h when exhausted, 24h normally)

---

## File Structure
```
/p/project1/ccstdl/<username>/
├── mamba/                     # Miniforge installation
├── eval_py311/                # Python 3.11 environment
├── harbor/                    # Harbor source
└── OpenThoughts-Agent/        # This repository
    ├── database/              # Upload module (unified_db)
    └── eval/JSC/
        ├── README.md              # This file
        ├── jsc_eval_harbor.sbatch # Main eval script
        ├── jsc_eval_config.yaml   # Harbor config
        ├── secret.env             # API keys (create this)
        ├── experiments/logs/      # Job logs
        └── jobs/                  # Evaluation results
```

---

## Quick Commands

```bash
# Activate environment (after first login)
activate_eval

# Submit job (without upload)
sbatch jsc_eval_harbor.sbatch <MODEL> <DATASET> [TASK_FILTER] [N_CONCURRENT] [N_ATTEMPTS]

# Submit job (with upload)
sbatch jsc_eval_harbor.sbatch <MODEL> <DATASET> [TASK_FILTER] [N_CONCURRENT] [N_ATTEMPTS] "true"

# Check queue
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# View results
ls jobs/
cat jobs/<RUN_TAG>/result.json
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| MODEL | `Qwen/Qwen2.5-Coder-7B-Instruct` | Model name |
| DATASET | `swebench-verified@1.0` | Dataset name |
| TASK_FILTER | (empty) | Optional: specific task ID |
| N_CONCURRENT | `8` | Number of concurrent tasks |
| N_ATTEMPTS | `1` | Number of attempts per task |
| ENABLE_UPLOAD | `false` | Set to `"true"` to upload results |

