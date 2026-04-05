# Jupiter v6 Listener Migration

## What's already done (in this repo, will arrive via `git pull`)

1. **`eval/clusters/jupiter.yaml`** — updated sbatch paths to shared `eval/MBZ/unified_eval_harbor.sbatch`
2. **`eval/jupiter/dcagent_eval_config.yaml`** — updated `jobs_dir` to `zhuang1_eval_jobs`
3. **`eval/jupiter/dcagent_eval_config_no_override.yaml`** — created (swebench/tb2 variant)
4. **`eval/MBZ/unified_eval_harbor.sbatch`** — cluster-agnostic v6 sbatch (shared across clusters)
5. **`eval/MBZ/unified_eval_harbor_dp.sbatch`** — cluster-agnostic DP sbatch
6. **`eval/MBZ/unified_eval_listener_v6.py`** — shared v6 listener
7. **`eval/baseline_model_configs.yaml`** — shared model configs

## Steps to run on Jupiter

### 1. Pull latest code
```bash
source ~/.bashrc; conda activate otagent
cd /e/scratch/jureap59/zhuang1/OpenThoughts-Agent
GIT_TERMINAL_PROMPT=0 git pull
```

### 2. Pin harbor to known-good commit
```bash
cd /e/scratch/jureap59/feuer1/harbor
git fetch && git checkout 6fdb92e7f5707c2b01214933f1622771784e6f67
# Reinstall in your conda env
pip install -e .
```

### 3. Install hf_transfer
```bash
pip install hf_transfer
```

### 4. Create jobs dir (if it doesn't exist)
```bash
mkdir -p /e/data1/datasets/playground/mmlaion/shared/zhuang1_eval_jobs
mkdir -p eval/jupiter/logs
```

### 5. Pre-download datasets
```bash
source ~/secrets.env
python eval/jupiter/snapshot_download.py DCAgent/dev_set_v2
python eval/jupiter/snapshot_download.py DCAgent2/terminal_bench_2
python eval/jupiter/snapshot_download.py DCAgent2/swebench-verified-random-100-folders
```

### 6. Verify secrets.env has all required keys
```bash
source ~/secrets.env
echo "DAYTONA_API_KEY: ${DAYTONA_API_KEY:0:12}..."
echo "SUPABASE_URL: ${SUPABASE_URL:0:20}..."
echo "HF_TOKEN: ${HF_TOKEN:0:8}..."
```

### 7. Dry-run
```bash
source ~/secrets.env && python eval/MBZ/unified_eval_listener_v6.py \
  --cluster-config eval/clusters/jupiter.yaml \
  --preset v2 \
  --priority-file eval/MBZ/lists/a1_retrained.txt \
  --baseline-model-config eval/baseline_model_configs.yaml \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --enable-thinking \
  --slurm-time 12:00:00 \
  --max-jobs-submitted 32 \
  --dry-run --once --verbose
```

### 8. Real run (example)
```bash
source ~/secrets.env && python eval/MBZ/unified_eval_listener_v6.py \
  --cluster-config eval/clusters/jupiter.yaml \
  --preset swebench \
  --priority-file eval/MBZ/lists/no_eval_models_latest.txt \
  --baseline-model-config eval/baseline_model_configs.yaml \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --enable-thinking \
  --slurm-time 12:00:00 \
  --max-jobs-submitted 32 \
  --pack-jobs \
  --stagger-delay 1 --chain-batch-size 10 \
  --no-disk-resume \
  --once
```

## Key differences from M2

| Setting | M2 | Jupiter |
|---------|-----|---------|
| Partition | `main` | `booster` |
| Account | (none) | `reformo` |
| Time limit | 24:00:00 | 12:00:00 |
| GPUs/node | 8 | 4 |
| Arch | x86_64 | aarch64 (GH200) |
| Internet on compute | yes | **no** (proxy required) |
| Conda env | otagent/otagent2 | otagent/otagent2 (different paths) |
| Harbor | local install | feuer1's shared install |
| HF cache | `~/.cache/huggingface/hub` | `/e/data1/datasets/playground/ot/hf_hub` |
| Jobs dir | `$PWD/jobs` | `/e/data1/.../zhuang1_eval_jobs` |
| Pre-download | optional (has internet) | **required** (no internet on compute) |

## Proxy note

Jupiter compute nodes have no internet. The v6 sbatch auto-detects proxy settings from `jupiter.yaml`:
- Uses proxychains for HF downloads on compute
- SSH tunnel via `jpbl-s01-02` login node
- `--pre-download` flag on listener pre-downloads models on login node before submission
