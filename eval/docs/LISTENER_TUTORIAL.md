# Eval Listener Tutorial

How to run the unified eval listener to submit model evaluation jobs on an HPC cluster.

## Overview

The listener (`eval/MBZ/unified_eval_listener_v4.py`) polls Supabase for pending evals, submits SLURM jobs, which then:
1. Start a vLLM server to serve the model
2. Run Harbor agent trials against a benchmark dataset
3. Upload results to Supabase + HuggingFace

## Prerequisites

### 1. Conda Environment

You need the `otagent` conda env. Set it up from the existing setup script or replicate these key packages:
- Python 3.12
- vLLM 0.13+ (use `otagent2` with vLLM 0.17+ for newer model architectures like Qwen3.5)
- Harbor installed from `git+https://github.com/laude-institute/harbor.git@penfever/temp-override`
- torch, transformers, huggingface_hub, supabase

### 2. Secrets File

Create `~/secrets.env` with these keys:
```bash
export DAYTONA_API_KEY='dtn_...'       # Daytona sandbox API key
export DAYTONA_TARGET='us'             # Daytona region (us, eu, RL)
export HF_TOKEN='hf_...'              # HuggingFace token for uploads
export SUPABASE_URL='https://...'      # Supabase project URL
export SUPABASE_ANON_KEY='...'         # Supabase anon key
export SUPABASE_SERVICE_ROLE_KEY='...' # Supabase service role key
```

### 3. Directory Structure

```
OpenThoughts-Agent/
├── eval/MBZ/
│   ├── unified_eval_listener_v4.py    # The listener
│   ├── unified_eval_harbor_v4.sbatch  # SLURM job script
│   ├── lists/                         # Priority/blacklist files
│   └── dcagent_eval_config_no_override.yaml
├── experiments/
│   ├── logs/                          # SLURM stdout logs (terminal_*.out, vllm_*.log)
│   └── listener_logs/                 # Listener daemon logs
└── database/unified_db/utils.py       # Supabase client
```

## Quick Start

### Run a One-Time Eval for Specific Models

**Step 1:** Create a priority file listing models (one per line):
```bash
cat > eval/MBZ/lists/my_models.txt << 'EOF'
Qwen/Qwen3-8B
deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
EOF
```

**Step 2:** Dry run to verify (no actual submission):
```bash
conda activate otagent
python eval/MBZ/unified_eval_listener_v4.py \
  --preset v2 \
  --priority-file eval/MBZ/lists/my_models.txt \
  --secrets-file ~/secrets.env \
  --enable-thinking \
  --tp-size 2 \
  --once --dry-run --verbose
```

**Step 3:** Submit for real:
```bash
python eval/MBZ/unified_eval_listener_v4.py \
  --preset v2 \
  --priority-file eval/MBZ/lists/my_models.txt \
  --secrets-file ~/secrets.env \
  --enable-thinking \
  --tp-size 2 \
  --once --verbose
```

**Step 4:** Monitor:
```bash
# Check SLURM queue
squeue -u $USER

# Watch job logs (replace JOB_ID)
tail -f experiments/logs/terminal_<JOB_ID>.out
tail -f experiments/logs/vllm_<JOB_ID>.log
```

## Presets

Presets bundle dataset + tuned defaults. **Always use a preset** as the starting point.

| Preset | Dataset | Time Limit | Notes |
|--------|---------|------------|-------|
| `v2` | `DCAgent/dev_set_v2` | 24h | Primary dev benchmark (~90 tasks) |
| `tb2` | `DCAgent2/terminal_bench_2` | 48h | Large terminal benchmark (~180 tasks) |
| `swebench` | `DCAgent2/swebench-verified-random-100-folders` | 24h | SWE-Bench, uses XML parser |
| `aider` | `DCAgent2/aider_polyglot` | 24h | Aider polyglot benchmark |
| `bfcl` | `DCAgent2/bfcl-parity` | 24h | BFCL function calling |
| `v1` | `DCAgent/dev_set_71_tasks` | 24h | Legacy dev set |

## Key Flags

### Required for most runs
| Flag | Description |
|------|-------------|
| `--preset <name>` | Benchmark preset (v2, tb2, swebench, etc.) |
| `--priority-file <path>` | File listing models to evaluate (one per line) |
| `--secrets-file <path>` | Path to secrets.env with Daytona/Supabase/HF keys |

### Execution control
| Flag | Description |
|------|-------------|
| `--once` | Run one iteration then exit (for one-off submissions) |
| `--dry-run` | Print what would be submitted without actually submitting |
| `--verbose` | Extra logging |

### Model/GPU configuration
| Flag | Default | Description |
|------|---------|-------------|
| `--tp-size <int>` | 1 | Tensor parallel size (GPUs per job). Use 2 for 7-14B models, 4 for 32B+ |
| `--conda-env <name>` | otagent | Conda env for the SLURM job. Use `otagent2` for Qwen3.5+ |
| `--enable-thinking` | off | Enable thinking/reasoning blocks. Most presets set this automatically |
| `--gpu-memory-util <float>` | 0.9 | vLLM GPU memory utilization (0.0-1.0) |

### Job management
| Flag | Default | Description |
|------|---------|-------------|
| `--max-jobs-submitted <int>` | 50 | Max concurrent SLURM jobs from this listener |
| `--batch-size <int>` | 20 | First N jobs run immediately, rest chain one-by-one |
| `--timeout-multiplier <float>` | 1.0 | Multiply per-task timeouts (use 2.0 for slow models) |
| `--blacklist-file <path>` | none | File listing models to never evaluate |
| `--slurm-partition <name>` | main | SLURM partition to submit to |

### Daytona/snapshot control
| Flag | Default | Description |
|------|---------|-------------|
| `--auto-snapshot` | off | Auto-create Daytona snapshots per environment (quota: 40/org) |

## Common Patterns

### One-time batch of specific models on dev_set_v2
```bash
python eval/MBZ/unified_eval_listener_v4.py \
  --preset v2 \
  --priority-file eval/MBZ/lists/my_models.txt \
  --blacklist-file eval/MBZ/lists/pruned_models_names.txt \
  --enable-thinking \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --secrets-file ~/secrets.env \
  --once --verbose
```

### Same models on terminal_bench_2
```bash
python eval/MBZ/unified_eval_listener_v4.py \
  --preset tb2 \
  --priority-file eval/MBZ/lists/my_models.txt \
  --blacklist-file eval/MBZ/lists/pruned_models_names.txt \
  --enable-thinking \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --secrets-file ~/secrets.env \
  --once --verbose
```

### Long-running daemon (polls every 2h)
```bash
nohup python eval/MBZ/unified_eval_listener_v4.py \
  --preset v2 \
  --priority-file eval/MBZ/lists/my_models.txt \
  --blacklist-file eval/MBZ/lists/pruned_models_names.txt \
  --enable-thinking \
  --timeout-multiplier 2.0 \
  --tp-size 2 \
  --max-jobs-submitted 50 \
  --secrets-file ~/secrets.env \
  --verbose &
```

### Using otagent2 for newer models (Qwen3.5+)
```bash
python eval/MBZ/unified_eval_listener_v4.py \
  --preset v2 \
  --priority-file eval/MBZ/lists/qwen35_models.txt \
  --conda-env otagent2 \
  --enable-thinking \
  --tp-size 2 \
  --secrets-file ~/secrets.env \
  --once --verbose
```

## How It Works (End-to-End Flow)

```
1. Listener reads priority file → fetches model list
2. Checks Supabase `sandbox_jobs` table for existing jobs (dedup)
3. For each new (model, dataset) pair:
   a. Creates "Pending" DB entry
   b. Runs: sbatch --partition <part> --gres gpu:<tp_size> unified_eval_harbor_v4.sbatch <model> <dataset> <dataset_id> <run_tag>
   c. Updates DB entry with SLURM job ID
4. SLURM job (sbatch script):
   a. Sources secrets.env → activates conda env
   b. Pre-flight: validates model architecture is supported by vLLM
   c. Downloads model weights if not cached
   d. Starts vLLM server (with retries)
   e. Runs: harbor jobs start -p <dataset_path> --model <model> --env daytona ...
   f. Harbor creates Daytona sandboxes, runs agent trials
   g. Checks error threshold → uploads results to Supabase + HuggingFace
   h. Updates DB entry to "Finished"
```

## Deduplication Logic

The listener skips models that already have jobs in the DB:
- **Finished** → skip (already done)
- **Started** < 24h ago → skip (in progress)
- **Started** > 24h ago → restart (stale)
- **Pending** > 48h ago → restart + scancel old job
- **No entry** → submit new job

## Troubleshooting

### Job fails immediately
Check `experiments/logs/terminal_<JOB_ID>.out` for:
- `Unsupported model architecture` → need newer vLLM (use `--conda-env otagent2`)
- `CUDA out of memory` → increase `--tp-size`
- `Invalid qos specification` → check partition supports the QOS

### vLLM won't start
Check `experiments/logs/vllm_<JOB_ID>.log`:
- Model download failures → pre-download with `huggingface_hub.snapshot_download()`
- Architecture not in ModelRegistry → upgrade vLLM

### Listener skips a model
- Already has a job in DB → check with:
  ```python
  from database.unified_db.utils import get_supabase_client, load_supabase_keys
  load_supabase_keys()
  c = get_supabase_client()
  r = c.table('sandbox_jobs').select('*').like('job_name', '%ModelName%').execute()
  for j in r.data: print(j['job_name'], j['status'])
  ```
- Model is in blacklist file
- Model not in priority file (filter_only mode)

### SLURM partition issues
- `main` partition: no `--qos` needed
- `lowprio` partition: needs `--qos=lowprio` (but jobs get preempted)
- The listener auto-handles this: it only adds `--qos` for non-main partitions

## Cluster-Specific Notes

When setting up on a new cluster, you need to:
1. Set up conda env (otagent) with vLLM + harbor + dependencies
2. Create `~/secrets.env` with valid credentials
3. Adjust `--slurm-partition` to match your cluster's partition names
4. Verify the sbatch script paths (`DCFT`, `HF_HOME`, cache dirs) match your filesystem
5. Pre-download model weights before submitting (avoids download during eval)
6. Test with `--dry-run --once` first, then `--once` for a single model before running the daemon
