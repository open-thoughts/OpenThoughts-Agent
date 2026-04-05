# Overlong Eval Jobs: Diagnosis & Upload Runbook

## Background

Some eval jobs hit the 24-hour SLURM time limit without completing all trials. These are marked as `is_overlong=true` in the database so the listener doesn't perpetually resubmit them.

### Root Cause

Slow models are **not more verbose** — they produce ~0.5x the output tokens of fast models. The bottleneck is **tool/environment execution**: 94% of agent wall time is spent waiting on sandbox commands (compilations, builds, installs) that don't lead to solutions. Trials hit the agent timeout ceiling and waste the maximum allowed duration.

Key metrics across 19 timed-out jobs:
- LLM inference time per trial: ~7 min (same as fast models)
- Agent execution per trial: **125 min** (vs 20 min for fast models)
- LLM % of agent execution: **6%** (vs 29% for fast models)
- 44% of trials end in `AgentTimeoutError`, consuming 88-96% of total wall time

### Detection

Overlong jobs are detected by: **no `finished_at` in result.json AND elapsed > 20 hours**. This catches the final timed-out attempt that was never retried. Earlier timed-out attempts show as `TIME LIMIT` in SLURM logs but were retried by the listener.

## Files

All files are relative to the repo root (`OpenThoughts-Agent/`).

### Scripts

| File | Purpose |
|------|---------|
| `scripts/database/upload_overlong_jobs.py` | Detect overlong jobs, check DB, upload with `is_overlong=true` |
| `scripts/database/manual_db_eval_push.py` | Manual single-job upload (supports `--overlong` flag) |
| `eval/MBZ/diagnose_slow_jobs.py` | Diagnostic script: timing breakdown, error distribution, per-model analysis |

### Modified (to support `is_overlong`)

| File | Change |
|------|--------|
| `database/unified_db/utils.py` | Added `is_overlong` param to `upload_job_and_trial_records()`, `upload_eval_results()`, `register_sandbox_job()` |
| `hpc/launch_utils.py` | Added `is_overlong` param to `sync_eval_to_database()`. Added trial `source` field parsing in `derive_benchmark_from_job_dir()` |
| `scripts/database/manual_db_eval_push.py` | Added `--overlong` CLI flag |

### Reports

| File | Purpose |
|------|---------|
| `eval/MBZ/SLOW_MODEL_DIAGNOSIS.md` | Generated diagnostic report (run `diagnose_slow_jobs.py` to regenerate) |

## Quick Start (copy-paste for another cluster)

### Prerequisites

```bash
# Source environment
source ~/secrets.env  # needs SUPABASE_URL, SUPABASE_ANON_KEY/SERVICE_ROLE_KEY, HF_TOKEN
conda activate otagent
cd /path/to/OpenThoughts-Agent
```

### 1. Dry run: see what would be uploaded

```bash
python scripts/database/upload_overlong_jobs.py --jobs-dir /path/to/jobs
```

### 2. Upload all overlong jobs (with HF traces, 8 parallel workers)

```bash
python scripts/database/upload_overlong_jobs.py \
    --upload --force --skip-db-check --parallel 8 \
    --jobs-dir /path/to/jobs
```

### 3. Upload with DB duplicate check (skips if model/benchmark already Finished)

```bash
python scripts/database/upload_overlong_jobs.py \
    --upload --force --parallel 8 \
    --jobs-dir /path/to/jobs
```

### 4. Upload a single job manually

```bash
python scripts/database/manual_db_eval_push.py \
    --job-dir /path/to/jobs/terminal_bench_2_model_name_20260330_014503 \
    --benchmark-name terminal_bench_2 \
    --overlong --force --forced-update --verbose
```

### 5. Skip HF upload (DB only)

```bash
python scripts/database/upload_overlong_jobs.py \
    --upload --force --skip-db-check --skip-hf --parallel 8 \
    --jobs-dir /path/to/jobs
```

### 6. Filter to specific benchmark

```bash
python scripts/database/upload_overlong_jobs.py \
    --upload --force --parallel 8 \
    --benchmark terminal_bench_2 \
    --jobs-dir /path/to/jobs
```

### 7. Generate diagnostic report

```bash
python eval/MBZ/diagnose_slow_jobs.py --output eval/MBZ/SLOW_MODEL_DIAGNOSIS.md
python eval/MBZ/diagnose_slow_jobs.py --benchmark tb2 --top 20  # stdout, filtered
```

## DB Schema

The `sandbox_jobs` table has:
- `is_overlong` (boolean, default false) — set to true for timed-out jobs
- `job_status` remains `"Finished"` so the listener treats them as done

```sql
-- If is_overlong column doesn't exist yet on the target cluster:
ALTER TABLE sandbox_jobs ADD COLUMN IF NOT EXISTS is_overlong BOOLEAN DEFAULT false;
```

## Files to Copy to Another Cluster

```bash
# From the repo root, these are the files needed:
scripts/database/upload_overlong_jobs.py
scripts/database/manual_db_eval_push.py
eval/MBZ/diagnose_slow_jobs.py
database/unified_db/utils.py          # has is_overlong support
hpc/launch_utils.py                   # has is_overlong passthrough + benchmark derivation fix
```

Or just `git pull` if the cluster has the same repo.
