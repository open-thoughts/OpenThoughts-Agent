# GLM-4.7 Trace Generation Pipeline

End-to-end pipeline for generating GLM-4.7 training traces across ~65 HuggingFace datasets on the Jupiter HPC cluster using Harbor.

## Overview

```
analyze_datasets.py  →  prepare_datasets.py  →  register_snapshots.py  →  launch_glm47_pairs.sh
     (Step 1)               (Step 2)                 (Step 3)                  (Step 4)
```

1. **Analyze** each candidate dataset: count tasks, count unique Dockerfiles
2. **Prepare** datasets: upsample small ones to 10K tasks, upload to HuggingFace
3. **Register** Daytona snapshots for all unique Docker environments
4. **Launch** trace generation jobs on Jupiter (2 concurrent, sliding-window dependencies)

## Prerequisites

```bash
# Environment setup (run on login node)
source ~/data_gen_secrets.env   # HF_TOKEN, DAYTONA_API_KEY
export PYTHONPATH="/e/scratch/jureap59/feuer1/OpenThoughts-Agent:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/scripts/harbor:/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/feuer1/OpenThoughts-Agent/data:$PYTHONPATH"
PYTHON="/lib/ld-linux-aarch64.so.1 /e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python3.12"
```

All scripts live in `/e/scratch/jureap59/etash/OpenThoughts-Agent/data/sbatches/`.

## Step 1: Analyze Datasets

**Script:** `analyze_datasets.py`
**Run on:** Login node (needs internet for HF downloads)

Iterates over the `DATASETS` list (defined in the script), downloads each from HuggingFace, counts tasks, and analyzes unique Docker environments.

```bash
$PYTHON analyze_datasets.py
```

**Download location:** HF cache and extracted tasks go to `/e/data1/datasets/playground/mmlaion/shared/guha1/tmp_glm47/` (playground, not scratch, to avoid quota limits). The script sets `TMPDIR` and `HF_HOME` accordingly. Extracted task directories are cleaned up after each dataset is analyzed; the HF cache persists for reuse by later steps.

**Outputs:** `dataset_analysis_report.jsonl` — one JSON record per dataset:
```json
{"repo_id": "DCAgent/exp_rpt_ghactions", "num_tasks": 10000, "unique_envs": 5, "hash_counts": {"17077f32bd04": 3614, ...}, "needs_upsample": false, "skip_reason": null, "error": null}
```

**Skip criteria:**
- `unique_envs > 5` — too many different Docker environments (would need too many snapshots)
- `num_tasks == 0` — empty dataset
- Any download/analysis error

**Resume-safe:** Already-analyzed datasets are skipped on re-run.

## Step 2: Prepare Datasets

**Script:** `prepare_datasets.py`
**Run on:** Login node (needs internet + HF_TOKEN)

Reads the analysis report. For datasets with <10K tasks, upsamples to 10K and uploads as `DCAgent/{name}_10k`. Datasets already at >=10K are used as-is.

```bash
$PYTHON prepare_datasets.py
```

**Outputs:**
- `launch_list.txt` — one HF repo ID per line (the final list to launch)
- `prepare_progress.jsonl` — tracks which datasets have been processed
- Upsampled datasets uploaded to HuggingFace (e.g., `DCAgent/exp_rpt_stack-rust_10k`)

**Resume-safe:** Checks `prepare_progress.jsonl` to skip already-processed datasets.

## Step 3: Register Daytona Snapshots

**Script:** `register_snapshots.py`
**Run on:** Login node (needs internet + DAYTONA_API_KEY)

Pre-creates Daytona snapshots for all unique Docker environments across all datasets. This lets `auto_snapshot=true` in harbor find pre-built snapshots instead of building from Dockerfile each time.

```bash
# Discover unique environments and cache locally
$PYTHON register_snapshots.py discover

# Check current snapshot status on both Daytona orgs
$PYTHON register_snapshots.py status

# Register missing snapshots
$PYTHON register_snapshots.py register

# List all snapshots on both orgs
$PYTHON register_snapshots.py list

# Clean up old/unneeded snapshots (frees quota)
$PYTHON register_snapshots.py cleanup
```

**How `register` works:**

1. Reads `dataset_analysis_report.jsonl` to get all needed environment hashes
2. Runs discovery: downloads datasets one at a time, extracts the Dockerfile from each unique environment, and caches it at `/e/scratch/jureap59/etash/glm47_tmp/dockerfile_cache/{hash}/` (skips already-cached hashes)
3. For **each org** (both org1 and org2), for **each unique hash**:
   - Computes snapshot name: `harbor__{hash}__snapshot`
   - Fast path: if already ACTIVE, skip. If PENDING, wait up to 10min. If ERROR, delete and recreate.
   - Slow path: calls Daytona API `snapshot.create()` with `Image.from_dockerfile()` pointing at the cached Dockerfile, then waits for ACTIVE state
   - Records result in `snapshot_registry.jsonl`
4. Resume-safe: skips hashes already marked ACTIVE in the registry from previous runs

Snapshots are registered on **both** orgs because the sbatch script load-balances across them (25%/75% weight), so every snapshot must exist on both.

**Key details:**
- Snapshot naming: `harbor__{environment_dir_hash_truncated(env_dir, 12)}__snapshot`
- This matches Harbor's internal `DaytonaEnvironment._get_auto_snapshot_name()`
- Two Daytona orgs with separate quotas (org1: 100, org2: 45)
- Cached environment dirs at `/e/scratch/jureap59/etash/glm47_tmp/dockerfile_cache/`

**Outputs:** `snapshot_registry.jsonl` — maps hash to snapshot name and status

**Snapshot quota management:** Each Daytona org has a snapshot limit (org1: 100, org2: 45). If you hit the limit, use `cleanup` to delete snapshots not needed by the current dataset list:

```bash
# See what's on each org (names + states)
$PYTHON register_snapshots.py list

# Check which needed snapshots are ACTIVE/MISSING/ERROR
$PYTHON register_snapshots.py status

# Delete all snapshots NOT in dataset_analysis_report.jsonl
# (compares existing snapshots against needed hashes, deletes the rest)
$PYTHON register_snapshots.py cleanup
```

`cleanup` loads the needed hashes from `dataset_analysis_report.jsonl`, lists all snapshots on both orgs, and deletes any snapshot whose name is not in the needed set. It handles Daytona rate limits with exponential backoff. After cleanup, re-run `register` to create any missing snapshots in the freed slots.

To delete a specific snapshot manually (e.g., from a Python shell):

```python
import asyncio
from daytona import AsyncDaytona, DaytonaConfig

async def delete_one(api_key, snapshot_name):
    client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))
    snap = await client.snapshot.get(snapshot_name)
    await client.snapshot.delete(snap)
    print(f"Deleted {snapshot_name}")
    await client.close()

asyncio.run(delete_one("dtn_ecfb75...", "harbor__abcdef123456__snapshot"))
```

## Step 4: Launch Trace Generation

**Script:** `launch_glm47_pairs.sh`
**Sbatch script:** `run_harbor_glm47_jupiter.sbatch`

Submits SLURM jobs with sliding-window concurrency. Each job runs 8-way data-parallel Harbor (8 shards across 32 nodes with 4 nodes per shard running vLLM).

```bash
# Launch all datasets from launch_list.txt, 2 concurrent
bash launch_glm47_pairs.sh --concurrency 2

# Dry run (preview without submitting)
bash launch_glm47_pairs.sh --dry-run --concurrency 2

# Start from a specific index (skip already-completed datasets)
bash launch_glm47_pairs.sh --concurrency 2 --start-idx 12

# Chain after existing running jobs

```

**Sliding-window dependencies:** Job N depends on job N-2 (for concurrency=2). As soon as one slot frees up, the next job starts immediately — no waiting for the entire batch.

**What each sbatch job does:**
1. Allocates 32 nodes (4 per shard x 8 shards)
2. Downloads dataset from HF, splits into 8 shards
3. Starts 8 vLLM servers (one per shard, 4-node tensor parallel)
4. Starts 8 Harbor processes via proxychains (for Daytona internet access)
5. Harbor runs each task: creates Daytona sandbox from snapshot, runs agent, verifies, collects traces
6. Exports and uploads traces to `DCAgent/{name}_glm_4.7_traces_jupiter`
7. Cleans up Ray, vLLM, temp files

**Outputs:**
- `launch_log.txt` — maps job IDs to dataset names
- Logs at `logs/harbor_glm47_jupiter_{JOBID}.out` (main job log)
- Logs at `logs/harbor_{dataset}_shard{N}_{JOBID}.log` (per-shard harbor log)
- Traces uploaded to HuggingFace
- Job-level results at `/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/{run_tag}/result.json`

## Monitoring

```bash
# Check running/pending GLM jobs
squeue -u guha1 --name=harbor_glm47_jupiter

# Check completed jobs
sacct -u guha1 --starttime=2026-03-08 --format=JobID,JobName%30,State%12,Elapsed -n | grep harbor_glm47 | grep -v "\."

# Check a specific job's progress
tail -20 logs/harbor_glm47_jupiter_JOBID.out

# Check per-shard progress
tail -20 logs/harbor_DATASET_shardN_JOBID.log

# Check error distribution from result.json
python3 -c "
import json
r = json.load(open('path/to/result.json'))
for eval_name, data in r['stats']['evals'].items():
    print(eval_name, data.get('exception_stats', {}).keys())
"
```

## Error Analysis

Each shard writes a `result.json` to its output directory at `/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/{run_tag}_shard{N}/result.json`.

### result.json Structure

```json
{
  "id": "...",
  "started_at": "...",
  "finished_at": "...",
  "n_total_trials": 1250,
  "stats": {
    "n_trials": 1250,
    "n_errors": 650,
    "evals": {
      "terminus-2__glm__shard_0": {
        "n_trials": 0,
        "n_errors": 650,
        "metrics": [{"mean": 0.0}],
        "reward_stats": {},
        "exception_stats": {
          "AgentTimeoutError": ["trial_name_1", "trial_name_2", ...],
          "DaytonaError": ["trial_name_3", ...],
          "EnvironmentStartTimeoutError": ["trial_name_4", ...]
        }
      }
    }
  }
}
```

Key fields:
- `stats.n_trials` — total trials attempted (top-level, use this one)
- `stats.n_errors` — total errors (top-level)
- `stats.n_trials - stats.n_errors` — successful trials
- `stats.evals.{name}.exception_stats` — error breakdown by type, each value is a list of trial names

### Common Error Types

| Error | Retryable? | Meaning |
|-------|-----------|---------|
| `AgentTimeoutError` | No | Agent exceeded its time limit — expected for hard tasks |
| `AgentEnvironmentTimeoutError` | No | Agent's environment interaction timed out |
| `DaytonaError` | Yes | Transient sandbox creation/communication failure |
| `EnvironmentStartTimeoutError` | Yes | Daytona sandbox took too long to start (overloaded) |
| `RuntimeError` | Maybe | Task-specific error (bad test script, etc.) |
| `DaytonaNotFoundError` | Yes | Snapshot not found or sandbox disappeared |

`AgentTimeoutError` is normal — it means the agent ran out of time, not that something broke. The retryable errors (`DaytonaError`, `EnvironmentStartTimeoutError`) are transient infrastructure issues that `harbor jobs resume` will retry.

### Quick Error Check (Single Shard)

```bash
python3 -c "
import json
r = json.load(open('path/to/shard0/result.json'))
s = r['stats']
print(f'Trials: {s[\"n_trials\"]}, Errors: {s[\"n_errors\"]}, Success: {s[\"n_trials\"] - s[\"n_errors\"]}')
for eval_name, data in s['evals'].items():
    for etype, trials in data.get('exception_stats', {}).items():
        print(f'  {etype}: {len(trials)}')
"
```

### Aggregate Across All 8 Shards for a Dataset

```bash
python3 -c "
import json, glob, os
from collections import defaultdict

base = '/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/'
dataset = 'exp_rle_detailed_10k'  # change this

files = glob.glob(base + dataset + '_jupiter_*_shard*/result.json')
totals = defaultdict(int)
errors = defaultdict(int)

for f in sorted(files):
    r = json.load(open(f))
    s = r['stats']
    totals['trials'] += s['n_trials']
    totals['errors'] += s['n_errors']
    for eval_name, data in s['evals'].items():
        for etype, trials in data.get('exception_stats', {}).items():
            errors[etype] += len(trials)

print(f'{dataset}: {totals[\"trials\"]} trials, {totals[\"errors\"]} errors, {totals[\"trials\"] - totals[\"errors\"]} success')
for etype, count in sorted(errors.items(), key=lambda x: -x[1]):
    print(f'  {etype}: {count}')
"
```

### Bulk Error Report Across All Completed Datasets

```bash
python3 -c "
import json, glob, os, re
from collections import defaultdict

base = '/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/'
# Group by dataset name (strip the hash and shard suffix)
datasets = defaultdict(lambda: defaultdict(int))

for f in sorted(glob.glob(base + '*/result.json')):
    dirname = os.path.basename(os.path.dirname(f))
    # Extract dataset name: everything before _jupiter_
    m = re.match(r'(.+?)_jupiter_.*_shard\d+', dirname)
    if not m:
        continue
    ds = m.group(1)
    r = json.load(open(f))
    s = r['stats']
    datasets[ds]['trials'] += s['n_trials']
    datasets[ds]['errors'] += s['n_errors']
    for eval_name, data in s['evals'].items():
        for etype, trials in data.get('exception_stats', {}).items():
            datasets[ds][etype] += len(trials)

for ds in sorted(datasets):
    d = datasets[ds]
    success = d['trials'] - d['errors']
    print(f'{ds}: {d[\"trials\"]} trials, {success} success, {d[\"errors\"]} errors')
    for k, v in sorted(d.items(), key=lambda x: -x[1]):
        if k not in ('trials', 'errors'):
            print(f'  {k}: {v}')
"
```

### Deciding What to Rerun

Rerun a dataset if it has a high count of retryable errors relative to total trials:
- **High `EnvironmentStartTimeoutError`** — Daytona was overloaded during the run; rerunning will retry these
- **High `DaytonaError`** — transient sandbox failures; rerunning will retry
- **`AgentTimeoutError` only** — no need to rerun, the agent just ran out of time (expected)
- **Low success count** — if success < expected (e.g., <7000 out of 10K), check which errors are retryable

Rule of thumb: if retryable errors (`DaytonaError` + `EnvironmentStartTimeoutError`) > 500, rerun.

## Rerunning Failed Jobs

Harbor supports resume — resubmitting the same dataset will skip already-completed trials and retry failed ones (DaytonaError, EnvironmentStartTimeout, etc.).

```bash
# Rerun a single dataset, chained after the last job
sbatch --dependency=afterany:LAST_JOB run_harbor_glm47_jupiter.sbatch DCAgent/dataset_name

# Common reasons to rerun:
# - High DaytonaError count (transient sandbox failures)
# - High EnvironmentStartTimeoutError (Daytona overloaded)
# - Job cancelled externally
```

## Key Paths

| Path | Description |
|------|-------------|
| `/e/scratch/jureap59/etash/OpenThoughts-Agent/data/sbatches/` | All pipeline scripts |
| `/e/scratch/jureap59/etash/harbor/` | Harbor source (branch: `guha1/add-file-retention`) |
| `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/` | OT-Agent repo (commons.py, tasks_parquet_converter) |
| `/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/` | Job output directories with result.json |
| `/e/scratch/jureap59/etash/glm47_tmp/dockerfile_cache/` | Cached environment dirs for snapshot registration |
| `/e/data1/datasets/playground/mmlaion/shared/guha1/tmp_glm47/` | HF cache and temp files |

## Daytona Configuration

Two orgs used for load balancing (configured in the sbatch script):
- **org1** (25% weight): `dtn_17868a...` — 100 snapshot limit
- **org2** (75% weight): `dtn_ecfb75...` — 45 snapshot limit

31 unique environment hashes across all 65 datasets, all pre-registered as snapshots on both orgs.
