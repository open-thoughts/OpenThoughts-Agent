# Harbor Evaluation Workflow — Jupiter HPC

End-to-end guide for running Harbor agent evaluations on the Jupiter cluster (JSC GH200).

---

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Snapshot Pre-creation](#3-snapshot-pre-creation)
4. [Launching Evaluations](#4-launching-evaluations)
5. [Monitoring Jobs](#5-monitoring-jobs)
6. [Uploading Results](#6-uploading-results)
7. [Database Schema](#7-database-schema)
8. [Common Pitfalls & Troubleshooting](#8-common-pitfalls--troubleshooting)
9. [Key File Paths Reference](#9-key-file-paths-reference)

---

## 1. Overview & Architecture

Harbor is a framework for evaluating AI agents against benchmark tasks. The evaluation pipeline has four main components:

```
┌──────────────┐     ┌──────────────────┐     ┌───────────┐     ┌──────────┐
│   Harbor     │────>│  Daytona Cloud   │────>│  Verifier  │────>│ Results  │
│   Agent      │     │  Sandbox (Docker) │     │  (tests/)  │     │  Upload  │
│  (terminus-2)│     │                  │     │            │     │ HF + DB  │
└──────────────┘     └──────────────────┘     └───────────┘     └──────────┘
       │                      │
       │                      │
  vLLM Server           Daytona API
  (GPU node)         (cloud sandboxes)
```

- **Agent**: `terminus-2` — the standard eval agent. Takes task instructions, executes in a sandbox, writes output.
- **Daytona**: Cloud sandbox provider. Each task runs in an isolated Docker container. Snapshots pre-built from `environment/Dockerfile`.
- **Verifier**: Runs `tests/test.sh` inside the container after the agent finishes. Writes reward (0 or 1) to `/logs/verifier/reward.txt`.
- **Supabase DB**: Stores benchmarks, tasks, jobs, trials, and model usage records.
- **HuggingFace Hub**: Stores agent traces (conversation logs) as HF datasets.

### Execution Flow (Slurm)

1. Sbatch starts vLLM server on GPU node (TP=4 across 4 GH200 GPUs)
2. SSH tunnel provides internet to compute node via SOCKS5 proxy
3. Harbor orchestrator runs N trials concurrently against Daytona sandboxes
4. Each trial: create sandbox → agent runs → verifier checks → record result
5. After all trials: check DaytonaError count → upload traces to HF → upload records to DB

---

## 2. Dataset Preparation

### Task Directory Structure

Each task is a directory with this layout:

```
<task_id>/
├── task.toml          # Configuration (timeouts, resources, metadata)
├── instruction.md     # Natural language task description for the agent
├── environment/       # Docker build context
│   ├── Dockerfile     # Container definition (required)
│   └── workspace/     # Optional files copied into container
├── tests/             # Verification scripts
│   ├── test.sh        # Main test runner (writes reward to /logs/verifier/reward.txt)
│   └── ...            # Test files, expected answers, etc.
└── solution/          # Optional reference solution
    └── solve.sh
```

### task.toml Format

```toml
version = "1.0"

[metadata]
author_name = "Your Name"
author_email = "you@example.com"
difficulty = "medium"
category = "reasoning"
tags = ["qa", "web-search"]
source = "gaia-benchmark/GAIA"

[verifier]
timeout_sec = 300

[verifier.env]
# Optional: env vars passed to verifier (e.g., for LLM judges)
# Values with ${VAR} are resolved from the host environment at runtime
OPENAI_API_KEY = "${OPENAI_API_KEY}"
MODEL_NAME = "openai/gpt-5-2025-08-07"

[agent]
timeout_sec = 600

[environment]
build_timeout_sec = 600
cpus = 1
memory_mb = 2048
storage_mb = 10240
```

### Three Verifier Patterns

**1. String Match (GAIA)**
```bash
# tests/test.sh
AGENT_ANSWER=$(cat /app/answer.txt | tr '[:upper:]' '[:lower:]' | xargs)
EXPECTED=$(cat /tests/expected_answer.txt | tr '[:upper:]' '[:lower:]' | xargs)
if [ "$AGENT_ANSWER" = "$EXPECTED" ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi
```

**2. Code Tests (Aider Polyglot)**
```bash
# tests/test.sh — runs language-specific test suite
# C++: cmake + ctest, Python: pytest, Go: go test, etc.
# Passes → reward=1, Fails → reward=0
```

**3. LLM Judge (FinanceAgent)**
```bash
# tests/test.sh — calls run_test.py which uses LiteLLM
# Sends agent answer + expected answer to gpt-5 for semantic comparison
# Writes judgment to /logs/verifier/judgment.json
# Writes reward (0 or 1) to /logs/verifier/reward.txt
```

The `[verifier.env]` section in task.toml supplies API keys to the LLM judge.

### Environment Hashing

Harbor uses `environment_dir_hash_truncated()` to compute a 12-character hex hash of the entire `environment/` directory. This hash determines the snapshot name used by Daytona.

```python
# From harbor/utils/container_cache.py
def environment_dir_hash(env_dir: Path) -> str:
    h = hashlib.sha256()
    for file_path in sorted(env_dir.rglob("*")):
        if file_path.is_file():
            rel = str(file_path.relative_to(env_dir))
            h.update(rel.encode("utf-8"))
            h.update(file_path.read_bytes())
    return h.hexdigest()

def environment_dir_hash_truncated(env_dir: Path, truncate: int = 12) -> str:
    return environment_dir_hash(env_dir)[:truncate]
```

Key details:
- Hashes **all files** in `environment/` (not just Dockerfile), including workspace files
- Files processed in sorted order for determinism
- Both relative path and file content contribute to the hash
- Tasks with identical `environment/` directories share a snapshot

### Pre-downloaded Datasets

These datasets are already downloaded and ready to use on the shared filesystem:

| Dataset | Tasks | Local Path |
|---------|------:|------------|
| Aider Polyglot | 225 | `/e/data1/.../guha1/datasets/DCAgent2_aider_polyglot` |
| BFCL Parity | 123 | `/e/data1/.../guha1/datasets/DCAgent2_bfcl-parity` |
| Terminal Bench v2 | 89 | `/e/data1/.../guha1/datasets/DCAgent2_terminal_bench_2` |
| Dev Set 71 | 70 | `/e/data1/.../guha1/datasets/DCAgent_dev_set_71_tasks` |
| Dev Set v2 | 100 | `/e/data1/.../guha1/datasets/DCAgent_dev_set_v2` |
| FinanceAgent | 50 | `/e/data1/.../guha1/datasets/financeagent` |
| FinanceAgent Terminal | 50 | `/e/data1/.../guha1/datasets/financeagent_terminal` |
| FinanceAgent Terminal v2 | 50 | `/e/data1/.../guha1/datasets/financeagent_terminal_v2` |
| FinanceAgent Terminal+Keys | 50 | `/e/data1/.../guha1/datasets/financeagent_terminal_withkeys` |
| GAIA (full) | 165 | `/e/data1/.../guha1/datasets/gaia` |
| GAIA 127 | 127 | `/e/data1/.../guha1/datasets/gaia_127` |
| GAIA 127 +Tools | 127 | `/e/data1/.../guha1/datasets/gaia_127_withtools` |
| MedAgentBench | 300 | `/e/data1/.../guha1/datasets/medagentbench` |

All paths above expand to `/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/<name>`.

### Downloading a Dataset from HuggingFace

Datasets are hosted as HF dataset repos (e.g., `DCAgent/dev_set_v2`). Download with `snapshot_download.py`:

```bash
PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"
DCFT="/e/scratch/jureap59/guha1/OpenThoughts-Agent"
REPO_ID="DCAgent2/aider_polyglot"

# Download to a local directory (real files, no symlinks — Daytona needs real files)
LOCAL_DIR="/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/DCAgent2_aider_polyglot"
$PYTHON "$DCFT/eval/jupiter/snapshot_download.py" "$REPO_ID" --local-dir "$LOCAL_DIR"

# Verify
ls "$LOCAL_DIR" | head -5
ls "$LOCAL_DIR"/$(ls "$LOCAL_DIR" | head -1)/  # Should show task.toml, instruction.md, environment/, tests/
```

The script uses `huggingface_hub.snapshot_download()` with `local_dir` to get real files (not symlinks). If the local dir already has valid task directories it skips re-downloading.

For local paths, no download needed — just point directly:
```bash
DATASET_PATH="/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/gaia_127"
```

### Checking Unique Dockerfiles / Environment Hashes

Before pre-creating snapshots, you need to know how many unique environments exist in your dataset. This tells you exactly which snapshots to create.

```bash
PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"
export PYTHONPATH="/e/scratch/jureap59/etash/harbor/src"

# Quick check: count unique environment hashes and list them
$PYTHON -c "
from pathlib import Path
from harbor.utils.container_cache import analyze_task_dockerfiles, get_task_environment_hash

dataset = Path('/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/gaia_127')
task_dirs = sorted([d for d in dataset.iterdir() if d.is_dir() and (d / 'instruction.md').exists()])

stats = analyze_task_dockerfiles(task_dirs)
print(f'Total tasks:           {stats.total_tasks}')
print(f'Tasks with Dockerfile: {stats.tasks_with_dockerfile}')
print(f'Without Dockerfile:    {stats.tasks_without_dockerfile}')
print(f'Unique env hashes:     {stats.unique_hashes}')
print()
print('Hash distribution (hash → task count):')
for h, count in stats.hash_counts.most_common():
    # Pick one representative task for this hash (needed for Dockerfile path later)
    for d in task_dirs:
        if get_task_environment_hash(d) == h:
            print(f'  harbor__{h}__snapshot  ({count} tasks)  e.g. {d.name}')
            break
"
```

Example output for GAIA-127:
```
Total tasks:           127
Tasks with Dockerfile: 127
Without Dockerfile:    0
Unique env hashes:     1
Hash distribution:
  harbor__92ea7b6dd33f__snapshot  (127 tasks)  e.g. 0383a3ee-47a7-41a4-b493-519bdefe0488
```

Example output for Aider Polyglot (many unique environments):
```
Total tasks:           225
Tasks with Dockerfile: 225
Without Dockerfile:    0
Unique env hashes:     225
Hash distribution:
  harbor__a1b2c3d4e5f6__snapshot  (1 tasks)  e.g. polyglot_cpp_allergies
  harbor__b2c3d4e5f6a7__snapshot  (1 tasks)  e.g. polyglot_go_bowling
  ...
```

### Creating a New Dataset from Scratch

```bash
# 1. Create dataset directory structure
mkdir -p my_dataset/task_001/{environment,tests,solution}

# 2. Write task.toml, instruction.md, Dockerfile, test.sh, solve.sh

# 3. Verify hash for a single task
$PYTHON -c "
from pathlib import Path
from harbor.utils.container_cache import environment_dir_hash_truncated
h = environment_dir_hash_truncated(Path('my_dataset/task_001/environment'))
print(f'Hash: {h}')
print(f'Snapshot name: harbor__{h}__snapshot')
"

# 4. Check unique snapshots across entire dataset (see section above)
```

---

## 3. Snapshot Pre-creation

### Why Pre-create

Without pre-creation, each unique environment hash triggers a Docker build inside Daytona on the first trial that uses it. This adds minutes of latency and can fail under load. Pre-creating snapshots ensures they're in `ACTIVE` state before the eval starts.

### Snapshot Naming Convention

```
harbor__{hash}__snapshot          # Regular evals
harbor__{hash}__RL__snapshot      # RL training evals (DAYTONA_TARGET=RL)
```

Where `{hash}` is the 12-char truncated SHA256 from `environment_dir_hash_truncated()`.

### Daytona Keys and Orgs

Three Daytona orgs are used. Keys are stored in `~/secrets.env` — **never hardcode them in scripts**.

| Key Name | Env Var | Use |
|----------|---------|-----|
| org1 | `DAYTONA_KEY_ORG1` | Data + eval |
| org2 | `DAYTONA_KEY_ORG2` | Eval only (more quota) |
| RL key | `DAYTONA_KEY_RL` | RL training only |

Snapshots must be pre-created on **both org1 and org2** for regular evals (the sbatch script randomly selects one with 3:1 weighting toward org2). For RL evals, pre-create on the RL org only.

### End-to-End: Compute Hashes → Pre-create on Both Orgs

Here's the full workflow from dataset to ready-to-eval:

**Step 1: Compute unique environment hashes** (see "Checking Unique Dockerfiles" above)

```bash
PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"
export PYTHONPATH="/e/scratch/jureap59/etash/harbor/src"
DATASET="/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/gaia_127"

# Get unique hashes and a representative Dockerfile path for each
$PYTHON -c "
from pathlib import Path
from harbor.utils.container_cache import analyze_task_dockerfiles, get_task_environment_hash

dataset = Path('$DATASET')
task_dirs = sorted([d for d in dataset.iterdir() if d.is_dir() and (d / 'instruction.md').exists()])
stats = analyze_task_dockerfiles(task_dirs)

print(f'Unique snapshots needed: {stats.unique_hashes}')
for h, count in stats.hash_counts.most_common():
    # Find one representative task for this hash
    for d in task_dirs:
        if get_task_environment_hash(d) == h:
            dockerfile = d / 'environment' / 'Dockerfile'
            print(f'  harbor__{h}__snapshot -> {dockerfile}  ({count} tasks)')
            break
"
```

**Step 2: Pre-create snapshots on both orgs**

The pre-creation script reads keys from `~/secrets.env` and creates snapshots on both Daytona orgs:

```python
#!/usr/bin/env python3
"""Pre-create Daytona snapshots for eval datasets on both orgs.

Keys are read from ~/secrets.env (DAYTONA_KEY_ORG1, DAYTONA_KEY_ORG2).
Edit the SNAPSHOTS dict below with output from step 1.
"""
import asyncio
import os

# Load secrets
with open(os.path.expanduser("~/secrets.env")) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            if line.startswith("export "):
                line = line[7:]
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip().strip('"').strip("'")

import sys
sys.path.insert(0, "/e/scratch/jureap59/etash/harbor/src")
sys.path.insert(0, "/e/scratch/jureap59/guha1/pip_packages")
sys.path.insert(0, "/e/scratch/jureap59/etash/pip_extras")

from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources
from daytona._async.snapshot import SnapshotState

# --- EDIT THIS: snapshot name → representative Dockerfile path ---
SNAPSHOTS = {
    "harbor__92ea7b6dd33f__snapshot": "/e/data1/.../gaia/task_abc/environment/Dockerfile",
    "harbor__bfc3340ef3c7__snapshot": "/e/data1/.../financeagent/task_0/environment/Dockerfile",
}

# Keys from environment (loaded from ~/secrets.env)
DAYTONA_KEYS = {
    "org1": os.environ["DAYTONA_KEY_ORG1"],
    "org2": os.environ["DAYTONA_KEY_ORG2"],
}


async def create_snapshot(client, name, dockerfile_path, org_name):
    """Create a single snapshot, handling already-exists gracefully."""
    try:
        snap = await client.snapshot.get(name)
        if snap.state == SnapshotState.ACTIVE:
            print(f"  [{org_name}] {name}: already ACTIVE, skipping")
            return True
        elif snap.state == SnapshotState.ERROR:
            print(f"  [{org_name}] {name}: ERROR state, deleting and recreating...")
            await client.snapshot.delete(snap)
    except Exception:
        pass  # Doesn't exist yet

    print(f"  [{org_name}] Creating {name} from {dockerfile_path}...")
    try:
        await client.snapshot.create(
            CreateSnapshotParams(
                name=name,
                image=Image.from_dockerfile(dockerfile_path),
                resources=Resources(cpu=1, memory=1, disk=3),
            )
        )
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"  [{org_name}] {name}: already exists (global), OK")
            return True
        print(f"  [{org_name}] {name}: create FAILED: {e}")
        return False

    # Poll for ACTIVE state (up to 10 minutes)
    for i in range(120):
        await asyncio.sleep(5)
        try:
            snap = await client.snapshot.get(name)
            if snap.state == SnapshotState.ACTIVE:
                print(f"  [{org_name}] {name}: ACTIVE (took ~{i*5}s)")
                return True
            elif snap.state == SnapshotState.ERROR:
                print(f"  [{org_name}] {name}: entered ERROR state")
                return False
        except Exception:
            pass
    print(f"  [{org_name}] {name}: TIMEOUT waiting for ACTIVE")
    return False


async def main():
    for org_name, api_key in DAYTONA_KEYS.items():
        print(f"\n=== {org_name} ({api_key[:12]}...) ===")
        client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="us"))
        try:
            for snap_name, dockerfile in SNAPSHOTS.items():
                await create_snapshot(client, snap_name, dockerfile, org_name)
        finally:
            await client.close()
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
```

**Step 3: Run it** (must use Python 3.12, login node only):

```bash
/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python precreate_snapshots.py
```

Expected output:
```
=== org1 (dtn_17868a19...) ===
  [org1] harbor__92ea7b6dd33f__snapshot: already ACTIVE, skipping
  [org1] harbor__bfc3340ef3c7__snapshot: ACTIVE (took ~30s)

=== org2 (dtn_ecfb7592...) ===
  [org2] harbor__92ea7b6dd33f__snapshot: already ACTIVE, skipping
  [org2] harbor__bfc3340ef3c7__snapshot: ACTIVE (took ~25s)

Done!
```

For datasets with many unique snapshots (like Aider Polyglot with 225), this can take a while. The snapshots are global — once created on an org, they stay cached.

### Automated Hash → Snapshot Script

For large datasets, you can combine hash computation + pre-creation into one script:

```bash
PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"
export PYTHONPATH="/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/guha1/pip_packages:/e/scratch/jureap59/etash/pip_extras"
DATASET="/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/DCAgent2_aider_polyglot"

# Generate snapshot dict and pre-create in one go
$PYTHON -c "
import asyncio, os, sys

# Load secrets
with open(os.path.expanduser('~/secrets.env')) as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            if line.startswith('export '): line = line[7:]
            k, v = line.split('=', 1)
            os.environ[k.strip()] = v.strip().strip('\"').strip(\"'\")

from pathlib import Path
from harbor.utils.container_cache import get_task_environment_hash
from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources
from daytona._async.snapshot import SnapshotState

dataset = Path('$DATASET')
task_dirs = sorted([d for d in dataset.iterdir() if d.is_dir() and (d / 'instruction.md').exists()])

# Build unique hash → Dockerfile mapping
snapshots = {}
for d in task_dirs:
    h = get_task_environment_hash(d)
    if h:
        snap_name = f'harbor__{h}__snapshot'
        if snap_name not in snapshots:
            snapshots[snap_name] = str(d / 'environment' / 'Dockerfile')

print(f'Found {len(snapshots)} unique snapshots to pre-create')

async def precreate_all():
    for org_name, key_env in [('org1', 'DAYTONA_KEY_ORG1'), ('org2', 'DAYTONA_KEY_ORG2')]:
        api_key = os.environ[key_env]
        print(f'\n=== {org_name} ({api_key[:12]}...) ===')
        client = AsyncDaytona(DaytonaConfig(api_key=api_key, target='us'))
        try:
            for name, dockerfile in snapshots.items():
                try:
                    snap = await client.snapshot.get(name)
                    if snap.state == SnapshotState.ACTIVE:
                        print(f'  [{org_name}] {name}: ACTIVE')
                        continue
                except Exception:
                    pass
                print(f'  [{org_name}] Creating {name}...')
                try:
                    await client.snapshot.create(CreateSnapshotParams(
                        name=name, image=Image.from_dockerfile(dockerfile),
                        resources=Resources(cpu=1, memory=1, disk=3)))
                except Exception as e:
                    if 'already exists' in str(e).lower():
                        print(f'  [{org_name}] {name}: already exists')
                        continue
                    print(f'  [{org_name}] {name}: FAILED: {e}')
                    continue
                # Poll for ACTIVE
                for i in range(120):
                    await asyncio.sleep(5)
                    snap = await client.snapshot.get(name)
                    if snap.state == SnapshotState.ACTIVE:
                        print(f'  [{org_name}] {name}: ACTIVE ({i*5}s)')
                        break
                    elif snap.state == SnapshotState.ERROR:
                        print(f'  [{org_name}] {name}: ERROR')
                        break
        finally:
            await client.close()

asyncio.run(precreate_all())
print('Done!')
"
```

Run with Python 3.12 (login node Python 3.9 is too old):
```bash
/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python precreate_snapshots.py
```

### RL Snapshot Pre-creation

For RL jobs, snapshot names include `__RL__`:
```
harbor__{hash}__RL__snapshot
```

Use the RL Daytona key and `DaytonaConfig(api_key=RL_KEY, target="RL")`.

---

## 4. Launching Evaluations

### 4a. Slurm Jobs (GPU Models via sbatch)

Use `unified_eval_harbor.sbatch` for models that need a local vLLM server.

**Positional Arguments:**
```
$1 = MODEL       # HF model name (e.g., mlfoundations-dev/some_model)
$2 = REPO_ID     # HF dataset repo or local path (starts with / for local)
$3 = BENCHMARK_ID  # Optional: DB benchmark UUID
$4 = RUN_TAG_ARG   # Optional: override run tag
```

**Environment Variables:**
```bash
EVAL_N_CONCURRENT=128       # Concurrent trials (default: 128)
EVAL_GPU_MEMORY_UTIL=0.95   # vLLM GPU memory utilization (default: 0.95)
EVAL_DAYTONA_THRESHOLD=3    # Max DaytonaErrors before skipping upload (default: 3)
EVAL_SNAPSHOT_NAME=...      # Force a specific snapshot template
EVAL_TIMEOUT_MULTIPLIER=... # Scale agent timeout
EVAL_OVERRIDE_MEMORY_MB=... # Override container memory
```

**Example Submission:**
```bash
sbatch --job-name="eval_mymodel_gaia" \
  eval/jupiter/unified_eval_harbor.sbatch \
  "mlfoundations-dev/my_model" \
  "/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/gaia_127"
```

**What the sbatch script does:**

1. **Environment setup**: Sources `jupiter.env`, `secrets.env`, sets PYTHONPATH, creates ld-linux wrappers (ARM GH200 needs system loader since conda has no exec perms on shared filesystem)
2. **Daytona key selection**: Randomly picks org1 (25%) or org2 (75%)
3. **SSH tunnel + proxychains**: Compute nodes have no internet; creates SOCKS5 tunnel to login node
4. **vLLM server**: Starts on port 8000 with TP=4, waits up to ~33 minutes for ready
5. **Dataset download**: Via proxychains (HF) or direct local path
6. **Start vs Resume logic**: If `$EVAL_JOBS_DIR/$RUN_TAG/config.json` exists, resumes (retries DaytonaError and EnvironmentStartTimeoutError trials). Otherwise starts fresh.
7. **DaytonaError check**: If error count > threshold, skips upload
8. **Upload**: Traces to HF, records to Supabase DB

**Auto-resume in sbatch:**

The sbatch script has built-in auto-resume. If you submit the same model+dataset combination and a previous job directory (`$EVAL_JOBS_DIR/$RUN_TAG/config.json`) exists, it automatically switches to `harbor jobs resume` instead of `harbor jobs start`. This means:

- **Re-submitting the same sbatch is safe** — it picks up where it left off
- Only trials with transient errors (DaytonaError, EnvironmentStartTimeoutError, DaytonaRateLimitError) are retried
- Completed trials are skipped
- You can resubmit after a Slurm timeout or OOM without losing progress

The run tag is derived from `${SAFE_REPO}_${SAFE_MODEL}` (e.g., `gaia_127_mlfoundations-dev_my_model`), so the same model+dataset always maps to the same job directory.

```bash
# First run: starts fresh
sbatch eval/jupiter/unified_eval_harbor.sbatch "my-org/my-model" "/path/to/dataset"

# Job times out or gets OOM-killed. Resubmit:
sbatch eval/jupiter/unified_eval_harbor.sbatch "my-org/my-model" "/path/to/dataset"
# ^ Automatically resumes, retrying only failed trials
```

**Manual harbor commands (for reference):**
```bash
# New job
harbor jobs start -p $DATASET_PATH --n-concurrent 128 --agent terminus-2 \
  --model "hosted_vllm/$MODEL" --env daytona \
  --agent-kwarg "api_base=http://localhost:8000/v1" \
  --agent-kwarg "key=fake_key" --n-attempts 3 \
  --job-name "$RUN_TAG" --config eval/jupiter/dcagent_eval_config.yaml

# Resume (retry transient errors only)
harbor jobs resume -p "$EXISTING_JOB_DIR" \
  --filter-error-type EnvironmentStartTimeoutError \
  --filter-error-type DaytonaError \
  --filter-error-type DaytonaRateLimitError
```

### 4b. Commercial Models (No GPU, Login Node / tmux)

Commercial API models (OpenAI, Anthropic) don't need vLLM — run directly on the login node inside a tmux session.

**Concrete commands to run commercial evals:**

```bash
# 1. Start a tmux session
tmux new-session -s commercial_evals

# 2. Set up environment
HARBOR_PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"
export PYTHONPATH="/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/guha1/pip_packages:/e/scratch/jureap59/etash/pip_extras"
EVAL_JOBS_DIR="/e/data1/datasets/playground/mmlaion/shared/guha1/eval_jobs"

source ~/secrets.env
# DAYTONA_API_KEY, OPENAI_API_KEY, HF_TOKEN loaded from secrets.env
export DAYTONA_TARGET="us"

# 3. Run a single model on a single dataset
$HARBOR_PYTHON -m harbor.cli.main jobs start \
  -p "/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/gaia_127" \
  --n-concurrent 32 \
  --agent terminus-2 \
  --model "openai/gpt-5-mini" \
  --env daytona \
  --ek auto_snapshot=true \
  --no-force-build \
  --n-attempts 1 \
  --job-name "gaia_127_openai_gpt-5-mini" \
  --jobs-dir "$EVAL_JOBS_DIR"

# 4. Resume a failed/partial run
$HARBOR_PYTHON -m harbor.cli.main jobs resume \
  -p "$EVAL_JOBS_DIR/gaia_127_openai_gpt-5-mini" \
  --filter-error-type EnvironmentStartTimeoutError \
  --filter-error-type DaytonaError
```

**Batch script for multiple models x datasets:**

The script at `/e/scratch/jureap59/etash/run_commercial_evals.sh` loops over models and datasets with auto-skip (if `result.json` exists) and auto-resume (if `config.json` exists):

```bash
#!/bin/bash
set -eo pipefail

HARBOR_PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"
export PYTHONPATH="/e/scratch/jureap59/etash/harbor/src:/e/scratch/jureap59/guha1/pip_packages:/e/scratch/jureap59/etash/pip_extras"
EVAL_JOBS_DIR="/e/data1/datasets/playground/mmlaion/shared/guha1/eval_jobs"

source ~/secrets.env
# Keys loaded: DAYTONA_API_KEY, OPENAI_API_KEY
export DAYTONA_TARGET="us"

MODELS=("openai/gpt-5-mini" "openai/gpt-5-nano" "openai/gpt-5")

DATASETS=(
    "gaia_127:/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/gaia_127"
    "financeagent:/e/data1/datasets/playground/mmlaion/shared/guha1/datasets/financeagent"
)

N_CONCURRENT=32

harbor_cmd() { $HARBOR_PYTHON -m harbor.cli.main "$@"; }

for ds_entry in "${DATASETS[@]}"; do
    DS_NAME="${ds_entry%%:*}"
    DS_PATH="${ds_entry##*:}"

    for MODEL in "${MODELS[@]}"; do
        SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
        JOB_NAME="${DS_NAME}_${SAFE_MODEL}"
        JOB_DIR="${EVAL_JOBS_DIR}/${JOB_NAME}"

        # Skip completed
        if [ -f "$JOB_DIR/result.json" ]; then
            echo "SKIP: $JOB_NAME already done"
            continue
        fi

        # Resume or start
        if [ -d "$JOB_DIR" ] && [ -f "$JOB_DIR/config.json" ]; then
            echo "RESUME: $JOB_NAME"
            harbor_cmd jobs resume -p "$JOB_DIR" \
                --filter-error-type EnvironmentStartTimeoutError \
                --filter-error-type DaytonaError
        else
            echo "START: $JOB_NAME"
            harbor_cmd jobs start -p "$DS_PATH" \
                --n-concurrent "$N_CONCURRENT" --agent terminus-2 \
                --model "$MODEL" --env daytona \
                --ek auto_snapshot=true --no-force-build \
                --n-attempts 1 --job-name "$JOB_NAME" \
                --jobs-dir "$EVAL_JOBS_DIR"
        fi
    done
done
```

Run in tmux:
```bash
tmux new-session -s commercial_evals
bash /e/scratch/jureap59/etash/run_commercial_evals.sh 2>&1 | tee /e/scratch/jureap59/etash/commercial_evals.log
# Ctrl-b d to detach, tmux attach -t commercial_evals to reattach
```

Key differences from Slurm:
- `--n-attempts 1` (commercial APIs are reliable, no retries needed)
- `--n-concurrent 32` (lower than Slurm's 128 — login node has limited resources)
- No vLLM, no SSH tunnel, no proxychains (login nodes have direct internet)
- Model specified directly (e.g., `openai/gpt-5-mini`), not as `hosted_vllm/...`

### 4c. Automated Eval Listener (Queue Management + Dependencies)

The `unified_eval_listener.py` is a long-running daemon that polls the Supabase DB for new models and auto-submits Slurm eval jobs. It handles queue management, deduplication, stale job detection, and Slurm dependency chains.

**How it works:**
1. Polls DB for recently registered models (configurable lookback window)
2. For each (model, dataset) pair, checks if a job already exists (Pending/Started/Finished)
3. Skips finished jobs, detects and auto-cancels stale jobs
4. Creates a "Pending" DB entry, then submits `sbatch`
5. Supports **sliding-window batch-size** to limit concurrent Slurm jobs

**Sliding-Window Dependencies (`--batch-size`):**

With `--batch-size N`, at most N jobs run concurrently. Job `i` depends on job `i-N` finishing (using Slurm's `afterany` dependency). As one job finishes, the next starts immediately — no waiting for entire waves.

```
--batch-size 4 with 10 jobs:

Job 0 ─────────────┐
Job 1 ─────────────┤ (run immediately, first 4)
Job 2 ─────────────┤
Job 3 ─────────────┤
Job 4 ──afterany:0──┤ (starts when job 0 finishes)
Job 5 ──afterany:1──┤ (starts when job 1 finishes)
Job 6 ──afterany:2──┤
...
```

**Example usage:**

```bash
PYTHON="/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python"

# Dry run: see what would be submitted
$PYTHON eval/jupiter/unified_eval_listener.py \
  --preset dev --dry-run --verbose --once

# Submit with batch-size (max 4 concurrent Slurm jobs)
$PYTHON eval/jupiter/unified_eval_listener.py \
  --preset dev --once --verbose --batch-size 4

# Submit with a priority file (only eval specific models)
$PYTHON eval/jupiter/unified_eval_listener.py \
  --preset aider --once --verbose \
  --priority-file eval/jupiter/priority_models.txt \
  --batch-size 4

# Long-running daemon (checks every 4 hours)
$PYTHON eval/jupiter/unified_eval_listener.py \
  --preset dev --verbose --check-hours 4

# Custom datasets (no preset)
$PYTHON eval/jupiter/unified_eval_listener.py \
  --datasets "DCAgent/dev_set_v2,DCAgent2/aider_polyglot" \
  --once --verbose --batch-size 2

# Add an explicit dependency (e.g., wait for another job first)
$PYTHON eval/jupiter/unified_eval_listener.py \
  --preset dev --once --dependency "afterany:12345"
```

**Available Presets:**

| Preset | Datasets | N_Concurrent | Description |
|--------|----------|-------------|-------------|
| `dev` | DCAgent/dev_set_71_tasks | 128 | Dev set (71 tasks) |
| `v2` | DCAgent/dev_set_v2 | 128 | Dev set v2 |
| `bfcl` | DCAgent2/bfcl-parity | 16 | Berkeley Function Calling |
| `aider` | DCAgent2/aider_polyglot | 64 | Aider Polyglot |
| `swebench` | DCAgent/swebench_verified_eval_set | 32 | SWE-bench verified |
| `tb2` | DCAgent/terminal_bench_v2 | 64 | Terminal Bench v2 |

**Manual sbatch with Slurm dependencies** (without the listener):

```bash
# Submit 3 jobs, max 2 concurrent:
JOB1=$(sbatch --parsable eval/jupiter/unified_eval_harbor.sbatch "model-A" "/path/to/dataset")
JOB2=$(sbatch --parsable eval/jupiter/unified_eval_harbor.sbatch "model-B" "/path/to/dataset")
JOB3=$(sbatch --parsable --dependency=afterany:$JOB1 eval/jupiter/unified_eval_harbor.sbatch "model-C" "/path/to/dataset")
# JOB3 starts only after JOB1 finishes (success or failure)

echo "Submitted: $JOB1, $JOB2, $JOB3"
squeue -u $USER  # Verify
```

**Run tag & auto-resume interaction:**

When using the listener or resubmitting manually, the run tag determines whether auto-resume kicks in:
- Run tag = `${SAFE_REPO}_${SAFE_MODEL}` (e.g., `gaia_127_mlfoundations-dev_my_model`)
- If `$EVAL_JOBS_DIR/$RUN_TAG/config.json` exists → auto-resumes
- If not → starts fresh

This means resubmitting the same (model, dataset) is always safe. The listener uses this to handle Slurm timeouts: if a job times out, the next poll detects it as stale and resubmits, and the sbatch auto-resumes.

---

## 5. Monitoring Jobs

### Slurm Jobs

```bash
# Check job status
squeue -u $USER

# View live log
tail -f eval/jupiter/logs/eval_<SLURM_JOB_ID>.out

# Check vLLM log
tail -f eval/jupiter/logs/vllm_<SLURM_JOB_ID>.log
```

### Job-Level result.json

Located at `$EVAL_JOBS_DIR/$RUN_TAG/result.json` (written when job completes):

```json
{
  "stats": {
    "n_trials": 127,
    "n_errors": 3,
    "evals": {
      "task_001__0": {
        "reward_stats": {"mean": 1.0, "std": 0.0},
        "exception_stats": {}
      },
      "task_002__0": {
        "reward_stats": {"mean": 0.0, "std": 0.0},
        "exception_stats": {"DaytonaError": ["trial_id_1"]}
      }
    }
  },
  "trial_results": [...]
}
```

### Trial-Level result.json

Each trial directory (`$RUN_TAG/task_name__attempt/result.json`) has:

```json
{
  "task_checksum": "abc123...",
  "task_name": "task_001",
  "verifier_result": {
    "reward": 1.0
  },
  "exception_info": null
}
```

When a trial fails:
```json
{
  "exception_info": {
    "exception_type": "DaytonaError",
    "exception_message": "Sandbox creation timed out"
  },
  "verifier_result": null
}
```

### Error Classification

| Error Type | Retryable | Description |
|-----------|-----------|-------------|
| `DaytonaError` | Yes | Sandbox creation/connection failures |
| `DaytonaRateLimitError` | Yes | API rate limits |
| `EnvironmentStartTimeoutError` | Yes | Container took too long to start |
| `AgentTimeoutError` | No | Agent exceeded its timeout |
| `ContextLength` / `LLMError` | No | Model issues (context overflow, API error) |
| `VerifierTimeoutError` | No | Verifier timed out |
| `SandboxBuildFailedError` | No | Dockerfile build failed |

### When to Retry a Job

**Retry (resume) when:**
- **DaytonaError / DaytonaRateLimitError / EnvironmentStartTimeoutError** — these are transient infrastructure failures (sandbox creation hiccups, rate limits, slow container starts). Retrying usually succeeds.
- **Slurm timeout or OOM kill** — the job was interrupted, not failed. Resubmit the same sbatch command; auto-resume picks up where it left off.
- **SSH tunnel failure** — if the proxy died mid-job, Daytona calls fail. Retry after ensuring SSH key is configured.
- **A few DaytonaErrors but most trials succeeded** — resume to fill in the gaps.

**Don't retry when:**
- **AgentTimeoutError** — the model genuinely couldn't solve the task in time. Retrying gives the same result. Consider increasing `EVAL_TIMEOUT_MULTIPLIER` if the timeout is too aggressive.
- **ContextLength / LLMError** — the model hit its context limit or returned invalid output. This is a model limitation, not infrastructure. Retrying won't help.
- **SandboxBuildFailedError** — the Dockerfile itself is broken. Fix the Dockerfile first.
- **VerifierTimeoutError** — the test suite is too slow. Fix the tests or increase verifier timeout in `task.toml`.
- **All trials have non-retryable errors** — the eval is done, the errors reflect real model/task behavior.

**Rule of thumb**: check the error distribution first. If most errors are DaytonaError, resume. If most are AgentTimeoutError, the results are final.

### Resuming Failed Trials

```bash
# Resume only retryable errors
harbor jobs resume -p "$JOB_DIR" \
  --filter-error-type DaytonaError \
  --filter-error-type EnvironmentStartTimeoutError \
  --filter-error-type DaytonaRateLimitError
```

`harbor jobs resume` skips completed trials and only retries trials matching the filter.

For Slurm jobs, just resubmit the same sbatch — auto-resume is built in:
```bash
sbatch eval/jupiter/unified_eval_harbor.sbatch "my-org/my-model" "/path/to/dataset"
```

### Quick Error Distribution Check

Read the job-level `result.json` (not individual trial dirs):

```bash
python3 -c "
import json
from pathlib import Path
from collections import Counter

result_path = Path('$EVAL_JOBS_DIR/$RUN_TAG/result.json')
data = json.loads(result_path.read_text())
stats = data.get('stats', {})

print(f'Total trials: {stats.get(\"n_trials\", \"?\")}')
print(f'Total errors: {stats.get(\"n_errors\", \"?\")}')

errors = Counter()
for eval_key, eval_data in stats.get('evals', {}).items():
    for exc_type, ids in eval_data.get('exception_stats', {}).items():
        errors[exc_type] += len(ids) if isinstance(ids, list) else 1

print(f'Error breakdown: {dict(errors)}')
print(f'Success: {stats.get(\"n_trials\", 0) - sum(errors.values())}')
"
```

---

## 6. Uploading Results

### upload_eval_results() — Full Reference

```python
upload_eval_results(
    job_dir,                          # Path to job directory (required)
    username="guha1",                 # Username for DB records (required)
    error_mode="skip_on_error",       # "skip_on_error" or "rollback_on_error" (required)

    # Auto-detected if not provided:
    agent_name=None,                  # Agent name (from trial config)
    agent_version=None,               # Agent version (from trial config)
    model_name=None,                  # Model name (from trial config)
    benchmark_name="gaia_127",        # Benchmark name
    benchmark_version_hash="abc...",  # SHA256 hash (64 chars)

    # Benchmark/task auto-registration:
    register_benchmark=True,          # Auto-register benchmark + tasks if not in DB

    # HuggingFace trace upload:
    hf_repo_id="DCAgent2/run_tag",   # HF dataset repo for traces
    hf_token=os.environ["HF_TOKEN"], # HF auth token
    hf_private=False,                 # Public by default
    hf_episodes="last",              # "last" or "all"

    # Other:
    git_commit_id=None,              # Optional git SHA
    forced_update=False,             # Allow overwriting existing DB records
)
```

### Two-Stage Upload

**Stage 1: HF Traces Upload**
- Exports agent conversation logs (trajectories) as a HuggingFace dataset
- Creates/updates a repo at `hf_repo_id` (e.g., `DCAgent2/gaia_127_openai_gpt-5-mini`)
- Returns the HF dataset URL

**Stage 2: DB Records Upload**
- Registers/finds: agent, model, benchmark in DB
- Creates `sandbox_jobs` entry with job metadata
- Creates `sandbox_trials` entries for each trial with timing, reward, exception info
- Creates `sandbox_trial_model_usage` entries for token tracking
- Links HF dataset URL to the job record

### register_benchmark=True

When `register_benchmark=True`, the upload function calls `register_benchmark_and_tasks_from_job()` which:

1. Registers the benchmark in `benchmarks` table if not found
2. Scans all trial directories for `result.json` → extracts `task_checksum`
3. Deduplicates tasks (important when `n_attempts > 1`)
4. Registers each unique task in `sandbox_tasks` table (with 3x retry)
5. Links tasks to benchmark in `sandbox_benchmark_tasks` table

This is critical for new benchmarks. Without it, trial uploads fail with FK constraint errors because the referenced `task_checksum` doesn't exist in `sandbox_tasks`.

### Error Modes

**`skip_on_error`** (recommended for most uploads):
- Continues uploading even if individual trials fail
- Job record always kept in DB
- Returns list of failed trials for debugging

**`rollback_on_error`** (atomic uploads):
- Deletes ALL job/trial/usage records on any error
- All-or-nothing semantics
- If HF upload fails, entire process aborts

### Example Upload Script

```python
#!/usr/bin/env python3
"""Upload eval results to HF + DB."""
import os, sys, hashlib

sys.path.insert(0, "eval/jupiter/dcagents-leaderboard")
from unified_db.utils import upload_eval_results

run_dir = "/e/data1/datasets/playground/mmlaion/shared/guha1/eval_jobs/gaia_127_openai_gpt-5-mini"
dataset_hf = "gaia_127"

# Stable benchmark version hash
benchmark_version_hash = hashlib.sha256(dataset_hf.encode()).hexdigest()

# HF repo ID (sanitized)
hf_repo_id = f"DCAgent2/gaia_127_openai_gpt-5-mini"

result = upload_eval_results(
    run_dir,
    username="guha1",
    error_mode="skip_on_error",
    hf_token=os.environ["HF_TOKEN"],
    hf_repo_id=hf_repo_id,
    register_benchmark=True,
    benchmark_name=dataset_hf,
    benchmark_version_hash=benchmark_version_hash,
)

print(f"Success: {result['success']}")
print(f"Job ID: {result.get('job_id')}")
print(f"Trials uploaded: {result.get('n_trials_uploaded')}")
if result.get('hf_dataset_url'):
    print(f"HF URL: {result['hf_dataset_url']}")
```

### Name Resolution: How Upload Finds the Right DB Records

The upload needs to match three things in the DB by **exact name**: agent, model, and benchmark. Getting the names wrong means it either fails or creates a duplicate entry.

**Benchmark name** — derived from `REPO_ID`:

```
REPO_ID                              → benchmark_name
─────────────────────────────────────────────────────
"DCAgent/dev_set_v2"                 → "dev_set_v2"       (split on "/", take last)
"/e/.../datasets/gaia_127"           → "gaia_127"         (split on "/", take last)
"/e/.../datasets/DCAgent_dev_set_v2" → "DCAgent_dev_set_v2"  ← WRONG if DB has "dev_set_v2"
```

The sbatch derives `benchmark_name = dataset_hf.split("/")[-1]`. So:
- HF repo IDs like `DCAgent/dev_set_v2` → `dev_set_v2` (correct)
- Local paths use the directory basename → could differ from what's in the DB

**If the name doesn't match an existing benchmark** and `register_benchmark=True`, it creates a **new** benchmark with that name. This is how you end up with both `dev_set_v2` and `DCAgent_dev_set_v2` in the DB.

**How to ensure you upload to the right benchmark:**

1. **Always pass `benchmark_name` explicitly** in manual uploads:
   ```python
   upload_eval_results(
       ...,
       benchmark_name="dev_set_v2",  # Must match what's in the DB
       benchmark_version_hash=hashlib.sha256("DCAgent/dev_set_v2".encode()).hexdigest(),
   )
   ```

2. **Check what's in the DB** before uploading:
   ```python
   from unified_db.utils import get_benchmark_by_name
   # Try the name you think it should be
   b = get_benchmark_by_name("dev_set_v2")
   print(b)  # If None, it doesn't exist yet
   ```

3. **For the sbatch script**, the `REPO_ID` (`$2`) controls the name. Use consistent `REPO_ID` values:
   - First run used `DCAgent/dev_set_v2` → benchmark "dev_set_v2" created
   - Later run must also use `DCAgent/dev_set_v2` (or a local path whose basename is `dev_set_v2`)
   - Using `/e/.../datasets/DCAgent_dev_set_v2` would create a separate "DCAgent_dev_set_v2" benchmark

**Model name** — auto-detected from trial config, with fallback chain:

```
Trial config model_name         → DB lookup
──────────────────────────────────────────────
"hosted_vllm/my-org/my-model"  → try "hosted_vllm/my-org/my-model", then strip to "my-org/my-model"
"openai/gpt-5-mini"            → try "openai/gpt-5-mini" directly
```

The upload strips the `hosted_vllm/` prefix automatically. If the model still isn't found, it tries to auto-register from HuggingFace `run_summary.json`.

**HF repo ID for traces** — the `sanitize_hf_repo_id()` function in the sbatch cleans the run tag for use as a HuggingFace dataset name:
- Replaces special characters with hyphens
- Collapses consecutive hyphens/dots
- Truncates long names with SHA1 suffix

---

## 7. Database Schema

### Entity Relationship Diagram

```
benchmarks ←──┐
              │ benchmark_id
sandbox_benchmark_tasks ──→ sandbox_tasks (PK: checksum)
                                  ↑
                                  │ task_checksum
agents ←─────── sandbox_jobs ──→ sandbox_trials
models ←─────── sandbox_jobs      │
                    ↑              │
                    │ job_id       │ trial_id
                    └──────────────┘
                                   │
                         sandbox_trial_model_usage
```

### Key Tables

**`benchmarks`**
| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| name | TEXT | Benchmark name (e.g., "gaia_127") |
| benchmark_version_hash | CHAR(64) | SHA256 of benchmark content |
| is_external | BOOLEAN | Whether externally hosted |

**`sandbox_tasks`**
| Column | Type | Description |
|--------|------|-------------|
| checksum | TEXT (PK) | SHA256 content hash (deduplication key) |
| source | TEXT | Benchmark name that registered this task |
| name | TEXT | Task name |
| instruction | TEXT | Task instruction text |
| agent_timeout_sec | NUMERIC | Agent execution timeout |
| verifier_timeout_sec | NUMERIC | Verifier timeout |
| path | TEXT | Task directory path |

**`sandbox_jobs`**
| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| job_name | TEXT | Run tag / job name |
| username | TEXT | Who ran the eval |
| agent_id | UUID (FK→agents) | Agent used |
| model_id | UUID (FK→models) | Model used |
| benchmark_id | UUID (FK→benchmarks) | Benchmark evaluated |
| n_trials | INTEGER | Number of trials |
| metrics | JSONB | Aggregate metrics |
| stats | JSONB | Detailed statistics |
| hf_traces_link | TEXT | HuggingFace dataset URL |
| job_status | ENUM | 'Pending', 'Started', 'Finished' |
| UNIQUE | | (agent_id, model_id, benchmark_id) |

**`sandbox_trials`**
| Column | Type | Description |
|--------|------|-------------|
| id | UUID (PK) | Auto-generated |
| trial_name | TEXT | Trial identifier |
| job_id | UUID (FK→sandbox_jobs) | Parent job |
| task_checksum | TEXT (FK→sandbox_tasks) | Task that was executed |
| reward | NUMERIC | Score (0 or 1) |
| started_at / ended_at | TIMESTAMP | Trial timing |
| environment_setup_* | TIMESTAMP | Sandbox setup timing |
| agent_execution_* | TIMESTAMP | Agent run timing |
| verifier_* | TIMESTAMP | Verification timing |
| exception_info | JSONB | Error details if failed |

**`sandbox_trial_model_usage`**
| Column | Type | Description |
|--------|------|-------------|
| trial_id | UUID (FK→sandbox_trials) | Trial |
| model_id | UUID (FK→models) | Model used |
| model_provider | TEXT | Provider (openai, anthropic, etc.) |
| n_input_tokens | INTEGER | Input tokens consumed |
| n_output_tokens | INTEGER | Output tokens consumed |
| PK | | (trial_id, model_id, model_provider) |

### Common DB Queries

```python
from unified_db.utils import get_supabase_client

client = get_supabase_client()

# Get all jobs for a benchmark
jobs = client.table('sandbox_jobs') \
    .select('*, benchmarks(name)') \
    .eq('benchmark_id', benchmark_uuid) \
    .execute()

# Get trial results for a job
trials = client.table('sandbox_trials') \
    .select('trial_name, reward, exception_info') \
    .eq('job_id', job_uuid) \
    .execute()

# Compute pass rate
rewards = [t['reward'] for t in trials.data if t['reward'] is not None]
pass_rate = sum(r > 0 for r in rewards) / len(rewards)
```

---

## 8. Common Pitfalls & Troubleshooting

### Python 3.9 on Login Node

The login node has Python 3.9 which is too old for Harbor and the Daytona SDK. Always use:
```bash
/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python  # Python 3.12
```

For RL snapshot work specifically:
```bash
/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl/bin/python3  # Python 3.12
```

### Missing SUPABASE Environment Variables

Upload will fail silently or error if `SUPABASE_URL`, `SUPABASE_ANON_KEY`, or `SUPABASE_SERVICE_ROLE_KEY` are not set. These are loaded from `~/secrets.env`:
```bash
source ~/secrets.env
# Or set KEYS=/path/to/secrets.env for auto-loading
```

### FK Constraint Errors (Missing Tasks)

If upload fails with errors like:
```
insert or update on table "sandbox_trials" violates foreign key constraint
"sandbox_trials_task_checksum_fkey"
```

The task checksums referenced by trials don't exist in `sandbox_tasks`. Fix: set `register_benchmark=True` in `upload_eval_results()` — this auto-registers all tasks from trial results.

### Stale Jobs That Look Running

If `squeue` shows a job but it crashed (e.g., OOM kill), check the log:
```bash
tail -50 eval/jupiter/logs/eval_<JOB_ID>.out
sacct -j <JOB_ID> --format=JobID,State,ExitCode,MaxRSS
```

The job directory may have partial results. Use `harbor jobs resume` to retry.

### RL Key vs Org Key Confusion

**Symptom**: `"Region not found"` error during RL snapshot operations.

**Cause**: Used org1/org2 key with `DAYTONA_TARGET=RL`. Only the RL key (`dtn_7ff746b0...`) has an RL region.

**Fix**: Ensure RL sbatch scripts use:
```bash
export DAYTONA_API_KEY="dtn_7ff746b032c547e741f0ef153ba7947b7d312c25711d4181423fcfe91cebb894"
```

### SSH Tunnel Failures

**Symptom**: `[proxy] Connectivity test failed` or Daytona timeouts on compute nodes.

**Fix**: Ensure `SSH_KEY` env var is set in `secrets.env`. The tunnel needs passwordless SSH to the login node (`jpbl-s01-02`).

### vLLM Server Won't Start

**Symptoms**: Health check loop times out after ~33 minutes.

Common causes:
- Model not in cache (`HF_HUB_CACHE`): pre-download via proxychains
- OOM: reduce `EVAL_GPU_MEMORY_UTIL` or use fewer GPUs
- Incompatible model: check `eval/jupiter/logs/vllm_<JOB_ID>.log`

### Snapshot Not Found Warnings

**Symptom**: Log shows `"not found (not global)"` for snapshots.

**Cause**: Snapshots weren't pre-created on the Daytona org being used.

**Fix**: Run `precreate_snapshots.py` on both orgs before launching evals.

### PYTHONPATH Pollution

The sbatch script explicitly `unset PYTHONPATH` before setting its own. If you source `jupiter.env` in your shell, it may set a PYTHONPATH with incompatible packages (e.g., numpy 2.4 breaking numba/vLLM). The sbatch handles this, but manual testing should be careful.

---

## 9. Key File Paths Reference

### Scripts

| Path | Description |
|------|-------------|
| `eval/jupiter/unified_eval_harbor.sbatch` | Main Slurm eval script |
| `eval/jupiter/dcagent_eval_config.yaml` | Harbor job config (jobs_dir, retry, agent settings) |
| `eval/jupiter/snapshot_download.py` | HF dataset download helper |
| `eval/jupiter/unified_eval_listener.py` | Auto-submit daemon (polls DB for new models) |
| `/e/scratch/jureap59/etash/run_commercial_evals.sh` | Commercial model eval (GAIA+FinanceAgent) |
| `/e/scratch/jureap59/etash/run_commercial_aider.sh` | Commercial model eval (Aider Polyglot) |
| `/e/scratch/jureap59/etash/precreate_snapshots.py` | Snapshot pre-creation script |

### Datasets

| Path | Description |
|------|-------------|
| `/e/data1/.../guha1/datasets/gaia_127` | GAIA 127-task subset |
| `/e/data1/.../guha1/datasets/gaia` | Full GAIA dataset |
| `/e/data1/.../guha1/datasets/financeagent` | FinanceAgent (50 tasks) |
| `/e/data1/.../guha1/datasets/DCAgent2_aider_polyglot` | Aider Polyglot (225 tasks) |
| `/e/data1/.../guha1/datasets/medagentbench` | MedAgentBench |

### Job Output

| Path | Description |
|------|-------------|
| `/e/data1/.../guha1/eval_jobs/` | All eval job directories |
| `eval/jupiter/logs/` | Slurm job logs and vLLM logs |
| `eval/jupiter/logs/upload_<JOB_ID>.log` | Upload logs |

### Infrastructure

| Path | Description |
|------|-------------|
| `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/` | Python 3.12 environment |
| `/e/scratch/jureap59/etash/harbor/src/` | Harbor source (editable install) |
| `/e/scratch/jureap59/guha1/pip_packages/` | Additional Python packages |
| `/e/data1/.../guha1/hub/` | Pre-downloaded HuggingFace models |
| `eval/jupiter/dcagents-leaderboard/` | DB utilities (unified_db/) |
| `~/secrets.env` | API keys (Daytona, OpenAI, HF, Supabase) |

### Harbor Config (dcagent_eval_config.yaml)

```yaml
jobs_dir: /e/data1/datasets/playground/mmlaion/shared/guha1/eval_jobs
n_attempts: 3
timeout_multiplier: 1.0
orchestrator:
  type: local
  n_concurrent_trials: 16
  quiet: false
  retry:
    max_retries: 3
    exclude_exceptions:
      - AgentTimeoutError
      - EnvironmentStartTimeoutError
      - SandboxBuildFailedError
      - VerifierTimeoutError
    wait_multiplier: 1.0
    min_wait_sec: 1.0
    max_wait_sec: 60.0
environment:
  type: daytona
  force_build: false
  delete: false
  kwargs:
    auto_snapshot: true
agents:
  - name: terminus-2
    trajectory_config:
      raw_content: true
      linear_history: true
```

Note: `n_concurrent_trials: 16` in the config is overridden by `--n-concurrent` on the CLI (128 for Slurm, 32 for commercial). The config's `exclude_exceptions` list prevents retries for non-transient errors.
