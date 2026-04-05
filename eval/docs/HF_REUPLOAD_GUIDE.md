# HF Trace Re-upload & Subagent Trace Fix

Guide for porting the HF upload fix and re-upload tooling to another cluster.

---

## Problem 1: HF uploads fail with XET permission denied

### Symptom
All HuggingFace result uploads fail with:
```
Permission denied (os error 13) at path "<HF_HOME>/xet/.../staging/shard-session/.tmpXXX"
```
DB uploads succeed (the pipeline uses `skip_on_error` mode), but HF trace repos are never created.

### Root cause
The sbatch sets `HF_HOME` to a shared read-only cache (for model/dataset downloads). HuggingFace's XET upload layer derives its staging directory from `HF_HOME`, landing at `<HF_HOME>/xet/`. If another user created that `xet/` subtree, your user can't write temp files there.

### Fix
Set `HF_XET_CACHE` to a **user-writable** location, separate from `HF_HOME`. Add this to your sbatch **after** the `HF_HOME`/`HF_HUB_CACHE` exports:

```bash
# XET staging cache — must be user-writable (shared HF_HOME/xet/ has permission issues)
export HF_XET_CACHE="/your/scratch/${USER}/hf_xet_cache"
mkdir -p "$HF_XET_CACHE"
```

**On Jupiter**, the change was made in `eval/jupiter/unified_eval_harbor.sbatch` at line ~138, right after:
```bash
export HF_HUB_CACHE="/e/data1/datasets/playground/ot/hf_hub"
export HF_HOME="/e/data1/datasets/playground/ot/hf_hub"
export HF_CACHE_DIR="$HF_HUB_CACHE"
```

Already-running jobs are NOT affected by this fix — they need the re-upload script below.

---

## Problem 2: Subagent traces were being uploaded (bloated HF datasets)

### Symptom
A job with 300 trials produces thousands of HF dataset rows instead of 300. The `Map:` progress lines show many batches of varying sizes (300, 238, 239, 240, 230, ...) instead of a single batch of ~300.

### Root cause
The upload call chain is:
```
sbatch upload section
  → upload_eval_results()           [database/unified_db/utils.py:4166]
    → upload_traces_to_hf()         [database/unified_db/utils.py:3909]
      → export_traces()             [harbor/utils/traces_utils.py:1422]
```

`upload_eval_results` has `hf_export_subagents: bool = False` (correct default). It passes this to `upload_traces_to_hf(export_subagents=...)`. **But** `upload_traces_to_hf` was NOT forwarding `export_subagents` to `export_traces()` — the parameter was silently dropped. So `export_traces()` always used its own default of `export_subagents=True`.

### Fix
In `database/unified_db/utils.py`, in the `upload_traces_to_hf` function (~line 4014), add `export_subagents` to the `export_traces()` call:

```python
# BEFORE (broken):
dataset = export_traces(
    root=job_dir,
    recursive=True,
    episodes=episodes,
    to_sharegpt=False,
    repo_id=None,
    push=False,
    verbose=verbose,
    success_filter=success_filter,
    include_verifier_output=include_verifier_output,
)

# AFTER (fixed):
dataset = export_traces(
    root=job_dir,
    recursive=True,
    episodes=episodes,
    to_sharegpt=False,
    repo_id=None,
    push=False,
    verbose=verbose,
    success_filter=success_filter,
    include_verifier_output=include_verifier_output,
    export_subagents=export_subagents,        # <-- THIS WAS MISSING
)
```

The `upload_traces_to_hf` function already accepts `export_subagents` as a parameter (line 3918) — it just wasn't passing it through. No signature changes needed.

### Verification
After the fix, a 300-trial job should produce exactly ~300 rows in the HF dataset (one per trial, last episode only), not thousands.

---

## Re-upload script: `eval/jupiter/reupload_hf.py`

This script re-uploads HF traces for jobs that already completed but failed the HF upload step. It also patches the Supabase DB record with the new HF URL.

### What it does (per job)
1. Reads `meta.env` from the job's run directory to get `RUN_TAG`, `DB_JOB_ID`, `MODEL`, etc.
2. Skips early if no `DB_JOB_ID` (no DB record to patch).
3. Calls `harbor.utils.traces_utils.export_traces()` **directly** (not through `upload_traces_to_hf`) with:
   - `episodes="last"` — only the final episode per trial
   - `export_subagents=False` — no subagent traces
   - `include_verifier_output=True`
4. Creates HF repo via `huggingface_hub.create_repo()` and pushes via `dataset.push_to_hub()`.
5. Patches `sandbox_jobs.hf_traces_link` in Supabase with the new HF URL.

### HF repo naming
- Format: `DCAgent2/<run_tag>-<random_8hex>`
- The random suffix prevents overwriting previous uploads for the same model/benchmark.
- Name part is sanitized to fit HF's 96-char limit (special chars → hyphens, truncation with sha1 tail if needed).

### Key dependencies / imports
```python
from database.unified_db.utils import get_supabase_client, load_supabase_keys
from harbor.utils.traces_utils import export_traces
from huggingface_hub import create_repo
```

### Environment setup
The script auto-configures these if not already set:
- `HF_XET_CACHE` → `<scratch>/<USER>/hf_xet_cache` (the permission fix)
- `DC_AGENT_SECRET_ENV` → `~/secrets.env` (so `load_supabase_keys()` can find credentials)
- `SUPABASE_KEY` → aliased to `SUPABASE_ANON_KEY` (Jupiter's secrets.env uses `SUPABASE_KEY`)

### Usage
```bash
PYTHON="/path/to/otagent/bin/python"

# Single job by SLURM ID
$PYTHON eval/jupiter/reupload_hf.py --job-ids 279050

# Multiple jobs
$PYTHON eval/jupiter/reupload_hf.py --job-ids 279050 279051 279052

# From a file of SLURM IDs
$PYTHON eval/jupiter/reupload_hf.py --job-ids-file eval/jupiter/lists/failed_uploads.txt

# Direct run directory paths
$PYTHON eval/jupiter/reupload_hf.py --run-dirs /path/to/eval_jobs/run_dir_name

# Scan ALL run dirs for missing HF uploads (skips dirs without DB_JOB_ID)
$PYTHON eval/jupiter/reupload_hf.py --scan-all

# Dry run (preview without uploading)
$PYTHON eval/jupiter/reupload_hf.py --scan-all --dry-run
```

### How it finds run directories from SLURM job IDs
1. **Strategy 1**: Parses `eval/jupiter/logs/eval_<JOB_ID>.out` for the `Run dir: <path>` line.
2. **Strategy 2**: Scans all `<EVAL_JOBS_DIR>/*/meta.env` files for matching `SLURM_JOB_ID`.

### Supabase details
- **Table**: `sandbox_jobs` (NOT `eval_jobs` — that doesn't exist)
- **Column**: `hf_traces_link` (string, stores the full HF dataset URL)
- **Client**: `get_supabase_client()` from `database.unified_db.utils` (NOT `_get_supabase_client`)
- **Credentials loader**: `load_supabase_keys()` — reads from file at `DC_AGENT_SECRET_ENV`
- Required env vars after loading: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`

### Job directory structure (what the script reads)
```
<EVAL_JOBS_DIR>/<run_tag>/
├── meta.env          # MODEL, REPO_ID, DB_JOB_ID, RUN_TAG, BENCHMARK_NAME, etc.
├── config.json       # Harbor job config
├── result.json       # Job result (status, metrics, exception_stats)
└── <task_name>/      # One per trial
    ├── config.json
    ├── result.json
    └── agent/
        ├── episode-0/
        ├── episode-1/   # export_traces with episodes="last" takes only the highest-numbered
        └── trajectory.json
```

### Porting to another cluster
To adapt `reupload_hf.py` for a different cluster, change:
1. `DEFAULT_EVAL_JOBS_DIR` — path to your eval jobs output directory
2. `DEFAULT_LOG_DIR` — path to your SLURM log directory
3. `SECRETS_ENV` — path to your secrets file
4. `HARBOR_SRC` — path to harbor source (for `export_traces` import)
5. The `HF_XET_CACHE` default path in `main()` — your user-writable scratch
6. HF org in `reupload_single()` — currently hardcoded as `DCAgent2/`

### Gotchas discovered during development
| Issue | Wrong | Correct |
|-------|-------|---------|
| Supabase table name | `eval_jobs` | `sandbox_jobs` |
| Supabase client function | `_get_supabase_client()` | `get_supabase_client()` |
| Credentials env var | Not set → warning + empty client | Set `DC_AGENT_SECRET_ENV=~/secrets.env` before calling `load_supabase_keys()` |
| Supabase key name alias | `SUPABASE_KEY` in secrets.env | Must alias to `SUPABASE_ANON_KEY` if secrets.env uses the short name |
| `export_subagents` not forwarded | `upload_traces_to_hf` silently ignored it | Must pass `export_subagents=export_subagents` to `export_traces()` |
| HF repo name length | No limit enforced → API rejects | Cap name part at 96 chars (HF limit), truncate with sha1 tail |
