---
name: datagen-launch-iris
description: Launch, monitor, and manually clean up a trajectory-generation (datagen) job on Marin's Iris TPU cluster via the OpenThoughts-Agent entrypoint. Use when asked to start, watch, rescue, or kill a datagen/tracegen run on Iris.
---

# datagen-launch-iris

End-to-end operation of a datagen (trajectory-generation) job through
`data/cloud/launch_tracegen_iris.py`. Covers launch → monitor → manual cleanup.
For **eval** jobs use the **eval-agentic-launch-iris** skill instead.

## Required info (ask if missing)

1. `tasks` — the task source for `--tasks_input_path`: an **HF dataset id** (e.g.
   `DCAgent/exp_rpt_e2egit-v2`; the launcher `snapshot_download`s it and
   auto-explodes a `task_binary` parquet into task dirs) **or** a local tasks
   directory. Not both.
2. `slug` — short dataset name used for the job name and HF repo (e.g.
   `e2egit-v2`).
3. Operating point — default is **S1** (Qwen3.5-122B-A10B-FP8, 32k, v5p-8,
   single-host). Don't change configs unless asked.

## Prerequisites

- **Launch from a Python 3.12 env** — the iris client writes the launcher's
  `sys.version_info` into the worker's `uv sync --python`. Use the otagent conda
  env: `source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent`
  (or call `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` directly).
- **Secrets**: `source /Users/benjaminfeuer/Documents/secrets.env` (provides
  `DAYTONA_API_KEY` for the host-side snapshot pre-build, `HF_TOKEN` for upload,
  `MARIN_HMAC_*` for runai_streamer). Also pass `--secrets-env <path>` so they
  reach the worker. Do not echo secret values.
- If a launch fails with `marin-iris client is too old`, run
  `git -C /Users/benjaminfeuer/Documents/marin pull --ff-only origin main` (iris
  is an editable install from that checkout) — **not** `uv sync`.

## Launch

```bash
cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent
source /Users/benjaminfeuer/Documents/secrets.env
TS=$(date +%Y%m%d-%H%M%S)
python data/cloud/launch_tracegen_iris.py \
  --harbor_config hpc/harbor_yaml/datagen/ctx32k_verified.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_5_122b_a10b_fp8_runai_v5p8_s1.yaml \
  --tasks_input_path <HF-dataset-id | /abs/tasks/dir> \
  --tpu v5p-8 --preemptible \
  --n_concurrent 64 --n_attempts 1 --health_max_attempts 600 \
  --job_name "qwen3.5-122b-32k-<slug>-${TS}" \
  --secrets-env /Users/benjaminfeuer/Documents/secrets.env \
  --gcs-output-dir gs://marin-models-us/ot-agent \
  --upload_hf_repo penfever/<slug>-qwen3.5-122b-32k-traces \
  --no-wait
```

Flag notes:
- **`--gcs-output-dir gs://marin-models-us/ot-agent` opts OUT of the region
  pin** (the launcher would otherwise auto-pin to the region with the most v5p-8
  capacity). Use it to avoid the stuck-PENDING trap when a single region's v5p-8
  pool has collapsed — iris then places on the first free v5p-8 in any US region.
  Keep output in the **US** multi-region bucket (matches the model bucket; never
  read/write GCS cross-region).
- `ctx32k_verified.yaml` = verifier ON + the `release_trial_payloads_in_memory`
  flag (bounds worker host-RAM so heavy/repo-based datasets don't OOM the
  container). Use the 32k config with the 32k S1 engine.
- `--health_max_attempts 600` is mandatory (122B-FP8 cold compile can take ~60
  min; the default 100 ≈ 50 min kills the job before first serve).
- `--n_concurrent 64` = `max_num_seqs(64) × DP(1)`.
- Image `:tpu` at/after digest `ae085bc8` (commit c2073e0e) **auto-uploads** the
  HF repo on a clean (state-4) completion — no manual rescue needed for those.

After submit, confirm placement:
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query \
  "SELECT job_id, state FROM jobs WHERE job_id='/benjaminfeuer/<job>'" -f csv
```
state 1=PENDING, 2=starting, 3=RUNNING, 4=SUCCEEDED, 5=FAILED, 6=KILLED.

## Monitor

```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py \
  /benjaminfeuer/<job> --output /tmp/<job>_history.md --refresh
```
Read the `.json` sidecar (it paginates the full history — don't eyeball
`--tail`): `total_runtime_s`, `iris_preemption_count`, `cycles[]` (each with
`did_serve`/`time_to_first_serve_s`), `serving_summary.gen_tps`/`.running`
(n/mean/max), `non_empty_trials` / `total_trial_dirs` (productive rate),
`harbor_exception_stats`. S1 baseline ≈ 400 mean / 1115 peak gen tok/s; short-task
datasets (nl2bash, e2egit) run lower gen tok/s by nature — judge by productive
trial rate, not tok/s alone.

**Did it auto-upload?** On a state-4 job from image `ae085bc8`+, check the repo
exists before rescuing:
```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -c \
 "from huggingface_hub import HfApi; print(HfApi().dataset_info('penfever/<slug>-qwen3.5-122b-32k-traces').lastModified)"
```

## Manual cleanup

**Kill a job** (only with explicit user permission for a RUNNING/placed job):
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job kill /benjaminfeuer/<job>
```

**Rescue banked traces** (any terminal job whose repo did NOT auto-create — e.g.
killed, OOM, or pre-`ae085bc8` image). Rsync the GCS job dir local, then push:
```bash
source /Users/benjaminfeuer/Documents/secrets.env
gsutil -m rsync -r gs://marin-models-us/ot-agent/<job>/<job>/ /tmp/<job>_traces/
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/harbor/make_and_upload_trace_dataset.py \
  --job_dir /tmp/<job>_traces \
  --repo_id penfever/<slug>-qwen3.5-122b-32k-traces \
  --episodes last --filter none --skip_register
```
(`--skip_register` = upload only, no Supabase row. Repo is public.)

**Daytona snapshot cap** — if a launch fails with `SnapshotCapExceeded` on the
shared `cli` org, delete ONLY broken (`MISSING`-state) `harbor__*` snapshots
(never `ACTIVE` ones — those may belong to running jobs, yours or a teammate's):
```bash
source /Users/benjaminfeuer/Documents/secrets.env
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python - <<'PY'
import os
from hpc.snapshot_manager import _parse_org_arg, _SnapshotManager, list_snapshots
org=_parse_org_arg(f"cli={os.environ['DAYTONA_API_KEY']}")
mgr=_SnapshotManager([org]); client=mgr._client(org)
for snaps in list_snapshots([org]).values():
    for s in snaps:
        if s.name.startswith("harbor__") and s.state=="MISSING":
            client.snapshot.delete(client.snapshot.get(s.name)); print("deleted", s.name)
PY
```
Do NOT run the broad `cleanup_unused_snapshots` against the shared `cli` org — it
deletes by your task set and would remove teammates' ACTIVE snapshots.

**Stuck PENDING** (no v5p-8 capacity): report it and relaunch UNPINNED (the
`--gcs-output-dir gs://marin-models-us/ot-agent` flag above). Kill the stuck
submission first only with user permission.

## Guardrails

- NEVER stop/restart/bounce a RUNNING job or the Iris cluster without explicit
  user permission in the current thread.
- NEVER read/write GCS across regions (cost). Keep everything in the
  bucket-matched US region.
- Rescue + snapshot-MISSING cleanup are safe maintenance; killing a running job
  and broad snapshot pruning are not — confirm first.
