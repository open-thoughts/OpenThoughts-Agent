---
name: eval-agentic-launch-iris
description: Launch, monitor, and manually clean up an eval job on Marin's Iris TPU cluster via the OpenThoughts-Agent entrypoint. Use when asked to start, watch, or kill a model evaluation (evalchemy / agent-harness benchmarks) on Iris.
---

# eval-agentic-launch-iris

End-to-end operation of an eval job through `eval/cloud/launch_eval_iris.py`
(the Iris analog of the SkyPilot `launch_eval_cloud.py` — same arg names/flow).
Covers launch → monitor → manual cleanup. For **datagen/tracegen** jobs use the
**datagen-launch-iris** skill instead.

## Required info (ask if missing)

1. `model` — model id for `--model` (HF id or a GCS/served path). Alternatively
   pass `--datagen_config <yaml>` and the launcher infers `--model` from its
   `engine.model`.
2. `dataset` — **for standard benchmarks, don't set this directly: use
   `--preset <name>` (see "Presets" below), which selects the dataset for you.**
   Pass an explicit dataset only for a *custom* benchmark or to override a
   preset: `--dataset <harbor slug>` **or** `--dataset_path <tasks dir | HF
   dataset id>` (mutually exclusive).
   - `--dataset` is a harbor-registry slug; harbor resolves/snapshots it itself.
   - `--dataset_path` is a local tasks dir **or** a bare HF dataset id of
     pre-built task folders (exactly one `/`, no leading `./`,`/`,`~`). The
     **worker's** `run_eval.py` resolves the HF id (`snapshot_download` +
     `convert_parquet_to_tasks`) — the launch host does NOT (see "Snapshots").
3. `harbor_config` — an eval harbor YAML from `hpc/harbor_yaml/eval/` (REQUIRED).
   - **`dcagent_eval_defaults.yaml` — USE THIS BY DEFAULT.** It is an exact,
     Iris-adapted port of the eval team's canonical config
     (`eval/configs/dcagent_eval_config.yaml`, the SLURM listener's default
     `EVAL_CONFIG_YAML`): terminus-2, `timeout_multiplier: 1.0`, `n_attempts: 3`,
     agent `max_timeout_sec: 7200`, verifier `max_timeout_sec: 14400`. Using it
     means our Iris numbers match the eval team's SLURM numbers. (The only
     deviation from their file is `force_build: true` — required because Iris
     builds sandboxes at runtime instead of using their prebuilt snapshot.)
   - `eval_ctx32k.yaml` / `eval_ctx131k.yaml` — terminus-2, but with
     **`timeout_multiplier: 8.0`** (and an 8GB/4GB sandbox override). This is the
     eval team's *extended-budget* mode, NOT the default — only use it when you
     deliberately want 8× the per-task timeout. For normal reg eval, prefer
     `dcagent_eval_defaults.yaml` so you don't silently inflate timeouts.
   - `eval_openhands_ctx32k_*` / `eval_mini_swe_ctx32k.yaml` /
     `swe_agent_ctx32k_eval_.yaml` — alternate agent harnesses (OpenHands /
     mini-SWE / SWE-agent). Only use these when reproducing a paper's preferred
     harness; the default reg eval for tb2/swebench is terminus-2.

## Presets (`--preset`, shared with the SLURM listener)

`--preset <name>` pulls run defaults from the shared catalog in
`eval/presets/` (one YAML per preset, the **same** catalog the SLURM
orchestrator `eval/unified_eval_listener.py` consumes). Choices:
`aider, bfcl, financeagent, gaia, medagentbench, swebench, swebench_full,
tb2, v1, v2`.

**Precedence: explicit CLI flags ALWAYS override preset values.** A preset only
fills in what you didn't pass.

What the Iris launcher does with each preset field:

- **Applied (Iris analogs):**
  - `datasets[0]` → `--dataset_path` (bare HF id, resolved on the worker), only
    when you gave neither `--dataset` nor `--dataset_path`. Extra datasets in the
    list are skipped (logged).
  - `n_concurrent` → `--n_concurrent`, only when you didn't pass `--n_concurrent`.
- **Applied (result-affecting agent kwargs — mapped exactly as the SLURM
  `unified_eval_harbor.sbatch` does):**
  - `agent_parser` → harbor `--agent-kwarg parser=<value>` (e.g. `swebench` →
    `parser=xml`), unless you already passed a `parser=` `--agent_kwarg`.
  - `enable_thinking: true` → harbor `--agent-kwarg enable_thinking=true`, unless
    you already passed an `enable_thinking=` `--agent_kwarg`.
- **Ignored (SLURM / vLLM-serve-only, no Iris analog):** `slurm_time`,
  `vllm_max_retries`, `gpu_memory_util`, `sbatch_script`, `check_hf_exists`,
  `log_suffix`, `error_threshold`, `config_yaml`, `agent_envs`, `auto_snapshot`.

The launcher prints a one-line `[eval-iris] preset <name>: applied {...};
ignored {...}` so the split is transparent. `--preset` composes with
`--harbor_config` (still required), `--model`, `--upload_to_database`, etc.;
e.g. `--preset swebench --harbor_config hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml
--model <id>` evaluates the swebench-verified-100 set with the xml parser.

## Core evals

The standard/core eval benchmarks are **presets** — launch them by name (the
preset sets the dataset, concurrency, and agent parser; you do not pass
`--dataset*`):

| Benchmark | Command | preset sets |
|---|---|---|
| SWE-bench-verified (random 100) | `--preset swebench` | `DCAgent2/swebench-verified-random-100-folders`, n_concurrent 32, `parser=xml`, enable_thinking |
| terminal-bench 2.0 | `--preset tb2` | `DCAgent2/terminal_bench_2`, n_concurrent 32, enable_thinking |

Both still require `--harbor_config hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml`
(terminus-2 @ 32k, eval-team default budget — the Cat 1 "reg eval" harness per
`eval/EVAL_GUIDE.md`), fit a v6e-4 for an 8B model, and should be launched
**with `--upload_to_database`** (Supabase sync, below). For full parity with the
eval team's SLURM runs, pass `--n_concurrent 128` (their CLI default).

(terminal-bench 2.0 also exists as the harbor-registry slug
`--dataset terminal-bench@2.0` — the same benchmark in slug form; the `tb2`
preset uses the HF task-folder form. Prefer the preset.)

## Snapshots — eval is the exception (no pre-build, MAIN Daytona org)

**Eval does NOT pre-build Daytona snapshots and does NOT touch
`hpc/snapshot_manager.ensure_snapshots`** (the shared-org 60-snapshot cap that
datagen lives under). The eval harbor configs in `hpc/harbor_yaml/eval/` all set
`environment.force_build: true`, so **harbor builds each task's sandbox at
runtime on the worker** — no launch-host prebuild, no cap, no `MISSING`-snapshot
cleanup dance. This is deliberate: agent benchmarks legitimately need one env
per task (swebench-verified-100 = 100 envs), which the 60-cap is wrong for.
Datagen is the opposite (`force_build: false` → it *does* pre-build; see
**datagen-launch-iris**).

**Always run eval out of the MAIN Daytona org.** The worker builds in the org
keyed by **`DAYTONA_API_KEY`** (the main org), carried via `--secrets-env`. Do
NOT point eval at the `DAYTONA_B_KEY` / `DAYTONA_RL_API_KEY` / `DAYTONA_DATA_API_KEY`
orgs — those are for other workloads. So: no `SnapshotCapExceeded` handling is
needed for eval, and there is nothing to prune on a shared snapshot org.

## Prerequisites

Same as datagen: launch from the **py3.12 otagent conda env**, `source
/Users/benjaminfeuer/Documents/secrets.env` (and pass `--secrets-env`), and
`git pull` the marin checkout if the iris client is reported too old. Harbor env
defaults to **daytona** (the only sandbox backend that works on iris workers).

## Launch

```bash
cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent
source /Users/benjaminfeuer/Documents/secrets.env
TS=$(date +%Y%m%d-%H%M%S)
python eval/cloud/launch_eval_iris.py \
  --preset <name> \                                  # e.g. swebench, tb2 — seeds dataset + concurrency + parser
  --harbor_config hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml \  # eval-team defaults (timeout_multiplier 1.0); use eval_ctx32k.yaml only for 8x budget
  --model <hf-or-gcs-model-id> \
  --tpu v6e-4 --preemptible \
  --job_name "eval-<model-slug>-<bench>-${TS}" \
  --secrets-env /Users/benjaminfeuer/Documents/secrets.env \
  --upload_to_database \
  --no-wait
# Custom benchmark (no preset): drop --preset and pass --dataset <harbor-slug>
# or --dataset_path <tasks dir | HF id>, plus --n_concurrent <N>.
```

Flag notes:
- **Supabase sync = `--upload_to_database`** (the opposite of datagen's
  `--skip_register`). It registers result abstracts to Supabase **and** uploads
  traces to HuggingFace; the HF repo is auto-derived from `--job_name` when
  `--upload_hf_repo` is omitted. Requires `SUPABASE_URL` +
  `SUPABASE_SERVICE_ROLE_KEY` in `--secrets-env` (carried automatically).
  Companion flags: `--upload_username` (attribution; defaults to
  `$UPLOAD_USERNAME`/current user), `--upload_error_mode
  {skip_on_error,rollback_on_error}`, `--upload_forced_update` (overwrite
  existing rows). There is no `--register`/`--skip_register` here — sync is OFF
  by default and turned ON solely by passing `--upload_to_database`.
- `--upload_hf_repo` alone (no `--upload_to_database`) does **HF-only** upload,
  no Supabase. Pass an explicit repo if you want a fixed destination.
- `--tpu` defaults to **v6e-4** for eval (vs v5p-8 for S1 datagen); set per the
  model's footprint.
- **Output mode**: by default eval outputs are **rsync'd back periodically to
  `--local-sync-dir`** while the job runs (so local eval-analysis tooling sees
  files). Pass `--output-mode gcs --gcs-output-dir gs://marin-models-us/ot-agent`
  to write straight to GCS instead (and, as with datagen, this opts out of the
  region pin — use it to dodge stuck-PENDING when a TPU pool has collapsed; keep
  the bucket in the US region).
- `--upload_hf_repo` pushes results to HF on completion (image `ae085bc8`+ wires
  harbor's `--export-push`); omit it if you only want local/GCS outputs.
- `--model` is optional only when `--datagen_config` is given (model inferred);
  otherwise it's required.
- See `eval/EVAL_GUIDE.md` for the benchmark/harness catalog and
  `scripts/iris/EVAL_GUIDE.md`/`README.md` for eval-analysis tooling.

Confirm placement (same as datagen):
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query \
  "SELECT job_id, state FROM jobs WHERE job_id='/benjaminfeuer/<job>'" -f csv
```

## Monitor

Same analyzer as datagen (it's job-agnostic):
```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py \
  /benjaminfeuer/<job> --output /tmp/<job>_history.md --refresh
```
For eval, the signal of interest is **completion + productive trial rate**
(`non_empty_trials`/`total_trial_dirs`) and the harness exception stats, more
than gen tok/s. Scores land in the synced outputs, not the analyzer sidecar:
- default mode → `--local-sync-dir` on the launch host (point eval-analysis
  tooling at it);
- `--output-mode gcs` → under `gs://marin-models-us/ot-agent/<job>/`.

Per-task progress / resume helpers live in `scripts/iris/check_progress.py` and
`check_resume_needed.py`.

## Manual cleanup

**Kill** (only with explicit user permission for a RUNNING job):
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job kill /benjaminfeuer/<job>
```

**Recover partial results**: eval outputs are already on the launch host
(`--local-sync-dir`) or in GCS (`--output-mode gcs`). To re-pull a GCS job dir:
`gsutil -m rsync -r gs://marin-models-us/ot-agent/<job>/<job>/ /tmp/<job>_eval/`.
If HF upload didn't fire (pre-`ae085bc8` image or non-state-4 exit) and you need
the traces on the Hub, use the same `make_and_upload_trace_dataset.py` recipe as
in **datagen-launch-iris** against the local job dir.

**Daytona snapshot cap**: N/A for eval — eval does not pre-build snapshots or
call `ensure_snapshots` (see "Snapshots" above), so `SnapshotCapExceeded` does
not arise here. If you somehow see it, you're on the wrong (datagen) path or the
wrong harbor config (eval configs use `force_build: true`).

**Stuck PENDING**: relaunch with `--output-mode gcs --gcs-output-dir
gs://marin-models-us/ot-agent` (unpinned) so iris places on any free TPU in the
US region. Kill the stuck submission first only with user permission.

## Guardrails

- NEVER stop/restart/bounce a RUNNING job or the Iris cluster without explicit
  user permission in the current thread.
- NEVER read/write GCS across regions. Keep outputs in the US bucket.
- ALWAYS run eval out of the MAIN Daytona org (`DAYTONA_API_KEY`) — never the
  B/RL/DATA orgs. Eval builds sandboxes at runtime (`force_build: true`); it does
  not pre-build or call `ensure_snapshots`.
- Match `--harbor_config` to the model's context window and the benchmark's
  harness (plain vs OpenHands/mini-SWE/SWE-agent) — a mismatch fails at runtime.
