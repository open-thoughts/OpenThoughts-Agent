---
name: eval-agentic-launch-iris
description: Launch, monitor, and manually clean up an eval job on Marin's Iris TPU or CoreWeave H100x8 GPU cluster via the OpenThoughts-Agent entrypoint. Use when asked to start, watch, or kill a model evaluation (evalchemy / agent-harness benchmarks) on Iris.
---

# eval-agentic-launch-iris

> **📍 Iris orientation — read first.** Read the Iris **tools catalog** (`.claude/ops/iris/ops.md`) and the Iris **ops directory** (`.claude/ops/iris/` — CoreWeave GPU in `ops.md`, TPU `marin` in `ops.md`) before acting.

Launch → monitor → manual cleanup of an eval job via `eval/cloud/launch_eval_iris.py` (Iris analog of the SkyPilot `launch_eval_cloud.py`). For **datagen/tracegen** use **datagen-launch-iris** instead.

> **🚪 Iris/cloud launchers bypass `hpc.launch` entirely** — the `python -m hpc.launch --job_type eval_listener` SLURM front door does NOT apply here. The Iris launcher resolves per-model serve config from `model_config/` (via `model_config/resolver.py` + `hpc/model_config_apply.py`): merges/forwards the model's `agent_kwargs`, and applies serve intrinsics (`max_model_len` / `limit_mm` / `extra_args`) on the worker. It also applies `n_attempts` from the preset (CLI-overridable via `--n_attempts`) and **ignores** SLURM-only fields plus `tp_size` (TPU chip count), `harbor_config` (CLI-required), `agent_name` (from harbor config). It prints `[eval-iris] preset <name>: applied {…}; ignored {…}`. Precedence: explicit CLI/`--preset` > `model_config/`. Edit the source at `model_config/<org>/<slug>.yaml` (not the generated registry).

## Required info

1. `model` — model id for `--model` (HF id or GCS/served path), OR pass `--datagen_config <yaml>` (model inferred from its `engine.model`).
2. `dataset` — for standard benchmarks, use `--preset <name>` (below), which selects the dataset. Pass an explicit dataset only for a *custom* benchmark or to override a preset:
   - `--dataset <harbor slug>` — harbor resolves/snapshots it.
   - `--dataset_path <tasks dir | HF dataset id>` (mutually exclusive with `--dataset`). A bare HF id has exactly one `/`, no leading `./`,`/`,`~`; the **worker's** `run_eval.py` resolves it (`snapshot_download` + `convert_parquet_to_tasks`) — the launch host does NOT.
3. `harbor_config` — REQUIRED, an eval harbor YAML from `hpc/harbor_yaml/eval/`:
   - **`dcagent_eval_defaults.yaml` — DEFAULT.** Iris-adapted port of the eval team's canonical config (`hpc/harbor_yaml/eval/configs/dcagent_eval_config.yaml`, the SLURM listener's `EVAL_CONFIG_YAML`): terminus-2, `timeout_multiplier: 1.0`, `n_attempts: 3`, agent `max_timeout_sec: 7200`, verifier `max_timeout_sec: 14400`. Iris numbers match the eval team's SLURM numbers. Only deviation: `force_build: true` (Iris builds sandboxes at runtime).
   - `eval_ctx32k.yaml` / `eval_ctx131k.yaml` — terminus-2 with **`timeout_multiplier: 8.0`** (8GB/4GB sandbox). Extended-budget mode — only for deliberate 8× timeout. Don't use for normal reg eval.
   - `eval_openhands_ctx32k_*` / `eval_mini_swe_ctx32k.yaml` / `swe_agent_ctx32k_eval_.yaml` — alternate harnesses (OpenHands / mini-SWE / SWE-agent). Only when reproducing a paper's harness.

## Presets (`--preset`, shared with the SLURM listener)

`--preset <name>` pulls run defaults from `eval/presets/` (one YAML per preset, **same** catalog the SLURM `eval/unified_eval_listener.py` consumes). Choices: `aider, bfcl, financeagent, gaia, medagentbench, swebench, swebench_full, tb2, v1, v2`. **Precedence: explicit CLI flags ALWAYS override preset values.**

What the Iris launcher does with each preset field:
- **Applied:** `datasets[0]` → `--dataset_path` (bare HF id, resolved on the worker) when neither `--dataset` nor `--dataset_path` was passed (extra datasets skipped, logged); `n_concurrent` → `--n_concurrent` when not passed.
- **Applied (agent kwargs, mapped as the SLURM `eval/jupiter/eval_harbor.sbatch` does):** `agent_parser` → harbor `--agent-kwarg parser=<value>` (e.g. swebench → `parser=xml`) unless you passed a `parser=`; each preset `agent_kwargs` list entry → its own `--agent-kwarg key=value` (your `--agent_kwarg` with the same key overrides).
- **Thinking (NOT a preset property):** the launcher resolves the model's `agent_kwargs` from `model_config/`, so thinking IS auto-applied per-model for models carrying `agent_kwargs: [extra_body={…enable_thinking:true}]`. For a model with **no `model_config/` entry**, thinking falls back to the served model's chat-template default (Qwen3 = ON). For a default-OFF template model not in `model_config/` (e.g. Qwen3.5/3.6), pass `--agent_kwarg 'extra_body={"chat_template_kwargs":{"enable_thinking":true}}'` (the live nested form vLLM applies; a bare `enable_thinking=true` is DEAD — terminus-2 has no such param). There is no `--enable-thinking` flag.
- **Ignored (SLURM/vLLM-serve-only):** `slurm_time`, `vllm_max_retries`, `gpu_memory_util`, `sbatch_script`, `check_hf_exists`, `log_suffix`, `error_threshold`, `config_yaml`, `agent_envs`, `auto_snapshot`.

`--preset` composes with `--harbor_config` (required), `--model`, `--upload_to_database`, etc.

## Core evals

The standard/core evals are **presets** — launch by name (preset sets dataset, concurrency, parser; do not pass `--dataset*`):

| Benchmark | Command | preset sets |
|---|---|---|
| SWE-bench-verified (random 100) | `--preset swebench` | `DCAgent2/swebench-verified-random-100-folders`, n_concurrent 32, `parser=xml` |
| terminal-bench 2.0 | `--preset tb2` | `DCAgent2/terminal_bench_2`, n_concurrent 32 |

Presets do **not** set thinking — see the note above. Both require `--harbor_config hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml` (terminus-2 @ 32k, eval-team default budget, the Cat 1 "reg eval" harness per `docs/EVAL_GUIDE.md`), fit a v6e-4 for an 8B model, and should launch **with `--upload_to_database`**. For full parity with the eval team's SLURM runs, pass `--n_concurrent 128` (their CLI default). (terminal-bench 2.0 also exists as slug `--dataset terminal-bench@2.0`; prefer the preset.)

## Snapshots — eval is the exception

Eval does NOT pre-build Daytona snapshots and does NOT call `hpc/snapshot_manager.ensure_snapshots`. Eval harbor configs set `environment.force_build: true` — **harbor builds each task's sandbox at runtime on the worker**, no launch-host prebuild, no 60-snapshot cap, no `SnapshotCapExceeded`. Datagen is the opposite (`force_build: false` → pre-builds; see **datagen-launch-iris**).

**Always run eval out of the MAIN Daytona org** (`DAYTONA_API_KEY`, carried via `--secrets-env`). Do NOT use `DAYTONA_B_KEY` / `DAYTONA_RL_API_KEY` / `DAYTONA_DATA_API_KEY` (other workloads).

## Prerequisites

Launch from the **py3.12 otagent conda env**, `source "$DC_AGENT_SECRET_ENV"` (see `.claude/secret.md`; pass `--secrets-env`), and `git pull` the marin checkout if the iris client is reported too old. Harbor env defaults to **daytona** (the only sandbox backend that works on iris workers).

## Launch

```bash
cd /Users/benjaminfeuer/Documents/OpenThoughts-Agent
source /Users/benjaminfeuer/miniconda3/etc/profile.d/conda.sh && conda activate otagent
source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"
TS=$(date +%Y%m%d-%H%M%S)
python eval/cloud/launch_eval_iris.py \
  --preset <name> \                                  # e.g. swebench, tb2 — seeds dataset + concurrency + parser
  --harbor_config hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml \  # eval-team defaults (timeout_multiplier 1.0); use eval_ctx32k.yaml only for 8x budget
  --model <hf-or-gcs-model-id> \
  --tpu v6e-4 --preemptible \
  --job_name "eval-<model-slug>-<bench>-${TS}" \
  --secrets-env "$DC_AGENT_SECRET_ENV" \
  --upload_to_database \
  --no-wait
# Custom benchmark (no preset): drop --preset and pass --dataset <harbor-slug>
# or --dataset_path <tasks dir | HF id>, plus --n_concurrent <N>.
```

### CoreWeave H100x8 GPU eval (single node, `cw-us-east-02a`)

Pass `--gpu H100x8` (mutually exclusive with `--tpu`) for one CoreWeave H100x8 node. The launcher defaults to the `gpu-8x` OT-Agent image, the `cw-us-east-02a` iris config, the `datagen` extra (not `datagen-tpu`), and skips the TPU iris-serve/`patch_tpu_inference` path. `export KUBECONFIG=~/.kube/coreweave-iris-gpu` first. Single-node only — do NOT pass `--replicas > 1` (task sharding + shared multi-node vLLM not implemented for GPU eval); `--gpu` is limited to `H100x8`. Use a model known to serve on the runtime (Qwen/Qwen3-32B works).

**Daytona/OpenCode against a separately served CoreWeave model:** use native federated ingress, not the peer controller's public host. Submit the serving job through Marin with `--target-cluster cw-us-east-02a` and forward the Marin login; wait until `iris --cluster=marin endpoints list <endpoint> --exact` shows the mirrored peer endpoint, then mint the scoped URL at Marin and pass only `https://iris.oa.dev/proxy/t/<token>/<endpoint>/v1` to the eval. A token minted with `iris --cluster=cw-us-east-02a endpoints mint` is peer-signed and cannot authorize the Marin parent route; `iris-cw-us-east-02a.oa.dev` is IP-locked and not Daytona-reachable. `scripts/iris/launch_external_opencode_eval.py` is the one-command Grug profile: it submits the parent-delegated serve, waits for the mirrored ready endpoint, mints at Marin, and submits the durable-S3 Harbor eval. Its defaults use the established Grug serve topology and CoreWeave S3 root; use explicit flags only to override that profile or attach an existing endpoint.

```bash
export KUBECONFIG=~/.kube/coreweave-iris-gpu
python eval/cloud/launch_eval_iris.py \
  --preset swebench \
  --harbor_config hpc/harbor_yaml/eval/dcagent_eval_defaults.yaml \
  --model Qwen/Qwen3-32B \
  --gpu H100x8 --replicas 1 \
  --n_concurrent 3 --n_attempts 1 \
  --harbor_extra_arg=--n-tasks=3 --harbor_extra_arg=--max-retries=0 \  # small subset for fast iteration
  --job_name "eval-<slug>-cw-gpu-${TS}" \
  --secrets-env "$DC_AGENT_SECRET_ENV" \
  --upload_to_database --no-wait
```

### Flag notes

- **Supabase sync = `--upload_to_database`** (opposite of datagen's `--skip_register`). Registers result abstracts to Supabase **and** uploads traces to HF (repo auto-derived from `--job_name` when `--upload_hf_repo` omitted). Requires `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` in `--secrets-env`. Companion flags: `--upload_username` (attribution; defaults `$UPLOAD_USERNAME`/current user), `--upload_error_mode {skip_on_error,rollback_on_error}`, `--upload_forced_update`. No `--register`/`--skip_register` — sync is OFF by default, ON solely via `--upload_to_database`.
- `--upload_hf_repo` alone (no `--upload_to_database`) = **HF-only**, no Supabase.
- `--upload_hf_repo` pushes results to HF on completion (image `ae085bc8`+ wires harbor's `--export-push`); omit for local/GCS-only.
- `--tpu` defaults to **v6e-4** for eval (vs v5p-8 for S1 datagen); set per model footprint.
- `--model` is optional only when `--datagen_config` is given (model inferred); otherwise required.
- See `docs/EVAL_GUIDE.md` (benchmark/harness catalog) and `scripts/iris/EVAL_GUIDE.md`/`README.md` (eval-analysis tooling).

### Output modes

- **TPU default**: outputs rsync'd back periodically to `--local-sync-dir` while the job runs (local eval-analysis tooling sees files). Pass `--output-mode gcs` (and OMIT `--gcs-output-dir`) to write straight to a co-located **single-region** bucket (`gs://marin-us-east5/ot-agent`, …). An explicit `--gcs-output-dir gs://marin-models-us/ot-agent` opts OUT of the pin (pricier multi-region) — only for the stuck-PENDING dodge when a TPU pool has collapsed.
- **GPU default**: `--output-mode local` — Harbor writes `trace_jobs` to pod-local NVMe and `run_eval --upload_to_database` registers to Supabase + HF **in-pod** before the ephemeral pod tears down (same path TPU/SLURM use). `--upload_to_database` IS supported on GPU. For durable raw Harbor artifacts: `--output-mode s3 --s3-output-dir s3://marin-us-east-02a/tmp/ttl=7d/ot-agent/evals/<user>` (CW object store). ⚠ Prefer deriving the output dir off `marin_prefix()` (`rigging.filesystem` — auto-resolves the storage root; don't hardcode the region bucket); the literal is a fallback. Never use `s3://marin-na` (R2) — pods can't reach it.
- **GPU storage creds**: the launcher WITHHOLDS the launch host's `AWS_*/LAION_*/MARIN_HMAC_*` from the pod (can't clobber the R2 creds the `cw-us-east-02a` cluster injects via the `iris-task-env` `envFrom` Secret). Do not re-add them.

Confirm placement (same as datagen):
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query \
  "SELECT job_id, state FROM jobs WHERE job_id='/benjaminfeuer/<job>'" -f csv
```

## Monitor

Same job-agnostic analyzer as datagen:
```bash
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python \
  /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_iris_harbor_job.py \
  /benjaminfeuer/<job> --output /tmp/<job>_history.md --resync
```
For eval, the signals of interest are **completion + productive trial rate** (`non_empty_trials`/`total_trial_dirs`) and the harness exception stats, more than gen tok/s. Scores land in the synced outputs, not the analyzer sidecar:
- default mode → `--local-sync-dir` on the launch host;
- `--output-mode gcs` → under the pinned single-region bucket (e.g. `gs://marin-us-east5/ot-agent/<job>/`; resolve with `python -m hpc.iris.job_output_resolver <job> --cluster …/marin.yaml`).

Per-task progress / resume helpers: `scripts/iris/check_progress.py` and `check_resume_needed.py`.

## Manual cleanup

**Kill** (only with explicit user permission for a RUNNING job):
```bash
/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job kill /benjaminfeuer/<job>
```

**Recover partial results**: outputs are already on the launch host (`--local-sync-dir`) or in GCS (`--output-mode gcs`). To re-pull a GCS job dir, resolve the recorded output prefix first (never hardcode): `OUT=$(python -m hpc.iris.job_output_resolver <job> --cluster …/marin.yaml)` then `gsutil -m rsync -r "$OUT/<job>/" /tmp/<job>_eval/`. If HF upload didn't fire and you need traces on the Hub, use the same `make_and_upload_trace_dataset.py` recipe as **datagen-launch-iris** against the local job dir.

**Daytona snapshot cap**: N/A for eval (no pre-build, no `ensure_snapshots`, eval configs use `force_build: true`). If you see `SnapshotCapExceeded`, you're on the wrong (datagen) path or wrong harbor config.

**Stuck PENDING**: relaunch with `--output-mode gcs --gcs-output-dir gs://marin-models-us/ot-agent` (unpinned — deliberate override drops the single-region pin so iris places on any free TPU in the US). Kill the stuck submission first only with user permission.

## Guardrails

- NEVER stop/restart/bounce a RUNNING job or the Iris cluster without explicit user permission in the current thread.
- NEVER read/write GCS across regions. Keep outputs in the US bucket.
- ALWAYS run eval out of the MAIN Daytona org (`DAYTONA_API_KEY`) — never the B/RL/DATA orgs. Eval builds sandboxes at runtime (`force_build: true`); it does not pre-build or call `ensure_snapshots`.
- Match `--harbor_config` to the model's context window and the benchmark's harness (plain vs OpenHands/mini-SWE/SWE-agent) — a mismatch fails at runtime.
