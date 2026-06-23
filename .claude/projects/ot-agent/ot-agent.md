# OpenThoughts-Agent — codebase, branches & launchers

Reference map of the OT-Agent repo: the git branch model, the companion repos, and every launcher
(HPC SFT/RL/datagen/eval/consolidate/pretokenize, plus the separate Cloud and Iris/TPU entry points).
Written 2026-06-14 from the live `penfever/working` tree — re-verify function/flag names against
`hpc/arguments.py` + the `*_launch_utils.py` modules if acting on this months later.

Repo: `open-thoughts/OpenThoughts-Agent`. Local clone: `/Users/benjaminfeuer/Documents/OpenThoughts-Agent`.

---

## 1. Branches & companion repos

- **`penfever/working` = the source of truth.** Day-to-day work + all clusters track this branch (origin `open-thoughts/OpenThoughts-Agent`). `main` exists but lags. Sync discipline: commit+push from the laptop, then `git pull` on the cluster (editable installs → live immediately, no reinstall). Many stale feature branches exist (`richard/*`, `rz/*`, `marianna/*`, `charlie/*`, etc.) — ignore unless explicitly referenced.
- **Companion repos** (also editable-installed on every cluster, edited locally + synced via git — never patched on the cluster):
  - **Harbor** `/Users/benjaminfeuer/Documents/harbor` — agent framework, environment backends (daytona/docker/modal), terminus-2 agent. SoT branch `penfever/working`, remote `marin` = `marin-community/harbor`.
  - **MarinSkyRL** `/Users/benjaminfeuer/Documents/MarinSkyRL` — the RL training framework (trainer, terminal_bench generator). SoT = `marin-community/MarinSkyRL` `penfever/working`. Full framework facts/gotchas: `.claude/projects/marinskyrl/marinskyrl.md`.
- **Editable-install model:** Harbor, MarinSkyRL, and OT-Agent are all `pip install -e .` on every cluster — after `git pull` the new source is active immediately.

### Top-level layout
`hpc/` (the launcher system — see below) · `sft/` (LLaMA-Factory configs + submodule) · `rl/` · `data/` (datagen pipelines) · `eval/` (eval listeners + per-cluster sbatch) · `database/` (Supabase unified DB) · `scripts/` (db/datagen/harbor/vllm utilities) · `prm/` (process reward models, e.g. `teacher_hint.py`) · `baselines/` (Sera/CoderForge) · `docker/` · `paper/`.

---

## 2. The unified HPC launcher — `hpc/launch.py`

**Single entry point** for all SLURM clusters: `python -m hpc.launch --job_type <type> …`. It detects the
cluster (`hpc.py:detect_hpc()`), loads the cluster dotenv, parses args (`arguments.py` + `arg_groups.py`),
and dispatches on a `JobType` enum (`arguments.py:8`) in `_main_dispatch()` (`launch.py:587`):

| `--job_type` | Launcher module | sbatch template |
|---|---|---|
| `sft` | `sft_launch_utils.py` | `sbatch_sft/universal_sft.sbatch` |
| `sft_mca` | `sft_launch_utils.py` (Megatron-Core-Adapter path, `USE_MCA=1`) | `sbatch_sft_mca/*.sbatch` |
| `pretokenize` | `pretokenize_launch_utils.py` | (reuses SFT template, 1 node) |
| `datagen` | `datagen_launch_utils.py` (`launch_datagen_job_v2`) | `sbatch_data/universal_{taskgen,tracegen}.sbatch` |
| `eval` | `eval_launch_utils.py` (`launch_eval_job_v2`) | `sbatch_eval/*.sbatch` |
| `consolidate` | `consolidate_launch_utils.py` | `sbatch_consolidate/*.sbatch` |
| `rl` | `rl_launch_utils.py` (`launch_rl_job`) | `sbatch_rl/universal_rl.sbatch` |

Every launcher follows the same 3-step shape: **(1) validate + resolve** (configs, model/dataset
pre-download for no-internet clusters), **(2) render the sbatch** from its template, **(3) submit** via
`launch_sbatch()` — optionally chaining `afterany:`/`afterok:` dependencies for restart chains. `--dry_run`
writes the sbatch without submitting. **Cloud and Iris are NOT job_types** — they're separate CLIs (§7–8).

Common flags across job types: `--job_name`, `--num_nodes`, `--time_limit`, `--max_restarts` (restart
chain), `--dependency afterany:<jid>`, `--experiments_dir`, `--dry_run`.

---

## 3. SFT launcher (LLaMA-Factory)

`sft_launch_utils.py` → `submit_sft_job(...)`. Builds the training config YAML (base YAML + CLI overrides via
`construct_config_yaml()`), pre-downloads dataset/model on the login node, renders
`sbatch_sft/universal_sft.sbatch`, and submits. Configs live under `sft/` (e.g. `sft/lf_configs/qwen3/`,
`sft/lf_configs/qwen3_5/`).

- **`SFT_MCA`** is the Megatron-Core-Adapter variant (sets `USE_MCA=1`, uses the `sbatch_sft_mca/` template, e.g. Vista) — use for the MCA training path; plain `sft` is the standard torchrun/accelerate path.
- **`pretokenize`** (`pretokenize_launch_utils.py:schedule_pretokenize`) runs a single-node tokenize-only pass building the Arrow cache; the SFT training job then chains `afterok:<pretok_jid>`. Trigger standalone (`--job_type pretokenize`) or inline (`--pretokenize true`).
- **Key flags:** `--train_config_path` (base YAML), `--model_name_or_path`, `--dataset` / `--dataset_dir`, `--hub_model_id` (HF upload target; auto-derives if unset), `--num_train_epochs`, `--global_batch_size` (launcher computes grad-accum from nodes×gpus), `--cutoff_len`, `--deepspeed`, `--num_nodes`, `--push_to_hub`.
- **Harbor-dataset tags (load-bearing):** Harbor/DCAgent datasets use `role`/`content` with `user`/`assistant`; LLaMA-Factory defaults to `from`/`value` with `human`/`gpt`. Always pass `--role_tag role --user_tag user --assistant_tag assistant --content_tag content` or the thinking preprocessor finds 0 assistant messages → garbage.
- Launch/cleanup specifics per cluster: skills `sft-launch-jupiter` / `sft-launch-leonardo` / `sft-job-cleanup`; runtime/env choice (8B vs 32B-ZeRO3 vs Qwen3.5): `.claude/ops/jupiter/ENVIRONMENT_MAP.md`.

---

## 4. RL launcher (SkyRL)

`rl_launch_utils.py` → `launch_rl_job(exp_args, hpc)` (config parsing in `rl_config_utils.py`). Orchestrates
a SkyRL GRPO/RLOO run: resolves the SkyRL Hydra config, **extracts tasks** from the HF dataset into a local
task dir, **pre-builds Daytona snapshots** on the login node (via `snapshot_manager.py`, to avoid a
multi-node race), selects the **runtime** (venv default vs Apptainer SIF opt-in), starts the Ray cluster,
renders `sbatch_rl/universal_rl.sbatch`, and submits with restart chaining. Configs under
`hpc/skyrl_yaml/{jupiter,leonardo,…}/`.

- **Key flags:** `--rl_config` (SkyRL Hydra YAML under `hpc/skyrl_yaml/`), `--skyrl_override key=value` (repeatable Hydra overrides), `--train_data`/`--val_data` (JSON list of HF repos), `--model`, `--num_nodes` (**required** — defaults to 1 otherwise; = total_gpus/gpus_per_node), `--tensor_parallel_size`, `--max_restarts`, `--rl_use_conda` (+ `--rl_conda_env`, Perlmutter), `--rl_container_sif` (+ `--rl_container_binds`, the MoE/80B SIF path).
- **Runtime selection:** dense FSDP2 (8B/32B) → the RL **venv**; Megatron/MoE/80B → the **SIF**. The launcher resolves `RL_ENV_DIR` / container vars in `universal_rl.sbatch`. To know which a job used, read its rendered `experiments/<job>/sbatch/*.sbatch`.
- **Concurrency / caps:** keep ≤6 RUNNING RL jobs per cluster (`.claude/projects/daytona/daytona.md`); a3 series is CONCLUDED (no relaunch). Launch/monitor/cleanup: skills `rl-agentic-launch-jupiter`, `rl-standard-launch-leonardo`, `rl-job-cleanup`, `monitor-job-tables`.

---

## 5. Datagen launcher (Harbor + vLLM)

`datagen_launch_utils.py` → `launch_datagen_job_v2(exp_args, hpc)` (vLLM/Harbor config in
`datagen_config_utils.py`, Harbor CLI in `harbor_utils.py`). Two stages, either or both:
1. **Task gen** (`--enable_task_gen`): runs `--datagen_script` (e.g. `data/<set>/generate.py`) → pushes structured tasks to `--datagen_target_repo`.
2. **Trace gen** (`--enable_trace_gen`, auto-on when `--tasks_input_path` is given): spins up a **vLLM server** (when the engine is `vllm_local`) and runs **Harbor** agent rollouts over the task set, recording traces → `--trace_target_repo`. Backend = Daytona (default) / docker / modal.

Stage 2 chains `afterok:` on stage 1 when both run. Templates: `sbatch_data/universal_taskgen.sbatch`,
`universal_tracegen.sbatch`.

- **Key flags:** `--datagen_config` (vLLM serve YAML under `hpc/datagen_yaml/`), `--trace_harbor_config`/`--harbor_config` (Harbor YAML under `hpc/harbor_yaml/datagen/`, match the model's `max_model_len`), `--tasks_input_path`, `--trace_target_repo`, `--trace-n-concurrent` (effective per-job ceiling ~128 — see daytona doc), `--num_nodes` (= the datagen YAML's `data_parallel_size`), `--daytona_api_key "$DAYTONA_DATA_API_KEY"` (mandatory), sandbox sizing (`--sandbox_cpu/_memory_gb/_disk_gb`).
- Two-step launch flow + the inode rule (extract tasks to `/e/data1/.../ot/tasks/`, NOT `$SCRATCH`) and MiniMax auto-advance: skills `datagen-launch`, `datagen-job-cleanup`, `datagen-reduce-dataset-snapshots`.

---

## 6. Agentic eval launcher

Two ways to run agentic eval:
- **Launcher-driven** — `eval_launch_utils.py:launch_eval_job_v2`: one sbatch that serves vLLM + runs Harbor over a benchmark, then uploads results (Supabase) + traces (HF). Template `sbatch_eval/*.sbatch`. Good for one-off / small jobs. Key flags mirror datagen trace-gen: `--model`, `--trace_agent_name`, `--harbor_dataset`, `--trace_harbor_config`, `--trace-n-concurrent`, `--trace-n-attempts`, `--upload_to_database`. On no-internet clusters a **Pinggy** SSH tunnel exposes the debugger (`hpc/pinggy_utils.py`; `--pinggy_*` flags).
- **Listener-driven** (the production sweep path) — `eval/jupiter/unified_eval_listener.py`: a polling listener that pulls a model list, submits eval sbatches, harvests `result.json`, and is resume-friendly. Driven with `--preset {tb2,v2,dev,swebench,bfcl,aider}`, `--priority-file`, `--require-priority-list`, `--n-concurrent 48`, `--harbor-config …`, `--batch-size`, `--pre-download`, `--once`. This is what the eval skills use.
- Defaults, the 15-min infra check, pinggy banks, and cleanup: skills `eval-agentic-launch` / `eval-agentic-cleanup`; standard (lm-eval/Delphi) eval: `eval-standard-launch` / `eval-standard-cleanup`.

---

## 7. Cloud launcher (SkyPilot) — separate entry

`cloud_launch_utils.py` (+ `cloud_providers.py`, `cloud_sync_utils.py`). **Not** a `--job_type`: it's a set
of argparse-based `CloudLauncher` subclasses (RL/SFT/datagen) that submit via **SkyPilot** instead of SLURM
— spinning up VMs/K8s pods on demand, file-mounting instead of using a shared FS, no sbatch. Providers
(per `cloud_providers.py`): **gcp, aws, azure, lambda, vast, runpod, cudo, paperspace, fluidstack,
kubernetes** (with aliases like `google`/`amazon`/`k8s`). Cloud-specific flags: `--cloud_provider`,
`--region`/`--zone`, `--cloud_accelerator` (e.g. `A100:1`), spot/retry knobs; the task-level flags
(model/dataset/training) are shared with the HPC launchers. `cloud_sync_utils.py` periodically pulls
remote outputs back to local.

---

## 8. Iris launcher (Google Cloud TPU) — separate entry

`iris_launch_utils.py` (+ `iris_fetch_daemon.py`, `iris_job_registry.py`). Submits containerized workloads
to **Marin Iris**, Google Cloud's **TPU** orchestration (v5e/v6e slices, not GPU). Also a standalone
argparse `IrisLauncher` (RL the primary use; datagen/eval less mature). Jobs write to **GCS**; the optional
**fetch daemon** polls the Iris controller and pulls completed outputs to `~/.ot-agent/runs/<job>/`, with a
local SQLite job registry. Verified iris flags: `--cluster-config`/`--cluster_config`,
`--task-image`/`--task_image`, `--tpu`, `--replicas`, `--priority`, `--max-retries`, `--cpu`, `--memory`,
`--disk`. Iris cluster/eval/job-lifecycle specifics live in `.claude/ops/iris/`.

---

## 9. Cluster detection & dotenv

`hpc.py:detect_hpc()` matches the hostname (FQDN regex; `NERSC_HOST` for Perlmutter) to a cluster config,
then `set_environment(hpc)` sources `hpc/dotenv/<cluster>.env` (HF/WANDB tokens, `HF_HUB_CACHE`, `DCFT`,
`DC_AGENT_TRAIN`, `CHECKPOINTS_DIR`, …). Recognized clusters (dotenv present): **jupiter** (GH200/aarch64,
no compute internet), **jureca**, **juwels**, **leonardo** (A100/x86, CINECA — **SLURM, NOT PBS**; the code
notes this explicitly), **capella**/**alpha** (ZIH), **vista**/**lonestar** (TACC), **perlmutter** (NERSC),
**frontier** (OLCF, AMD MI300X), **polaris** (ALCF), **nyutorch**/**nyugreene** (NYU), **dip**, **oumi**
(local). No match → `ValueError`. Per-cluster access/paths/preamble/gotchas live in `.claude/ops/<cluster>/`
(jupiter, leonardo, torch, iris, all). The **internet split** is the key cluster axis: JSC (Jupiter/Jureca/
Juwels) + Leonardo have NO compute-node internet → models/datasets are pre-downloaded on the login node and
jobs reach HF/Daytona via proxychains over an SSH SOCKS tunnel; TACC/NERSC/NYU compute nodes have internet.
Two no-internet pitfalls: **`push_to_hub` defaults to `false` on JSC** (override with `--push_to_hub true`
— proxychains *does* give compute nodes HF Hub access, though W&B does NOT work through proxychains); and
some clusters carry **faulty-node exclusion lists** in `hpc.py` (a job stuck PENDING may be waiting on
excluded nodes). Critical env vars come from `hpc/dotenv/<cluster>.env` (`HF_TOKEN`, `WANDB_TOKEN`,
`HF_HUB_CACHE`, `DCFT`, `DC_AGENT_TRAIN`, `CHECKPOINTS_DIR`).

---

## 10. Shared infrastructure (one-liners)

- `ray_utils.py` — builds/health-checks/​tears-down the SLURM-backed Ray cluster (RL/eval/datagen); NCCL env, port/hostname resolution, object-store sizing.
- `vllm_utils.py` — `VLLMServer` lifecycle (start, OpenAI-API health check, shutdown); TP/PP, quantization; auto-injects the Ray head IP for DP>1 (never `127.0.0.1`).
- `snapshot_manager.py` — pre-builds/cleans Daytona Docker snapshots (region- + env-hash-keyed); enforces the hard `max_new_snapshots`/`max_org_snapshots` caps (see daytona doc).
- `resume_manager.py` — detects an existing job dir, classifies config drift (mutable vs fatal), and advises resume / mutate / bail; lets datagen/eval resume after interruption.
- `proxychains_setup.py` (+ `proxychains.conf.template`) — generates the SOCKS proxychains config for no-internet compute nodes.
- `upload.py` / `hf_utils.py` — HF Hub upload + repo-id sanitization (96-char limit) / HF-path detection.
- `harbor_utils.py` — Harbor CLI invocation + job tracking (shared by datagen + eval).
- `pinggy_utils.py` — Pinggy SSH-tunnel setup for remote debugger access on no-internet clusters.
- `cloud_sync_utils.py` / `iris_fetch_daemon.py` — pull remote (cloud VM / GCS) outputs back to local.

> Operational how-tos (launch/monitor/cleanup per job type) live in `.claude/skills/`; cluster particulars
> in `.claude/ops/<cluster>/`; dependency facts in `.claude/projects/{daytona,marinskyrl}/`. This doc is the
> codebase/launcher map that ties them together.
