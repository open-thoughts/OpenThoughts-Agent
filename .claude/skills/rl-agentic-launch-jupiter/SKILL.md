---
name: rl-agentic-launch-jupiter
description: >-
  Launch / relaunch agentic RL (SkyRL terminal_bench + Harbor + Daytona) on JSC Jupiter (GH200).
  Covers the dense 8B/32B FSDP2 arms (seqnorm, TIS, shaped, symclip, lrboost, loopshape) and the
  MoE/80B Megatron arms (Qwen3-Coder-30B-A3B, Qwen3-Next-80B-A3B) — the exact `python -m hpc.launch
  --job_type rl` flag set, which flags vary per arm (config / model_path / train_data / num_nodes),
  runtime+SIF selection, the Daytona RL-org + chain-restart conventions, and the standing constraints
  (≤6 RL/cluster, a3 CONCLUDED, TIMEOUT restarts are normal). Use when asked to launch / relaunch /
  refill an agentic SkyRL RL run on Jupiter. Reference: notes/ot-agent/rl_experiments.md,
  .claude/ops/jupiter/{ops.md,ENVIRONMENT_MAP.md}.
---

# rl-agentic-launch-jupiter

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL code/config/sbatch edits (OpenThoughts-Agent +
> MarinSkyRL) go in the local Mac checkouts → commit → push → `git pull` on the cluster. **NEVER** hand-edit,
> `git commit`, or leave divergent/untracked changes on a cluster; no patch-by-rsync (vLLM is the only
> exception — built from source per-cluster). Bake this into every subagent you dispatch from this skill.

Agentic RL on Jupiter runs through **`python -m hpc.launch --job_type rl`** (SkyRL framework, GRPO,
FSDP2 or Megatron). Each rollout is a real **Harbor** agent episode against a **Daytona** sandbox
(the `terminal_bench` generator), with a colocated vLLM rollout engine. GH200 96GB, **4 GPUs/node**.

**Access boilerplate (ssh, the pre-launch preamble, key paths) lives in `.claude/ops/jupiter/ops.md` —
do not duplicate it; run that preamble first.** **Runtime / SIF selection detail lives in
`.claude/ops/jupiter/ENVIRONMENT_MAP.md`** — summarized in §3 here, deferred there for the gotchas.

## 1. The canonical launch

> **🚧 SUBMIT FROM THE REPO DIR WITH `DCFT` SET — the sbatch WORKDIR guard hard-fails otherwise.**
> Before launching/resuming, `cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent && export DCFT=$PWD`
> (the ops.md preamble does this). The generated `universal_rl.sbatch` resolves `WORKDIR` from
> `DCFT_PRIVATE → DCFT → $PWD`; if you submit from `$HOME` or a scratch subdir with `DCFT` unset, the
> guard detects the wrong dir (missing `hpc/shell_utils/triton_cache.sh` marker) and **`exit 1`s
> immediately** with a `FATAL: WORKDIR=... is not the OpenThoughts-Agent repo root` message. This bug
> silently broke 3 relaunches (#217, stageC, datagen) before the guard existed. If you ever see that
> FATAL, you submitted from the wrong place — `cd` to the repo, `export DCFT=$PWD`, resubmit.

```bash
python -m hpc.launch --job_type rl \
  --rl_config ./hpc/skyrl_yaml/jupiter/<cfg>.yaml \
  --model_path <hf-or-local-model> \
  --train_data '["<HF-repo-or-/abs/task/dir>"]' \
  --num_nodes N \
  --time_limit 11:59:00 \
  --max_restarts K \
  --reservation reformo \
  --experiments_dir /e/data1/datasets/playground/ot-baf \
  --job_name <name>
```
**What VARIES per arm:** `--rl_config` (the recipe), `--model_path`, `--train_data`, `--num_nodes`
(must match the config's GPU budget — see §2). **What is essentially fixed on Jupiter:**
`--time_limit 11:59:00` (booster QOS caps wall at 12h — chain past it with `--max_restarts`),
`--reservation reformo` (the `jureap59` booster QOS is suspended → `InvalidQOS`; `reformo` is the
runnable account/reservation), `--experiments_dir /e/data1/datasets/playground/ot-baf` (the `ot-baf`
personal data root; `/ot` is read-only-for-you), and `--max_restarts K` (chain-resume, see §5).

- **`--train_data` is a JSON-list string** `'["..."]'` — an HF repo (`DCAgent/…`, `laion/…`,
  `SankalpKJ/…`) **or** a pre-extracted local task dir (`/e/scratch/jureap59/feuer1/tasks/<name>`).
- **`--job_name <name>`** controls the experiment dir name; set it explicitly so chain-restart and
  cleanup land on a predictable un-suffixed dir. Without it the launcher derives a long auto name.
- **`--skyrl_override '++a.b.c=val'`** appends a Hydra override (last-wins over the base yaml).
  Used for per-arm tweaks without forking a config: sampling (`generator.sampling_params.temperature=1.0`,
  `…top_p`, `…top_k`, `…min_p`), Harbor sandbox sizing
  (`++terminal_bench_config.harbor.override_{cpus,memory_mb,storage_mb}`), context bumps
  (`++generator.engine_init_kwargs.max_model_len=…`). Pass **`++`-prefixed, struct-safe** keys — a
  bare top-level key (e.g. `enable_db_registration=false`) risks a Hydra `ConfigKeyError` and is
  redundant (see §4).
- **Launch from the `otagent` conda env** (`/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python`),
  NOT the RL venv — task extraction imports `google.cloud.storage`, which the RL venv lacks. (The
  launcher then selects the RL venv or a SIF for the actual *training* — §3.)

## 2. Config map + node count (`num_nodes` MUST match the config)
`num_nodes = GPUs / 4`. Pick the config, then set `--num_nodes` to its budget:

| Config (`hpc/skyrl_yaml/jupiter/…`) | Model | GPUs → `--num_nodes` |
|---|---|---|
| `56GPU_seqnorm_tis.yaml` (+ `extra/56GPU_seqnorm.yaml`, `extra/56GPU_seqnorm_tis_shaped.yaml`) | dense 8B | 56 → **14** |
| `extra/56GPU_seqnorm_tis_untrunc_symclip.yaml` | dense 8B (symclip) | 56 → **14** |
| `extra/56GPU_seqnorm_tis_untrunc_symclip_loopshape.yaml` | dense 8B (symclip+loopshape) | 56 → **14** |
| `extra/56GPU_seqnorm_tis_untrunc_lrboost.yaml` | dense 8B (lr-boost) | 56 → **14** |
| `56GPU_shaped.yaml` (`extra/24GPU_shaped.yaml`) | dense 8B (shaped reward) | 56→14 / 24→**6** |
| `24GPU_base_131k.yaml` / `extra/24GPU_base_old.yaml` | dense 8B | 24 → **6** |
| `64GPU_base_32b.yaml`, `extra/64GPU_base_32b_fp8.yaml`, `extra/48GPU_*_32b.yaml`, `extra/128GPU_base_32b.yaml` | dense 32B | 64→16 / 48→12 / 128→32 |
| **`24GPU_qwen3_coder_30b_a3b.yaml`** | **Qwen3-Coder-30B-A3B (MoE)** | 24 → **6** |
| **`extra/128GPU_qwen3_next_80b_a3b.yaml`** | **Qwen3-Next-80B-A3B (MoE, prod)** | 64 → **16** (name is historical; header = 64 GPU/16 node) |
| `extra/16GPU_mixtral_8x7b.yaml` | Mixtral-8x7B (MoE bring-up) | 16 → **4** |

General rule (per memory): 24GPU→6, 48GPU→12, 56GPU→14, 64GPU→16, 96GPU→24, 128GPU→32.
The dense-arm models are typically `laion/GLM-4_7-swesmith-…-fixthink` (an 8B; this is the a3-series
pre-RL base) or `Qwen/Qwen3-8B`/`Qwen3-32B`; common train sets are `exp_rpt_pymethods2test-large`,
`code-contests-sandboxes-with-tests`, `swesmith-oracle-filtered`, `swe_rebench_patched_oracle`.

## 3. Runtime / SIF selection (summary — detail in ENVIRONMENT_MAP.md)
The **launcher** picks venv-vs-SIF for the *training* process (`hpc/sbatch_rl/universal_rl.sbatch`).
To confirm which a job used, read its rendered sbatch (`experiments/<job>/sbatch/*.sbatch`) for
`apptainer exec … <sif>` vs the venv python — don't assume.
- **Dense 8B/32B FSDP2** (seqnorm / TIS / shaped / symclip / lrboost / loopshape) → **RL venv**
  `$WORKDIR/envs/rl` (vLLM 0.16-era, **torch 2.9**). Default RL runtime.
- **MoE — Qwen3-Coder-30B-A3B** and **prod 80B Qwen3-Next-80B-A3B (R3+TIS)** → **SIF
  `skyrl_megatron_vllm_r3baked.sif`** (vLLM 0.16.0, **torch 2.9**, overlays baked in).
- **torch≥2.10 / DCP / torch-native CP / Mixtral-multinode work** → **SIF
  `skyrl_megatron_vllm0202rc0_r3.sif`** (vLLM 0.20.2rc0, **torch 2.11**); stack the
  **`skyrl_titan_overlay.img`** when torchtitan-0.2.2 / `_StridedShard` (CP+EP) is needed.

> `torch` version is the reliable discriminator, **not** `vllm.__version__` (which lies). See
> ENVIRONMENT_MAP §4 for the verify-before-you-trust probes, and §2c/§2d for the SIF gotchas
> (FlashInfer-sampler env, Triton libcuda linker path, `pg_options→backend_options`).

## 4. Agentic infra conventions
- **Daytona uses the RL-org key** for RL rollouts (distinct from the eval-org key). The launch preamble
  / `hpc/dotenv/jupiter.env` set it; you generally don't pass it on the CLI.
- **Pinggy is for EVALS, not RL** — `--pinggy_persistent_url` / `--pinggy_token` are eval-path flags.
  (A few very old RL commands in the ref doc carry a pinggy flag, but the agentic RL path does not use
  pinggy tunnels; treat pinggy as eval-only. *(verify if a new RL config starts requiring it.)*)
- **`enable_db_registration: false`** — the launcher **auto-injects** `++trainer.enable_db_registration=false`
  for RL. Do NOT also pass a bare `--skyrl_override enable_db_registration=false` (Hydra resolves it as a
  bare top-level key → struct ConfigKeyError risk, and it's redundant). DB registration is a **manual
  cleanup step**, not a launch flag.
- **Daytona snapshots:** a new task set builds snapshots on first launch; caps are **HARD**
  (10 new / 60 org — RL org observed at 40, server-side). Registry hits (snapshots already ACTIVE) cost 0
  new. **At the org cap, clean STALE snapshots first (autonomous) — do NOT raise the cap:**
  `python scripts/daytona/daytona_snapshot_manager.py --api-key-env DAYTONA_RL_API_KEY --delete-stale --yes --stale-days 2`
  (audit-only without `--delete-stale`; deletes only idle/unprotected `harbor__*` envs — safe). Only a
  single dataset legitimately needing >`max_new_snapshots` unique envs escalates → ask. Full procedure +
  caveats → `.claude/projects/daytona/daytona.md` § "How to clean stale snapshots".
- **MoE / 80B placement:** the MoE configs carry their own FSDP/EP sizing in-yaml
  (Coder-30B: EP=4×FSDP=4=16 policy GPUs + 4 TP=2 vLLM engines = 24 GPU/6 nodes; 80B: 8 TP=4 engines +
  8-node FSDP shard = 64 GPU/16 nodes). The **80B yaml sets `policy_strict_spread_pg: true`** (opt-in
  anti-affinity that reserves the policy PG up front to dodge the two-PACK-PG init-OOM race); leave it
  as-configured. Honor the MoE FSDP/EP divisibility constraint (`fsdp_size` must divide
  `num_experts // ep_size`) — don't hand-edit node/EP counts.

## 5. Chain-restart (`--max_restarts K`)
`--max_restarts K` submits a head job + K `afterany`-dependent restart links. A link that hits the
**12h wall TIMEOUT auto-resumes from the latest checkpoint** in the next link — **TIMEOUT is the
NORMAL terminal state of a healthy chain, not a failure.** Typical `K` is **5–6**.
- **A fresh `python -m hpc.launch` with the SAME `--job_name` forks to `<dir>_2` at step 0** if the
  original exp dir's `configs/*.json` exists (the dedup resume-manager engages only for datagen/eval,
  not RL). To *resume* an existing chain instead of forking: either resubmit the existing generated
  sbatch (`experiments/<dir>/sbatch/*_rl.sbatch`) via `--dependency=afterany`, or move the original
  `configs/*.json` aside so the dedup lands on the un-suffixed dir. (`--dry_run` regenerates that
  config → re-move after a dry-run, or skip it.)
- **Relaunching auto-resumes from prior failed-run checkpoints** (silent). For a *clean* ablation,
  delete the checkpoint dir first; for a chain extension, leave it.
- Always **`scancel` the previous failed/superseded chain** before resubmitting (cancel-before-resubmit).

## 6. Standing constraints (do NOT violate)
- **Daytona RL concurrency ≤ 6 RUNNING per cluster.** PENDING restart links don't count. Don't launch a
  7th concurrent RL job on Jupiter.
- **The a3 series is CONCLUDED — do NOT launch, refill, or auto-advance a3 rows.** (a3 = binary reward
  + RLOO-n + token_mean; uninformative.) Successor arms are the seqnorm / TIS / shaped / symclip /
  loopshape ablations above. *(Exception per memory: `DCAgent/r2egym-patched-full-oracle` is a separate
  snapshot-optimized variant — not the a3 row — and launches fine.)*
- **Never alter config/hparams mid-series.** If a controlled ablation needs a change, propose a separate
  experiment; do not mutate the in-flight arm.
- **TIMEOUT restarts are expected** (§5) — do not treat a chain's TIMEOUT links as failures or salvage them.

## 7. After launch
- **Monitor:** `monitor-cron-sweep` (the 3-cluster RL/eval/datagen health-sweep cadence;
  entropy / log_ratio / grad_norm are mandatory progress columns).
- **On completion → `rl-job-cleanup`** (best-ckpt selection, HF upload from the login node, the
  **manual** Supabase DB registration, trace export + `parse_skyrl_metrics`). DB registration is a
  cleanup step here — `enable_db_registration` stays false at launch (§4).
- **Behavior analysis:** `analyze-rl-behavior` for a post-hoc arm comparison.

---

## Operating notes (folded from memory 2026-06-14)

- **Every `hpc.launch --job_type rl` MUST pass `--num_nodes N`** = total_gpus_in_config / gpus_per_node (Jupiter GH200 = 4/node): `64GPU_base_32b.yaml`→16, `48GPU`→12, `24GPU`→6. The generated sbatch `#SBATCH --nodes=1` default is misleading — the CLI override controls `-N` at submit. Derive N from the yaml (`policy_num_nodes`, `ref_num_nodes`, `num_inference_engines × inference_engine_tensor_parallel_size`). **If an RL job fails <15min with no clear error, check NODE COUNT before blaming Daytona/OOM** — single-node allocation is a common fast-fail.
- **Launch a3/agentic RL from the `otagent` conda env, NOT `envs/rl`.** Task-extraction (`extract_tasks_from_parquet`) imports `google.cloud.storage`, which `envs/rl` lacks → `ImportError` before submit. Training itself still runs in `envs/rl` via srun. (Same reason the in-run Step-8 trace upload fails `exit 1` from `envs/rl` — re-run `make_and_upload_trace_dataset` from otagent during cleanup; traces aren't lost.)
- **Relaunching with the same `--job_name` SILENTLY auto-resumes** from any prior `checkpoints/global_step_N/` (even a failed mid-training attempt). For a CLEAN ablation restart, `rm -rf <exp>/<job>/<job>/checkpoints/` BEFORE relaunching. For resume-after-transient-crash, leave it intact. Suspiciously-fast step numbers (step ≫ elapsed/expected) = an unintended inherited checkpoint.
- **vLLM DP>1 (ray backend): never hardcode `--data-parallel-address 127.0.0.1`** — Ray registers the head only under its real IPv4 → `127.0.0.1` gives `AssertionError: DP master node missing or dead`. `hpc/vllm_utils.py` `VLLMServer.start()` auto-injects the head IP for DP>1; don't add the flag to new yamls. If overriding, use the real Ray head IPv4.
- MoE/EP config constraint (`fsdp_size` must divide `num_experts//ep_size`) and the a3-resume dry-run gotcha (resolved in code) live in `.claude/projects/marinskyrl/marinskyrl.md`.
