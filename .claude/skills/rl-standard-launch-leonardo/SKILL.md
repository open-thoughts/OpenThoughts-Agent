---
name: rl-standard-launch-leonardo
description: >-
  Launch, relaunch, or sweep STANDARD (non-agentic) SkyRL RL on CINECA Leonardo —
  GRPO on math/reasoning datasets (gsm8k, MATH/aime) and on-policy distillation
  (OPD, teacher→student) — via raw `sbatch` of the `hpc/skyrl_yaml/leonardo/*` run
  scripts inside the writable apptainer SANDBOX + uv `marin_venv` (NOT `python -m
  hpc.launch`, NOT a `.sif`, NOT `--rl_use_conda`). Use when asked to run/relaunch
  a gsm8k or OPD GRPO canary, throughput/accuracy grid, or multi-node RL on
  Leonardo A100-64GB. Covers the GRPO/OPD knobs, the grid cell structure, the
  1-node-vs-multi-node layout, the A100-64GB ceilings, and the no-internet/offline
  + gcc/HOME/Ray-temp-dir gotchas. For agentic Harbor+Daytona RL, this is the WRONG
  skill (Daytona needs internet — infeasible on Leonardo).
---

# rl-standard-launch-leonardo

> **⚠ Before submitting, VERIFY checkpoint/export paths resolve to `$WORK` (`$CHECKPOINTS_DIR`), NOT
> `$SF`/`$SCRATCH_FAST`** — scratch is 1 TB/over-quota; a ckpt write there fails `OSError [Errno 122] Disk
> quota exceeded` mid-run (NOT an OOM). This wiped all 12 Delphi #6279 RL cells on 2026-06-19 (an sbatch
> hardcoded `$SF/rl_ckpts`). See `.claude/ops/leonardo/ops.md` "WRITE-PATH MANDATE"; bake the check into
> every launch subagent.

Standard **NON-agentic** SkyRL RL on CINECA Leonardo (A100-64GB, no compute-node
internet): GRPO on local math/reasoning parquet (gsm8k, Hendrycks MATH via the
`aime` env) and on-policy distillation (OPD, teacher-logit KL). This is the
offline, fully-local path — **no Harbor, no Daytona, no terminal_bench, no
proxyserver**. (Agentic OPD / TBench RL is INFEASIBLE on Leonardo — Daytona is a
cloud API and compute nodes have no internet.)

Authoritative source docs (this skill distills them — read for full numbers):
- `notes/RL/gsm8k_grid_leonardo/` — `grid.md` (throughput tables), `accuracy_grid.md`
  (pass@8 to convergence), `grid_experiment_log.md` (bottleneck analysis + methodology),
  `scripts/` (per-cell `run_*.sh`).
- `notes/RL/opd_grid_leonardo/` — `leonardo_opd_qwen3_plan.md` (OPD mechanism + standup),
  `grid.md` (speed×convergence), `throughput_grid.md` (pass@8-per-GPU-hour).
- Generic Leonardo access boilerplate (ssh/2FA, preamble, code/data paths, step-ca
  cert, login-node killer) lives in **`.claude/ops/leonardo/ops.md`** and `CLAUDE.md`
  → "CINECA Leonardo Access". Point there; do not duplicate.

> ⚠️ **The launch mechanism is NOT `hpc.launch`.** Unlike SFT (which uses
> `python -m hpc.launch`) and unlike Perlmutter RL (`--rl_use_conda`), standard
> Leonardo RL is launched by **`sbatch`-ing a wrapper script** in
> `hpc/skyrl_yaml/leonardo/` that `singularity exec`s a writable **sandbox dir**
> (not a `.sif`) + an external **uv** venv, and invokes the SkyRL entrypoint
> directly. The run scripts say so explicitly: *"does NOT go through the OTA
> hpc.py launcher and has NO Harbor/Daytona/terminal_bench/agentic dependency."*

## 1. Cluster + env facts (the canary setup)

A100-**64GB**, 4 GPUs/node, x86_64, SLURM. Account `AIFAC_5C0_290`, partition
`boost_usr_prod`. QOS: **`boost_qos_dbg`** (≤30 min, ≤2 nodes — canary/grid cells)
or **`normal`** (longer walls, more nodes; **24h max**). See ops.md for the rest.

The RL runtime (from `reference_leonardo_marinskyrl_canary`):
- **MarinSkyRL** = `marin-community/MarinSkyRL` `main @ 9bb6d5e` (the `penfever/working`
  line merged to main; the remote has NO penfever fork), cloned to
  `$WORK/code/MarinSkyRL` (`$WORK = /leonardo_work/AIFAC_5C0_290/bfeuer00`). Distinct
  from the old `$WORK/code/SkyRL`. Container `--pwd` is `MarinSkyRL/skyrl-train`, so
  any SkyRL-side fix goes to **MarinSkyRL `main`** (NOT `penfever/SkyRL`).
- **Image = a writable apptainer SANDBOX dir**, NOT a `.sif`:
  `$SF/marinskyrl_sandbox` (`$SF = /leonardo_scratch/fast/AIFAC_5C0_290/bfeuer00`),
  built from `docker://anyscale/ray:2.51.1-slim-py312-cu128`. `mksquashfs` OOM-kills
  on the Lustre login node + hits fatal `lustre.lov` xattr errors → `.sif` deferred;
  a sandbox dir execs fine via `singularity exec --nv`. Binary is `/usr/bin/singularity`
  (SingularityPRO 4.3.1; no `apptainer` on PATH).
- **uv, not conda** (SkyRL-native): venv at `$SF/marin_venv` (`uv sync --extra vllm`
  against the committed `uv.lock` → torch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.3).
  conda can't satisfy the pinned cu128/flashinfer/torch/vllm/flash-attn graph. The
  run scripts use `$VENV_PY=$VENV/bin/python`. (`--rl_use_conda` is **Perlmutter-only**;
  it does NOT apply on Leonardo — verified, the canary uses the uv venv directly.)

### The 3 standing gotchas (the sbatch wrappers already handle these — keep them)
1. **No compute-node internet** → fully offline: `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`, `WANDB_MODE=offline`, `HF_HOME`/`HF_HUB_CACHE` =
   `$WORK/data/hub`. **Pre-stage model + dataset parquet on the LOGIN node first.**
2. **gcc for Triton JIT** — the ray base image ships no compiler. The wrapper binds
   the host miniforge `$WORK/miniforge3/envs/otagent/bin` onto PATH and exports
   `CC`/`CXX` (gcc 14.3.0) into the container. Also `RAY_USAGE_STATS_ENABLED=0`.
3. **HOME is read-only in-container** (`${HOME}` → `/leonardo` RO fs). SkyRL defaults
   `export_path` / several caches to `${HOME}/...` → point `HOME=$SF/canary_home` +
   `ckpt_path`/`export_path` at writable `$SF`. The `Traceback`/`FileNotFoundError:
   '/leonardo/home'`/`Read-only file system` lines under `--no-home` are **benign
   engine-init noise** (tvm_ffi dlpack, vLLM usage_lib telemetry) — ignore them;
   engine init completes past them. (If ever fatal: `VLLM_NO_USAGE_STATS=1`,
   `DO_NOT_TRACK=1`, `XDG_CACHE_HOME`/`XDG_CONFIG_HOME` on `$SF`.)

## 2. Pre-launch (login node, tmux)

Run the standard Leonardo preamble (ops.md), then pre-stage offline data:
```bash
ssh Leonardo                              # step-ca cert; 2FA once, socket ~8h (ops.md)
# preamble: conda activate otagent; cd $WORK/code/OpenThoughts-Agent; git pull;
#           source hpc/dotenv/leonardo.env && source ~/secrets.env  (ops.md)
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL && GIT_TERMINAL_PROMPT=0 git pull
# Pre-stage on the LOGIN node (compute has no internet):
hf download Qwen/Qwen2.5-1.5B-Instruct    # → $WORK/data/hub  (the model the cell uses)
# gsm8k parquet → $WORK/data/gsm8k/{train,validation}.parquet
#   (built with MarinSkyRL examples/gsm8k/gsm8k_dataset.py)
# MATH:  build via hpc/skyrl_yaml/leonardo/math_dataset.py → $WORK/data/math/
```
All code edits: edit local Mac → commit/push → `ssh Leonardo 'git pull'` (CLAUDE.md
sync discipline — never patch remote files). SkyRL fixes → MarinSkyRL `main`, pulled
on `$WORK/code/MarinSkyRL`.

## 3. Launch — single node (the canary + grid cells)

```bash
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/hpc/skyrl_yaml/leonardo
sbatch sbatch_gsm8k_canary.sh             # bare canary: 1 node × 4 A100, ≤30 min
```
The sbatch sets `DATA_DIR / MODEL_PATH / NUM_GPUS=4 / CKPT_DIR` + the offline env,
then `singularity exec --nv --no-home --bind /leonardo_work,/leonardo_scratch
--pwd $MARIN $SANDBOX bash run_gsm8k_canary.sh`. `run_gsm8k_canary.sh` calls
`$VENV_PY -m skyrl_train.entrypoints.main_base` (NOT `main_tbench`) with the GRPO knobs.

**Canary GRPO config** (`run_gsm8k_canary.sh`, Qwen2.5-1.5B-Instruct):
`advantage_estimator=grpo`, `strategy=fsdp2`, `colocate_all=true`, `backend=vllm`,
`run_engines_locally=true`, `weight_sync_backend=nccl`, `async_engine=true`,
4 inference engines × TP1, `use_kl_loss=false`, `lr=1e-6`, `n_samples_per_prompt=4`,
`train_batch_size=32`, `max_prompt_length=512`, `max_generate_length=512`,
`gpu_memory_utilization=0.70`, `env_class=gsm8k`, `epochs=1`, `logger=console`
(offline → console, not wandb). Reference run: job **44478923** COMPLETED, 233-step
epoch, 9.58 s/step, reward 0.14→0.64, pass@4 0.78.

### Grid-cell overrides (the `"$@"` passthrough)
`run_gsm8k_canary.sh` ends in `"$@"` → trailing hydra `key=val` args override (last-wins).
**But `sbatch_gsm8k_canary.sh` does NOT forward `"$@"`** — for grid cells use
**`sbatch_gsm8k_grid.sh`** (canary sbatch + passthrough + **fresh per-cell `CKPT_DIR`**,
`rm -rf`'d before launch — see the Wave-2 cross-cell-resume hazard in `grid_experiment_log.md`).
```bash
sbatch --job-name=grid_cudagraph sbatch_gsm8k_grid.sh generator.enforce_eager=false
sbatch --job-name=grid_tbs128    sbatch_gsm8k_grid.sh trainer.train_batch_size=128 trainer.policy_mini_batch_size=128
```
The `--job-name=grid_<cell>` is load-bearing: `sbatch_gsm8k_grid.sh` derives the
fresh ckpt dir from it (`CKPT_DIR=$SF/grid_ckpts/${SLURM_JOB_NAME#grid_}`). Per-cell
ready-made scripts live in `notes/RL/gsm8k_grid_leonardo/scripts/run_<cell>.sh`;
launchers `launch_throughput_grid.sh` / `launch_accuracy_grid.sh` + cell catalogs
`*_grid_cells.txt` are in `hpc/skyrl_yaml/leonardo/`.

## 4. Grid structure + what the cells vary

The grids are **one-factor-at-a-time** sweeps off a base config + combined "promising
corner" cells + ceiling probes (full tables/numbers in the source docs). Two distinct
gsm8k grids:

- **Throughput grid** (`gsm8k_grid_leonardo/grid.md`, 18 cells): maximize sec/step /
  eff tok/s off the canary base. Varies `train_batch_size` (32→512), `n_samples_per_prompt`
  (4→16), `gpu_memory_utilization` (0.70→0.85), `enforce_eager` (CUDA graphs), engine
  layout (4×TP1 vs 2×TP2 vs 1×TP4), `micro_*_batch_size_per_gpu`, `reshard_after_forward`,
  `colocate_all`. **Winners:** `enforce_eager=false` (CUDA graphs) = −29% sec/step,
  near-free, always on; **4×TP1 > 2×TP2 > 1×TP4**; **colocated > disaggregated** at 4 GPU.
  Bottleneck MIGRATES with width: base width is generation-bound (cudagraph fixes it),
  beyond ~tbs128 it's `policy_train` compute-bound; **never memory-bound** (KV <11%).
- **Accuracy grid** (`gsm8k_grid_leonardo/accuracy_grid.md`, 20 cells): maximize
  **pass@8 to convergence** off the throughput winner `combo_C`. Varies **lr** (the
  dominant lever; GRPO knee at **1e-5**, 3e-7 undertrains, 3e-5 unstable), `n_samples`
  (n8 = efficiency winner), `max_generate_length` (gen1024 winner), `use_kl_loss`/
  `kl_loss_coef`, rollout temperature, `eps_clip_high` (DAPO), entropy bonus
  (`use_entropy_loss=true, entropy_loss_coef=0.01` = the anti-collapse stabilizer
  winner), reward shaping. **Best:** `combo_acc` (lr1e-5 + n8 + gen1024) → pass@8 ~0.97
  but entropy collapses; `combo_acc_stab` (+ entbonus) holds ~0.95–0.98 WITHOUT collapse.

**gsm8k-specific:** short CoT (~245–268 tok), so gen512 truncates little; the gsm8k
env reward is exact-match ±1; lr knee 1e-5 (two orders ABOVE the OPD knee). Extensions
in `grid.md` cover MATH/`aime` (real long-CoT, `max_generate_length=4096`,
`run_math_grid.sh` + `math_dataset.py`) and model-size saturation (1.5B→32B; **32B OOMs
single-node 4×A100-64GB** → needs multi-node).

## 5. OPD — on-policy distillation (teacher→student)

A separate non-agentic RL mode: student (Qwen3-1.7B) generates rollouts; per-token
reward = −KL(student‖teacher) over the student's own tokens. Entrypoint
**`examples.on_policy_distillation_logits.main_on_policy_distill_logits`** (NOT
`main_base`, NOT the agentic `main_tbench_opd_logits` which needs Daytona). Key knobs:
`advantage_estimator=no_op`, `policy_loss_type=importance_sampling`,
`use_kl_in_reward=true`, `use_kl_loss=false`; the FSDP **ref worker is loaded with the
teacher model** + a separate **vLLM-served teacher** supplies top-K logprobs
(`teacher.top_k_logprobs`).
```bash
sbatch sbatch_opd_qwen3.sh                         # smoke defaults (2 nodes, ≤90 min)
sbatch --job-name=opd_q3_full --time=08:00:00 sbatch_opd_qwen3.sh \
  MAX_STEPS=60 EPOCHS=2 TRAIN_BATCH_SIZE=64 MINI_BATCH_SIZE=64 N_SAMPLES=8 MAX_GEN_LEN=1024 TOPK=128
```
`run_opd_qwen3_32b_to_1p7b.sh` is env-knob-sized (one script, smoke + full).
**Layout (2 nodes × 4 A100-64GB):** student colocated (FSDP2 ↔ 4× vLLM TP1) on node-0;
teacher **Qwen3-32B TP2** (32B bf16 ≈ 64GB > one 64GB card → needs TP≥2), own Ray PACK
PG on node-1; 2 GPUs spare. Student/teacher share the Qwen3 tokenizer → retokenization
is a no-op (`Qwen3-1.5B does not exist`; 1.7B is the nearest size — documented substitution).

**OPD is overwhelmingly teacher-score-bound** (90–97% of every step: the 32B teacher
scoring every rollout token via `prompt_logprobs=top_k`). So the speed lever is **`top_k`**
(superlinear: ~1.6–1.85× per doubling; topk16 ≈ 2.2× faster than topk128, near-lossless
on pass@8 at the right lr) and scored-token count (`max_generate_length`, `n_samples`).
**lr is the convergence lever (orthogonal to top_k): OPD knee = 3e-5** (1e-5 too low →
nothing learns, 1e-4 diverges). **Recommended production OPD:** `lr=3e-5, top_k=64,
n_samples=8, gen=1024, teacher TP2, cudagraph off` → pass@8 ~0.75–0.78 @ ~832 s/step.
(cudagraph and teacher-TP4 are NON-levers for OPD — student gen is <2% of the step;
FP8 teacher is speed-NEGATIVE on A100. See `opd_grid_leonardo/throughput_grid.md`.)

## 6. Multi-node (`--nodes N`, Ray bring-up)

Use **`sbatch_gsm8k_grid_multinode.sh`** (gsm8k) / `sbatch_math_grid_multinode.sh` /
`sbatch_opd_qwen3.sh` (OPD is already 2-node). It starts a Ray head inside the
container on node-0, attaches workers on nodes 1..N-1, then launches the trainer on
the head with `RAY_ADDRESS` set. Multi-node gotchas it handles (keep them):
- **Cross-node fabric pinned to InfiniBand `ib0`**: `NCCL_SOCKET_IFNAME=ib0`,
  `GLOO_SOCKET_IFNAME=ib0`; head IP resolved from `ib0` (not the `eno*` mgmt addr).
- **Ray `--temp-dir=/tmp`** (not Lustre scratch): the AF_UNIX plasma-store socket path
  `<temp-dir>/session_<ts>/sockets/plasma_store` **cannot exceed 107 bytes**, and the
  Lustre scratch root is already ~55 chars → a temp-dir there overflows. (verify the
  exact `RAY_TMP` value in the script before relaunch.)
- For gsm8k/1.5B, **multi-node generator scaling does NOT help** (it's train-bound, not
  gen-bound; 1→2 nodes regresses on cross-node FSDP, 8 nodes only breaks even; new
  limiter = `sync_weights`). Multi-node pays off only for big models (≥32B, which OOM
  single-node) or genuinely gen-bound small-model long-CoT.

## 7. Monitoring + completion

- **Monitor** detached, self-terminating: poll the `%x_%j.out` for the per-step
  `WANDB_MIRROR kind=train step=N metrics={...}` JSON lines (offline → stdout). Watch
  `timing/step`, `timing/generate`, `timing/policy_train`, `timing/sync_weights`;
  GRPO reward + `policy/policy_entropy` (collapse guard, mandatory); grad_norm. OPD:
  `distill/token_kl_mean` (should DECREASE), `teacher/chosen_logprob_mean`, entropy.
  Sweep cadence + cross-cluster checks → **`monitor-cron-sweep`**.
- **Resume:** these run scripts set `resume_mode=null` (fresh) per cell to avoid the
  cross-cell stale-`global_step` resume that voided gsm8k Wave-2. For a genuine resume,
  set `resume_mode=latest` and keep `ckpt_path` stable; for a clean re-run `rm -rf`
  the ckpt dir first.
- **24h wall:** `boost_usr_prod` caps at `23:59:00`; OPD full runs (~24 min/step) only
  fit ~18–20 steps per slot → ckpt every few steps and chain `--dependency=afterany:`.
- **Completion → run `rl-job-cleanup`** (cancel pending retries, consolidate/upload to
  HF via the Leonardo sbatch-tunnel — login-node 100s killer, see ops.md — DB register,
  trace/metrics steps, then `rm -rf` the dirs). Throughput/accuracy *measurement* runs
  with throwaway ckpts skip the upload (no production winner) — only clean up disk.

## 8. Guardrails

- **Launch via `sbatch hpc/skyrl_yaml/leonardo/sbatch_*.sh`, NOT `python -m hpc.launch`,
  NOT a `.sif`, NOT `--rl_use_conda`** — Leonardo standard RL = sandbox-dir + uv venv.
- **Fully offline** — pre-stage model + parquet on the login node; `HF_HUB_OFFLINE=1`,
  `WANDB_MODE=offline`; the `/leonardo/home` RO `FileNotFoundError`/`Traceback` lines
  are benign engine-init noise (§1.3).
- **Grid cells need `sbatch_gsm8k_grid.sh` (the `"$@"`-forwarding wrapper) + a unique
  `--job-name=grid_<cell>`** → fresh per-cell ckpt dir; the bare canary sbatch does NOT
  forward overrides. Never share a ckpt dir across cells (Wave-2 lesson).
- **A100-64GB single-node ceilings:** `gpu_memory_utilization` ≤ **0.85** (≥0.90 OOMs
  eval); dense ≥32B and MoE 30B-A3B OOM single-node (32B at FSDP4 backward / weight-sync
  broadcast) → multi-node/disaggregated.
- **Never alter hparams mid-series** (controlled grid) — flag + propose a separate cell.
  **Entropy/log-ratio/grad-norm are mandatory monitoring columns.**
- **Multi-node:** `ib0` NICs + Ray `--temp-dir` short path (107-byte AF_UNIX limit).
- **SkyRL fixes → MarinSkyRL `main`** (the `--pwd`), pushed + pulled on Leonardo; never
  patch remote files.
- **Agentic RL (Harbor/Daytona/TBench) is INFEASIBLE on Leonardo** (no compute-node
  internet) — that is a different skill, not this one.
