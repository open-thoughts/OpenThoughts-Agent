# Leonardo runtime / environment map

**Purpose:** which conda env / venv / container to use for which workstream on **CINECA Leonardo**, and the
version facts that bite. Leonardo differs from Jupiter on every axis that matters for binaries:
**x86_64** (not aarch64), **A100-64GB** (not GH200), **CUDA 12.x / cu128 wheels** (not cu130), and
**SingularityPRO** (`/usr/bin/singularity`, **no `apptainer`, no podman**). So Jupiter SIFs/wheels do NOT
transfer — Leonardo builds its own. Access/preamble/paths/HF-upload live in `ops.md`; this is the runtime map.

Last verified: **2026-06-14** (from `notes/leonardo.md` + the MarinSkyRL canary). Re-confirm with §3 if acting on this months later.

---

## 0. TL;DR — the discriminators

> - **otagent conda** = the default runtime for orchestration, eval, datagen, AND agentic work — **torch 2.9.1+cu128, vLLM 0.16.0 (built from source), flash_attn 2.8.3+cu128torch2.9**.
> - **sft-qwen35 conda** = Qwen3.5 (9B/27B) SFT only — **torch 2.9.1+cu128, transformers ≥5.3, deepspeed ≥0.18**.
> - **MarinSkyRL uv venv inside a singularity SANDBOX dir** = SkyRL RL (validated non-agentic gsm8k GRPO) — **torch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.3** (a DIFFERENT, SkyRL-native stack from otagent).
> - Compute nodes have **no internet** → pre-download models on the login node; jobs use proxychains over an SSH SOCKS tunnel.

---

## 1. Decision table — which runtime for which workstream

| Workstream | Runtime | Stack |
|---|---|---|
| Orchestration, `hpc.launch`, eval listener, datagen, HF uploads, Supabase | **`otagent` conda** | torch 2.9.1+cu128, vLLM 0.16.0 (src), FA2 2.8.3 |
| Agentic eval / datagen (Harbor + Daytona over proxychains) | **`otagent` conda** | same |
| **SkyRL RL** (gsm8k GRPO canary; SkyRL-native examples) | **MarinSkyRL uv venv in a singularity sandbox dir** | torch 2.8.0+cu128, vLLM 0.11.0, FA 2.8.3 |
| **SFT (Qwen3.5 9B/27B)** | **`sft-qwen35` conda** | torch 2.9.1+cu128, transformers 5.3+, deepspeed 0.18+ |
| Compilers (for any from-source build) | **conda/mamba**, NOT system modules | gcc/gxx 14 + conda nvcc; `module load cuda/12.2` only for llama.cpp |

---

## 2. The runtimes in detail

### 2a. `otagent` conda — orchestration + eval + datagen + agentic (the default)
- **Path:** `/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/` (activate via the `ops.md` preamble).
- **Stack:** **torch 2.9.1+cu128**, **vLLM 0.16.0 built FROM SOURCE** against that torch (the from-source vLLM build recipe + env scrubbing is in `notes/leonardo.md` "Install"), **flash_attn 2.8.3+cu128torch2.9** (prebuilt x86 wheel from mjun0812 release **v0.9.0**, NOT the aarch64 wheel Jupiter uses).
- **Use for:** everything that isn't SkyRL-RL or Qwen3.5-SFT — `hpc.launch`, the unified eval listener (`eval/leonardo/unified_eval_harbor.sbatch`, TP=4, 1 node, 24h), datagen, uploads. `proxychains` is provided by this env (compute-node internet).
- **Build gotchas (from-source vLLM):** set `CUDA_HOME=$CONDA_PREFIX`, `CUDACXX/CMAKE_CUDA_COMPILER=$CONDA_PREFIX/bin/nvcc`, `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:…`; `mamba install cmake ninja`; `VLLM_TARGET_DEVICE=cuda … uv pip install vllm==0.16.0 --no-build-isolation`. `-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler` if the host compiler is rejected.

### 2b. `sft-qwen35` conda — Qwen3.5 (9B/27B) SFT
- **Create:** `conda create -n sft-qwen35 python=3.12 && pip install uv`; `uv pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.9.1" …`; then in `sft/llamafactory`: `uv pip install "transformers>=5.3.0" "accelerate>=1.12" "deepspeed>=0.18" "datasets>=4.4" "peft>=0.18" "trl>=0.29" liger-kernel hf-kernels`.
- **Why separate:** Qwen3.5's hybrid GatedDeltaNet+Attention arch needs **transformers ≥5.3**, which the default LLaMA-Factory/otagent stack (transformers 4.x) can't load. Use for any `sft/lf_configs/qwen3_5/*` run.
- **Leonardo-specific:** the launcher doesn't wire conda activation here the way it does on Jupiter — **hand-patch the sbatch** (conda activate + WORKDIR) per CLAUDE.md "SFT Launch on Leonardo". Cleanup follows the 8B path (Qwen3.5 writes root safetensors → skip consolidate; copy `preprocessor_config.json` from the base before upload). HF upload uses the `ops.md` sbatch-tunnel (or the login-node nohup fallback when the cert is expired).

### 2c. MarinSkyRL — uv venv inside a singularity SANDBOX (RL runtime)
First successful non-agentic SkyRL RL on Leonardo (2026-06-05): a full gsm8k GRPO epoch on
Qwen2.5-1.5B-Instruct, job 44478923 COMPLETED 0:0 (reward 0.14→0.64, pass@4 0.78, weight-sync working).
NO Harbor/Daytona/agentic yet (proxyserver deferred).
- **Repo:** `marin-community/MarinSkyRL` `penfever/working` at `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/MarinSkyRL` (distinct from the removed old `/code/SkyRL`). Non-agentic recipe: `skyrl-train/examples/gsm8k/` via `python -m skyrl_train.entrypoints.main_base` (NOT main_tbench).
- **Image = a writable singularity SANDBOX DIR, NOT a `.sif`:** `singularity build --sandbox $SCRATCH_FAST/marinskyrl_sandbox docker://anyscale/ray:2.51.1-slim-py312-cu128`. **`mksquashfs` OOM-kills on the Lustre login node + hits fatal `lustre.lov` xattr errors → `.sif` packaging is deferred** (needs a non-login, xattr-clean build host). A sandbox dir execs fine via `singularity exec --nv`. Binary is `/usr/bin/singularity` (SingularityPRO 4.3.1).
- **Env = uv, not conda** (SkyRL-native): the venv is **`$SCRATCH_FAST/marin_venv`** (distinct from the `marinskyrl_sandbox` image dir), built via `uv 0.9.4` + `uv sync --extra vllm` against the committed `uv.lock` → **torch 2.8.0+cu128, vLLM 0.11.0, flash-attn 2.8.3**. conda can't satisfy the pinned cu128/flashinfer/torch/vllm/flash-attn graph. MarinSkyRL is **`pip install -e`'d (editable)** into this venv.
- **⚠️ STALE EDITABLE-INSTALL TRAP — regenerate the editable install after any structural package change (2026-06-21, the thinking-RL grid 16-node wipeout).** MarinSkyRL is editable-installed in `marin_venv`, so setuptools wrote a **meta-path finder** (`marin_venv/lib/python*/site-packages/__editable___skyrl_train_*_finder.py`) . When a `git pull` **ADDS a subpackage** (here `skyrl_train/dataset/`, added 2026-06-19 *after* the `pip install -e` ran) the install goes STALE: on a Ray actor `skyrl_train.<new>` resolves as an empty **"unknown location"** namespace → `ImportError: cannot import name <X> from 'skyrl_train.<new>' (unknown location)`. (The exact internal is finder-style-dependent and not worth chasing — some finders statically enumerate subpackages at install time, this one FS-walks children at import; either way the stale install mis-resolves the new subpackage on actors, and a clean reinstall clears it.)
  - **Manifests on Ray ACTORS at MULTI-NODE scale, not the driver.** The trainer driver runs with `--pwd=$MARIN` so the repo root is on its cwd/sys.path and resolves fine; remote-node FSDP policy-worker actors have no helpful cwd → hit the stale finder → crash at `FSDPPolicyWorkerBase.__init__`. The head looks healthy (driver tokenizes, engines init); a 2-node smoke can get lucky on actor placement; a 16-node run spreads workers to remote nodes → **all cells fail identically at worker init**.
  - **`PYTHONPATH=$MARIN` does NOT fix it** (meta-path precedes sys.path for the claimed namespace; that 7b36d9d1 commit is harmless but inert here).
  - **FIX = regenerate the editable install** (env maintenance on the already-committed code — like rebuilding a SIF, NOT a code/divergence change). ⚠️ The exact form matters (verified 2026-06-22, the 4-cell long-run launch): `<python> -m uv` and `-m pip` BOTH FAIL — this venv has neither as a module; `uv` is the standalone container binary. Run, in the sandbox: `singularity exec --nv -B /leonardo_work,/leonardo_scratch $SF/marinskyrl_sandbox bash -lc "cd $MARIN/skyrl-train && uv pip install --python $SF/marin_venv/bin/python -e . --no-deps"` — note it installs against **`$MARIN/skyrl-train`** (the package root where the finder lives), NOT `$MARIN`, and is offline (`--no-deps`; re-enumerates the finder). This trap has recurred 3× (grid ×2 + the 4-cell launch) — regenerate proactively after any MarinSkyRL pull (next bullet).
  - **DETERMINISTIC pre-launch GATE** (do this before spending N nodes): from a NON-repo cwd inside the sandbox+venv, `python -c "from skyrl_train.dataset import PromptDataset; print('OK')"` must succeed (replicates the actor import context — no helpful cwd). ⚠️ Do NOT grep the finder file for the subpackage name — this finder style FS-walks children and never lists them literally, so a grep returns 0 *even when healthy*; the import repro is the ONLY reliable gate. If it fails with `unknown location`, the install is stale → regenerate before launching.
  - **STANDING RULE:** after any MarinSkyRL pull that adds/moves/renames a `skyrl_train` subpackage, regenerate the editable install + run the import repro. Treat it as part of the cluster-sync, same as `git pull`.
- **Gotchas:** Triton JIT needs a C compiler the ray base image lacks → bind the host miniforge **gcc 14.3.0** onto PATH + set `CC`/`CXX` inside the container; also `RAY_USAGE_STATS_ENABLED=0`. **`HOME`/`trainer.export_path` default to `${HOME}`** which is read-only `/leonardo` inside the container → point `HOME` at a writable scratch tmp, but **`trainer.export_path`/checkpoints MUST go to `$WORK` (`$CHECKPOINTS_DIR`), NOT `$SCRATCH_FAST`** — scratch is 1 TB/over-quota and a ckpt write there fails `OSError [Errno 122] Disk quota exceeded` (see ops.md "WRITE-PATH MANDATE"; the 2026-06-19 Delphi RL grid wipeout). Scratch = ephemeral caches only.
- **Configs:** `hpc/skyrl_yaml/leonardo/{run_gsm8k_canary.sh, sbatch_gsm8k_canary.sh, note.txt}` (commit `44afed58`). sbatch = `--account=AIFAC_5C0_290 --partition=boost_usr_prod`, 1 node `--gres=gpu:a100:4` (debug QOS `boost_qos_dbg` ≤30min). Prestage on the LOGIN node (compute has no internet): model → `$WORK/data/hub`, gsm8k parquet → `$WORK/data/gsm8k`. `WANDB_MODE=offline`.

---

### 2d. `skyrl_megatron_vllm0202rc0_r3_sandbox` — the Jupiter prod-SIF cross-cluster twin (RECIPE; build pending)
The x86/A100 analogue of Jupiter's `skyrl_megatron_vllm0202rc0_r3.sif`: SkyRL editable (`penfever/SkyRL @ 2ab513a6`) + Megatron-core 0.14.0 + TE + flash-attn + **vLLM fork `penfever/working @ 5d7319dd1`** (0.20.2rc0 + native R3 routed-experts capture + the DCP GQA-LSE fp32 fix). Built as a **writable singularity sandbox dir** at `$WORK/containers/skyrl_megatron_vllm0202rc0_r3_sandbox/` (NOT a `.sif` — mksquashfs OOM/xattr blocker on Lustre login).
- **CUDA-13 via FORWARD COMPATIBILITY (the gating question — NOT a flat blocker).** Leonardo A100 nodes load NVIDIA *kernel driver* **`535.274.02`** (native CUDA ≤12.2, verified `srun nvidia-smi`), and `singularity --nv` binds *that host* `libcuda` — a container cannot replace the kernel driver. **But the *toolkit* can be CUDA-13:** CUDA **Forward Compatibility** (the `cuda-compat-13` package, bundled in NGC cu13 images at `/usr/local/cuda/compat`) ships a *newer userspace* `libcuda.so` that lets a cu13 toolkit run on an older **datacenter** driver — and the A100 qualifies (forward-compat is a datacenter-GPU feature). The trick is LD ordering: `LD_LIBRARY_PATH=/usr/local/cuda/compat` must precede the `--nv`-bound host lib. ⚠️ The earlier "CUDA-13 infeasible" verdict was **WRONG** — it conflated the host *driver's* native ceiling with the *toolkit*, checked only host `module avail` (no cuda-compat *module*), and invoked *minor*-version compat; it never tried installing cu13 + the forward-compat libs **into the container**. The ONE real caveat: forward-compat has a per-CUDA-version **minimum-driver floor** — whether the 535 branch is within cu13's floor must be **verified empirically** (run a cu13 kernel on an A100 via the compat libs). (For reference, CUDA-12 *minor*-forward-compat already lets otagent's torch 2.9.1+cu128 run A100 kernels at cap (8,0) under this driver.)
- **Matched vs differs:** matched (both paths) = vLLM fork commit `5d7319dd1`, R3 native capture, DCP fp32 fix, model archs (Gemma4/Qwen3Moe/Qwen3Next), SkyRL/Megatron/TE stack, `TORCH_CUDA_ARCH_LIST=8.0` (A100, vs 9.0 GH200), GDN/FlashQLA overlay **omitted** (deferred; vanilla GDN still runs). torch/CUDA floor depends on the forward-compat outcome: **cu13/torch-2.9 (true parity) if forward-compat clears 535**, else **torch-2.8/cu12.9 (fallback)**.
- **Recipe (committed):** `.claude/ops/leonardo/sif_build/recipes/{README_vllm0202rc0_r3_leonardo.md, build_vllm0202rc0_r3_leonardo.sbatch}`. Two-phase sbatch (A: SkyRL+Megatron+flash-attn; B: vLLM-from-source against in-base torch 2.8 via `use_existing_torch.py`, arch 8.0, offline wheelhouses). **Build the sandbox on WORK, not SCRATCH_FAST** (SCRATCH_FAST was 3.7 T / 1 T over quota, grace=none, 2026-06-16). Runtime env: `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `LIBRARY_PATH=/.singularity.d/libs` for tp>1.
- **FINAL DECISION (2026-06-16, user-approved):** (1) **PREFER the true CUDA-13 / torch-2.9 Jupiter twin** — NGC 25.09 base + `cuda-compat-13` forward-compat libs (arch 8.0), pending the empirical forward-compat test on the 535 driver. (2) **SANCTIONED FALLBACK:** if forward-compat can't clear the 535 driver ("if we have no other option"), drop to the **torch-2.8 / NGC-25.06 / CUDA-12.9.1** twin (recipe already written; differs from Jupiter only in the torch/CUDA floor). Escalating a CINECA driver upgrade (→≥580.65) is **not** required — the fallback is acceptable.
- **✅ FORWARD-COMPAT GATE PASSED (2026-06-16, STAGE 1 empirically verified) → the true CUDA-13/torch-2.9 twin is GO; fallback NOT needed.** Built the cu13 base as a **writable sandbox dir** (`$WORK/containers/pytorch_2509_sbx`, NGC `nvcr.io/nvidia/pytorch:25.09-py3`, 19 G) — note a packed `.sif` pull **FATAL'd** at `while creating squashfs: create command failed: signal: killed` (the documented login-node mksquashfs OOM/kill + `lustre.lov` xattr blocker; `singularity build --sandbox` avoids the squash step and succeeds, with `TMPDIR`/`SINGULARITY_TMPDIR` forced onto GPFS WORK — the default `TMPDIR=/scratch_local` is Lustre and triggers the xattr storm). On an A100 (`srun … boost_qos_dbg`, host driver **535.274.02**) under `singularity exec --nv -B /leonardo_work`, **a real CUDA-13 fp32 matmul executed**: `torch 2.9.0a0+…nv25.09`, `torch.version.cuda 13.0`, `is_available True`, cap `(8,0)`, `2048² matmul maxerr=6.8e-2` (normal tf32 tolerance — real tensor-core compute). `/proc/self/maps` confirms torch loaded **`/usr/local/cuda-13.0/compat/lib.real/libcuda.so.580.82.07`** (the bundled forward-compat userspace), NOT a host 535 libcuda. So the **535 branch is within cu13's forward-compat floor** on the A100 (datacenter GPU). The NGC image already wires the compat libs via its own ldconfig (`compat/lib` → `lib.real`), so no manual `LD_LIBRARY_PATH` reorder was even needed under `--nv`; if a future image doesn't, set `SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat/lib.real`. Test artifacts: `$WORK/containers/{fwdcompat_test.py,fwdcompat_run.sh,check_libcuda.py}`.

### 2e. `evalchemy-marin` conda — standard / pass@k downstream evals (Delphi #6279)

- **The ONE canonical evalchemy clone:** `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/evalchemy-marin` —
  remote `origin = github.com/marin-community/evalchemy.git`, branch `main`. The **`evalchemy-marin` conda env**
  is the editable install against it (lm-eval v0.4.12 standard). All recent Delphi #6279 pass@k + standard
  `SCORES.md` rows used this env+clone; the live `evalchemy_eval.sbatch` `cd`s here.
- **Use for:** MATH500 / gsm8k / AIME24 (`eval/evalchemy/*.sbatch`) + native pass@k (`eval.eval --num_samples N --pass_at_k …`; unbiased estimator in `eval/passk.py`). The standalone `passatk/` driver was removed 2026-06-22 (superseded by native pass@k).
- **CLEANUP (2026-06-18):** the two redundant clones were **deleted** — `code/evalchemy` (a stale
  `mlfoundations/evalchemy` clone, frozen at clone-time HEAD `6ed67415`, paired with the deprecated 0.4.9.1
  `evalchemy` conda env; cut over to `evalchemy-marin` on 2026-06-15 after MATH500 validated 9.6% vs 10.2%
  grader-byte-identical) and `code/evalchemy-resume-test` (a *linked git worktree* of `code/evalchemy` on
  branch `feuer/resume-manager` @ `d42f2c38`, **fully merged** into `evalchemy-marin` main `3f1618c2` —
  no unmerged work lost). Do NOT re-create either; `evalchemy-marin` is the only clone.

## 3. VERIFY before you trust

```bash
# otagent
/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/python -c "import torch,vllm; print('otagent', torch.__version__, vllm.__version__)"   # → 2.9.1+cu128 / 0.16.0
# sft-qwen35
/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/sft-qwen35/bin/python -c "import torch,transformers; print('sft', torch.__version__, transformers.__version__)"  # → 2.9.1+cu128 / 5.3+
# SkyRL sandbox venv
singularity exec --nv $SCRATCH_FAST/marinskyrl_sandbox python -c "import torch,vllm; print('skyrl', torch.__version__, vllm.__version__)"   # → 2.8.0+cu128 / 0.11.0
# SkyRL editable-install freshness (the §2c stale-finder trap) — from a NON-repo cwd so it matches a Ray actor:
cd /tmp && singularity exec --nv -B /leonardo_work,/leonardo_scratch $SCRATCH_FAST/marinskyrl_sandbox $SCRATCH_FAST/marin_venv/bin/python -c "from skyrl_train.dataset import PromptDataset; print('editable OK', PromptDataset)"   # MUST print OK; if ImportError 'unknown location' → regenerate: singularity exec --nv -B /leonardo_work,/leonardo_scratch $SCRATCH_FAST/marinskyrl_sandbox bash -lc "cd $MARIN/skyrl-train && uv pip install --python $SCRATCH_FAST/marin_venv/bin/python -e . --no-deps"  (NOT `-m uv`/`-m pip` — not modules here; uv = standalone container binary)
```

---

## 4. Hardware / filesystems (the constraints that shape the above)

- **Booster nodes:** 32-core Ice Lake (x86_64) @ 2.6GHz, **4× A100-64GB**, 512 GB RAM, 3456 nodes. CUDA 12.2/12.3/12.6 modules; gcc 12.2 system / conda gcc 14.
- **Filesystems:** HOME `/leonardo/home/userexternal/bfeuer00` (50 GB, NFS, persistent) — too small for envs; **WORK** `/leonardo_work/AIFAC_5C0_290` (~1.465 PB shared GPFS, persistent — code/envs/HF-cache/experiments live here); **SCRATCH/fast** `/leonardo_scratch/fast/AIFAC_5C0_290` (1 TB shared Lustre, **auto-purged**, fastest — vLLM/Triton/FlashInfer caches + job tmp; often OVER quota, may need project cleanup). Env vars in `hpc/dotenv/leonardo.env`: `HF_HOME`/`HF_HUB_CACHE=$WORK/data/hub`, `VLLM_CONFIG_ROOT`/`TRITON_CACHE_DIR`/`FLASHINFER_WORKSPACE_BASE=$SCRATCH_FAST/vllm_cache`.
- **Account:** `AIFAC_5C0_290` (the expired `EUHPC_E03_068` / `CMPNS_E03_068` are dead — do NOT use). Partition `boost_usr_prod`, max wall 24h (`--time 23:59:00`), debug QOS `boost_qos_dbg` ≤30min. Budget: `saldo -b`; storage: `cinQuota`/`cindata`.

---

## 5. Canonical env set — anything else is cruft (inventory reconciled 2026-06-21)

These are ALL the runtimes we intentionally keep. If you find an env/sandbox/clone NOT on this list, it's a candidate to retire (audit before assuming).

**conda (`$WORK/miniforge3/envs/`):** `otagent` (default — orch/eval/datagen/agentic), `sft-qwen35` (Qwen3.5 SFT), `evalchemy-marin` (Delphi #6279 evals), `ajudge` (LLM-judge tool — a documented dependency, rarely used).
**RL runtime (`$SF`):** `marinskyrl_sandbox` (singularity sandbox) + `marin_venv` (uv venv, editable MarinSkyRL) — torch 2.8/vLLM 0.11. **This is what live multi-node RL uses** (not the cu13 twin).
**cu13 cross-cluster twin (`$WORK/containers/`):** `skyrl_megatron_vllm0202rc0_r3_sandbox` (the Jupiter prod-SIF twin, 22G) + `pytorch_2509_sbx` (its NGC cu13 base, 19G) — both fully built, **PARKED**: no live RL uses them (the recipe is committed under `sif_build/recipes/`, so they're rebuildable). Keep only while the cross-cluster-twin option is open; retire if that effort is abandoned.
**code clones (`$WORK/code/`):** `OpenThoughts-Agent`, `MarinSkyRL`, `harbor`, `evalchemy-marin`, `ajudge`.

**RETIRED 2026-06-21** (orphaned, no live ref): the deprecated **`evalchemy` 0.4.9.1 conda env** (superseded by `evalchemy-marin` 2026-06-15), the conda `pkgs` cache, stale `$SF` dirs (`ray_tmp`/`canary_home`/`apptainer_cache`), the one-off forward-compat/build artifacts, and the unidentified **`code/abb`** clone (700M). The **cu13 twin pair is KEPT (parked)** per user decision 2026-06-21 — negligible footprint, option value for the cross-cluster twin. Earlier deletions (per §2e): `code/evalchemy`, `code/evalchemy-resume-test`, `code/SkyRL`.
