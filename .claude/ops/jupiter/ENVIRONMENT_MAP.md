# Jupiter runtime / environment map

**Purpose:** there are **6 distinct runtimes** on Jupiter that can run SkyRL/vLLM code, with
**two different vLLM lines** (0.16-era vs 0.20.2rc0) and **two torch lines** (2.9 vs 2.11).
`vllm.__version__` is **unreliable** (reports `dev`, `0.1.dev16577+g…`, or `0.20.2rc0` for what
are different builds), so people (and subagents) keep mis-identifying which vLLM a run used. This
doc is the source of truth. **Verify before you trust** — see §4.

Last verified: **2026-06-13** (versions probed live). Re-confirm with §4 if acting on this months later.

---

## 0. TL;DR — the one rule that prevents the mistake

> **`torch` version is the reliable discriminator, not `vllm.__version__`.**
> - **torch 2.9.0+cu130  → OLD stack → vLLM 0.16-era** (RL venv, `*_r3baked.sif`, `skyrl_megatron*.sif`)
> - **torch 2.11.0+cu130 → NEW stack → vLLM 0.20.2rc0** (otagent conda, `*vllm0202rc0_r3.sif`)
>
> DCP (`decode_context_parallel_size`/`get_dcp_group`), torch-native CP (`context_parallel`), and
> anything needing torch≥2.10 ONLY exist in the **NEW stack**. If a parity/feature test "fails" on
> a torch-2.9 runtime, you tested the wrong vLLM.

---

## 1. Decision table — which runtime for which workstream

| Workstream | Runtime | vLLM / torch |
|---|---|---|
| **Local** Harbor/SkyRL code, launching, uploads, eval/datagen orchestration, count tooling, Supabase | **`otagent` conda (local Mac AND Jupiter)** | (local: no GPU) / Jupiter otagent: vLLM `0.1.dev…041cfa68e`, **torch 2.11** |
| **Standard dense RL** (8B/32B FSDP2: a3, seqnorm/TIS/shaped/symclip/lrboost/loopshape ablations) | **RL venv** `$WORKDIR/envs/rl` | vLLM `dev` (**0.16-era**), **torch 2.9** |
| **MoE / Megatron RL** (Qwen3-Coder-30B-A3B; prod 80B Qwen3-Next-80B-A3B R3+TIS) | **SIF `skyrl_megatron_vllm_r3baked.sif`** (baked overlays) | vLLM **0.16.0**, **torch 2.9** (NGC) |
| **NEW torch2.11 / vLLM-0.20.2rc0 work** (FSDP2-CP feature, DCP feature, Mixtral multi-node, anything needing torch≥2.10) | **SIF `skyrl_megatron_vllm0202rc0_r3.sif`** | vLLM **0.20.2rc0**, **torch 2.11** |
| Datagen / eval (Harbor + Daytona, no training) | **`otagent` conda** | torch 2.11 (vLLM only as serving lib if needed) |
| **SFT (Qwen3.5 — 9B/27B)** | **`sft-qwen35` conda** (see §2f) | **transformers 5.3.0 / torch 2.9.1+cu130** |
| Axolotl SFT (Sera/CoderForge) | `sera-axolotl` conda | torch 2.9.1+cu130 |
| Curator datagen | `curator` conda | — |

> **The launcher chooses venv vs SIF for RL.** `hpc/sbatch_rl/universal_rl.sbatch` resolves
> `RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"` (venv path) and a container var
> (`RL_CONTAINER` / `RL_CONTAINER_OVERLAYS`). Dense FSDP2 → venv; Megatron/MoE/80B → SIF.
> **To know which a specific job used, read its rendered sbatch** (`experiments/<job>/sbatch/*.sbatch`)
> for `apptainer exec … <sif>` vs the venv python — don't assume.

---

## 2. The runtimes in detail

### 2a. `otagent` conda — orchestration + local dev
- **Local Mac:** `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` (per CLAUDE.md; symlinks don't work in the sandbox — use the full path).
- **Jupiter:** `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/` (torch 2.11.0+cu130; has a vLLM `0.1.dev…041cfa68e` build).
- **Use for:** Harbor/SkyRL code, `hpc.launch`, HF uploads, eval/datagen listeners, `count_snapshots_from_tasks.py`, Supabase, trace upload + `parse_skyrl_metrics.py` (needs `google.cloud.storage` + matplotlib — the RL venv lacks these).
- **Load (Jupiter, non-interactive):** `/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python …` (full path; `$DCFT_ACTIVATE_ENV` doesn't work over non-interactive ssh).
- **Local-dev gotcha:** Harbor's `__init__.py` fails without the package installed, so to test modules in isolation locally, mock the package first: `import sys,types; m=types.ModuleType('harbor'); m.__path__=['/Users/benjaminfeuer/Documents/harbor/src/harbor']; m.__version__='0.0.0'; sys.modules['harbor']=m; sys.path.insert(0,'/Users/benjaminfeuer/Documents/harbor/src')`. System python + conda base lack loguru/pydantic — always use the otagent env.
- **`flash_attn` (FA2) is INSTALLED (2026-06-14):** version `2.8.3+cu130torch2.11` (prebuilt wheel `flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl`, from mjun0812/flash-attention-prebuild-wheels release tag **v0.9.22**: `https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.22/flash_attn-2.8.3%2Bcu130torch2.11-cp312-cp312-manylinux_2_34_aarch64.whl`). Installed `--no-deps` so torch/vLLM untouched. Exact match to otagent's torch 2.11.0+cu130 / cp312 / aarch64 / glibc 2.34. `import flash_attn; import flash_attn_2_cuda; from flash_attn import flash_attn_func` all clean → `attn: fa2` SFT configs no longer fall back to eager.

### 2b. RL venv `$WORKDIR/envs/rl` — standard dense RL training
- **Path:** `/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl` (resolved via `RL_ENV_DIR` in `hpc/sbatch_rl/universal_rl.sbatch`).
- **Versions:** vLLM `dev` (**0.16-era**), **torch 2.9.0+cu130**.
- **Use for:** the standard 8B/32B FSDP2 RL ablations (a3 series, the seqnorm/TIS/shaped/symclip/lrboost/loopshape arms). This is the **default RL rollout+train runtime** — and the one DCP/CP-on-0.20.2rc0 do NOT live in.
- **Load:** `$WORKDIR/envs/rl/bin/python …` (it's a venv, NOT a conda env — `conda activate` is wrong; see memory `reference_jupiter_rl_venv_path`).

### 2c. `skyrl_megatron_vllm_r3baked.sif` — MoE + prod 80B RL (OLD stack)
- **Path:** `/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm_r3baked.sif` (9.7 GB, 2026-06-08).
- **Versions:** vLLM **0.16.0**, **torch 2.9** (NGC base), transformers 5.10.1. **Overlays baked in** (`vllm_http_overlay` + `fla_tilelang_overlay`) — no `--overlay` needed (see memory `reference_80b_baked_sif_overlay_fix`).
- **Use for:** Qwen3-Coder-30B-A3B MoE RL, and the **production 80B** Qwen3-Next-80B-A3B R3+TIS run (validated job 662928).
- **Load:** `apptainer exec --nv <sif> python …` (the `--nv` + baked `LD_LIBRARY_PATH`/`TRITON_LIBCUDA_PATH` are required or TransformerEngine import fails on `libcuda.so` — see `reference_skyrl_megatron_container`).

### 2d. `skyrl_megatron_vllm0202rc0_r3.sif` — NEW torch2.11 / vLLM 0.20.2rc0 ⭐
- **Path:** `/e/scratch/jureap59/feuer1/containers/skyrl_megatron_vllm0202rc0_r3.sif` (11.6 GB, 2026-06-12). **NEWEST, active.**
- **Versions (verified 2026-06-13):** vLLM **0.20.2rc0**, **torch 2.11.0+cu130**. `decode_context_parallel_size` field present; `get_dcp_group` imports. Source = `mlfoundations/vllm` `v0.20.2rc0-306-g3e3a1c45d` (local tree at `/Users/benjaminfeuer/Documents/vllm`, branch `v2-migration`).
- **Use for:** the FSDP2 torch-native **CP** feature (Stage 1+ pinned torch≥2.10 here), the vLLM **DCP** feature, the Mixtral-8x7B multi-node stand-up (first multi-node run on this SIF, 2026-06-12). **This is the runtime any DCP / CP / torch≥2.10 test MUST use** — NOT the venv or the r3baked SIF.
- **Load:** `apptainer exec --nv <sif> python …`. Migrating multi-node RL here surfaces 3 torch-2.11 fixes the OLD SIF doesn't need (vLLM-bind-removal when R3 off, `NCCL_P2P_DISABLE=1`, `pg_options→backend_options`) — see `reference_new_sif_torch211_multinode_fixes`. vLLM lives at `/opt/vllm_build/vllm/` here (NOT `dist-packages/vllm`), which shadows R3 single-file binds — remove the binds when R3 is OFF.
- **GOTCHA — set `VLLM_USE_FLASHINFER_SAMPLER=0` for any direct `vllm.LLM`/engine call on this SIF.** This SIF ships **no `flashinfer`**, but vLLM 0.20.2rc0 defaults `VLLM_USE_FLASHINFER_SAMPLER=True` and `TopKTopPSampler.__init__` unconditionally imports `FlashInferBackend` (→ top-imports flashinfer) when the flag is on — crashing with `ModuleNotFoundError: No module named 'flashinfer'` *before* its graceful fallback. Setting `VLLM_USE_FLASHINFER_SAMPLER=0` uses the PyTorch/Triton sampler (irrelevant to greedy parity). Bit any direct-engine smoke (DCP Stage-3); the normal SkyRL launch path may not hit it, but any standalone `vllm.LLM(...)` on this SIF will.
- **GOTCHA — multi-GPU (tp>1) spawned workers fail Triton linking** with `ld: cannot find -l:libcuda.so.1` on GH200/aarch64. Fix: add the `--nv` driver-stub dir to the linker path — `export LIBRARY_PATH=/.singularity.d/libs:${LIBRARY_PATH:-}`. Needed for any ≥2-GPU vLLM smoke on this SIF (DCP parity / Stage-1 instrumentation hit this).
- **GOTCHA — `VLLM_ATTENTION_BACKEND` is a dead/ignored env var on 0.20.2rc0** (logs "Unknown vLLM environment variable"); the engine auto-selects FlashAttention-3. Don't rely on it to pin a backend.

### 2e. OLD SIFs — DELETED 2026-06-13 ✅
- `skyrl_megatron_vllm.sif` (9.5 GB, 2026-06-04) and `skyrl_megatron.sif` (8.7 GB, 2026-06-04)
  were **deleted 2026-06-13** (~18 GB reclaimed). Pre-delete safety check: the only remaining
  references were **stale comments** (80B yaml line 536, `rl_launch_utils.py` overlay-logic
  comments, the Polaris `ALCC/96GPU_base_80b.yaml`) — the 80B yaml's *active* `container.sif`
  (line 599) is `skyrl_megatron_vllm_r3baked.sif`, and no running job referenced the old SIFs.
- As of 2026-06-13 only two SIFs remained: `skyrl_megatron_vllm0202rc0_r3.sif` (§2d) and
  `skyrl_megatron_vllm_r3baked.sif` (§2c). **SUPERSEDED — see §2g**: the CP-variant SIF chain
  (`_cp` → `_cp_fixb` → `_cp_fixb2` → `_cp_fixb3`) plus `_r3_fixb` / `_r3_v2migration` were
  built afterward off §2d for the #232 FSDP2-CP work.

### 2g. CP-variant SIF chain — `skyrl_megatron_vllm0202rc0_r3_cp*.sif` (#232 FSDP2-CP) ⭐
All built off §2d (`vllm0202rc0_r3`, torch 2.11 / vLLM 0.20.2rc0) for the #232 FSDP2 context-parallel
(ring-SDPA) + R3 work. Path prefix: `/e/scratch/jureap59/feuer1/containers/`. Load like §2d
(`apptainer exec --nv … python …`; same flashinfer/libcuda/attention-backend gotchas apply).
- **`skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif`** (11.6 GB, 2026-06-19) — ⭐ **CANONICAL CP+R3 SIF.**
  `.vllm_commit = 4d167a4af` (`penfever/working` with the merged **#237** rank-symmetric R3-capture
  fix baked in). **Use this for ALL new CP and/or R3 runs** (the #232 cp2 / cp2_r3 rungs point here).
- `skyrl_megatron_vllm0202rc0_r3_cp_fixb2.sif` — ❌ **DELETED 2026-06-19** (~11.6 GB reclaimed). Superseded
  by `_cp_fixb3` (lacked the #237 fix). Removed after the cp2 chain (926043-048) was cancelled at the
  CP-MoE-forward-bug hold point; no live job referenced it. Use `_cp_fixb3` for all CP/R3 runs.
- `_cp_fixb.sif` / `_cp.sif` (earlier 2026-06-19) — older CP-iteration SIFs, superseded by `_cp_fixb3`.
- `_r3_fixb.sif` / `_r3_v2migration.sif` — non-CP R3 variants (NOT rebuilt with the #237 fix as of
  2026-06-19; a separate rebake is needed if a non-CP R3 run must carry the fix).
- **KNOWN CP BUG (2026-06-19, separate from the SIF — it's a SkyRL fix):** Qwen3-**MoE** crashes in the
  CP≥2 policy forward (`model_wrapper.py:668` passes a dict attention_mask that MoE's `create_causal_mask`
  can't consume → `AttributeError: 'dict' has no attribute 'ndim'`). Dense-Qwen3 & CP1 are unaffected.
  Fix is in the SkyRL host clone (`OpenThoughts-Agent/SkyRL/skyrl-train/skyrl_train/model_wrapper.py`),
  NOT the SIF — editable install, no SIF rebuild. See `agent_logs/2026-06-19_cp2_forward_dict_ndim_bug.md`.

### 2f. `sft-qwen35` conda — Qwen3.5 SFT (and the other SFT/datagen condas)
- **Path (Jupiter):** `/e/scratch/jureap59/feuer1/miniforge3/envs/sft-qwen35/`
- **Versions (Jupiter-verified 2026-06-13):** **torch 2.9.1+cu130, transformers 5.3.0.** (Uses deepspeed `zero3_bf16` for sharding; the Leonardo recipe in CLAUDE.md cites deepspeed ≥0.18 / torch ≥2.10 — expect minor per-cluster drift, so probe before assuming.)
- **Why a separate env:** **Qwen3.5 (9B/27B) uses a hybrid GatedDeltaNet + Attention architecture that does NOT exist in transformers 4.x.** It requires **transformers ≥ 5.3**, so the default LLaMA-Factory/`otagent` SFT stack (transformers 4.x) physically cannot load or train Qwen3.5 — hence this dedicated env. Use it for **any `sft/lf_configs/qwen3_5/*` run** (9B/27B at 32k/131k).
- **Load:** `/e/scratch/jureap59/feuer1/miniforge3/envs/sft-qwen35/bin/python …` (conda env under `miniforge3/envs`; full path over non-interactive ssh). The launcher (`hpc.launch --train_config_path sft/lf_configs/qwen3_5/...`) wires conda activation on Jupiter; on **Leonardo** you must hand-patch the sbatch (conda activate + WORKDIR) per CLAUDE.md "SFT Launch on Leonardo".
- **Launch + cleanup specifics:**
  - Pass the Harbor role/content tags: `--role_tag role --user_tag user --assistant_tag assistant --content_tag content` (or the thinking preprocessor finds 0 assistant messages → garbage).
  - Qwen3.5 writes **full safetensors at the checkpoint root on completion → SKIP the `consolidate` step** (per memory `feedback_qwen35_9b_no_consolidate`); follow the **8B SFT cleanup checklist**, not the 32B one.
  - **Before HF upload, copy `preprocessor_config.json` from the base model** into the checkpoint (LLaMA-Factory doesn't emit it; vLLM needs it) — 8B-checklist step 1b.
  - HF-only / DB-register per the SFT checklist; uploads default PUBLIC to `laion/`.

**Other condas on Jupiter (brief):**
- `sera-axolotl` — Sera/CoderForge axolotl SFT (torch 2.9.1+cu130; see CLAUDE.md "Axolotl SFT on Jupiter" for the install recipe + mandatory env patches).
- `curator` — Curator sharded datagen jobs only (`run_curator_datagen_sharded.sbatch`, `--account=reformo`).

---

## 3. Overlay images (`containers/*.img`)

Overlays stack onto a SIF at `apptainer exec --overlay <img>` (or are baked in). On GPFS they can
FUSE-mount-timeout on the Ray head — the prod 80B SIF bakes them in to avoid that.

| Overlay | Size | Provides | Status |
|---|---|---|---|
| `vllm_http_overlay.img` | 0.5 G | vLLM HTTP `routed_experts` serialization (R3 routing capture) | baked into both `*r3*.sif` |
| `fla_tilelang_overlay.img` | 4 G | tilelang 0.1.8 + FlashQLA fused GatedDeltaNet kernels (FSDP2-EP Stage 8) | baked into `*r3*.sif` |
| `skyrl_titan_overlay.img` | 4 G (2026-06-13) | torchtitan (for CP **+EP** / MoE expert-parallel path — CP Stage-6 TEST3) | overlay (stack when CP+EP) |

**R3 routing-capture works on the existing SIF (no rebuild) — RESOLVED 2026-06-08.** The
`vllm_http_overlay` already serializes `routed_experts` over `/chat/completions` on stock-0.16; the only
thing that ever blocked R3 capture was the flag `enable_return_routed_experts=False` (now true). For
**Qwen3-Next** specifically, the vLLM **Ray Compiled-DAG** backend deadlocks on the hybrid arch when
capture is on → run with the **mp executor backend** (`generator.inference_engine_mp_backend: true`),
which ran the identical config clean (12/12 rounds, TP=4). Also needed: an undersized hybrid-kv-buffer
fix + defensive clip (`gmr_fix`/`scheduler_fix`/`capturer_fix` single-file binds; SkyRL flag on branch
`r3-mp-backend-qwen3next-20260608`). The FSDP2 router-replay hook EXISTS and ran a full GRPO backprop step
on the 80B (do NOT repeat the "Megatron-only / no FSDP2 replay" claim).

---

## 4. VERIFY before you trust — copy-paste probes

**SIF:**
```bash
C=/e/scratch/jureap59/feuer1/containers
apptainer exec --nv $C/<sif>.sif python -c "import vllm,torch; print('vllm',vllm.__version__,'torch',torch.__version__); \
from vllm.engine.arg_utils import EngineArgs; print('DCP field', 'decode_context_parallel_size' in EngineArgs.__dataclass_fields__)"
```
**venv / conda:**
```bash
/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl/bin/python -c "import vllm,torch; print(vllm.__version__, torch.__version__)"   # → dev 2.9.0  (OLD)
/e/scratch/jureap59/feuer1/miniforge3/envs/otagent/bin/python   -c "import torch; print(torch.__version__)"                          # → 2.11.0     (NEW)
```
**Read torch first:** 2.9 ⇒ 0.16-era (no DCP/CP); 2.11 ⇒ 0.20.2rc0 (has DCP/CP). If a feature that
requires torch≥2.10 "isn't there" or "fails parity," confirm you're on a torch-2.11 runtime before
concluding it's a real defect.

---

## 5. Known gotcha that motivated this doc

The vLLM **DCP rollout-parity** test (Stage 3 of the DCP plan) was run **twice on torch-2.9 / vLLM-0.16
runtimes** (the RL venv + the r3baked SIF) and reported a NO-GO — but DCP's real target is the
**torch-2.11 / 0.20.2rc0 SIF** (`skyrl_megatron_vllm0202rc0_r3.sif`), which was never tested in those
runs. Always pin the parity/feature smoke to the SIF that actually carries the feature (§2d), and
record the verified `vllm`+`torch` versions in the result.
