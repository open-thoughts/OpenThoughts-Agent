# Leonardo runtime: `skyrl_megatron_vllm0202rc0_r3` — the x86/A100 cross-cluster twin

Leonardo (CINECA) analogue of Jupiter's prod SIF `skyrl_megatron_vllm0202rc0_r3.sif`
(`.claude/ops/jupiter/sif_build/recipes/README_vllm0202rc0_r3_sif.md`). Carries the
**same logical stack** — SkyRL editable + Megatron-core + TransformerEngine + our
canonical vLLM fork `penfever/working @ 5d7319dd1` (0.20.2rc0 mainline + R3 native
routed-experts capture + the DCP GQA-LSE fp32 fix) — so models + DCP/CP features are
cross-cluster compatible. **Built for x86_64 / A100 (`TORCH_CUDA_ARCH_LIST=8.0`)**,
not GH200/aarch64.

> ⚠️ **NOT a byte-for-byte twin — it CANNOT be.** See "Feasibility / the CUDA-13
> blocker" below. The hard difference is the base: Jupiter rides **NGC 25.09 /
> CUDA 13.0 / torch 2.9.0a0+nv25.09**; Leonardo's GPU driver caps it at CUDA 12.x,
> so this twin rides **NGC 25.06 / CUDA 12.9.1 / torch 2.8.0a0**. The vLLM fork
> commit, R3 capture, DCP fp32 fix, model archs, and the SkyRL/Megatron/TE training
> stack are matched; the torch/CUDA floor is one NGC step back.

---

## Feasibility / the CUDA-13 blocker (the #1 gate — settled 2026-06-16)

**Leonardo A100 booster nodes run NVIDIA driver `535.274.02` → max CUDA `12.2`**
(verified via `srun … nvidia-smi`, job 47057124). System `module avail` tops out at
`cuda/12.6`; there is **no CUDA 13 module and no `cuda-compat` package** on the system,
and the driver is admin-pinned (a CINECA action, out of scope).

- **Why NGC 25.09 / CUDA 13.0 (Jupiter's base) is INFEASIBLE here.** CUDA *Enhanced
  Compatibility* lets a newer CUDA **minor** run on an older **same-major** driver —
  it does **NOT** bridge a major bump. CUDA 13.0 on a 12.2 (535) driver fails at
  context init. NGC 25.07/25.08 already cut over to CUDA 13.0, so 25.06 is the newest
  NGC PyTorch container that stays in CUDA major 12.
- **Why CUDA 12.9 (NGC 25.06) IS feasible.** Verified empirically: the existing
  `otagent` env runs **torch 2.9.1+cu128 (CUDA 12.8)** with real matmul kernels on
  this exact 535 driver (capability (8,0), A100). cu128/cu129 are CUDA-major-12 and
  run on the 535 driver by minor-version forward-compat — the standard, supported
  path. NGC 25.06 = CUDA 12.9.1 sits in the same regime.
- **Closest-achievable base chosen: `nvcr.io/nvidia/pytorch:25.06-py3`** — Ubuntu
  24.04, CUDA 12.9.1, Python 3.12, **torch 2.8.0a0+5228986c39**, TransformerEngine
  2.4, apex, cuDNN/NCCL from the CUDA-DL 25.06 line. It does **not** bundle
  Megatron-core, flash-attn, or SkyRL — those are added in the build (phase A).

**Decision:** build the closest-achievable twin on NGC 25.06 (torch 2.8 / cu12.9).
This is one NGC minor behind Jupiter's torch 2.9. If true torch-2.9/CUDA-13 parity is
required, that is **blocked on a Leonardo driver upgrade to ≥580.65** — a human/CINECA
decision, not buildable from our side. Report-and-stop, do not force an incompatible
CUDA-13 base.

---

## Packaging: writable SANDBOX DIR on WORK, NOT a packed `.sif`

Per the env map + the MarinSkyRL runtime: `mksquashfs` OOM-kills on the Lustre login
node and hits fatal `lustre.lov` xattr errors, so `.sif` packing is deferred. We build
a **writable singularity sandbox directory** and `singularity exec --nv` it directly
(same as `marinskyrl_sandbox`).

**Target WORK, not SCRATCH_FAST.** SCRATCH_FAST `/leonardo_scratch/fast/AIFAC_5C0_290`
is **3.7 TB used against a 1 TB quota (grace=none, already over)** as of 2026-06-16 — do
not stage a multi-hundred-GB sandbox there. WORK `/leonardo_work/AIFAC_5C0_290`
(9.9 T / 100 T) has headroom and is persistent (where containers/envs live). GPFS has no
Lustre xattr issue. Output:

```
/leonardo_work/AIFAC_5C0_290/bfeuer00/containers/skyrl_megatron_vllm0202rc0_r3_sandbox/
```

Builder = SingularityPRO 4.3.1 `/usr/bin/singularity` (NO apptainer, NO podman, NO root,
`--fakeroot` not assumed). The sandbox build + the in-sandbox `pip install` run with
`singularity build --sandbox` / `singularity exec --writable`.

---

## Ingredients

| Ingredient | Identity | Notes |
|---|---|---|
| **Base image** | `docker://nvcr.io/nvidia/pytorch:25.06-py3` | x86_64, CUDA 12.9.1, torch 2.8.0a0+5228986c39, Python 3.12, TE 2.4, apex. **Closest CUDA-12 NGC base** (see feasibility). NGC pull needs internet → pull on the **login node** (compute has none). NGC may need an `NGC_API_KEY`/`SINGULARITY_DOCKER_*` login for `nvcr.io`; the base is public-ish but rate-limited — if pull fails, `singularity remote login docker://nvcr.io`. |
| **Megatron-core + flash-attn (phase A)** | Megatron-core `0.14.0` (match Jupiter), `flash-attn 2.7.4.post1` (match Jupiter prod SIF) | NOT in NGC 25.06 → built/installed into the sandbox. flash-attn for x86/A100 sm_80: prefer a prebuilt cu12/torch2.8 x86 wheel if available; else `pip install flash-attn --no-build-isolation` (long compile; A100 sm_80). |
| **SkyRL editable (phase A)** | `penfever/SkyRL @ 2ab513a6`, editable at `/opt/SkyRL` | Match Jupiter prod SIF. rsync the local SkyRL clone or `git clone`+`checkout 2ab513a6` on the login node, then `pip install -e .` for `skyrl_train` + `skyrl_gym`. (Leonardo's RL runtime today is MarinSkyRL `penfever/working`; this twin pins the **Jupiter prod SIF's** SkyRL `2ab513a6` for cross-cluster parity. If parity should instead track MarinSkyRL, swap the pin — flagged, not assumed.) |
| **vLLM fork (phase B)** | `penfever/working @ 5d7319dd100b424c73d1bb9b2ba7b52a44ee811b` (`git describe` = `v0.20.2rc0-328-g5d7319dd1`) | Built **from source against the in-sandbox NGC torch 2.8** via `use_existing_torch.py` (strips the fork's `torch==2.11.0` pin in `requirements/cuda.txt`). `TORCH_CUDA_ARCH_LIST=8.0` (A100). Carries native R3 (`routed_experts_capturer.py` + emission in the four chat/completion serving+protocol files) AND the DCP GQA-LSE fp32 fix (`v1/attention/ops/common.py` fp32 accumulate + `out_fp32`; `v1/attention/backends/flash_attn.py` `_forward_with_dcp` out_fp32). **No separate patch** — the fork carries both natively. |
| **GDN overlay** | **OMITTED** (deferred) | Jupiter merges `fla_tilelang_overlay.img` (FlashQLA/tilelang/tvm_ffi) for fused Qwen3-Next GatedDeltaNet. No such image exists on Leonardo; building FlashQLA for sm_80 is a separate effort. Qwen3-Next still **runs** (vanilla GDN path); only the fused fast-path is absent. Follow-up if Qwen3-Next RL is needed on Leonardo. |

---

## Why build vLLM from source (not a wheel)

Same rationale as Jupiter: the only prebuilt 0.20.2 wheels are `cu130torch2.11`, which
mismatch this sandbox's torch 2.8 / cu12.9 and would shatter TE/apex/flash-attn (no NGC
2.11 cu12 wheel). So compile the fork from source against the in-sandbox NGC torch 2.8
with `use_existing_torch.py`, exactly as Jupiter does against its NGC torch 2.9.

---

## Build steps (encoded in `build_vllm0202rc0_r3_leonardo.sbatch`)

Run in **sbatch** (NOT the login node — login killer + fragility). The NGC pull and the
offline-wheelhouse pre-download are **login-node** prep (compute has no internet).

**Login-node prep (manual, before sbatch):**
1. `singularity pull` / `build` the NGC 25.06 base `.sif` into `$WORK/containers/` (needs internet).
2. rsync the vLLM fork source (`penfever/working @ 5d7319dd1`, exclude `.git/build/*.so`) to `$WORK/sif_build/vllm_src/`.
3. rsync/clone SkyRL `@ 2ab513a6` to `$WORK/sif_build/skyrl_src/`.
4. Pre-download offline wheelhouses (build-deps + vLLM runtime pure-python deps) using the **base image's own pip** for ABI match (same two wheelhouses Jupiter uses — `setuptools_scm`/`setuptools`/`wheel`/`packaging` + the ~32 missing `common.txt` pure-python deps, excluding torch/triton/transformers/tokenizers/numpy/pillow/nvidia). Optionally a flash-attn + megatron-core wheel.

**sbatch (compute node, offline):**
1. `singularity build --sandbox $WORK/containers/…_sandbox $WORK/containers/pytorch_25.06.sif`.
2. **Phase A** (`singularity exec --writable`): install Megatron-core 0.14.0, flash-attn 2.7.4.post1 (sm_80), SkyRL `-e /opt/SkyRL` (skyrl_train+skyrl_gym). Confirm `import skyrl_train, skyrl_gym`.
3. **Phase B** (`singularity exec --writable`): stage vLLM fork into `/opt/vllm_build`; `use_existing_torch.py`; install build-deps + runtime-deps OFFLINE from the staged wheelhouses; `pip install --no-build-isolation --no-deps -v -e .` with `TORCH_CUDA_ARCH_LIST=8.0`, `MAX_JOBS` bounded, `SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc0`.
4. Validate in-sandbox (see acceptance).

The Jupiter sbatch's hard-won offline-build lessons are reused verbatim (escaped-quotes
only in the `bash -lc` block; `--no-deps` so pip never pulls torch 2.11 / clobbers numpy;
two-pass runtime-dep install; non-fatal version probes). See its header comments.

---

## Acceptance / validation (asserted in the sbatch + a separate A100 parity smoke)

In-sandbox (`singularity exec --nv`):
- `torch.__version__` ~ `2.8.0a0…`, `torch.version.cuda` = `12.9`, `torch.cuda.is_available()`, capability `(8, 0)`.
- `vllm.__version__` renders `0.20.2rc0` (dev tree may show `0.20.2rc0.devN+g5d7319dd1`).
- `ModelRegistry.get_supported_archs()` ⊇ `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`.
- R3 native: `routed_experts` present in the four `entrypoints/openai/{chat_completion,completion}/{serving,protocol}.py` + `model_executor/layers/fused_moe/routed_experts_capturer.py` exists.
- **DCP fp32 fix baked:** grep `vllm/v1/attention/ops/common.py` for `out_fp32` + fp32-accumulate in `cp_lse_ag_out_rs`/`cp_lse_ag_out_ar`; grep `vllm/v1/attention/backends/flash_attn.py` `_forward_with_dcp` for `out_fp32=True`.
- transformers `gemma4`/`gemma4_text`/`qwen3_next`/`qwen3_moe` in `CONFIG_MAPPING_NAMES` (NGC 25.06 transformers may be < Jupiter's 5.10.1 — if an arch is missing, `pip install -U transformers` in-sandbox to match Jupiter, flagged).
- `import skyrl_train, skyrl_gym` OK.

Runtime gotchas (carried from the 0.20.2rc0 SIF on Jupiter): set
`VLLM_USE_FLASHINFER_SAMPLER=0` (no flashinfer baked), `LIBRARY_PATH=/.singularity.d/libs`
for tp>1 Triton linking; `VLLM_ATTENTION_BACKEND=FLASH_ATTN`.

**DCP parity smoke (A100, separate sbatch — do NOT launch an RL run):** 2-node tp=8,
dcp=1 vs dcp=2, Qwen3-Coder-30B-A3B, greedy temp=0, R3 on → expect token mismatch
≈ **6.94%** (the validated DCP+R3 parity on `penfever/working`). This is the
cross-cluster correctness check; the architectural bf16-tie floor is identical to
Jupiter's.

---

## Notes / gotchas

- **Compilers come from the NGC base**, which already has nvcc 12.9 + a matching gcc. Do
  NOT pull in conda gcc/nvcc inside the sandbox (that's the from-scratch-env path, not the
  NGC-base path). If a host-compiler rejection appears, add
  `-DCMAKE_CUDA_FLAGS=--allow-unsupported-compiler`.
- `use_existing_torch.py` MUST run before `pip install` or pip pulls torch 2.11 and
  breaks TE/apex/flash-attn.
- Build scratch: use the sandbox on WORK; node-local `/tmp` is too small for sandbox +
  extracts. Do not write the sandbox to over-quota SCRATCH_FAST.
- vLLM kernel compile is the long pole (~1.5–2.5 h on 32 Ice Lake cores for sm_80; A100
  build node has 32 cores/node, fewer than Jupiter's 64–72 → expect longer; give 6 h wall,
  within the 24 h `boost_usr_prod` cap). `boost_qos_dbg` (≤30 min) is too short for the
  build — use the normal QOS.
- Phase A (Megatron/flash-attn build) can itself take 30–90 min if flash-attn compiles
  from source; prefer a prebuilt x86 cu12/torch2.8 flash-attn 2.7.4 wheel to skip it.
- Do NOT disturb the in-flight delphi evals / evalchemy jobs or the otagent / MarinSkyRL /
  sft-qwen35 envs — this is a NEW sandbox alongside them.
