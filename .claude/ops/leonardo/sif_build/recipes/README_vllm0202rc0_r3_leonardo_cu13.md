# Leonardo runtime: `skyrl_megatron_vllm0202rc0_r3` — the TRUE cu13/torch-2.9 twin

Leonardo (CINECA) analogue of Jupiter's prod SIF `skyrl_megatron_vllm0202rc0_r3.sif`,
built on the **SAME NGC 25.09 / CUDA-13 / torch-2.9 base** as Jupiter for **true
cross-cluster parity** — made possible on Leonardo's older A100 driver by **CUDA
forward compatibility**. This supersedes the torch-2.8 fallback recipe
(`README_vllm0202rc0_r3_leonardo.md`), which remains documented as the sanctioned
fallback but is NOT the path taken.

Carries the same logical stack — SkyRL editable (`penfever/SkyRL @ 2ab513a6`) +
Megatron-core 0.14.0 + TransformerEngine + flash-attn + our canonical vLLM fork
`penfever/working @ 5d7319dd1` (0.20.2rc0 + native R3 routed-experts capture + the
DCP GQA-LSE fp32 fix) — built for **x86_64 / A100 (`TORCH_CUDA_ARCH_LIST=8.0`)**.

---

## The forward-compat gate (STAGE 1 — PASSED 2026-06-16)

Leonardo A100 booster nodes load NVIDIA **kernel driver `535.274.02`** (native CUDA
≤12.2; verified `srun nvidia-smi`). `singularity --nv` binds *that host* kernel driver
and a container cannot replace the kernel module. **But the CUDA *toolkit* can be 13.0
via forward compatibility:** NGC cu13 images bundle `cuda-compat-13` at
`/usr/local/cuda-13.0/compat/lib.real/` — a **newer userspace** `libcuda.so.580.82.07`
that lets a cu13 toolkit run on an older **datacenter** driver. The A100 is a datacenter
GPU, so it qualifies.

**Empirically verified on an A100 under the 535 driver:**
- Base built as a sandbox dir: `singularity build --sandbox $WORK/containers/pytorch_2509_sbx docker://nvcr.io/nvidia/pytorch:25.09-py3` (19 G).
  - ⚠️ A **packed `.sif` pull FATALs**: `while creating squashfs: create command failed: signal: killed` (login-node mksquashfs OOM/kill + `lustre.lov` xattr errors). The `--sandbox` build skips the squash step and succeeds — **provided `TMPDIR`/`SINGULARITY_TMPDIR` are forced onto GPFS `$WORK`**; the default `TMPDIR=/scratch_local` is Lustre and triggers the `lustre.lov` xattr storm.
- `srun … boost_qos_dbg --gres=gpu:1`, then `singularity exec --nv -B /leonardo_work pytorch_2509_sbx python -c "<matmul>"`:
  - `torch 2.9.0a0+…nv25.09`, `torch.version.cuda == 13.0`, `is_available True`, cap `(8, 0)`.
  - A real fp32 `2048²` matmul executed: `maxerr=6.8e-2` (normal tf32 tolerance — real tensor-core compute, not a stub).
  - `/proc/self/maps` confirms torch loaded **`/usr/local/cuda-13.0/compat/lib.real/libcuda.so.580.82.07`** (the forward-compat userspace), NOT a host 535 libcuda.
- **Conclusion:** the 535 branch is **within cu13's forward-compat minimum-driver floor** on the A100. The earlier "CUDA-13 infeasible" verdict (in the torch-2.8 README) was wrong — it conflated the host kernel driver's native ceiling with the toolkit and never tried the in-container compat libs.

**The NGC image wires the compat libs via its own ldconfig** under `singularity exec --nv`,
so no manual `LD_LIBRARY_PATH` reorder was even needed. The run convention still EXPORTs
`SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat/lib.real` defensively (see below).

---

## Packaging: writable SANDBOX DIR on WORK, NOT a packed `.sif`

Same as the MarinSkyRL runtime + the fallback recipe: `mksquashfs` OOM-kills / hits
`lustre.lov` xattr errors on the Lustre login node, so `.sif` packing is deferred. We
build/exec a **writable singularity sandbox directory** on **WORK (GPFS)**, not Lustre.

```
/leonardo_work/AIFAC_5C0_290/bfeuer00/containers/skyrl_megatron_vllm0202rc0_r3_sandbox/
```

**Target WORK, not SCRATCH_FAST** (SCRATCH_FAST is 3.7 T / 1 T over quota, grace=none).
Builder = SingularityPRO 4.3.1 `/usr/bin/singularity` (NO apptainer/podman/root).

---

## Ingredients

| Ingredient | Identity | Notes |
|---|---|---|
| **Base image** | `docker://nvcr.io/nvidia/pytorch:25.09-py3` (built as sandbox `pytorch_2509_sbx`) | x86_64, CUDA 13.0, torch 2.9.0a0+nv25.09, Python 3.12, TE, apex. **SAME base as Jupiter** → true parity. Runs on the 535 A100 driver via the bundled `cuda-compat-13`. NGC pull/build on the **login node** (compute has none); force `TMPDIR`/`SINGULARITY_TMPDIR` onto GPFS WORK. |
| **Megatron-core (phase A)** | Megatron-core `0.14.0` (match Jupiter) — **prebuilt cp312 x86_64 wheel, pre-staged in the offline `accel_wheels` wheelhouse**, installed `--no-deps` | **NOT in the NGC base** (verified: `ModuleNotFoundError: No module named 'megatron'`), so it cannot come "for free" from the base; and compute nodes have no internet, so it must be a pre-downloaded offline wheel. `--no-deps` because its only non-extra runtime deps (`torch`, `numpy`, `packaging`) are already in the base (we keep the base's numpy 2.1.0 over megatron's conservative `numpy<2.0.0` pin — same as Jupiter's prod stack). |
| **flash-attn (phase A)** | from the NGC base if present, else accel wheelhouse, else source (sm_80) | NGC 25.09 already ships flash-attn; phase A only compiles it if absent. |
| **SkyRL editable (phase A)** | `penfever/SkyRL @ 2ab513a6`, editable at `/opt/SkyRL` | Match Jupiter prod SIF. Cloned on the login node. (Leonardo's live RL today is MarinSkyRL `penfever/working`; this twin pins the **Jupiter prod-SIF** SkyRL `2ab513a6` for cross-cluster parity — swap the pin if parity should instead track MarinSkyRL.) |
| **vLLM fork (phase B)** | `penfever/working @ 5d7319dd100b424c73d1bb9b2ba7b52a44ee811b` | Built **from source against the in-base NGC torch 2.9 (cu13)** via `use_existing_torch.py` (strips the fork's `torch==2.11.0` pin). `TORCH_CUDA_ARCH_LIST=8.0`. Carries native R3 + the DCP GQA-LSE fp32 fix (no separate patch). |
| **GDN overlay** | **OMITTED** (deferred) | Same as Jupiter's FlashQLA/tilelang overlay — no Leonardo image; Qwen3-Next still runs the vanilla GDN path. Follow-up if Qwen3-Next RL is needed on Leonardo. |

---

## Login-node prep (manual, before sbatch — compute nodes have no internet)

1. **Build the cu13 base sandbox** (NOT a .sif): with `SINGULARITY_TMPDIR`/`SINGULARITY_CACHEDIR`/`TMPDIR` on GPFS WORK,
   `singularity build --sandbox $WORK/containers/pytorch_2509_sbx docker://nvcr.io/nvidia/pytorch:25.09-py3`.
2. **Clone the vLLM fork**: `git clone --branch penfever/working https://github.com/mlfoundations/vllm.git $WORK/sif_build/vllm_clone && (cd … && git checkout 5d7319dd1)`.
3. **Clone SkyRL**: `git clone https://github.com/penfever/SkyRL.git $WORK/sif_build/skyrl_clone && (cd … && git checkout 2ab513a6)`.
4. **Offline wheelhouses** built with the **cu13 base image's own pip** (ABI match) via `singularity exec pytorch_2509_sbx pip download …`. Pip refuses to write to the sandbox-mounted `/leonardo_work` (read-only without `--writable`); **bind the target dir** to a writable in-container path, e.g. `-B $ACCEL_WHEELS:/whl … -d /whl`:
   - `build_deps_wheels`: `setuptools_scm setuptools wheel packaging`.
   - `runtime_deps_wheels`: the vLLM `common.txt` pure-python deps (named in the sbatch's two-pass install) + `opencv-python-headless --no-deps`. (The resolver also drops torch/numpy/nvidia wheels into the dir; they are NOT installed — the install commands name only the pure-python deps and use `--no-deps`/`use_existing_torch`.)
   - `accel_wheels` (**MANDATORY** — megatron-core is NOT in the NGC base):
     ```bash
     singularity exec --no-home -B "$ACCEL_WHEELS":/whl "$BASE_SBX" \
       python -m pip download "megatron-core==0.14.0" --no-deps --only-binary=:all: -d /whl
     ```
     This fetches the prebuilt `megatron_core-0.14.0-cp312-cp312-manylinux_2_24_x86_64…whl` (~2.2 MB). Optionally also stage a prebuilt `flash_attn` wheel here if you want to skip the base's flash-attn / a source compile.

---

## Build steps (encoded in `build_vllm0202rc0_r3_leonardo_cu13.sbatch`)

Run in **sbatch** (`boost_usr_prod`, normal QOS — the `boost_qos_dbg` ≤30 min cap is too
short for the ~1.5–2.5 h vLLM compile; give 6 h wall). Steps:
1. `cp -a pytorch_2509_sbx → skyrl_megatron_vllm0202rc0_r3_sandbox` (the base is already a
   sandbox dir, so we copy — no `singularity build` / .sif conversion).
2. Stage SkyRL + vLLM fork + wheelhouses into `/opt/…` in the sandbox.
3. **Phase A** (`singularity exec --writable`, no `--nv`): Megatron-core 0.14.0 **from the
   offline `accel_wheels` wheelhouse, `--no-deps`** (NOT a pip-install-from-internet — compute
   has no network); flash-attn (base, else wheelhouse, else source); SkyRL `-e ./skyrl-train`
   + `./skyrl-gym`.
4. **Phase B** (`singularity exec --writable`, no `--nv`): `use_existing_torch.py`; install
   build-deps + runtime-deps OFFLINE; `pip install --no-build-isolation --no-deps -v -e .`
   with `TORCH_CUDA_ARCH_LIST=8.0`, `SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc0`, `MAX_JOBS=24`.
5. Validate under `--nv` with the compat LD path exported.

---

## Acceptance / validation (asserted in the sbatch)

In-sandbox (`SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat/lib.real singularity exec --nv`):
- `torch.__version__` ~ `2.9.0a0…nv25.09`, `torch.version.cuda == 13.0`, `is_available`, cap `(8, 0)`.
- `LOADED_LIBCUDA` == the compat `libcuda.so.580.82.07` (forward-compat confirmed at runtime).
- `vllm.__version__` renders `0.20.2rc0` (dev tree may show `0.20.2rc0.devN+g5d7319dd1`).
- `ModelRegistry.get_supported_archs()` ⊇ `Gemma4ForConditionalGeneration`, `Gemma4ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`.
- R3 native: `routed_experts` in the four chat/completion serving+protocol files + `routed_experts_capturer.py` exists.
- DCP fp32 fix baked: `out_fp32` in `v1/attention/ops/common.py` + `_forward_with_dcp` `out_fp32=True` in `v1/attention/backends/flash_attn.py`.
- `import skyrl_train, skyrl_gym`, `megatron.core`, `flash_attn` OK.

**Run convention (cu13 forward-compat):**
```bash
SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat/lib.real \
  /usr/bin/singularity exec --nv \
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env LIBRARY_PATH=/.singularity.d/libs \
  $WORK/containers/skyrl_megatron_vllm0202rc0_r3_sandbox <cmd>
```
(`LIBRARY_PATH=/.singularity.d/libs` for tp>1 Triton linking; flashinfer sampler off — none baked.)

**DCP parity smoke (A100, separate sbatch — do NOT launch a full RL run):** 2-node tp=8,
dcp=1 vs dcp=2, Qwen3-Coder-30B-A3B, greedy temp=0, R3 on → expect token mismatch ≈ **6.94%**
(the validated DCP+R3 parity on `penfever/working`; the bf16-tie floor is identical to Jupiter's).

---

## Notes / gotchas

- **megatron-core is NOT in the NGC 25.09 base** and compute nodes have no internet — it MUST be a pre-staged offline wheel in `accel_wheels`, installed `--no-deps`. The first cu13 build attempt (job `47061331`, 2026-06-17) FAILED here: `accel_wheels` was empty, so phase A fell to `pip install megatron-core==0.14.0` which hit `Network is unreachable` → `No matching distribution found`. Fix = stage the wheel on the login node (prep step 4) + the preflight now hard-FATALs if it's missing.
- **Compilers come from the NGC base** (nvcc 13.0 + matching gcc). Do NOT pull conda gcc/nvcc into the sandbox.
- `use_existing_torch.py` MUST run before `pip install` or pip pulls torch 2.11 and breaks TE/apex/flash-attn.
- Build scratch (`SINGULARITY_TMPDIR`/`TMPDIR`) on WORK GPFS — NOT Lustre `/scratch_local` (xattr storm), NOT over-quota SCRATCH_FAST, NOT tiny node-local `/tmp`.
- vLLM kernel compile is the long pole (~1.5–2.5 h on 32 Ice Lake cores, sm_80). `boost_qos_dbg` (≤30 min) is too short — use normal QOS, 6 h wall.
- Do NOT disturb the in-flight Delphi SFT / delphi+pass@k evals or the otagent / MarinSkyRL / sft-qwen35 envs — this is a NEW sandbox alongside them.
