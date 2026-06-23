#!/bin/bash
# =============================================================================
# build_vllm0202rc0_r3_leonardo_cu13.sbatch
#
# Leonardo (CINECA, x86_64 / A100) TRUE CUDA-13 / torch-2.9 cross-cluster twin of
# Jupiter's prod SIF skyrl_megatron_vllm0202rc0_r3.sif. Builds a WRITABLE
# SINGULARITY SANDBOX DIR (NOT a packed .sif -- mksquashfs OOM/`lustre.lov` xattr
# blocker on the Lustre login node; the .sif pull FATALs at "create command
# failed: signal: killed") carrying:
#   - SkyRL editable (penfever/SkyRL @ 2ab513a6) at /opt/SkyRL
#   - Megatron-core 0.14.0 + flash-attn (sm_80)
#   - vLLM fork penfever/working @ 5d7319dd1 (0.20.2rc0 + native R3 + DCP fp32 fix),
#     built from source against the in-base NGC torch 2.9 (cu13).
#
# BASE: nvcr.io/nvidia/pytorch:25.09-py3 (CUDA 13.0, torch 2.9.0a0+nv25.09, TE, apex)
#   -- SAME base as Jupiter's twin -> TRUE parity, NOT the torch-2.8 fallback.
#   Runs on Leonardo's 535.274.02 A100 driver via CUDA FORWARD COMPATIBILITY
#   (cuda-compat-13: /usr/local/cuda-13.0/compat/lib.real/libcuda.so.580.82.07).
#   STAGE-1 GATE PASSED 2026-06-16: a real cu13 matmul executed on an A100 under
#   this driver; torch loaded the compat libcuda.so.580.82.07 (see
#   README_vllm0202rc0_r3_leonardo_cu13.md / ENVIRONMENT_MAP.md §2d). The NGC image
#   wires the compat libs via its own ldconfig under `singularity exec --nv`, so no
#   manual LD reorder was needed; we still EXPORT the compat path defensively below.
#
# A100, so TORCH_CUDA_ARCH_LIST=8.0 (NOT 9.0). SingularityPRO 4.3.1, no apptainer,
# no root. Run via sbatch (NOT the fragile login node). Compute nodes have NO
# internet -> the NGC sandbox build, the fork/SkyRL clones, and the offline
# wheelhouses are prepared on the LOGIN node first (see README "Login-node prep").
#
# OUTPUT: $CONTAINERS/skyrl_megatron_vllm0202rc0_r3_sandbox/  (a directory)
# =============================================================================
#SBATCH --job-name=build_l_vllm0202rc0_r3_cu13
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/sif_build/logs/build_l0202_cu13_%j.out
#SBATCH --error=/leonardo_work/AIFAC_5C0_290/bfeuer00/sif_build/logs/build_l0202_cu13_%j.err

set -euo pipefail

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
CONTAINERS=$WORK/containers
BASE_SBX=$CONTAINERS/pytorch_2509_sbx                        # NGC 25.09 cu13 sandbox, built on login node
SANDBOX=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_sandbox    # OUTPUT (a dir, on WORK/GPFS)
VLLM_SRC=$WORK/sif_build/vllm_clone                          # cloned fork @ 5d7319dd1
SKYRL_SRC=$WORK/sif_build/skyrl_clone                        # SkyRL @ 2ab513a6
BUILD_DEPS_WHEELS=$WORK/sif_build/build_deps_wheels          # setuptools_scm/setuptools/wheel/packaging
RUNTIME_DEPS_WHEELS=$WORK/sif_build/runtime_deps_wheels      # vLLM common.txt pure-python deps
ACCEL_WHEELS=$WORK/sif_build/accel_wheels                    # optional prebuilt flash_attn/megatron wheels

# The forward-compat libcuda path (cu13 userspace driver shim for the 535 host kernel).
COMPAT=/usr/local/cuda-13.0/compat/lib.real

# Build tmp/cache on WORK (NOT over-quota SCRATCH_FAST; NOT tiny node-local /tmp; NOT
# the default Lustre /scratch_local -> lustre.lov xattr storm).
BUILD=$WORK/sif_build/job_$SLURM_JOB_ID
mkdir -p "$BUILD" "$WORK/sif_build/logs"
export SINGULARITY_TMPDIR=$BUILD/sgtmp
export SINGULARITY_CACHEDIR=$BUILD/sgcache
export TMPDIR=$BUILD/sgtmp
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
echo "BASE_SBX=$BASE_SBX"; [ -d "$BASE_SBX" ] || { echo "FATAL: NGC cu13 base sandbox not found (build it on the login node: singularity build --sandbox $BASE_SBX docker://nvcr.io/nvidia/pytorch:25.09-py3, with TMPDIR on GPFS)"; exit 2; }
[ -d "$VLLM_SRC/vllm" ] || { echo "FATAL: vLLM fork source not at $VLLM_SRC (git clone penfever/working, checkout 5d7319dd1)"; exit 2; }
cat "$VLLM_SRC/.vllm_commit" 2>/dev/null || true
[ -d "$SKYRL_SRC" ] || { echo "FATAL: SkyRL source not at $SKYRL_SRC (clone penfever/SkyRL, checkout 2ab513a6)"; exit 2; }
[ -d "$BUILD_DEPS_WHEELS" ] || { echo "FATAL: build-deps wheelhouse not at $BUILD_DEPS_WHEELS"; exit 3; }
[ -d "$RUNTIME_DEPS_WHEELS" ] || { echo "FATAL: runtime-deps wheelhouse not at $RUNTIME_DEPS_WHEELS"; exit 3; }
# megatron-core is NOT in the NGC base and compute nodes have no internet, so the
# 0.14.0 wheel MUST be pre-staged in the accel wheelhouse on the login node.
ls "$ACCEL_WHEELS"/megatron_core-0.14.0-*.whl >/dev/null 2>&1 || { echo "FATAL: megatron-core 0.14.0 wheel not in accel wheelhouse $ACCEL_WHEELS -- run login-node prep: singularity exec -B \$ACCEL_WHEELS:/whl \$BASE_SBX python -m pip download megatron-core==0.14.0 --no-deps --only-binary=:all: -d /whl"; exit 3; }
nvidia-smi | head -3 || true
df -h "$WORK" | tail -1

SINGULARITY=/usr/bin/singularity   # SingularityPRO 4.3.1; NO apptainer

# ---------------------------------------------------------------------------
# 1. Copy the cu13 base sandbox -> the build sandbox (on WORK/GPFS). We copy
#    rather than `singularity build` because the base is ALREADY a sandbox dir
#    (no .sif to convert). cp -a preserves perms/symlinks (incl. compat/lib.real).
# ---------------------------------------------------------------------------
# RESUME=1: skip the destructive [1] base-sandbox clone + [1b] staging and reuse
# an EXISTING fully-staged $SANDBOX. The clone+stage of the multi-GB NGC 25.09 cu13
# tree on GPFS is the build's long pole (~9-10h); PHASE A/B only ADD into the
# sandbox (megatron/flash-attn/skyrl/vllm installs), they never depend on a fresh
# clone. So a re-run after a late-stage FATAL (e.g. the GPFS-stall transient that
# tripped 47181886's in-container megatron-wheel ls at 15:59) can resume straight
# into PHASE A against the intact staged sandbox. ONLY set RESUME=1 when $SANDBOX
# is a clean, fully-staged base where PHASE A has NOT yet pip-installed anything
# (i.e. the failure was before/at the first install step).
if [ "${RESUME:-0}" = "1" ]; then
  echo "=== [1] RESUME=1 -> reusing existing staged sandbox (skip clone + staging) ==="
  [ -d "$SANDBOX/opt/accel_wheels" ] || { echo "FATAL (resume): $SANDBOX/opt/accel_wheels missing -- sandbox not staged, cannot RESUME"; exit 2; }
  echo "reusing sandbox at $SANDBOX"
else
echo "=== [1] cloning cu13 base sandbox -> build sandbox ==="
rm -rf "$SANDBOX"
cp -a "$BASE_SBX" "$SANDBOX"
echo "sandbox at $SANDBOX"

echo "=== [1b] staging SkyRL + vLLM fork + wheelhouses into sandbox ==="
mkdir -p "$SANDBOX/opt/SkyRL" "$SANDBOX/opt/vllm_build" \
         "$SANDBOX/opt/build_deps_wheels" "$SANDBOX/opt/runtime_deps_wheels" \
         "$SANDBOX/opt/accel_wheels"
# NB: no /whl bind mountpoint is created -- the host bind into the --writable
# sandbox is silently dropped on SingularityPRO 4.3.1 (no overlay/underlay), so
# PHASE A installs the accel wheels from the staged /opt/accel_wheels copy below.
cp -a "$SKYRL_SRC"/.            "$SANDBOX/opt/SkyRL/"
cp -a "$VLLM_SRC"/.            "$SANDBOX/opt/vllm_build/"
cp -a "$BUILD_DEPS_WHEELS"/.   "$SANDBOX/opt/build_deps_wheels/"
cp -a "$RUNTIME_DEPS_WHEELS"/. "$SANDBOX/opt/runtime_deps_wheels/"
[ -d "$ACCEL_WHEELS" ] && cp -a "$ACCEL_WHEELS"/. "$SANDBOX/opt/accel_wheels/" || echo "(no accel wheelhouse -- phase A will compile flash-attn from source)"
fi

# PREFLIGHT ASSERT (host-side, runs in seconds): the megatron-core install is the
# FIRST PHASE A step, and it can only see the IN-SANDBOX staged wheelhouse (the
# host bind is unsupported here). Verify the staged megatron wheel actually landed
# in $SANDBOX/opt/accel_wheels BEFORE we spend ~2h on the base-sandbox clone +
# vLLM compile. A staging miss must fail HERE, not 2h later inside the container
# (cu13 build 47137387 burned ~1h55 before the in-container FATAL). Glob -> stat
# the first match to confirm it exists AND is non-empty.
_mega_staged=$(ls "$SANDBOX"/opt/accel_wheels/megatron_core-0.14.0-*.whl 2>/dev/null | head -1)
if [ -z "$_mega_staged" ] || [ ! -s "$_mega_staged" ]; then
  echo "FATAL (preflight): megatron-core 0.14.0 wheel did NOT stage into $SANDBOX/opt/accel_wheels" >&2
  echo "  host wheelhouse $ACCEL_WHEELS:" >&2; ls -la "$ACCEL_WHEELS" 2>&1 | sed "s/^/    /" >&2
  echo "  staged $SANDBOX/opt/accel_wheels:" >&2; ls -la "$SANDBOX"/opt/accel_wheels 2>&1 | sed "s/^/    /" >&2
  exit 4
fi
echo "preflight OK: staged megatron wheel -> $_mega_staged ($(stat -c%s "$_mega_staged" 2>/dev/null || stat -f%z "$_mega_staged") bytes)"

# ---------------------------------------------------------------------------
# 2. PHASE A: SkyRL editable + Megatron-core 0.14.0 + flash-attn (sm_80).
#    NB: bash -lc single-quoted block -> ESCAPED DOUBLE QUOTES (\") only.
#    These phases run WITHOUT --nv (no GPU needed to compile); the cu13 toolkit
#    + nvcc come from the base. SETUPTOOLS_SCM not needed for these.
# ---------------------------------------------------------------------------
echo "=== [2] PHASE A: SkyRL + Megatron-core + flash-attn ==="
# Install the offline accel wheels from the IN-SANDBOX staged copy
# (/opt/accel_wheels) ONLY -- do NOT bind the host wheelhouse into the writable
# sandbox. On Leonardo's SingularityPRO 4.3.1 (no overlay/underlay) a user bind
# into a `--writable` sandbox dir SILENTLY DROPS even when the mountpoint is
# pre-created: the bind never appears inside the container, so `/whl` shows up as
# "No such file or directory" and the megatron-core install fails at PHASE A start
# (cu13 build 47137387, the 4th recurrence of this class). The staged
# /opt/accel_wheels is a plain dir in the writable sandbox tree and is therefore
# always visible inside; the host-side preflight assert above (after the cp -a)
# guarantees the megatron wheel is actually present in that staged path before we
# spend ~2h on the base-sandbox clone + compile.
$SINGULARITY exec --writable --no-home \
  --env TORCH_CUDA_ARCH_LIST="8.0" \
  --env MAX_JOBS=24 \
  --env FLASH_ATTENTION_FORCE_BUILD=TRUE \
  --env PIP_NO_BUILD_ISOLATION=1 \
  "$SANDBOX" bash -lc '
    set -euo pipefail
    python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda)"
    nvcc --version | tail -2
    echo "--- megatron-core 0.14.0 (OFFLINE wheel, --no-deps) ---"
    # The NGC 25.09 base does NOT ship megatron-core (verified), and compute nodes
    # have NO internet, so megatron-core 0.14.0 MUST come from the offline accel
    # wheelhouse. The prebuilt cp312 x86_64 manylinux wheel matches the base ABI
    # exactly. Install --no-deps: megatron-core 0.14.0's only NON-extra runtime
    # deps are torch / numpy / packaging, all already in the base (the base's
    # numpy 2.1.0 vs megatron's conservative numpy<2.0.0 pin is intentionally
    # ignored -- the Jupiter prod stack runs megatron-core on the NGC numpy 2.x).
    #
    # Install from the IN-SANDBOX staged wheelhouse /opt/accel_wheels ONLY. The
    # host-bind /whl path was removed: a user bind into this --writable sandbox is
    # silently dropped on SingularityPRO 4.3.1 (no overlay/underlay), so /whl is
    # never present inside the container. The host-side preflight assert (after the
    # cp -a stage) already guaranteed the wheel is in $SANDBOX/opt/accel_wheels.
    # Detect the staged wheel with a short RETRY. 47181886 FAILED here at 15:59
    # with "ls: /opt/accel_wheels: No such file or directory" EVEN THOUGH the host
    # had just stat'd that exact wheel (preflight OK, 2170140 bytes) -- the in-
    # container view of the GPFS-backed sandbox dir transiently returned ENOENT
    # during the 15:20-16:00 /leonardo_work GPFS stall (same window that killed 10
    # pass@k jobs). A single ls was enough to fall through to exit 4 and waste the
    # ~10h clone. Retry a few times with a short sleep so a momentary FS stall does
    # not masquerade as a missing wheel; the host-side preflight already proved the
    # wheel is staged, so any miss here is FS-transient, not a real absence.
    MEGA_LINKS=""
    for _try in 1 2 3 4 5 6; do
      if ls /opt/accel_wheels/megatron_core-0.14.0-*.whl >/dev/null 2>&1; then
        MEGA_LINKS=/opt/accel_wheels
        break
      fi
      echo "  (attempt $_try) /opt/accel_wheels not yet visible in container -- waiting 10s (FS stall?)" >&2
      sleep 10
    done
    if [ -n "$MEGA_LINKS" ]; then
      echo "megatron-core wheel found in $MEGA_LINKS"
      python -m pip install --no-index --no-deps --find-links "$MEGA_LINKS" \
        "megatron-core==0.14.0"
    else
      # GUARD the diagnostics: under `set -euo pipefail` an `ls` of a missing dir
      # exits 2 and would trip pipefail, self-aborting this block at exit 2 BEFORE
      # the intended `exit 4` ever ran (that masked the real cause on 47137387 as a
      # bare 2:0). `set +e` here lets both listings print and the explicit exit 4 fire.
      set +e
      echo "FATAL: megatron-core 0.14.0 wheel not found in /opt/accel_wheels (staged)" >&2
      echo "  /opt/accel_wheels:" >&2; ls -la /opt/accel_wheels 2>&1 | sed "s/^/    /" >&2
      exit 4
    fi
    echo "--- flash-attn (base ships it; build from source only if absent) ---"
    # NGC 25.09 already ships flash-attn; only compile if genuinely missing.
    if python -c "import flash_attn" 2>/dev/null; then
      echo "flash-attn already present in base"
    elif ls /opt/accel_wheels/flash_attn-*.whl >/dev/null 2>&1; then
      python -m pip install --no-index --no-deps --find-links /opt/accel_wheels flash-attn
    else
      echo "WARN: flash-attn absent from base AND accel wheelhouse -- compiling from source (long; sm_80)"
      python -m pip install --no-build-isolation flash-attn
    fi
    echo "--- SkyRL editable (skyrl_train + skyrl_gym) ---"
    cd /opt/SkyRL
    python -m pip install --no-build-isolation --no-deps -e ./skyrl-train || \
      python -m pip install --no-build-isolation --no-deps -e .
    [ -d ./skyrl-gym ] && python -m pip install --no-build-isolation --no-deps -e ./skyrl-gym || true
    echo "--- phase A smoke ---"
    python -c "import megatron.core as mc; print(\"megatron-core\", mc.__version__)"
    python -c "import flash_attn; print(\"flash_attn\", flash_attn.__version__)"
    python -c "import skyrl_train, skyrl_gym; print(\"skyrl_train+skyrl_gym import OK\")"
  '
echo "PHASE A complete."

# ---------------------------------------------------------------------------
# 3. PHASE B: build vLLM fork 0.20.2rc0 (penfever/working @ 5d7319dd1) FROM
#    SOURCE against the in-base NGC torch 2.9 (cu13), arch 8.0. All OFFLINE.
#    use_existing_torch strips the torch==2.11 pin; --no-deps so pip never pulls
#    torch 2.11 / bumps numpy; two-pass runtime-dep install; non-fatal probes.
# ---------------------------------------------------------------------------
echo "=== [3] PHASE B: build vLLM fork from source (arch 8.0, existing cu13 torch) ==="
$SINGULARITY exec --writable --no-home \
  --env MAX_JOBS=24 \
  --env CMAKE_BUILD_TYPE=Release \
  --env VLLM_TARGET_DEVICE=cuda \
  --env TORCH_CUDA_ARCH_LIST="8.0" \
  --env CCACHE_DISABLE=1 \
  --env PIP_NO_BUILD_ISOLATION=1 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION=0.20.2rc0 \
  --env SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM=0.20.2rc0 \
  "$SANDBOX" bash -lc '
    set -euo pipefail
    cd /opt/vllm_build
    echo "--- pre-build torch / nvcc ---"
    python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda)"
    nvcc --version | tail -2
    echo "--- strip torch pins so the build uses the in-base NGC torch 2.9 ---"
    python use_existing_torch.py
    echo "--- install PEP517 build-backend deps OFFLINE ---"
    python -m pip install --no-index --find-links /opt/build_deps_wheels \
      setuptools_scm setuptools wheel packaging
    python -c "import setuptools_scm; print(\"setuptools_scm import OK\")"
    echo "--- install python build deps (no torch) ---"
    python -m pip install --no-build-isolation -r requirements/build.txt || true
    echo "--- install vLLM RUNTIME deps OFFLINE (two passes, both --no-index) ---"
    python -m pip install --no-index --find-links /opt/runtime_deps_wheels \
      "cachetools" "blake3" "py-cpuinfo" "openai>=2.0.0" \
      "prometheus-fastapi-instrumentator>=7.0.0" "tiktoken>=0.6.0" \
      "lm-format-enforcer==0.11.3" "llguidance>=1.3.0,<1.4.0" "outlines_core==0.2.14" \
      "diskcache==5.6.3" "xgrammar>=0.2.0,<1.0.0" "partial-json-parser" "msgspec" \
      "gguf>=0.17.0" "mistral_common>=1.11.2" "compressed-tensors==0.15.0.1" \
      "depyf==0.20.0" "watchfiles" "pybase64" "cbor2" "ijson" "setproctitle" \
      "openai-harmony>=0.0.3" "anthropic>=0.71.0" \
      "model-hosting-container-standards>=0.1.14,<1.0.0" "mcp" \
      "opentelemetry-sdk>=1.27.0" "opentelemetry-exporter-otlp>=1.27.0" \
      "opentelemetry-semantic-conventions-ai>=0.4.1" "fastsafetensors>=0.2.2"
    python -m pip install --no-deps --no-index --find-links /opt/runtime_deps_wheels \
      "opencv-python-headless>=4.13.0" || true
    echo "--- runtime-dep import smoke (offline) ---"
    python -c "import cachetools, blake3, openai, xgrammar, gguf, mistral_common, compressed_tensors, msgspec, mcp, fastsafetensors; print(\"runtime-dep imports OK\")"
    echo "--- build + install vLLM (editable, no-build-isolation, --no-deps, existing torch, arch 8.0) ---"
    python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tail -120
    python -c "import vllm; print(\"vllm\", vllm.__version__)"
  '
echo "PHASE B (vLLM build) complete."

echo "=== [3c] pruning vLLM build artifacts (keep /opt/vllm_build source for editable import) ==="
rm -rf "$SANDBOX/opt/vllm_build/build" "$SANDBOX/opt/vllm_build/.deps" \
       "$SANDBOX"/root/.cache/pip "$SANDBOX"/opt/vllm_build/*.egg-info/SOURCES.txt 2>/dev/null || true
rm -rf "$SANDBOX/opt/build_deps_wheels" "$SANDBOX/opt/runtime_deps_wheels" "$SANDBOX/opt/accel_wheels" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Validate under --nv with the forward-compat LD path exported (the RUN
#    CONVENTION). SINGULARITYENV_LD_LIBRARY_PATH puts compat/lib.real first;
#    NGC's own ldconfig already wires it, this is belt-and-suspenders.
# ---------------------------------------------------------------------------
echo "=== [4] validating sandbox (cu13 forward-compat run convention) ==="
SINGULARITYENV_LD_LIBRARY_PATH=$COMPAT $SINGULARITY exec --nv \
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$SANDBOX" python - <<'PYEOF' || echo "WARN: validation block exited non-zero (sandbox already built; inspect output above)"
import importlib, os
def chk(n):
    try:
        m=importlib.import_module(n); print(f"IMPORT OK  {n}  v={getattr(m,'__version__','n/a')}"); return m
    except Exception as e:
        print(f"IMPORT FAIL {n}: {type(e).__name__}: {e}"); return None
print("--- base stack (cu13 forward-compat) ---")
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(),
                    "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")
# confirm the compat libcuda is the one loaded
try:
    for l in open("/proc/self/maps").read().splitlines():
        if "libcuda.so" in l: print("LOADED_LIBCUDA:", l.split()[-1]); break
except Exception: pass
chk("vllm"); import vllm; print("VLLM_VERSION:", vllm.__version__)
chk("transformers")
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
    print("transformers gemma4:", "gemma4" in C, "gemma4_text:", "gemma4_text" in C,
          "qwen3_next:", "qwen3_next" in C, "qwen3_moe:", "qwen3_moe" in C)
except Exception as e: print("transformers config probe:", type(e).__name__, e)
chk("megatron.core"); chk("flash_attn"); chk("skyrl_train"); chk("skyrl_gym")
print("--- vLLM model registry archs ---")
try:
    from vllm import ModelRegistry
    archs=set(ModelRegistry.get_supported_archs())
    for a in ("Gemma4ForConditionalGeneration","Gemma4ForCausalLM","Qwen3MoeForCausalLM","Qwen3NextForCausalLM"):
        print(f"  REGISTRY {a}: {a in archs}")
except Exception as e: print("registry probe FAIL:", type(e).__name__, e)
print("--- R3 routed_experts native capture ---")
sp="/opt/vllm_build"
for rel in ("vllm/entrypoints/openai/chat_completion/serving.py",
            "vllm/entrypoints/openai/chat_completion/protocol.py",
            "vllm/entrypoints/openai/completion/serving.py",
            "vllm/entrypoints/openai/completion/protocol.py",
            "vllm/model_executor/layers/fused_moe/routed_experts_capturer.py"):
    p=os.path.join(sp,rel)
    print(f"  {rel}: " + (f"routed_experts_count={open(p).read().count('routed_experts')}" if os.path.exists(p) else "MISSING"))
print("--- DCP GQA-LSE fp32 fix baked ---")
common=os.path.join(sp,"vllm/v1/attention/ops/common.py")
fa=os.path.join(sp,"vllm/v1/attention/backends/flash_attn.py")
if os.path.exists(common):
    txt=open(common).read(); print(f"  common.py out_fp32={txt.count('out_fp32')} (expect>0)  float32_acc_present={'torch.float32' in txt}")
else: print("  common.py MISSING")
if os.path.exists(fa):
    print(f"  flash_attn.py _forward_with_dcp out_fp32=True present: {'out_fp32=True' in open(fa).read()}")
else: print("  flash_attn.py MISSING")
PYEOF

echo "=== $(date) BUILD DONE -> $SANDBOX ==="
echo "RUN CONVENTION (cu13 forward-compat):"
echo "  SINGULARITYENV_LD_LIBRARY_PATH=$COMPAT $SINGULARITY exec --nv \\"
echo "    --env VLLM_ATTENTION_BACKEND=FLASH_ATTN --env VLLM_USE_FLASHINFER_SAMPLER=0 \\"
echo "    --env LIBRARY_PATH=/.singularity.d/libs $SANDBOX <cmd>   # LIBRARY_PATH for tp>1 Triton"
echo "DCP parity smoke (separate sbatch, 2-node tp=8 dcp=1 vs dcp=2, Qwen3-Coder-30B-A3B, greedy, R3 on) -> expect ~6.94% token mismatch."
# NO .sif packing (mksquashfs OOM/xattr on Lustre login) -- the sandbox dir is the deliverable.
