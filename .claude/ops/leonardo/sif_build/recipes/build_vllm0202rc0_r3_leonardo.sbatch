#!/bin/bash
# =============================================================================
# build_vllm0202rc0_r3_leonardo.sbatch
#
# Leonardo (CINECA, x86_64 / A100) cross-cluster twin of Jupiter's prod SIF
# skyrl_megatron_vllm0202rc0_r3.sif. Builds a WRITABLE SINGULARITY SANDBOX DIR
# (NOT a packed .sif -- mksquashfs OOM/xattr blocker on Lustre login) carrying:
#   - SkyRL editable (penfever/SkyRL @ 2ab513a6) at /opt/SkyRL
#   - Megatron-core 0.14.0 + flash-attn 2.7.4.post1 (sm_80)
#   - vLLM fork penfever/working @ 5d7319dd1 (0.20.2rc0 + native R3 + DCP fp32 fix),
#     built from source against the in-base NGC torch 2.8 (cu12.9).
#
# BASE: nvcr.io/nvidia/pytorch:25.06-py3 (CUDA 12.9.1, torch 2.8.0a0, TE 2.4, apex).
#   NGC 25.09/CUDA13 (Jupiter's base) is INFEASIBLE: Leonardo driver 535.274.02
#   caps CUDA at 12.2; CUDA 13 needs >=580.65. 25.06 is the newest CUDA-12 NGC base.
#   See README_vllm0202rc0_r3_leonardo.md "Feasibility / the CUDA-13 blocker".
#
# A100, so TORCH_CUDA_ARCH_LIST=8.0 (NOT 9.0). SingularityPRO 4.3.1, no apptainer,
# no root. Run via sbatch (NOT the fragile login node). Compute nodes have NO
# internet -> the NGC pull, the fork/SkyRL rsync, and the offline wheelhouses are
# prepared on the LOGIN node first (see README "Login-node prep").
#
# OUTPUT: $CONTAINERS/skyrl_megatron_vllm0202rc0_r3_sandbox/  (a directory)
# =============================================================================
#SBATCH --job-name=build_l_vllm0202rc0_r3
#SBATCH --account=AIFAC_5C0_290
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/AIFAC_5C0_290/bfeuer00/sif_build/logs/build_l0202_%j.out
#SBATCH --error=/leonardo_work/AIFAC_5C0_290/bfeuer00/sif_build/logs/build_l0202_%j.err

set -euo pipefail

WORK=/leonardo_work/AIFAC_5C0_290/bfeuer00
CONTAINERS=$WORK/containers
BASE_SIF=$CONTAINERS/pytorch_25.06.sif                       # NGC 25.06, pulled on login node
SANDBOX=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_sandbox    # OUTPUT (a dir, on WORK/GPFS)
VLLM_SRC=$WORK/sif_build/vllm_src                            # rsync'd fork @ 5d7319dd1
SKYRL_SRC=$WORK/sif_build/skyrl_src                          # SkyRL @ 2ab513a6
BUILD_DEPS_WHEELS=$WORK/sif_build/build_deps_wheels          # setuptools_scm/setuptools/wheel/packaging
RUNTIME_DEPS_WHEELS=$WORK/sif_build/runtime_deps_wheels      # ~32 missing common.txt pure-python deps
# Optional prebuilt accel wheels (skip a long source compile if present):
ACCEL_WHEELS=$WORK/sif_build/accel_wheels                    # flash_attn 2.7.4.post1 + megatron_core 0.14.0 wheels

# Build tmp/cache on WORK (NOT over-quota SCRATCH_FAST; NOT tiny node-local /tmp).
BUILD=$WORK/sif_build/job_$SLURM_JOB_ID
mkdir -p "$BUILD" "$WORK/sif_build/logs"
export SINGULARITY_TMPDIR=$BUILD/sgtmp
export SINGULARITY_CACHEDIR=$BUILD/sgcache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
echo "BASE_SIF=$BASE_SIF"; ls -la "$BASE_SIF" 2>&1 || { echo "FATAL: NGC base not found (pull it on the login node: singularity pull $BASE_SIF docker://nvcr.io/nvidia/pytorch:25.06-py3)"; exit 2; }
[ -d "$VLLM_SRC/vllm" ] || { echo "FATAL: vLLM fork source not at $VLLM_SRC (rsync penfever/working @ 5d7319dd1, exclude .git/build/*.so)"; exit 2; }
cat "$VLLM_SRC/.vllm_commit" 2>/dev/null || true
[ -d "$SKYRL_SRC" ] || { echo "FATAL: SkyRL source not at $SKYRL_SRC (clone+checkout 2ab513a6 on the login node)"; exit 2; }
[ -d "$BUILD_DEPS_WHEELS" ] || { echo "FATAL: build-deps wheelhouse not at $BUILD_DEPS_WHEELS"; exit 3; }
[ -d "$RUNTIME_DEPS_WHEELS" ] || { echo "FATAL: runtime-deps wheelhouse not at $RUNTIME_DEPS_WHEELS"; exit 3; }
nvidia-smi | head -3 || true
df -h "$WORK" | tail -1

SINGULARITY=/usr/bin/singularity   # SingularityPRO 4.3.1; NO apptainer

# ---------------------------------------------------------------------------
# 1. Writable sandbox from the NGC 25.06 base, on WORK (GPFS)
# ---------------------------------------------------------------------------
echo "=== [1] building sandbox from NGC 25.06 base ==="
rm -rf "$SANDBOX"
$SINGULARITY build --sandbox "$SANDBOX" "$BASE_SIF"
echo "sandbox built at $SANDBOX"

# Stage sources INTO the sandbox (writable; editable installs + pip build write here).
echo "=== [1b] staging SkyRL + vLLM fork + wheelhouses into sandbox ==="
mkdir -p "$SANDBOX/opt/SkyRL" "$SANDBOX/opt/vllm_build" \
         "$SANDBOX/opt/build_deps_wheels" "$SANDBOX/opt/runtime_deps_wheels" \
         "$SANDBOX/opt/accel_wheels"
cp -a "$SKYRL_SRC"/.            "$SANDBOX/opt/SkyRL/"
cp -a "$VLLM_SRC"/.            "$SANDBOX/opt/vllm_build/"
cp -a "$BUILD_DEPS_WHEELS"/.   "$SANDBOX/opt/build_deps_wheels/"
cp -a "$RUNTIME_DEPS_WHEELS"/. "$SANDBOX/opt/runtime_deps_wheels/"
[ -d "$ACCEL_WHEELS" ] && cp -a "$ACCEL_WHEELS"/. "$SANDBOX/opt/accel_wheels/" || echo "(no accel wheelhouse -- phase A will compile from the staged sources)"

# ---------------------------------------------------------------------------
# 2. PHASE A: SkyRL editable + Megatron-core 0.14.0 + flash-attn 2.7.4.post1
#    (the training stack NGC 25.06 does NOT bundle). All OFFLINE.
#    NB: bash -lc single-quoted block -> use ESCAPED DOUBLE QUOTES (\") only in
#    python -c snippets; a bare single quote anywhere closes the outer quote.
# ---------------------------------------------------------------------------
echo "=== [2] PHASE A: SkyRL + Megatron-core + flash-attn ==="
$SINGULARITY exec --writable --no-home \
  --env TORCH_CUDA_ARCH_LIST="8.0" \
  --env MAX_JOBS=24 \
  --env FLASH_ATTENTION_FORCE_BUILD=TRUE \
  --env PIP_NO_BUILD_ISOLATION=1 \
  "$SANDBOX" bash -lc '
    set -euo pipefail
    python -c "import torch;print(\"torch\",torch.__version__,\"cuda\",torch.version.cuda,\"cap\",torch.cuda.get_device_capability() if torch.cuda.is_available() else \"n/a\")"
    nvcc --version | tail -2
    echo "--- megatron-core 0.14.0 + flash-attn 2.7.4.post1 ---"
    # Prefer prebuilt accel wheels (offline); fall back to source compile.
    if ls /opt/accel_wheels/*.whl >/dev/null 2>&1; then
      python -m pip install --no-index --find-links /opt/accel_wheels --no-deps \
        "megatron-core==0.14.0" "flash-attn==2.7.4.post1" || \
        python -m pip install --no-index --find-links /opt/accel_wheels /opt/accel_wheels/*.whl
    else
      echo "WARN: no accel wheelhouse -- compiling flash-attn from source (long; sm_80)"
      python -m pip install --no-build-isolation "megatron-core==0.14.0"
      python -m pip install --no-build-isolation "flash-attn==2.7.4.post1"
    fi
    echo "--- SkyRL editable install (skyrl_train + skyrl_gym) ---"
    cd /opt/SkyRL
    # SkyRL is a monorepo; install the train + gym packages editable, no torch churn.
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
#    SOURCE against the in-base NGC torch 2.8 (cu12.9), arch 8.0. All OFFLINE.
#    Reuses the Jupiter sbatch's offline-build recipe verbatim (use_existing_torch
#    strips the torch==2.11 pin; --no-deps so pip never pulls torch 2.11 / bumps
#    numpy; two-pass runtime-dep install; non-fatal version probes).
# ---------------------------------------------------------------------------
echo "=== [3] PHASE B: build vLLM fork from source (arch 8.0, existing torch) ==="
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
    echo "--- strip torch pins so the build uses the in-base NGC torch 2.8 ---"
    python use_existing_torch.py
    echo "--- install PEP517 build-backend deps OFFLINE ---"
    python -m pip install --no-index --find-links /opt/build_deps_wheels \
      setuptools_scm setuptools wheel packaging
    python -c "import setuptools_scm; print(\"setuptools_scm import OK\")"
    python -c "import importlib.metadata as md; print(\"setuptools_scm\", md.version(\"setuptools-scm\"))" || true
    echo "--- install python build deps (no torch) ---"
    python -m pip install --no-build-isolation -r requirements/build.txt || true
    echo "--- install vLLM RUNTIME deps OFFLINE (two passes, both --no-index) ---"
    # PASS A: named deps + their pure-python transitives (resolver ON, offline).
    #   mistral_common WITHOUT [image] (the extra pins numpy>=2 -> unsatisfiable
    #   against the in-base numpy). MUST succeed (missing runtime import dooms serve).
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
    # PASS B: opencv-python-headless --no-deps (its numpy>=2 pin is conservative;
    #   runs on the in-base numpy; --no-deps so numpy is NOT bumped).
    python -m pip install --no-deps --no-index --find-links /opt/runtime_deps_wheels \
      "opencv-python-headless>=4.13.0" || true
    echo "--- runtime-dep import smoke (offline) ---"
    python -c "import cachetools, blake3, openai, xgrammar, gguf, mistral_common, compressed_tensors, msgspec, mcp, fastsafetensors; print(\"runtime-dep imports OK\")"
    echo "--- build + install vLLM (editable, no-build-isolation, --no-deps, existing torch, arch 8.0) ---"
    # --no-deps is CRITICAL: without it pip resolves cuda.txt (torch==2.11.0,
    # tilelang==0.1.9) against PyPI -> clobbers torch 2.8 + (offline) Network
    # unreachable. The runtime deps were just installed offline above.
    python -m pip install --no-build-isolation --no-deps -v -e . 2>&1 | tail -120
    python -c "import vllm; print(\"vllm\", vllm.__version__)"
  '
echo "PHASE B (vLLM build) complete."

# Prune build artifacts (keep the editable source for import).
echo "=== [3c] pruning vLLM build artifacts (keep /opt/vllm_build source for editable import) ==="
rm -rf "$SANDBOX/opt/vllm_build/build" "$SANDBOX/opt/vllm_build/.deps" \
       "$SANDBOX"/root/.cache/pip "$SANDBOX"/opt/vllm_build/*.egg-info/SOURCES.txt 2>/dev/null || true
# Drop the staged wheelhouses from the final sandbox (no longer needed).
rm -rf "$SANDBOX/opt/build_deps_wheels" "$SANDBOX/opt/runtime_deps_wheels" "$SANDBOX/opt/accel_wheels" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. Validate (informational; the sandbox is already written -> guard non-fatal).
# ---------------------------------------------------------------------------
echo "=== [4] validating sandbox ==="
$SINGULARITY exec --nv \
  --env VLLM_ATTENTION_BACKEND=FLASH_ATTN \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$SANDBOX" python - <<'PYEOF' || echo "WARN: validation block exited non-zero (sandbox already built; inspect output above)"
import importlib, os
def chk(n):
    try:
        m=importlib.import_module(n); print(f"IMPORT OK  {n}  v={getattr(m,'__version__','n/a')}"); return m
    except Exception as e:
        print(f"IMPORT FAIL {n}: {type(e).__name__}: {e}"); return None
print("--- base stack ---")
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(),
                    "cap", torch.cuda.get_device_capability() if torch.cuda.is_available() else "n/a")
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
echo "Run it:  $SINGULARITY exec --nv --env VLLM_ATTENTION_BACKEND=FLASH_ATTN --env VLLM_USE_FLASHINFER_SAMPLER=0 $SANDBOX <cmd>"
echo "DCP parity smoke (separate sbatch, 2-node tp=8 dcp=1 vs dcp=2, Qwen3-Coder-30B-A3B, greedy, R3 on) -> expect ~6.94% token mismatch."
# Note: NO .sif packing (mksquashfs OOM/xattr on Lustre login) -- the sandbox dir is the deliverable.
