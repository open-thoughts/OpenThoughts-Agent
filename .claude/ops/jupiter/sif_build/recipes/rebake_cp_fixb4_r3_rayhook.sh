#!/bin/bash
# =============================================================================
# rebake_cp_fixb4_r3_rayhook.sh  (LOGIN-NODE surgical rebake, NOT sbatch)
#
# R3 ray-executor EngineCore init fix: rebakes the live CP+R3 prod SIF so its
#   editable vLLM source advances 4d167a4af (penfever/working, baked into _cp_fixb3)
#   -> c5832db29 (penfever/working, adds the in-EngineCore setup-hook strip in
#   vllm/v1/executor/ray_utils.py before the nested ray.init).
#
# Delta 4d167a4af..c5832db29 is PYTHON-ONLY (git diff --name-only):
#   vllm/v1/executor/ray_utils.py   <- the fix (adds _strip_leaked_setup_hook_job_config)
# NO csrc/CMake/.cu/.cpp/.h/pyproject/setup.py/requirements changed -> compiled
# CUDA/C++ kernels are byte-identical. vLLM is installed EDITABLE (-e .,
# source at /opt/vllm_build/vllm), so swapping the .py source + clearing pyc fully
# applies the fix WITHOUT a recompile. Mirrors rebake_237_cp_fixb3.sh.
#
# c5832db29 DESCENDS FROM 4d167a4af (verified: git merge-base --is-ancestor), so
# checking it out on the clone is a clean fast-forward, no rebase.
#
# IN  = skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif (live; #237 epilogue -- read-only)
# OUT = skyrl_megatron_vllm0202rc0_r3_cp_fixb4.sif (new; validated, built ALONGSIDE fixb3)
#
# PREREQ (supervisor, before running): push penfever/working so c5832db29 is on the
#   remote, then on the clone:  cd /e/scratch/jureap59/feuer1/vllm && git fetch && \
#   git checkout c5832db29 .   (this script fetches + checks out the target itself).
# =============================================================================
set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
IN_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_cp_fixb4.sif
VLLM_SRC=/e/scratch/jureap59/feuer1/vllm
TARGET_COMMIT=c5832db29e933cafde40d06ebffb50f3b8ceab57
RU_REL=vllm/v1/executor/ray_utils.py
SIG='_strip_leaked_setup_hook_job_config'

STAMP=$(date +%Y%m%d_%H%M%S)
BUILD=/tmp/sifrebake_fixb4_$STAMP
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "=== $(date) host=$(hostname) FIXB4 R3-RAYHOOK REBAKE STAMP=$STAMP ==="
echo "IN_SIF =$IN_SIF ($(stat -c%s "$IN_SIF") bytes)"
echo "OUT_SIF=$OUT_SIF"
echo "VLLM_SRC=$VLLM_SRC TARGET=$TARGET_COMMIT"
[ -d "$VLLM_SRC/vllm" ] || { echo "FATAL: vLLM fork source not found at $VLLM_SRC"; exit 2; }

echo "=== [0] fetch + checkout the target fork commit on the clone ==="
( cd "$VLLM_SRC" && git fetch --all --quiet && git checkout "$TARGET_COMMIT" )
GOT=$(cd "$VLLM_SRC" && git rev-parse HEAD)
echo "VLLM_SRC HEAD=$GOT"
[ "$GOT" = "$TARGET_COMMIT" ] || { echo "FATAL: clone HEAD $GOT != target $TARGET_COMMIT"; exit 2; }
grep -qF "$SIG" "$VLLM_SRC/$RU_REL" || { echo "FATAL: fix signature missing in source ray_utils.py"; exit 2; }
echo "source ray_utils fix-sig count: $(grep -cF "$SIG" "$VLLM_SRC/$RU_REL")"
df -h /tmp | tail -1; df -i /tmp | tail -1

echo "=== [1] sandbox from live CP prod SIF (_cp_fixb3) ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$IN_SIF"
echo "sandbox built."
DST=$SANDBOX/opt/vllm_build
[ -d "$DST/vllm" ] || { echo "FATAL: /opt/vllm_build/vllm missing in sandbox"; exit 3; }

echo "=== [1b] pre-swap signature (expect fix-sig ABSENT in baked _cp_fixb3) ==="
grep -cF "$SIG" "$DST/$RU_REL" || true
head -1 "$DST/.vllm_commit" 2>/dev/null || echo "(no .vllm_commit)"

echo "=== [2] swap the patched ray_utils.py; stamp commit ==="
cp -a "$VLLM_SRC/$RU_REL" "$DST/$RU_REL"
echo "  swapped $RU_REL"
echo "$TARGET_COMMIT" > "$DST/.vllm_commit"
( cd "$VLLM_SRC" && git describe --tags 2>/dev/null ) >> "$DST/.vllm_commit" || true

echo "=== [2b] clear stale bytecode for the swapped module ==="
rm -f "$DST/vllm/v1/executor/__pycache__/ray_utils."*.pyc 2>/dev/null || true

echo "=== [2c] post-swap signature (expect fix-sig present) ==="
grep -cF "$SIG" "$DST/$RU_REL"

echo "=== [2d] in-sandbox import + signature checks ==="
apptainer exec --writable --nv --no-home --env VLLM_USE_FLASHINFER_SAMPLER=0 --env VLLM_ATTENTION_BACKEND=FLASH_ATTN "$SANDBOX" bash -lc '
  set -euo pipefail
  python -c "import vllm; print(\"vllm\", vllm.__version__, \"from\", vllm.__file__)"
  python -c "import torch; print(\"torch\", torch.__version__)"
  python -c "from vllm.v1.executor import ray_utils as r; print(\"ray_utils import OK\", r.__file__); assert hasattr(r, \"_strip_leaked_setup_hook_job_config\"), \"helper missing\"; print(\"helper present OK\")"
  python -c "from vllm import ModelRegistry; a=set(ModelRegistry.get_supported_archs()); print({k:(k in a) for k in [\"Qwen3MoeForCausalLM\",\"Qwen3NextForCausalLM\"]})"
'

echo "=== [3] building NEW SIF -> $OUT_SIF ==="
TMP_OUT=$BUILD/out.sif
rm -f "$TMP_OUT"
apptainer build "$TMP_OUT" "$SANDBOX"
mv -f "$TMP_OUT" "$OUT_SIF"
echo "NEW SIF built: $(stat -c%s "$OUT_SIF") bytes"; ls -la "$OUT_SIF"

echo "=== [4] validating fresh SIF ==="
apptainer exec --nv --env VLLM_USE_FLASHINFER_SAMPLER=0 --env VLLM_ATTENTION_BACKEND=FLASH_ATTN "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation exited non-zero (SIF already written)"
import os
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "from", vllm.__file__)
sp="/opt/vllm_build"
ru=os.path.join(sp,"vllm/v1/executor/ray_utils.py")
txt=open(ru).read()
print("ray_utils exists:", os.path.exists(ru))
print("FIX sig _strip_leaked_setup_hook_job_config count:", txt.count("_strip_leaked_setup_hook_job_config"))
from vllm.v1.executor import ray_utils as r
print("ray_utils import OK:", r.__file__)
print("helper present:", hasattr(r, "_strip_leaked_setup_hook_job_config"))
print("in-SIF .vllm_commit:", open(os.path.join(sp,".vllm_commit")).read().strip().splitlines()[0] if os.path.exists(os.path.join(sp,".vllm_commit")) else "none")
import skyrl_train, skyrl_gym; print("skyrl_train OK")
from vllm import ModelRegistry
archs=set(ModelRegistry.get_supported_archs())
for a in ("Qwen3MoeForCausalLM","Qwen3NextForCausalLM"):
    print("  REGISTRY", a, a in archs)
print("ALL CORE CHECKS DONE")
PYEOF

echo "=== md5 ==="; md5sum "$OUT_SIF"
echo "=== removing build scratch ==="
rm -rf "$BUILD" 2>/dev/null || true
echo "=== $(date) FIXB4 REBAKE COMPLETE OUT_SIF=$OUT_SIF ==="
