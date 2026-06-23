#!/bin/bash
# =============================================================================
# rebake_237_cp_fixb3.sh  (LOGIN-NODE surgical rebake, NOT sbatch)
#
# #237 MERGE: rebakes the live CP+R3 prod SIF so its editable vLLM source advances
#   07e9fbca3 (feuer/r3-rank-symmetry-norope dev, OLD "FIX B") -> 4d167a4af
#   (penfever/working, MERGED; carries c7f1da624 "rank-symmetric R3 capture epilogue").
#
# Delta 07e9fbca3..4d167a4af is PYTHON-ONLY (git diff --name-only):
#   vllm/model_executor/layers/fused_moe/routed_experts_capturer.py   <- the fix
#   tests/r3_237_smoke.sbatch, tests/r3_compare_captures.py, tests/r3_rank_symmetry_repro.py (additive)
# NO csrc/CMake/.cu/.cpp/.h/pyproject/setup.py/requirements changed -> compiled
# CUDA/C++ kernels are byte-identical. vLLM is installed EDITABLE (-e .,
# source at /opt/vllm_build/vllm), so swapping the .py source + clearing pyc fully
# applies the merge WITHOUT a recompile. Mirrors rebake_vllm0202rc0_r3_cp_fixb.sh.
#
# IN  = skyrl_megatron_vllm0202rc0_r3_cp_fixb2.sif (live; USED BY RUNNING job 926043 -- read-only)
# OUT = skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif (new; validated, swapped by hand later)
# =============================================================================
set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
IN_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_cp_fixb2.sif
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_cp_fixb3.sif
VLLM_SRC=/e/scratch/jureap59/feuer1/vllm
TARGET_COMMIT=4d167a4af0550bc3edf288444cf81a5e28ad23ee
SIG='#237 (rank-symmetric epilogue)'

STAMP=$(date +%Y%m%d_%H%M%S)
BUILD=/tmp/sifrebake_237_$STAMP
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "=== $(date) host=$(hostname) #237 MERGE REBAKE STAMP=$STAMP ==="
echo "IN_SIF =$IN_SIF ($(stat -c%s "$IN_SIF") bytes)"
echo "OUT_SIF=$OUT_SIF"
echo "VLLM_SRC=$VLLM_SRC TARGET=$TARGET_COMMIT"
[ -d "$VLLM_SRC/vllm" ] || { echo "FATAL: vLLM fork source not found at $VLLM_SRC"; exit 2; }
GOT=$(cd "$VLLM_SRC" && git rev-parse HEAD)
echo "VLLM_SRC HEAD=$GOT"
[ "$GOT" = "$TARGET_COMMIT" ] || { echo "FATAL: clone HEAD $GOT != target $TARGET_COMMIT"; exit 2; }
CAP_REL=vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
grep -qF "$SIG" "$VLLM_SRC/$CAP_REL" || { echo "FATAL: merged signature missing in source capturer"; exit 2; }
echo "source capturer merged-sig count: $(grep -cF "$SIG" "$VLLM_SRC/$CAP_REL")"
df -h /tmp | tail -1; df -i /tmp | tail -1

echo "=== [1] sandbox from live CP prod SIF ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$IN_SIF"
echo "sandbox built."
DST=$SANDBOX/opt/vllm_build
[ -d "$DST/vllm" ] || { echo "FATAL: /opt/vllm_build/vllm missing in sandbox"; exit 3; }

echo "=== [1b] pre-swap signatures (expect OLD FIX B present, merged-sig absent) ==="
grep -c "#237 FIX B (rank-symmetric epilogue)" "$DST/$CAP_REL" || true
grep -cF "$SIG" "$DST/$CAP_REL" || true
head -1 "$DST/.vllm_commit" 2>/dev/null || echo "(no .vllm_commit)"

echo "=== [2] swap the merged capturer + additive tests; stamp commit ==="
cp -a "$VLLM_SRC/$CAP_REL" "$DST/$CAP_REL"
echo "  swapped $CAP_REL"
for t in r3_237_smoke.sbatch r3_compare_captures.py r3_rank_symmetry_repro.py ; do
  [ -f "$VLLM_SRC/tests/$t" ] && cp -a "$VLLM_SRC/tests/$t" "$DST/tests/$t" || true
done
echo "$TARGET_COMMIT" > "$DST/.vllm_commit"
( cd "$VLLM_SRC" && git describe --tags 2>/dev/null ) >> "$DST/.vllm_commit" || true

echo "=== [2b] clear stale bytecode for the swapped module ==="
rm -f "$DST/vllm/model_executor/layers/fused_moe/__pycache__/routed_experts_capturer."*.pyc 2>/dev/null || true

echo "=== [2c] post-swap signatures (expect merged-sig present, OLD FIX B gone) ==="
grep -cF "$SIG" "$DST/$CAP_REL"
grep -c "#237 FIX B (rank-symmetric epilogue)" "$DST/$CAP_REL" || true
grep -c "_device_staging = torch.empty_like" "$DST/$CAP_REL" || true

echo "=== [2d] in-sandbox import + signature checks ==="
apptainer exec --writable --nv --no-home --env VLLM_USE_FLASHINFER_SAMPLER=0 --env VLLM_ATTENTION_BACKEND=FLASH_ATTN "$SANDBOX" bash -lc '
  set -euo pipefail
  python -c "import vllm; print(\"vllm\", vllm.__version__, \"from\", vllm.__file__)"
  python -c "import torch; print(\"torch\", torch.__version__)"
  CAP=/opt/vllm_build/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
  python -c "import vllm.model_executor.layers.fused_moe.routed_experts_capturer as r; print(\"capturer import OK\", r.__file__)"
  python -c "src=open(\"'"$DST/$CAP_REL"'\").read(); print(\"merged-sig count\", src.count(\"#237 (rank-symmetric epilogue)\")); print(\"device_staging present\", \"_device_staging = torch.empty_like\" in src)"
  python -c "from vllm import ModelRegistry; a=set(ModelRegistry.get_supported_archs()); print({k:(k in a) for k in [\"Qwen3MoeForCausalLM\",\"Qwen3NextForCausalLM\",\"Gemma4ForCausalLM\"]})"
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
cap=os.path.join(sp,"vllm/model_executor/layers/fused_moe/routed_experts_capturer.py")
txt=open(cap).read()
print("capturer exists:", os.path.exists(cap))
print("MERGED #237 (rank-symmetric epilogue) count:", txt.count("#237 (rank-symmetric epilogue)"))
print("OLD #237 FIX B count (expect 0):", txt.count("#237 FIX B (rank-symmetric epilogue)"))
print("rank-symmetric _device_staging present:", "_device_staging = torch.empty_like" in txt)
import vllm.model_executor.layers.fused_moe.routed_experts_capturer as r
print("capturer import OK:", r.__file__)
print("in-SIF .vllm_commit:", open(os.path.join(sp,".vllm_commit")).read().strip().splitlines()[0] if os.path.exists(os.path.join(sp,".vllm_commit")) else "none")
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as C
print("transformers qwen3_moe:", "qwen3_moe" in C, "qwen3_next:", "qwen3_next" in C)
import skyrl_train, skyrl_gym; print("skyrl_train OK")
from vllm import ModelRegistry
archs=set(ModelRegistry.get_supported_archs())
for a in ("Qwen3MoeForCausalLM","Qwen3NextForCausalLM","Gemma4ForCausalLM"):
    print("  REGISTRY", a, a in archs)
print("ALL CORE CHECKS DONE")
PYEOF

echo "=== md5 ==="; md5sum "$OUT_SIF"
echo "=== removing build scratch ==="
rm -rf "$BUILD" 2>/dev/null || true
echo "=== $(date) #237 MERGE REBAKE COMPLETE OUT_SIF=$OUT_SIF ==="
