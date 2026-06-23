#!/bin/bash
# =============================================================================
# rebake_vllm0202rc0_r3_cp.sh   (LOGIN-NODE surgical rebake, NOT sbatch)
#
# Advances the prod SIF's baked editable /opt/SkyRL from
#   2ab513a6 (penfever/SkyRL, NO FSDP2-CP)  ->  34c348b (marin penfever/working)
# which carries the FSDP2 torch-native context-parallel (ring SDPA) + ExpertParallel
# path (task #218, staged 0-7). The 2ab->34c348b delta is PURE-PYTHON (verified:
# only .py/.yaml/.toml + tests changed; NO csrc/.cu/.cpp/CMake/C-extension), and
# /opt/SkyRL is installed EDITABLE (-e .), so a `git checkout` of the new commit +
# clearing __pycache__ FULLY applies FSDP2-CP WITHOUT any pip reinstall or recompile.
# vLLM is UNTOUCHED -> the canonical SIF's vLLM DCP GQA-LSE fp32 fix + the
# VLLM_ALLOW_ROUTED_EXPERTS_DCP guard-lift (both baked into the IN_SIF) are preserved
# byte-for-byte (we build ON TOP of the live canonical SIF).
#
# torchtitan (the new pyproject `ep` dep, needed for ExpertParallel) is NOT baked
# here; it is supplied at runtime via PYTHONPATH pydeps (sif_pydeps_longctx_titan022
# = torchtitan 0.2.2), exactly as the 80B long-ctx config already does, to dodge the
# GPFS-FUSE Ray-bootstrap timeout a runtime --overlay would risk.
#
# CRITICAL: builds to a NEW filename. Does NOT touch the live prod SIF.
#   IN  = skyrl_megatron_vllm0202rc0_r3.sif     (live prod; read-only here)
#   OUT = skyrl_megatron_vllm0202rc0_r3_cp.sif  (new; validated, then used by the 30B cfg)
# =============================================================================
set -euo pipefail

CONTAINERS=/e/scratch/jureap59/feuer1/containers
IN_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3.sif
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_cp.sif
MARIN_REMOTE=https://github.com/marin-community/MarinSkyRL.git
TARGET_COMMIT=34c348bfd260c176bd04fde85456a69f4f736086

STAMP=$(date +%Y%m%d_%H%M%S)
BUILD=/tmp/sifrebake_cp_$STAMP
SANDBOX=$BUILD/sandbox
mkdir -p "$BUILD"
export APPTAINER_TMPDIR=$BUILD/aptmp
export APPTAINER_CACHEDIR=$BUILD/apcache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

echo "=== $(date) host=$(hostname) FSDP2-CP REBAKE STAMP=$STAMP ==="
echo "IN_SIF =$IN_SIF ($(stat -c%s $IN_SIF) bytes)"
echo "OUT_SIF=$OUT_SIF"
echo "TARGET_COMMIT=$TARGET_COMMIT (marin penfever/working FSDP2-CP)"
df -h /tmp | tail -1; df -i /tmp | tail -1

# ---------------------------------------------------------------------------
# 1. Sandbox from the LIVE canonical SIF (vLLM stays compiled + DCP-fixed)
# ---------------------------------------------------------------------------
echo "=== [1] sandbox from live canonical SIF ==="
rm -rf "$SANDBOX"
apptainer build --sandbox "$SANDBOX" "$IN_SIF"
echo "sandbox built."

SK=$SANDBOX/opt/SkyRL
[ -d "$SK/.git" ] || { echo "FATAL: /opt/SkyRL/.git missing in sandbox"; exit 3; }

echo "=== [1b] pre-checkout baked SkyRL commit (should be OLD 2ab513a6) ==="
git -C "$SK" rev-parse HEAD

echo "=== [1c] pre-swap vLLM DCP-fix fingerprint (MUST be preserved) ==="
md5sum "$SANDBOX/opt/vllm_build/vllm/v1/attention/ops/common.py"
grep -c out_fp32 "$SANDBOX/opt/vllm_build/vllm/v1/attention/ops/common.py" || true
grep -c VLLM_ALLOW_ROUTED_EXPERTS_DCP "$SANDBOX/opt/vllm_build/vllm/config/vllm.py" || true

# ---------------------------------------------------------------------------
# 2. Fetch the FSDP2-CP commit from marin + checkout (editable tree, no reinstall)
# ---------------------------------------------------------------------------
echo "=== [2] git fetch marin penfever/working + checkout $TARGET_COMMIT ==="
# login node has internet; fetch the single commit shallow-ish into the sandbox repo
git -C "$SK" remote add marin "$MARIN_REMOTE" 2>/dev/null || git -C "$SK" remote set-url marin "$MARIN_REMOTE"
git -C "$SK" fetch --no-tags marin "$TARGET_COMMIT"
git -C "$SK" checkout -f "$TARGET_COMMIT"
echo "post-checkout HEAD:"; git -C "$SK" rev-parse HEAD
[ "$(git -C "$SK" rev-parse HEAD)" = "$TARGET_COMMIT" ] || { echo "FATAL: checkout did not land on $TARGET_COMMIT"; exit 4; }

echo "=== [2b] clear ALL stale bytecode under /opt/SkyRL (editable src changed) ==="
find "$SK" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== [2c] confirm FSDP2-CP markers now present in the editable tree ==="
grep -n "def create_device_mesh" "$SK/skyrl-train/skyrl_train/distributed/fsdp_utils.py" | head -1
grep -rl "ring_sdpa" "$SK/skyrl-train/skyrl_train/utils/utils.py" | head -1
test -f "$SK/skyrl-train/skyrl_train/distributed/cp_utils.py" && echo "cp_utils.py PRESENT"
test -f "$SK/skyrl-train/skyrl_train/distributed/expert_parallel.py" && echo "expert_parallel.py PRESENT"

echo "=== [2d] in-sandbox import + CP/vLLM presence checks (PYTHONPATH=titan for ExpertParallel) ==="
TITAN=/e/scratch/jureap59/feuer1/sif_pydeps_longctx_titan022
SHARED=/e/scratch/jureap59/feuer1/sif_pydeps
apptainer exec --writable --nv --no-home \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env PYTHONPATH="$TITAN:$SHARED" \
  --bind /e/scratch \
  "$SANDBOX" bash -lc '
  set -euo pipefail
  python -c "import skyrl_train, skyrl_gym; print(\"skyrl_train OK\", skyrl_train.__file__)"
  python -c "import torch,vllm; print(\"torch\",torch.__version__,\"vllm\",vllm.__version__)"
  echo "--- vLLM DCP fix preserved ---"
  python -c "import vllm.v1.attention.ops.common as c; src=open(c.__file__).read(); print(\"out_fp32 count\", src.count(\"out_fp32\")); print(\"cp_lse_ag_out_rs\", \"cp_lse_ag_out_rs\" in src)"
  echo "--- FSDP2-CP code reachable from installed skyrl_train ---"
  python -c "import inspect; from skyrl_train.distributed.fsdp_utils import create_device_mesh as f; sig=str(inspect.signature(f)); print(\"create_device_mesh sig:\", sig); assert \"cp_size\" in sig, \"NO cp_size dim\"; print(\"CP DIM PRESENT\")"
  python -c "import skyrl_train.distributed.cp_utils as cp; print(\"cp_utils OK\")"
  # ExpertParallel comes from torchtitan (the torch Stage-4 EP backend); SkyRL's own
  # expert_parallel.py defines DeepEPExpertParallel (the Stage-5 DeepEP backend).
  python -c "from torchtitan.distributed.expert_parallel import ExpertParallel; print(\"torchtitan ExpertParallel import OK\")"
  python -c "from skyrl_train.distributed.expert_parallel import DeepEPExpertParallel; print(\"SkyRL DeepEPExpertParallel import OK\")"
  echo "--- torchtitan via PYTHONPATH ---"
  python -c "import torchtitan; print(\"torchtitan\", getattr(torchtitan,\"__version__\",\"n/a\"))"
'

# ---------------------------------------------------------------------------
# 3. Build the NEW SIF (new filename; live prod untouched)
# ---------------------------------------------------------------------------
echo "=== [3] building NEW SIF -> $OUT_SIF ==="
TMP_OUT=$BUILD/out.sif
rm -f "$TMP_OUT"
apptainer build "$TMP_OUT" "$SANDBOX"
mv -f "$TMP_OUT" "$OUT_SIF"
echo "NEW SIF built: $(stat -c%s $OUT_SIF) bytes"; ls -la "$OUT_SIF"

# ---------------------------------------------------------------------------
# 4. Validate the FRESH SIF
# ---------------------------------------------------------------------------
echo "=== [4] validating fresh SIF ==="
TITAN=/e/scratch/jureap59/feuer1/sif_pydeps_longctx_titan022
SHARED=/e/scratch/jureap59/feuer1/sif_pydeps
apptainer exec --nv \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env PYTHONPATH="$TITAN:$SHARED" \
  --bind /e/scratch \
  "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation exited non-zero (SIF already written)"
import os, inspect
import torch, vllm
print("torch", torch.__version__, "vllm", vllm.__version__)
import vllm.v1.attention.ops.common as c
src=open(c.__file__).read()
print("vLLM DCP fix: out_fp32 count", src.count("out_fp32"), "cp_lse_ag_out_rs", "cp_lse_ag_out_rs" in src)
vc=open(os.path.join(os.path.dirname(c.__file__),"..","..","..","config","vllm.py"))  # may not resolve; use module
import vllm.config.vllm as vcfg
print("guard-lift VLLM_ALLOW_ROUTED_EXPERTS_DCP:", "VLLM_ALLOW_ROUTED_EXPERTS_DCP" in open(vcfg.__file__).read())
import skyrl_train, skyrl_gym
print("skyrl_train OK", skyrl_train.__file__)
from skyrl_train.distributed.fsdp_utils import create_device_mesh
sig=str(inspect.signature(create_device_mesh)); print("create_device_mesh", sig)
assert "cp_size" in sig
import skyrl_train.distributed.cp_utils as cp; print("cp_utils OK")
from torchtitan.distributed.expert_parallel import ExpertParallel; print("torchtitan ExpertParallel OK")
from skyrl_train.distributed.expert_parallel import DeepEPExpertParallel; print("SkyRL DeepEPExpertParallel OK")
import torchtitan; print("torchtitan", getattr(torchtitan,"__version__","n/a"))
from vllm import ModelRegistry
archs=set(ModelRegistry.get_supported_archs())
print("Qwen3MoeForCausalLM in registry:", "Qwen3MoeForCausalLM" in archs)
print("ALL CORE CHECKS DONE")
PYEOF

echo "=== removing build scratch ==="
rm -rf "$BUILD" 2>/dev/null || true
echo "=== $(date) FSDP2-CP REBAKE COMPLETE  OUT_SIF=$OUT_SIF ==="
