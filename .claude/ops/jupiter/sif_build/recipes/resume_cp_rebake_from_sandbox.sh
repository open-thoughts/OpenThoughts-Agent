#!/bin/bash
# Resume the FSDP2-CP rebake from the EXISTING sandbox (sandbox extract + git
# checkout of marin 34c348b already completed; the first run only aborted on a
# wrong ExpertParallel probe BEFORE building the SIF). Re-runs the corrected
# probe + builds + validates the NEW SIF. SANDBOX path passed as $1.
set -euo pipefail
SANDBOX=${1:?usage: resume_cp_rebake_from_sandbox.sh <sandbox_dir>}
CONTAINERS=/e/scratch/jureap59/feuer1/containers
OUT_SIF=$CONTAINERS/skyrl_megatron_vllm0202rc0_r3_cp.sif
TITAN=/e/scratch/jureap59/feuer1/sif_pydeps_longctx_titan022
SHARED=/e/scratch/jureap59/feuer1/sif_pydeps
SK=$SANDBOX/opt/SkyRL

echo "=== $(date) RESUME from $SANDBOX ==="
[ -d "$SK/.git" ] || { echo "FATAL: $SK/.git missing"; exit 3; }
echo "sandbox SkyRL HEAD:"; git -C "$SK" rev-parse HEAD
[ "$(git -C "$SK" rev-parse HEAD)" = "34c348bfd260c176bd04fde85456a69f4f736086" ] || { echo "FATAL: sandbox not at 34c348b"; exit 4; }

echo "=== corrected in-sandbox probe ==="
apptainer exec --writable --nv --no-home \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env PYTHONPATH="$TITAN:$SHARED" \
  --bind /e/scratch \
  "$SANDBOX" bash -lc '
  set -euo pipefail
  python -c "import skyrl_train, skyrl_gym; print(\"skyrl_train OK\")"
  python -c "import inspect; from skyrl_train.distributed.fsdp_utils import create_device_mesh as f; s=str(inspect.signature(f)); assert \"cp_size\" in s; print(\"CP DIM PRESENT\", s)"
  python -c "import skyrl_train.distributed.cp_utils; print(\"cp_utils OK\")"
  python -c "from torchtitan.distributed.expert_parallel import ExpertParallel; ExpertParallel(); print(\"torchtitan ExpertParallel instantiates OK\")"
  python -c "from skyrl_train.distributed.expert_parallel import DeepEPExpertParallel; print(\"SkyRL DeepEPExpertParallel OK\")"
  python -c "import vllm.v1.attention.ops.common as c; src=open(c.__file__).read(); print(\"vLLM DCP fix out_fp32 count\", src.count(\"out_fp32\"))"
'

echo "=== build NEW SIF -> $OUT_SIF ==="
STAMP=$(date +%Y%m%d_%H%M%S)
TMP_OUT=/tmp/cp_out_$STAMP.sif
export APPTAINER_TMPDIR=/tmp/aptmp_$STAMP; export APPTAINER_CACHEDIR=/tmp/apcache_$STAMP
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
rm -f "$TMP_OUT"
apptainer build "$TMP_OUT" "$SANDBOX"
mv -f "$TMP_OUT" "$OUT_SIF"
echo "NEW SIF: $(stat -c%s $OUT_SIF) bytes"; ls -la "$OUT_SIF"

echo "=== validate fresh SIF ==="
apptainer exec --nv \
  --env VLLM_USE_FLASHINFER_SAMPLER=0 \
  --env PYTHONPATH="$TITAN:$SHARED" \
  --bind /e/scratch \
  "$OUT_SIF" python - <<'PYEOF' || echo "WARN: validation nonzero (SIF written)"
import os, inspect, torch, vllm
print("torch", torch.__version__, "vllm", vllm.__version__)
import vllm.v1.attention.ops.common as c
src=open(c.__file__).read()
print("vLLM DCP fix out_fp32 count", src.count("out_fp32"), "cp_lse_ag_out_rs", "cp_lse_ag_out_rs" in src)
import vllm.config.vllm as vcfg
print("guard-lift VLLM_ALLOW_ROUTED_EXPERTS_DCP:", "VLLM_ALLOW_ROUTED_EXPERTS_DCP" in open(vcfg.__file__).read())
import skyrl_train, skyrl_gym; print("skyrl_train OK")
from skyrl_train.distributed.fsdp_utils import create_device_mesh
print("create_device_mesh", str(inspect.signature(create_device_mesh)))
assert "cp_size" in str(inspect.signature(create_device_mesh))
import skyrl_train.distributed.cp_utils; print("cp_utils OK")
from torchtitan.distributed.expert_parallel import ExpertParallel; print("torchtitan ExpertParallel OK")
import torchtitan; print("torchtitan", getattr(torchtitan,"__version__","n/a"))
from vllm import ModelRegistry
print("Qwen3MoeForCausalLM in registry:", "Qwen3MoeForCausalLM" in set(ModelRegistry.get_supported_archs()))
print("ALL CORE CHECKS DONE")
PYEOF

echo "=== md5 of new SIF ==="; md5sum "$OUT_SIF"
echo "=== cleanup sandbox + scratch ==="
rm -rf "$SANDBOX" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" 2>/dev/null || true
echo "=== $(date) RESUME COMPLETE OUT_SIF=$OUT_SIF ==="
