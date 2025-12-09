#!/bin/bash
#SBATCH --job-name=check-vllm-env
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:15:00
#SBATCH --mem=32G

# Simple diagnostic batch job for confirming that the dcagent environment
# can import vLLM/Torch CUDA extensions on an H200 node.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dcagent}"

_activate_conda() {
    local conda_root=""
    if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}/miniconda3" ]]; then
        conda_root="${SCRATCH}/miniconda3"
    elif [[ -d "${HOME}/miniconda3" ]]; then
        conda_root="${HOME}/miniconda3"
    fi

    if [[ -z "$conda_root" ]]; then
        echo "ERROR: Could not find miniconda3 under \$SCRATCH or \$HOME." >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
}

echo "=== Activating conda environment (${CONDA_ENV}) ==="
_activate_conda

echo
echo "=== Runtime environment ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo
nvidia-smi || true

echo
echo "=== vLLM package + attention backends ==="
python - <<'PY'
import importlib
import vllm
print("vllm package:", vllm.__file__)

def report(module_name, attr=None):
    try:
        module = importlib.import_module(module_name)
        target = module
        if attr:
            target = getattr(module, attr)
        path = getattr(target, "__file__", repr(target))
        print(f"{module_name}{'.' + attr if attr else ''}: {path}")
    except Exception as exc:  # pragma: no cover - diagnostic script
        print(f"{module_name}{'.' + attr if attr else ''}: FAILED -> {exc}")

report("vllm.attention.backends.flash_attn", "_flash_attn_mha")
report("vllm.attention.backends.flash_attn_triton", "_flash_attn_triton")
report("vllm", "_C")
PY

echo
echo "=== PyTorch CUDA info ==="
python - <<'PY'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA runtime reported by torch:", torch.version.cuda)
if not torch.cuda.is_available():
    print("CUDA is not available inside this job.")
else:
    device = torch.device("cuda")
    idx = device.index or 0
    print("Device name:", torch.cuda.get_device_name(idx))
    print("Device capability:", torch.cuda.get_device_capability(idx))
    print("Current BLAS handle:", torch.backends.cuda.matmul.get_blas_batch_limit())
PY
