#!/bin/bash
#SBATCH --job-name=build-flash-attn
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

# Batch job that compiles and installs flash-attention from source inside the
# requested conda environment. The build runs with --no-build-isolation so that
# it can re-use the already provisioned CUDA/Torch stack.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-dcagent}"
FLASH_ATTN_REPO="${FLASH_ATTN_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-main}"
FLASH_ATTN_BUILD_DIR="${FLASH_ATTN_BUILD_DIR:-${SCRATCH:-${HOME}}/flash-attention}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"
USE_PIP_CUDA="${USE_PIP_CUDA:-1}"
ALLOW_SYSTEM_NVCC="${ALLOW_SYSTEM_NVCC:-0}"

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

_sync_repo() {
    local repo_path="$1"
    if [[ ! -d "$repo_path/.git" ]]; then
        echo "Cloning $FLASH_ATTN_REPO to $repo_path"
        git clone "$FLASH_ATTN_REPO" "$repo_path"
    fi

    echo "Checking out ${FLASH_ATTN_REF}"
    git -C "$repo_path" fetch --all --tags
    git -C "$repo_path" checkout "$FLASH_ATTN_REF"
    git -C "$repo_path" submodule update --init --recursive
}

_prepend_path() {
    local var_name="$1"
    local new_value="$2"
    local current="${!var_name-}"
    if [[ -z "$new_value" ]]; then
        return
    fi
    if [[ -n "$current" ]]; then
        export "${var_name}=${new_value}:${current}"
    else
        export "${var_name}=${new_value}"
    fi
}

_configure_pip_cuda() {
    if [[ "${USE_PIP_CUDA}" == "0" ]]; then
        echo "USE_PIP_CUDA=0 -> leaving system CUDA paths untouched."
        return
    fi

    local info=""
    if ! info="$(python - <<'PY'
import importlib.util
import importlib.metadata
import pathlib

def pkg_dir(name):
    spec = importlib.util.find_spec(name)
    if not spec or not spec.origin:
        return ""
    return pathlib.Path(spec.origin).parent

def find_nvcc():
    candidates = []
    pkg = pkg_dir("nvidia.cuda_nvcc")
    if pkg:
        candidates.append(pkg)
    try:
        dist = importlib.metadata.distribution("nvidia-cuda-nvcc-cu12")
    except importlib.metadata.PackageNotFoundError:
        dist = None
    if dist:
        candidates.append(pathlib.Path(dist.locate_file("")))
        candidates.append(pathlib.Path(dist.locate_file("bin")))
    for base in candidates:
        if not base:
            continue
        paths = []
        base = pathlib.Path(base)
        if base.name == "bin":
            paths.append(base / "nvcc")
        paths.append(base / "bin" / "nvcc")
        for candidate in paths:
            if candidate.exists():
                return candidate
        try:
            hit = next((p for p in base.rglob("nvcc") if p.is_file()), None)
        except (PermissionError, FileNotFoundError):
            hit = None
        if hit:
            return hit
    return ""

runtime = pkg_dir("nvidia.cuda_runtime")
if not runtime:
    raise SystemExit(1)
nvcc_path = find_nvcc()
print(runtime)
print(nvcc_path)
PY
)"; then
        echo "WARNING: Could not locate pip-installed CUDA runtime; continuing with system CUDA." >&2
        return
    fi

    local cuda_root nvcc_bin nvcc_dir
    cuda_root="$(awk 'NR==1' <<<"$info")"
    nvcc_bin="$(awk 'NR==2' <<<"$info")"
    nvcc_dir=""

    echo "Configuring pip-installed CUDA runtime at: ${cuda_root}"
    export CUDA_HOME="${cuda_root}"
    export CUDA_PATH="${cuda_root}"
    export CUDATOOLKIT_ROOT_DIR="${cuda_root}"

    if [[ -n "$nvcc_bin" && -x "$nvcc_bin" ]]; then
        nvcc_dir="$(dirname "$nvcc_bin")"
        echo "Using pip-installed nvcc at: ${nvcc_bin}"
    elif [[ -x "${cuda_root}/bin/nvcc" ]]; then
        nvcc_dir="${cuda_root}/bin"
        nvcc_bin="${cuda_root}/bin/nvcc"
        echo "Using pip CUDA runtime-provided nvcc at: ${nvcc_bin}"
    fi

    if [[ -z "$nvcc_dir" ]]; then
        if [[ "${ALLOW_SYSTEM_NVCC}" == "1" ]]; then
            echo "WARNING: pip CUDA runtime lacks nvcc; falling back to system PATH (ALLOW_SYSTEM_NVCC=1)." >&2
        else
            cat >&2 <<'EOF'
ERROR: pip-installed CUDA runtime does not provide nvcc (missing bin/nvcc).
Install the nvcc wheel (e.g. `pip install nvidia-cuda-nvcc-cu12`) or set USE_PIP_CUDA=0 / ALLOW_SYSTEM_NVCC=1 to continue with the system toolkit.
EOF
            exit 1
        fi
    fi

    local lib32="${cuda_root}/lib"
    local lib64="${cuda_root}/lib64"
    local include_dir="${cuda_root}/include"

    if [[ -d "$lib32" ]]; then
        _prepend_path LD_LIBRARY_PATH "$lib32"
        _prepend_path LIBRARY_PATH "$lib32"
    fi
    if [[ -d "$lib64" ]]; then
        _prepend_path LD_LIBRARY_PATH "$lib64"
        _prepend_path LIBRARY_PATH "$lib64"
    fi
    if [[ -d "$include_dir" ]]; then
        _prepend_path CPATH "$include_dir"
    fi
    if [[ -n "$nvcc_dir" && -d "$nvcc_dir" ]]; then
        _prepend_path PATH "$nvcc_dir"
    elif [[ -d "${cuda_root}/bin" ]]; then
        _prepend_path PATH "${cuda_root}/bin"
    fi

    python - <<'PY'
import importlib.metadata
pkg = "nvidia-cuda-runtime-cu12"
try:
    version = importlib.metadata.version(pkg)
except importlib.metadata.PackageNotFoundError:
    version = "<unknown>"
print(f"Detected {pkg} version: {version}")
PY
}

echo "=== Activating conda environment (${CONDA_ENV}) ==="
_activate_conda
_configure_pip_cuda

echo
echo "=== Runtime environment ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Python: $(which python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "FLASH_ATTN_REPO=${FLASH_ATTN_REPO}"
echo "FLASH_ATTN_REF=${FLASH_ATTN_REF}"
echo "FLASH_ATTN_BUILD_DIR=${FLASH_ATTN_BUILD_DIR}"
echo
nvidia-smi || true

echo
echo "=== CUDA paths ==="
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "CUDA_PATH=${CUDA_PATH:-<unset>}"
echo "CUDATOOLKIT_ROOT_DIR=${CUDATOOLKIT_ROOT_DIR:-<unset>}"
echo "PATH entries with CUDA:"
tr ':' '\n' <<< "${PATH}" | grep -i 'cuda' || true
if command -v nvcc >/dev/null 2>&1; then
    echo "nvcc=$(command -v nvcc)"
    nvcc --version || true
else
    echo "nvcc=<not found in PATH>"
fi

mkdir -p "$FLASH_ATTN_BUILD_DIR"
_sync_repo "$FLASH_ATTN_BUILD_DIR"

# Default to Hopper (SM90) if the user did not provide an explicit arch list.
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="90"
fi

echo
echo "=== Building flash-attention (TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}) ==="
pushd "$FLASH_ATTN_BUILD_DIR" >/dev/null
python -m pip install --no-build-isolation -v ${PIP_EXTRA_ARGS} .
popd >/dev/null

echo
echo "=== Installed flash-attention version ==="
python - <<'PY'
import importlib
import pkg_resources
name = "flash-attn"
spec = importlib.util.find_spec("flash_attn")
print("Module path:", getattr(spec, "origin", spec))
dist = pkg_resources.get_distribution(name)
print("Installed distribution:", dist)
PY

echo
echo "flash-attention build completed successfully."
