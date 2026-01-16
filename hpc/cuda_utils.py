"""CUDA environment detection utilities for HPC clusters.

This module provides utilities to automatically detect and configure CUDA
environment variables, particularly for clusters like Perlmutter where the
CUDA SDK has a complex directory structure.

Usage:
    from hpc.cuda_utils import setup_cuda_environment

    # In Python (e.g., SFTJobRunner):
    cuda_env = setup_cuda_environment()
    for key, value in cuda_env.items():
        os.environ[key] = value

    # Or get bash export statements for sbatch templates:
    from hpc.cuda_utils import get_cuda_exports_bash
    bash_code = get_cuda_exports_bash()
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Optional


def detect_cuda_home() -> Optional[str]:
    """Derive CUDA_HOME from nvcc when the module doesn't set it.

    Returns:
        Path to CUDA installation, or None if not found.
    """
    # First check if CUDA_HOME is already set
    if cuda_home := os.environ.get("CUDA_HOME"):
        return cuda_home

    # Try to find nvcc and derive CUDA_HOME from it
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        # nvcc is typically at CUDA_HOME/bin/nvcc
        return str(Path(nvcc_path).parent.parent)
    return None


def get_cuda_math_libs_path(cuda_home: str) -> Optional[str]:
    """Find CUDA math_libs path from SDK directory structure.

    NERSC/Perlmutter layout example:
        cuda_home = /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/cuda/12.6
        math_libs = /opt/nvidia/hpc_sdk/Linux_x86_64/24.9/math_libs/12.6/lib64

    Args:
        cuda_home: Path to CUDA installation.

    Returns:
        Path to math_libs/lib64, or None if not found.
    """
    cuda_path = Path(cuda_home)
    cuda_version = cuda_path.name  # e.g., "12.6"
    sdk_root = cuda_path.parent.parent  # e.g., /opt/nvidia/hpc_sdk/Linux_x86_64/24.9

    math_lib_path = sdk_root / "math_libs" / cuda_version / "lib64"
    if math_lib_path.is_dir():
        return str(math_lib_path)
    return None


def get_conda_curand_path() -> Optional[str]:
    """Find NVIDIA curand wheels bundled with conda env.

    Returns:
        Path to curand lib directory, or None if not found.
    """
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return None

    # Try common Python versions
    for py_ver in ["3.12", "3.11", "3.10", "3.9"]:
        curand_path = Path(conda_prefix) / f"lib/python{py_ver}/site-packages/nvidia/curand/lib"
        if curand_path.is_dir():
            return str(curand_path)
    return None


def _prepend_path(env_vars: Dict[str, str], key: str, path: str) -> None:
    """Prepend path to an environment variable.

    Args:
        env_vars: Dictionary of environment variables to update.
        key: Environment variable name.
        path: Path to prepend.
    """
    existing = env_vars.get(key) or os.environ.get(key, "")
    if existing:
        env_vars[key] = f"{path}:{existing}"
    else:
        env_vars[key] = path


def setup_cuda_environment() -> Dict[str, str]:
    """Detect and configure CUDA environment variables.

    This replicates the complex CUDA detection logic from Perlmutter's
    sbatch template, covering:
    - CUDA_HOME detection from nvcc
    - CUDA lib64 paths
    - CUDA math_libs paths (HPC SDK structure)
    - Conda curand wheel paths
    - Conda compiler paths (CC, CXX, CUDAHOSTCXX)

    Returns:
        Dictionary of environment variables to export.
    """
    env_vars: Dict[str, str] = {}

    # Step 1: Detect CUDA_HOME
    cuda_home = detect_cuda_home()
    if cuda_home:
        env_vars["CUDA_HOME"] = cuda_home

        # Step 2: Add CUDA lib64 to library paths
        cuda_lib = Path(cuda_home) / "lib64"
        if cuda_lib.is_dir():
            _prepend_path(env_vars, "LIBRARY_PATH", str(cuda_lib))

        # Step 3: Add math_libs path (for HPC SDK installations)
        math_libs = get_cuda_math_libs_path(cuda_home)
        if math_libs:
            _prepend_path(env_vars, "LIBRARY_PATH", math_libs)
            _prepend_path(env_vars, "LD_LIBRARY_PATH", math_libs)

    # Step 4: Add conda curand libs
    curand_path = get_conda_curand_path()
    if curand_path:
        _prepend_path(env_vars, "LIBRARY_PATH", curand_path)
        _prepend_path(env_vars, "LD_LIBRARY_PATH", curand_path)

    # Step 5: Add conda lib and compiler paths
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = Path(conda_prefix) / "lib"
        if conda_lib.is_dir():
            _prepend_path(env_vars, "LIBRARY_PATH", str(conda_lib))
            _prepend_path(env_vars, "LD_LIBRARY_PATH", str(conda_lib))

        # Set compiler environment variables
        conda_bin = Path(conda_prefix) / "bin"
        gcc = conda_bin / "x86_64-conda-linux-gnu-gcc"
        gxx = conda_bin / "x86_64-conda-linux-gnu-g++"
        gfortran = conda_bin / "x86_64-conda-linux-gnu-gfortran"

        if gcc.is_file():
            env_vars["CC"] = str(gcc)
        if gxx.is_file():
            env_vars["CXX"] = str(gxx)
            env_vars["CUDAHOSTCXX"] = str(gxx)
        if gfortran.is_file():
            env_vars["FC"] = str(gfortran)

    return env_vars


def get_cuda_exports_bash() -> str:
    """Generate bash export statements for CUDA environment setup.

    This can be used in sbatch templates as an alternative to the
    Python-based setup.

    Returns:
        Multi-line string of bash export statements.
    """
    env_vars = setup_cuda_environment()
    if not env_vars:
        return "# No CUDA environment variables detected"

    lines = ["# CUDA environment variables (auto-detected)"]
    for key, value in env_vars.items():
        lines.append(f'export {key}="{value}"')
    return "\n".join(lines)


def apply_cuda_environment() -> None:
    """Apply CUDA environment variables to the current process.

    This is a convenience function that calls setup_cuda_environment()
    and applies the results to os.environ.
    """
    cuda_env = setup_cuda_environment()
    for key, value in cuda_env.items():
        os.environ[key] = value


if __name__ == "__main__":
    # When run directly, print the detected environment
    print("Detecting CUDA environment...")
    env = setup_cuda_environment()
    if env:
        print("\nDetected environment variables:")
        for key, value in sorted(env.items()):
            print(f"  {key}={value}")
    else:
        print("No CUDA environment variables detected.")
