"""Model-specific utilities for HPC launchers.

This module provides utilities for handling model-specific requirements
that differ from the standard vLLM workflow, such as:
- Custom tokenizer downloads (GPT-OSS tiktoken)
- Model-specific environment variables
- Special initialization requirements

These utilities are shared across all execution paths:
- Local runners (data/local/run_tracegen.py, eval/local/run_eval.py)
- Cloud launchers (data/cloud/launch_tracegen_cloud.py, eval/cloud/launch_eval_cloud.py)
- HPC SLURM launchers (hpc/launch.py --job_type datagen)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Tiktoken encoding files required for OpenAI GPT-OSS models
# See: https://github.com/vllm-project/vllm/issues/22525
GPT_OSS_TIKTOKEN_FILES = [
    "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
]


def is_gpt_oss_model(model: Optional[str]) -> bool:
    """Check if the model is an OpenAI GPT-OSS model requiring tiktoken setup.

    Args:
        model: Model identifier (e.g., "openai/gpt-oss-120b")

    Returns:
        True if model requires GPT-OSS tiktoken setup
    """
    if not model:
        return False
    return model.lower().startswith("openai/gpt-oss")


def setup_gpt_oss_tiktoken(
    cache_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[Path, Dict[str, str]]:
    """Download tiktoken encoding files for GPT-OSS models.

    OpenAI's GPT-OSS models require tiktoken encoding files that are normally
    downloaded at runtime. In containerized/offline environments, these must
    be pre-downloaded and mounted.

    See: https://github.com/vllm-project/vllm/issues/22525

    Args:
        cache_dir: Directory to store encoding files. Defaults to ~/.cache/tiktoken_encodings
        verbose: Print progress messages

    Returns:
        Tuple of (encodings_dir, env_vars_dict)
        - encodings_dir: Path to directory containing encoding files
        - env_vars_dict: Dict with TIKTOKEN_ENCODINGS_BASE set
    """
    import urllib.request
    import urllib.error

    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "tiktoken_encodings"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env_vars = {"TIKTOKEN_ENCODINGS_BASE": str(cache_dir)}

    for url in GPT_OSS_TIKTOKEN_FILES:
        filename = url.split("/")[-1]
        local_path = cache_dir / filename

        if local_path.exists():
            if verbose:
                print(f"[tiktoken] Found cached: {filename}")
            continue

        if verbose:
            print(f"[tiktoken] Downloading: {filename}")

        try:
            urllib.request.urlretrieve(url, local_path)
            if verbose:
                print(f"[tiktoken] Downloaded: {local_path}")
        except urllib.error.URLError as e:
            print(f"[tiktoken] Warning: Failed to download {filename}: {e}", file=sys.stderr)
            # Don't fail - vLLM might still work if files exist elsewhere

    return cache_dir, env_vars


def get_model_specific_env_vars(model: Optional[str], verbose: bool = True) -> Dict[str, str]:
    """Get model-specific environment variables.

    This is the main entry point for model-specific setup. It checks the model
    and returns any required environment variables.

    Args:
        model: Model identifier
        verbose: Print progress messages

    Returns:
        Dict of environment variables to set for vLLM
    """
    env_vars: Dict[str, str] = {}

    # GPT-OSS models need tiktoken encodings
    if is_gpt_oss_model(model):
        _, tiktoken_env = setup_gpt_oss_tiktoken(verbose=verbose)
        env_vars.update(tiktoken_env)

    return env_vars
