"""Environment setup for Iris task containers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def default_secrets_env() -> str | None:
    """Return the default launch-host secrets file if one exists."""
    default = os.environ.get("OT_AGENT_SECRETS_ENV") or os.path.expanduser(
        "~/Documents/secrets.env"
    )
    return default if os.path.isfile(default) else None


def load_secrets_env_into_os_environ(secrets_env: str | None) -> int:
    """Read ``secrets_env`` (KEY=VALUE) into ``os.environ`` on the launch host."""
    if not secrets_env:
        return 0
    path = Path(secrets_env).expanduser().resolve()
    if not path.is_file():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if not k:
            continue
        os.environ[k] = v  # file overrides shell
        loaded += 1
    return loaded


def apply_iris_runtime_env(
    *,
    env_vars: dict[str, str],
    args: argparse.Namespace,
    remote_output_dir: str,
    extras: list[str],
) -> None:
    """Apply OT-Agent Iris runtime defaults in-place to ``env_vars``."""
    # iris-serve gating. iris runs the entrypoint on EVERY VM of the slice
    # (one task/VM; adjust_tpu_replicas scales replicas=1 -> vm_count), so
    # the worker's LocalHarborRunner.run() must (a) bring up ONE cross-host
    # Ray cluster via scripts/vllm/start_vllm_iris_controller.py instead of
    # the SLURM/single-host start_vllm_ray_controller.py, and (b) gate
    # harbor to the driver rank (IRIS_TASK_ID==0). This env var is the
    # signal - it is only ever set here, on the iris entrypoint path. The
    # rendezvous dir is a shared gs:// location (under the job's GCS output
    # prefix) where rank 0 publishes the Ray head IP for the worker ranks.
    env_vars.setdefault("OT_AGENT_IRIS_SERVE", "1")
    env_vars.setdefault(
        "OT_AGENT_IRIS_RENDEZVOUS_DIR",
        f"{remote_output_dir.rstrip('/')}/_ray_rendezvous",
    )

    # Wire the iris controller's XLA persistent cache to the same
    # region-matched bucket we picked above. The worker appends
    # /<cpu_tag>/<model_tag>/ to namespace per-microarch (the only
    # axis the JAX cache key doesn't already discriminate; see
    # [[xla-persistent-cache-cross-host-poison]]). One shared base
    # across all jobs is fine - JAX hashes the HLO into per-config
    # subdirs of its own beneath the dir we hand it. Disable with
    # OT_AGENT_XLA_CACHE_BASE=disabled in the environment.
    if args.gcs_output_dir:
        cache_root = args.gcs_output_dir.rstrip("/").rsplit("/ot-agent", 1)[0]
        env_vars.setdefault(
            "OT_AGENT_XLA_CACHE_BASE",
            f"{cache_root}/ot-agent/xla_cache",
        )

    # OT-Agent's build_support.py syncs the sft/llamafactory git submodule
    # at every setuptools.build_meta call (i.e. every editable install),
    # even when no sft-* extra is being installed. Inside the iris worker
    # container there's no git remote configured for that submodule, so
    # the sync errors out with exit 128. The build_support helper already
    # supports an escape hatch - opt in when no sft-* extra is requested.
    if not any(e.startswith("sft-") for e in extras):
        env_vars.setdefault("OT_AGENT_SKIP_SFT_SYNC", "1")

    # Force uv and subprocesses to use /app/.venv and copied wheels at runtime.
    env_vars.setdefault("UV_PROJECT_ENVIRONMENT", "/app/.venv")
    env_vars.setdefault("VIRTUAL_ENV", "/app/.venv")
    env_vars.setdefault("UV_LINK_MODE", "copy")
    env_vars.setdefault("OT_AGENT_INHERIT_SUBPROC_LOGS", "1")

    env_vars.setdefault("VLLM_SKIP_FLAG_DISCOVERY", "1")
    env_vars.setdefault("VLLM_SKIP_RAY_PROBE", "1")
    env_vars.setdefault("MODEL_IMPL_TYPE", "vllm")

    # Run:AI Model Streamer config for S3-compatible safetensor reads.
    env_vars.setdefault("RUNAI_STREAMER_S3_USE_VIRTUAL_ADDRESSING", "False")
    env_vars.setdefault("AWS_EC2_METADATA_DISABLED", "true")

    _forward_launcher_env(env_vars)
    _load_worker_secrets_env(env_vars, getattr(args, "secrets_env", None))
    _alias_s3_credentials(env_vars)


def _forward_launcher_env(env_vars: dict[str, str]) -> None:
    launcher_env_passthrough = (
        "DAYTONA_API_KEY",
        "DAYTONA_JWT_TOKEN",
        "DAYTONA_ORGANIZATION_ID",
        "DAYTONA_API_URL",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    )
    for key in launcher_env_passthrough:
        value = os.environ.get(key)
        if value:
            env_vars.setdefault(key, value)


def _load_worker_secrets_env(env_vars: dict[str, str], secrets_env: str | None) -> None:
    if not secrets_env:
        return
    secrets_path = Path(secrets_env).expanduser().resolve()
    if not secrets_path.exists():
        raise FileNotFoundError(f"--secrets-env file not found: {secrets_path}")
    loaded: list[str] = []
    for raw_line in secrets_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue  # malformed; skip
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        if not k:
            continue
        env_vars[k] = v  # file values override passthrough
        loaded.append(k)
    print(
        f"[iris] Secrets:    loaded {len(loaded)} entries from "
        f"{secrets_path}: {', '.join(sorted(loaded))}",
        flush=True,
    )


def _alias_s3_credentials(env_vars: dict[str, str]) -> None:
    endpoint_in_yaml = env_vars.get("AWS_ENDPOINT_URL")
    is_marin_endpoint = (
        endpoint_in_yaml is not None and "storage.googleapis.com" in endpoint_in_yaml
    )
    has_marin = (
        "MARIN_HMAC_ACCESS_ID" in env_vars
        and "MARIN_HMAC_SECRET" in env_vars
    )
    has_laion = "LAION_ENDPOINT" in env_vars
    aliased: list[str] = []

    if has_marin and (is_marin_endpoint or not has_laion):
        env_vars.setdefault("AWS_ENDPOINT_URL", "https://storage.googleapis.com")
        env_vars["AWS_ACCESS_KEY_ID"] = env_vars["MARIN_HMAC_ACCESS_ID"]
        env_vars["AWS_SECRET_ACCESS_KEY"] = env_vars["MARIN_HMAC_SECRET"]
        aliased = [
            "AWS_ENDPOINT_URL ← https://storage.googleapis.com",
            "AWS_ACCESS_KEY_ID ← MARIN_HMAC_ACCESS_ID",
            "AWS_SECRET_ACCESS_KEY ← MARIN_HMAC_SECRET",
        ]
    elif has_laion:
        if "AWS_ENDPOINT_URL" not in env_vars:
            env_vars["AWS_ENDPOINT_URL"] = env_vars["LAION_ENDPOINT"]
            aliased.append("AWS_ENDPOINT_URL ← LAION_ENDPOINT")
        if "LAION_ACCESS_KEY" in env_vars:
            env_vars["AWS_ACCESS_KEY_ID"] = env_vars["LAION_ACCESS_KEY"]
            aliased.append("AWS_ACCESS_KEY_ID ← LAION_ACCESS_KEY")
        if "LAION_SECRET_KEY" in env_vars:
            env_vars["AWS_SECRET_ACCESS_KEY"] = env_vars["LAION_SECRET_KEY"]
            aliased.append("AWS_SECRET_ACCESS_KEY ← LAION_SECRET_KEY")

    if aliased:
        print(
            f"[iris] Aliased for runai_streamer S3 against "
            f"{env_vars.get('AWS_ENDPOINT_URL', '<unset>')}: "
            f"{', '.join(aliased)}",
            flush=True,
        )
