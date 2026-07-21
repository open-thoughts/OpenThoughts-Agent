#!/usr/bin/env python3
"""Launch an external-endpoint OpenCode eval on CoreWeave Iris.

The evaluated model is served separately (for example by Marin's vLLM fork),
so this is deliberately a small launch-host wrapper around ``run_eval`` rather
than the self-serving ``eval.cloud.launch_eval_iris`` path.  It mints the
endpoint capability URL *on the launch host* through the supported Iris CLI;
the generic task image intentionally does not include the Iris controller
client.  The minted URL is passed only as a task environment variable and is
never printed or written to disk.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IRIS_BIN = "/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris"
_CAPABILITY_URL_RE = re.compile(r"https://[^\s]+")


def mint_capability_api_base(
    *, iris_bin: str, cluster: str, endpoint_name: str, ttl_hours: float
) -> str:
    """Mint and validate a scoped capability URL without exposing its token."""
    result = subprocess.run(
        [
            iris_bin,
            "--cluster",
            cluster,
            "endpoints",
            "mint",
            endpoint_name,
            "--ttl-hours",
            str(ttl_hours),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Iris endpoint mint failed (exit {result.returncode}): "
            f"{result.stderr[-800:]}"
        )
    urls = _CAPABILITY_URL_RE.findall(result.stdout)
    if len(urls) != 1:
        raise RuntimeError(
            "Iris endpoint mint returned no unambiguous capability URL; refusing to launch."
        )
    # Iris returns a scoped proxy base and directs callers to append the app
    # path. OpenCode speaks the OpenAI-compatible ``/v1`` API.
    return f"{urls[0].rstrip('/')}/v1"


def require_secret(env: dict[str, str], name: str) -> str:
    value = env.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is required; source the supplied secrets environment first."
        )
    return value


def build_submit_command(
    args: argparse.Namespace, env: dict[str, str], api_base: str
) -> list[str]:
    """Build the Iris submission without placing the scoped URL in argv."""
    task_env = {
        "DAYTONA_API_KEY": require_secret(env, "DAYTONA_API_KEY"),
        "HF_TOKEN": require_secret(env, "HF_TOKEN"),
        "OPENAI_API_KEY": require_secret(env, "OPENAI_API_KEY"),
        # Capability-proxy URLs authenticate in their path. Retain the static
        # bearer for a sidecar endpoint, but do not require it for Iris-minted
        # no-auth capability URLs.
        "OPENCODE_DUMMY_KEY": env.get(
            "IRIS_INGRESS_API_KEY", "capability-url-no-auth-header"
        ),
        "EXTERNAL_AGENT_API_BASE": api_base,
    }
    command = [
        args.iris_bin,
        "--cluster",
        args.cluster,
        "job",
        "run",
        "--task-image",
        args.task_image,
        "--enable-extra-resources",
        "--cpu",
        str(args.cpu),
        "--memory",
        args.memory,
        "--disk",
        args.disk,
        "--priority",
        args.priority,
        "--max-retries",
        "0",
        "--no-wait",
        "--job-name",
        args.job_name,
    ]
    for key, value in task_env.items():
        command.extend(["-e", key, value])
    command.extend(
        [
            "--",
            "bash",
            "-lc",
            "set -eu\n"
            'test -n "${EXTERNAL_AGENT_API_BASE:?missing minted endpoint URL}"\n'
            "exec python -m eval.local.run_eval "
            f"--harbor_config {args.harbor_config} "
            f"--datagen_config {args.datagen_config} "
            f"--model {args.model} "
            f"--dataset_path {args.dataset_path} "
            '--agent_kwarg "api_base=${EXTERNAL_AGENT_API_BASE}" '
            f"--n_concurrent {args.n_concurrent} "
            f"--n_attempts {args.n_attempts} "
            f"--upload_hf_repo {args.upload_hf_repo}",
        ]
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--task-image", required=True)
    parser.add_argument(
        "--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS_BIN)
    )
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument("--secrets-env", required=True)
    parser.add_argument(
        "--harbor-config",
        default="hpc/harbor_yaml/eval/configs/eval_opencode_ctx64k.yaml",
    )
    parser.add_argument(
        "--datagen-config", default="hpc/datagen_yaml/grug_external_endpoint.yaml"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--n-concurrent", type=int, default=6)
    parser.add_argument("--n-attempts", type=int, default=3)
    parser.add_argument("--upload-hf-repo", required=True)
    parser.add_argument("--cpu", type=float, default=8)
    parser.add_argument("--memory", default="64GB")
    parser.add_argument("--disk", default="64GB")
    parser.add_argument(
        "--priority", choices=["production", "interactive", "batch"], default="batch"
    )
    args = parser.parse_args()

    secret_path = Path(args.secrets_env).expanduser()
    if not secret_path.is_file():
        raise SystemExit(f"secrets environment file not found: {secret_path}")
    for line in secret_path.read_text().splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    api_base = mint_capability_api_base(
        iris_bin=args.iris_bin,
        cluster=args.cluster,
        endpoint_name=args.endpoint_name,
        ttl_hours=args.ttl_hours,
    )
    submit = build_submit_command(args, dict(os.environ), api_base)
    # Never log ``submit``: it contains task environment values, including the scoped URL.
    result = subprocess.run(submit, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
