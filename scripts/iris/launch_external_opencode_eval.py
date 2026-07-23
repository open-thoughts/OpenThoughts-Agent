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
import shlex
import subprocess
import sys
import time
from urllib.parse import urlsplit
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IRIS_BIN = "/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris"
_CAPABILITY_URL_RE = re.compile(r"https://[^\s]+")
DEFAULT_PARENT_CLUSTER = "marin"
DEFAULT_PARENT_INGRESS_HOST = "https://iris.oa.dev"
DEFAULT_MIRROR_TIMEOUT_SECONDS = 180.0
DEFAULT_MIRROR_POLL_SECONDS = 3.0
# This wrapper exclusively submits external-endpoint evals to CoreWeave
# ``cw-us-east-02a``.  Match the Iris datagen launcher's durable CW object-store
# convention instead of leaving Harbor's trial directory on the ephemeral pod.
DEFAULT_S3_OUTPUT_ROOT = "s3://marin-us-east-02a/iris"
# OpenAI-compatible installed agents require a non-empty value, but the scoped
# capability URL is the credential. Do not use a sidecar bearer here.
DUMMY_API_KEY = "capability-url-no-auth-header"


def _is_mirrored_parent_endpoint(stdout: str, endpoint_name: str) -> bool:
    """Return whether the parent listed ``endpoint_name`` as a peer endpoint."""
    for line in stdout.splitlines():
        columns = line.split()
        if len(columns) >= 3 and columns[0] == endpoint_name:
            return columns[2] != "local"
    return False


def wait_for_parent_endpoint_mirror(
    *,
    iris_bin: str,
    parent_cluster: str,
    endpoint_name: str,
    timeout_seconds: float = DEFAULT_MIRROR_TIMEOUT_SECONDS,
    poll_seconds: float = DEFAULT_MIRROR_POLL_SECONDS,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> None:
    """Wait for FederationSync before minting a parent-scoped endpoint URL."""
    deadline = monotonic() + timeout_seconds
    while True:
        result = subprocess.run(
            [
                iris_bin,
                "--cluster",
                parent_cluster,
                "endpoints",
                "list",
                endpoint_name,
                "--exact",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                "Iris parent endpoint lookup failed "
                f"(exit {result.returncode}): {result.stderr[-800:]}"
            )
        if _is_mirrored_parent_endpoint(result.stdout, endpoint_name):
            return
        if monotonic() >= deadline:
            raise RuntimeError(
                f"endpoint {endpoint_name!r} did not mirror onto parent "
                f"{parent_cluster!r} within {timeout_seconds:.0f}s; the serving job "
                "must be submitted through the parent with --target-cluster."
            )
        sleep(poll_seconds)


def mint_capability_api_base(
    *,
    iris_bin: str,
    parent_cluster: str,
    parent_ingress_host: str,
    endpoint_name: str,
    ttl_hours: float,
) -> str:
    """Mint a parent-scoped capability URL without exposing its token."""
    result = subprocess.run(
        [
            iris_bin,
            "--cluster",
            parent_cluster,
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
    minted_host = urlsplit(urls[0]).hostname
    expected_host = urlsplit(parent_ingress_host).hostname
    if not expected_host or minted_host != expected_host:
        raise RuntimeError(
            "Iris parent mint returned a URL for an unexpected ingress host; "
            "refusing to submit an unreachable eval."
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


def load_secrets_env(path: Path, environ: dict[str, str]) -> None:
    """Load the selected secret file, replacing any stale inherited values."""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            environ[key] = value.strip().strip('"').strip("'")


def durable_harbor_jobs_dir(*, s3_output_root: str, iris_job_name: str) -> str:
    """Return the durable Harbor jobs root for one external-endpoint eval.

    This deliberately follows the Iris datagen launcher pattern: object-store
    artifacts are isolated by the Iris job name, while ``run_eval`` receives a
    ``--jobs-dir`` root and Harbor creates its own deterministic run directory
    beneath it.  Keeping the ``trace_jobs`` component makes terminal-job
    recovery discoverable with the standard CoreWeave artifact readers.
    """
    root = s3_output_root.rstrip("/")
    if not root.startswith("s3://"):
        raise ValueError("--s3-output-dir must start with s3://")
    if not iris_job_name or "/" in iris_job_name:
        raise ValueError("--job-name must be a non-empty Iris job-name component")
    return f"{root}/{iris_job_name}/trace_jobs"


def build_submit_command(
    args: argparse.Namespace, env: dict[str, str], api_base: str
) -> list[str]:
    """Build the Iris submission without placing the scoped URL in argv."""
    task_env = {
        "DAYTONA_API_KEY": require_secret(env, "DAYTONA_API_KEY"),
        "HF_TOKEN": require_secret(env, "HF_TOKEN"),
        "OPENAI_API_KEY": require_secret(env, "OPENAI_API_KEY"),
        "OPENCODE_DUMMY_KEY": DUMMY_API_KEY,
        "EXTERNAL_AGENT_API_BASE": api_base,
    }
    durable_jobs_dir = durable_harbor_jobs_dir(
        s3_output_root=args.s3_output_dir, iris_job_name=args.job_name
    )
    # ``run_eval`` needs a real local directory for its short-lived endpoint
    # metadata and process logs.  Harbor's long-lived trial artifacts are sent
    # to ``durable_jobs_dir`` instead (the same split as Iris datagen S3 mode).
    work_dir = f"/tmp/ot-agent-runs/{args.job_name}"
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
            f"--experiments_dir {shlex.quote(work_dir)} "
            f"--harbor_extra_arg={shlex.quote(f'--jobs-dir={durable_jobs_dir}')} "
            f"--upload_hf_repo {args.upload_hf_repo}",
        ]
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--parent-cluster", default=DEFAULT_PARENT_CLUSTER)
    parser.add_argument(
        "--parent-ingress-host", default=DEFAULT_PARENT_INGRESS_HOST
    )
    parser.add_argument("--task-image", required=True)
    parser.add_argument(
        "--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS_BIN)
    )
    parser.add_argument("--ttl-hours", type=float, default=24.0)
    parser.add_argument(
        "--mirror-timeout-seconds", type=float, default=DEFAULT_MIRROR_TIMEOUT_SECONDS
    )
    parser.add_argument("--secrets-env", required=True)
    parser.add_argument(
        "--harbor-config",
        default="hpc/harbor_yaml/eval/configs/eval_opencode_ctx64k.yaml",
    )
    parser.add_argument(
        "--datagen-config", default="scratch_grug_external_endpoint.yaml"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-path", required=True)
    # One coordinator is intentionally task-sharded; Iris GPU eval replicas are
    # not.  A high Harbor concurrency feeds the separately served DP=8 model
    # without launching duplicate full-dataset evaluations.
    parser.add_argument("--n-concurrent", type=int, default=256)
    parser.add_argument("--n-attempts", type=int, default=3)
    parser.add_argument(
        "--s3-output-dir",
        default=DEFAULT_S3_OUTPUT_ROOT,
        help=(
            "Durable CoreWeave object-store root for raw Harbor artifacts. "
            "Each launch writes under <root>/<job-name>/trace_jobs/."
        ),
    )
    parser.add_argument("--upload-hf-repo", required=True)
    parser.add_argument("--cpu", type=float, default=32)
    parser.add_argument("--memory", default="128GB")
    parser.add_argument("--disk", default="128GB")
    parser.add_argument(
        "--priority", choices=["production", "interactive", "batch"], default="batch"
    )
    args = parser.parse_args()

    # Reject an invalid durable destination before minting a capability token or
    # contacting the controller.  ``build_submit_command`` repeats this check
    # because it is also called directly by unit tests and other tooling.
    try:
        durable_harbor_jobs_dir(
            s3_output_root=args.s3_output_dir, iris_job_name=args.job_name
        )
    except ValueError as error:
        parser.error(str(error))

    secret_path = Path(args.secrets_env).expanduser()
    if not secret_path.is_file():
        raise SystemExit(f"secrets environment file not found: {secret_path}")
    # File values deliberately replace inherited variables.  A stale exported
    # DAYTONA_API_KEY otherwise produces a full, but completely invalid, eval.
    load_secrets_env(secret_path, os.environ)

    wait_for_parent_endpoint_mirror(
        iris_bin=args.iris_bin,
        parent_cluster=args.parent_cluster,
        endpoint_name=args.endpoint_name,
        timeout_seconds=args.mirror_timeout_seconds,
    )
    api_base = mint_capability_api_base(
        iris_bin=args.iris_bin,
        parent_cluster=args.parent_cluster,
        parent_ingress_host=args.parent_ingress_host,
        endpoint_name=args.endpoint_name,
        ttl_hours=args.ttl_hours,
    )
    submit = build_submit_command(args, dict(os.environ), api_base)
    # Never log ``submit``: it contains task environment values, including the scoped URL.
    result = subprocess.run(submit, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
