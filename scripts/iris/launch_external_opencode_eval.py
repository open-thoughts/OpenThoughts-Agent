#!/usr/bin/env python3
"""Launch a federated external-endpoint OpenCode eval on CoreWeave Iris.

This is the one-command Grug profile: it submits the serving job to the Marin
parent, waits for its peer endpoint to mirror back to the parent, mints the
parent-scoped capability URL, and then submits the durable Harbor eval.  The
capability URL is passed only as a task environment variable; it is never
printed or persisted by this launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from hpc.iris.capability_tokens import (
    persist_token_duration_policy,
    resolve_token_duration_policy,
)


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
DEFAULT_KUBECONFIG = "/Users/benjaminfeuer/.kube/coreweave-iris-gpu"
DEFAULT_MARIN_REPO = "/Users/benjaminfeuer/Documents/marin"
DEFAULT_SERVE_MODEL = "laion/grug-67b-a2b-sft-s3-agentic-step1903"
DEFAULT_DATASET_PATH = "DCAgent/dev_set_v2"
DEFAULT_HF_TRACE_REPO = "laion/grug-agentic-eval-v2-traces"
# This image carries the Harbor fixes validated for the canonical Grug eval. A
# mutable tag could silently revert those runtime fixes between launch and retry.
DEFAULT_TASK_IMAGE = (
    "ghcr.io/open-thoughts/openthoughts-agent@sha256:"
    "a172ed5a83df94775684bc66ea6afa7b530162c21d872b3bd185ec663eb9f1a2"
)
DEFAULT_SERVE_READY_TIMEOUT_SECONDS = 1800.0
# The serve watches only model-inference requests (not readiness/health probes)
# and frees the H100x8 slice after this idle interval.  The Marin-side flag is
# intentionally distinct from the seven-day capability-token and wall-clock TTLs.
DEFAULT_SERVE_IDLE_TIMEOUT_HOURS = 1.0
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
                f"(exit {result.returncode}); inspect client diagnostics locally."
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
            f"Iris endpoint mint failed (exit {result.returncode}); "
            "the launcher will not expose controller diagnostics that may contain "
            "a capability URL."
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


def default_job_name(now: datetime | None = None) -> str:
    """Return a unique, descriptive Iris job name for the default Grug profile."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    return f"grug-agentic-eval-v2-65k-{timestamp}"


def default_secrets_env(environ: dict[str, str]) -> str:
    """Match the Iris launcher's safe local secrets-file convention."""
    return (
        environ.get("DC_AGENT_SECRET_ENV")
        or environ.get("OT_AGENT_SECRETS_ENV")
        or str(Path.home() / "Documents/secrets.env")
    )


def build_serve_command(args: argparse.Namespace) -> list[str]:
    """Build the parent-delegated Marin serve submission for this eval."""
    command = [
        "uv",
        "run",
        "--project",
        "lib/marin",
        "marin-serve",
        "iris",
        args.serve_model,
        "--cluster",
        args.parent_cluster,
        "--target-cluster",
        args.cluster,
        "--gpu",
        "H100x8",
        "--name",
        args.serve_name,
        "--endpoint-name",
        args.endpoint_name,
        "--max-model-len",
        str(args.serve_max_model_len),
        "--max-num-batched-tokens",
        str(args.serve_max_num_batched_tokens),
        "--tensor-parallel-size",
        str(args.serve_tensor_parallel_size),
        "--dtype",
        "bfloat16",
        "--vllm-source",
        "marin-fork",
        "--proxy-timeout",
        "600",
        "--cpu",
        "48",
        "--memory",
        "1024g",
        "--disk",
        "512g",
        "--wait-timeout",
        str(args.serve_ready_timeout_seconds),
        "--idle-timeout-hours",
        str(args.serve_idle_timeout_hours),
        "--no-wait",
        f"--vllm-arg=--data-parallel-size={args.serve_data_parallel_size}",
        f"--vllm-arg=--max-num-seqs={args.serve_max_num_seqs}",
        "--vllm-arg=--enable-expert-parallel",
        "--vllm-arg=--enable-auto-tool-choice",
        "--vllm-arg=--tool-call-parser=hermes",
        f"--vllm-arg=--served-model-name={args.serve_model}",
    ]
    for value in args.serve_vllm_arg:
        command.append(f"--vllm-arg={value}")
    return command


def build_worker_command(
    args: argparse.Namespace, durable_jobs_dir: str, work_dir: str
) -> str:
    """Return a shell-safe eval command while preserving runtime URL expansion."""
    before_api_base = [
        "python",
        "-m",
        "eval.local.run_eval",
        "--harbor_config",
        args.harbor_config,
        "--datagen_config",
        args.datagen_config,
        "--model",
        args.model,
        "--dataset_path",
        args.dataset_path,
    ]
    after_api_base = [
        "--n_concurrent",
        str(args.n_concurrent),
        "--n_attempts",
        str(args.n_attempts),
        "--job_name",
        args.job_name,
        "--experiments_dir",
        work_dir,
        f"--harbor_extra_arg=--jobs-dir={durable_jobs_dir}",
        "--upload_hf_repo",
        args.upload_hf_repo,
    ]
    return (
        "set -eu\n"
        'test -n "${EXTERNAL_AGENT_API_BASE:?missing minted endpoint URL}"\n'
        f"exec {shlex.join(before_api_base)} "
        '--agent_kwarg "api_base=${EXTERNAL_AGENT_API_BASE}" '
        f"{shlex.join(after_api_base)}"
    )


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
    task_env["OT_AGENT_CAPABILITY_TOKEN_DURATION_POLICY"] = (
        args.capability_token_duration_policy_json
    )
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
        "--timeout",
        str(args.timeout),
        "--no-wait",
        "--job-name",
        args.job_name,
    ]
    for key, value in task_env.items():
        command.extend(["-e", key, value])
    command.extend(
        ["--", "bash", "-lc", build_worker_command(args, durable_jobs_dir, work_dir)]
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", default=default_job_name())
    parser.add_argument(
        "--existing-endpoint",
        help="Use an already-running federated endpoint instead of submitting a new serve.",
    )
    parser.add_argument(
        "--serve-name", help="Serving job name (default: <job-name>-serve)."
    )
    parser.add_argument(
        "--endpoint-name", help="Endpoint name (default: /serve/<serve-name>)."
    )
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--parent-cluster", default=DEFAULT_PARENT_CLUSTER)
    parser.add_argument("--parent-ingress-host", default=DEFAULT_PARENT_INGRESS_HOST)
    parser.add_argument("--kubeconfig", default=DEFAULT_KUBECONFIG)
    parser.add_argument("--marin-repo", default=DEFAULT_MARIN_REPO)
    parser.add_argument("--serve-model", default=DEFAULT_SERVE_MODEL)
    parser.add_argument("--serve-max-model-len", type=int, default=65536)
    parser.add_argument("--serve-max-num-batched-tokens", type=int, default=7168)
    parser.add_argument("--serve-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--serve-data-parallel-size", type=int, default=8)
    parser.add_argument("--serve-max-num-seqs", type=int, default=32)
    parser.add_argument(
        "--serve-ready-timeout-seconds",
        type=float,
        default=DEFAULT_SERVE_READY_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--serve-idle-timeout-hours",
        type=float,
        default=DEFAULT_SERVE_IDLE_TIMEOUT_HOURS,
        help="Stop a newly created serve after this long without a model-inference request.",
    )
    parser.add_argument(
        "--serve-vllm-arg",
        action="append",
        default=[],
        help="Extra raw vLLM flag for Marin serve; repeatable.",
    )
    parser.add_argument("--task-image", default=DEFAULT_TASK_IMAGE)
    parser.add_argument(
        "--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS_BIN)
    )
    parser.add_argument(
        "--ttl-hours",
        type=float,
        default=None,
        help="Requested scoped-token TTL. Defaults to the controller maximum.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="Iris eval timeout in seconds (0 derives the minted-token lifetime).",
    )
    parser.add_argument(
        "--mirror-timeout-seconds", type=float, default=DEFAULT_MIRROR_TIMEOUT_SECONDS
    )
    parser.add_argument("--secrets-env", default=default_secrets_env(dict(os.environ)))
    parser.add_argument(
        "--harbor-config",
        default="hpc/harbor_yaml/eval/configs/eval_opencode_ctx64k.yaml",
    )
    parser.add_argument(
        "--datagen-config", default="hpc/datagen_yaml/grug_external_endpoint.yaml"
    )
    parser.add_argument("--model", help="Eval model id (default: vllm/<serve-model>).")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH)
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
    parser.add_argument("--upload-hf-repo", default=DEFAULT_HF_TRACE_REPO)
    parser.add_argument("--cpu", type=float, default=32)
    parser.add_argument("--memory", default="128GB")
    parser.add_argument("--disk", default="128GB")
    parser.add_argument(
        "--priority", choices=["production", "interactive", "batch"], default="batch"
    )
    args = parser.parse_args()

    args.serve_name = args.serve_name or f"{args.job_name}-serve"
    args.endpoint_name = (
        args.existing_endpoint or args.endpoint_name or f"/serve/{args.serve_name}"
    )
    args.model = args.model or f"vllm/{args.serve_model}"

    requested_ttl_seconds = (
        None if args.ttl_hours is None else int(args.ttl_hours * 3600)
    )
    try:
        token_policy = resolve_token_duration_policy(
            agent="opencode",
            timeout_seconds=args.timeout,
            requested_token_ttl_seconds=requested_ttl_seconds,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.timeout = token_policy.effective_timeout_seconds
    args.ttl_hours = token_policy.effective_token_ttl_seconds / 3600.0
    args.capability_token_duration_policy_json = json.dumps(
        token_policy.to_dict(), sort_keys=True
    )
    policy_path = persist_token_duration_policy(
        job_name=args.job_name, policy=token_policy
    )
    print(
        "[external-eval] capability-token policy: "
        f"timeout={args.timeout}s manifest={policy_path}",
        flush=True,
    )

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

    submit_environment = dict(os.environ)
    submit_environment["KUBECONFIG"] = args.kubeconfig
    if not args.existing_endpoint:
        marin_repo = Path(args.marin_repo).expanduser()
        if not marin_repo.is_dir():
            parser.error(f"Marin checkout not found: {marin_repo}")
        print(
            f"[external-eval] submitting federated serve {args.serve_name}", flush=True
        )
        serve_result = subprocess.run(
            build_serve_command(args),
            cwd=marin_repo,
            env=submit_environment,
            check=False,
        )
        if serve_result.returncode:
            return serve_result.returncode

    print(
        f"[external-eval] waiting for parent-mirrored ready endpoint {args.endpoint_name}",
        flush=True,
    )
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
    print(f"[external-eval] submitting durable eval {args.job_name}", flush=True)
    result = subprocess.run(submit, cwd=REPO_ROOT, env=submit_environment, check=False)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
