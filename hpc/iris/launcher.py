"""IrisLauncher - base class for submitting OT-Agent jobs to a Marin Iris cluster.

This is the Iris analog of ``hpc.cloud_launch_utils.CloudLauncher`` (which
targets SkyPilot). It exists in parallel rather than as a "provider" plugin
because Iris and SkyPilot disagree on key abstractions (workdir bind mount
vs file_mounts, in-cluster scheduling vs bring-up-a-VM, autostop vs job
timeout). Trying to share one interface created leaky bolts; two clean
modules is cheaper to reason about.

Backend-agnostic helpers under ``hpc/`` (e.g. ``arg_groups``,
``harbor_utils``, ``datagen_config_utils``) are reused as-is.

Output handling - GCS only. The workload writes directly to
``--gcs-output-dir/<job-name>/`` (default
``gs://marin-eu-west4/ot-agent/``; override with ``$OT_AGENT_GCS_OUTPUT_ROOT``
or the flag). A local fetch daemon (``hpc.iris_fetch_daemon``, planned)
polls the iris controller and pulls completed jobs into
``~/.ot-agent/runs/<job-name>/``. The previous "rsync from worker
workdir" mode was removed on 2026-05-22: the worker workdir is on
ephemeral tmpfs and iris GCs it at task end, so any laptop-side rsync
loop is fragile by construction; see
``notes/marin/flows/iris-outputs-redesign.md`` for the post-mortem.

Multi-host slices (TPU vm_count > 1) are scaffolded but only validated
on v6e-8. Confirm cross-host JAX init + coscheduling before relying on
larger slices.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hpc.local_paths import PATHS as LOCAL_PATHS, ensure as ensure_local_paths
from hpc.iris_job_registry import register_submission, get_latest_by_job_name
from hpc.iris.accelerator import (
    DEFAULT_IRIS_JOB_API,
    IrisJobApi,
    ResolvedIrisAccelerator,
)
from hpc.iris.bootstrap import wrap_task_command
from hpc.iris.env import (
    apply_iris_runtime_env,
    default_secrets_env,
    load_secrets_env_into_os_environ,
)
from hpc.iris.capability_tokens import (
    infer_harbor_agent,
    persist_token_duration_policy,
    resolve_token_duration_policy,
)
from hpc.iris.outputs import (
    DEFAULT_GCS_OUTPUT_ROOT,
    DEFAULT_LOCAL_OUTPUT_ROOT,
    DEFAULT_S3_OUTPUT_ROOT,
    resolve_output_mode,
    resolve_remote_output_dir,
    resolve_work_output_dir,
    validate_output_args,
)
from hpc.iris.regions import (
    assert_yaml_regions_match_pin,
    discover_region_for_tpu,
    output_bucket_for_region,
    region_from_output_bucket,
)
from hpc.iris.settings import (
    DEFAULT_CLUSTER_CONFIG,
    DEFAULT_DISK,
    DEFAULT_GPU_CLUSTER_CONFIG,
    DEFAULT_GPU_DISK,
    DEFAULT_GPU_TASK_IMAGE,
    DEFAULT_PRIORITY,
    DEFAULT_TASK_IMAGE,
)


class IrisLauncher:
    """Base class for OT-Agent launchers targeting Marin Iris.

    Subclasses override:
      - ``add_task_specific_args(parser)``
      - ``normalize_paths(args)``
      - ``build_task_command(args, remote_output_dir) -> list[str]``
      - ``build_env(args) -> dict[str, str]``  (optional override)
    """

    task_name: str = "ot-iris"
    job_name_prefix: str = "iris"
    default_n_concurrent: int = 16
    default_tpu: str = "v6e-4"

    # Daytona is the only sandbox backend that works without DinD on iris.
    # Users may still pass --harbor_env docker; iris workers don't mount
    # /var/run/docker.sock so the job will fail at runtime — by design.
    default_harbor_env: str = "daytona"
    enforce_capability_token_duration: bool = False

    def __init__(self, repo_root: Path, iris_api: IrisJobApi = DEFAULT_IRIS_JOB_API):
        self.repo_root = Path(repo_root).resolve()
        self.iris_api = iris_api

    def setup_scripts(self, args: argparse.Namespace) -> list[str] | None:
        """Resolve the Iris task setup contract for this launch.

        Baked-venv images carry an image-specific environment that must not be
        replaced by Iris's default ``uv sync``. An explicit empty list is Iris's
        bring-your-own-image sentinel; ``None`` requests its normal setup.
        """
        return [] if getattr(args, "baked_venv", False) else None

    # ------------------------------------------------------------------
    # Argument parsing
    # ------------------------------------------------------------------

    def create_argument_parser(self, description: str = "") -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=description or self.task_name)
        self._add_iris_common_args(parser)
        self.add_task_specific_args(parser)
        return parser

    def _add_iris_common_args(self, parser: argparse.ArgumentParser) -> None:
        g = parser.add_argument_group("iris")
        g.add_argument(
            "--cluster-config",
            "--cluster_config",
            default=None,
            help="Path to the iris cluster YAML (default: marin for TPU, "
            "cw-us-east-02a for GPU, resolved in the marin repo).",
        )
        g.add_argument(
            "--cluster",
            default=None,
            help="Named iris cluster: resolves lib/iris/config/<name>.yaml in "
            "the marin repo (e.g. --cluster cw-rno2a). Convenience over "
            "--cluster-config; --cluster-config wins if both are given.",
        )
        g.add_argument(
            "--task-image",
            "--task_image",
            default=None,
            help="Container image for the task (default: "
            f"{DEFAULT_TASK_IMAGE} for TPU, {DEFAULT_GPU_TASK_IMAGE} for GPU).",
        )
        g.add_argument(
            "--tpu",
            default=None,
            help=f"TPU variant (default: {self.default_tpu} when --gpu is omitted).",
        )
        g.add_argument(
            "--gpu",
            default=None,
            help="GPU variant in Iris format, e.g. H100x8. Mutually "
            "exclusive with --tpu (selects the CoreWeave GPU path).",
        )
        g.add_argument(
            "--replicas",
            type=int,
            default=1,
            help="Replica count passed to iris submit (default 1). "
            "For a multi-host TPU slice iris REQUIRES one task "
            "per VM and auto-scales replicas=1 -> vm_count "
            "(iris adjust_tpu_replicas); those tasks form ONE "
            "JAX device mesh (jax.distributed.initialize) that a "
            "single engine spans — this is required, not "
            "duplication. Use N*vm_count to request N slices. "
            "NOTE: run_tracegen currently runs harbor on EVERY "
            "task; the per-host duplication is the harbor layer, "
            "not the replica count (fix = gate harbor to the "
            "driver rank, see #69).",
        )
        g.add_argument(
            "--cpu",
            type=float,
            default=8.0,
            help="CPU cores for the entrypoint task (default 8).",
        )
        g.add_argument(
            "--memory",
            default="256GB",
            help="Memory for the entrypoint task (default 256GB). "
            "v6e workers have 720GB total, so 256GB covers "
            "HF weight loading for models up to ~120B bf16 "
            "or ~400B AWQ-4-bit with comfortable headroom. "
            "Bump for larger models; drop to 64GB for small "
            "smokes if you want to be polite to the queue.",
        )
        g.add_argument(
            "--disk",
            default=None,
            help=f"Ephemeral disk (default {DEFAULT_DISK} on marin TPU, "
            f"{DEFAULT_GPU_DISK} on CoreWeave GPU). marin's v6e/v5p "
            "workers cap per-VM disk at 100GB; requests above that "
            "queue forever waiting on an autoscaler that can't "
            "provision a larger-disk worker (on TPU, for weights "
            ">100GB use --load-format runai_streamer + gs://-hosted "
            "weights instead of bumping disk). CoreWeave GPU nodes "
            "have large local disk, so the GPU default is higher to "
            "fit big model weight downloads without an ephemeral-"
            "storage eviction.",
        )
        g.add_argument(
            "--priority",
            default=DEFAULT_PRIORITY,
            choices=["production", "interactive", "batch"],
            help="Iris priority band (default interactive).",
        )
        g.add_argument(
            "--max-retries",
            "--max_retries",
            type=int,
            default=0,
            help="Max retries on failure (does NOT cover preemption — iris retries "
            "preemptions automatically up to its own limit).",
        )
        g.add_argument(
            "--timeout",
            type=int,
            default=0,
            help="Job timeout in seconds (0 = no timeout).",
        )
        g.add_argument(
            "--preemptible",
            dest="preemptible",
            action="store_true",
            default=None,
            help="Force scheduling on preemptible workers (overrides iris heuristic).",
        )
        g.add_argument(
            "--no-preemptible",
            dest="preemptible",
            action="store_false",
            help="Force scheduling on non-preemptible workers.",
        )
        g.add_argument(
            "--no-wait",
            dest="no_wait",
            action="store_true",
            default=False,
            help="Submit and detach instead of streaming logs.",
        )
        g.add_argument(
            "--extras",
            action="append",
            default=None,
            help="OpenThoughts-Agent extras to install in the iris worker's "
            "/app/.venv via `uv sync --extra <name>`. Repeatable. "
            "Default: ['datagen-tpu'] for TPU and ['datagen'] for GPU. "
            "Pass --extras '' to install no extras.",
        )

        og = parser.add_argument_group("outputs")
        og.add_argument(
            "--output-mode",
            "--output_mode",
            choices=["auto", "gcs", "s3", "local"],
            default=os.environ.get("OT_AGENT_OUTPUT_MODE", "auto"),
            help="Where workload outputs are written. 'auto' uses GCS for TPU "
            "and pod-local for GPU (Harbor writes trace_jobs to local NVMe, "
            "run_eval registers to Supabase/HF in-pod). 'gcs' writes to "
            "--gcs-output-dir/<job-name> (TPU); 's3' writes durable Harbor "
            "artifacts to --s3-output-dir/<job-name> (CoreWeave R2); 'local' "
            "writes to --local-output-dir/<job-name> (GPU, default).",
        )
        og.add_argument(
            "--gcs-output-dir",
            "--gcs_output_dir",
            default=os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT", DEFAULT_GCS_OUTPUT_ROOT),
            help=f"GCS prefix for workload outputs; workload writes to "
            f"<this>/<job-name>/. Defaults to $OT_AGENT_GCS_OUTPUT_ROOT or "
            f"{DEFAULT_GCS_OUTPUT_ROOT}. The fetch daemon "
            f"(hpc.iris_fetch_daemon) pulls completed jobs from here into "
            f"{LOCAL_PATHS.runs}/<job-name>/.",
        )
        og.add_argument(
            "--s3-output-dir",
            "--s3_output_dir",
            default=os.environ.get("OT_AGENT_S3_OUTPUT_ROOT", DEFAULT_S3_OUTPUT_ROOT),
            help="S3-compatible prefix for durable GPU outputs, e.g. "
            "s3://marin-us-east-02a/tmp/ttl=7d/ot-agent/evals/<user>. Defaults "
            "to $OT_AGENT_S3_OUTPUT_ROOT. The pod uses the cluster-injected "
            "creds+endpoint (iris-task-env envFrom); launch-host storage creds "
            "are withheld so they cannot clobber them. NOTE: default store moved "
            "R2 (s3://marin-na) -> CW (s3://marin-us-east-02a) 2026-07-05 (marin "
            "c7caecc95a); s3://marin-na is no longer reachable from pods.",
        )
        og.add_argument(
            "--local-output-dir",
            "--local_output_dir",
            default=os.environ.get(
                "OT_AGENT_LOCAL_OUTPUT_ROOT", DEFAULT_LOCAL_OUTPUT_ROOT
            ),
            help="Pod-local runtime scratch root used with --output-mode local/s3. "
            f"Defaults to {DEFAULT_LOCAL_OUTPUT_ROOT}.",
        )

        rg = parser.add_argument_group("resume")
        rg.add_argument(
            "--resume-from",
            "--resume_from",
            dest="resume_from",
            default=None,
            help="Resume harbor state from a previously-submitted iris job "
            "(by job_name; looked up in the local registry "
            f"at {LOCAL_PATHS.state}/iris_jobs.db). The new iris job gets a "
            "fresh timestamped name (for iris-level uniqueness), but the "
            "harbor --job_name and --jobs-dir are routed at the old job's "
            "GCS path so harbor's _maybe_init_existing_job picks up the "
            "existing trial results and only runs the unmatched remaining "
            "trials. No config gating: per the user's direction (2026-05-24), "
            "OT-Agent and harbor already validate compatibility on resume.",
        )

        sg = parser.add_argument_group("secrets")
        # Default to $OT_AGENT_SECRETS_ENV, then ~/Documents/secrets.env if it
        # exists — the canonical location for the user's credentials file.
        # File values override the os.environ passthrough below, so this is
        # the safer-by-default path: without it, a stale shell-cached
        # DAYTONA_API_KEY can ride into the iris worker and cause harbor's
        # auto_snapshot path to fail with "Sandbox not found" even when the
        # snapshot is ACTIVE on the right org.
        _default_secrets = default_secrets_env()
        sg.add_argument(
            "--secrets-env",
            "--secrets_env",
            default=_default_secrets,
            help="Path to a KEY=VALUE env file (~/Documents/secrets.env style). "
            "Every entry is loaded into the iris task's env_vars at submit "
            "time. Pairs with the hardcoded launcher passthrough list "
            "(DAYTONA_API_KEY, OPENAI_API_KEY, etc.) — file values win on "
            "conflict, explicit `-e` iris-CLI flags can't override since we "
            "use IrisClient.submit() directly. Lines starting with '#' and "
            "blank lines are ignored; leading 'export ' is stripped. "
            "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env "
            "if it exists.",
        )
        # NOTE: --dry-run / --dry_run is provided by hpc.arg_groups.add_model_compute_args
        # which subclass launchers call from add_task_specific_args. We don't redeclare
        # it here to avoid argparse conflicts.

    def _resolve_cluster_config_default(
        self, default_config: str = DEFAULT_CLUSTER_CONFIG
    ) -> str:
        """Find the marin repo's cluster config relative to common locations."""
        candidates = [
            Path.home() / "Documents/marin" / default_config,
            Path.home() / "dev/marin" / default_config,
            Path(os.environ.get("MARIN_REPO", "")) / default_config,
            Path(os.environ.get("MARIN_ROOT", "")) / default_config,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return default_config

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        raise NotImplementedError

    @staticmethod
    def load_secrets_env_into_os_environ(secrets_env: Optional[str]) -> int:
        """Read ``secrets_env`` (KEY=VALUE) into ``os.environ`` on the launch host."""
        return load_secrets_env_into_os_environ(secrets_env)

    def normalize_paths(self, args: argparse.Namespace) -> None:
        """Subclass hook: validate/normalize paths and infer defaults."""

    def build_task_command(
        self, args: argparse.Namespace, remote_output_dir: str
    ) -> List[str]:
        """Subclass hook: build the ``python data/...py ...`` invocation."""
        raise NotImplementedError

    def build_env(self, args: argparse.Namespace) -> dict:
        """Subclass hook: env vars to inject into the iris task container.

        HF_TOKEN, WANDB_API_KEY, HF_DATASETS_TRUST_REMOTE_CODE, and
        TOKENIZERS_PARALLELISM are auto-injected by iris workers — no need
        to add them here.
        """
        return {}

    def pre_submit_precache(
        self, args: argparse.Namespace, *, remote_output_dir: str
    ) -> dict:
        """Subclass hook: pre-cache artifacts + return extra env, after region pin.

        Runs in ``run()`` AFTER the region pin (so ``args._pinned_region`` is
        known) and BEFORE ``build_task_command`` (so a subclass may rewrite
        args.model / args.dataset_path onto region-local cloud URIs). The
        returned dict is merged into the task env with ``setdefault`` semantics
        (an explicit build_env / runtime value wins). Default: no-op — the base
        launcher and every non-eval subclass keep byte-identical behavior.
        """
        return {}

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def _derive_job_name(self, args: argparse.Namespace) -> str:
        job_name = getattr(args, "job_name", None)
        if job_name:
            return job_name
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"{self.job_name_prefix}-{ts}"

    def _normalize_accelerator_args(
        self, args: argparse.Namespace
    ) -> ResolvedIrisAccelerator:
        """Resolve the default TPU vs explicit GPU accelerator choice."""
        accelerator = ResolvedIrisAccelerator.from_args(
            args,
            default_tpu=self.default_tpu,
            iris_api=self.iris_api,
        )
        args.tpu = accelerator.tpu
        args.gpu = accelerator.gpu
        self._apply_accelerator_defaults(args, accelerator)
        args._resolved_iris_accelerator = accelerator
        return accelerator

    def _apply_accelerator_defaults(
        self,
        args: argparse.Namespace,
        accelerator: ResolvedIrisAccelerator,
    ) -> None:
        if args.task_image is None:
            args.task_image = (
                DEFAULT_GPU_TASK_IMAGE if accelerator.is_gpu else DEFAULT_TASK_IMAGE
            )
        elif accelerator.is_gpu and args.task_image == DEFAULT_TASK_IMAGE:
            raise SystemExit(
                "--gpu requires a GPU task image. Omit --task-image to use "
                f"{DEFAULT_GPU_TASK_IMAGE}, or pass an explicit GPU-capable image."
            )

        if args.cluster_config is None and getattr(args, "cluster", None):
            args.cluster_config = self._resolve_cluster_config_default(
                f"lib/iris/config/{args.cluster}.yaml"
            )

        if args.cluster_config is None:
            config = (
                DEFAULT_GPU_CLUSTER_CONFIG
                if accelerator.is_gpu
                else DEFAULT_CLUSTER_CONFIG
            )
            args.cluster_config = self._resolve_cluster_config_default(config)

        if args.disk is None:
            args.disk = DEFAULT_GPU_DISK if accelerator.is_gpu else DEFAULT_DISK

    def resolved_accelerator(self, args: argparse.Namespace) -> ResolvedIrisAccelerator:
        accelerator = getattr(args, "_resolved_iris_accelerator", None)
        if accelerator is None:
            accelerator = self._normalize_accelerator_args(args)
        return accelerator

    def run(self, args: argparse.Namespace) -> int:
        accelerator = self._normalize_accelerator_args(args)
        self.normalize_paths(args)
        if self.enforce_capability_token_duration:
            try:
                args.agent = infer_harbor_agent(
                    getattr(args, "agent", None), getattr(args, "harbor_config", None)
                )
                args._capability_token_duration_policy = resolve_token_duration_policy(
                    agent=args.agent, timeout_seconds=args.timeout
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            args.timeout = (
                args._capability_token_duration_policy.effective_timeout_seconds
            )
        accelerator = self.resolved_accelerator(args)

        output_mode = resolve_output_mode(args, accelerator_kind=accelerator.kind)
        args.output_mode = output_mode
        validate_output_args(args, output_mode, accelerator_kind=accelerator.kind)

        # Resolve --resume-from up front: the recorded job's output bucket is
        # authoritative for BOTH the harbor identity (routed below) and the region
        # pin. Forward region discovery is skipped on resume (we don't re-discover a
        # region for an existing job), so we derive the pin from that bucket instead.
        resume_target = getattr(args, "resume_from", None)
        resume_prev = get_latest_by_job_name(resume_target) if resume_target else None
        if resume_target and resume_prev is None:
            raise SystemExit(
                f"--resume-from {resume_target!r}: no record found in "
                f"{LOCAL_PATHS.state}/iris_jobs.db. Available recent jobs:\n"
                "  python -c 'from hpc.iris_job_registry import list_all; "
                "[print(r.job_name) for r in list_all(limit=20)]'"
            )

        # Dynamic region discovery + pin. When the user did NOT explicitly
        # set --gcs-output-dir or $OT_AGENT_GCS_OUTPUT_ROOT (i.e., the value
        # is still the cross-continent default), query iris for which
        # region has v5p/v6e capacity for this TPU spec, override the
        # output bucket to the matching co-located SINGLE-region bucket, and
        # pin the iris job to that region so preempt-retry can't cross-region
        # failover. On --resume-from we don't re-discover (the existing job's
        # bucket is authoritative) but we DO pin to that bucket's region so a
        # resumed single-region job can't be re-placed cross-region either.
        #
        # Discovery is MANDATORY in this branch: if it errors or finds no
        # mapped region, we RAISE rather than silently fall back to the static
        # cross-continent default — that silent fallback previously placed a
        # TPU in one continent while pinning output to a bucket in another,
        # causing cross-region egress. To opt out, pass an explicit
        # --gcs-output-dir (which sets user_set_output and skips this block).
        args._pinned_region = None
        user_set_output = bool(
            os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT")
            or args.gcs_output_dir != DEFAULT_GCS_OUTPUT_ROOT
        )
        if (
            output_mode == "gcs"
            and not user_set_output
            and not resume_target
            and accelerator.is_tpu
        ):
            try:
                region, rows = discover_region_for_tpu(
                    args.cluster_config, accelerator.primary_tpu
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Region discovery failed ({exc}). Refusing to fall back to the static "
                    f"default {DEFAULT_GCS_OUTPUT_ROOT}: iris would place the TPU wherever it has "
                    "capacity while output stays pinned to that fixed bucket, causing cross-region "
                    "egress. Fix region discovery (e.g. iris CLI query-format compatibility) or "
                    "pass an explicit --gcs-output-dir to opt out of the pin."
                ) from exc
            if region is None:
                candidates = [r.get("region") for r in rows if r.get("region")]
                raise RuntimeError(
                    f"No region with a co-located single-region output bucket has "
                    f"--tpu={accelerator.primary_tpu} capacity (regions seen: {candidates or 'none'}). "
                    f"Refusing the static {DEFAULT_GCS_OUTPUT_ROOT} fallback to avoid cross-region "
                    "egress. Wait for capacity in a mapped region, or pass an explicit "
                    "--gcs-output-dir to opt out of the pin."
                )
            bucket = output_bucket_for_region(region)
            args.gcs_output_dir = f"{bucket}/ot-agent"
            args._pinned_region = region
            summary = ", ".join(
                f"{r['region']}: {r.get('unassigned', 0)} warm / "
                f"{r.get('total', 0)} total"
                for r in rows
                if r.get("region")
            )
            print(
                f"[iris] Region pin: --tpu={accelerator.primary_tpu} → {region} "
                f"(bucket {bucket}). Capacity: {summary or 'none reported'}.",
                flush=True,
            )
        elif resume_target and output_mode == "gcs":
            # Resume skips forward discovery, but a resumed job that writes to a
            # SINGLE-region bucket must still be pinned to that bucket's region so a
            # preempt-retry can't be re-placed cross-region (egressing every write).
            # Legacy multi-region buckets (marin-models-{us,eu}) map to None -> stay
            # unpinned: any in-continent placement is read/write-local there.
            resumed_region = region_from_output_bucket(resume_prev.gcs_output_dir)
            if resumed_region:
                args._pinned_region = resumed_region
                print(
                    f"[iris] Resume region pin: {resumed_region} "
                    f"(from recorded bucket {resume_prev.gcs_output_dir}).",
                    flush=True,
                )

        # Fail-fast on cross-region YAML model paths. Iris just pinned the
        # job to a region; if a YAML hardcodes a GCS bucket in the wrong
        # continent (e.g. model_path=gs://marin-models-us/... but pinned
        # region=europe-west4), every weight read would cross continents.
        # We don't auto-rewrite — that masks broken mirrors. We refuse to
        # submit and tell the user exactly which field is wrong.
        if args._pinned_region:
            yaml_attrs = (
                "datagen_config",
                "harbor_config",
                "eval_config",
                "config",
                "harbor_yaml",
            )
            yaml_paths = [
                Path(getattr(args, attr))
                for attr in yaml_attrs
                if getattr(args, attr, None)
            ]
            assert_yaml_regions_match_pin(yaml_paths, args._pinned_region)

        # --resume-from: look up a previously-submitted job and route the
        # new task's harbor command at the old GCS path. The iris-level
        # job_name still gets a fresh timestamp (iris rejects duplicates);
        # the harbor-level identity (jobs_dir + job_name) is preserved so
        # harbor's _maybe_init_existing_job (job.py:203) finds existing
        # trial results and skips them. We stash the old harbor identity
        # on the args namespace so subclass build_task_command can read it.
        if resume_target:
            prev = resume_prev
            ts = time.strftime("%Y%m%d-%H%M%S")
            args.job_name = (
                getattr(args, "job_name", None) or f"{prev.job_name}-resume-{ts}"
            )
            args._harbor_job_name_override = prev.job_name
            args._resume_gcs_output_dir = prev.gcs_output_dir
            print(
                f"[iris] Resume mode: harbor job_name={prev.job_name}  "
                f"gcs={prev.gcs_output_dir}",
                flush=True,
            )

        job_name = self._derive_job_name(args)
        # Persist the derived job_name back onto args so build_task_command bakes it
        # into the worker command as ``--job_name``. On iris this name is IDENTICAL
        # across every preempt-retry of the job (the retry re-runs the same baked
        # command), which is what makes the harbor identity DETERMINISTIC per job:
        #   - harbor jobs_dir/<job_name>/ run dir is the SAME each serve, so harbor's
        #     _maybe_init_existing_job finds the prior serve's trials and CONTINUES
        #     instead of re-running the whole dataset from task 1 in a fresh dir;
        #   - generate_served_model_id(job_name) / default_job_name() no longer fall
        #     back to their per-serve time-based values (the run_tracegen worker only
        #     got a stable name when --job_name was passed — it never was).
        # Resume mode already set args.job_name (+ _harbor_job_name_override) above;
        # _derive_job_name returns that, so this is a no-op there.
        args.job_name = job_name
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

        remote_output_dir = resolve_remote_output_dir(
            args,
            job_name=job_name,
            output_mode=output_mode,
            resume_target=resume_target,
        )
        work_output_dir = resolve_work_output_dir(
            args,
            job_name=job_name,
            output_mode=output_mode,
            remote_output_dir=remote_output_dir,
        )
        args._work_output_dir = work_output_dir
        # Harbor's --jobs-dir ROOT. Harbor writes each job under <jobs-dir>/<job-name>/,
        # and run_eval's in-pod DB/HF upload reads <_jobs_dir_path>/<job-name>/, so the
        # jobs-dir root must be the parent (no job-name) for those to line up.
        #   - s3:   durable object-store root (s3://marin-us-east-02a/.../<user>); harbor appends <job>.
        #   - local: pod-local scratch root; run_eval reads it back in-pod.
        #   - gcs:  historical behavior (remote_output_dir already includes <job>).
        if output_mode == "s3":
            args._harbor_jobs_dir = args.s3_output_dir
        elif output_mode == "local":
            # Distinct from the experiments/work dir (local_output_dir/<job>) so
            # Harbor's job dir (<root>/<job_name>/) holds ONLY trial subdirs +
            # config.json/result.json — run_eval's _extract_job_metadata globs
            # job-dir subdirs for trials, so a colliding experiments `logs/` dir
            # would be mistaken for a trial and crash the in-pod DB upload.
            args._harbor_jobs_dir = f"{args.local_output_dir.rstrip('/')}/harbor_jobs"
        else:
            args._harbor_jobs_dir = remote_output_dir

        # Make sure the local managed tree exists so the daemon (and any
        # downstream consumers) find LOCAL_PATHS.runs/ on first run.
        ensure_local_paths(
            LOCAL_PATHS.home,
            LOCAL_PATHS.state,
            LOCAL_PATHS.runs,
            LOCAL_PATHS.logs,
        )

        # Pre-submit artifact pre-cache (offline staging). Runs after the region
        # pin so a subclass can target the region-local mirror; may rewrite
        # args.model / args.dataset_path and returns extra env (e.g. offline
        # flags). Default no-op -> byte-identical for non-eval launchers.
        precache_env = self.pre_submit_precache(
            args, remote_output_dir=remote_output_dir
        )

        command = self.build_task_command(args, remote_output_dir)
        env_vars = self.build_env(args)
        if self.enforce_capability_token_duration:
            # Keep the same secret-free policy with the remote job as well as
            # the local control-plane manifest. This must never contain the
            # capability URL or its JWT.
            env_vars["OT_AGENT_CAPABILITY_TOKEN_DURATION_POLICY"] = json.dumps(
                args._capability_token_duration_policy.to_dict(), sort_keys=True
            )
        if precache_env:
            for _k, _v in precache_env.items():
                env_vars.setdefault(_k, _v)

        # Default extras: datagen-tpu (TPU) / datagen (GPU); allow override via
        # repeated --extras or --extras '' (single empty) to install nothing extra.
        if args.extras is None:
            extras = accelerator.default_extras
        else:
            extras = [e for e in args.extras if e]

        apply_iris_runtime_env(
            env_vars=env_vars,
            args=args,
            accelerator=accelerator,
            output_mode=output_mode,
            remote_output_dir=remote_output_dir,
            extras=extras,
        )

        local_dest = LOCAL_PATHS.runs / job_name
        token_policy_path = None
        if self.enforce_capability_token_duration:
            token_policy_path = persist_token_duration_policy(
                job_name=job_name, policy=args._capability_token_duration_policy
            )

        print(f"[iris] Job:        /{user}/{job_name}", flush=True)
        print(f"[iris] Cluster:    {args.cluster_config}", flush=True)
        print(f"[iris] Image:      {args.task_image}", flush=True)
        print(f"[iris] {accelerator.label}", flush=True)
        print(f"[iris] Priority:   {args.priority}", flush=True)
        print(f"[iris] Extras:     {extras or '(none)'}", flush=True)
        print(
            f"[iris] Output:     {remote_output_dir}  (mode={output_mode})", flush=True
        )
        if token_policy_path is not None:
            policy = args._capability_token_duration_policy
            print(
                "[iris] Capability-token policy: "
                f"required={policy.token_required} timeout={policy.effective_timeout_seconds}s "
                f"manifest={token_policy_path}",
                flush=True,
            )
        if output_mode == "gcs":
            print(
                f"[iris] Fetch dest: {local_dest}/  (via hpc.iris_fetch_daemon)",
                flush=True,
            )
        else:
            print(
                f"[iris] Work dir:   {work_output_dir}  (pod-local runtime state)",
                flush=True,
            )
            print(
                f"[iris] Jobs dir:   {args._harbor_jobs_dir}  (harbor --jobs-dir)",
                flush=True,
            )
        print(f"[iris] Command:    {shlex.join(command)}", flush=True)

        if args.dry_run:
            print("[iris] --dry-run: not submitting", flush=True)
            return 0

        if accelerator.vm_count > 1:
            print(
                "[iris] NOTE: multi-host TPU slice (vm_count > 1). Validated on v6e-8 "
                "(2026-05-22 smoke #10); larger slices need their own validation pass.",
                file=sys.stderr,
                flush=True,
            )

        # Defer the heavy iris imports so --dry-run / --help stay snappy.
        from iris.client import IrisClient
        from iris.cluster.config import load_config
        from iris.cluster.composer import provider_bundle
        from iris.cluster.local_cluster import LocalCluster
        from iris.cluster.types import EnvironmentSpec, Entrypoint
        from iris.cli.job import build_job_constraints
        from iris.cli.main import client_credentials, resolve_cluster_name
        from iris.rpc import job_pb2

        # Tunnel to the controller via the current pydantic config API
        # (mirrors iris.cli.connect.require_controller_url's SSH-tunnel branch).
        config = load_config(args.cluster_config)
        cluster_name = resolve_cluster_name(
            config, None, Path(args.cluster_config).stem
        )
        credentials = client_credentials(config, cluster_name)
        bundle = provider_bundle(config)
        if config.controller.controller_kind() == "local":
            local_cluster = LocalCluster(config)
            controller_address = local_cluster.start()
        else:
            controller_address = (
                config.controller_address()
                or bundle.controller.discover_controller(config.controller)
            )

        with bundle.controller.tunnel(address=controller_address) as controller_url:
            resources = accelerator.build_resources(
                cpu=args.cpu, memory=args.memory, disk=args.disk
            )
            tpu_variants = list(accelerator.tpu_variants)
            # --replicas defaults to 1; for a multi-host TPU iris's
            # adjust_tpu_replicas (in client.submit) auto-scales 1 -> vm_count
            # because every VM in the slice must run a task to join the one
            # JAX device mesh. So this does NOT reduce the task count on a
            # multi-host slice (that is required for the mesh) — pass N*vm_count
            # to request N slices. The per-host *harbor* duplication is a
            # separate run_tracegen issue (run harbor on the driver rank only).
            # GPU is single-node (vm_count 1); resolve_multinode_defaults is a
            # no-op passthrough there.
            replicas, coscheduling = accelerator.resolve_multinode_defaults(
                args.replicas
            )
            resources_proto = resources.to_proto()
            # Pin the job to the region we discovered at submit time, so
            # preempt-retries land back in the same continent and our
            # output bucket stays local.
            pinned_region = getattr(args, "_pinned_region", None)
            constraints = build_job_constraints(
                resources_proto=resources_proto,
                tpu_variants=tpu_variants,
                replicas=replicas,
                regions=(pinned_region,) if pinned_region else None,
                zone=None,
                preemptible=args.preemptible,
            )

            priority_band = job_pb2.PRIORITY_BAND_UNSPECIFIED
            if args.priority:
                # Map name → enum the same way iris/cli/job.py does.
                _PRIO = {
                    "production": job_pb2.PRIORITY_BAND_PRODUCTION,
                    "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
                    "batch": job_pb2.PRIORITY_BAND_BATCH,
                }
                priority_band = _PRIO.get(args.priority, priority_band)

            client = IrisClient.remote(
                controller_url, workspace=self.repo_root, credentials=credentials
            )

            wrapped = wrap_task_command(
                command,
                extras=extras,
                needs_tpu_runtime_patch=accelerator.needs_tpu_runtime_patch,
            )
            entrypoint = Entrypoint.from_command(*wrapped)

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(
                    env_vars=env_vars,
                    extras=extras,
                    setup_scripts=self.setup_scripts(args),
                ),
                constraints=constraints,
                coscheduling=coscheduling,
                replicas=replicas,
                max_retries_failure=args.max_retries,
                # Iris auto-retries on preemption; leave at default (1000).
                task_image=args.task_image,
                priority_band=priority_band,
                timeout=None
                if args.timeout == 0
                else _seconds_to_duration(args.timeout),
            )
            full_job_id = str(job.job_id)
            print(f"[iris] Submitted: {full_job_id}", flush=True)

            if output_mode == "gcs":
                # Record the job in the local registry so the fetch daemon
                # knows where to pull outputs from on completion. Failures
                # here are non-fatal — the job is already submitted, and the
                # user can re-register later via `python -m hpc.iris_fetch_daemon
                # fetch <job-id>` once that module lands.
                try:
                    register_submission(
                        job_id=full_job_id,
                        job_name=job_name,
                        submitted_at_iso=datetime.now(timezone.utc).isoformat(),
                        gcs_output_dir=remote_output_dir,
                        local_dest=local_dest,
                        cluster_config=str(args.cluster_config),
                    )
                except Exception as e:
                    print(
                        f"[iris] WARN: could not register job locally: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
            else:
                print(
                    f"[iris] {output_mode.upper()} output mode: GPU eval registers to "
                    "Supabase/HF in-pod (run_eval --upload_to_database); skipped GCS "
                    "fetch-daemon registration.",
                    flush=True,
                )

            if args.no_wait:
                return 0

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
            except KeyboardInterrupt:
                print(
                    f"[iris] Terminating job {full_job_id}...",
                    file=sys.stderr,
                    flush=True,
                )
                client.terminate_job(job.job_id)
                exit_code = 130

            print(f"[iris] Job exit: {exit_code}", flush=True)
            return exit_code


# Imported lazily inside .run() to keep CLI startup fast, but tiny enough
# to define here.
def _seconds_to_duration(secs: int):
    # Duration moved from iris.cluster.types to rigging.timing on a
    # marin/iris refactor; iris.client imports from rigging.timing now.
    from rigging.timing import Duration

    return Duration.from_seconds(secs)
