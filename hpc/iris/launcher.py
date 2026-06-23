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
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hpc.local_paths import PATHS as LOCAL_PATHS, ensure as ensure_local_paths
from hpc.iris_job_registry import register_submission, get_latest_by_job_name
from hpc.iris.bootstrap import wrap_task_command
from hpc.iris.env import (
    apply_iris_runtime_env,
    default_secrets_env,
    load_secrets_env_into_os_environ,
)
from hpc.iris.outputs import (
    DEFAULT_GCS_OUTPUT_ROOT,
    resolve_remote_output_dir,
    validate_output_args,
)
from hpc.iris.regions import (
    assert_yaml_regions_match_pin,
    discover_region_for_tpu,
    gcs_bucket_for_region,
    parse_tpu_vm_count,
)
from hpc.iris.settings import DEFAULT_CLUSTER_CONFIG, DEFAULT_PRIORITY, DEFAULT_TASK_IMAGE


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

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

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
        g.add_argument("--cluster-config", "--cluster_config",
                       default=self._resolve_cluster_config_default(),
                       help="Path to the iris cluster YAML (default: marin via lib/iris/config/marin.yaml in the marin repo).")
        g.add_argument("--task-image", "--task_image",
                       default=DEFAULT_TASK_IMAGE,
                       help=f"Container image for the task (default: {DEFAULT_TASK_IMAGE}).")
        g.add_argument("--tpu", default=self.default_tpu,
                       help=f"TPU variant (default: {self.default_tpu}).")
        g.add_argument("--replicas", type=int, default=1,
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
                            "driver rank, see #69).")
        g.add_argument("--cpu", type=float, default=8.0,
                       help="CPU cores for the entrypoint task (default 8).")
        g.add_argument("--memory", default="256GB",
                       help="Memory for the entrypoint task (default 256GB). "
                            "v6e workers have 720GB total, so 256GB covers "
                            "HF weight loading for models up to ~120B bf16 "
                            "or ~400B AWQ-4-bit with comfortable headroom. "
                            "Bump for larger models; drop to 64GB for small "
                            "smokes if you want to be polite to the queue.")
        g.add_argument("--disk", default="100GB",
                       help="Ephemeral disk (default 100GB). marin's v6e/v5p "
                            "workers cap per-VM disk at 100GB; requests above "
                            "that queue forever waiting on the autoscaler which "
                            "can't provision a larger-disk worker. For models "
                            "whose weights exceed 100GB, use --load-format "
                            "runai_streamer + gs://-hosted weights instead of "
                            "bumping disk.")
        g.add_argument("--priority", default=DEFAULT_PRIORITY,
                       choices=["production", "interactive", "batch"],
                       help="Iris priority band (default interactive).")
        g.add_argument("--max-retries", "--max_retries", type=int, default=0,
                       help="Max retries on failure (does NOT cover preemption — iris retries "
                            "preemptions automatically up to its own limit).")
        g.add_argument("--timeout", type=int, default=0,
                       help="Job timeout in seconds (0 = no timeout).")
        g.add_argument("--preemptible", dest="preemptible", action="store_true", default=None,
                       help="Force scheduling on preemptible workers (overrides iris heuristic).")
        g.add_argument("--no-preemptible", dest="preemptible", action="store_false",
                       help="Force scheduling on non-preemptible workers.")
        g.add_argument("--no-wait", dest="no_wait", action="store_true", default=False,
                       help="Submit and detach instead of streaming logs.")
        g.add_argument("--extras", action="append", default=None,
                       help="OpenThoughts-Agent extras to install in the iris worker's "
                            "/app/.venv via `uv sync --extra <name>`. Repeatable. "
                            "Default: ['datagen-tpu'] (matches the :tpu task image's "
                            "intended dep set). Pass --extras '' to install no extras.")

        og = parser.add_argument_group("outputs")
        og.add_argument("--gcs-output-dir", "--gcs_output_dir",
                        default=os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT", DEFAULT_GCS_OUTPUT_ROOT),
                        help=f"GCS prefix for workload outputs; workload writes to "
                             f"<this>/<job-name>/. Defaults to $OT_AGENT_GCS_OUTPUT_ROOT or "
                             f"{DEFAULT_GCS_OUTPUT_ROOT}. The fetch daemon "
                             f"(hpc.iris_fetch_daemon) pulls completed jobs from here into "
                             f"{LOCAL_PATHS.runs}/<job-name>/.")

        rg = parser.add_argument_group("resume")
        rg.add_argument("--resume-from", "--resume_from", dest="resume_from", default=None,
                        help="Resume harbor state from a previously-submitted iris job "
                             "(by job_name; looked up in the local registry "
                             f"at {LOCAL_PATHS.state}/iris_jobs.db). The new iris job gets a "
                             "fresh timestamped name (for iris-level uniqueness), but the "
                             "harbor --job_name and --jobs-dir are routed at the old job's "
                             "GCS path so harbor's _maybe_init_existing_job picks up the "
                             "existing trial results and only runs the unmatched remaining "
                             "trials. No config gating: per the user's direction (2026-05-24), "
                             "OT-Agent and harbor already validate compatibility on resume.")

        sg = parser.add_argument_group("secrets")
        # Default to $OT_AGENT_SECRETS_ENV, then ~/Documents/secrets.env if it
        # exists — the canonical location for the user's credentials file.
        # File values override the os.environ passthrough below, so this is
        # the safer-by-default path: without it, a stale shell-cached
        # DAYTONA_API_KEY can ride into the iris worker and cause harbor's
        # auto_snapshot path to fail with "Sandbox not found" even when the
        # snapshot is ACTIVE on the right org.
        _default_secrets = default_secrets_env()
        sg.add_argument("--secrets-env", "--secrets_env", default=_default_secrets,
                        help="Path to a KEY=VALUE env file (~/Documents/secrets.env style). "
                             "Every entry is loaded into the iris task's env_vars at submit "
                             "time. Pairs with the hardcoded launcher passthrough list "
                             "(DAYTONA_API_KEY, OPENAI_API_KEY, etc.) — file values win on "
                             "conflict, explicit `-e` iris-CLI flags can't override since we "
                             "use IrisClient.submit() directly. Lines starting with '#' and "
                             "blank lines are ignored; leading 'export ' is stripped. "
                             "Defaults to $OT_AGENT_SECRETS_ENV, else ~/Documents/secrets.env "
                             "if it exists.")
        # NOTE: --dry-run / --dry_run is provided by hpc.arg_groups.add_model_compute_args
        # which subclass launchers call from add_task_specific_args. We don't redeclare
        # it here to avoid argparse conflicts.

    def _resolve_cluster_config_default(self) -> str:
        """Find the marin repo's cluster config relative to common locations."""
        candidates = [
            Path.home() / "Documents/marin" / DEFAULT_CLUSTER_CONFIG,
            Path.home() / "dev/marin" / DEFAULT_CLUSTER_CONFIG,
            Path(os.environ.get("MARIN_REPO", "")) / DEFAULT_CLUSTER_CONFIG,
            Path(os.environ.get("MARIN_ROOT", "")) / DEFAULT_CLUSTER_CONFIG,
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        return DEFAULT_CLUSTER_CONFIG

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

    def build_task_command(self, args: argparse.Namespace, remote_output_dir: str) -> List[str]:
        """Subclass hook: build the ``python data/...py ...`` invocation."""
        raise NotImplementedError

    def build_env(self, args: argparse.Namespace) -> dict:
        """Subclass hook: env vars to inject into the iris task container.

        HF_TOKEN, WANDB_API_KEY, HF_DATASETS_TRUST_REMOTE_CODE, and
        TOKENIZERS_PARALLELISM are auto-injected by iris workers — no need
        to add them here.
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

    def run(self, args: argparse.Namespace) -> int:
        self.normalize_paths(args)

        validate_output_args(args)

        # Dynamic region discovery + pin. When the user did NOT explicitly
        # set --gcs-output-dir or $OT_AGENT_GCS_OUTPUT_ROOT (i.e., the value
        # is still the cross-continent default), query iris for which
        # region has v5p/v6e capacity for this TPU spec, override the
        # output bucket to the matching multi-region bucket, and pin the
        # iris job to that region so preempt-retry can't cross-continent
        # failover. Skipped on --resume-from (the existing job's bucket
        # is authoritative) and when no TPU is requested.
        args._pinned_region = None
        user_set_output = bool(
            os.environ.get("OT_AGENT_GCS_OUTPUT_ROOT")
            or args.gcs_output_dir != DEFAULT_GCS_OUTPUT_ROOT
        )
        if (
            not user_set_output
            and not getattr(args, "resume_from", None)
            and getattr(args, "tpu", None)
        ):
            try:
                region, rows = discover_region_for_tpu(args.cluster_config, args.tpu)
            except Exception as exc:
                print(
                    f"[iris] Region discovery failed ({exc}); falling back to "
                    f"static default {DEFAULT_GCS_OUTPUT_ROOT}. Cross-region "
                    "egress possible if iris schedules outside that bucket's "
                    "region.",
                    file=sys.stderr, flush=True,
                )
            else:
                if region is None:
                    print(
                        f"[iris] No workers visible for --tpu={args.tpu}; falling "
                        f"back to static default {DEFAULT_GCS_OUTPUT_ROOT}.",
                        file=sys.stderr, flush=True,
                    )
                else:
                    bucket = gcs_bucket_for_region(region)
                    args.gcs_output_dir = f"{bucket}/ot-agent"
                    args._pinned_region = region
                    summary = ", ".join(
                        f"{r['region']}: {r.get('unassigned', 0)} warm / "
                        f"{r.get('total', 0)} total"
                        for r in rows if r.get('region')
                    )
                    print(
                        f"[iris] Region pin: --tpu={args.tpu} → {region} "
                        f"(bucket {bucket}). Capacity: {summary or 'none reported'}.",
                        flush=True,
                    )

        # Fail-fast on cross-region YAML model paths. Iris just pinned the
        # job to a region; if a YAML hardcodes a GCS bucket in the wrong
        # continent (e.g. model_path=gs://marin-models-us/... but pinned
        # region=europe-west4), every weight read would cross continents.
        # We don't auto-rewrite — that masks broken mirrors. We refuse to
        # submit and tell the user exactly which field is wrong.
        if args._pinned_region:
            yaml_attrs = ("datagen_config", "harbor_config", "eval_config",
                          "config", "harbor_yaml")
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
        resume_target = getattr(args, "resume_from", None)
        if resume_target:
            prev = get_latest_by_job_name(resume_target)
            if prev is None:
                raise SystemExit(
                    f"--resume-from {resume_target!r}: no record found in "
                    f"{LOCAL_PATHS.state}/iris_jobs.db. Available recent jobs:\n"
                    "  python -c 'from hpc.iris_job_registry import list_all; "
                    "[print(r.job_name) for r in list_all(limit=20)]'"
                )
            ts = time.strftime("%Y%m%d-%H%M%S")
            args.job_name = getattr(args, "job_name", None) or f"{prev.job_name}-resume-{ts}"
            args._harbor_job_name_override = prev.job_name
            args._resume_gcs_output_dir = prev.gcs_output_dir
            print(
                f"[iris] Resume mode: harbor job_name={prev.job_name}  "
                f"gcs={prev.gcs_output_dir}",
                flush=True,
            )

        job_name = self._derive_job_name(args)
        user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"

        remote_output_dir = resolve_remote_output_dir(
            args,
            job_name=job_name,
            resume_target=resume_target,
        )

        # Make sure the local managed tree exists so the daemon (and any
        # downstream consumers) find LOCAL_PATHS.runs/ on first run.
        ensure_local_paths(
            LOCAL_PATHS.home, LOCAL_PATHS.state, LOCAL_PATHS.runs, LOCAL_PATHS.logs,
        )

        command = self.build_task_command(args, remote_output_dir)
        env_vars = self.build_env(args)

        # Default extras = ["datagen-tpu"]; allow override via repeated --extras
        # or --extras '' (single empty) to install nothing extra.
        if args.extras is None:
            extras = ["datagen-tpu"]
        else:
            extras = [e for e in args.extras if e]

        apply_iris_runtime_env(
            env_vars=env_vars,
            args=args,
            remote_output_dir=remote_output_dir,
            extras=extras,
        )

        vm_count = parse_tpu_vm_count(args.tpu)

        local_dest = LOCAL_PATHS.runs / job_name

        print(f"[iris] Job:        /{user}/{job_name}", flush=True)
        print(f"[iris] Cluster:    {args.cluster_config}", flush=True)
        print(f"[iris] Image:      {args.task_image}", flush=True)
        print(f"[iris] TPU:        {args.tpu}  (vm_count={vm_count})", flush=True)
        print(f"[iris] Priority:   {args.priority}", flush=True)
        print(f"[iris] Extras:     {extras or '(none)'}", flush=True)
        print(f"[iris] Output:     {remote_output_dir}", flush=True)
        print(f"[iris] Fetch dest: {local_dest}/  (via hpc.iris_fetch_daemon)", flush=True)
        print(f"[iris] Command:    {shlex.join(command)}", flush=True)

        if args.dry_run:
            print("[iris] --dry-run: not submitting", flush=True)
            return 0

        if vm_count > 1:
            print(
                "[iris] NOTE: multi-host TPU slice (vm_count > 1). Validated on v6e-8 "
                "(2026-05-22 smoke #10); larger slices need their own validation pass.",
                file=sys.stderr, flush=True,
            )

        # Defer the heavy iris imports so --dry-run / --help stay snappy.
        from iris.client import IrisClient
        from iris.cluster.config import IrisConfig
        from iris.cluster.types import EnvironmentSpec, Entrypoint
        from iris.cli.job import build_resources, build_job_constraints, resolve_multinode_defaults, build_tpu_alternatives
        from iris.rpc import job_pb2

        # Tunnel to the controller via the documented IrisConfig pattern
        # (see lib/iris/.../cluster/config.py:IrisConfig docstring).
        iris_config = IrisConfig.load(args.cluster_config)
        bundle = iris_config.provider_bundle()
        controller_proto = iris_config.proto.controller
        if controller_proto.WhichOneof("controller") == "local":
            from iris.cluster.providers.local.cluster import LocalCluster
            local_cluster = LocalCluster(iris_config.proto)
            controller_address = local_cluster.start()
        else:
            controller_address = (
                iris_config.controller_address()
                or bundle.controller.discover_controller(controller_proto)
            )

        with bundle.controller.tunnel(controller_address) as controller_url:
            resources = build_resources(args.tpu, None, cpu=args.cpu, memory=args.memory, disk=args.disk)
            tpu_variants = build_tpu_alternatives(args.tpu)
            primary_tpu = tpu_variants[0] if tpu_variants else None
            # --replicas defaults to 1; for a multi-host TPU iris's
            # adjust_tpu_replicas (in client.submit) auto-scales 1 -> vm_count
            # because every VM in the slice must run a task to join the one
            # JAX device mesh. So this does NOT reduce the task count on a
            # multi-host slice (that is required for the mesh) — pass N*vm_count
            # to request N slices. The per-host *harbor* duplication is a
            # separate run_tracegen issue (run harbor on the driver rank only).
            replicas, coscheduling = resolve_multinode_defaults(
                primary_tpu, None, args.replicas
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

            client = IrisClient.remote(controller_url, workspace=self.repo_root)

            wrapped = wrap_task_command(command, extras=extras)
            entrypoint = Entrypoint.from_command(*wrapped)

            job = client.submit(
                entrypoint=entrypoint,
                name=job_name,
                resources=resources,
                environment=EnvironmentSpec(env_vars=env_vars, extras=extras),
                constraints=constraints,
                coscheduling=coscheduling,
                replicas=replicas,
                max_retries_failure=args.max_retries,
                # Iris auto-retries on preemption; leave at default (1000).
                task_image=args.task_image,
                priority_band=priority_band,
                timeout=None if args.timeout == 0 else _seconds_to_duration(args.timeout),
            )
            full_job_id = str(job.job_id)
            print(f"[iris] Submitted: {full_job_id}", flush=True)

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
                print(f"[iris] WARN: could not register job locally: {e}", file=sys.stderr, flush=True)

            if args.no_wait:
                return 0

            try:
                status = job.wait(stream_logs=True, timeout=float("inf"))
                exit_code = 0 if status.state == job_pb2.JOB_STATE_SUCCEEDED else 1
            except KeyboardInterrupt:
                print(f"[iris] Terminating job {full_job_id}...", file=sys.stderr, flush=True)
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
