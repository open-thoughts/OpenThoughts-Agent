#!/usr/bin/env python3
"""Launch OpenThoughts trace generation on Marin's Iris TPU cluster.

Iris analog of ``data/cloud/launch_tracegen_cloud.py``. See
``eval/cloud/launch_eval_iris.py`` and ``hpc/iris_launch_utils.py`` for
the shared design notes (rsync vs gcs outputs, daytona-default Harbor env,
docker-not-gated, multi-host TPU is scaffolded but untested).
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import List

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

import yaml  # noqa: E402

from hpc.iris_launch_utils import IrisLauncher  # noqa: E402
from hpc.iris.outputs import IrisOutputPaths  # noqa: E402
from hpc.cloud_launch_utils import repo_relative, infer_harbor_env_from_config  # noqa: E402
from hpc.arg_groups import (  # noqa: E402
    add_harbor_args,
    add_harbor_env_arg,
    add_ingress_literal_args,
    add_model_compute_args,
    add_hf_upload_args,
    add_tasks_input_arg,
)
from hpc.launch_utils import PROJECT_ROOT  # noqa: E402


def _env_vars_from_datagen_yaml(path: Path) -> dict:
    """Lift a top-level ``env_vars:`` block from a datagen YAML, if present.

    Lets a per-config knob override launcher-side env defaults (e.g.
    `MODEL_IMPL_TYPE=auto` for models that prefer the JAX-native flax_nnx
    path). Returns ``{}`` if no env_vars block is declared.
    """
    if not path or not path.exists():
        return {}
    try:
        with path.open() as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        print(
            f"[tracegen-iris] WARNING: failed to parse {path} for env_vars: {e}",
            file=sys.stderr,
        )
        return {}
    env = cfg.get("env_vars") or {}
    if not isinstance(env, dict):
        return {}
    return {str(k): str(v) for k, v in env.items()}


class TracegenIrisLauncher(IrisLauncher):
    """Iris launcher for data/local/run_tracegen.py."""

    task_name = "ot-tracegen-iris"
    job_name_prefix = "tracegen-iris"
    default_n_concurrent = 64
    # Trace generation runs for days on preemptible capacity. Keep Iris's
    # cumulative job-level failure guard from aborting after one hard failure;
    # per-task --max-retries remains the bound for deterministic failures, and
    # Iris separately permits 1000 infrastructure preemptions per task.
    default_max_task_failures = 10
    enforce_capability_token_duration = True

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        add_harbor_args(parser, config_required=True)

        add_model_compute_args(
            parser,
            model_required=False,
            default_n_concurrent=self.default_n_concurrent,
            default_n_attempts=1,
            n_attempts_help="Times to run each task for repeated trials (default: 1).",
        )

        add_harbor_env_arg(
            parser,
            default=self.default_harbor_env,
            legacy_names=["--trace-env", "--trace_env"],
        )

        parser.add_argument(
            "--datagen_config",
            required=True,
            help="Datagen config with vLLM settings (required).",
        )
        parser.add_argument(
            "--datagen-config", dest="datagen_config", help=argparse.SUPPRESS
        )

        add_tasks_input_arg(parser, required=True)

        parser.add_argument(
            "--skip-daytona-snapshot-prebuild",
            action="store_true",
            help="Skip launch-host task extraction and Daytona snapshot prebuild. "
            "The worker still receives harbor_env=daytona and handles its task set on-cluster.",
        )

        # Health-check passthrough for the run_tracegen.py worker. These are
        # registered by hpc.arg_groups.add_ray_vllm_args on the local runner
        # but NOT auto-registered on the launcher (add_ray_vllm_args adds
        # local-only flags like --host/--ray_port that the launcher must
        # not touch). Re-declare just the health knobs here so users can
        # extend the startup-wait timeout for slow XLA compiles (e.g.
        # large MoE models on cold-cache iris workers).
        parser.add_argument(
            "--health_max_attempts",
            "--health-max-attempts",
            type=int,
            default=None,
            help="Override the worker's run_tracegen.py "
            "--health_max_attempts (default 100 on the "
            "worker; 100 × --health_retry_delay seconds "
            "is the hard cap on cold-start XLA compile + "
            "engine warmup). Bump for large MoE models.",
        )
        parser.add_argument(
            "--health_retry_delay",
            "--health-retry-delay",
            type=int,
            default=None,
            help="Override the worker's run_tracegen.py "
            "--health_retry_delay (default 30s).",
        )

        # NOTE: --job_name comes from add_harbor_args above.

        add_hf_upload_args(parser)

        # Literal-token capture + controller-ingress passthrough to the worker.
        add_ingress_literal_args(parser)

        parser.add_argument(
            "--baked-venv",
            "--baked_venv",
            dest="baked_venv",
            action="store_true",
            help="Run the worker from the image's baked /opt/openthoughts/.venv instead of "
            "letting Iris run its default setup or rebuild /app/.venv from uv.lock. Required for "
            "images whose venv carries deps NOT in the locked pin (e.g. the gpu-glm52 "
            "image's fork vLLM for GLM-5.2, which a `uv sync --reinstall` would clobber "
            "back to the pinned version).",
        )

    def normalize_paths(self, args: argparse.Namespace) -> None:
        # --gpus drives vLLM tensor_parallel_size. On the GPU path it is the whole-node
        # GPU count (H100x8 -> 8, so a TP=8 serve sees all 8 GPUs); on TPU it is the
        # slice's chip/core count. Derive from the already-resolved accelerator (run()
        # resolves it before normalize_paths). Previously this only parsed --tpu and fell
        # back to 1 on GPU, which starved Ray of GPUs and broke multi-GPU serves.
        if args.gpus is None:
            accelerator = self.resolved_accelerator(args)
            if accelerator.is_gpu:
                args.gpus = accelerator.gpu_count
            else:
                args.gpus = int(args.tpu.rsplit("-", 1)[-1])

        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        # --tasks_input_path is overloaded: local FS path | HF dataset id | harbor slug.
        # Only run it through repo_relative when it's actually a local FS path that
        # exists under the repo. HF ids like ``mlfoundations-dev/foo`` would otherwise
        # be resolved against CWD and either rewritten into a bogus repo-relative
        # path or raise ValueError if CWD escapes the repo.
        if not args.tasks_input_path.startswith("/"):
            if (self.repo_root / args.tasks_input_path).exists():
                args.tasks_input_path = repo_relative(
                    args.tasks_input_path, self.repo_root
                )

        infer_harbor_env_from_config(
            args, args.harbor_config, log_prefix="[tracegen-iris]"
        )

        # Resolve per-model serve config from model_config/ (single source of
        # truth). agent_kwargs are merged + forwarded here; max_model_len /
        # limit_mm_per_prompt / extra_args are applied downstream by
        # run_tracegen.py on the iris worker (same model_config/). tp_size is
        # IGNORED on iris (tp derives from the TPU chip count). When --model isn't
        # passed (model inferred from the datagen config on the worker), resolution
        # is deferred to the worker.
        from hpc.model_config_apply import apply_to_launcher

        apply_to_launcher(args, log_prefix="[tracegen-iris]", iris=True)

        if args.harbor_env == "docker":
            print(
                "[tracegen-iris] WARNING: --harbor_env=docker on an iris worker requires "
                "/var/run/docker.sock mounted into the task container; iris workers don't "
                "do that by default. Job will likely fail. Use --harbor_env=daytona.",
                file=sys.stderr,
            )

        # Load --secrets-env into os.environ on the launch host BEFORE
        # the snapshot pre-build hook (which reads os.environ via
        # get_daytona_api_key_override). The same file is also parsed into
        # the iris worker's env_vars later in run() so the worker sees the
        # same values.
        loaded = self.load_secrets_env_into_os_environ(
            getattr(args, "secrets_env", None)
        )
        if loaded:
            print(
                f"[tracegen-iris] Secrets:    loaded {loaded} entries from "
                f"{args.secrets_env} into os.environ for launch-host hooks",
                flush=True,
            )

        # Pre-build Daytona snapshots on the launch host so harbor's
        # `auto_snapshot=true` short-circuits to an existing ACTIVE snapshot
        # at trial time instead of falling through to the declarative-build
        # path — which produces "Sandbox not found. Please build the
        # environment first." on every trial. The SLURM datagen/eval launchers
        # do this via hpc/{datagen,eval}_launch_utils.py + hpc/launch.py's
        # convert_parquet_to_tasks step; the iris launcher was missing both
        # (the silent prereq that took Q4-v8 out tonight).
        # No-op when harbor_env != "daytona" or DAYTONA_API_KEY is unset.
        if (
            args.harbor_env == "daytona"
            and args.tasks_input_path
            and not args.skip_daytona_snapshot_prebuild
        ):
            from hpc.hf_utils import resolve_dataset_path
            from hpc.launch_utils import (
                convert_parquet_to_tasks,
                get_daytona_api_key_override,
                maybe_prebuild_daytona_snapshots,
            )
            from hpc.snapshot_manager import OrgConfig

            api_key = get_daytona_api_key_override(vars(args))
            orgs = [OrgConfig(name="cli", api_key=api_key)] if api_key else []
            if not orgs:
                print(
                    "[tracegen-iris] WARNING: harbor_env=daytona but DAYTONA_API_KEY unset on "
                    "launch host; skipping snapshot pre-build. Expect 'Sandbox not found' "
                    "at trial time unless snapshots are already warm in Daytona.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                from hpc.hf_utils import is_raw_tasks_directory

                resolved_tasks = resolve_dataset_path(
                    args.tasks_input_path, verbose=True
                )
                # HF datasets ship a single tasks.parquet; explode into task
                # directories so snapshot_manager can hash the environments.
                # hpc/launch.py:232 + hpc/datagen_launch_utils.py do this.
                if not is_raw_tasks_directory(resolved_tasks):
                    resolved_tasks = convert_parquet_to_tasks(
                        snapshot_dir=resolved_tasks,
                        dataset_identifier=args.tasks_input_path,
                    )
                print(
                    f"[tracegen-iris] Pre-building Daytona snapshots from {resolved_tasks} ...",
                    flush=True,
                )
                maybe_prebuild_daytona_snapshots(
                    [resolved_tasks],
                    harbor_env=args.harbor_env,
                    orgs=orgs,
                )
        elif args.skip_daytona_snapshot_prebuild:
            print(
                "[tracegen-iris] Skipping launch-host Daytona snapshot pre-build; "
                "the worker will prepare tasks on-cluster.",
                flush=True,
            )

    def build_task_command(
        self, args: argparse.Namespace, output_paths: IrisOutputPaths
    ) -> List[str]:
        cmd: List[str] = [
            "python",
            "data/local/run_tracegen.py",
            "--harbor_config",
            args.harbor_config,
            "--datagen_config",
            args.datagen_config,
            "--tasks_input_path",
            args.tasks_input_path,
        ]

        if args.model:
            cmd.extend(["--model", args.model])

        cmd.extend(
            [
                "--agent",
                args.agent,
                "--n_concurrent",
                str(args.n_concurrent),
                "--n_attempts",
                str(args.n_attempts),
                "--gpus",
                str(args.gpus),
                "--experiments_dir",
                output_paths.work_output_dir,
            ]
        )

        # Health-check passthrough (run_tracegen.py worker-side args).
        if args.health_max_attempts is not None:
            cmd.extend(["--health_max_attempts", str(args.health_max_attempts)])
        if args.health_retry_delay is not None:
            cmd.extend(["--health_retry_delay", str(args.health_retry_delay)])

        if args.harbor_env:
            cmd.extend(["--harbor_env", args.harbor_env])

        # Literal-capture + controller-ingress passthrough (run_tracegen.py worker args).
        if getattr(args, "record_literal", False):
            cmd.append("--record_literal")
        if getattr(args, "ingress_mode", "pinggy") and args.ingress_mode != "pinggy":
            cmd.extend(["--ingress_mode", args.ingress_mode])
        if getattr(args, "ingress_host", None):
            cmd.extend(["--ingress_host", args.ingress_host])

        # When --resume-from is active, the iris-level job_name is fresh
        # (timestamped for iris uniqueness) but the harbor identity must be
        # the OLD job's name so harbor's _maybe_init_existing_job picks up
        # the existing config.json / trials. IrisLauncher.run() stashes the
        # old name on args._harbor_job_name_override.
        harbor_job_name = (
            getattr(args, "_harbor_job_name_override", None) or args.job_name
        )
        if harbor_job_name:
            cmd.extend(["--job_name", harbor_job_name])
        if args.dry_run:
            cmd.append("--dry_run")

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent_kwarg", kwarg])
        # Keep Harbor's durable job root separate from the pod-local runtime
        # directory. Harbor appends the stable job name below this root.
        cmd.append(f"--harbor_extra_arg=--jobs-dir={output_paths.harbor_jobs_dir}")
        for extra in args.harbor_extra_arg:
            # `=` form for argparse accept of `-`-prefixed values; see
            # the same comment in eval/cloud/launch_eval_iris.py.
            cmd.append(f"--harbor_extra_arg={extra}")

        if args.upload_hf_repo:
            cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload_hf_token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload_hf_private")

        if getattr(args, "baked_venv", False):
            # IrisLauncher.setup_scripts() also passes the explicit no-setup
            # sentinel. This shell entrypoint selects the baked venv, while
            # PYTHONPATH=/app keeps the synced worker code authoritative over
            # its editable install.
            bash = (
                "cd /app && export PYTHONPATH=/app:${PYTHONPATH:-} && "
                "export PATH=/opt/openthoughts/.venv/bin:$PATH && exec "
                + shlex.join(cmd)
            )
            return ["bash", "-c", bash]

        return cmd

    def build_env(self, args: argparse.Namespace) -> dict:
        """Lift a per-config ``env_vars:`` block from the datagen YAML.

        Returned entries are merged BEFORE the launcher's setdefaults in
        IrisLauncher.run(), so they win — e.g. setting
        ``env_vars: {MODEL_IMPL_TYPE: auto}`` in a gemma4 YAML overrides
        the launcher-wide default of `MODEL_IMPL_TYPE=vllm` (which is
        the right default only for AWQ workloads). Other env defaults
        the launcher provides (UV_LINK_MODE, OT_AGENT_SKIP_SFT_SYNC,
        etc) remain in effect via setdefault.
        """
        env = super().build_env(args)
        # args.datagen_config has been normalized to a repo-relative string in
        # normalize_paths(); resolve it against repo_root for file open().
        if args.datagen_config:
            yaml_path = self.repo_root / args.datagen_config
            env.update(_env_vars_from_datagen_yaml(yaml_path))
        return env


def main() -> None:
    launcher = TracegenIrisLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch data/local/run_tracegen.py on a Marin Iris TPU worker."
    )
    args = parser.parse_args()
    sys.exit(launcher.run(args))


if __name__ == "__main__":
    main()
