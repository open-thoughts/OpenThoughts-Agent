#!/usr/bin/env python3
"""Launch OpenThoughts evals on Marin's Iris TPU cluster.

Iris analog of ``eval/cloud/launch_eval_cloud.py``. Shape mirrors the SkyPilot
launcher exactly so muscle memory carries over — same arg names, same flow.
The differences are all behind the IrisLauncher base.

Output handling: by default outputs are rsync'd back to ``--local-sync-dir``
periodically while the job runs, so downstream eval-analysis tooling sees
local files. Pass ``--output-mode gcs --gcs-output-dir gs://...`` to skip
the rsync layer and have the workload write straight to GCS instead.

Harbor environment: defaults to ``daytona`` (the only sandbox backend that
works on iris workers without DinD). Passing ``--harbor_env docker`` is not
gated — the job will fail at runtime because iris doesn't mount
/var/run/docker.sock into task containers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from hpc.iris_launch_utils import IrisLauncher
from hpc.iris.accelerator import ResolvedIrisAccelerator
from hpc.cloud_launch_utils import repo_relative, infer_harbor_env_from_config
from hpc.arg_groups import (
    add_harbor_args,
    add_harbor_env_arg,
    add_model_compute_args,
    add_hf_upload_args,
    add_database_upload_args,
    add_ingress_literal_args,
)
from hpc.harbor_utils import load_harbor_config
from hpc.datagen_config_utils import parse_datagen_config
from hpc.hf_utils import is_hf_dataset_path
from hpc.launch_utils import PROJECT_ROOT
from eval.presets import load_presets
from database.unified_db.infra_errors import INFRA_ERROR_TYPES

# Default re-fire filter: the non-benign INFRA error types a warm-dir re-fire
# should DELETE-and-re-run (so the harbor auto-resume actually re-runs them
# instead of keeping their errored trial dirs — the campaign no-op bug). This is
# the shared INFRA_ERROR_TYPES set (the single source of truth the Jupiter/
# Leonardo sbatches' `harbor jobs resume --filter-error-type` derives from) UNION
# two POLICY.md non-benign types not (yet) in that set. BENIGN types
# (AgentTimeoutError, ContextLengthExceededError, SummarizationTimeoutError/Error,
# BadRequestError, NonZeroAgentExitCodeError, VerifierRuntimeError) are
# deliberately EXCLUDED — they are valid "not-solved" results, not infra flakes,
# and must NOT be re-run.
DEFAULT_REFIRE_ERROR_TYPES = sorted(
    set(INFRA_ERROR_TYPES) | {"DaytonaValidationError", "VerifierTimeoutError"}
)

# Preset fields with no Iris analog (SLURM orchestrator / vLLM-serve only, or fields
# Iris forces via a different channel). Listed explicitly so the applied/ignored split
# is transparent and a newly added preset field fails loudly here rather than being
# silently dropped. Kept in sync with the listener's `build_config` preset-reading
# surface (eval/unified_eval_listener.py:4347+) so Iris cannot silently drift.
_PRESET_IGNORED_FIELDS = frozenset(
    {
        # --- SLURM / vLLM-serve only (no Iris equivalent) ---
        "slurm_time",
        "slurm_partition",
        "slurm_account",
        "vllm_max_retries",
        "gpu_memory_util",
        "sbatch_script",
        "check_hf_exists",
        "log_suffix",
        "error_threshold",
        "config_yaml",
        "agent_envs",
        "auto_snapshot",
        # --- Iris forces these via a different channel (not preset-driven) ---
        # harbor_config: Iris REQUIRES --harbor_config on the CLI (the listener may pick
        #   it up from cluster-config / size-selection; Iris does not).
        # agent_name: Iris infers the agent from the harbor config's agents[0].name
        #   (the listener has a --agent-name fallback).
        # tp_size: Iris derives --gpus from the TPU chip count (no tensor-parallel concept).
        "harbor_config",
        "agent_name",
        "tp_size",
    }
)


def _cli_has(*flags: str) -> bool:
    """Whether any of the given flags was passed on the command line.

    Used to honor "CLI overrides preset" for args that carry a non-None
    default (e.g. --n_concurrent), where the parsed value alone can't
    distinguish an explicit pass from the default.
    """
    for arg in sys.argv[1:]:
        token = arg.split("=", 1)[0]
        if token in flags:
            return True
    return False


class EvalIrisLauncher(IrisLauncher):
    """Iris launcher for eval/local/run_eval.py."""

    task_name = "ot-eval-iris"
    job_name_prefix = "eval-iris"
    default_n_concurrent = 16

    def required_runtime_extras(
        self,
        args: argparse.Namespace,
        accelerator: ResolvedIrisAccelerator,
    ) -> list[str]:
        """Keep S3 trace persistence in the locked GPU-eval environment."""
        del args
        # CoreWeave agentic evals always pass Harbor an s3:// jobs directory.
        # ``cloud`` owns the lock-pinned s3fs backend required by fsspec/upath.
        return ["cloud"] if accelerator.is_gpu else []

    def add_task_specific_args(self, parser: argparse.ArgumentParser) -> None:
        """Mirror EvalCloudLauncher's args exactly so users don't have to relearn flags."""
        add_harbor_args(parser, config_required=True)

        add_model_compute_args(
            parser,
            model_required=False,  # Can be inferred from datagen_config
            default_n_concurrent=self.default_n_concurrent,
            default_n_attempts=3,
            n_attempts_help="Times to run each task for standard error calculation (default: 3).",
        )

        # Default to daytona; docker passes through and fails organically on iris.
        add_harbor_env_arg(
            parser,
            default=self.default_harbor_env,
            legacy_names=["--eval-env", "--eval_env"],
        )

        parser.add_argument(
            "--datagen_config", help="Optional datagen config to seed defaults."
        )
        parser.add_argument(
            "--datagen-config", dest="datagen_config", help=argparse.SUPPRESS
        )

        parser.add_argument(
            "--preset",
            choices=sorted(load_presets().keys()),
            default=None,
            help="Eval preset from eval/presets/ (shared with the SLURM listener). "
            "Seeds --dataset_path, --n_concurrent, agent parser, and agent_kwargs; "
            "explicit CLI flags always override preset values.",
        )

        parser.add_argument(
            "--dataset", help="Harbor dataset slug (exclusive with --dataset_path)."
        )
        parser.add_argument(
            "--dataset_path", help="Path to tasks directory (exclusive with --dataset)."
        )
        parser.add_argument(
            "--dataset-path", dest="dataset_path", help=argparse.SUPPRESS
        )

        parser.add_argument(
            "--ray_object_store_gb",
            "--ray-object-store-gb",
            type=float,
            default=None,
            help="Ray object store (plasma) size in GB.",
        )

        parser.add_argument(
            "--hf-offline-mode",
            "--hf_offline_mode",
            dest="hf_offline_mode",
            choices=["auto", "strict", "off"],
            default="auto",
            help="Pre-cache the model+dataset to the region-local GCS mirror and run "
            "HF-offline (HF_HUB_OFFLINE=1) when present. "
            "'auto' (default): cache-HIT => offline + region-local runai_streamer "
            "serve; any MISS => today's ONLINE behavior (launch never blocked). "
            "'strict': a MISS fails LOUD at launch with the mirror command. "
            "'off': skip entirely (byte-identical to the pre-precache behavior).",
        )

        # NOTE: --job_name comes from add_harbor_args above.

        # Re-fire errored-trial filter. On a warm-dir re-fire (same --job_name /
        # --resume-from onto an existing run dir), a plain relaunch triggers
        # harbor AUTO-RESUME, which KEEPS infra-errored trial dirs (they have
        # exception_info + no reward -> counted "done") and re-runs only truly-
        # missing trials -> the errored trials NEVER re-run (the campaign no-op).
        # These types are deleted before auto-resume so they DO re-run — the
        # gs://-capable analog of the Jupiter/Leonardo sbatch's
        # `harbor jobs resume --filter-error-type`. Repeatable; defaults to the
        # non-benign infra set. Pass 'none' (or '') to DISABLE (plain auto-resume).
        parser.add_argument(
            "--refire-filter-error-type",
            "--refire_filter_error_type",
            dest="refire_filter_error_types",
            action="append",
            default=None,
            help="Infra exception type to delete-and-re-run on a warm-dir "
            "re-fire (repeatable). Default: the non-benign infra set "
            f"{DEFAULT_REFIRE_ERROR_TYPES}. Pass 'none' to DISABLE pruning "
            "(errored trials will NOT be re-run). A fresh launch with no "
            "existing run dir is unaffected either way.",
        )

        add_hf_upload_args(parser)
        add_database_upload_args(parser)

        # Controller-ingress passthrough to the worker. REQUIRED for installed
        # agents (opencode/openhands/…) whose CLI runs INSIDE the Daytona sandbox
        # and must call the job's own co-located vLLM: the served api_base is a
        # VPC-internal advertise-host IP unreachable from Daytona, so the sandbox
        # reaches it only via the native controller-ingress capability URL
        # (--ingress_mode controller --ingress_host https://iris.oa.dev). Without
        # these the worker defaults to the inert pinggy path and opencode gets an
        # unreachable api_base -> every turn connect-fails -> silent 0. Mirrors
        # the proven datagen path (data/cloud/launch_tracegen_iris.py:113); run_eval.py
        # consumes them via getattr, so no worker change is needed.
        add_ingress_literal_args(parser)

    def _validate_gpu_mode(self, args: argparse.Namespace) -> None:
        accelerator = self.resolved_accelerator(args)
        if not accelerator.is_gpu:
            return

        if accelerator.gpu_variant != "H100" or accelerator.gpu_count != 8:
            raise SystemExit(
                "GPU eval is currently limited to --gpu H100x8 on cw-use02a-h100-8x."
            )

        if (args.replicas or 1) > 1:
            raise SystemExit(
                "GPU eval replicas > 1 need task sharding and are not supported yet. "
                "Use --replicas 1 for the single-node CoreWeave H100x8 path on "
                "cw-use02a-h100-8x."
            )

        # NOTE: --upload_to_database IS supported on the GPU path. Harbor writes
        # trace_jobs to the pod-local jobs dir (or durable R2 with --output-mode
        # s3) and run_eval performs the Supabase + HF registration in-pod before
        # teardown — the same code path the TPU eval uses. (PR #33 blocked this;
        # unblocking it is the fix for defect #1.)

    def _apply_preset(self, args: argparse.Namespace) -> None:
        """Resolve a named preset onto the launcher args.

        CLI flags always win over preset values. Result-affecting fields
        (agent parser, agent_kwargs) become harbor agent-kwargs so the built
        harbor command carries them. SLURM/serve-only fields are ignored with a
        one-line transparency log.
        """
        if not args.preset:
            return

        preset = load_presets()[args.preset]
        applied: dict[str, object] = {}
        ignored: dict[str, object] = {}

        # --dataset_path from datasets[0], only if the user gave no dataset.
        datasets = preset.get("datasets") or []
        if datasets and not args.dataset and not args.dataset_path:
            args.dataset_path = datasets[0]
            applied["dataset_path"] = datasets[0]
            if len(datasets) > 1:
                print(
                    f"[eval-iris] preset {args.preset}: using first of "
                    f"{len(datasets)} datasets; skipped {datasets[1:]}"
                )
        elif datasets:
            ignored["datasets"] = datasets

        # --n_concurrent, only if not explicitly passed on the CLI.
        if "n_concurrent" in preset:
            if not _cli_has("--n_concurrent", "--n-concurrent"):
                args.n_concurrent = preset["n_concurrent"]
                applied["n_concurrent"] = preset["n_concurrent"]
            else:
                ignored["n_concurrent"] = preset["n_concurrent"]

        # --n_attempts (standard-error repeat count), only if not explicitly passed.
        # The listener applies this from preset (build_config n_attempts); Iris must
        # too so a preset that tunes it (e.g. a high-variance benchmark wanting n=5)
        # actually takes effect rather than silently defaulting to 3.
        if "n_attempts" in preset:
            if not _cli_has("--n_attempts", "--n-attempts"):
                args.n_attempts = preset["n_attempts"]
                applied["n_attempts"] = preset["n_attempts"]
            else:
                ignored["n_attempts"] = preset["n_attempts"]

        # Result-affecting agent kwargs → harbor --agent-kwarg, replicating how
        # eval/jupiter/eval_harbor.sbatch maps them (parser=<v>, plus the preset's
        # generic agent_kwargs list — e.g. thinking via the live nested
        # extra_body.chat_template_kwargs.enable_thinking form).
        existing_kwarg_keys = {kw.split("=", 1)[0] for kw in (args.agent_kwarg or [])}

        agent_parser = preset.get("agent_parser")
        if agent_parser:
            if "parser" in existing_kwarg_keys:
                ignored["agent_parser"] = agent_parser
            else:
                args.agent_kwarg.append(f"parser={agent_parser}")
                applied["agent_parser"] = f"parser={agent_parser}"
                existing_kwarg_keys.add("parser")

        # Generic preset agent-kwargs passthrough. Presets carry result-affecting
        # kwargs as full `key=value` strings (e.g. thinking is delivered as the
        # live nested extra_body form the RL rollouts use, which terminus-2's
        # `extra_body` param folds into the request — there is no dedicated
        # enable_thinking flag anymore). A caller-supplied --agent-kwarg with the
        # same key always wins (not clobbered).
        for kw in preset.get("agent_kwargs") or []:
            key = kw.split("=", 1)[0]
            if key in existing_kwarg_keys:
                ignored.setdefault("agent_kwargs", []).append(kw)
            else:
                args.agent_kwarg.append(kw)
                applied.setdefault("agent_kwargs", []).append(kw)
                existing_kwarg_keys.add(key)

        for key, value in preset.items():
            if key in _PRESET_IGNORED_FIELDS:
                ignored[key] = value

        print(
            f"[eval-iris] preset {args.preset}: applied {applied}; "
            f"ignored (SLURM/serve-only or CLI-overridden) {ignored}"
        )

    def normalize_paths(self, args: argparse.Namespace) -> None:
        self._validate_gpu_mode(args)
        self._apply_preset(args)
        if args.dataset and args.dataset_path:
            raise ValueError("Specify either --dataset or --dataset-path (not both).")
        if not args.dataset and not args.dataset_path:
            raise ValueError(
                "Must provide --dataset or --dataset-path for eval workloads."
            )

        # --gpus is the downstream run_eval.py knob for vLLM tensor_parallel_size.
        # Ask the resolved Iris accelerator for the OT-A runtime device count:
        #  - GPU (CoreWeave H100x8): the GPU count.
        #  - TPU: the variant's TRUE chip count via the accelerator's
        #    tpu_topology.chip_count (get_tpu_topology, resolved at
        #    from_args()), NOT the "-N" suffix. TPU naming is a trap — v5p-N
        #    counts CORES not chips (v5p-8 = 4 chips) so a naive suffix parse
        #    overcounts on v5p and sets TP too high -> fit-fail; v6e uses
        #    chips-naming so both agree. The topology is authoritative per family.
        if args.gpus is None:
            args.gpus = self.resolved_accelerator(args).downstream_eval_device_count

        args.harbor_config = repo_relative(args.harbor_config, self.repo_root)
        if args.datagen_config:
            args.datagen_config = repo_relative(args.datagen_config, self.repo_root)
        # Capture the original HF dataset repo id (if any) BEFORE any path
        # rewriting, so pre_submit_precache can mirror it. repo_relative only
        # applies to in-repo local task dirs; an HF-id (org/repo) passes through
        # untouched (the GPU-path HF-dataset guard) and a remote gs://|s3:// URI
        # also passes through (the worker's resolve_dataset_path handles those).
        from hpc.hf_utils import is_hf_dataset_path

        args._orig_dataset_repo = None
        if args.dataset_path and is_hf_dataset_path(args.dataset_path):
            args._orig_dataset_repo = args.dataset_path
        elif (
            args.dataset_path
            and not args.dataset_path.startswith("/")
            and not args.dataset_path.startswith(("gs://", "s3://"))
        ):
            args.dataset_path = repo_relative(args.dataset_path, self.repo_root)

        infer_harbor_env_from_config(args, args.harbor_config, log_prefix="[eval-iris]")

        if not args.agent:
            harbor_cfg = load_harbor_config(args.harbor_config)
            agents = harbor_cfg.get("agents", [])
            if agents and isinstance(agents, list) and len(agents) > 0:
                inferred_agent = agents[0].get("name")
                if inferred_agent:
                    args.agent = inferred_agent
                    print(
                        f"[eval-iris] Inferred --agent={inferred_agent} from harbor config"
                    )

        if not args.model and args.datagen_config:
            try:
                parsed = parse_datagen_config(args.datagen_config)
                if parsed.model:
                    args.model = parsed.model
                    print(
                        f"[eval-iris] Inferred --model={parsed.model} from datagen config"
                    )
            except Exception as e:
                print(
                    f"[eval-iris] Warning: Could not parse datagen config for model: {e}"
                )

        if not args.model:
            raise ValueError(
                "Must provide --model or --datagen_config (to infer model from engine.model)"
            )
        if not args.agent:
            raise ValueError(
                "Must provide --agent or ensure harbor config has agents[0].name"
            )

        # Resolve per-model serve config from model_config/ (single source of
        # truth). Runs AFTER _apply_preset so a chosen preset wins over the
        # model_config; explicit CLI flags win over both. agent_kwargs are merged
        # + forwarded here; max_model_len / limit_mm_per_prompt / extra_args are
        # applied downstream by run_eval.py on the iris worker (same model_config/).
        # tp_size + harbor_config are IGNORED on iris (tp derives from the TPU
        # chip count; --harbor_config is CLI-required).
        from hpc.model_config_apply import apply_to_launcher

        apply_to_launcher(args, log_prefix="[eval-iris]", iris=True)

        # Resolve the re-fire infra-error filter. Unset -> the default non-benign
        # set (ON; harmless on a fresh launch — there is no run dir to prune).
        # An explicit 'none'/'off'/'' DISABLES pruning (plain auto-resume). Any
        # explicit type(s) REPLACE the default. The resolved list is baked into
        # the run_eval command in build_task_command.
        raw_refire = getattr(args, "refire_filter_error_types", None)
        if raw_refire is None:
            args.refire_filter_error_types = list(DEFAULT_REFIRE_ERROR_TYPES)
        elif any(str(v).strip().lower() in ("", "none", "off") for v in raw_refire):
            args.refire_filter_error_types = []
            print(
                "[eval-iris] re-fire errored-trial pruning DISABLED "
                "(--refire-filter-error-type none): a warm-dir re-fire will NOT "
                "re-run infra-errored trials."
            )
        else:
            args.refire_filter_error_types = list(raw_refire)
        if args.refire_filter_error_types:
            print(
                "[eval-iris] re-fire filter (delete + re-run these on an existing "
                f"run dir): {args.refire_filter_error_types}"
            )

        if args.harbor_env == "docker":
            print(
                "[eval-iris] WARNING: --harbor_env=docker on an iris worker requires "
                "/var/run/docker.sock mounted into the task container; iris workers don't "
                "do that by default. Job will likely fail. Use --harbor_env=daytona.",
                file=sys.stderr,
            )

        # Load --secrets-env into os.environ on the launch host (these also
        # reach the worker via the iris submit's --secrets-env).
        loaded = self.load_secrets_env_into_os_environ(
            getattr(args, "secrets_env", None)
        )
        if loaded:
            print(
                f"[eval-iris] Secrets:    loaded {loaded} entries from "
                f"{args.secrets_env} into os.environ for launch-host hooks",
                flush=True,
            )

        # Eval deliberately does NOT pre-build Daytona snapshots and does NOT
        # call hpc.snapshot_manager.ensure_snapshots (the shared-org 60-snapshot
        # cap). The eval harbor configs in hpc/harbor_yaml/eval/ set
        # `environment.force_build: true`, so harbor builds each task's sandbox
        # at runtime on the worker, in the MAIN Daytona org (DAYTONA_API_KEY,
        # forwarded via --secrets-env). The worker's run_eval.py resolves an
        # HF-id `--dataset_path` itself (snapshot_download + parquet convert).
        # This is the eval exception to the snapshot-cap discipline: agent
        # benchmarks legitimately need one env per task (100+), which the cap is
        # wrong for. (Datagen uses force_build: false and DOES pre-build, via
        # data/cloud/launch_tracegen_iris.py.)

    def pre_submit_precache(
        self, args: argparse.Namespace, *, remote_output_dir: str
    ) -> dict:
        """Ensure the model + dataset are region-cached, then wire the offline plan.

        CoreWeave GPU workers intentionally do *not* use the region mirror as
        a RunAI-streamer source.  They have ample node-local NVMe and the
        normal Hugging Face/vLLM path materializes weights there once per pod;
        this is both more reliable than S3 range streaming and matches the
        production GPU datagen path.  TPU workers retain the mirror/offline
        path because their per-VM disk is constrained.

        Runs after the region pin. On a full cache-HIT: serve the model from the
        region-local GCS mirror via runai_streamer (``args._vllm_model_uri``),
        route the dataset read through GCS (``args.dataset_path`` -> gs://), and
        return HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 / the GCS S3 endpoint.
        On a MISS (auto mode) it returns {} -> the launch runs online, unchanged.
        """
        if getattr(args, "gpu", None):
            print(
                "[eval-iris] CoreWeave GPU: serving from the pod-local Hugging Face "
                "cache (RunAI S3 streaming disabled by default).",
                flush=True,
            )
            return {}

        mode = getattr(args, "hf_offline_mode", "auto")
        if mode == "off":
            return {}
        from hpc.iris.precache import precache_for_eval

        dataset_repos = (
            [args._orig_dataset_repo]
            if getattr(args, "_orig_dataset_repo", None)
            else []
        )
        result = precache_for_eval(
            args.model,
            dataset_repos,
            region=getattr(args, "_pinned_region", None),
            mode=mode,
            verbose=True,
        )
        for note in result.notes:
            print(f"[eval-iris] {note}", flush=True)
        if not result.offline_ok:
            return {}
        # Serve the model from the region-local mirror via runai_streamer, but
        # KEEP args.model as the HF id so the worker still resolves model_config/
        # (max_model_len, tool/reasoning parser, ...) from the registry by id.
        args._vllm_model_uri = result.model_serve_uri
        if dataset_repos and result.dataset_uris:
            args.dataset_path = result.dataset_uris[0]
            print(
                f"[eval-iris] dataset -> {args.dataset_path} (offline GCS read)",
                flush=True,
            )
        print(
            f"[eval-iris] model serve URI -> {args._vllm_model_uri} "
            "(runai_streamer); HF_HUB_OFFLINE=1",
            flush=True,
        )
        return result.env

    def build_task_command(
        self, args: argparse.Namespace, remote_output_dir: str
    ) -> List[str]:
        cmd: List[str] = [
            "python",
            "eval/local/run_eval.py",
            "--harbor_config",
            args.harbor_config,
            "--model",
            args.model,
        ]

        # Offline: serve the vLLM model from the region-local mirror URI while
        # --model stays the HF id (for model_config resolution). Only set when
        # pre_submit_precache confirmed the mirror is present.
        vllm_model_uri = getattr(args, "_vllm_model_uri", None)
        if vllm_model_uri:
            cmd.extend(["--vllm_model_uri", vllm_model_uri])

        if args.datagen_config:
            cmd.extend(["--datagen_config", args.datagen_config])
        if args.dataset:
            cmd.extend(["--dataset", args.dataset])
        elif args.dataset_path:
            cmd.extend(["--dataset_path", args.dataset_path])

        # For s3/local output modes the runtime scratch (endpoint.json, logs)
        # lives on pod-local disk; for gcs it is the same remote path.
        work_output_dir = getattr(args, "_work_output_dir", remote_output_dir)

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
                work_output_dir,
            ]
        )

        if args.harbor_env:
            cmd.extend(["--harbor_env", args.harbor_env])

        # Controller-ingress passthrough (run_eval.py worker args, consumed via
        # getattr). Byte-for-byte the datagen pattern
        # (data/cloud/launch_tracegen_iris.py:246-249). Only forwarded when set to
        # a non-default; a bare launch keeps the byte-identical legacy command.
        _ingress_mode = getattr(args, "ingress_mode", None)
        if _ingress_mode and _ingress_mode != "pinggy":
            cmd.extend(["--ingress_mode", _ingress_mode])
        _ingress_host = getattr(args, "ingress_host", None)
        if _ingress_host:
            cmd.extend(["--ingress_host", _ingress_host])

        if args.job_name:
            cmd.extend(["--job_name", args.job_name])
        if args.dry_run:
            cmd.append("--dry_run")

        if args.ray_object_store_gb is not None:
            cmd.extend(["--ray_object_store_gb", str(args.ray_object_store_gb)])

        for kwarg in args.agent_kwarg:
            cmd.extend(["--agent_kwarg", kwarg])
        # Auto-inject --jobs-dir so harbor writes outputs under the durable
        # jobs root (harbor appends the job-name below it). With harbor's UPath
        # patch (penfever/otagent-latest @ dc41d295a4) remote schemes route all
        # per-job/per-trial writes through fsspec (GCS/S3) instead of local
        # /app/trace_jobs/; for --output-mode local it is a fast pod-local dir
        # that run_eval's in-pod --upload_to_database then reads. User
        # --harbor_extra_arg entries follow below so an explicit
        # --harbor_extra_arg=--jobs-dir=... wins.
        harbor_jobs_dir = getattr(args, "_harbor_jobs_dir", remote_output_dir)
        cmd.append(f"--harbor_extra_arg=--jobs-dir={harbor_jobs_dir}")
        for extra in args.harbor_extra_arg:
            # Use the `=` form so argparse on the worker side accepts values
            # that start with `-` (e.g. --harbor_extra_arg=--n-tasks). The
            # space form `--harbor_extra_arg --n-tasks` trips argparse's
            # "looks like an option" heuristic and gets rejected with
            # "argument --harbor_extra_arg: expected one argument".
            cmd.append(f"--harbor_extra_arg={extra}")

        # Bake the resolved re-fire filter into the in-pod run_eval command. On a
        # warm-dir re-fire, run_eval prunes trials with these error types before
        # auto-resume so they re-run. Empty (disabled) -> nothing baked ->
        # run_eval's default (no pruning) -> byte-identical to the old behavior.
        for _et in getattr(args, "refire_filter_error_types", None) or []:
            cmd.extend(["--refire_filter_error_type", _et])

        if args.upload_to_database:
            cmd.append("--upload_to_database")
        if args.upload_username:
            cmd.extend(["--upload_username", args.upload_username])
        if args.upload_error_mode:
            cmd.extend(["--upload_error_mode", args.upload_error_mode])
        if args.upload_hf_repo:
            cmd.extend(["--upload_hf_repo", args.upload_hf_repo])
        if args.upload_hf_token:
            cmd.extend(["--upload_hf_token", args.upload_hf_token])
        if args.upload_hf_private:
            cmd.append("--upload_hf_private")
        if args.upload_hf_episodes:
            cmd.extend(["--upload_hf_episodes", args.upload_hf_episodes])
        if args.upload_forced_update:
            cmd.append("--upload_forced_update")

        return cmd


def main() -> None:
    launcher = EvalIrisLauncher(PROJECT_ROOT)
    parser = launcher.create_argument_parser(
        description="Launch eval/local/run_eval.py on a Marin Iris TPU worker."
    )
    args = parser.parse_args()
    sys.exit(launcher.run(args))


if __name__ == "__main__":
    main()
