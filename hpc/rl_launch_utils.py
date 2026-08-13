"""
RL Training Launch Utilities

Helpers for computing distributed RL training parameters and deriving paths
for SkyRL-based reinforcement learning jobs.

This module provides:
- RLJobConfig: Configuration dataclass for RL training jobs
- launch_rl_job(): Main entry point for submitting RL jobs
- RLJobRunner: Class for executing RL jobs from sbatch
- resolve_rl_train_data(): Extracts HF datasets to local task directories
- Helper functions for computing inference engines, tensor parallelism, etc.
"""

from __future__ import annotations

import json
import os
import shlex
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hpc.hf_utils import is_hf_dataset_path
from hpc.launch_utils import get_daytona_api_key_override
from hpc.rl_config_utils import get_skyrl_command_preview
from hpc.rl_paths import (
    LATEST_CHECKPOINT_FILE,
    RLLaunchIntent,
    RLPathManager,
    RLRunPaths,
    hydra_override_values,
)


# Default Apptainer bind mounts for the RL container runtime mode.
# These GPFS roots on Jupiter are NOT auto-bound by Apptainer (only $HOME,
# $PWD, /tmp, /proc, /sys, /dev are) and cover code, SIF, tasks, checkpoints,
# HF cache, and experiments. See scope_rl_via_apptainer_launcher.md §3.
DEFAULT_RL_CONTAINER_BINDS: List[str] = ["/e/scratch", "/e/data1"]


# Candidate directory NAMES for the RL training repo, in probe order. The repo
# was historically always cloned to a dir literally named "SkyRL", but its
# contents may be replaced by MarinSkyRL while keeping (or changing) the dir
# name. The Python import name (skyrl_train) is unaffected by the dir name and
# must NOT be touched — only filesystem PATH resolution. "SkyRL" stays first so
# any existing SkyRL-named deployment resolves byte-identically.
RL_REPO_DIR_CANDIDATES: List[str] = ["SkyRL", "MarinSkyRL"]


def resolve_rl_repo_dir(parent: str) -> str:
    """Resolve the RL training repo directory under ``parent``.

    The Jupiter deployment is about to have its RL repo *contents* replaced
    with MarinSkyRL while keeping the directory NAME ``SkyRL`` (to satisfy
    hardcoded paths); future setups may instead name the dir ``MarinSkyRL``.
    This helper lets the launcher consume either, while staying byte-identical
    for existing ``SkyRL``-only deployments.

    Precedence:
      (a) explicit override env var ``RL_REPO_DIR`` (full path) if set — honored
          verbatim, regardless of whether it exists yet;
      (b) probe ``parent`` for ``SkyRL`` then ``MarinSkyRL`` (``RL_REPO_DIR_CANDIDATES``
          order) and return the first that exists as a directory;
      (c) fall back to ``<parent>/SkyRL`` (the historical literal) for
          byte-identical back-compat when neither candidate exists.

    Args:
        parent: Parent directory that contains (or will contain) the repo dir.

    Returns:
        Absolute-or-relative path to the resolved repo directory (joins
        ``parent`` with the resolved dir name; does not normalize ``parent``).
    """
    override = os.environ.get("RL_REPO_DIR")
    if override:
        return override
    for name in RL_REPO_DIR_CANDIDATES:
        candidate = os.path.join(parent, name)
        if os.path.isdir(candidate):
            return candidate
    # Back-compat fallback: the historical literal, even if it doesn't exist
    # yet (e.g. setup_rl_env.sh is about to clone into it).
    return os.path.join(parent, RL_REPO_DIR_CANDIDATES[0])


def _resolve_skyrl_home() -> Optional[str]:
    """Resolve the on-disk RL repo dir for SKYRL_HOME-based path construction.

    Honors ``SKYRL_HOME`` (precedence (a)) when it points at an existing dir.
    If ``SKYRL_HOME`` is set but missing, or unset, falls back to probing its
    parent (or CWD's parent) for the ``{SkyRL, MarinSkyRL}`` candidate dirs via
    :func:`resolve_rl_repo_dir`. Returns None only when nothing is set and no
    candidate exists (callers then skip adding the path, as before).

    This keeps existing SkyRL-named deployments byte-identical: ``SKYRL_HOME``
    is set by the dotenv to ``<parent>/SkyRL`` and exists, so it is returned
    unchanged.
    """
    skyrl_home = os.environ.get("SKYRL_HOME")
    if skyrl_home:
        if os.path.isdir(skyrl_home):
            return skyrl_home
        # SKYRL_HOME set but absent under that exact name — probe its parent for
        # the alternate dir name (e.g. dotenv hardcoded .../SkyRL but only
        # .../MarinSkyRL exists on this box).
        parent = os.path.dirname(skyrl_home.rstrip("/"))
        if parent:
            resolved = resolve_rl_repo_dir(parent)
            if os.path.isdir(resolved):
                return resolved
        # Nothing better found; preserve prior behavior (use SKYRL_HOME as-is).
        return skyrl_home
    return None


def build_apptainer_prefix(
    sif: str,
    binds: Optional[List[str]] = None,
    pythonpath: Optional[str] = None,
) -> List[str]:
    """Build the ``apptainer exec --nv`` command prefix for the RL runtime.

    Mirrors the SFT-MCA precedent (``hpc/sbatch_sft_mca/vista_train_mca.sbatch``
    line 139: ``srun singularity exec --nv --bind ... <sif> ...``).

    The returned list is meant to be *prepended* to a command (Ray ``ray start``,
    the ray.init() wait script, or the SkyRL driver), so that the command runs
    inside the SIF using the container's own Python install. A ``PYTHONPATH``
    is prepended via ``--env`` so the live host SkyRL/harbor source overrides
    the install baked into the container (live-source bind, §3 of the design doc).

    Intentional omissions per the design doc:
    - NO ``--cleanenv``: host env (UV_USE_IO_URING, NCCL_*, etc.) must survive.
    - NO ``--net``/``--network``: the container shares the host network so Ray's
      ``--node-ip-address``/``--address`` and VLLM_HOST_IP work unchanged.

    Args:
        sif: Absolute path to the Apptainer/Singularity SIF image.
        binds: Bind-mount sources (default: DEFAULT_RL_CONTAINER_BINDS). Each is
            passed as ``--bind <src>`` (DST==SRC).
        pythonpath: Value for an ``--env PYTHONPATH=...`` flag, prepended so the
            bind-mounted host source wins over the in-SIF install. If None, no
            PYTHONPATH override is injected (apptainer passes host env by default).

    Returns:
        List of command tokens ending with the SIF path, e.g.::

            ["apptainer", "exec", "--nv",
             "--bind", "/e/scratch", "--bind", "/e/data1",
             "--env", "PYTHONPATH=...", "<sif>"]
    """
    if binds is None:
        binds = DEFAULT_RL_CONTAINER_BINDS
    prefix: List[str] = ["apptainer", "exec", "--nv"]
    # Writable/RO overlays composed over the read-only SIF rootfs. Sourced from
    # the RL_CONTAINER_OVERLAYS env var (colon-separated paths, each mounted
    # read-only via `--overlay <path>:ro`). Used for the 80B Qwen3-Next run,
    # which stacks the P1 vLLM-HTTP overlay + the Stage-8 fla_tilelang overlay
    # over skyrl_megatron_vllm.sif. Each overlay is an independent ext3 image;
    # they compose by stacking `--overlay` flags. Empty/unset => no overlays
    # (byte-identical to the prior non-overlay path).
    overlays_env = os.environ.get("RL_CONTAINER_OVERLAYS", "")
    for ov in (p for p in overlays_env.split(":") if p):
        prefix.extend(["--overlay", f"{ov}:ro"])
    for b in binds:
        prefix.extend(["--bind", b])
    if pythonpath:
        prefix.extend(["--env", f"PYTHONPATH={pythonpath}"])
    # Force WANDB_MODE into the container explicitly. Although apptainer passes
    # host env by default (no --cleanenv), the SkyRL wandb.init() inside the SIF
    # was NOT seeing the host WANDB_MODE=offline (dotenv-set) on the 80B run
    # (job 599463) and tried an online init that timed out after 90s against
    # wandb.ai through proxychains (CommError: "Run initialization has timed
    # out"). Pinning it on the apptainer --env list guarantees offline mode
    # inside the container, removing the network dependency entirely; the .out
    # log carries all train metrics and the offline run can be `wandb sync`'d
    # later. Defaults to offline but honors an explicit host override.
    wandb_mode = os.environ.get("WANDB_MODE", "offline")
    prefix.extend(["--env", f"WANDB_MODE={wandb_mode}"])
    # Raise Ray's raylet-startup grace window. On busy 6-node GH200 allocations
    # the raylet (head AND worker) intermittently fails to finish registering
    # with the local GCS inside Ray's default 30s (`RAY_raylet_start_wait_time_s`),
    # so `ray start` aborts with "The current node timed out during startup ...
    # the GCS has become overloaded" → the driver then loops on
    # "Failed to connect to GCS ... within 5 seconds" until the 600s wait window
    # expires (job 930367, #232 cp2). This is slow-to-form, NOT unreachable — the
    # driver runs ON the head node and still can't reach its own 6379. Ray's own
    # error message recommends raising this config. apptainer passes it via --env
    # so it reaches the in-SIF `ray start`; host `env KEY=val` prefixes do NOT
    # cross the container boundary (and the proxychains path drops ray_env_vars
    # entirely), making this --env injection the only point that reaches every
    # ray invocation (head, worker, wait/poll scripts) uniformly. Honors an
    # explicit host override.
    raylet_wait = os.environ.get("RAY_raylet_start_wait_time_s", "120")
    prefix.extend(["--env", f"RAY_raylet_start_wait_time_s={raylet_wait}"])
    prefix.append(sif)
    return prefix


def _build_container_pythonpath() -> str:
    """Build the in-container PYTHONPATH for the Apptainer RL runtime mode.

    Prepends the live host SkyRL (skyrl-train), harbor, and workdir paths so the
    bind-mounted host source wins over the install baked into the SIF (the SIF
    installed our SkyRL fork ``--no-deps``). Resolved at runtime inside the
    sbatch job from environment set by the dotenv + sbatch (SKYRL_HOME, DCFT,
    WORKDIR), so it reflects the actual cluster paths. Mirrors
    ``universal_rl.sbatch:143`` (``PYTHONPATH="$WORKDIR:..."``).

    Returns:
        Colon-joined PYTHONPATH string. Includes the existing $PYTHONPATH tail
        so nothing already on the host path is dropped.
    """
    parts: List[str] = []

    # Extra pure-python deps installed into a bind-mounted dir for SIFs built
    # --no-deps (e.g. hydra-core + antlr4, which the qwen3_next/main_tbench
    # entrypoint imports but skyrl_megatron_vllm.sif lacks). Sourced from
    # RL_CONTAINER_PYDEPS (colon-separated); prepended FIRST so it wins. Unset
    # => no-op (byte-identical to the prior path).
    pydeps = os.environ.get("RL_CONTAINER_PYDEPS", "")
    for p in (x for x in pydeps.split(":") if x):
        parts.append(p)

    # Resolve SKYRL_HOME with {SkyRL, MarinSkyRL} dir-name hardening (honors an
    # explicit SKYRL_HOME / RL_REPO_DIR override first; see resolve_rl_repo_dir).
    skyrl_home = _resolve_skyrl_home()
    if skyrl_home:
        parts.append(os.path.join(skyrl_home, "skyrl-train"))

    # Harbor lives as a sibling of OpenThoughts-Agent ($DCFT/../harbor on
    # Jupiter); fall back to $HARBOR_HOME if set explicitly.
    harbor_home = os.environ.get("HARBOR_HOME")
    if not harbor_home:
        dcft = os.environ.get("DCFT")
        if dcft:
            harbor_home = os.path.join(os.path.dirname(dcft.rstrip("/")), "harbor")
    if harbor_home:
        # Harbor uses a src/ layout (importable package at harbor/src/harbor),
        # so the importable root is harbor/src — add it when present (current
        # marin harbor). Keep the repo root too for the flat-layout fallback.
        harbor_src = os.path.join(harbor_home, "src")
        if os.path.isdir(os.path.join(harbor_src, "harbor")):
            parts.append(harbor_src)
        parts.append(harbor_home)

    workdir = os.environ.get("WORKDIR") or os.environ.get("DCFT")
    if workdir:
        parts.append(workdir)

    existing = os.environ.get("PYTHONPATH")
    if existing:
        parts.append(existing)

    # Flatten any colon-joined entries (e.g. the inherited $PYTHONPATH is itself
    # a colon-list) and drop entries containing shell metacharacters. On Jupiter
    # the inherited PYTHONPATH can still carry an UNEXPANDED dotenv tail like
    #   $(resolve_rl_repo_dir "$DCFT")/skyrl-train:${DCFT_PRIVATE:-$DCFT}${PYTHONPATH:+:$PYTHONPATH}
    # (when resolve_rl_repo.sh wasn't sourced in the shell that first exported
    # PYTHONPATH, the `$()`/`${}` substitutions never ran). Such an entry is a
    # bogus import path AND — because it contains spaces, `"`, and `)` — corrupts
    # the `apptainer exec --env PYTHONPATH=<value>` argument: apptainer mis-reads
    # the value tail as the SIF image path and dies with
    #   FATAL: could not open image .../OpenThoughts-Agent/"...")/skyrl-train:...
    # → the ray head exits 255 before producing output (80B step-4 651533-651541
    # / 651960). The real importable roots (sif_pydeps, SkyRL/skyrl-train,
    # harbor/src, harbor, OTA) are all added above and never contain shell
    # syntax, so dropping the dirty entries is safe and keeps imports intact.
    _SHELL_META = set(" \t$()`\"'{}")
    flat: List[str] = []
    for entry in parts:
        for sub in entry.split(":"):
            if not sub:
                continue
            if any(ch in _SHELL_META for ch in sub):
                continue  # unexpanded/garbled path entry — skip
            if not sub.startswith("/"):
                continue  # only absolute paths are valid in-container imports;
                          # drops stray fragments left by splitting a garbled
                          # entry (e.g. the "+" from "${PYTHONPATH:+:...}")
            flat.append(sub)

    # De-dup while preserving order.
    seen = set()
    deduped = []
    for p in flat:
        if p and p not in seen:
            seen.add(p)
            deduped.append(p)
    return ":".join(deduped)


def prebuild_daytona_snapshots(
    resolved_train_data: List[str],
    max_new_snapshots: int = 10,
    max_org_snapshots: int = 60,
    build_region: str = "us",
    target_region: str = "",
    build_timeout: float = 600.0,
) -> None:
    """DEPRECATED shim: prefer ``hpc.snapshot_manager.ensure_snapshots``.

    Backward-compat wrapper around the new unified snapshot manager.
    Resolves a single org from ``DAYTONA_API_KEY`` (matching prior behavior),
    then delegates to ``ensure_snapshots``. Any direct callers (e.g. external
    scripts) keep working unchanged.
    """
    from hpc.snapshot_manager import ensure_snapshots, load_orgs_from_env, SnapshotCapExceeded

    if not os.environ.get("DAYTONA_API_KEY", ""):
        print("WARNING: DAYTONA_API_KEY not set; skipping snapshot pre-build.")
        return
    orgs = load_orgs_from_env(["default"])
    for org in orgs:
        org.target = build_region

    try:
        ensure_snapshots(
            resolved_train_data,
            orgs,
            max_new_snapshots=max_new_snapshots,
            max_org_snapshots=max_org_snapshots,
            target_region=target_region,
            build_timeout=build_timeout,
        )
    except SnapshotCapExceeded as exc:
        # Preserve the prior ValueError contract for any caller that catches it.
        raise ValueError(str(exc)) from exc


def resolve_rl_train_data(
    train_data: List[str],
    scratch_dir: Optional[str] = None,
    on_exist: str = "skip",
    verbose: bool = True,
) -> List[str]:
    """Resolve train_data paths, extracting HF datasets to local task directories.

    SkyRL's TerminalBenchTaskDataset expects local directory paths where each
    subdirectory is a task containing an instruction.md file. This function:
    1. Detects HuggingFace dataset identifiers (e.g., "org/repo-name")
    2. Extracts them to $SCRATCH/tasks/<repo-name>/ using extract_tasks_from_parquet
    3. Fixes permissions (chmod) to ensure tasks are readable
    4. Returns local filesystem paths for all datasets

    Args:
        train_data: List of dataset paths (local paths or HF repo IDs).
        scratch_dir: Base directory for extracted tasks (default: $SCRATCH/tasks or /tmp/tasks).
        on_exist: How to handle existing task directories ("skip", "overwrite", "error").
        verbose: Whether to print status messages.

    Returns:
        List of resolved local filesystem paths.

    Example:
        >>> resolve_rl_train_data(["penfever/my-dataset", "/local/path/tasks"])
        ['/scratch/tasks/my-dataset', '/local/path/tasks']
    """
    if not train_data:
        return []

    # Determine scratch directory for extracted tasks
    # IMPORTANT: Must use a shared filesystem visible to all compute nodes.
    # /tmp is local to each node and will NOT work for multi-node jobs.
    if scratch_dir is None:
        # Try multiple fallbacks in order of preference:
        # 1. $SCRATCH - standard HPC scratch directory
        # 2. $DCFT - project directory (set in dotenv files)
        # 3. $DCFT_PRIVATE - private project directory variant
        # 4. $HOME - user's home directory (usually shared on HPC)
        # 5. /tmp - LAST RESORT (local to each node, will fail on multi-node!)
        for env_var in ["SCRATCH", "DCFT", "DCFT_PRIVATE", "HOME"]:
            if os.environ.get(env_var):
                scratch_dir = os.environ[env_var]
                break
        else:
            scratch_dir = "/tmp"
            print("[rl_launch_utils] WARNING: Using /tmp for task extraction. "
                  "This is local to each node and may fail on multi-node jobs. "
                  "Set $SCRATCH, $DCFT, or $DCFT_PRIVATE to a shared filesystem path.")
    tasks_base = Path(scratch_dir) / "tasks"

    resolved_paths = []

    for data_path in train_data:
        if is_hf_dataset_path(data_path):
            # It's a HuggingFace dataset - extract to local directory
            # Extract repo name from "org/repo-name" -> "repo-name"
            repo_name = data_path.split("/")[-1]
            output_dir = tasks_base / repo_name

            if verbose:
                print(f"[rl_launch_utils] Extracting HF dataset: {data_path}")
                print(f"[rl_launch_utils] Output directory: {output_dir}")

            # Check if already extracted (when on_exist="skip")
            if on_exist == "skip" and output_dir.exists() and any(output_dir.iterdir()):
                if verbose:
                    print(f"[rl_launch_utils] Tasks already extracted, skipping: {output_dir}")
                resolved_paths.append(str(output_dir))
                continue

            # Run extract_tasks_from_parquet
            cmd = [
                sys.executable, "-m", "scripts.datagen.extract_tasks_from_parquet",
                "--parquet", data_path,
                "--output_dir", str(output_dir),
                "--on_exist", on_exist,
            ]

            if verbose:
                print(f"[rl_launch_utils] Running: {' '.join(cmd)}")

            # A stalled HF download INSIDE extract_tasks_from_parquet (mid-download socket
            # hang — seen 2026-07-13 on rno2a v0: ranks 3,6 had a byte-frozen `.incomplete`
            # parquet) used to block subprocess.run FOREVER, so the node never `ray start`ed and
            # the head hit its "Only N/8 Ray nodes joined within 1800s" gang-timeout → TERMINAL
            # (--max-retries does NOT rescue gang-formation). This mirrors the stage_model fix
            # (start_rl_iris_controller.py, 2026-07-13): a per-attempt timeout converts a stall
            # into a retry; HF resumes the partial `.incomplete` shard on the next attempt, so a
            # killed-mid-download attempt loses nothing. 600s covers a clean extract yet fits
            # several retries inside the 1800s gang-join budget.
            EXTRACT_ATTEMPT_TIMEOUT_S = 600
            last_err = ""
            for attempt in range(1, 7):
                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=EXTRACT_ATTEMPT_TIMEOUT_S,
                    )
                    if verbose and result.stdout:
                        print(result.stdout)
                    break
                except subprocess.TimeoutExpired:
                    last_err = (f"extract_tasks_from_parquet stalled > {EXTRACT_ATTEMPT_TIMEOUT_S}s "
                                "(mid-download socket hang); killed, retrying (HF resumes the partial shard)")
                    print(f"[rl_launch_utils] extract attempt {attempt}/6 TIMED OUT for {data_path}: {last_err}")
                    time.sleep(min(30, 2 ** attempt))
                except subprocess.CalledProcessError as e:
                    last_err = f"stdout: {e.stdout}\n  stderr: {e.stderr}"
                    print(f"[rl_launch_utils] extract attempt {attempt}/6 failed for {data_path}:")
                    print(f"  {last_err}")
                    time.sleep(min(30, 2 ** attempt))
            else:
                raise RuntimeError(
                    f"Failed to extract HF dataset after 6 attempts: {data_path}: {last_err}"
                )

            # Fix permissions on extracted tasks (chmod -R a+rX)
            _fix_task_permissions(output_dir, verbose=verbose)

            resolved_paths.append(str(output_dir))
        else:
            # It's a local path - fix permissions just in case
            local_path = Path(data_path)
            if local_path.exists():
                _fix_task_permissions(local_path, verbose=verbose)
            resolved_paths.append(data_path)

    return resolved_paths


def _fix_task_permissions(task_dir: Path, verbose: bool = True) -> None:
    """Fix permissions on task directory to ensure files are readable.

    Runs chmod -R a+rX on the directory to make all files readable
    and directories traversable.

    IDEMPOTENCY GUARD (added 2026-06-19): the recursive ``chmod -R a+rX`` over a
    large task tree (e.g. the 5000-task ``exp_rpt_pymethods2test-large``, ~100K
    inodes) issues ~100K GPFS metadata WRITES on EVERY launch, even when the tree
    is already world-readable from a prior successful launch. Under GPFS metadata
    contention that recursive chmod has wedged the launcher in uninterruptible
    D-state for 35+ min before ``sbatch`` is ever reached. So we first do a SINGLE
    cheap ``stat`` of the top-level dir (fast even under contention, ~0.004s) and
    SHORT-CIRCUIT when its perms already satisfy ``a+rX`` for a directory
    (group+other readable AND group+other traversable). We only skip when perms are
    VERIFIED already-correct; any dir that genuinely needs the fix (top-level bits
    missing, or not yet stat-able) still gets the full recursive chmod. This is a
    conservative top-level check: if the recursive walk were ever interrupted it
    could leave inner files unfixed, but in practice the tree is written atomically
    by extraction and re-chmod'd as a unit, so a correct top-level dir implies a
    correct tree (and the prior-launch path proves it). NEVER use find/du here.

    Args:
        task_dir: Path to task directory.
        verbose: Whether to print status messages.
    """
    if not task_dir.exists():
        return

    # Cheap idempotency probe: a single stat on the top-level dir. The bits
    # `chmod a+rX` sets on a DIRECTORY are S_IRGRP|S_IXGRP|S_IROTH|S_IXOTH (the
    # owner already has them post-extraction). If all four are present, the
    # recursive chmod would be a ~100K-write no-op -> skip it.
    _RX_BITS = stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    try:
        mode = task_dir.stat().st_mode
        if (mode & _RX_BITS) == _RX_BITS:
            if verbose:
                print(f"[rl_launch_utils] Permissions already a+rX on top-level "
                      f"dir, skipping recursive chmod: {task_dir}")
            return
    except OSError as e:
        # stat failed (race / transient FS) -> fall through to the chmod, which
        # is the safe, correctness-preserving default.
        if verbose:
            print(f"[rl_launch_utils] stat probe failed on {task_dir} ({e}); "
                  f"running recursive chmod to be safe.")

    if verbose:
        print(f"[rl_launch_utils] Fixing permissions on: {task_dir}")

    try:
        subprocess.run(
            ["chmod", "-R", "a+rX", str(task_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # Don't fail the whole job for permission issues
        print(f"[rl_launch_utils] Warning: chmod failed on {task_dir}: {e.stderr}")


def compute_num_inference_engines(
    num_nodes: int,
    gpus_per_node: int,
    tensor_parallel_size: int = 1,
) -> int:
    """
    Compute the number of vLLM inference engines for distributed RL training.

    In SkyRL, inference engines are used for rollout generation. The total number
    is determined by dividing the total GPU count by the tensor parallel size.

    Args:
        num_nodes: Number of nodes in the job.
        gpus_per_node: Number of GPUs per node.
        tensor_parallel_size: Tensor parallel size for vLLM (default: 1).

    Returns:
        Number of inference engines.

    Example:
        >>> compute_num_inference_engines(num_nodes=2, gpus_per_node=4, tensor_parallel_size=1)
        8
        >>> compute_num_inference_engines(num_nodes=2, gpus_per_node=4, tensor_parallel_size=2)
        4
    """
    total_gpus = num_nodes * gpus_per_node
    return total_gpus // tensor_parallel_size


def get_tensor_parallel_size(
    gpus_per_node: int,
    model_size_hint: Optional[str] = None,
) -> int:
    """
    Determine appropriate tensor parallel size based on model and GPU configuration.

    For most models, TP=1 is sufficient. Larger models (70B+) may need TP=2 or TP=4.

    Args:
        gpus_per_node: Number of GPUs per node.
        model_size_hint: Optional hint about model size (e.g., "7B", "70B", "405B").

    Returns:
        Recommended tensor parallel size.
    """
    # Default to TP=1 for most models
    if model_size_hint is None:
        return 1

    # Parse model size from hint
    model_size_hint = model_size_hint.upper()
    if "405B" in model_size_hint or "400B" in model_size_hint:
        # Very large models need high TP
        return min(8, gpus_per_node)
    elif "70B" in model_size_hint or "72B" in model_size_hint:
        # Large models benefit from TP=2 or TP=4
        return min(4, gpus_per_node)
    elif "32B" in model_size_hint or "34B" in model_size_hint:
        # Medium-large models may benefit from TP=2
        return min(2, gpus_per_node)
    else:
        # Smaller models (7B, 8B, 14B, etc.) work fine with TP=1
        return 1


def build_rl_env_vars(
    exp_args: Dict[str, Any],
    run_paths: RLRunPaths,
    hpc: Optional[Any] = None,
) -> Dict[str, str]:
    """
    Build environment variables dictionary for RL training jobs.

    Args:
        exp_args: Experiment arguments dictionary.
        run_paths: Validated durable RL paths.
        hpc: Optional HPC configuration object.

    Returns:
        Dictionary of environment variable name -> value.
    """
    env_vars = {}

    # Tensor parallel and inference engine settings
    num_nodes = int(exp_args.get("num_nodes", 1))
    gpus_per_node = int(exp_args.get("gpus_per_node", 4))
    tensor_parallel_size = int(exp_args.get("tensor_parallel_size", 1))

    env_vars["TENSOR_PARALLEL_SIZE"] = str(tensor_parallel_size)
    env_vars["NUM_INFERENCE_ENGINES"] = str(
        compute_num_inference_engines(num_nodes, gpus_per_node, tensor_parallel_size)
    )

    # Policy nodes (can be different from total nodes for asymmetric setups)
    policy_num_nodes = exp_args.get("policy_num_nodes")
    if policy_num_nodes is not None:
        env_vars["POLICY_NUM_NODES"] = str(policy_num_nodes)
    else:
        env_vars["POLICY_NUM_NODES"] = str(num_nodes)

    env_vars["SKYRL_EXPORT_PATH"] = str(run_paths.export_dir)

    # Inherit all HPC-specific environment variables (WANDB_MODE, GLOO_USE_IPV6, etc.)
    if hpc is not None and hasattr(hpc, "env_vars"):
        hpc_env = hpc.env_vars or {}
        for key, value in hpc_env.items():
            env_vars[key] = value

    return env_vars


def get_rl_env_exports(exp_args: Dict[str, Any], run_paths: RLRunPaths, hpc: Optional[Any] = None) -> str:
    """
    Generate shell export statements for RL environment variables.

    Args:
        exp_args: Experiment arguments dictionary.
        run_paths: Validated durable RL paths.
        hpc: Optional HPC configuration object.

    Returns:
        Multi-line string of export statements.
    """
    env_vars = build_rl_env_vars(exp_args, run_paths, hpc)

    if not env_vars:
        return "# No RL-specific environment variables"

    lines = ["# RL training environment variables"]
    for key, value in env_vars.items():
        # shlex.quote keeps values with shell-special characters intact —
        # notably the JSON RAY_object_spilling_config blob, whose inner
        # double-quotes would otherwise prematurely close a naive
        # `export KEY="value"` and mangle the spill config. shlex.quote is a
        # no-op for plain values, so this is byte-identical for all other vars.
        lines.append(f"export {key}={shlex.quote(str(value))}")

    return "\n".join(lines)


def get_rl_env_activation(exp_args: Dict[str, Any]) -> str:
    """
    Generate shell code for RL environment activation.

    Supports two modes:
    1. Conda environment (--rl_use_conda --rl_conda_env NAME)
    2. venv created by setup_rl_env.sh (default)

    Args:
        exp_args: Experiment arguments dictionary.

    Returns:
        Multi-line shell script for environment activation.
    """
    # Apptainer runtime mode (OPT-IN): when --rl_container_sif is set the Python
    # comes from inside the SIF (and the bind-mounted host SkyRL/harbor), so the
    # host venv/conda activation is skipped entirely. The three command seams
    # (Ray head/worker, ray.init() wait, SkyRL driver) are wrapped in
    # `apptainer exec --nv` downstream. Mirrors SFT-MCA, which `conda activate`s
    # *inside* the container, not on the host.
    container_sif = exp_args.get("rl_container_sif")
    if container_sif:
        return (
            "# RL Apptainer runtime mode (--rl_container_sif set):\n"
            "# Host venv/conda activation SKIPPED. Python is provided by the SIF\n"
            f"# ({container_sif}); Ray + SkyRL run via `apptainer exec --nv`.\n"
            'echo "RL runtime: Apptainer SIF (host venv/conda activation skipped)"'
        )

    use_conda = exp_args.get("rl_use_conda", False)
    conda_env = exp_args.get("rl_conda_env", "dcagent-rl")

    if use_conda:
        return f'''# Using conda environment for RL: {conda_env}
echo "Activating conda environment: {conda_env}"
# Disable unbound variable check during conda operations (conda scripts reference unset vars)
set +u
# Initialize conda for non-interactive shell (required before conda activate)
if [[ -n "${{CONDA_EXE:-}}" ]]; then
  # Use CONDA_EXE to find conda.sh
  CONDA_BASE=$(dirname $(dirname "$CONDA_EXE"))
  source "$CONDA_BASE/etc/profile.d/conda.sh"
elif [[ -f "${{HOME}}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${{HOME}}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${{HOME}}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${{HOME}}/anaconda3/etc/profile.d/conda.sh"
elif command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: Could not find conda installation for initialization"
  set -u
  exit 1
fi
conda activate {conda_env}
# Re-enable unbound variable check
set -u'''
    else:
        return '''# Using venv for RL (created by ./hpc/setup_rl_env.sh)
# IMPORTANT: Deactivate conda environment to prevent import conflicts,
# but KEEP conda paths in PATH because the venv's python symlink may point
# to the conda Python that was used when the venv was created.
set +u  # conda deactivate may reference unset variables
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "Deactivating conda environment: $CONDA_PREFIX"
  # Deactivate all stacked conda environments
  while [[ -n "${CONDA_PREFIX:-}" ]]; do
    conda deactivate 2>/dev/null || break
  done
  # Unset the environment name variable so imports don't get confused
  unset CONDA_DEFAULT_ENV
  # NOTE: We keep CONDA_PREFIX and conda paths in PATH because the venv's
  # python binary is often a symlink to the conda Python.
fi
set -u

RL_ENV_DIR="${RL_ENV_DIR:-$WORKDIR/envs/rl}"
if [[ -d "$RL_ENV_DIR" ]]; then
  echo "Activating RL environment: $RL_ENV_DIR"
  source "$RL_ENV_DIR/bin/activate"
elif [[ -n "${DCFT_RL_ENV:-}" ]] && [[ -d "$DCFT_RL_ENV" ]]; then
  echo "Activating RL environment from DCFT_RL_ENV: $DCFT_RL_ENV"
  source "$DCFT_RL_ENV/bin/activate"
else
  echo "Warning: RL environment not found at $RL_ENV_DIR"
  echo "Run ./hpc/setup_rl_env.sh to create it, or set DCFT_RL_ENV"
fi

# Verify we're using the correct Python
echo "Python executable: $(which python)"
echo "Python path check: $(python -c 'import sys; print(sys.executable)')"'''


# =============================================================================
# RLJobConfig and Job Submission
# =============================================================================


@dataclass
class RLJobConfig:
    """Configuration for an RL training job (serialized to JSON for sbatch).

    This dataclass contains all information needed to run an RL training job
    via the universal_rl.sbatch template and RLJobRunner.
    """

    job_name: str
    experiments_dir: str
    cluster_name: str

    # SkyRL settings
    skyrl_entrypoint: str
    trials_dir: str
    skyrl_hydra_args: List[str] = field(default_factory=list)

    # Model and data
    model_path: str = ""
    train_data: List[str] = field(default_factory=list)
    val_data: List[str] = field(default_factory=list)

    # Resource allocation
    num_nodes: int = 1
    gpus_per_node: int = 4
    cpus_per_node: int = 48
    tensor_parallel_size: int = 1

    # Networking
    ray_port: int = 6379
    master_port: int = 12345

    # Paths
    checkpoints_dir: Optional[str] = None
    export_path: Optional[str] = None

    # Cluster-specific flags
    needs_ssh_tunnel: bool = False
    needs_cuda_detection: bool = False

    # Pinggy tunnel settings (for cloud backends with installed agents)
    pinggy_persistent_url: Optional[str] = None
    pinggy_token: Optional[str] = None

    # Ingress mode: "pinggy" (default, legacy) or "controller" (auth-gated
    # controller ingress). ingress_host is the public controller-ingress host.
    ingress_mode: str = "pinggy"
    ingress_host: Optional[str] = None

    # Literal-token capture: co-locate harbor's RecordProxy in front of the vLLM
    # engines and route the agent endpoint through it. Default off = no proxy.
    record_literal: bool = False

    # Agent/environment info (for needs_pinggy_tunnel decision)
    agent_name: str = "terminus-2"
    harbor_env: str = "daytona"

    proxychains_binary: Optional[str] = None

    # Apptainer/Singularity RL runtime mode (OPT-IN). When container_sif is set,
    # the SkyRL trainer + Ray head/workers run inside the SIF via
    # `apptainer exec --nv` instead of activating the host venv/conda. See
    # scope_rl_via_apptainer_launcher.md.
    container_sif: Optional[str] = None
    container_binds: List[str] = field(default_factory=list)

    # Ray object store size in GB (default: 40)
    ray_object_store_gb: float = 40.0

    # Post-training trace upload settings
    trace_upload_enabled: bool = False
    trace_upload_repo_org: str = "DCAgent"
    trace_upload_episodes: str = "last"
    trace_upload_dataset_type: str = "SFT"
    trace_upload_cleanup: bool = True


@dataclass(frozen=True)
class RLLaunchArtifacts:
    """Resolved launch script and the exact SkyRL command it contains."""

    sbatch_path: Path
    skyrl_entrypoint: str
    hydra_args: tuple[str, ...]


def build_skyrl_command_string(config: RLJobConfig) -> str:
    """Build the full SkyRL command string for the sbatch template.

    Args:
        config: RLJobConfig with entrypoint and hydra args.

    Returns:
        Shell command string with proper line continuations.
    """
    parts = [f"python -m {config.skyrl_entrypoint}"]

    for arg in config.skyrl_hydra_args:
        parts.append(f"  {arg}")

    return " \\\n".join(parts)


def _build_rl_container_env(container: Mapping[str, Any], exp_args: dict) -> str:
    """Build the `{rl_container_env}` sbatch block from a yaml `container:` section.

    Side effect: when ``container.sif`` is set and ``--rl_container_sif`` was NOT
    passed on the CLI, populates ``exp_args["rl_container_sif"]`` (and
    ``rl_container_binds`` from ``container.binds``) so the existing Apptainer
    runtime mode activates downstream (host venv/conda activation skipped; Ray +
    SkyRL run via ``apptainer exec --nv``). An explicit CLI ``--rl_container_sif``
    always wins.

    Emits ``export`` lines for:
      - ``RL_CONTAINER_OVERLAYS`` — colon-joined ``container.overlays`` (each
        mounted ``--overlay <p>:ro`` by build_apptainer_prefix).
      - ``RL_CONTAINER_PYDEPS``  — ``container.pydeps`` (prepended to the
        in-container PYTHONPATH by _build_container_pythonpath).
      - one line per ``container.extra_env`` key (e.g. SKYRL_GDN_MASK_FLA,
        PYTORCH_CUDA_ALLOC_CONF). ``APPTAINERENV_`` mirroring is the author's
        responsibility (list both keys in extra_env) — kept verbatim, no magic.

    Returns a comment-only stub when ``container`` is empty/absent, so configs
    without a container section are byte-identical to the prior path.

    Args:
        container: The parsed yaml ``container:`` mapping (or empty dict).
        exp_args: Experiment args dict (mutated to set rl_container_sif/binds).

    Returns:
        Multi-line shell string of export statements (or a single comment line).
    """
    if not container:
        return "# (no container section in RL yaml — host venv/conda runtime)"

    # SIF: populate the existing CLI-flag plumbing if not explicitly overridden.
    sif = container.get("sif")
    if sif and not exp_args.get("rl_container_sif"):
        exp_args["rl_container_sif"] = sif
        binds = container.get("binds")
        if binds and not exp_args.get("rl_container_binds"):
            exp_args["rl_container_binds"] = list(binds)

    lines: List[str] = []

    overlays = container.get("overlays") or []
    if overlays:
        joined = ":".join(str(o) for o in overlays)
        lines.append(f'export RL_CONTAINER_OVERLAYS="{joined}"')

    pydeps = container.get("pydeps")
    if pydeps:
        lines.append(f'export RL_CONTAINER_PYDEPS="{pydeps}"')

    extra_env = container.get("extra_env") or {}
    for key, value in extra_env.items():
        # bool -> shell-friendly literal (True/False kept as-is for SKYRL_* flags
        # that test truthiness via int(); most callers use 1/0 or strings).
        if isinstance(value, bool):
            value = int(value)
        lines.append(f'export {key}="{value}"')

    if not lines:
        return "# (container section present but defined no overlays/pydeps/extra_env)"

    return "\n".join(lines)


def validate_trace_upload_environment(
    terminal_bench: Mapping[str, Any], container: Mapping[str, Any]
) -> None:
    """Reject trace uploads from a runtime configured for offline Hub access."""
    trace_upload = terminal_bench.get("trace_upload") or {}
    if not trace_upload.get("enabled"):
        return

    extra_env = container.get("extra_env") or {}
    offline_keys = ("HF_HUB_OFFLINE", "APPTAINERENV_HF_HUB_OFFLINE")
    enabled = [key for key in offline_keys if str(extra_env.get(key, "")).lower() in {"1", "true", "yes"}]
    if enabled:
        raise ValueError(
            "terminal_bench.trace_upload.enabled=true conflicts with "
            f"container.extra_env {', '.join(enabled)}; disable trace upload or remove offline Hub mode"
        )


def construct_rl_sbatch_script(exp_args: dict, hpc) -> RLLaunchArtifacts:
    """Construct RL sbatch script using the universal template system.

    This follows the same pattern as construct_sft_sbatch_script() but for RL jobs.

    Args:
        exp_args: Experiment arguments dictionary.
        hpc: HPC cluster configuration.

    Returns:
        Generated sbatch path and the exact SkyRL command.
    """
    from hpc.launch_utils import (
        resolve_job_and_paths,
        substitute_template,
        build_sbatch_directives,
        resolve_conda_activate,
    )
    from hpc.rl_config_utils import (
        apply_context_budget_overrides,
        build_skyrl_hydra_args,
        extract_terminal_bench_agent_env,
        parse_rl_config,
    )

    print("\n=== RL MODE (Universal Launcher) ===")

    # Parse RL config YAML
    rl_config_path = exp_args.get("rl_config")
    if not rl_config_path:
        raise ValueError("--rl_config is required for RL jobs")

    parsed = parse_rl_config(rl_config_path, model_override=exp_args.get("model_path"))
    print(f"Loaded RL config from: {parsed.config_path}")
    container = parsed.raw.get("container") or {}
    validate_trace_upload_environment(parsed.terminal_bench or {}, container)

    # --- RL container section (Apptainer SIF + overlays + pydeps + extra env) ---
    # Optional top-level `container:` block in the RL yaml. When present it lets a
    # config self-describe its Apptainer runtime (the Qwen3-Next-80B case), so the
    # standard `python -m hpc.launch` path reproduces the previously hand-baked
    # sbatch without env-vars-at-launch-time or a giant --skyrl_override string.
    #
    #   container:
    #     sif: /abs/path/to/image.sif            # -> --rl_container_sif (host venv skip)
    #     binds: ["/e/scratch", "/e/data1"]      # -> --rl_container_binds (default if omitted)
    #     overlays: ["/abs/a.img", "/abs/b.img"] # -> RL_CONTAINER_OVERLAYS (colon-joined, :ro)
    #     pydeps: "/abs/sif_pydeps"              # -> RL_CONTAINER_PYDEPS (PYTHONPATH prepend)
    #     extra_env:                             # -> verbatim `export K=V` lines
    #       SKYRL_GDN_MASK_FLA: 1
    #       PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
    #
    # An explicit --rl_container_sif CLI flag still wins (only fills if unset), so
    # nothing changes for configs without a `container:` section.
    rl_container_env_block = _build_rl_container_env(container, exp_args)

    # Extract agent name and harbor_env from terminal_bench config
    yaml_agent_name, yaml_harbor_env = extract_terminal_bench_agent_env(parsed)

    # CLI overrides YAML for harbor_env
    harbor_env = exp_args.get("harbor_env") or yaml_harbor_env or "daytona"
    agent_name = yaml_agent_name  # Agent name comes from YAML only

    print(f"Terminal bench: agent={agent_name}, harbor_env={harbor_env}")

    # Resolve train_data: extract HF datasets to local task directories
    # This must happen BEFORE building Hydra args so the local paths are used
    train_data_raw = exp_args.get("train_data") or []
    if isinstance(train_data_raw, str):
        # Handle JSON string from CLI
        import ast
        try:
            train_data_raw = ast.literal_eval(train_data_raw)
        except (ValueError, SyntaxError):
            train_data_raw = [train_data_raw]

    if train_data_raw:
        print(f"Resolving train_data: {train_data_raw}")
        resolved_train_data = resolve_rl_train_data(train_data_raw)
        exp_args["train_data"] = resolved_train_data
        print(f"Resolved train_data: {resolved_train_data}")

        # Pre-build Daytona snapshots for RL region (train_data only; val_data
        # snapshots are not pre-built due to capacity constraints). Routes
        # through the unified hook in hpc.launch_utils; the caller assembles
        # `orgs` explicitly (no magical job_type defaults).
        if os.environ.get("DAYTONA_API_KEY") and resolved_train_data:
            from hpc.launch_utils import maybe_prebuild_daytona_snapshots
            from hpc.snapshot_manager import OrgConfig, load_orgs_from_env

            api_key_override = get_daytona_api_key_override(exp_args)
            if api_key_override and api_key_override != os.environ.get("DAYTONA_API_KEY", ""):
                orgs = [OrgConfig(name="cli", api_key=api_key_override)]
            else:
                orgs = load_orgs_from_env(["default"])
            maybe_prebuild_daytona_snapshots(
                resolved_train_data,
                harbor_env=harbor_env,
                orgs=orgs,
            )

    # Resolve val_data similarly (eval datasets may also be HF repos)
    # Check CLI first, then fall back to YAML config default
    val_data_raw = exp_args.get("val_data")
    if val_data_raw is None:
        # Get default from YAML config
        val_data_raw = parsed.data.get("val_data", [])
    if isinstance(val_data_raw, str):
        import ast
        try:
            val_data_raw = ast.literal_eval(val_data_raw)
        except (ValueError, SyntaxError):
            val_data_raw = [val_data_raw]

    if val_data_raw:
        print(f"Resolving val_data: {val_data_raw}")
        resolved_val_data = resolve_rl_train_data(val_data_raw)
        exp_args["val_data"] = resolved_val_data
        print(f"Resolved val_data: {resolved_val_data}")

    # Pre-download model for RL jobs
    # SkyRL's FSDP and DeepSpeed strategies don't have built-in pre-download logic
    # (only Megatron does), so we always pre-download HF models to avoid issues with:
    # - Multiple workers trying to download simultaneously
    # - Network timeouts on compute nodes
    # - Auth issues in distributed settings
    from hpc.checkpoint_utils import pre_download_model, is_huggingface_repo
    model_path = exp_args.get("model_path") or parsed.model.get("model_name_or_path", "")
    if model_path and is_huggingface_repo(model_path):
        print(f"Pre-downloading model for SkyRL: {model_path}")
        result = pre_download_model(model_path)
        exp_args["model_path"] = result.local_path
        print(f"Model available at: {result.local_path}")
    elif model_path:
        exp_args["model_path"] = model_path

    # Resolve launcher artifact paths before the durable RL paths. A collision
    # may rename the artifact directory, while RLPathManager must still inspect
    # the canonical directory and its numbered siblings.
    job_setup = resolve_job_and_paths(
        exp_args,
        job_type_label="RL",
    )
    job_name = job_setup.job_name
    exp_paths = job_setup.paths
    experiments_subdir = str(exp_paths.root)

    skyrl_overrides = exp_args.get("skyrl_override") or []
    parsed, passthrough_overrides = apply_context_budget_overrides(
        parsed, skyrl_overrides
    )
    print(f"Resolved context budget: {parsed.context_budget.as_dict()}")

    run_paths = RLPathManager(
        job_name,
        exp_paths.canonical_root or exp_paths.root,
        exp_paths.root,
    ).resolve(
        trainer_config=parsed.trainer,
        terminal_bench_config=parsed.terminal_bench or {},
        skyrl_overrides=passthrough_overrides,
        launch_intent=(
            RLLaunchIntent.FRESH
            if exp_args.get("overwrite_output_dir") or exp_args.get("allow_overwrite")
            else RLLaunchIntent.AUTO
        ),
    )
    print(run_paths.describe())

    hydra_args = build_skyrl_hydra_args(parsed, exp_args, hpc, run_paths=run_paths)
    if passthrough_overrides:
        hydra_args.extend(passthrough_overrides)
        print(f"Applied {len(passthrough_overrides)} non-context CLI overrides")

    # Extract config values
    num_nodes = int(exp_args.get("num_nodes") or 1)
    gpus_per_node = int(exp_args.get("gpus_per_node") or hpc.gpus_per_node)
    cpus_per_node = int(exp_args.get("cpus_per_node") or hpc.cpus_per_node)

    # Build RLJobConfig
    job_config = RLJobConfig(
        job_name=job_name,
        experiments_dir=experiments_subdir,
        cluster_name=hpc.name,
        skyrl_entrypoint=parsed.entrypoint,
        skyrl_hydra_args=hydra_args,
        model_path=exp_args.get("model_path", ""),
        train_data=exp_args.get("train_data", []),
        val_data=exp_args.get("val_data", []),
        num_nodes=num_nodes,
        gpus_per_node=gpus_per_node,
        cpus_per_node=cpus_per_node,
        tensor_parallel_size=parsed.tensor_parallel_size,
        ray_port=int(exp_args.get("ray_port") or 6379),
        master_port=int(exp_args.get("master_port") or 12345),
        export_path=str(run_paths.export_dir),
        trials_dir=str(run_paths.trials_dir),
        needs_ssh_tunnel=hpc.needs_ssh_tunnel,
        needs_cuda_detection=getattr(hpc, "needs_cuda_detection", False),
        # Pinggy tunnel settings (for cloud backends with installed agents)
        pinggy_persistent_url=exp_args.get("pinggy_persistent_url"),
        pinggy_token=exp_args.get("pinggy_token"),
        ingress_mode=exp_args.get("ingress_mode") or "pinggy",
        ingress_host=exp_args.get("ingress_host"),
        record_literal=bool(exp_args.get("record_literal")),
        agent_name=agent_name,
        harbor_env=harbor_env,
        container_sif=exp_args.get("rl_container_sif"),
        container_binds=list(exp_args.get("rl_container_binds") or DEFAULT_RL_CONTAINER_BINDS)
        if exp_args.get("rl_container_sif") else [],
        ray_object_store_gb=float(exp_args.get("ray_object_store_gb", 40.0)),
    )

    # Populate trace upload settings from parsed terminal_bench config
    if parsed.terminal_bench:
        tu = parsed.terminal_bench.get("trace_upload", {})
        job_config.trace_upload_enabled = bool(tu.get("enabled", False))
        job_config.trace_upload_repo_org = tu.get("repo_org", "DCAgent")
        job_config.trace_upload_episodes = tu.get("episodes", "last")
        job_config.trace_upload_dataset_type = tu.get("dataset_type", "SFT")
        job_config.trace_upload_cleanup = bool(tu.get("cleanup", True))

    # Write config JSON
    config_dir = exp_paths.configs
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{job_name}_rl_config.json"
    config_path.write_text(json.dumps(asdict(job_config), indent=2))
    print(f"Wrote RL job config to {config_path}")

    # Load and populate universal template
    template_path = Path(__file__).parent / "sbatch_rl" / "universal_rl.sbatch"
    if not template_path.exists():
        raise FileNotFoundError(f"RL sbatch template not found: {template_path}")
    template_text = template_path.read_text()

    # Build cluster-specific SBATCH directives
    sbatch_directives = build_sbatch_directives(hpc, exp_args)

    # Generate RL environment exports
    rl_env_exports = get_rl_env_exports(exp_args, run_paths, hpc)

    # Generate CUDA setup code
    cuda_setup = ""
    if getattr(hpc, "needs_cuda_detection", False):
        cuda_setup = """# CUDA path detection (Perlmutter and similar)
if [[ -d /opt/nvidia/hpc_sdk ]]; then
    export CUDA_HOME=$(ls -d /opt/nvidia/hpc_sdk/*/Linux_x86_64/cuda/* 2>/dev/null | head -1)
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi"""

    # Build SkyRL command
    skyrl_command = build_skyrl_command_string(job_config)

    # Generate RL environment activation code (conda or venv)
    # Containerized clusters (EmpireAI/Pyxis): skip host activation — env is in the .sqsh.
    if hpc.is_containerized:
        rl_env_activation = (
            '# RL Pyxis/Enroot container mode (hpc.container_image set):\n'
            '# Host venv/conda activation SKIPPED. Python is inside the container.\n'
            'echo "RL runtime: Pyxis container (host venv/conda activation skipped)"'
        )
    else:
        rl_env_activation = get_rl_env_activation(exp_args)

    substitutions = {
        "time_limit": exp_args.get("time_limit") or "24:00:00",
        "num_nodes": str(num_nodes),
        "cpus_per_node": str(cpus_per_node),
        "experiments_dir": experiments_subdir,
        "job_name": job_name,
        "sbatch_extra_directives": "\n".join(sbatch_directives),
        "module_commands": hpc.get_module_commands(),
        "conda_activate": resolve_conda_activate(hpc, exp_args),
        "cluster_env_file": hpc.dotenv_filename,
        "cuda_setup": cuda_setup,
        "nccl_exports": hpc.get_nccl_exports(),
        "rl_container_env": rl_container_env_block,
        # LATE re-emit of the SAME container.extra_env block, placed after
        # {rl_env_exports} in the template so config extra_env wins by shell
        # last-write over the hardcoded `export TORCH_NCCL_*` defaults and the
        # hpc.env_vars block (idempotent / no-op for non-colliding configs).
        "rl_container_env_late": rl_container_env_block,
        "rl_env_exports": rl_env_exports,
        "ray_env_exports": hpc.get_ray_env_exports(experiments_subdir),
        "rl_env_activation": rl_env_activation,
        "ssh_tunnel_setup": hpc.get_ssh_tunnel_setup(),
        "proxy_setup": hpc.get_proxy_setup(),
        "ray_port": str(job_config.ray_port),
        "master_port": str(job_config.master_port),
        "gpus_per_node": str(gpus_per_node),
        "config_path": str(config_path),
        "skyrl_command": skyrl_command,
        "email_address": os.environ.get("EMAIL_ADDRESS", ""),
        "harbor_env": job_config.harbor_env,
        "daytona_api_key_override": get_daytona_api_key_override(exp_args),
    }

    sbatch_text = substitute_template(template_text, substitutions)

    # Write sbatch script
    sbatch_dir = exp_paths.sbatch
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    sbatch_output = sbatch_dir / f"{job_name}_rl.sbatch"
    sbatch_output.write_text(sbatch_text)
    os.chmod(sbatch_output, 0o750)
    print(f"Wrote RL sbatch script to {sbatch_output}")

    return RLLaunchArtifacts(
        sbatch_path=sbatch_output,
        skyrl_entrypoint=parsed.entrypoint,
        hydra_args=tuple(hydra_args),
    )


def check_rl_environment() -> Optional[Path]:
    """Check if the RL environment exists and return its path.

    The RL environment is separate from the main environment due to
    dependency conflicts between SkyRL (torch 2.8, vllm 0.11.0) and
    datagen (torch 2.9, vllm 0.11.2).

    Returns:
        Path to RL environment if found, None otherwise.
    """
    # Check common locations
    candidates = []

    # DCFT_RL_ENV explicit override
    if os.environ.get("DCFT_RL_ENV"):
        candidates.append(Path(os.environ["DCFT_RL_ENV"]))

    # Standard location relative to DCFT
    if os.environ.get("DCFT"):
        candidates.append(Path(os.environ["DCFT"]) / "envs" / "rl")

    # Standard location relative to this file
    candidates.append(Path(__file__).parent.parent / "envs" / "rl")

    for candidate in candidates:
        if candidate.exists() and (candidate / "bin" / "activate").exists():
            return candidate

    return None


def launch_rl_job(exp_args: dict, hpc) -> Optional[str]:
    """Launch RL training job using universal template system.

    This is the main entry point for RL job submission from hpc/launch.py.

    Args:
        exp_args: Experiment arguments dictionary from CLI.
        hpc: HPC cluster configuration.

    Returns:
        Job ID if submitted, None if dry_run.
    """
    from hpc.launch_utils import launch_sbatch

    # Check for RL environment
    rl_env_path = check_rl_environment()
    if rl_env_path:
        print(f"RL environment found: {rl_env_path}")
    else:
        print("\n" + "=" * 60)
        print("WARNING: RL environment not found!")
        print("The RL environment is required for SkyRL training.")
        print("Create it with: ./hpc/setup_rl_env.sh")
        print("Or set DCFT_RL_ENV to point to an existing environment.")
        print("=" * 60 + "\n")

    # Construct the sbatch script
    artifacts = construct_rl_sbatch_script(exp_args, hpc)
    sbatch_path = str(artifacts.sbatch_path)

    # Get dependency if specified
    dependency = exp_args.get("dependency")

    # Dry run handling
    if exp_args.get("dry_run"):
        print(f"\nDRY RUN: RL sbatch script written to {sbatch_path}")
        if dependency:
            print(f"  Would submit with dependency: {dependency}")

        print("\nSkyRL command preview:")
        print(
            get_skyrl_command_preview(
                artifacts.skyrl_entrypoint, list(artifacts.hydra_args)
            )
        )

        return None

    # Chain max_restarts jobs with afterany dependencies so the RL job
    # auto-resumes from its latest checkpoint on preemption or failure.
    current_dependency = dependency
    max_restarts = int(exp_args.get("max_restarts") or 0)
    for i in range(max_restarts):
        job_id = launch_sbatch(sbatch_path, dependency=current_dependency)
        job_id = job_id.strip().split()[-1]
        current_dependency = f"afterany:{job_id}"
        print(f"  Restart {i + 1}/{max_restarts} queued: {job_id}")

    # Submit the final (or only) job
    job_id = launch_sbatch(sbatch_path, dependency=current_dependency)
    job_id = job_id.strip().split()[-1]
    print(f"\nRL job submitted: {job_id}")
    if max_restarts > 0:
        print(f"  ({max_restarts} auto-restart(s) chained with afterany dependencies)")

    return job_id


# =============================================================================
# RLJobRunner - Runs within sbatch
# =============================================================================


class RLJobRunner:
    """Runner for RL training jobs executed from sbatch.

    This class is instantiated within the sbatch script and handles:
    - Ray cluster setup (using shared RayCluster utility)
    - Environment configuration
    - SkyRL execution

    Usage (from sbatch):
        python -m hpc.rl_launch_utils --config /path/to/config.json
    """

    def __init__(self, config: RLJobConfig):
        self.config = config
        self._hpc = None

    def _get_hpc(self):
        """Lazy-load HPC configuration."""
        if self._hpc is None:
            from hpc.hpc import detect_hpc, clusters
            if self.config.cluster_name:
                for c in clusters:
                    if c.name.lower() == self.config.cluster_name.lower():
                        self._hpc = c
                        break
                if self._hpc is None:
                    raise ValueError(f"Unknown cluster: {self.config.cluster_name}")
            else:
                self._hpc = detect_hpc()
        return self._hpc

    def _already_complete_on_disk(self) -> bool:
        """True iff the canonical checkpoint already reached max_steps.

        Reads ``trainer.ckpt_path`` + ``trainer.max_steps`` from the job's Hydra
        args and compares ``<ckpt_path>/latest_ckpt_global_step.txt`` (the atomic
        completed-step marker SkyRL writes after each step) against max_steps.
        Returns True only when BOTH are resolvable and ``completed >= max_steps``.

        Fully defensive: any missing/unparseable input -> False (fall through to
        normal training). The marker is the same file the trainer's resume-at-max
        guard keys off, so the two guards agree on what "complete" means.
        """
        hydra_args = list(getattr(self.config, "skyrl_hydra_args", []) or [])
        hydra_values = hydra_override_values(hydra_args)
        max_steps_raw = hydra_values.get("trainer.max_steps")
        ckpt_path = hydra_values.get("trainer.ckpt_path")
        if not max_steps_raw or not ckpt_path:
            return False
        try:
            max_steps = int(max_steps_raw)
        except (TypeError, ValueError):
            return False
        if max_steps <= 0:
            return False

        marker = Path(ckpt_path) / LATEST_CHECKPOINT_FILE
        try:
            completed = int(marker.read_text().strip())
        except (OSError, ValueError):
            return False

        if completed >= max_steps:
            print(
                f"[RLJobRunner] Completion marker {marker} reports global_step "
                f"{completed} >= trainer.max_steps {max_steps}.",
                flush=True,
            )
            return True
        return False

    def run(self) -> int:
        """Execute the RL training job.

        After training (success or failure), launches trace upload as a subprocess
        and waits for it to complete before returning. This ensures the upload
        finishes before SLURM kills the job allocation.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        print(f"=== RLJobRunner: {self.config.job_name} ===", flush=True)

        # Resume-overshoot CHAIN guard. Each link in the afterany auto-restart
        # chain runs this before bringing up Ray. If the canonical checkpoint dir
        # already records a completed global_step >= max_steps, the run is DONE:
        # skip Ray + training entirely and return 0. Without this, every queued
        # afterany successor would still spin up a 14/16-node Ray cluster just to
        # resume-and-immediately-exit (the trainer's own resume-at-max guard makes
        # that exit clean, but it still wastes the node slot + bring-up time). This
        # short-circuits the whole remaining chain at ~zero cost. afterany fires
        # the successor regardless of the predecessor's exit status, so the marker
        # check — not the exit code — is what actually terminates the chain.
        if self._already_complete_on_disk():
            print(
                "[RLJobRunner] Canonical checkpoint already at/past max_steps — "
                "run is COMPLETE. Skipping Ray bring-up and training; exiting 0 "
                "(short-circuits the remaining afterany restart chain).",
                flush=True,
            )
            return 0

        training_exit_code = 1
        try:
            self._setup_environment()
            training_exit_code = self._run_with_ray()
        except Exception as e:
            print(f"RL job failed: {e}", file=sys.stderr, flush=True)
            import traceback
            traceback.print_exc()

        # On a crash, preserve Ray logs BEFORE the (potentially slow) trace
        # upload. The sbatch EXIT-trap also preserves them, but it only runs
        # after this Python process returns — and this process then blocks on
        # the trace upload below. If the wall clock kills the job during that
        # upload, the trap never completes and the crash evidence (the dead
        # worker's python-core-worker-*.log) is lost. Preserving here first
        # guarantees the evidence survives even if the upload is later killed.
        if training_exit_code != 0:
            self._preserve_ray_logs_on_crash()

        # Upload traces after training (success or failure — partial traces are valuable)
        upload_proc = self._launch_trace_upload(training_exit_code)
        if upload_proc is not None:
            print("[RLJobRunner] Waiting for trace upload to complete...", flush=True)
            upload_exit_code = upload_proc.wait()
            if upload_exit_code == 0:
                print("[RLJobRunner] Trace upload completed successfully.", flush=True)
                if self.config.trace_upload_cleanup:
                    trials_dir = Path(self.config.trials_dir)
                    if trials_dir.exists():
                        import shutil
                        print(f"[RLJobRunner] Cleaning up traces directory: {trials_dir}", flush=True)
                        shutil.rmtree(trials_dir, ignore_errors=True)
                        print("[RLJobRunner] Traces directory removed.", flush=True)
            else:
                print(f"[RLJobRunner] Trace upload failed with exit code {upload_exit_code}.", flush=True)

        return training_exit_code

    def _preserve_ray_logs_on_crash(self) -> None:
        """Best-effort: preserve Ray logs immediately after a training crash,
        before the (potentially slow) trace upload, so a wall-clock kill can't
        destroy the crash evidence.

        Mirrors the sbatch EXIT-trap's ``cleanup_ray_logs`` exactly (head-node
        ``/tmp/ray`` rsync + the shared ``collect_worker_ray_logs.sh`` for
        worker nodes) but runs early. Bounded by a timeout so it can never
        itself hang the job, and fully non-fatal — any failure is logged and
        ignored. Running twice (here + the EXIT trap) is harmless: rsync is
        additive.
        """
        job_dir = Path(self.config.experiments_dir) / self.config.job_name
        dest = job_dir / "ray_logs"
        repo_root = Path(__file__).resolve().parents[1]
        collector = repo_root / "hpc" / "ray" / "collect_worker_ray_logs.sh"

        # Same steps as the sbatch trap, in the same order.
        script = (
            'mkdir -p "$DEST"; '
            'if [[ -d /tmp/ray ]]; then rsync -a --ignore-errors /tmp/ray/ "$DEST/" 2>/dev/null || true; fi; '
            'for d in /tmp/ray_logs /tmp/ray_tmp; do '
            '  if [[ -d "$d" ]]; then rsync -a --ignore-errors "$d/" "$DEST/$(basename "$d")/" 2>/dev/null || true; fi; '
            'done; '
            'if [[ -f "$COLLECTOR" ]]; then source "$COLLECTOR"; collect_worker_ray_logs "$DEST"; fi'
        )
        env = {**os.environ, "DEST": str(dest), "COLLECTOR": str(collector)}

        print(
            f"[RLJobRunner] Crash detected (exit!=0) — preserving Ray logs to {dest} "
            f"BEFORE trace upload (so a wall-clock kill can't lose crash evidence)...",
            flush=True,
        )
        try:
            subprocess.run(
                ["bash", "-c", script],
                env=env,
                # Bounded short (120s, was 600s): on a Ray-bringup failure the
                # node raylets are already dead/unresponsive, so the worker-log
                # rsync/collector just blocks until this timeout — adding a full
                # extra 10min of wedged-allocation on top of the failed bringup
                # (job 930367 hung ~1h holding 6 nodes while its chain queued
                # behind it). 120s is ample to grab whatever logs are reachable;
                # the EXIT-trap cleanup_ray_logs runs again afterward as a belt.
                timeout=120,
                stdout=sys.stdout,
                stderr=subprocess.STDOUT,
            )
            print("[RLJobRunner] Crash-time Ray log preservation complete.", flush=True)
        except subprocess.TimeoutExpired:
            print(
                "[RLJobRunner] Crash-time Ray log preservation timed out (120s); continuing.",
                flush=True,
            )
        except Exception as e:
            print(
                f"[RLJobRunner] Crash-time Ray log preservation failed (non-fatal): {e}",
                flush=True,
            )

    def _launch_trace_upload(self, training_exit_code: int) -> Optional[subprocess.Popen]:
        """Launch post-training trace upload as a subprocess.

        Args:
            training_exit_code: Exit code from training (logged but doesn't gate upload).

        Returns:
            Popen handle if upload was launched, None if skipped.
        """
        if not self.config.trace_upload_enabled:
            print("[RLJobRunner] Trace upload disabled, skipping.", flush=True)
            return None

        trials_dir = Path(self.config.trials_dir)
        if not trials_dir.exists():
            print(f"[RLJobRunner] No trials directory found at {trials_dir}, skipping upload.", flush=True)
            return None

        repo_id = f"{self.config.trace_upload_repo_org}/{self.config.job_name}"

        # Log file for upload output
        log_dir = Path(self.config.experiments_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.config.job_name}_trace_upload.log"

        cmd = [
            sys.executable, "-m", "scripts.harbor.make_and_upload_trace_dataset",
            "--job_dir", str(trials_dir),
            "--repo_id", repo_id,
            "--episodes", self.config.trace_upload_episodes,
            "--dataset_type", self.config.trace_upload_dataset_type,
        ]

        print(f"[RLJobRunner] Launching trace upload (training exit code: {training_exit_code}):", flush=True)
        print(f"  repo_id: {repo_id}", flush=True)
        print(f"  trials_dir: {trials_dir}", flush=True)
        print(f"  episodes: {self.config.trace_upload_episodes}", flush=True)
        print(f"  log: {log_path}", flush=True)

        try:
            log_fh = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            return proc
        except Exception as e:
            print(f"[RLJobRunner] Failed to launch trace upload: {e}", flush=True)
            return None

    def _setup_environment(self) -> None:
        """Configure environment variables for RL training."""
        # Set common environment variables
        os.environ["TENSOR_PARALLEL_SIZE"] = str(self.config.tensor_parallel_size)
        os.environ["NUM_INFERENCE_ENGINES"] = str(
            compute_num_inference_engines(
                self.config.num_nodes,
                self.config.gpus_per_node,
                self.config.tensor_parallel_size,
            )
        )
        os.environ["POLICY_NUM_NODES"] = str(self.config.num_nodes)

        if self.config.export_path:
            os.environ["SKYRL_EXPORT_PATH"] = self.config.export_path

        # vLLM settings
        # RL uses its own conda env (dcagent-rl/otagent-rl) pinned to an older
        # vLLM wheel that still uses the V1 engine. The eval/datagen path
        # (hpc/vllm_utils.py) uses the newer wheel and opts into V2.
        os.environ["VLLM_USE_V1"] = "1"

        # Ensure WandB directory is writable
        from hpc.wandb_launch_utils import ensure_wandb_dir
        wandb_dir = ensure_wandb_dir(
            experiments_dir=self.config.experiments_dir,
            verbose=True,
        )
        os.environ["WANDB_DIR"] = wandb_dir

        # HuggingFace Hub settings for checkpoint uploads
        # Pass through HF_TOKEN if set (needed for hub uploads and private model access)
        hf_token = os.environ.get("HF_TOKEN")
        hf_hub_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
        if hf_token:
            # HF_TOKEN is already in environment, just log it's available
            print(f"  HF_TOKEN=****{hf_token[-4:] if len(hf_token) > 4 else '****'}", flush=True)
        if hf_hub_cache:
            os.environ["HF_HUB_CACHE"] = hf_hub_cache
            print(f"  HF_HUB_CACHE={hf_hub_cache}", flush=True)

        # Supabase credentials for database registration callback
        # KEYS should point to a file with SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
        keys_path = os.environ.get("KEYS")
        if keys_path:
            print(f"  KEYS={keys_path} (Supabase credentials for DB registration)", flush=True)
        else:
            # Also check if Supabase vars are set directly
            supabase_url = os.environ.get("SUPABASE_URL")
            if supabase_url:
                print(f"  SUPABASE_URL={supabase_url[:30]}... (direct Supabase config)", flush=True)

        print("Environment configured:", flush=True)
        print(f"  TENSOR_PARALLEL_SIZE={os.environ['TENSOR_PARALLEL_SIZE']}", flush=True)
        print(f"  NUM_INFERENCE_ENGINES={os.environ['NUM_INFERENCE_ENGINES']}", flush=True)
        print(f"  POLICY_NUM_NODES={os.environ['POLICY_NUM_NODES']}", flush=True)
        print(f"  WANDB_DIR={wandb_dir}", flush=True)

    def _run_with_ray(self) -> int:
        """Run SkyRL training with managed Ray cluster.

        Uses RayCluster.from_slurm() to properly start Ray across all SLURM nodes
        using srun, ensuring all nodes join the cluster before training begins.
        """
        from hpc.ray_utils import (
            RayCluster,
            RayClusterConfig,
            compute_ray_memory_from_slurm,
        )

        hpc = self._get_hpc()
        setattr(self.config, "proxychains_binary", getattr(hpc, "proxychains_binary", None))
        num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", self.config.num_nodes))

        # Use config values (from CLI overrides) instead of cluster defaults
        gpus_per_node = self.config.gpus_per_node or hpc.gpus_per_node
        cpus_per_node = self.config.cpus_per_node or hpc.cpus_per_node

        # Compute Ray memory limit from SLURM allocation (prevents OOM from over-detection)
        ray_memory = compute_ray_memory_from_slurm()
        if ray_memory:
            print(f"[RLJobRunner] Ray memory limit: {ray_memory / (1024**3):.1f} GB", flush=True)

        ray_cfg = RayClusterConfig(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            cpus_per_node=cpus_per_node,
            ray_port=self.config.ray_port,
            srun_export_env=hpc.get_srun_export_env(),
            ray_env_vars=hpc.get_ray_env_vars(),
            memory_per_node=ray_memory,
            object_store_memory=int(self.config.ray_object_store_gb * 1024 * 1024 * 1024),
            disable_cpu_bind=getattr(hpc, "disable_cpu_bind", False),
            gpu_bind=getattr(hpc, "gpu_bind", "none"),
            proxychains_binary=getattr(hpc, "proxychains_binary", None),
            # Apptainer RL runtime mode (OPT-IN): wrap ray start / ray.init()
            # wait scripts in `apptainer exec --nv` when a SIF is configured.
            container_sif=getattr(self.config, "container_sif", None),
            container_binds=list(getattr(self.config, "container_binds", []) or []),
            container_pythonpath=_build_container_pythonpath()
            if getattr(self.config, "container_sif", None) else "",
        )

        print(f"Starting Ray cluster with {num_nodes} nodes, {gpus_per_node} GPUs/node", flush=True)

        with RayCluster.from_slurm(ray_cfg) as ray_cluster:
            # Set RAY_ADDRESS for SkyRL to connect
            os.environ["RAY_ADDRESS"] = ray_cluster.address
            print(f"Ray cluster ready at {ray_cluster.address}", flush=True)
            print(f"Total GPUs available: {ray_cluster.total_gpus}", flush=True)

            # Enable distributed containers for multi-node local backend jobs
            # This allows Harbor to spread container workload across all Ray nodes
            local_backends = {"podman_hpc", "docker", "apptainer"}
            if ray_cluster.total_nodes > 1 and self.config.harbor_env in local_backends:
                os.environ["HARBOR_DISTRIBUTED_CONTAINERS"] = "1"
                print(f"[RLJobRunner] Enabled distributed {self.config.harbor_env} "
                      f"across {ray_cluster.total_nodes} nodes", flush=True)

            # Controller-ingress mode: replace the pinggy tunnel with a stable
            # public URL fronting the controller proxy. Only reroutes cloud
            # backends (needs_pinggy_tunnel); local backends keep direct vLLM.
            # In the default "pinggy" mode this branch is skipped entirely, so
            # the legacy path below is byte-identical.
            if self.config.ingress_mode == "controller":
                from hpc.pinggy_utils import needs_pinggy_tunnel
                from hpc.ingress_utils import (
                    capability_api_base,
                    controller_registration_plan,
                    inject_ingress_agent_key,
                    register_controller_endpoint,
                )

                if needs_pinggy_tunnel(self.config.agent_name, self.config.harbor_env):
                    if not self.config.ingress_host:
                        raise ValueError(
                            "--ingress-mode controller requires --ingress-host "
                            "(the public controller-ingress host)."
                        )
                    # Register the co-located upstream with the iris controller under
                    # ENDPOINT_ACCESS_LINK, then mint a scoped capability token and
                    # build the /proxy/t/<token>/<name>/v1 api_base. The upstream is
                    # the RecordProxy (record_literal: controller -> RecordProxy ->
                    # vLLM, so literal tokens are captured) or raw vLLM otherwise; the
                    # proxy binds 0.0.0.0 so the (remote) controller reaches it at
                    # IRIS_ADVERTISE_HOST. record_literal off = maybe_serve_literal_proxy
                    # is a null CM (no proxy) and the plan registers raw vLLM's port.
                    from hpc.literal_proxy_utils import (
                        maybe_serve_literal_proxy,
                        DEFAULT_LITERAL_PROXY_PORT,
                    )

                    endpoint_name, register_address = controller_registration_plan(
                        self.config.job_name,
                        record_literal=self.config.record_literal,
                        proxy_port=DEFAULT_LITERAL_PROXY_PORT,
                    )
                    _RL_VLLM_LOCAL = "http://localhost:8000/v1"
                    with maybe_serve_literal_proxy(
                        self.config.record_literal,
                        _RL_VLLM_LOCAL,
                        experiments_dir=self.config.experiments_dir,
                        job_name=self.config.job_name,
                        host="0.0.0.0",
                    ):
                        # The leased EndpointClient must stay alive for the whole
                        # skyrl run; _run_skyrl runs synchronously here, so close
                        # (stop renewal + unregister) after it.
                        registration = register_controller_endpoint(
                            endpoint_name, register_address
                        )
                        try:
                            # Mint + build the capability api_base AFTER register
                            # (the mint resolves the just-registered endpoint).
                            api_base = capability_api_base(
                                self.config.ingress_host, endpoint_name
                            )
                            os.environ["HARBOR_MODEL_ENDPOINT"] = api_base
                            injected = inject_ingress_agent_key()
                            print(
                                f"[RLJobRunner] ingress_mode=controller "
                                f"record_literal={self.config.record_literal}: registered "
                                f"{endpoint_name} -> {register_address} "
                                f"(id={registration.endpoint_id}, access=LINK); "
                                f"HARBOR_MODEL_ENDPOINT=/proxy/t/<token>/{endpoint_name}/v1 "
                                f"(dummy key injected={injected})",
                                flush=True,
                            )
                            return self._run_skyrl()
                        finally:
                            registration.close()
                else:
                    print(
                        "[RLJobRunner] ingress_mode=controller but local backend "
                        "(no ingress needed), using local vLLM",
                        flush=True,
                    )
                    return self._run_skyrl()

            # Check if Pinggy tunnel is needed for installed agents in cloud backends
            from hpc.pinggy_utils import (
                needs_pinggy_tunnel,
                PinggyTunnel,
                PinggyConfig,
            )

            has_url = bool(self.config.pinggy_persistent_url)
            has_token = bool(self.config.pinggy_token)
            needs_tunnel = needs_pinggy_tunnel(self.config.agent_name, self.config.harbor_env)
            use_pinggy = has_url and has_token and needs_tunnel

            print(f"[RLJobRunner] Pinggy check: url={has_url}, token={has_token}, "
                  f"needs_tunnel={needs_tunnel} (agent={self.config.agent_name}, "
                  f"env={self.config.harbor_env})", flush=True)

            # Co-locate harbor's RecordProxy in front of SkyRL's vLLM (localhost:8000)
            # when --record_literal is set, so agent completions are captured to a
            # literal.jsonl log. Default off = a null context manager that starts no
            # server and leaves the tunnel port + HARBOR_MODEL_ENDPOINT exactly as
            # today (byte-identical). When on, the tunnel fronts the proxy port and
            # the local path routes Harbor through the proxy.
            from hpc.literal_proxy_utils import (
                maybe_serve_literal_proxy,
                DEFAULT_LITERAL_PROXY_PORT,
            )

            _RL_VLLM_LOCAL = "http://localhost:8000/v1"
            with maybe_serve_literal_proxy(
                self.config.record_literal,
                _RL_VLLM_LOCAL,
                experiments_dir=self.config.experiments_dir,
                job_name=self.config.job_name,
            ) as literal_endpoint:
                if use_pinggy:
                    # SkyRL's vLLM HTTP endpoint typically runs on port 8000; when
                    # capturing literal tokens the tunnel must front the proxy port.
                    # The tunnel must be started BEFORE SkyRL so the port is available.
                    vllm_port = (
                        DEFAULT_LITERAL_PROXY_PORT if self.config.record_literal else 8000
                    )

                    pinggy_cfg = PinggyConfig(
                        persistent_url=self.config.pinggy_persistent_url,
                        token=self.config.pinggy_token,
                        local_port=vllm_port,
                        local_host="localhost",
                    )

                    log_dir = Path(self.config.experiments_dir) / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    pinggy_log = log_dir / f"{self.config.job_name}_pinggy.log"

                    print(f"[RLJobRunner] Starting Pinggy tunnel: localhost:{vllm_port} -> "
                          f"{self.config.pinggy_persistent_url}", flush=True)

                    with PinggyTunnel(pinggy_cfg, log_path=pinggy_log) as tunnel:
                        # Set environment variable for SkyRL/Harbor to use public endpoint
                        # Terminal bench reads this to configure the hosted_vllm backend
                        os.environ["HARBOR_MODEL_ENDPOINT"] = tunnel.public_endpoint
                        print(f"[RLJobRunner] HARBOR_MODEL_ENDPOINT={tunnel.public_endpoint}", flush=True)
                        return self._run_skyrl()
                else:
                    if self.config.record_literal:
                        # Route Harbor through the co-located proxy instead of vLLM directly.
                        os.environ["HARBOR_MODEL_ENDPOINT"] = literal_endpoint
                        print(f"[RLJobRunner] record_literal: HARBOR_MODEL_ENDPOINT={literal_endpoint}", flush=True)
                        return self._run_skyrl()
                    print("[RLJobRunner] No Pinggy tunnel needed, using local vLLM", flush=True)
                    return self._run_skyrl()

    def _run_skyrl(self) -> int:
        """Execute SkyRL training.

        Returns:
            Exit code from SkyRL process.
        """
        # Build command. In the default (host venv/conda) path we use
        # sys.executable so we run the same Python as the current process
        # (which the sbatch activated). In Apptainer mode the Python comes
        # from inside the SIF, so we use a bare "python" prefixed with the
        # apptainer-exec list.
        container_sif = getattr(self.config, "container_sif", None)
        if container_sif:
            apptainer_prefix = build_apptainer_prefix(
                container_sif,
                binds=self.config.container_binds or None,
                pythonpath=_build_container_pythonpath(),
            )
            cmd = apptainer_prefix + ["python", "-m", self.config.skyrl_entrypoint]
        else:
            cmd = [sys.executable, "-m", self.config.skyrl_entrypoint]
        cmd.extend(self.config.skyrl_hydra_args)

        print("\nRunning SkyRL:", flush=True)
        if container_sif:
            print(f"  Runtime: Apptainer SIF {container_sif}", flush=True)
        else:
            print(f"  Python: {sys.executable}", flush=True)
        print(f"  Entrypoint: {self.config.skyrl_entrypoint}", flush=True)
        print(f"  Args: {len(self.config.skyrl_hydra_args)} Hydra arguments", flush=True)

        # Change to SKYRL_HOME if set (resolved with {SkyRL, MarinSkyRL}
        # dir-name hardening; honors SKYRL_HOME / RL_REPO_DIR override first).
        skyrl_home = _resolve_skyrl_home()
        cwd = None
        if skyrl_home:
            cwd = os.path.join(skyrl_home, "skyrl-train")
            if os.path.isdir(cwd):
                print(f"  Working dir: {cwd}", flush=True)
            else:
                cwd = None

        # Proxychains stays OUTSIDE the apptainer exec: egress is handled at the
        # host layer (proxychains4 -f <conf> apptainer exec ... python ...), so
        # the container needn't know about proxychains. See design doc §5.
        if self.config.proxychains_binary:
            print(f"Using proxychains binary: {self.config.proxychains_binary}", flush=True)
            cmd = [f'{self.config.proxychains_binary}', '-f', "$PROXYCHAINS_CONF_FILE"] + cmd

        srun_cmd = cmd

        print(f"\nExecuting command with srun: {' '.join(srun_cmd)}", flush=True)

        result = subprocess.run(srun_cmd, cwd=cwd)
        return result.returncode


def run_rl_job_main():
    """Entry point for running RL jobs from sbatch.

    This is invoked by the sbatch script via:
        python -m hpc.rl_launch_utils --config /path/to/config.json
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run RL training job")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config_dict = json.load(f)

    config = RLJobConfig(**config_dict)
    runner = RLJobRunner(config)
    sys.exit(runner.run())


if __name__ == "__main__":
    run_rl_job_main()
