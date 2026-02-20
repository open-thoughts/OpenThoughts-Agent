#!/usr/bin/env python3
"""
Unified Eval Listener - Consolidates all eval listeners and sbatch scripts.

This script replaces:
  - aider_eval_listener.py, bfcl_eval_listener.py, swebench_eval_listener.py
  - v2_eval_listener.py, v2_eval_listener_prio.py, tb2_eval_listener.py
  - dev_eval_listener.py
  - Uses unified_eval_harbor.sbatch (replaces all individual sbatch scripts)

Features:
  - Preset configurations for each benchmark (aider, bfcl, swebench, v2, tb2)
  - Priority file hot-reload (changes take effect without restart)
  - Configurable sbatch parameters (n-concurrent, daytona-threshold, etc.)

Usage Examples:
  # Use a preset (replaces individual listener scripts)
  python unified_eval_listener.py --preset bfcl
  python unified_eval_listener.py --preset swebench
  python unified_eval_listener.py --preset v2 --priority-file priority_models.txt

  # Custom configuration with sbatch params
  python unified_eval_listener.py \\
    --datasets "DCAgent/dev_set_v2" \\
    --n-concurrent 128 \\
    --daytona-threshold 5

  # Override preset concurrency
  python unified_eval_listener.py --preset v2 --n-concurrent 64

  # Dry run to preview
  python unified_eval_listener.py --preset v2 --dry-run --once

Environment Variables (all optional, CLI args take precedence):
  EVAL_LISTENER_LOOKBACK_DAYS       Days to look back for models (default: 100)
  EVAL_LISTENER_CHECK_HOURS         Hours between iterations (default: 4.0)
  EVAL_LISTENER_SBATCH              SBATCH script to use
  EVAL_LISTENER_LOG_DIR             Log directory (default: experiments/listener_logs)
  EVAL_LISTENER_DATASETS            Comma/space/newline list of HF dataset repos
  EVAL_LISTENER_PRIORITY_FILE       Path to priority models file (hot-reloaded)
  EVAL_LISTENER_DRY_RUN             "1" or "true" to enable dry run mode
  EVAL_LISTENER_REQUIRE_PRIORITY_LIST  "1" or "true" to require priority list
  EVAL_LISTENER_CHECK_HF_EXISTS     "1" or "true" to validate HF model existence
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add leaderboard utilities to path
sys.path.insert(0, "/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard")

from unified_db.utils import get_supabase_client

# ---------- Preset Definitions ----------
# Each preset can configure:
#   - datasets: list of HF dataset repos
#   - sbatch_script: sbatch script to use (default: unified_eval_harbor.sbatch)
#   - log_suffix: suffix for log file
#   - check_hf_exists: validate model exists on HuggingFace
#   - n_concurrent: Harbor --n-concurrent (default: 64)
#   - n_attempts: Harbor --n-attempts (default: 3)
#   - gpu_memory_util: VLLM --gpu-memory-utilization (default: 0.9)
#   - daytona_threshold: Max DaytonaErrors before abort (default: 3)
#   - vllm_max_retries: VLLM startup retries (default: 5)
#   - agent_parser: Agent parser type (default: "", use "xml" for swebench)
#   - slurm_time: SLURM time limit (default: "24:00:00")
PRESETS: Dict[str, Dict] = {
    "aider": {
        "datasets": ["DCAgent2/aider_polyglot"],
        "log_suffix": "aider",
        "n_concurrent": 32,
        "daytona_threshold": 3,
    },
    "bfcl": {
        "datasets": ["DCAgent/dev_set_v2"],
        "log_suffix": "v2",
        "n_concurrent": 32,
        "daytona_threshold": 10,
        "vllm_max_retries": 20,
        "enable_thinking": True,
    },
    "swebench": {
        "datasets": ["DCAgent2/swebench-verified-random-100-folders"],
        "check_hf_exists": True,
        "log_suffix": "swebench",
        "n_concurrent": 64,
        "daytona_threshold": 15,
        "agent_parser": "xml",
        "gpu_memory_util": 0.95,
        "vllm_max_retries": 20,
    },
    "v2": {
        "datasets": ["DCAgent/dev_set_v2"],
        "log_suffix": "v2",
        "n_concurrent": 32,
        "daytona_threshold": 10,
        "vllm_max_retries": 20,
        "enable_thinking": True,
    },
    "tb2": {
        "datasets": ["DCAgent2/terminal_bench_2"],
        "log_suffix": "tb2",
        "n_concurrent": 32,
        "daytona_threshold": 10,
        "slurm_time": "48:00:00",
        "gpu_memory_util": 0.95,
    },
    "dev": {
        "datasets": [],  # Must provide via args/env
        "log_suffix": "dev",
    },
}

# ---------- Constants ----------
HF_URL_RE = re.compile(r'https?://(?:www\.)?huggingface\.co/([^/\s]+)/([^/\s#?]+)')
JOB_STATUS_PENDING = "Pending"
JOB_STATUS_STARTED = "Started"
JOB_STATUS_FINISHED = "Finished"
DEFAULT_STALE_JOB_HOURS = 24
DEFAULT_STALE_PENDING_HOURS = 168
DEFAULT_LOOKBACK_DAYS = 100
DEFAULT_CHECK_HOURS = 12.0
DEFAULT_LOG_DIR = "experiments/listener_logs"

# Sbatch parameter defaults
DEFAULT_N_CONCURRENT = 64
DEFAULT_N_ATTEMPTS = 3
DEFAULT_GPU_MEMORY_UTIL = 0.9
DEFAULT_DAYTONA_THRESHOLD = 3
DEFAULT_VLLM_MAX_RETRIES = 5
DEFAULT_AGENT_PARSER = ""
DEFAULT_SLURM_TIME = "24:00:00"
DEFAULT_AGENT_NAME = "terminus-2"
DEFAULT_SLURM_PARTITION = "gh"
DEFAULT_ENABLE_THINKING = False
DEFAULT_SBATCH_SCRIPT = "unified_eval_harbor.sbatch"


# ---------- Configuration ----------
@dataclass
class ListenerConfig:
    """Configuration for the eval listener."""
    datasets: List[str]
    sbatch_script: str
    log_file: Optional[Path]
    lookback_days: int
    check_interval_hours: float
    stale_job_hours: int
    stale_pending_hours: int
    priority_file: Optional[str]
    require_priority_list: bool
    priority_models: Set[str]
    check_hf_exists: bool
    dry_run: bool
    run_once: bool
    verbose: bool
    # Sbatch parameters (passed to sbatch via env vars)
    n_concurrent: int = DEFAULT_N_CONCURRENT
    n_attempts: int = DEFAULT_N_ATTEMPTS
    gpu_memory_util: float = DEFAULT_GPU_MEMORY_UTIL
    daytona_threshold: int = DEFAULT_DAYTONA_THRESHOLD
    vllm_max_retries: int = DEFAULT_VLLM_MAX_RETRIES
    agent_parser: str = DEFAULT_AGENT_PARSER
    slurm_time: str = DEFAULT_SLURM_TIME
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    agent_name: str = DEFAULT_AGENT_NAME
    slurm_partition: str = DEFAULT_SLURM_PARTITION
    upload_username: str = ""
    log_prefix: str = "[unified-eval-listener]"

    @property
    def check_interval_seconds(self) -> int:
        return int(self.check_interval_hours * 60 * 60)


# ---------- Logging ----------
_LOG_FILE: Optional[Path] = None


def set_log_file(path: Optional[Path]) -> None:
    global _LOG_FILE
    _LOG_FILE = path


def log(msg: str, prefix: str = "[unified-eval-listener]") -> None:
    """Log a message to stdout and optionally to file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{prefix} {ts}  {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        try:
            with _LOG_FILE.open("a") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ---------- Priority Models Loading ----------
def load_priority_models(filepath: Optional[str]) -> Set[str]:
    """
    Load priority models from a text file.

    File format:
      - One model per line (HuggingFace format: org/model)
      - Lines starting with # are comments
      - Blank lines are ignored

    Returns:
        Set of model names (exact match). Empty set if file is missing or empty.
    """
    if not filepath:
        return set()

    path = Path(filepath)
    if not path.exists():
        log(f"Priority file not found: {filepath}")
        return set()

    models: Set[str] = set()
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                models.add(line)
        log(f"Loaded {len(models)} model(s) from priority file: {filepath}")
        return models
    except Exception as e:
        log(f"ERROR reading priority file {filepath}: {e}")
        return set()


# ---------- HuggingFace Utilities ----------
def check_hf_model_exists(model_name: str) -> bool:
    """
    Check if a model exists on HuggingFace Hub.

    Args:
        model_name: HF model name (e.g., "org/model-name")

    Returns:
        True if model exists and is accessible, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        return False

    try:
        from huggingface_hub import model_info
        model_info(model_name)
        return True
    except Exception as e:
        log(f"HF check failed for {model_name}: {e}")
        return False


def _parse_hf_from_str(val: Optional[str]) -> Optional[str]:
    """Parse HuggingFace model name from a string (URL or org/repo)."""
    if not isinstance(val, str):
        return None
    m = HF_URL_RE.search(val)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def resolve_hf_model_name(model_row: Dict) -> Optional[str]:
    """
    Resolve HF model name from a database model row.

    Checks multiple fields in order of priority.
    """
    # Check name field first
    v = model_row.get("name")
    if isinstance(v, str) and "/" in v and not v.startswith("hosted_vllm/"):
        return v

    # Check other URL fields
    for field in ("weights_location", "training_parameters", "url", "hf_url"):
        vv = model_row.get(field)
        if isinstance(vv, str):
            name = _parse_hf_from_str(vv)
            if name:
                return name

    # Check training_parameters as JSON
    vv = model_row.get("training_parameters")
    if isinstance(vv, str):
        try:
            obj = json.loads(vv)
        except Exception:
            obj = None
    else:
        obj = vv

    if isinstance(obj, dict):
        for sval in obj.values():
            if isinstance(sval, str):
                name = _parse_hf_from_str(sval)
                if name:
                    return name

    return None


# ---------- Dataset Parsing ----------
def parse_datasets(s: str) -> List[str]:
    """
    Parse dataset list from string.

    Supports comma, space, or newline separated values.
    Normalizes HF URLs to org/repo format.
    """
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    out = []
    for p in parts:
        m = HF_URL_RE.search(p)
        out.append(f"{m.group(1)}/{m.group(2)}" if m else p)

    # Dedup while preserving order
    seen: Set[str] = set()
    uniq: List[str] = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def dataset_repo_name(dataset_hf: str) -> str:
    """Convert 'org/repo' or HF URL to 'repo' (just the repo name)."""
    if not dataset_hf:
        return dataset_hf
    m = HF_URL_RE.search(dataset_hf)
    if m:
        return m.group(2)
    if "/" in dataset_hf:
        return dataset_hf.rsplit("/", 1)[-1]
    return dataset_hf


# ---------- Database Operations ----------
_BENCH_CACHE: Dict[str, Optional[str]] = {}


def _iso(dt: datetime) -> str:
    """Convert datetime to ISO format string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _time_filters(q, since_iso: str):
    """Apply time filter to Supabase query (handles both column names)."""
    try:
        return q.gte('creation_time', since_iso)
    except Exception:
        return q.gte('created_at', since_iso)


def fetch_recent_models(days: int) -> List[Dict]:
    """Fetch recent models from Supabase within the lookback window."""
    client = get_supabase_client()
    since = _iso(datetime.now(timezone.utc) - timedelta(days=days))
    try:
        resp = _time_filters(client.table('models').select('*'), since).execute()
        rows = list(resp.data or [])
    except Exception as e:
        log(f"ERROR: failed querying models by time: {e}")
        return []

    # Filter out precomputed models
    out: List[Dict] = []
    for r in rows:
        if r.get("created_by") == "precomputed_hf":
            continue
        out.append(r)
    return out


def resolve_benchmark_id(dataset_hf: str) -> Optional[str]:
    """
    Look up benchmark ID from database for a given dataset.

    Caches results for performance.
    """
    repo_name = dataset_repo_name(dataset_hf)
    if repo_name in _BENCH_CACHE:
        return _BENCH_CACHE[repo_name]

    try:
        client = get_supabase_client()
        resp = (
            client.table('benchmarks')
            .select('id,name')
            .eq('name', repo_name)
            .limit(1)
            .execute()
        )
        rows = resp.data or []
        bench_id = rows[0]['id'] if rows else None
        _BENCH_CACHE[repo_name] = bench_id
        if not bench_id:
            log(f"No benchmark row found for dataset '{dataset_hf}' (wanted name='{repo_name}').")
        return bench_id
    except Exception as e:
        log(f"ERROR resolving benchmark id for dataset '{dataset_hf}': {e}")
        return None


def check_job_status(
    model_id: str, benchmark_id: Optional[str]
) -> Tuple[bool, Optional[str], Optional[datetime], Optional[datetime], Optional[str]]:
    """
    Check if a job exists for (model_id, benchmark_id) and its status.

    Returns:
        (job_exists, job_status, started_at, submitted_at, slurm_job_id)
    """
    if not benchmark_id:
        return (False, None, None, None, None)

    try:
        client = get_supabase_client()
        q = (
            client.table('sandbox_jobs')
            .select('id,job_status,started_at,submitted_at,slurm_job_id')
            .eq('model_id', model_id)
            .eq('benchmark_id', benchmark_id)
            .order('created_at', desc=True)
            .limit(1)
        )
        data = (q.execute().data) or []

        if not data:
            return (False, None, None, None, None)

        job = data[0]
        job_status = job.get('job_status')
        started_at_str = job.get('started_at')
        submitted_at_str = job.get('submitted_at')
        slurm_job_id = job.get('slurm_job_id')

        started_at = None
        if started_at_str:
            try:
                started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
            except Exception:
                pass

        submitted_at = None
        if submitted_at_str:
            try:
                submitted_at = datetime.fromisoformat(submitted_at_str.replace('Z', '+00:00'))
            except Exception:
                pass

        return (True, job_status, started_at, submitted_at, slurm_job_id)

    except Exception as e:
        log(f"WARNING: sandbox_jobs check failed for model_id={model_id}, benchmark_id={benchmark_id}: {e}")
        return (False, None, None, None, None)  # fail-open


def is_job_stale(started_at: Optional[datetime], hours: int = DEFAULT_STALE_JOB_HOURS) -> bool:
    """Check if a job started more than the specified hours ago."""
    if not started_at:
        # If started_at is null but job exists with status='Started', treat as stale
        return True
    now = datetime.now(timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    age = now - started_at
    return age > timedelta(hours=hours)


def should_start_job(
    model_id: str,
    benchmark_id: Optional[str],
    stale_hours: int = DEFAULT_STALE_JOB_HOURS,
    stale_pending_hours: int = DEFAULT_STALE_PENDING_HOURS,
) -> Tuple[bool, str, Optional[str]]:
    """
    Determine if a job should be started based on DB status.

    Returns:
        (should_start, reason, slurm_job_id)
        slurm_job_id is provided so the caller can scancel stale jobs.
    """
    job_exists, job_status, started_at, submitted_at, slurm_job_id = check_job_status(
        model_id, benchmark_id
    )

    if not job_exists:
        return (True, "no existing job", None)

    if job_status == JOB_STATUS_FINISHED:
        return (False, "job finished", slurm_job_id)

    if job_status == JOB_STATUS_PENDING:
        # Job submitted but not yet running - check if stale using separate pending threshold
        if is_job_stale(submitted_at, stale_pending_hours):
            submitted_str = submitted_at.isoformat() if submitted_at else "null"
            return (True, f"stale pending job (submitted_at={submitted_str})", slurm_job_id)
        else:
            submitted_str = submitted_at.isoformat() if submitted_at else "null"
            return (False, f"job pending in SLURM queue (submitted_at={submitted_str})", slurm_job_id)

    if job_status == JOB_STATUS_STARTED:
        if is_job_stale(started_at, stale_hours):
            started_str = started_at.isoformat() if started_at else "null"
            return (True, f"stale job (started_at={started_str})", slurm_job_id)
        else:
            started_str = started_at.isoformat() if started_at else "null"
            return (False, f"job in progress (started_at={started_str})", slurm_job_id)

    # Unknown status - start job to be safe
    return (True, f"unknown job status: {job_status}", slurm_job_id)


# ---------- Job Submission ----------
@dataclass
class SbatchParams:
    """Parameters passed to sbatch via environment variables."""
    n_concurrent: int = DEFAULT_N_CONCURRENT
    n_attempts: int = DEFAULT_N_ATTEMPTS
    gpu_memory_util: float = DEFAULT_GPU_MEMORY_UTIL
    daytona_threshold: int = DEFAULT_DAYTONA_THRESHOLD
    vllm_max_retries: int = DEFAULT_VLLM_MAX_RETRIES
    agent_parser: str = DEFAULT_AGENT_PARSER
    slurm_time: str = DEFAULT_SLURM_TIME
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    agent_name: str = DEFAULT_AGENT_NAME
    slurm_partition: str = DEFAULT_SLURM_PARTITION
    upload_username: str = ""

    def to_env(self) -> Dict[str, str]:
        """Convert to environment variables for sbatch."""
        env = {
            "EVAL_N_CONCURRENT": str(self.n_concurrent),
            "EVAL_N_ATTEMPTS": str(self.n_attempts),
            "EVAL_GPU_MEMORY_UTIL": str(self.gpu_memory_util),
            "EVAL_DAYTONA_THRESHOLD": str(self.daytona_threshold),
            "EVAL_VLLM_MAX_RETRIES": str(self.vllm_max_retries),
            "EVAL_AGENT_PARSER": self.agent_parser,
            "EVAL_SLURM_TIME": self.slurm_time,
            "EVAL_ENABLE_THINKING": "true" if self.enable_thinking else "false",
            "EVAL_AGENT_NAME": self.agent_name,
        }
        if self.upload_username:
            env["EVAL_UPLOAD_USERNAME"] = self.upload_username
        return env

    def __str__(self) -> str:
        """String representation for logging."""
        parts = [
            f"n_concurrent={self.n_concurrent}",
            f"n_attempts={self.n_attempts}",
            f"gpu_memory_util={self.gpu_memory_util}",
            f"daytona_threshold={self.daytona_threshold}",
            f"vllm_max_retries={self.vllm_max_retries}",
        ]
        if self.agent_parser:
            parts.append(f"agent_parser={self.agent_parser}")
        if self.slurm_time != DEFAULT_SLURM_TIME:
            parts.append(f"slurm_time={self.slurm_time}")
        if self.enable_thinking:
            parts.append("enable_thinking=True")
        if self.agent_name != DEFAULT_AGENT_NAME:
            parts.append(f"agent_name={self.agent_name}")
        if self.slurm_partition != DEFAULT_SLURM_PARTITION:
            parts.append(f"slurm_partition={self.slurm_partition}")
        if self.upload_username:
            parts.append(f"upload_username={self.upload_username}")
        return ", ".join(parts)


def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    """Run a command and return exit code and output."""
    # Merge with current environment if extra env vars provided
    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=run_env
    )
    out_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line.rstrip())
    code = proc.wait()
    return code, "\n".join(out_lines)


def generate_run_tag(dataset_hf: str, model_hf: str) -> str:
    """
    Generate a unique RUN_TAG for the job.

    Format: {safe_repo}_{safe_model}_{timestamp}
    """
    safe_repo = dataset_repo_name(dataset_hf).replace("-", "_").replace(".", "_")
    safe_model = model_hf.split("/")[-1].replace("-", "_").replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe_repo}_{safe_model}_{timestamp}"


def cancel_slurm_job(slurm_job_id: str, dry_run: bool = False) -> bool:
    """Cancel a SLURM job via scancel. Returns True if successful."""
    if dry_run:
        log(f"[DRY RUN] Would cancel SLURM job {slurm_job_id}")
        return True
    code, out = _run(["scancel", slurm_job_id])
    if code == 0:
        log(f"Cancelled SLURM job {slurm_job_id}")
        return True
    else:
        log(f"WARNING: scancel failed for job {slurm_job_id}: {out}")
        return False


def submit_eval(
    hf_model_name: str,
    dataset_hf: str,
    benchmark_id: Optional[str],
    sbatch_script: str,
    sbatch_params: Optional[SbatchParams] = None,
    dry_run: bool = False,
    upload_username: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Submit sbatch job and create a Pending DB entry.

    sbatch args:
      $1 = model HF name
      $2 = dataset HF repo (org/repo)
      $3 = benchmark_id (uuid)  [optional]
      $4 = job_name (RUN_TAG)

    Environment variables (from sbatch_params):
      EVAL_N_CONCURRENT, EVAL_N_ATTEMPTS, EVAL_GPU_MEMORY_UTIL,
      EVAL_DAYTONA_THRESHOLD, EVAL_VLLM_MAX_RETRIES, EVAL_AGENT_PARSER,
      EVAL_SLURM_TIME, EVAL_AGENT_NAME

    Returns:
        (slurm_job_id, job_name) if successful, ("DRY_RUN", job_name) if dry run, (None, None) on failure
    """
    # Generate unique job name
    job_name = generate_run_tag(dataset_hf, hf_model_name)

    cmd = ["sbatch"]
    if sbatch_params:
        cmd.extend(["--time", sbatch_params.slurm_time])
        cmd.extend(["--partition", sbatch_params.slurm_partition])
    cmd.append(sbatch_script)
    cmd.extend([hf_model_name, dataset_hf])
    if benchmark_id:
        cmd.append(str(benchmark_id))
    cmd.append(job_name)  # 4th arg: job_name (RUN_TAG)

    # Get env vars from params
    env_vars = sbatch_params.to_env() if sbatch_params else {}

    if dry_run:
        log(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        if sbatch_params:
            log(f"[DRY RUN] With params: {sbatch_params}")
        return ("DRY_RUN", job_name)

    code, out = _run(cmd, env=env_vars)
    log(f"sbatch: {' '.join(cmd)}\n{out}")

    if code != 0:
        return (None, None)

    m = re.search(r"Submitted batch job (\d+)", out)
    slurm_job_id = m.group(1) if m else None

    if not slurm_job_id:
        log("ERROR: Could not parse SLURM job ID from sbatch output")
        return (None, None)

    # Create Pending DB entry
    try:
        from unified_db.utils import create_job_entry_pending
        agent = sbatch_params.agent_name if sbatch_params else DEFAULT_AGENT_NAME
        result = create_job_entry_pending(
            job_name=job_name,
            model_hf=hf_model_name,
            benchmark_hf=dataset_hf,
            agent_name=agent,
            slurm_job_id=slurm_job_id,
            username=upload_username or "listener",
            config={"agent": agent, "env": "daytona"},
        )
        if result.get("success"):
            log(f"Created Pending DB entry for job {job_name}")
        else:
            log(f"WARNING: Failed to create Pending DB entry: {result.get('error')}")
    except Exception as e:
        log(f"WARNING: Exception creating Pending DB entry: {e}")

    return (slurm_job_id, job_name)


# ---------- Main Listener Class ----------
class EvalListener:
    """Unified eval listener that handles all benchmark configurations."""

    def __init__(self, config: ListenerConfig):
        self.config = config
        set_log_file(config.log_file)

    def run_iteration(self) -> int:
        """
        Run one check iteration.

        Returns:
            Number of jobs submitted (or would submit in dry-run mode)
        """
        # Hot-reload priority models from file (enables editing during long runs)
        if self.config.priority_file:
            new_priority = load_priority_models(self.config.priority_file)
            if new_priority != self.config.priority_models:
                log(f"Priority list reloaded: {len(new_priority)} model(s)")
                self.config.priority_models = new_priority

        log("Checking for new models...")
        models = fetch_recent_models(self.config.lookback_days)
        log(f"Found {len(models)} model(s) in window. Filtering...")

        # Check if we should skip all models due to require_priority_list
        if not self.config.priority_models and self.config.require_priority_list:
            log("No priority list configured and --require-priority-list is set. Skipping all models.")
            return 0

        submissions: List[Tuple[str, str, str, Optional[str], str, Optional[str]]] = []
        # (model_id, hf_model_name, dataset_hf, benchmark_id, reason, slurm_job_id)

        # Track stats
        skipped_not_in_priority = 0
        skipped_hf_not_exists = 0

        # Resolve all benchmarks up front (once per loop)
        dataset_to_bench: Dict[str, Optional[str]] = {
            ds: resolve_benchmark_id(ds) for ds in self.config.datasets
        }

        for m in models:
            model_id = str(m.get("id"))
            if not model_id:
                continue

            hf_model = resolve_hf_model_name(m)
            if not hf_model:
                if self.config.verbose:
                    log(f"Skip: cannot resolve HF model for id={model_id}, name={m.get('name')}")
                continue

            # Priority filtering (exact match)
            if self.config.priority_models and hf_model not in self.config.priority_models:
                skipped_not_in_priority += 1
                continue

            # HuggingFace existence check
            if self.config.check_hf_exists:
                if not check_hf_model_exists(hf_model):
                    log(f"Skip: model not found on HuggingFace: {hf_model} (model_id={model_id})")
                    skipped_hf_not_exists += 1
                    continue

            for dataset_hf in self.config.datasets:
                bench_id = dataset_to_bench.get(dataset_hf)

                # Check DB status to decide if we should start
                should_start, reason, old_slurm_job_id = should_start_job(
                    model_id, bench_id, self.config.stale_job_hours,
                    stale_pending_hours=self.config.stale_pending_hours,
                )

                if should_start:
                    submissions.append((model_id, hf_model, dataset_hf, bench_id, reason, old_slurm_job_id))
                elif self.config.verbose:
                    log(f"Skip: model={hf_model}, dataset={dataset_hf}, reason={reason}")

        # Log filtering stats
        if self.config.priority_models and skipped_not_in_priority > 0:
            log(f"Skipped {skipped_not_in_priority} model(s) not in priority list")
        if self.config.check_hf_exists and skipped_hf_not_exists > 0:
            log(f"Skipped {skipped_hf_not_exists} model(s) not found on HuggingFace")

        if not submissions:
            log("No eligible (model, dataset) pairs to submit.")
            return 0

        prefix = "[DRY RUN] Would submit" if self.config.dry_run else "Submitting"
        log(f"{prefix} {len(submissions)} eval(s)...")

        # Create sbatch params from config
        sbatch_params = SbatchParams(
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            gpu_memory_util=self.config.gpu_memory_util,
            daytona_threshold=self.config.daytona_threshold,
            vllm_max_retries=self.config.vllm_max_retries,
            agent_parser=self.config.agent_parser,
            slurm_time=self.config.slurm_time,
            enable_thinking=self.config.enable_thinking,
            agent_name=self.config.agent_name,
            slurm_partition=self.config.slurm_partition,
            upload_username=self.config.upload_username,
        )

        submitted = 0
        for mid, hf_model, dataset_hf, bench_id, reason, old_slurm_job_id in submissions:
            dry_prefix = "[DRY RUN] " if self.config.dry_run else ""
            log(f"{dry_prefix}Submitting: model={hf_model}, dataset={dataset_hf}, reason={reason}")

            # Cancel stale Pending SLURM job before resubmission
            if reason.startswith("stale pending") and old_slurm_job_id:
                cancel_slurm_job(old_slurm_job_id, dry_run=self.config.dry_run)

            slurm_job_id, job_name = submit_eval(
                hf_model,
                dataset_hf,
                bench_id,
                self.config.sbatch_script,
                sbatch_params=sbatch_params,
                dry_run=self.config.dry_run,
                upload_username=self.config.upload_username,
            )

            if slurm_job_id:
                if self.config.dry_run:
                    log(f"  -> Would submit as SLURM job (job_name={job_name})")
                else:
                    log(f"  -> Submitted as SLURM job {slurm_job_id} (job_name={job_name})")
                submitted += 1
            else:
                log(f"  -> Submission failed")

            if not self.config.dry_run:
                time.sleep(1)

        return submitted

    def run(self) -> None:
        """Main event loop."""
        # Log configuration
        hdr = (
            f"lookback={self.config.lookback_days}d, "
            f"every {self.config.check_interval_hours}h, "
            f"sbatch={self.config.sbatch_script}"
        )
        log(f"Starting listener for datasets={self.config.datasets}: {hdr}")
        log(
            f"Job logic: restart if 'Started' and started_at > {self.config.stale_job_hours}h ago, "
            f"restart+scancel if 'Pending' and submitted_at > {self.config.stale_pending_hours}h ago, "
            f"skip if 'Finished'"
        )
        log(f"Dry run mode: {self.config.dry_run}")
        log(f"Run once mode: {self.config.run_once}")
        log(f"Check HF exists: {self.config.check_hf_exists}")
        log(f"Require priority list: {self.config.require_priority_list}")

        if self.config.priority_models:
            log(f"Priority filtering: {len(self.config.priority_models)} model(s) in list")
            if self.config.priority_file:
                log(f"Priority file: {self.config.priority_file} (hot-reloaded each iteration)")
            if self.config.verbose:
                for m in sorted(self.config.priority_models):
                    log(f"  - {m}")
        else:
            log("Priority filtering: disabled (no priority file or empty)")

        # Log sbatch parameters
        sbatch_params = SbatchParams(
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            gpu_memory_util=self.config.gpu_memory_util,
            daytona_threshold=self.config.daytona_threshold,
            vllm_max_retries=self.config.vllm_max_retries,
            agent_parser=self.config.agent_parser,
            slurm_time=self.config.slurm_time,
            enable_thinking=self.config.enable_thinking,
            agent_name=self.config.agent_name,
            slurm_partition=self.config.slurm_partition,
        )
        log(f"Sbatch params: {sbatch_params}")

        while True:
            try:
                self.run_iteration()

                # Exit after one iteration if requested
                if self.config.run_once or self.config.dry_run:
                    mode = "DRY RUN" if self.config.dry_run else "ONCE"
                    log(f"[{mode}] Complete. Exiting after one iteration.")
                    break

                hours = self.config.check_interval_hours
                log(f"Sleeping for {hours} hours...\n")
                time.sleep(self.config.check_interval_seconds)

            except KeyboardInterrupt:
                log("Interrupted by user. Exiting.")
                sys.exit(0)
            except Exception as e:
                log(f"ERROR in main loop: {e}. Backing off 30s.")
                time.sleep(30)


# ---------- CLI Argument Parsing ----------
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified Eval Listener - Run models on benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets available: aider, bfcl, swebench, v2, tb2, dev

Examples:
  # Replace bfcl_eval_listener.py
  python unified_eval_listener.py --preset bfcl

  # Replace v2_eval_listener_prio.py with priority filtering
  python unified_eval_listener.py --preset v2 \\
    --priority-file priority_models.txt

  # Replace swebench_eval_listener.py (HF checking enabled by preset)
  python unified_eval_listener.py --preset swebench

  # Custom: multiple datasets + custom options
  python unified_eval_listener.py \\
    --datasets "DCAgent/dev_set_v2,DCAgent2/bfcl-parity" \\
    --sbatch-script custom_eval.sbatch \\
    --check-hf-exists

  # Dry run to preview
  python unified_eval_listener.py --preset v2 --dry-run

  # Single iteration mode
  python unified_eval_listener.py --preset bfcl --once
        """,
    )

    # Preset configuration
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration (aider, bfcl, swebench, v2, tb2, dev)",
    )

    # Dataset configuration
    parser.add_argument(
        "--datasets", "-d",
        help="Comma/space separated HF dataset repos (overrides preset)",
    )
    parser.add_argument(
        "--sbatch-script", "-s",
        help="SBATCH script to use (overrides preset)",
    )
    parser.add_argument(
        "--log-file",
        help="Log file path (default: auto-generated based on preset)",
    )

    # Timing configuration
    parser.add_argument(
        "--lookback-days",
        type=int,
        help=f"Days to look back for models (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--check-hours",
        type=float,
        help=f"Hours between iterations (default: {DEFAULT_CHECK_HOURS})",
    )
    parser.add_argument(
        "--stale-hours",
        type=int,
        help=f"Hours before 'Started' job is stale (default: {DEFAULT_STALE_JOB_HOURS})",
    )
    parser.add_argument(
        "--stale-pending-hours",
        type=int,
        help=f"Hours before 'Pending' job is stale (default: {DEFAULT_STALE_PENDING_HOURS})",
    )

    # Priority filtering
    parser.add_argument(
        "--priority-file",
        help="Path to priority models file (one model per line)",
    )
    parser.add_argument(
        "--require-priority-list",
        action="store_true",
        help="Skip all models when priority list is empty/missing",
    )

    # Validation options
    parser.add_argument(
        "--check-hf-exists",
        action="store_true",
        help="Validate model exists on HuggingFace before submit",
    )

    # Eval parameters (passed to sbatch via env vars)
    parser.add_argument(
        "--n-concurrent",
        type=int,
        help=f"Harbor concurrent jobs (default: {DEFAULT_N_CONCURRENT}, preset overrides)",
    )
    parser.add_argument(
        "--n-attempts",
        type=int,
        help=f"Retry attempts per task (default: {DEFAULT_N_ATTEMPTS})",
    )
    parser.add_argument(
        "--gpu-memory-util",
        type=float,
        help=f"VLLM GPU memory fraction (default: {DEFAULT_GPU_MEMORY_UTIL})",
    )
    parser.add_argument(
        "--daytona-threshold",
        type=int,
        help=f"Max DaytonaErrors before abort (default: {DEFAULT_DAYTONA_THRESHOLD})",
    )
    parser.add_argument(
        "--vllm-max-retries",
        type=int,
        help=f"VLLM startup retries (default: {DEFAULT_VLLM_MAX_RETRIES})",
    )
    parser.add_argument(
        "--agent-parser",
        help=f"Agent parser type (default: \"{DEFAULT_AGENT_PARSER}\", use \"xml\" for swebench)",
    )
    parser.add_argument(
        "--slurm-time",
        help=f"SLURM time limit (default: \"{DEFAULT_SLURM_TIME}\")",
    )
    parser.add_argument(
        "--agent-name",
        help=f"Agent name for harbor and DB entries (default: \"{DEFAULT_AGENT_NAME}\")",
    )
    parser.add_argument(
        "--slurm-partition",
        help=f"SLURM partition (default: \"{DEFAULT_SLURM_PARTITION}\")",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking blocks for model inference (default: False)",
    )
    parser.add_argument(
        "--upload-username",
        help="Username for DB entries and result uploads (default: current OS user)",
    )

    # Execution mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview mode, no actual submission (implies --once)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run single iteration and exit",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def _env_bool(name: str) -> bool:
    """Get boolean from environment variable."""
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def build_config(args: argparse.Namespace) -> ListenerConfig:
    """Build configuration from args, env vars, and preset defaults."""

    # Start with preset if specified
    preset_config: Dict = {}
    if args.preset:
        preset_config = PRESETS.get(args.preset, {})

    # Resolve datasets: CLI > ENV > Preset
    datasets_str = args.datasets or os.getenv("EVAL_LISTENER_DATASETS") or ""
    if datasets_str:
        datasets = parse_datasets(datasets_str)
    else:
        datasets = preset_config.get("datasets", [])

    if not datasets:
        print("ERROR: No datasets specified. Use --datasets, EVAL_LISTENER_DATASETS, or --preset")
        sys.exit(2)

    # Resolve sbatch script: CLI > ENV > Preset > Default
    sbatch_script = (
        args.sbatch_script
        or os.getenv("EVAL_LISTENER_SBATCH")
        or preset_config.get("sbatch_script")
        or DEFAULT_SBATCH_SCRIPT
    )

    # Resolve timing: CLI > ENV > Default
    lookback_days = (
        args.lookback_days
        if args.lookback_days is not None
        else int(os.getenv("EVAL_LISTENER_LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS)))
    )
    check_hours = (
        args.check_hours
        if args.check_hours is not None
        else float(os.getenv("EVAL_LISTENER_CHECK_HOURS", str(DEFAULT_CHECK_HOURS)))
    )
    stale_hours = args.stale_hours if args.stale_hours is not None else DEFAULT_STALE_JOB_HOURS
    stale_pending_hours = args.stale_pending_hours if args.stale_pending_hours is not None else DEFAULT_STALE_PENDING_HOURS

    # Resolve log file
    log_dir = Path(os.getenv("EVAL_LISTENER_LOG_DIR", DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        suffix = preset_config.get("log_suffix", "unified")
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{suffix}_eval_listener_{current_time}.log"

    # Resolve priority file: CLI > ENV
    priority_file = args.priority_file or os.getenv("EVAL_LISTENER_PRIORITY_FILE")
    priority_models = load_priority_models(priority_file)

    # Resolve boolean flags: CLI > ENV > Preset
    require_priority = args.require_priority_list or _env_bool("EVAL_LISTENER_REQUIRE_PRIORITY_LIST")
    dry_run = args.dry_run or _env_bool("EVAL_LISTENER_DRY_RUN")
    check_hf_exists = (
        args.check_hf_exists
        or _env_bool("EVAL_LISTENER_CHECK_HF_EXISTS")
        or preset_config.get("check_hf_exists", False)
    )

    # Resolve sbatch parameters: CLI > Preset > Default
    # Helper to get value with priority: CLI arg > Preset > Default
    def _resolve(cli_val, preset_key: str, default):
        if cli_val is not None:
            return cli_val
        return preset_config.get(preset_key, default)

    n_concurrent = _resolve(args.n_concurrent, "n_concurrent", DEFAULT_N_CONCURRENT)
    n_attempts = _resolve(args.n_attempts, "n_attempts", DEFAULT_N_ATTEMPTS)
    gpu_memory_util = _resolve(args.gpu_memory_util, "gpu_memory_util", DEFAULT_GPU_MEMORY_UTIL)
    daytona_threshold = _resolve(args.daytona_threshold, "daytona_threshold", DEFAULT_DAYTONA_THRESHOLD)
    vllm_max_retries = _resolve(args.vllm_max_retries, "vllm_max_retries", DEFAULT_VLLM_MAX_RETRIES)
    agent_parser = _resolve(args.agent_parser, "agent_parser", DEFAULT_AGENT_PARSER)
    slurm_time = _resolve(args.slurm_time, "slurm_time", DEFAULT_SLURM_TIME)
    agent_name = _resolve(args.agent_name, "agent_name", DEFAULT_AGENT_NAME)
    slurm_partition = _resolve(args.slurm_partition, "slurm_partition", DEFAULT_SLURM_PARTITION)
    # enable_thinking: CLI flag > Preset > Default (CLI is action="store_true" so check explicitly)
    enable_thinking = args.enable_thinking or preset_config.get("enable_thinking", DEFAULT_ENABLE_THINKING)

    # Resolve upload_username: CLI > ENV > current OS user
    upload_username = (
        args.upload_username
        or os.getenv("EVAL_UPLOAD_USERNAME")
        or getpass.getuser()
    )

    return ListenerConfig(
        datasets=datasets,
        sbatch_script=sbatch_script,
        log_file=log_file,
        lookback_days=lookback_days,
        check_interval_hours=check_hours,
        stale_job_hours=stale_hours,
        stale_pending_hours=stale_pending_hours,
        priority_file=priority_file,
        require_priority_list=require_priority,
        priority_models=priority_models,
        check_hf_exists=check_hf_exists,
        dry_run=dry_run,
        run_once=args.once,
        verbose=args.verbose,
        # Sbatch parameters
        n_concurrent=n_concurrent,
        n_attempts=n_attempts,
        gpu_memory_util=gpu_memory_util,
        daytona_threshold=daytona_threshold,
        vllm_max_retries=vllm_max_retries,
        agent_parser=agent_parser,
        slurm_time=slurm_time,
        enable_thinking=enable_thinking,
        agent_name=agent_name,
        slurm_partition=slurm_partition,
        upload_username=upload_username,
    )


# ---------- Main ----------
def main() -> None:
    args = parse_args()
    config = build_config(args)
    listener = EvalListener(config)
    listener.run()


if __name__ == "__main__":
    main()
