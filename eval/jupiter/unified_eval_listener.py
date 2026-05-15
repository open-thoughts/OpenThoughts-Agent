#!/usr/bin/env python3
"""
Unified Eval Listener for Jupiter Cluster (JSC GH200)

Long-running daemon that polls Supabase for new models and submits SLURM eval jobs.
Supports preset-based configuration for multiple benchmarks.

Reuses core logic from eval/tacc/dev_eval_listener.py and DB operations from
database/unified_db (unified_db.utils).

Env overrides:
  EVAL_LISTENER_LOOKBACK_DAYS   (int, default "100")
  EVAL_LISTENER_CHECK_HOURS     (float, default "4")
  EVAL_LISTENER_SBATCH          (default "eval/jupiter/unified_eval_harbor.sbatch")
  EVAL_LISTENER_LOG_DIR         (default "eval/jupiter/logs")
  EVAL_LISTENER_DATASETS        comma/space/newline list of HF dataset repos
  EVAL_LISTENER_PRIORITY_FILE   path to priority models file
  EVAL_LISTENER_STALE_HOURS     (float, default "24")
  EVAL_LISTENER_STALE_PENDING_HOURS (float, default "6")

Usage:
  # Dry run with preset
  python eval/jupiter/unified_eval_listener.py --preset bfcl --dry-run --verbose

  # Single iteration with priority file
  python eval/jupiter/unified_eval_listener.py --preset bfcl --once --verbose \\
    --priority-file eval/jupiter/priority_models.txt

  # Long-running daemon
  python eval/jupiter/unified_eval_listener.py --preset dev --verbose
"""
import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: project root (for database.unified_db and eval.tacc imports)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

# Project root: so we can import database.unified_db.* and eval.tacc.*
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Imports from database/unified_db
# ---------------------------------------------------------------------------
from database.unified_db.utils import (  # noqa: E402
    get_supabase_client,
    create_job_entry_pending,
    load_supabase_keys,
)

# ---------------------------------------------------------------------------
# Imports from eval/tacc/dev_eval_listener.py — reuse all the core logic
# ---------------------------------------------------------------------------
from eval.tacc.dev_eval_listener import (  # noqa: E402
    _parse_datasets,
    _parse_hf_from_str,
    _resolve_hf_model_name,
    _dataset_repo_name,
    resolve_benchmark_id_for_dataset,
    fetch_recent_models,
    check_job_status,
    is_job_stale,
    HF_URL_RE,
)

# ---------------------------------------------------------------------------
# Presets — hardcoded configs for common benchmark suites
# ---------------------------------------------------------------------------
PRESETS: Dict[str, Dict] = {
    "bfcl": {
        "datasets": ["DCAgent2/bfcl-parity"],
        "description": "Berkeley Function Calling Leaderboard eval",
        "n_concurrent": 16,
        "gpu_memory_util": 0.95,
        "snapshot_name": "harbor_bfcl_python_v2",
        "daytona_api_key": "dtn_17868a1955b56a52cb367af6dd3c6e93ee531b2073df801784273435c0e0fc6c",
    },
    "aider": {
        "datasets": ["DCAgent2/aider_polyglot"],
        "description": "Aider code editing eval",
        "n_concurrent": 64,
        "gpu_memory_util": 0.95,
    },
    "swebench": {
        "datasets": ["DCAgent/swebench_verified_eval_set"],
        "description": "SWE-bench verified eval",
        "n_concurrent": 32,
        "gpu_memory_util": 0.95,
    },
    "v2": {
        "datasets": ["DCAgent/dev_set_v2"],
        "description": "Dev set v2 eval",
        "n_concurrent": 128,
        "gpu_memory_util": 0.95,
    },
    "tb2": {
        "datasets": ["DCAgent2/terminal_bench_2"],
        "description": "Terminal Bench v2 eval",
        "n_concurrent": 64,
        "gpu_memory_util": 0.95,
    },
    "dev": {
        "datasets": ["DCAgent/dev_set_71_tasks"],
        "description": "Dev set (71 tasks) eval",
        "n_concurrent": 128,
        "gpu_memory_util": 0.95,
    },
}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRINT_PREFIX = "[eval-listener]"


# ---------------------------------------------------------------------------
# Harbor config parsing — extract eval config fields for dedup
# ---------------------------------------------------------------------------
def parse_harbor_eval_config(path: Optional[str]) -> Dict[str, Any]:
    """Parse eval-relevant config fields from a Harbor YAML config.

    Returns dict with keys: timeout_multiplier, override_cpus,
    override_memory_mb, override_storage_mb (only if set).
    """
    if not path or not os.path.isfile(path):
        return {}
    try:
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        log(f"WARNING: failed to parse harbor config {path}: {e}")
        return {}
    result: Dict[str, Any] = {}
    if cfg.get("timeout_multiplier") is not None:
        result["timeout_multiplier"] = float(cfg["timeout_multiplier"])
    env_cfg = cfg.get("environment") or {}
    for key in ("override_cpus", "override_memory_mb", "override_storage_mb"):
        if env_cfg.get(key) is not None:
            result[key] = int(env_cfg[key])
    return result


JOB_STATUS_PENDING = "Pending"
JOB_STATUS_STARTED = "Started"
JOB_STATUS_FINISHED = "Finished"


# ---------------------------------------------------------------------------
# Datagen config loading — extract vLLM serving params from datagen YAML
# ---------------------------------------------------------------------------

def load_datagen_config_env(path: Optional[str]) -> Dict[str, str]:
    """Load a datagen/serving YAML and extract vLLM params as EVAL_VLLM_* env vars.

    This provides global defaults for vLLM serving when running evals with
    custom TP/DP/PP configurations (e.g., multi-GPU serving on Jupiter/Leonardo).
    """
    if not path or not os.path.isfile(path):
        return {}

    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        log(f"WARNING: failed to load datagen config from {path}: {e}")
        return {}

    server = data.get("vllm_server", {})
    backend = data.get("backend", {})
    env: Dict[str, str] = {}

    # Core serving params
    if server.get("tensor_parallel_size") is not None:
        env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] = str(server["tensor_parallel_size"])
    if server.get("pipeline_parallel_size") is not None:
        env["EVAL_VLLM_PIPELINE_PARALLEL_SIZE"] = str(server["pipeline_parallel_size"])
    if server.get("data_parallel_size") is not None:
        env["EVAL_VLLM_DATA_PARALLEL_SIZE"] = str(server["data_parallel_size"])
    if server.get("max_model_len") is not None:
        env["EVAL_VLLM_MAX_MODEL_LEN"] = str(server["max_model_len"])
    if server.get("gpu_memory_utilization") is not None:
        env["EVAL_GPU_MEMORY_UTIL"] = str(server["gpu_memory_utilization"])
    if server.get("swap_space") is not None:
        env["EVAL_VLLM_SWAP_SPACE"] = str(server["swap_space"])
    if server.get("max_num_seqs") is not None:
        env["EVAL_VLLM_MAX_NUM_SEQS"] = str(server["max_num_seqs"])
    if server.get("trust_remote_code"):
        env["EVAL_VLLM_TRUST_REMOTE_CODE"] = "1"
    if server.get("tool_call_parser"):
        env["EVAL_VLLM_TOOL_CALL_PARSER"] = server["tool_call_parser"]
    if server.get("reasoning_parser"):
        env["EVAL_VLLM_REASONING_PARSER"] = server["reasoning_parser"]
    if server.get("enable_expert_parallel"):
        env["EVAL_VLLM_ENABLE_EXPERT_PARALLEL"] = "1"
    if server.get("extra_args"):
        # Convert list to space-separated string
        extra = server["extra_args"]
        if isinstance(extra, list):
            extra = " ".join(str(a) for a in extra)
        env["EVAL_VLLM_EXTRA_ARGS"] = extra

    # Backend params (healthcheck)
    if backend.get("healthcheck_max_attempts") is not None:
        env["EVAL_HEALTHCHECK_MAX_ATTEMPTS"] = str(backend["healthcheck_max_attempts"])
    if backend.get("healthcheck_retry_delay") is not None:
        env["EVAL_HEALTHCHECK_RETRY_DELAY"] = str(backend["healthcheck_retry_delay"])

    log(f"Loaded datagen config from {path}: {', '.join(f'{k}={v}' for k, v in env.items())}")
    return env


# ---------------------------------------------------------------------------
# Baseline model config mapping — per-model vLLM overrides
# ---------------------------------------------------------------------------
_BASELINE_MODEL_CONFIGS: Optional[Dict[str, Dict[str, Any]]] = None
_BASELINE_MODEL_PATTERNS: Optional[List[Dict[str, Any]]] = None


def load_baseline_model_configs(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load baseline model -> vLLM config mapping from YAML file.

    Returns dict mapping HF model name to vLLM serving params.
    Also loads pattern-based fallback configs (stored in _BASELINE_MODEL_PATTERNS).
    """
    global _BASELINE_MODEL_CONFIGS, _BASELINE_MODEL_PATTERNS
    if _BASELINE_MODEL_CONFIGS is not None:
        return _BASELINE_MODEL_CONFIGS

    if not path or not os.path.isfile(path):
        _BASELINE_MODEL_CONFIGS = {}
        _BASELINE_MODEL_PATTERNS = []
        return _BASELINE_MODEL_CONFIGS

    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _BASELINE_MODEL_CONFIGS = data.get("models", {})
        _BASELINE_MODEL_PATTERNS = data.get("patterns", [])
        log(f"Loaded {len(_BASELINE_MODEL_CONFIGS)} baseline model config(s) and "
            f"{len(_BASELINE_MODEL_PATTERNS)} pattern(s) from {path}")
    except Exception as e:
        log(f"WARNING: failed to load baseline model configs from {path}: {e}")
        _BASELINE_MODEL_CONFIGS = {}
        _BASELINE_MODEL_PATTERNS = []

    return _BASELINE_MODEL_CONFIGS


def _match_pattern_config(hf_model: str) -> Optional[Dict[str, Any]]:
    """Try to match a model name against pattern-based configs.

    Patterns are checked in order; first match wins.
    Each pattern has a 'match' field (regex or substring) and config fields.
    """
    import re
    if not _BASELINE_MODEL_PATTERNS:
        return None
    for pattern_entry in _BASELINE_MODEL_PATTERNS:
        pattern = pattern_entry.get("match", "")
        if not pattern:
            continue
        if re.search(pattern, hf_model):
            return {k: v for k, v in pattern_entry.items() if k != "match"}
    return None


def get_vllm_env_overrides(hf_model: str, configs: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Get vLLM env var overrides for a model from the baseline config mapping.

    Tries exact model name match first, then falls back to pattern matching.
    Returns dict of EVAL_VLLM_* env vars to pass to the eval script.
    """
    cfg = configs.get(hf_model)
    if not cfg:
        cfg = _match_pattern_config(hf_model)
    if not cfg:
        return {}

    env = {}
    if cfg.get("tensor_parallel_size") is not None:
        env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] = str(cfg["tensor_parallel_size"])
    if cfg.get("pipeline_parallel_size") is not None:
        env["EVAL_VLLM_PIPELINE_PARALLEL_SIZE"] = str(cfg["pipeline_parallel_size"])
    if cfg.get("data_parallel_size") is not None:
        env["EVAL_VLLM_DATA_PARALLEL_SIZE"] = str(cfg["data_parallel_size"])
    if cfg.get("max_model_len") is not None:
        env["EVAL_VLLM_MAX_MODEL_LEN"] = str(cfg["max_model_len"])
    if cfg.get("swap_space") is not None:
        env["EVAL_VLLM_SWAP_SPACE"] = str(cfg["swap_space"])
    if cfg.get("trust_remote_code"):
        env["EVAL_VLLM_TRUST_REMOTE_CODE"] = "1"
    if cfg.get("tool_call_parser"):
        env["EVAL_VLLM_TOOL_CALL_PARSER"] = cfg["tool_call_parser"]
    if cfg.get("reasoning_parser"):
        env["EVAL_VLLM_REASONING_PARSER"] = cfg["reasoning_parser"]
    if cfg.get("extra_args"):
        env["EVAL_VLLM_EXTRA_ARGS"] = cfg["extra_args"]

    return env


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_LOG_FILE: Optional[Path] = None
_VERBOSE: bool = False


def _init_logging(log_dir: str) -> None:
    global _LOG_FILE
    p = Path(log_dir)
    p.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = p / "eval_listener.log"


def log(msg: str, verbose_only: bool = False) -> None:
    if verbose_only and not _VERBOSE:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{PRINT_PREFIX} {ts}  {msg}"
    print(line, flush=True)
    if _LOG_FILE:
        try:
            with _LOG_FILE.open("a") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Secrets loading (env files)
# ---------------------------------------------------------------------------
def _load_secrets(path: Optional[str] = None) -> None:
    """Load secrets from env file, then call unified_db's load_supabase_keys."""
    path = (
        path
        or os.environ.get("DC_AGENT_SECRET_ENV")
        or os.environ.get("KEYS")
        or os.path.expanduser("~/secrets.env")
    )
    if path and os.path.isfile(os.path.expanduser(path)):
        with open(os.path.expanduser(path)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip("'\"")
    # Also call unified_db's loader for any additional key sources
    try:
        load_supabase_keys()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Optional HF existence check
# ---------------------------------------------------------------------------
def check_hf_model_exists(hf_model: str) -> bool:
    """Check whether a model exists on HuggingFace Hub. Returns True on error (fail-open)."""
    try:
        from huggingface_hub import model_info  # type: ignore

        model_info(hf_model)
        return True
    except Exception as exc:
        exc_str = str(exc).lower()
        if "404" in exc_str or "not found" in exc_str:
            return False
        log(f"WARNING: HF existence check failed for {hf_model}: {exc}")
        return True


# ---------------------------------------------------------------------------
# Priority file handling
# ---------------------------------------------------------------------------
def load_priority_models(path: Optional[str]) -> Optional[List[str]]:
    """Load a priority models file (one HF model name per line). Returns None if no file."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        log(f"Priority file not found: {path}")
        return None
    models: List[str] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = HF_URL_RE.search(line)
            if m:
                models.append(f"{m.group(1)}/{m.group(2)}")
            else:
                models.append(line)
    log(f"Loaded {len(models)} model(s) from priority file {path}", verbose_only=True)
    return models if models else None



# ---------------------------------------------------------------------------
# Pending job tracking (extends TACC logic with Pending status + stale detection)
# ---------------------------------------------------------------------------
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def scancel_job(slurm_job_id: str) -> bool:
    """Cancel a SLURM job. Returns True on success."""
    try:
        result = subprocess.run(
            ["scancel", slurm_job_id],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            log(f"scancel {slurm_job_id}: success")
            return True
        else:
            log(f"scancel {slurm_job_id}: failed ({result.stderr.strip()})")
            return False
    except Exception as e:
        log(f"scancel {slurm_job_id}: exception ({e})")
        return False


def _config_matches_eval(job_config: Optional[Dict], eval_config: Dict[str, Any]) -> bool:
    """Check if a DB job's config JSONB matches the current eval config fields.

    Compares: timeout_multiplier, override_cpus, override_memory_mb, override_storage_mb.
    A job with no config is treated as defaults (timeout=1.0, no overrides).
    If eval_config is empty (no harbor config), any job config matches (backwards compat).
    """
    if not eval_config:
        return True  # no config constraints — any existing job counts

    job_cfg = job_config or {}
    job_env = job_cfg.get("environment") or {}

    # timeout_multiplier: top-level in config JSONB
    if "timeout_multiplier" in eval_config:
        job_tm = job_cfg.get("timeout_multiplier")
        # Treat None/missing as 1.0
        job_tm = float(job_tm) if job_tm is not None else 1.0
        if float(eval_config["timeout_multiplier"]) != job_tm:
            return False

    # Environment overrides: nested under config.environment
    for key in ("override_cpus", "override_memory_mb", "override_storage_mb"):
        if key in eval_config:
            job_val = job_env.get(key)
            # Treat None/missing as the default (None means no override)
            job_val = int(job_val) if job_val is not None else None
            eval_val = int(eval_config[key])
            if job_val != eval_val:
                return False

    return True


def should_start_job(
    model_id: str,
    benchmark_id: Optional[str],
    eval_config: Optional[Dict[str, Any]] = None,
    stale_started_hours: float = 24.0,
    stale_pending_hours: float = 6.0,
) -> Tuple[bool, str]:
    """
    Determine if a job should be started based on DB status.
    Extends TACC's check_job_status with Pending handling, auto-scancel,
    and config-aware deduplication (timeout, resource overrides).
    """
    # First, quick check: does any job exist at all for this model+benchmark?
    job_exists, job_status, started_at = check_job_status(model_id, benchmark_id)

    if not job_exists:
        return (True, "no existing job")

    # If we have eval config constraints, do a config-aware check against
    # all recent jobs for this model+benchmark (not just the latest one).
    ec = eval_config or {}

    if job_status == JOB_STATUS_FINISHED:
        if ec:
            # Check if any Finished job matches our eval config
            try:
                client = get_supabase_client()
                q = (
                    client.table("sandbox_jobs")
                    .select("config, metrics")
                    .eq("model_id", model_id)
                    .eq("benchmark_id", benchmark_id)
                    .eq("job_status", "Finished")
                    .order("created_at", desc=True)
                    .limit(10)
                )
                rows = (q.execute().data) or []
                matching = [r for r in rows if _config_matches_eval(r.get("config"), ec)]
                if not matching:
                    return (True, "no finished job with matching config")
                # Check if the matching finished job has metrics
                if matching[0] and not matching[0].get("metrics"):
                    return (True, "finished with matching config but metrics cleared")
            except Exception as e:
                log(f"WARNING: config-aware check failed: {e}")
        else:
            # No eval config — original behavior: check latest finished job for metrics
            try:
                client = get_supabase_client()
                q = (
                    client.table("sandbox_jobs")
                    .select("metrics")
                    .eq("model_id", model_id)
                    .eq("benchmark_id", benchmark_id)
                    .eq("job_status", "Finished")
                    .order("created_at", desc=True)
                    .limit(1)
                )
                data = (q.execute().data) or []
                if data and not data[0].get("metrics"):
                    return (True, "finished but metrics cleared (invalid errors)")
            except Exception:
                pass
        return (False, "job finished")

    if job_status == JOB_STATUS_STARTED:
        if ec:
            # Check if the in-progress job matches our config
            try:
                client = get_supabase_client()
                q = (
                    client.table("sandbox_jobs")
                    .select("config, started_at")
                    .eq("model_id", model_id)
                    .eq("benchmark_id", benchmark_id)
                    .eq("job_status", "Started")
                    .order("created_at", desc=True)
                    .limit(5)
                )
                rows = (q.execute().data) or []
                matching = [r for r in rows if _config_matches_eval(r.get("config"), ec)]
                if not matching:
                    return (True, "no in-progress job with matching config")
            except Exception as e:
                log(f"WARNING: config-aware check failed: {e}")
        if is_job_stale(started_at, int(stale_started_hours)):
            ts_str = started_at.isoformat() if started_at else "null"
            return (True, f"stale started job (started_at={ts_str})")
        else:
            ts_str = started_at.isoformat() if started_at else "null"
            return (False, f"job in progress (started_at={ts_str})")

    if job_status == JOB_STATUS_PENDING:
        if ec:
            # Check if the pending job matches our config
            try:
                client = get_supabase_client()
                q = (
                    client.table("sandbox_jobs")
                    .select("config, slurm_job_id, created_at")
                    .eq("model_id", model_id)
                    .eq("benchmark_id", benchmark_id)
                    .eq("job_status", "Pending")
                    .order("created_at", desc=True)
                    .limit(5)
                )
                rows = (q.execute().data) or []
                matching = [r for r in rows if _config_matches_eval(r.get("config"), ec)]
                if not matching:
                    return (True, "no pending job with matching config")
            except Exception as e:
                log(f"WARNING: config-aware check failed: {e}")
        if is_job_stale(started_at, int(stale_pending_hours)):
            ts_str = started_at.isoformat() if started_at else "null"
            # Try to get slurm_job_id for auto-scancel
            try:
                client = get_supabase_client()
                q = (
                    client.table("sandbox_jobs")
                    .select("slurm_job_id")
                    .eq("model_id", model_id)
                    .eq("benchmark_id", benchmark_id)
                    .eq("job_status", "Pending")
                    .order("created_at", desc=True)
                    .limit(1)
                )
                data = (q.execute().data) or []
                if data and data[0].get("slurm_job_id"):
                    log(f"Auto-cancelling stale pending SLURM job {data[0]['slurm_job_id']}")
                    scancel_job(data[0]["slurm_job_id"])
            except Exception:
                pass
            return (True, f"stale pending job (submitted_at={ts_str})")
        else:
            ts_str = started_at.isoformat() if started_at else "null"
            return (False, f"job pending (submitted_at={ts_str})")

    # Unknown status
    return (True, f"unknown job status: {job_status}")


# ---------------------------------------------------------------------------
# Pending job creation (uses database/unified_db create_job_entry_pending)
# ---------------------------------------------------------------------------
def create_pending_job(
    model_hf: str,
    dataset_hf: str,
    benchmark_id: Optional[str],
    model_id: str,
    slurm_job_id: str = "pending",
    eval_config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Create a Pending job entry in sandbox_jobs before sbatch submission.
    Uses database/unified_db's create_job_entry_pending().
    Returns the DB job row id, or None on failure.
    """
    # Build config JSONB matching the schema the sbatch will also write
    job_config: Optional[Dict[str, Any]] = None
    if eval_config:
        job_config = {}
        if "timeout_multiplier" in eval_config:
            job_config["timeout_multiplier"] = eval_config["timeout_multiplier"]
        env_overrides = {}
        for key in ("override_cpus", "override_memory_mb", "override_storage_mb"):
            if key in eval_config:
                env_overrides[key] = eval_config[key]
        if env_overrides:
            job_config["environment"] = env_overrides
    try:
        result = create_job_entry_pending(
            job_name=f"pending_{model_hf.replace('/', '_')}_{_dataset_repo_name(dataset_hf)}",
            model_hf=model_hf,
            benchmark_hf=dataset_hf,
            agent_name="terminus-2",
            slurm_job_id=slurm_job_id,
            username=os.environ.get("USER", "jupiter"),
            config=job_config,
        )
        if result.get("success") and result.get("job"):
            db_id = result["job"].get("id")
            log(f"Created Pending job entry: {db_id}", verbose_only=True)
            return str(db_id)
        else:
            log(f"WARNING: create_job_entry_pending returned: {result.get('error', 'unknown')}")
            return None
    except Exception as e:
        log(f"WARNING: failed to create Pending job entry: {e}")
        return None


def update_pending_job_slurm_id(db_job_id: str, slurm_job_id: str) -> None:
    """Update the Pending job entry with the SLURM job ID after successful sbatch."""
    try:
        client = get_supabase_client()
        client.table("sandbox_jobs").update(
            {"slurm_job_id": slurm_job_id}
        ).eq("id", db_job_id).execute()
        log(f"Updated job {db_job_id} with slurm_job_id={slurm_job_id}", verbose_only=True)
    except Exception as e:
        log(f"WARNING: failed to update job {db_job_id} with slurm_job_id: {e}")


# ---------------------------------------------------------------------------
# sbatch submission
# ---------------------------------------------------------------------------
def _run(cmd: List[str], env: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
    )
    out_lines: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line.rstrip())
    code = proc.wait()
    return code, "\n".join(out_lines)


def submit_eval(
    hf_model_name: str,
    dataset_hf: str,
    benchmark_id: Optional[str],
    sbatch_script: str,
    sbatch_env: Optional[Dict[str, str]] = None,
    dry_run: bool = False,
    reservation: Optional[str] = None,
    dependency: Optional[str] = None,
    remote_host: Optional[str] = None,
    remote_workdir: Optional[str] = None,
) -> Optional[str]:
    """
    Submit a batch job (SLURM sbatch or PBS qsub). Returns job_id if successful.

    Detects scheduler from script extension: .pbs -> qsub, else -> sbatch.
    If remote_host is set, submits via SSH (e.g., for Polaris where Supabase is
    blocked but the listener runs locally with DB access).

    SLURM: positional args $1=model, $2=dataset, $3=benchmark_id
    PBS: env vars EVAL_MODEL, EVAL_REPO_ID, EVAL_BENCHMARK_ID via qsub -v
    """
    use_pbs = sbatch_script.endswith(".pbs")

    if use_pbs:
        # PBS Pro: pass all params as env vars via qsub -v
        all_vars = dict(sbatch_env or {})
        all_vars["EVAL_MODEL"] = hf_model_name
        all_vars["EVAL_REPO_ID"] = dataset_hf
        if benchmark_id:
            all_vars["EVAL_BENCHMARK_ID"] = str(benchmark_id)
        v_flag = ",".join(f"{k}={v}" for k, v in all_vars.items())
        cmd = ["qsub", "-v", v_flag]
        if dependency:
            # PBS dependency syntax: -W depend=afterany:jobid
            cmd.extend(["-W", f"depend={dependency}"])
        cmd.append(sbatch_script)
        job_id_pattern = r"(\d+\.\w+)"  # PBS job IDs: 12345.polaris-pbs-01
    else:
        # SLURM: --export for env vars, positional args for model/dataset
        export_parts = ["ALL"]
        if sbatch_env:
            for k, v in sbatch_env.items():
                export_parts.append(f"{k}={v}")
        export_flag = ",".join(export_parts)
        cmd = ["sbatch", f"--export={export_flag}"]
        if reservation:
            cmd.append(f"--reservation={reservation}")
        if dependency:
            cmd.append(f"--dependency={dependency}")
        cmd.extend([sbatch_script, hf_model_name, dataset_hf])
        if benchmark_id:
            cmd.append(str(benchmark_id))
        job_id_pattern = r"Submitted batch job (\d+)"

    # Wrap in SSH if submitting to a remote cluster
    if remote_host:
        # Quote each arg for the remote shell, then join into a single command string
        import shlex
        shell_cmd = " ".join(shlex.quote(c) for c in cmd)
        if remote_workdir:
            shell_cmd = f"cd {shlex.quote(remote_workdir)} && {shell_cmd}"
        cmd = ["ssh", remote_host, shell_cmd]

    if dry_run:
        log(f"[DRY RUN] Would run: {' '.join(cmd)}")
        if sbatch_env:
            for k, v in sbatch_env.items():
                log(f"  env: {k}={v}", verbose_only=True)
        return "DRY_RUN"

    code, out = _run(cmd, env=sbatch_env if not remote_host else None)
    scheduler = "qsub" if use_pbs else "sbatch"
    log(f"{scheduler}: {' '.join(cmd)}\n{out}")

    if code != 0:
        return None

    m = re.search(job_id_pattern, out)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Eval Listener for Jupiter cluster",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets:
  bfcl      Berkeley Function Calling Leaderboard
  aider     Aider code editing
  swebench  SWE-bench verified
  v2        Dev set v2
  tb2       Terminal Bench v2
  dev       Dev set (71 tasks)

Examples:
  %(prog)s --preset dev --dry-run --verbose
  %(prog)s --preset bfcl --once --priority-file models.txt
  %(prog)s --datasets DCAgent/dev_set_v2,DCAgent/bfcl_eval_set --verbose
""",
    )

    # Dataset selection
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preset", choices=list(PRESETS.keys()), help="Use a preset benchmark configuration")
    g.add_argument("--datasets", help="Comma/space separated list of HF dataset repos")

    # Sbatch
    p.add_argument(
        "--sbatch-script",
        default=os.getenv("EVAL_LISTENER_SBATCH", "eval/jupiter/unified_eval_harbor.sbatch"),
        help="Path to sbatch script (default: %(default)s)",
    )

    # Timing
    p.add_argument("--lookback-days", type=int, default=int(os.getenv("EVAL_LISTENER_LOOKBACK_DAYS", "100")))
    p.add_argument("--check-hours", type=float, default=float(os.getenv("EVAL_LISTENER_CHECK_HOURS", "4")))
    p.add_argument("--stale-started-hours", type=float, default=float(os.getenv("EVAL_LISTENER_STALE_HOURS", "24")))
    p.add_argument("--stale-pending-hours", type=float, default=float(os.getenv("EVAL_LISTENER_STALE_PENDING_HOURS", "6")))

    # Priority file
    p.add_argument("--priority-file", default=os.getenv("EVAL_LISTENER_PRIORITY_FILE"))
    p.add_argument("--require-priority-list", action="store_true")

    # Sbatch params (passed as env vars to sbatch)
    p.add_argument("--n-concurrent", type=int, default=None)
    p.add_argument("--gpu-memory-util", type=float, default=None)
    p.add_argument("--daytona-threshold", type=int, default=3)
    p.add_argument("--harbor-config", default=None,
                   help="Path to Harbor YAML config (passed as EVAL_HARBOR_CONFIG to sbatch)")
    p.add_argument("--reservation", default=os.getenv("EVAL_LISTENER_RESERVATION"), help="SLURM reservation name")
    p.add_argument("--max-jobs", type=int, default=None, help="Maximum number of SLURM jobs to submit in one iteration")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Max concurrent jobs via sliding-window SLURM dependencies. "
                        "Job N depends on job N-batch_size finishing (afterany), "
                        "so at most batch-size jobs run at once. As one finishes, "
                        "the next starts immediately.")
    p.add_argument("--dependency", default=None, help="SLURM dependency string (e.g., 'afterany:123:456')")
    p.add_argument("--remote-host", default=None,
                   help="SSH host for remote job submission (e.g., ALCFPolaris). "
                        "Listener runs locally (with DB access) and submits jobs via SSH.")
    p.add_argument("--remote-workdir", default=None,
                   help="Working directory on remote host (cd before qsub/sbatch)")

    # Baseline model configs
    p.add_argument("--baseline-model-configs", default=None,
                   help="Path to YAML mapping baseline models to vLLM serving params "
                        "(e.g., eval/baseline_model_configs.yaml)")
    p.add_argument("--datagen-config", default=None,
                   help="Path to datagen/serving YAML (e.g., hpc/datagen_yaml/qwen3_8b_vllm_serve_32k_4xGH200.yaml). "
                        "Extracts vLLM serving params (TP, DP, PP, max_model_len, etc.) as global defaults.")

    # Modes
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--check-hf", action="store_true")
    p.add_argument("--pre-download", action="store_true",
                   help="Pre-download all model weights on login node before submitting jobs. "
                        "Essential for no-internet compute nodes (Leonardo, Jupiter).")
    p.add_argument("--log-dir", default=os.getenv("EVAL_LISTENER_LOG_DIR", "eval/jupiter/logs"))

    return p


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    global _VERBOSE

    parser = build_parser()
    args = parser.parse_args()
    _VERBOSE = args.verbose

    _load_secrets()
    _init_logging(args.log_dir)

    # Resolve datasets
    datasets: List[str] = []
    preset_config: Dict = {}
    if args.preset:
        preset_config = PRESETS[args.preset]
        datasets = list(preset_config["datasets"])
        log(f"Using preset '{args.preset}': {preset_config['description']}")
    elif args.datasets:
        datasets = _parse_datasets(args.datasets)
    else:
        raw = os.getenv("EVAL_LISTENER_DATASETS", "")
        if raw:
            datasets = _parse_datasets(raw)

    if not datasets:
        log("ERROR: No datasets specified. Use --preset, --datasets, or EVAL_LISTENER_DATASETS.")
        sys.exit(2)

    # Sbatch env vars — start with datagen config defaults (if provided),
    # then layer on CLI overrides.
    sbatch_env: Dict[str, str] = {}

    # Load datagen config as base defaults for vLLM serving params
    if args.datagen_config:
        sbatch_env.update(load_datagen_config_env(args.datagen_config))

    # CLI overrides take precedence
    n_concurrent = args.n_concurrent or preset_config.get("n_concurrent", 128)
    gpu_mem_util = args.gpu_memory_util or preset_config.get("gpu_memory_util", 0.95)
    sbatch_env["EVAL_N_CONCURRENT"] = str(n_concurrent)
    # Only override gpu_mem_util if explicitly set via CLI or preset (don't clobber datagen)
    if args.gpu_memory_util is not None or "EVAL_GPU_MEMORY_UTIL" not in sbatch_env:
        sbatch_env["EVAL_GPU_MEMORY_UTIL"] = str(gpu_mem_util)
    sbatch_env["EVAL_DAYTONA_THRESHOLD"] = str(args.daytona_threshold)

    # Harbor config (CLI > preset > sbatch default)
    harbor_config = args.harbor_config or preset_config.get("harbor_config")
    if harbor_config:
        sbatch_env["EVAL_HARBOR_CONFIG"] = harbor_config

    # Parse eval-relevant config from harbor YAML for dedup + sbatch env vars
    eval_config = parse_harbor_eval_config(harbor_config)
    if eval_config.get("timeout_multiplier") is not None:
        sbatch_env["EVAL_TIMEOUT_MULTIPLIER"] = str(eval_config["timeout_multiplier"])
    if eval_config.get("override_memory_mb") is not None:
        sbatch_env["EVAL_OVERRIDE_MEMORY_MB"] = str(eval_config["override_memory_mb"])
    # Preset-specific env vars (snapshot, daytona key)
    if preset_config.get("snapshot_name"):
        sbatch_env["EVAL_SNAPSHOT_NAME"] = preset_config["snapshot_name"]
    if preset_config.get("daytona_api_key"):
        sbatch_env["DAYTONA_API_KEY"] = preset_config["daytona_api_key"]

    # Load baseline model configs for per-model vLLM overrides
    baseline_configs = load_baseline_model_configs(args.baseline_model_configs)

    check_interval_seconds = int(args.check_hours * 3600)

    if eval_config:
        log(f"Eval config for dedup: {eval_config}")
    log(
        f"Starting listener: datasets={datasets}, lookback={args.lookback_days}d, "
        f"interval={args.check_hours}h, sbatch={args.sbatch_script}, "
        f"n_concurrent={n_concurrent}, gpu_mem={gpu_mem_util}"
    )
    if args.dry_run:
        log("DRY RUN mode — no jobs will be submitted")

    while True:
        try:
            priority_models = load_priority_models(args.priority_file)

            log("Checking for new models...")
            models = fetch_recent_models(args.lookback_days)
            log(f"Found {len(models)} model(s) in window.")

            dataset_to_bench: Dict[str, Optional[str]] = {
                ds: resolve_benchmark_id_for_dataset(ds) for ds in datasets
            }

            submissions: List[Tuple[str, str, str, Optional[str], str]] = []

            for m in models:
                model_id = str(m.get("id", ""))
                if not model_id:
                    continue

                hf_model = _resolve_hf_model_name(m)
                if not hf_model:
                    log(f"Skip: cannot resolve HF model for id={model_id}, name={m.get('name')}", verbose_only=True)
                    continue

                if args.require_priority_list and priority_models is not None:
                    if hf_model not in priority_models:
                        continue

                if args.check_hf and not check_hf_model_exists(hf_model):
                    log(f"Skip (not found on HF): {hf_model}")
                    continue

                for dataset_hf in datasets:
                    bench_id = dataset_to_bench.get(dataset_hf)
                    start, reason = should_start_job(
                        model_id, bench_id,
                        eval_config=eval_config,
                        stale_started_hours=args.stale_started_hours,
                        stale_pending_hours=args.stale_pending_hours,
                    )
                    if start:
                        submissions.append((model_id, hf_model, dataset_hf, bench_id, reason))
                    else:
                        log(f"Skip: model={hf_model}, dataset={dataset_hf}, reason={reason}", verbose_only=True)

            if not submissions:
                log("No eligible (model, dataset) pairs to submit.")
            else:
                if priority_models:
                    priority_set = set(priority_models)
                    submissions.sort(key=lambda x: (0 if x[1] in priority_set else 1, x[1]))

                if args.max_jobs and len(submissions) > args.max_jobs:
                    log(f"Capping submissions from {len(submissions)} to {args.max_jobs} (--max-jobs)")
                    submissions = submissions[:args.max_jobs]

                log(f"Submitting {len(submissions)} eval(s)...")

                # Pre-download setup (for no-internet compute nodes)
                pre_download = args.pre_download
                if pre_download:
                    from huggingface_hub import snapshot_download
                    downloaded_models: set = set()
                    # Pre-download datasets (same for all submissions, download once)
                    downloaded_datasets: set = set()
                    failed_datasets: set = set()
                    for _, _, dataset_hf, _, _ in submissions:
                        if dataset_hf not in downloaded_datasets and dataset_hf not in failed_datasets:
                            log(f"  Pre-downloading dataset {dataset_hf}...")
                            try:
                                path = snapshot_download(repo_id=dataset_hf, repo_type="dataset")
                                log(f"  Dataset cached at {path}")
                                downloaded_datasets.add(dataset_hf)
                            except Exception as e:
                                log(f"  ERROR: Failed to download dataset {dataset_hf}: {e}")
                                failed_datasets.add(dataset_hf)

                # Sliding-window dependency: job N depends on job N-batch_size
                # so at most batch_size jobs run concurrently. As one finishes,
                # the next starts immediately (no waiting for entire wave).
                batch_size = args.batch_size
                all_job_ids: List[str] = []

                if batch_size and batch_size > 0:
                    log(f"Using sliding-window batch-size={batch_size}: "
                        f"first {batch_size} run immediately, rest chain one-by-one")

                failed_models: set = set() if pre_download else set()
                for idx, (mid, hf_model, dataset_hf, bench_id, reason) in enumerate(submissions):
                    # Skip if dataset download failed
                    if pre_download and dataset_hf in failed_datasets:
                        log(f"  Skipping {hf_model} (dataset {dataset_hf} download failed)")
                        all_job_ids.append(f"FAILED_{idx}")
                        continue
                    # Pre-download this model before submitting (download-then-submit per model)
                    if pre_download and hf_model not in downloaded_models:
                        if hf_model in failed_models:
                            log(f"  Skipping {hf_model} (pre-download already failed)")
                            all_job_ids.append(f"FAILED_{idx}")
                            continue
                        log(f"  Pre-downloading model {hf_model}...")
                        try:
                            path = snapshot_download(repo_id=hf_model, repo_type="model")
                            log(f"  Cached at {path}")
                            downloaded_models.add(hf_model)
                        except Exception as e:
                            log(f"  WARNING: Failed to download {hf_model}: {e} — skipping submission")
                            failed_models.add(hf_model)
                            all_job_ids.append(f"FAILED_{idx}")
                            continue

                    log(f"Submitting [{idx+1}/{len(submissions)}]: model={hf_model}, dataset={dataset_hf}, reason={reason}")

                    db_job_id = create_pending_job(hf_model, dataset_hf, bench_id, mid,
                                                    eval_config=eval_config)

                    job_env = dict(sbatch_env)
                    if db_job_id:
                        job_env["EVAL_DB_JOB_ID"] = db_job_id

                    # Apply per-model vLLM overrides from baseline config mapping
                    vllm_overrides = get_vllm_env_overrides(hf_model, baseline_configs)
                    if vllm_overrides:
                        log(f"  Applying baseline model vLLM overrides: {list(vllm_overrides.keys())}", verbose_only=True)
                        job_env.update(vllm_overrides)

                    # Build dependency: job N depends on job N-batch_size
                    job_dependency = args.dependency
                    if batch_size and batch_size > 0 and idx >= batch_size:
                        dep_job = all_job_ids[idx - batch_size]
                        dep_str = f"afterany:{dep_job}"
                        if job_dependency:
                            job_dependency = f"{job_dependency},{dep_str}"
                        else:
                            job_dependency = dep_str
                        log(f"  Depends on job {dep_job} (slot {idx - batch_size + 1})", verbose_only=True)

                    slurm_id = submit_eval(
                        hf_model, dataset_hf, bench_id,
                        args.sbatch_script,
                        sbatch_env=job_env,
                        dry_run=args.dry_run,
                        reservation=args.reservation,
                        dependency=job_dependency,
                        remote_host=args.remote_host,
                        remote_workdir=args.remote_workdir,
                    )

                    if slurm_id and slurm_id != "DRY_RUN":
                        log(f"  -> Submitted as SLURM job {slurm_id}")
                        all_job_ids.append(slurm_id)
                        if db_job_id:
                            update_pending_job_slurm_id(db_job_id, slurm_id)
                    elif slurm_id == "DRY_RUN":
                        log(f"  -> [DRY RUN] Would submit")
                        all_job_ids.append(f"DRY_{idx}")
                    else:
                        log(f"  -> Submission failed")
                        all_job_ids.append(f"FAILED_{idx}")

                    time.sleep(1)

                n_submitted = len([j for j in all_job_ids if not j.startswith('FAILED')])
                n_failed = len([j for j in all_job_ids if j.startswith('FAILED')])
                log(f"Submitted {n_submitted} jobs, skipped {n_failed} (download failures)")
                if pre_download and failed_models:
                    log(f"  Failed models: {sorted(failed_models)}")

            if args.once:
                log("Single iteration complete (--once). Exiting.")
                break

            hours = check_interval_seconds / 3600.0
            log(f"Sleeping for {hours} hours...\n")
            time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            log(f"ERROR in main loop: {e}. Backing off 30s.")
            import traceback
            traceback.print_exc()
            time.sleep(30)


if __name__ == "__main__":
    main()
