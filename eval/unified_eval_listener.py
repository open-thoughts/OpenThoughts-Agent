#!/usr/bin/env python3
"""
Unified Eval Listener v6 - Polls Supabase for models and submits SLURM eval jobs.

Based on v5, with disk-based resume replacing the v5 DB-based DaytonaError resume.
Scans the eval jobs directory for incomplete/error-heavy jobs and resubmits them
with the same run_tag so harbor auto-resumes (skips completed trials, retries failed ones).

Key v6 features over v5:
  - Disk-based resume: scans --jobs-dir for incomplete/errored job dirs
  - Persistent sliding-window batch_size across iterations
  - hf_overrides support in baseline model configs
  - Supabase queries wrapped in try/except for resilience
Uses unified_eval_harbor.sbatch as the SLURM job template.


===============================================================================
FLAG REFERENCE
===============================================================================

--- Preset & Dataset Selection ---

--preset, -p {aider,bfcl,medagentbench,gaia,financeagent,swebench,v2,tb2,v1}
    Load a named preset that bundles dataset, concurrency, error threshold, and
    other defaults tuned for a specific benchmark. CLI flags override any preset
    value. Almost all runs should start with a preset.

    Preset details:
      aider     Dataset: DCAgent2/aider_polyglot.       n_concurrent=32, error_threshold=10, thinking=on, config_yaml=no_override.
      bfcl      Dataset: DCAgent2/bfcl-parity.          n_concurrent=32, error_threshold=10, thinking=on, vllm_retries=20, config_yaml=no_override.
      medagentbench  Dataset: DCAgent/medagentbench.   n_concurrent=32, error_threshold=10, thinking=on, vllm_retries=20, config_yaml=no_override.
      gaia      Dataset: DCAgent/gaia_127.             n_concurrent=32, error_threshold=10, thinking=on, vllm_retries=20, config_yaml=no_override.
      financeagent  Dataset: DCAgent/financeagent_terminal.  n_concurrent=16 (SEC_EDGAR 10req/s cap; empirical p95 ~6.8 req/s aggregate at n=16, safe margin), error_threshold=10, thinking=on, vllm_retries=20, config_yaml=no_override.
      swebench  Dataset: DCAgent2/swebench-verified-*.   n_concurrent=32, error_threshold=15, thinking=on, vllm_retries=20,
                agent_parser=xml, gpu_mem=0.95, config_yaml=no_override. HF existence check on.
      v2        Dataset: DCAgent/dev_set_v2.             n_concurrent=32, error_threshold=10, thinking=on, vllm_retries=20.
      tb2       Dataset: DCAgent2/terminal_bench_2.      n_concurrent=32, error_threshold=10, thinking=on,
                gpu_mem=0.95, slurm_time=48h, config_yaml=no_override.
      v1        Dataset: DCAgent/dev_set_71_tasks.       n_concurrent=32, error_threshold=10, thinking=on, vllm_retries=20.

    Tuning: Pick the preset matching your benchmark. Override individual params
    with CLI flags (e.g. --n-concurrent 64 to double concurrency).

--datasets, -d <str>
    Comma- or space-separated list of HuggingFace dataset repos. Overrides the
    preset's dataset list. Use this for one-off evals against custom datasets.
    Example: --datasets "DCAgent/dev_set_v2,DCAgent2/terminal_bench_2"

--sbatch-script, -s <path>
    Path to the sbatch template. Default: unified_eval_harbor_v4.sbatch (or
    whatever the preset specifies). Only change this if you have a custom sbatch.


--- Model Filtering ---

--priority-file <path>
    Text file listing HuggingFace model names (org/model), one per line.
    Lines starting with # are comments; blank lines are ignored.
    File order = submission priority: earlier lines are submitted first.
    Hot-reloaded every iteration — edit the file without restarting the listener.

    Env: EVAL_LISTENER_PRIORITY_FILE

--priority-mode {filter_only,priority_first}    [default: filter_only]
    filter_only   — Only evaluate models IN the priority file. All others skipped.
    priority_first — Evaluate ALL models, but submit priority models first.

    Tuning: Use filter_only (default) when you have a curated list of models to
    evaluate. Use priority_first when you want to evaluate everything but ensure
    specific models get SLURM slots first.

    Env: EVAL_LISTENER_PRIORITY_MODE

--require-priority-list
    Safety flag. If set and no priority file is loaded (missing file or empty),
    the listener skips ALL models instead of evaluating everything. Prevents
    accidental mass submissions when a priority file path is misconfigured.

    Env: EVAL_LISTENER_REQUIRE_PRIORITY_LIST="1"

--blacklist-file <path>                          [v4 NEW]
    Text file listing models that should NEVER be submitted, same format as
    --priority-file (one model per line, # comments, blank lines ignored).
    Blacklist overrides priority: if a model appears in both files, it is blocked.
    Hot-reloaded every iteration, same as --priority-file.

    Tuning: Use this to permanently exclude known-bad models (e.g. broken
    checkpoints, models that consistently OOM, duplicates you don't want to
    re-evaluate). Faster than removing them from the priority file because
    the blacklist is checked first — no DB queries wasted on blocked models.

    Env: EVAL_LISTENER_BLACKLIST_FILE

--check-hf-exists
    Before submitting, validate that the model actually exists on HuggingFace Hub.
    Adds a network round-trip per model but prevents wasted SLURM jobs on typos
    or deleted models. The swebench preset enables this by default.

    Env: EVAL_LISTENER_CHECK_HF_EXISTS="1"


--- Timing & Lifecycle ---

--lookback-days <int>                            [default: 1000]
    How far back to query the Supabase `models` table (by creation_time).
    Priority models bypass this window — they are always fetched by name
    regardless of when they were added.

    Tuning: Keep this large (default 1000) to catch old models. Reduce only if
    DB queries are slow and you know all target models are recent.

    Env: EVAL_LISTENER_LOOKBACK_DAYS

--check-hours <float>                            [default: 4.0]
    Hours to sleep between iterations. Each iteration re-queries the DB, hot-
    reloads priority/blacklist files, and submits any new jobs.

    Tuning: For active development with frequent model uploads, use 1-2h.
    For stable production runs, 4-12h is fine. Ignored when --once is set.

    Env: EVAL_LISTENER_CHECK_HOURS

--stale-hours <int>                              [default: 24]
    A job in "Started" status older than this is considered stale and will be
    resubmitted. Covers cases where the sbatch job crashed without updating
    the DB to Finished.

    Tuning: Set to at least 1.5x your SLURM time limit. If --slurm-time is
    24:00:00, keep this at 24 (default). If you use --slurm-time 48:00:00
    (like tb2), bump to 48-72.

--stale-pending-hours <int>                      [default: 48]
    A job in "Pending" status older than this is considered stale. The listener
    will scancel the old SLURM job (if tracked) and resubmit.

    Tuning: Should be >= --stale-hours. Default of 48h gives Pending jobs extra
    time to get through the SLURM queue before being killed.


--- Sbatch / vLLM Parameters (passed to sbatch via env vars) ---

--n-concurrent <int>                             [default: 64, preset overrides]
    Number of concurrent Harbor evaluation jobs inside the sbatch. Controls how
    many sandbox tasks run in parallel against the vLLM server.

    Tuning: Depends on model size and GPU memory.
      - 7-8B models on GH200 (96GB): 32-64 is safe.
      - 32B models: 8-16 (higher causes vLLM queue buildup → AgentTimeoutError).
      - 131K context models: 4-8 (KV cache fills fast at high concurrency).
    If you see many AgentTimeoutErrors, reduce this. If eval is slow and vLLM
    GPU utilization is low, increase it.

--n-attempts <int>                               [default: 3]
    Number of retry attempts per Harbor task. If a task fails (e.g. sandbox
    timeout), Harbor retries it up to this many times.

    Tuning: 3 is good for most benchmarks. Raise to 5 for flaky benchmarks.
    Lowering to 1 speeds up runs but increases noise from transient failures.

--gpu-memory-util <float>                        [default: 0.9]
    Fraction of GPU memory allocated to vLLM via --gpu-memory-utilization.

    Tuning:
      - 0.90 (default): safe for 7-8B models on GH200 (96GB). Leaves headroom
        for GPU memory variance across nodes.
      - 0.95: used by swebench/tb2 presets for larger models or when you need
        maximum KV cache capacity. Risk: some GH200 nodes have slightly less
        available memory and will OOM at 0.95 (use --exclude in sbatch).
      - Never go above 0.95. Below 0.85 wastes memory.

--error-threshold <int>                          [default: 3, preset overrides]
    Maximum number of "invalid" errors allowed before the sbatch script aborts
    result upload. Invalid = any error type EXCEPT AgentTimeoutError,
    ContextLengthExceededError, SummarizationTimeout, SummarizationTimeoutError.

    Tuning: Controls quality gating. Low values (3) are strict — a few
    DaytonaErrors or unexpected crashes abort the upload. Higher values (10-15)
    are more tolerant, appropriate for benchmarks where some sandbox flakiness
    is expected.
      - aider: 3 (strict, small dataset)
      - v2/tb2: 10 (moderate, larger datasets with occasional flakes)
      - swebench: 15 (lenient, swebench sandboxes are flakier)

    --daytona-threshold is a backward-compatible alias for this flag.

--vllm-max-retries <int>                         [default: 5, preset overrides]
    Number of times the sbatch script retries starting the vLLM server.
    vLLM occasionally fails to start on first attempt (port conflicts,
    CUDA initialization issues).

    Tuning: 5 is fine for quick detection of real failures. Presets like v2
    and swebench use 20 for more resilience on busy clusters.

--agent-parser <str>                             [default: "" (none)]
    Parser type for Harbor agent output. Set to "xml" for swebench (which
    uses XML-structured agent responses). Leave empty for all other benchmarks.

    Tuning: Only change this if you're adding a new benchmark with a custom
    agent output format. The swebench preset sets this automatically.

--slurm-time <str>                               [default: "24:00:00"]
    SLURM wall-clock time limit for the sbatch job. Format: HH:MM:SS.

    Tuning: 24h is enough for most benchmarks. tb2 preset uses 48h because
    terminal_bench_2 tasks are longer-running. If jobs are hitting the time
    limit and getting killed, increase this and also bump --stale-hours.

--slurm-partition <str>                          [default: "gh"]
    SLURM partition to submit jobs to. On TACC, "gh" is the GH200 GPU partition.

--agent-name <str>                               [default: "terminus-2"]
    Agent name written to DB entries and used by Harbor for evaluation config.
    This determines which agent implementation Harbor uses to run the eval tasks.

--enable-thinking
    Enable thinking/reasoning blocks in vLLM model inference. Most presets
    enable this by default. Only disable if the model doesn't support thinking
    or you want to test non-thinking mode.

--upload-username <str>                          [default: current OS user]
    Username recorded in DB entries and result uploads. Auto-detected from
    the OS user if not specified.

    Env: EVAL_UPLOAD_USERNAME


--- v3 Enhancement: Per-Listener SLURM Throttle ---

--max-jobs-submitted <int>                       [default: 20]
    Maximum number of active SLURM jobs this listener instance is allowed to
    have running simultaneously. The listener tracks which SLURM job IDs it
    submitted and checks squeue to count only those still active.

    Tuning: This is PER-LISTENER, not global. Multiple listeners can run in
    parallel with independent budgets. Set based on your fair-share allocation:
      - Single listener: 10-20 is typical.
      - Multiple listeners: split your budget (e.g. v2=10, swebench=5).
    When the limit is reached, the listener queues submissions by priority
    order and drops the lowest-priority ones.

    Env: EVAL_LISTENER_MAX_JOBS


--- v3 Enhancement: Daytona Resource Pre-flight ---

--check-daytona-resources
    Enable Daytona API sandbox count check at startup and each iteration.
    If active sandboxes are at or above the limit, the listener skips that
    iteration entirely. Requires DAYTONA_API_KEY in environment.

    Tuning: Enable this in production to prevent overwhelming the Daytona
    sandbox pool. Not needed for small-scale or development runs.

--daytona-sandbox-limit <int>                    [default: 2000]
    Maximum expected active sandboxes. The listener skips submissions when
    the active count reaches this number.

--daytona-warning-buffer <float>                 [default: 0.9]
    Fraction of the sandbox limit at which a warning is logged. At 0.9 with
    limit=2000, warns when active sandboxes reach 1800.


--- v3 Enhancement: Model Retry Tracking ---

--track-model-retries
    Enable tracking of how many times each model has been started. Models
    exceeding the retry threshold are deprioritized (moved to end of the
    submission queue, not blocked entirely).

    Tuning: Enable this for long-running listeners to prevent repeatedly
    resubmitting models that keep failing. The sbatch script appends to the
    shared log when transitioning a job from Pending → Started.

--model-retry-threshold <int>                    [default: 5]
    Number of eval starts before a model is deprioritized. Deprioritized
    models are still submitted, just last in the queue (and may be dropped
    if --max-jobs-submitted truncates the list).

    Tuning: 3-5 for strict environments. Higher (10+) if transient failures
    are common and you want to give models more chances.

--eval-starts-log <path>                         [default: auto-generated]
    Path to the shared append-only log file where eval starts are recorded.
    Auto-generated with a benchmark+timestamp suffix if not specified.
    Multiple listeners using the same log file will share retry counts.

    Tuning: If you run multiple listeners for the same benchmark and want
    shared retry tracking, point them at the same log file.


--- v3 Enhancement: Timeout-Config-Sensitive Dedup ---

--timeout-aware
    Change job dedup logic to check model + benchmark + agent + timeout_multiplier
    instead of just model + benchmark. This allows running the same model with
    different timeout configurations without one blocking the other.

    Tuning: Enable when running A/B experiments with different timeout settings.
    When disabled (default), two listeners submitting the same model with
    different --timeout-multiplier values will conflict (one sees the other's
    job and skips).

--timeout-multiplier <float>                     [default: 1.0]
    Harbor timeout multiplier, passed to the sbatch job and stored in the DB
    job config. Values >1.0 give tasks more time; <1.0 makes them stricter.

    Tuning: Use with --timeout-aware for controlled experiments:
      --timeout-multiplier 0.25   (aggressive timeout, fast failures)
      --timeout-multiplier 1.0    (default)
      --timeout-multiplier 2.0    (lenient, for slow models)
      --timeout-multiplier 4.0    (very lenient, for debugging)


--- Execution Mode ---

--dry-run
    Preview mode: runs one full iteration (DB queries, filtering, status checks)
    but does NOT submit any sbatch jobs. Logs what WOULD be submitted. Implies
    --once. Use this to verify your flags before a real run.

    Env: EVAL_LISTENER_DRY_RUN="1"

--once
    Run a single iteration and exit. Useful for cron-triggered runs or one-shot
    submissions. Without this, the listener loops forever (sleeping --check-hours
    between iterations).

--verbose, -v
    Enable detailed logging: shows every model skipped (with reason), priority
    list contents, blacklist contents, and per-model DB status checks.

--log-file <path>
    Explicit log file path. Default: auto-generated in experiments/listener_logs/
    with a preset+timestamp name.

    Env: EVAL_LISTENER_LOG_DIR (for the directory)


===============================================================================
ENVIRONMENT VARIABLES (all optional, CLI args take precedence)
===============================================================================

  EVAL_LISTENER_LOOKBACK_DAYS         Days to look back for models (default: 1000)
  EVAL_LISTENER_CHECK_HOURS           Hours between iterations (default: 4.0)
  EVAL_LISTENER_SBATCH                SBATCH script to use
  EVAL_LISTENER_LOG_DIR               Log directory (default: experiments/listener_logs)
  EVAL_LISTENER_DATASETS              Comma/space/newline list of HF dataset repos
  EVAL_LISTENER_PRIORITY_FILE         Path to priority models file (hot-reloaded)
  EVAL_LISTENER_BLACKLIST_FILE        Path to blacklist models file (hot-reloaded) [v4]
  EVAL_LISTENER_DRY_RUN               "1" or "true" to enable dry run mode
  EVAL_LISTENER_REQUIRE_PRIORITY_LIST "1" or "true" to require priority list
  EVAL_LISTENER_PRIORITY_MODE         "filter_only" or "priority_first"
  EVAL_LISTENER_CHECK_HF_EXISTS       "1" or "true" to validate HF model existence
  EVAL_LISTENER_MAX_JOBS              Per-listener SLURM job limit (default: 20)
  EVAL_UPLOAD_USERNAME                Username for DB entries (default: OS user)
  DAYTONA_API_KEY                     Required for --check-daytona-resources


===============================================================================
QUICK START EXAMPLES
===============================================================================

  # Most common: evaluate priority models on dev_set_v2
  python unified_eval_listener_v4.py --preset v2 \\
    --priority-file v2_priority_models_richard.txt

  # Preview what would be submitted (no actual jobs)
  python unified_eval_listener_v4.py --preset v2 --dry-run --once \\
    --priority-file v2_priority_models_richard.txt --verbose

  # Block known-bad models
  python unified_eval_listener_v4.py --preset v2 \\
    --priority-file v2_priority_models_richard.txt \\
    --blacklist-file bad_models.txt

  # Full v3/v4 features enabled
  python unified_eval_listener_v4.py --preset v2 \\
    --priority-file v2_priority_models_richard.txt \\
    --blacklist-file bad_models.txt \\
    --error-threshold 10 --max-jobs-submitted 15 \\
    --check-daytona-resources \\
    --track-model-retries --model-retry-threshold 3 \\
    --timeout-aware --timeout-multiplier 2.0

  # Two listeners with independent SLURM budgets
  python unified_eval_listener_v4.py --preset v2 --max-jobs-submitted 10 &
  python unified_eval_listener_v4.py --preset swebench --max-jobs-submitted 5 &

  # A/B timeout experiment (requires --timeout-aware on both)
  python unified_eval_listener_v4.py --preset v2 --timeout-aware \\
    --timeout-multiplier 1.0 --max-jobs-submitted 10 &
  python unified_eval_listener_v4.py --preset v2 --timeout-aware \\
    --timeout-multiplier 2.0 --max-jobs-submitted 5 &
"""

import argparse
import getpass
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Add leaderboard utilities to path
# Add project root to path for database.unified_db imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Ensure ${DCFT} in baseline-yaml extra_args expands at submit time (see
# _build_env -> EVAL_VLLM_EXTRA_ARGS). load_cluster_config() may override
# this when paths.project_root is supplied via --cluster-config.
_DCFT_DEFAULT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
os.environ.setdefault("DCFT", _DCFT_DEFAULT)

from database.unified_db.utils import get_supabase_client, load_supabase_keys
from eval.presets import load_presets


# ---------------------------------------------------------------------------
# Secrets loading (Jupiter-specific: load ~/secrets.env at import time)
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
    # Alias SUPABASE_KEY -> SUPABASE_ANON_KEY if the latter is missing
    # (some secrets.env files use the shorter name)
    if os.environ.get("SUPABASE_KEY") and not os.environ.get("SUPABASE_ANON_KEY"):
        os.environ["SUPABASE_ANON_KEY"] = os.environ["SUPABASE_KEY"]
    try:
        load_supabase_keys()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Harbor config parsing -- extract eval config fields for dedup
# ---------------------------------------------------------------------------
def parse_harbor_eval_config(path: Optional[str]) -> Dict:
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
    result: Dict = {}
    if cfg.get("timeout_multiplier") is not None:
        result["timeout_multiplier"] = float(cfg["timeout_multiplier"])
    env_cfg = cfg.get("environment") or {}
    for key in ("override_cpus", "override_memory_mb", "override_storage_mb"):
        if env_cfg.get(key) is not None:
            result[key] = int(env_cfg[key])
    return result


# ---------------------------------------------------------------------------
# Baseline model config mapping -- per-model vLLM overrides
# ---------------------------------------------------------------------------
_BASELINE_MODEL_CONFIGS: Optional[Dict[str, Dict]] = None
_BASELINE_MODEL_PATTERNS: Optional[List[Dict]] = None


def load_baseline_model_configs(path: Optional[str]) -> Dict[str, Dict]:
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

        # Start with per-model entries
        per_model = data.get("models", {})

        # Expand groups: each group has a "models" list + shared config fields.
        # Group config is the base; per-model entries are merged on top (override wins).
        expanded: Dict[str, Dict] = {}
        for group in data.get("groups", []):
            model_names = group.get("models", [])
            shared_cfg = {k: v for k, v in group.items() if k != "models"}
            for name in model_names:
                expanded[name] = dict(shared_cfg)  # copy so mutations are isolated

        # Merge per-model overrides on top of group defaults
        for name, overrides in per_model.items():
            if name in expanded:
                expanded[name].update(overrides)
            else:
                expanded[name] = dict(overrides)

        _BASELINE_MODEL_CONFIGS = expanded
        _BASELINE_MODEL_PATTERNS = data.get("patterns", [])
        n_groups = len(data.get("groups", []))
        log(f"Loaded {len(_BASELINE_MODEL_CONFIGS)} baseline model config(s) "
            f"({n_groups} group(s), {len(per_model)} override(s)) and "
            f"{len(_BASELINE_MODEL_PATTERNS)} pattern(s) from {path}")
    except Exception as e:
        log(f"WARNING: failed to load baseline model configs from {path}: {e}")
        _BASELINE_MODEL_CONFIGS = {}
        _BASELINE_MODEL_PATTERNS = []

    return _BASELINE_MODEL_CONFIGS


def _match_pattern_config(hf_model: str) -> Optional[Dict]:
    """Try to match a model name against pattern-based configs.

    Patterns are checked in order; first match wins.
    Each pattern has a 'match' field (regex or substring) and config fields.
    """
    if not _BASELINE_MODEL_PATTERNS:
        return None
    for pattern_entry in _BASELINE_MODEL_PATTERNS:
        pattern = pattern_entry.get("match", "")
        if not pattern:
            continue
        if re.search(pattern, hf_model):
            return {k: v for k, v in pattern_entry.items() if k != "match"}
    return None


def get_vllm_env_overrides(hf_model: str, configs: Dict[str, Dict]) -> Dict[str, str]:
    """Get vLLM env var overrides for a model from the baseline config mapping.

    Tries exact model name match first, then falls back to pattern matching.
    Returns dict of EVAL_VLLM_* env vars to pass to the eval script.
    """
    match_source = None
    cfg = configs.get(hf_model)
    if cfg:
        match_source = "exact/group"
    else:
        cfg = _match_pattern_config(hf_model)
        if cfg:
            match_source = "pattern"
    if not cfg:
        return {}

    log(f"  Baseline config [{match_source}] for {hf_model}: {cfg}")

    env: Dict[str, str] = {}
    if cfg.get("tensor_parallel_size") is not None:
        env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] = str(cfg["tensor_parallel_size"])
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
        # Expand ${DCFT}/... etc. so absolute paths flow into the sbatch
        # (vLLM does not do shell expansion on its own arguments).
        env["EVAL_VLLM_EXTRA_ARGS"] = os.path.expandvars(cfg["extra_args"])
    if cfg.get("hf_overrides"):
        env["EVAL_VLLM_HF_OVERRIDES"] = cfg["hf_overrides"]

    return env


def get_conda_env_override(hf_model: str, configs: Dict[str, Dict]) -> Optional[str]:
    """Get conda_env override for a model from the baseline config mapping.

    Tries exact/group match first, then pattern match. Returns the conda_env
    string (e.g. "otagent2") or None if no override is configured.
    """
    cfg = configs.get(hf_model)
    if not cfg:
        cfg = _match_pattern_config(hf_model)
    if cfg and cfg.get("conda_env"):
        return cfg["conda_env"]
    return None


# ---------------------------------------------------------------------------
# API model config mapping -- per-model serving config for hosted endpoints
# (Together AI, OpenAI, Anthropic ...). Models matching this mapping are
# dispatched to eval/unified_eval_api_harbor.sbatch instead of the regular
# vLLM sbatch — no GPU allocation, no local vLLM server.
# ---------------------------------------------------------------------------
_API_MODEL_CONFIGS: Optional[Dict[str, Any]] = None


def load_api_model_configs(path: Optional[str]) -> Dict[str, Any]:
    """Load per-model API serving config from YAML.

    Returns a dict with keys:
      - api_models: dict[model_name -> {api_base, api_key_env, agent_kwargs, n_concurrent_cap}]
      - patterns: list[{regex, api_base, api_key_env, n_concurrent_cap}]
      - preset_n_concurrent_caps: dict[preset_name -> int]

    If the file is missing or malformed, returns an empty dict (i.e. no
    models match → no behavior change for existing fires).
    """
    global _API_MODEL_CONFIGS
    if _API_MODEL_CONFIGS is not None:
        return _API_MODEL_CONFIGS

    if not path or not os.path.isfile(path):
        _API_MODEL_CONFIGS = {}
        return _API_MODEL_CONFIGS

    try:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _API_MODEL_CONFIGS = {
            "api_models": data.get("api_models", {}) or {},
            "patterns": data.get("patterns", []) or [],
            "preset_n_concurrent_caps": data.get("preset_n_concurrent_caps", {}) or {},
        }
        log(f"Loaded {len(_API_MODEL_CONFIGS['api_models'])} API model entries "
            f"and {len(_API_MODEL_CONFIGS['patterns'])} pattern(s) from {path}")
    except Exception as e:
        log(f"WARNING: failed to load API model configs from {path}: {e}")
        _API_MODEL_CONFIGS = {}
    return _API_MODEL_CONFIGS


def get_api_config(hf_model: str, configs: Optional[Dict[str, Any]] = None) -> Optional[Dict]:
    """Resolve the API serving config for a model name, or None if it should
    use the regular vLLM path.

    Lookup order:
      1. Exact match in api_models.
      2. Regex pattern fallback in patterns (first match wins).
      3. None (model is not API-served).

    Returns dict with keys: api_base, api_key_env, agent_kwargs (dict),
    n_concurrent_cap (int).
    """
    cfg_root = configs if configs is not None else (_API_MODEL_CONFIGS or {})
    if not cfg_root:
        return None

    api_models = cfg_root.get("api_models", {})
    cfg = api_models.get(hf_model)
    if not cfg:
        for pat in cfg_root.get("patterns", []):
            regex = pat.get("regex")
            if regex and re.search(regex, hf_model):
                cfg = {k: v for k, v in pat.items() if k != "regex"}
                break
    if not cfg:
        return None

    return {
        "api_base": cfg.get("api_base"),
        "api_key_env": cfg.get("api_key_env"),
        "agent_kwargs": cfg.get("agent_kwargs", {}) or {},
        "n_concurrent_cap": int(cfg.get("n_concurrent_cap", 32)),
        # litellm_model is what LiteLLM expects (with provider prefix);
        # falls back to the lookup key when omitted.
        "litellm_model": cfg.get("litellm_model") or hf_model,
    }


def _build_api_agent_kwargs_string(agent_kwargs: Dict[str, Any]) -> str:
    """Serialize a dict of agent kwargs into newline-separated `key=value`
    pairs for EVAL_API_AGENT_KWARGS. The sbatch splits on newlines and adds
    each as one --agent-kwarg arg, so JSON values (with quotes/braces/commas)
    survive without shell quoting hell.
    """
    if not agent_kwargs:
        return ""
    return "\n".join(f"{k}={v}" for k, v in agent_kwargs.items())


# ---------- v6: Disk-Based Resume Scanner ----------

# Infrastructure errors that harbor's resume filters will retry
INFRA_ERROR_TYPES = {
    "DaytonaError",
    "DaytonaAuthenticationError",
    "DaytonaAuthorizationError",
    "DaytonaNotFoundError",
    "EnvironmentStartTimeoutError",
    "DaytonaRateLimitError",
    "CancelledError",
    "SandboxBuildFailedError",
    "AgentEnvironmentTimeoutError",
}


def _parse_job_dir(job_dir: Path) -> Optional[Dict]:
    """Parse a harbor job directory, extracting model/dataset/progress info.

    Returns dict with keys: run_tag, hf_model, dataset, n_completed, n_total,
    finished_at, infra_errors, total_errors, db_job_id, slurm_job_id, resume_count.
    Returns None if dir is not a valid harbor job dir.
    """
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None

    run_tag = job_dir.name
    info: Dict = {
        "run_tag": run_tag,
        "hf_model": None,
        "dataset": None,
        "n_completed": 0,
        "n_total": 0,
        "finished_at": None,
        "infra_errors": 0,
        "total_errors": 0,
        "db_job_id": None,
        "slurm_job_id": None,
        "resume_count": 0,
    }

    # Parse config.json for model and dataset
    try:
        import json as _json
        config = _json.loads(config_path.read_text())
        agents = config.get("agents", [])
        if agents and isinstance(agents, list):
            model_name = agents[0].get("model_name", "")
            # Strip "hosted_vllm/" prefix
            if model_name.startswith("hosted_vllm/"):
                model_name = model_name[len("hosted_vllm/"):]
            info["hf_model"] = model_name or None
        datasets = config.get("datasets", [])
        if datasets and isinstance(datasets, list):
            ds_path = datasets[0].get("path", "")
            if ds_path:
                # Extract dataset name from path: /e/.../DCAgent_dev_set_v2 → DCAgent/dev_set_v2
                ds_name = Path(ds_path).name  # e.g., "DCAgent_dev_set_v2"
                # Convert first underscore to slash (org/name convention)
                parts = ds_name.split("_", 1)
                if len(parts) == 2:
                    info["dataset"] = f"{parts[0]}/{parts[1]}"
                else:
                    info["dataset"] = ds_name
    except Exception:
        pass

    # Parse meta.env for DB_JOB_ID, SLURM_JOB_ID, RESUME_COUNT
    meta_path = job_dir / "meta.env"
    if meta_path.exists():
        try:
            for line in meta_path.read_text().splitlines():
                if line.startswith("DB_JOB_ID="):
                    info["db_job_id"] = line.split("=", 1)[1].strip() or None
                elif line.startswith("SLURM_JOB_ID="):
                    info["slurm_job_id"] = line.split("=", 1)[1].strip() or None
                elif line.startswith("RESUME_COUNT="):
                    try:
                        info["resume_count"] = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("MODEL=") and not info["hf_model"]:
                    info["hf_model"] = line.split("=", 1)[1].strip() or None
        except Exception:
            pass

    # Parse result.json for progress
    result_path = job_dir / "result.json"
    if result_path.exists():
        try:
            import json as _json
            result = _json.loads(result_path.read_text())
            info["n_total"] = result.get("n_total_trials", 0)
            stats = result.get("stats", {})
            info["n_completed"] = stats.get("n_trials", 0)
            info["finished_at"] = result.get("finished_at")

            # Count infrastructure errors
            infra_count = 0
            total_err_count = 0
            for eval_data in stats.get("evals", {}).values():
                for exc_type, ids in eval_data.get("exception_stats", {}).items():
                    n = len(ids) if isinstance(ids, list) else 1
                    total_err_count += n
                    if exc_type in INFRA_ERROR_TYPES:
                        infra_count += n
            info["infra_errors"] = infra_count
            info["total_errors"] = total_err_count
        except Exception:
            pass

    return info


def scan_jobs_dir_for_resume(
    jobs_dir: str,
    dataset_prefixes: List[str],
    active_slurm_ids: Set[str],
    infra_error_threshold: int = 3,
    max_resume_count: int = 5,
) -> List[Dict]:
    """Scan eval jobs directory for jobs that need to be resumed.

    Args:
        jobs_dir: Path to the eval jobs directory
        dataset_prefixes: List of dataset name prefixes to filter (e.g., ["dev_set_v2"])
        active_slurm_ids: Set of SLURM job IDs currently in squeue
        infra_error_threshold: Min infra errors to trigger resume for PARTIAL jobs
        max_resume_count: Skip dirs with RESUME_COUNT >= this (prevent infinite loops)

    Returns:
        List of dicts with keys: hf_model, dataset, run_tag, reason, db_job_id
    """
    jobs_path = Path(jobs_dir)
    if not jobs_path.is_dir():
        log(f"[v6-resume] Jobs dir not found: {jobs_dir}")
        return []

    # Build prefix patterns from dataset names
    # "DCAgent/dev_set_v2" → "dev_set_v2_"
    # Must normalize hyphens/dots to underscores to match generate_run_tag() output
    dir_prefixes = []
    for ds in dataset_prefixes:
        # Dataset format: "DCAgent/dev_set_v2" or "DCAgent2/terminal_bench_2"
        ds_short = ds.split("/")[-1] if "/" in ds else ds
        ds_safe = ds_short.replace("-", "_").replace(".", "_")
        dir_prefixes.append(f"{ds_safe}_")

    candidates = []
    scanned = 0
    skipped_active = 0
    skipped_done = 0
    skipped_resume_limit = 0

    for entry in sorted(jobs_path.iterdir()):
        if not entry.is_dir():
            continue

        # Filter by dataset prefix
        if not any(entry.name.startswith(p) for p in dir_prefixes):
            continue

        info = _parse_job_dir(entry)
        if info is None:
            continue
        scanned += 1

        # Skip if SLURM job still running
        if info["slurm_job_id"] and info["slurm_job_id"] in active_slurm_ids:
            skipped_active += 1
            continue

        # Skip if resume count too high
        if info["resume_count"] >= max_resume_count:
            skipped_resume_limit += 1
            continue

        # Classify job state
        n_completed = info["n_completed"]
        n_total = info["n_total"]
        finished_at = info["finished_at"]
        infra_errors = info["infra_errors"]

        reason = None

        if n_total == 0 and not (jobs_path / entry.name / "result.json").exists():
            # EARLY_KILL: killed before any trial completed
            reason = f"early_kill (no result.json, resume #{info['resume_count']+1})"
        elif n_completed < n_total and finished_at is None:
            # INCOMPLETE: SLURM killed mid-run
            reason = f"incomplete ({n_completed}/{n_total} trials, resume #{info['resume_count']+1})"
        elif n_completed < n_total and finished_at is not None:
            # PARTIAL: harbor finished but some trials failed
            if infra_errors > infra_error_threshold:
                reason = f"partial ({n_completed}/{n_total}, {infra_errors} infra errors, resume #{info['resume_count']+1})"
        elif n_completed == n_total:
            # DONE: all trials completed
            if infra_errors > infra_error_threshold:
                reason = f"done_with_errors ({n_completed}/{n_total}, {infra_errors} infra errors, resume #{info['resume_count']+1})"
            else:
                skipped_done += 1
                continue
        else:
            continue

        if reason and info["hf_model"]:
            candidates.append({
                "hf_model": info["hf_model"],
                "dataset": info["dataset"],
                "run_tag": info["run_tag"],
                "reason": f"v6_resume: {reason}",
                "db_job_id": info["db_job_id"],
            })

    log(f"[v6-resume] Scanned {scanned} job dirs: "
        f"{len(candidates)} resume candidates, "
        f"{skipped_active} still running, "
        f"{skipped_done} completed, "
        f"{skipped_resume_limit} at resume limit")

    return candidates


# ---------- Preset Definitions ----------
# Presets are defined as one YAML per preset in eval/presets/ and loaded here
# via the shared eval.presets catalog (the same catalog the Iris launcher
# consumes). Each preset is a flat mapping that can configure:
#   - datasets: list of HF dataset repos
#   - sbatch_script: sbatch script to use (default: unified_eval_harbor_v4.sbatch)
#   - log_suffix: suffix for log file
#   - check_hf_exists: validate model exists on HuggingFace
#   - n_concurrent: Harbor --n-concurrent (default: 64)
#   - n_attempts: Harbor --n-attempts (default: 3)
#   - gpu_memory_util: VLLM --gpu-memory-utilization (default: 0.9)
#   - error_threshold: Max invalid errors before abort (default: 3)
#   - vllm_max_retries: VLLM startup retries (default: 5)
#   - agent_parser: Agent parser type (default: "", use "xml" for swebench)
#   - slurm_time: SLURM time limit (default: "24:00:00")
# To add/edit a preset, change the YAML in eval/presets/ — no code change here.
PRESETS: Dict[str, Dict] = load_presets()

# ---------- Cluster Config ----------
_CLUSTER_CONFIG_REQUIRED_KEYS = ["cluster_name", "slurm_partition", "paths"]
_CLUSTER_CONFIG_REQUIRED_PATHS = ["eval_jobs_dir", "sbatch_script"]

# Global cluster config (set by --cluster-config, None = use hardcoded defaults)
_CLUSTER_CONFIG: Optional[Dict[str, Any]] = None


def load_cluster_config(path: str) -> Dict[str, Any]:
    """Load and validate a cluster config YAML.

    Returns the parsed config dict.  Raises SystemExit on validation failure.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        print(f"ERROR: Cluster config not found: {path}")
        sys.exit(2)

    with open(path) as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        print(f"ERROR: Cluster config must be a YAML mapping, got {type(cfg).__name__}")
        sys.exit(2)

    # Expand $USER / ${USER} and ~ in all string values (paths, conda env dirs, etc.)
    def _expand(obj):
        if isinstance(obj, str):
            return os.path.expandvars(os.path.expanduser(obj))
        elif isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj
    cfg = _expand(cfg)

    for key in _CLUSTER_CONFIG_REQUIRED_KEYS:
        if key not in cfg:
            print(f"ERROR: Cluster config missing required key: {key}")
            sys.exit(2)

    paths = cfg.get("paths", {})
    for key in _CLUSTER_CONFIG_REQUIRED_PATHS:
        if key not in paths:
            print(f"ERROR: Cluster config paths.{key} is required")
            sys.exit(2)

    # Cluster config wins over the script-relative DCFT default so that
    # baseline-yaml extra_args ('${DCFT}/eval/configs/...') expand to the
    # cluster's checkout root, not whatever directory the listener was
    # launched from.
    if paths.get("project_root"):
        os.environ["DCFT"] = paths["project_root"]

    return cfg


def _cc_get(key: str, default: Any = None) -> Any:
    """Get a top-level key from the cluster config, or *default* if not loaded."""
    if _CLUSTER_CONFIG is None:
        return default
    return _CLUSTER_CONFIG.get(key, default)


def _cc_path(key: str, default: Any = None) -> Any:
    """Get a paths.* key from the cluster config, or *default*."""
    if _CLUSTER_CONFIG is None:
        return default
    return _CLUSTER_CONFIG.get("paths", {}).get(key, default)


# ---------- Constants ----------
HF_URL_RE = re.compile(r'https?://(?:www\.)?huggingface\.co/([^/\s]+)/([^/\s#?]+)')
JOB_STATUS_PENDING = "Pending"
JOB_STATUS_STARTED = "Started"
JOB_STATUS_FINISHED = "Finished"
JOB_STATUS_FAILED = "Failed"
DEFAULT_STALE_JOB_HOURS = 24
DEFAULT_STALE_PENDING_HOURS = 48
DEFAULT_LOOKBACK_DAYS = 1000
DEFAULT_CHECK_HOURS = 4.0
DEFAULT_LOG_DIR = "experiments/listener_logs"

# Sbatch parameter defaults
DEFAULT_N_CONCURRENT = 64
DEFAULT_N_ATTEMPTS = 3
DEFAULT_GPU_MEMORY_UTIL = 0.9
DEFAULT_ERROR_THRESHOLD = 10
DEFAULT_VLLM_MAX_RETRIES = 10
DEFAULT_AGENT_PARSER = ""
DEFAULT_SLURM_TIME = "12:00:00"
DEFAULT_AGENT_NAME = "terminus-2"
DEFAULT_SLURM_PARTITION = "booster"
DEFAULT_SLURM_ACCOUNT = ""  # empty = use sbatch header default
DEFAULT_ENABLE_THINKING = False
DEFAULT_TP_SIZE = 1
DEFAULT_SBATCH_SCRIPT = "eval/unified_eval_harbor.sbatch"

# Fallback defaults (used when no --cluster-config is provided).
# Empty strings force explicit cluster config — no implicit Jupiter defaults.
_FALLBACK_EVAL_JOBS_DIR = ""
_FALLBACK_HF_CACHE = ""
_FALLBACK_EVAL_LOGS_DIR = "eval/logs"

# Conda env paths: env name → prefix directory (passed as OTAGENT_DIR to sbatch)
# Overridden by cluster_config["conda_envs"] when --cluster-config is used.
CONDA_ENV_PATHS: Dict[str, str] = {}

# Dataset repo name → short benchmark tag for SLURM job names (squeue readability)
_BENCH_SHORT: Dict[str, str] = {
    "dev_set_v2": "v2",
    "swebench-verified-random-100-folders": "swe",
    "terminal_bench_2": "tb2",
    "swebench-verified": "swefull",
    "aider_polyglot": "aider",
    "bfcl-parity": "bfcl",
    "medagentbench": "medagent",
    "gaia_127": "gaia",
    "financeagent_terminal": "finance",
    "dev_set_71_tasks": "v1",
}

# Enhancement 2: SLURM job submission throttle
DEFAULT_MAX_JOBS_SUBMITTED = 20

# Enhancement 3: Daytona resource pre-flight check
DEFAULT_DAYTONA_SANDBOX_LIMIT = 2000
DEFAULT_DAYTONA_WARNING_BUFFER = 0.9

# Enhancement 5: Timeout-config-sensitive dedup
DEFAULT_TIMEOUT_MULTIPLIER = 1.0


# ---------- Configuration ----------
@dataclass
class ListenerConfig:
    """Configuration for the eval listener.

    Core fields:
        datasets             HF dataset repos to evaluate against.
        sbatch_script        Path to the sbatch script to submit.
        priority_models      Ordered list of HF model names from the priority file.
                             File order = submission priority (first = highest).
        priority_file        Path to the priority file (hot-reloaded each iteration).

    Sbatch parameters (forwarded to sbatch via env vars):
        n_concurrent         Harbor --n-concurrent.
        n_attempts           Harbor --n-attempts.
        gpu_memory_util      VLLM --gpu-memory-utilization.
        error_threshold      Max invalid errors before aborting upload (v3 Enhancement 1).
                             Replaces v2's daytona_threshold. Env var kept as
                             EVAL_DAYTONA_THRESHOLD for sbatch backward compat.
        agent_name           Agent name for harbor and DB entries.
        timeout_multiplier   Harbor timeout multiplier (v3 Enhancement 5).

    v3 enhancement fields:
        max_jobs_submitted       Per-listener SLURM job limit (Enhancement 2).
                                 Each listener tracks its own submitted job IDs and
                                 only counts those still active in squeue.
        check_daytona_resources  Enable Daytona API pre-flight check (Enhancement 3).
        daytona_sandbox_limit    Max expected active sandboxes for pre-flight check.
        daytona_warning_buffer   Fraction of limit to trigger warning (e.g. 0.95).
        timeout_aware            Enable config-sensitive job dedup (Enhancement 5).
    """
    datasets: List[str]
    sbatch_script: str
    log_file: Optional[Path]
    lookback_days: int
    check_interval_hours: float
    stale_job_hours: int
    stale_pending_hours: int
    priority_file: Optional[str]
    require_priority_list: bool
    priority_models: List[str]
    check_hf_exists: bool
    dry_run: bool
    run_once: bool
    verbose: bool
    # Priority mode: "filter_only" (skip non-priority) or "priority_first" (all models, priority first)
    priority_mode: str = "filter_only"
    # Sbatch parameters (passed to sbatch via env vars)
    n_concurrent: int = DEFAULT_N_CONCURRENT
    n_attempts: int = DEFAULT_N_ATTEMPTS
    gpu_memory_util: float = DEFAULT_GPU_MEMORY_UTIL
    error_threshold: int = DEFAULT_ERROR_THRESHOLD
    vllm_max_retries: int = DEFAULT_VLLM_MAX_RETRIES
    agent_parser: str = DEFAULT_AGENT_PARSER
    slurm_time: str = DEFAULT_SLURM_TIME
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    agent_name: str = DEFAULT_AGENT_NAME
    slurm_partition: str = DEFAULT_SLURM_PARTITION
    slurm_account: str = DEFAULT_SLURM_ACCOUNT
    tp_size: int = DEFAULT_TP_SIZE
    dp_size: int = 1  # vLLM native data-parallel replicas (total GPUs = tp_size * dp_size)
    upload_username: str = ""
    log_prefix: str = "[unified-eval-listener-v6]"
    # v3 Enhancement 2: Per-listener SLURM throttle
    max_jobs_submitted: int = DEFAULT_MAX_JOBS_SUBMITTED
    # v3 Enhancement 3: Daytona pre-flight
    check_daytona_resources: bool = False
    daytona_sandbox_limit: int = DEFAULT_DAYTONA_SANDBOX_LIMIT
    daytona_warning_buffer: float = DEFAULT_DAYTONA_WARNING_BUFFER
    # v3 Enhancement 5: Timeout-config-sensitive dedup
    timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER
    timeout_aware: bool = False
    # Config YAML for harbor (overrides vs no-overrides)
    config_yaml: str = "dcagent_eval_config.yaml"
    # Max output tokens override (None = use sbatch default 16384)
    max_output_tokens: Optional[int] = None
    # Model blacklist
    blacklist_file: Optional[str] = None
    blacklisted_models: Set[str] = field(default_factory=set)
    # Daytona auto_snapshot: None = use YAML config default, True/False = override
    auto_snapshot: Optional[bool] = None
    # Comma-separated KEY=VALUE pairs forwarded into the Daytona sandbox via --ae
    agent_envs: Optional[str] = None
    # Pinggy tunnel config (for installed agents that run in Daytona sandbox)
    pinggy_url: Optional[str] = None
    pinggy_token: Optional[str] = None
    # Per-model vLLM overrides (baseline model configs)
    baseline_model_configs: Optional[str] = None
    # Per-model API serving config (Together AI / OpenAI / Anthropic)
    api_model_config: Optional[str] = None
    # API sbatch script (used when get_api_config() matches a model)
    api_sbatch_script: str = "eval/unified_eval_api_harbor.sbatch"
    # Harbor config path
    harbor_config: Optional[str] = None
    # Parsed eval config from harbor YAML (for config-aware dedup)
    eval_config: Dict = field(default_factory=dict)
    # Pre-download model weights before submitting jobs
    pre_download: bool = False
    # Sliding-window batch dependencies
    batch_size: Optional[int] = None
    # Conda env selector (otagent / otagent2)
    conda_env: str = "otagent"
    # v6: Disk-based resume
    jobs_dirs: List[str] = field(default_factory=list)  # Set from CLI or EVAL_JOBS_DIR env var
    enable_disk_resume: bool = True
    resume_infra_error_threshold: int = 10
    max_resume_count: int = 5
    force_reeval: bool = False  # Bypass DB status check (submit even if Finished/Started)
    resume_only: bool = False  # Only submit resume jobs, skip fresh submissions
    submission_delay: float = 1.0  # Seconds to sleep between sbatch submissions
    stagger_delay: int = 0  # Minutes between job starts via SLURM after: dependency chain (0 = disabled)
    chain_batch_size: int = 1  # Jobs per stagger batch (1 = every job waits, 10 = fire 10 then wait)
    pack_jobs: bool = False  # Pack multiple jobs onto same node via --nodelist
    # DP: data-parallel multi-node eval
    dp_nodes: int = 0  # 0 = single-node (default), >0 = use DP sbatch with N nodes
    dp_sbatch_script: str = "eval/unified_eval_harbor_dp.sbatch"
    # Inherit: seed _submitted_jobs from previous listener logs
    inherit_log: Optional[List[str]] = None
    # Cluster config (loaded from --cluster-config YAML)
    cluster_config: Optional[Dict[str, Any]] = None

    @property
    def check_interval_seconds(self) -> int:
        return int(self.check_interval_hours * 60 * 60)


# ---------- Logging ----------
_LOG_FILE: Optional[Path] = None
_VERBOSE: bool = False


def set_log_file(path: Optional[Path]) -> None:
    global _LOG_FILE
    _LOG_FILE = path


def log(msg: str, prefix: str = "[unified-eval-listener-v6]", verbose_only: bool = False) -> None:
    """Log a message to stdout and optionally to file.

    If verbose_only=True, the message is only emitted when _VERBOSE is set.
    """
    if verbose_only and not _VERBOSE:
        return
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
def load_priority_models(filepath: Optional[str]) -> List[str]:
    """
    Load priority models from a text file, preserving file order as rank.

    File order determines submission priority: models listed earlier are
    submitted first. When the per-listener SLURM job limit truncates the
    submission list, higher-priority (earlier) models are kept.

    File format:
      - One model per line (HuggingFace format: org/model)
      - Lines starting with # are comments
      - Blank lines are ignored

    Returns:
        Ordered list of model names (duplicates removed, order preserved).
        Empty list if file is missing or empty.
    """
    if not filepath:
        return []

    path = Path(filepath)
    if not path.exists():
        log(f"Priority file not found: {filepath}")
        return []

    models: List[str] = []
    seen: Set[str] = set()
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                if line not in seen:
                    seen.add(line)
                    models.append(line)
        log(f"Loaded {len(models)} model(s) from priority file: {filepath}")
        return models
    except Exception as e:
        log(f"ERROR reading priority file {filepath}: {e}")
        return []


# ---------- Model Blacklist Loading ----------
def load_blacklist(filepath: Optional[str]) -> Set[str]:
    """Load blacklisted models from a text file. Same format as priority file."""
    return set(load_priority_models(filepath))


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
    """Fetch recent models from Supabase within the lookback window.

    Filters out:
      - Models with created_by == "precomputed_hf"
      - Models with a non-empty "duplicate_of" field (v3: prevents duplicate
        eval submissions when the same HF model appears under multiple DB rows)
    """
    client = get_supabase_client()
    since = _iso(datetime.now(timezone.utc) - timedelta(days=days))
    try:
        resp = _time_filters(client.table('models').select('*'), since).execute()
        rows = list(resp.data or [])
    except Exception as e:
        log(f"ERROR: failed querying models by time: {e}")
        return []

    # Filter out precomputed models and duplicates
    out: List[Dict] = []
    skipped_dupes = 0
    for r in rows:
        if r.get("created_by") == "precomputed_hf":
            continue
        if r.get("duplicate_of"):
            skipped_dupes += 1
            continue
        out.append(r)
    if skipped_dupes:
        log(f"Filtered out {skipped_dupes} duplicate model(s) (duplicate_of set)")
    return out


def fetch_priority_models(priority_names: List[str]) -> List[Dict]:
    """Fetch models by name from Supabase, bypassing the lookback window.

    This ensures priority models are always evaluated even if they were
    registered long ago (outside the lookback window).

    Filters out:
      - Models with created_by == "precomputed_hf"
      - Models with a non-empty "duplicate_of" field
    """
    if not priority_names:
        return []

    client = get_supabase_client()
    try:
        resp = (
            client.table('models')
            .select('*')
            .in_('name', priority_names)
            .execute()
        )
        rows = list(resp.data or [])
    except Exception as e:
        log(f"ERROR: failed querying priority models by name: {e}")
        return []

    out: List[Dict] = []
    for r in rows:
        if r.get("created_by") == "precomputed_hf":
            continue
        if r.get("duplicate_of"):
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
    """Check if a job exists for (model_id, benchmark_id) and its status.

    Delegates to check_job_status_v3 (single-ID, non-timeout-aware path).
    Kept as a thin wrapper for backward compatibility with callers that don't
    need timeout-aware or duplicate-group queries.
    """
    return check_job_status_v3(model_id, benchmark_id)


# ---------- Cross-Duplicate Aggregation ----------
_DUP_GROUP_CACHE: Dict[str, List[str]] = {}


def get_duplicate_group_ids(table: str, entity_id: str) -> List[str]:
    """Get all IDs in the duplicate group for a model or benchmark.

    Given an entity_id, finds the canonical ID and all its duplicates.
    - If entity has duplicate_of set, canonical = duplicate_of
    - Otherwise canonical = entity_id
    - Then finds all rows WHERE duplicate_of = canonical_id
    - Returns [canonical_id] + [all duplicate IDs]

    Results are cached per (table, entity_id).
    """
    cache_key = f"{table}:{entity_id}"
    if cache_key in _DUP_GROUP_CACHE:
        return _DUP_GROUP_CACHE[cache_key]

    try:
        client = get_supabase_client()

        # Step 1: Find the canonical ID
        resp = client.table(table).select('id,duplicate_of').eq('id', entity_id).limit(1).execute()
        rows = resp.data or []
        if not rows:
            _DUP_GROUP_CACHE[cache_key] = [entity_id]
            return [entity_id]

        canonical_id = rows[0].get('duplicate_of') or entity_id

        # Step 2: Find all duplicates of the canonical
        resp2 = client.table(table).select('id').eq('duplicate_of', canonical_id).execute()
        dup_ids = [r['id'] for r in (resp2.data or [])]

        group = list(set([canonical_id] + dup_ids))
        # Cache for all members of the group
        for gid in group:
            _DUP_GROUP_CACHE[f"{table}:{gid}"] = group
        return group

    except Exception as e:
        log(f"WARNING: Failed to get duplicate group for {table}/{entity_id}: {e}")
        _DUP_GROUP_CACHE[cache_key] = [entity_id]
        return [entity_id]


# ---------- v3 Enhancement 5: Timeout-Config-Sensitive Job Dedup ----------
def check_job_status_v3(
    model_id: str,
    benchmark_id: Optional[str],
    timeout_aware: bool = False,
    agent_name: str = DEFAULT_AGENT_NAME,
    timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER,
    duplicate_model_ids: Optional[List[str]] = None,
    duplicate_benchmark_ids: Optional[List[str]] = None,
) -> Tuple[bool, Optional[str], Optional[datetime], Optional[datetime], Optional[str]]:
    """
    Check if a job exists for (model_id, benchmark_id) and its status.

    When timeout_aware=True, filters to only match jobs with the same
    agent_name and timeout_multiplier in their config.

    When duplicate_model_ids/duplicate_benchmark_ids are provided, queries
    across the entire duplicate group using .in_() instead of .eq().

    Returns:
        (job_exists, job_status, started_at, submitted_at, slurm_job_id)
    """
    if not benchmark_id:
        return (False, None, None, None, None)

    # Determine which IDs to query
    model_ids = duplicate_model_ids if duplicate_model_ids else [model_id]
    bench_ids = duplicate_benchmark_ids if duplicate_benchmark_ids else [benchmark_id]

    try:
        client = get_supabase_client()
        q = client.table('sandbox_jobs').select(
            'id,job_status,started_at,submitted_at,slurm_job_id,config'
        )

        # Use .in_() for duplicate groups, .eq() for singles
        if len(model_ids) == 1:
            q = q.eq('model_id', model_ids[0])
        else:
            q = q.in_('model_id', model_ids)

        if len(bench_ids) == 1:
            q = q.eq('benchmark_id', bench_ids[0])
        else:
            q = q.in_('benchmark_id', bench_ids)

        q = q.order('created_at', desc=True).limit(50)
        data = (q.execute().data) or []

        if not data:
            return (False, None, None, None, None)

        # Filter to matching config if timeout_aware
        for job in data:
            if timeout_aware:
                config = job.get('config')
                if isinstance(config, str):
                    try:
                        config = json.loads(config)
                    except Exception:
                        config = {}
                if not isinstance(config, dict):
                    config = {}

                job_agent = config.get('agent', DEFAULT_AGENT_NAME)
                job_tm = config.get('timeout_multiplier', DEFAULT_TIMEOUT_MULTIPLIER)

                # Skip if agent_name or timeout_multiplier don't match
                if job_agent != agent_name or float(job_tm) != float(timeout_multiplier):
                    continue

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

        # No matching job found
        return (False, None, None, None, None)

    except Exception as e:
        log(f"WARNING: sandbox_jobs v3 check failed for model_id={model_id}, benchmark_id={benchmark_id}: {e}")
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


def _config_matches_eval(job_config: Optional[Dict], eval_config: Dict) -> bool:
    """Check if a DB job's config JSONB matches the current eval config fields.

    Compares: timeout_multiplier, override_cpus, override_memory_mb, override_storage_mb.
    A job with no config is treated as defaults (timeout=1.0, no overrides).
    If eval_config is empty (no harbor config), any job config matches (backwards compat).
    """
    if not eval_config:
        return True  # no config constraints -- any existing job counts

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
    stale_hours: int = DEFAULT_STALE_JOB_HOURS,
    stale_pending_hours: int = DEFAULT_STALE_PENDING_HOURS,
    timeout_aware: bool = False,
    agent_name: str = DEFAULT_AGENT_NAME,
    timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER,
    duplicate_model_ids: Optional[List[str]] = None,
    duplicate_benchmark_ids: Optional[List[str]] = None,
    eval_config: Optional[Dict] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Determine if a job should be started based on DB status.

    When timeout_aware=True (v3 Enhancement 5), uses check_job_status_v3()
    which filters jobs by agent_name and timeout_multiplier in config. This
    allows running the same model with different configs without one blocking
    the other.

    When duplicate_model_ids/duplicate_benchmark_ids are provided, checks
    across the entire duplicate group for existing jobs.

    When eval_config is provided (from harbor YAML), performs config-aware
    dedup: checks that existing jobs match the current resource overrides
    (timeout_multiplier, override_cpus, override_memory_mb, override_storage_mb).

    Returns:
        (should_start, reason, slurm_job_id)
        slurm_job_id is provided so the caller can scancel stale jobs.
    """
    job_exists, job_status, started_at, submitted_at, slurm_job_id = check_job_status_v3(
        model_id, benchmark_id,
        timeout_aware=timeout_aware,
        agent_name=agent_name,
        timeout_multiplier=timeout_multiplier,
        duplicate_model_ids=duplicate_model_ids,
        duplicate_benchmark_ids=duplicate_benchmark_ids,
    )

    if not job_exists:
        return (True, "no existing job", None)

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
                    return (True, "no finished job with matching config", slurm_job_id)
                if matching[0] and not matching[0].get("metrics"):
                    return (True, "finished with matching config but metrics cleared", slurm_job_id)
            except Exception as e:
                log(f"WARNING: config-aware check failed: {e}")
        # v6: DaytonaError resume is now handled by disk-based scanning in
        # scan_jobs_dir_for_resume(), not here. This avoids the circular dependency
        # where DB stats only exist after upload, but upload is skipped on error.

        return (False, "job finished", slurm_job_id)

    if job_status == JOB_STATUS_PENDING:
        if ec:
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
                    return (True, "no pending job with matching config", slurm_job_id)
            except Exception as e:
                log(f"WARNING: config-aware check failed: {e}")
        # Job submitted but not yet running - check if stale using separate pending threshold
        if is_job_stale(submitted_at, stale_pending_hours):
            submitted_str = submitted_at.isoformat() if submitted_at else "null"
            return (True, f"stale pending job (submitted_at={submitted_str})", slurm_job_id)
        else:
            submitted_str = submitted_at.isoformat() if submitted_at else "null"
            return (False, f"job pending in SLURM queue (submitted_at={submitted_str})", slurm_job_id)

    if job_status == JOB_STATUS_STARTED:
        if ec:
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
                    return (True, "no in-progress job with matching config", slurm_job_id)
            except Exception as e:
                log(f"WARNING: config-aware check failed: {e}")
        if is_job_stale(started_at, stale_hours):
            started_str = started_at.isoformat() if started_at else "null"
            return (True, f"stale job (started_at={started_str})", slurm_job_id)
        else:
            started_str = started_at.isoformat() if started_at else "null"
            return (False, f"job in progress (started_at={started_str})", slurm_job_id)

    # Unknown status - start job to be safe
    return (True, f"unknown job status: {job_status}", slurm_job_id)


# ---------- v3 Enhancement 2: Per-Listener SLURM Job Throttle ----------
def get_active_slurm_job_ids() -> Set[str]:
    """Return set of SLURM job IDs currently queued/running for this user.

    Used by EvalListener to determine which of its submitted jobs are still
    active. The listener intersects this with its internal _submitted_jobs
    set to get a per-listener active count.
    """
    try:
        user = getpass.getuser()
        code, out = _run(["squeue", "-u", user, "--noheader", "-h", "-o", "%i"])
        if code != 0:
            log(f"WARNING: squeue failed (exit {code}), returning empty set")
            return set()
        return {line.strip() for line in out.strip().split('\n') if line.strip()}
    except Exception as e:
        log(f"WARNING: Failed to query squeue: {e}")
        return set()


def get_active_model_dataset_pairs(
    log_dir: str = "eval/logs",
) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, str]]:
    """Return (active_models, active_model_dataset_pairs, active_run_tags) for all active SLURM eval jobs.

    Queries squeue for all active jobs (RUNNING/PENDING/COMPLETING), then parses
    each job's eval log file to extract the model name, dataset, and run_tag.

    Args:
        log_dir: Directory containing eval log files ({job_name}_{slurm_id}.out).

    Returns:
        active_models: Set of HF model names currently running/queued.
        active_pairs: Set of (hf_model, dataset_hf) tuples currently running/queued.
        active_run_tags: Dict mapping run_tag → slurm_job_id for active jobs.
    """
    active_models: Set[str] = set()
    active_pairs: Set[Tuple[str, str]] = set()
    active_run_tags: Dict[str, str] = {}

    try:
        user = getpass.getuser()
        code, out = _run(["squeue", "-u", user, "--noheader", "-o", "%i %j"])
        if code != 0:
            log(f"WARNING: squeue failed (exit {code}), returning empty active sets")
            return active_models, active_pairs, active_run_tags
    except Exception as e:
        log(f"WARNING: Failed to query squeue: {e}")
        return active_models, active_pairs, active_run_tags

    log_path = Path(log_dir)
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        job_id, job_name = parts[0], parts[1]

        # Try to parse the eval log file for this job
        log_file = log_path / f"{job_name}_{job_id}.out"
        if not log_file.exists():
            continue

        model = dataset = run_tag = None
        try:
            with open(log_file, "r") as f:
                for i, fline in enumerate(f):
                    if i > 200:  # Only scan first 200 lines
                        break
                    if fline.startswith("Model: "):
                        model = fline.strip()[7:]
                    elif fline.startswith("Dataset: "):
                        dataset = fline.strip()[9:]
                    elif fline.startswith("Run tag: "):
                        run_tag = fline.strip()[9:]
                    if model and dataset and run_tag:
                        break
        except (OSError, IOError):
            continue

        if model:
            active_models.add(model)
        if model and dataset:
            # Normalize dataset: local path → HF name (e.g. /e/.../DCAgent_dev_set_v2 → DCAgent/dev_set_v2)
            ds_normalized = dataset
            if "/" in dataset and not dataset.startswith("/"):
                # Already HF format like DCAgent/dev_set_v2 or DCAgent2/terminal_bench_2
                ds_normalized = dataset
            active_pairs.add((model, ds_normalized))
        if run_tag:
            active_run_tags[run_tag] = job_id

    return active_models, active_pairs, active_run_tags


def _parse_job_ids_from_single_log(log_path: str) -> Set[str]:
    """Parse SLURM job IDs from a single listener log file.

    Matches two patterns:
      1. "-> Submitted as SLURM job 293324 (job_name=...)"   — direct submissions
      2. "[inherit-log] Inherited jobs: 293324,293325,..."    — inherited from previous log
    """
    job_ids: Set[str] = set()
    try:
        with open(log_path, "r") as f:
            for line in f:
                if "Submitted as SLURM job" in line:
                    parts = line.split("Submitted as SLURM job ")
                    if len(parts) >= 2:
                        jid = parts[1].split()[0].strip()
                        if jid.isdigit():
                            job_ids.add(jid)
                elif "[inherit-log] Inherited jobs:" in line:
                    # Parse comma-separated job IDs
                    parts = line.split("Inherited jobs:")
                    if len(parts) >= 2:
                        for jid in parts[1].strip().split(","):
                            jid = jid.strip()
                            if jid.isdigit():
                                job_ids.add(jid)
    except (OSError, IOError) as e:
        log(f"WARNING: Failed to read log {log_path}: {e}")
    return job_ids


def parse_submitted_jobs_from_logs(log_paths: List[str]) -> Set[str]:
    """Parse SLURM job IDs from one or more listener logs.

    Aggregates across all logs, then filters to jobs still active in squeue.
    """
    all_job_ids: Set[str] = set()
    for lp in log_paths:
        ids = _parse_job_ids_from_single_log(lp)
        log(f"[inherit-log] Parsed {len(ids)} job(s) from {lp}")
        all_job_ids |= ids

    active_ids = get_active_slurm_job_ids()
    still_active = all_job_ids & active_ids
    log(f"[inherit-log] Total: {len(all_job_ids)} job(s) across {len(log_paths)} log(s), "
        f"{len(still_active)} still active in squeue")
    return still_active


# ---------- v3 Enhancement 3: Daytona Resource Pre-flight Check ----------
def check_daytona_resources(sandbox_limit: int, warning_buffer: float) -> bool:
    """
    Check Daytona resource usage via API.

    Called at listener startup and optionally each iteration when
    --check-daytona-resources is enabled. Requires DAYTONA_API_KEY in env.

    Returns True if OK to proceed, False if active sandboxes >= sandbox_limit.
    Logs a warning when active sandboxes >= sandbox_limit * warning_buffer.
    """
    # Daytona migrated /api/sandbox/paginated (page-based) -> /api/sandbox
    # (cursor-based) on 2026-06-25; daytona SDK >= 0.180.0 exposes the new
    # endpoint via `daytona.list()`, which returns an iterator that handles
    # cursor pagination internally. The iterator does not expose a `.total`
    # attribute (unlike the deprecated `list_sandboxes_paginated`), so to
    # check whether we're at the sandbox cap we walk the iterator and
    # short-circuit once we hit `sandbox_limit + 1` (no need to enumerate
    # every active sandbox in the org).
    try:
        from daytona import Daytona, DaytonaConfig, ListSandboxesQuery, SandboxState
    except ImportError:
        log("WARNING: daytona SDK not installed (>=0.180.0 required), skipping resource check")
        return True

    api_key = os.environ.get("DAYTONA_API_KEY")
    api_url = os.environ.get("DAYTONA_API_URL")
    if not api_key:
        log("WARNING: DAYTONA_API_KEY not set, skipping resource check")
        return True

    try:
        cfg_kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_url:
            cfg_kwargs["api_url"] = api_url
        client = Daytona(DaytonaConfig(**cfg_kwargs))

        # Walk the iterator, counting up to sandbox_limit + 1 (we only need
        # to know whether we're at the cap). The SDK transparently fetches
        # subsequent cursor pages as we iterate.
        query = ListSandboxesQuery(states=[SandboxState.STARTED], limit=100)
        active_count = 0
        for _ in client.list(query):
            active_count += 1
            if active_count > sandbox_limit:
                break

        threshold = int(sandbox_limit * warning_buffer)
        if active_count >= sandbox_limit:
            log(f"ERROR: Daytona resources at limit: {active_count}/{sandbox_limit} active sandboxes "
                f"({active_count/sandbox_limit:.1%})")
            return False
        elif active_count >= threshold:
            log(f"WARNING: Daytona resources at {active_count}/{sandbox_limit} active sandboxes "
                f"({active_count/sandbox_limit:.1%}) - approaching limit!")
            return True
        else:
            log(f"Daytona resources OK: {active_count}/{sandbox_limit} active sandboxes "
                f"({active_count/sandbox_limit:.1%})")
            return True
    except Exception as e:
        log(f"WARNING: Daytona resource check failed: {e}")
        return True  # fail-open


# ---------- Job Submission ----------
def _resolve_agent_name_from_config_yaml(config_yaml: str) -> Optional[str]:
    """Return agents[0].name from a harbor eval config YAML, or None if not resolvable.

    Mirrors the sbatch's own yaml resolution (eval/configs/, eval/<cluster>/, eval/MBZ/)
    so the listener and sbatch always agree on the runtime agent name.
    """
    if not config_yaml:
        return None
    candidates = []
    if os.path.isabs(config_yaml):
        candidates.append(config_yaml)
    else:
        cluster_name = (_CLUSTER_CONFIG or {}).get("cluster_name")
        for base in ("eval/configs", f"eval/{cluster_name}" if cluster_name else None, "eval/MBZ"):
            if base:
                candidates.append(os.path.join(base, config_yaml))
    for path in candidates:
        try:
            if os.path.isfile(path):
                with open(path) as f:
                    cfg = yaml.safe_load(f) or {}
                agents = cfg.get("agents") or []
                if agents and isinstance(agents, list):
                    name = agents[0].get("name")
                    if name:
                        return name
                return None
        except Exception:
            continue
    return None


@dataclass
class SbatchParams:
    """Parameters passed to the sbatch script via environment variables.

    The listener converts these to EVAL_* env vars via to_env(), which the
    sbatch script reads at startup.

    v3 additions:
        error_threshold   Mapped to EVAL_DAYTONA_THRESHOLD (name kept for compat).
                          Controls the unified invalid error threshold.
        timeout_multiplier  Mapped to EVAL_TIMEOUT_MULTIPLIER. Passed to harbor
                            --timeout-multiplier and stored in DB job config.

    Cluster config additions (v6):
        When a cluster config YAML is loaded (--cluster-config), to_env() also
        exports EVAL_PROJECT_ROOT, EVAL_HF_CACHE, EVAL_HARBOR_SRC,
        EVAL_DATASETS_DIRS, EVAL_PROXY_ENABLED, EVAL_LOGIN_NODE,
        EVAL_PROXYCHAINS_BIN, EVAL_CUDA_HOME, EVAL_ARCH, EVAL_GPUS_PER_NODE,
        and EVAL_LOGS_DIR so sbatch scripts can be cluster-agnostic.
    """
    n_concurrent: int = DEFAULT_N_CONCURRENT
    n_attempts: int = DEFAULT_N_ATTEMPTS
    gpu_memory_util: float = DEFAULT_GPU_MEMORY_UTIL
    error_threshold: int = DEFAULT_ERROR_THRESHOLD
    vllm_max_retries: int = DEFAULT_VLLM_MAX_RETRIES
    agent_parser: str = DEFAULT_AGENT_PARSER
    slurm_time: str = DEFAULT_SLURM_TIME
    enable_thinking: bool = DEFAULT_ENABLE_THINKING
    agent_name: str = DEFAULT_AGENT_NAME
    slurm_partition: str = DEFAULT_SLURM_PARTITION
    slurm_account: str = DEFAULT_SLURM_ACCOUNT
    tp_size: int = DEFAULT_TP_SIZE
    dp_size: int = 1  # vLLM native data-parallel replicas (total GPUs = tp_size * dp_size)
    upload_username: str = ""
    timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER  # v3 Enhancement 5
    config_yaml: str = "dcagent_eval_config.yaml"
    max_output_tokens: Optional[int] = None  # None = use sbatch default (16384)
    auto_snapshot: Optional[bool] = None  # None = use YAML default
    agent_envs: Optional[str] = None  # Comma-separated KEY=VALUE pairs for --ae
    pinggy_url: Optional[str] = None  # Pinggy persistent URL (e.g., xxx.a.pinggy.link)
    pinggy_token: Optional[str] = None  # Pinggy auth token

    def get_effective_agent_name(self) -> str:
        """Resolve the runtime agent name, preferring the config_yaml's agents[0].name.

        The harbor config YAML is the ground truth for which agent actually runs
        (the sbatch reads it too). The listener's `--agent-name` CLI flag is only
        used as a fallback when the yaml is missing or lacks an agents block.
        Without this resolution, the DB Pending entry ends up with the wrong
        agent_id for installed-agent runs (aider / openhands / mini-swe-agent /
        swe-agent), because --agent-name defaults to terminus-2.
        """
        yaml_name = _resolve_agent_name_from_config_yaml(self.config_yaml)
        return yaml_name or self.agent_name

    def to_env(self) -> Dict[str, str]:
        """Convert to environment variables for sbatch."""
        env = {
            "EVAL_N_CONCURRENT": str(self.n_concurrent),
            "EVAL_N_ATTEMPTS": str(self.n_attempts),
            "EVAL_GPU_MEMORY_UTIL": str(self.gpu_memory_util),
            "EVAL_DAYTONA_THRESHOLD": str(self.error_threshold),
            "EVAL_VLLM_MAX_RETRIES": str(self.vllm_max_retries),
            "EVAL_AGENT_PARSER": self.agent_parser,
            "EVAL_SLURM_TIME": self.slurm_time,
            "EVAL_ENABLE_THINKING": "true" if self.enable_thinking else "false",
            "EVAL_AGENT_NAME": self.get_effective_agent_name(),
        }
        # Always send tp_size so build_vllm_cmd.sh doesn't fall back to its own default
        env["EVAL_VLLM_TENSOR_PARALLEL_SIZE"] = str(self.tp_size)
        if self.dp_size > 1:
            env["EVAL_VLLM_DATA_PARALLEL_SIZE"] = str(self.dp_size)
        if self.upload_username:
            env["EVAL_UPLOAD_USERNAME"] = self.upload_username
        # Enhancement 5: Pass timeout multiplier
        if self.timeout_multiplier != DEFAULT_TIMEOUT_MULTIPLIER:
            env["EVAL_TIMEOUT_MULTIPLIER"] = str(self.timeout_multiplier)
        # Pass config YAML (no-override for tb2/swebench)
        if self.config_yaml and self.config_yaml != "dcagent_eval_config.yaml":
            env["EVAL_CONFIG_YAML"] = self.config_yaml
        # Max output tokens override
        if self.max_output_tokens is not None:
            env["EVAL_MAX_OUTPUT_TOKENS"] = str(self.max_output_tokens)
        # Daytona auto_snapshot override (None = use YAML default)
        if self.auto_snapshot is not None:
            env["EVAL_AUTO_SNAPSHOT"] = "true" if self.auto_snapshot else "false"
        # Agent env vars forwarded into sandbox via --ae
        if self.agent_envs:
            env["EVAL_AGENT_ENVS"] = self.agent_envs
        # Pinggy tunnel config (for installed agents)
        if self.pinggy_url:
            env["EVAL_PINGGY_URL"] = self.pinggy_url
        if self.pinggy_token:
            env["EVAL_PINGGY_TOKEN"] = self.pinggy_token
        # Forward EVAL_JOBS_DIR to sbatch (default to user-writable location)
        fallback_jobs_dir = _cc_path("eval_jobs_dir", _FALLBACK_EVAL_JOBS_DIR)
        env["EVAL_JOBS_DIR"] = os.environ.get("EVAL_JOBS_DIR", fallback_jobs_dir)

        # --- Cluster config env vars (for sbatch parameterization) ---
        cc = _CLUSTER_CONFIG
        if cc:
            paths = cc.get("paths", {})
            proxy = cc.get("proxy", {})
            hw = cc.get("hardware", {})

            if cc.get("cluster_name"):
                env["EVAL_CLUSTER_NAME"] = cc["cluster_name"]
            if paths.get("project_root"):
                env["EVAL_PROJECT_ROOT"] = paths["project_root"]
            if paths.get("hf_cache"):
                env["EVAL_HF_CACHE"] = paths["hf_cache"]
            if paths.get("harbor_src"):
                env["EVAL_HARBOR_SRC"] = paths["harbor_src"]
            if paths.get("datasets_dirs"):
                env["EVAL_DATASETS_DIRS"] = ":".join(paths["datasets_dirs"])
            if paths.get("eval_logs_dir"):
                env["EVAL_LOGS_DIR"] = paths["eval_logs_dir"]

            env["EVAL_PROXY_ENABLED"] = "true" if proxy.get("enabled") else "false"
            if proxy.get("login_node"):
                env["EVAL_LOGIN_NODE"] = proxy["login_node"]
            if proxy.get("proxychains_bin"):
                env["EVAL_PROXYCHAINS_BIN"] = proxy["proxychains_bin"]

            if hw.get("cuda_home"):
                env["EVAL_CUDA_HOME"] = hw["cuda_home"]
            if hw.get("arch"):
                env["EVAL_ARCH"] = hw["arch"]
            if hw.get("gpus_per_node"):
                env["EVAL_GPUS_PER_NODE"] = str(hw["gpus_per_node"])
            if hw.get("cpus_per_node"):
                env["EVAL_CPUS_PER_NODE"] = str(hw["cpus_per_node"])

        return env

    def __str__(self) -> str:
        """String representation for logging."""
        parts = [
            f"n_concurrent={self.n_concurrent}",
            f"n_attempts={self.n_attempts}",
            f"gpu_memory_util={self.gpu_memory_util}",
            f"error_threshold={self.error_threshold}",
            f"vllm_max_retries={self.vllm_max_retries}",
        ]
        if self.agent_parser:
            parts.append(f"agent_parser={self.agent_parser}")
        if self.slurm_time != DEFAULT_SLURM_TIME:
            parts.append(f"slurm_time={self.slurm_time}")
        if self.tp_size != DEFAULT_TP_SIZE:
            parts.append(f"tp_size={self.tp_size}")
        if self.dp_size > 1:
            parts.append(f"dp_size={self.dp_size}")
        if self.enable_thinking:
            parts.append("enable_thinking=True")
        if self.agent_name != DEFAULT_AGENT_NAME:
            parts.append(f"agent_name={self.agent_name}")
        if self.slurm_partition != DEFAULT_SLURM_PARTITION:
            parts.append(f"slurm_partition={self.slurm_partition}")
        if self.upload_username:
            parts.append(f"upload_username={self.upload_username}")
        if self.timeout_multiplier != DEFAULT_TIMEOUT_MULTIPLIER:
            parts.append(f"timeout_multiplier={self.timeout_multiplier}")
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


def get_idle_nodes(partition: str) -> List[str]:
    """Get list of idle nodes on a SLURM partition."""
    code, out = _run(["sinfo", "-p", partition, "-N", "--format=%N %t", "--noheader"])
    if code != 0:
        return []
    nodes = []
    for line in out.strip().split("\n"):
        parts = line.split()
        if len(parts) >= 2 and parts[1].strip() == "idle":
            nodes.append(parts[0].strip())
    return nodes


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


def submit_eval(
    hf_model_name: str,
    dataset_hf: str,
    benchmark_id: Optional[str],
    sbatch_script: str,
    sbatch_params: Optional[SbatchParams] = None,
    dry_run: bool = False,
    upload_username: str = "",
    timeout_multiplier: float = DEFAULT_TIMEOUT_MULTIPLIER,
    vllm_overrides: Optional[Dict[str, str]] = None,
    dependency: Optional[str] = None,
    eval_config: Optional[Dict] = None,
    conda_env: str = "otagent",
    run_tag_override: Optional[str] = None,
    dp_nodes: int = 0,
    nodelist: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    skip_gpu_request: bool = False,
    sbatch_model_override: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Create a Pending DB entry, then submit sbatch job and update with SLURM ID.

    sbatch positional args:
      $1 = model HF name
      $2 = dataset HF repo (org/repo)
      $3 = benchmark_id (uuid)  [optional]
      $4 = job_name (RUN_TAG)

    Environment variables (from SbatchParams.to_env()):
      EVAL_N_CONCURRENT, EVAL_N_ATTEMPTS, EVAL_GPU_MEMORY_UTIL,
      EVAL_DAYTONA_THRESHOLD, EVAL_VLLM_MAX_RETRIES, EVAL_AGENT_PARSER,
      EVAL_SLURM_TIME, EVAL_ENABLE_THINKING, EVAL_AGENT_NAME,
      EVAL_STARTS_LOG (v3), EVAL_TIMEOUT_MULTIPLIER (v3)

    The Pending DB entry includes timeout_multiplier in its config dict
    so that timeout-aware dedup (Enhancement 5) can match on it.

    Args:
        vllm_overrides: Optional dict of EVAL_VLLM_* env vars from baseline
                        model config. Merged into sbatch env vars.
        dependency: Optional SLURM dependency string (e.g. 'afterany:12345').
        eval_config: Optional harbor eval config dict for DB job config.
        run_tag_override: v5 - reuse original run_tag for resume (triggers sbatch auto-resume).

    Returns:
        (slurm_job_id, job_name) if successful, ("DRY_RUN", job_name) if dry run, (None, None) on failure
    """
    # Generate unique job name, or reuse for resume
    if run_tag_override:
        job_name = run_tag_override
        log(f"  [v5] Reusing run_tag for resume: {job_name}")
    else:
        job_name = generate_run_tag(dataset_hf, hf_model_name)

    # Early return for dry-run — no DB writes, no sbatch
    if dry_run:
        log(f"[DRY RUN] Would submit: model={hf_model_name} dataset={dataset_hf} job={job_name}")
        if sbatch_params:
            log(f"[DRY RUN] With params: {sbatch_params}")
        if vllm_overrides:
            log(f"[DRY RUN] vLLM overrides: {list(vllm_overrides.keys())}", verbose_only=True)
        return ("DRY_RUN", job_name)

    # Step 1: Create or reuse DB entry BEFORE sbatch submission.
    # Use the effective agent name (derived from config_yaml when available) so
    # installed-agent runs don't get terminus-2's agent_id stamped on their row.
    agent = sbatch_params.get_effective_agent_name() if sbatch_params else DEFAULT_AGENT_NAME
    tm = sbatch_params.timeout_multiplier if sbatch_params else timeout_multiplier
    config: Dict = {"agent": agent, "env": "daytona", "timeout_multiplier": tm, "run_tag": job_name}
    # Include harbor eval config fields in DB entry for config-aware dedup
    if eval_config:
        if "timeout_multiplier" in eval_config:
            config["timeout_multiplier"] = eval_config["timeout_multiplier"]
        env_overrides = {}
        for key in ("override_cpus", "override_memory_mb", "override_storage_mb"):
            if key in eval_config:
                env_overrides[key] = eval_config[key]
        if env_overrides:
            config["environment"] = env_overrides

    db_job_id: Optional[str] = None
    try:
        from database.unified_db.utils import (
            create_job_entry_pending, get_supabase_client,
            get_model_by_name, get_benchmark_by_name,
            get_job_by_model_benchmark, update_sandbox_job,
        )

        if run_tag_override:
            # v6 resume: find existing DB entry and update its job_name to match
            # the resume run_tag, so sbatch's update_job_status_to_started() can
            # find it by name. Also reset status to Pending for clean state.
            model_row = get_model_by_name(hf_model_name)
            bench_name = dataset_hf.split("/")[-1] if "/" in dataset_hf else dataset_hf
            bench_row = get_benchmark_by_name(bench_name)
            if model_row and bench_row:
                existing = get_job_by_model_benchmark(model_row['id'], bench_row['id'])
                if existing:
                    old_name = existing.get('job_name', '?')
                    old_status = existing.get('job_status', '?')
                    db_job_id = str(existing['id'])
                    # Update the existing entry: reset to Pending with new job_name
                    update_sandbox_job(db_job_id, {
                        "job_name": job_name,
                        "job_status": "Pending",
                        "slurm_job_id": "pending",
                        "config": config,
                        "submitted_at": datetime.now().isoformat(),
                        "started_at": None,
                    })
                    log(f"  [v6] Reused DB entry {db_job_id}: {old_status} '{old_name}' → Pending '{job_name}'",
                        verbose_only=True)

        if not db_job_id:
            # Normal path: create new Pending entry
            result = create_job_entry_pending(
                job_name=job_name,
                model_hf=hf_model_name,
                benchmark_hf=dataset_hf,
                agent_name=agent,
                slurm_job_id="pending",
                username=upload_username or "listener",
                config=config,
            )
            if result.get("success") and result.get("job"):
                db_job_id = str(result["job"].get("id"))
                log(f"Created Pending DB entry: {db_job_id}", verbose_only=True)
            else:
                log(f"WARNING: Failed to create Pending DB entry: {result.get('error')}")
    except Exception as e:
        log(f"WARNING: Exception creating Pending DB entry: {e}")

    # Step 2: Build sbatch command
    cmd = ["sbatch"]
    if sbatch_params:
        cmd.extend(["--time", sbatch_params.slurm_time])
        cmd.extend(["--partition", sbatch_params.slurm_partition])
        if sbatch_params.slurm_account:
            cmd.extend(["--account", sbatch_params.slurm_account])
        if skip_gpu_request:
            # API mode: no GPU, modest CPU/mem (sbatch header sets sane defaults).
            cmd.extend(["--cpus-per-task", "8"])
            cmd.extend(["--mem", "16G"])
        else:
            # Request GPUs, CPUs, and memory proportional to total GPU count (TP × DP).
            # Per-model baseline config can override TP (e.g. 32B models need TP=4).
            effective_tp = sbatch_params.tp_size
            if vllm_overrides and "EVAL_VLLM_TENSOR_PARALLEL_SIZE" in vllm_overrides:
                effective_tp = int(vllm_overrides["EVAL_VLLM_TENSOR_PARALLEL_SIZE"])
            effective_dp = sbatch_params.dp_size
            total_gpus = effective_tp * effective_dp
            cmd.extend(["--gres", f"gpu:{total_gpus}"])
            cc = _CLUSTER_CONFIG or {}
            hw = cc.get("hardware", {})
            gpus_per_node = int(hw.get("gpus_per_node", 8))
            cpus_per_node = int(hw.get("cpus_per_node", 96))
            mem_per_node_mb = int(hw.get("mem_per_node_mb", 1860000))
            cpus_needed = (cpus_per_node * total_gpus) // gpus_per_node
            mem_needed = (mem_per_node_mb * total_gpus) // gpus_per_node
            cmd.extend(["--cpus-per-task", str(cpus_needed)])
            cmd.extend(["--mem", f"{mem_needed}M"])
    # Override sbatch --output to use cluster-configured log dir
    cc_logs = (_CLUSTER_CONFIG or {}).get("paths", {}).get("eval_logs_dir", _FALLBACK_EVAL_LOGS_DIR)
    cmd.extend(["--output", f"{cc_logs}/%x_%j.out"])
    if nodelist:
        cmd.extend(["--nodelist", nodelist])
    if dependency:
        cmd.append(f"--dependency={dependency}")
    # DP: request multiple nodes
    if dp_nodes > 0:
        cmd.extend(["--nodes", str(dp_nodes)])
    # Benchmark-aware SLURM job name for squeue readability
    repo_name = dataset_repo_name(dataset_hf)
    bench_tag = _BENCH_SHORT.get(repo_name, repo_name[:12])
    if run_tag_override:
        job_prefix = "res_dp_" if dp_nodes > 0 else "res_"
    else:
        job_prefix = "eval_dp_" if dp_nodes > 0 else "eval_"
    slurm_job_name = os.environ.get("EVAL_SLURM_JOB_NAME", "data")
    cmd.extend(["--job-name", slurm_job_name])
    cmd.append(sbatch_script)
    # For API mode the sbatch arg is the LiteLLM-prefixed name (e.g.
    # together_ai/moonshotai/Kimi-K2.5) while the DB row keeps the unprefixed
    # registered name. For vLLM mode they're the same.
    sbatch_model_arg = sbatch_model_override or hf_model_name
    cmd.extend([sbatch_model_arg, dataset_hf])
    # $3 = benchmark_id (or empty placeholder), $4 = run_tag override
    cmd.append(str(benchmark_id) if benchmark_id else "")
    cmd.append(job_name)  # 4th arg: job_name (RUN_TAG)

    # Get env vars from params and merge vllm overrides + harbor config env vars
    env_vars = sbatch_params.to_env() if sbatch_params else {}
    # Pass-through whitelist for agent-specific host env overrides (e.g. custom
    # swe-agent config URL). Listed here rather than forwarding full environ to
    # avoid leaking unrelated vars into the SLURM job.
    for _passthru in ("SWEAGENT_CONFIG", "EVAL_USE_GLM5_PROXY"):
        if _passthru in os.environ:
            env_vars[_passthru] = os.environ[_passthru]
    if db_job_id:
        env_vars["EVAL_DB_JOB_ID"] = db_job_id
    if vllm_overrides:
        env_vars.update(vllm_overrides)
    # Pass harbor eval config fields as sbatch env vars
    if eval_config:
        if eval_config.get("timeout_multiplier") is not None:
            env_vars["EVAL_TIMEOUT_MULTIPLIER"] = str(eval_config["timeout_multiplier"])
        if eval_config.get("override_memory_mb") is not None:
            env_vars["EVAL_OVERRIDE_MEMORY_MB"] = str(eval_config["override_memory_mb"])

    # Pass conda env path so sbatch uses the right Python/vLLM installation
    otagent_dir = CONDA_ENV_PATHS.get(conda_env)
    if otagent_dir:
        env_vars["OTAGENT_DIR"] = otagent_dir
    # Merge extra env vars (e.g. EVAL_VLLM_PORT from pack-jobs port planner)
    if extra_env:
        env_vars.update(extra_env)
    # DP: let sbatch compute NUM_SHARDS from GPUS_PER_NODE / TP_SIZE * nodes
    # Do NOT pass EVAL_NUM_SHARDS — it would override the per-node shard calculation

    # Step 3: Run sbatch
    code, out = _run(cmd, env=env_vars)
    log(f"sbatch: {' '.join(cmd)}\n{out}")

    if code != 0:
        # sbatch failed; pending entry remains (will be detected as stale later)
        return (None, None)

    m = re.search(r"Submitted batch job (\d+)", out)
    slurm_job_id = m.group(1) if m else None

    if not slurm_job_id:
        log("ERROR: Could not parse SLURM job ID from sbatch output")
        return (None, None)

    # Step 4: Update pending entry with actual SLURM job ID
    if db_job_id:
        update_pending_job_slurm_id(db_job_id, slurm_job_id)

    return (slurm_job_id, job_name)


# ---------- Main Listener Class ----------
class EvalListener:
    """Unified eval listener v3 that handles all benchmark configurations.

    Lifecycle:
      1. run() logs config, runs Daytona pre-flight (if enabled), enters main loop
      2. Each iteration: hot-reload priority file, fetch models, filter, build
         submissions list, sort by priority rank, apply retry deprioritization,
         throttle to per-listener SLURM limit, submit
      3. Sleep for check_interval_hours, then repeat

    Per-listener SLURM tracking:
      _submitted_jobs tracks SLURM job IDs submitted by THIS listener instance.
      Each iteration, completed jobs are pruned via squeue intersection. This
      allows multiple listeners to run in parallel with independent job budgets.
    """

    def __init__(self, config: ListenerConfig):
        self.config = config
        self._submitted_jobs: Set[str] = set()  # SLURM job IDs submitted by THIS listener
        self._dep_chain: List[str] = []  # Persistent sliding-window dependency chain across iterations
        self._resume_run_tags: Dict[str, str] = {}  # v6: hf_model → run_tag for disk resume
        set_log_file(config.log_file)
        # Seed _submitted_jobs from previous listener logs (--inherit-log)
        if config.inherit_log:
            inherited = parse_submitted_jobs_from_logs(config.inherit_log)
            self._submitted_jobs = inherited
            if inherited:
                # Log the inherited IDs so future --inherit-log on THIS log picks them up
                log(f"[inherit-log] Inherited {len(inherited)} active job(s)")
                log(f"[inherit-log] Inherited jobs: {','.join(sorted(inherited))}")

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

        # Hot-reload blacklist from file
        if self.config.blacklist_file:
            new_blacklist = load_blacklist(self.config.blacklist_file)
            if new_blacklist != self.config.blacklisted_models:
                log(f"Blacklist reloaded: {len(new_blacklist)} model(s)")
                self.config.blacklisted_models = new_blacklist

        # v6: Clear per-iteration resume state
        self._resume_run_tags = {}

        log("Checking for new models...")

        # Optimization: in filter_only mode with a priority file, skip the
        # expensive fetch_recent_models() (which returns ALL models in the
        # lookback window) and only fetch priority models by name.
        if (self.config.priority_mode == "filter_only"
                and self.config.priority_models):
            models = fetch_priority_models(self.config.priority_models)
            log(f"Fetched {len(models)} priority model(s) directly (filter_only mode, skipped full scan).")
        else:
            models = fetch_recent_models(self.config.lookback_days)
            log(f"Found {len(models)} model(s) in lookback window.")

            # Priority models bypass lookback window.
            # Fetch priority models by name regardless of creation_time, then merge.
            if self.config.priority_models:
                priority_models_from_db = fetch_priority_models(self.config.priority_models)
                seen_ids = {str(m.get("id")) for m in models}
                added = 0
                for pm in priority_models_from_db:
                    if str(pm.get("id")) not in seen_ids:
                        models.append(pm)
                        seen_ids.add(str(pm.get("id")))
                        added += 1
                if added:
                    log(f"Added {added} priority model(s) outside lookback window.")

        log(f"Total {len(models)} model(s) to check. Filtering...")

        # Check if we should skip all models due to require_priority_list
        if not self.config.priority_models and self.config.require_priority_list:
            log("No priority list configured and --require-priority-list is set. Skipping all models.")
            return 0

        submissions: List[Tuple[str, str, str, Optional[str], str, Optional[str]]] = []
        # (model_id, hf_model_name, dataset_hf, benchmark_id, reason, slurm_job_id)
        finished_in_db: Set[str] = set()  # v6: models DB considers done (skip for resume)

        # v6: Build set of (model, dataset) pairs currently running in squeue (by parsing eval logs).
        # Used to prevent both resume and fresh submissions for already-running models.
        active_models, active_pairs, active_run_tags = get_active_model_dataset_pairs(
            log_dir=_cc_path("eval_logs_dir", _FALLBACK_EVAL_LOGS_DIR),
        )
        if active_pairs:
            log(f"[v6-active] Found {len(active_pairs)} (model, dataset) pair(s) currently active in squeue")
            if self.config.verbose:
                for m, d in sorted(active_pairs):
                    log(f"  [v6-active] {m} on {d}")

        # Track stats
        skipped_not_in_priority = 0
        skipped_hf_not_exists = 0

        # Resolve all benchmarks up front (once per loop)
        dataset_to_bench: Dict[str, Optional[str]] = {
            ds: resolve_benchmark_id(ds) for ds in self.config.datasets
        }

        # Precompute benchmark duplicate groups for cross-duplicate aggregation
        bench_dup_groups: Dict[str, List[str]] = {}
        for ds, bench_id in dataset_to_bench.items():
            if bench_id:
                bench_dup_groups[bench_id] = get_duplicate_group_ids('benchmarks', bench_id)

        for m in models:
            model_id = str(m.get("id"))
            if not model_id:
                continue

            hf_model = resolve_hf_model_name(m)
            if not hf_model:
                if self.config.verbose:
                    log(f"Skip: cannot resolve HF model for id={model_id}, name={m.get('name')}")
                continue

            # Blacklist check (overrides priority)
            if hf_model in self.config.blacklisted_models:
                if self.config.verbose:
                    log(f"Skip: model={hf_model} is blacklisted")
                continue

            # Priority handling depends on mode
            is_priority = bool(self.config.priority_models and hf_model in self.config.priority_models)

            if self.config.priority_mode == "filter_only":
                # Only evaluate models in the priority list
                if self.config.priority_models and not is_priority:
                    skipped_not_in_priority += 1
                    continue
            # priority_first: don't skip, just track is_priority for sorting

            # HuggingFace existence check
            if self.config.check_hf_exists:
                if not check_hf_model_exists(hf_model):
                    log(f"Skip: model not found on HuggingFace: {hf_model} (model_id={model_id})")
                    skipped_hf_not_exists += 1
                    continue

            # Compute model duplicate group for cross-duplicate aggregation
            model_dup_ids = get_duplicate_group_ids('models', model_id)

            for dataset_hf in self.config.datasets:
                bench_id = dataset_to_bench.get(dataset_hf)

                # Get benchmark duplicate group (precomputed above)
                bench_dup_ids = bench_dup_groups.get(bench_id) if bench_id else None

                # Check DB status to decide if we should start
                # (Enhancement 5: timeout-aware, cross-duplicate aggregation, config-aware dedup)
                if self.config.force_reeval:
                    should_start, reason, old_slurm_job_id = True, "force-reeval", None
                else:
                    should_start, reason, old_slurm_job_id = should_start_job(
                        model_id, bench_id, self.config.stale_job_hours,
                        stale_pending_hours=self.config.stale_pending_hours,
                        timeout_aware=self.config.timeout_aware,
                        agent_name=self.config.agent_name,
                        timeout_multiplier=self.config.timeout_multiplier,
                        duplicate_model_ids=model_dup_ids,
                        duplicate_benchmark_ids=bench_dup_ids,
                        eval_config=self.config.eval_config if self.config.eval_config else None,
                    )

                if should_start:
                    # v6: Skip if (model, dataset) already running in squeue (even if DB says "no existing job",
                    # e.g. when DB entry was deleted but SLURM job is still active)
                    # Bypass this check in force-reeval mode.
                    if not self.config.force_reeval and (hf_model, dataset_hf) in active_pairs:
                        if self.config.verbose:
                            log(f"Skip: model={hf_model}, dataset={dataset_hf}, reason=currently running in squeue")
                        continue
                    submissions.append((model_id, hf_model, dataset_hf, bench_id, reason, old_slurm_job_id))
                else:
                    # Track models the DB considers done (for v6 resume filtering)
                    if "finished" in reason:
                        finished_in_db.add(hf_model)
                    if self.config.verbose:
                        log(f"Skip: model={hf_model}, dataset={dataset_hf}, reason={reason}")

        # Log filtering stats
        if self.config.priority_mode == "filter_only" and self.config.priority_models and skipped_not_in_priority > 0:
            log(f"Skipped {skipped_not_in_priority} model(s) not in priority list")
        if self.config.check_hf_exists and skipped_hf_not_exists > 0:
            log(f"Skipped {skipped_hf_not_exists} model(s) not found on HuggingFace")

        # v6: Disk-based resume — scan jobs dir for incomplete/errored jobs
        resume_submissions = []
        if self.config.enable_disk_resume and self.config.jobs_dirs:
            # Always query squeue (even in dry-run) for accurate filtering
            active_slurm = get_active_slurm_job_ids()
            # Build dataset prefixes from config
            ds_prefixes = []
            for ds in self.config.datasets:
                ds_short = ds.split("/")[-1] if "/" in ds else ds
                ds_prefixes.append(ds_short)
            # Scan all configured jobs directories
            resume_candidates = []
            for jdir in self.config.jobs_dirs:
                resume_candidates.extend(scan_jobs_dir_for_resume(
                    jobs_dir=jdir,
                    dataset_prefixes=ds_prefixes,
                    active_slurm_ids=active_slurm,
                    infra_error_threshold=self.config.resume_infra_error_threshold,
                    max_resume_count=self.config.max_resume_count,
                ))
            # Filter resume candidates through blacklist and priority (same as normal models)
            if self.config.blacklisted_models:
                before = len(resume_candidates)
                resume_candidates = [rc for rc in resume_candidates
                                     if rc["hf_model"] not in self.config.blacklisted_models]
                skipped_bl = before - len(resume_candidates)
                if skipped_bl:
                    log(f"[v6-resume] Filtered out {skipped_bl} blacklisted resume candidate(s)")
            if self.config.priority_mode == "filter_only" and self.config.priority_models:
                before = len(resume_candidates)
                resume_candidates = [rc for rc in resume_candidates
                                     if rc["hf_model"] in self.config.priority_models]
                skipped_prio = before - len(resume_candidates)
                if skipped_prio:
                    log(f"[v6-resume] Filtered out {skipped_prio} non-priority resume candidate(s)")

            # v6: Filter out (model, dataset) pairs currently running in squeue.
            # This prevents resuming old dirs when a job for the same model+dataset is active.
            # Bypassed by --force-reeval (mirrors the fresh-fire path behavior at the active_pairs
            # check earlier — needed e.g. when an active job uses a different scaffold than the
            # resume target, so the (model, dataset) collision is a false positive).
            if active_pairs and not self.config.force_reeval:
                before = len(resume_candidates)
                resume_candidates = [rc for rc in resume_candidates
                                     if (rc["hf_model"], rc.get("dataset", "")) not in active_pairs]
                skipped_active = before - len(resume_candidates)
                if skipped_active:
                    log(f"[v6-resume] Filtered out {skipped_active} currently-running resume candidate(s)")

            # Filter out models that DB already considers finished (stale disk dirs
            # from older runs that have been superseded by a successful resubmission)
            if finished_in_db:
                before = len(resume_candidates)
                resume_candidates = [rc for rc in resume_candidates
                                     if rc["hf_model"] not in finished_in_db]
                skipped_fin = before - len(resume_candidates)
                if skipped_fin:
                    log(f"[v6-resume] Filtered out {skipped_fin} already-finished-in-DB resume candidate(s)")

            # Dedup: pick the most recent dir per model (reverse so latest timestamp wins).
            seen_resume_models: Set[str] = set()
            for rc in reversed(resume_candidates):
                if rc["hf_model"] not in seen_resume_models:
                    seen_resume_models.add(rc["hf_model"])
                    # Use a sentinel model_id since we don't have it from DB
                    resume_submissions.append(
                        ("__resume__", rc["hf_model"], rc["dataset"] or "",
                         None, rc["reason"], None)
                    )
                    # Store run_tag mapping for submit_eval
                    self._resume_run_tags[rc["hf_model"]] = rc["run_tag"]
            if resume_submissions:
                log(f"[v6-resume] Adding {len(resume_submissions)} resume job(s) (priority over new models)")

        # v6: Resume takes priority — remove normal submissions for models
        # that already have a resume candidate (avoid duplicate fresh + resume).
        resume_model_set = {s[1] for s in resume_submissions}  # s[1] = hf_model
        if resume_model_set:
            before = len(submissions)
            submissions = [s for s in submissions if s[1] not in resume_model_set]
            skipped_dup = before - len(submissions)
            if skipped_dup:
                log(f"[v6-resume] Suppressed {skipped_dup} fresh submission(s) in favor of resume")

        # --resume-only: drop all fresh submissions, keep only resume jobs
        if self.config.resume_only:
            if submissions:
                log(f"[v6-resume] --resume-only: dropping {len(submissions)} fresh submission(s)")
                submissions = []

        if not submissions and not resume_submissions:
            log("No eligible (model, dataset) pairs to submit.")
            return 0

        # Prepend resume submissions (higher priority than new models)
        submissions = resume_submissions + submissions

        # Sort submissions by priority file order (earlier in file = higher priority).
        # Models not in the priority list get lowest rank (submitted last).
        if self.config.priority_models:
            priority_rank = {m: i for i, m in enumerate(self.config.priority_models)}
            fallback_rank = len(self.config.priority_models)
            submissions.sort(key=lambda s: priority_rank.get(s[1], fallback_rank))
            if self.config.priority_mode == "priority_first":
                n_priority = sum(1 for s in submissions if s[1] in priority_rank)
                n_non_priority = len(submissions) - n_priority
                log(f"Priority-first ordering: {n_priority} priority + {n_non_priority} non-priority submissions")

        prefix = "[DRY RUN] Would submit" if self.config.dry_run else "Submitting"
        log(f"{prefix} {len(submissions)} eval(s)...")

        # Enhancement 2: Per-listener SLURM job submission throttle.
        # Track which SLURM job IDs this listener submitted. Prune finished ones
        # via squeue, then cap new submissions at remaining slots.
        if not self.config.dry_run:
            active_ids = get_active_slurm_job_ids()
            # Prune jobs that are no longer in squeue (finished/failed/cancelled)
            still_active = self._submitted_jobs & active_ids
            finished = len(self._submitted_jobs) - len(still_active)
            self._submitted_jobs = still_active
            active_count = len(self._submitted_jobs)
            remaining_slots = self.config.max_jobs_submitted - active_count
            log(f"Listener SLURM jobs: {active_count} active "
                f"({finished} finished since last check), "
                f"{remaining_slots} slots available (max {self.config.max_jobs_submitted})")
            if remaining_slots <= 0:
                log(f"WARNING: At per-listener job limit "
                    f"({active_count}/{self.config.max_jobs_submitted}), "
                    f"skipping all submissions this iteration")
                return 0
            if len(submissions) > remaining_slots:
                log(f"Capping submissions from {len(submissions)} to {remaining_slots} "
                    f"(per-listener limit: {self.config.max_jobs_submitted})")
                submissions = submissions[:remaining_slots]

        # Create sbatch params from config
        sbatch_params = SbatchParams(
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            gpu_memory_util=self.config.gpu_memory_util,
            error_threshold=self.config.error_threshold,
            vllm_max_retries=self.config.vllm_max_retries,
            agent_parser=self.config.agent_parser,
            slurm_time=self.config.slurm_time,
            enable_thinking=self.config.enable_thinking,
            agent_name=self.config.agent_name,
            slurm_partition=self.config.slurm_partition,
            slurm_account=self.config.slurm_account,
            tp_size=self.config.tp_size,
            dp_size=self.config.dp_size,
            upload_username=self.config.upload_username,
            timeout_multiplier=self.config.timeout_multiplier,
            config_yaml=self.config.config_yaml,
            max_output_tokens=self.config.max_output_tokens,
            auto_snapshot=self.config.auto_snapshot,
            agent_envs=self.config.agent_envs,
            pinggy_url=self.config.pinggy_url,
            pinggy_token=self.config.pinggy_token,
        )

        # Load baseline model configs for per-model vLLM overrides
        baseline_configs = load_baseline_model_configs(self.config.baseline_model_configs)

        # Load per-model API serving configs (no-op if file missing — get_api_config returns None)
        api_configs = load_api_model_configs(self.config.api_model_config)

        # Add harbor config env vars to sbatch params
        if self.config.harbor_config:
            # Will be merged in submit_eval via eval_config, but also pass path
            pass  # harbor config fields are passed via eval_config to submit_eval

        # Pre-download setup (for no-internet compute nodes)
        if self.config.pre_download:
            from huggingface_hub import snapshot_download
            downloaded_models: set = set()

        # Sliding-window dependency tracking (persistent across iterations)
        # self._dep_chain carries job IDs from previous iterations so new jobs
        # respect the concurrency limit even across sleep cycles.
        batch_size = self.config.batch_size
        if batch_size and batch_size > 0:
            if not self.config.dry_run:
                active_ids = get_active_slurm_job_ids()
            else:
                active_ids = set()
            active_in_chain = sum(1 for jid in self._dep_chain if jid in active_ids)
            log(f"Sliding-window batch-size={batch_size}: "
                f"{active_in_chain} active jobs in dependency chain from previous iterations")

        # Node packing: query idle nodes and track GPU + port slots per node
        pack_node_list: List[str] = []
        pack_gpus_per_node = 8
        pack_node_gpu_used: Dict[int, int] = {}  # node_idx -> GPUs used so far
        pack_node_port_next: Dict[int, int] = {}  # node_idx -> next available port offset
        pack_node_idx = 0
        if self.config.pack_jobs:
            cc = self.config.cluster_config or {}
            hw = cc.get("hardware", {})
            pack_gpus_per_node = int(hw.get("gpus_per_node", 8))
            pack_node_list = get_idle_nodes(self.config.slurm_partition)
            if pack_node_list:
                log(f"Pack mode: {len(pack_node_list)} idle nodes, {pack_gpus_per_node} GPUs/node")
            else:
                log("Pack mode: no idle nodes found, falling back to default scheduling")

        submitted = 0
        for idx, (mid, hf_model, dataset_hf, bench_id, reason, old_slurm_job_id) in enumerate(submissions):

            # Pre-download this model before submitting (download-then-submit per model)
            # Uses the shared HF cache so compute nodes (no internet) find it via HF_HUB_OFFLINE=1
            if self.config.pre_download and hf_model not in downloaded_models:
                hf_cache = os.environ.get("HF_HUB_CACHE", _cc_path("hf_cache", _FALLBACK_HF_CACHE))
                log(f"  Pre-downloading model {hf_model} to {hf_cache}...")
                try:
                    # Run snapshot_download in a worker thread with a hard wall
                    # timeout. etag_timeout=120 + max_workers=8 keep slow HF
                    # metadata/file fetches from stalling (the default 10s etag
                    # retries serially and wedged an entire fire for hours on a
                    # slow link — 2026-06 ablation batch). Critically, we drive
                    # the executor manually and shutdown(wait=False) in finally:
                    # the old `with ThreadPoolExecutor()` block's implicit
                    # shutdown(wait=True) re-blocked on the still-running (un-
                    # killable) download thread AFTER the 300s timeout fired,
                    # which was the actual indefinite hang.
                    import concurrent.futures
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    try:
                        future = executor.submit(
                            snapshot_download, repo_id=hf_model, repo_type="model",
                            cache_dir=hf_cache, etag_timeout=120, max_workers=8,
                        )
                        path = future.result(timeout=600)  # 10 minute hard cap
                        log(f"  Cached at {path}")
                    finally:
                        executor.shutdown(wait=False)
                except concurrent.futures.TimeoutError:
                    log(f"  WARNING: Pre-download of {hf_model} timed out after 600s, skipping (will retry next iteration)")
                except Exception as e:
                    log(f"  WARNING: Failed to download {hf_model}: {e}")
                downloaded_models.add(hf_model)

            dry_prefix = "[DRY RUN] " if self.config.dry_run else ""
            prio_tag = " [PRIORITY]" if (self.config.priority_mode == "priority_first"
                                         and self.config.priority_models
                                         and hf_model in self.config.priority_models) else ""
            # v6: Pretty-print resume reason
            display_reason = reason
            log(f"{dry_prefix}Submitting [{idx+1}/{len(submissions)}]: model={hf_model}, dataset={dataset_hf}, reason={display_reason}{prio_tag}")

            # Cancel stale Pending SLURM job before resubmission
            if reason.startswith("stale pending") and old_slurm_job_id:
                cancel_slurm_job(old_slurm_job_id, dry_run=self.config.dry_run)

            # Per-model vLLM overrides from baseline config mapping
            vllm_overrides = get_vllm_env_overrides(hf_model, baseline_configs)
            if vllm_overrides:
                log(f"  Applying baseline model vLLM overrides: {list(vllm_overrides.keys())}", verbose_only=True)

            # Per-model conda env override (e.g. otagent2 for Qwen3.5)
            model_conda_env = get_conda_env_override(hf_model, baseline_configs) or self.config.conda_env
            if model_conda_env != self.config.conda_env:
                log(f"  Using conda env '{model_conda_env}' for {hf_model}")

            # Build sliding-window dependency using persistent chain.
            # Look back batch_size positions in self._dep_chain. If that job is
            # still active (running/pending in squeue), depend on it. If it already
            # finished, the concurrency slot is free — no dependency needed.
            job_dependency: Optional[str] = None
            if batch_size and batch_size > 0:
                chain_pos = len(self._dep_chain)  # where this job will be appended
                if chain_pos >= batch_size:
                    dep_candidate = self._dep_chain[chain_pos - batch_size]
                    if not self.config.dry_run and dep_candidate in active_ids:
                        job_dependency = f"afterany:{dep_candidate}"
                        log(f"  Depends on job {dep_candidate} (chain pos {chain_pos - batch_size})", verbose_only=True)

            # Stagger chain: jobs wait N minutes after the previous batch STARTS.
            # Uses SLURM "after:jobid+minutes" so jobs start sequentially even when
            # many nodes are idle (prevents Daytona sandbox creation burst).
            # With chain_batch_size=K, K jobs fire together, then the next K wait
            # for the first job of the previous batch to have started + delay.
            if self.config.stagger_delay > 0 and self._dep_chain:
                cbs = max(self.config.chain_batch_size, 1)
                chain_len = len(self._dep_chain)
                # Determine which batch this job belongs to
                current_batch = chain_len // cbs
                if current_batch > 0:
                    # Depend on the first job of the previous batch
                    prev_batch_first = (current_batch - 1) * cbs
                    prev_job = self._dep_chain[prev_batch_first]
                    if not prev_job.startswith(("DRY_", "FAILED_")):
                        stagger_dep = f"after:{prev_job}+{self.config.stagger_delay}"
                        if job_dependency:
                            job_dependency = f"{job_dependency},{stagger_dep}"
                        else:
                            job_dependency = stagger_dep
                        if chain_len % cbs == 0:
                            log(f"  Stagger: batch {current_batch} boundary, wait {self.config.stagger_delay}m after job {prev_job} starts")
                        else:
                            log(f"  Stagger: batch {current_batch} (pos {chain_len % cbs}/{cbs}), wait {self.config.stagger_delay}m after job {prev_job} starts", verbose_only=True)

            # v6: Extract run_tag_override for disk-based resume
            resume_run_tag = self._resume_run_tags.get(hf_model)

            # DP: use DP sbatch when dp_nodes > 0
            actual_sbatch = self.config.dp_sbatch_script if self.config.dp_nodes > 0 else self.config.sbatch_script

            # API mode: dispatch hosted-endpoint models to the API sbatch
            # (no GPU, no vLLM). Vllm-served models hit the else branch
            # below with byte-identical behavior to the pre-API listener.
            api_cfg = get_api_config(hf_model, api_configs)
            is_api_mode = api_cfg is not None
            target_node = None
            pack_port = None
            extra_env: Dict[str, str] = {}
            sbatch_model_override: Optional[str] = None
            if is_api_mode:
                actual_sbatch = self.config.api_sbatch_script
                sbatch_model_override = api_cfg["litellm_model"]
                # Resolve preset → cap (honor n_concurrent_cap from yaml + per-preset cap)
                preset_for_dataset = next(
                    (name for name, p in PRESETS.items()
                     if dataset_hf in p.get("datasets", [])),
                    None,
                )
                preset_caps = (api_configs or {}).get("preset_n_concurrent_caps", {}) or {}
                model_cap = api_cfg.get("n_concurrent_cap", self.config.n_concurrent)
                preset_cap = preset_caps.get(preset_for_dataset, model_cap)
                n_eff = min(self.config.n_concurrent, model_cap, preset_cap)
                ak_string = _build_api_agent_kwargs_string(api_cfg.get("agent_kwargs", {}))
                extra_env.update({
                    "EVAL_API_BASE": api_cfg["api_base"] or "",
                    "EVAL_API_KEY_ENV": api_cfg["api_key_env"] or "",
                    "EVAL_API_AGENT_KWARGS": ak_string,
                    "EVAL_N_CONCURRENT": str(n_eff),
                    "EVAL_DB_MODEL_NAME": hf_model,
                })
                log(f"  [API] {hf_model} → {sbatch_model_override} "
                    f"(api_base={api_cfg['api_base']}, key_env={api_cfg['api_key_env']}, "
                    f"n_concurrent={n_eff}; caps: model={model_cap}, preset={preset_cap}/{preset_for_dataset})")
            elif pack_node_list:
                # Node packing: assign a target node and port based on GPU slots
                effective_tp = self.config.tp_size
                if vllm_overrides and "EVAL_VLLM_TENSOR_PARALLEL_SIZE" in vllm_overrides:
                    effective_tp = int(vllm_overrides["EVAL_VLLM_TENSOR_PARALLEL_SIZE"])
                effective_dp = self.config.dp_size
                total_gpus = effective_tp * effective_dp
                # Find a node with enough free GPU slots
                while pack_node_idx < len(pack_node_list):
                    used = pack_node_gpu_used.get(pack_node_idx, 0)
                    if used + total_gpus <= pack_gpus_per_node:
                        target_node = pack_node_list[pack_node_idx]
                        pack_node_gpu_used[pack_node_idx] = used + total_gpus
                        # Assign a non-overlapping port for this job
                        port_offset = pack_node_port_next.get(pack_node_idx, 0)
                        pack_port = 10000 + port_offset
                        pack_node_port_next[pack_node_idx] = port_offset + max(effective_dp, 1)
                        break
                    pack_node_idx += 1
                if target_node:
                    log(f"  Pack: {target_node} (GPUs {pack_node_gpu_used[pack_node_idx]}/{pack_gpus_per_node}, port {pack_port})", verbose_only=True)

            # Pass listener-assigned port to sbatch when packing (vLLM mode only)
            if pack_port is not None:
                extra_env["EVAL_VLLM_PORT"] = str(pack_port)

            slurm_job_id, job_name = submit_eval(
                hf_model,
                dataset_hf,
                bench_id,
                actual_sbatch,
                sbatch_params=sbatch_params,
                dry_run=self.config.dry_run,
                upload_username=self.config.upload_username,
                timeout_multiplier=self.config.timeout_multiplier,
                vllm_overrides=vllm_overrides if (vllm_overrides and not is_api_mode) else None,
                dependency=job_dependency,
                eval_config=self.config.eval_config if self.config.eval_config else None,
                conda_env=model_conda_env,
                run_tag_override=resume_run_tag,
                dp_nodes=0 if is_api_mode else self.config.dp_nodes,
                nodelist=target_node,
                extra_env=extra_env,
                skip_gpu_request=is_api_mode,
                sbatch_model_override=sbatch_model_override,
            )

            if slurm_job_id:
                if self.config.dry_run:
                    node_str = f" on {target_node}" if target_node else ""
                    log(f"  -> Would submit as SLURM job (job_name={job_name}){node_str}")
                    self._dep_chain.append(f"DRY_{idx}")
                else:
                    log(f"  -> Submitted as SLURM job {slurm_job_id} (job_name={job_name})")
                    self._submitted_jobs.add(slurm_job_id)
                    self._dep_chain.append(slurm_job_id)
                    # Fold freshly-submitted JID into active_ids so later jobs in
                    # the same iteration see it when computing sliding-window deps.
                    # Without this, --batch-size silently has no effect in --once
                    # mode with many models per priority file (active_ids was
                    # snapshotted before this loop began).
                    if batch_size and batch_size > 0:
                        active_ids.add(slurm_job_id)
                submitted += 1
            else:
                log(f"  -> Submission failed")
                self._dep_chain.append(f"FAILED_{idx}")

            if not self.config.dry_run and self.config.submission_delay > 0:
                time.sleep(self.config.submission_delay)

        return submitted

    def run(self) -> None:
        """Main event loop."""
        # Log configuration
        hdr = (
            f"lookback={self.config.lookback_days}d, "
            f"every {self.config.check_interval_hours}h, "
            f"sbatch={self.config.sbatch_script}"
        )
        log(f"Starting listener v3 for datasets={self.config.datasets}: {hdr}")
        log(
            f"Job logic: restart if 'Started' and started_at > {self.config.stale_job_hours}h ago, "
            f"restart+scancel if 'Pending' and submitted_at > {self.config.stale_pending_hours}h ago, "
            f"skip if 'Finished'"
        )
        log(f"Dry run mode: {self.config.dry_run}")
        log(f"Run once mode: {self.config.run_once}")
        if self.config.force_reeval:
            log("WARNING: --force-reeval is ON — bypassing DB status checks, will re-submit even if Finished")
        log(f"Check HF exists: {self.config.check_hf_exists}")
        log(f"Require priority list: {self.config.require_priority_list}")

        if self.config.priority_models:
            mode_desc = "filter_only (skip non-priority)" if self.config.priority_mode == "filter_only" else "priority_first (all models, priority first)"
            log(f"Priority mode: {mode_desc}, {len(self.config.priority_models)} model(s) in list")
            if self.config.priority_file:
                log(f"Priority file: {self.config.priority_file} (hot-reloaded each iteration)")
            if self.config.verbose:
                for m in sorted(self.config.priority_models):
                    log(f"  - {m}")
        else:
            log("Priority: disabled (no priority file or empty)")

        if self.config.blacklisted_models:
            log(f"Blacklist: {len(self.config.blacklisted_models)} model(s) from {self.config.blacklist_file}")
            if self.config.verbose:
                for m in sorted(self.config.blacklisted_models):
                    log(f"  - {m}")
        else:
            log("Blacklist: disabled (no blacklist file or empty)")

        # Log sbatch parameters
        sbatch_params = SbatchParams(
            n_concurrent=self.config.n_concurrent,
            n_attempts=self.config.n_attempts,
            gpu_memory_util=self.config.gpu_memory_util,
            error_threshold=self.config.error_threshold,
            vllm_max_retries=self.config.vllm_max_retries,
            agent_parser=self.config.agent_parser,
            slurm_time=self.config.slurm_time,
            enable_thinking=self.config.enable_thinking,
            agent_name=self.config.agent_name,
            slurm_partition=self.config.slurm_partition,
            slurm_account=self.config.slurm_account,
            tp_size=self.config.tp_size,
            dp_size=self.config.dp_size,
            timeout_multiplier=self.config.timeout_multiplier,
            config_yaml=self.config.config_yaml,
            max_output_tokens=self.config.max_output_tokens,
            auto_snapshot=self.config.auto_snapshot,
            agent_envs=self.config.agent_envs,
            pinggy_url=self.config.pinggy_url,
            pinggy_token=self.config.pinggy_token,
        )
        log(f"Sbatch params: {sbatch_params}")

        # Log v3 enhancement status
        log(f"[v3] Max SLURM jobs per listener: {self.config.max_jobs_submitted}")
        log(f"[v3] Daytona resource check: {'enabled' if self.config.check_daytona_resources else 'disabled'}")
        log(f"[v3] Timeout-aware dedup: {'enabled' if self.config.timeout_aware else 'disabled'}")
        if self.config.timeout_multiplier != DEFAULT_TIMEOUT_MULTIPLIER:
            log(f"[v3] Timeout multiplier: {self.config.timeout_multiplier}")
        if self.config.stagger_delay > 0:
            log(f"[v6] Stagger delay: {self.config.stagger_delay}m between batches of {self.config.chain_batch_size} jobs (SLURM after: chain)")

        # Enhancement 3: Daytona resource pre-flight check at startup
        if self.config.check_daytona_resources:
            ok = check_daytona_resources(
                self.config.daytona_sandbox_limit,
                self.config.daytona_warning_buffer,
            )
            if not ok:
                log("ERROR: Daytona resources at limit. Exiting.")
                sys.exit(1)

        while True:
            try:
                # Enhancement 3: Optional per-iteration Daytona resource check
                if self.config.check_daytona_resources:
                    ok = check_daytona_resources(
                        self.config.daytona_sandbox_limit,
                        self.config.daytona_warning_buffer,
                    )
                    if not ok:
                        log("WARNING: Daytona resources at limit, skipping this iteration")
                        if self.config.run_once or self.config.dry_run:
                            break
                        hours = self.config.check_interval_hours
                        log(f"Sleeping for {hours} hours...\n")
                        time.sleep(self.config.check_interval_seconds)
                        continue

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
        description="Unified Eval Listener v4 - Run models on benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
See the module docstring (top of file) for detailed flag reference with tuning
guidance. Quick summary below.

Presets: aider, bfcl, medagentbench, gaia, financeagent, swebench, v2, tb2, v1

v4 new:  --blacklist-file PATH   Block models from eval (overrides priority list)
v3 opt-in enhancements (all backward compatible):
  --error-threshold N             Unified invalid error threshold
  --max-jobs-submitted N          Per-listener SLURM job limit
  --check-daytona-resources       Daytona sandbox pre-flight check
  --track-model-retries           Deprioritize repeatedly-started models
  --timeout-aware                 Dedup by model+benchmark+timeout_multiplier

Examples:
  python unified_eval_listener_v4.py --preset v2 \\
    --priority-file priority_models.txt

  python unified_eval_listener_v4.py --preset v2 \\
    --priority-file priority_models.txt \\
    --blacklist-file bad_models.txt

  python unified_eval_listener_v4.py --preset v2 --dry-run --once --verbose
        """,
    )

    # Preset configuration
    parser.add_argument(
        "--preset", "-p",
        choices=list(PRESETS.keys()),
        help="Use a preset configuration (aider, bfcl, medagentbench, gaia, financeagent, swebench, v2, tb2, v1)",
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
    parser.add_argument(
        "--log-dir",
        help=f"Directory for listener logs (default: {DEFAULT_LOG_DIR}, env: EVAL_LISTENER_LOG_DIR)",
    )

    # Cluster configuration
    parser.add_argument(
        "--cluster-config",
        help="Path to cluster config YAML (e.g. eval/clusters/jupiter.yaml). "
             "Provides cluster-specific defaults for SLURM, paths, proxy, and hardware. "
             "CLI flags still override cluster config values.",
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
    parser.add_argument(
        "--blacklist-file",
        help="Path to blacklisted models file (one model per line). "
             "Models in this file are never submitted. Overrides priority list.",
    )
    parser.add_argument(
        "--priority-mode",
        choices=["filter_only", "priority_first"],
        help='Priority mode: "filter_only" (default) only evaluates priority models; '
             '"priority_first" evaluates all models but submits priority ones first',
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
    # Enhancement 1: Unified error threshold (with backward-compat alias)
    parser.add_argument(
        "--error-threshold",
        type=int,
        dest="error_threshold",
        help=f"Max invalid errors before abort upload (default: {DEFAULT_ERROR_THRESHOLD})",
    )
    parser.add_argument(
        "--daytona-threshold",
        type=int,
        dest="error_threshold_compat",
        help=f"Alias for --error-threshold (backward compat, default: {DEFAULT_ERROR_THRESHOLD})",
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
        "--slurm-account",
        help="SLURM account for job submission (e.g. 'reformo'). "
             "Overrides the #SBATCH --account in the sbatch script.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        choices=[1, 2, 4],
        help=f"vLLM tensor parallel size — number of GPUs per model "
             f"(default: {DEFAULT_TP_SIZE})",
    )
    parser.add_argument(
        "--dp-size",
        type=int,
        default=1,
        choices=[1, 2, 4, 8],
        help="vLLM native data-parallel size — number of model replicas. "
             "Total GPUs = tp_size × dp_size. vLLM load-balances requests "
             "across replicas internally. (default: 1)",
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

    # v3 Enhancement 2: Per-listener SLURM job throttle
    parser.add_argument(
        "--max-jobs-submitted",
        type=int,
        help=f"Per-listener SLURM job limit. Each listener tracks its own "
             f"submitted jobs independently (default: {DEFAULT_MAX_JOBS_SUBMITTED})",
    )

    # v3 Enhancement 3: Daytona resource pre-flight check
    parser.add_argument(
        "--check-daytona-resources",
        action="store_true",
        help="Query Daytona API for active sandbox count; skip if at limit. "
             "Requires DAYTONA_API_KEY in env",
    )
    parser.add_argument(
        "--daytona-sandbox-limit",
        type=int,
        help=f"Max expected active sandboxes (default: {DEFAULT_DAYTONA_SANDBOX_LIMIT})",
    )
    parser.add_argument(
        "--daytona-warning-buffer",
        type=float,
        help=f"Warn when active sandboxes reach this fraction of limit "
             f"(default: {DEFAULT_DAYTONA_WARNING_BUFFER})",
    )

    # v3 Enhancement 5: Timeout-config-sensitive dedup
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        help=f"Harbor timeout multiplier, stored in DB job config "
             f"(default: {DEFAULT_TIMEOUT_MULTIPLIER})",
    )
    parser.add_argument(
        "--timeout-aware",
        action="store_true",
        help="Dedup jobs by model+benchmark+agent+timeout_multiplier instead "
             "of just model+benchmark. Allows same model with different configs",
    )

    # Baseline model configs (per-model vLLM overrides)
    parser.add_argument(
        "--baseline-model-configs",
        help="Path to YAML mapping baseline models to vLLM serving params "
             "(e.g., eval/baseline_model_configs.yaml)",
    )

    # API model config (per-model serving config for Together / OpenAI / Anthropic)
    parser.add_argument(
        "--api-model-config",
        default="eval/configs/api_model_configs.yaml",
        help="Path to YAML mapping API-served models (together_ai/, openai/, "
             "anthropic/...) to api_base + key env + agent kwargs. Models that "
             "match are dispatched to eval/unified_eval_api_harbor.sbatch (no GPU). "
             "Missing file is OK — listener falls back to vLLM path for all models.",
    )

    # Harbor config
    parser.add_argument(
        "--harbor-config",
        help="Path to Harbor YAML config (parsed for timeout_multiplier, "
             "resource overrides; passed as EVAL_HARBOR_CONFIG to sbatch)",
    )
    parser.add_argument(
        "--config-yaml",
        help="Override Harbor eval config YAML filename (e.g. 'ablation_interleaved_true_16k.yaml'). "
             "Overrides preset default. Resolved from eval/configs/ on the compute node.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override max output tokens for LLM calls (default: 16384). "
             "Sets both max_tokens and model_info.max_output_tokens in the agent.",
    )

    # Pre-download model weights
    parser.add_argument(
        "--pre-download",
        action="store_true",
        help="Pre-download all model weights on login node before submitting jobs. "
             "Essential for no-internet compute nodes (Leonardo, Jupiter).",
    )

    # Sliding-window batch dependencies
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Max concurrent jobs via sliding-window SLURM dependencies. "
             "Job N depends on job N-batch_size finishing (afterany), "
             "so at most batch-size jobs run at once.",
    )

    # Conda environment selector
    parser.add_argument(
        "--conda-env",
        default="otagent",
        help="Conda environment to use for eval jobs. 'otagent2' has vLLM 0.17+ "
             "for Qwen3.5 and newer architectures. Available envs are defined in "
             "the cluster config YAML. (default: otagent)",
    )

    # v6: Disk-based resume
    parser.add_argument(
        "--jobs-dir",
        nargs="+",
        default=None,  # resolved in build_config from cluster config / env / fallback
        help="Path(s) to eval jobs directories for disk-based resume scanning. "
             "Can specify multiple dirs. (default: $EVAL_JOBS_DIR or cluster config paths.eval_jobs_dir)",
    )
    parser.add_argument(
        "--no-disk-resume",
        action="store_true",
        help="Disable v6 disk-based resume scanning.",
    )
    parser.add_argument(
        "--resume-only",
        action="store_true",
        help="Only submit resume jobs from disk scan, skip all fresh submissions.",
    )
    parser.add_argument(
        "--force-reeval",
        action="store_true",
        help="Force re-evaluation: bypass DB status check (submit even if Finished/Started). "
             "Use with --priority-file to re-run specific models.",
    )
    parser.add_argument(
        "--dp-nodes",
        type=int,
        default=0,
        help="Use DP (data-parallel) eval with N SLURM nodes. "
             "0 = single-node (default). Each node runs shards_per_node vLLM replicas (4/TP).",
    )
    parser.add_argument(
        "--inherit-log",
        nargs="+",
        default=None,
        help="Path(s) to previous listener log file(s). Seeds _submitted_jobs with SLURM IDs "
             "still active in squeue. Supports multiple logs for chained takeovers. "
             "Future --inherit-log on THIS listener's log will also pick up inherited IDs.",
    )
    parser.add_argument(
        "--submission-delay",
        type=float,
        default=1.0,
        help="Seconds to sleep between sbatch submissions (default: 1.0). "
             "Increase to avoid Daytona rate limits (e.g. 30 for 600 sandboxes/min).",
    )
    parser.add_argument(
        "--stagger-delay",
        type=int,
        default=0,
        help="Minutes between job starts via SLURM 'after:' dependency chain (default: 0 = disabled). "
             "Each batch of --chain-batch-size jobs waits N minutes after the previous batch STARTS. "
             "Prevents Daytona sandbox burst when many pending jobs start simultaneously. "
             "Minimum 1 (SLURM after: granularity is minutes).",
    )
    parser.add_argument(
        "--chain-batch-size",
        type=int,
        default=1,
        help="Jobs per stagger batch (default: 1). With --stagger-delay=1 --chain-batch-size=10, "
             "10 jobs fire immediately, then the next 10 wait 1 minute after the first batch starts. "
             "Only meaningful when --stagger-delay > 0.",
    )
    parser.add_argument(
        "--pack-jobs",
        action="store_true",
        help="Pack multiple jobs onto the same node. Queries idle nodes and assigns "
             "jobs round-robin so that GPUs_PER_NODE / TP_SIZE jobs share one node.",
    )
    parser.add_argument(
        "--resume-error-threshold",
        type=int,
        default=10,
        help="Min infrastructure errors to trigger resume for completed jobs. "
             "(default: 3)",
    )
    parser.add_argument(
        "--max-resume-count",
        type=int,
        default=5,
        help="Max times to resume a job dir before giving up. "
             "(default: 5)",
    )

    # Execution mode
    snapshot_group = parser.add_mutually_exclusive_group()
    snapshot_group.add_argument(
        "--auto-snapshot",
        action="store_true",
        default=None,
        dest="auto_snapshot",
        help="Enable Daytona auto_snapshot (overrides YAML config)",
    )
    snapshot_group.add_argument(
        "--no-auto-snapshot",
        action="store_false",
        dest="auto_snapshot",
        help="Disable Daytona auto_snapshot (overrides YAML config)",
    )
    parser.add_argument(
        "--pinggy-url",
        type=str,
        default=None,
        help="Pinggy persistent URL for tunnel (e.g., dadccqeqqf.a.pinggy.link). Used by installed agents.",
    )
    parser.add_argument(
        "--pinggy-token",
        type=str,
        default=None,
        help="Pinggy auth token for SSH tunnel.",
    )
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
    """Build configuration from args, env vars, cluster config, and preset defaults.

    Resolution order for most fields:
        CLI flag > Preset > Cluster config > Hardcoded default
    """
    global _CLUSTER_CONFIG, CONDA_ENV_PATHS

    # --- Load cluster config (if provided) ---
    cluster_config: Optional[Dict[str, Any]] = None
    if args.cluster_config:
        cluster_config = load_cluster_config(args.cluster_config)
        _CLUSTER_CONFIG = cluster_config
        # Override CONDA_ENV_PATHS from cluster config
        if cluster_config.get("conda_envs"):
            CONDA_ENV_PATHS = cluster_config["conda_envs"]

    # Helper: get cluster config value
    def _cc(key: str, default: Any = None) -> Any:
        if cluster_config is None:
            return default
        return cluster_config.get(key, default)

    def _cc_p(key: str, default: Any = None) -> Any:
        if cluster_config is None:
            return default
        return cluster_config.get("paths", {}).get(key, default)

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

    # Resolve sbatch script: CLI > ENV > Preset > Cluster config > Default
    sbatch_script = (
        args.sbatch_script
        or os.getenv("EVAL_LISTENER_SBATCH")
        or preset_config.get("sbatch_script")
        or _cc_p("sbatch_script", DEFAULT_SBATCH_SCRIPT)
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

    # Resolve log file (CLI --log-dir > ENV > Cluster config > default)
    log_dir = Path(
        args.log_dir
        or os.getenv("EVAL_LISTENER_LOG_DIR")
        or _cc_p("listener_logs_dir", DEFAULT_LOG_DIR)
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    suffix = preset_config.get("log_suffix", "unified")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    resume_tag = "_resume" if args.resume_only else ""
    dryrun_tag = "_dryrun" if args.dry_run else ""

    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = log_dir / f"{suffix}_eval_listener_v6{resume_tag}{dryrun_tag}_{current_time}.log"

    # Resolve priority file: CLI > ENV
    priority_file = args.priority_file or os.getenv("EVAL_LISTENER_PRIORITY_FILE")
    priority_models = load_priority_models(priority_file)

    # Resolve blacklist file: CLI > ENV
    blacklist_file = args.blacklist_file or os.getenv("EVAL_LISTENER_BLACKLIST_FILE")
    blacklisted_models = load_blacklist(blacklist_file)

    # Resolve priority mode: CLI > ENV > default
    priority_mode = (
        args.priority_mode
        or os.getenv("EVAL_LISTENER_PRIORITY_MODE")
        or "filter_only"
    )

    # Resolve boolean flags: CLI > ENV > Preset
    require_priority = args.require_priority_list or _env_bool("EVAL_LISTENER_REQUIRE_PRIORITY_LIST")
    dry_run = args.dry_run or _env_bool("EVAL_LISTENER_DRY_RUN")
    check_hf_exists = (
        args.check_hf_exists
        or _env_bool("EVAL_LISTENER_CHECK_HF_EXISTS")
        or preset_config.get("check_hf_exists", False)
    )

    # Resolve sbatch parameters: CLI > Preset > Cluster config > Default
    def _resolve(cli_val, preset_key: str, default):
        if cli_val is not None:
            return cli_val
        return preset_config.get(preset_key, default)

    n_concurrent = _resolve(args.n_concurrent, "n_concurrent", DEFAULT_N_CONCURRENT)
    n_attempts = _resolve(args.n_attempts, "n_attempts", DEFAULT_N_ATTEMPTS)
    gpu_memory_util = _resolve(args.gpu_memory_util, "gpu_memory_util", DEFAULT_GPU_MEMORY_UTIL)

    # Enhancement 1: Resolve error_threshold with backward compat
    error_threshold_cli = args.error_threshold
    if error_threshold_cli is None:
        error_threshold_cli = getattr(args, 'error_threshold_compat', None)
    error_threshold = _resolve(error_threshold_cli, "error_threshold", DEFAULT_ERROR_THRESHOLD)

    vllm_max_retries = _resolve(args.vllm_max_retries, "vllm_max_retries", DEFAULT_VLLM_MAX_RETRIES)
    agent_parser = _resolve(args.agent_parser, "agent_parser", DEFAULT_AGENT_PARSER)
    slurm_time = _resolve(args.slurm_time, "slurm_time", _cc("slurm_time", DEFAULT_SLURM_TIME))
    agent_name = _resolve(args.agent_name, "agent_name", DEFAULT_AGENT_NAME)
    slurm_partition = _resolve(args.slurm_partition, "slurm_partition", _cc("slurm_partition", DEFAULT_SLURM_PARTITION))
    slurm_account = _resolve(args.slurm_account, "slurm_account", _cc("slurm_account", DEFAULT_SLURM_ACCOUNT))
    tp_size = _resolve(args.tp_size, "tp_size", DEFAULT_TP_SIZE)
    dp_size = args.dp_size if args.dp_size else 1
    enable_thinking = args.enable_thinking or preset_config.get("enable_thinking", DEFAULT_ENABLE_THINKING)

    # Resolve upload_username: CLI > ENV > current OS user
    upload_username = (
        args.upload_username
        or os.getenv("EVAL_UPLOAD_USERNAME")
        or getpass.getuser()
    )

    # Enhancement 2: SLURM throttle
    max_jobs_submitted = (
        args.max_jobs_submitted
        if args.max_jobs_submitted is not None
        else int(os.getenv("EVAL_LISTENER_MAX_JOBS", str(DEFAULT_MAX_JOBS_SUBMITTED)))
    )

    # Enhancement 3: Daytona resource check
    check_daytona = args.check_daytona_resources
    daytona_sandbox_limit = (
        args.daytona_sandbox_limit
        if args.daytona_sandbox_limit is not None
        else DEFAULT_DAYTONA_SANDBOX_LIMIT
    )
    daytona_warning_buffer = (
        args.daytona_warning_buffer
        if args.daytona_warning_buffer is not None
        else DEFAULT_DAYTONA_WARNING_BUFFER
    )

    # Enhancement 5: Timeout-config-sensitive dedup
    timeout_multiplier = (
        args.timeout_multiplier
        if args.timeout_multiplier is not None
        else DEFAULT_TIMEOUT_MULTIPLIER
    )
    timeout_aware = args.timeout_aware

    # auto_snapshot: CLI > Preset > None (use YAML default)
    auto_snapshot = args.auto_snapshot
    if auto_snapshot is None:
        auto_snapshot = preset_config.get("auto_snapshot")

    # Config YAML: CLI > Preset > Default
    config_yaml = args.config_yaml or preset_config.get("config_yaml", "dcagent_eval_config.yaml")

    # Harbor config (parse eval-relevant fields for config-aware dedup)
    harbor_config = args.harbor_config or preset_config.get("harbor_config")
    eval_config = parse_harbor_eval_config(harbor_config)

    # Baseline model configs for per-model vLLM overrides
    baseline_model_configs_path = args.baseline_model_configs

    # API model configs for per-model API serving (no-op if file missing)
    api_model_config_path = args.api_model_config

    # Pre-download model weights
    pre_download = args.pre_download

    # Sliding-window batch dependencies
    batch_size = args.batch_size

    # Conda env selector
    conda_env = args.conda_env

    # Resolve jobs_dirs: CLI > ENV > Cluster config > Fallback
    fallback_jobs_dir = _cc_p("eval_jobs_dir", _FALLBACK_EVAL_JOBS_DIR)
    jobs_dirs = args.jobs_dir or [os.environ.get("EVAL_JOBS_DIR", fallback_jobs_dir)]

    # Resolve DP sbatch script: Cluster config > Default
    dp_sbatch_script = _cc_p("dp_sbatch_script", "eval/unified_eval_harbor_dp.sbatch")

    # agent_envs: resolve from preset. Format: "KEY_NAME,KEY_NAME=ENVVAR,KEY=literal"
    # - "SERPAPI_API_KEY" → reads os.environ["SERPAPI_API_KEY"], passes as SERPAPI_API_KEY=<value>
    # - "MODEL_API_KEY=OPENAI_API_KEY" → reads os.environ["OPENAI_API_KEY"], passes as MODEL_API_KEY=<value>
    # - "MODEL_FOR_TOOLS=openai/gpt-5.2" → literal value, passes as-is
    raw_agent_envs = preset_config.get("agent_envs", "")
    resolved_agent_envs = None
    if raw_agent_envs:
        resolved_pairs = []
        for item in raw_agent_envs.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                key, val = item.split("=", 1)
                env_val = os.environ.get(val)
                if env_val is not None:
                    resolved_pairs.append(f"{key}={env_val}")
                else:
                    resolved_pairs.append(f"{key}={val}")
            else:
                env_val = os.environ.get(item, "")
                if env_val:
                    resolved_pairs.append(f"{item}={env_val}")
                else:
                    log(f"WARNING: agent_envs: env var {item} not set, skipping")
        resolved_agent_envs = ",".join(resolved_pairs) if resolved_pairs else None

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
        priority_mode=priority_mode,
        check_hf_exists=check_hf_exists,
        dry_run=dry_run,
        run_once=args.once,
        verbose=args.verbose,
        # Sbatch parameters
        n_concurrent=n_concurrent,
        n_attempts=n_attempts,
        gpu_memory_util=gpu_memory_util,
        error_threshold=error_threshold,
        vllm_max_retries=vllm_max_retries,
        agent_parser=agent_parser,
        slurm_time=slurm_time,
        enable_thinking=enable_thinking,
        agent_name=agent_name,
        slurm_partition=slurm_partition,
        slurm_account=slurm_account,
        tp_size=tp_size,
        dp_size=dp_size,
        upload_username=upload_username,
        # Enhancement 2
        max_jobs_submitted=max_jobs_submitted,
        # Enhancement 3
        check_daytona_resources=check_daytona,
        daytona_sandbox_limit=daytona_sandbox_limit,
        daytona_warning_buffer=daytona_warning_buffer,
        # Enhancement 5
        timeout_multiplier=timeout_multiplier,
        timeout_aware=timeout_aware,
        config_yaml=config_yaml,
        max_output_tokens=args.max_output_tokens,
        auto_snapshot=auto_snapshot,
        blacklist_file=blacklist_file,
        blacklisted_models=blacklisted_models,
        # New features
        baseline_model_configs=baseline_model_configs_path,
        api_model_config=api_model_config_path,
        harbor_config=harbor_config,
        eval_config=eval_config,
        pre_download=pre_download,
        batch_size=batch_size,
        conda_env=conda_env,
        # v6: Disk-based resume
        jobs_dirs=jobs_dirs,
        enable_disk_resume=not args.no_disk_resume,
        resume_infra_error_threshold=args.resume_error_threshold,
        max_resume_count=args.max_resume_count,
        force_reeval=args.force_reeval,
        resume_only=args.resume_only,
        submission_delay=args.submission_delay,
        stagger_delay=max(args.stagger_delay, 0),
        chain_batch_size=max(args.chain_batch_size, 1),
        pack_jobs=args.pack_jobs,
        dp_nodes=args.dp_nodes,
        dp_sbatch_script=dp_sbatch_script,
        inherit_log=args.inherit_log,
        # Cluster config
        cluster_config=cluster_config,
        agent_envs=resolved_agent_envs,
        pinggy_url=args.pinggy_url,
        pinggy_token=args.pinggy_token,
    )


# ---------- Main ----------
def main() -> None:
    global _VERBOSE
    _load_secrets()
    args = parse_args()
    config = build_config(args)
    _VERBOSE = config.verbose
    if config.cluster_config:
        log(f"[v6] Cluster config: {config.cluster_config.get('cluster_name', '?')}")
    listener = EvalListener(config)
    listener.run()


if __name__ == "__main__":
    main()
