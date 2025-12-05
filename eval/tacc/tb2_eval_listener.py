#!/usr/bin/env python3
"""
Eval Listener — run ALL recent models on one or more benchmark datasets (HF repos).

Env:
  EVAL_LISTENER_LOOKBACK_DAYS   (int, default "100")
  EVAL_LISTENER_CHECK_HOURS     (float, default "12")
  EVAL_LISTENER_SBATCH          (default "tacc_eval_sandbox.sbatch")
  EVAL_LISTENER_LOG_DIR         (default "experiments/listener_logs")
  EVAL_LISTENER_DATASETS        comma/space/newline list of HF dataset repos
                                e.g. "org1/dsA,org2/dsB" or multiline

Behavior:
- Fetch recent models (time window only).
- For each model, submit sbatch(model_hf_name, dataset_hf, benchmark_id) for every dataset.
- Dedupe per (model_id, benchmark_id) in DB based on job_status:
  * No job exists → start job
  * Job exists with job_status='Started' and started_at > 24 hours ago → restart job
  * Job exists with job_status='Finished' → skip

Note: The sbatch script/job itself is responsible for creating the DB row with all required fields.
"""
import os
import sys

sys.path.insert(0, "/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard")

import json
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from unified_db.utils import get_supabase_client  # your code

# ---------- Config ----------
LOOKBACK_DAYS = int(os.getenv("EVAL_LISTENER_LOOKBACK_DAYS", "100"))
CHECK_INTERVAL_HOURS = float(os.getenv("EVAL_LISTENER_CHECK_HOURS", "4"))
CHECK_INTERVAL_SECONDS = int(CHECK_INTERVAL_HOURS * 60 * 60)

SBATCH_SCRIPT = os.getenv("EVAL_LISTENER_SBATCH", "tb2_eval_harbor.sbatch")
LOG_DIR = Path(os.getenv("EVAL_LISTENER_LOG_DIR", "experiments/listener_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "eval_listener.log"

DATASETS_RAW='DCAgent2/terminal_bench_2'
#DATASETS_RAW = os.getenv("EVAL_LISTENER_DATASETS", "")
HF_URL_RE = re.compile(r'https?://(?:www\.)?huggingface\.co/([^/\s]+)/([^/\s#?]+)')

PRINT_PREFIX = "[eval-listener]"

# Job status constants
JOB_STATUS_STARTED = "Started"
JOB_STATUS_FINISHED = "Finished"
STALE_JOB_HOURS = 24

def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{PRINT_PREFIX} {ts}  {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass

# ---------- Dataset list parsing ----------
def _parse_datasets(s: str) -> List[str]:
    # Split on commas, whitespace, or newlines; normalize full URLs to org/repo
    parts = [p.strip() for p in re.split(r"[,\s]+", s) if p.strip()]
    out = []
    for p in parts:
        m = HF_URL_RE.search(p)
        out.append(f"{m.group(1)}/{m.group(2)}" if m else p)
    # dedup while preserving order
    seen = set()
    uniq = []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq

DATASETS: List[str] = _parse_datasets(DATASETS_RAW)

# ---------- Helpers: HF parsing & benchmark resolution ----------
def _parse_hf_from_str(val: Optional[str]) -> Optional[str]:
    if not isinstance(val, str):
        return None
    m = HF_URL_RE.search(val)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None

def _resolve_hf_model_name(model_row: Dict) -> Optional[str]:
    v = model_row.get("name")
    if isinstance(v, str) and "/" in v and not v.startswith("hosted_vllm/"):
        return v
    for field in ("weights_location", "training_parameters", "url", "hf_url"):
        vv = model_row.get(field)
        if isinstance(vv, str):
            name = _parse_hf_from_str(vv)
            if name:
                return name
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

def _dataset_repo_name(dataset_hf: str) -> str:
    """
    Convert 'org/repo' or HF URL to 'repo'.
    """
    if not dataset_hf:
        return dataset_hf
    m = HF_URL_RE.search(dataset_hf)
    if m:
        return m.group(2)
    if "/" in dataset_hf:
        return dataset_hf.rsplit("/", 1)[-1]
    return dataset_hf

# Cache benchmark_id lookups in memory for speed
_BENCH_CACHE: Dict[str, Optional[str]] = {}

def resolve_benchmark_id_for_dataset(dataset_hf: str) -> Optional[str]:
    """
    Look up benchmarks.id where benchmarks.name == repo_name(dataset_hf).
    Returns None if no matching benchmark row exists.
    """
    repo_name = _dataset_repo_name(dataset_hf)
    if repo_name in _BENCH_CACHE:
        return _BENCH_CACHE[repo_name]
    try:
        client = get_supabase_client()
        resp = client.table('benchmarks').select('id,name').eq('name', repo_name).limit(1).execute()
        rows = resp.data or []
        bench_id = rows[0]['id'] if rows else None
        _BENCH_CACHE[repo_name] = bench_id
        if not bench_id:
            log(f"No benchmark row found for dataset '{dataset_hf}' (wanted name='{repo_name}').")
        return bench_id
    except Exception as e:
        log(f"ERROR resolving benchmark id for dataset '{dataset_hf}': {e}")
        return None

# ---------- Supabase: models & jobs ----------
def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

def _time_filters(q, since_iso: str):
    try:
        return q.gte('creation_time', since_iso)
    except Exception:
        return q.gte('created_at', since_iso)

def fetch_recent_models(days: int) -> List[Dict]:
    client = get_supabase_client()
    since = _iso(datetime.now(timezone.utc) - timedelta(days=days))
    try:
        resp = _time_filters(client.table('models').select('*'), since).execute()
        rows = list(resp.data or [])
    except Exception as e:
        log(f"ERROR: failed querying models by time: {e}")
        return []
    out: List[Dict] = []
    for r in rows:
        if r.get("created_by") == "precomputed_hf":
            continue
        out.append(r)
    return out

def check_job_status(model_id: str, benchmark_id: Optional[str]) -> Tuple[bool, Optional[str], Optional[datetime]]:
    """
    Check if a job exists for (model_id, benchmark_id) and its status.
    
    Returns:
        (job_exists, job_status, started_at)
        - job_exists: True if a job row exists
        - job_status: 'Started', 'Finished', or other enum values
        - started_at: datetime when job actually started (for staleness check)
    """
    if not benchmark_id:
        # No benchmark row → treat as no job exists
        return (False, None, None)
    
    try:
        client = get_supabase_client()
        q = (
            client.table('sandbox_jobs')
            .select('id,job_status,started_at')
            .eq('model_id', model_id)
            .eq('benchmark_id', benchmark_id)
            .order('created_at', desc=True)
            .limit(1)
        )
        data = (q.execute().data) or []
        
        if not data:
            return (False, None, None)
        
        job = data[0]
        job_status = job.get('job_status')
        started_at_str = job.get('started_at')
        
        started_at = None
        if started_at_str:
            try:
                started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
            except Exception:
                pass
        
        return (True, job_status, started_at)
    
    except Exception as e:
        log(f"WARNING: sandbox_jobs check failed for model_id={model_id}, benchmark_id={benchmark_id}: {e}")
        return (False, None, None)  # fail-open

def is_job_stale(started_at: Optional[datetime], hours: int = STALE_JOB_HOURS) -> bool:
    """Check if a job started more than the specified hours ago."""
    if not started_at:
        # If started_at is null but job exists with status='Started', treat as stale
        return True
    now = datetime.now(timezone.utc)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=timezone.utc)
    age = now - started_at
    return age > timedelta(hours=hours)

def should_start_job(model_id: str, benchmark_id: Optional[str]) -> Tuple[bool, str]:
    """
    Determine if a job should be started based on DB status.
    
    Returns:
        (should_start, reason)
    """
    job_exists, job_status, started_at = check_job_status(model_id, benchmark_id)
    
    if not job_exists:
        return (True, "no existing job")
    
    if job_status == JOB_STATUS_FINISHED:
        return (False, "job finished")
    
    if job_status == JOB_STATUS_STARTED:
        if is_job_stale(started_at):
            started_str = started_at.isoformat() if started_at else "null"
            return (True, f"stale job (started_at={started_str})")
        else:
            started_str = started_at.isoformat() if started_at else "null"
            return (False, f"job in progress (started_at={started_str})")
    
    # Unknown status - start job to be safe
    return (True, f"unknown job status: {job_status}")

# ---------- sbatch ----------
def _run(cmd: List[str]) -> Tuple[int, str]:
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_lines = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line.rstrip())
    code = proc.wait()
    return code, "\n".join(out_lines)

def submit_eval(hf_model_name: str, dataset_hf: str, benchmark_id: Optional[str]) -> Optional[str]:
    """
    Submit sbatch job. The job itself will create the DB row when it starts.
    
    sbatch args:
      $1 = model HF name
      $2 = dataset HF repo (org/repo)
      $3 = benchmark_id (uuid)  [optional]
    
    Returns:
        slurm_job_id if successful, None otherwise
    """
    cmd = ["sbatch", SBATCH_SCRIPT, hf_model_name, dataset_hf]
    if benchmark_id:
        cmd.append(str(benchmark_id))
    
    code, out = _run(cmd)
    log(f"sbatch: {' '.join(cmd)}\n{out}")
    
    if code != 0:
        return None
    
    m = re.search(r"Submitted batch job (\d+)", out)
    slurm_job_id = m.group(1) if m else None
    
    return slurm_job_id

# ---------- Main ----------
def main():
    if not DATASETS:
        log("ERROR: EVAL_LISTENER_DATASETS is empty. Set it to one or more HF dataset repos.")
        sys.exit(2)

    hdr = f"lookback={LOOKBACK_DAYS}d, every {CHECK_INTERVAL_HOURS}h, sbatch={SBATCH_SCRIPT}"
    log(f"Starting listener for datasets={DATASETS}: {hdr}")
    log(f"Job logic: restart if 'Started' and started_at > {STALE_JOB_HOURS}h ago, skip if 'Finished'")

    while True:
        try:
            log("Checking for new models...")
            models = fetch_recent_models(LOOKBACK_DAYS)
            log(f"Found {len(models)} model(s) in window. Filtering...")

            submissions: List[Tuple[str, str, str, Optional[str], str]] = []  
            # (model_id, hf_model_name, dataset_hf, benchmark_id, reason)

            # Resolve all benchmarks up front (once per loop)
            dataset_to_bench: Dict[str, Optional[str]] = {ds: resolve_benchmark_id_for_dataset(ds) for ds in DATASETS}

            for m in models:
                model_id = str(m.get("id"))
                if not model_id:
                    continue

                hf_model = _resolve_hf_model_name(m)
                if not hf_model:
                    log(f"Skip: cannot resolve HF model for id={model_id}, name={m.get('name')}")
                    continue

                for dataset_hf in DATASETS:
                    bench_id = dataset_to_bench.get(dataset_hf)
                    
                    # Check DB status to decide if we should start
                    should_start, reason = should_start_job(model_id, bench_id)
                    
                    if should_start:
                        submissions.append((model_id, hf_model, dataset_hf, bench_id, reason))
                    else:
                        # Log skipped jobs at debug level
                        pass  # Could add verbose logging here if needed

            if not submissions:
                log("No eligible (model, dataset) pairs to submit.")
            else:
                log(f"Submitting {len(submissions)} eval(s)...")
                for mid, hf_model, dataset_hf, bench_id, reason in submissions:
                    log(f"Submitting: model={hf_model}, dataset={dataset_hf}, reason={reason}")
                    job_id = submit_eval(hf_model, dataset_hf, bench_id)
                    if job_id:
                        log(f"  → Submitted as SLURM job {job_id}")
                    else:
                        log(f"  → Submission failed")
                    time.sleep(1)

            hours = CHECK_INTERVAL_SECONDS / 3600.0
            log(f"Sleeping for {hours} hours...\n")
            time.sleep(CHECK_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            log("Interrupted by user. Exiting.")
            sys.exit(0)
        except Exception as e:
            log(f"ERROR in main loop: {e}. Backing off 30s.")
            time.sleep(30)

if __name__ == "__main__":
    main()

