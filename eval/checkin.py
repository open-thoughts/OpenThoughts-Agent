#!/usr/bin/env python3
"""Check-in dashboard: 'what's changed since I last looked?'

Shows newly-ended jobs, jobs about to hit walltime, with a per-row
RESUME / UPLOAD / WAIT recommendation. Persists last-seen state in
`eval/.local_notes/checkin_state.json`.

Usage:
    python eval/checkin.py                       # show diff + commit state
    python eval/checkin.py --no-commit           # show diff, don't update state
    python eval/checkin.py --reset               # forget all state
    python eval/checkin.py --warn-elapsed-h 10   # flag RUNNING jobs > 10h elapsed
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "database" / "unified_db"))
from eval.check_resume_needed import scan  # noqa: E402

DEFAULT_JOBS_DIR = "/e/data1/datasets/playground/mmlaion/shared/zhuang1_eval_jobs"
STATE_FILE = REPO / "eval" / ".local_notes" / "checkin_state.json"
DEFAULT_LOG_DIR = "eval/logs"
WALLTIME_CAP_H = 12  # SLURM walltime cap on Jupiter

# Exception types user does NOT want resumes to retry (these dominate failures
# but resume can't fix them — the model just can't do those tasks)
USER_EXCLUDED_EXC = {
    "SummarizationTimeoutError",
    "AgentTimeoutError",
    "ContextLengthExceededError",
}
TRUE_INFRA_EXC = {
    "DaytonaError",
    "DaytonaAuthenticationError",
    "DaytonaNotFoundError",
    "RewardFileNotFoundError",
    "RewardFileEmptyError",
    "EnvironmentSetupError",
    "AgentSetupError",
    "EngineError",
    "VerifierTimeoutError",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def db_finished_run_tags(run_tags: List[str]) -> Dict[str, str]:
    """Return {run_tag: 'Finished'|'Started'|...} for run_tags found in sandbox_jobs.

    Uses a single batched query. Empty dict if Supabase env not set.
    """
    if not run_tags or not os.environ.get("SUPABASE_URL"):
        return {}
    try:
        from supabase import create_client
        sb = create_client(
            os.environ["SUPABASE_URL"],
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"],
        )
    except Exception:
        return {}

    out: Dict[str, str] = {}
    # Batch in chunks of 50 to avoid URL-length limits
    for i in range(0, len(run_tags), 50):
        batch = run_tags[i:i + 50]
        try:
            r = sb.table("sandbox_jobs").select("job_name,job_status").in_("job_name", batch).execute()
        except Exception:
            continue
        for row in r.data:
            tag = row.get("job_name")
            status = row.get("job_status")
            # Prefer Finished over Started if multiple rows exist for same tag
            if tag and (tag not in out or status == "Finished"):
                out[tag] = status
    return out


def _parse_iso(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_state() -> Dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_check": None, "seen": {}}


def save_state(state: Dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def squeue_running() -> Dict[str, Tuple[str, str]]:
    """Return {jid: (elapsed, time_left)} for current user's RUNNING jobs."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", ""),
             "--format=%i|%T|%M|%L", "--noheader"],
            text=True, timeout=10,
        )
    except Exception:
        return {}
    res = {}
    for line in out.strip().splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 4 and parts[1] == "RUNNING":
            res[parts[0]] = (parts[2], parts[3])
    return res


def _hms_to_hours(s: str) -> float:
    """Parse SLURM 'D-HH:MM:SS' or 'HH:MM:SS' or 'MM:SS' into hours."""
    if "-" in s:
        d, rest = s.split("-", 1)
        days = int(d)
    else:
        days, rest = 0, s
    parts = rest.split(":")
    if len(parts) == 3:
        h, m, sec = (int(x) for x in parts)
    elif len(parts) == 2:
        h, m, sec = 0, int(parts[0]), int(parts[1])
    else:
        return 0.0
    return days * 24 + h + m / 60 + sec / 3600


def score_dir(run_dir: Path, n_total: int) -> Tuple[int, Dict]:
    """Compute resume-worth score from per-trial result.json files.

    Score = 3 * missing_tasks + 1 * incomplete_dirs + 1 * true_infra_errors.
    SummarizationTimeoutError + AgentTimeoutError + ContextLengthExceededError
    are explicitly NOT counted (model can't do those tasks).
    """
    detail = {"missing_tasks": 0, "incomplete_dirs": 0, "true_infra": 0,
              "summarization": 0, "agent_timeout": 0, "ctx_len": 0,
              "valid_reward": 0, "n_dirs": 0}
    if not run_dir.is_dir():
        return 0, detail

    task_attempted: set = set()
    for d in run_dir.iterdir():
        if not d.is_dir() or "__" not in d.name or d.name.startswith("_"):
            continue
        m = re.match(r"^(.+)__[A-Za-z0-9]{6,9}$", d.name)
        if not m:
            continue
        task = m.group(1)
        task_attempted.add(task)
        detail["n_dirs"] += 1
        rj = d / "result.json"
        if not rj.exists():
            detail["incomplete_dirs"] += 1
            continue
        try:
            data = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        exc = (data.get("exception_info") or {}).get("exception_type")
        vr = data.get("verifier_result") or {}
        rew = (vr.get("rewards") or {}).get("reward")
        if exc in TRUE_INFRA_EXC:
            detail["true_infra"] += 1
        elif exc == "SummarizationTimeoutError":
            detail["summarization"] += 1
        elif exc == "AgentTimeoutError":
            detail["agent_timeout"] += 1
        elif exc == "ContextLengthExceededError":
            detail["ctx_len"] += 1
        elif rew is not None:
            detail["valid_reward"] += 1

    detail["missing_tasks"] = max(0, _tasks_in_dataset_for_dir(run_dir, n_total) - len(task_attempted))
    score = 3 * detail["missing_tasks"] + detail["incomplete_dirs"] + detail["true_infra"]
    return score, detail


def _tasks_in_dataset_for_dir(run_dir: Path, n_total: int) -> int:
    """Best-effort: total UNIQUE tasks the dataset expects.

    n_total in result.json is total ATTEMPTS expected (= n_tasks * n_attempts).
    So n_unique_tasks = n_total / n_attempts. If we can't determine n_attempts,
    fall back to assuming 3.
    """
    config = run_dir / "config.json"
    n_attempts = 3
    if config.exists():
        try:
            data = json.loads(config.read_text())
            n_attempts = int(data.get("n_attempts") or 3)
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return max(0, n_total // max(1, n_attempts))


def recommend(score: int, detail: Dict, n_fires: int) -> Tuple[str, str]:
    """Return (action, reason) — action ∈ {RESUME, UPLOAD, WAIT}."""
    if n_fires >= 2:
        return "UPLOAD", f"already resumed (n_fires={n_fires}); 1-resume policy → upload"
    if score >= 3:
        bits = []
        if detail["missing_tasks"]: bits.append(f"{detail['missing_tasks']} missing task(s)")
        if detail["incomplete_dirs"]: bits.append(f"{detail['incomplete_dirs']} incomplete dir(s)")
        if detail["true_infra"]: bits.append(f"{detail['true_infra']} infra-err")
        return "RESUME", f"score={score} ({', '.join(bits)})"
    bits = [f"{detail['summarization']} summarization", f"{detail['agent_timeout']} agent-to",
            f"{detail['ctx_len']} ctx-len", f"{detail['valid_reward']} valid"]
    return "UPLOAD", f"score={score} (gap not recoverable; long-tail: {', '.join(b for b in bits if not b.startswith('0'))})"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-dir", default=DEFAULT_JOBS_DIR)
    ap.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    ap.add_argument("--no-commit", action="store_true",
                    help="Don't update last-check state (default: commit after run)")
    ap.add_argument("--reset", action="store_true",
                    help="Wipe all state and exit")
    ap.add_argument("--warn-elapsed-h", type=float, default=10.0,
                    help="Flag RUNNING jobs with elapsed > this many hours (default: 10)")
    ap.add_argument("--preset", default=None, help="Filter by preset (tb2, swebench, v2)")
    args = ap.parse_args()

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print(f"State wiped: {STATE_FILE}")
        return 0

    state = load_state()
    last_check = state.get("last_check")
    seen = state.get("seen", {})

    print(f"=== Check-in @ {_now()} ===")
    if last_check:
        delta = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last_check)
        print(f"Last check: {last_check}  (Δ {int(delta.total_seconds()/60)} min ago)")
    else:
        print("Last check: (first run)")
    print()

    rows = scan(
        jobs_dirs=[Path(args.jobs_dir)],
        preset_filter=args.preset,
        infra_error_threshold=10,
        max_resume_count=5,
        max_total_fires=2,
        log_dir=args.log_dir,
        show_rejected=False,
    )

    running = squeue_running()

    # Cross-check DB: any run_tag with Finished sandbox_jobs is already-uploaded
    db_status = db_finished_run_tags([r["run_tag"] for r in rows])

    new_ended = []   # newly classified as needs-action since last check
    about_to_cap = []
    in_flight_healthy = []
    auto_cleared = 0  # count of items skipped because DB row is Finished

    for r in rows:
        run_tag = r["run_tag"]
        status = r["status"]
        prev = seen.get(run_tag, {})
        prev_status = prev.get("status")

        # IN_FLIGHT → check elapsed, flag if approaching walltime
        if status == "IN_FLIGHT":
            jid = r["slurm_jid"]
            if jid in running:
                elapsed_h = _hms_to_hours(running[jid][0])
                left = running[jid][1]
                if elapsed_h >= args.warn_elapsed_h:
                    about_to_cap.append({**r, "elapsed_h": elapsed_h, "time_left": left})
                else:
                    in_flight_healthy.append({**r, "elapsed_h": elapsed_h, "time_left": left})
            continue

        # Skip dirs that already completed naturally (finished_at set AND status DONE)
        # — those auto-uploaded in their sbatch and need no user action.
        if status in ("DONE", "DONE_WITH_ERRORS", "AT_RESUME_LIMIT") and r.get("finished_at"):
            seen[run_tag] = {"status": status, "n_completed": r["n_completed"],
                             "n_total": r["n_total"], "n_fires": r["n_fires"],
                             "last_seen": _now()}
            continue

        # Skip dirs where DB row is Finished (= already manually uploaded).
        # The on-disk result.json may still have finished_at=None because
        # batch_upload_eval auto-sets it in memory only.
        if db_status.get(run_tag) == "Finished":
            seen[run_tag] = {"status": "DB_FINISHED",
                             "n_completed": r["n_completed"],
                             "n_total": r["n_total"], "n_fires": r["n_fires"],
                             "last_seen": _now()}
            if prev_status != "DB_FINISHED":
                auto_cleared += 1
            continue

        # Status transitioned (e.g., was IN_FLIGHT, now INCOMPLETE/DONE/EARLY_KILL)
        if status != prev_status:
            run_dir = Path(r["jobs_dir"]) / run_tag
            score, detail = score_dir(run_dir, r["n_total"] or 267)
            action, reason = recommend(score, detail, r["n_fires"])
            new_ended.append({**r, "score": score, "detail": detail,
                              "action": action, "reason": reason,
                              "prev_status": prev_status})

        # Update seen state
        seen[run_tag] = {
            "status": status, "n_completed": r["n_completed"],
            "n_total": r["n_total"], "n_fires": r["n_fires"],
            "last_seen": _now(),
        }

    # ----- Print report -----
    if new_ended:
        print(f"### NEWLY ENDED / NEEDS ACTION ({len(new_ended)})\n")
        print(f"{'Status':<18}{'JID':<10}{'Action':<8}{'Cov':<10}{'Score':<7}Model | Reason")
        print("-" * 130)
        for r in new_ended:
            cov = f"{r['n_completed']}/{r['n_total']}"
            model_short = (r["hf_model"] or "").split("/")[-1][:40]
            run_dir = Path(r["jobs_dir"]) / r["run_tag"]
            print(f"{r['status']:<18}{r['slurm_jid'] or '-':<10}{r['action']:<8}{cov:<10}{r['score']:<7}{model_short} | {r['reason']}")
            print(f"    └─ run_dir: {run_dir}")
        print()

    if about_to_cap:
        print(f"### ABOUT TO HIT WALLTIME (>{args.warn_elapsed_h}h elapsed) ({len(about_to_cap)})\n")
        print(f"{'JID':<10}{'Elapsed':<10}{'TimeLeft':<10}{'Cov':<10}Model")
        print("-" * 100)
        for r in about_to_cap:
            cov = f"{r['n_completed']}/{r['n_total']}"
            model_short = (r["hf_model"] or "").split("/")[-1][:50]
            print(f"{r['slurm_jid']:<10}{r['elapsed_h']:.1f}h     {r['time_left']:<10}{cov:<10}{model_short}")
        print()

    if not new_ended and not about_to_cap:
        msg = f"No new actions. {len(in_flight_healthy)} job(s) still in flight (healthy)"
        if auto_cleared:
            msg += f"; {auto_cleared} auto-cleared (DB Finished)"
        print(msg + ".")
    else:
        msg = f"({len(in_flight_healthy)} other jobs in flight, healthy"
        if auto_cleared:
            msg += f"; {auto_cleared} auto-cleared via DB"
        print(msg + ")")

    if not args.no_commit:
        state["last_check"] = dt.datetime.now(dt.timezone.utc).isoformat()
        state["seen"] = seen
        save_state(state)
        print(f"\nState committed → {STATE_FILE}")
    else:
        print("\n(--no-commit: state NOT updated)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
