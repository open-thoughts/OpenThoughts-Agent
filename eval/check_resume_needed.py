#!/usr/bin/env python3
"""Inspect harbor eval job dirs and report which need resume.

Reuses _parse_job_dir + INFRA_ERROR_TYPES + PRESETS + get_active_model_dataset_pairs
from eval.unified_eval_listener so classification matches the listener's resume scanner.

Usage:
    # Human-readable table for all jobs across both presets (eval-org dir)
    python eval/check_resume_needed.py \
        --jobs-dir /e/data1/datasets/playground/mmlaion/shared/zhuang1_eval_jobs

    # Only show jobs that would resume (skip DONE / IN_FLIGHT / PARTIAL_OK)
    python eval/check_resume_needed.py --jobs-dir <dir> --needs-resume-only

    # Filter by preset
    python eval/check_resume_needed.py --jobs-dir <dir> --preset tb2

    # CSV output for orchestrator consumption
    python eval/check_resume_needed.py --jobs-dir <dir> --csv > /tmp/candidates.csv

    # Permanently exclude run dirs from resume (writes .no-resume marker file).
    # Rejected dirs are hidden from inspector output by default.
    python eval/check_resume_needed.py --jobs-dir <dir> --reject <run_tag1> --reject <run_tag2>

    # Bulk-reject every dir currently classified as needs-resume:
    python eval/check_resume_needed.py --jobs-dir <dir> --reject-needs-resume --yes

    # Show rejected dirs (default: hidden):
    python eval/check_resume_needed.py --jobs-dir <dir> --show-rejected

Status taxonomy:
    DONE              n_completed == n_total, infra_errors <= threshold      (no resume)
    DONE_WITH_ERRORS  n_completed == n_total, infra_errors > threshold       (RESUME)
    INCOMPLETE        n_completed < n_total, finished_at is None             (RESUME — 12h cap case)
    PARTIAL           n_completed < n_total, finished_at != None,
                      infra_errors > threshold                                (RESUME)
    PARTIAL_OK        n_completed < n_total, finished_at != None,
                      infra_errors <= threshold                               (no resume — gave up)
    EARLY_KILL        no result.json, n_total == 0                           (RESUME)
    IN_FLIGHT         (model, dataset) is currently running in squeue        (no resume — wait)
    AT_RESUME_LIMIT   total fires >= --max-total-fires (1 orig + 1 resume    (NO RESUME — 24h cap)
                      = 2 by default), OR resume_count >= --max-resume-count
    REJECTED          .no-resume marker file present in run dir              (NO RESUME — user-rejected)
    UNKNOWN           failed to parse                                         (skip)

Total-fires accounting (24h cap):
    Each original fire + each resume produces one slurm log file
    (`eval/logs/data_<JID>.out`) with a `Run tag: <run_tag>` line. We count those
    matches across all logs and combine with `meta.env`'s RESUME_COUNT (which is
    only updated when sbatch reaches its post-harbor write — SIGTERM-killed jobs
    don't update it). Effective fire count = max(log-count, resume_count + 1).
    Default `--max-total-fires 2` enforces the 1-original-plus-1-resume hard cap;
    bump if you want more headroom or set higher if old logs are pruned.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Make the listener importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

# Import strictly the things we need; the listener's top-level is mostly defs/constants
from eval.unified_eval_listener import (  # noqa: E402
    INFRA_ERROR_TYPES,
    PRESETS,
    _parse_job_dir,
    get_active_model_dataset_pairs,
)

DEFAULT_INFRA_ERROR_THRESHOLD = 10  # matches listener CLI default --resume-error-threshold
DEFAULT_MAX_RESUME_COUNT = 5  # matches listener CLI default --max-resume-count
DEFAULT_MAX_TOTAL_FIRES = 2  # 1 original + 1 resume = 24h on a 12h-cap cluster


# ---- Status classification ---------------------------------------------------

STATUS_DONE = "DONE"
STATUS_DONE_WITH_ERRORS = "DONE_WITH_ERRORS"
STATUS_INCOMPLETE = "INCOMPLETE"
STATUS_PARTIAL = "PARTIAL"
STATUS_PARTIAL_OK = "PARTIAL_OK"
STATUS_EARLY_KILL = "EARLY_KILL"
STATUS_IN_FLIGHT = "IN_FLIGHT"
STATUS_AT_RESUME_LIMIT = "AT_RESUME_LIMIT"
STATUS_REJECTED = "REJECTED"
STATUS_UNKNOWN = "UNKNOWN"

NEEDS_RESUME = {
    STATUS_DONE_WITH_ERRORS,
    STATUS_INCOMPLETE,
    STATUS_PARTIAL,
    STATUS_EARLY_KILL,
}

# Marker file written inside a run dir to permanently exclude it from resume.
# By default the inspector skips dirs with this marker entirely (they don't
# appear in the table or CSV). Pass --show-rejected to see them.
REJECT_MARKER = ".no-resume"


def reject_marker_path(job_dir: Path) -> Path:
    return job_dir / REJECT_MARKER


def is_rejected(job_dir: Path) -> bool:
    return reject_marker_path(job_dir).exists()


def mark_rejected(job_dir: Path, reason: str = "") -> bool:
    """Touch the .no-resume marker. Returns True on first-time mark, False if
    already marked. Writes reason + UTC timestamp into the file for forensics."""
    marker = reject_marker_path(job_dir)
    if marker.exists():
        return False
    from datetime import datetime, timezone
    body = f"# Rejected at {datetime.now(timezone.utc).isoformat()} UTC\n"
    if reason:
        body += f"# Reason: {reason}\n"
    marker.write_text(body)
    return True


def classify(
    info: Dict,
    job_dir: Path,
    active_pairs: Set[Tuple[str, str]],
    active_run_tags: Dict[str, str],
    infra_error_threshold: int,
    max_resume_count: int,
    max_total_fires: int,
    n_fires: int,
) -> str:
    """Classify a parsed job dir. Mirrors scan_jobs_dir_for_resume's logic
    but adds IN_FLIGHT, PARTIAL_OK, AT_RESUME_LIMIT for full-spectrum reporting.

    AT_RESUME_LIMIT triggers when EITHER:
      - resume_count from meta.env >= max_resume_count (matches listener), OR
      - n_fires (slurm-log count) >= max_total_fires (24h hard cap).
    The latter is robust against SIGTERM-killed jobs that never wrote meta.env.
    """
    # Active by run-tag → IN_FLIGHT (covers the case where meta.env hasn't been
    # written yet; the listener's main loop also does this via active_pairs).
    if info["run_tag"] in active_run_tags:
        return STATUS_IN_FLIGHT
    # Active by (model, dataset) pair from squeue.
    if info["hf_model"] and info["dataset"]:
        if (info["hf_model"], info["dataset"]) in active_pairs:
            return STATUS_IN_FLIGHT

    if info["resume_count"] >= max_resume_count:
        return STATUS_AT_RESUME_LIMIT
    if n_fires >= max_total_fires:
        return STATUS_AT_RESUME_LIMIT

    n_completed = info["n_completed"]
    n_total = info["n_total"]
    finished_at = info["finished_at"]
    infra_errors = info["infra_errors"]

    if n_total == 0 and not (job_dir / "result.json").exists():
        return STATUS_EARLY_KILL

    if n_completed < n_total and finished_at is None:
        return STATUS_INCOMPLETE

    if n_completed < n_total and finished_at is not None:
        if infra_errors > infra_error_threshold:
            return STATUS_PARTIAL
        return STATUS_PARTIAL_OK

    if n_total > 0 and n_completed == n_total:
        if infra_errors > infra_error_threshold:
            return STATUS_DONE_WITH_ERRORS
        return STATUS_DONE

    return STATUS_UNKNOWN


# ---- Slurm-log fire count ---------------------------------------------------

def build_run_tag_fire_index(log_dir: Path) -> Dict[str, List[str]]:
    """Walk eval/logs/data_*.out, return run_tag -> [JID, ...]. Count = len(list).

    A run_tag appears once per fire. Original fire + each resume each produce
    one slurm log. Hits the first ~200 lines of each log to find the
    `Run tag: <run_tag>` line written near sbatch start.
    """
    index: Dict[str, List[str]] = {}
    if not log_dir.is_dir():
        return index
    for f in log_dir.glob("data_*.out"):
        # JID is the suffix after "data_" and before ".out"
        try:
            jid = f.stem.split("_", 1)[1]
        except IndexError:
            jid = f.stem
        try:
            with open(f, "r", errors="replace") as fh:
                for i, line in enumerate(fh):
                    if i > 200:
                        break
                    if line.startswith("Run tag: "):
                        rt = line[len("Run tag: "):].strip()
                        if rt:
                            index.setdefault(rt, []).append(jid)
                        break
        except (OSError, IOError):
            continue
    return index


def fires_for_run_tag(index: Dict[str, List[str]], run_tag: str, resume_count: int) -> int:
    """Effective fire count for a run_tag.

    Combines two sources, taking the max so we err on the conservative side
    (more fires counted = more likely to hit the hard cap, fail safe):
      * log_count: number of `data_*.out` slurm logs whose `Run tag: <X>` line
        matches. Truth source IF logs aren't pruned.
      * rc_count: meta.env's RESUME_COUNT + 1. RESUME_COUNT is the number of
        prior fires that reached sbatch line ~1108; the +1 reflects whichever
        fire most recently wrote the value. Undercounts if SIGTERM kills the
        sbatch before line ~1108 (the typical 12h-cap scenario).
      * floor of 1: a parsed run dir means at least one fire happened, even
        if both log + meta.env signals are missing (logs pruned, meta.env not
        written). Prevents a 0-count classification of an obviously-fired dir.
    """
    log_count = len(index.get(run_tag, []))
    rc_count = (resume_count or 0) + 1
    return max(log_count, rc_count, 1)


# ---- Preset / dataset prefix helpers ----------------------------------------

def preset_dataset_prefixes(preset_filter: Optional[str]) -> List[Tuple[str, str]]:
    """Return [(preset_name, dir_prefix), ...]. dir_prefix matches scan_jobs_dir_for_resume's
    normalization: dataset short name with hyphens/dots → underscores, plus trailing '_'."""
    out: List[Tuple[str, str]] = []
    for name, cfg in PRESETS.items():
        if preset_filter and name != preset_filter:
            continue
        for ds in cfg.get("datasets", []):
            ds_short = ds.split("/")[-1] if "/" in ds else ds
            ds_safe = ds_short.replace("-", "_").replace(".", "_")
            out.append((name, f"{ds_safe}_"))
    return out


def detect_preset(run_tag: str, prefix_map: List[Tuple[str, str]]) -> Optional[str]:
    for name, prefix in prefix_map:
        if run_tag.startswith(prefix):
            return name
    return None


# ---- Main scan loop ---------------------------------------------------------

def scan(
    jobs_dirs: List[Path],
    preset_filter: Optional[str],
    infra_error_threshold: int,
    max_resume_count: int,
    max_total_fires: int,
    log_dir: str,
    show_rejected: bool = False,
) -> List[Dict]:
    """Walk each --jobs-dir, classify every dir matching a preset prefix.

    Rejected dirs (marked with REJECT_MARKER) are excluded by default; pass
    show_rejected=True to include them with status=REJECTED.

    Returns a list of result dicts (sorted: needs-resume first, then by run_tag).
    """
    prefix_map = preset_dataset_prefixes(preset_filter)
    if not prefix_map:
        sys.stderr.write(f"ERROR: no presets matched '{preset_filter}'\n")
        sys.exit(2)

    # Pull active SLURM state once
    active_models, active_pairs, active_run_tags = get_active_model_dataset_pairs(log_dir=log_dir)

    # Build run_tag → [JIDs] index from slurm logs once (24h-cap accounting)
    fire_index = build_run_tag_fire_index(Path(log_dir))

    rows: List[Dict] = []
    n_skipped_rejected = 0
    for jd in jobs_dirs:
        if not jd.is_dir():
            sys.stderr.write(f"WARN: skipping missing jobs-dir {jd}\n")
            continue
        for entry in sorted(jd.iterdir()):
            if not entry.is_dir():
                continue
            preset = detect_preset(entry.name, prefix_map)
            if preset is None:
                continue
            info = _parse_job_dir(entry)
            if info is None:
                continue
            rejected = is_rejected(entry)
            if rejected and not show_rejected:
                n_skipped_rejected += 1
                continue
            n_fires = fires_for_run_tag(fire_index, info["run_tag"], info["resume_count"])
            fire_jids = ",".join(fire_index.get(info["run_tag"], []))
            if rejected:
                status = STATUS_REJECTED
            else:
                status = classify(
                    info, entry, active_pairs, active_run_tags,
                    infra_error_threshold, max_resume_count,
                    max_total_fires, n_fires,
                )
            slurm_jid = active_run_tags.get(info["run_tag"]) or (info["slurm_job_id"] or "")
            rows.append({
                "preset": preset,
                "status": status,
                "slurm_jid": slurm_jid,
                "run_tag": info["run_tag"],
                "hf_model": info["hf_model"] or "",
                "dataset": info["dataset"] or "",
                "n_completed": info["n_completed"],
                "n_total": info["n_total"],
                "infra_errors": info["infra_errors"],
                "total_errors": info["total_errors"],
                "finished_at": info["finished_at"] or "",
                "resume_count": info["resume_count"],
                "n_fires": n_fires,
                "fire_jids": fire_jids,
                "db_job_id": info["db_job_id"] or "",
                "jobs_dir": str(jd),
            })

    if n_skipped_rejected:
        sys.stderr.write(f"(hidden: {n_skipped_rejected} rejected dir(s); pass --show-rejected to display)\n")
    rows.sort(key=lambda r: (r["status"] not in NEEDS_RESUME, r["preset"], r["run_tag"]))
    return rows


# ---- Output formatters -------------------------------------------------------

def print_csv(rows: List[Dict]) -> None:
    fieldnames = [
        "preset", "status", "slurm_jid", "run_tag", "hf_model", "dataset",
        "n_completed", "n_total", "infra_errors", "total_errors",
        "finished_at", "resume_count", "n_fires", "fire_jids",
        "db_job_id", "jobs_dir",
    ]
    w = csv.DictWriter(sys.stdout, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)


def print_table(rows: List[Dict]) -> None:
    if not rows:
        print("(no rows)")
        return

    # Truncate model name for terminal width
    def short_model(m: str) -> str:
        if len(m) <= 60:
            return m
        return m[:28] + "..." + m[-29:]

    def short_run_tag(t: str) -> str:
        if len(t) <= 56:
            return t
        return t[:24] + "..." + t[-29:]

    header = f"{'STATUS':<17} {'PRESET':<9} {'SLURM':<7} {'PROGRESS':<10} {'INFRA':>5} {'TOT_ERR':>7} {'RC':>2} {'FIRES':>5} {'MODEL':<60}  {'RUN_TAG'}"
    print(header)
    print("-" * len(header))
    for r in rows:
        prog = f"{r['n_completed']}/{r['n_total']}"
        print(
            f"{r['status']:<17} "
            f"{r['preset']:<9} "
            f"{(r['slurm_jid'] or '-'):<7} "
            f"{prog:<10} "
            f"{r['infra_errors']:>5} "
            f"{r['total_errors']:>7} "
            f"{r['resume_count']:>2} "
            f"{r['n_fires']:>5} "
            f"{short_model(r['hf_model']):<60}  "
            f"{short_run_tag(r['run_tag'])}"
        )

    # Summary by status
    print()
    print("Summary:")
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for s in sorted(counts.keys()):
        marker = " (RESUME)" if s in NEEDS_RESUME else ""
        print(f"  {s:<17} {counts[s]:>4}{marker}")


# ---- CLI --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Inspect harbor eval job dirs and report resume status.")
    p.add_argument(
        "--jobs-dir", action="append", required=True,
        help="Path to harbor jobs dir (repeatable).",
    )
    p.add_argument(
        "--preset", default=None,
        help=f"Filter by preset. Choices: {', '.join(PRESETS.keys())}. Default: all.",
    )
    p.add_argument(
        "--needs-resume-only", action="store_true",
        help="Drop rows that don't need resume (DONE / IN_FLIGHT / PARTIAL_OK / AT_RESUME_LIMIT).",
    )
    p.add_argument(
        "--infra-error-threshold", type=int, default=DEFAULT_INFRA_ERROR_THRESHOLD,
        help="Min infra errors to trigger PARTIAL/DONE_WITH_ERRORS resume. Default: 10 (matches listener).",
    )
    p.add_argument(
        "--max-resume-count", type=int, default=DEFAULT_MAX_RESUME_COUNT,
        help="Treat resume_count (from meta.env) >= this as AT_RESUME_LIMIT. Default: 5.",
    )
    p.add_argument(
        "--max-total-fires", type=int, default=DEFAULT_MAX_TOTAL_FIRES,
        help=("Treat n_fires (slurm-log count, robust to SIGTERM) >= this as AT_RESUME_LIMIT. "
              "Default: 2 (1 original + 1 resume = 24h hard cap on a 12h-walltime cluster)."),
    )
    p.add_argument(
        "--log-dir", default="eval/logs",
        help="Slurm log dir (parsed for active model+dataset+run_tag). Default: eval/logs.",
    )
    p.add_argument(
        "--csv", action="store_true",
        help="Emit CSV instead of human-readable table.",
    )
    p.add_argument(
        "--reject", action="append", default=[], metavar="RUN_TAG",
        help=("Mark a run dir as rejected (writes .no-resume marker file). "
              "Repeatable. Rejected dirs are hidden from inspector output by default. "
              "Specify either bare run_tag (search every --jobs-dir) or "
              "absolute path to the run dir."),
    )
    p.add_argument(
        "--reject-needs-resume", action="store_true",
        help=("Reject ALL run dirs currently classified as needs-resume "
              "(DONE_WITH_ERRORS / INCOMPLETE / PARTIAL / EARLY_KILL). "
              "Useful for nuking legacy candidates in bulk."),
    )
    p.add_argument(
        "--reject-reason", default="",
        help="Free-form reason recorded in the marker file (used with --reject / --reject-needs-resume).",
    )
    p.add_argument(
        "--show-rejected", action="store_true",
        help="Include rejected dirs in the output (status=REJECTED). "
             "By default rejected dirs are hidden entirely.",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Skip confirmation prompt for bulk --reject-needs-resume.",
    )
    args = p.parse_args()

    jobs_dirs = [Path(x).resolve() for x in args.jobs_dir]

    # --- Reject handling --------------------------------------------------
    # Two modes:
    #   1) --reject <run_tag> [--reject ...]  → reject those run_tags directly,
    #      skip the scan, exit. Path-form (absolute) is also accepted.
    #   2) --reject-needs-resume              → scan first, reject every row
    #      currently classified as needing resume.

    if args.reject and not args.reject_needs_resume:
        n_marked = 0
        n_already = 0
        n_missing = 0
        for ident in args.reject:
            target: Optional[Path] = None
            ip = Path(ident)
            if ip.is_absolute() and ip.is_dir():
                target = ip
            else:
                # Search each --jobs-dir for a matching run dir
                for jd in jobs_dirs:
                    cand = jd / ident
                    if cand.is_dir():
                        target = cand
                        break
            if target is None:
                sys.stderr.write(f"WARN: run_tag not found in any --jobs-dir: {ident}\n")
                n_missing += 1
                continue
            if mark_rejected(target, reason=args.reject_reason):
                print(f"REJECTED  {target}")
                n_marked += 1
            else:
                print(f"already-rejected  {target}")
                n_already += 1
        print(f"\nDone: {n_marked} newly rejected, {n_already} already rejected, {n_missing} missing.",
              file=sys.stderr)
        return 0 if n_missing == 0 else 1

    rows = scan(
        jobs_dirs=jobs_dirs,
        preset_filter=args.preset,
        infra_error_threshold=args.infra_error_threshold,
        max_resume_count=args.max_resume_count,
        max_total_fires=args.max_total_fires,
        log_dir=args.log_dir,
        show_rejected=args.show_rejected,
    )

    if args.reject_needs_resume:
        targets = [r for r in rows if r["status"] in NEEDS_RESUME]
        if not targets:
            print("No needs-resume rows to reject.", file=sys.stderr)
            return 0
        print(f"About to reject {len(targets)} run dir(s):", file=sys.stderr)
        for r in targets:
            print(f"  {r['status']:18s} {r['preset']:8s} {r['run_tag']}", file=sys.stderr)
        if not args.yes and not _confirm_destructive():
            print("Aborted.", file=sys.stderr)
            return 1
        n_marked = 0
        n_already = 0
        for r in targets:
            target = Path(r["jobs_dir"]) / r["run_tag"]
            if mark_rejected(target, reason=args.reject_reason or "bulk --reject-needs-resume"):
                print(f"REJECTED  {target}")
                n_marked += 1
            else:
                n_already += 1
        print(f"\nDone: {n_marked} newly rejected, {n_already} already rejected.", file=sys.stderr)
        return 0

    if args.needs_resume_only:
        rows = [r for r in rows if r["status"] in NEEDS_RESUME]

    if args.csv:
        print_csv(rows)
    else:
        print_table(rows)
    return 0


def _confirm_destructive() -> bool:
    """Prompt user for confirmation when --reject-needs-resume affects multiple dirs."""
    try:
        ans = input("Proceed? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in ("y", "yes")


if __name__ == "__main__":
    sys.exit(main())
