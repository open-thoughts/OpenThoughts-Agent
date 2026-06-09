#!/usr/bin/env python3
"""Fire chunked resume listeners against a list of (model, preset) pairs.

Reads a CSV produced by `eval/check_resume_needed.py --csv` (or accepts a
priority file directly), chunks the candidate models, and fires one
unified_eval_listener.py invocation per chunk with --resume-only +
--priority-file + --max-jobs-submitted = chunk size. Sleeps between chunks
to keep the sbatch rate under the 6/min cap and ramp Daytona load smoothly.

Per-chunk listener invocation matches the original-fire flag set so
config.json conflict (Harbor FileExistsError) is avoided. **You must pass
the same sizing / yaml / conda-env you used on the original fire.**

Usage:
    # Dry-run — show planned invocations without firing
    python eval/resume_chunked.py \
        --csv /tmp/cands.csv --preset tb2 --org eval \
        --tp-size 2 --dp-size 2 --timeout-multiplier 16.0 \
        --jobs-dir /e/data1/datasets/playground/mmlaion/shared/zhuang1_eval_jobs \
        --tag-prefix etashguha_resume_20260507 --dry-run

    # Fire 4 at a time, 120s sleep between chunks
    python eval/resume_chunked.py \
        --csv /tmp/cands.csv --preset tb2 --org eval \
        --tp-size 2 --dp-size 2 --timeout-multiplier 16.0 \
        --jobs-dir /e/data1/datasets/playground/mmlaion/shared/zhuang1_eval_jobs \
        --tag-prefix etashguha_resume_20260507 \
        --chunk-size 4 --sleep-between 120

The orchestrator runs each listener invocation foreground; you'll see the
"Submitted batch job N" lines per chunk in the listener log path it prints.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent

# These come straight from the inspector — restate here only so we don't
# import the listener (and its log spam) just for two strings.
NEEDS_RESUME = {"DONE_WITH_ERRORS", "INCOMPLETE", "PARTIAL", "EARLY_KILL"}
STATUS_AT_LIMIT = "AT_RESUME_LIMIT"

# Org → dotenv map. Add new orgs here as needed.
ORG_DOTENVS = {
    "eval": "hpc/dotenv/jupiter_eval.env",
    "data": "hpc/dotenv/jupiter_eval_data_org.env",
}


def parse_csv(csv_path: Path, preset: Optional[str]) -> Tuple[List[dict], List[dict]]:
    """Return (resume_rows, at_limit_rows) from inspector CSV, filtered by preset.

    resume_rows: status in NEEDS_RESUME — these will be fired.
    at_limit_rows: status == AT_RESUME_LIMIT — these will be WARNED about and skipped
                   (24h hard cap or other resume-count limit). The orchestrator
                   surfaces these so the user is aware of dropouts.
    """
    resume_rows: List[dict] = []
    at_limit_rows: List[dict] = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if preset and row.get("preset") != preset:
                continue
            status = row.get("status", "")
            if status in NEEDS_RESUME:
                resume_rows.append(row)
            elif status == STATUS_AT_LIMIT:
                at_limit_rows.append(row)
    return resume_rows, at_limit_rows


def parse_priority_file(path: Path) -> List[str]:
    return [
        ln.strip() for ln in path.read_text().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def chunkify(items: List[str], n: int) -> List[List[str]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def build_listener_argv(
    py: str,
    preset: str,
    priority_file: Path,
    chunk_size: int,
    args: argparse.Namespace,
    log_path: Path,
) -> List[str]:
    cmd = [
        py, "eval/unified_eval_listener.py",
        "--cluster-config", args.cluster_config,
        "--preset", preset,
        "--priority-file", str(priority_file),
        "--baseline-model-config", args.baseline_model_config,
        "--conda-env", args.conda_env,
        "--tp-size", str(args.tp_size),
        "--dp-size", str(args.dp_size),
        "--timeout-multiplier", str(args.timeout_multiplier),
        "--slurm-partition", args.slurm_partition,
        "--slurm-time", args.slurm_time,
        "--max-jobs-submitted", str(chunk_size),
        "--n-concurrent", str(args.n_concurrent),
        "--jobs-dir", args.jobs_dir,
        "--max-resume-count", str(args.max_resume_count),
        "--resume-only",
        "--auto-snapshot",
        "--pre-download",
        "--stagger-delay", "2",
        "--chain-batch-size", str(chunk_size),
        "--once",
        "--force-reeval",
    ]
    # Pass --resume-error-threshold only when explicitly set, so we don't shadow
    # the listener's own default. Setting this to -1 promotes DONE dirs (with
    # infra_errors=0) into DONE_WITH_ERRORS classification, which is the only
    # way to resume a dir that has n_completed == n_total but is genuinely
    # stuck-at-full (finished_at=None). See RESUME_RUNBOOK.md "Resuming
    # stuck-at-full DONE dirs" for rationale.
    if args.resume_error_threshold is not None:
        cmd += ["--resume-error-threshold", str(args.resume_error_threshold)]
    if args.enable_thinking:
        cmd.append("--enable-thinking")
    # Cat 3 (preferred-harness reproduction): pass through Pinggy URL/token so the
    # sbatch opens a tunnel and the unified_eval_harbor.sbatch resume sed-patch
    # can rewrite the dir's config.json from the (dead) old Pinggy URL to this
    # new one. Each fire needs its own free pair. resume_chunked.py is single-
    # pair per invocation; for multi-model batches, fire one chunk per pair.
    if args.pinggy_url:
        cmd += ["--pinggy-url", args.pinggy_url]
    if args.pinggy_token:
        cmd += ["--pinggy-token", args.pinggy_token]
    if args.config_yaml:
        cmd += ["--config-yaml", args.config_yaml]
    if args.agent_parser is not None:
        cmd += ["--agent-parser", args.agent_parser]
    return cmd


def run_chunk(
    chunk_idx: int,
    n_chunks: int,
    chunk: List[str],
    preset: str,
    args: argparse.Namespace,
    log_dir: Path,
    tmp_dir: Path,
    py: str,
    dotenv_abs: Path,
) -> int:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    pf = tmp_dir / f"{args.tag_prefix}_{preset}_chunk{chunk_idx:02d}_{ts}.txt"
    pf.write_text("\n".join(chunk) + "\n")

    log_path = log_dir / f"resume_{args.tag_prefix}_{preset}_chunk{chunk_idx:02d}_{ts}.log"

    listener_cmd = build_listener_argv(py, preset, pf, len(chunk), args, log_path)

    # We invoke listener inside a bash shell so we can `. dotenv` first; this matches
    # the canonical Jupiter fire pattern and ensures Daytona / SLURM env vars are set.
    cmd_str = (
        f"cd {shlex.quote(str(_REPO_ROOT))} && "
        f". {shlex.quote(str(dotenv_abs))} >/dev/null 2>&1 && "
        + " ".join(shlex.quote(c) for c in listener_cmd)
    )
    print(
        f"\n[chunk {chunk_idx}/{n_chunks}] preset={preset} size={len(chunk)} "
        f"priority_file={pf} log={log_path}"
    )
    for m in chunk:
        print(f"  - {m}")
    if args.dry_run:
        print(f"  [DRY-RUN] would run: {cmd_str}")
        print(f"  [DRY-RUN] would log to: {log_path}")
        return 0

    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            ["bash", "-c", cmd_str], stdout=logf, stderr=subprocess.STDOUT,
        )
    rc = proc.wait()
    print(f"  → listener exited rc={rc}")
    if rc == 0:
        # Print the JIDs from the log for visibility
        try:
            txt = log_path.read_text()
            for line in txt.splitlines():
                if "Submitted batch job" in line:
                    print(f"  {line.strip()}")
        except Exception:
            pass
    return rc


def main() -> int:
    p = argparse.ArgumentParser(description="Chunked resume orchestrator (PR #27 listener).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Inspector CSV path (output of check_resume_needed.py --csv).")
    src.add_argument("--priority-file", help="Pre-built priority file with model names (one per line).")

    p.add_argument("--preset", required=True, help="Preset to fire (one per orchestrator run, e.g. tb2 / swebench).")
    p.add_argument("--org", required=True, choices=sorted(ORG_DOTENVS.keys()),
                   help="Daytona org. Routes to the matching dotenv.")
    p.add_argument("--jobs-dir", required=True, help="Path to harbor jobs dir for --jobs-dir on the listener.")
    p.add_argument("--tag-prefix", required=True,
                   help="Used in priority-file + log paths for traceability (e.g. etashguha_resume_20260507).")

    p.add_argument("--chunk-size", type=int, default=4,
                   help="Models per listener invocation. Also passed as --max-jobs-submitted. Default: 4.")
    p.add_argument("--sleep-between", type=int, default=120,
                   help="Seconds to sleep between chunks (Daytona ramp + sbatch rate). Default: 120.")

    # Listener flag pass-through (must match the original fire to avoid config.json conflict)
    p.add_argument("--cluster-config", default="eval/clusters/jupiter.yaml")
    p.add_argument("--baseline-model-config", default="eval/configs/baseline_model_configs_minimal.yaml")
    p.add_argument("--conda-env", default="otagent-fix")
    p.add_argument("--tp-size", type=int, required=True)
    p.add_argument("--dp-size", type=int, default=2)
    p.add_argument("--timeout-multiplier", type=float, required=True)
    p.add_argument("--n-concurrent", type=int, default=32)
    p.add_argument("--enable-thinking", action="store_true", default=True)
    p.add_argument("--slurm-partition", default="booster")
    p.add_argument("--slurm-time", default="11:59:00")
    p.add_argument("--max-resume-count", type=int, default=1,
                   help=("Listener-side --max-resume-count (defense-in-depth backstop alongside the "
                         "inspector's slurm-log-based --max-total-fires filter). Default: 1, "
                         "meaning the listener will also refuse to resume a dir whose meta.env "
                         "shows >= 1 prior resume completed. Bump if you raised --max-total-fires."))
    p.add_argument("--pinggy-url", default=None,
                   help=("Cat 3 only. Pinggy URL (e.g. dadccqeqqf.a.pinggy.link) for the "
                         "tunnel back to the served model. Pair with --pinggy-token. The "
                         "sbatch will sed-rewrite the existing dir's config.json's stale "
                         "Pinggy URL to this one. Single pair per orchestrator invocation; "
                         "for multi-model resumes, fire one invocation per (model, pair)."))
    p.add_argument("--pinggy-token", default=None,
                   help="Cat 3 only. Pinggy auth token, paired with --pinggy-url.")
    p.add_argument("--config-yaml", default=None,
                   help=("Cat 3 only. Harbor config YAML (e.g. dcagent_eval_config_swe_agent.yaml). "
                         "Listener passes this to the sbatch for scaffold/parser selection."))
    p.add_argument("--agent-parser", default=None,
                   help=("Cat 3 only. Agent parser override (typically empty string '' for "
                         "swe-agent which doesn't use a parser). Empty string = disable parser."))
    p.add_argument("--resume-error-threshold", type=int, default=None,
                   help=("Listener-side --resume-error-threshold passthrough. When omitted, the "
                         "listener uses its own default (10). Pass -1 to enable resuming "
                         "stuck-at-full DONE dirs (n_completed==n_total, finished_at=None, "
                         "infra_errors below normal threshold) — the listener's resume scanner "
                         "skips DONE dirs unless infra_errors > threshold, so threshold=-1 makes "
                         "any infra_errors >= 0 promote the dir to DONE_WITH_ERRORS classification "
                         "and become resume-eligible. Combined with the per-trial G12 sed patch, "
                         "harbor will only run the actual missing trials and then finalize+upload."))

    p.add_argument("--python", default="/e/scratch/jureap59/zhuang1/conda/envs/otagent-fix/bin/python",
                   help="Python interpreter to invoke listener with.")
    p.add_argument("--log-dir", default="experiments/listener_logs",
                   help="Where to write listener logs.")
    p.add_argument("--tmp-dir", default="/tmp",
                   help="Where to write per-chunk priority files.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned invocations but don't fire.")
    args = p.parse_args()

    # Resolve dotenv
    dotenv_rel = ORG_DOTENVS[args.org]
    dotenv_abs = (_REPO_ROOT / dotenv_rel).resolve()
    if not dotenv_abs.is_file():
        sys.stderr.write(f"ERROR: dotenv not found: {dotenv_abs}\n")
        return 2

    # Build candidate model list
    if args.csv:
        resume_rows, at_limit_rows = parse_csv(Path(args.csv), args.preset)

        # Surface AT_RESUME_LIMIT rows as warnings — these are the "already at 24h cap"
        # entries the user wanted called out. We do NOT silently drop them.
        if at_limit_rows:
            sys.stderr.write(
                f"\n*** WARNING: {len(at_limit_rows)} (preset={args.preset}) row(s) at resume/time-fire limit — SKIPPING ***\n"
            )
            for r in at_limit_rows:
                sys.stderr.write(
                    f"  SKIP  n_fires={r.get('n_fires','?')} resume_count={r.get('resume_count','?')}  "
                    f"{r.get('hf_model','')}  run_tag={r.get('run_tag','')}\n"
                )
            sys.stderr.write(
                "  (Bump --max-total-fires on the inspector if intentional; otherwise treat these as terminated.)\n\n"
            )

        models: List[str] = []
        seen = set()
        for r in resume_rows:
            m = r.get("hf_model", "").strip()
            if m and m not in seen:
                seen.add(m)
                models.append(m)
    else:
        models = parse_priority_file(Path(args.priority_file))

    if not models:
        print("(no resume candidates — nothing to fire)")
        return 0

    print(f"Resume orchestrator")
    print(f"  preset             {args.preset}")
    print(f"  org                {args.org}  ({dotenv_rel})")
    print(f"  jobs-dir           {args.jobs_dir}")
    print(f"  candidates         {len(models)} model(s)")
    print(f"  chunk-size         {args.chunk_size}")
    print(f"  sleep-between      {args.sleep_between}s")
    print(f"  sizing             tp={args.tp_size} dp={args.dp_size} tm={args.timeout_multiplier} n_concurrent={args.n_concurrent}")
    print(f"  conda-env          {args.conda_env}")
    print(f"  dry-run            {args.dry_run}")

    chunks = chunkify(models, args.chunk_size)
    print(f"  → {len(chunks)} chunk(s)")

    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.tmp_dir).resolve()

    fail = 0
    for i, chunk in enumerate(chunks, start=1):
        rc = run_chunk(
            chunk_idx=i, n_chunks=len(chunks),
            chunk=chunk, preset=args.preset, args=args,
            log_dir=log_dir, tmp_dir=tmp_dir,
            py=args.python, dotenv_abs=dotenv_abs,
        )
        if rc != 0:
            fail += 1
            print(f"  WARNING: chunk {i} listener returned non-zero ({rc}); continuing")
        if i < len(chunks):
            if args.dry_run:
                print(f"  [DRY-RUN] would sleep {args.sleep_between}s before next chunk")
            else:
                print(f"  sleeping {args.sleep_between}s before next chunk...")
                time.sleep(args.sleep_between)

    print(f"\nDone. Chunks fired: {len(chunks)}, failed: {fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
