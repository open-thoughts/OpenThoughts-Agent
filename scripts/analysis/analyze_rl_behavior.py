#!/usr/bin/env python3
"""Orchestrate the full RL behavioral-analysis pipeline.

Runs the existing analysis scripts plus the new behavioral_delta /
trace_pair_render / eval_temporal_overlay tools in the right order, with
each step writing into ``--output-dir/<step>/``. Each step is skipped if
its output already exists (use ``--force`` to re-run, ``--skip step1,step2``
to opt out individually).

The pipeline maps to the four research questions:

    Q1 (what changed):     behavioral_delta  +  summarize_conversations
                           Optionally update_hf_failure_modes upstream
                           for both --baseline and --post-rl-eval if
                           --annotate-failure-modes is set.

    Q2 (attribution):      temporal_trace_analysis  +  parse_skyrl_metrics
                           Side-by-side temporal plots of RL reward and
                           skyrl training metrics (KL, grad norm, ...).

    Q3 (persistence):      eval_temporal_overlay  +  trace_pair_render
                           +  post_training_comparison
                           Eval markers overlaid on RL time axis; same-
                           task pairs rendered side-by-side; stage-level
                           summary table.

    Q4 (eval impact):      solve_rate_by_context  +  the Q1+Q3 outputs
                           Did behavior shifts move trace distribution
                           into bins that have different solve rates? If
                           not, eval-stable behavior change is expected.

Example:

    python -m scripts.analysis.analyze_rl_behavior \\
        --rl-traces        penfever/rl-train-traces-foo \\
        --baseline-eval    penfever/eval-pre-rl-foo \\
        --post-rl-eval     penfever/eval-post-rl-foo \\
        --post-rl-eval-ts  2026-05-28T18:30 \\
        --training-log-dir /scratch/skyrl-logs/foo/ \\
        --output-dir       /Users/me/Documents/notes/rl-behavior-foo/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class Step:
    """One pipeline step. ``runner`` is invoked when the step's output is missing."""

    name: str
    description: str
    question: str  # "Q1" .. "Q4"
    output_marker: Path  # if this exists and is non-empty, step is skipped
    runner: Callable[[], int]
    optional: bool = False  # if True, missing inputs cause skip-with-warning, not fail


def _run_subprocess(cmd: List[str]) -> int:
    print(f"\n[orchestrator] $ {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def _module_runner(module: str, extra: List[str]) -> Callable[[], int]:
    """Build a runner that invokes ``python -m <module> <extra>``."""
    def _run() -> int:
        return _run_subprocess([sys.executable, "-m", module] + extra)
    return _run


def _build_steps(args: argparse.Namespace) -> List[Step]:
    out = Path(args.output_dir).expanduser().resolve()
    steps: List[Step] = []

    # Q1: behavioral delta (baseline vs post-RL eval). Requires both repos
    # to have been failure-mode-annotated already (or --annotate-failure-modes).
    if args.baseline_eval and args.post_rl_eval:
        delta_md = out / "Q1_behavioral_delta" / "report.md"
        steps.append(
            Step(
                name="Q1.behavioral_delta",
                description="Macro-metric + failure-mode diff baseline-eval vs post-RL eval",
                question="Q1",
                output_marker=delta_md,
                runner=_module_runner(
                    "scripts.analysis.behavioral_delta",
                    [
                        "--before", args.baseline_eval,
                        "--after", args.post_rl_eval,
                        "--output", str(delta_md),
                        *(["--max-rows", str(args.max_rows)] if args.max_rows else []),
                    ],
                ),
            )
        )

    # Q1: LLM judge diff (GPT-5 same-task before/after pair classification).
    # Gated on --llm-judge because it costs API money. Runs AFTER behavioral_delta
    # in execution order (the orchestrator runs steps in append order); uses the
    # same baseline/post-RL pair the resolver chose.
    #
    # TWO complementary probes (each `--llm-judge-max-pairs` tasks):
    #   1. MOST-CHANGED (behavior-delta) → Q1_llm_judge_most_changed/ — the
    #      tasks whose behavior shifted most. Biased toward where the policy
    #      changed; answers "when behavior changed, was it for the better?"
    #   2. RANDOM (uniform, without replacement, DISJOINT from #1 via its
    #      selected_tasks.json) → Q1_llm_judge_random/ — answers "did behavior
    #      change in general?" The disjoint random draw is the unbiased control
    #      for the most-changed sample's selection bias.
    if args.baseline_eval and args.post_rl_eval and args.llm_judge:
        mc_md = out / "Q1_llm_judge_most_changed" / "report.md"
        steps.append(
            Step(
                name="Q1.llm_judge_most_changed",
                description=(
                    f"LLM judge ({args.llm_judge_model}) — {args.llm_judge_max_pairs} "
                    f"MOST-behaviorally-changed pairs (selection-biased)"
                ),
                question="Q1",
                output_marker=mc_md,
                runner=_module_runner(
                    "scripts.analysis.llm_judge_diff",
                    [
                        "--before", args.baseline_eval,
                        "--after", args.post_rl_eval,
                        "--output", str(mc_md),
                        "--selection", "behavior-delta",
                        "--max-pairs", str(args.llm_judge_max_pairs),
                        "--model", args.llm_judge_model,
                        "--max-concurrent", str(args.llm_judge_concurrent),
                        *(["--max-rows", str(args.max_rows)] if args.max_rows else []),
                    ],
                ),
                optional=True,
            )
        )
        # Random control: disjoint from the most-changed set (excludes its
        # selected_tasks.json). Runs after #1 in append order, so that file
        # exists; if #1 was skipped/never-run the exclude file is simply absent
        # (no exclusion) and this is a plain uniform sample.
        rand_md = out / "Q1_llm_judge_random" / "report.md"
        steps.append(
            Step(
                name="Q1.llm_judge_random",
                description=(
                    f"LLM judge ({args.llm_judge_model}) — {args.llm_judge_max_pairs} "
                    f"RANDOM pairs (uniform, disjoint from most-changed; 'changed in general?')"
                ),
                question="Q1",
                output_marker=rand_md,
                runner=_module_runner(
                    "scripts.analysis.llm_judge_diff",
                    [
                        "--before", args.baseline_eval,
                        "--after", args.post_rl_eval,
                        "--output", str(rand_md),
                        "--selection", "random",
                        "--seed", "0",
                        "--exclude-tasks-file",
                        str(out / "Q1_llm_judge_most_changed" / "selected_tasks.json"),
                        "--max-pairs", str(args.llm_judge_max_pairs),
                        "--model", args.llm_judge_model,
                        "--max-concurrent", str(args.llm_judge_concurrent),
                        *(["--max-rows", str(args.max_rows)] if args.max_rows else []),
                    ],
                ),
                optional=True,
            )
        )

    # NOTE: Q1.rl_summary (formerly summarize_conversations on RL-time
    # traces) is intentionally NOT a step here. summarize_conversations.py
    # only accepts a positional JSONL path (no HF id, no --output flag),
    # and its outputs — per-row token/turn/reward stats — are already
    # covered by Q2.temporal_trace_analysis (binned + overall) and
    # Q1.behavioral_delta (per-dataset macros). Re-add if a JSONL-only
    # workflow comes back.

    # Q2: temporal_trace_analysis on the RL-time traces.
    if args.rl_traces:
        temp_dir = out / "Q2_temporal"
        temp_marker = temp_dir / "temporal_summary.json"
        temp_plot = temp_dir / "temporal_summary.png"
        temp_dir.mkdir(parents=True, exist_ok=True)
        steps.append(
            Step(
                name="Q2.temporal_trace_analysis",
                description="Time-binned reward + behavioral stats over RL training",
                question="Q2",
                output_marker=temp_marker,
                runner=_module_runner(
                    "scripts.analysis.temporal_trace_analysis",
                    [
                        args.rl_traces,
                        "--bin-hours", str(args.rl_bin_hours),
                        "--output", str(temp_marker),
                        "--plot", str(temp_plot),
                    ],
                ),
            )
        )

    # Q2: parse_skyrl_metrics on training logs. The script writes
    # ``<ts>_metrics_report.md`` (timestamped) inside its output dir
    # and has no --output-filename override. Wrap the runner to drop a
    # ``summary.md`` symlink at the most recent report, so our marker-
    # based skip-detection and INDEX.md cross-link both work.
    if args.training_log_dir:
        skyrl_dir = out / "Q2_skyrl_metrics"
        skyrl_marker = skyrl_dir / "summary.md"

        def _run_skyrl(args=args, skyrl_dir=skyrl_dir, marker=skyrl_marker) -> int:
            rc = _run_subprocess([
                sys.executable, "-m", "scripts.analysis.parse_skyrl_metrics",
                args.training_log_dir, str(skyrl_dir),
            ])
            if rc != 0:
                return rc
            reports = sorted(skyrl_dir.glob("*_metrics_report.md"))
            if not reports:
                print(
                    "[orchestrator] parse_skyrl_metrics succeeded (rc=0) but no "
                    "*_metrics_report.md was produced; INDEX link will be missing.",
                    file=sys.stderr,
                )
                return rc
            latest = reports[-1]
            if marker.exists() or marker.is_symlink():
                marker.unlink()
            try:
                marker.symlink_to(latest.name)  # relative symlink stays valid if dir moves
            except OSError:
                # Fall back to copy on filesystems that can't symlink.
                shutil.copy2(latest, marker)
            return rc

        steps.append(
            Step(
                name="Q2.parse_skyrl_metrics",
                description="KL / grad-norm / LR / vLLM stats from SkyRL training logs",
                question="Q2",
                output_marker=skyrl_marker,
                runner=_run_skyrl,
            )
        )

    # Q3: eval_temporal_overlay. Needs the post-RL eval to carry a timestamp.
    if args.rl_traces and args.post_rl_eval:
        overlay_dir = out / "Q3_temporal_overlay"
        overlay_png = overlay_dir / "overlay.png"
        # Build eval-traces specs: baseline (if present) without ts, post-rl with --post-rl-eval-ts.
        eval_specs: List[str] = []
        post_spec = args.post_rl_eval
        if args.post_rl_eval_ts:
            post_spec = f"{post_spec}@post-RL:{args.post_rl_eval_ts}"
        eval_specs.extend(["--eval-traces", post_spec])
        if args.baseline_eval and args.baseline_eval_ts:
            eval_specs.extend(
                ["--eval-traces", f"{args.baseline_eval}@baseline:{args.baseline_eval_ts}"]
            )
        if args.extra_eval:
            for s in args.extra_eval:
                eval_specs.extend(["--eval-traces", s])
        steps.append(
            Step(
                name="Q3.eval_temporal_overlay",
                description="RL reward curve with post-RL eval-checkpoint markers",
                question="Q3",
                output_marker=overlay_png,
                runner=_module_runner(
                    "scripts.analysis.eval_temporal_overlay",
                    [
                        "--rl-traces", args.rl_traces,
                        "--bin-hours", str(args.rl_bin_hours),
                        "--output", str(overlay_png),
                        *eval_specs,
                        *(["--max-rows", str(args.max_rows)] if args.max_rows else []),
                    ],
                ),
                optional=not args.post_rl_eval_ts,
            )
        )

    # Q3: trace_pair_render baseline vs post-RL.
    if args.baseline_eval and args.post_rl_eval:
        pair_html = out / "Q3_trace_pairs" / "pairs.html"
        steps.append(
            Step(
                name="Q3.trace_pair_render",
                description="Same-task side-by-side render (highest-reward trials)",
                question="Q3",
                output_marker=pair_html,
                runner=_module_runner(
                    "scripts.analysis.trace_pair_render",
                    [
                        "--before", args.baseline_eval,
                        "--after", args.post_rl_eval,
                        "--output", str(pair_html),
                        "--top-n", str(args.pair_top_n),
                        *(["--max-rows", str(args.max_rows)] if args.max_rows else []),
                    ],
                ),
            )
        )

    # Q3/Q4: post_training_comparison takes a list of model eval dirs. We
    # provide baseline + post-RL as a minimal pair; downstream user can
    # extend the JSON config to add SFT-only / weak-SFT / etc.
    if args.post_training_config:
        ptc_md = out / "Q3_post_training_comparison" / "comparison.md"
        steps.append(
            Step(
                name="Q3.post_training_comparison",
                description="Per-task scores + failure-mode + context-length across post-training stages",
                question="Q3",
                output_marker=ptc_md,
                runner=_module_runner(
                    "scripts.analysis.post_training_comparison",
                    [
                        "--config", args.post_training_config,
                        "--output", str(ptc_md),
                    ],
                ),
            )
        )

    # Q4: solve_rate_by_context for baseline vs post-RL.
    if args.baseline_eval and args.post_rl_eval:
        sr_dir = out / "Q4_solve_rate_by_context"
        sr_dir.mkdir(parents=True, exist_ok=True)
        sr_png = sr_dir / "solve_rate.png"
        steps.append(
            Step(
                name="Q4.solve_rate_by_context",
                description="Solve / timeout / error rates binned by context length",
                question="Q4",
                output_marker=sr_png,
                runner=_module_runner(
                    "scripts.analysis.solve_rate_by_context",
                    [
                        args.baseline_eval,
                        args.post_rl_eval,
                        "--plot", str(sr_png),
                    ],
                ),
            )
        )

    # Optional upstream: annotate failure modes via GPT-5 on both eval repos.
    # Inserted at the FRONT (must run before Q1 behavioral_delta picks up the labels).
    if args.annotate_failure_modes:
        prefix: List[Step] = []
        for label, repo in [("baseline", args.baseline_eval), ("post-rl", args.post_rl_eval)]:
            if not repo:
                continue
            marker = out / f"Q0_failure_mode_{label}" / "done.txt"
            # update_hf_failure_modes writes back to the HF repo when --push
            # is set; we just record the run-attempt locally.
            def _annotate(repo=repo, marker=marker) -> int:
                rc = _run_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "scripts.analysis.update_hf_failure_modes",
                        repo,
                        "--resume",
                        "--push",
                    ]
                )
                if rc == 0:
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text(f"annotated {repo} at {datetime.utcnow().isoformat()}\n")
                return rc
            prefix.append(
                Step(
                    name=f"Q0.annotate_failure_modes.{label}",
                    description=f"GPT-5 failure-mode annotation on {repo}",
                    question="Q0",
                    output_marker=marker,
                    runner=_annotate,
                    optional=True,
                )
            )
        steps = prefix + steps

    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rl-traces",
        help="RL-time training-trace dataset (HF id, JSONL, or dir). Drives Q2.",
    )
    parser.add_argument(
        "--model-repo",
        help=(
            "Trained-model HF repo (id or URL). When provided, the orchestrator "
            "calls scripts.analysis.auto_resolve.resolve() to autofill the "
            "post-rl-eval / baseline-eval / *-ts / training-log-dir flags from "
            "Supabase (models + sandbox_jobs tables) and the HF Hub. Explicit "
            "CLI values still win on conflict."
        ),
    )
    parser.add_argument(
        "--baseline-eval",
        help="Pre-RL eval traces. Pair with --post-rl-eval for Q1/Q3/Q4 diffs.",
    )
    parser.add_argument("--baseline-eval-ts", help="ISO timestamp for the baseline eval marker (Q3 overlay)")
    parser.add_argument(
        "--post-rl-eval",
        help="Post-RL eval traces. Pair with --baseline-eval for Q1/Q3/Q4 diffs.",
    )
    parser.add_argument("--post-rl-eval-ts", help="ISO timestamp for the post-RL eval marker (Q3 overlay)")
    parser.add_argument(
        "--extra-eval",
        action="append",
        default=[],
        help="Additional eval-checkpoint specs for the Q3 overlay: '<source>[@label][:ts]'",
    )
    parser.add_argument(
        "--training-log-dir",
        help="Directory of SkyRL training logs (drives Q2 attribution analysis)",
    )
    parser.add_argument(
        "--no-fetch-training-logs",
        action="store_true",
        help="When --model-repo is set, skip the HF training_logs/ snapshot step",
    )
    parser.add_argument(
        "--eval-selection",
        choices=("largest-delta", "largest-abs-delta", "latest", "benchmark"),
        default="largest-delta",
        help=(
            "How auto-resolve picks among multiple eval-jobs: "
            "'largest-delta' (default — match post/baseline by benchmark, "
            "pick the pair with the biggest positive score gain — usually "
            "the most interesting), 'largest-abs-delta' (rank by |delta|, "
            "catches regressions too), 'latest' (most recent post-RL job), "
            "or 'benchmark' (pin to --eval-benchmark)."
        ),
    )
    parser.add_argument(
        "--eval-benchmark",
        help="Benchmark UUID or name to pin to (used with --eval-selection=benchmark)",
    )
    parser.add_argument(
        "--eval-score-key",
        help="Which metrics entry to use for score-delta ranking (default: 'accuracy' then a fallback chain)",
    )
    parser.add_argument(
        "--list-evals",
        action="store_true",
        help=(
            "When set with --model-repo, print all available eval pairs "
            "(matched + post-only + baseline-only) and exit without running "
            "the pipeline. Useful for choosing --eval-benchmark."
        ),
    )
    parser.add_argument(
        "--post-training-config",
        help="JSON config for post_training_comparison (Base/SFT/RL stages); see that script's docs",
    )
    parser.add_argument(
        "--annotate-failure-modes",
        action="store_true",
        help="Run update_hf_failure_modes on baseline + post-rl FIRST (GPT-5 API; can be slow)",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help=(
            "Enable Q1.llm_judge_diff (GPT-5 same-task pair classification). "
            "Off by default since each pair costs API money. Caches results "
            "in <output-dir>/Q1_llm_judge_diff/llm_judge_cache.json — re-runs "
            "are cheap."
        ),
    )
    parser.add_argument(
        "--llm-judge-model",
        default="openai/gpt-5-2025-08-07",
        help="LiteLLM-style model id for the LLM judge step (default GPT-5)",
    )
    parser.add_argument(
        "--llm-judge-max-pairs",
        type=int,
        default=20,
        help="Pairs judged PER probe (default 20). Applied to BOTH the most-changed "
             "and the random-control judge runs, so total GPT-5 pairs ≈ 2× this.",
    )
    parser.add_argument(
        "--llm-judge-concurrent",
        type=int,
        default=4,
        help="Max concurrent LLM-judge API calls (default 4)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root directory where each step's output is written",
    )
    parser.add_argument("--rl-bin-hours", type=float, default=4.0, help="Hours per RL temporal bin")
    parser.add_argument("--pair-top-n", type=int, default=25, help="Number of side-by-side trace pairs to render")
    parser.add_argument("--max-rows", type=int, default=None, help="Per-source row cap (for smoke testing)")
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated step names to skip (e.g. 'Q2.parse_skyrl_metrics,Q4.solve_rate_by_context')",
    )
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated step names to run exclusively (everything else skipped)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run steps even if their output marker exists")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned steps without running anything",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize HuggingFace URLs to bare ``owner/name`` so sub-script calls
    # to ``datasets.load_dataset`` succeed regardless of whether the caller
    # passed a full https URL.
    from scripts.analysis.auto_resolve import parse_hf_url as _parse_hf_url

    def _norm_hf(value):
        if not value:
            return value
        try:
            return _parse_hf_url(value)
        except Exception:
            return value

    for _attr in ("rl_traces", "model_repo", "baseline_eval", "post_rl_eval"):
        if getattr(args, _attr, None):
            setattr(args, _attr, _norm_hf(getattr(args, _attr)))

    # --list-evals mode: print the candidate eval pairs and exit.
    if args.list_evals:
        if not args.model_repo:
            print("[orchestrator] --list-evals requires --model-repo", file=sys.stderr)
            return 2
        from scripts.analysis.auto_resolve import list_evals_for_model
        info = list_evals_for_model(
            rl_traces=args.rl_traces or "",
            model_repo=args.model_repo,
            score_key=args.eval_score_key,
        )
        if "error" in info:
            print(f"[orchestrator] {info['error']}", file=sys.stderr)
            return 2
        print(f"model_id={info['model_id']}  base_model_id={info.get('base_model_id')}")
        if info["matched"]:
            print(f"\nMatched pairs (sorted by delta desc; pass --eval-benchmark <id> to pin):")
            print(f"  {'benchmark':<38} {'post':>8} {'base':>8} {'delta':>8}")
            for r in info["matched"]:
                ps = "—" if r["post_score"] is None else f"{r['post_score']:.4f}"
                bs = "—" if r["baseline_score"] is None else f"{r['baseline_score']:.4f}"
                d = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
                print(f"  {r['benchmark_id']:<38} {ps:>8} {bs:>8} {d:>8}")
        if info["post_only"]:
            print(f"\nPost-RL only ({len(info['post_only'])} benchmark(s); no baseline pair):")
            for r in info["post_only"]:
                print(f"  {r['benchmark_id']:<38} score={r['post_score']}")
        if info["baseline_only"]:
            print(f"\nBaseline only ({len(info['baseline_only'])} benchmark(s); no post-RL pair):")
            for r in info["baseline_only"]:
                print(f"  {r['benchmark_id']:<38} score={r['baseline_score']}")
        return 0

    # Autofill from supabase + HF if --model-repo was provided.
    # CLI-set values win on conflict (resolver only fills blanks).
    if args.model_repo:
        if not args.rl_traces:
            print(
                "[orchestrator] --model-repo set but --rl-traces missing; "
                "Q2 temporal analysis won't run without RL-time traces.",
                file=sys.stderr,
            )
        from scripts.analysis.auto_resolve import resolve as _resolve
        resolved = _resolve(
            rl_traces=args.rl_traces or "",
            model_repo=args.model_repo,
            eval_selection=args.eval_selection,
            eval_benchmark=args.eval_benchmark,
            eval_score_key=args.eval_score_key,
            fetch_training_logs=not args.no_fetch_training_logs,
        )
        fills = resolved.as_dict()
        applied: List[str] = []
        for key, value in fills.items():
            if value is None:
                continue
            attr = key  # post_rl_eval, post_rl_eval_ts, baseline_eval, baseline_eval_ts, training_log_dir
            if getattr(args, attr, None):
                continue  # CLI already set; don't clobber
            setattr(args, attr, value)
            applied.append(f"--{attr.replace('_', '-')}={value}")
        # Persist the resolver report alongside the pipeline plan for transparency.
        (args.output_dir / "auto_resolve.json").write_text(
            json.dumps(
                {
                    "model_repo": args.model_repo,
                    "rl_traces": args.rl_traces,
                    "applied": applied,
                    "resolver_fills": fills,
                    "notes": resolved.notes,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if applied:
            print(f"\n[orchestrator] auto-resolve applied {len(applied)} arg(s):")
            for a in applied:
                print(f"  {a}")
        if resolved.notes:
            print("[orchestrator] auto-resolve notes:")
            for n in resolved.notes:
                print(f"  - {n}")

    # Re-normalize after the resolver — the resolver may have filled
    # ``baseline_eval`` / ``post_rl_eval`` with full https URLs.
    for _attr in ("rl_traces", "model_repo", "baseline_eval", "post_rl_eval"):
        if getattr(args, _attr, None):
            setattr(args, _attr, _norm_hf(getattr(args, _attr)))

    steps = _build_steps(args)
    if not steps:
        print(
            "[orchestrator] No steps planned. Provide at least one of "
            "--rl-traces, --baseline-eval+--post-rl-eval, or --post-training-config.",
            file=sys.stderr,
        )
        return 2

    skip_set = {s.strip() for s in args.skip.split(",") if s.strip()}
    only_set = {s.strip() for s in args.only.split(",") if s.strip()}

    plan_path = args.output_dir / "pipeline_plan.json"
    plan = []
    for step in steps:
        status = "planned"
        if only_set and step.name not in only_set:
            status = "skipped:not-in-only"
        elif step.name in skip_set:
            status = "skipped:--skip"
        elif step.output_marker.exists() and step.output_marker.stat().st_size > 0 and not args.force:
            status = f"skipped:already-present ({step.output_marker})"
        plan.append({
            "name": step.name,
            "question": step.question,
            "description": step.description,
            "output_marker": str(step.output_marker),
            "status": status,
            "optional": step.optional,
        })
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    # Print plan summary.
    print(f"\n[orchestrator] Plan ({args.output_dir}/pipeline_plan.json):")
    for p in plan:
        print(f"  [{p['status']:30s}] {p['name']:40s} {p['question']:4s} {p['description']}")
    if args.dry_run:
        return 0

    failures: List[str] = []
    for step, p in zip(steps, plan):
        if p["status"].startswith("skipped"):
            continue
        rc = step.runner()
        if rc != 0:
            (failures if not step.optional else []).append(f"{step.name} (rc={rc})")
            print(
                f"[orchestrator] step {step.name} failed (rc={rc})"
                + (" — non-fatal, optional step" if step.optional else ""),
                file=sys.stderr,
            )

    # Write the final index report.
    index_path = args.output_dir / "INDEX.md"
    index = ["# RL Behavioral Analysis", "", f"Generated {datetime.utcnow().isoformat()}Z", ""]
    for q_label, q_name in [
        ("Q1", "What model behaviors are changing as a result of RL?"),
        ("Q2", "Are reward changes attributable to behavior changes or other factors?"),
        ("Q3", "Do behavioral changes persist in post-RL eval traces?"),
        ("Q4", "Do those changes affect eval results? If not, is it expected?"),
    ]:
        index += [f"## {q_label}. {q_name}", ""]
        for step, p in zip(steps, plan):
            if step.question != q_label:
                continue
            link = step.output_marker.relative_to(args.output_dir) if step.output_marker.exists() else None
            if link:
                index.append(f"- **{step.name}** — [{link}]({link}) — {step.description}")
            else:
                index.append(f"- **{step.name}** — _(not produced)_ — {step.description}")
        index.append("")
    index_path.write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\n[orchestrator] index → {index_path}")
    print(f"[orchestrator] plan → {plan_path}")

    if failures:
        print(f"[orchestrator] {len(failures)} required step(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
