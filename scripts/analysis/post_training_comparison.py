#!/usr/bin/env python3
"""Compare eval results across post-training stages.

Analyzes 5 model checkpoints on 71-task dev set:
  1. Base (Qwen3-8B-Base)
  2. Base -> Strong SFT
  3. Base -> Weak SFT (Qwen3-8B instruct)
  4. Base -> Weak SFT -> Strong SFT
  5. Base -> Weak SFT -> Strong SFT -> RL

Reports: per-task best scores, failure modes, context lengths,
task-level agreement/disagreement between models, and whether
models converge to a common skill ceiling.

Usage:
    python -m scripts.analysis.post_training_comparison \
        --output /Users/benjaminfeuer/Documents/notes/ot-agent/post_training_comparison.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from scripts.analysis.utils import (
    extract_conversation_text,
    extract_error_type,
    extract_reward,
    ei_common_tasks,
    filter_ei,
    mean_reward_per_trial,
    mean_reward_per_trial_ei,
)

# ── Dataset definitions ──

MODELS = {
    "base": {
        "repo": "DCAgent2/dcagent-dev-set-71-tasks-qwen-qwen3-8b-base-20251120-143150",
        "label": "Base (Qwen3-8B-Base)",
        "short": "base",
        "pipeline": "base",
    },
    "base_strong_sft": {
        "repo": "DCAgent2/dev_set_71_tasks_qwen3base_GLM_4_7_swesmith_sandboxes_with_tests_oracle_verifiefd45c7e6",
        "label": "Base → Strong SFT",
        "short": "b→sSFT",
        "pipeline": "base → strong_sft",
    },
    "base_weak_sft": {
        "repo": "DCAgent2/TEST_DCAgent_dev_set_71_tasks_Qwen_Qwen3-8B_thinking_false_20260215_073417",
        "label": "Base → Weak SFT",
        "short": "b→wSFT",
        "pipeline": "base → weak_sft",
    },
    "base_weak_strong_sft": {
        "repo": "DCAgent2/dev_set_71_tasks_GLM_4_7_swesmith_sandboxes_with_tests_oracle_verified_120s_max03c50f55",
        "label": "Base → Weak SFT → Strong SFT",
        "short": "b→w→sSFT",
        "pipeline": "base → weak_sft → strong_sft",
    },
    "base_weak_strong_rl": {
        "repo": "DCAgent2/dev_set_71_tasks_rl_swesmith_fixthink_pymethods2test_45_20260302_185735",
        "label": "Base → Weak SFT → Strong SFT → RL",
        "short": "b→w→s→RL",
        "pipeline": "base → weak_sft → strong_sft → rl",
    },
}


def load_all_datasets() -> dict[str, list[dict]]:
    """Load all datasets and return as {model_key: [rows]}."""
    datasets = {}
    for key, info in MODELS.items():
        print(f"Loading {info['label']}...")
        ds = load_dataset(info["repo"], split="train")
        datasets[key] = [dict(row) for row in ds]
        print(f"  {len(ds)} rows, {len(set(r['task'] for r in datasets[key]))} tasks")
    return datasets


def best_score_per_task(rows: list[dict]) -> dict[str, float]:
    """For each task, return the best (max) reward across episodes.

    If the dataset has no 'result' column, returns empty dict.
    """
    task_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reward = extract_reward(row)
        if reward is not None:
            task_scores[row["task"]].append(reward)
    return {task: max(scores) for task, scores in task_scores.items()}


def mean_score_per_task_ei(rows: list[dict]) -> dict[str, float]:
    """Mean score per task after dropping infrastructure errors."""
    clean = filter_ei(rows)
    task_scores: dict[str, list[float]] = defaultdict(list)
    for row in clean:
        reward = extract_reward(row)
        values = reward if reward is not None else 0.0
        task_scores[row["task"]].append(values)
    return {task: np.mean(scores) for task, scores in task_scores.items()}


def failure_mode_distribution(rows: list[dict]) -> Counter:
    """Count failure modes across all rows."""
    modes = Counter()
    for row in rows:
        reward = extract_reward(row)
        error = extract_error_type(row)
        if error:
            modes[error] += 1
        elif reward is not None:
            if reward == 1.0:
                modes["success (1.0)"] += 1
            elif reward == 0.0:
                modes["fail (0.0)"] += 1
            else:
                modes[f"partial ({reward:.2f})"] += 1
        else:
            modes["no_result"] += 1
    return modes


def conversation_lengths(rows: list[dict]) -> np.ndarray:
    """Return array of conversation character lengths."""
    return np.array([len(extract_conversation_text(r)) for r in rows])


def turn_counts(rows: list[dict]) -> np.ndarray:
    """Return array of turn counts per row."""
    counts = []
    for row in rows:
        convs = row.get("conversations") or row.get("messages") or []
        counts.append(len(convs))
    return np.array(counts)


def compute_task_level_agreement(
    model_scores: dict[str, dict[str, float]],
    common_tasks: set[str],
    threshold: float = 0.5,
) -> dict:
    """Analyze task-level agreement: which tasks do all models solve, none solve, etc."""
    all_solve = []
    none_solve = []
    some_solve = []
    unique_solvers: dict[str, list[str]] = defaultdict(list)

    for task in sorted(common_tasks):
        solvers = []
        for model_key, scores in model_scores.items():
            if model_key == "base":
                continue  # base has no result column
            score = scores.get(task, 0.0)
            if score >= threshold:
                solvers.append(model_key)

        if len(solvers) == len(model_scores) - 1:  # all non-base models
            all_solve.append(task)
        elif len(solvers) == 0:
            none_solve.append(task)
        else:
            some_solve.append((task, solvers))
            if len(solvers) == 1:
                unique_solvers[solvers[0]].append(task)

    return {
        "all_solve": all_solve,
        "none_solve": none_solve,
        "some_solve": some_solve,
        "unique_solvers": dict(unique_solvers),
    }


def format_report(
    datasets: dict[str, list[dict]],
    output_path: Path,
) -> str:
    """Generate the full comparison report."""
    lines = []
    lines.append("# Post-Training Stage Comparison — 71-Task Dev Set")
    lines.append("")
    lines.append("## Models Compared")
    lines.append("")
    lines.append("| Key | Pipeline | Rows | Tasks |")
    lines.append("|-----|----------|------|-------|")
    for key, info in MODELS.items():
        rows = datasets[key]
        n_tasks = len(set(r["task"] for r in rows))
        lines.append(f"| {info['short']} | {info['pipeline']} | {len(rows)} | {n_tasks} |")
    lines.append("")

    # ── 1. Overall scores ──
    lines.append("## 1. Overall Scores (Best-of-N per Task)")
    lines.append("")

    model_scores = {}
    for key in MODELS:
        model_scores[key] = best_score_per_task(datasets[key])

    # Find common tasks
    all_tasks = [set(r["task"] for r in datasets[key]) for key in MODELS]
    common_tasks = set.intersection(*all_tasks)
    lines.append(f"Common tasks across all models: **{len(common_tasks)}**")
    lines.append("")

    scored_models = {k: v for k, v in model_scores.items() if v}  # exclude base (no results)

    lines.append("| Model | Mean Score | Median | Tasks ≥0.5 | Tasks =1.0 | Tasks =0.0 |")
    lines.append("|-------|-----------|--------|-----------|-----------|-----------|")
    for key, scores in scored_models.items():
        task_scores = [scores.get(t, 0.0) for t in common_tasks]
        mean_s = np.mean(task_scores)
        med_s = np.median(task_scores)
        ge_half = sum(1 for s in task_scores if s >= 0.5)
        eq_one = sum(1 for s in task_scores if s == 1.0)
        eq_zero = sum(1 for s in task_scores if s == 0.0)
        info = MODELS[key]
        lines.append(
            f"| {info['short']} | {mean_s:.3f} | {med_s:.3f} | "
            f"{ge_half}/{len(common_tasks)} | {eq_one}/{len(common_tasks)} | "
            f"{eq_zero}/{len(common_tasks)} |"
        )
    lines.append("")

    # ── 1b. Harbor-style mean reward (flat mean across all trials) ──
    lines.append("### Mean Reward per Trial (Harbor 'accuracy' metric)")
    lines.append("")
    lines.append("Flat mean of all trial rewards (errors=0). EI = infrastructure errors dropped.")
    lines.append("")
    lines.append("| Model | Mean Reward | Mean Reward (EI) | Trials | Trials (EI) |")
    lines.append("|-------|------------|-----------------|--------|------------|")
    for key in MODELS:
        mr = mean_reward_per_trial(datasets[key])
        mr_ei = mean_reward_per_trial_ei(datasets[key])
        n_ei = len(filter_ei(datasets[key]))
        info = MODELS[key]
        mr_str = f"{mr:.3f}" if mr is not None else "N/A"
        mr_ei_str = f"{mr_ei:.3f}" if mr_ei is not None else "N/A"
        lines.append(
            f"| {info['short']} | {mr_str} | {mr_ei_str} | {len(datasets[key])} | {n_ei} |"
        )
    lines.append("")

    # ── 1c. EI-filtered common tasks ──
    ei_tasks = ei_common_tasks(datasets)
    lines.append(f"### EI-Filtered Common Tasks: {len(ei_tasks)}")
    lines.append("")
    lines.append("Tasks where all models have valid (non-infra-error) results.")
    lines.append("")

    model_mean_ei = {}
    for key in MODELS:
        model_mean_ei[key] = mean_score_per_task_ei(datasets[key])

    scored_ei = {k: v for k, v in model_mean_ei.items() if v}
    lines.append("| Model | Mean (EI common) | Median | >=0.5 | =1.0 | =0.0 |")
    lines.append("|-------|-----------------|--------|-------|------|------|")
    for key, scores_dict in scored_ei.items():
        scores = [scores_dict.get(t, 0.0) for t in ei_tasks]
        if scores:
            mean_s = np.mean(scores)
            med_s = np.median(scores)
            ge_half = sum(1 for s in scores if s >= 0.5)
            eq_one = sum(1 for s in scores if s == 1.0)
            eq_zero = sum(1 for s in scores if s == 0.0)
            lines.append(
                f"| {MODELS[key]['short']} | {mean_s:.3f} | {med_s:.3f} | "
                f"{ge_half}/{len(ei_tasks)} | {eq_one}/{len(ei_tasks)} | "
                f"{eq_zero}/{len(ei_tasks)} |"
            )
    lines.append("")

    # ── 2. Failure mode breakdown ──
    lines.append("## 2. Failure Mode Breakdown (All Rows)")
    lines.append("")

    all_failure_modes = set()
    model_failures = {}
    for key in MODELS:
        fm = failure_mode_distribution(datasets[key])
        model_failures[key] = fm
        all_failure_modes.update(fm.keys())

    # Sort modes for display
    mode_order = sorted(all_failure_modes, key=lambda m: (
        0 if "success" in m else 1 if "partial" in m else 2 if "fail" in m else 3
    ))

    header = "| Failure Mode | " + " | ".join(MODELS[k]["short"] for k in MODELS) + " |"
    sep = "|" + "|".join(["---"] * (len(MODELS) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for mode in mode_order:
        row_parts = [f" {mode} "]
        for key in MODELS:
            count = model_failures[key].get(mode, 0)
            total = len(datasets[key])
            if count > 0:
                row_parts.append(f" {count} ({100*count/total:.0f}%) ")
            else:
                row_parts.append(" - ")
        lines.append("|" + "|".join(row_parts) + "|")
    lines.append("")

    # ── 3. Context / conversation length stats ──
    lines.append("## 3. Context Length & Turn Count Statistics")
    lines.append("")
    lines.append("| Model | Mean Chars | Median Chars | P90 Chars | Mean Turns | Median Turns | P90 Turns |")
    lines.append("|-------|-----------|-------------|----------|-----------|-------------|----------|")
    for key in MODELS:
        clens = conversation_lengths(datasets[key])
        turns = turn_counts(datasets[key])
        info = MODELS[key]
        lines.append(
            f"| {info['short']} | {np.mean(clens):,.0f} | {np.median(clens):,.0f} | "
            f"{np.percentile(clens, 90):,.0f} | {np.mean(turns):.1f} | "
            f"{np.median(turns):.0f} | {np.percentile(turns, 90):.0f} |"
        )
    lines.append("")

    # ── 4. Task-level agreement analysis (EI-filtered) ──
    lines.append(f"## 4. Task-Level Agreement — EI-filtered ({len(ei_tasks)} tasks)")
    lines.append("")

    agreement = compute_task_level_agreement(model_mean_ei, ei_tasks)

    n_scored = len([k for k in model_mean_ei if model_mean_ei[k]])
    lines.append(f"- **All {n_scored} scored models solve (>=0.5):** {len(agreement['all_solve'])} tasks")
    lines.append(f"- **No model solves:** {len(agreement['none_solve'])} tasks")
    lines.append(f"- **Some but not all solve:** {len(agreement['some_solve'])} tasks")
    lines.append("")

    if agreement["unique_solvers"]:
        lines.append("### Uniquely Solved Tasks (only 1 model solves)")
        lines.append("")
        for model_key, tasks in sorted(agreement["unique_solvers"].items()):
            info = MODELS[model_key]
            lines.append(f"- **{info['short']}**: {len(tasks)} unique solves — `{'`, `'.join(tasks[:5])}`{'...' if len(tasks) > 5 else ''}")
        lines.append("")

    # ── 5. Per-task score comparison (EI-filtered) ──
    lines.append(f"## 5. Per-Task Score Comparison — EI-filtered ({len(ei_tasks)} tasks)")
    lines.append("")
    lines.append("Tasks where models diverge most (sorted by score variance):")
    lines.append("")

    task_variance = []
    for task in ei_tasks:
        scores_for_task = []
        for key in scored_ei:
            scores_for_task.append(scored_ei[key].get(task, 0.0))
        var = np.var(scores_for_task)
        task_variance.append((task, var, scores_for_task))

    task_variance.sort(key=lambda x: -x[1])

    header = "| Task | " + " | ".join(MODELS[k]["short"] for k in scored_ei) + " | Var |"
    sep = "|" + "|".join(["---"] * (len(scored_ei) + 2)) + "|"
    lines.append(header)
    lines.append(sep)
    for task, var, scores in task_variance[:20]:
        task_short = task[:30]
        score_strs = [f" {s:.2f} " for s in scores]
        lines.append(f"| `{task_short}` |" + "|".join(score_strs) + f"| {var:.3f} |")
    lines.append("")

    # ── 6. Pairwise improvement analysis (EI-filtered) ──
    lines.append(f"## 6. Pairwise Improvement Analysis (EI-filtered, {len(ei_tasks)} tasks)")
    lines.append("")
    lines.append("Effect of each post-training stage (EI-filtered mean scores):")
    lines.append("")

    comparisons = [
        ("base_weak_sft", "base_weak_strong_sft", "Adding Strong SFT to Weak SFT"),
        ("base_strong_sft", "base_weak_strong_sft", "Adding Weak SFT before Strong SFT"),
        ("base_weak_strong_sft", "base_weak_strong_rl", "Adding RL after SFT"),
        ("base_strong_sft", "base_weak_strong_rl", "Full pipeline vs Direct Strong SFT"),
        ("base_weak_sft", "base_weak_strong_rl", "Full pipeline vs Weak SFT alone"),
    ]

    for key_a, key_b, description in comparisons:
        if key_a not in scored_ei or key_b not in scored_ei:
            continue
        scores_a = scored_ei[key_a]
        scores_b = scored_ei[key_b]

        improved = 0
        regressed = 0
        unchanged = 0
        deltas = []

        for task in ei_tasks:
            sa = scores_a.get(task, 0.0)
            sb = scores_b.get(task, 0.0)
            delta = sb - sa
            deltas.append(delta)
            if delta > 0.01:
                improved += 1
            elif delta < -0.01:
                regressed += 1
            else:
                unchanged += 1

        mean_delta = np.mean(deltas)
        lines.append(f"### {description}")
        lines.append(f"- {MODELS[key_a]['short']} -> {MODELS[key_b]['short']}")
        lines.append(f"- Mean Δ: **{mean_delta:+.3f}**")
        lines.append(f"- Improved: {improved}, Regressed: {regressed}, Unchanged: {unchanged}")
        lines.append("")

    # ── 7. Convergence analysis (EI-filtered) ──
    lines.append(f"## 7. Convergence Analysis (EI-filtered, {len(ei_tasks)} tasks)")
    lines.append("")
    lines.append("Do the strongest models converge to the same solution set?")
    lines.append("")

    if "base_weak_strong_sft" in scored_ei and "base_weak_strong_rl" in scored_ei:
        sft_scores = scored_ei["base_weak_strong_sft"]
        rl_scores = scored_ei["base_weak_strong_rl"]

        both_solve = 0
        sft_only = 0
        rl_only = 0
        neither = 0
        sft_only_tasks = []
        rl_only_tasks = []

        for task in ei_tasks:
            sft_pass = sft_scores.get(task, 0.0) >= 0.5
            rl_pass = rl_scores.get(task, 0.0) >= 0.5
            if sft_pass and rl_pass:
                both_solve += 1
            elif sft_pass:
                sft_only += 1
                sft_only_tasks.append(task)
            elif rl_pass:
                rl_only += 1
                rl_only_tasks.append(task)
            else:
                neither += 1

        lines.append("### SFT-only vs SFT+RL (best two models)")
        lines.append(f"- Both solve: **{both_solve}** tasks")
        lines.append(f"- SFT-only solves (RL regresses): **{sft_only}** tasks")
        lines.append(f"- RL-only solves (SFT can't): **{rl_only}** tasks")
        lines.append(f"- Neither solves: **{neither}** tasks")
        lines.append("")

        if sft_only_tasks:
            lines.append(f"SFT-only tasks: `{'`, `'.join(t[:25] for t in sft_only_tasks[:10])}`")
        if rl_only_tasks:
            lines.append(f"RL-only tasks: `{'`, `'.join(t[:25] for t in rl_only_tasks[:10])}`")
        lines.append("")

    if "base_strong_sft" in scored_ei and "base_weak_strong_sft" in scored_ei:
        direct = scored_ei["base_strong_sft"]
        staged = scored_ei["base_weak_strong_sft"]

        both_solve = 0
        direct_only = 0
        staged_only = 0
        neither = 0

        for task in ei_tasks:
            d_pass = direct.get(task, 0.0) >= 0.5
            s_pass = staged.get(task, 0.0) >= 0.5
            if d_pass and s_pass:
                both_solve += 1
            elif d_pass:
                direct_only += 1
            elif s_pass:
                staged_only += 1
            else:
                neither += 1

        lines.append("### Direct Strong SFT vs Staged (Weak->Strong) SFT")
        lines.append(f"- Both solve: **{both_solve}** tasks")
        lines.append(f"- Direct-only: **{direct_only}** tasks")
        lines.append(f"- Staged-only: **{staged_only}** tasks")
        lines.append(f"- Neither: **{neither}** tasks")
        lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Post-training stage comparison analysis")
    parser.add_argument(
        "--output",
        default="/Users/benjaminfeuer/Documents/notes/ot-agent/post_training_comparison.md",
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    datasets = load_all_datasets()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = format_report(datasets, output_path)
    print(f"\nReport written to {output_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
