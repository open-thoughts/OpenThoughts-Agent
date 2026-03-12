#!/usr/bin/env python3
"""Method ablation comparison across SFT and RL variants.

Compares multiple variants within the same post-training stage to understand
how training data, thinking mode, and verbosity affect performance.

Accepts model definitions via --config JSON file or uses built-in defaults.

Usage:
    python -m scripts.analysis.method_ablation \
        --output /Users/benjaminfeuer/Documents/notes/ot-agent/method_ablation.md
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

# ── Default model definitions ──

DEFAULT_MODELS = {
    # --- SFT variants (base → weak SFT → strong SFT) ---
    "sft_A": {
        "repo": "DCAgent2/dev_set_71_tasks_GLM_4_7_swesmith_sandboxes_with_tests_oracle_verified_120s_max03c50f55",
        "label": "SFT-A: swesmith+oracle",
        "short": "SFT-A",
        "group": "SFT",
        "notes": "swesmith-sandboxes, oracle-verified, 120s, 131k maxeps",
    },
    "sft_B": {
        "repo": "DCAgent2/dev_set_71_tasks_exp_tas_optimal_combined_traces_20260224_124603",
        "label": "SFT-B: optimal-combined",
        "short": "SFT-B",
        "group": "SFT",
        "notes": "optimal combined traces",
    },
    "sft_C": {
        "repo": "penfever/eval__openthoughts-tb-dev__r2egym-nl2bash-stack__lambda",
        "label": "SFT-C: r2egym+nl2bash+stack (cloud eval)",
        "short": "SFT-C",
        "group": "SFT",
        "notes": "r2egym+nl2bash+stack, cloud eval on lambda",
    },
    # --- RL variants (base → weak SFT → strong SFT → RL) ---
    "rl_A": {
        "repo": "DCAgent2/dev_set_71_tasks_rl_swesmith_fixthink_pymethods2test_45_20260302_185735",
        "label": "RL-A: thinking, chatty",
        "short": "RL-A",
        "group": "RL",
        "notes": "thinking, chatty, swesmith-fixthink + pymethods2test, 45 steps",
    },
    "rl_B": {
        "repo": "DCAgent2/dev_set_71_tasks_rl_rl_conf_24GP_base_yaml_mode_path_exp_tas_opti_comb_trac_tra998a8eff",
        "label": "RL-B: thinking, concise",
        "short": "RL-B",
        "group": "RL",
        "notes": "thinking, more concise, optimal combined traces",
    },
    "rl_C": {
        "repo": "DCAgent2/dev_set_71_tasks_rl_rl_conf_24GP_base_noth_yaml_mode_path_r2eg_nl2b_stac_bugs_t3c184dcc",
        "label": "RL-C: no thinking",
        "short": "RL-C",
        "group": "RL",
        "notes": "no thinking, rl-conf 24GPU base nothink, r2egym+nl2bash+stack-bugs",
    },
}


def load_all_datasets(models: dict) -> dict[str, list[dict]]:
    datasets = {}
    for key, info in models.items():
        print(f"Loading {info['label']}...")
        ds = load_dataset(info["repo"], split="train")
        datasets[key] = [dict(row) for row in ds]
        n_tasks = len(set(r["task"] for r in datasets[key]))
        print(f"  {len(ds)} rows, {n_tasks} tasks")
    return datasets


def best_score_per_task(rows: list[dict]) -> dict[str, float]:
    task_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reward = extract_reward(row)
        if reward is not None:
            task_scores[row["task"]].append(reward)
    return {task: max(scores) for task, scores in task_scores.items()}


def mean_score_per_task(rows: list[dict]) -> dict[str, float]:
    task_scores: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        reward = extract_reward(row)
        if reward is not None:
            task_scores[row["task"]].append(reward)
    return {task: np.mean(scores) for task, scores in task_scores.items()}


def mean_score_per_task_ei(rows: list[dict]) -> dict[str, float]:
    """Mean score per task after dropping infrastructure errors."""
    clean = filter_ei(rows)
    task_scores: dict[str, list[float]] = defaultdict(list)
    for row in clean:
        reward = extract_reward(row)
        values = reward if reward is not None else 0.0
        task_scores[row["task"]].append(values)
    return {task: np.mean(scores) for task, scores in task_scores.items()}


def failure_mode_summary(rows: list[dict]) -> dict[str, int]:
    """Collapse into major categories."""
    cats = Counter()
    for row in rows:
        reward = extract_reward(row)
        error = extract_error_type(row)
        if error:
            cats[error] += 1
        elif reward is not None:
            if reward == 1.0:
                cats["success"] += 1
            elif reward == 0.0:
                cats["fail"] += 1
            else:
                cats["partial"] += 1
        else:
            cats["no_result"] += 1
    return dict(cats)


def conversation_stats(rows: list[dict]) -> dict:
    char_lens = [len(extract_conversation_text(r)) for r in rows]
    turn_lens = [len(r.get("conversations") or r.get("messages") or []) for r in rows]
    arr_c = np.array(char_lens)
    arr_t = np.array(turn_lens)
    return {
        "mean_chars": int(np.mean(arr_c)),
        "median_chars": int(np.median(arr_c)),
        "p90_chars": int(np.percentile(arr_c, 90)),
        "max_chars": int(np.max(arr_c)),
        "mean_turns": round(float(np.mean(arr_t)), 1),
        "median_turns": int(np.median(arr_t)),
        "p90_turns": int(np.percentile(arr_t, 90)),
    }


def format_report(
    models: dict,
    datasets: dict[str, list[dict]],
    output_path: Path,
) -> str:
    lines = []
    lines.append("# Method Ablation — SFT & RL Variants on 71-Task Dev Set")
    lines.append("")

    # Group info
    groups = defaultdict(list)
    for key, info in models.items():
        groups[info.get("group", "unknown")].append(key)

    # ── Model overview ──
    lines.append("## Models")
    lines.append("")
    lines.append("| Key | Group | Description | Notes |")
    lines.append("|-----|-------|-------------|-------|")
    for key, info in models.items():
        lines.append(f"| {info['short']} | {info.get('group','')} | {info['label']} | {info.get('notes','')} |")
    lines.append("")

    # ── Compute scores ──
    model_best = {}
    model_mean = {}
    for key in models:
        model_best[key] = best_score_per_task(datasets[key])
        model_mean[key] = mean_score_per_task(datasets[key])

    all_task_sets = [set(r["task"] for r in datasets[key]) for key in models]
    common_tasks = sorted(set.intersection(*all_task_sets))
    lines.append(f"**Common tasks: {len(common_tasks)}**")
    lines.append("")

    # ── 1. Score summary ──
    lines.append("## 1. Score Summary (Best-of-N, Common Tasks)")
    lines.append("")
    lines.append("| Model | Mean | Median | ≥0.5 | =1.0 | =0.0 | Partial |")
    lines.append("|-------|------|--------|------|------|------|---------|")
    for key in models:
        scores = [model_best[key].get(t, 0.0) for t in common_tasks]
        mean_s = np.mean(scores)
        med_s = np.median(scores)
        ge_half = sum(1 for s in scores if s >= 0.5)
        eq_one = sum(1 for s in scores if s == 1.0)
        eq_zero = sum(1 for s in scores if s == 0.0)
        partial = sum(1 for s in scores if 0 < s < 1.0)
        lines.append(
            f"| {models[key]['short']} | {mean_s:.3f} | {med_s:.3f} | "
            f"{ge_half} | {eq_one} | {eq_zero} | {partial} |"
        )
    lines.append("")

    # ── 1b. Harbor-style mean reward (flat mean across all trials) ──
    lines.append("### Mean Reward per Trial (Harbor 'accuracy' metric)")
    lines.append("")
    lines.append("Flat mean of all trial rewards (errors=0). Matches Harbor leaderboard metric.")
    lines.append("")
    lines.append("| Model | Mean Reward | Mean Reward (EI) | Trials | Trials (EI) |")
    lines.append("|-------|------------|-----------------|--------|------------|")
    for key in models:
        mr = mean_reward_per_trial(datasets[key])
        mr_ei = mean_reward_per_trial_ei(datasets[key])
        n_ei = len(filter_ei(datasets[key]))
        mr_str = f"{mr:.3f}" if mr is not None else "N/A"
        mr_ei_str = f"{mr_ei:.3f}" if mr_ei is not None else "N/A"
        lines.append(
            f"| {models[key]['short']} | {mr_str} | {mr_ei_str} | {len(datasets[key])} | {n_ei} |"
        )
    lines.append("")

    # ── 1c. EI-filtered common tasks ──
    ei_tasks = ei_common_tasks(datasets)
    lines.append(f"### EI-Filtered Common Tasks: {len(ei_tasks)}")
    lines.append("")
    lines.append("Tasks where all models have valid (non-infra-error) results.")
    lines.append("")

    model_mean_ei = {}
    for key in models:
        model_mean_ei[key] = mean_score_per_task_ei(datasets[key])

    lines.append("| Model | Mean (EI common) | Median | >=0.5 | =1.0 | =0.0 |")
    lines.append("|-------|-----------------|--------|-------|------|------|")
    for key in models:
        scores = [model_mean_ei[key].get(t, 0.0) for t in ei_tasks]
        if scores:
            mean_s = np.mean(scores)
            med_s = np.median(scores)
            ge_half = sum(1 for s in scores if s >= 0.5)
            eq_one = sum(1 for s in scores if s == 1.0)
            eq_zero = sum(1 for s in scores if s == 0.0)
            lines.append(
                f"| {models[key]['short']} | {mean_s:.3f} | {med_s:.3f} | "
                f"{ge_half} | {eq_one} | {eq_zero} |"
            )
        else:
            lines.append(f"| {models[key]['short']} | N/A | N/A | - | - | - |")
    lines.append("")

    # Mean-of-N (consistency)
    lines.append("## 2. Consistency (Mean-of-N, Common Tasks)")
    lines.append("")
    lines.append("Mean-of-N penalizes models that succeed rarely; best-of-N rewards any success.")
    lines.append("")
    lines.append("| Model | Best-of-N Mean | Mean-of-N Mean | Gap (luck factor) |")
    lines.append("|-------|---------------|---------------|-------------------|")
    for key in models:
        best_scores = [model_best[key].get(t, 0.0) for t in common_tasks]
        mean_scores = [model_mean[key].get(t, 0.0) for t in common_tasks]
        best_avg = np.mean(best_scores)
        mean_avg = np.mean(mean_scores)
        gap = best_avg - mean_avg
        lines.append(f"| {models[key]['short']} | {best_avg:.3f} | {mean_avg:.3f} | {gap:.3f} |")
    lines.append("")

    # ── 3. Failure modes ──
    lines.append("## 3. Failure Mode Breakdown")
    lines.append("")
    all_modes = set()
    model_fm = {}
    for key in models:
        fm = failure_mode_summary(datasets[key])
        model_fm[key] = fm
        all_modes.update(fm.keys())
    mode_order = ["success", "partial", "fail", "AgentTimeoutError",
                  "ContextLengthExceededError", "no_result"]
    mode_order = [m for m in mode_order if m in all_modes]
    mode_order += sorted(all_modes - set(mode_order))

    header = "| Mode | " + " | ".join(models[k]["short"] for k in models) + " |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(models) + 1)) + "|")
    for mode in mode_order:
        parts = [f" {mode} "]
        for key in models:
            count = model_fm[key].get(mode, 0)
            total = len(datasets[key])
            if count > 0:
                parts.append(f" {count} ({100*count/total:.0f}%) ")
            else:
                parts.append(" - ")
        lines.append("|" + "|".join(parts) + "|")
    lines.append("")

    # ── 4. Context length ──
    lines.append("## 4. Context Length & Verbosity")
    lines.append("")
    lines.append("| Model | Mean Chars | Median | P90 | Max | Mean Turns | Med Turns | P90 Turns |")
    lines.append("|-------|-----------|--------|-----|-----|-----------|----------|----------|")
    for key in models:
        s = conversation_stats(datasets[key])
        lines.append(
            f"| {models[key]['short']} | {s['mean_chars']:,} | {s['median_chars']:,} | "
            f"{s['p90_chars']:,} | {s['max_chars']:,} | {s['mean_turns']} | "
            f"{s['median_turns']} | {s['p90_turns']} |"
        )
    lines.append("")

    # ── 5. Within-group comparisons (EI-filtered common tasks) ──
    for group_name, group_keys in sorted(groups.items()):
        if len(group_keys) < 2:
            continue
        lines.append(f"## 5. Within-Group Pairwise: {group_name} Variants (EI-filtered)")
        lines.append("")

        for i, ka in enumerate(group_keys):
            for kb in group_keys[i+1:]:
                sa = model_mean_ei[ka]
                sb = model_mean_ei[kb]
                improved = regressed = unchanged = 0
                deltas = []
                for task in ei_tasks:
                    a_s = sa.get(task, 0.0)
                    b_s = sb.get(task, 0.0)
                    d = b_s - a_s
                    deltas.append(d)
                    if d > 0.01:
                        improved += 1
                    elif d < -0.01:
                        regressed += 1
                    else:
                        unchanged += 1
                lines.append(f"### {models[ka]['short']} → {models[kb]['short']}")
                lines.append(f"- Tasks compared: {len(ei_tasks)} (EI-filtered)")
                lines.append(f"- Mean Δ: **{np.mean(deltas):+.3f}**")
                lines.append(f"- Improved: {improved}, Regressed: {regressed}, Unchanged: {unchanged}")
                lines.append("")
        lines.append("")

    # ── 6. Cross-group: best SFT vs best RL (EI-filtered) ──
    lines.append("## 6. Cross-Group: SFT vs RL (EI-filtered)")
    lines.append("")

    sft_keys = groups.get("SFT", [])
    rl_keys = groups.get("RL", [])

    if sft_keys and rl_keys:
        # Find best SFT and best RL by EI-filtered mean score
        best_sft_key = max(sft_keys, key=lambda k: np.mean([model_mean_ei[k].get(t, 0.0) for t in ei_tasks]))
        best_rl_key = max(rl_keys, key=lambda k: np.mean([model_mean_ei[k].get(t, 0.0) for t in ei_tasks]))

        lines.append(f"Best SFT: **{models[best_sft_key]['short']}**, Best RL: **{models[best_rl_key]['short']}**")
        lines.append(f"(on {len(ei_tasks)} EI-filtered common tasks)")
        lines.append("")

        sa = model_mean_ei[best_sft_key]
        sb = model_mean_ei[best_rl_key]

        both = sft_only_tasks = rl_only_tasks = neither = 0
        sft_only_list = []
        rl_only_list = []
        for task in ei_tasks:
            s_pass = sa.get(task, 0.0) >= 0.5
            r_pass = sb.get(task, 0.0) >= 0.5
            if s_pass and r_pass:
                both += 1
            elif s_pass:
                sft_only_tasks += 1
                sft_only_list.append(task)
            elif r_pass:
                rl_only_tasks += 1
                rl_only_list.append(task)
            else:
                neither += 1

        lines.append(f"- Both solve: **{both}**")
        lines.append(f"- SFT-only: **{sft_only_tasks}** — `{'`, `'.join(t[:25] for t in sft_only_list[:8])}`")
        lines.append(f"- RL-only: **{rl_only_tasks}** — `{'`, `'.join(t[:25] for t in rl_only_list[:8])}`")
        lines.append(f"- Neither: **{neither}**")
        lines.append("")

        # Oracle ensemble: best of any model per task (EI-filtered)
        lines.append(f"### Oracle Ensemble (best of ALL {len(models)} models per task, EI-filtered)")
        lines.append("")
        oracle_scores = []
        for task in ei_tasks:
            best = max(model_mean_ei[k].get(task, 0.0) for k in models)
            oracle_scores.append(best)
        oracle_mean = np.mean(oracle_scores) if oracle_scores else 0.0
        oracle_ge_half = sum(1 for s in oracle_scores if s >= 0.5)
        oracle_eq_one = sum(1 for s in oracle_scores if s == 1.0)

        lines.append(f"- Oracle mean: **{oracle_mean:.3f}**")
        lines.append(f"- Oracle >=0.5: **{oracle_ge_half}/{len(ei_tasks)}**")
        lines.append(f"- Oracle =1.0: **{oracle_eq_one}/{len(ei_tasks)}**")

        # Compare oracle to best single model
        best_single_key = max(models.keys(), key=lambda k: np.mean([model_mean_ei[k].get(t, 0.0) for t in ei_tasks]))
        best_single_mean = np.mean([model_mean_ei[best_single_key].get(t, 0.0) for t in ei_tasks])
        lines.append(f"- Best single model: **{models[best_single_key]['short']}** ({best_single_mean:.3f})")
        lines.append(f"- Oracle lift over best single: **{oracle_mean - best_single_mean:+.3f}**")
        lines.append("")

    # ── 7. Per-task detail (high variance, EI-filtered) ──
    lines.append(f"## 7. Highest-Variance Tasks — EI-filtered ({len(ei_tasks)} tasks)")
    lines.append("")

    task_var = []
    for task in ei_tasks:
        scores = [model_mean_ei[k].get(task, 0.0) for k in models]
        task_var.append((task, np.var(scores), scores))
    task_var.sort(key=lambda x: -x[1])

    header = "| Task | " + " | ".join(models[k]["short"] for k in models) + " | Var |"
    lines.append(header)
    lines.append("|" + "|".join(["---"] * (len(models) + 2)) + "|")
    for task, var, scores in task_var[:25]:
        task_short = task[:28]
        score_strs = [f" {s:.2f} " for s in scores]
        lines.append(f"| `{task_short}` |" + "|".join(score_strs) + f"| {var:.3f} |")
    lines.append("")

    # ── 8. Skill ceiling analysis (EI-filtered) ──
    lines.append(f"## 8. Skill Ceiling Analysis (EI-filtered, {len(ei_tasks)} tasks)")
    lines.append("")

    solved_by: dict[str, set[str]] = {}
    for key in models:
        solved_by[key] = {t for t in ei_tasks if model_mean_ei[key].get(t, 0.0) >= 0.5}

    all_solved = set.union(*solved_by.values()) if solved_by else set()
    all_unsolved = set(ei_tasks) - all_solved

    lines.append(f"- Tasks solved by at least one model: **{len(all_solved)}/{len(ei_tasks)}**")
    lines.append(f"- Tasks unsolved by all models: **{len(all_unsolved)}/{len(ei_tasks)}**")
    lines.append("")

    solve_count = Counter()
    for task in ei_tasks:
        n = sum(1 for k in models if model_mean_ei[k].get(task, 0.0) >= 0.5)
        solve_count[n] += 1

    lines.append("### Distribution of Solve Counts")
    lines.append("")
    lines.append("| # Models Solving | # Tasks |")
    lines.append("|-----------------|---------|")
    for n in range(len(models) + 1):
        lines.append(f"| {n} | {solve_count.get(n, 0)} |")
    lines.append("")

    lines.append("### Unique Contributions")
    lines.append("")
    lines.append("| Model | Total Solved | Uniquely Solved | Example Unique Tasks |")
    lines.append("|-------|-------------|----------------|---------------------|")
    for key in models:
        my_solved = solved_by[key]
        others_solved = set.union(*(solved_by[k] for k in models if k != key)) if len(models) > 1 else set()
        unique = my_solved - others_solved
        examples = ", ".join(f"`{t[:20]}`" for t in sorted(unique)[:4])
        lines.append(f"| {models[key]['short']} | {len(my_solved)} | {len(unique)} | {examples or '-'} |")
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="Method ablation comparison")
    parser.add_argument(
        "--output",
        default="/Users/benjaminfeuer/Documents/notes/ot-agent/method_ablation.md",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional JSON file with model definitions (overrides defaults)",
    )
    args = parser.parse_args()

    if args.config:
        models = json.loads(Path(args.config).read_text())
    else:
        models = DEFAULT_MODELS

    datasets = load_all_datasets(models)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = format_report(models, datasets, output_path)
    print(f"\nReport written to {output_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
