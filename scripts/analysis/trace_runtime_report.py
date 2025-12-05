#!/usr/bin/env python3
"""Aggregate runtime statistics from eval trace outputs and plot comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib.pyplot as plt


TimeUnit = str

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
_CANDIDATE_SIBLING_ROOT = REPO_ROOT.parent / "evaltraces"
_CANDIDATE_LOCAL_ROOT = REPO_ROOT / "evaltraces"
if _CANDIDATE_SIBLING_ROOT.exists():
    DEFAULT_TRACE_ROOT = _CANDIDATE_SIBLING_ROOT
elif _CANDIDATE_LOCAL_ROOT.exists():
    DEFAULT_TRACE_ROOT = _CANDIDATE_LOCAL_ROOT
else:
    DEFAULT_TRACE_ROOT = _CANDIDATE_SIBLING_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan experiment traces under a root directory (default: evaltraces), "
            "collect job and task runtimes, export a master JSON summary, and "
            "generate an overlapping bar plot."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_TRACE_ROOT,
        help=(
            "Root directory containing experiment subdirectories. "
            f"(default: {DEFAULT_TRACE_ROOT})"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Target path for the summary JSON. Defaults to <root>/trace_runtime_summary.json."
        ),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the runtime comparison plot. Defaults to "
            "<root>/trace_runtime_summary.png."
        ),
    )
    parser.add_argument(
        "--output-task-boxplot",
        type=Path,
        default=None,
        help=(
            "Target path for the single-task runtime boxplot. Defaults to "
            "<root>/trace_runtime_task_boxplot.png."
        ),
    )
    parser.add_argument(
        "--output-runtime-correlation-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the runtime vs. accuracy correlation plot. Defaults to "
            "<root>/trace_runtime_task_accuracy_correlation.png."
        ),
    )
    parser.add_argument(
        "--output-trace-length-correlation-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the trace-length vs. accuracy correlation plot. Defaults to "
            "<root>/trace_length_task_accuracy_correlation.png."
        ),
    )
    parser.add_argument(
        "--output-prompt-response-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the average prompt/response length plot. Defaults to "
            "<root>/trace_prompt_response_lengths.png."
        ),
    )
    parser.add_argument(
        "--output-overall-prompt-response-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the overall average prompt/response length plot. Defaults to "
            "<root>/trace_prompt_response_lengths_overall.png."
        ),
    )
    parser.add_argument(
        "--output-task-prompt-response-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the per-task prompt/response length plot. Defaults to "
            "<root>/trace_prompt_response_lengths_by_task.png."
        ),
    )
    parser.add_argument(
        "--output-response-commands",
        type=Path,
        default=None,
        help=(
            "Path to write aggregated response command JSON. Defaults to "
            "<root>/trace_response_commands.json."
        ),
    )
    parser.add_argument(
        "--flat-structure",
        action="store_true",
        help=(
            "Expect job directories directly under each experiment directory "
            "instead of under <experiment>/trace_jobs/."
        ),
    )
    parser.add_argument(
        "--output-keystroke-frequency",
        type=Path,
        default=None,
        help=(
            "Path to write keystroke term frequency JSON. Defaults to "
            "<root>/trace_keystroke_frequencies.json."
        ),
    )
    parser.add_argument(
        "--output-task-solvability-figure",
        type=Path,
        default=None,
        help=(
            "Target path for the per-task solvability plot. Defaults to "
            "<root>/trace_task_solvability.png."
        ),
    )
    parser.add_argument(
        "--output-correlation-table",
        type=Path,
        default=None,
        help=(
            "Target path for the episode/runtime correlation table (CSV). "
            "Defaults to <root>/trace_runtime_episode_correlation.csv."
        ),
    )
    parser.add_argument(
        "--time-unit",
        choices=("seconds", "minutes", "hours"),
        default="minutes",
        help="Unit to display in the visualization (seconds, minutes, or hours).",
    )
    return parser.parse_args()


def iso_diff_seconds(start: Optional[str], finish: Optional[str]) -> Optional[float]:
    """Parse ISO timestamps and return the positive runtime in seconds."""
    if not start or not finish:
        return None
    if isinstance(start, str) and start.endswith("Z"):
        start = start[:-1] + "+00:00"
    if isinstance(finish, str) and finish.endswith("Z"):
        finish = finish[:-1] + "+00:00"
    try:
        start_dt = datetime.fromisoformat(start)
        finish_dt = datetime.fromisoformat(finish)
    except ValueError:
        return None
    diff = (finish_dt - start_dt).total_seconds()
    if diff <= 0:
        return None
    return diff


def average_top_fraction(values: List[float], fraction: float) -> Optional[float]:
    """Average of the top fraction of values (e.g., top 10%)."""
    if not values:
        return None
    if fraction <= 0:
        return None
    sorted_values = sorted(values, reverse=True)
    count = max(1, ceil(len(sorted_values) * fraction))
    top_slice = sorted_values[:count]
    if not top_slice:
        return None
    return sum(top_slice) / len(top_slice)


def percentile_value(values: List[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if percentile <= 0:
        return min(values)
    if percentile >= 1:
        return max(values)
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    position = (n - 1) * percentile
    lower_index = floor(position)
    upper_index = ceil(position)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = position - lower_index
    return lower_value * (1 - weight) + upper_value * weight


def coerce_int(value: object) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def extract_json_object(text: str) -> Optional[Dict]:
    """Extract first JSON object from text that may contain leading commentary."""
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if not stripped:
        return None
    for idx, char in enumerate(stripped):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(stripped[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


@dataclass
class JobRuntime:
    job_name: str
    runtime_seconds: Optional[float]
    task_runtimes_seconds: List[float] = field(default_factory=list)
    task_runs: List["TaskRun"] = field(default_factory=list)
    agent_timeout_total_count: int = 0
    agent_timeout_runtime_samples: List[float] = field(default_factory=list)
    prompt_char_total: int = 0
    prompt_count_total: int = 0
    response_char_total: int = 0
    response_count_total: int = 0
    positive_rewards_by_task: Dict[str, int] = field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.task_runtimes_seconds)

    @property
    def agent_timeout_average_runtime(self) -> Optional[float]:
        if not self.agent_timeout_runtime_samples:
            return None
        return sum(self.agent_timeout_runtime_samples) / len(
            self.agent_timeout_runtime_samples
        )

    @property
    def average_prompt_characters(self) -> Optional[float]:
        if self.prompt_count_total == 0:
            return None
        return self.prompt_char_total / self.prompt_count_total

    @property
    def average_response_characters(self) -> Optional[float]:
        if self.response_count_total == 0:
            return None
        return self.response_char_total / self.response_count_total


@dataclass
class TaskRun:
    task_name: str
    runtime_seconds: float
    episode_count: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    is_success: Optional[bool] = None
    prompt_char_total: int = 0
    prompt_count: int = 0
    response_char_total: int = 0
    response_count: int = 0
    average_prompt_length: Optional[float] = None
    average_response_length: Optional[float] = None


def _rank_values(values: List[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        value = indexed[i][1]
        while j < len(indexed) and indexed[j][1] == value:
            j += 1
        rank = (i + j - 1) / 2 + 1  # average rank (1-based)
        for k in range(i, j):
            original_index = indexed[k][0]
            ranks[original_index] = rank
        i = j
    return ranks


def spearman_rho(pairs: List[Tuple[float, float]]) -> Optional[float]:
    if len(pairs) < 2:
        return None
    x_values = [pair[0] for pair in pairs]
    y_values = [pair[1] for pair in pairs]
    rank_x = _rank_values(x_values)
    rank_y = _rank_values(y_values)
    n = len(pairs)
    mean_rank_x = sum(rank_x) / n
    mean_rank_y = sum(rank_y) / n
    cov = sum(
        (rx - mean_rank_x) * (ry - mean_rank_y)
        for rx, ry in zip(rank_x, rank_y)
    )
    std_x = sqrt(sum((rx - mean_rank_x) ** 2 for rx in rank_x))
    std_y = sqrt(sum((ry - mean_rank_y) ** 2 for ry in rank_y))
    if std_x == 0 or std_y == 0:
        return None
    return cov / (std_x * std_y)


@dataclass
class ExperimentRuntime:
    name: str
    jobs: List[JobRuntime]

    def as_summary(self) -> Dict[str, object]:
        all_task_runtimes = [
            seconds for job in self.jobs for seconds in job.task_runtimes_seconds
        ]
        overall_runtime_seconds: Optional[float] = None
        job_runtimes = [job.runtime_seconds for job in self.jobs if job.runtime_seconds]
        if job_runtimes:
            overall_runtime_seconds = max(job_runtimes)

        max_task_runtime_seconds = max(all_task_runtimes) if all_task_runtimes else None
        avg_task_runtime_seconds: Optional[float] = None
        avg_top_10_task_runtime_seconds = None
        avg_top_25_task_runtime_seconds = None
        if all_task_runtimes:
            avg_task_runtime_seconds = sum(all_task_runtimes) / len(all_task_runtimes)
            avg_top_10_task_runtime_seconds = average_top_fraction(
                all_task_runtimes, 0.10
            )
            avg_top_25_task_runtime_seconds = average_top_fraction(
                all_task_runtimes, 0.25
            )

        timeout_total_count = sum(
            job.agent_timeout_total_count for job in self.jobs
        )
        timeout_runtimes = [
            sample
            for job in self.jobs
            for sample in job.agent_timeout_runtime_samples
        ]
        timeout_average_runtime = (
            sum(timeout_runtimes) / len(timeout_runtimes)
            if timeout_runtimes
            else None
        )

        prompt_char_total = sum(job.prompt_char_total for job in self.jobs)
        prompt_count_total = sum(job.prompt_count_total for job in self.jobs)
        response_char_total = sum(job.response_char_total for job in self.jobs)
        response_count_total = sum(job.response_count_total for job in self.jobs)
        average_prompt_chars = (
            prompt_char_total / prompt_count_total
            if prompt_count_total
            else None
        )
        average_response_chars = (
            response_char_total / response_count_total
            if response_count_total
            else None
        )

        return {
            "experiment": self.name,
            "job_count": len(self.jobs),
            "task_count": len(all_task_runtimes),
            "overall_runtime_seconds": overall_runtime_seconds,
            "max_task_runtime_seconds": max_task_runtime_seconds,
            "avg_task_runtime_seconds": avg_task_runtime_seconds,
            "avg_top_10_task_runtime_seconds": avg_top_10_task_runtime_seconds,
            "avg_top_25_task_runtime_seconds": avg_top_25_task_runtime_seconds,
            "agent_timeout_error_count": timeout_total_count,
            "agent_timeout_average_runtime": timeout_average_runtime,
            "average_prompt_characters": average_prompt_chars,
            "average_response_characters": average_response_chars,
            "jobs": [
                {
                    "job_name": job.job_name,
                    "runtime_seconds": job.runtime_seconds,
                    "task_count": job.task_count,
                    "task_runtimes_seconds": job.task_runtimes_seconds,
                    "task_details": [
                        {
                            "task_name": task_run.task_name,
                            "runtime_seconds": task_run.runtime_seconds,
                            "episode_count": task_run.episode_count,
                            "n_input_tokens": task_run.input_tokens,
                            "n_output_tokens": task_run.output_tokens,
                            "is_success": task_run.is_success,
                            "prompt_char_total": task_run.prompt_char_total,
                            "prompt_count": task_run.prompt_count,
                            "response_char_total": task_run.response_char_total,
                            "response_count": task_run.response_count,
                        }
                        for task_run in job.task_runs
                    ],
                    "agent_timeout_error_count": job.agent_timeout_total_count,
                    "agent_timeout_average_runtime": job.agent_timeout_average_runtime,
                    "prompt_char_total": job.prompt_char_total,
                    "prompt_count": job.prompt_count_total,
                    "average_prompt_characters": job.average_prompt_characters,
                    "response_char_total": job.response_char_total,
                    "response_count": job.response_count_total,
                    "average_response_characters": job.average_response_characters,
                }
                for job in self.jobs
            ],
        }


def load_job(
    job_dir: Path,
    *,
    response_records: Optional[List[Dict[str, object]]] = None,
    experiment_name: Optional[str] = None,
    keystroke_terms: Optional[Dict[str, Counter]] = None,
    solvability_records: Optional[Dict[str, Dict[str, bool]]] = None,
) -> JobRuntime:
    """Collect runtime information from a single job directory."""
    job_result_path = job_dir / "result.json"
    job_payload_data: Optional[Dict[str, object]] = None
    job_runtime_seconds: Optional[float] = None
    positive_trials: set[str] = set()
    pos_reward_counts: Dict[str, int] = {}
    pos_reward_counts: Dict[str, int] = {}
    if job_result_path.is_file():
        try:
            with job_result_path.open("r", encoding="utf-8") as handle:
                job_payload_data = json.load(handle)
            job_payload = job_payload_data
            job_runtime_seconds = iso_diff_seconds(
                job_payload.get("started_at"), job_payload.get("finished_at")
            )
            task_name_value = job_payload.get("task_name")
            if isinstance(task_name_value, str) and task_name_value:
                job_task_name = task_name_value.strip()
            stats = job_payload.get("stats")
            if isinstance(stats, dict):
                positive_reward_trials = stats.get("positive_reward_trials")
                if isinstance(positive_reward_trials, list):
                    for entry in positive_reward_trials:
                        if isinstance(entry, str):
                            trimmed = entry.strip()
                            if trimmed:
                                positive_trials.add(trimmed)
                                prefix = trimmed.split("__", 1)[0]
                                prefix = prefix.strip()
                                if prefix:
                                    pos_reward_counts[prefix] = pos_reward_counts.get(prefix, 0) + 1
        except (OSError, json.JSONDecodeError):
            job_runtime_seconds = None

    task_runtimes: List[float] = []
    task_runtime_by_dir: Dict[str, float] = {}
    task_runs: List[TaskRun] = []
    job_prompt_char_total = 0
    job_prompt_count_total = 0
    job_response_char_total = 0
    job_response_count_total = 0
    job_positive_reward_counts: Dict[str, int] = dict(pos_reward_counts)
    def process_task_directory(task_dir: Path, task_payload: Dict[str, object]) -> None:
        nonlocal job_prompt_char_total
        nonlocal job_prompt_count_total
        nonlocal job_response_char_total
        nonlocal job_response_count_total
        agent_execution = task_payload.get("agent_execution") or {}
        runtime_seconds = iso_diff_seconds(
            agent_execution.get("started_at"), agent_execution.get("finished_at")
        )
        if runtime_seconds is not None:
            task_runtimes.append(runtime_seconds)
            task_runtime_by_dir[task_dir.name] = runtime_seconds
            task_name = task_payload.get("task_name")
            if not isinstance(task_name, str) or not task_name.strip():
                task_name = task_dir.name.split("__", 1)[0]
            task_name = task_name.strip()
            episode_count = 0
            prompt_char_total = 0
            prompt_count = 0
            response_char_total = 0
            response_count = 0
            child_task_complete = False
            agent_dir = task_dir / "agent"
            if agent_dir.is_dir():
                for episode_dir in agent_dir.iterdir():
                    if not episode_dir.is_dir():
                        continue
                    try:
                        episode_num = int(episode_dir.name.split("-")[-1])
                    except ValueError:
                        continue
                    episode_count += 1
                    prompt_path = episode_dir / "prompt.txt"
                    if prompt_path.is_file():
                        try:
                            prompt_text = prompt_path.read_text(encoding="utf-8", errors="ignore")
                            prompt_char_total += len(prompt_text)
                            prompt_count += 1
                        except OSError:
                            pass
                    response_path = episode_dir / "response.txt"
                    if response_path.is_file():
                        try:
                            response_text = response_path.read_text(encoding="utf-8", errors="ignore")
                            response_char_total += len(response_text)
                            response_count += 1
                            if response_records is not None and experiment_name:
                                parsed_response = extract_json_object(response_text)
                                if isinstance(parsed_response, dict):
                                    response_records.append(
                                        {
                                            "experiment": experiment_name,
                                            "job": job_dir.name,
                                            "task_name": task_name,
                                            "episode": episode_dir.name,
                                            "response": parsed_response,
                                        }
                                    )
                                    if parsed_response.get("task_complete") is True:
                                        child_task_complete = True
                                    if keystroke_terms is not None:
                                        commands = parsed_response.get("commands")
                                        if isinstance(commands, list):
                                            for command in commands:
                                                if isinstance(command, dict):
                                                    keystrokes = command.get("keystrokes")
                                                    if isinstance(keystrokes, str):
                                                        tokens = [token.strip() for token in keystrokes.split()]
                                                    else:
                                                        continue
                                                elif isinstance(command, str):
                                                    tokens = [token.strip() for token in command.split()]
                                                else:
                                                    continue
                                                for token in tokens:
                                                    if not token:
                                                        continue
                                                    keystroke_terms.setdefault("all", Counter())[token] += 1
                                                    if experiment_name:
                                                        lower_name = experiment_name.lower()
                                                        if experiment_name.startswith("claude-") or lower_name.startswith("gpt-"):
                                                            keystroke_terms.setdefault("claude_gpt", Counter())[token] += 1
                                                        if experiment_name.startswith("Qwen3-8B") or lower_name.startswith("qwen3-8b"):
                                                            keystroke_terms.setdefault("qwen3_8b", Counter())[token] += 1
                        except OSError:
                            pass
            agent_result = task_payload.get("agent_result") or {}
            input_tokens = coerce_int(agent_result.get("n_input_tokens"))
            output_tokens = coerce_int(agent_result.get("n_output_tokens"))
            verifier_result = task_payload.get("verifier_result") or {}
            reward_success: Optional[bool] = None
            reward_values_available = False
            if isinstance(verifier_result, dict):
                rewards_payload = verifier_result.get("rewards")
                if isinstance(rewards_payload, dict):
                    numeric_values: List[float] = []
                    for value in rewards_payload.values():
                        numeric_value: Optional[float] = None
                        if isinstance(value, bool):
                            numeric_value = float(int(value))
                        elif isinstance(value, (int, float)):
                            numeric_value = float(value)
                        elif isinstance(value, str):
                            stripped = value.strip()
                            if stripped:
                                try:
                                    numeric_value = float(stripped)
                                except ValueError:
                                    numeric_value = None
                        if numeric_value is not None:
                            numeric_values.append(numeric_value)
                    if numeric_values:
                        reward_success = any(val >= 0.1 for val in numeric_values)
                        reward_values_available = True

            if not reward_values_available:
                print(
                    f"[trace_runtime_report] Warning: No usable verifier reward found for "
                    f"{experiment_name}/{job_dir.name}/{task_dir.name}; marking as failure.",
                    file=sys.stderr,
                )
            success_flag: Optional[bool] = reward_success if reward_success is not None else False
            task_completed = bool(success_flag)
            average_prompt_length = (
                prompt_char_total / prompt_count if prompt_count else None
            )
            average_response_length = (
                response_char_total / response_count if response_count else None
            )
            task_runs.append(
                TaskRun(
                    task_name=task_name,
                    runtime_seconds=runtime_seconds,
                    episode_count=episode_count,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    is_success=success_flag if success_flag is not None else False,
                    prompt_char_total=prompt_char_total,
                    prompt_count=prompt_count,
                    response_char_total=response_char_total,
                    response_count=response_count,
                    average_prompt_length=average_prompt_length,
                    average_response_length=average_response_length,
                )
            )
            job_prompt_char_total += prompt_char_total
            job_prompt_count_total += prompt_count
            job_response_char_total += response_char_total
            job_response_count_total += response_count
            if (
                solvability_records is not None
                and experiment_name
                and task_name
            ):
                entry = solvability_records.setdefault(task_name, {})
                entry.setdefault(experiment_name, False)
                entry[experiment_name] = entry[experiment_name] or task_completed
            if task_completed and pos_reward_counts:
                prefix = task_name
                if prefix:
                    job_positive_reward_counts[prefix] = job_positive_reward_counts.get(prefix, 0) + pos_reward_counts.get(prefix, 0)

    for child in job_dir.iterdir():
        if not child.is_dir():
            continue
        task_result_path = child / "result.json"
        if not task_result_path.is_file():
            continue
        try:
            with task_result_path.open("r", encoding="utf-8") as handle:
                task_payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        process_task_directory(child, task_payload)

    if not task_runs and job_payload_data:
        process_task_directory(job_dir, job_payload_data)

    timeout_total_count = 0
    timeout_runtime_samples: List[float] = []
    if job_payload_data:
        stats = job_payload_data.get("stats")
        if isinstance(stats, dict):
            exception_stats = stats.get("exception_stats")
            if isinstance(exception_stats, dict):
                timeout_entries = exception_stats.get("AgentTimeoutError")
                if isinstance(timeout_entries, list):
                    timeout_total_count = len(timeout_entries)
                    for entry in timeout_entries:
                        if not isinstance(entry, str):
                            continue
                        runtime_value = task_runtime_by_dir.get(entry)
                        if runtime_value is not None:
                            timeout_runtime_samples.append(runtime_value)

    return JobRuntime(
        job_name=job_dir.name,
        runtime_seconds=job_runtime_seconds,
        task_runtimes_seconds=task_runtimes,
        task_runs=task_runs,
        agent_timeout_total_count=timeout_total_count,
        agent_timeout_runtime_samples=timeout_runtime_samples,
        prompt_char_total=job_prompt_char_total,
        prompt_count_total=job_prompt_count_total,
        response_char_total=job_response_char_total,
        response_count_total=job_response_count_total,
        positive_rewards_by_task=job_positive_reward_counts,
    )


def discover_experiments(
    root: Path,
    *,
    response_records: Optional[List[Dict[str, object]]] = None,
    keystroke_terms: Optional[Dict[str, Counter]] = None,
    solvability_records: Optional[Dict[str, Dict[str, bool]]] = None,
    flat_structure: bool = False,
) -> List[ExperimentRuntime]:
    experiments: List[ExperimentRuntime] = []
    for experiment_dir in sorted(root.iterdir()):
        if not experiment_dir.is_dir():
            continue
        job_parent_dirs: List[Path] = []
        if flat_structure:
            job_parent_dirs.append(experiment_dir)
        else:
            trace_jobs_dir = experiment_dir / "trace_jobs"
            if trace_jobs_dir.is_dir():
                job_parent_dirs.append(trace_jobs_dir)
        if not job_parent_dirs:
            continue

        jobs: List[JobRuntime] = []
        for job_parent in job_parent_dirs:
            for job_dir in sorted(job_parent.iterdir()):
                if not job_dir.is_dir():
                    continue
                jobs.append(
                    load_job(
                        job_dir,
                        response_records=response_records,
                        experiment_name=experiment_dir.name,
                        keystroke_terms=keystroke_terms,
                        solvability_records=solvability_records,
                    )
                )
        if jobs:
            experiments.append(ExperimentRuntime(name=experiment_dir.name, jobs=jobs))
    return experiments


def aggregate_by_metric(values: List[float], metric: str) -> Optional[float]:
    if not values:
        return None
    if metric == "overall_average":
        return sum(values) / len(values)
    if metric == "percentile_75":
        return percentile_value(values, 0.75)
    if metric == "percentile_90":
        return percentile_value(values, 0.90)
    if metric == "percentile_99":
        return percentile_value(values, 0.99)
    raise ValueError(f"Unknown metric '{metric}'")


def compute_correlations_for_feature(
    runtime_by_task: Dict[str, List[float]],
    feature_by_task: Dict[str, List[float]],
    metrics: List[str],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for metric in metrics:
        pairs: List[Tuple[float, float]] = []
        for task_name, feature_values in feature_by_task.items():
            if not feature_values:
                continue
            runtime_values = runtime_by_task.get(task_name)
            if not runtime_values:
                continue
            runtime_metric = aggregate_by_metric(runtime_values, metric)
            feature_metric = aggregate_by_metric(feature_values, metric)
            if runtime_metric is None or feature_metric is None:
                continue
            pairs.append((feature_metric, runtime_metric))
        rho = spearman_rho(pairs)
        results.append(
            {
                "metric": metric,
                "spearman_rho": rho,
                "pair_count": len(pairs),
            }
        )
    return results


def compute_runtime_correlations(
    experiments: List[ExperimentRuntime],
) -> Dict[str, List[Dict[str, object]]]:
    runtime_by_task: Dict[str, List[float]] = defaultdict(list)
    feature_maps: Dict[str, Dict[str, List[float]]] = {
        "episode_count": defaultdict(list),
        "n_input_tokens": defaultdict(list),
        "n_output_tokens": defaultdict(list),
    }

    for experiment in experiments:
        for job in experiment.jobs:
            for task_run in job.task_runs:
                runtime_by_task[task_run.task_name].append(task_run.runtime_seconds)
                feature_maps["episode_count"][task_run.task_name].append(task_run.episode_count)
                if task_run.input_tokens is not None:
                    feature_maps["n_input_tokens"][task_run.task_name].append(
                        float(task_run.input_tokens)
                    )
                if task_run.output_tokens is not None:
                    feature_maps["n_output_tokens"][task_run.task_name].append(
                        float(task_run.output_tokens)
                    )

    metrics = [
        "overall_average",
        "percentile_75",
        "percentile_90",
        "percentile_99",
    ]

    correlation_map: Dict[str, List[Dict[str, object]]] = {}
    for feature_name, feature_by_task in feature_maps.items():
        correlation_map[feature_name] = compute_correlations_for_feature(
            runtime_by_task, feature_by_task, metrics
        )
    return correlation_map


def compute_feature_accuracy_correlation(
    experiments: List[ExperimentRuntime],
    feature: str,
) -> Dict[str, Optional[Dict[str, object]]]:
    def feature_value(task_run: TaskRun) -> Optional[float]:
        if feature == "runtime":
            return float(task_run.runtime_seconds)
        if feature == "prompt":
            return task_run.average_prompt_length
        if feature == "response":
            return task_run.average_response_length
        if feature == "episodes":
            return float(task_run.episode_count)
        raise ValueError(f"Unsupported feature for correlation: {feature}")

    samples: List[Tuple[float, float]] = []
    for experiment in experiments:
        for job in experiment.jobs:
            for task_run in job.task_runs:
                value = feature_value(task_run)
                if value is None:
                    continue
                if task_run.is_success is None:
                    continue
                samples.append(
                    (
                        value,
                        1.0 if task_run.is_success else 0.0,
                    )
                )

    if not samples:
        return {}

    feature_values = [value for value, _ in samples]
    metrics = {
        "overall": ("ge", None),
        "percentile_75": ("ge", 0.75),
        "percentile_90": ("ge", 0.90),
        "percentile_25": ("ge", 0.25),
        "percentile_10": ("ge", 0.10),
        "percentile_low_25": ("le", 0.25),
        "percentile_low_10": ("le", 0.10),
    }

    results: Dict[str, Optional[Dict[str, object]]] = {}
    for label, (mode, percentile) in metrics.items():
        if percentile is None:
            selected = samples
        else:
            threshold = percentile_value(feature_values, percentile)
            if threshold is None:
                selected = []
            else:
                if mode == "ge":
                    selected = [
                        pair for pair in samples if pair[0] >= threshold
                    ]
                else:
                    selected = [
                        pair for pair in samples if pair[0] <= threshold
                    ]
        rho = spearman_rho(selected)
        results[label] = {
            "spearman_rho": rho,
            "pair_count": len(selected),
        }
    return results


def compute_per_task_accuracy_correlations(
    experiments: List[ExperimentRuntime],
    feature: str,
) -> List[Dict[str, object]]:
    def feature_value(task_run: TaskRun) -> Optional[float]:
        if feature == "runtime":
            return float(task_run.runtime_seconds)
        if feature == "episodes":
            return float(task_run.episode_count)
        if feature == "prompt":
            return task_run.average_prompt_length
        if feature == "response":
            return task_run.average_response_length
        raise ValueError(f"Unsupported feature for correlation: {feature}")

    pairs_by_task: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for experiment in experiments:
        for job in experiment.jobs:
            for task_run in job.task_runs:
                value = feature_value(task_run)
                if value is None:
                    continue
                if task_run.is_success is None:
                    continue
                pairs_by_task[task_run.task_name].append(
                    (value, 1.0 if task_run.is_success else 0.0)
                )

    task_entries: List[Dict[str, object]] = []
    for task_name, pairs in pairs_by_task.items():
        rho = spearman_rho(pairs)
        task_entries.append(
            {
                "task_name": task_name,
                "spearman_rho": rho,
                "pair_count": len(pairs),
            }
        )
    task_entries.sort(
        key=lambda entry: (
            entry["spearman_rho"] if entry["spearman_rho"] is not None else 0.0
        ),
        reverse=True,
    )
    return task_entries


def compute_per_experiment_feature_correlations(
    experiments: List[ExperimentRuntime],
    feature: str,
) -> List[Dict[str, object]]:
    def feature_value(task_run: TaskRun) -> Optional[float]:
        if feature == "prompt":
            return task_run.average_prompt_length
        if feature == "response":
            return task_run.average_response_length
        if feature == "runtime":
            return float(task_run.runtime_seconds)
        if feature == "episodes":
            return float(task_run.episode_count)
        raise ValueError(f"Unsupported feature for correlation: {feature}")

    results: List[Dict[str, object]] = []
    for experiment in experiments:
        pairs: List[Tuple[float, float]] = []
        for job in experiment.jobs:
            for task_run in job.task_runs:
                value = feature_value(task_run)
                if value is None or task_run.is_success is None:
                    continue
                pairs.append((value, 1.0 if task_run.is_success else 0.0))
        rho = spearman_rho(pairs)
        results.append(
            {
                "experiment": experiment.name,
                "spearman_rho": rho,
                "pair_count": len(pairs),
            }
        )
    results.sort(
        key=lambda entry: entry["spearman_rho"]
        if entry["spearman_rho"] is not None
        else 0.0,
        reverse=True,
    )
    return results


def aggregate_prompt_response_stats(
    experiments: List[ExperimentRuntime],
) -> Dict[str, object]:
    total_prompt_chars = 0
    total_prompt_count = 0
    total_response_chars = 0
    total_response_count = 0
    per_task: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "prompt_chars": 0.0,
            "prompt_count": 0.0,
            "response_chars": 0.0,
            "response_count": 0.0,
        }
    )
    positive_reward_counts: Dict[str, int] = defaultdict(int)

    for experiment in experiments:
        for job in experiment.jobs:
            total_prompt_chars += job.prompt_char_total
            total_prompt_count += job.prompt_count_total
            total_response_chars += job.response_char_total
            total_response_count += job.response_count_total
            solver = getattr(job, "positive_rewards_by_task", {})
            for task_id, count in solver.items():
                positive_reward_counts[task_id] += count
            for task_run in job.task_runs:
                stats = per_task[task_run.task_name]
                stats["prompt_chars"] += task_run.prompt_char_total
                stats["prompt_count"] += task_run.prompt_count
                stats["response_chars"] += task_run.response_char_total
                stats["response_count"] += task_run.response_count

    per_task_list: List[Dict[str, object]] = []
    for task_name, stats in per_task.items():
        prompt_count = stats["prompt_count"]
        response_count = stats["response_count"]
        per_task_list.append(
            {
                "task_name": task_name,
                "average_prompt_characters": (
                    stats["prompt_chars"] / prompt_count if prompt_count else None
                ),
                "average_response_characters": (
                    stats["response_chars"] / response_count if response_count else None
                ),
                "prompt_count": int(prompt_count),
                "response_count": int(response_count),
                "positive_reward_trials": positive_reward_counts.get(task_name, 0),
            }
        )

    per_task_list.sort(
        key=lambda entry: (
            entry.get("average_prompt_characters") or 0.0,
            entry.get("average_response_characters") or 0.0,
        ),
        reverse=True,
    )

    return {
        "total_prompt_chars": total_prompt_chars,
        "total_prompt_count": total_prompt_count,
        "total_response_chars": total_response_chars,
        "total_response_count": total_response_count,
        "per_task": per_task_list,
    }


def write_correlation_table(
    correlation_map: Dict[str, List[Dict[str, object]]],
    per_task_runtime_corr: List[Dict[str, object]],
    per_task_trace_corr: List[Dict[str, object]],
    per_task_prompt_corr: List[Dict[str, object]],
    per_task_response_corr: List[Dict[str, object]],
    prompt_length_corr: Dict[str, Optional[Dict[str, object]]],
    response_length_corr: Dict[str, Optional[Dict[str, object]]],
    per_experiment_prompt_corr: List[Dict[str, object]],
    per_experiment_response_corr: List[Dict[str, object]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "metric", "spearman_rho", "pair_count"])
        for feature, entries in correlation_map.items():
            for entry in entries:
                rho = entry.get("spearman_rho")
                writer.writerow(
                    [
                        feature,
                        entry.get("metric"),
                        f"{rho:.6f}" if isinstance(rho, float) else "",
                        entry.get("pair_count", 0),
                    ]
                )
        writer.writerow([])
        writer.writerow(["feature", "metric", "spearman_rho", "pair_count"])
        for metric, entry in (prompt_length_corr or {}).items():
            if not isinstance(entry, dict):
                continue
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    "prompt_length",
                    metric,
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["feature", "metric", "spearman_rho", "pair_count"])
        for metric, entry in (response_length_corr or {}).items():
            if not isinstance(entry, dict):
                continue
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    "response_length",
                    metric,
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["task_name (runtime)", "spearman_rho", "pair_count"])
        for entry in per_task_runtime_corr:
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    entry.get("task_name"),
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["task_name (trace_length)", "spearman_rho", "pair_count"])
        for entry in per_task_trace_corr:
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    entry.get("task_name"),
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["task_name (prompt_length)", "spearman_rho", "pair_count"])
        for entry in per_task_prompt_corr:
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    entry.get("task_name"),
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["task_name (response_length)", "spearman_rho", "pair_count"])
        for entry in per_task_response_corr:
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    entry.get("task_name"),
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["experiment (prompt_length)", "spearman_rho", "pair_count"])
        for entry in per_experiment_prompt_corr:
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    entry.get("experiment"),
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )
        writer.writerow([])
        writer.writerow(["experiment (response_length)", "spearman_rho", "pair_count"])
        for entry in per_experiment_response_corr:
            rho = entry.get("spearman_rho")
            writer.writerow(
                [
                    entry.get("experiment"),
                    f"{rho:.6f}" if isinstance(rho, float) else "",
                    entry.get("pair_count", 0),
                ]
            )


def build_summary(
    experiments: List[ExperimentRuntime],
    generated_at: str,
    correlation_map: Dict[str, List[Dict[str, object]]],
    runtime_accuracy_correlation: Dict[str, Optional[Dict[str, object]]],
    per_task_runtime_correlations: List[Dict[str, object]],
    per_task_trace_correlations: List[Dict[str, object]],
    prompt_stats: Dict[str, object],
    prompt_length_accuracy_correlation: Dict[str, Optional[Dict[str, object]]],
    response_length_accuracy_correlation: Dict[str, Optional[Dict[str, object]]],
    per_task_prompt_correlations: List[Dict[str, object]],
    per_task_response_correlations: List[Dict[str, object]],
    per_experiment_prompt_correlations: List[Dict[str, object]],
    per_experiment_response_correlations: List[Dict[str, object]],
    task_solvability: List[Dict[str, object]],
) -> Dict[str, object]:
    timeout_total_count = 0
    timeout_runtime_sum = 0.0
    timeout_sample_count = 0
    for experiment in experiments:
        for job in experiment.jobs:
            timeout_total_count += job.agent_timeout_total_count
            timeout_runtime_sum += sum(job.agent_timeout_runtime_samples)
            timeout_sample_count += len(job.agent_timeout_runtime_samples)

    timeout_average_runtime = (
        timeout_runtime_sum / timeout_sample_count
        if timeout_sample_count
        else None
    )
    total_prompt_count = prompt_stats.get("total_prompt_count", 0) or 0
    total_response_count = prompt_stats.get("total_response_count", 0) or 0
    overall_average_prompt_chars = (
        prompt_stats.get("total_prompt_chars", 0) / total_prompt_count
        if total_prompt_count
        else None
    )
    overall_average_response_chars = (
        prompt_stats.get("total_response_chars", 0) / total_response_count
        if total_response_count
        else None
    )

    return {
        "generated_at": generated_at,
        "experiment_count": len(experiments),
        "experiments": [experiment.as_summary() for experiment in experiments],
        "episode_runtime_correlation": correlation_map.get("episode_count", []),
        "input_token_runtime_correlation": correlation_map.get("n_input_tokens", []),
        "output_token_runtime_correlation": correlation_map.get("n_output_tokens", []),
        "agent_timeout_error_total_count": timeout_total_count,
        "agent_timeout_error_average_runtime": timeout_average_runtime,
        "runtime_accuracy_correlation": runtime_accuracy_correlation,
        "prompt_length_accuracy_correlation": prompt_length_accuracy_correlation,
        "response_length_accuracy_correlation": response_length_accuracy_correlation,
        "per_task_runtime_accuracy_correlation": per_task_runtime_correlations,
        "per_task_trace_length_accuracy_correlation": per_task_trace_correlations,
        "per_task_prompt_length_accuracy_correlation": per_task_prompt_correlations,
        "per_task_response_length_accuracy_correlation": per_task_response_correlations,
        "per_experiment_prompt_length_accuracy_correlation": per_experiment_prompt_correlations,
        "per_experiment_response_length_accuracy_correlation": per_experiment_response_correlations,
        "average_prompt_characters_overall": overall_average_prompt_chars,
        "average_response_characters_overall": overall_average_response_chars,
        "per_task_prompt_response_average": prompt_stats.get("per_task", []),
        "task_solvability": task_solvability,
    }


UNIT_FACTORS: Dict[TimeUnit, float] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
}


FONT_SCALE = 0.7
MAX_LABEL_LENGTH = 56


def _warn_skipped_plot(reason: str, output_path: Path) -> None:
    print(
        f"[trace_runtime_report] Skipping {output_path}: {reason}",
        file=sys.stderr,
    )


def _wrap_label(text: str, width: int = 30) -> str:
    if len(text) <= MAX_LABEL_LENGTH:
        return text
    return text[: MAX_LABEL_LENGTH - 1] + "…"


def _scaled_font_size() -> float:
    base = float(plt.rcParams.get("font.size", 10.0))
    return base * FONT_SCALE


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    if output_path.suffix.lower() != ".pdf":
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=200)


def plot_runtimes(
    summaries: List[Dict[str, object]],
    time_unit: TimeUnit,
    output_path: Path,
) -> bool:
    factor = UNIT_FACTORS[time_unit]
    filtered = [
        summary
        for summary in summaries
        if summary.get("overall_runtime_seconds") is not None
    ]
    if not filtered:
        _warn_skipped_plot("no runtime data available", output_path)
        return False

    filtered.sort(
        key=lambda item: item["overall_runtime_seconds"] or 0.0, reverse=True
    )
    labels = [item["experiment"] for item in filtered]
    wrapped_labels = [_wrap_label(label, width=56) for label in labels]

    metrics = [
        (
            [
                (item["overall_runtime_seconds"] or 0.0) / factor for item in filtered
            ],
            f"Overall job runtime ({time_unit})",
            "#1f77b4",
        ),
        (
            [
                (item.get("max_task_runtime_seconds") or 0.0) / factor
                for item in filtered
            ],
            f"Max single-task runtime ({time_unit})",
            "#ff7f0e",
        ),
        (
            [
                (item.get("avg_task_runtime_seconds") or 0.0) / factor
                for item in filtered
            ],
            f"Average single-task runtime ({time_unit})",
            "#2ca02c",
        ),
        (
            [
                (item.get("avg_top_25_task_runtime_seconds") or 0.0) / factor
                for item in filtered
            ],
            f"Average top 25% single-task runtime ({time_unit})",
            "#9467bd",
        ),
        (
            [
                (item.get("avg_top_10_task_runtime_seconds") or 0.0) / factor
                for item in filtered
            ],
            f"Average top 10% single-task runtime ({time_unit})",
            "#d62728",
        ),
    ]

    base_positions = np.arange(len(labels))
    n_metrics = len(metrics)
    bar_height = min(0.15, max(0.06, 0.6 / max(n_metrics, 1)))
    offsets = np.linspace(-(n_metrics - 1) / 2.0, (n_metrics - 1) / 2.0, n_metrics)
    offsets *= bar_height

    row_height = max(0.55, n_metrics * (bar_height + 0.02))
    fig_height = min(36.0, max(6.0, len(labels) * row_height))

    fig, ax = plt.subplots(figsize=(12.0, fig_height), constrained_layout=True)

    for (values, label, color), offset in zip(metrics, offsets):
        positions = base_positions + offset
        ax.barh(
            positions,
            values,
            height=bar_height * 0.9,
            color=color,
            alpha=0.7,
            label=label,
        )

    ax.set_yticks(base_positions)
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()
    scaled_font = _scaled_font_size()
    ax.set_xlabel(f"Runtime ({time_unit})", fontsize=scaled_font)
    ax.set_title("Experiment Runtime Comparison", fontsize=scaled_font)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=scaled_font)
    ax.margins(x=0.02, y=0.01)
    ax.tick_params(axis="both", labelsize=scaled_font)

    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def plot_task_boxplot(
    summaries: List[Dict[str, object]],
    time_unit: TimeUnit,
    output_path: Path,
) -> bool:
    factor = UNIT_FACTORS[time_unit]
    task_to_runtimes: Dict[str, List[float]] = {}

    for summary in summaries:
        for job in summary.get("jobs", []):
            for detail in job.get("task_details") or []:
                task_name = detail.get("task_name")
                runtime_seconds = detail.get("runtime_seconds")
                if not task_name or runtime_seconds is None:
                    continue
                task_to_runtimes.setdefault(task_name, []).append(runtime_seconds / factor)

    if not task_to_runtimes:
        _warn_skipped_plot("no per-task runtime samples available", output_path)
        return False

    sorted_items = sorted(
        task_to_runtimes.items(),
        key=lambda item: (sum(item[1]) / len(item[1])) if item[1] else 0.0,
        reverse=True,
    )
    labels = [name for name, _ in sorted_items]
    data = [values for _, values in sorted_items]

    wrapped_labels = [_wrap_label(label, width=64) for label in labels]

    row_height = 0.45
    fig_height = min(36.0, max(6.5, len(labels) * row_height + 2.0))

    fig, ax = plt.subplots(figsize=(11.0, fig_height), constrained_layout=True)
    box = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        labels=wrapped_labels,
        showfliers=True,
    )

    scaled_font = _scaled_font_size()
    color = "#1f77b4"
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    for median in box["medians"]:
        median.set_color("#d62728")
        median.set_linewidth(1.5)

    ax.set_xlabel(f"Single-task runtime ({time_unit})", fontsize=scaled_font)
    ax.set_title("Single-task Runtime Distribution per Task Across Experiments", fontsize=scaled_font)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.margins(x=0.05, y=0.02)
    ax.tick_params(axis="both", labelsize=scaled_font)

    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def plot_task_correlation_bars(
    per_task_correlations: List[Dict[str, object]],
    output_path: Path,
    *,
    title: str,
) -> bool:
    entries = [
        entry
        for entry in per_task_correlations
        if entry.get("spearman_rho") is not None
    ]
    if not entries:
        _warn_skipped_plot("no correlation values to plot", output_path)
        return False

    # Already sorted descending when generated; keep that order.
    labels = [entry["task_name"] for entry in entries]
    wrapped_labels = [_wrap_label(label, width=56) for label in labels]
    values = [entry["spearman_rho"] for entry in entries]
    colors = [
        "#2ca02c" if value >= 0 else "#d62728"
        for value in values
    ]

    height = max(6.0, len(labels) * 0.25)
    fig, ax = plt.subplots(
        figsize=(12.0, height), constrained_layout=True
    )
    y_positions = range(len(labels))
    ax.barh(y_positions, values, color=colors, alpha=0.8)
    ax.axvline(0, color="#444444", linewidth=1.0)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()  # Highest positive at top
    min_value = min(values)
    max_value = max(values)
    margin = max(0.05, (max_value - min_value) * 0.05)
    ax.set_xlim(min_value - margin, max_value + margin)
    scaled_font = _scaled_font_size()
    ax.set_xlabel("Spearman correlation (success vs. feature)", fontsize=scaled_font)
    ax.set_title(title, fontsize=scaled_font)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=scaled_font)

    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def plot_experiment_prompt_response_lengths(
    experiments_summary: List[Dict[str, object]],
    output_path: Path,
) -> bool:
    entries = [
        (
            item["experiment"],
            item.get("average_prompt_characters"),
            item.get("average_response_characters"),
        )
        for item in experiments_summary
        if item.get("average_prompt_characters") is not None
        or item.get("average_response_characters") is not None
    ]
    if not entries:
        _warn_skipped_plot(
            "no prompt/response averages per experiment available", output_path
        )
        return False

    # Sort by prompt average descending (fallback to response, then name).
    entries.sort(
        key=lambda entry: (
            entry[1] if entry[1] is not None else -1,
            entry[2] if entry[2] is not None else -1,
        ),
        reverse=True,
    )

    labels = [entry[0] for entry in entries]
    wrapped_labels = [_wrap_label(label, width=64) for label in labels]
    prompt_values = [entry[1] or 0.0 for entry in entries]
    response_values = [entry[2] or 0.0 for entry in entries]

    base_positions = np.arange(len(labels))
    bar_height = 0.35
    offsets = np.array([-0.5, 0.5]) * bar_height
    row_height = 0.5
    fig_height = min(36.0, max(6.0, len(labels) * row_height + 2.0))

    fig, ax = plt.subplots(figsize=(11.0, fig_height), constrained_layout=True)

    ax.barh(
        base_positions + offsets[0],
        prompt_values,
        height=bar_height * 0.9,
        color="#1f77b4",
        alpha=0.7,
        label="Prompt average length",
    )
    ax.barh(
        base_positions + offsets[1],
        response_values,
        height=bar_height * 0.9,
        color="#ff7f0e",
        alpha=0.7,
        label="Response average length",
    )

    ax.set_yticks(base_positions)
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()
    scaled_font = _scaled_font_size()
    ax.set_xlabel("Average characters per episode", fontsize=scaled_font)
    ax.set_title("Average Prompt and Response Length per Experiment", fontsize=scaled_font)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=scaled_font)
    ax.margins(x=0.05, y=0.02)
    ax.tick_params(axis="both", labelsize=scaled_font)

    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def plot_overall_prompt_response_lengths(
    average_prompt: Optional[float],
    average_response: Optional[float],
    output_path: Path,
) -> bool:
    if average_prompt is None and average_response is None:
        _warn_skipped_plot(
            "no overall prompt or response averages available", output_path
        )
        return False
    prompt_value = average_prompt or 0.0
    response_value = average_response or 0.0
    labels = ["Prompt", "Response"]
    values = [prompt_value, response_value]
    colors = ["#1f77b4", "#ff7f0e"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0), constrained_layout=True)
    ax.bar(labels, values, color=colors, alpha=0.8)
    scaled_font = _scaled_font_size()
    ax.set_ylabel("Average characters per episode", fontsize=scaled_font)
    ax.set_title("Overall Average Prompt/Response Lengths", fontsize=scaled_font)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=scaled_font)
    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def plot_task_prompt_response_lengths(
    per_task_stats: List[Dict[str, object]],
    output_path: Path,
) -> bool:
    entries = [
        entry
        for entry in per_task_stats
        if entry.get("average_prompt_characters") is not None
        or entry.get("average_response_characters") is not None
    ]
    if not entries:
        _warn_skipped_plot(
            "no per-task prompt or response averages available", output_path
        )
        return False

    entries.sort(
        key=lambda entry: (
            entry.get("average_prompt_characters") or 0.0,
            entry.get("average_response_characters") or 0.0,
        ),
        reverse=True,
    )

    labels = [entry["task_name"] for entry in entries]
    wrapped_labels = [_wrap_label(label, width=32) for label in labels]
    prompt_values = [entry.get("average_prompt_characters") or 0.0 for entry in entries]
    response_values = [
        entry.get("average_response_characters") or 0.0 for entry in entries
    ]

    base_positions = np.arange(len(labels))
    bar_height = 0.35
    offsets = np.array([-0.5, 0.5]) * bar_height
    row_height = 0.5
    fig_height = min(36.0, max(6.0, len(labels) * row_height + 2.0))

    fig, ax = plt.subplots(figsize=(11.0, fig_height), constrained_layout=True)
    ax.barh(
        base_positions + offsets[0],
        prompt_values,
        height=bar_height * 0.9,
        color="#1f77b4",
        alpha=0.7,
        label="Prompt average length",
    )
    ax.barh(
        base_positions + offsets[1],
        response_values,
        height=bar_height * 0.9,
        color="#ff7f0e",
        alpha=0.7,
        label="Response average length",
    )
    ax.set_yticks(base_positions)
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()
    scaled_font = _scaled_font_size()
    ax.set_xlabel("Average characters per episode", fontsize=scaled_font)
    ax.set_title("Average Prompt and Response Length per Task", fontsize=scaled_font)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.legend(fontsize=scaled_font)
    ax.margins(x=0.05, y=0.02)
    ax.tick_params(axis="both", labelsize=scaled_font)
    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def plot_task_solvability(
    solvability_data: List[Dict[str, object]],
    output_path: Path,
) -> bool:
    entries = [
        entry
        for entry in solvability_data
        if entry.get("experiments_total")
    ]
    if not entries:
        _warn_skipped_plot("no task solvability records available", output_path)
        return False

    entries = sorted(
        entries,
        key=lambda item: item.get("percent_solved", 0.0),
        reverse=True,
    )
    labels = [entry["task"] for entry in entries]
    wrapped_labels = [_wrap_label(label, width=32) for label in labels]
    percents = [entry.get("percent_solved", 0.0) for entry in entries]
    totals = [entry.get("experiments_total", 0) for entry in entries]

    base_positions = np.arange(len(labels))
    row_height = 0.45
    fig_height = min(36.0, max(6.0, len(labels) * row_height + 2.0))

    fig, ax = plt.subplots(
        figsize=(11.0, fig_height), constrained_layout=True
    )
    bars = ax.barh(
        base_positions,
        percents,
        color="#1f77b4",
        alpha=0.7,
    )
    ax.set_yticks(base_positions)
    ax.set_yticklabels(wrapped_labels)
    ax.invert_yaxis()
    scaled_font = _scaled_font_size()
    ax.set_xlabel("Percent experiments solved (%)", fontsize=scaled_font)
    ax.set_title("Task Solvability Across Experiments", fontsize=scaled_font)
    ax.set_xlim(0, 100)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=scaled_font)

    # Annotate bars with counts
    for bar, total, percent in zip(bars, totals, percents):
        ax.text(
            percent + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{percent:.1f}% ({total})",
            va="center",
            fontsize=max(1.0, scaled_font * 0.8),
        )

    _save_figure(fig, output_path)
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root directory not found: {root}")

    response_records: List[Dict[str, object]] = []
    keystroke_terms: Dict[str, Counter[str]] = {
        "all": Counter(),
        "claude_gpt": Counter(),
        "qwen3_8b": Counter(),
    }
    solvability_records: Dict[str, Dict[str, bool]] = defaultdict(dict)
    experiments = discover_experiments(
        root,
        response_records=response_records,
        keystroke_terms=keystroke_terms,
        solvability_records=solvability_records,
        flat_structure=args.flat_structure,
    )
    task_solvability: List[Dict[str, object]] = []
    for task_name, mapping in solvability_records.items():
        total = len(mapping)
        solved = sum(1 for solved_flag in mapping.values() if solved_flag)
        percent = (solved / total * 100.0) if total else 0.0
        task_solvability.append(
            {
                "task": task_name,
                "experiments_total": total,
                "experiments_solved": solved,
                "percent_solved": percent,
            }
        )
    task_solvability.sort(key=lambda item: item["percent_solved"], reverse=True)
    prompt_stats = aggregate_prompt_response_stats(experiments)
    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    correlation_map = compute_runtime_correlations(experiments)
    runtime_accuracy_correlation = compute_feature_accuracy_correlation(
        experiments, feature="runtime"
    )
    per_task_runtime_correlations = compute_per_task_accuracy_correlations(
        experiments, feature="runtime"
    )
    per_task_trace_correlations = compute_per_task_accuracy_correlations(
        experiments, feature="episodes"
    )
    prompt_length_accuracy_correlation = compute_feature_accuracy_correlation(
        experiments, feature="prompt"
    )
    response_length_accuracy_correlation = compute_feature_accuracy_correlation(
        experiments, feature="response"
    )
    per_task_prompt_correlations = compute_per_task_accuracy_correlations(
        experiments, feature="prompt"
    )
    per_task_response_correlations = compute_per_task_accuracy_correlations(
        experiments, feature="response"
    )
    per_experiment_prompt_correlations = compute_per_experiment_feature_correlations(
        experiments, feature="prompt"
    )
    per_experiment_response_correlations = compute_per_experiment_feature_correlations(
        experiments, feature="response"
    )
    summary = build_summary(
        experiments,
        generated_at,
        correlation_map,
        runtime_accuracy_correlation,
        per_task_runtime_correlations,
        per_task_trace_correlations,
        prompt_stats,
        prompt_length_accuracy_correlation,
        response_length_accuracy_correlation,
        per_task_prompt_correlations,
        per_task_response_correlations,
        per_experiment_prompt_correlations,
        per_experiment_response_correlations,
        task_solvability,
    )
    summary["response_command_record_count"] = len(response_records)
    all_terms = keystroke_terms["all"]
    filtered_terms = keystroke_terms["claude_gpt"]
    summary["keystroke_term_count"] = len(all_terms)
    summary["top_keystroke_terms"] = [
        {"term": term, "count": count}
        for term, count in all_terms.most_common(50)
    ]
    filtered_terms = keystroke_terms["claude_gpt"]
    summary["claude_gpt_keystroke_term_count"] = len(filtered_terms)
    summary["top_claude_gpt_keystroke_terms"] = [
        {"term": term, "count": count}
        for term, count in filtered_terms.most_common(50)
    ]
    qwen_terms = keystroke_terms["qwen3_8b"]
    summary["qwen3_8b_keystroke_term_count"] = len(qwen_terms)
    summary["top_qwen3_8b_keystroke_terms"] = [
        {"term": term, "count": count}
        for term, count in qwen_terms.most_common(50)
    ]

    output_json = (
        args.output_json
        if args.output_json is not None
        else root / "trace_runtime_summary.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    output_response_commands = (
        args.output_response_commands
        if args.output_response_commands is not None
        else root / "trace_response_commands.json"
    )
    output_response_commands.parent.mkdir(parents=True, exist_ok=True)
    with output_response_commands.open("w", encoding="utf-8") as handle:
        json.dump(response_records, handle, indent=2)

    output_keystroke_frequency = (
        args.output_keystroke_frequency
        if args.output_keystroke_frequency is not None
        else root / "trace_keystroke_frequencies.json"
    )
    output_keystroke_frequency.parent.mkdir(parents=True, exist_ok=True)
    keystroke_list = {
        "all": [
            {"term": term, "count": count}
            for term, count in keystroke_terms["all"].most_common()
        ],
        "claude_gpt": [
            {"term": term, "count": count}
            for term, count in keystroke_terms["claude_gpt"].most_common()
        ],
        "qwen3_8b": [
            {"term": term, "count": count}
            for term, count in keystroke_terms["qwen3_8b"].most_common()
        ],
    }
    with output_keystroke_frequency.open("w", encoding="utf-8") as handle:
        json.dump(keystroke_list, handle, indent=2)

    output_correlation = (
        args.output_correlation_table
        if args.output_correlation_table is not None
        else root / "trace_runtime_episode_correlation.csv"
    )
    write_correlation_table(
        correlation_map,
        per_task_runtime_correlations,
        per_task_trace_correlations,
        per_task_prompt_correlations,
        per_task_response_correlations,
        prompt_length_accuracy_correlation,
        response_length_accuracy_correlation,
        per_experiment_prompt_correlations,
        per_experiment_response_correlations,
        output_correlation,
    )

    if not experiments:
        print(
            f"[trace_runtime_report] No experiments with trace data found under {root}. "
            "No figures were generated.",
            file=sys.stderr,
        )
        return

    figures_generated: List[bool] = []

    output_figure = (
        args.output_figure
        if args.output_figure is not None
        else root / "trace_runtime_summary.png"
    )
    figures_generated.append(
        plot_runtimes(summary["experiments"], args.time_unit, output_figure)
    )

    output_boxplot = (
        args.output_task_boxplot
        if args.output_task_boxplot is not None
        else root / "trace_runtime_task_boxplot.png"
    )
    figures_generated.append(
        plot_task_boxplot(summary["experiments"], args.time_unit, output_boxplot)
    )

    output_runtime_corr = (
        args.output_runtime_correlation_figure
        if args.output_runtime_correlation_figure is not None
        else root / "trace_runtime_task_accuracy_correlation.png"
    )
    figures_generated.append(
        plot_task_correlation_bars(
            per_task_runtime_correlations,
            output_runtime_corr,
            title="Per-task Correlation: Runtime vs. Success",
        )
    )

    output_trace_corr = (
        args.output_trace_length_correlation_figure
        if args.output_trace_length_correlation_figure is not None
        else root / "trace_length_task_accuracy_correlation.png"
    )
    figures_generated.append(
        plot_task_correlation_bars(
            per_task_trace_correlations,
            output_trace_corr,
            title="Per-task Correlation: Trace Length vs. Success",
        )
    )

    output_prompt_response = (
        args.output_prompt_response_figure
        if args.output_prompt_response_figure is not None
        else root / "trace_prompt_response_lengths.png"
    )
    figures_generated.append(
        plot_experiment_prompt_response_lengths(
            summary["experiments"], output_prompt_response
        )
    )

    output_overall_prompt_response = (
        args.output_overall_prompt_response_figure
        if args.output_overall_prompt_response_figure is not None
        else root / "trace_prompt_response_lengths_overall.png"
    )
    figures_generated.append(
        plot_overall_prompt_response_lengths(
            summary.get("average_prompt_characters_overall"),
            summary.get("average_response_characters_overall"),
            output_overall_prompt_response,
        )
    )

    output_task_prompt_response = (
        args.output_task_prompt_response_figure
        if args.output_task_prompt_response_figure is not None
        else root / "trace_prompt_response_lengths_by_task.png"
    )
    figures_generated.append(
        plot_task_prompt_response_lengths(
            summary.get("per_task_prompt_response_average", []),
            output_task_prompt_response,
        )
    )

    output_task_solvability = (
        args.output_task_solvability_figure
        if args.output_task_solvability_figure is not None
        else root / "trace_task_solvability.png"
    )
    figures_generated.append(
        plot_task_solvability(task_solvability, output_task_solvability)
    )

    if not any(figures_generated):
        print(
            "[trace_runtime_report] No figures were generated from the available data. "
            "Verify that the traces contain runtime and prompt/response statistics.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
