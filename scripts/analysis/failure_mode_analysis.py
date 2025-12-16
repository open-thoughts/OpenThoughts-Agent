#!/usr/bin/env python3
"""
Failure mode analysis helper.

Given a HuggingFace dataset repository that mirrors our eval trace layout
(each trial directory contains result.json, agent/, verifier/, etc.),
download the repo, distill the relevant artifacts per trial, and ask GPT‑5
to judge what the task was, how the agent attempted it, and why it
succeeded or failed.  The OpenAI queries are batched so that we can
process large repositories efficiently.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from huggingface_hub import snapshot_download
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class TrialContext:
    """Minimal context we send to the judge model per trial."""

    trial_id: str
    task_name: str
    prompt_excerpt: str
    agent_excerpt: str
    verifier_excerpt: str
    status: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch failure-mode analysis via GPT-5 judge")
    parser.add_argument("repo_id", help="HuggingFace repo id containing eval traces")
    parser.add_argument(
        "--repo-type", default="dataset", choices=["dataset", "model"], help="Repo type for snapshot download"
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="HF snapshot cache directory (defaults to HuggingFace standard cache)",
    )
    parser.add_argument(
        "--output",
        default="failure_mode_report.md",
        help="Path where the aggregated markdown report will be written",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of trials to analyze per OpenAI request",
    )
    parser.add_argument(
        "--max-trials",
        type=int,
        default=None,
        help="Optional cap on the number of trials to process",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.1",
        help="OpenAI model id to use for judging (default: gpt-5.1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the judge model",
    )
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Optional OpenAI API key (falls back to OPENAI_API_KEY env var)",
    )
    return parser.parse_args()


def snapshot_repo(repo_id: str, repo_type: str, revision: str | None, cache_dir: str | None) -> Path:
    target_dir = snapshot_download(
        repo_id,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=None,
        local_dir_use_symlinks=False,
    )
    logger.info("Downloaded %s to %s", repo_id, target_dir)
    return Path(target_dir)


def read_text(path: Path, limit: int = 1800) -> str:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[missing: {path}]"
    except Exception as exc:  # pragma: no cover - defensive
        return f"[error reading {path}: {exc}]"
    content = content.strip()
    if not content:
        return "[empty]"
    if limit and len(content) > limit:
        return f"{content[:limit]}\n...[truncated]..."
    return content


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("Missing JSON: %s", path)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON %s: %s", path, exc)
    return {}


def iter_trial_dirs(root: Path) -> Iterable[Path]:
    """Yield directories that contain a result.json (one per trial)."""
    seen: set[Path] = set()
    for result_file in sorted(root.rglob("result.json")):
        trial_dir = result_file.parent
        if trial_dir in seen:
            continue
        seen.add(trial_dir)
        yield trial_dir


def latest_episode_dir(agent_dir: Path) -> Path | None:
    episodes = sorted(p for p in agent_dir.glob("episode-*") if p.is_dir())
    return episodes[-1] if episodes else None


def gather_trial_context(trial_dir: Path) -> TrialContext | None:
    result_json = load_json(trial_dir / "result.json")
    task_name = result_json.get("task_name") or trial_dir.name
    agent_dir = trial_dir / "agent"
    verifier_dir = trial_dir / "verifier"

    prompt_excerpt = read_text(agent_dir / "episode-0" / "prompt.txt")
    last_episode = latest_episode_dir(agent_dir)
    agent_excerpt = ""
    if last_episode:
        agent_excerpt = read_text(last_episode / "response.txt", limit=1400)
    else:
        agent_excerpt = "[missing agent transcript]"

    verifier_excerpt = read_text(verifier_dir / "test-stdout.txt") if verifier_dir.exists() else "[no verifier output]"
    status = None
    if result_json:
        agent_result = result_json.get("verifier_result") or {}
        if "rewards" in agent_result:
            status = f"reward={agent_result['rewards']}"
        elif "exception_info" in result_json and result_json["exception_info"]:
            status = f"exception={result_json['exception_info'].get('exception_type')}"

    return TrialContext(
        trial_id=trial_dir.name,
        task_name=task_name,
        prompt_excerpt=prompt_excerpt,
        agent_excerpt=agent_excerpt,
        verifier_excerpt=verifier_excerpt,
        status=status,
    )


def batched(seq: Sequence[TrialContext], batch_size: int) -> Iterable[Sequence[TrialContext]]:
    for idx in range(0, len(seq), batch_size):
        yield seq[idx : idx + batch_size]


SYSTEM_PROMPT = (
    "You are an expert judge analyzing LLM agent traces to identify failure modes. "
    "For each trial you receive the task prompt, rough agent behavior, and verifier output. "
    "Extract:\n"
    "1. What was the task?\n"
    "2. How did the agent attempt it? Highlight key commands or strategies.\n"
    "3. Why did it succeed or fail? Reference verifier cues or observable mistakes.\n"
    "Respond as JSON array where each item has keys: "
    "'trial_id', 'task', 'approach', 'outcome'. Keep outcomes concise."
)


def build_user_prompt(batch: Sequence[TrialContext]) -> str:
    sections: List[str] = []
    for trial in batch:
        section = textwrap.dedent(
            f"""
            ### Trial ID: {trial.trial_id}
            Task Name: {trial.task_name}
            Additional Status: {trial.status or 'n/a'}

            Task Prompt (excerpt):
            {trial.prompt_excerpt}

            Agent Transcript (excerpt):
            {trial.agent_excerpt}

            Verifier / Result Output (excerpt):
            {trial.verifier_excerpt}
            """
        ).strip()
        sections.append(section)

    instructions = (
        "Analyze each trial independently. Return a JSON array (no code fences) "
        "with one object per trial matching the requested schema."
    )
    return f"{instructions}\n\n{'\n\n'.join(sections)}"


def strip_code_fence(payload: str) -> str:
    text = payload.strip()
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 2:
            # parts[1] may include 'json\n'
            content = parts[1]
            if "\n" in content:
                content = content.split("\n", 1)[1]
            return content.strip()
    return text


def judge_batch(
    client: OpenAI, batch: Sequence[TrialContext], model: str, temperature: float
) -> List[dict]:
    prompt = build_user_prompt(batch)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "[]"
    content = strip_code_fence(content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse judge output as JSON: %s\nPayload: %s", exc, content)
        raise

    # Ensure mapping back to trial IDs in current batch order.
    records = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        records.append(entry)
    return records


def write_markdown_report(repo_id: str, output_path: Path, analyses: List[dict]) -> None:
    lines = [
        f"# Failure Mode Analysis",
        "",
        f"- **Source repo**: `{repo_id}`",
        f"- **Trials analyzed**: {len(analyses)}",
        "",
    ]
    for entry in analyses:
        trial_id = entry.get("trial_id", "unknown")
        task = entry.get("task", "").strip()
        approach = entry.get("approach", "").strip()
        outcome = entry.get("outcome", "").strip()
        lines.extend(
            [
                f"## {trial_id}",
                "",
                f"**Task**: {task or 'n/a'}",
                "",
                f"**Agent Approach**: {approach or 'n/a'}",
                "",
                f"**Outcome / Failure Mode**: {outcome or 'n/a'}",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report to %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    snapshot_path = snapshot_repo(args.repo_id, args.repo_type, args.revision, args.cache_dir)
    trial_dirs = [path for path in iter_trial_dirs(snapshot_path)]
    if not trial_dirs:
        raise RuntimeError(f"No trial directories discovered under {snapshot_path}")

    contexts: List[TrialContext] = []
    for trial_dir in trial_dirs:
        ctx = gather_trial_context(trial_dir)
        if ctx:
            contexts.append(ctx)
        if args.max_trials and len(contexts) >= args.max_trials:
            break

    if not contexts:
        raise RuntimeError("No valid trials to analyze")

    logger.info("Collected %d trials for analysis", len(contexts))

    api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key not provided. Use --openai-api-key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key)
    analyses: List[dict] = []

    for batch in batched(contexts, args.batch_size):
        logger.info("Judging batch of %d trials...", len(batch))
        batch_results = judge_batch(client, batch, model=args.model, temperature=args.temperature)
        if len(batch_results) != len(batch):
            logger.warning(
                "Judge returned %d entries for a batch of %d trials; check alignment.",
                len(batch_results),
                len(batch),
            )
        analyses.extend(batch_results)

    output_path = Path(args.output).expanduser().resolve()
    write_markdown_report(args.repo_id, output_path, analyses)


if __name__ == "__main__":
    main()
