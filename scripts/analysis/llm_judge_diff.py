#!/usr/bin/env python3
"""LLM-judge pairwise comparison of baseline vs post-RL same-task traces.

Answers research question (3) qualitatively: **what specifically changed
between a baseline trace and a post-RL trace on the same task?** The
behavioral_delta script gives macro counts (think tokens up, tool calls
down, etc.); this step uses GPT-5 to give a human-readable judgment per
pair, with tags drawn from a small fixed vocabulary so we can aggregate.

Per pair, the judge returns:

    {
      "task": "<task_id>",
      "behavior_change": "<one-paragraph summary>",
      "tags": ["more-verbose", "different-tool-strategy", ...],
      "confidence": "<high|medium|low>",
      "winner": "<baseline|post-rl|tie>",
      "winner_reason": "<why>"
    }

Pairs are selected from same-task before/after pairs ranked by the
behavioral-delta magnitude (largest-delta first — those are most likely
to surface meaningful differences). Results are cached per (task, before-
trial, after-trial) so re-runs are cheap.

The LLM client uses ``ajudge.llms.litellm_llm.LiteLLM`` if available, with
a graceful fallback to direct OpenAI SDK calls if ajudge is not
importable.

Usage:
    python -m scripts.analysis.llm_judge_diff \\
        --before  penfever/eval-pre-rl \\
        --after   penfever/eval-post-rl \\
        --output  /path/llm_judge_diff.md \\
        --max-pairs 30
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.trace_pair_render import select_pairs  # noqa: E402
from scripts.analysis.utils import (  # noqa: E402
    Trace,
    group_by_task,
    load_traces,
)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_TAG_VOCABULARY: List[str] = [
    "more-verbose",
    "less-verbose",
    "more-tool-calls",
    "fewer-tool-calls",
    "different-tool-strategy",
    "more-careful-planning",
    "more-impulsive",
    "fewer-tool-errors",
    "more-tool-errors",
    "more-self-correction",
    "less-self-correction",
    "more-thinking-budget",
    "less-thinking-budget",
    "different-failure-mode",
    "different-solution-strategy",
    "more-defensive-coding",
    "more-decisive",
    "more-hesitant",
    "shorter-trace",
    "longer-trace",
    "ran-out-of-context",
    "gave-up-early",
    "completed-successfully",
    "no-substantive-change",
]


_SYSTEM_PROMPT = (
    "You are an expert RL researcher reviewing pairs of agent traces. "
    "For each pair, both traces are the same task — one was produced by a "
    "baseline model, the other by the post-RL model. Your job is to "
    "identify what changed in the agent's behavior. Be specific and "
    "concrete. Cite tokens / patterns from the traces when relevant."
)


_USER_PROMPT_TEMPLATE = """You are comparing two agent traces on the same task.

TASK: {task}

BASELINE TRACE:
- reward: {baseline_reward}
- turns: {baseline_turns}
- tool calls: {baseline_tool_calls}
- tool errors: {baseline_tool_errors}
- assistant tokens: {baseline_asst_tokens}
- think tokens: {baseline_think_tokens}
- self-correction hits: {baseline_self_corr}

POST-RL TRACE:
- reward: {after_reward}
- turns: {after_turns}
- tool calls: {after_tool_calls}
- tool errors: {after_tool_errors}
- assistant tokens: {after_asst_tokens}
- think tokens: {after_think_tokens}
- self-correction hits: {after_self_corr}

--- BASELINE TRACE (transcript, truncated) ---
{baseline_text}

--- POST-RL TRACE (transcript, truncated) ---
{after_text}

---

Output a single JSON object with these fields:
- "behavior_change": One paragraph (2-5 sentences) summarizing the concrete
  behavioral difference between the two traces. Focus on *what the agent
  did differently*, not whether it succeeded.
- "tags": A list of 1-4 tag strings from this fixed vocabulary:
  {tag_vocabulary}
  Use only tags that genuinely apply. Don't reach.
- "confidence": "high" / "medium" / "low" — how sure you are the
  behavior change is real (vs. random sampling noise).
- "winner": "baseline" / "post-rl" / "tie" — which trace handled the
  task better. Tie is acceptable.
- "winner_reason": One short sentence justifying the winner verdict.

Reply with ONLY the JSON object. No prose before or after.
"""


# ---------------------------------------------------------------------------
# Trace truncation
# ---------------------------------------------------------------------------

def _trace_text(trace: Trace, max_chars: int) -> str:
    """Compact, judge-friendly transcript: ``[role] content`` per line.

    Truncates from the MIDDLE — we keep the start (task framing, initial
    plan) and end (final answer / failure) because the middle is usually
    the most redundant. If the trace is shorter than the budget, no
    truncation happens.
    """
    messages = trace.raw.get("messages") or trace.raw.get("conversations") or []
    if not isinstance(messages, list):
        return ""
    lines = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        content = msg.get("content")
        if isinstance(content, list):
            content = "\n".join(
                (c.get("text", "") if isinstance(c, dict) else str(c)) for c in content
            )
        elif not isinstance(content, str):
            content = str(content) if content is not None else ""
        # Add structured tool_calls (OpenAI / Anthropic format) inline so the
        # judge can see them even if they aren't in the text content.
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            content = content + "\n[STRUCTURED TOOL_CALLS]: " + json.dumps(tool_calls)[:400]
        lines.append(f"[{role}] {content}")
    full = "\n\n".join(lines)
    if len(full) <= max_chars:
        return full
    # Keep the first 60% from the start, last 40% from the end.
    head_budget = int(max_chars * 0.6)
    tail_budget = max_chars - head_budget - 50  # 50 chars for the marker
    head = full[:head_budget]
    tail = full[-tail_budget:] if tail_budget > 0 else ""
    return head + f"\n\n[... {len(full) - head_budget - tail_budget} chars elided ...]\n\n" + tail


# ---------------------------------------------------------------------------
# LLM client (prefers ajudge.LiteLLM, falls back to openai SDK)
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    model: str = "openai/gpt-5-2025-08-07"
    request_timeout_sec: float = 240.0
    max_concurrent: int = 4


class _LiteLLMClient:
    """Wrapper around ajudge.llms.litellm_llm.LiteLLM for batched async calls."""

    def __init__(self, cfg: LLMConfig) -> None:
        from ajudge.llms.litellm_llm import LiteLLM
        self._llm = LiteLLM(model_id=cfg.model, request_timeout=cfg.request_timeout_sec)
        self._sem = asyncio.Semaphore(cfg.max_concurrent)

    async def call(self, system: str, user: str) -> Tuple[str, Dict[str, Any]]:
        """Return (content, metadata-dict)."""
        async with self._sem:
            resp = await self._llm.call(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ]
            )
        return resp.content, {
            "model": resp.model,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_sec": resp.latency_sec,
        }


class _OpenAIFallbackClient:
    """Bare OpenAI-SDK client used when ajudge isn't importable."""

    def __init__(self, cfg: LLMConfig) -> None:
        from openai import AsyncOpenAI
        # Strip the ``openai/`` provider prefix that ajudge / LiteLLM use.
        self._model = cfg.model.split("/", 1)[1] if "/" in cfg.model else cfg.model
        self._client = AsyncOpenAI()
        self._sem = asyncio.Semaphore(cfg.max_concurrent)
        self._timeout = cfg.request_timeout_sec

    async def call(self, system: str, user: str) -> Tuple[str, Dict[str, Any]]:
        async with self._sem:
            resp = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                ),
                timeout=self._timeout,
            )
        choice = resp.choices[0]
        content = choice.message.content or ""
        return content, {
            "model": resp.model,
            "input_tokens": getattr(resp.usage, "prompt_tokens", 0),
            "output_tokens": getattr(resp.usage, "completion_tokens", 0),
            "latency_sec": 0.0,
        }


def _build_client(cfg: LLMConfig):
    try:
        return _LiteLLMClient(cfg)
    except ImportError:
        return _OpenAIFallbackClient(cfg)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _pair_cache_key(task: str, before: Trace, after: Trace, model: str) -> str:
    """Stable cache key for a (task, before-trial, after-trial, model) tuple.

    Uses the trial identifier when available, falling back to a hash of the
    conversation text. This lets re-runs reuse judgments unless the
    underlying traces change.
    """
    def _trial_id(t: Trace) -> str:
        raw = t.raw or {}
        for k in ("trial_name", "trial_id", "id", "task"):
            v = raw.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        # Fall back to a content hash.
        return hashlib.sha256((t.conversation or "")[:4096].encode("utf-8")).hexdigest()[:16]

    payload = "|".join([model, task, _trial_id(before), _trial_id(after)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Judgment
# ---------------------------------------------------------------------------

@dataclass
class PairJudgment:
    task: str
    behavior_change: str
    tags: List[str]
    confidence: str
    winner: str
    winner_reason: str
    raw_response: str
    metadata: Dict[str, Any]


_JSON_FENCED_RE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _parse_judgment_json(raw: str, task: str) -> Optional[Dict[str, Any]]:
    """Try several strategies to coerce the model's reply into a dict."""
    # Strategy 1: raw is itself JSON.
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    # Strategy 2: fenced ```json``` block.
    m = _JSON_FENCED_RE.search(raw)
    if m:
        try:
            data = json.loads(m.group(1).strip())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    # Strategy 3: first {...} substring.
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start : end + 1])
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    return None


async def _judge_pair(
    client,
    task: str,
    before: Trace,
    after: Trace,
    trace_char_budget: int,
) -> PairJudgment:
    bb = before.behavioral
    ab = after.behavioral
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        task=task,
        baseline_reward=before.reward,
        baseline_turns=before.turns,
        baseline_tool_calls=bb.tool_calls_total,
        baseline_tool_errors=bb.tool_errors,
        baseline_asst_tokens=bb.assistant_tokens,
        baseline_think_tokens=bb.think_tokens,
        baseline_self_corr=bb.self_correction_hits,
        after_reward=after.reward,
        after_turns=after.turns,
        after_tool_calls=ab.tool_calls_total,
        after_tool_errors=ab.tool_errors,
        after_asst_tokens=ab.assistant_tokens,
        after_think_tokens=ab.think_tokens,
        after_self_corr=ab.self_correction_hits,
        baseline_text=_trace_text(before, trace_char_budget),
        after_text=_trace_text(after, trace_char_budget),
        tag_vocabulary=", ".join(_TAG_VOCABULARY),
    )

    raw, meta = await client.call(_SYSTEM_PROMPT, user_prompt)
    parsed = _parse_judgment_json(raw, task) or {}

    return PairJudgment(
        task=task,
        behavior_change=str(parsed.get("behavior_change") or "").strip(),
        tags=[str(t) for t in parsed.get("tags") or [] if isinstance(t, (str, int))],
        confidence=str(parsed.get("confidence") or "low").strip().lower(),
        winner=str(parsed.get("winner") or "tie").strip().lower(),
        winner_reason=str(parsed.get("winner_reason") or "").strip(),
        raw_response=raw,
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Pair selection: largest behavioral delta first.
# ---------------------------------------------------------------------------

def _behavior_delta_score(before: Trace, after: Trace) -> float:
    bb, ab = before.behavioral, after.behavioral
    score = 0.0
    for bv, av in [
        (bb.tool_calls_total, ab.tool_calls_total),
        (bb.tool_errors, ab.tool_errors),
        (bb.assistant_tokens, ab.assistant_tokens),
        (bb.think_tokens, ab.think_tokens),
        (bb.self_correction_hits, ab.self_correction_hits),
        (float(before.turns), float(after.turns)),
    ]:
        scale = max(abs(bv), abs(av), 1.0)
        score += abs(av - bv) / scale
    return score


# ---------------------------------------------------------------------------
# Aggregation + report writing
# ---------------------------------------------------------------------------

def _write_markdown_report(
    judgments: List[PairJudgment],
    output_path: Path,
    before_label: str,
    after_label: str,
    model_id: str,
    n_pairs_attempted: int,
    selection: str = "",
) -> None:
    n = len(judgments)
    tag_counts: Counter[str] = Counter()
    for j in judgments:
        tag_counts.update(j.tags)
    winner_counts = Counter(j.winner for j in judgments)
    confidence_counts = Counter(j.confidence for j in judgments)

    lines = [
        "# LLM-Judge Pair Diff Report",
        "",
        f"- **Before**: `{before_label}`",
        f"- **After**:  `{after_label}`",
        f"- **Judge model**: `{model_id}`",
        f"- **Selection**: `{selection}`"
        + ("  ⚠️ most-shifted tasks — biased toward where the policy changed; "
           "NOT an unbiased win-rate over the benchmark" if selection == "behavior-delta"
           else "  uniform sample — estimates 'did behavior change in general'"
           if selection == "random" else ""),
        f"- **Pairs judged**: {n} (attempted {n_pairs_attempted})",
        "",
        "## Win-rate table",
        "",
        "| winner | count | share |",
        "|---|---|---|",
    ]
    for w in ("baseline", "post-rl", "tie"):
        c = winner_counts.get(w, 0)
        lines.append(f"| {w} | {c} | {(c / n if n else 0):.1%} |")
    lines.append("")

    lines += [
        "## Confidence distribution",
        "",
        "| confidence | count |",
        "|---|---|",
    ]
    for c in ("high", "medium", "low"):
        lines.append(f"| {c} | {confidence_counts.get(c, 0)} |")
    lines.append("")

    lines += [
        "## Tag distribution (across all pair judgments)",
        "",
        "Each row = number of judgments that included this tag. A single pair "
        "can carry multiple tags.",
        "",
        "| tag | count | share of pairs |",
        "|---|---|---|",
    ]
    for tag, cnt in tag_counts.most_common():
        lines.append(f"| `{tag}` | {cnt} | {(cnt / n if n else 0):.1%} |")
    lines.append("")

    # Top 5 most interesting pair judgments — sort by behavioral-delta score
    # we computed during selection (and now stored in metadata).
    sorted_judg = sorted(
        judgments,
        key=lambda j: j.metadata.get("behavior_delta_score", 0.0),
        reverse=True,
    )
    lines += [
        "## Top 5 most-shifted pair judgments (verbatim)",
        "",
    ]
    for j in sorted_judg[:5]:
        lines.append(f"### Task `{j.task}` — winner: **{j.winner}** · confidence: {j.confidence}")
        lines.append("")
        lines.append(f"**Tags**: {', '.join('`' + t + '`' for t in j.tags) if j.tags else '_(none)_'}")
        lines.append("")
        lines.append(f"> {j.behavior_change}")
        lines.append("")
        if j.winner_reason:
            lines.append(f"_Winner reason_: {j.winner_reason}")
            lines.append("")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json_sidecar(
    judgments: List[PairJudgment],
    sidecar_path: Path,
    model_id: str,
) -> None:
    sidecar_path.write_text(
        json.dumps(
            {
                "model": model_id,
                "n_judgments": len(judgments),
                "judgments": [
                    {
                        "task": j.task,
                        "behavior_change": j.behavior_change,
                        "tags": j.tags,
                        "confidence": j.confidence,
                        "winner": j.winner,
                        "winner_reason": j.winner_reason,
                        "metadata": j.metadata,
                    }
                    for j in judgments
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _run_async(args: argparse.Namespace) -> int:
    print(f"[llm-judge-diff] loading traces (before={args.before}, after={args.after})...")
    before = load_traces(args.before, max_rows=args.max_rows)
    after = load_traces(args.after, max_rows=args.max_rows)
    bb = group_by_task(before)
    aa = group_by_task(after)
    # Load the exclusion set (e.g. the most-changed run's selected tasks) so a
    # 'random' draw is disjoint from it. Tolerant of a missing/unreadable file.
    exclude: List[str] = []
    if args.exclude_tasks_file:
        try:
            payload = json.loads(Path(args.exclude_tasks_file).read_text(encoding="utf-8"))
            exclude = list(payload.get("tasks", payload) if isinstance(payload, dict) else payload)
            print(f"[llm-judge-diff] excluding {len(exclude)} task(s) from {args.exclude_tasks_file}")
        except (OSError, json.JSONDecodeError) as e:
            print(f"[llm-judge-diff] WARN: could not read --exclude-tasks-file ({e}); no exclusion",
                  file=sys.stderr)
    # Select the tasks to judge. 'behavior-delta' = the most-shifted tasks
    # (biased toward where the policy changed); 'random' = a uniform
    # without-replacement sample ("did behavior change in general?").
    pairs = select_pairs(bb, aa, args.max_pairs, prefer=args.selection,
                         exclude=exclude, seed=args.seed)
    if not pairs:
        print("[llm-judge-diff] no common tasks between before/after (after exclusion)", file=sys.stderr)
        return 2
    print(f"[llm-judge-diff] selection={args.selection} seed={args.seed}: "
          f"selected {len(pairs)} pair(s) for LLM judgment")
    # Record which tasks were judged, so a downstream 'random' run can exclude
    # this set (disjoint, without replacement across the two probes).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "selected_tasks.json").write_text(
        json.dumps({"selection": args.selection, "seed": args.seed,
                    "tasks": [task for task, _, _ in pairs]}, indent=2),
        encoding="utf-8",
    )

    cfg = LLMConfig(
        model=args.model,
        request_timeout_sec=args.request_timeout,
        max_concurrent=args.max_concurrent,
    )
    client = _build_client(cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.output.parent / "llm_judge_cache.json"
    cache = _load_cache(cache_path)
    cache_hits = 0

    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def _judge_with_cache(idx: int, task: str, b: Trace, a: Trace) -> Tuple[int, PairJudgment]:
        key = _pair_cache_key(task, b, a, args.model)
        score = _behavior_delta_score(b, a)
        was_cached = key in cache and not args.force
        if was_cached:
            cached = cache[key]
            nonlocal cache_hits
            cache_hits += 1
            j = PairJudgment(
                task=cached["task"],
                behavior_change=cached.get("behavior_change", ""),
                tags=list(cached.get("tags", [])),
                confidence=cached.get("confidence", "low"),
                winner=cached.get("winner", "tie"),
                winner_reason=cached.get("winner_reason", ""),
                raw_response=cached.get("raw_response", ""),
                metadata=dict(cached.get("metadata", {})),
            )
        else:
            async with semaphore:
                j = await _judge_pair(client, task, b, a, args.trace_char_budget)
            cache[key] = {
                "task": j.task,
                "behavior_change": j.behavior_change,
                "tags": j.tags,
                "confidence": j.confidence,
                "winner": j.winner,
                "winner_reason": j.winner_reason,
                "raw_response": j.raw_response,
                "metadata": j.metadata,
            }
            _save_cache(cache_path, cache)
        # Stamp the behavior-delta score so the report can rank "top 5 most
        # shifted" without re-running the comparison.
        j.metadata["behavior_delta_score"] = score
        print(
            f"  [{idx+1}/{len(pairs)}] task={task} winner={j.winner} "
            f"tags={','.join(j.tags) or '(none)'} ({'cache' if was_cached else 'fresh'})"
        )
        return idx, j

    tasks_to_run = [
        _judge_with_cache(i, task, b, a) for i, (task, b, a) in enumerate(pairs)
    ]
    results: List[Tuple[int, PairJudgment]] = await asyncio.gather(*tasks_to_run, return_exceptions=False)
    judgments = [j for _, j in sorted(results, key=lambda x: x[0])]
    print(f"[llm-judge-diff] {cache_hits} cache hit(s) of {len(pairs)} pair(s)")

    _write_markdown_report(
        judgments,
        args.output,
        before_label=args.before,
        after_label=args.after,
        model_id=args.model,
        n_pairs_attempted=len(pairs),
        selection=args.selection,
    )
    sidecar = args.output.with_suffix(".json")
    _write_json_sidecar(judgments, sidecar, args.model)
    print(f"[llm-judge-diff] wrote {args.output} (+ {sidecar.name})")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--before", required=True, help="Pre-RL trace source (HF id / JSONL / dir)")
    parser.add_argument("--after",  required=True, help="Post-RL trace source (HF id / JSONL / dir)")
    parser.add_argument("--output", type=Path, required=True, help="Markdown report path (sidecar .json also written)")
    parser.add_argument("--max-pairs", type=int, default=20,
                        help="Number of pairs to judge (default 20, capped for cost)")
    parser.add_argument("--selection", choices=("behavior-delta", "random", "flips", "reward-delta"),
                        default="behavior-delta",
                        help="How to pick the judged tasks: 'behavior-delta' (the most-shifted "
                             "tasks — biased toward where the policy changed) or 'random' (a "
                             "uniform without-replacement sample — 'did behavior change in general?'). "
                             "Default behavior-delta.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --selection random (reproducible sample)")
    parser.add_argument("--exclude-tasks-file", type=Path, default=None,
                        help="JSON file with a list of task ids to exclude from the candidate set "
                             "(e.g. the most-changed run's selected_tasks.json, so a random draw is "
                             "disjoint from it). Missing/unreadable file = no exclusion.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap rows loaded per side (smoke testing only)")
    parser.add_argument("--model", default="openai/gpt-5-2025-08-07",
                        help="LiteLLM-style model id (default openai/gpt-5-2025-08-07)")
    parser.add_argument("--max-concurrent", type=int, default=4,
                        help="Max concurrent API calls (default 4)")
    parser.add_argument("--request-timeout", type=float, default=240.0,
                        help="Per-request timeout in seconds")
    parser.add_argument("--trace-char-budget", type=int, default=12000,
                        help="Per-trace character budget passed to the judge (truncated from middle)")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cache; re-judge every pair")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("[llm-judge-diff] OPENAI_API_KEY not set; cannot call the judge", file=sys.stderr)
        return 2
    return asyncio.run(_run_async(args))


def main() -> None:
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
