#!/usr/bin/env python3
"""
D1 Trace Hints: Extract hints from prior agent episodes and augment instructions.

Analyzes failed/partial episodes to extract what the agent learned, then adds
those hints to the task instruction for the next training round.

Usage:
    python d1_trace_hints.py \
        --trace-dataset DCAgent/stackexchange-tezos-sandboxes_glm_4.7_traces_jupiter \
        --sandbox-dataset DCAgent/stackexchange-tezos-sandboxes \
        --output-repo DCAgent/d1-hints-tezos \
        --model gpt-4o-mini \
        --push
"""

from __future__ import annotations

import sys
# Remove editable install finders that point to feuer1's home
sys.meta_path = [f for f in sys.meta_path if 'editable' not in str(getattr(f, '__module__', '')).lower()]
sys.path.insert(0, "/e/scratch/jureap59/guha1/OpenThoughts-Agent")

import argparse
import io
import logging
import tarfile
import tempfile
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, load_dataset
from data.completions import run_completions
from data.commons import upload_tasks_to_hf
from d1_harden import extract_instructions_from_sandbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXTRACT_HINTS_PROMPT = """You are analyzing an agent's attempt at solving a coding task.
The agent tried but may have failed or only partially succeeded.

Task instruction:
{{instruction}}

Agent's conversation (last few messages):
{{conversation_tail}}

Task result: {{result}}

Based on the agent's attempt, extract 2-3 concise hints that would help a future
agent solve this task more effectively. Focus on:
- What approach worked or didn't work
- Key files/commands that were important
- Common pitfalls to avoid
- The correct sequence of steps

Output ONLY the hints as a bulleted list. No preamble."""

AUGMENT_INSTRUCTION_PROMPT = """You are augmenting a coding task instruction with hints from prior attempts.

Original instruction:
{{instruction}}

Hints from prior agent attempts:
{{hints}}

Create an augmented version of the instruction that naturally incorporates these hints.
The hints should be woven into the task description as "suggestions" or "notes" —
not as spoilers. The agent should still need to figure out the solution.

Output ONLY the augmented instruction."""


def group_traces_by_task(ds: Dataset) -> dict[str, list[dict]]:
    """Group trace rows by task name."""
    groups = defaultdict(list)
    for row in ds:
        task = row.get("task", row.get("trial_name", "").split("__")[0])
        groups[task].append(row)
    return dict(groups)


def extract_conversation_tail(conversations: list[dict], n_messages: int = 6) -> str:
    """Get last N messages as a string."""
    tail = conversations[-n_messages:] if len(conversations) > n_messages else conversations
    parts = []
    for msg in tail:
        role = msg.get("role", "?")
        content = msg.get("content", "")[:500]
        parts.append(f"[{role}]: {content}")
    return "\n\n".join(parts)


def main():
    p = argparse.ArgumentParser(description="D1: Augment instructions with trace hints")
    p.add_argument("--trace-dataset", required=True, help="HF trace dataset (with conversations)")
    p.add_argument("--sandbox-dataset", required=True, help="HF sandbox dataset (with task_binary)")
    p.add_argument("--output-repo", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    # Load traces
    log.info(f"Loading traces: {args.trace_dataset}")
    trace_ds = load_dataset(args.trace_dataset, split="train")
    task_traces = group_traces_by_task(trace_ds)
    log.info(f"  {len(task_traces)} unique tasks with traces")

    # Load sandboxes
    log.info(f"Loading sandboxes: {args.sandbox_dataset}")
    sandbox_ds = load_dataset(args.sandbox_dataset, split="train")
    tasks = extract_instructions_from_sandbox(sandbox_ds)
    log.info(f"  {len(tasks)} sandbox tasks")

    # Step 1: Extract hints from traces
    log.info("Step 1: Extracting hints from traces...")
    hint_rows = []
    for task in tasks[:args.limit]:
        task_name = Path(task["path"]).name if task["path"] else ""
        traces = task_traces.get(task_name, [])
        if not traces:
            hint_rows.append({
                "instruction": task["instruction"],
                "conversation_tail": "(no prior traces available)",
                "result": "unknown",
            })
            continue
        # Use the first trace (could also pick best/worst)
        trace = traces[0]
        convos = trace.get("conversations", [])
        result = str(trace.get("result", "unknown"))
        hint_rows.append({
            "instruction": task["instruction"],
            "conversation_tail": extract_conversation_tail(convos),
            "result": result,
        })

    hint_ds = Dataset.from_list(hint_rows)
    result = run_completions(
        dataset=hint_ds, model=args.model, map_type="chat",
        map_config={"user_message": EXTRACT_HINTS_PROMPT, "output_column": "hints"},
        temperature=0.5,
    )
    hints_list = result.dataset["hints"]
    log.info(f"  Extracted hints for {len(hints_list)} tasks")

    # Step 2: Augment instructions with hints
    log.info("Step 2: Augmenting instructions with hints...")
    augment_rows = []
    for task, hints in zip(tasks[:args.limit], hints_list):
        augment_rows.append({
            "instruction": task["instruction"],
            "hints": hints or "(no hints available)",
        })

    augment_ds = Dataset.from_list(augment_rows)
    result2 = run_completions(
        dataset=augment_ds, model=args.model, map_type="chat",
        map_config={"user_message": AUGMENT_INSTRUCTION_PROMPT, "output_column": "augmented_instruction"},
        temperature=0.5,
    )
    augmented = result2.dataset["augmented_instruction"]
    log.info(f"  Augmented {len(augmented)} instructions")

    # Step 3: Write tasks
    log.info("Step 3: Writing augmented tasks...")
    out_dir = Path(tempfile.mkdtemp(prefix="d1_hints_"))
    for i, (task, new_instr) in enumerate(zip(tasks[:args.limit], augmented)):
        if not new_instr or len(new_instr.strip()) < 50:
            new_instr = task["instruction"]
        task_dir = out_dir / f"hints-{i:05d}"
        task_dir.mkdir()
        for name, content in task["files"].items():
            fpath = task_dir / name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("instruction.md"):
                fpath.write_text(new_instr)
            else:
                fpath.write_bytes(content)

    log.info(f"Created {len(augmented)} hint-augmented tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)

    log.info("Done!")


if __name__ == "__main__":
    main()
