#!/usr/bin/env python3
"""
D1 Constraining: Add tool restrictions, step limits, and brevity requirements.

Takes sandbox datasets and adds operational constraints to instructions
(e.g., "solve in under 5 commands", "use only built-in tools", "no pip install").

Usage:
    python d1_constrain.py \
        --sandbox-dataset DCAgent/stackexchange-tezos-sandboxes \
        --output-repo DCAgent/d1-constrain-tezos \
        --model gpt-4o-mini \
        --push
"""

from __future__ import annotations

import sys
# Remove editable install finders that point to feuer1's home
sys.meta_path = [f for f in sys.meta_path if 'editable' not in str(getattr(f, '__module__', '')).lower()]
sys.path.insert(0, "/e/scratch/jureap59/guha1/OpenThoughts-Agent")

import argparse
import logging
import tempfile
from pathlib import Path

from datasets import Dataset, load_dataset
from data.completions import run_completions
from data.commons import upload_tasks_to_hf
from d1_harden import extract_instructions_from_sandbox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CONSTRAIN_PROMPT = """You are an expert at adding operational constraints to coding tasks.

Given the following task, add realistic constraints that make it more challenging
to execute. Choose 2-3 constraints from this list:

Tool restrictions:
- "Use only built-in shell commands (no pip/npm/apt install)"
- "Do not use any external HTTP requests or downloads"
- "Use only standard library — no third-party packages"

Step/time limits:
- "Complete the task in 5 or fewer shell commands"
- "Your solution must execute in under 30 seconds"
- "Minimize the number of file writes"

Brevity/style:
- "Write the solution as a single shell script"
- "Keep all code under 50 lines"
- "No comments — code must be self-documenting"

Error handling:
- "Handle all edge cases gracefully with informative error messages"
- "The solution must be idempotent (safe to run multiple times)"

Original task:
{{instruction}}

Provide the FULL task with constraints added naturally into the text. Do not list
constraints separately — weave them into the task description. Output ONLY the
constrained task."""


def main():
    p = argparse.ArgumentParser(description="D1: Add constraints to task instructions")
    p.add_argument("--sandbox-dataset", required=True)
    p.add_argument("--output-repo", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    log.info(f"Loading sandbox: {args.sandbox_dataset}")
    ds = load_dataset(args.sandbox_dataset, split="train")

    log.info("Extracting instructions...")
    tasks = extract_instructions_from_sandbox(ds)
    log.info(f"  Extracted {len(tasks)} tasks")

    instructions = [t["instruction"] for t in tasks[:args.limit]]
    constrain_ds = Dataset.from_dict({"instruction": instructions})

    log.info(f"Constraining {len(instructions)} instructions with {args.model}...")
    result = run_completions(
        dataset=constrain_ds,
        model=args.model,
        map_type="chat",
        map_config={"user_message": CONSTRAIN_PROMPT, "output_column": "constrained_instruction"},
        temperature=0.7,
    )
    constrained = result.dataset["constrained_instruction"]

    log.info("Rebuilding sandboxes with constrained instructions...")
    out_dir = Path(tempfile.mkdtemp(prefix="d1_constrain_"))
    for i, (task, new_instr) in enumerate(zip(tasks[:args.limit], constrained)):
        if not new_instr or len(new_instr.strip()) < 50:
            new_instr = task["instruction"]
        task_dir = out_dir / f"constrained-{i:05d}"
        task_dir.mkdir()
        for name, content in task["files"].items():
            fpath = task_dir / name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("instruction.md"):
                fpath.write_text(new_instr)
            else:
                fpath.write_bytes(content)

    log.info(f"Created {len(constrained)} constrained tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)

    log.info("Done!")


if __name__ == "__main__":
    main()
