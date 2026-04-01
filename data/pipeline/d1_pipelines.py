#!/usr/bin/env python3
"""
D1 Pipeline variants: Compose harden, constrain, and mix in different orders.

Variants:
    d1_harden_then_constrain  — harden instructions, then add constraints
    d1_constrain_then_harden  — constrain first, then harden
    d1_all_three              — mix → harden → constrain
    d1_hardened_vs_original   — side-by-side hardened vs raw control

Usage:
    python d1_pipelines.py harden-then-constrain \
        --sandbox-dataset DCAgent/stackexchange-tezos-sandboxes \
        --output-repo DCAgent/d1-htc-tezos --push

    python d1_pipelines.py all-three \
        --sandbox-datasets DCAgent/stackexchange-tezos-sandboxes DCAgent/exp_rpt_stack-bash \
        --output-repo DCAgent/d1-all-tezos-bash --push

    python d1_pipelines.py hardened-vs-original \
        --sandbox-dataset DCAgent/stackexchange-tezos-sandboxes \
        --output-repo DCAgent/d1-hvo-tezos --push
"""

from __future__ import annotations

import sys
# Remove editable install finders that point to feuer1's home
sys.meta_path = [f for f in sys.meta_path if 'editable' not in str(getattr(f, '__module__', '')).lower()]
sys.path.insert(0, "/e/scratch/jureap59/guha1/OpenThoughts-Agent")

import argparse
import logging
import random
import tempfile
from pathlib import Path

from datasets import Dataset, load_dataset
from data.completions import run_completions
from data.commons import upload_tasks_to_hf
from d1_harden import HARDEN_PROMPT, extract_instructions_from_sandbox
from d1_constrain import CONSTRAIN_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _apply_llm(instructions: list[str], prompt: str, column: str, model: str) -> list[str]:
    """Apply an LLM prompt to a list of instructions."""
    ds = Dataset.from_dict({"instruction": instructions})
    result = run_completions(
        dataset=ds, model=model, map_type="chat",
        map_config={"user_message": prompt, "output_column": column},
        temperature=0.7,
    )
    return result.dataset[column]


def _write_tasks(tasks: list[dict], instructions: list[str], out_dir: Path, prefix: str):
    """Write tasks to disk with new instructions."""
    for i, (task, instr) in enumerate(zip(tasks, instructions)):
        if not instr or len(instr.strip()) < 50:
            instr = task["instruction"]
        task_dir = out_dir / f"{prefix}-{i:05d}"
        task_dir.mkdir()
        for name, content in task["files"].items():
            fpath = task_dir / name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("instruction.md"):
                fpath.write_text(instr)
            else:
                fpath.write_bytes(content)


def run_harden_then_constrain(args):
    """Harden instructions, then add constraints."""
    ds = load_dataset(args.sandbox_dataset, split="train")
    tasks = extract_instructions_from_sandbox(ds)[:args.limit]
    instructions = [t["instruction"] for t in tasks]

    log.info(f"Step 1: Hardening {len(instructions)} instructions...")
    hardened = _apply_llm(instructions, HARDEN_PROMPT, "hardened", args.model)

    log.info(f"Step 2: Constraining hardened instructions...")
    # Use hardened as input for constraining
    constrained = _apply_llm(hardened, CONSTRAIN_PROMPT.replace("{{instruction}}", "{{instruction}}"), "constrained", args.model)

    out_dir = Path(tempfile.mkdtemp(prefix="d1_htc_"))
    _write_tasks(tasks, constrained, out_dir, "htc")
    log.info(f"Created {len(constrained)} tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)


def run_constrain_then_harden(args):
    """Constrain first, then harden."""
    ds = load_dataset(args.sandbox_dataset, split="train")
    tasks = extract_instructions_from_sandbox(ds)[:args.limit]
    instructions = [t["instruction"] for t in tasks]

    log.info(f"Step 1: Constraining {len(instructions)} instructions...")
    constrained = _apply_llm(instructions, CONSTRAIN_PROMPT, "constrained", args.model)

    log.info(f"Step 2: Hardening constrained instructions...")
    hardened = _apply_llm(constrained, HARDEN_PROMPT.replace("{{instruction}}", "{{instruction}}"), "hardened", args.model)

    out_dir = Path(tempfile.mkdtemp(prefix="d1_cth_"))
    _write_tasks(tasks, hardened, out_dir, "cth")
    log.info(f"Created {len(hardened)} tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)


def run_all_three(args):
    """Mix → harden → constrain pipeline."""
    rng = random.Random(args.seed)

    # Step 1: Mix
    log.info("Step 1: Mixing sources...")
    all_tasks = []
    for repo in args.sandbox_datasets:
        ds = load_dataset(repo, split="train")
        tasks = extract_instructions_from_sandbox(ds)
        all_tasks.extend(tasks)
        log.info(f"  {repo}: {len(tasks)} tasks")

    rng.shuffle(all_tasks)
    tasks = all_tasks[:args.limit]
    instructions = [t["instruction"] for t in tasks]
    log.info(f"  Mixed: {len(tasks)} tasks")

    # Step 2: Harden
    log.info(f"Step 2: Hardening {len(instructions)} instructions...")
    hardened = _apply_llm(instructions, HARDEN_PROMPT, "hardened", args.model)

    # Step 3: Constrain
    log.info(f"Step 3: Constraining hardened instructions...")
    constrained = _apply_llm(hardened, CONSTRAIN_PROMPT.replace("{{instruction}}", "{{instruction}}"), "constrained", args.model)

    out_dir = Path(tempfile.mkdtemp(prefix="d1_all_"))
    _write_tasks(tasks, constrained, out_dir, "all")
    log.info(f"Created {len(constrained)} tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)


def run_hardened_vs_original(args):
    """Side-by-side: hardened and original in same dataset (for ablation)."""
    ds = load_dataset(args.sandbox_dataset, split="train")
    tasks = extract_instructions_from_sandbox(ds)

    # Take half for hardening, half original
    half = min(args.limit // 2, len(tasks))
    original_tasks = tasks[:half]
    harden_tasks = tasks[:half]

    instructions = [t["instruction"] for t in harden_tasks]
    log.info(f"Hardening {len(instructions)} instructions (control: {len(original_tasks)} original)...")
    hardened = _apply_llm(instructions, HARDEN_PROMPT, "hardened", args.model)

    out_dir = Path(tempfile.mkdtemp(prefix="d1_hvo_"))

    # Write originals
    _write_tasks(original_tasks, [t["instruction"] for t in original_tasks], out_dir, "original")
    # Write hardened
    _write_tasks(harden_tasks, hardened, out_dir, "hardened")

    total = len(original_tasks) + len(harden_tasks)
    log.info(f"Created {total} tasks ({len(original_tasks)} original + {len(harden_tasks)} hardened)")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)


def main():
    p = argparse.ArgumentParser(description="D1: Pipeline variants for hardening")
    sub = p.add_subparsers(dest="mode", required=True)

    # harden-then-constrain
    htc = sub.add_parser("harden-then-constrain")
    htc.add_argument("--sandbox-dataset", required=True)
    htc.add_argument("--output-repo", required=True)

    # constrain-then-harden
    cth = sub.add_parser("constrain-then-harden")
    cth.add_argument("--sandbox-dataset", required=True)
    cth.add_argument("--output-repo", required=True)

    # all-three (mix + harden + constrain)
    at = sub.add_parser("all-three")
    at.add_argument("--sandbox-datasets", nargs="+", required=True)
    at.add_argument("--output-repo", required=True)
    at.add_argument("--seed", type=int, default=42)

    # hardened-vs-original
    hvo = sub.add_parser("hardened-vs-original")
    hvo.add_argument("--sandbox-dataset", required=True)
    hvo.add_argument("--output-repo", required=True)

    for sp in [htc, cth, at, hvo]:
        sp.add_argument("--model", default="gpt-4o-mini")
        sp.add_argument("--limit", type=int, default=10000)
        sp.add_argument("--push", action="store_true")

    args = p.parse_args()
    {
        "harden-then-constrain": run_harden_then_constrain,
        "constrain-then-harden": run_constrain_then_harden,
        "all-three": run_all_three,
        "hardened-vs-original": run_hardened_vs_original,
    }[args.mode](args)


if __name__ == "__main__":
    main()
