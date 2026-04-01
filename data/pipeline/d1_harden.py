#!/usr/bin/env python3
"""
D1 Hardening: Add requirements/edge cases/complexity to task instructions.

Takes sandbox datasets, extracts instructions, uses an LLM to create harder
versions, then rebuilds the sandboxes with hardened instructions.

Usage:
    python d1_harden.py \
        --sandbox-dataset DCAgent/stackexchange-tezos-sandboxes \
        --output-repo DCAgent/d1-harden-tezos \
        --model gpt-4o-mini \
        --limit 10000 \
        --push
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import tarfile
import tempfile
from pathlib import Path

# Remove editable install finders that point to feuer1's home
sys.meta_path = [f for f in sys.meta_path if 'editable' not in str(getattr(f, '__module__', '')).lower()]
sys.path.insert(0, "/e/scratch/jureap59/guha1/OpenThoughts-Agent")

from datasets import Dataset, load_dataset
from data.completions import run_completions
from data.commons import upload_tasks_to_hf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HARDEN_PROMPT = """You are an expert at making coding tasks more challenging and complex.

Given the following task/instruction, create a harder version of it. The harder version should:
1. Add additional requirements or constraints
2. Require more edge cases to be handled
3. Introduce additional complexity (e.g., performance requirements, error handling, testing requirements)
4. Be more specific about expected behavior in corner cases
5. Potentially require integration with additional components

Keep the core task the same but make it significantly more challenging to complete correctly.

Original task:
{{instruction}}

Provide ONLY the harder version of the task. Do not include any explanation or preamble."""


def extract_instructions_from_sandbox(ds: Dataset) -> list[dict]:
    """Extract instruction.md and other files from task_binary."""
    tasks = []
    for row in ds:
        tb = row.get("task_binary", b"")
        if not tb:
            continue
        try:
            files = {}
            with tarfile.open(fileobj=io.BytesIO(tb)) as tar:
                for member in tar.getmembers():
                    f = tar.extractfile(member)
                    if f:
                        files[member.name] = f.read()
            instruction = ""
            for name, content in files.items():
                if name.endswith("instruction.md"):
                    instruction = content.decode("utf-8", errors="replace")
            if instruction:
                tasks.append({"instruction": instruction, "path": row.get("path", ""), "files": files})
        except Exception:
            continue
    return tasks


def rebuild_sandbox_with_new_instruction(task: dict, new_instruction: str) -> bytes:
    """Rebuild task_binary tar with the new instruction."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in task["files"].items():
            if name.endswith("instruction.md"):
                content = new_instruction.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser(description="D1: Harden task instructions")
    p.add_argument("--sandbox-dataset", required=True, help="HF sandbox dataset")
    p.add_argument("--output-repo", required=True, help="Output HF repo")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM for hardening")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--push", action="store_true")
    args = p.parse_args()

    log.info(f"Loading sandbox: {args.sandbox_dataset}")
    ds = load_dataset(args.sandbox_dataset, split="train")

    log.info("Extracting instructions...")
    tasks = extract_instructions_from_sandbox(ds)
    log.info(f"  Extracted {len(tasks)} tasks")

    instructions = [t["instruction"] for t in tasks[:args.limit]]
    harden_ds = Dataset.from_dict({"instruction": instructions})

    log.info(f"Hardening {len(instructions)} instructions with {args.model}...")
    result = run_completions(
        dataset=harden_ds,
        model=args.model,
        map_type="chat",
        map_config={"user_message": HARDEN_PROMPT, "output_column": "hardened_instruction"},
        temperature=0.7,
    )
    hardened = result.dataset["hardened_instruction"]

    log.info("Rebuilding sandboxes with hardened instructions...")
    out_dir = Path(tempfile.mkdtemp(prefix="d1_harden_"))
    for i, (task, new_instr) in enumerate(zip(tasks[:args.limit], hardened)):
        if not new_instr or len(new_instr.strip()) < 50:
            new_instr = task["instruction"]  # fallback to original
        task_dir = out_dir / f"hardened-{i:05d}"
        task_dir.mkdir()
        for name, content in task["files"].items():
            fpath = task_dir / name
            fpath.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith("instruction.md"):
                fpath.write_text(new_instr)
            else:
                fpath.write_bytes(content)

    log.info(f"Created {len(hardened)} hardened tasks in {out_dir}")

    if args.push:
        upload_tasks_to_hf(str(out_dir), args.output_repo)

    log.info("Done!")


if __name__ == "__main__":
    main()
