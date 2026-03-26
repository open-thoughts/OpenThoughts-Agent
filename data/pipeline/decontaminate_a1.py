#!/usr/bin/env python3
"""
Decontamination for A1 trace datasets against harbor eval benchmarks.

Compares the task instructions in A1 trace datasets against the instruction.md
files from eval benchmarks (terminal_bench_2, swebench-verified, dev_set_v2).
Removes any A1 trace rows where the task instruction is too similar to an eval task.

Uses fuzzy string matching (rapidfuzz) on the first user message in each
conversation (which contains the task instruction embedded in the system prompt).

Usage:
    # Check contamination levels (dry run):
    python decontaminate_a1.py check --threshold 80

    # Decontaminate all trained A1 datasets and push clean versions:
    python decontaminate_a1.py clean --threshold 80 --push

    # Decontaminate a single dataset:
    python decontaminate_a1.py clean-one \
        --dataset DCAgent/exp_rpt_stack-bash_glm_4.7_traces_jupiter \
        --threshold 80 --push
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import multiprocessing as mp
import os
import re
import tempfile
from functools import partial
from pathlib import Path
from typing import List, Set, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, list_repo_files, hf_hub_download
from rapidfuzz import fuzz, process
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 80.0
DEFAULT_CSV = Path.home() / "datagen" / "dataset_mappings.csv"
DEFAULT_HF_ORG = "DCAgent"

# Eval benchmark repos (folder-based harbor task datasets)
EVAL_BENCHMARKS = {
    "terminal_bench_2": {
        "repo": "DCAgent2/terminal_bench_2",
        "repo_type": "dataset",
    },
    "swebench_verified_100": {
        "repo": "DCAgent2/swebench-verified-random-100-folders",
        "repo_type": "dataset",
    },
    "dev_set_v2": {
        "repo": "DCAgent/dev_set_v2",
        "repo_type": "dataset",
    },
}


# ---------------------------------------------------------------------------
# Load eval benchmark instructions
# ---------------------------------------------------------------------------

def load_eval_instructions(cache_dir: str | None = None) -> dict[str, list[str]]:
    """Download instruction.md files from all eval benchmarks."""
    all_instructions = {}
    api = HfApi()

    for bench_name, bench_info in EVAL_BENCHMARKS.items():
        repo = bench_info["repo"]
        log.info(f"Loading eval instructions from {repo}...")

        files = list(api.list_repo_files(repo, repo_type="dataset"))
        instruction_files = [f for f in files if f.endswith("/instruction.md")]

        instructions = []
        for f in tqdm(instruction_files, desc=f"  {bench_name}"):
            try:
                local_path = hf_hub_download(
                    repo, f, repo_type="dataset",
                    cache_dir=cache_dir,
                )
                with open(local_path) as fh:
                    text = fh.read().strip()
                if len(text) > 50:  # skip trivially short instructions
                    instructions.append(text)
            except Exception as e:
                log.warning(f"  Could not load {f}: {e}")

        all_instructions[bench_name] = instructions
        log.info(f"  {bench_name}: {len(instructions)} instructions loaded")

    return all_instructions


# ---------------------------------------------------------------------------
# Extract task instruction from A1 trace conversations
# ---------------------------------------------------------------------------

def extract_task_instruction(conversations: list[dict]) -> str:
    """Extract the task-specific instruction from a trace conversation.

    The first user message contains the system prompt + task instruction.
    We extract just the task-specific part (after the generic prompt template).
    """
    if not conversations:
        return ""

    first_msg = conversations[0].get("content", "")

    # The task instruction is typically after "Task:" or after the JSON format block
    # Try to find the actual task content
    markers = [
        "## Task\n",
        "## Task:\n",
        "**Task:**\n",
        "Task:\n",
        "Here is your task:\n",
        "Your task is to",
        "\n\n---\n\n",  # separator between prompt template and task
    ]

    for marker in markers:
        idx = first_msg.find(marker)
        if idx >= 0:
            return first_msg[idx:].strip()

    # If no marker found, use the second half of the message
    # (first half is usually the generic system prompt)
    if len(first_msg) > 500:
        return first_msg[len(first_msg) // 2:].strip()

    return first_msg.strip()


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

def check_contamination(
    task_instruction: str,
    eval_instructions: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[bool, float, str]:
    """Check if a task instruction is contaminated against eval benchmarks.

    Returns (is_contaminated, best_score, best_match_preview).
    """
    if not task_instruction or len(task_instruction) < 50:
        return False, 0.0, ""

    # Use rapidfuzz for efficient matching
    result = process.extractOne(
        task_instruction[:2000],  # cap length for performance
        eval_instructions,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )

    if result is None:
        return False, 0.0, ""

    match_str, score, _ = result
    return True, score, match_str[:100]


def check_contamination_batch(
    instructions: list[str],
    eval_instructions: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    n_workers: int | None = None,
) -> list[bool]:
    """Check contamination for a batch of instructions. Returns mask of contaminated indices."""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 16)

    fn = partial(
        _check_one,
        eval_instructions=eval_instructions,
        threshold=threshold,
    )

    contaminated = []
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(fn, instructions, chunksize=100),
            total=len(instructions),
            desc="Checking contamination",
        ))

    return results


def _check_one(instruction: str, eval_instructions: list[str], threshold: float) -> bool:
    is_cont, _, _ = check_contamination(instruction, eval_instructions, threshold)
    return is_cont


# ---------------------------------------------------------------------------
# Decontaminate a single dataset
# ---------------------------------------------------------------------------

def decontaminate_dataset(
    repo_id: str,
    eval_instructions: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[Dataset, int, int]:
    """Load a trace dataset, remove contaminated rows, return clean dataset."""
    log.info(f"Loading {repo_id}...")
    ds = load_dataset(repo_id, split="train")
    original_size = len(ds)

    # Extract task instructions from conversations
    log.info(f"  Extracting instructions from {original_size} rows...")
    instructions = []
    for row in ds:
        convos = row.get("conversations", [])
        instructions.append(extract_task_instruction(convos))

    # Check contamination
    log.info(f"  Checking against {len(eval_instructions)} eval instructions (threshold={threshold})...")
    contaminated = check_contamination_batch(instructions, eval_instructions, threshold)

    # Filter
    keep_indices = [i for i, is_cont in enumerate(contaminated) if not is_cont]
    clean_ds = ds.select(keep_indices)
    n_removed = original_size - len(clean_ds)

    log.info(f"  {repo_id}: {original_size} -> {len(clean_ds)} ({n_removed} removed, {n_removed/original_size*100:.1f}%)")
    return clean_ds, original_size, n_removed


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def run_check(args):
    """Check contamination levels across all A1 datasets (dry run)."""
    eval_instructions = _load_all_eval_instructions()

    registry = _load_registry(args.csv)

    results = []
    for name, repo in sorted(registry.items()):
        try:
            ds = load_dataset(repo, split="train")
            instructions = [extract_task_instruction(row.get("conversations", [])) for row in ds]
            contaminated = check_contamination_batch(instructions, eval_instructions, args.threshold)
            n_cont = sum(contaminated)
            pct = n_cont / len(ds) * 100 if ds else 0
            results.append((name, len(ds), n_cont, pct))
            log.info(f"{name}: {n_cont}/{len(ds)} contaminated ({pct:.1f}%)")
        except Exception as e:
            log.warning(f"{name}: ERROR - {e}")
            results.append((name, 0, 0, 0))

    # Summary
    print(f"\n{'='*70}")
    print(f"{'Dataset':<40} {'Total':>6} {'Contam':>7} {'%':>6}")
    print(f"{'-'*70}")
    total_rows, total_cont = 0, 0
    for name, total, cont, pct in sorted(results, key=lambda x: -x[3]):
        if cont > 0:
            print(f"{name:<40} {total:>6} {cont:>7} {pct:>5.1f}%")
        total_rows += total
        total_cont += cont
    print(f"{'-'*70}")
    print(f"{'TOTAL':<40} {total_rows:>6} {total_cont:>7} {total_cont/total_rows*100 if total_rows else 0:>5.1f}%")


def run_clean(args):
    """Decontaminate all A1 datasets and optionally push clean versions."""
    eval_instructions = _load_all_eval_instructions()
    registry = _load_registry(args.csv)

    for name, repo in sorted(registry.items()):
        try:
            clean_ds, orig, removed = decontaminate_dataset(repo, eval_instructions, args.threshold)
            if removed > 0 and args.push:
                clean_repo = f"{repo}_decontaminated"
                log.info(f"  Pushing clean dataset to {clean_repo}...")
                clean_ds.push_to_hub(clean_repo, private=False)
            elif removed == 0:
                log.info(f"  {name}: clean (no contamination)")
        except Exception as e:
            log.warning(f"{name}: ERROR - {e}")


def run_clean_one(args):
    """Decontaminate a single dataset."""
    eval_instructions = _load_all_eval_instructions()
    clean_ds, orig, removed = decontaminate_dataset(args.dataset, eval_instructions, args.threshold)

    if removed > 0 and args.push:
        clean_repo = f"{args.dataset}_decontaminated"
        log.info(f"Pushing to {clean_repo}...")
        clean_ds.push_to_hub(clean_repo, private=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_all_eval_instructions() -> list[str]:
    """Load and flatten all eval instructions."""
    all_instr = load_eval_instructions()
    flat = []
    for bench, instrs in all_instr.items():
        flat.extend(instrs)
    log.info(f"Total eval instructions: {len(flat)}")
    return flat


def _load_registry(csv_path: str | None = None) -> dict[str, str]:
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
    registry = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "").strip() == "ON HF":
                link = row["hf_sandbox_or_trace_link"].strip()
                if link and "{" not in link:
                    registry[row["a1_name"].strip()] = link
    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Decontaminate A1 datasets against eval benchmarks")
    sub = p.add_subparsers(dest="mode", required=True)

    # Check mode
    cp = sub.add_parser("check", help="Check contamination levels (dry run)")
    cp.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    cp.add_argument("--csv", default=None)

    # Clean all mode
    clp = sub.add_parser("clean", help="Decontaminate all A1 datasets")
    clp.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    clp.add_argument("--csv", default=None)
    clp.add_argument("--push", action="store_true")

    # Clean one mode
    cop = sub.add_parser("clean-one", help="Decontaminate single dataset")
    cop.add_argument("--dataset", required=True, help="HF dataset repo")
    cop.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    cop.add_argument("--push", action="store_true")

    args = p.parse_args()
    {"check": run_check, "clean": run_clean, "clean-one": run_clean_one}[args.mode](args)


if __name__ == "__main__":
    main()
