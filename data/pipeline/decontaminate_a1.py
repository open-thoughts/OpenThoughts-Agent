#!/usr/bin/env python3
"""
Decontamination for A1 sandbox datasets against harbor eval benchmarks.

Extracts instruction.md from each task's task_binary in A1 sandbox datasets
and fuzzy-matches against instruction.md files from eval benchmarks
(terminal_bench_2, swebench-verified, dev_set_v2).

Usage:
    # Check contamination levels (dry run):
    python decontaminate_a1.py check --threshold 80

    # Check a single sandbox dataset:
    python decontaminate_a1.py check-one \
        --dataset DCAgent/exp_rpt_stack-bash \
        --threshold 80

    # Decontaminate all and push clean versions:
    python decontaminate_a1.py clean --threshold 80 --push
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import multiprocessing as mp
import tarfile
from functools import partial
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from rapidfuzz import fuzz, process
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 80.0
DEFAULT_CSV = Path.home() / "datagen" / "dataset_mappings.csv"

# Eval benchmark repos (folder-based harbor task datasets)
EVAL_BENCHMARKS = {
    "terminal_bench_2": "DCAgent2/terminal_bench_2",
    "swebench_verified_100": "DCAgent2/swebench-verified-random-100-folders",
    "dev_set_v2": "DCAgent/dev_set_v2",
}


# ---------------------------------------------------------------------------
# Load eval benchmark instructions from folder-based repos
# ---------------------------------------------------------------------------

def load_eval_instructions(cache_dir: str | None = None) -> list[str]:
    """Download all instruction.md files from eval benchmarks."""
    api = HfApi()
    all_instructions = []

    for bench_name, repo in EVAL_BENCHMARKS.items():
        log.info(f"Loading eval instructions from {repo}...")
        files = list(api.list_repo_files(repo, repo_type="dataset"))
        instr_files = [f for f in files if f.endswith("/instruction.md")]

        count = 0
        for f in tqdm(instr_files, desc=f"  {bench_name}"):
            try:
                local_path = hf_hub_download(repo, f, repo_type="dataset", cache_dir=cache_dir)
                with open(local_path) as fh:
                    text = fh.read().strip()
                if len(text) > 50:
                    all_instructions.append(text)
                    count += 1
            except Exception as e:
                pass

        log.info(f"  {bench_name}: {count} instructions")

    log.info(f"Total eval instructions: {len(all_instructions)}")
    return all_instructions


# ---------------------------------------------------------------------------
# Extract instruction.md from sandbox task_binary
# ---------------------------------------------------------------------------

def extract_instruction_from_task_binary(task_binary: bytes) -> str:
    """Extract instruction.md content from a tar-archived task_binary."""
    if not task_binary:
        return ""
    try:
        with tarfile.open(fileobj=io.BytesIO(task_binary)) as tar:
            for member in tar.getmembers():
                if member.name.endswith("instruction.md"):
                    f = tar.extractfile(member)
                    if f:
                        return f.read().decode("utf-8", errors="replace").strip()
    except Exception:
        pass
    return ""


def extract_instructions_from_dataset(ds: Dataset) -> list[str]:
    """Extract instruction.md from all rows in a sandbox dataset."""
    instructions = []
    for row in tqdm(ds, desc="  Extracting instructions"):
        tb = row.get("task_binary", b"")
        instructions.append(extract_instruction_from_task_binary(tb))
    return instructions


# ---------------------------------------------------------------------------
# Also extract from trace datasets (conversations column)
# ---------------------------------------------------------------------------

def extract_instruction_from_conversation(conversations: list[dict]) -> str:
    """Extract task instruction from trace conversation's second user message.

    In harbor traces, the first user message is the system prompt + initial terminal.
    The actual task instruction is often in the first message after markers like
    'Task:' or embedded in the terminal output section.
    """
    if not conversations:
        return ""

    # Concatenate all user messages and look for task content
    all_user = " ".join(
        msg.get("content", "")[:3000]
        for msg in conversations
        if msg.get("role") == "user"
    )

    # The instruction is usually between "Task:" and the terminal output
    # Try to find it
    markers = ["## Task\n", "## Task:\n", "**Task:**", "Task:\n", "Your task:"]
    for marker in markers:
        idx = all_user.find(marker)
        if idx >= 0:
            # Get text from marker to next section or end
            text = all_user[idx + len(marker):]
            # Stop at terminal output or next section
            for stop in ["\n**Current Terminal", "\n## ", "\n```"]:
                stop_idx = text.find(stop)
                if stop_idx > 0:
                    text = text[:stop_idx]
            return text.strip()

    # Fallback: return chunk from middle of first message (skip system prompt)
    first = conversations[0].get("content", "")
    if len(first) > 1000:
        return first[500:].strip()
    return first.strip()


# ---------------------------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------------------------

# No truncation — compare full instruction text for accuracy


def _check_one(instruction: str, eval_instructions: list[str], threshold: float) -> Tuple[bool, float]:
    """Check one instruction against all eval instructions.

    Compares full instruction text — no truncation.
    """
    if not instruction or len(instruction) < 50:
        return False, 0.0

    result = process.extractOne(
        instruction,
        eval_instructions,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=threshold,
    )
    if result is None:
        return False, 0.0
    return True, result[1]


def check_contamination_batch(
    instructions: list[str],
    eval_instructions: list[str],
    threshold: float = DEFAULT_THRESHOLD,
    n_workers: int | None = None,
) -> list[Tuple[bool, float]]:
    """Check contamination for a batch. Returns list of (is_contaminated, score)."""
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 16)

    fn = partial(_check_one, eval_instructions=eval_instructions, threshold=threshold)

    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap(fn, instructions, chunksize=100),
            total=len(instructions),
            desc="  Checking contamination",
        ))
    return results


# ---------------------------------------------------------------------------
# Load sandbox or trace dataset and extract instructions
# ---------------------------------------------------------------------------

def load_and_extract(repo_id: str) -> Tuple[Dataset, list[str]]:
    """Load a dataset and extract task instructions.

    Auto-detects whether it's a sandbox dataset (has task_binary column)
    or a trace dataset (has conversations column).
    """
    ds = load_dataset(repo_id, split="train")
    cols = ds.column_names

    if "task_binary" in cols:
        log.info(f"  Sandbox dataset ({len(ds)} rows)")
        instructions = extract_instructions_from_dataset(ds)
    elif "conversations" in cols:
        log.info(f"  Trace dataset ({len(ds)} rows)")
        instructions = [
            extract_instruction_from_conversation(row.get("conversations", []))
            for row in tqdm(ds, desc="  Extracting from conversations")
        ]
    else:
        log.warning(f"  Unknown format. Columns: {cols}")
        instructions = [""] * len(ds)

    non_empty = sum(1 for i in instructions if len(i) > 50)
    log.info(f"  Extracted {non_empty}/{len(instructions)} non-empty instructions")
    return ds, instructions


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------

def _load_registry(csv_path: str | None) -> dict[str, str]:
    """Load A1 sandbox/trace dataset mapping from CSV."""
    csv_path = Path(csv_path) if csv_path else DEFAULT_CSV
    registry = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            status = row.get("status", "").strip()
            if status in ("ON HF", "SANDBOX EXISTS", "COMPLETED CHECK UPLOAD"):
                link = row["hf_sandbox_or_trace_link"].strip()
                if link and "{" not in link:
                    registry[row["a1_name"].strip()] = link
    return registry


def run_check(args):
    """Check contamination levels across all A1 datasets."""
    eval_instructions = load_eval_instructions()
    registry = _load_registry(args.csv)

    results = []
    for name, repo in sorted(registry.items()):
        log.info(f"\n{name} ({repo}):")
        try:
            ds, instructions = load_and_extract(repo)
            checks = check_contamination_batch(instructions, eval_instructions, args.threshold)
            n_cont = sum(1 for is_cont, _ in checks if is_cont)
            max_score = max((s for _, s in checks), default=0)
            pct = n_cont / len(ds) * 100 if ds else 0
            results.append((name, len(ds), n_cont, pct, max_score))
            log.info(f"  Result: {n_cont}/{len(ds)} contaminated ({pct:.1f}%), max_score={max_score:.1f}")
        except Exception as e:
            log.warning(f"  ERROR: {e}")
            results.append((name, 0, 0, 0, 0))

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Dataset':<40} {'Total':>6} {'Contam':>7} {'%':>6} {'MaxScore':>9}")
    print(f"{'-'*80}")
    total_rows, total_cont = 0, 0
    for name, total, cont, pct, max_s in sorted(results, key=lambda x: -x[3]):
        flag = " ***" if cont > 0 else ""
        print(f"{name:<40} {total:>6} {cont:>7} {pct:>5.1f}% {max_s:>8.1f}{flag}")
        total_rows += total
        total_cont += cont
    print(f"{'-'*80}")
    pct_total = total_cont / total_rows * 100 if total_rows else 0
    print(f"{'TOTAL':<40} {total_rows:>6} {total_cont:>7} {pct_total:>5.1f}%")


def run_check_one(args):
    """Check contamination for a single dataset."""
    eval_instructions = load_eval_instructions()
    log.info(f"\nChecking {args.dataset}:")
    ds, instructions = load_and_extract(args.dataset)
    checks = check_contamination_batch(instructions, eval_instructions, args.threshold)

    n_cont = sum(1 for is_cont, _ in checks if is_cont)
    pct = n_cont / len(ds) * 100 if ds else 0
    log.info(f"\nResult: {n_cont}/{len(ds)} contaminated ({pct:.1f}%)")

    # Show contaminated examples
    if n_cont > 0:
        log.info("\nContaminated examples:")
        for i, (is_cont, score) in enumerate(checks):
            if is_cont:
                log.info(f"  Row {i}: score={score:.1f}, instruction={instructions[i][:150]}...")
                if sum(1 for j in range(i + 1) if checks[j][0]) >= 5:
                    log.info(f"  ... and {n_cont - 5} more")
                    break


def run_clean(args):
    """Decontaminate all A1 datasets and optionally push clean versions."""
    eval_instructions = load_eval_instructions()
    registry = _load_registry(args.csv)

    for name, repo in sorted(registry.items()):
        log.info(f"\n{name} ({repo}):")
        try:
            ds, instructions = load_and_extract(repo)
            checks = check_contamination_batch(instructions, eval_instructions, args.threshold)

            keep = [i for i, (is_cont, _) in enumerate(checks) if not is_cont]
            n_removed = len(ds) - len(keep)

            if n_removed > 0:
                clean_ds = ds.select(keep)
                log.info(f"  Removed {n_removed} contaminated rows ({len(ds)} -> {len(clean_ds)})")
                if args.push:
                    clean_repo = f"{repo}_decontaminated"
                    clean_ds.push_to_hub(clean_repo, private=False)
                    log.info(f"  Pushed to {clean_repo}")
            else:
                log.info(f"  Clean (0 contaminated)")
        except Exception as e:
            log.warning(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Decontaminate A1 sandbox/trace datasets")
    sub = p.add_subparsers(dest="mode", required=True)

    cp = sub.add_parser("check", help="Check contamination (dry run)")
    cp.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    cp.add_argument("--csv", default=None)

    cop = sub.add_parser("check-one", help="Check single dataset")
    cop.add_argument("--dataset", required=True)
    cop.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)

    clp = sub.add_parser("clean", help="Decontaminate all and push")
    clp.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    clp.add_argument("--csv", default=None)
    clp.add_argument("--push", action="store_true")

    args = p.parse_args()
    {"check": run_check, "check-one": run_check_one, "clean": run_clean}[args.mode](args)


if __name__ == "__main__":
    main()
