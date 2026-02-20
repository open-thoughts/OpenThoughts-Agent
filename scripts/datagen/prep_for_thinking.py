#!/usr/bin/env python3
"""
prep_for_thinking.py - Prepare trace datasets for Qwen3 SFT training.

Reformats assistant message content to match LLaMA-Factory's Qwen3
ReasoningTemplate thought_words format:

    <think>\\n{thinking_content}\\n</think>\\n\\n{json_content}

Handles diverse input formats from multiple trace generation pipelines:

  - Proper <think>content</think>{json}  (Kimi-K2T, GLM-4.7, etc.)
  - <think>\\ncontent\\n</think>\\n\\n{json}  (minimax-m2 - already close)
  - content</think>{json}  (DCAgent datasets - missing opening tag)
  - {json} or \\n{json}  (glm46 - no think tags at all)
  - Think tags + markdown fences (```json...```)
  - Fallback: strip everything before first { for unrecognized formats

Importable API:
    from scripts.datagen.prep_for_thinking import reformat_assistant_content
    from scripts.datagen.prep_for_thinking import preprocess_dataset_for_thinking

CLI usage:
    # Dry run (preview stats and samples)
    python -m scripts.datagen.prep_for_thinking \\
        --source DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k \\
        --target penfever/test-output \\
        --dry-run

    # Full run (target defaults to source if omitted)
    python -m scripts.datagen.prep_for_thinking \\
        --source DCAgent2/GLM-4.7-r2egym_sandboxes-maxeps-131k \\
        --target penfever/GLM-4.7-r2egym-thinking-formatted \\
        --private
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

# Target format components (LLaMA-Factory qwen3 ReasoningTemplate thought_words)
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"


# ---------------------------------------------------------------------------
# Core formatting utilities (no heavy dependencies)
# ---------------------------------------------------------------------------


def _clean_rest(rest: str) -> str:
    """Strip markdown code fences and surrounding whitespace from the JSON portion."""
    rest = rest.strip()
    rest = re.sub(r"^```\w*\n?", "", rest)
    rest = re.sub(r"\n?```\s*$", "", rest)
    return rest.strip()


def _build_output(thinking: str, rest: str) -> str:
    """Build output in Qwen3 thought_words format."""
    if thinking:
        return f"{THINK_OPEN}{thinking}{THINK_CLOSE}{rest}"
    return f"{THINK_OPEN}{THINK_CLOSE}{rest}"


def reformat_assistant_content(content: str) -> Tuple[str, str]:
    """Reformat a single assistant message to Qwen3 thought_words format.

    Returns:
        (reformatted_content, format_label) where format_label describes
        which input format was detected.
    """
    # 1. Already in target format
    if content.startswith(THINK_OPEN) and THINK_CLOSE in content:
        return content, "already_qwen3"

    # 2. Proper <think>...</think> tags (with or without inner newlines)
    m = re.match(r"<think>(.*?)</think>(.*)", content, re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        rest = _clean_rest(m.group(2))
        if rest.startswith("{"):
            return _build_output(thinking, rest), "think_tags"
        json_pos = rest.find("{")
        if json_pos >= 0:
            return _build_output(thinking, rest[json_pos:]), "think_tags_extracted_json"
        return _build_output(thinking, rest), "think_tags_no_json"

    # 3. Orphaned </think> (missing opening tag, e.g. DCAgent datasets)
    close_idx = content.find("</think>")
    if close_idx >= 0:
        thinking = content[:close_idx].strip()
        rest = _clean_rest(content[close_idx + len("</think>"):])
        if rest.startswith("{"):
            return _build_output(thinking, rest), "orphaned_close_think"
        json_pos = rest.find("{")
        if json_pos >= 0:
            return _build_output(thinking, rest[json_pos:]), "orphaned_close_think_extracted_json"
        return _build_output(thinking, rest), "orphaned_close_think_no_json"

    # 4. Bare/malformed <think> with no </think> (e.g. model went straight to JSON,
    #    or doubled tags like <think><think>)
    bare_think = re.match(r"^(?:<think>)+\s*(.*)", content.lstrip(), re.DOTALL)
    if bare_think:
        after_tag = bare_think.group(1)
        after_tag_clean = _clean_rest(after_tag)
        json_pos = after_tag_clean.find("{")
        if json_pos >= 0:
            thinking = after_tag_clean[:json_pos].strip()
            return _build_output(thinking, after_tag_clean[json_pos:]), "bare_open_think"

    # 5. No think tags - just JSON (possibly with leading whitespace/newlines)
    stripped = content.strip()
    stripped = _clean_rest(stripped)
    if stripped.startswith("{"):
        return f"{THINK_OPEN}{THINK_CLOSE}{stripped}", "no_think_json"

    # 6. Fallback: strip everything before first {
    json_pos = stripped.find("{")
    if json_pos >= 0:
        return f"{THINK_OPEN}{THINK_CLOSE}{stripped[json_pos:]}", "fallback_strip_to_json"

    # 7. No JSON found at all - return unchanged
    return content, "no_json_unchanged"


# ---------------------------------------------------------------------------
# Parquet-level processing (memory-efficient, uses pyarrow directly)
# ---------------------------------------------------------------------------


def _process_conversations_list(
    conversations_list: list,
    role_tag: str,
    content_tag: str,
    stats: Dict[str, int],
) -> list:
    """Process a batch of conversations (list of list-of-dicts) in place."""
    processed = []
    for convs in conversations_list:
        new_convs = []
        for msg in convs:
            if isinstance(msg, dict) and msg.get(role_tag) == "assistant":
                reformatted, fmt = reformat_assistant_content(msg[content_tag])
                stats[fmt] = stats.get(fmt, 0) + 1
                new_convs.append({**msg, content_tag: reformatted})
            else:
                new_convs.append(msg)
        processed.append(new_convs)
    return processed


def _process_parquet_file(
    input_path: str,
    output_path: str,
    conversations_col: str,
    role_tag: str,
    content_tag: str,
    stats: Dict[str, int],
) -> int:
    """Process a single parquet file, one row-group at a time.

    Returns the number of rows processed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(input_path)
    writer = None
    total_rows = 0

    try:
        for rg_idx in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(rg_idx)
            total_rows += table.num_rows

            if conversations_col not in table.column_names:
                # No conversations column — copy through unchanged
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                continue

            # Convert conversations column to Python, process, convert back
            convs_list = table.column(conversations_col).to_pylist()
            processed = _process_conversations_list(
                convs_list, role_tag, content_tag, stats,
            )

            col_idx = table.column_names.index(conversations_col)
            table = table.set_column(
                col_idx, conversations_col, pa.array(processed),
            )

            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return total_rows


# ---------------------------------------------------------------------------
# Dataset-level processing (requires `datasets` library)
# ---------------------------------------------------------------------------


def preprocess_dataset_for_thinking(
    dataset,
    conversations_col: str = "conversations",
    role_tag: str = "role",
    content_tag: str = "content",
) -> tuple:
    """Process a HF Dataset split, reformatting assistant messages for Qwen3.

    This in-memory version is used by the CLI push-to-hub workflow.  For the
    HPC launch path, use :func:`preprocess_local_dataset` instead (processes
    parquet files one at a time without loading the full dataset into memory).

    Args:
        dataset: A ``datasets.Dataset`` object.
        conversations_col: Column containing the conversation list.
        role_tag: Key for the role field inside each message dict.
        content_tag: Key for the content field inside each message dict.

    Returns:
        (processed_dataset, stats) where *stats* is a dict mapping format
        labels to counts.
    """
    from datasets import Dataset as _Dataset

    stats: Dict[str, int] = {}
    new_rows: list[dict] = []

    for row in dataset:
        convs = row[conversations_col]
        processed_convs = []

        for msg in convs:
            if msg[role_tag] == "assistant":
                reformatted, fmt = reformat_assistant_content(msg[content_tag])
                stats[fmt] = stats.get(fmt, 0) + 1
                processed_convs.append({**msg, content_tag: reformatted})
            else:
                processed_convs.append(msg)

        new_rows.append({**row, conversations_col: processed_convs})

    return _Dataset.from_list(new_rows), stats


def preprocess_local_dataset(
    dataset_path: str,
    *,
    conversations_col: str = "conversations",
    role_tag: str = "role",
    content_tag: str = "content",
    output_dir: str | None = None,
) -> str:
    """Preprocess a local dataset for Qwen3, processing parquet files one at a time.

    This is the main entry point used by ``hpc/launch.py`` to preprocess
    datasets after they are downloaded but before LlamaFactory sees them.
    It never loads the full dataset into memory — each parquet shard is read,
    processed, and written independently.

    Args:
        dataset_path: Path to a local HF-format dataset (snapshot directory).
        conversations_col: Column containing conversations.
        role_tag: Key for role in message dicts.
        content_tag: Key for content in message dicts.
        output_dir: Where to save the processed dataset.  If *None*, saves
            to ``{dataset_path}_thinking_preprocessed``.

    Returns:
        Path to the saved processed dataset directory.
    """
    if output_dir is None:
        output_dir = dataset_path.rstrip("/") + "_thinking_preprocessed"

    # Skip if already preprocessed
    marker = os.path.join(output_dir, ".thinking_preprocessed")
    if os.path.exists(marker):
        print(f"[prep_for_thinking] Already preprocessed: {output_dir}")
        return output_dir

    # Discover parquet files (preserving subdirectory structure)
    parquet_files: List[Tuple[str, str]] = []  # (relative_dir, filename)
    for root, _dirs, files in os.walk(dataset_path):
        for fname in sorted(files):
            if fname.endswith(".parquet"):
                rel_dir = os.path.relpath(root, dataset_path)
                parquet_files.append((rel_dir, fname))

    if not parquet_files:
        print(f"[prep_for_thinking] WARNING: No parquet files found in {dataset_path}")
        return dataset_path

    print(
        f"[prep_for_thinking] Processing {len(parquet_files)} parquet file(s) "
        f"from {dataset_path}"
    )

    all_stats: Dict[str, int] = {}
    total_rows = 0

    for rel_dir, fname in parquet_files:
        input_path = os.path.join(dataset_path, rel_dir, fname)
        out_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        output_path = os.path.join(out_dir, fname)

        rows = _process_parquet_file(
            input_path, output_path,
            conversations_col, role_tag, content_tag,
            all_stats,
        )
        total_rows += rows
        print(f"  {os.path.join(rel_dir, fname)}: {rows} rows")

    # Copy non-parquet files (README, .gitattributes, etc.) so the directory
    # looks like a valid HF dataset repo.
    for root, _dirs, files in os.walk(dataset_path):
        for fname in files:
            if fname.endswith(".parquet"):
                continue
            rel = os.path.relpath(os.path.join(root, fname), dataset_path)
            dest = os.path.join(output_dir, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if not os.path.exists(dest):
                import shutil
                shutil.copy2(os.path.join(root, fname), dest)

    # Print stats
    total_msgs = sum(all_stats.values())
    print(f"[prep_for_thinking] Done: {total_rows} rows, {total_msgs} assistant messages")
    for fmt, count in sorted(all_stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_msgs if total_msgs else 0
        print(f"  {fmt:40s} {count:6d}  ({pct:.1f}%)")

    # Write marker so we don't re-process on retry
    with open(marker, "w") as f:
        f.write("1\n")

    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reformat trace datasets to Qwen3 thought_words format for SFT",
    )
    p.add_argument(
        "--source",
        required=True,
        help="Source HF dataset repo (org/name) or local path",
    )
    p.add_argument(
        "--target",
        default=None,
        help="Target HF dataset repo (org/name). Defaults to --source (overwrite in place).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create target repo as private",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token (defaults to HF_TOKEN env var)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Process data but don't push. Print stats and samples instead.",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Process only first N rows (for testing)",
    )
    p.add_argument(
        "--role-tag",
        default="role",
        help="Key for role in conversation messages (default: role)",
    )
    p.add_argument(
        "--content-tag",
        default="content",
        help="Key for content in conversation messages (default: content)",
    )
    p.add_argument(
        "--conversations-column",
        default="conversations",
        help="Column name for conversations (default: conversations)",
    )
    return p.parse_args()


def main() -> None:
    from datasets import Dataset, DatasetDict, load_dataset
    from huggingface_hub import create_repo

    args = parse_args()
    if args.target is None:
        args.target = args.source
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"[prep] Loading source: {args.source}")
    ds = load_dataset(args.source)
    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    all_stats: Dict[str, int] = {}
    processed_splits: dict[str, Dataset] = {}

    for split_name, split_ds in ds.items():
        if args.max_rows:
            split_ds = split_ds.select(range(min(args.max_rows, len(split_ds))))

        print(f"[prep] Processing split '{split_name}' ({len(split_ds)} rows)...")
        processed, stats = preprocess_dataset_for_thinking(
            split_ds,
            args.conversations_column,
            args.role_tag,
            args.content_tag,
        )
        processed_splits[split_name] = processed

        for k, v in stats.items():
            all_stats[k] = all_stats.get(k, 0) + v

    total = sum(all_stats.values())
    print(f"\n[prep] Format detection stats ({total} assistant messages):")
    for fmt, count in sorted(all_stats.items(), key=lambda x: -x[1]):
        pct = 100 * count / total if total else 0
        print(f"  {fmt:40s} {count:6d}  ({pct:.1f}%)")

    if args.dry_run:
        print("\n[prep] Sample reformatted messages:")
        sample_split = list(processed_splits.values())[0]
        shown = 0
        for i, row in enumerate(sample_split):
            if shown >= 3:
                break
            convs = row[args.conversations_column]
            for msg in convs:
                if msg[args.role_tag] == "assistant":
                    content = msg[args.content_tag]
                    preview = content[:500] + "..." if len(content) > 500 else content
                    print(f"\n  Row {i}, first assistant turn:")
                    print(f"  {repr(preview)}")
                    shown += 1
                    break
        print("\n[prep] Dry run complete. No data pushed.")
        return

    result_ds = DatasetDict(processed_splits)
    print(f"\n[prep] Creating/using target repo: {args.target}")
    create_repo(
        repo_id=args.target,
        repo_type="dataset",
        private=args.private,
        token=token,
        exist_ok=True,
    )

    print(f"[prep] Pushing to {args.target}...")
    result_ds.push_to_hub(
        args.target,
        private=args.private,
        token=token,
        commit_message=f"Reformat to Qwen3 thought_words format (from {args.source})",
    )
    print(f"[prep] Done. https://huggingface.co/datasets/{args.target}")


if __name__ == "__main__":
    main()
