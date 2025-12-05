"""
Utility to inspect Harbor trace directories using Harbor's trace utilities.

Example:
    python -m scripts.datagen.print_trace_contents \
        --trace_dir /Users/benjaminfeuer/Documents/evaltraces/tezos-0014_copy7__dHXz7gB \
        --limit 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from harbor.utils import traces_utils


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _format_messages(messages: Sequence[Mapping[str, object]]) -> Iterable[str]:
    for msg in messages or []:
        role = str(msg.get("role", "unknown"))
        content = _stringify(msg.get("content", ""))
        yield f"[{role}] {content}"


def _format_reasoning(reasoning_payload) -> str:
    if not reasoning_payload:
        return ""
    return _stringify(reasoning_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Harbor trace contents using harbor.utils.traces_utils"
    )
    parser.add_argument(
        "--trace_dir",
        required=True,
        help="Path to a Harbor trace directory (job root or specific trial directory).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under trace_dir for trials (default: False).",
    )
    parser.add_argument(
        "--include_reasoning",
        action="store_true",
        help="Whether to include reasoning content when exporting traces.",
    )
    parser.add_argument(
        "--sharegpt",
        action="store_true",
        help="Also produce ShareGPT formatted conversations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace_root = Path(args.trace_dir).expanduser().resolve()
    if not trace_root.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_root}")

    dataset = traces_utils.export_traces(
        trace_root,
        recursive=args.recursive,
        episodes="last",
        to_sharegpt=args.sharegpt,
        verbose=True,
        include_reasoning=args.include_reasoning,
    )

    total = len(dataset)
    if total == 0:
        print(f"[trace-printer] No episodes found in {trace_root}")
        return

    rows = list(dataset)
    row_to_print = None
    for row in reversed(rows):
        conversations = row.get("conversations") or []
        if conversations:
            row_to_print = row
            break
    if row_to_print is None:
        row_to_print = rows[-1]

    header = [
        "Episode (last)",
        f"name={row_to_print.get('episode')}",
        f"task={row_to_print.get('task')}",
        f"agent={row_to_print.get('agent')}",
        f"model={row_to_print.get('model')}",
    ]
    print("=" * 80)
    print(" | ".join(header))
    print("-" * 80)
    for line in _format_messages(row_to_print.get("conversations") or []):
        print(line)

    if args.sharegpt and row_to_print.get("conversations_sharegpt"):
        print("\n[ShareGPT format]")
        for share_msg in row_to_print["conversations_sharegpt"]:
            print(f"[{share_msg.get('from')}] {share_msg.get('value')}")

    if args.include_reasoning:
        reasoning = row_to_print.get("reasoning_content") or row_to_print.get("reasoning")
        if reasoning:
            print("\n[Reasoning]")
            print(_format_reasoning(reasoning))
    print()


if __name__ == "__main__":
    main()
