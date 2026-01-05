#!/usr/bin/env python3
"""Compute total tokenizer length of the `conversations` field for a HF dataset."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given a HuggingFace dataset repo, load a split and report the aggregate "
            "token count of the serialized `conversations` field using the Qwen3 tokenizer."
        )
    )
    parser.add_argument(
        "repo_id",
        help="Dataset repository on HuggingFace Hub (e.g. org/name).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional dataset config/name.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load (default: train).",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3-8B",
        help="Tokenizer repo to use (default: Qwen/Qwen3-8B).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code=True to AutoTokenizer.from_pretrained (needed for some custom tokenizers).",
    )
    return parser.parse_args()


def serialize_conversations(raw_value: Any) -> str:
    if raw_value is None:
        return ""
    if isinstance(raw_value, str):
        return raw_value
    try:
        return json.dumps(raw_value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        return str(raw_value)


def main() -> None:
    args = parse_args()

    print(f"[tokens] Loading dataset {args.repo_id} (split={args.split}, config={args.config})...")
    dataset = (
        load_dataset(args.repo_id, args.config, split=args.split)
        if args.config
        else load_dataset(args.repo_id, split=args.split)
    )

    print(f"[tokens] Loading tokenizer {args.tokenizer}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code or ("Qwen3" in args.tokenizer),
    )

    total_tokens = 0
    missing_count = 0

    print("[tokens] Iterating over samples...")
    for example in dataset:
        if "conversations" not in example:
            missing_count += 1
            continue
        serialized = serialize_conversations(example["conversations"])
        # len of encode without special tokens to match raw text size
        total_tokens += len(tokenizer.encode(serialized, add_special_tokens=False))

    processed = len(dataset) if hasattr(dataset, "__len__") else "unknown"

    if missing_count:
        print(f"[tokens] Warning: skipped {missing_count} example(s) missing `conversations`.")

    print(f"[tokens] Examples processed: {processed}")
    print(f"[tokens] Aggregate tokens: {total_tokens:,}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted.")
