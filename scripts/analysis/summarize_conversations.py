#!/usr/bin/env python3
"""Summarize conversation statistics from a JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import os
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, Iterable, Iterator

# Make tokenizers robust on HPC nodes with strict thread limits
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover - import guard for informative error message
    AutoTokenizer = None  # type: ignore[assignment]

DEFAULT_TOKENIZER_ID = "Qwen/Qwen3-8B"
EXTRA_TEXT_MARKER = "Extra text detected before JSON object"
THINK_MARKERS = ("<think>", "</think>")
TECHNICAL_DIFFICULTIES_MARKER = "Technical difficulties. Please continue with the task."
ACTION_DENIED_MARKER = "NONE of the actions you just requested were performed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute summary statistics over conversations stored in a JSONL file."
        )
    )
    parser.add_argument(
        "jsonl_path",
        help="Path to the JSONL file produced by filter_latest_episodes.py",
    )
    parser.add_argument(
        "--tokenizer-id",
        default=DEFAULT_TOKENIZER_ID,
        help=(
            "Hugging Face tokenizer identifier to use (default: "
            f"{DEFAULT_TOKENIZER_ID})."
        ),
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Failed to parse JSON on line {line_number}: {exc}"
                ) from exc


def _extract_from_message_content(content) -> str:
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, dict):
        # Common schema in which each item has a type and text field.
        text = content.get("text")
        if isinstance(text, str):
            parts.append(text)
    elif isinstance(content, list):
        for item in content:
            parts.append(_extract_from_message_content(item))
    return "\n".join(p for p in parts if p)


def extract_conversation_text(record) -> str:
    if isinstance(record, dict):
        messages = record.get("messages") or record.get("conversations")
        if isinstance(messages, list):
            collected: list[str] = []
            for message in messages:
                if isinstance(message, dict):
                    if "content" in message:
                        collected.append(_extract_from_message_content(message["content"]))
                    for key in ("value", "text"):
                        value = message.get(key)
                        if isinstance(value, str):
                            collected.append(value)
                elif isinstance(message, str):
                    collected.append(message)
            combined = "\n".join(chunk for chunk in collected if chunk)
            if combined:
                return combined
        for field in ("conversation", "text", "prompt", "content"):
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value
    return json.dumps(record, ensure_ascii=False)


def count_turns(record) -> int:
    """Estimate the number of turns in a record."""
    if isinstance(record, dict):
        messages = record.get("messages") or record.get("conversations")
        if isinstance(messages, list):
            return len(messages)
        turn_count = record.get("turn_count")
        if isinstance(turn_count, int):
            return turn_count
    return 0


def compute_stats(
    records: Iterable[Dict], tokenizer
) -> dict:
    stats = {
        "count": 0,
        "total_tokens": 0,
        "max_tokens": None,
        "min_tokens": None,
        "total_turns": 0,
        "max_turns": None,
        "min_turns": None,
        "count_extra_text": 0,
        "count_think": 0,
        "count_technical_difficulties": 0,
        "count_action_denied": 0,
        "reward_count": 0,
        "reward_total": 0.0,
        "reward_max": None,
        "reward_min": None,
    }

    for record in records:
        text = extract_conversation_text(record)
        token_count = len(
            tokenizer.encode(text, add_special_tokens=False, return_tensors=None)
        )
        turns = count_turns(record)

        stats["count"] += 1
        stats["total_tokens"] += token_count
        stats["total_turns"] += turns

        if stats["max_tokens"] is None or token_count > stats["max_tokens"]:
            stats["max_tokens"] = token_count
        if stats["min_tokens"] is None or token_count < stats["min_tokens"]:
            stats["min_tokens"] = token_count
        if stats["max_turns"] is None or turns > stats["max_turns"]:
            stats["max_turns"] = turns
        if stats["min_turns"] is None or turns < stats["min_turns"]:
            stats["min_turns"] = turns

        if EXTRA_TEXT_MARKER in text:
            stats["count_extra_text"] += 1
        if any(marker in text for marker in THINK_MARKERS):
            stats["count_think"] += 1
        if TECHNICAL_DIFFICULTIES_MARKER in text:
            stats["count_technical_difficulties"] += 1
        if ACTION_DENIED_MARKER in text:
            stats["count_action_denied"] += 1

        reward = None
        if isinstance(record, dict):
            reward_candidate = record.get("reward")
            if reward_candidate is None:
                metrics = record.get("metrics")
                if isinstance(metrics, dict):
                    reward_candidate = metrics.get("reward")
            if reward_candidate is not None:
                try:
                    reward = float(reward_candidate)
                except (TypeError, ValueError):
                    reward = None
        if reward is not None:
            stats["reward_count"] += 1
            stats["reward_total"] += reward
            if stats["reward_max"] is None or reward > stats["reward_max"]:
                stats["reward_max"] = reward
            if stats["reward_min"] is None or reward < stats["reward_min"]:
                stats["reward_min"] = reward

    return stats


if __name__ == "__main__":
    args = parse_args()
    path = Path(args.jsonl_path)
    if not path.exists():
        raise SystemExit(f"Input file not found: {path}")

    if AutoTokenizer is None:
        raise SystemExit(
            "transformers package is required to tokenize conversations. Install it via 'pip install transformers'."
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
           args.tokenizer_id, trust_remote_code=True
        )
    except Exception:
        # Fallback to slow tokenizer if fast/Rust backend cannot init thread pool
        os.environ["TRANSFORMERS_NO_FAST_TOKENIZER"] = "1"
        tokenizer = AutoTokenizer.from_pretrained(
           args.tokenizer_id, trust_remote_code=True, use_fast=False
        )

    stats = compute_stats(iter_jsonl(path), tokenizer)

    if stats["count"] == 0:
        print("No conversations processed.")
    else:
        average = stats["total_tokens"] / stats["count"]
        average_turns = stats["total_turns"] / stats["count"]
        print(f"File: {path}")
        print(f"Tokenizer: {args.tokenizer_id}")
        print(f"Conversations processed: {stats['count']}")
        print(f"Max tokens: {stats['max_tokens']}")
        print(f"Min tokens: {stats['min_tokens']}")
        print(f"Average tokens: {average:.2f}")
        print(f"Max turns: {stats['max_turns']}")
        print(f"Min turns: {stats['min_turns']}")
        print(f"Average turns: {average_turns:.2f}")
        print(
            f"Contains '{EXTRA_TEXT_MARKER}': {stats['count_extra_text']}"
        )
        print(
            "Contains '<think>' or '</think>': "
            f"{stats['count_think']}"
        )
        print(
            f"Contains '{TECHNICAL_DIFFICULTIES_MARKER}': "
            f"{stats['count_technical_difficulties']}"
        )
        print(
            f"Contains '{ACTION_DENIED_MARKER}': "
            f"{stats['count_action_denied']}"
        )
        print(
            f"Trials with reward: {stats['reward_count']} (missing: {stats['count'] - stats['reward_count']})"
        )
        if stats["reward_count"]:
            reward_avg = stats["reward_total"] / stats["reward_count"]
            print(f"Reward max: {stats['reward_max']}")
            print(f"Reward min: {stats['reward_min']}")
            print(f"Reward average: {reward_avg:.4f}")
