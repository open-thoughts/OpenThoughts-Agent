#!/usr/bin/env python3
"""
Common utilities for neulab/agent-data-collection dataset subsets
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _extract_std_question(entry: dict) -> Optional[str]:
    details = entry.get("details")
    if isinstance(details, dict):
        desc = details.get("task_description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()

    content = entry.get("content")
    if isinstance(content, list):
        for chunk in content:
            text = chunk.get("content")
            if isinstance(text, str) and text.strip():
                return text.strip()
    text_id = entry.get("id")
    if isinstance(text_id, str):
        return text_id.strip()
    return None


def _extract_raw_question(entry: dict) -> Optional[str]:
    for key in ("confirmed_task", "task_description"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    details = entry.get("details")
    if isinstance(details, dict):
        desc = details.get("task_description")
        if isinstance(desc, str) and desc.strip():
            return desc.strip()

    actions = entry.get("actions")
    if isinstance(actions, list):
        for action in actions:
            text = action.get("template")
            if isinstance(text, str) and text.strip():
                return text.strip()
    return None


def _load_mind2web_questions() -> List[str]:
    repo_id = "neulab/agent-data-collection"
    files: List[tuple[str, Callable[[dict], Optional[str]]]] = [
        ("mind2web/full_std.jsonl", _extract_std_question),
        ("mind2web/full_raw.jsonl", _extract_raw_question),
    ]
    questions: List[str] = []
    seen: set[str] = set()

    for filename, extractor in files:
        try:
            local_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
            )
        except Exception as exc:
            print(f"Warning: unable to download {filename} ({exc}); skipping.")
            continue

        for record in _iter_jsonl(local_path):
            question = extractor(record)
            if not question:
                continue
            if question in seen:
                continue
            seen.add(question)
            questions.append(question)

    print(f"Extracted {len(questions)} questions from Mind2Web jsonl files")
    return questions


SPECIAL_SUBSET_HANDLERS = {
    "mind2web": _load_mind2web_questions,
}


def extract_questions_from_subset(subset_name: str) -> List[str]:
    """
    Load a subset from neulab/agent-data-collection and extract the first human question from each conversation.

    Args:
        subset_name: Name of the subset to load

    Returns:
        List of instruction strings (first human message from each conversation)
    """
    if subset_name in SPECIAL_SUBSET_HANDLERS:
        return SPECIAL_SUBSET_HANDLERS[subset_name]()

    print(f"Loading neulab/agent-data-collection subset: {subset_name}")
    try:
        dataset = load_dataset("neulab/agent-data-collection", subset_name, split="train")
    except Exception as e:
        print(f"Error loading subset {subset_name}: {e}")
        return []

    print(f"Loaded {len(dataset)} examples")

    questions = []
    for i, item in enumerate(dataset):
        conversations = item.get("conversations", [])

        instruction = ""
        for turn in conversations:
            if turn.get("from") == "human":
                instruction = turn.get("value", "")
                break

        if not instruction:
            print(f"Warning: No instruction found in conversation {i}")
            instruction = "No instruction found"

        questions.append(instruction)

    print(f"Extracted {len(questions)} questions from {subset_name}")
    return questions
