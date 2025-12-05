#!/usr/bin/env python3
"""
Join multiple Hugging Face dataset repos (with identical schema) in order and
push the concatenated result to a target repo.

Example:
  python -m scripts.datagen.join_hf_repos \
    --sources DCAgent/code_contests-GLM-4.6-traces DCAgent/code_contests-GLM-4.6-traces-chunk001 \
    --target DCAgent/code_contests-GLM-4.6-traces-joined \
    --private
"""

from __future__ import annotations

import argparse
import os
from typing import List

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from huggingface_hub import HfApi, create_repo

try:
    from data.commons import clean_empty_structs  # best-effort cleanup before push
except Exception:
    def clean_empty_structs(dataset: Dataset) -> Dataset:  # type: ignore
        return dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Join HF dataset repos (identical schema) and push to target repo",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source HF dataset repos (org/name), in concatenation order",
    )
    p.add_argument(
        "--target",
        required=True,
        help="Target HF dataset repo (org/name)",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create target repo as private",
    )
    p.add_argument(
        "--token",
        default=None,
        help="HF token (defaults to HF_TOKEN environment variable)",
    )
    p.add_argument(
        "--commit_message",
        default="Join datasets",
        help="Commit message for push_to_hub",
    )
    return p.parse_args()


def _ensure_repo(repo_id: str, private: bool, token: str | None) -> None:
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, token=token, exist_ok=True)


def _load_all(sources: List[str]) -> List[DatasetDict]:
    dsds: List[DatasetDict] = []
    for src in sources:
        dsd = load_dataset(src)  # Expect DatasetDict
        if isinstance(dsd, Dataset):  # Fallback if a single split dataset is returned
            dsd = DatasetDict({"train": dsd})
        dsds.append(dsd) 
    return dsds


def _merge_by_split(sources: List[DatasetDict]) -> DatasetDict:
    base = sources[0]
    merged: dict[str, Dataset] = {}
    for split in base.keys():
        parts: List[Dataset] = []
        for dsd in sources:
            if split not in dsd:
                raise ValueError(f"Split '{split}' missing in one of the sources")
            parts.append(dsd[split])
        out = concatenate_datasets(parts, info=parts[0].info)
        out = clean_empty_structs(out)
        merged[split] = out
    return DatasetDict(merged)


def main() -> None:
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    print(f"[join] Loading {len(args.sources)} source repo(s)...")
    src_dsds = _load_all(args.sources)

    # Basic schema sanity check: ensure all have the same set of splits
    base_splits = set(src_dsds[0].keys())
    for i, dsd in enumerate(src_dsds[1:], start=2):
        splits = set(dsd.keys())
        if splits != base_splits:
            raise ValueError(f"Source #{i} splits {sorted(splits)} != base splits {sorted(base_splits)}")

    print("[join] Concatenating splits in order...")
    merged = _merge_by_split(src_dsds)

    print(f"[join] Creating/using target repo: {args.target}")
    _ensure_repo(args.target, args.private, token)

    print(f"[join] Pushing merged dataset to {args.target}...")
    merged.push_to_hub(args.target, private=args.private, token=token, commit_message=args.commit_message)  # type: ignore[arg-type]
    print(f"[join] Done. https://huggingface.co/datasets/{args.target}")


if __name__ == "__main__":
    main()

