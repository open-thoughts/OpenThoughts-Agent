#!/usr/bin/env python3
"""Build raw JSONL subsets of allenai/Sera-4.5A-Full-T1 for axolotl SFT.

v3 preserves the original row schema verbatim (no shareGPT flattening). Each
output row is the original Sera row plus a `source` field pointing back to
allenai/Sera-4.5A-Full-T1.

Target repos on HF (laion/):
  - Sera-4.5A-Full-T1-v3          (full 72,118 rows)
  - Sera-4.5A-Full-T1-v3-316
  - Sera-4.5A-Full-T1-v3-1000
  - Sera-4.5A-Full-T1-v3-3160
  - Sera-4.5A-Full-T1-v3-10000
  - Sera-4.5A-Full-T1-v3-31600

Skipped: 100000 (dataset has only 72,118 rows).

Uploads raw JSONL (via HfApi.upload_file) to mirror the allenai repo layout —
no HF-datasets parquet conversion. Axolotl consumes this with:

    datasets:
      - path: laion/Sera-4.5A-Full-T1-v3-<SIZE>
        data_files:
          - sera-4.5a-full-t1_v3_<SIZE>.jsonl
        type: chat_template
        field_messages: messages
        ds_type: json
        message_field_training: train
"""
import argparse
import json
import os
import random
import shutil
from pathlib import Path

from huggingface_hub import HfApi


SERA_JSONL = "/Users/benjaminfeuer/.cache/huggingface/hub/datasets--allenai--Sera-4.5A-Full-T1/snapshots/6e97fe0156fc2a89ee11bb565f4e0e21617ef9ca/sera-4.5a-full-t1_72118_string_enriched.jsonl"
SOURCE_TAG = "allenai/Sera-4.5A-Full-T1"
TARGET_BASE = "laion/Sera-4.5A-Full-T1-v3"
SUBSET_SIZES = [316, 1000, 3160, 10000, 31600]   # 100000 skipped (dataset is 72k)
STAGING_DIR = Path("/Users/benjaminfeuer/Documents/scripts_dataset_build/_sera_v3_staging")
SEED = 42


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def add_train_key(messages):
    """Mirror sera/datagen/data/postprocess/utils.py::add_train_key — only
    assistant messages contribute to loss. Mutates in-place."""
    for m in messages:
        m["train"] = (m.get("role") == "assistant")
    return messages


def write_subsets(jsonl_path, staging_dir, sizes, full_size, seed):
    """Two-pass: (1) build random permutation, (2) write each subset + full copy."""
    staging_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    idxs = list(range(full_size))
    random.shuffle(idxs)

    # Build index sets per subset size for O(1) lookup during streaming
    size_to_idxset = {}
    for s in sizes:
        if s > full_size:
            print(f"[skip] size={s} (only {full_size} rows)", flush=True)
            continue
        size_to_idxset[s] = set(idxs[:s])

    # Output files: one per size + one for full
    size_to_file = {}
    for s in size_to_idxset:
        p = staging_dir / f"sera-4.5a-full-t1_v3_{s}.jsonl"
        if p.exists():
            p.unlink()
        size_to_file[s] = open(p, "w")
    full_path = staging_dir / f"sera-4.5a-full-t1_v3_{full_size}.jsonl"
    if full_path.exists():
        full_path.unlink()
    full_f = open(full_path, "w")

    counts = {s: 0 for s in size_to_idxset}
    counts["full"] = 0

    with open(jsonl_path) as src:
        for i, line in enumerate(src):
            if i % 10000 == 0 and i > 0:
                print(f"  streamed {i}/{full_size}", flush=True)
            # Decode → add source → add `train` field per message → re-encode
            row = json.loads(line)
            row["source"] = SOURCE_TAG
            # messages is stored as a JSON-string in the raw layout; re-serialize
            # after injecting `train: bool` so axolotl's `message_field_training:
            # train` works out of the box (matches SERA's add_train_key).
            try:
                msgs = json.loads(row["messages"])
                add_train_key(msgs)
                row["messages"] = json.dumps(msgs, ensure_ascii=False)
            except Exception as e:
                print(f"  [warn] line {i}: failed to inject train key: {e}", flush=True)
            out = json.dumps(row, ensure_ascii=False) + "\n"

            # Full
            full_f.write(out)
            counts["full"] += 1

            # Subsets
            for s, idxset in size_to_idxset.items():
                if i in idxset:
                    size_to_file[s].write(out)
                    counts[s] += 1

    full_f.close()
    for f in size_to_file.values():
        f.close()

    for k, v in counts.items():
        label = f"full ({full_size})" if k == "full" else f"subset {k}"
        print(f"[write] {label}: {v} rows", flush=True)

    # Return path map
    paths = {s: staging_dir / f"sera-4.5a-full-t1_v3_{s}.jsonl" for s in size_to_idxset}
    paths["full"] = full_path
    return paths


def make_readme(target_repo, size, full_size):
    return f"""---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - sft
  - agent
  - swe-bench
  - axolotl
  - openai-messages
---

# {target_repo}

Subset of [allenai/Sera-4.5A-Full-T1](https://huggingface.co/datasets/allenai/Sera-4.5A-Full-T1).

**Size**: {size:,} rows (full dataset: {full_size:,} rows).

**Format**: Raw JSONL, OpenAI-native messages layout. Preserves the original `messages`
field (as JSON string), `instance_id`, `rollout_patch`, `func_name`, `func_path`,
`problem_statement`, `target_patch`, `docker_image`. Adds a `source` field pointing
back to the parent dataset.

Each assistant message carries a native `tool_calls` array (OpenAI tool-calling format)
and a `train: bool` flag for per-message loss masking — these are **not** flattened
into shareGPT. Intended for direct consumption by [axolotl](https://github.com/axolotl-ai-cloud/axolotl)
with `type: chat_template`, `chat_template: chatml`, `message_field_training: train`.

Sampling: deterministic random, seed=42, row-indexed into the full dataset.

## Usage (axolotl)

```yaml
datasets:
  - path: {target_repo}
    data_files:
      - sera-4.5a-full-t1_v3_{size}.jsonl
    type: chat_template
    field_messages: messages
    ds_type: json
    message_field_training: train
chat_template: chatml
```
"""


def push_dataset(api, target_repo, jsonl_path, size, full_size):
    print(f"[push] creating repo {target_repo}", flush=True)
    api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

    readme_tmp = jsonl_path.parent / f"README_{size}.md"
    readme_tmp.write_text(make_readme(target_repo, size, full_size))

    print(f"[push] uploading {jsonl_path.name} ({jsonl_path.stat().st_size / 1e9:.2f} GB) -> {target_repo}", flush=True)
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo=jsonl_path.name,
        repo_id=target_repo,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(readme_tmp),
        path_in_repo="README.md",
        repo_id=target_repo,
        repo_type="dataset",
    )
    readme_tmp.unlink()
    print(f"[push]  done {target_repo}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true", help="Don't upload the full (72118) dataset")
    ap.add_argument("--only-full", action="store_true", help="Only upload the full dataset (skip subsets)")
    ap.add_argument("--build-only", action="store_true", help="Write JSONL files locally, skip HF uploads")
    args = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    full_size = count_lines(SERA_JSONL)
    print(f"[src] {SERA_JSONL} has {full_size} rows", flush=True)

    paths = write_subsets(SERA_JSONL, STAGING_DIR, SUBSET_SIZES, full_size, SEED)

    if args.build_only:
        print("[build-only] skipping uploads", flush=True)
        return

    if not args.only_full:
        for size in SUBSET_SIZES:
            if size > full_size:
                continue
            repo = f"{TARGET_BASE}-{size}"
            push_dataset(api, repo, paths[size], size, full_size)

    if not args.skip_full:
        push_dataset(api, TARGET_BASE, paths["full"], full_size, full_size)


if __name__ == "__main__":
    main()
