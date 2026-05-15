#!/usr/bin/env python3
"""Build CoderForge-Preview v3 row-subsets from the pre-tokenized trajectories.

Source: togethercomputer/CoderForge-Preview/trajectories-tokenized_qwencoder
        (4 slugs × 224 shards = 896 parquets, ~413k rows total; tokenizer matches Qwen3-8B).

Axolotl auto-detects pre-tokenized datasets via the presence of `input_ids` +
`attention_mask` + `labels`. The upstream data has input_ids + labels but NOT
attention_mask, so we inject attention_mask = [1] * len(input_ids) per row.

Schema per subset row:
  trajectory_id (str), reward (double), chat_template_applied (str),
  input_ids (list[int32]), labels (list[int64]), attention_mask (list[int8]),
  source (str, always "togethercomputer/CoderForge-Preview/trajectories-tokenized_qwencoder")

Target repos on HF (laion/):
  - CoderForge-Preview-v3           (full ~413k rows)
  - CoderForge-Preview-v3-316
  - CoderForge-Preview-v3-1000
  - CoderForge-Preview-v3-3160
  - CoderForge-Preview-v3-10000
  - CoderForge-Preview-v3-31600
  - CoderForge-Preview-v3-100000

Streaming one shard at a time keeps memory bounded.
"""
import argparse
import gc
import glob
import json
import os
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, snapshot_download


SRC_REPO = "togethercomputer/CoderForge-Preview"
SRC_SUBSET_DIR = "trajectories-tokenized_qwencoder"
SOURCE_TAG = f"{SRC_REPO}/{SRC_SUBSET_DIR}"
TARGET_BASE = "laion/CoderForge-Preview-v3"
SIZES = [316, 1000, 3160, 10000, 31600, 100000]   # full uploaded as TARGET_BASE separately
# Per the CoderForge-Preview blog (https://www.together.ai/blog/coderforge-preview)
# the authors trained only on successful+test-passing+license-filtered trajectories —
# this is the `filtered_reward1` slug (155,144 rows). The other 3 slugs (R2E_Gym,
# SWE_Rebench, SWE_Smith) are the raw pre-filter pool, NOT their training set.
# We sample only from filtered_reward1 to match their methodology.
SLUGS = ["filtered_reward1"]
SEED = 42

STAGING = Path("/e/data1/datasets/playground/ot-baf/_cf_v3_staging")

# Schema we emit per row (adds attention_mask + source to source schema).
OUT_SCHEMA = pa.schema([
    ("trajectory_id", pa.string()),
    ("reward", pa.float64()),
    ("chat_template_applied", pa.string()),
    ("input_ids", pa.list_(pa.int32())),
    ("attention_mask", pa.list_(pa.int8())),
    ("labels", pa.list_(pa.int64())),
    ("source", pa.string()),
])


def count_source_rows(source_dir):
    """Count total rows across all shards — needed for global indexing."""
    total = 0
    per_slug = {}
    for slug in SLUGS:
        shards = sorted(glob.glob(f"{source_dir}/{slug}-*-of-*.parquet"))
        n = 0
        for p in shards:
            n += pq.ParquetFile(p).metadata.num_rows
        per_slug[slug] = n
        total += n
        print(f"  {slug}: {len(shards)} shards, {n:,} rows")
    return total, per_slug


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
  - pretokenized
---

# {target_repo}

Row-subset of the pre-tokenized trajectories in
[togethercomputer/CoderForge-Preview](https://huggingface.co/datasets/togethercomputer/CoderForge-Preview)
(`trajectories-tokenized_qwencoder` subset).

**Size**: {size:,} rows (source: {full_size:,} across 4 slugs).

**Format**: native pre-tokenized data for Qwen3 (tokenizer shared with Qwen2.5-Coder / Qwen3-Coder / Qwen3-8B).
Per row columns:
- `input_ids: list[int32]`
- `attention_mask: list[int8]` (all 1s; added by this subsetter so axolotl's
  auto-detection of pre-tokenized datasets triggers — upstream only had
  input_ids + labels)
- `labels: list[int64]` (with `-100` masks already applied)
- `chat_template_applied: str` (decoded render for debugging)
- `trajectory_id: str`, `reward: float64`
- `source: str` (always `"togethercomputer/CoderForge-Preview/trajectories-tokenized_qwencoder"`)

Sampled deterministically (seed=42) from a concatenation of all 4 source slugs
(R2E_Gym, SWE_Rebench, SWE_Smith, filtered_reward1). Row subsets are nested.

## Usage (axolotl)

```yaml
datasets:
  - path: {target_repo}
chat_template: chatml
sequence_len: 32768   # sequences in the upstream data can exceed 80k tokens; axolotl truncates
```

Axolotl detects the pre-tokenized columns and skips the chat_template renderer.
"""


def stream_and_subset(source_dir, sizes, full_size, seed, staging):
    """Single streaming pass: for each row, (a) write to `full` parquet, and
    (b) write to each size parquet whose index-set contains the global row
    index. Returns the full path + per-size paths.
    """
    staging.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    idxs = list(range(full_size))
    random.shuffle(idxs)

    idxset = {s: set(idxs[:s]) for s in sizes if s <= full_size}
    out_paths = {s: staging / f"cf_v3_{s}.parquet" for s in idxset}
    full_path = staging / f"cf_v3_{full_size}.parquet"
    for p in list(out_paths.values()) + [full_path]:
        if p.exists():
            p.unlink()

    writers = {s: pq.ParquetWriter(out_paths[s], OUT_SCHEMA, compression="snappy") for s in idxset}
    full_writer = pq.ParquetWriter(full_path, OUT_SCHEMA, compression="snappy")

    counts = {s: 0 for s in idxset}
    counts["full"] = 0
    global_i = 0

    for slug in SLUGS:
        shards = sorted(glob.glob(f"{source_dir}/{slug}-*-of-*.parquet"))
        for si, sp in enumerate(shards):
            if si % 20 == 0:
                print(f"  [{slug}] shard {si}/{len(shards)} (global_i={global_i}, full_written={counts['full']})", flush=True)
            tbl = pq.read_table(sp, columns=["trajectory_id", "reward", "chat_template_applied", "input_ids", "labels"])
            rows = tbl.to_pylist()
            buf_full = []
            buf_size = {s: [] for s in idxset}
            for r in rows:
                # Inject attention_mask, source
                r["attention_mask"] = [1] * len(r["input_ids"])
                r["source"] = SOURCE_TAG
                buf_full.append(r)
                for s in idxset:
                    if global_i in idxset[s]:
                        buf_size[s].append(r)
                global_i += 1
            # Flush per-shard (keeps memory bounded)
            if buf_full:
                out_tbl = pa.Table.from_pylist(buf_full, schema=OUT_SCHEMA)
                full_writer.write_table(out_tbl)
                counts["full"] += len(buf_full)
                del out_tbl
            for s in idxset:
                if buf_size[s]:
                    out_tbl = pa.Table.from_pylist(buf_size[s], schema=OUT_SCHEMA)
                    writers[s].write_table(out_tbl)
                    counts[s] += len(buf_size[s])
                    del out_tbl
            del tbl, rows, buf_full, buf_size
            gc.collect()

    for w in writers.values():
        w.close()
    full_writer.close()

    for k, v in counts.items():
        label = f"full ({full_size})" if k == "full" else f"{k}"
        print(f"  [out] {label}: {v} rows, {out_paths.get(k, full_path).stat().st_size / 1e9:.2f} GB", flush=True)

    return out_paths, full_path


def push_dataset(api, target_repo, parquet_path, size, full_size, token):
    print(f"[push] creating {target_repo}", flush=True)
    api.create_repo(repo_id=target_repo, repo_type="dataset", exist_ok=True)

    readme_tmp = parquet_path.parent / f"README_{size}.md"
    readme_tmp.write_text(make_readme(target_repo, size, full_size))

    path_in_repo = f"data/train-00000-of-00001.parquet"   # HF default layout
    print(f"[push] uploading {parquet_path.name} ({parquet_path.stat().st_size / 1e9:.2f} GB) -> {target_repo}", flush=True)
    api.upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo=path_in_repo,
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
    print(f"[push] done {target_repo}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-only", action="store_true", help="Download source shards and exit")
    ap.add_argument("--skip-download", action="store_true", help="Assume source shards are already local")
    ap.add_argument("--build-only", action="store_true", help="Write subset parquets, skip uploads")
    ap.add_argument("--skip-full", action="store_true", help="Don't upload the full (413k) dataset")
    ap.add_argument("--only-small", action="store_true", help="Only upload 316 + 1000 (skip larger)")
    args = ap.parse_args()

    token = os.environ["HF_TOKEN"]
    api = HfApi(token=token)

    # Download upstream shards to local HF cache
    if not args.skip_download:
        print("[download] snapshot_download trajectories-tokenized_qwencoder/", flush=True)
        src_root = snapshot_download(
            repo_id=SRC_REPO,
            repo_type="dataset",
            allow_patterns=[f"{SRC_SUBSET_DIR}/*.parquet"],
            token=token,
            max_workers=16,
        )
        source_dir = os.path.join(src_root, SRC_SUBSET_DIR)
    else:
        # Discover from HF_HOME cache
        hits = glob.glob(f"{os.environ.get('HF_HOME','~/.cache/huggingface')}/hub/datasets--togethercomputer--CoderForge-Preview/snapshots/*/{SRC_SUBSET_DIR}")
        if not hits:
            raise SystemExit("--skip-download but no cached source dir found")
        source_dir = sorted(hits)[-1]
    print(f"[src] {source_dir}", flush=True)

    if args.download_only:
        return

    full_size, _ = count_source_rows(source_dir)
    print(f"[src] total rows: {full_size:,}", flush=True)

    sizes = [316, 1000] if args.only_small else SIZES
    out_paths, full_path = stream_and_subset(source_dir, sizes, full_size, SEED, STAGING)

    if args.build_only:
        print("[build-only] skipping uploads", flush=True)
        return

    for s in sizes:
        push_dataset(api, f"{TARGET_BASE}-{s}", out_paths[s], s, full_size, token)

    if not args.skip_full:
        push_dataset(api, TARGET_BASE, full_path, full_size, full_size, token)


if __name__ == "__main__":
    main()
