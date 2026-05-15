"""Build the SERAlike-swesmith dataset by rewriting `instruction.md` inside
each task tarball of `DCAgent/swesmith-sandboxes-with_tests-25k`.

Strategy:
- Stream rows from the source dataset one at a time (binary tarballs are big).
- For each row: gunzip+untar the task_binary, replace `instruction.md` with a
  SERAlike vague-prompt + abstain, repack with the same layout, emit.
- Bug-type per row is sampled deterministically from the 51-prompt list using
  a SHA-256 hash of the row's `path` field (so the rewrite is reproducible and
  the seed is the row id, not a global counter).
- Output is sharded JSON-lines-of-bytes — one .parquet per shard — written under
  ./out/ and uploaded to HF separately by upload.py.

Usage:
    python -m data.seralike.rewrite --limit 50    # smoke test
    python -m data.seralike.rewrite              # full 25k
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from data.seralike.bug_prompts import BUG_PROMPTS, render_instruction


SOURCE_DATASET = "DCAgent/swesmith-sandboxes-with_tests-25k"
OUT_DIR = Path(__file__).parent / "out"
SHARD_ROWS = 1000  # rows per parquet shard


def pick_bug(path_id: str) -> tuple[str, str]:
    """Deterministic mapping path → (label, paragraph) using SHA-256."""
    h = hashlib.sha256(path_id.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % len(BUG_PROMPTS)
    return BUG_PROMPTS[idx]


def rewrite_tarball(blob: bytes, new_instruction_md: str) -> bytes:
    """Open a gzipped tar in-memory, swap `instruction.md`, repack to a new gzipped tar.
    Preserves all other files and their metadata (mode/mtime).
    """
    src_buf = io.BytesIO(blob)
    src_gz = gzip.GzipFile(fileobj=src_buf, mode="rb")
    src_tar = tarfile.open(fileobj=src_gz, mode="r")

    dst_buf = io.BytesIO()
    # mtime=0 makes the gzip header deterministic; same byte output for same input
    dst_gz = gzip.GzipFile(fileobj=dst_buf, mode="wb", mtime=0)
    dst_tar = tarfile.open(fileobj=dst_gz, mode="w")

    new_bytes = new_instruction_md.encode("utf-8")
    found = False

    for member in src_tar.getmembers():
        if member.name == "instruction.md":
            new_member = tarfile.TarInfo(name="instruction.md")
            new_member.size = len(new_bytes)
            new_member.mode = member.mode
            new_member.mtime = member.mtime
            new_member.uid = member.uid
            new_member.gid = member.gid
            new_member.uname = member.uname
            new_member.gname = member.gname
            dst_tar.addfile(new_member, fileobj=io.BytesIO(new_bytes))
            found = True
        else:
            extracted = src_tar.extractfile(member) if member.isfile() else None
            dst_tar.addfile(member, fileobj=extracted)

    dst_tar.close()
    dst_gz.close()
    src_tar.close()
    src_gz.close()

    if not found:
        raise RuntimeError("instruction.md not found in source tarball")

    return dst_buf.getvalue()


def write_shard(rows: list[dict], shard_idx: int, out_dir: Path) -> Path:
    """Write a list of dict rows to a parquet shard file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shard-{shard_idx:05d}.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_path, compression="zstd")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N rows (smoke test). Default: all.")
    ap.add_argument("--out_dir", type=Path, default=OUT_DIR)
    ap.add_argument("--shard_rows", type=int, default=SHARD_ROWS)
    ap.add_argument("--source", type=str, default=SOURCE_DATASET)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_jsonl = args.out_dir / "metadata.jsonl"
    print(f"Source: {args.source}")
    print(f"Out:    {args.out_dir}")
    print(f"Limit:  {args.limit or 'no limit'}")

    ds = load_dataset(args.source, split="train", streaming=True)

    rows_in_shard = []
    shard_idx = 0
    n_total = 0
    bug_label_counter: dict[str, int] = {}
    t0 = time.time()

    with metadata_jsonl.open("w") as meta_f:
        for row in ds:
            if args.limit and n_total >= args.limit:
                break

            path = row["path"]
            label, paragraph = pick_bug(path)
            new_md = render_instruction(label, paragraph)
            new_blob = rewrite_tarball(row["task_binary"], new_md)

            rows_in_shard.append({
                "path": path,
                "task_binary": new_blob,
                "bug_label": label,
                "source_dataset": args.source,
            })
            bug_label_counter[label] = bug_label_counter.get(label, 0) + 1
            meta_f.write(json.dumps({"path": path, "bug_label": label}) + "\n")

            n_total += 1
            if len(rows_in_shard) >= args.shard_rows:
                p = write_shard(rows_in_shard, shard_idx, args.out_dir)
                rate = n_total / (time.time() - t0 + 1e-9)
                print(f"  wrote shard {shard_idx} -> {p.name}  ({n_total} total, {rate:.1f} rows/s)")
                rows_in_shard = []
                shard_idx += 1

        if rows_in_shard:
            p = write_shard(rows_in_shard, shard_idx, args.out_dir)
            rate = n_total / (time.time() - t0 + 1e-9)
            print(f"  wrote shard {shard_idx} -> {p.name}  ({n_total} total, {rate:.1f} rows/s)")

    # Summary
    print(f"\nDone. {n_total} rows in {shard_idx + (1 if rows_in_shard else 0)} shards.")
    print("Bug-label distribution (top 15):")
    for label, c in sorted(bug_label_counter.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {c:5}  {label}")


if __name__ == "__main__":
    main()
