#!/usr/bin/env python3
"""Convert allenai/Sera-4.5A-Full-T1 to shareGPT format + upload to laion/Sera-4.5A-Full-T1-v2 + size subsets.
Memory-bounded: writes to parquet in 500-row chunks, then loads via pyarrow for upload.
"""
import os, json, sys, argparse, random, gc
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


SERA_JSONL = "/Users/benjaminfeuer/.cache/huggingface/hub/datasets--allenai--Sera-4.5A-Full-T1/snapshots/6e97fe0156fc2a89ee11bb565f4e0e21617ef9ca/sera-4.5a-full-t1_72118_string_enriched.jsonl"
SOURCE_TAG = "allenai/Sera-4.5A-Full-T1"
TARGET_REPO = "laion/Sera-4.5A-Full-T1-v2"
SUBSET_SIZES = [316, 1000, 3160, 10000, 31600, 100000]
STAGING_DIR = Path("/Users/benjaminfeuer/Documents/scripts_dataset_build/_sera_staging")
CHUNK_SIZE = 500


def _content_to_str(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                if p.get("type") == "text" and "text" in p:
                    parts.append(p["text"])
                elif "content" in p:
                    parts.append(str(p["content"]))
                else:
                    parts.append(json.dumps(p))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(c or "")


def _map_role(r):
    return "assistant" if r == "assistant" else "user"


def _merge_consecutive(turns):
    out = []
    for t in turns:
        if out and out[-1]["role"] == t["role"]:
            out[-1]["content"] = out[-1]["content"] + "\n\n" + t["content"]
        else:
            out.append(t)
    return out


def to_sharegpt(raw_messages_json):
    try:
        msgs = json.loads(raw_messages_json)
    except Exception:
        return None
    turns = []
    for m in msgs:
        role = _map_role(m.get("role") or "")
        content = _content_to_str(m.get("content"))
        if not content.strip():
            continue
        turns.append({"role": role, "content": content})
    turns = _merge_consecutive(turns)
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": ""})
        turns = _merge_consecutive(turns)
    # LLaMA-Factory expects the conversation to end with an assistant turn
    # (the training target). Sera traces often end with a tool observation
    # (which maps to user in shareGPT), so drop trailing user turns.
    while turns and turns[-1]["role"] == "user":
        turns.pop()
    return turns


SCHEMA = pa.schema([
    ("conversations", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("source", pa.string()),
    ("instance_id", pa.string()),
])


def write_full_parquet(out_path):
    print(f"[sera] Parsing JSONL and writing to {out_path} (chunk_size={CHUNK_SIZE})", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    chunk = []
    total = 0
    with open(SERA_JSONL) as f:
        for i, line in enumerate(f):
            if i % 5000 == 0 and i > 0:
                print(f"  parsed {i} rows ({total} valid, {len(chunk)} in buffer)", flush=True)
            try:
                row = json.loads(line)
            except Exception:
                continue
            convo = to_sharegpt(row.get("messages", ""))
            if not convo or len(convo) < 2:
                continue
            chunk.append({
                "conversations": convo,
                "source": SOURCE_TAG,
                "instance_id": row.get("instance_id") or f"sera-{i}",
            })
            if len(chunk) >= CHUNK_SIZE:
                tbl = pa.Table.from_pylist(chunk, schema=SCHEMA)
                writer.write_table(tbl)
                total += len(chunk)
                chunk.clear()
                del tbl
                gc.collect()
    if chunk:
        tbl = pa.Table.from_pylist(chunk, schema=SCHEMA)
        writer.write_table(tbl)
        total += len(chunk)
    writer.close()
    print(f"[sera] Wrote {total} rows to {out_path}", flush=True)
    return total


def write_subset_parquet(full_path, subset_path, indices):
    index_set = set(indices)
    print(f"[sera] Writing subset {subset_path.name} ({len(index_set)} rows)", flush=True)
    writer = pq.ParquetWriter(subset_path, SCHEMA, compression="snappy")
    pf = pq.ParquetFile(full_path)
    cursor = 0
    written = 0
    for batch in pf.iter_batches(batch_size=500):
        rows = batch.to_pylist()
        keep = []
        for r in rows:
            if cursor in index_set:
                keep.append(r)
            cursor += 1
        if keep:
            tbl = pa.Table.from_pylist(keep, schema=SCHEMA)
            writer.write_table(tbl)
            written += len(keep)
        if written >= len(index_set):
            break
    writer.close()
    return written


def push(path, repo_id, token, private=False):
    from datasets import Dataset
    print(f"[sera] Loading {path.name} for push to {repo_id}", flush=True)
    ds = Dataset.from_parquet(str(path))
    print(f"[sera]  -> {len(ds)} rows, pushing...", flush=True)
    ds.push_to_hub(repo_id, token=token, private=private)
    print(f"[sera]  -> pushed {repo_id}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-subsets", action="store_true")
    args = ap.parse_args()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    full_path = STAGING_DIR / "sera_full.parquet"
    if not full_path.exists():
        n = write_full_parquet(full_path)
    else:
        n = pq.ParquetFile(full_path).metadata.num_rows
        print(f"[sera] Using existing {full_path} ({n} rows)", flush=True)

    token = os.environ["HF_TOKEN"]

    if not args.skip_full:
        push(full_path, TARGET_REPO, token)

    if args.skip_subsets:
        return

    random.seed(42)
    all_idxs = list(range(n))
    random.shuffle(all_idxs)

    for size in SUBSET_SIZES:
        if size > n:
            print(f"[sera] Skip size {size} (only {n} rows)", flush=True)
            continue
        subset_path = STAGING_DIR / f"sera_{size}.parquet"
        indices = sorted(all_idxs[:size])
        write_subset_parquet(full_path, subset_path, indices)
        push(subset_path, f"{TARGET_REPO}-{size}", token)


if __name__ == "__main__":
    main()
