#!/usr/bin/env python3
"""Convert togethercomputer/CoderForge-Preview (4 subsets) to unified shareGPT + upload.
Memory-bounded: streams each parquet shard, writes to unified output parquet in 500-row chunks.
"""
import os, json, sys, argparse, random, glob, gc
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


CACHE_DIR = "/Users/benjaminfeuer/.cache/huggingface/hub/datasets--togethercomputer--CoderForge-Preview"
TARGET_REPO = "laion/CoderForge-Preview-v2"
SUBSET_SIZES = [316, 1000, 3160, 10000, 31600, 100000]
SUBSETS = ["R2E_Gym", "SWE_Rebench", "SWE_Smith", "filtered_reward1"]
STAGING_DIR = Path("/Users/benjaminfeuer/Documents/scripts_dataset_build/_cf_staging")
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


def _msg_with_tool_calls_to_str(m):
    parts = []
    content = _content_to_str(m.get("content"))
    if content and content.strip():
        parts.append(content)
    tc = m.get("tool_calls")
    if tc:
        for t in tc if isinstance(tc, list) else [tc]:
            try:
                parts.append("<tool_call>\n" + json.dumps(t) + "\n</tool_call>")
            except Exception:
                parts.append(str(t))
    return "\n".join(parts)


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
        content = _msg_with_tool_calls_to_str(m)
        if not content.strip():
            continue
        turns.append({"role": role, "content": content})
    turns = _merge_consecutive(turns)
    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": ""})
        turns = _merge_consecutive(turns)
    return turns


SCHEMA = pa.schema([
    ("conversations", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("source", pa.string()),
    ("instance_id", pa.string()),
])


def find_traj_dir():
    snaps = glob.glob(f"{CACHE_DIR}/snapshots/*/trajectories")
    if not snaps:
        raise SystemExit("No CoderForge trajectories dir")
    return sorted(snaps)[-1]


def write_full_parquet(traj_dir, out_path):
    print(f"[cf] Streaming {traj_dir} -> {out_path} (chunk_size={CHUNK_SIZE})", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    chunk = []
    total = 0
    source_counts = {}
    for subset in SUBSETS:
        paths = sorted(glob.glob(f"{traj_dir}/{subset}-*-of-*.parquet"))
        print(f"[cf] {subset}: {len(paths)} shards", flush=True)
        for pi, p in enumerate(paths):
            if pi % 30 == 0:
                print(f"  {subset} shard {pi}/{len(paths)} (total={total})", flush=True)
            tbl = pq.read_table(p, columns=["trajectory_id", "messages"])
            for r in tbl.to_pylist():
                convo = to_sharegpt(r["messages"])
                if not convo or len(convo) < 2:
                    continue
                src = f"togethercomputer/CoderForge-Preview/{subset}"
                chunk.append({
                    "conversations": convo,
                    "source": src,
                    "instance_id": r["trajectory_id"] or f"{subset}-{total}",
                })
                source_counts[src] = source_counts.get(src, 0) + 1
                if len(chunk) >= CHUNK_SIZE:
                    out_tbl = pa.Table.from_pylist(chunk, schema=SCHEMA)
                    writer.write_table(out_tbl)
                    total += len(chunk)
                    chunk.clear()
                    del out_tbl
                    gc.collect()
            del tbl
    if chunk:
        out_tbl = pa.Table.from_pylist(chunk, schema=SCHEMA)
        writer.write_table(out_tbl)
        total += len(chunk)
    writer.close()
    print(f"[cf] Wrote {total} rows to {out_path}", flush=True)
    for s, n in source_counts.items():
        print(f"  {s}: {n}", flush=True)
    return total


def write_subset_parquet(full_path, subset_path, indices):
    index_set = set(indices)
    print(f"[cf] Writing subset {subset_path.name} ({len(index_set)} rows)", flush=True)
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
    print(f"[cf] Loading {path.name} for push to {repo_id}", flush=True)
    ds = Dataset.from_parquet(str(path))
    print(f"[cf]  -> {len(ds)} rows, pushing...", flush=True)
    ds.push_to_hub(repo_id, token=token, private=private)
    print(f"[cf]  -> pushed {repo_id}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-full", action="store_true")
    ap.add_argument("--skip-subsets", action="store_true")
    args = ap.parse_args()

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    full_path = STAGING_DIR / "cf_full.parquet"
    if not full_path.exists():
        traj_dir = find_traj_dir()
        n = write_full_parquet(traj_dir, full_path)
    else:
        n = pq.ParquetFile(full_path).metadata.num_rows
        print(f"[cf] Using existing {full_path} ({n} rows)", flush=True)

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
            print(f"[cf] Skip size {size} (only {n} rows)", flush=True)
            continue
        subset_path = STAGING_DIR / f"cf_{size}.parquet"
        indices = sorted(all_idxs[:size])
        write_subset_parquet(full_path, subset_path, indices)
        push(subset_path, f"{TARGET_REPO}-{size}", token)


if __name__ == "__main__":
    main()
