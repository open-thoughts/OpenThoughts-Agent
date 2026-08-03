#!/usr/bin/env python3
"""Fast generator for nemotron-cpp-v2: same transform + keep-filter as
generate.py, but validation runs inside a FEW long-lived containers (one bash
loop per shard) instead of two `docker run` cold-starts per task — ~50x faster.

Procedure:
  1. transform all source tasks -> in-memory artifacts (drop external-lib/no-test)
  2. materialize each candidate to a scratch workdir:
       cand/<id>/test_solution.cpp            (the rewired agent-linked test)
       cand/<id>/gold/<hdr>                   (the gold header)
  3. for each shard, run ONE container that, per task, compiles+runs:
       (a) GOLD : g++ -I cand/<id>/gold        -> reward
       (b) EMPTY: g++ -I emptydir              -> reward
     writes cand/<id>/result.txt = "<gold> <empty>"
  4. KEEP iff gold==1 AND empty==0; assemble kept Harbor task dirs; upload.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from data.nemotron_cpp_v2 import generate as G
from data.nemotron_cpp_v2.transform import transform

PREFIX = G.PREFIX

# bash run inside the container: validate every task dir under /cand.
# For each task: compile gold, run; compile empty, run; write result.txt.
CONTAINER_SCRIPT = r"""
set -u
mkdir -p /empty
for d in /cand/*/; do
  [ -f "$d/result.txt" ] && continue
  hdr=$(cat "$d/hdr.txt")
  g=0; e=0
  # GOLD
  if g++ -std=c++17 -I"$d/gold" -o /tmp/rg "$d/test_solution.cpp" -lgtest -lgtest_main -pthread >/dev/null 2>&1; then
    if timeout 60 /tmp/rg --gtest_output=json:/tmp/gg.json >/dev/null 2>&1; [ -f /tmp/gg.json ]; then
      g=$(python3 -c "import json;d=json.load(open('/tmp/gg.json'));t=int(d.get('tests',0));f=int(d.get('failures',0))+int(d.get('errors',0));print(1 if (t>0 and f==0) else 0)" 2>/dev/null || echo 0)
    fi
  fi
  rm -f /tmp/gg.json
  # EMPTY (only if gold passed; saves time)
  if [ "$g" = "1" ]; then
    if g++ -std=c++17 -I/empty -o /tmp/re "$d/test_solution.cpp" -lgtest -lgtest_main -pthread >/dev/null 2>&1; then
      if timeout 60 /tmp/re --gtest_output=json:/tmp/ge.json >/dev/null 2>&1; [ -f /tmp/ge.json ]; then
        e=$(python3 -c "import json;d=json.load(open('/tmp/ge.json'));t=int(d.get('tests',0));f=int(d.get('failures',0))+int(d.get('errors',0));print(1 if (t>0 and f==0) else 0)" 2>/dev/null || echo 0)
      fi
    fi
    rm -f /tmp/ge.json
  fi
  echo "$g $e" > "$d/result.txt"
done
echo SHARD_DONE
"""


def run_shard(image: str, shard_dir: Path) -> None:
    subprocess.run(
        ["docker", "run", "--rm", "-v", f"{shard_dir}:/cand", image, "bash", "-c", CONTAINER_SCRIPT],
        capture_output=True, text=True, timeout=7200,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shards", type=int, default=10)
    ap.add_argument("--docker-image", default="nemo-cpp-gtest:v2")
    ap.add_argument("--target-repo", default="laion/exp_rpt_nemotron-cpp-v2")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    df = G.load_source()
    n = len(df) if args.limit <= 0 else min(args.limit, len(df))
    print(f"source rows: {len(df)}  processing: {n}", flush=True)

    # transform
    cands = []  # (task_id, instr, meta, art)
    skip: dict[str, int] = {}
    for i in range(n):
        instr, test, meta = G.extract_task(df.iloc[i]["task_binary"])
        art, why = transform(instr, test)
        if art is None:
            k = why.split(":")[0]
            skip[k] = skip.get(k, 0) + 1
            continue
        cands.append((f"{PREFIX}-{i:06d}", instr, meta, art))
    print(f"transformed: {len(cands)}  skipped: {skip}", flush=True)

    # materialize candidates into shard dirs
    cand_root = args.scratch / "cand"
    if cand_root.exists():
        shutil.rmtree(cand_root)
    shard_dirs = [cand_root / f"shard_{s:02d}" for s in range(args.shards)]
    for sd in shard_dirs:
        sd.mkdir(parents=True, exist_ok=True)
    for idx, (tid, instr, meta, art) in enumerate(cands):
        sd = shard_dirs[idx % args.shards]
        td = sd / tid
        (td / "gold").mkdir(parents=True, exist_ok=True)
        (td / "test_solution.cpp").write_text(art["new_test"])
        (td / "gold" / art["hdr"]).write_text(art["header_src"])
        (td / "hdr.txt").write_text(art["hdr"])
    print(f"materialized {len(cands)} candidates across {args.shards} shards", flush=True)

    # validate shards in parallel (one long-lived container each)
    with ThreadPoolExecutor(max_workers=args.shards) as ex:
        futs = {ex.submit(run_shard, args.docker_image, sd): sd for sd in shard_dirs}
        done = 0
        for fut in as_completed(futs):
            fut.result()
            done += 1
            print(f"  shard {done}/{args.shards} complete", flush=True)

    # collect results
    by_id = {tid: (instr, meta, art) for (tid, instr, meta, art) in cands}
    kept = []
    fail: dict[str, int] = {}
    for sd in shard_dirs:
        for td in sd.iterdir():
            rf = td / "result.txt"
            if not rf.exists():
                fail["no_result"] = fail.get("no_result", 0) + 1
                continue
            parts = rf.read_text().split()
            g = parts[0] if parts else "0"
            e = parts[1] if len(parts) > 1 else "0"
            tid = td.name
            if g == "1" and e == "0":
                instr, meta, art = by_id[tid]
                kept.append((tid, instr, meta, art))
            elif g != "1":
                fail["gold_not_1"] = fail.get("gold_not_1", 0) + 1
            else:
                fail["empty_not_0"] = fail.get("empty_not_0", 0) + 1
    print(f"\nKEPT (gold->1 AND empty->0): {len(kept)}/{len(cands)}  fail={fail}", flush=True)

    # assemble
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tid, instr, meta, art in kept:
        G.write_task_dir(args.out_dir, tid, instr, meta, art)
    print(f"wrote {len(kept)} task dirs to {args.out_dir}", flush=True)

    report = {"source_repo": G.SRC_REPO, "processed": n, "transformed": len(cands),
              "skip_reasons": skip, "kept": len(kept), "fail_reasons": fail,
              "target_repo": args.target_repo}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))

    if not args.no_upload and kept:
        from scripts.harbor import tasks_parquet_converter as tpc
        from huggingface_hub import HfApi
        tasks = tpc.find_tasks(args.out_dir, recursive=True)
        pq = args.out_dir.parent / "nemotron_cpp_v2.parquet"
        tpc.to_parquet(args.out_dir, pq, tasks, compression="gz")
        print(f"parquet: {pq} ({pq.stat().st_size/1e6:.1f} MB, {len(tasks)} tasks)", flush=True)
        api = HfApi()
        api.create_repo(args.target_repo, repo_type="dataset", exist_ok=True)
        api.upload_file(path_or_fileobj=str(pq), path_in_repo="tasks.parquet",
                        repo_id=args.target_repo, repo_type="dataset")
        print(f"uploaded -> https://huggingface.co/datasets/{args.target_repo}", flush=True)

    print("\nDONE.", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
