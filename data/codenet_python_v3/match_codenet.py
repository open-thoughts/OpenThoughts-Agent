#!/usr/bin/env python3
"""Recover the lost CodeNet problem_id mapping for the codenet-python tasks via
embedding nearest-neighbour, and emit the authoritative clean I/O + reference solution.

Why
---
``laion/exp_rpt_codenet-python-v2`` dropped ``problem_id`` and ships zero-byte test I/O.
Its embedded instruction examples are unreliable (non-stdin framing + LLM-hallucinated
outputs; only ~55% self-consistent). The authoritative I/O is real CodeNet
(``windchimeran/codenet_python``, small/Accepted split: 2861 problems, each with clean
``input``/``output`` and an accepted reference ``code``; CodeNet's own code passes its own
I/O ~94%).

This script embeds each task's ``instruction.md`` (HTML/markdown stripped) and each
CodeNet ``problem_description``, with OpenAI ``text-embedding-3-small``, and assigns each
task its nearest CodeNet problem. Output is a parquet mapping:
    path, problem_id, sim, input, output, code

Downstream (``build_v3.py --match_map``) writes the matched clean I/O into
``tests/inputs``/``tests/outputs`` and the reference ``code`` into ``solution/solve.sh``.
"""
from __future__ import annotations
import argparse
import gzip
import io
import tarfile
import re
import html
import sys
import numpy as np
import pandas as pd
from openai import OpenAI

EMB_MODEL = "text-embedding-3-small"


def strip_markup(t: str) -> str:
    t = re.sub(r"```.*?```", " ", t, flags=re.S)   # drop fenced code (often hallucinated)
    t = re.sub(r"<[^>]+>", " ", t)                 # html tags (CodeNet desc)
    t = html.unescape(t)
    t = re.sub(r"[#*`_$\\]", " ", t)               # markdown / latex noise
    return re.sub(r"\s+", " ", t).strip()


def embed_all(client: OpenAI, texts: list[str], batch: int = 256) -> np.ndarray:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = [t[:6000] for t in texts[i : i + batch]]
        r = client.embeddings.create(model=EMB_MODEL, input=chunk)
        out.extend([d.embedding for d in r.data])
        if (i // batch) % 10 == 0:
            print(f"  embedded {i+len(chunk)}/{len(texts)}", flush=True)
    a = np.array(out, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def task_instruction(task_binary: bytes) -> str:
    t = tarfile.open(fileobj=io.BytesIO(gzip.decompress(task_binary)))
    return t.extractfile("instruction.md").read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks_parquet", required=True)
    ap.add_argument("--problems_parquet", required=True,
                    help="windchimeran per-problem map: problem_id, code, input, output, problem_description")
    ap.add_argument("--out_map", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = OpenAI()

    prob = pd.read_parquet(args.problems_parquet)
    prob = prob[prob["input"].str.len() > 0].reset_index(drop=True)
    print(f"embedding {len(prob)} CodeNet problems...", flush=True)
    pemb = embed_all(client, [strip_markup(x) for x in prob["problem_description"]])

    df = pd.read_parquet(args.tasks_parquet)
    if args.limit:
        df = df.iloc[: args.limit]
    print(f"embedding {len(df)} tasks...", flush=True)
    temb = embed_all(client, [strip_markup(task_instruction(b)) for b in df["task_binary"]])

    # nearest neighbour (chunked matmul to bound memory)
    best = np.empty(len(df), dtype=np.int64)
    bsim = np.empty(len(df), dtype=np.float32)
    step = 1024
    for i in range(0, len(df), step):
        s = temb[i : i + step] @ pemb.T
        best[i : i + step] = s.argmax(1)
        bsim[i : i + step] = s.max(1)

    rows = []
    for i, path in enumerate(df["path"].tolist()):
        pr = prob.iloc[best[i]]
        rows.append({
            "path": path,
            "problem_id": pr["problem_id"],
            "sim": float(bsim[i]),
            "input": pr["input"],
            "output": pr["output"],
            "code": pr["code"],
        })
    out = pd.DataFrame(rows)
    out.to_parquet(args.out_map, index=False)
    print(f"wrote {len(out)} -> {args.out_map}")
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8):
        print(f"  sim>={thr}: {(out['sim']>=thr).sum()}")
    print("sim quantiles:", out["sim"].quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
