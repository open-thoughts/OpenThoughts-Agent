#!/usr/bin/env python3
"""SELECT (not just count) the "thinking-preserving" subset of the DCAgent2
GLM-4.7-swesmith trace dataset and PUSH it to laion/.

Keep a row IFF BOTH:
  (1) agent task_complete  -- an ASSISTANT-role message contains the literal
      `"task_complete": true`. (The same string appears in the USER
      instruction of EVERY trace -> scope to role=="assistant".)
  (2) fits 32k, thinking-preserving -- plain-concat token count via
      extract_conversation_text + Qwen/Qwen3-8B tokenizer < 32768.
      (NOT apply_chat_template, which strips historical thinking and
       over-counts fits.)

Keeps ALL columns so the result is SFT-ready. Pushes to
laion/GLM-4.7-swesmith-oracle_verified-complete-lt32k (public).

Run with the otagent python; source secrets.env first.
"""
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

# extract_conversation_text lives in scripts/analysis/utils.py
sys.path.insert(0, "/Users/benjaminfeuer/Documents/OpenThoughts-Agent")
from scripts.analysis.utils import extract_conversation_text

SRC = "DCAgent2/GLM-4.7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k"
DST = "laion/GLM-4.7-swesmith-oracle_verified-complete-lt32k"
THRESH = 32768

print("loading tokenizer (Qwen/Qwen3-8B)...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

print("loading dataset (non-streaming, split=train)...", flush=True)
ds = load_dataset(SRC, split="train")
n = len(ds)
print(f"rows: {n}", flush=True)


def agent_complete(conv):
    return any(
        (m.get("role") == "assistant")
        and ('"task_complete": true' in (m.get("content") or ""))
        for m in conv
    )


def fits_32k_plain(row):
    text = extract_conversation_text(row)
    return len(tok(text, add_special_tokens=False)["input_ids"]) < THRESH


n_complete = 0
n_fit_plain = 0
n_both = 0
keep_idx = []

for i in range(n):
    row = ds[i]
    conv = row.get("conversations") or []
    complete = agent_complete(conv)
    fit = fits_32k_plain(row)
    if complete:
        n_complete += 1
    if fit:
        n_fit_plain += 1
    if complete and fit:
        n_both += 1
        keep_idx.append(i)
    if (i + 1) % 1000 == 0:
        print(
            f"  {i+1}/{n}  complete={n_complete} fit32k(plain)={n_fit_plain} both={n_both}",
            flush=True,
        )

print("\n==== FILTER RESULT ====", flush=True)
print(f"total traces:                                  {n}")
print(f"(1) agent task_complete:true [assistant]:      {n_complete}  ({100*n_complete/n:.1f}%)")
print(f"(2) <32768 tok [plain/extract_conversation]:   {n_fit_plain}  ({100*n_fit_plain/n:.1f}%)")
print(f">>> BOTH (complete AND <32k plain):            {n_both}  ({100*n_both/n:.1f}%)")

# Sanity gate per the task spec.
if not (7300 <= n_both <= 7340):
    print(
        f"\n!! UNEXPECTED kept count {n_both} (expected ~7318). "
        "Aborting before push.",
        flush=True,
    )
    sys.exit(1)

print(f"\nselecting {len(keep_idx)} rows (all columns preserved)...", flush=True)
filtered = ds.select(keep_idx)
print(f"filtered dataset: {filtered}", flush=True)
print(f"columns: {filtered.column_names}", flush=True)

print(f"\npushing to {DST} (public)...", flush=True)
filtered.push_to_hub(DST, private=False)
print("PUSH COMPLETE", flush=True)
print(f"kept={n_both} -> {DST}", flush=True)
