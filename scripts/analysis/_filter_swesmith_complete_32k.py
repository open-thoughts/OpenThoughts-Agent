#!/usr/bin/env python3
"""One-off: count traces in the DCAgent2 GLM-4.7-swesmith trace dataset that are
(1) agent-marked task_complete:true AND (2) tokenize to < 32768 tokens.

task_complete detection: the literal `"task_complete": true` appears in the USER
instruction of every trace ("...include \"task_complete\": true in your..."), so we
scope detection to ASSISTANT messages (the agent's actual terminal action). Token
count uses the Qwen3-8B tokenizer (the model is a Qwen3-8B SFT) via apply_chat_template
(training-faithful), with a plain-concat count reported as a cross-check.
"""
import sys
from datasets import load_dataset
from transformers import AutoTokenizer

DS = "DCAgent2/GLM-4.7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k"
THRESH = 32768

print("loading tokenizer (Qwen/Qwen3-8B)...", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

print("loading dataset (non-streaming)...", flush=True)
ds = load_dataset(DS, split="train")
n = len(ds)
print(f"rows: {n}", flush=True)

def agent_complete(conv):
    # agent emitted task_complete:true in an assistant turn (exclude the user-instruction confound)
    return any((m.get("role") == "assistant") and ('"task_complete": true' in (m.get("content") or "")) for m in conv)

n_complete = 0
n_fit_ct = 0          # apply_chat_template < THRESH
n_fit_plain = 0       # plain concat < THRESH
n_both_ct = 0         # complete AND chat-template < THRESH  (THE ANSWER)
n_both_plain = 0
errs = 0

# batch the plain-concat path for speed; chat-template per row (handles roles)
for i in range(n):
    row = ds[i]
    conv = row.get("conversations") or []
    complete = agent_complete(conv)
    if complete:
        n_complete += 1
    # chat-template token count (training-faithful)
    try:
        ct = len(tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=False))
    except Exception:
        # fallback: plain join if template rejects the message shape
        ct = len(tok(("\n".join((m.get("content") or "") for m in conv)), add_special_tokens=False)["input_ids"])
        errs += 1
    plain = len(tok(("\n".join((m.get("content") or "") for m in conv)), add_special_tokens=False)["input_ids"])
    if ct < THRESH: n_fit_ct += 1
    if plain < THRESH: n_fit_plain += 1
    if complete and ct < THRESH: n_both_ct += 1
    if complete and plain < THRESH: n_both_plain += 1
    if (i + 1) % 1000 == 0:
        print(f"  {i+1}/{n}  complete={n_complete} fit32k(ct)={n_fit_ct} both(ct)={n_both_ct}", flush=True)

print("\n==== RESULT ====", flush=True)
print(f"total traces:                         {n}")
print(f"(1) agent task_complete:true:         {n_complete}  ({100*n_complete/n:.1f}%)")
print(f"(2) <32768 tok [apply_chat_template]: {n_fit_ct}  ({100*n_fit_ct/n:.1f}%)")
print(f"    <32768 tok [plain concat]:        {n_fit_plain}")
print(f">>> BOTH (complete AND <32k, chat-template): {n_both_ct}  ({100*n_both_ct/n:.1f}%)")
print(f">>> BOTH (complete AND <32k, plain concat):  {n_both_plain}")
print(f"(apply_chat_template fallbacks: {errs})")
