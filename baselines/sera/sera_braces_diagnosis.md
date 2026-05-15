# Sera v3 eval collapse — brace bug diagnosis

**Date:** 2026-04-24
**Agent:** background Claude
**Target:** `laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v3` (SLURM 391242, eval 392457 in progress on pinggy pair 10)

## TL;DR

The `}}}` triple-brace emission is a **symptom**, not the root cause.

**Real failure mode: long-context degeneracy.** Sera v3 executes 1-3 correct tool-call turns, then — once the prompt accumulates large tool observations (≥20 KB in a single tool message) — collapses into a repeating-token attractor (`4.4.4.4…`, `for the for the…`, `\n\n\n…`). Across all 38 trials submitted so far in eval 392457, 6 have completed, **all with reward=None / no patch produced**. Greedy decoding does not fix it.

The `}}}` bug seen in 1 trial (`pydata__xarray-3305__op4h4iD`) is a minor sampling glitch where `}}\n` got sampled in place of `}\n` at the critical brace-close position (see §2 probability probe). This only matters when vLLM applies `temperature=0.6` from the model's `generation_config.json`; under `temperature=0` greedy, the correct `}\n` has p=0.9969.

Even with greedy, though, the model still collapses at step 3+ with long observations — so fixing temperature doesn't fix the eval.

## Evidence

### 1. Training data is clean

Sera-4.6-Lite-T2-v4-316 source JSONL (on Jupiter):
`/e/data1/datasets/playground/ot-baf/hf_hub/datasets--laion--Sera-4.6-Lite-T2-v4-316/snapshots/2df0e5321c676c5010ba43d4be6b74fb13dfe8a4/sera-4.6-lite-t2_v4_316.jsonl`

- 316 rows, 11 325 `<tool_call>{...}</tool_call>` payloads.
- **0 malformed JSON** in tool_call payloads.
- **0 payloads ending in `}}}`.**
- The 12 occurrences of `}}}` in the file are all inside legitimate nested-dict Python snippets embedded in tool content/observations, not in model-output positions.

→ Hypothesis C (training-data corruption) ruled out.

### 2. Per-token probability probe (greedy model on HF checkpoint)

Script: `/e/scratch/jureap59/feuer1/sera_probe_compare.py` (job 392619 output `/e/scratch/jureap59/feuer1/sera_probe_392619.out`).

Seed position: prompt + `<think>…</think>\n\n<tool_call>\n{"name": "str_replace_editor", "arguments": {"command": "view", "path": "/testbed"}`.

Model must next emit `}\n` (one close of inner object, then newline leading into `}` for outer), then eventually `\n</tool_call>`.

**Top-12 next tokens after single `}` already emitted (eval prompt context):**
| prob | token |
|---:|---|
| **0.9969** | `'}\n'` — CORRECT |
| 0.0017 | `'}\n\n'` |
| 0.0004 | `'},'` |
| 0.0003 | `']}\n'` |
| 0.0002 | `'}}\n'` — the bug token |

→ under greedy, `}\n` wins decisively (emits `}}` total; valid). Under temp=0.6 + top_k=20, the `}}\n` option is inside the top-k and can be sampled → yields `}}}` total. Hypothesis B validated (sampling can produce the triple brace).

**Top-12 next tokens after valid `}}` emitted (eval prompt, should close with `\n</tool_call>`):**
| prob | token |
|---:|---|
| **0.9353** | `<|im_end|>` — model ends turn WITHOUT `</tool_call>` |
| 0.0043 | `<tool_response>` |
| 0.0036 | `' \n'` |
| … | random garbage (踏实, rollback, 党的建设, touchdown) |
| — | `\n</tool_call>` is NOT in top 12 |

→ Model never learned to emit the literal `</tool_call>` close tag at this position; it jumps straight to `<|im_end|>`. The hermes parser tolerates this (it extracts the JSON body preceding `<|im_end|>`), so the first-turn path "works by accident". This is under-training, not a bug in rendering.

### 3. Live-vLLM turn-1 replay (`/e/scratch/jureap59/feuer1/curl_probe*.py`)

Prompt: system + user (eval-style, 3.7 KB). Tools: str_replace_editor / bash / submit, tool_choice=auto.

| temperature sent | valid tool_calls / 3 reps | notes |
|---:|---:|---|
| 0.0 (explicit) | 3/3 | clean `{"command":"view","path":"/testbed"}` every time |
| 0.6 (explicit) | 3/3 | varies between `bash find` and `str_replace_editor view`, all JSON valid |
| None (server default → 0.6 from generation_config) | 3/3 | same as above |

→ Turn-1 is robust. `}}}` appears only occasionally at sampling > 0; the hermes tool-call path in vLLM parses JSON before emitting `tool_calls`, so if the model emits `}}}` the parser fails and it surfaces as raw content (which is what we saw in the xarray-3305 trial — `tool_calls=None`).

### 4. Step-3 replay (progressive truncation) — the real failure

Prompt: same as astropy-14369 step 3 = system + user + {assistant+tool}×3, with all prior tool outputs intact. Temperature=0.0 greedy, tools on.

Script: `/e/scratch/jureap59/feuer1/curl_probe4.py`

| upto-N msgs | cumulative chars | behaviour |
|---:|---:|---|
| 4 (1 tool resp, 3.5 KB) | 7 269 | valid `bash find …` |
| 6 (2 tool resps, 6.0 KB) | 9 780 | valid `str_replace_editor view table.py` |
| 7 (+assistant) | 9 780 | valid `str_replace_editor view table_helpers.py` |
| **8 (+19 689-char tool response of table.py)** | **29 469** | `** DEGENERATE **` repeated `4.4.4.4…` until finish_reason=length |

The collapse is deterministic (happens under greedy). Triggered by large tool observations.

## Hypotheses — final verdict

| H | Statement | Verdict |
|---|---|---|
| A | Eval prompt pushes model to different basin | partial; long-context specifically |
| B | Temperature>0 samples `}}}` | true but minor; greedy also fails |
| C | Training data has `}}}` rows | false |
| D | 316 rows × 6 epochs insufficient signal | **TRUE, primary cause** |
| E | Chat-template renders extra `}` | false (verified: `tok.apply_chat_template` → 2 braces) |

## Fix plan

F1 (temperature=0 override via `generation_config.json` patch) — **will not fix** the degeneracy per §4.

F2 (clean training data) — no dirty rows to clean per §1.

F3 (**scale to Sera-4.6-Lite-T2-v4-1000**, 3× data, 6 epochs) — **launching as Sera v6**.

F4 (scale to v4-3160) — reserve as backup.

### Sera v6 — F3 launch (SLURM 392641, RUNNING)

- **HF name** (planned): `laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v6`
- **Config**: `/e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_sera_v4_1000_v6.yaml` — mirrors `qwen3_8b_sera_v4_316_v3.yaml` (chat_template=tokenizer_default, num_epochs=6, lr=1e-5) but with `Sera-4.6-Lite-T2-v4-1000` dataset.
- **Sbatch**: `/e/scratch/jureap59/feuer1/code/axolotl_sbatch/sera_v4_1000_v6.sbatch` — 4 nodes GH200 booster/reformo, zero3_bf16.
- **Submitted**: 2026-04-24 ~20:50 UTC, SLURM 392641, running on `jpbo-112-[26-28,31]`.
- **Expected wall**: v3 (316 rows × 6 epochs × 1 node) took ~56 min; v6 (1000 rows × 6 epochs × 4 nodes) ≈ similar (3.17× rows / 4× GPUs = 0.79× wall), call it ~1 h.

## Post-training checklist (when 392641 exits 0)

1. Strip FSDP prefixes & checkpoints:
   ```bash
   SRC=/e/data1/datasets/playground/ot-baf/checkpoints/sera-v4-1000-axolotl__Qwen3-8B-v6
   DST=${SRC}-converted
   rm -rf $SRC/checkpoint-* $SRC/.cache
   python /e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py $SRC $DST
   ```
2. Secret scan (grep for `sk-`, `AKIA`, `ghp_`, `hf_`).
3. Upload: `huggingface-cli upload-large-folder laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v6 $DST --repo-type=model`
4. **Tokenizer restore** (MANDATORY per `feedback_axolotl_restore_tokenizer`): overwrite `tokenizer_config.json` / `tokenizer.json` / `vocab.json` / `merges.txt` with stock `Qwen/Qwen3-8B` files; delete any `chat_template.jinja` on the repo. Also consider replacing `generation_config.json` with `do_sample: false, temperature: 0` to pin greedy at serve time regardless of what the client sends (see §2 — avoids the small-but-real `}}}` sampling glitch).
5. DB register:
   ```bash
   python scripts/database/manual_db_push.py \
     --hf-model-id laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v6 \
     --base-model Qwen/Qwen3-8B \
     --dataset-name laion/Sera-4.6-Lite-T2-v4-1000
   ```
6. **Smoke replay** first: rerun `/e/scratch/jureap59/feuer1/sera_probe_compare.py` pointed at the v6 converted path. Expect top-12 after `..."path":"/testbed"}}` to include `\n</tool_call>` (not `<|im_end|>`). Also rerun `/e/scratch/jureap59/feuer1/curl_probe4.py`-style step-3 replay; expect non-degenerate output at upto=8.
7. **Eval**: swebench-verified-random-100-folders on pinggy pair 9 (pair 10 busy with 392457). Copy the 392457 config under a new experiments_dir `..._v6`, swap the model path / hf name. Target: non-zero reward.

## Operational

- Pinggy pair 10 is BUSY with running Sera v3 eval 392457 (let it finish, per hard rules).
- Use pair 9 for the v6 eval when training completes.
- Don't cancel 392457.

### Timeline

- 2026-04-24 20:25 UTC: diagnosis completed. Training data probed on Jupiter; per-token probe job 392619 ran; live-vLLM curl probe confirmed turn-1 robust and turn-3 degenerate.
- Next: launch Sera v6 SFT on v4-1000 dataset.
