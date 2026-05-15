# Sera v3 axolotl retrain — diagnosis & launch

**Date:** 2026-04-24
**Predecessor:** `laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v2` (SLURM 389486), 0% on swebench-verified-random-100-folders. Emitted malformed `{"name": "view", ... }}}` with 3 closing braces and wrong tool name.

## Training-data probe (first 5 rows of `sera-4.6-lite-t2_v4_316.jsonl` on Jupiter)

| Metric | Value |
|---|---|
| Total `<tool_call>` blocks scanned | **140** |
| Malformed JSON bodies | **0** |
| Rows containing `}}}` anywhere | **0** |
| Tool names | `str_replace_editor`=68, `bash`=62, `submit`=10 |
| Roles | `system`=5, `user`=145, `assistant`=140 |
| Structured `tool_calls` field | 0 (all tool calls inlined as `<tool_call>…</tool_call>` text) |
| `<tool_response>` in `role=user` | 140 / in `role=tool` 0 |
| Train mask | `assistant: True` only (correct) |
| Message keys | `{role, content, train}` only |

Matches SWE-agent harness schema (iter 16 called out `str_replace_editor`/`bash`/`submit`). Matches allenai/SERA-8B's training format. **Training data is clean** — the v2 SFT failure is a model-quality issue, not a data-regeneration blocker.

## Failure mode analysis

v2 output `{"name": "view", "arguments": {"command": "view", "path": "/testbed"}}}`:
1. Wrong tool name: `"view"` is the value of `arguments.command`, not the outer tool name. Model collapsed nested levels.
2. Three `}` instead of two: nesting-stack miscount under pressure.

Both are classic under-training symptoms at size=316. Effective grad updates at 3 epochs × 316 rows ÷ 8 grad-accum = **~120 updates**. allenai/SERA-8B (32.3% reference) trained on the full 36,083-row set (~100× more data). Our fair comparison point lives higher up the size ladder.

## Decision: retrain as v3 with `num_epochs: 6`

Single-knob change, per the "pick ONE" rule. Doubles gradient passes on the tiny dataset to let the model actually latch onto tool-name binding + JSON close-brace discipline without touching anything else that could regress v2's structural gains (no whitespace collapse, valid `<think>` / `<tool_call>` scaffolding). LR stays at 1e-5 (already mild); lowering it would undertrain further. Masking is already verified correct by axolotl preprocess (see below).

Files written on Jupiter:
- Config: `/e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_sera_v4_316_v3.yaml` (`num_epochs: 6`, `output_dir: …-v3`, `dataset_prepared_path: …sera-v4-316-v3`)
- Sbatch: `/e/scratch/jureap59/feuer1/code/axolotl_sbatch/sera_v4_316_v3.sbatch` (copy of v2 sbatch, only CONFIG + job-name changed)

Preprocess pre-flight check (login-node `axolotl preprocess`) printed `<tool_call>(151657, 151657)` and `str_replace_editor(1284, 1284)` inside the loss — training signal healthy.

## SLURM job

**391109** `sera-v4-316-axolotl__Qwen3-8B-v3`, 1 node, booster/reformo — already RUNNING on `jpbo-066-36` at 13:45 UTC. Wall estimate ~56 min (v2 took 31 min for 3 epochs; 6 epochs ≈ 2×).

## Next actions for the user (post-SFT)

When 391109 exits 0:

1. **Convert** (strip `_checkpoint_wrapped_module.` FSDP prefixes):
   ```bash
   SRC=$CHECKPOINTS_DIR/sera-v4-316-axolotl__Qwen3-8B-v3
   DST=$CHECKPOINTS_DIR/sera-v4-316-axolotl__Qwen3-8B-v3-converted
   rm -rf $SRC/checkpoint-* $SRC/.cache
   python /e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py $SRC $DST
   ```
2. **Secret scan** `$DST` (grep for `sk-…`/`AKIA…`/`ghp_…`/`hf_…`).
3. **Upload**: `huggingface-cli upload-large-folder laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v3 $DST --repo-type=model`
4. **Tokenizer restore** (CRITICAL; per `feedback_axolotl_restore_tokenizer`): overwrite `tokenizer_config.json` / `tokenizer.json` / `vocab.json` / `merges.txt` with stock Qwen/Qwen3-8B files AND `api.delete_file("chat_template.jinja")` on the repo.
5. **DB register**:
   ```bash
   python scripts/database/manual_db_push.py \
     --hf-model-id laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v3 \
     --base-model Qwen/Qwen3-8B \
     --dataset-name laion/Sera-4.6-Lite-T2-v4-316
   ```
6. **Eval** via same harness as iter 16 (swe-agent, 4xGH200 num_replicas=4, hermes parser ON, function_calling, pinggy). Pass signal: `<tool_call>` JSON parses cleanly with tool name in `{str_replace_editor, bash, submit}`.

## Contingency

If v3 still emits malformed JSON / wrong tool names → the size=316 ladder rung is fundamentally too small for this SFT recipe. Escalation path (NOT launched here — needs user approval): regenerate `sera-v4-3160` tokenized cache and train at size=3160 (2 nodes, ~1h wall per v4 README node ladder).
