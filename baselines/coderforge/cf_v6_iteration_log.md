# CF v6 Iteration Log

Goal: Achieve non-zero reward on `swebench-verified-random-100-folders` eval with a CoderForge-trained Qwen3-8B.

## Background

CF v3 (pre-tokenized) and CF v5 (wrapper-stripped, no `<think>`) both produced
garbage output at eval time (`8888...`, `0.0.0.0...`) despite clean training
loss. Root cause (verified by stock-model probe): Qwen3-8B assigns ~100% prior
to `<think>` (id 151667) as the first token after `<|im_start|>assistant\n`.
CF's training data had zero `<think>` blocks — every assistant turn fought
the prior and destroyed long-context coherence.

Sera v4 works (grad_norm ~0.3, coherent output) because it renders
`<think>reasoning</think>\n\n<prose>\n\n<tool_call>{JSON}</tool_call>`.

## v6 design

- `<think>REASONING</think>` injected at the start of every assistant turn,
  sourced from (in order): synthetic `think`-tool `thought` arg → natural
  assistant content → fallback "Let me examine the problem...".
- Tool calls rendered as native OpenHands XML:
  `<function=NAME>\n<parameter=K>V</parameter>\n...\n</function>`
  (NOT `<tool_call>` wrapper — eval harness sets `disable_tool_calls: true`).
- Tool observations → `role: user` with `<tool_response>...</tool_response>`.
- `chat_template: tokenizer_default` (Qwen3-8B's own template, preserves
  think blocks across multi-turn).
- LR 1e-5, sequence_len 32768, zero3_bf16, num_epochs=1 (smoke test).

## Iterations

### v6-316 (job 392618) — submitted 2026-04-23

**Config**:
- Dataset: `laion/CoderForge-Preview-v6-316` (316 rendered trajectories,
  17,905 assistant turns, 16,938 XML function calls, 0 `<tool_call>`
  leaks).
- Sbatch: `/e/scratch/jureap59/feuer1/code/axolotl_sbatch/cf_v6_316.sbatch`
  (4 nodes, 4 GH200/node = 16 GPUs, zero3_bf16, 11:59 wall)
- Axolotl config:
  `/e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_cf_v6_316.yaml`
  - num_epochs=1, LR 1e-5, grad_accum 8, bs 1, warmup 0.1875, weight_decay 0.01,
    max_grad_norm 1.0, cosine scheduler, flash_attention, gradient_checkpointing.
- Base model: Qwen/Qwen3-8B (cached on Jupiter)

**Status**: SUBMITTED. Awaiting run.

**Next**: monitor grad_norm (target < 5.0, Sera v4 was ~0.3), loss. On success,
strip `_checkpoint_wrapped_module` prefixes with
`convert_axolotl_checkpoint.py`, restore stock Qwen3 tokenizer (delete
`chat_template.jinja`, copy `tokenizer_config.json` / `tokenizer.json` /
`vocab.json` / `merges.txt`), run `cf_probe_replay.py` to sanity check
coherence before uploading to HF.

## Expected trigger points

- **Smoke-test garbage** → diagnose: check token tags, loss curve, grad_norm.
  Try v7 with higher LR (3e-5) or lower (3e-6) or different think-synthesis.
- **Smoke-test coherent** → upload `laion/CoderForge-Preview-v6-316-axolotl__Qwen3-8B`,
  manual_db_push with `--base-model Qwen/Qwen3-8B`, launch swebench eval.
- **Eval reward=0 across 15+ trials with coherent outputs** → scale to
  v6-1000 with 3 epochs. Re-eval.
