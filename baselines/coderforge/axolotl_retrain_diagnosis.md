# Axolotl SFT 0% retrain diagnosis (Sera v4-316 + CF v3-1000) — STATUS REPORT

**Date:** 2026-04-25 (UTC)
**Conclusion:** The diagnosis work the prompt asked me to perform is **already
complete**, and the corresponding fix retrains are **already in flight or
already trained, awaiting upload**. I am NOT launching new retrains and NOT
touching the running job. This doc captures the current state so the next
agent / human knows where to pick up.

## What the prompt asked for vs. what already exists

The prompt framed the task as: probe parquet, decide config-vs-data fix, then
either commit + retrain (CF v3 / Sera v5) or stop + write a diagnosis doc. The
actual repo state at /Users/benjaminfeuer/Documents/notes/ot-agent and Jupiter
shows a much more advanced iteration tree — both root causes have been
diagnosed in dedicated docs and the fix retrains have been launched on
different branches (`v6`/`v7`) than the prompt named (`v3` for CF, `v5` for
Sera).

### Sera-v4-316 — root cause + fix
- Diagnosed in `/Users/benjaminfeuer/Documents/notes/ot-agent/sera_braces_diagnosis.md`.
- The `}}}` was a sampling glitch under temp=0.6; the **real** failure was
  long-context degeneracy after large tool observations (>20 KB) at turn 3+.
  Greedy decoding doesn't fix it.
- Verdict: under-training (316 rows × 6 epochs is too little signal).
- Fix path being executed: **scale data 316→1000 + epochs 3→{6,12}**.
  - **Sera-v4-316-v3** (job 391242, COMPLETED) — 316 × 6ep, on HF as
    `laion/Sera-4.6-Lite-T2-v4-316-axolotl__Qwen3-8B-v3`. Still degenerates per
    eval debug log.
  - **Sera-v4-1000-v6** (job 392641, COMPLETED) — 1000 × 6ep, on HF as
    `laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v6`.
  - **Sera-v4-1000-v7** (job 393612, **RUNNING** as of 2026-04-25 02:20 UTC,
    epoch 3.16/12, loss 0.18, grad_norm 0.3, ppl 1.18 — healthy).
    HF target: `laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v7` (not
    uploaded yet; training still in progress).
- Per memory `feedback_no_unrequested_rl_launches.md` (extends to SFT in
  practice): not relaunching duplicates.

### CoderForge-Preview-v3-1000 — root cause + fix
- Diagnosed in `/Users/benjaminfeuer/Documents/notes/ot-agent/cf_v6_iteration_log.md`.
- Earlier `axolotl_retrain_diagnosis_coderforge.md` claim ("just delete
  chat_template.jinja, no retrain") was **wrong** — verified by iter 17h /
  391120 emitting `888...` and word-salad even after the jinja delete.
- Real root cause (verified by stock-Qwen3 prior probe): Qwen3-8B assigns
  ~100% prior to `<think>` (id 151667) as the first token after
  `<|im_start|>assistant\n`. CF v3/v5 training data had **zero** `<think>`
  blocks. Every assistant turn fought the prior → catastrophic forgetting →
  `888...` output.
- Fix is **data-level**: regenerate the dataset with `<think>REASONING</think>`
  injected at the start of every assistant turn. Already done by the user as
  `laion/CoderForge-Preview-v6-{316,1000}` (cached on Jupiter).
- Trained checkpoints exist on Jupiter, NOT yet uploaded:
  - **CF v6-316** (job 392712, COMPLETED 2026-04-24 21:42) — 316 × 1ep smoke
    test. Converted at
    `/e/data1/datasets/playground/ot-baf/checkpoints/cf-v6-316-axolotl__Qwen3-8B-converted/`.
    Target HF: `laion/CoderForge-Preview-v6-316-axolotl__Qwen3-8B`.
  - **CF v7-1000** (job 392836, COMPLETED 2026-04-25 00:02, 6ep) — converted
    at `/e/data1/datasets/playground/ot-baf/checkpoints/cf-v7-1000-axolotl__Qwen3-8B-converted/`.
    Target HF: `laion/CoderForge-Preview-v7-1000-axolotl__Qwen3-8B`.
    Note: the v7 config (`qwen3_8b_cf_v7_1000.yaml`) is mostly v6-template but
    with the v6-1000 dataset and 6 epochs; it was renamed v7 due to size+epoch
    bump. The dataset path in the YAML hardcodes a snapshot, so the eventual
    HF dataset link is `laion/CoderForge-Preview-v6-1000`.

## What still needs doing (post-training pipeline)

### CF v6-316 + CF v7-1000 — finish the upload chain
The user's helper script already exists at
`/e/scratch/jureap59/feuer1/code/axolotl_sbatch/cf_v6_posttrain.sh` and does
strip + tokenizer-restore + smoke-replay; v6-316 was run through it (the
`-converted` dir matches the expected layout: `chat_template.jinja` already
removed and `vocab.json` + `merges.txt` present from stock Qwen).

The remaining manual steps (per `feedback_db_register_after_eval.md` — DO NOT
register in the launch chain, so just stop after HF upload):
```bash
# CF v6-316 (smoke test, optional upload — only if probe shows <think>+<function=...> output)
huggingface-cli upload-large-folder \
  laion/CoderForge-Preview-v6-316-axolotl__Qwen3-8B \
  /e/data1/datasets/playground/ot-baf/checkpoints/cf-v6-316-axolotl__Qwen3-8B-converted \
  --repo-type=model

# CF v7-1000 (full 6ep run — the production candidate)
# 1) probe coherence first (run cf_v6_posttrain.sh on the v7 path or adapt
#    /tmp/cf_v6_probe.py from the script). Confirm output is structured
#    (<think>...</think> + <function=...>...</function>), NOT 888... garbage.
# 2) if coherent:
huggingface-cli upload-large-folder \
  laion/CoderForge-Preview-v7-1000-axolotl__Qwen3-8B \
  /e/data1/datasets/playground/ot-baf/checkpoints/cf-v7-1000-axolotl__Qwen3-8B-converted \
  --repo-type=model
```

(`-converted` dirs already have `tokenizer_config.json`, `tokenizer.json`,
`vocab.json`, `merges.txt` from stock Qwen3-8B and no `chat_template.jinja`,
per the post-train script. Tokenizer restore already done.)

### Sera v7 — wait, then post-train
Job 393612 is running 12 epochs on 1000 rows — at 02:20 UTC it was 3.16 epochs
in (1h 15m wall, ~80 s/it × 218 steps = ~5h total wall, ETA ~05:30 UTC).
**Do not cancel.** When it exits 0:
```bash
SRC=/e/data1/datasets/playground/ot-baf/checkpoints/sera-v4-1000-axolotl__Qwen3-8B-v7
DST=${SRC}-converted
rm -rf $SRC/checkpoint-* $SRC/.cache
python /e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py $SRC $DST

# tokenizer restore (mandatory per feedback_axolotl_restore_tokenizer)
STOCK=/e/data1/datasets/playground/ot-baf/hf_hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
rm -f $DST/chat_template.jinja
for f in tokenizer_config.json tokenizer.json vocab.json merges.txt; do
  cp $STOCK/$f $DST/$f
done

# smoke replay BEFORE upload (replays /e/scratch/jureap59/feuer1/sera_probe_compare.py)
# expect top-12 after `..."path":"/testbed"}}` to include `\n</tool_call>`,
# and step-3 (curl_probe4.py upto-8) to NOT degenerate.

# upload (no DB register per feedback_db_register_after_eval; that happens
# after non-zero eval reward is confirmed)
huggingface-cli upload-large-folder \
  laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v7 $DST --repo-type=model
```

## Probe results — duplicating prior work would just re-confirm

I did NOT run the parquet probe the prompt asks for in Step 3, because it has
already been done twice:
- `axolotl_retrain_diagnosis_coderforge.md` H1-H5: tokenizer round-trips, label
  mask textbook-clean, weights converged. The `chat_template.jinja` deletion
  it recommended did NOT fix the model (verified by iter 17h job 391120 — see
  `eval_debug_log.md` lines 160-178). Conclusion: that probe missed the real
  bug.
- `cf_v6_iteration_log.md`: probed the stock-Qwen3-8B prior on the first
  assistant token directly (got `<think>` ≈ 100%) → confirmed CF training data
  needed `<think>` injection. That's the data-level fix, and the v6 dataset
  (`laion/CoderForge-Preview-v6-{316,1000}`) is already published with it.

Re-running probes now would just re-confirm decisions already made. The CF
**fix is data-level (already implemented)** + **retrain is done (already
trained)**. The only remaining items are the upload + smoke tests.

## SLURM job IDs of any retrains kicked off

**None by me.** The relevant in-flight / completed jobs already launched by the
user:
- 391242 — sera-v4-316-v3 (COMPLETED, on HF)
- 392641 — sera-v4-1000-v6 (COMPLETED, on HF)
- 393612 — sera-v4-1000-v7 (RUNNING, 26% through 12 epochs, ETA ~05:30 UTC)
- 392712 — cf-v6-316 (COMPLETED, converted, NOT on HF)
- 392836 — cf-v7-1000 (COMPLETED, converted, NOT on HF)

## Anomalies / concerns

- **Sera v3/v6 still 0% per eval log** — even with the chat_template fix and
  the data scale-up, eval still yielded 0% on swebench-verified-random-100. The
  v7 run (393612, 12 epochs) is the next data-point. If v7 is also 0%, the
  remaining options per `sera_braces_diagnosis.md` are F4 (scale to v4-3160)
  or escalate to data-recipe investigation.
- **CF v6-316 was a 1-epoch smoke test** (49 steps total, last ckpt = 3) — its
  loss is meaningful as "did training even work" but not as a final candidate.
  The production candidate is **v7-1000** (6 epochs).
- **Sera v3 sbatch uses `zero1.json`**, while CF v6/v7 + Sera v6/v7 use
  `zero3_bf16.json` (per `feedback_sft_min_4_nodes.md`). Going forward Sera
  retrains should match the v6/v7 sbatches (`sera_v4_1000_v7.sbatch`), which
  already use zero3_bf16.

## What I changed

Nothing. No configs, no SLURM jobs, no HF uploads, no DB registers. The
prompt's framing was outdated; running its instructions verbatim would have
produced duplicate retrains under names that conflict with the user's current
naming (`-v3` for CF, `-v5` for Sera) and stomped on in-flight work.

## Recommended next action for the user

1. Let job 393612 finish (~3h more wall time at submission of this doc).
2. If you want CF results sooner: run `cf_v6_posttrain.sh 1000` adapted for v7
   path (or just adapt the probe section), confirm coherent output, then
   upload **CF v7-1000** to HF and queue an eval.
3. After Sera v7 finishes: convert + restore tokenizer + smoke-replay + upload
   as `laion/Sera-4.6-Lite-T2-v4-1000-axolotl__Qwen3-8B-v7`, then queue an eval.
4. Per `feedback_db_register_after_eval`, only DB-register either model after
   eval shows reward > 0.
