# SERA v4 Baseline — Jupiter Reproduction

Faithful re-creation of the SERA SFT training pipeline for Qwen3-8B on the JSC Jupiter cluster (GH200, aarch64, CUDA 13), matching Ai2's methodology end-to-end including the Hermes/Qwen3 `<tool_call>` wire-format pre-render that axolotl's stock chatml preset drops.

- **Upstream paper / blog**: https://allenai.org/papers/opencodingagents
- **Upstream code**: https://github.com/allenai/SERA (data-prep) + https://github.com/allenai/SERA-SWE-agent (eval harness)
- **Upstream model**: https://huggingface.co/allenai/SERA-8B
- **Sibling baseline**: `/baselines/coderforge` (same axolotl env, CoderForge-Preview data source)

---

## Iteration history

We use **flat, monotonically-incrementing version suffixes** on HF repos (`-v2`, `-v3`, …). The `Sera-4.6-Lite-T2-v4` segment names the *dataset recipe*; the trailing `-vN` segment is the *training run*. Versions 4 and 5 are skipped to keep the run-version distinct from the dataset-version and to leave room for past artifacts.

| Run | HF suffix | Data source | Rows × epochs | SLURM | Eval | Notes |
|---|---|---|---:|---:|---|---|
| **i1** | (deprecated) | `allenai/Sera-4.5A-Full-T1` | — | — | — | Prototype only |
| **i2** | (deleted) | `allenai/Sera-4.5A-Full-T1` → shareGPT via custom converter | — | — | 3/295 (1%) | Converter silently dropped `tool_calls` → 100% harness-use failures |
| **i3** | (deleted) | `allenai/Sera-4.5A-Full-T1` JSONL → axolotl `chat_template: chatml` | — | — | 0/297 (0%) | chatml preset ignores OpenAI `tool_calls` field → 0 `<tool_call>` tokens in loss (verified via preprocessed-cache decode, 2026-04-22) |
| **i4** | (no suffix; deprecated) | `laion/Sera-4.6-Lite-T2-v4-316` (pre-rendered via `transform_traj_hermes`) | 316 × 3 | (initial) | 0/89 (0%) | Train-time `chat_template: chatml` (bare) mismatched stock Qwen3 template at inference (which strips `<think>` from non-last assistant turns) → multi-turn OOD drift, whitespace collapse after ~2 turns |
| **i5** | `-v2` | `Sera-4.6-Lite-T2-v4-316` | 316 × 3 | 389486 | 0/67 (0%) | **Structural recovery** — coherent `<tool_call>{JSON}</tool_call>` output; but invalid JSON (`"arguments": {…}}}`, 3 closing braces) and wrong tool names (`"view"` instead of `"str_replace_editor"`). Fix applied: `chat_template: tokenizer_default` |
| **i6** | `-v3` | `Sera-4.6-Lite-T2-v4-316` | 316 × **6** | 391242 | 0% | Doubled epochs (`num_epochs: 3 → 6`) to address i5's 120-grad-update floor. Stand-alone training-data probe confirmed clean (0 malformed bodies, valid tool names). Failure mode shifted: turn-1 robust, but the model collapses into a repeating-token attractor (`4.4.4.4…`, `for the for the…`, `\n\n\n…`) at turn ≥3 once a tool observation exceeds ≈20 KB. Greedy decoding does NOT fix it. See `notes/ot-agent/sera_braces_diagnosis.md` for the per-token probability probes that validated this. |
| **i7** | `-v6` | `Sera-4.6-Lite-T2-v4-1000` | 1 000 × 6 | 392641 | 0% | First retrain on the 1 000-row rung. Same long-context degeneracy as i6 — the 3× data + same epoch count was insufficient to escape the attractor. |
| **i8** | `-v7` | `Sera-4.6-Lite-T2-v4-1000` | 1 000 × **12** | 393612 | ~0% | Doubled epochs again (6 → 12) on the 1 000-row data. Long-context collapse is mostly gone; the new dominant failure is **schema-confusion in the `<tool_call>` envelope** (doubly-nested `name`/`arguments` blocks; 4 closing braces vs. 3 opening braces). Fully diagnosed in `eval/README.md` — training-data probe scanned 34 793 `<tool_call>` blocks with 0 doubly-nested rows, so it is **not** a data bug. Hypothesis: still under-trained on the schema distinction between the outer envelope (`{"name": tool, "arguments": {…}}`) and the inner `str_replace_editor` sub-command schema (`{command, path, …}`); the model conflates the two under long-context pressure. |
| **i9** | `-v8` (**current**, ablation series) | `ethanlshen/sera-subset` (mixed stage1+stage2, 48 216 rows, seed=42 shuffle), subset to 316 / 1 000 / 3 160 / 10 000 | × 3 | 404574–404577 | pending | **First run on the SERA-author-blessed pre-rendered dataset.** Supersedes our `subset_sera_v4.py` port. Authors' fully-preprocessed mix of two stages: stage1 (22 972 unresolved rollouts, threshold 0.88) + stage2 (25 244 resolved soft-t0 rollouts). Per-author hyperparameters: 3 epochs, lr 1e-5, b1 0.9 / b2 0.95, wd 0.01, **48-step absolute warmup** (not ratio), global batch 32 on 8 nodes × 4 GPUs (micro=1, grad_accum=1, dp=32), `chat_template: chatml`. Walltime 11:59 per Jupiter's `part_boos+` 12h QoS cap (user spec said 47:59 — Perlmutter-style; Jupiter doesn't allow). **Caveat at SIZE=316:** total grad steps ≈ 30 < warmup floor 48, so LR never exits linear ramp — kept per author spec for controlled comparison across sizes. Preprocess on login node confirmed `<tool_call>` (151657), `</tool_call>` (151658), `<think>` (151667), `<tool_response>` (151666) all in loss with positive label IDs. |

### Naming caveats

- The README and the `notes/` diagnosis docs occasionally referred to runs as `v4-v2` / `v4-v3` mid-iteration; flat `-v2` / `-v3` is the canonical HF-repo form. `-v4` and `-v5` are intentionally unused so that the dataset-version `v4` stays unambiguous.
- The next Sera retrain (after `-v7`) should use `-v8` (skip `-v4`/`-v5`).

### What's still open (`-v7`)

The doubly-nested-name pathology in `i8`/`-v7` is the active blocker. We have three escalation paths, none yet executed (and none should be without explicit user approval):

1. **Parser shim** in harbor's `function_calling` parser — recursively unwrap `{"name": X, "arguments": {"name": Y, "arguments": Z}}` patterns. Highest near-term leverage; unblocks evaluation of every `-v*` weight without retraining. See `eval/README.md` § "Diagnostic — `FormatError` storm at eval time (2026-04-25)".
2. **Scale to `Sera-4.6-Lite-T2-v4-3160`** — next size-ladder rung (2 nodes, ~1 h wall per the node-ladder table below).
3. **Schema-disambiguating data augment** — explicitly distinguish the envelope vs the `str_replace_editor.command` schema in synthetic training rows.

### Additional gotcha (2026-04-23, still applies)

After v4 training completed and models were uploaded to HF, the first eval cycle still showed 0/89 pass. Root cause: **axolotl saves a stripped-down `tokenizer_config.json` + a bare 4-line `chat_template.jinja` that don't handle `tool_calls` or `role:tool` at serve time**, so vLLM silently dropped every `tool_calls` field in the incoming OpenAI-format messages and passed raw tool observations to the model. The SFT'd model had learned the correct wire format but the served prompt was malformed. Ai2 sidesteps this by shipping SERA-8B with tokenizer files **byte-identical** to stock `Qwen/Qwen3-8B` — they overwrote axolotl's bare config at publish time.

**Always apply the tokenizer-restore step** (§ Post-training, step 4) after any axolotl SFT on a Qwen3/Qwen2.5 base.

---

## Upstream `sera/datagen/train` compliance

This section documents, file-for-file, what the upstream `https://github.com/allenai/SERA/tree/main/sera/datagen/train` directory documents, and how our reproduction mirrors or deviates from it. The directory contains a short README, two thin shell entrypoints, four axolotl/llamafactory/unsloth configs, five DeepSpeed presets, and one post-hoc checkpoint converter — that's the whole "training instructions" surface the authors publish.

### File-by-file compliance

| Upstream file | Purpose | Our mirror | Notes |
|---|---|---|---|
| `README.md` (786 B) | Says: "primarily axolotl, validate with llamafactory + unsloth; configs in `train_config/`; frameworks not in SERA deps; install your favorite. Apply `convert_axolotl_checkpoint.py` post-hoc or vLLM/sglang won't load." | This README + `convert_axolotl_checkpoint.py` (Post-training § 1) | Every claim verified end-to-end. We additionally document the byte-identical-tokenizer-restore step that upstream performs but does not document (§ Post-training § 4 below). |
| `train_axolotl_8b.sh` (16 B) | Literally `axolotl train $1`. | `sbatch/axolotl_sera_v4.sbatch` calls `axolotl train $CFG` from inside an `accelerate launch` wrapper for multi-node rendezvous. | Single-node SERA-8B path is otherwise byte-equivalent. Multi-node is our addition (§ Launcher deviation). |
| `train_axolotl_32b.sh` (230 B) | `axolotl train $1 --launcher torchrun -- --nnodes 2 --nproc_per_node 8 --rdzv_id "" --rdzv_backend "" --rdzv_endpoint ""` (rdzv fields left for the user). | Not yet exercised. | We have not done a 32B SERA SFT yet; size-ladder is 8B-only as of i8/`-v7`. When we do, we will base it on this 2-node 8-GPU template and replace `torchrun` with `accelerate launch` (same reason as 8B — Jupiter inter-node `torchrun` c10d rendezvous fails). |
| `convert_axolotl_checkpoint.py` (3.2 KB) | Strips `_checkpoint_wrapped_module.` prefix from state-dict keys. | `baselines/sera/convert_axolotl_checkpoint.py` (local mirror, byte-for-byte). Canonical copy on Jupiter at `/e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py`. | Verbatim. We invoke it as Post-training § 1 on every run. |
| `train_config/axolotl_qwen3_8b.yaml` (1.2 KB) | The 8B SFT recipe (full hyperparameters). | `configs/template_qwen3_8b_sera_v4.yaml` (this directory). | Full hyperparameter compliance table below. |
| `train_config/axolotl_qwen3_32b.yaml` | 32B variant. | Not yet mirrored. | Will be `configs/template_qwen3_32b_sera_v4.yaml` when we do a 32B run. |
| `train_config/axolotl_qwen25_32b.yaml` | Qwen2.5-32B variant (their original base model). | Not used. | We only target Qwen3-8B; Qwen2.5 is upstream-historical. |
| `train_config/llamafactory_qwen3_full_sft.yaml` | Cross-validation against LLaMA-Factory. | Not used for SERA. | We use LLaMA-Factory for *other* SFTs in this repo (Qwen3.5 baselines), but for SERA we stay on the primary axolotl path the authors recommend. |
| `train_config/unsloth_qwen3_moe_qlora.yaml` | Unsloth QLoRA cross-validation. | Not used. | LoRA is incompatible with our full-SFT comparison goal. |
| `train_unsloth.sh`, `train_unsloth_lora.py` | Unsloth entrypoint + script. | Not used. | Same. |
| `filter_dataset_hf.py` (14 KB) | Filters a raw trajectory dataset to the SFT-ready set. | Not used. | We pull `allenai/Sera-4.6-Lite-T2` directly — that's already the post-filter set per the Together AI blog (file `sera-4.6-lite-t2_36083_string_enriched.jsonl`). The 4.5A-Full-T1 dataset would have required this filter; v4.6-Lite-T2 does not. |
| `deepspeed_configs/zero1.json` | ZeRO-1, no offload. | Used only by the deprecated i3-era `sbatch/axolotl_sera_v3.sbatch`. | OOMs at `sequence_len: 32768` on Jupiter GH200 96 GB once we disable CCE — see deviation below. |
| `deepspeed_configs/zero2.json` | ZeRO-2, default offload. | Not used. | Default offloads optimizer state to CPU → DeepSpeedCPUAdam JIT-compile → fails on Jupiter's GCC 14.3 with `-march=armv9-a+...+nossbs+nopauth`. |
| `deepspeed_configs/zero3.json` | ZeRO-3, default offload. | Not used. | Same CPUAdam issue. |
| `deepspeed_configs/zero3_bf16_cpuoffload_all.json` | ZeRO-3 + full CPU offload. | Not used. | Same CPUAdam issue. |
| `deepspeed_configs/zero3_bf16_cpuoffload_params.json` | ZeRO-3 + params-only CPU offload. | Not used. | Same CPUAdam issue. |
| (no upstream equivalent) | — | `zero3_bf16.json` (in this directory; ZeRO-3 + bf16 + no CPU offload). | This is our addition. Keeps Adam on GPU (sidesteps the `-march=armv9-a` CPUAdam compile bug) while still sharding params + grads + optimizer state. Required for 32k sequence length on GH200. See § Config gotchas. |

### Hyperparameter compliance — `axolotl_qwen3_8b.yaml`

Every numeric and structural field in upstream's Qwen3-8B config, mapped to our `template_qwen3_8b_sera_v4.yaml`:

| Field | Upstream value | Our value | Match? |
|---|---|---|---|
| `base_model` | `Qwen3-8B` | `Qwen/Qwen3-8B` | ✓ (same model, fully-qualified path) |
| `load_in_8bit` / `load_in_4bit` | `false` / `false` | `false` / `false` | ✓ |
| `chat_template` | `chatml` | `chatml` (i4) → `tokenizer_default` (i5+) | ✗ — see deviation 1 below |
| Data type | `chat_template` | `chat_template` | ✓ |
| `field_messages` | `messages` | `messages` | ✓ |
| `message_field_training` | `train` | `train` | ✓ |
| `ds_type` | `json` | `json` | ✓ |
| `sequence_len` | `32768` | `32768` | ✓ |
| `gradient_accumulation_steps` | `8` | `8` | ✓ |
| `micro_batch_size` | `1` | `1` | ✓ |
| `num_epochs` | `3` | `3` (i4–i5) → `6` (i6, i7) → `12` (i8) | ✗ — see deviation 2 below |
| `optimizer` | `adamw_torch` | `adamw_torch` | ✓ |
| `lr_scheduler` | `cosine` | `cosine` | ✓ |
| `learning_rate` | `1e-5` | `1e-5` | ✓ |
| `adam_beta1` / `adam_beta2` | `0.9` / `0.95` | `0.9` / `0.95` | ✓ |
| `weight_decay` | `0.01` | `0.01` | ✓ |
| `warmup_ratio` | `0.1875` | `0.1875` | ✓ |
| `bf16` | `auto` | `auto` | ✓ |
| `tf32` | `false` | `false` | ✓ |
| `gradient_checkpointing` | `true` | `true` | ✓ |
| `activation_offloading` | `true` | `true` | ✓ |
| `flash_attention` | `true` | `true` | ✓ (via the prebuilt aarch64 wheel from `mjun0812/flash-attention-prebuild-wheels`) |
| `evals_per_epoch` | `0` | `0` | ✓ |
| `save_strategy` | `epoch` | `epoch` | ✓ |
| `logging_steps` | `1` | `1` | ✓ |
| `loss_watchdog_threshold` / `loss_watchdog_patience` | `5.0` / `3` | `5.0` / `3` | ✓ |
| `plugins.CutCrossEntropyPlugin` | enabled | **disabled** | ✗ — see deviation 3 below |
| `deepspeed` | `zero1.json` | `zero3_bf16.json` | ✗ — see deviation 4 below |
| `wandb_*` | unset | unset (`WANDB_MODE=offline` in sbatch) | ✓ (by intent — Jupiter has no W&B network access) |
| `hub_model_id` / `hub_strategy` | unset | unset | ✓ (we additionally hard-omit; under `HF_HUB_OFFLINE=1` the in-train `init_hf_repo` would crash) |
| `max_grad_norm` | unset | `1.0` (explicit) | ✗ — see deviation 5 below |
| `dataset.path` | `# FILL IN` | `laion/Sera-4.6-Lite-T2-v4-__SIZE__` | ✓ (we filled it in as instructed) |
| `output_dir` | `# FILL IN` | `$CHECKPOINTS_DIR/sera-v4-${SIZE}-axolotl__Qwen3-8B[-vN]` | ✓ |

### Deviations (each justified)

1. **`chat_template: chatml` → `tokenizer_default`** (i5+). The upstream `chatml` preset works **iff** the training data already inlines tool calls as `<tool_call>{JSON}</tool_call>` text inside `content` (which our `subset_sera_v4.py` ensures, mirroring upstream's `sera/datagen/data/postprocess/utils.py::transform_traj_hermes`). i4 used `chatml` per upstream and produced multi-turn whitespace collapse because the bare chatml render at train time differs from the stock Qwen3 template at inference (Qwen3's template strips `<think>` from non-last assistant turns; chatml does not). `tokenizer_default` makes the train-time render byte-identical to the served render, eliminating the OOD shift on turn ≥ 2. This is not a hyperparameter change in spirit — it's a render-fidelity fix that's a no-op for upstream (whose inference path uses chatml as well, since they republish the model with stock Qwen3 tokenizer files).

2. **`num_epochs: 3 → 6 → 12`** on size-ladder rungs. Upstream trains 3 epochs on the **full 36 083-row dataset** (≈13.5k gradient updates at GA=8). On size-ladder rungs (316, 1 000) at 3 epochs we have 120 / 375 grad updates, far below the SGD floor required to lock the `<tool_call>` envelope schema in. We doubled to 6 epochs (i6, i7) and 12 epochs (i8) to keep total grad-updates approximately 1k–1.5k. **Once we reach the 3 160-row+ ladder rungs, we will revert to upstream's 3 epochs.**

3. **`CutCrossEntropyPlugin` disabled.** Upstream enables it. On aarch64 + torch 2.9.1+cu130 + FA2, CCE causes a bf16 gradient explosion (`grad_norm` 9.8e+11 within 3–7 steps) → loss NaN → silently masked as 0 by axolotl's loss-watchdog. Confirmed reproducibly on Jupiter GH200; the same env on x86_64 H100 doesn't show this (so this is genuinely an aarch64-specific axolotl/CCE/torch interaction). We compensate by raising VRAM headroom via `zero3_bf16.json` (deviation 4); the 8B at 32k seq fits in GH200 96 GB without CCE.

4. **`zero1.json` → `zero3_bf16.json`** (custom). `zero1.json` OOMs at `sequence_len: 32768` on GH200 96 GB once CCE is disabled (CCE compresses peak activation memory; without it ZeRO-1 alone is insufficient). The two upstream ZeRO-3 variants both CPU-offload optimizer state, which triggers DeepSpeedCPUAdam's JIT compile against Jupiter's GCC 14.3 — the compiler rejects `-march=armv9-a+...+nossbs+nopauth` and the build fails. Our `zero3_bf16.json` (in this directory) is a ZeRO-3 + bf16 config with all CPU-offload disabled, sharding params + grads + optimizer state on GPU. This is a strict superset of ZeRO-1's memory model from a correctness standpoint.

5. **`max_grad_norm: 1.0` set explicitly.** Upstream leaves it unset (axolotl default is also 1.0). Belt-and-suspenders against the bf16 grad explosion mode in deviation 3 above; if CCE is ever re-enabled accidentally, the explicit ceiling clips it.

### Post-training step the upstream README does not document

The upstream `README.md` only documents `convert_axolotl_checkpoint.py` as a post-hoc step. **It does not document** that axolotl's saved `tokenizer_config.json` (~665 B) and `chat_template.jinja` (4 lines) are bare templates that drop `tool_calls` and `role: tool` at vLLM serve time, producing 0% pass rate on SWE-bench harness evals despite training being healthy. We discovered this on 2026-04-23 by comparing our uploaded checkpoint's tokenizer files to `allenai/SERA-8B`'s — Ai2's are **byte-identical** to stock `Qwen/Qwen3-8B`, with no `chat_template.jinja` sibling present, indicating the upstream authors performed a manual tokenizer-restore + jinja-delete between training and publishing the released SERA-8B model.

Our Post-training § 4 below scripts that restore. We flag this as a documentation gap upstream and an integration point we'd be happy to upstream as a PR if useful.

### Pre-train data prep

Upstream documents the dataset filter (`filter_dataset_hf.py`) and post-process renderer (`sera/datagen/data/postprocess/utils.py::transform_traj_hermes`). Our `scripts_dataset_build/subset_sera_v4.py` is a faithful port of `transform_traj_hermes`:
- Assistant turns: `content = original <think>…</think>\n{prose}\n\n<tool_call>\n{"name":"…","arguments":{…}}\n</tool_call>` (one block per OpenAI tool_call, concatenated with `\n`).
- Tool turns: rewritten to `role: "user"` with content wrapped as `<tool_response>\n{text}\n</tool_response>`.
- System / user turns: `content` passed through unchanged (we flatten `[{"type":"text","text":…}]` to a string).
- `train: bool` per message (True only for assistant), drops upstream metadata fields (`thought`, `action`, `agent`, `message_type`, `tool_call_ids`, `cache_control`).

We pull `allenai/Sera-4.6-Lite-T2` rather than running `filter_dataset_hf.py` because Together AI's blog post identifies `sera-4.6-lite-t2_36083_string_enriched.jsonl` as the post-filter SFT set (vs. `Sera-4.5A-Full-T1`, the 72k-row pre-filter pool we mistakenly used in i2/i3).

---

## Dataset

Uploaded to `laion/Sera-4.6-Lite-T2-v4(-<SIZE>)` as raw JSONL after pre-rendering via `transform_traj_hermes`:

| Dataset | Rows | Used by |
|---|---:|---|
| `laion/Sera-4.6-Lite-T2-v4` | 36 083 | (full — not yet uploaded) |
| `laion/Sera-4.6-Lite-T2-v4-316` | 316 | i4, i5 (`-v2`), i6 (`-v3`) |
| `laion/Sera-4.6-Lite-T2-v4-1000` | 1 000 | i7 (`-v6`), i8 (`-v7`, current) |
| `laion/Sera-4.6-Lite-T2-v4-3160` | 3 160 | (pending rollout — F4 fallback per `sera_braces_diagnosis.md`) |
| `laion/Sera-4.6-Lite-T2-v4-10000` | 10 000 | (pending rollout) |
| `laion/Sera-4.6-Lite-T2-v4-31600` | 31 600 | (pending rollout) |

Each row carries `messages: list[{role, content, train}]`, where:
- Roles are `system | user | assistant` (upstream `role: tool` observations are rewritten to `role: user` with `<tool_response>…</tool_response>` wrapping).
- `content` for assistant turns contains the original `<think>…</think>` block + visible prose + one or more `<tool_call>\n{JSON}\n</tool_call>` blocks (appended per upstream `transform_traj_hermes`).
- `train: bool` on each message is the per-message loss mask consumed by axolotl's `message_field_training: train`.
- Plus `instance_id`, `source="allenai/Sera-4.6-Lite-T2"`, and `docker_image`/`problem_statement` passthroughs.

Sampling: deterministic random, seed=42, row-indexed into the full 36,083-row source. Row subsets are nested.

### Why `Sera-4.6-Lite-T2` (not `Sera-4.5A-Full-T1`)?

Discovered 2026-04-23 while investigating the v3 eval failure: the Together AI blog references `Sera-4.6-Lite-T2` as the SFT training set (file: `sera-4.6-lite-t2_36083_string_enriched.jsonl`). The 4.5A-Full-T1 dataset (72k rows, used by v2/v3) is the raw pre-filter trajectory pool, not the filtered+deduplicated set used in the paper.

---

## Environment (Jupiter, aarch64, CUDA 13)

`sera-axolotl` conda env on Jupiter. Core pins after install:

| Package | Version |
|---|---|
| python | 3.12.13 |
| torch | 2.9.1+cu130 |
| axolotl | 0.16.0.dev0 (from v0.16.1 tag) |
| transformers | 5.5.0 |
| accelerate | 1.13.0 |
| deepspeed | 0.18.9 |
| datasets | 4.5.0 |
| flash-attn | 2.8.3+cu130torch2.9 (prebuilt wheel) |
| triton | 3.5.1 |

Notable absences (excluded by axolotl's aarch64 filter in `setup.py`): `torchao`, `fla-core`, `flash-linear-attention`.

### Install recipe

Axolotl has two issues to work around on aarch64 + torch 2.9:

1. **`torchao==0.17.0` hard-pins `torch==2.8.0`** → would clobber our Jupiter-safe `torch==2.9.1+cu130`. Axolotl's `setup.py` already filters torchao on aarch64, but uv's prebuilt-wheel metadata resolution doesn't run that code. Fix: force a source build with `--no-build-isolation` so the filter runs.
2. **flash-attn** has no prebuilt aarch64 wheel for torch 2.9. Use [mjun0812/flash-attention-prebuild-wheels](https://github.com/mjun0812/flash-attention-prebuild-wheels) release `v0.7.16`.

```bash
conda activate sera-axolotl  # already has torch 2.9.1+cu130

# Axolotl v0.16.1
mkdir -p /e/scratch/jureap59/feuer1/code && cd /e/scratch/jureap59/feuer1/code
git clone https://github.com/axolotl-ai-cloud/axolotl.git && cd axolotl
git checkout v0.16.1

uv pip install "setuptools>=64" wheel "setuptools_scm>=8" "packaging==26.0"
uv pip install -e . --no-build-isolation

# Deepspeed (excluded from axolotl deps)
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
uv pip install "deepspeed>=0.18.6,<0.19.0" --no-build-isolation

# flash-attn prebuilt wheel
cd /tmp
WHL=flash_attn-2.8.3+cu130torch2.9-cp312-cp312-manylinux_2_34_aarch64.whl
wget -q "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/$WHL"
uv pip install --no-deps "./$WHL"

# axolotl-ai-cloud's cut-cross-entropy fork (required by import chain even though we disable the plugin in config)
uv pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@63b15e6" --no-build-isolation
```

### Mandatory axolotl source patches

1. **`src/axolotl/utils/callbacks/qat.py`** — guard torchao imports in a `try/except ImportError:` block; assign unreachable stub classes so `isinstance()` checks in `toggle_fake_quant` still return False. Without this, `from axolotl.core.builders import …` fails on aarch64.

2. **`convert_axolotl_checkpoint.py`** — kept at `/e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py` on Jupiter (also tracked locally as `baselines/sera/convert_axolotl_checkpoint.py`). Strips `_checkpoint_wrapped_module.` prefixes from state_dict keys so vLLM / sglang can load the checkpoint.

---

## Data subsetter — faithful `transform_traj_hermes` port

`subset_sera_v4.py` (in `scripts_dataset_build/`) reads the upstream `allenai/Sera-4.6-Lite-T2` JSONL and applies the same render logic as Ai2's `sera/datagen/data/postprocess/utils.py::transform_traj_hermes`:

- Assistant turns → `content` = original `<think>…</think>\n{prose}\n\n<tool_call>\n{"name": "…", "arguments": {…}}\n</tool_call>` (one `<tool_call>` block per OpenAI tool_call, concatenated with `\n`).
- Tool turns → `role: "user"` with content wrapped as `<tool_response>\n{text}\n</tool_response>`.
- System / user turns → pass through `content` unchanged (flatten list-of-dicts `[{"type":"text","text":…}]` to string if needed).

Adds `train: bool` per message (True only for assistant), drops upstream metadata fields (`thought`, `action`, `agent`, `message_type`, `tool_call_ids`, `cache_control`) that axolotl doesn't read.

Run locally (reads `HF_TOKEN` from env):
```bash
source /Users/benjaminfeuer/Documents/secrets.env
cd /Users/benjaminfeuer/Documents/scripts_dataset_build
python subset_sera_v4.py
```

The default `SUBSET_SIZES = [316, 1000]` — edit the constant to upload more sizes on the ladder.

---

## Axolotl config

- `configs/template_qwen3_8b_sera_v4.yaml` — axolotl config template (sed-substitute `__SIZE__` for a specific subset).

Key lines:
```yaml
base_model: Qwen/Qwen3-8B
chat_template: chatml   # passes <tool_call>/<tool_response> wire tokens through verbatim because the subsetter already rendered them into content
datasets:
  - path: laion/Sera-4.6-Lite-T2-v4-__SIZE__
    data_files:
      - sera-4.6-lite-t2_v4___SIZE__.jsonl
    type: chat_template
    field_messages: messages
    ds_type: json
    message_field_training: train
sequence_len: 32768
num_epochs: 3
learning_rate: 1e-5
lr_scheduler: cosine
warmup_ratio: 0.1875
gradient_accumulation_steps: 8
micro_batch_size: 1
weight_decay: 0.01
max_grad_norm: 1.0
save_strategy: epoch
deepspeed: /e/scratch/jureap59/feuer1/code/axolotl/deepspeed_configs/zero3_bf16.json
# hub_model_id and hub_strategy are intentionally absent — with HF_HUB_OFFLINE=1
# set on Jupiter compute nodes, init_hf_repo would crash on job start.
```

### Config gotchas

- **Omit `hub_model_id` / `hub_strategy`** — transformers' `init_hf_repo` runs at job start and calls `create_repo` → HF API → `OfflineModeIsEnabled` crash under `HF_HUB_OFFLINE=1`. Push manually after training.
- **Disable `CutCrossEntropyPlugin`** — on aarch64+torch2.9+FA2, CCE causes bf16 grad explosion (grad_norm 9.8e+11) → loss → NaN → masked as 0 after 3-7 steps. Comment out under `plugins:`.
- **Set `max_grad_norm: 1.0` explicitly** as belt-and-suspenders.
- **Use `zero3_bf16.json`** — not `zero1` (OOM without CCE), not `zero2/zero3` defaults (offload Adam to CPU → `DeepSpeedCPUAdam` JIT compile → GCC 14.3 rejects `-march=armv9-a+…+nossbs+nopauth`).

---

## sbatch env (Jupiter compute)

`sbatch/axolotl_sera_v4.sbatch` — SLURM template. Sed-substitute `__SIZE__` and `__NODES__` before submitting. Key env block:

```bash
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export GCC_HOME=/e/software/default/stages/2026/software/GCCcore/14.3.0
export CC=$GCC_HOME/bin/gcc
export CXX=$GCC_HOME/bin/g++         # Triton JIT kernels need these
export PATH=$CUDA_HOME/bin:$GCC_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$GCC_HOME/lib64:${LD_LIBRARY_PATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export WANDB_MODE=offline
export AXOLOTL_DO_NOT_TRACK=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1

# Optional for multi-node: force IB-interconnect FQDN to avoid localhost:29500 / IPv6 failures on some nodes
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)-interconnect-1.jupiter.internal
export NCCL_SOCKET_IFNAME=ib0 NCCL_IB_TIMEOUT=60
```

Node ladder (empirical, 3-epoch walls on Jupiter GH200):

| Subset | Nodes | Approx total wall |
|---|---|---|
| 316 | 1 | ~28 min |
| 1000 | 1 | ~1h17m |
| 3160 | 2 | ~1h |
| 10000 | 4 | ~2h |
| 31600 | 8 | ~3h |

---

## Pre-download (required on no-internet clusters)

Jupiter compute nodes have no internet. Run `axolotl preprocess` once on the login node for each subset so the tokenized dataset cache lives under `dataset_prepared_path`:

```bash
conda activate sera-axolotl
source /e/scratch/jureap59/feuer1/OpenThoughts-Agent/hpc/dotenv/jupiter.env
source ~/secrets.env
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
cd /e/scratch/jureap59/feuer1/code/axolotl

SIZE=316
CFG=/e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_sera_v4_${SIZE}.yaml
sed "s/__SIZE__/${SIZE}/g" \
  /e/scratch/jureap59/feuer1/code/axolotl_configs/template_qwen3_8b_sera_v4.yaml > $CFG
axolotl preprocess $CFG
```

### Training-cache sanity check (verify `<tool_call>` is in loss)

This is the single most useful guardrail. If it fails, the subsetter or template is broken:

```python
from datasets import load_from_disk
import glob, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
p = glob.glob(f"/e/data1/datasets/playground/ot-baf/axolotl_dataset_cache/sera-v4-316/*")[0]
ds = load_from_disk(p)
ids = np.array(ds[0]["input_ids"])
labels = np.array(ds[0]["labels"])
tc = tok.convert_tokens_to_ids("<tool_call>")  # 151657
in_loss = int(((ids == tc) & (labels != -100)).sum())
print(f"<tool_call> tokens in loss (row 0): {in_loss}")  # must be > 0
```

Expected: 52 for row 0 of sera-v4-316; 23 for row 0 of sera-v4-1000 (varies per row; floor is ~10 assistant turns × 1 tool_call each).

---

## Training launch

```bash
SIZE=316 ; NODES=1
cd /e/scratch/jureap59/feuer1/code/axolotl_sbatch
sed -e "s/__SIZE__/$SIZE/g" -e "s/__NODES__/$NODES/g" \
  template_sera_v4.sbatch > sera_v4_${SIZE}.sbatch
sbatch sera_v4_${SIZE}.sbatch
```

Multi-node launch uses `srun --ntasks-per-node=1` + `accelerate launch --num_machines=$SLURM_JOB_NUM_NODES` with rendezvous on the first node of the allocation and DeepSpeed ZeRO-3-bf16. Chain retries with `--dependency=afterany:$J1` if you expect timeouts or transient NCCL failures.

---

## Post-training (axolotl → vLLM-servable)

Four steps. All required; step 4 is the newest and most commonly forgotten.

### 1. Strip FSDP prefixes

Axolotl with gradient checkpointing writes state_dict keys prefixed with `_checkpoint_wrapped_module.` — vLLM / sglang won't load these.

```bash
SRC=$CHECKPOINTS_DIR/sera-v4-${SIZE}-axolotl__Qwen3-8B
DST=$CHECKPOINTS_DIR/sera-v4-${SIZE}-axolotl__Qwen3-8B-converted

# First remove intermediate checkpoint-N dirs so we don't upload cruft
rm -rf $SRC/checkpoint-* $SRC/.cache

python /e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py $SRC $DST
# Converted keys should have no `_checkpoint_wrapped_module.` and total ~399 on Qwen3-8B
```

### 2. Secret scan

```bash
grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34})' $DST
```

### 2b. ★ Rewrite (or delete) axolotl's `README.md` BEFORE upload ★

Axolotl writes a `README.md` to the checkpoint root with YAML frontmatter that **HF Hub rejects as invalid metadata**, causing `upload-large-folder` to commit-loop forever ("Failed to commit: Invalid metadata in README.md. Will retry with less files in next batch."). The bytes of all other files upload fine via LFS, but the commit never lands — the repo stays empty except for `.gitattributes`.

The two offending fields in axolotl's frontmatter are typically:

```yaml
datasets:
- /e/data1/datasets/playground/ot-baf/.../<size>.jsonl   # local FS path — HF expects org/name
model-index:
- name: e/data1/datasets/playground/ot-baf/checkpoints/<run-name>   # path-like name with extra slashes
```

HF validates these as either dataset-repo IDs or HF-conformant names. A leading `/` or multi-slash path fails validation. Fix is to **overwrite the README.md before the upload** (axolotl's auto-generated content is mostly throwaway anyway):

```bash
cat > $DST/README.md <<'MD'
---
library_name: transformers
base_model: Qwen/Qwen3-8B
tags:
- generated_from_trainer
- axolotl
- sera
- sft
- qwen3
license: apache-2.0
---

# <run-name>

SFT of Qwen/Qwen3-8B on … See \`baselines/sera/README.md\` for full reproduction details.

## Hyperparameters

- learning_rate: 1e-5
- batch_size: 32 (global; micro=1, grad_accum=1, dp=32)
- num_epochs: 3
- warmup_steps: 48
- adam_beta1: 0.9, adam_beta2: 0.95
- weight_decay: 0.01
- sequence_len: 32768
- chat_template: chatml
- bf16, deepspeed zero3 (no CPU offload)
MD
```

If you've already kicked off `upload-large-folder` and it's stuck in the "Invalid metadata in README.md" retry loop, kill the process, fix the README, and re-launch. The LFS bytes are already on HF's content-addressed store and will be skipped on the second pass — only the commit needs to retry.

### 3. Upload weights to HF

```bash
source ~/secrets.env
huggingface-cli upload-large-folder \
  laion/Sera-4.6-Lite-T2-v4-${SIZE}-axolotl__Qwen3-8B \
  $DST --repo-type=model
```

### 4. ★ Restore stock Qwen3-8B tokenizer files AND delete `chat_template.jinja` (CRITICAL) ★

Axolotl saves a stripped-down `tokenizer_config.json` (~665 bytes) **and** a bare 4-line `chat_template.jinja`. Both must be fixed; restoring only `tokenizer_config.json` is **not enough**, because HF transformers + vLLM both prioritize a sibling `chat_template.jinja` file over the `chat_template` string embedded inside `tokenizer_config.json`. If the jinja file is left in place, you'll see the same degenerate output (`\n\n]\n\n]\n\n]…` to max tokens, `tool_calls=None`, SWE-agent logs `Your last output did not use any tool calls!`) as a fully-broken tokenizer.

The bare template:
- Does NOT loop `message.tool_calls` → drops tool-call structured fields from incoming OpenAI messages at serve time
- Does NOT wrap `role: "tool"` content in `<tool_response>…</tool_response>`
- Does NOT declare `<tool_call>`/`</tool_call>`/`<think>` in `added_tokens_decoder`

vLLM loads these on server startup and renders every SWE-agent prompt with the broken template → model sees a malformed prompt with no tool-call wire tokens → out-of-distribution → degenerate loops → 0% pass rate on SWE-bench harness evals, even though training was healthy. First confirmed 2026-04-23 (eval debug log: `notes/ot-agent/eval_debug_log.md`).

Ai2 sidesteps this: SERA-8B's `tokenizer_config.json` is **byte-identical** to stock `Qwen/Qwen3-8B`, AND the SERA-8B repo has **no** `chat_template.jinja` sibling. Mirror both:

```bash
source ~/secrets.env
python - <<'PY'
import os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN'])
REPO = "laion/Sera-4.6-Lite-T2-v4-${SIZE}-axolotl__Qwen3-8B-v7"  # adjust per run
BASE = "Qwen/Qwen3-8B"

# (a) Overwrite the four tokenizer files with stock base-model versions.
for f in ['tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']:
    local = hf_hub_download(BASE, f, token=os.environ['HF_TOKEN'])
    api.upload_file(path_or_fileobj=local, path_in_repo=f,
                    repo_id=REPO, repo_type='model',
                    commit_message='Restore stock Qwen3-8B tokenizer (axolotl bare template fix)')
    print(f"  ✓ overwrote {f}")

# (b) Delete axolotl's bare chat_template.jinja so HF/vLLM fall back to the
#     restored tokenizer_config.json["chat_template"]. Stock Qwen/Qwen3-8B
#     has no chat_template.jinja, and allenai/SERA-8B also has none — mirror that.
try:
    api.delete_file(path_in_repo='chat_template.jinja',
                    repo_id=REPO, repo_type='model',
                    commit_message='Remove axolotl bare chat_template.jinja — fall back to tokenizer_config.json template')
    print("  ✓ deleted chat_template.jinja")
except Exception as e:
    print(f"  · chat_template.jinja already absent ({e})")
PY
```

**Verify both** — failing either check leaves the broken-template foot-gun in place:
1. `api.model_info(REPO, files_metadata=True)` → `tokenizer_config.json` should be ~9 700 B (not ~665), and there should be **no** `chat_template.jinja` entry.
2. The `chat_template` field embedded inside `tokenizer_config.json` should be ~4 168 chars and contain both the substrings `"tool_calls"` and `"tool_response"`.

### 4b. (Optional but recommended) Pin greedy decoding via `generation_config.json`

Per the per-token probe in `notes/ot-agent/sera_braces_diagnosis.md` § 2, the small-but-real `}}}` sampling glitch happens because vLLM honors the model's `generation_config.json` defaults (`temperature: 0.6, top_k: 20`) when the client sends no temperature override. Under those samplers the bug-token `}}\n` (p ≈ 0.0002) is inside top-k and occasionally fires; under greedy (`temperature: 0`) the correct `}\n` wins decisively (p ≈ 0.997).

If you don't trust every downstream client to send `temperature=0`, overwrite the served `generation_config.json` to pin greedy:

```python
import json, tempfile, os
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(token=os.environ['HF_TOKEN'])
local = hf_hub_download(REPO, 'generation_config.json', token=os.environ['HF_TOKEN'])
cfg = json.load(open(local))
cfg.update({'do_sample': False, 'temperature': 0.0, 'top_k': 1})
with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
    json.dump(cfg, f, indent=2); patched = f.name
api.upload_file(path_or_fileobj=patched, path_in_repo='generation_config.json',
                repo_id=REPO, repo_type='model',
                commit_message='Pin greedy decoding (do_sample=false, temperature=0) at serve time')
```

This does NOT fix the long-context degeneracy or the doubly-nested-name pathology — those are training-signal problems, not sampling glitches. It just removes the `}}}` red herring so debug attention lands on the real bugs.

### 5. Register in the unified DB

```bash
python scripts/database/manual_db_push.py \
  --hf-model-id laion/Sera-4.6-Lite-T2-v4-${SIZE}-axolotl__Qwen3-8B \
  --base-model Qwen/Qwen3-8B \
  --dataset-name laion/Sera-4.6-Lite-T2-v4-${SIZE}
```

### 6. Clean up local checkpoint dirs

```bash
rm -rf $SRC   # pre-convert checkpoint; DST is what we uploaded. Frees ~16 GB per model.
```

---

## Evaluation

SERA was trained on **SWE-agent JSON tool-calling format**:
`<think>...</think>...<tool_call>{"name":"<tool>","arguments":{...}}</tool_call>`. Pair it with the matching harbor harness (`swe_agent_ctx32k_eval_.yaml`) and the SERA team's SWE-agent scaffold config — DO NOT use the OpenHands harness for SERA.

### Pinggy and Daytona

Both clusters route the vLLM endpoint through pinggy. URL+token pairs are catalogued in `/Users/benjaminfeuer/Documents/notes/ot-agent/pinggy_bank.md`. Pairs 1–7 are reserved for sibling experiments; 8–10 are the rotating slots for ad-hoc evals (8 may be in use by a labmate — confirm first).

Daytona key MUST be `$DAYTONA_BASE_API_KEY` — `DAYTONA_API_KEY` is the RL-org key and blocks declarative builds (failure mode: `DaytonaValidationError`).

### Critical: bump `agent.max_requeries`

SWE-agent's default is `agent.max_requeries: 3`. After 3 consecutive `FormatError` / `_BlockedActionError` / `BashIncorrectSyntaxError`, the agent emits **"Exit due to repeated format/blocklist/bash syntax errors"** and forfeits — even if it was making real progress. Sera-trained 8B models hit this floor often, so we override the SWE-agent config via the `SWEAGENT_CONFIG` env var pointing to a patched gist of SERA's [e2e.yaml](https://github.com/allenai/SERA/blob/main/sera/configs/sweagent/e2e.yaml) with `max_requeries: 50`:

```
SWEAGENT_CONFIG=https://gist.githubusercontent.com/penfever/f327eabde26934630ee2aea1a59bb511/raw/sera_e2e_patched.yaml
```

Harbor's `swe_agent.py` reads `os.environ["SWEAGENT_CONFIG"]` — if it's a URL, harbor curls it into the daytona container. If it's a path, it's read directly inside the container. The URL form is portable across clusters.

To inject the env var into a launcher-generated sbatch:
```bash
SB=experiments/<run_dir>/sbatch/<job_name>__eval.sbatch
sed -i '/^set -eo pipefail$/a export SWEAGENT_CONFIG="https://gist.githubusercontent.com/penfever/f327eabde26934630ee2aea1a59bb511/raw/sera_e2e_patched.yaml"' "$SB"
sbatch "$SB"
```

If you need a different override, edit the gist (or fork it) and update the URL.

### Launch — Perlmutter (A100, has internet)

```bash
ssh perlmutter
source ~/.bashrc; source ~/secrets.env; module load conda; conda activate dcagent
cd $SCRATCH/OpenThoughts-Agent && git pull
source hpc/dotenv/perlmutter.env
git submodule update --init --remote sft/llamafactory
cd $SCRATCH/SkyRL && git pull && cd $SCRATCH/harbor && git pull
cd $SCRATCH/OpenThoughts-Agent

python -m hpc.launch \
  --job_type eval \
  --model_path laion/<sera_v7_hub_id> \
  --tasks_input_path DCAgent2/swebench-verified-random-100-folders \
  --trace_harbor_config hpc/harbor_yaml/eval/swe_agent_ctx32k_eval_.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_8b_vllm_serve_32k_4xA100.yaml \
  --trace_agent_name swe-agent \
  --daytona_api_key "$DAYTONA_BASE_API_KEY" \
  --pinggy_persistent_url <pair-N URL> --pinggy_token <pair-N token> \
  --time_limit 11:59:00 \
  --num_nodes 1 --gpus_per_node 4 \
  --trace_n_concurrent 16

# After hpc.launch prints the SLURM job id, patch the sbatch with SWEAGENT_CONFIG
# (see above) and re-sbatch — the launcher submits without the override by default.
```

`--datagen_config` MUST be the A100 variant on Perlmutter; the GH200 yaml uses `--all2all-backend pplx` which crashes on A100.

### Launch — Jupiter (GH200, no internet on compute)

Replace `qwen3_8b_vllm_serve_32k_4xA100.yaml` → `qwen3_8b_vllm_serve_32k_4xGH200.yaml`. Otherwise identical. Login node has direct HF Hub access; the launcher pre-downloads the model into `$HF_HUB_CACHE` before submitting.

### Health check (15 min after submit)

Per `feedback_eval_15min_infra_check.md`, every active eval gets a 15-minute infra cadence:
- vLLM endpoint responds (check `vllm_endpoint.json` → `endpoint_url`).
- Pinggy tunnel alive (`curl -sSL https://<pair>.a.pinggy.link/health`).
- Daytona reachable (look for `DaytonaValidationError` or `Bearer token is invalid` in trial logs).
- Trial throughput accumulating (`find $TRIALS -maxdepth 2 -name result.json | wc -l` rising).

### Aggregating results

```bash
P=experiments/<run_dir>/trace_jobs/<run_tag>
find $P -maxdepth 1 -mindepth 1 -type d | wc -l        # total trials launched
find $P -maxdepth 2 -name result.json | wc -l           # trials completed
python3 -c "
import json, glob
n = ok = 0
for r in glob.glob('$P/*/result.json'):
    d = json.load(open(r))
    n += 1
    if (d.get('verifier_result') or {}).get('rewards', {}).get('reward', 0) > 0: ok += 1
print(f'pass: {ok}/{n} = {100*ok/max(n,1):.1f}%')
"
```

### Failure-mode notes (from prior iterations)

Infrastructure / harness:

- **Wrong harness (OpenHands instead of SWE-agent)** → 0% pass; the model emits `<tool_call>{...}</tool_call>` JSON, but the OpenHands harness expects `<function=NAME>...</function>` XML and parses nothing.
- **Default `max_requeries: 3`** → trials forfeit prematurely with the format/blocklist/bash exit even though the agent was producing valid actions on most turns.
- **`DaytonaValidationError` flood** → wrong daytona key; switch to `DAYTONA_BASE_API_KEY`.
- **`SummarizationTimeoutError` on every trial** → wrong harness (`terminus-2` instead of `swe-agent`).
- **Pinggy bank conflict** → if the same persistent URL is in use by another concurrent eval, vLLM endpoint binds but the agent can't reach it. Use a free pair (verify with `find experiments -name vllm_endpoint.json | xargs grep pinggy`).
- **Bare `chat_template.jinja` left on the repo** → see Post-training § 4 above. Symptom: `tool_calls=None`, model loops `\n\n]\n\n]\n\n]…` to max tokens. Skipping step 4's *deletion* (not just overwrite) of `chat_template.jinja` is the most common cause of "trained model that evaluates 0%".

Model-quality, in escalating order of difficulty to fix:

- **`}}}` triple-brace tool-call (sampling glitch)** — only manifests under `temperature > 0`. Top-token `}\n` has p ≈ 0.997 vs bug-token `}}\n` p ≈ 0.0002 (probe in `notes/ot-agent/sera_braces_diagnosis.md` § 2). Pin greedy via Post-training § 4b to eliminate.
- **Missing `</tool_call>` close-tag** — model jumps from `}}` straight to `<|im_end|>` (p ≈ 0.94). Hermes parser tolerates this so turn-1 looks healthy by accident; under-training symptom that disappears at higher data scale.
- **Long-context degeneracy** (i6, i7) — at turn ≥3 once a single tool observation crosses ≈20 KB, model collapses into a repeating-token attractor (`4.4.4.4…`, `for the for the…`, `\n\n\n…`). Deterministic under greedy. Diagnosis: `notes/ot-agent/sera_braces_diagnosis.md` § 4. Fix path: scale data + epochs (i7 was the first run where this mostly disappeared).
- **Schema confusion in `<tool_call>` envelope (i7-current)** — model emits `{"name": "str_replace_editor", "arguments": {"name": "view", "arguments": {…}}}}` (doubly-nested envelope; 4 closing braces vs 3 opening). vLLM's Hermes parser fails the JSON parse → empty `tool_calls` → SWE-agent `FunctionCallingFormatError` → forfeits after `max_requeries`. Training-data probe (34 793 `<tool_call>` blocks scanned) ruled out a data bug. Diagnosis + escalation paths in `eval/README.md` § "Diagnostic — `FormatError` storm at eval time (2026-04-25)".

---

## Files in this directory

- `README.md` — this doc.
- `configs/template_qwen3_8b_sera_v3.yaml` — legacy i3-era axolotl template (kept for reference only; deprecated).
- `configs/template_qwen3_8b_sera_v4.yaml` — v4-dataset axolotl training config template; current basis for all live runs.
- `sbatch/axolotl_sera_v3.sbatch` — legacy i3-era SLURM template (uses `zero1.json`; deprecated — successors all use `zero3_bf16.json`).
- `sbatch/axolotl_sera_v4.sbatch` — current SLURM template (sed-substitute `__SIZE__` and `__NODES__`).
- `subset_sera_v3.py` — legacy i3-era subsetter (deprecated).
- `subset_sera_v4.py` — local mirror of the i4+ subsetter that does the `transform_traj_hermes` pre-render.
- `convert_axolotl_checkpoint.py` — FSDP-prefix stripper (local copy; canonical lives at `/e/scratch/jureap59/feuer1/code/axolotl/` on Jupiter).
- `convert_sera.py` — earlier shareGPT-format converter from i2; deprecated.
- `zero2_no_offload.json` — custom DeepSpeed config exploring zero2 without CPU offload (superseded by `zero3_bf16.json`, kept for reference).
- `eval/` — eval-time configs and diagnostics. See `eval/README.md` for the SWE-agent scaffold configs (upstream + `max_requeries: 50` patch) and the doubly-nested-name pathology write-up.

The 1 000-row + 12-epoch axolotl configs and sbatches actually used for `-v6` / `-v7` (e.g. `qwen3_8b_sera_v4_1000_v6.yaml`, `qwen3_8b_sera_v4_1000_v7.yaml`, `sera_v4_1000_v6.sbatch`, `sera_v4_1000_v7.sbatch`) live on Jupiter under `/e/scratch/jureap59/feuer1/code/axolotl_configs/` and `axolotl_sbatch/` respectively. They are sed-edited derivatives of the v4 templates in this directory; nothing structural changed besides `num_epochs`, `__SIZE__`, dataset path, and node count.

Subsetter (v4) lives at `/Users/benjaminfeuer/Documents/scripts_dataset_build/subset_sera_v4.py` alongside other data-build scripts; the local copy in this directory mirrors it.

## 04-29-26 — SERA v8 (sera-subset-mixed)

SERA authors (`ethanlshen`) sent a fully preprocessed dataset, sera-subset, with two
JSONL files: stage1 (22972 rows unresolved) + stage2 (25244 rows resolved). Mixed
by concatenation + shuffle (seed=42), then subset to 316 / 1000 / 3160 / 10000.

Hyperparameters per author spec: 3 epochs, lr 1e-5, b1 0.9, b2 0.95, wd 0.01, warmup
48 steps, batch size 32 (global), seq 32768, chat_template chatml, message_field_training train.
8 nodes × 4 GPUs = 32 GPUs → micro=1, grad_accum=1, dp=32 → global batch 32.
zero3_bf16 (no CPU offload — Jupiter GCC 14.3 rejects DeepSpeedCPUAdam's `-march=armv9-a`).
CCE plugin disabled (bf16 grad explosion on aarch64 + torch 2.9 + FA2).

```
# Source data
huggingface-cli download ethanlshen/sera-subset \
  --local-dir /e/data1/datasets/playground/ot-baf/sera_subset/raw --repo-type=model

# Mixed subsets at /e/data1/datasets/playground/ot-baf/sera_subset/subsets/sera_subset_mixed_<SIZE>.jsonl
# Sizes: 316, 1000, 3160, 10000 (seed=42 shuffle)

# Per-size axolotl preprocess (login-node, populates offline cache for compute):
for SIZE in 316 1000 3160 10000; do
  axolotl preprocess /e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_sera_subset_${SIZE}_v8.yaml
done

# Submit (4 jobs, 8 nodes each, no dependency, parallel):
for SIZE in 316 1000 3160 10000; do
  sbatch /e/scratch/jureap59/feuer1/code/axolotl_sbatch/sera_subset_${SIZE}_v8.sbatch
done
```

**Walltime note**: user spec said 47:59:00 (Perlmutter-style) but Jupiter's `part_boos+`
QoS caps at 12h. Sbatches use `--time=11:59:00`; this is comfortable margin since the
v4 node ladder shows 10000 rows × 3 epochs on 8 nodes finishes well under 2h.

Submitted 2026-04-29 ~02:50 UTC:
- 404574 = sera-subset-mixed-316-axolotl__Qwen3-8B-v8
- 404575 = sera-subset-mixed-1000-axolotl__Qwen3-8B-v8
- 404576 = sera-subset-mixed-3160-axolotl__Qwen3-8B-v8
- 404577 = sera-subset-mixed-10000-axolotl__Qwen3-8B-v8

HF target repos:
- `laion/sera-subset-mixed-316-axolotl__Qwen3-8B-v8`
- `laion/sera-subset-mixed-1000-axolotl__Qwen3-8B-v8`
- `laion/sera-subset-mixed-3160-axolotl__Qwen3-8B-v8`
- `laion/sera-subset-mixed-10000-axolotl__Qwen3-8B-v8`

Note: at SIZE=316, total grad steps ≈ 316 × 3 / 32 ≈ 30, less than the warmup floor (48).
Warmup never completes — LR stays in linear-ramp regime for all 30 steps. Per author spec.