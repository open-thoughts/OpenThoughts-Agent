# CoderForge v3 Baseline — Jupiter Reproduction

Faithful re-creation of the `togethercomputer/CoderForge-Preview` SFT training pipeline for Qwen3-8B on the JSC Jupiter cluster (GH200, aarch64, CUDA 13).

**Sibling to** `/baselines/sera` — same axolotl env, same cluster config, different data source.

## Iteration history

| Version | Dataset | Config notes | Outcome |
|---|---|---|---|
| **v2** (deprecated) | `convert_coderforge.py` → shareGPT via custom flattening, trained with LLaMA-Factory | Tool calls preserved as inline `<tool_call>…</tool_call>` in `content`. Extra conversion step introduced template drift | Deprecated in favor of v3's pre-tokenized path |
| **v3-316** (2026-04-22, `laion/CoderForge-Preview-v3-316-axolotl__Qwen3-8B`) | `laion/CoderForge-Preview-v3-316` (pre-tokenized, 316 rows) | axolotl `chat_template: chatml` (default for template) | SWE-bench eval: 0/? (0%). Model emits pure token garbage — pages of `"888888..."` or `"with with with..."` word salad |
| **v3-1000** (2026-04-23, `laion/CoderForge-Preview-v3-1000-axolotl__Qwen3-8B`) | `laion/CoderForge-Preview-v3-1000` (pre-tokenized, 1000 rows) | same | Same failure mode: 0/16 completions, all AgentTimeoutError, every model completion was degenerate repetition. Training log itself was clean (loss 0.81→0.27, exit 0 in 47min) — bug was inference-time, not training-time |
| **v3-1000-fixed** (2026-04-24, same HF repo, **current**) | same training data — **no retrain needed** | Deleted stale `chat_template.jinja` from the HF repo. Axolotl saved a 292-byte bare chatml template alongside the stock-restored `tokenizer_config.json`. HF/vLLM prioritize the standalone `.jinja` over `tokenizer_config["chat_template"]`, so inference rendered prompts with the bare template (drops `tool_calls`, no `<tool_response>` wrapping) → model saw OOD input → emitted garbage. Same root cause as Sera v4 "additional gotcha" (2026-04-23). Patched `configs/template_qwen3_8b_cf_v3.yaml`: `chat_template: chatml → tokenizer_default` to prevent future retrains from re-introducing the bare jinja | Eval validation in progress |

### The `chat_template.jinja` bug (2026-04-24 diagnosis)

Same bug as Sera v4 but caught later because CF's pre-tokenized training data means the issue is **purely inference-time**: the weights trained correctly (labels masked right, loss curve clean, tokenizer IDs stock Qwen3-8B), but the wrong chat template at serve time corrupted every prompt.

- **Stock `Qwen/Qwen3-8B`** has NO `chat_template.jinja` file. Its 4168-char template lives inside `tokenizer_config.json["chat_template"]`.
- **Axolotl output** has a tiny 292-byte `chat_template.jinja` (naive `<|im_start|>role\ncontent<|im_end|>`). Added by axolotl even when the base model doesn't use one.
- **HF Transformers + vLLM** load `chat_template.jinja` in preference to `tokenizer_config["chat_template"]` when both exist. The restored tokenizer_config was effectively ignored.

**Fix** (applied to `laion/CoderForge-Preview-v3-1000-axolotl__Qwen3-8B` and `laion/CoderForge-Preview-v3-316-axolotl__Qwen3-8B` on 2026-04-24): delete `chat_template.jinja` from the HF repo. `feedback_axolotl_restore_tokenizer.md` was updated to mandate this step.

## Why v3?

Our earlier **v2** converter (`convert_coderforge.py`, producing `laion/CoderForge-Preview-v2-*`) ran the raw CF trajectories through a shareGPT flattening pipeline and trained on them with LLaMA-Factory. The converter DID preserve tool calls (serialized as inline `<tool_call>…</tool_call>` in `content`) — unlike our Sera v2 which silently dropped them. But the extra conversion step still introduces drift:
- We re-render tool calls into hermes-style text
- The chat-template application at training time isn't guaranteed to match the upstream tokenizer exactly

**v3 skips all of that.** CoderForge already publishes a pre-tokenized subset (`trajectories-tokenized_qwencoder`) with `input_ids` + `labels` + `chat_template_applied` in native format. Our v3 row-subsets those and trains on the tokens directly via axolotl, which auto-detects pre-tokenized datasets.

## Dataset

Row-subsets of `togethercomputer/CoderForge-Preview/trajectories-tokenized_qwencoder`, sampled **only from the `filtered_reward1` slug** (155,144 rows).

### Why only `filtered_reward1`?

The [Together AI CoderForge-Preview blog](https://www.together.ai/blog/coderforge-preview) states the authors trained **only on successful trajectories**: filtered to those whose final patches pass all repository tests, with SWE-Bench overlap and non-permissive-license trajectories removed. That filtered set is exactly `filtered_reward1` — **155,144 trajectories** across R2E-Gym + SWE-Smith + SWE-Rebench.

The other 3 slugs in the HF dataset (`R2E_Gym`=32,964, `SWE_Rebench`=77,169, `SWE_Smith`=148,001) are the **raw pre-filter trajectory pool**, not what the upstream SFT was run on. Training on the union would confound our v3 reproduction with unfiltered/failed/licensed-out trajectories.

| Dataset | Rows |
|---|---|
| `laion/CoderForge-Preview-v3` | 155,144 (full filtered_reward1) |
| `laion/CoderForge-Preview-v3-316` | 316 |
| `laion/CoderForge-Preview-v3-1000` | 1,000 |
| `laion/CoderForge-Preview-v3-3160` | 3,160 |
| `laion/CoderForge-Preview-v3-10000` | 10,000 |
| `laion/CoderForge-Preview-v3-31600` | 31,600 |
| `laion/CoderForge-Preview-v3-100000` | 100,000 |

Sampling: deterministic random, seed=42, indexed into `filtered_reward1` (all 224 shards, read in order). Row subsets are nested.

Each row carries:
- `input_ids: list[int32]` — pre-tokenized via the Qwen2.5-Coder / Qwen3 tokenizer (confirmed identical to `Qwen/Qwen3-8B`).
- `attention_mask: list[int8]` — all 1s, **added by our subsetter** so axolotl's pre-tokenized auto-detection (`_is_dataset_already_tokenized` checks `input_ids + attention_mask + labels`) triggers. Upstream only had `input_ids + labels`.
- `labels: list[int64]` — with `-100` masks already applied upstream (assistant-only loss).
- `chat_template_applied: str` — decoded render, useful for debugging.
- `trajectory_id: str`, `reward: float64` (upstream fields preserved).
- `source: str` — always `togethercomputer/CoderForge-Preview/trajectories-tokenized_qwencoder`.

Long sequences: upstream trajectories range up to ~80k tokens. Our training config uses `sequence_len: 32768`, matching Sera v3 — axolotl truncates at collate.

## Environment

Shared with `baselines/sera` — the `sera-axolotl` conda env on Jupiter (see `baselines/sera/README.md` for install recipe). No additional deps needed for CoderForge v3.

## Axolotl config

- `configs/template_qwen3_8b_cf_v3.yaml` — axolotl config template (sed-substitute `__SIZE__`).

Key line:
```yaml
datasets:
  - path: laion/CoderForge-Preview-v3-__SIZE__
    ds_type: parquet
    # no `type:` — axolotl's _is_dataset_already_tokenized() catches
    # (input_ids + attention_mask + labels) and skips chat_template rendering.
```

All other hparams match Sera v3 (chatml template, sequence_len=32768, lr=1e-5, grad_accum=8, 3 epochs, cosine schedule, warmup_ratio=0.1875, CutCrossEntropyPlugin) so Sera-vs-CoderForge ablations are apples-to-apples.

## Training launch

- `sbatch/axolotl_cf_v3.sbatch` — SLURM template. Sed-substitute `__SIZE__` and `__NODES__`.

Node sizing (same ladder as Sera v3):

| Subset | Nodes |
|---|---|
| 316 | 1 |
| 1000 | 1 |
| 3160 | 2 |
| 10000 | 4 |
| 31600 | 8 |
| 100000 | 16 |
| full | 32 |

**Rollout order (per user direction):** launch SFT for 316 + 1000 first. Hold 3160 and larger until initial eval results come back on the small runs.

Render + submit:
```bash
SIZE=316
NODES=1
cd /e/scratch/jureap59/feuer1/code/axolotl_sbatch
sed -e "s/__SIZE__/$SIZE/g" -e "s/__NODES__/$NODES/g" template_cf.sbatch > cf_v3_${SIZE}.sbatch
sbatch cf_v3_${SIZE}.sbatch
```

Pre-download on login node first (compute nodes are offline):
```bash
conda activate sera-axolotl
source /e/scratch/jureap59/feuer1/OpenThoughts-Agent/hpc/dotenv/jupiter.env
source ~/secrets.env
export CUDA_HOME=/e/software/default/stages/2026/software/CUDA/13
export PATH=$CUDA_HOME/bin:$PATH
cd /e/scratch/jureap59/feuer1/code/axolotl

# Renders a per-size axolotl config if missing:
sed "s/__SIZE__/316/g" /e/scratch/jureap59/feuer1/code/axolotl_configs/template_qwen3_8b_cf_v3.yaml \
  > /e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_cf_v3_316.yaml

axolotl preprocess /e/scratch/jureap59/feuer1/code/axolotl_configs/qwen3_8b_cf_v3_316.yaml
```

## Post-training (axolotl → vLLM)

See `/Users/benjaminfeuer/Documents/OpenThoughts-Agent/CLAUDE.md` for the canonical axolotl-checkpoint-to-vLLM procedure. Short version:

1. Axolotl's FSDP wrapper leaves `_checkpoint_wrapped_module.` prefixes in state_dict keys. vLLM / sglang can't load these.
2. Use upstream's `convert_axolotl_checkpoint.py` (lives at `/e/scratch/jureap59/feuer1/code/axolotl/convert_axolotl_checkpoint.py`) to strip them.
3. **Restore stock Qwen3-8B tokenizer files** (overwrite `tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`) AND **delete `chat_template.jinja`** before upload. See `feedback_axolotl_restore_tokenizer.md`.
4. **Patch the auto-generated `README.md`** before upload — axolotl writes the local jsonl file path into `datasets:` frontmatter, which HF rejects as `Invalid metadata`. Replace with the real HF dataset id (e.g. `datasets:\n- laion/CoderForge-Preview-v6-1000`).
5. `huggingface-cli upload-large-folder` the cleaned checkpoint.
6. **HOLD `manual_db_push.py` until the eval shows non-zero reward** (see `feedback_db_register_after_eval.md`). The DB is a permanent registry; don't seed it with broken-at-eval-time models.

---

## Evaluation

CoderForge v6+ data was generated to match the **OpenHands XML tool-call format**:
`<think>...</think>...<function=NAME><parameter=K>V</parameter></function>`. Pair it with the matching harbor harness (`openhands_ctx32k_eval_.yaml`) and `--trace_agent_name openhands` — DO NOT use the SWE-agent harness.

Mismatched harness is a leading 0% failure mode: the SWE-agent harness expects JSON-in-`<tool_call>` blocks, sees `<function=...>` instead, parses nothing, and every trial forfeits.

### Why CF needs the OpenHands harness specifically

- v3 (pre-tokenized) and v5 (wrapper-stripped, no `<think>`) both produced **garbage output** at eval time. Stock Qwen3-8B assigns ~100% prior to `<think>` as the first token after `<|im_start|>assistant`; CF's pre-tokenized v3 data trained the model to skip `<think>` → catastrophic forgetting at inference.
- v6 (subset_coderforge_v6.py) regenerates training data with `<think>...</think>` injected and tool calls rendered as native OpenHands XML, so the model emits OpenHands format AND keeps Qwen3's `<think>` prior intact.
- The OpenHands harness's `disable_tool_calls: true` setting tells the agent framework to expect XML-parsed function calls, not Hermes-style JSON-wrapped ones. Wrong harness → no parsed actions → AgentTimeoutError on every trial.

### Pinggy and Daytona

See `baselines/sera/README.md` for the shared notes. TL;DR:
- Use `$DAYTONA_BASE_API_KEY` (NOT `DAYTONA_API_KEY` — that's the RL-org key, blocks declarative builds).
- Pinggy URL+token pairs in `/Users/benjaminfeuer/Documents/notes/ot-agent/pinggy_bank.md`. Default to free pairs from 8/9/10.

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
  --model_path laion/<cf_v6plus_hub_id> \
  --tasks_input_path DCAgent2/swebench-verified-random-100-folders \
  --trace_harbor_config hpc/harbor_yaml/eval/openhands_ctx32k_eval_.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_8b_vllm_serve_32k_4xA100.yaml \
  --trace_agent_name openhands \
  --daytona_api_key "$DAYTONA_BASE_API_KEY" \
  --pinggy_persistent_url <pair-N URL> --pinggy_token <pair-N token> \
  --time_limit 11:59:00 \
  --num_nodes 1 --gpus_per_node 4 \
  --trace_n_concurrent 16
```

### Launch — Jupiter (GH200, no internet on compute)

Replace `qwen3_8b_vllm_serve_32k_4xA100.yaml` → `qwen3_8b_vllm_serve_32k_4xGH200.yaml`. Otherwise identical.

### Health check + aggregating results

Same protocol as Sera (see `baselines/sera/README.md` → "Health check" and "Aggregating results"). 15-min infra cadence per `feedback_eval_15min_infra_check.md` while any eval is RUNNING.

### Failure modes seen so far

- **Wrong harness (SWE-agent instead of OpenHands)** → 0% pass; CF model emits `<function=NAME>` XML but harness parses nothing.
- **`chat_template.jinja` left in the HF repo** → vLLM uses axolotl's bare 292-byte template instead of stock Qwen3-8B's 4168-byte one → drops `tool_calls` + `role:tool` at serve time → corrupted prompts → garbage output. Always delete `chat_template.jinja` post-upload (see step 3 above).
- **Long-context degeneracy** — even with the right harness + correct data, CF v7 (1000 × 6 epochs) produced clean short-context output but collapsed at the 8.5k+ token eval prompts. CF v8 (1000 × 12 epochs) is the parallel test of the under-training hypothesis (matches Sera v7).

---

## Files in this directory

- `README.md` — this doc.
- `subset_coderforge_v3.py` — builds row-subsets of `trajectories-tokenized_qwencoder`, injects `attention_mask` + `source` columns, uploads as `laion/CoderForge-Preview-v3(-<SIZE>)`.
- `configs/template_qwen3_8b_cf_v3.yaml` — axolotl training config template.
- `sbatch/axolotl_cf_v3.sbatch` — SLURM sbatch template (shares semantics with Sera's).
