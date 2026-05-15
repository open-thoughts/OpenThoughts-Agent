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