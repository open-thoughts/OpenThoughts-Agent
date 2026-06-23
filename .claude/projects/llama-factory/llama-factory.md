# LLaMA-Factory — dependency overview

The SFT training backend behind `hpc/launch.py --job_type sft`. Written 2026-06-14 from notes + the
submodule. Launch/cleanup how-tos are in the `sft-launch-*` / `sft-job-cleanup` skills; runtime/env choice
(8B vs 32B-ZeRO3 vs Qwen3.5) is in `.claude/ops/jupiter/ENVIRONMENT_MAP.md`. This is the framework overview.

- **Submodule:** `sft/llamafactory` (detached HEAD); init via `git submodule update --init --remote sft/llamafactory`. Driven by `hpc/sft_launch_utils.py` → `llamafactory-cli train <config.yaml>`.
- **OT-Agent configs:** `sft/lf_configs/` — families `qwen2_5/`, `qwen3/` (dense 32B/1.7B), `qwen3_5/` (9B/27B, needs transformers ≥5.3), `openai/`, `delphi/`, `deepspeed/` templates.

---

## Config model (the lf_configs YAML knobs)

- **Model/attn:** `model_name_or_path`, `attn ∈ {fa2, fa3, sdpa, eager, hf:org/repo}` (fallback chain eager→sdpa→fa2→fa3→kernel), `enable_liger_kernel`, `optim: adamw_torch_fused`.
- **Method:** `stage: sft`, `finetuning_type: full`, `deepspeed: <ds_z3_config.json>`.
- **Data:** `template` (qwen3 / qwen3_5), `cutoff_len`, `packing`, `neat_packing`, `model_capacity` (decouples model `max_position_embeddings` from `cutoff_len` so a large cutoff doesn't auto-RoPE-scale), `rope_scaling: yarn`.
- **ShareGPT tags (load-bearing):** Harbor/DCAgent data uses `role`/`content` with `user`/`assistant`; LLaMA-Factory defaults to `from`/`value` with `human`/`gpt`. **Always pass** `role_tag: role`, `user_tag: user`, `assistant_tag: assistant`, `content_tag: content` or the thinking preprocessor finds 0 assistant messages → garbage.
- **Hparams:** `learning_rate`, `num_train_epochs`, `lr_scheduler_type: cosine`, `warmup_ratio`, `bf16`, `gradient_checkpointing`.

---

## Kernels / performance work (status)

- **Liger kernel** — IN PRODUCTION (`enable_liger_kernel: true`): fuses CE+linear, SwiGLU/GEGLU MLP, RMSNorm/LayerNorm for llama/mistral/qwen2/qwen3/qwen3_moe/gemma/glm4/… **Disable for Qwen3.5** (hybrid GatedDeltaNet+Attention not supported — `qwen3_5/*.yaml` sets `enable_liger_kernel: false`).
- **Cut-Cross-Entropy (CCE)** — aspirational (plan drafted in `cut-cross-entropy-lf.md`, needs `ml-cross-entropy`); not yet wired. (Axolotl uses a CCE fork but disables the plugin — see that doc.)
- **Neat packing** — in-codebase but **gated on transformers ≤ 4.52.4**: it monkey-patches `_get_unpad_data` with integer attention masks; transformers 4.53.0+ casts 2D masks to bool → cross-sequence contamination / illegal memory access. Use standard `packing: true` + `neat_packing: false` (allows cross-seq attention) unless you pin old transformers.
- **torch.compile** — NOT default; per `torch_compile.md` it helps stable-shape compute-bound DDP, but **hurts** FSDP + variable-length long-context (graph breaks, regresses step time) and adds little over FA2. Avoid for OT-Agent long-context SFT.
- **TransformerEngine FP8** (`fp8_backend: te`) — experimental, integration pending.

---

## DeepSpeed / consolidation

- **ZeRO-3 (+ optional CPU offload)** is required for 32B+ dense (`deepspeed: examples/deepspeed/ds_z3*.json`); it writes **sharded** checkpoints. **ZeRO-3 is incompatible with MoE** (assertion error) → use ZeRO-2 for MoE.
- **8B vs 32B output distinction** (drives cleanup): 8B writes **full safetensors at the checkpoint root** → no consolidate, upload directly; 32B/ZeRO-3 writes `global_stepN/` shards → must **consolidate to safetensors first** (`hpc.launch --job_type consolidate`). Qwen3.5 also writes root safetensors → skip consolidate (8B path) + copy `preprocessor_config.json` from the base before upload. (See `sft-job-cleanup`.)
- DeepSpeed `pin_memory` CUDA-error crashes are patched in `extras/deepspeed_utils.py` (falls back to CPU-contiguous).

---

## MoE training (the gap)

LLaMA-Factory supports 20+ MoE models but has **no expert parallelism / Megablocks / Tutel** → ~2–3× slower
than dense (random expert access kills bandwidth), and ZeRO-3 is unavailable for MoE. `moe_training_frameworks_comparison.md`
ranks alternatives (DeepSpeed-MoE `ep_size` 2–3×, Megablocks grouped-GEMM 3–5× but a broken sparse path,
Tutel 2–8×, Megatron `mcore_adapter` full-FT-only ~33% faster). **For Qwen3-MoE (30B-A3B) SFT, plain LF +
ZeRO-2 is slow** — this is why MoE RL uses the SkyRL+Megatron/SIF path, not LF.

---

## Gotchas

- **Qwen3.5 needs transformers ≥ 5.3** (hybrid arch absent from 4.x) → the dedicated `sft-qwen35` conda env (`.claude/ops/jupiter/ENVIRONMENT_MAP.md` §2f); Liger off.
- **`attn: eager` OOMs at 32k** on an 80/96GB GPU (Qwen3-32B) → use `fa2` (or `sdpa` if FA2 missing; on Jupiter otagent now has FA2 installed). A config shipping `attn: fa2` falls back to eager if the env lacks flash_attn — verify.
- **ShareGPT role/content tags** (above) — silent garbage if omitted.
- **Post-train chat template:** confirm the served template handles tool_calls / role:tool (this is the bug that bit the axolotl baselines — see that doc; LF's `template: qwen3` handles it during training, double-check after consolidation).
- **neat_packing × transformers version** (above).

Key code: `src/llamafactory/{train/sft/trainer.py, model/model_utils/{attention,liger_kernel,packing}.py, hparams/{model_args,data_args}.py, extras/deepspeed_utils.py}`.
