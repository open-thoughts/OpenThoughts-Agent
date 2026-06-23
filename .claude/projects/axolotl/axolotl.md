# Axolotl — dependency overview

The SFT trainer used **only** for the Sera + CoderForge baseline reproductions. Everything else uses
LLaMA-Factory. Axolotl is deployed when the upstream paper's wire format requires it (Sera's OpenAI-native
`messages` with `tool_calls`+`train` fields; CoderForge's pre-tokenized `input_ids`/`labels`). Written
2026-06-14 from `notes/axolotl.md` + `baselines/`.

- **Version:** axolotl **v0.16.1**, in the **`sera-axolotl`** conda env (torch 2.9.1+cu130, Jupiter GH200 aarch64). Canonical install `/e/scratch/jureap59/feuer1/code/axolotl/`.
- **Configs:** `baselines/sera/configs/template_qwen3_8b_sera_v4.yaml`, `baselines/coderforge/configs/template_qwen3_8b_cf_v3.yaml` (templated with `__SIZE__`/`__NODES__` via `sed` before submit).
- **Launch:** SLURM sbatch (`baselines/{sera,coderforge}/sbatch/axolotl_*.sbatch`), multi-node `srun + accelerate launch`, default **4 nodes** (16 GH200) + `zero3_bf16.json`, `booster`/`--account reformo`. **Not** wired into `hpc.launch` — it's its own sbatch path. Compute has no internet → run `axolotl preprocess` on the login node per size first to populate the offline HF cache.

---

## Load-bearing gotchas

1. **Post-train tokenizer/chat-template restoration (the 0%-SWE-bench bug).** Axolotl saves a stripped `tokenizer_config.json` + a bare 4-line `chat_template.jinja` that **don't handle `tool_calls` / `role: tool`** at serve time → vLLM silently drops every `tool_calls` field → the model is OOD at inference (training was healthy) → **0% on SWE-bench.** **Fix:** after training, overwrite the four tokenizer files (`tokenizer_config.json`, `tokenizer.json`, `vocab.json`, `merges.txt`) with the **stock base-model** versions AND delete `chat_template.jinja` from the HF repo (Ai2's released SERA-8B is byte-identical to stock Qwen3-8B tokenizer — mirror that). Script in `baselines/sera/README.md` §Post-training. (This is why the `feedback_axolotl_restore_tokenizer` memory existed.)
2. **DeepSpeed = custom `zero3_bf16.json`, no CPU offload.** `zero1.json` OOMs at 32k on 96GB once CCE is off; default zero2/zero3 offload Adam to CPU → `DeepSpeedCPUAdam` JIT → GCC 14.3 rejects the armv9 flags. ZeRO-3-bf16 keeps Adam on GPU and shards params/grads/moments.
3. **`CutCrossEntropyPlugin` disabled** (commented in YAML): on aarch64+torch2.9+FA2 it causes bf16 grad explosion (`grad_norm ~1e11` in 3–7 steps) → NaN loss masked as 0. Also set `max_grad_norm: 1.0` explicitly. (The CCE fork is still installed because import paths need it.)
4. **Two mandatory in-place env patches:** (a) `src/axolotl/utils/callbacks/qat.py` — wrap the torchao import in `try/except ImportError` (axolotl's setup filters torchao on aarch64 but qat.py imports it unconditionally → `builders` import fails); (b) `convert_axolotl_checkpoint.py` — strips `_checkpoint_wrapped_module.` FSDP prefixes from state_dict keys so vLLM/sglang can load (required post-train step; canonical copy on Jupiter).
5. **Omit `hub_model_id`/`hub_strategy`** from the config: under `HF_HUB_OFFLINE=1` the in-training `init_hf_repo()` crashes `OfflineModeIsEnabled` — push manually after.
6. **sbatch needs the compiler env** (`CUDA_HOME`, `GCC_HOME`, `CC`/`CXX` for Triton JIT), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, offline flags, `NCCL_SOCKET_IFNAME=ib0`.

---

## Versioning

Flat monotonic suffix on HF repos: `-v2`, `-v3`, `-v6`, `-v7`, `-v8` (v4/v5 intentionally skipped to keep
the dataset-recipe version distinct from the run version). The dataset segment (e.g. `Sera-4.6-Lite-T2-v4-316`)
is separate from the run version (`-v3`). In-flight runs keep their names; the NEXT retrain takes the next
flat number. (See `.claude/skills/sft-job-cleanup` operating notes; matches `feedback_baseline_model_versioning`.)

Install recipe (env build + the two patches + the prebuilt aarch64 flash-attn wheel + the CCE fork) is in
`notes/axolotl.md`.
