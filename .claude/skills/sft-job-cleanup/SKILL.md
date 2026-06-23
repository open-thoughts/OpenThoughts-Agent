---
name: sft-job-cleanup
description: >-
  Publish + clean up a finished LLaMA-Factory SFT job on a no-internet HPC cluster (Jupiter/Leonardo):
  cancel pending retries, drop intermediate checkpoints, HF-upload the model to its configured
  --hub_model_id, register in Supabase via manual_db_push (--training-type SFT default), and free disk.
  Covers the 8B path (root safetensors, direct upload), the 32B/ZeRO-3 path (consolidate shards →
  safetensors first), the Qwen3.5 preprocessor_config copy, the don't-upload-partials policy, and the
  hf-upload gotchas (tmux not nohup, `hf upload` not `-large-folder`, Leonardo sbatch-tunnel not login node).
  Use when an SFT fine-tune finishes and needs uploading + registering, or "run the SFT cleanup checklist".
  Distinct from RL cleanup (rl-job-cleanup) and datagen cleanup (datagen-job-cleanup).
---

# sft-job-cleanup

After an SFT job completes on a no-internet cluster (Jupiter, Leonardo), publish the model + clean up.

## Recognition heuristic — which path? (check the checkpoint root first)
```bash
ls $CHECKPOINTS_DIR/<job_name>/ | grep -E 'safetensors|global_step'
```
- `model-*.safetensors` at root → **8B path** (also **Qwen3.5** — full safetensors at root, no consolidate).
- `global_stepN/` + `zero_to_fp32.py`, no root safetensors → **32B path** (DeepSpeed ZeRO-3 shards — must consolidate first).

## Cross-cutting upload rules (apply to both paths)
- **`hf upload`, NEVER `hf upload-large-folder`** (deprecated stub + deadlocks on HF LFS 429s). Wrap any
  non-trivial upload in **`tmux`**, not `nohup`/`disown`.
- **`--private` is a no-value flag** — omit it (default public per `feedback_hf_public_default`); `--private false` is a CLI parse error.
- **Jupiter**: login node has direct internet → `hf upload` from the login node (in tmux) works.
- **Leonardo**: the login node SIGKILLs long processes at ~100s → use the **sbatch compute-node + SSH-tunnel** upload
  (see `sft-launch-leonardo` §11 / CLAUDE.md "Leonardo HF Upload — Use sbatch, NOT the Login Node").
- **Don't upload partials** (`feedback_no_partial_checkpoint_uploads`): if training didn't reach 100%, relaunch and let
  chain restarts auto-resume from the latest checkpoint. Salvage-upload a partial ONLY when explicitly OK'd
  (then name it `laion/<job_name>-<step>-<size>`).
- **Tokenizer sanity check (pre-upload, MANDATORY):** verify `tokenizer_config.json`'s `extra_special_tokens` is a **dict**, not a list — `python -c "import json;d=json.load(open('<ckpt>/tokenizer_config.json'));assert isinstance(d.get('extra_special_tokens',{}),dict), 'LIST — coerce to {}'"`. If it's a list, set it to `{}` and re-save before upload. (**ROOT CAUSE, confirmed 2026-06-14:** SFT-save envs run **transformers 5.x** (otagent 5.8.1 / sft-qwen35 5.3 / sera-axolotl 5.5), whose tokenizer save folds `additional_special_tokens` into `extra_special_tokens` as a **list**; the SkyRL RL-load env runs **transformers 4.57.6**, which calls `.keys()` on it → `'list' object has no attribute 'keys'` in `get_tokenizer`, killing every SFT→RL handoff. The older 4.x-saved models have `{}` and load fine. Coercing to `{}` makes the model loadable by both. See `agent_logs/2026-06-14_swesmith_coldstart_rl_list_keys_crash.md`.)

---

## 8B SFT Job Cleanup Checklist

**0. Cancel pending retries** (so stale restarts don't fire mid-upload):
```bash
squeue -u $USER --format='%i %j %T' | grep <job_name> | grep PENDING | awk '{print $1}' | xargs -r scancel
```

**1. Remove intermediate checkpoints** (don't upload cruft):
```bash
rm -rf $CHECKPOINTS_DIR/<job_name>/checkpoint-*  $CHECKPOINTS_DIR/<job_name>/.cache
```

**1b. Qwen3.5 only — copy `preprocessor_config.json` from the base model** (LLaMA-Factory doesn't emit it;
vLLM needs it or the model fails to load / produces garbage):
```bash
cp /path/to/Qwen3.5-9B/preprocessor_config.json  $CHECKPOINTS_DIR/<job_name>/   # or the -27B base
```

**2. Upload model weights to HuggingFace.** **Naming:** full final upload (training reached 100%) → the
configured `--hub_model_id` from the launch command (`laion/<descriptive_name>`, NO step/size suffix — do
NOT use the job name verbatim). (Partial salvage, uncommon/only-if-OK'd → `laion/<job_name>-<step>-<size>`.)
```bash
# Jupiter login node (direct internet). On LEONARDO use the §11 sbatch-tunnel — login-node hf upload dies at ~100s.
source ~/secrets.env
tmux new-session -d -s hf_upload_<short> \
    "source ~/secrets.env && hf upload <hub_model_id> $CHECKPOINTS_DIR/<job_name> . \
        --repo-type=model 2>&1 | tee $CHECKPOINTS_DIR/<job_name>/upload.log"
# tmux attach -t hf_upload_<short>  (Ctrl-b d to detach)
```
Wait for it to finish and verify the repo exists on HF Hub.

**3. Register in the unified DB** (W&B optional — Jupiter has no W&B; SFT is the DEFAULT `--training-type`, no flag needed):
```bash
python scripts/database/manual_db_push.py \
  --hf-model-id <hub_model_id> --base-model <base_model_hf> \
  --dataset-name <dataset_name>          # comma-separated for multi-dataset → sets dataset_names
```
**SKIP for HF-only series** (e.g. Delphi #6279 — YAMLs set `enable_db_registration: false`; do not register, and
do not pass an anchor as `--base-model` since that auto-creates a base-model row).

**4. Clean up the experiments dir** — only after 1–3 succeed:
```bash
rm -rf $EXPERIMENTS_DIR/<job_name>
```

---

## 32B SFT Job Cleanup Checklist (DeepSpeed ZeRO-3 — consolidate first)

For 32B SFT (any ZeRO-3 run without `stage3_gather_16bit_weights_on_model_save: true`), the trainer writes
sharded ZeRO-3 state into `checkpoint-N/global_stepN/` instead of consolidated safetensors at root —
consolidate before uploading. Steps mirror 8B except consolidate + the manual upload location.

**0. Cancel pending retries** (same as 8B).

**1. Verify training reached 100%** — `trainer_log.jsonl` shows `current_steps == total_steps`. Per
`feedback_no_partial_checkpoint_uploads`, default policy is don't salvage partials (relaunch + resume); only proceed if explicitly OK'd as a partial.

**2. Consolidate** ZeRO-3 shards → fp32 state_dict → safetensors:
```bash
python -m hpc.launch --job_type consolidate \
  --consolidate_input $CHECKPOINTS_DIR/<job_name> \
  --consolidate_output_repo <hub_model_id> \
  --consolidate_workdir <writable_workdir>/<job_name> \
  --time_limit 02:00:00 --num_nodes 1
```
Produces `<workdir>/<job_name>/final_repo/` (`model-NNNN-of-MMMM.safetensors` + tokenizer + config at root).
The consolidate job also attempts an HF push at the end — **do NOT rely on it** (historical `BrokenPipeError`
at `api.create_commit` on big 32B uploads). Treat consolidate as done once `final_repo/` is fully written;
upload manually (step 3).

**3. Manually upload from `final_repo/`** (NOT the original checkpoint dir — it still holds ZeRO-3 shards).
Naming same as 8B (full → `--consolidate_output_repo`/`--hub_model_id`, no suffix):
```bash
# Jupiter login node. On LEONARDO use the §11 sbatch-tunnel (131GB → ~4 min). tmux; hf upload (not -large-folder).
source ~/secrets.env
tmux new-session -d -s hf_upload_<short> \
    "source ~/secrets.env && hf upload <hub_model_id> <consolidate_workdir>/<job_name>/final_repo . \
        --repo-type=model 2>&1 | tee <consolidate_workdir>/<job_name>/upload.log"
```

**4. Register in the unified DB** (same as 8B step 3; SFT is the default; skip for HF-only series).

**5. Clean up** — only after 2–4 succeed, remove BOTH the sharded checkpoint dir AND the consolidate workdir
(32B sharded ckpt ~700GB + workdir ~200GB):
```bash
rm -rf $CHECKPOINTS_DIR/<job_name>  <consolidate_workdir>/<job_name>
```

> Launch-side details (preamble, configs, sbatch patching, the no-internet pre-download) live in the
> **`sft-launch-jupiter`** / **`sft-launch-leonardo`** skills; this skill is the post-run publish + cleanup.

---

## Operating notes (folded from memory 2026-06-14)

- **Run the FULL cleanup checklist AUTONOMOUSLY on completion** (COMPLETED + 100%): cancel pending chain → drop checkpoints → HF upload → DB register → clean exp dir. Do NOT ask "want me to do (a)/(b)/(c)?" — just do it (user: "don't block that process on me"; "complete the entire checklist, now and in the future"). No eval-gate — DB-register is part of the mechanical follow-through. **Exception:** brief-flag (don't block) only on obvious anomalies (NaN loss, huge step gap). Cancel decisions on RUNNING jobs stay user-driven.
- **Multi-dataset DB registration:** pass the full comma-separated list to `--dataset-name` so `dataset_names` is populated (not just one `dataset_id`). Known limitation: the script stores it as a single string and does NOT trigger the `multiple_datasets` path (`dataset_id` ends up null) — verify the right field after registering. Single-dataset `--dataset-name` works fine and populates `dataset_id`.
- **Baseline model versioning (Sera/CoderForge):** flat monotonic `-v5`/`-v6`/`-v7` in HF repo names + README iteration tables, NOT nested `v4-v2`/`v4-v3`. In-flight runs keep their existing names; the NEXT retrain uses the new scheme (next Sera = v5, skipping v4 to avoid colliding with existing v4 artifacts).
