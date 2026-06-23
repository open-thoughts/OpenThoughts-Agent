---
name: rl-job-cleanup
description: >-
  Preserve + publish a finished RL (SkyRL/GRPO) training checkpoint after the job terminates
  (completed at max_steps OR early-stopped/scancelled) on an HPC cluster (Jupiter/Leonardo/Perlmutter).
  Covers: cancel pending retries, pick the BEST checkpoint by trailing-5 EMA of reward across the full
  restart chain, flatten weights to repo root, secret-scan, `hf upload` to laion/<job>-<step>-<size>,
  Supabase DB register (--training-type RL + cross-user FK safety pre-check), upload training traces to
  penfever/<job>, parse metrics, and clean up. Use when an RL run needs its model uploaded + registered,
  or when asked to "run the RL cleanup checklist". Distinct from SFT cleanup (that's a different flow).
---

# rl-job-cleanup

After an RL job terminates (early or completed), follow these steps to preserve and publish the
checkpoint. The final HF artifact is `laion/<job_name>-<step>-<size>` (weights at repo root) +
a companion trace dataset `penfever/<job_name>`, with a Supabase `models` row (`training_type=RL`).

**Cross-cutting rules (from CLAUDE.md / memory — they bite):**
- **`hf upload`, NEVER `hf upload-large-folder`** (the latter is a deprecated stub + deadlocks on HF LFS
  429s). Wrap long uploads in **`tmux`**, not `nohup`. (See CLAUDE.md "HF Uploads + Long-Running Login-Node Commands".)
- **`--private` is a no-value flag** — do NOT pass `--private false` (CLI parse error). Default policy is
  PUBLIC; just omit it (`feedback_hf_public_default`).
- Run trace upload + `parse_skyrl_metrics.py` from the **`otagent` conda env** (they need
  `google.cloud.storage` + matplotlib; the RL venv lacks them).
- On **Leonardo**, login-node `hf upload` is SIGKILLed at ~100s — use the sbatch+tunnel upload pattern
  (CLAUDE.md "Leonardo HF Upload — Use sbatch, NOT the Login Node").

## 0. Cancel pending retries
Before anything else, cancel queued retry jobs for the same run so they don't start mid-upload:
```bash
squeue -u $USER --format='%.18i %.80j %.8T' | grep <job_name>
scancel <retry_job_ids>
```

## 1. Locate the best checkpoint — trailing-5 EMA of reward (NOT single-step max)
```bash
# NOTE: there is an empty exports/ at the base level — ignore it. Real HF-exportable ckpts are nested:
ls -lt $EXPERIMENTS_DIR/<job_name>/<job_name>/exports/ | head -10
```
Use the **EMA of `reward/avg_raw_reward` over a trailing-5 window** — single-step max overfits one lucky step; EMA picks the most sustained-good region.

Rules:
- **EMA across ALL steps in chronological order, regardless of chain restarts.** Collect `step` lines
  from EVERY `.out` (`.out` is canonical per `feedback_sft_status_via_out_not_jsonl`), sort by
  `trainer/global_step`. Do NOT compute per-chain-link EMA (chain boundaries aren't training-meaningful).
- Standard 5-period EMA: `α = 2/(5+1) = 1/3`; `EMA_n = α·reward_n + (1−α)·EMA_{n−1}`, `EMA_1 = reward_1`.
- **Never select the first saved checkpoint** (`global_step_5` with `hf_save_interval: 5`) — EMA not
  warmed up. Start from the second-saved step (typically 10).
- Among saved-and-aligned ckpts (multiples of `hf_save_interval`, excluding the first), upload the one
  with the highest EMA. (If the job was scancelled before a save-aligned max-step, the latest saved ckpt
  is the largest multiple of `hf_save_interval` reached — cap selection there.)

```python
import json, glob, re
rewards = {}  # step -> avg_raw_reward
for fn in glob.glob(f"{EXP_DIR}/logs/*.out"):
    for line in open(fn):
        m = re.search(r'trainer/global_step":\s*(\d+).*avg_raw_reward":\s*([\d.eE+-]+)', line)
        if m:
            step, r = int(m.group(1)), float(m.group(2))
            rewards.setdefault(step, r)  # first-seen wins (chain links may overlap)
steps = sorted(rewards)
alpha = 1/3
ema = {}; prev = rewards[steps[0]]
for s in steps:
    prev = alpha * rewards[s] + (1 - alpha) * prev
    ema[s] = prev
SAVE_EVERY = 5  # match hf_save_interval
aligned_eligible = [s for s in steps if s % SAVE_EVERY == 0 and s >= 2 * SAVE_EVERY]
best = max(aligned_eligible, key=ema.get)
print(f"best EMA={ema[best]:.4f} at step={best} (reward at that step={rewards[best]:.4f})")
```
Upload the checkpoint at `exports/global_step_<best>/`.

## 2. Locate the W&B run (optional)
From the job logs / `trainer_log.jsonl`: `https://wandb.ai/dogml/OpenThoughts-Agent/runs/<run_id>`. (Jupiter has no W&B — fine to omit.)

## 3. Flatten model files to the upload-dir ROOT
HF model files MUST be at the base of the uploaded dir — not nested in `policy/`:
```bash
UPLOAD_DIR=/e/scratch/jureap59/feuer1/upload_staging/<job_name>-<step>
mkdir -p $UPLOAD_DIR
cp $EXPORT_DIR/policy/* $UPLOAD_DIR/
ls $UPLOAD_DIR/   # safetensors, config.json, tokenizer files all at root
```

## 4. Copy the launch config for reproducibility
```bash
cp hpc/skyrl_yaml/<config_used>.yaml $UPLOAD_DIR/rl_config.yaml
```

## 5. Scan for secrets (before upload — HF runs TruffleHog post-upload; catch it first)
```bash
trufflehog filesystem $UPLOAD_DIR --no-update                                   # if installed
trufflehog filesystem $EXPERIMENTS_DIR/<job_name>/<job_name> --no-update         # logs/traces too
# fallback:
grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34}|eyJ[a-zA-Z0-9._-]+)' $UPLOAD_DIR
```
Remove/redact anything found before proceeding.

## 6. Upload to HuggingFace — `laion/<job_name>-<step>-<size>`
Append BOTH the global step AND the base-model size suffix (`-20-32B`, `-30-8B`). The size suffix is required.
```bash
# tmux for long uploads. OMIT --private (no-value flag; default public).
hf upload laion/<job_name>-<step>-<size> $UPLOAD_DIR . --repo-type=model
```
**Why the `-<step>-<size>` repo:** the SkyRL trainer auto-pushes intermediates to a *canonical* repo
`laion/<job_name>` with the WRONG layout (weights under `checkpoints/step_N/`, not root) + auto-registers
it. We bypass that by uploading the manually-flattened export to the `-<step>-<size>` repo (weights at root).

## 7. Register in the DB (`--training-type RL`) — with cross-user FK safety
First delete the trainer's auto-registered duplicate **IF SAFE**, then push the correct row.

**CRITICAL — cross-user FK safety (pre-check BEFORE any delete):** if any **other-user** row in
`sandbox_jobs` / `sandbox_trial_model_usage` / anywhere FKs the auto-row, **STOP** — do NOT delete it and
do NOT mutate the FK'd rows; surface the conflict and leave the duplicate `models` row (one row of noise
≫ breaking someone else's evals). (This happened to `zhuang1` on 2026-05-26.) Restrict ALL writes to rows
you own (`feedback_supabase_filter_username`).
```python
other_users_fk = (c.table("sandbox_jobs").select("id,username,model_id")
    .eq("model_id", auto_row_id).neq("username", os.environ.get("USER","<you>")).execute())
if other_users_fk.data:
    print(f"SKIPPING auto-row delete — {len(other_users_fk.data)} other-user rows FK'd.")
else:
    c.table("models").delete().eq("name", "laion/<job_name>").execute()
    # optional, ONLY if pre-check passed: HfApi().delete_repo("laion/<job_name>", repo_type="model")
```
Then register the `-<step>-<size>` repo (`--training-type RL` is REQUIRED — the script defaults to SFT):
```bash
python scripts/database/manual_db_push.py \
  --hf-model-id laion/<job_name>-<step>-<size> \
  --base-model <base_model_hf> \
  --dataset-name <dataset_name> \        # comma-separated for multi-dataset → sets dataset_names
  --training-type RL                      # --wandb-run optional (defaults to now)
```
**Verify `--base-model` CAREFULLY** — the exact HF repo RL trained *from*, NOT a default. It's encoded in
the job-name suffix (`__GLM-4_7-swesmith-san` → `laion/GLM-4_7-swesmith-sandboxes-with_tests-…`); cross-check
the RL config YAML's `trainer.policy.model.path` (in the `.out` launch cmd) or `notes/ot-agent/rl_experiments.md`.
Getting it wrong corrupts the base_model_id tree used for size classification + RL-bump analysis.

## 8. Upload RL traces → `penfever/<job_name>`
From the **otagent** env. **Always pass `--skip_register` for RL** (RL trace datasets are NOT
Supabase-registered — the model is registered separately in step 7; only datagen registers its traces):
```bash
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job_dir "$EXPERIMENTS_DIR/<job_name>/<job_name>" \
  --repo_id penfever/<job_name> --episodes last --skip_register
```
Then add a **"Training Traces"** section to `$UPLOAD_DIR/README.md` (append if a model card exists, don't
overwrite) linking `penfever/<job_name>`, so the lineage is discoverable from the model page — it rides
along with the step-9 re-upload:
```markdown
## Training Traces
Training-time Daytona/Harbor rollouts: **[penfever/<job_name>](https://huggingface.co/datasets/penfever/<job_name>)**
(the `last` episode of each trial — the rollouts the policy trained on after rollback/truncation).
```

## 9. Parse metrics + preserve training logs (re-upload alongside the model)
Especially important on Jupiter (no W&B):
```bash
python scripts/analysis/parse_skyrl_metrics.py \
  $EXPERIMENTS_DIR/<job_name>/logs $UPLOAD_DIR/training_logs \
  --trace_jobs_dir $EXPERIMENTS_DIR/<job_name>/<job_name>/trace_jobs
cp $EXPERIMENTS_DIR/<job_name>/<job_name>/trainer_log.jsonl $UPLOAD_DIR/training_logs/ 2>/dev/null
cp $EXPERIMENTS_DIR/<job_name>/logs/<job_name>_*.out $UPLOAD_DIR/training_logs/
hf upload laion/<job_name>-<step>-<size> $UPLOAD_DIR . --repo-type=model   # additive
```
Produces `metrics.csv`, `vllm_metrics.csv`, `trial_stats.csv`, `report.md`, `reward_plot.png`.
**WARNING:** never use `huggingface_hub.upload_folder()` without `delete_patterns=[]` — it deletes files
absent locally and will clobber the weights. `hf upload` is additive (safe).

## 10. Clean up the experiments dir
Only after ALL above succeed, `rm -rf` the local job dir to free disk (detach a large GPFS `rm` with
nohup/tmux; never `du`/`find` to size it first — `feedback_cleanup_subagent_no_du_detach_rm`).

---

## Operating notes (folded from memory 2026-06-14)

- **Run the FULL checklist end-to-end, including Steps 8 & 9** (don't stop after DB register — those are minutes, not the hours of the model upload):
  - **Step 8 — trace upload:** `make_and_upload_trace_dataset --job_dir <exp>/<job>/<job> --repo_id penfever/<job_name> --episodes last`. It reads the inner `<job>/<job>` subdir (where Harbor writes `trace_jobs/`). **NEVER subsample/cap — upload the FULL trial set** (slowness is OK; an incomplete dataset is not). Run from the **otagent** env (needs `google.cloud.storage`).
  - **Step 9 — metrics + log re-upload:** `parse_skyrl_metrics.py` → CSV/report.md/reward_plot.png in `training_logs/` + copy trainer_log.jsonl/*.out, then re-upload via `hf upload`/`upload-large-folder` (NOT the `upload_folder` Python API — it deletes non-listed files and clobbers weights). The `training_logs/` in the HF repo are Jupiter's only W&B-equivalent.
- **Step 8 GPFS-hang FIXED (harbor `94379963`):** `iter_trial_dirs` used `root.rglob("*")` → stat'd the entire subtree (700k–900k stats on collapsed runs, 44-min stalls). Now prunes via `os.walk` + in-place `dirnames[:]` (stops descending at the trial dir). CLI unchanged. **Second, distinct bug — unbounded RAM:** `make_and_upload_trace_dataset` buffers the whole dataset and pushes only at the end (~146 GB RSS on ~24k rows → OOM rc=137 → 0 shards on hub); `chunk_size` doesn't bound peak RAM. Fix = incremental shard-push (in progress). Until it lands, large-run trace uploads on the shared login node may OOM — do NOT "fix" by sampling.
