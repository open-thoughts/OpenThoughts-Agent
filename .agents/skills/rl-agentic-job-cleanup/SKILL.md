---
name: rl-agentic-job-cleanup
description: >-
  Preserve + publish a finished RL (SkyRL/GRPO) training checkpoint after the job terminates
  (completed at max_steps OR early-stopped/scancelled) on an HPC cluster (Jupiter/Leonardo/Perlmutter).
  Covers: cancel pending retries, pick the BEST checkpoint by trailing-5 EMA of reward across the full
  restart chain, flatten weights to repo root, secret-scan, `hf upload` to laion/<job>-<step>-<size>,
  Supabase DB register (--training-type RL + cross-user FK safety pre-check), upload training traces to
  penfever/<job>, parse metrics, and clean up. Use when an RL run needs its model uploaded + registered,
  or when asked to "run the RL cleanup checklist". Distinct from SFT cleanup (that's a different flow).
---

> ⚠ **Do not add comments to YAMLs. Report your recommendations directly to the supervisor.**

# rl-agentic-job-cleanup

After an RL job terminates, publish `laion/<job_name>-<step>-<size>` (weights at repo root), trace dataset
`penfever/<job_name>`, and a Supabase `models` row (`training_type=RL`).

## Rules
- **`hf upload`, NEVER `hf upload-large-folder`** (deprecated stub; deadlocks on HF LFS 429s). Wrap long uploads in **`tmux`**, not `nohup`.
- **`--private` is a no-value flag** — do NOT pass `--private false`. Default is PUBLIC; omit it.
- Run trace upload + `parse_skyrl_metrics.py` from the **`otagent` conda env** (the RL venv lacks `google.cloud.storage` + matplotlib).
- On **Leonardo**, login-node `hf upload` is SIGKILLed at ~100s — use the sbatch+tunnel upload pattern.

## 0. Cancel pending retries
```bash
squeue -u $USER --format='%.18i %.80j %.8T' | grep <job_name>
scancel <retry_job_ids>
```

## 1. Select the checkpoint — trailing-5 reward EMA
```bash
# NOTE: there is an empty exports/ at the base level — ignore it. Real HF-exportable ckpts are nested:
ls -lt $EXPERIMENTS_DIR/<job_name>/<job_name>/exports/ | head -10
```
Use the **EMA of `reward/avg_raw_reward` over a trailing-5 window**.

Rules:
- **EMA across ALL chronological steps, regardless of chain restarts.** Collect every `.out` and sort by
  `trainer/global_step`; do not compute it per-chain link.
- Standard 5-period EMA: `α = 2/(5+1) = 1/3`; `EMA_n = α·reward_n + (1−α)·EMA_{n−1}`, `EMA_1 = reward_1`.
- **Never select the first saved checkpoint** (`global_step_5` with `hf_save_interval: 5`) — EMA not warmed
  up. Start from the second-saved step (typically 10).
- Among saved, aligned checkpoints (multiples of `hf_save_interval`, excluding the first), upload the highest
  EMA. If scancelled before a save-aligned max-step, cap at the latest saved multiple.

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
From the job logs / `trainer_log.jsonl`: `https://wandb.ai/dogml/OpenThoughts-Agent/runs/<run_id>`. (Jupiter has no W&B — omit.)

## 3. Flatten model files to the upload-dir root
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

## 5. Scan for secrets before upload
```bash
trufflehog filesystem $UPLOAD_DIR --no-update                                   # if installed
trufflehog filesystem $EXPERIMENTS_DIR/<job_name>/<job_name> --no-update         # logs/traces too
# fallback:
grep -rIE '(sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|ghp_[a-zA-Z0-9]{36}|hf_[a-zA-Z0-9]{34}|eyJ[a-zA-Z0-9._-]+)' $UPLOAD_DIR
```
Redact before proceeding (the wrapper emits a JSON finding record even when clean):
```bash
python -m scripts.harbor.secret_redaction "$UPLOAD_DIR"
```

## 6. Upload to HuggingFace — `laion/<job_name>-<step>-<size>`
Include the global step and base-model size suffix (`-20-32B`, `-30-8B`).
```bash
# tmux for long uploads. OMIT --private (no-value flag; default public).
hf upload laion/<job_name>-<step>-<size> $UPLOAD_DIR . --repo-type=model
```
> The SkyRL trainer auto-pushes `laion/<job_name>` with weights under `checkpoints/step_N/`. Upload the
> manually flattened export to `-<step>-<size>` instead.

## 7. Register in the DB (`--training-type RL`) — with cross-user FK safety
Delete the trainer's auto-registered duplicate **only if safe**, then push the correct row. If any **other-user**
row in `sandbox_jobs`, `sandbox_trial_model_usage`, or elsewhere FKs the auto-row, stop; do not delete or mutate
the FK'd rows. Restrict all writes to rows you own.
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
**Verify `--base-model`** is the exact HF repo trained from, not a default. Cross-check the job-name suffix,
`trainer.policy.model.path` in the `.out` launch command, or `notes/ot-agent/rl_experiments.md`.

## 8. Upload RL traces → `penfever/<job_name>`
From the **otagent** env, always pass `--skip_register` for RL; register the model separately in step 7.
```bash
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job_dir "$EXPERIMENTS_DIR/<job_name>/<job_name>" \
  --repo_id penfever/<job_name> --episodes last --skip_register
```
**Never subsample or cap**: upload the full trial set. The script reads the inner `<job>/<job>` `trace_jobs/`.
For image-backed Jupiter runs, pass the same inner run root. The exporter detects `artifact_store.img`, refuses an unsafe mount while a writer lock is active, mounts it read-only after the link exits, and unmounts it when export finishes. Legacy bare `trace_jobs/` runs keep the existing path.

> `make_and_upload_trace_dataset` buffers the full dataset; `chunk_size` does not bound peak RAM. Large login-node
> uploads can OOM; do not respond by sampling.

Then add a **"Training Traces"** section to `$UPLOAD_DIR/README.md` (append if a model card exists, don't
overwrite) linking `penfever/<job_name>`:
```markdown
## Training Traces
Training-time Daytona/Harbor rollouts: **[penfever/<job_name>](https://huggingface.co/datasets/penfever/<job_name>)**
(the `last` episode of each trial — the rollouts the policy trained on after rollback/truncation).
```

## 9. Parse metrics and preserve training logs
```bash
python scripts/analysis/parse_skyrl_metrics.py \
  $EXPERIMENTS_DIR/<job_name>/logs $UPLOAD_DIR/training_logs \
  --trace_jobs_dir $EXPERIMENTS_DIR/<job_name>/<job_name>/trace_jobs
cp $EXPERIMENTS_DIR/<job_name>/<job_name>/trainer_log.jsonl $UPLOAD_DIR/training_logs/ 2>/dev/null
cp $EXPERIMENTS_DIR/<job_name>/logs/<job_name>_*.out $UPLOAD_DIR/training_logs/
hf upload laion/<job_name>-<step>-<size> $UPLOAD_DIR . --repo-type=model   # additive
```
Produces `metrics.csv`, `vllm_metrics.csv`, `trial_stats.csv`, `report.md`, `reward_plot.png`.
The metrics reader also detects a missing `<run>/trace_jobs` beside `artifact_store.img` and mounts the inactive image read-only for trial statistics.
**WARNING:** never use `huggingface_hub.upload_folder()` without `delete_patterns=[]` — it deletes files absent
locally and clobbers the weights. `hf upload` is additive (safe).

## 10. Verify the published model and write the completion record
The model is complete only when its remote repo has weights, `README.md` with **Training Traces**, and
redacted `training_logs/`. Record every artifact as **present**, **absent**, or **not applicable** in the
cleanup handoff; an absent required artifact is not a completed cleanup.
```bash
hf api repo-info laion/<job_name>-<step>-<size> --repo-type model --expand siblings \
 | python -c 'import json,sys; files=[x["rfilename"] for x in json.load(sys.stdin)["siblings"]]; print({"weights": any(x.endswith(".safetensors") for x in files), "readme": "README.md" in files, "training_logs": any(x.startswith("training_logs/") for x in files)})'
```
Fetch `README.md` and verify it contains `## Training Traces` and the exact `penfever/<job_name>` URL.

## 11. Clean up the experiments dir
After all prior steps succeed, `rm -rf` the local job dir. Detach a large GPFS removal with `nohup` or `tmux`;
do not `du` or `find` it first.
