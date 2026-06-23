---
name: sft-launch-jupiter
description: >-
  Launch SFT (LLaMA-Factory via hpc.launch) on JSC Jupiter (GH200). Covers the standard
  preamble, the vanilla Qwen3-8B launch, the 8B config variants (bs96 node-scaling,
  large/small-dataset optimized opt100k/opt1k, nothink, 131k), the 32B no-upload +
  consolidate→upload flow, the Qwen3.5 variant (separate sft-qwen35 conda env), dataset
  mixing/concatenation, custom flags, and where checkpoints go. Use when asked to SFT / launch
  a finetune / train a model on Jupiter. Reference: notes/jsc/jsc.md, CLAUDE.md, notes/ot-agent/sft_experiments.md.
---

# sft-launch-jupiter

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL code/config/sbatch edits go in the local Mac
> checkout (`~/Documents/OpenThoughts-Agent`) → commit → push → `git pull` on the cluster. **NEVER** hand-edit,
> `git commit`, or leave divergent/untracked changes on a cluster; no patch-by-rsync. New/changed configs are
> authored locally + synced, never on the cluster. Bake this into every subagent you dispatch from this skill.

SFT on Jupiter runs through **`python -m hpc.launch --job_type sft`** (LLaMA-Factory backend; `accelerate`
launcher, multi-node via DeepSpeed ZeRO-3). Conda env **`otagent`** for everything except Qwen3.5
(→ `sft-qwen35`). GH200, 4 GPUs/node.

## 1. Preamble (run FIRST, every session — pulls latest code + submodule)
```bash
source ~/.bashrc; source ~/secrets.env; cd /e/scratch/jureap59/feuer1/harbor; git stash; git pull; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent/SkyRL; git stash; git pull; conda activate otagent; \
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent; git pull; \
git submodule update --init --remote sft/llamafactory; source hpc/dotenv/jupiter.env;
```
(`git submodule update … sft/llamafactory` is essential — SFT won't run on a stale submodule.)

> **🚧 SUBMIT FROM THE REPO DIR WITH `DCFT` SET — the sbatch WORKDIR guard hard-fails otherwise.**
> The preamble above already does this (`cd …/OpenThoughts-Agent` + `source hpc/dotenv/jupiter.env`,
> which exports `DCFT`). Do NOT skip it. The generated `universal_sft.sbatch` resolves `WORKDIR` from
> `DCFT_PRIVATE → DCFT → $PWD`; submitting from `$HOME` or a scratch subdir with `DCFT` unset trips the
> guard (missing `hpc/shell_utils/triton_cache.sh` marker) and **`exit 1`s immediately** with a `FATAL:
> WORKDIR=... is not the OpenThoughts-Agent repo root` message. If you see that FATAL, re-run the
> preamble (`cd` repo + `source hpc/dotenv/jupiter.env`) and resubmit.

## 2. Vanilla launch (Qwen3-8B, 32k, single dataset → HF)
```bash
python -m hpc.launch --train_config_path sft/lf_configs/qwen3/32k_base.yaml \
  --time_limit 11:59:00 --num_nodes 2 --gpus_per_node 4 \
  --dataset DCAgent/perturbed-docker-exp-freelancer-tasks_glm_4.7_traces \
  --role_tag role --user_tag user --assistant_tag assistant --content_tag content \
  --hub_model_id laion/perturbed-docker-exp-freelancer-tasks_glm_4.7_traces
```
- **`--time_limit` max is `11:59:00`** — the Jupiter `booster` QOS caps wall at **12 h**
  (`QOSMaxWallDurationPerJobLimit` rejects more). For longer training, chain-restart with `--max_restarts N`
  (auto-resumes from the latest checkpoint — but see the save-interval caveat in §10).
- **HF upload + DB registration are COMPLETION/CLEANUP steps on Jupiter — not launch flags.** `push_to_hub`
  defaults **False** on Jupiter (compute nodes have no direct internet), so `--hub_model_id` alone does NOT
  auto-push; and **`--upload_to_database` is wired for the EVAL path only** (no SFT hook consumes it — it's a
  no-op for SFT). So after the run, do the **8B/32B SFT Cleanup Checklist**: `hf upload` from the *login* node
  + `manual_db_push.py` to register in Supabase. (Still pass `--hub_model_id` — it sets the canonical repo
  name the cleanup uploads to. `--push_to_hub true` can force a proxychains upload from compute, but the
  login-node cleanup upload is the reliable path.)
- **`--role_tag role --user_tag user --assistant_tag assistant --content_tag content` is MANDATORY for
  Harbor/DCAgent datasets** (they use `role`/`content` with `user`/`assistant`). LLaMA-Factory defaults to
  `from`/`value`+`human`/`gpt`; without these tags the thinking preprocessor finds **0 assistant messages →
  garbage training.** (Older ShareGPT sets like `tbench_oracle_solutions` use `--role_tag from --user_tag
  human --assistant_tag gpt --content_tag value`.)
- `--hub_model_id laion/<name>` → auto-push to HF on completion (8B writes consolidated safetensors at root).
  Public by default. **OMIT it for 32B** (see §6).
- `--time_limit 47:59:00` (Jupiter SFT max is generous; chain-restart auto-resumes from latest ckpt).

## 3. Checkpoints → `/e/data1/datasets/playground/ot/checkpoints`
SFT checkpoints belong in **`/e/data1/datasets/playground/ot/checkpoints/<job_name>`** (per jsc.md). NOTE:
`hpc/dotenv/jupiter.env` defaults `CHECKPOINTS_DIR=$DCFT_DATA/checkpoints` (= `…/ot-baf/checkpoints`). **Prefer
`--output_dir /e/data1/datasets/playground/ot/checkpoints/<job_name>`** — an `export CHECKPOINTS_DIR=…` gets
**clobbered by `source hpc/dotenv/jupiter.env`** in the preamble (the source runs after your export and resets
it to ot-baf), so the explicit flag is the reliable way to land in canonical `ot/checkpoints`. Dry-run and
confirm the rendered `output_dir` before the real submit.

## 4. Qwen3-8B config map (`sft/lf_configs/qwen3/` + `…/extra/`)
| Config | Use |
|---|---|
| `32k_base.yaml` | default 32k Qwen3-8B (thinking) |
| `32k_base_nothink.yaml`, `32k_base_2ep.yaml` | nothink variant / 2-epoch |
| `131k_base.yaml`, `131k_base_nothink.yaml` | 131k context |
| `extra/32k_base_bs96.yaml` | **bs96 node-scaling** (global_batch_size=96; see §5) |
| `extra/32k_base_bs96_opt1k.yaml` | **small datasets (<1k rows)** — opt: 7 epochs, lr 4e-5, warmup 0.1 |
| `extra/32k_base_bs96_opt100k.yaml` | **large datasets (≈11k+ rows)** — opt: 5 epochs, lr 4e-5, warmup 0.1 |
| `extra/131k_base_bs96.yaml`, `extra/32k_base_bs96_nothink.yaml` | 131k / nothink bs96 |
| `extra/32k_base_{1_7b,4b,14b,8b_base}.yaml`, `…30b_coder*` | other sizes / Qwen3-Coder-30B |

## 5. Scaling nodes for speed — the bs96 configs
The `bs96` configs fix **`global_batch_size: 96`** and auto-derive
`gradient_accumulation_steps = global_batch_size // (num_nodes * gpus_per_node)`. So **adding nodes shrinks
grad-accum → fewer micro-steps per optimizer step → faster wall-clock per epoch**, at a fixed effective batch.
E.g. `--num_nodes 4` (16 GPUs) → accum 6; `--num_nodes 8` (32 GPUs) → accum 3. Use a bs96 config + raise
`--num_nodes` to speed up a run without changing the effective batch (and thus the optimization dynamics).
`--num_nodes` must divide cleanly into 96/(gpus_per_node) — keep `96 % (num_nodes*4) == 0`.

## 6. 32B — NO HF upload at train time; consolidate → upload separately
32B (`sft/lf_configs/qwen3/32k_base_32b.yaml`, `…_32b_nothink.yaml`, `extra/32k_base_32b_bs96*.yaml`) trains
with DeepSpeed ZeRO-3 and writes **sharded `global_stepN/` state, NOT consolidated safetensors at root** — so
**launch WITHOUT `--hub_model_id`** (a direct push would be incomplete). Flow:
1. Train (no `--hub_model_id`), e.g.:
   `python -m hpc.launch --train_config_path sft/lf_configs/qwen3/32k_base_32b.yaml --time_limit 47:59:00 --num_nodes 4 --gpus_per_node 4 --dataset <ds> --role_tag role --user_tag user --assistant_tag assistant --content_tag content`
2. **Consolidate** (ZeRO-3 shards → fp32 → safetensors `final_repo/`):
   `python -m hpc.launch --job_type consolidate --consolidate_input $CKPT/<job_name> --consolidate_output_repo laion/<name> --consolidate_workdir <writable_wd>/<name> --time_limit 02:00:00 --num_nodes 1`
3. **Manually `hf upload` from `final_repo/`** (the consolidate auto-push has hit `BrokenPipeError` on big 32B
   uploads — don't rely on it; upload manually in a tmux session). Full steps: CLAUDE.md "32B SFT Job Cleanup
   Checklist". (Recognition: `ls $CKPT/<job>/` shows `global_stepN/`+`zero_to_fp32.py`, no root safetensors → 32B path.)

## 7. Qwen3.5 variant — different conda env (`sft-qwen35`)
Qwen3.5 (9B/27B) uses a hybrid GatedDeltaNet+Attention arch **not in transformers 4.x** → requires the
`sft-qwen35` env (transformers ≥5.3). Run the preamble but **`conda activate sft-qwen35`** instead of `otagent`.
Configs: `sft/lf_configs/qwen3_5/{32k_9b,131k_9b,32k_27b,131k_27b,8k_27b,32k_27b_bs96}.yaml`. After training:
copy `preprocessor_config.json` from the base model into the ckpt before upload (LLaMA-Factory doesn't emit it;
vLLM needs it); **9B writes full safetensors at root → SKIP consolidate** (per `feedback_qwen35_9b_no_consolidate`);
27B follows the 32B consolidate flow. (On Leonardo this env needs sbatch conda/WORKDIR patches — see CLAUDE.md;
on Jupiter the launcher handles activation.)

## 8. Dataset mixing & concatenation
`--dataset` is **repeatable** (append) → pass it multiple times for multi-dataset runs:
- **Concatenate:** `--dataset A --dataset B --mix_strategy concat` (just stacks all rows).
- **Interleave (mix by sampling):** `--mix_strategy interleave_under` (stop at smallest) or `interleave_over`
  (oversample to largest) + `--interleave_probs 0.7,0.3` (comma-separated sampling weights, per dataset order).
- For DB registration of a multi-dataset model, use comma-separated `--dataset-name` at cleanup (per
  `feedback_multi_dataset_db_registration`).

## 9. Custom flags (common)
`--num_nodes` / `--gpus_per_node` (4 on GH200); `--partition` booster. **Account: leave it default `reformo`
— do NOT pass `--account jureap59`.** The `jureap59` booster QOS is **suspended** (→ `Reason=InvalidQOS`, never
schedules); `reformo`/`normal` is the only runnable booster account for this user, and `hpc.py` hardwires it for
Jupiter SFT. SFT therefore shares the reformo allocation with the RL/datagen jobs — to schedule a queued SFT
faster you must **free a reformo slot** (not switch accounts) or get an admin QOS change.
`--time_limit HH:MM:SS`; `--seed N` (seeding experiments); `--overwrite_output_dir true` (re-run a job-name from
scratch instead of resuming); `--max_restarts N` (chain restarts).

> **`--overwrite_output_dir true` string-coercion bug (FIXED 2026-06-13, commit in `hpc/arguments.py`).** Job
> 858288 FAILED on every rank at ~8.5 min with `ValueError: Some keys are not used by the HfArgumentParser:
> ['overwrite_output_dir']`. Root cause: `overwrite_output_dir` is an `Optional[bool]` field defaulting to
> `None`, and the argparse builder only routed fields through `parse_bool_flag` when `isinstance(field.default,
> bool)` — which `None` is not. So `--overwrite_output_dir true` was stored as the **string `'true'`**, written
> into the rendered train config as `overwrite_output_dir: 'true'`, and rejected by LLaMA-Factory's
> `HfArgumentParser` (it expects a bool TrainingArgument). Fix: `_add_dataclass_arguments` now detects bool from
> the **type annotation** (`bool`/`Optional[bool]`), so any `Optional[bool]` CLI flag parses to a real bool. The
> documented `--overwrite_output_dir true` form is now safe. (Same bug latently affected any other
> `Optional[bool]` field passed on the CLI, e.g. `--do_train true`, `--packing true` — all fixed by the same
> change.) Make sure the cluster has pulled the fix (`git pull`) before relying on the flag.

> **`overwrite_output_dir` written into the LF train config — transformers v5 (FIXED 2026-06-13, `hpc/launch.py`
> `7b259c84`).** Even after the bool-coercion fix above, job 860017 hit the **identical** error
> `ValueError: Some keys are not used by the HfArgumentParser: ['overwrite_output_dir']` (~3 min in, during arg
> parse). Different root cause: the launcher MERGES `overwrite_output_dir` into `base_config` (it's a
> `LlamaFactoryArgs` field, used by `_configure_output_and_logging` for the resume/overwrite preflight guards),
> then DUMPS the whole `base_config` into `train_config.yaml`. On the **torch-2.11 / transformers 5.8 stack**,
> `overwrite_output_dir` was **removed from `Seq2SeqTrainingArguments`** (also why you see "warmup_ratio …
> removed in v5.2" warnings), so LLaMA-Factory's `HfArgumentParser(allow_extra_keys=False)` no longer recognizes
> it and rejects the YAML. Fix: `_strip_launcher_only_keys()` pops `overwrite_output_dir` (a launcher-only
> control flag) from `base_config` right before `_write_train_config`, AFTER all in-launcher consumers have run.
> The `--overwrite_output_dir true` CLI flag still works — its preflight semantics are unchanged; it just no
> longer leaks into the LF config. Verify post-launch: `grep -c overwrite_output_dir
> experiments/<exp>/configs/*_train_config.yaml` must print `0`.

Arbitrary LLaMA-Factory/trainer keys can be
overridden on the CLI (the launcher passes them through) — e.g. learning rate / epochs / cutoff_len if you need
to deviate from a config. `--dataset_dir` to point at a non-default datasets root.

## 10. After it completes — cleanup
8B: CLAUDE.md "8B SFT Job Cleanup Checklist" (rm intermediate `checkpoint-*` + `.cache`, `hf upload`, DB register
via `manual_db_push.py`, rm exp dir). 32B: the "32B SFT Job Cleanup Checklist" (the consolidate→upload flow, §6).
Both: **DB registration is a manual cleanup step** via `manual_db_push.py` (the `--upload_to_database` launch
flag is eval-only — a no-op for SFT here); HF upload is the login-node `hf upload` (compute has no internet).
Uploads default PUBLIC to `laion/`. Live
status: tail the `.out` for `{'loss':…, 'grad_norm':…}` step lines (trainer_log.jsonl is unreliable mid-run, per
`feedback_sft_status_via_out_not_jsonl`).

## 11. Trap: `AF_UNIX path too long` at dataset format-conversion (TMPDIR path-doubling)
Symptom: every rank dies ~2-3 min in, BEFORE the first `{'loss':…}` step, during HF
`datasets` format conversion (`dataset.map(num_proc=…)`). The map spins a `SyncManager`
that binds an **AF_UNIX socket under `$TMPDIR`**; the kernel caps `sun_path` at **108 bytes**,
so an over-long TMPDIR → `OSError: AF_UNIX path too long` → all ranks crash. (Killed jobs
860080 SIGTERM + 860229 exit 1, swesmith #223 stage 2.)

Root cause (fixed 2026-06-14): `hpc/sbatch_sft/universal_sft.sbatch` prepended `$DCFT/` to
`{experiments_dir}`, but the launcher (`hpc/launch_utils.py:708`, via `resolve_workspace_path`)
already substitutes `{experiments_dir}` as an **absolute** path. So `$DCFT/{abs}` doubled the
path (~172 chars) → `_TMPROOT` / `TMPDIR` / `WANDB_DIR` / the `mkdir`s all doubled. Fix removed
the `$DCFT/` prefix at lines 121/128/129/130/141 (the `#SBATCH --output=` line 6 and the RL
template already treat it as absolute).

Two levers if you ever see it again:
- **`SFT_KEEP_TMPDIR_LOCAL=1`** — template escape hatch (line 146): keeps TMPDIR on node-local
  `/tmp` (short socket path). Pass by exporting it in the launch shell BEFORE `python -m hpc.launch`
  (sbatch inherits the submit env via default `--export=ALL`): `export SFT_KEEP_TMPDIR_LOCAL=1`.
- Confirm the **rendered** sbatch (`<exp_dir>/sbatch/*.sbatch`) shows an un-doubled, short
  `_TMPROOT=` / `TMPDIR=` before it runs.

(Note: `hpc/sbatch_consolidate/capella_consolidate.sbatch:50-51` still has the same `$DCFT/{experiments_dir}`
prepend — harmless for consolidate, no `dataset.map`, but the same latent doubling.)

---

## Operating notes (folded from memory 2026-06-14)

- **Axolotl SFT (Sera/CoderForge): default `--nodes=4` (16 GH200) + `zero3_bf16.json`** even for small 1-epoch iteration jobs — cuts per-step ~4× (vs ~50min/1-node) and dodges the 1-node/zero1 step-5 OOM at 8B/32k. NOT `zero1.json` (replicates full model, no headroom). Only scale down to 2 nodes if the 4-node queue is congested. (LLaMA-Factory Qwen3.5 SFT may differ per `sft_experiments.md`.)
- **HPO sweeps: always pass `--job_name <short>`** (e.g. `100k_baseline`, `100k_epochs4`) so the SLURM name + exp dir + checkpoint dir all match. Without it the launcher derives a long dataset-string name that breaks resume-from-checkpoint.
- **SFT scaling-ablation workflow:** one SSH session → preamble → queue all sizes back-to-back (record every JID) → ScheduleWakeup at 600s + 1800s for early health (PENDING-past-30min, arrow-cache race, ENOSPC, NCCL, OOM are the common early failures — only caught by tailing `.out`) → switch to a 2h **CronCreate** once all are RUNNING-advancing/PENDING-with-ETA → fire the per-size post-training flow as EACH lands (don't wait for all): 32B = consolidate (`--time_limit 06:00:00`, NOT 24h — QOS rejects) → `manual_db_push`; 8B = SKIP consolidate, upload root safetensors. Checkpoint dirname carries the `__Qwen3-Nb` suffix even though `--job_name` doesn't. Verify the Supabase row after, then `rm -rf` exp + consolidate dirs.
