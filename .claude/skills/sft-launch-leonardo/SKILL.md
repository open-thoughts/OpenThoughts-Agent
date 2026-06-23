---
name: sft-launch-leonardo
description: Launch, monitor, and clean up a LLaMA-Factory SFT job on the CINECA Leonardo cluster (A100-64GB, no internet on compute nodes). Use when asked to run/relaunch/upload an SFT fine-tune on Leonardo — covers the pretokenization gotcha (large mixes time out tokenizing inside the 24h training job), the no-internet pre-download + offline-dataset handling, the mandatory sbatch post-patch, --max_restarts resume, the login-node-killer HF-upload workaround, and the 8B/32B consolidate+upload checklists.
---

# sft-launch-leonardo

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** ALL code/config/sbatch edits go in the local Mac
> checkout (`~/Documents/OpenThoughts-Agent`) → commit → push → `git pull` on the cluster. **NEVER** hand-edit,
> `git commit`, or leave divergent/untracked changes on a cluster; no patch-by-rsync. If launching/fixing a run
> needs a new or changed config (e.g. a `_dsfs` variant), author it locally + sync — never on the cluster.
> Bake this rule into every subagent you dispatch from this skill.

End-to-end SFT on CINECA Leonardo via `python -m hpc.launch --train_config_path …`.

> **⚠ Checkpoints/exports → `$WORK` (`$CHECKPOINTS_DIR`), NEVER `$SF`/`$SCRATCH_FAST`** (1 TB/over-quota →
> `OSError [Errno 122] Disk quota exceeded` mid-run; wiped the Delphi #6279 RL grid 2026-06-19). Verify the
> sbatch's ckpt/export paths before submitting — see `ops/leonardo/ops.md` "WRITE-PATH MANDATE".

Leonardo is **no-internet-on-compute-nodes** + a **login-node process killer** +
**24h max wall** — three constraints that drive almost every quirk below. Read
§5 (pretokenization) and §10 (canary blockers) before launching anything large;
they are the two ways a run silently wastes a 24h slot.

Authoritative source docs (this skill distills them — read for full context):
- `experiments/active/delphi/rl-scaling-laws-6279/SFT_LEONARDO_INSTRUCTIONS.md`
  (the launch template + Delphi-specific dataset/template handling + canary blockers).
- `OpenThoughts-Agent/CLAUDE.md` → "CINECA Leonardo Access", "SFT Launch on Leonardo",
  "Leonardo HF Upload", "8B/32B SFT Job Cleanup Checklist".

## 1. Cluster facts

- A100-**64GB** GPUs, 4/node; SLURM; **no internet on compute nodes** (login nodes have
  direct internet). Account `AIFAC_5C0_290`, partition `boost_usr_prod`, **max wall `23:59:00`** (1-day limit).
- **Compilers come from conda** (GCC 15.2, CUDA 13.2) — do NOT `module load gcc cuda` (system modules too old).
- Key paths:
  - Code (`$DCFT`): `/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent`
  - `$HF_HUB_CACHE`/`$HF_HOME`: `/leonardo_work/AIFAC_5C0_290/bfeuer00/data/hub`
  - `$CHECKPOINTS_DIR`/experiments: `/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments`
  - conda base: `/leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3`

## 2. Conda env: `otagent` vs `sft-qwen35` (pick by architecture)

- **`otagent`** (transformers 4.x) — dense Qwen3 / Llama-3-tokenizer models. Use this by default.
- **`sft-qwen35`** (transformers ≥5.3.0, torch 2.10+, deepspeed 0.18+) — ONLY for the
  **Qwen3.5 hybrid GDN+Attention** arch (9B/27B). Using it elsewhere drags in CCE/DeepSpeed pins you don't want.
- The CLAUDE.md SFT preamble/patch snippets say `sft-qwen35`; **override to `otagent`** for dense-Qwen3 jobs
  (e.g. the Delphi midtrained checkpoints) in BOTH the preamble (§3) and the sbatch patch (§7).

**Qwen3.5 hybrid (9B/27B) — concrete launch.** Keep `sft-qwen35` in the preamble (§3) AND the sbatch patch
(§7 — do NOT swap to `otagent`). Configs: `sft/lf_configs/qwen3_5/{32k_9b,131k_9b,32k_27b,131k_27b}.yaml`
(require transformers ≥5.3.0 — Qwen3.5's GDN+Attention arch isn't in transformers 4.x). Pass `DISABLE_VERSION_CHECK=1`:
```bash
DISABLE_VERSION_CHECK=1 python -m hpc.launch \
  --train_config_path sft/lf_configs/qwen3_5/32k_9b.yaml \
  --time_limit 23:59:00 --num_nodes 4 --gpus_per_node 4 \
  --dataset DCAgent/exp_tas_optimal_combined_traces \
  --role_tag role --user_tag user --assistant_tag assistant --content_tag content \
  --hub_model_id laion/exp_tas_optimal_combined_traces-Qwen3.5-9B
```
(Global `--role_tag` flags are fine for a SINGLE homogeneous dataset like this; for mixed-schema MIXES use
the registry `dataset_info.json` per-dataset tags instead — §4. Qwen3.5 cleanup follows the **8B path** —
root safetensors, no consolidate — plus copy `preprocessor_config.json` from the base, per §12.)

## 3. Pre-launch preamble (login node, tmux)

```bash
ssh Leonardo   # step-ca cert; complete 2FA once, socket persists ~8h
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh && \
conda activate otagent && \   # or sft-qwen35 for Qwen3.5 hybrid
cd /leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent && GIT_TERMINAL_PROMPT=0 git pull && \
git submodule update --init --remote sft/llamafactory && \
source hpc/dotenv/leonardo.env && source ~/secrets.env
```

## 4. Launch command (template)

```bash
DISABLE_VERSION_CHECK=1 python -m hpc.launch \
  --train_config_path sft/lf_configs/<family>/<cfg>.yaml \
  --time_limit 23:59:00 \
  --num_nodes 4 --gpus_per_node 4 \
  --model_path <hf_base_model> \
  --dataset_dir <registry dir, e.g. sft/delphi> \
  --dataset <instr>[,<warmup>] [--mix_strategy interleave_under --interleave_probs 0.9,0.1] \
  --hub_model_id <hub_model_id> \
  --internet_node \          # skip launcher pre-download (everything pre-cached in §6) — see canary blocker #3
  --max_restarts 2           # 24h-wall resume chain — see §9
# For MULTI-NODE: use a config with `data_shared_file_system: true` (e.g. <cfg>_dsfs.yaml) — see §5 (the real
#   "24h timeout" cause). `--pretokenize` / `--pretokenize_bprod` is an OPTIONAL optimization, NOT the multi-node fix.
# push_to_hub stays FALSE (set in config / canary blocker #5); upload from login node post-run (§11).
```
- Config files: `sft/lf_configs/<family>/*.yaml`. They bake `template`, `cutoff_len`,
  `num_train_epochs`, `learning_rate`, full-FT + DeepSpeed. **Do NOT pass epochs/LR/template/cutoff as CLI flags
  for a controlled set** — vary only `--model_path`, `--hub_model_id`, `--dataset`. Always `--dry_run` the first cell
  to confirm the rendered config (model, template, epochs, LR, role tags, push_to_hub) before submitting the batch.
- **Dataset parse config lives in the registry `dataset_info.json`, NOT in CLI `--role_tag` flags.** Harbor/DCAgent
  datasets that mix schemas (`messages` role/content vs `conversations` from/value vs `conversation` role/content)
  MUST be registered with per-dataset `columns`/`tags` — a single global `--role_tag` silently yields 0 assistant
  turns on the mismatched sources. Launch with `--dataset_dir <registry>` and reference datasets by registered name.

## 5. ⚠️ Multi-node dataset prep — the `data_shared_file_system` cache race (the REAL "24h timeout")

**Symptom:** a *tiny* model (e.g. 447M) multi-node SFT job sits idle and gets killed at
the 24h wall with state `TIMEOUT`, having never checkpointed — which *looks* like "it
spent 24h tokenizing." **It did not.** (Verified 2026-06-12 on the Delphi grid: tokenizing
the 555k/428k 90/10 mixes takes **~65 s** with `preprocessing_num_workers: 16`, not hours.)

**Real root cause — an HF-datasets cache RACE across nodes.** LLaMA-Factory gates dataset
prep with `main_process_first(..., local=(not data_args.data_shared_file_system))`. With the
default **`data_shared_file_system: false`**, the barrier is **per-node**, so *each node's*
local-rank-0 (rank 0 AND rank 4 on a 2-node job) runs `datasets.map` **simultaneously against
the same shared GPFS cache dir**. They race: one rank hits `FileNotFoundError` on a parquet
cache shard mid-map, the surviving ranks' collective hangs, the **NCCL watchdog SIGABRTs at
the 600s/30-min mark**, and the slurm step never releases the allocation → the job idles until
the 24h wall (hence the misleading TIMEOUT). It's **intermittent** (timing-dependent) — cells
that "got lucky" on cache timing run fine, which is why only some cells of a grid hang.

**Fix — `data_shared_file_system: true`** (config knob, NOT an hparam/methodology change — same
tokens/loss): global barrier, only global-rank-0 tokenizes+writes the shared cache, everyone
else waits then reads. Apply via a per-run config copy (e.g. `sft/lf_configs/delphi/4k_sft_dsfs.yaml`
= base + that one line) so the shared base yaml + in-flight cells stay byte-identical.
**Grid-wide exposure:** every cell sharing `data_shared_file_system: false` is exposed; harden
future/PENDING launches with the `_dsfs` config (the running ones just got lucky). It is safe to
do mid-grid because it changes no training math (infra/correctness only).

### Pretokenization (a real optimization — but it was NOT the fix above)
The built-in `pretokenize` job type exists and is genuinely useful for *legitimately* slow prep
or to decouple tokenization from the training walltime — but **don't reach for it before
confirming tokenization is actually the bottleneck** (it usually isn't; the 24h hang is almost
always the cache race above or another collective hang, NOT slow tokenization).
- `--pretokenize` on the SFT launch → `schedule_pretokenize` runs a standalone tokenize job
  (`--pretokenize_bprod` puts it on the `boost_qos_bprod` 128-node fast queue) and the SFT job
  depends on it; or `--job_type pretokenize` standalone. It writes a deterministic
  `tokenized_path` (`<dir>/<dataset>_<model>_tokenized`) that the SFT job auto-detects + reuses
  (`hpc/launch.py:_maybe_assign_tokenized_path` → "Found pre-tokenized dataset … — reusing it").
- Reuse keys on (dataset, model); a cell already at `tokenized_path` reuses with no flag.
- **Diagnostic order when a cell hangs with no train step:** (1) check `data_shared_file_system`
  (the usual culprit on multi-node), (2) check for the §9.4 schema-key `KeyError`, (3) only then
  suspect genuinely-slow tokenization → pretokenize.

## 6. No-internet-on-compute handling

Compute nodes can't reach HF Hub. Three consequences:
1. **Pre-download model + datasets on the login node first** (detached tmux, with retry):
   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1
   hf download <hf_base_model> --repo-type model       # monitor by $HF_HUB_CACHE size growth, not a file
   hf download <dataset_repo> --repo-type dataset      # unset HF_HUB_ENABLE_HF_TRANSFER if hf_transfer not installed
   ```
2. **`--internet_node`** — passes to `hpc.launch` to **skip the launcher's own pre-download**
   (`_materialize_dataset_and_model` does `snapshot_download(repo_id="<registered-name>")` → 404 on a
   registry name → looks like a stall). Everything is already cached by step 1, so skip it.
3. **Offline dataset *loading*:** LF tries to fetch a dataset *loading script* over the network for
   `hf_hub_url` registry entries → fails offline. **Repoint each dataset in the registry `dataset_info.json`
   from `hf_hub_url` to a local `file_name` parquet dir** (LF `load_from=file`, fully offline). Also strip the
   ~6 *global* schema-tag keys the launcher injects into the train_config (they `KeyError` on heterogeneous
   mixes — let LF use the per-dataset registry tags).
4. **`push_to_hub: false`** in the config (creating the HF repo at train start hits the unreachable hub →
   `OfflineModeIsEnabled` crash). The model saves to local disk; upload from the login node post-run (§11).

## 7. MANDATORY sbatch post-patch (the launcher does NOT do this for Leonardo SFT)

After `hpc.launch` generates the sbatch and BEFORE `sbatch`-ing it, patch it (the CLAUDE.md snippet
activates `sft-qwen35` — change to `otagent` for dense Qwen3):
```bash
SBATCH=experiments/<exp_dir>/sbatch/<job_name>_sft.sbatch
# 1. conda activation (otagent, NOT sft-qwen35, for dense Qwen3)
sed -i 's|# No conda activation configured|source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh\nconda activate otagent|' $SBATCH
# 2. WORKDIR (defaults to $PWD, wrong on compute nodes)
sed -i 's|WORKDIR="$PWD"|WORKDIR="/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent"|' $SBATCH
# 3. DCFT
sed -i 's|export DCFT="$WORKDIR"|export DCFT="/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent"|' $SBATCH
# 4. fix doubled paths
sed -i 's|\$DCFT//leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/experiments|/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent/experiments|g' $SBATCH
sbatch $SBATCH
```
Sanity-check before submit: `grep -n "conda activate\|WORKDIR=\|export DCFT" $SBATCH`
→ expect `conda activate otagent`, literal WORKDIR/DCFT paths, no `$DCFT//leonardo_work` doubling.
**A fast (~20s) ExitCode-1 failure is almost always a missing/incorrect post-patch** (the script
couldn't activate conda or `cd` to WORKDIR). Re-patch and resubmit — Slurm snapshots the script at submit,
so re-submit after editing (don't expect a queued job to pick up edits).

⚠️ **The `sed` patterns above can SILENTLY NO-OP on a newer launcher template** (verified 2026-06-12). The new
template no longer emits literal `WORKDIR="$PWD"` / `export DCFT="$WORKDIR"` lines — it resolves `WORKDIR` at
runtime from `$DCFT`/`$DCFT_PRIVATE` env vars via a dotenv-autoload loop. On the compute node neither is set, so
the loop falls through to `$PWD` (= `$HOME`), `$HOME/hpc/dotenv/leonardo.env` doesn't exist, `$DCFT` stays unset,
`WORKDIR=$HOME`, and `source "$WORKDIR/hpc/shell_utils/triton_cache.sh"` → "No such file" → `set -e` exit 1 in ~22s.
So instead of the brittle line-match `sed`, **inject literal `export DCFT=<code path>` and `export DCFT_PRIVATE=<code path>`
immediately before the `conda activate` line** in the generated sbatch (the §5-intent adapted to the new template).
After patching, `grep` the spooled script (`scontrol write batch_script <jobid> -`) to confirm both exports are present.

## 8. Multi-node launcher: accelerate (not torchrun)

Leonardo multi-node SFT uses `accelerate launch` (`training_launcher="accelerate"` in `hpc.py`, already wired) —
torchrun's c10d rendezvous fails on Leonardo's inter-node TCP. `--num_nodes 4 --gpus_per_node 4` = 16 A100-64GB;
DeepSpeed ZeRO-3 (`ds_z3_accelerate.json`) handles up to ~9.7B full-FT. Small dense models (≤~2.5B) fit 1–2 nodes
and queue faster — but keep node count identical within a comparison set.

## 9. 24h wall + `--max_restarts` resume

`--max_restarts N` submits an N-deep `afterany` chain that auto-resumes from the latest checkpoint, so a cell
gets N+1 walltime slots. **It is MUTUALLY EXCLUSIVE with `--overwrite_output_dir`** (`hpc.launch` hard-errors):
overwrite wipes the checkpoint dir each restart → restart-from-scratch. So drop `--overwrite_output_dir`; the chain
relies on checkpoint-resume (`save_total_limit: 1` keeps the latest). For a genuine clean re-run, `rm -rf` the
output dir first instead of re-adding overwrite. **Caveat:** resume only helps if the job *checkpoints* — a job that
TIMEOUTs in tokenization (§5) never saves, so resume re-tokenizes forever. Fix tokenization (§5) first.

## 10. Canary-discovered blockers — apply to EVERY cell

(From `SFT_LEONARDO_INSTRUCTIONS.md` §9; all verified.)
1. **HF pre-download stalls** → pre-stage in detached tmux with retry BEFORE launch (§6.1).
2. **`upath` missing** in `otagent` → `pip install universal_pathlib` (now persistent).
3. **Launcher can't resolve REGISTERED names** for pre-download → pass **`--internet_node`** (§6.2).
4. **Offline dataset loading** → repoint registry to local `file_name` parquet + strip global schema-tag keys (§6.3).
5. **`push_to_hub: true` crashes at repo-create** offline → set **`push_to_hub: false`**, upload from login node (§11).
6. **Template × tokenizer mismatch** (the top silent ruin): use the template whose control tokens match the
   tokenizer. For Llama-3-family tokenizers (e.g. Delphi: `<|start_header_id|>`/`<|eot_id|>`, NO ChatML) use a
   **llama3-family** template, NOT `qwen3` (ChatML `<|im_start|>` would shred every example). `--dry_run` + eyeball
   the first rendered example of EACH source (an instruction turn AND a warmup `<think>` example) before launching.

## 11. HF upload on Leonardo — use sbatch, NOT the login node

Leonardo's login node **SIGKILLs any long process at ~100s** (kills `nohup`, `tmux`, `systemd-run` alike — the
killer is process-agnostic). The login node has direct internet, but you can't run a multi-minute upload there.
**Upload via an sbatch job on a compute node + an SSH tunnel back to login.**

Pre-flight (from local Mac — the step-ca cert expires ~12h):
```bash
step ssh certificate 'bfeuer00' --provisioner cineca-hpc ~/.ssh/leonardo_daytona --no-password --insecure
ssh-keygen -R login.leonardo.cineca.it && rsync -avz -e 'ssh -i ~/.ssh/leonardo_daytona -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
  ~/.ssh/leonardo_daytona ~/.ssh/leonardo_daytona.pub ~/.ssh/leonardo_daytona-cert.pub bfeuer00@login.leonardo.cineca.it:~/.ssh/
ssh-keygen -L -f ~/.ssh/leonardo_daytona-cert.pub | grep Valid   # confirm not expired
```

sbatch upload template (compute node + `start_proxy_tunnel.sh` SOCKS forward → `hf upload`):
```bash
cat > /leonardo_work/AIFAC_5C0_290/bfeuer00/upload_<job>.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=hf_upload_<short>
#SBATCH --output=<workdir>/upload_logs/upload_sbatch.log
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --gres=gpu:1
#SBATCH --partition=boost_usr_prod --account=AIFAC_5C0_290 --qos=boost_qos_dbg
set -e
source /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/etc/profile.d/conda.sh
conda activate otagent && source ~/secrets.env
DCFT=/leonardo_work/AIFAC_5C0_290/bfeuer00/code/OpenThoughts-Agent
unset LD_PRELOAD; export PATH="/leonardo_work/AIFAC_5C0_290/bfeuer00/proxychains/bin:${PATH}"
CMD_PREFIX=$(bash "$DCFT/eval/leonardo/start_proxy_tunnel.sh")
cd <upload_dir>   # $CHECKPOINTS_DIR/<job> (8B) or <workdir>/final_repo (32B)
$CMD_PREFIX /leonardo_work/AIFAC_5C0_290/bfeuer00/miniforge3/envs/otagent/bin/hf upload <hub_model_id> . . --repo-type=model
EOF
sbatch /leonardo_work/AIFAC_5C0_290/bfeuer00/upload_<job>.sbatch
```
- Always **`hf upload`** (sequential, 3-arg form), NEVER `hf upload-large-folder` (deprecation stub + deadlocks on HF LFS 429s). 131GB 32B → ~4 min through the tunnel; 30-min wall fits `boost_qos_dbg`; resume is automatic (`.cache/huggingface/`).
- **Fallback when the step-ca cert is expired** (`Permission denied (publickey)` → sbatch dies in ~19s): the LOGIN node has direct HF internet, so a **detached `nohup hf upload` on the login node** works as a one-off (it survives long enough for a single-model upload) — used 2026-06-12 when the cert lapsed. Prefer the sbatch path when the cert is fresh.

## 12. SFT cleanup checklists (from CLAUDE.md)

**Recognition heuristic** — after training completes, check the checkpoint root:
`ls $CHECKPOINTS_DIR/<job>/ | grep -E 'safetensors|global_step'`
- `model-*.safetensors` at root → **8B path** (also Qwen3.5 — no consolidate). 
- `global_stepN/` + `zero_to_fp32.py`, no safetensors → **32B path** (DeepSpeed ZeRO-3 shards — must consolidate).

### 8B (root safetensors)
0. Cancel pending retries: `squeue -u $USER --format='%i %j %T' | grep <job> | grep PENDING | awk '{print $1}' | xargs -r scancel`
1. `rm -rf $CHECKPOINTS_DIR/<job>/checkpoint-*  $CHECKPOINTS_DIR/<job>/.cache` (drop intermediate ckpts).
   *(Qwen3.5 only: also `cp <base>/preprocessor_config.json $CHECKPOINTS_DIR/<job>/`.)*
2. **Upload** to `<hub_model_id>` (the launch's configured `--hub_model_id`, no suffix for a full run) — via the §11 sbatch-tunnel (Leonardo). `--private` is a no-value flag; default public, so omit it.
3. **Register in DB**: `python scripts/database/manual_db_push.py --hf-model-id <hub_model_id> --base-model <base_hf> --dataset-name <name|comma,list>` (SFT is the default `--training-type`). **SKIP for HF-only series** (e.g. Delphi #6279 — `enable_db_registration: false`).
4. `rm -rf $EXPERIMENTS_DIR/<job>` only after 1–3 succeed.

### 32B (DeepSpeed ZeRO-3 — consolidate first)
0. Cancel pending retries (as above).
1. Verify `trainer_log.jsonl` shows `current_steps == total_steps` (don't salvage partials; relaunch + resume instead).
2. **Consolidate** shards → safetensors:
   ```bash
   python -m hpc.launch --job_type consolidate \
     --consolidate_input $CHECKPOINTS_DIR/<job> \
     --consolidate_output_repo <hub_model_id> \
     --consolidate_workdir <writable_workdir>/<job> \
     --time_limit 02:00:00 --num_nodes 1
   ```
   Produces `<workdir>/<job>/final_repo/` (safetensors + tokenizer + config at root). The consolidate job's own
   HF auto-push can `BrokenPipeError` on big 32B uploads — treat consolidate as done once `final_repo/` is fully
   written, and upload manually (step 3).
3. **Upload from `final_repo/`** (NOT the original sharded dir) via the §11 sbatch-tunnel.
4. **Register in DB** (same as 8B step 3; skip for HF-only series).
5. `rm -rf $CHECKPOINTS_DIR/<job>` AND `<workdir>/<job>` after 2–4 succeed (sharded ckpt ~700GB + workdir ~200GB).

> ## ⛔ Per-series no-DB exception
> Some series are **HF-upload ONLY** (e.g. the Delphi RL scaling-laws #6279 grid — YAMLs set
> `enable_db_registration: false`). For those, run the checklist **through the HF-upload step only** and
> **SKIP `manual_db_push.py`** (and do NOT register the base/anchor checkpoints either — passing one as
> `--base-model` auto-creates a base-model row). Honor any documented no-DB decision.

## 13. Guardrails

- **Multi-node → `data_shared_file_system: true`** (§5) — the #1 silent 24h "timeout": a tiny model idling to the wall is almost always the per-node HF-datasets cache race (NCCL hang), NOT slow tokenization (~65s). Diagnose in order: dsfs → §9.4 schema-KeyError → only then pretokenize.
- **No internet on compute** — pre-download model+datasets on the login node, `--internet_node`, `push_to_hub: false`,
  registry datasets repointed to local `file_name` parquet (§6).
- **Always patch the generated sbatch** (§7) — fast ExitCode-1 = missing patch.
- **`--max_restarts` ⊥ `--overwrite_output_dir`** (§9); use resume, not overwrite.
- **Uploads go through the sbatch-tunnel, never a >100s login-node process** (§11); `hf upload`, never `upload-large-folder`.
- **`--dry_run` the first cell** of any batch; verify template×tokenizer (§10.6) — a mismatch silently trains garbage.
- Use **`otagent`** unless it's the Qwen3.5 hybrid arch (then `sft-qwen35`) — §2.
- NEVER cancel/relaunch another user's job; restrict DB writes to your own rows.
