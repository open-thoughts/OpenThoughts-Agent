---
name: datagen-launch
description: >-
  Launch a datagen (trace-generation) job on an HPC cluster (Jupiter/Leonardo/Perlmutter) via the
  OpenThoughts-Agent `hpc.launch --job_type datagen` entrypoint — the cluster-AGNOSTIC general flow:
  extract tasks from a parquet, then submit a managed vLLM-serve + Harbor/Daytona trace run that uploads
  the trajectories to an HF repo. Use when asked to start a datagen / trace-generation job, generate
  agent traces from a task dataset, or advance / start a row in the MiniMax-M2.7 datagen tracker. For the
  Iris TPU path use **datagen-launch-iris** instead; per-cluster particulars (ssh, paths, conda env, which
  vllm-serve config matches the cluster's GPUs, JSC pre-download) live in `.claude/ops/<cluster>/`.
---

# datagen-launch

End-to-end launch of a datagen (trace-generation) job through
`python -m hpc.launch --job_type datagen`. A datagen job stands up a managed vLLM endpoint serving a
teacher model, runs Harbor agent rollouts against a set of tasks in Daytona sandboxes, and uploads the
resulting trajectories to an HF dataset repo. There is **no model checkpoint** — the artifact is the
trace dataset.

This skill is **cluster-AGNOSTIC**. It documents the shared flow, flags, and gotchas. Defer everything
cluster-specific — how to `ssh` / which login node, repo path, conda env activation, JSC login-node
pre-download, and which `datagen_config` (vllm-serve) YAML matches the cluster's GPUs (H200 vs GH200 vs
MI250X) and the model — to `.claude/ops/<cluster>/`. Do NOT inline cluster-specific values here.

Sibling skills: **datagen-launch-iris** (Iris TPU path), **datagen-job-cleanup** (post-run upload + disk
cleanup), **datagen-reduce-dataset-snapshots** (shrink a task dataset's Daytona snapshot count under cap).

## Required info (ask if missing)
1. **Dataset / tasks** — an HF parquet repo id to extract tasks from (e.g. `DCAgent/r2egym_sandboxes`),
   or an already-extracted local tasks dir.
2. **Teacher model + operating point** — determines the `datagen_config` (vllm-serve) YAML and the
   `trace_harbor_config` context-length YAML. Pick the cluster-matched config from `.claude/ops/<cluster>/`;
   don't change configs unless asked.
3. **Target HF repo** — `--trace_target_repo` (default `penfever/` or `DCAgent2/` org per the series).

## The 2-step flow

**Step 1 — extract tasks from the parquet** (writes one task dir per row):
```bash
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet <hf-parquet-repo-or-local> \
  --output_dir <tasks dir> \
  --on_exist overwrite
```

**Step 2 — submit the datagen job:**

> **🚧 SUBMIT FROM THE REPO DIR WITH `DCFT` SET — the sbatch WORKDIR guard hard-fails otherwise.**
> Before launching/relaunching, `cd <repo> && export DCFT=$PWD` (on Jupiter:
> `/e/scratch/jureap59/feuer1/OpenThoughts-Agent`). The generated `universal_tracegen.sbatch` /
> `universal_taskgen.sbatch` resolve `WORKDIR` from `DCFT_PRIVATE → DCFT → $PWD`; if you submit from
> `$HOME` or a scratch subdir with `DCFT` unset, the guard detects the wrong dir (missing
> `hpc/shell_utils/triton_cache.sh` marker) and **`exit 1`s immediately** with a `FATAL: WORKDIR=... is
> not the OpenThoughts-Agent repo root` message. This bug silently broke 3 relaunches (#217, stageC,
> datagen) before the guard existed. If you see that FATAL, `cd` to the repo, `export DCFT=$PWD`, resubmit.

```bash
python -m hpc.launch \
  --job_type datagen \
  --datagen_config <vllm-serve cfg>.yaml \
  --trace_harbor_config hpc/harbor_yaml/datagen/<ctx>.yaml \
  --tasks_input_path <tasks dir> \
  --trace_target_repo <hf repo> \
  --daytona_api_key "$DAYTONA_DATA_API_KEY" \
  --num_nodes 1 \
  --trace-n-concurrent N
```

## CRITICAL gotchas (get these wrong → silent total failure)

- **`--daytona_api_key "$DAYTONA_DATA_API_KEY"` is MANDATORY.** This must be the datagen-org Daytona key,
  NOT the default RL-org key. The RL-org key **rejects declarative sandbox builds**, so every trial
  instant-fails (the job looks "running" but produces zero real trajectories). Source the key from your
  secrets env (see CLAUDE.md "Datagen Daytona Key (CRITICAL)" / `.claude/ops/<cluster>/`). Do NOT echo the
  value. (The exact env-var name `DAYTONA_DATA_API_KEY` — verify against your local secrets env / ops doc.)
- **harbor_config needs `auto_snapshot: true`.** This attaches the prebuilt Daytona snapshot instead of
  doing a slow per-trial declarative build. Use a `hpc/harbor_yaml/datagen/<ctx>.yaml` that sets it.
- **Do NOT export `DAYTONA_TARGET`.** Leaving it set misroutes the run.
- **Per-job effective concurrency caps at ~100–128.** This is a per-job ceiling (not aggregate
  contention), so `--trace-n-concurrent` (and the config's `seqs` / `max_num_seqs`) above ~128 is wasted —
  it does not speed the job up. Parallel datagen *jobs* are fine; over-fanning a single job is not.
- **Daytona snapshot org cap is a HARD, server-side limit** (e.g. 40 or 60 per org depending on org).
  Never raise the cap or convert its `ValueError`/`SnapshotCapExceeded` into a warning. If a dataset
  overflows: clean **stale/MISSING** snapshots, shrink the dataset's snapshot footprint with
  **datagen-reduce-dataset-snapshots**, or reuse a dataset whose snapshots already exist (0 new). If still
  blocked, ask the user. Snapshots are keyed by sandbox **environment**, not dataset — they're shared, so
  do NOT reclaim them per-run. **Clean stale at the cap (autonomous):**
  `python scripts/daytona/daytona_snapshot_manager.py --api-key-env DAYTONA_DATA_API_KEY --delete-stale --yes --stale-days 2`
  (audit-only without `--delete-stale`; deletes only idle/unprotected `harbor__*` envs). Full procedure +
  caveats → `.claude/projects/daytona/daytona.md` § "How to clean stale snapshots".

## Chunking (long datasets)

For large task sets, split into chunked sub-jobs so each gets its own vLLM endpoint, walltime, and HF
`_chunk{N}` repo (avoids one giant job timing out and stranding traces, and keeps `trace_jobs/` inode use
bounded per job):
- `--chunk_size <S>` → ~`ceil(num_tasks / S)` chunks, each a separate job + serve + `_chunk{N}` HF repo.
- `--chunk_array_max <M>` → at most M chunks run concurrently (rolling `afterany` dependency chain).
- Alternatively, a manual `--dependency "afterany:<jobid>"` chain serializes individually-launched jobs.
- Route `trace_jobs/` off inode-tight scratch via `--experiments_dir <path>` where required by the
  cluster (see `.claude/ops/<cluster>/`); without it datagen can write thousands of per-trial dirs per
  chunk onto a quota-tight filesystem.
- Respect the cluster's QOS max walltime via `--time_limit` (see ops doc).
- The snapshot org cap still applies across chunks — see the HARD-cap gotcha above.

## Verify it's running

After submit, confirm the job placed and is producing **real** trajectories (a served `/v1/models`
healthcheck alone does NOT prove generation works):
- Check the job state (`squeue`/`sacct` per `.claude/ops/<cluster>/`).
- Tail the run's `_vllm.log` for a successful serve, then confirm trial dirs under
  `<run>/trace_jobs/<inner>/` are accumulating multi-step trajectories (avg turns > 1), not 1-turn
  exception stubs. The real-vs-failed check is detailed in **datagen-job-cleanup** step 2.

## MiniMax-M2.7 tracker workflow (the "datagen #N" queue)

The MiniMax-M2.7 131k series is driven by a **canonical tracker** that is the source of truth — follow its
**Launch Command** and **Process Notes EXACTLY**, not the ad-hoc command grab-bag. (Tracker path lives in
the `reference-minimax-datagen-tracker` memory note; verify the current path there.) Key conventions:
- One dataset in flight at a time (datasets strictly sequential).
- Per-dataset autonomous N→N+1 loop: extract → launch chunks → wait for ALL chunks to complete →
  consolidate the `_chunk{N}` repos into the final `<slug>-…-traces` repo (`scripts/datagen/join_hf_repos.py`)
  → verify (row count == sum of chunks, non-empty, realness: avg turns > 1, sane exception rate) → **DELETE
  the `_chunk{N}` repos** (only after the consolidated repo is verified) → clean local exp/task dirs
  (detached `rm`) → launch dataset N+1. Advance WITHOUT blocking (on chunk-completion / on the cron).
- **Oversized pre-check:** before extracting a row, query the HF dataset-viewer `/size` API
  (`https://datasets-server.huggingface.co/size?dataset=<repo>`); skip rows >~10k rows (each oversized
  extraction burns ~90–100k GPFS inodes).
- Reward-realness caveat: LLM-judge datasets can score all-0.0 if the judge API key is invalid/unplumbed —
  inspect the reward distribution at consolidation, not just avg turns.

## On completion
- Upload + verify traces and free disk with **datagen-job-cleanup** (handles the TIMEOUT-stranded-traces
  case, the one-level `trace_jobs/` nesting, the real-vs-failed sanity check, and safe disk cleanup).
- If a dataset's snapshot footprint blocks future launches, shrink it with
  **datagen-reduce-dataset-snapshots**.

## Guardrails
- NEVER launch with the wrong (RL-org) Daytona key — it silently zeroes the run.
- NEVER raise a Daytona snapshot org cap or downgrade its error to a warning.
- NEVER change experimental configs / hparams mid-series; flag and propose a separate run instead.
- Keep cluster-specific values (paths, env, configs, ssh) in `.claude/ops/<cluster>/`, not inlined here.

---

## Operating notes (folded from memory 2026-06-14)

- **Curator sharded datagen: pass `--save-every 700`** (5th positional arg), not 200 → ~4 checkpoints/12h-job instead of ~14, each checkpoint has a serial GPU-wasting pause.
- **FD-exhaustion / libuv SIGABRT at ~1h on boundary-hugging datasets:** `model_info.max_input_tokens` does NOTHING when `enable_summarize=false` (our RL/trace default) — nothing truncates the growing multi-turn prompt → `VLLMValidationError: 32769 input tokens` overflow. Lowering the token budget is INERT. The real levers: (a) drop `n_concurrent_trials` (e.g. 900→500), then (b) disable litellm's async logging worker on the hosted_vllm path (its `clear_queue exceeded max_time` backlog leaks FDs ∝ overflow_rate × concurrency). The HTTP-400 fix closed the retry-storm vector but not this one.
- Daytona per-job concurrency ceiling (~128) and snapshot caps live in `.claude/projects/daytona/daytona.md`.
