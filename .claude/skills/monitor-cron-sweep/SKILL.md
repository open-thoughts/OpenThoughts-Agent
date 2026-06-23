---
name: monitor-cron-sweep
description: >-
  Produce a comprehensive cross-cluster job-status update for a recurring N-hourly cluster sweep. Gather
  squeue/sacct on each cluster (validating against false-drain), bucket every active + recently-terminated
  job by type (RL / SFT / datagen / eval / catch-all), pull each type's signals, render them in the
  job_monitor_table.md formats, and flag completions (→ the matching cleanup skill), genuine failures
  (→ diagnose + agent_logs), and per-type health red-flags. Cluster-AGNOSTIC — ssh strings, code/log/exp
  paths, concurrency caps, gpu-mem ceilings live in `.claude/ops/<cluster>/`. Use for "run a cluster sweep",
  the N-hourly cron, or "give me a status update on all jobs".
---

# monitor-cron-sweep

Deploy this each cron sweep to produce ONE comprehensive update across all active clusters.

> **⚠ STEP 0 — READ `.claude/ops/<cluster>/ops.md` FIRST, every sweep, for each cluster you'll touch.**
> Not optional and not "only when something breaks." The ops doc carries the binding gotchas that
> repeatedly bite when skipped: the **GPFS `find`/`du` ban** (stat-walks stall the SSH session for minutes —
> locate logs via `scontrol show job <id> -o` `StdOut=`/`%Z` + depth-1 `ls`, never `find /e/scratch`), the
> **login01 fork-saturation false-drain** (re-check via login02/03/04), **inode allocations / cleanup-isn't-
> done-until-rm'd**, the **SIF/Ray-actor/NCCL debugging tooling** (ptrace blocked → faulthandler; the
> `opCount dead` false-positive), the **sig53/EDQUOT** traps, and shell idioms (`sacct -S now-Nhours`, simple
> single-string ssh). Reading it first prevents re-deriving — or re-violating — these mid-sweep.

> **⚠ Local clone = ground truth (CLAUDE.md §Always).** Any code/config fix this sweep performs — or
> dispatches a subagent to perform (reactive relaunches, cleanups, eval-grid fixes) — is edited in the local
> Mac checkout → commit → push → `git pull` on the cluster. **NEVER** hand-edit, `git commit`, or leave
> divergent/untracked changes on a cluster; no patch-by-rsync (vLLM excepted — built from source per-cluster).
> **Bake this rule into EVERY subagent prompt you dispatch** — the recurring Leonardo-clone divergence came from
> in-cluster edits/commits during reactive relaunches.

> **Formats** (the exact per-type tables + which metrics to pull + per-type red-flags) live in
> **`/Users/benjaminfeuer/Documents/notes/ot-agent/job_monitor_table.md`** — read it; this skill is the
> *process* that fills it. **Cluster particulars** (ssh invocation, code/experiments/log paths, the RL
> concurrency cap, gpu-mem ceiling, dotenv) live in **`.claude/ops/<cluster>/ops.md`** — read the ops for
> each cluster you sweep. No cluster-specific values are inlined here.

## 1. Gather (per cluster)
- `squeue -u <user> -t RUNNING` (running) + `sacct -u <user> -S <-Nh> -X` (terminal states since last sweep).
- **Validate squeue succeeded** — an empty result can be a **false-drain flake** (saturated login node /
  slurmctld timeout). Before trusting "drained", re-check via `sacct` and/or a different login node.
- **LIVENESS CHECK — `RUNNING` is NOT proof of progress (catches silent wedges).** A job can hold its
  allocation for hours while hung (engine deadlock, NCCL stall, generation-buffer wedge) — squeue still
  says RUNNING. For EVERY RUNNING job, `stat -c%y <StdOut>` and compare the log mtime to "now": if a job
  has emitted NOTHING for materially longer than its expected cadence (RL step / SFT log interval / eval
  trial — minutes, not hours), treat it as a **suspected silent hang** and investigate (tail the log, grep
  the ray-worker logs for EngineDead/NCCL-timeout/Watchdog/RPC-timeout around the last-output timestamp).
  A multi-hour-stale log on a multi-node job is a wedge burning nodes → diagnose + (with permission, since
  it's RUNNING) kill+relaunch. **Never report a RUNNING job as "healthy" without confirming its log is
  live** — this is how 914214 (stageC) sat wedged at buffer 35/64 for 6.5h on 14 nodes across multiple
  sweeps. Put the log-mtime (or "last output N min ago") in the table so staleness is visible at a glance.
- **INODE HEADROOM CHECK (each sweep) — `jutil project dataquota -p <project>` + `df -i`.** Inodes (file
  COUNT), not bytes, are the binding constraint; the shared `datasets` project on `/e/data1/datasets`
  (where `…/playground/ot-baf` lives) runs chronically near/over its soft limit. See the per-allocation
  limits + how-to-check in **`ops/jupiter/ops.md` → "Inode allocations" (`#inode-allocations`)**. If a
  project is at/over its inode soft limit, that's a sweep red-flag → trigger the cleanup-reclaim step (§4).
- ssh string + paths → `ops/<cluster>`.

## 2. Bucket every job by type
By job-name prefix / run-tag: `rl__*` → **RL**, `sft__*` → **SFT**, `datagen__*` → **Datagen**,
`eval-*` / eval run-tags → **Eval**, everything else (consolidate, pretokenize, hf_upload, SIF build,
DCP/CP/GPU-CI smoke, measurement/grid probes) → **Catch-all**.

## 3. Render per `job_monitor_table.md` (unify cross-cluster per type)
For each bucket, pull the type's signals and render its table. **Unify all clusters' runs of a type into
ONE table.** Extraction pointers:
- **RL** — Step (`.out` tqdm `Training Step Progress: N/M` or `trainer/global_step`) + reward/grad/entropy/
  TIS from the WANDB_MIRROR lines (chain-restart logs may have step but not the dict — scan the chain's logs).
  Apply the collapse-signal rule.
- **SFT** — Step + `{'loss','grad_norm'}` from the `.out` (NOT trainer_log.jsonl); total steps from the config/banner.
- **Datagen** — chunks done/total (squeue+sacct) + `result.json` count + avg_turns (realness gate: ≈1.0 = dead) + exc%.
- **Eval** — `result.json`/total + pass-rate + top exception + the 4 infra checks (`eval-agentic-launch` §4 for greps).
- **Catch-all** — one line each: State / Elapsed / human note.
(The cleanup skills below carry deeper extraction snippets if you need them; don't reinvent.)

## 4. Flag + hand off (the value of the sweep)
- **Completion → the matching cleanup skill** (don't inline the whole checklist unless the sweep directive
  says to act): RL → **`rl-job-cleanup`**, SFT → **`sft-job-cleanup`** (upload + DB register), datagen (all
  chunks done) → **`datagen-job-cleanup`** (consolidate + advance the tracker), eval → **`eval-agentic-cleanup`**
  (only if auto-upload/register failed). For RL, recognize **resume-overshoot**: a clean COMPLETED at
  `max_steps` means done → cleanup; spurious past-max chain links should be cancelled.
  **CLEANUP IS NOT DONE UNTIL THE ARTIFACT DIR IS `rm`'d.** Uploading to HF then leaving the experiment's
  `trace_jobs/`/`tasks/`/already-pushed-`exports/` subtrees on `/e/data1/.../ot-baf` is the #1 inode leak
  (subagents habitually skip the delete → the `datasets` project blew past its inode soft limit). Every
  cleanup handoff (and every cleanup subagent prompt) MUST: confirm the artifact is on HF, then **delete the
  on-disk trees** (detached `rm` per the GPFS-delete discipline in `ops/jupiter/ops.md`), and **verify inode
  reclaim** (`df -i` / `jutil`). Limits + the offender list → `ops/jupiter/ops.md` (`#inode-allocations`).
- **HF-only / non-agentic SFT (Delphi #6279 + any `enable_db_registration: false` series) — "move the chains" (3 legs, autonomous, every sweep, no asking):**
  1. **SFT completes → HF upload** via `sft-cleanup-hf-only` (NOT `sft-job-cleanup`; upload, **no DB**).
  2. **upload completes → `eval-standard-launch`** for the newly-uploaded cell(s).
  3. **eval completes → record scores** in the experiment tracker — Delphi midtrained-cell grid → `main_sft_evals/SCORES.md`; **base-model SFT grid (#6279 rows 2&4) → `base_sft_evals/grid.md`** (one row per base×recipe cell).
  Each sweep, advance whichever leg is pending (catch up backlog: completed-SFT-not-uploaded, uploaded-not-evaled, evaled-not-recorded). Idempotent — skip done legs. **This chain applies to BOTH the `main_sft_evals` (27 midtrained cells) AND the `base_sft_evals` (9 base × 2 recipes = 18 cells) series** — both are HF-only `enable_db_registration: false`.
- **Standalone eval-grid trackers (self-describing — harvest pending rows every sweep, no asking):** any tracker
  markdown that holds `⏳ pending` rows with a recorded `eval job` id — e.g.
  `experiments/active/delphi/rl-scaling-laws-6279/baseline_evals/grid.md` (Qwen3 dense-family baseline), `…/base_sft_evals/grid.md`
  (the 18-cell base-model SFT grid, #6279 rows 2&4), `…/pass_at_k_sft_evals/grid.md`, and `…/main_sft_evals/SCORES.md`. For each pending row: `sacct -j <jobid> --format=State` → on `COMPLETED`, harvest
  per the convention's §5.2 D/E (rsync the per-task `results_*.json` to the tracker's `<RUN>/` dir, **verify the
  JSON has numeric scores — a COMPLETED job can carry an empty `results:{}`**, extract MATH500/AIME24-mean±se/gsm8k,
  fill the row, flip to ✅). On a failure state, diagnose per §3.3 of `EVAL_CONVENTION.md` + log. The tracker file
  carries the jobids, so no run-state needs to live here — just read the grid each sweep and advance pending rows.
- **Chain-restart TIMEOUT** (12h/24h wall) with a successor RUNNING/PENDING → **normal, not a failure** —
  note the successor.
- **Genuine FAILED** (exit≠0, not a wall TIMEOUT) → diagnose (read the first real traceback, often masked by
  the elastic summary) + a dated **`agent_logs/`** entry; recurring identical failures ≠ transient.
- **RL collapse rule** (≥2 signals fire same step) → flag for cancel+salvage per `rl-job-cleanup`.
- **Eval** stall/zombie/instant-fail red-flags → act per `job_monitor_table.md` Eval section; **`DCAgent2/*`
  measurement runs are EXEMPT** (report as calibration, not production).

## 5. Respect the standing constraints (reference, don't relitigate)
- **RL concurrency cap per cluster** (value in `ops/<cluster>`); **a3 series CONCLUDED** — do NOT
  launch/refill a3 rows; **`enable_db_registration: false`** in YAMLs → DB registration is the manual
  cleanup step only; **Daytona snapshot org cap is HARD** — at the cap clean STALE snapshots, never raise
  it; **cross-user FK safety** before ANY Supabase delete/mutate (restrict to rows you own). Full policy
  set → the cron sweep directive + the cleanup skills.

## 6. Output + record
Post the bucketed tables (RL/SFT/datagen/eval/catch-all) + a short **"actions taken / flagged"** summary
(completions cleaned or handed off, failures diagnosed, health flags, anything launched). Maintain the
experiment log (`notes/claude/claude_experiments.md`) + the relevant tracker (a3 / MiniMax datagen / Delphi).
Skip clusters that are unreachable (note it) rather than blocking.

---

## Operating notes (folded from memory 2026-06-14)

- **Scope = Jupiter + Leonardo** (Perlmutter DROPPED 2026-06-05, not in use — do NOT ssh it until re-enabled). Each cron pass covers BOTH clusters in the same response.
- **Validate that `squeue` succeeded before trusting a 0-count.** A slurmctld timeout prints `slurm_load_jobs error: Socket timed out` / `error:` with NO job lines → a naive `squeue | grep -c <name>` reads 0 → false "drained". Treat an errored squeue as UNKNOWN (keep waiting). Prefer a positive done-signal: confirm terminal state via `sacct -j <ids> --format=State` (slurmdbd survives slurmctld outages). This is mandatory before any destructive datagen consolidate+delete — a false-drain once re-leaked secrets by re-uploading an unscrubbed chunk. (login01 fork-saturation is a second false-empty-squeue cause → re-check via login02/03/04, see `ops/jupiter/ops.md`.)
- **Spike-mitigation ablations OVERRIDE the cancel-on-collapse rule.** When a job_name contains `zclip`/`staleclip`/`stale_clip`/`z_clip`/`maxgn09_hint`/`shaped_entropy` or any spike-mitigation tag, do NOT autonomously scancel on the 2/4 collapse signals — observing whether the mechanism damps the spike IS the experiment. Still REPORT the signals prominently (mark "ablation observation, not actionable"). Standard runs (a3/a2/a1-base, no tag) DO follow the cancel+salvage rule. A real crash/NaN/SIGSEGV is still a genuine failure → diagnose.
