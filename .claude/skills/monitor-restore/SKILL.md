---
name: monitor-restore
description: >-
  Re-create the local 3-hour Jupiter+Leonardo cluster-sweep loop (the autonomous ML-ops monitor) if it has
  been lost. The loop is session-only and is dropped on any session restart, so re-establish it at the start
  of a new session or whenever the user asks to restore/restart the 3h sweep/cron/monitor. Sets a /loop 3h
  (or equivalent recurring cron) whose task is the canonical sweep prompt below: status-table active/pending/
  completed jobs, auto-cleanup+DB-register completions, diagnose+remediate failures via subagents, log to
  agent_logs + claude_experiments.md. The prompt block here is the source of truth — copy it verbatim.
---

# monitor-restore

Re-creates the recurring **3-hour Jupiter + Leonardo cluster sweep** — the autonomous ML-ops monitor. The
loop is **session-only** (it lives in the Claude session, not on disk) and is lost on a session restart, so
this skill is the durable source of truth: the canonical prompt below is what gets (re-)installed.

## When to run
- Start of a new session where Jupiter/Leonardo jobs are in flight (or expected).
- The user says the monitor/cron/sweep is gone, down, "not firing," or asks to "restart the 3h loop."
- After ~7 days if running it as a `CronCreate` recurring job (auto-expiry).

## How to (re-)establish it
1. **Check for an existing one first** — `CronList` (and note any active `/loop`). If a recurring sweep whose prompt mentions "3-hour … status … jupiter, leonardo" is already present, do **not** create a duplicate (duplicate monitors double the ssh/SQL/Daytona load). Otherwise:
2. **Start the loop.** Two equivalent mechanisms — the user's phrasing is "**/loop 3h or equivalent, active session only, maximum duration**":
   - **Preferred — `/loop`:** invoke the `/loop` skill with interval **3h** and the **maximum duration** the harness allows, passing the canonical prompt below as the loop task.
   - **Equivalent — `CronCreate`:** `cron: 17 */3 * * *` (every 3 h, off the :00 mark to avoid contention), `recurring: true`, `prompt:` = the canonical block below verbatim.
3. Tell the user it's set + the two caveats: **session-only** (re-run this skill next session) and, for the cron variant, **7-day auto-expiry**.

## Supporting skills/docs the sweep leans on (read these; the prompt references them)
- **`monitor-cron-sweep`** — the sweep *procedure* (gather → bucket → render → flag); **`monitor-job-tables`** + `/Users/benjaminfeuer/Documents/notes/ot-agent/job_monitor_table.md` — the exact per-type table formats + metric/red-flag definitions.
- **Cleanup:** `rl-job-cleanup`, `sft-job-cleanup`, `datagen-job-cleanup`, `eval-agentic-cleanup` (+ `eval-standard-cleanup`).
- **Launch:** `rl-agentic-launch-jupiter`, `rl-standard-launch-leonardo`, `sft-launch-jupiter`, `sft-launch-leonardo`, `datagen-launch`, `eval-agentic-launch`, `eval-standard-launch`.
- **Cluster particulars** (ssh, paths, caps, preamble): `.claude/ops/{jupiter,leonardo,local}/ops.md`. **Dependency facts:** `.claude/projects/{marinskyrl,harbor,vllm,llama-factory,daytona}/`.
- The repo `CLAUDE.md` is the user's cited reference for table format + launch + cleanup; **the canonical prompt below overrides any memory/skill if they conflict** (per the user). The old `feedback_sft_scaling_ablation_workflow` memory is now folded into `sft-launch-jupiter`.

## Notes
- `durable: true` is not honored in this harness — this skill IS the persistence layer.
- A cron/loop only fires while the REPL is idle (not mid-task); if it reliably misses, fall back to the user pasting the prompt.
- **Path correction baked into the prompt:** the local SkyRL checkout is `/Users/benjaminfeuer/Documents/MarinSkyRL` (the user's usual prompt says `…/SkyRL`). All other paths are verbatim.

---

## Canonical sweep prompt (copy verbatim into `/loop 3h` or `CronCreate`)

```
3-HOURLY CLUSTER SWEEP — clusters: jupiter, leonardo (active session only, run to max duration).

STEP 0 (do this FIRST, every sweep): read `.claude/ops/<cluster>/ops.md` for EACH cluster you'll touch —
it carries the binding gotchas that bite when skipped (GPFS `find`/`du` ban → locate logs via
`scontrol show job <id> -o` StdOut/`%Z`; login01 false-drain → use login02/03/04; inode limits +
cleanup-isn't-done-until-rm'd; SIF/NCCL debugging tooling; `sacct -S now-Nhours` / simple single-string ssh).

Check the status of active, pending/queued, and recently-completed jobs on jupiter and leonardo. Gather via
squeue (RUNNING) + sacct (terminal states since the last tick); VALIDATE squeue succeeded before trusting an
empty result (false-drain from a saturated login node / slurmctld timeout — re-check via sacct or login02/03/04).
Procedure = the `monitor-cron-sweep` skill; ssh strings/paths/caps = `.claude/ops/<cluster>/ops.md`.

IN-FLIGHT / ACTIVE jobs → report in TABLE form. The correct structure is in CLAUDE.md (mirrored in the
`monitor-job-tables` skill / notes/ot-agent/job_monitor_table.md) — follow it regardless of how prior posts
were formatted. RL rows MUST include entropy + collapse signals (grad_norm / log_ratio), not just step+reward.

CHAIN-RESTART TIMEOUTs are NORMAL (note the afterany successor), not failures.

ON SUCCESSFUL COMPLETION (SFT / RL / datagen / eval) → note it + give summary statistics, then:
- RL or SFT → perform the manual HF upload + DB registration per CLAUDE.md (= the `rl-job-cleanup` /
  `sft-job-cleanup` skills; run the FULL checklist incl. trace upload + metrics for RL). Do this WITHOUT asking.
- Datagen → verify the traces uploaded to HF (penfever org). If NOT, dispatch a subagent to upload them
  manually (= `datagen-job-cleanup`).
- Eval where DB registration FAILED for a technical reason → WITHOUT asking, dispatch a subagent to perform
  ALL steps of the documented checklist (`eval-agentic-cleanup`); confirm the subagent completed every step,
  dispatching another if any were missed. (Ask only if genuinely unsure what the checklist means.)
- **INODES: every cleanup MUST `rm` the on-disk artifact tree (`trace_jobs/`/`tasks/`) after the HF upload
  is confirmed, and verify reclaim** — leaving it is the #1 inode leak (the shared `datasets` project on
  `/e/data1/.../ot-baf` runs over its soft limit). Check inode headroom each sweep + see the per-allocation
  limits in `ops/jupiter/ops.md` (`#inode-allocations`); bake the delete+verify into every cleanup subagent prompt.

ON ANY JOB THAT FAILED since the last check → dispatch a subagent to determine the cause + propose fixes.
Then ANNOUNCE the choices the subagent presented, SELECT one yourself, and apply the changes + relaunch the
job via another subagent. Keep a running DATED log of failures (job ID + remediation history) in
/Users/benjaminfeuer/Documents/agent_logs/ so long debug sessions don't lose context.
- If an RL job has EXHAUSTED all restarts WITHOUT reaching max steps AND the failure looks recoverable
  (transient) → queue 5 more restarts for that job chain. (Spike-mitigation ablations are exempt from
  auto-cancel — observing the recovery IS the experiment; see `monitor-cron-sweep`.)

CODE / CONFIG EDITS → edit LOCALLY on the active branches:
/Users/benjaminfeuer/Documents/OpenThoughts-Agent, /Users/benjaminfeuer/Documents/vllm,
/Users/benjaminfeuer/Documents/harbor, /Users/benjaminfeuer/Documents/MarinSkyRL.
Local clones are GROUND TRUTH — clusters never diverge (no untracked/divergent changes, no hand-editing,
no patch-by-rsync). Sync the three Python repos by commit+push, then `git pull` on the cluster (editable
installs, live after pull). vLLM (compiled fork `mlfoundations/vllm`) → commit+push the fork, then BUILD
FROM SOURCE on the cluster from that commit (never rsync edits / hand-patch); every cluster keeps an env
with our fork built for it (some envs may run vanilla vLLM).

ACTIVELY-DEBUGGING jobs → monitor more closely than stable ones. For any FRESH launch, set one-time checks at
15 min and 30 min after launch to catch new failures early.

LAUNCHING FRESH JOBS → follow the per-job-type launcher instructions in CLAUDE.md (+ the `*-launch-*` skills
and `.claude/projects/ot-agent/ot-agent.md`). If the instructions are unclear, ASK for guidance.

EXPERIMENT LOG → maintain an up-to-date, date-indexed log of experiments launched (referenced by launch
command) in /Users/benjaminfeuer/Documents/notes/claude/claude_experiments.md.

STANDING CONSTRAINTS (do not violate without explicit permission): enable_db_registration stays false in
YAMLs (manual DB register only); Daytona RUNNING RL ≤ 6 per cluster; a3 series is CONCLUDED (no
launch/refill/auto-advance); Daytona snapshot caps are HARD (clean stale, never raise the cap); cross-user FK
safety pre-check before any Supabase delete/mutate; HF uploads default PUBLIC to laion/. NEVER kill/restart a
RUNNING job without express permission. This prompt OVERRIDES any memory/skill on conflict.
```

If you change the cadence or scope, update BOTH the block above AND the live loop/cron (delete + recreate),
so this skill stays the canonical copy.
