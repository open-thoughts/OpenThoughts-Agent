---
name: monitor-restore-iris-cron
description: Re-register the every-3-hours Iris job-monitor cron (status check + datagen auto-rescue/keep-2-in-flight) if it has been lost. Use at the start of a new session, after a restart, or when the user asks to restore/check the datagen-eval monitoring cron.
---

# monitor-restore-iris-cron

The recurring cron that watches all of `benjaminfeuer`'s Iris jobs is
**session-only** (it lives in the Claude session, not on disk) and recurring
crons **auto-expire after 7 days**. So it is routinely lost on a session
restart. This skill is the durable source of truth for re-creating it — the
cron prompt below is canonical; copy it verbatim into `CronCreate`.

## When to run
- Start of a new session where Iris jobs are in flight.
- The user says the monitor/cron is gone, down, or "not firing."
- After ~7 days (expiry).

## Steps
1. **Check if it already exists** — call `CronList`. If a recurring job whose
   prompt mentions "status check on ALL Iris jobs for user benjaminfeuer" is
   present, do nothing (don't create a duplicate — duplicate monitors cause
   redundant SQL/tunnel load). If a stale **datagen-only** variant exists
   (prompt mentions only `qwen3.5-122b-32k-%`), `CronDelete` it and recreate
   with the all-jobs prompt below.
2. **If absent, call `CronCreate`** with:
   - `cron`: `23 */3 * * *`  (every 3 h at :23 — off the :00/:30 marks)
   - `recurring`: `true`
   - `prompt`: the exact text in the fenced block below.
3. Tell the user the new job id, and the two caveats: **session-only** (dies
   when this Claude session exits — re-run this skill next session) and
   **7-day auto-expiry**.

## Notes
- `durable: true` is NOT honored in this harness (it still creates a
  session-only job), so don't rely on it — this skill IS the persistence layer.
- The cron only fires while the REPL is idle (not mid-task). If it reliably
  misses, the fallback is the user pasting the prompt manually, or an external
  launchd monitor (out of scope here).
- It tracks ALL `/benjaminfeuer/%` jobs but the autonomous write actions
  (auto-rescue, keep-2-in-flight) are **datagen-only**; eval jobs are
  monitor-only (they self-sync to Supabase+HF). See the `datagen-launch-iris` and
  `eval-agentic-launch-iris` skills.

## Canonical cron prompt (copy verbatim into CronCreate)

```
Every-3-hours status check on ALL Iris jobs for user benjaminfeuer (datagen + eval + anything else).

1. Active jobs (ALL job types):
   /Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin query "SELECT job_id, state FROM jobs WHERE state IN (1,2,3) AND job_id LIKE '/benjaminfeuer/%' ORDER BY job_id DESC LIMIT 20" -f csv
   Also query state IN (4,5,6) LIMIT 8 to catch jobs that went terminal since the last tick.

2. For each ACTIVE job, run the analyzer (job-type-agnostic):
   /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/iris/analyze_job_history.py <job_id> --output /tmp/$(basename <job_id>)_history.md --refresh
   Report from the .json sidecar: runtime_h, iris_preemption_count, cycles total/served, samples (serving_summary.gen_tps.n), gen tok/s mean/peak, Running mean/peak, non_empty/total trials = rate, t_first_serve, top harbor_exception_stats.

3. Print `## Iris jobs status — <ISO UTC>`: one line per job (name + state + CLOSED/PARTIAL/OPEN/DEAD), a compact metrics block, and a survival check (past cold compile? throughput sane? traces/results landing on HF?). Classify each job by job_id prefix and apply the right treatment:

   A. **Datagen** (`qwen3.5-122b-32k-%`): S1 baseline gen tok/s mean~400/peak~1115 (short-task datasets run lower — judge by productive trial rate). HF repo `penfever/<slug>-qwen3.5-122b-32k-traces`. Image `ae085bc8`+ auto-uploads on state-4 success — verify the repo self-created before rescuing. Watch the OOM-fix (heavy datasets) and stuck-PENDING (unpinned relaunch via --gcs-output-dir gs://marin-models-us/ot-agent). **The standing actions in §4-5 below apply to datagen jobs ONLY.**

   B. **Eval** (`eval-%`): launched per the eval-agentic-launch-iris skill. These **auto-sync to Supabase + HF on completion** (`--upload_to_database`), build sandboxes at runtime (force_build:true, MAIN Daytona org) and need **NO rescue and NO keep-2-in-flight** — they are one-off. **ALWAYS report the leading metric (running accuracy / mean reward) for each in-flight eval** — pull the `<done>/<total> Mean: <X>` figure from the harbor progress line in `/Users/benjaminfeuer/Documents/marin/.venv/bin/iris --cluster=marin job logs <job_id>` (the analyzer sidecar does NOT capture it). Also report productive trial rate + harness exceptions, and on a terminal job whether results landed (Supabase/HF). Do NOT auto-relaunch eval jobs.

   C. **Other** job types: report state + a one-line health read; take no autonomous write action.

STANDING ACTIONS — DATAGEN JOBS ONLY (override read-only; see memories auto_rescue_banked_trials, datagen_keep_two_in_flight):
4. AUTO-RESCUE (datagen only): if a datagen job is terminal (4/5/6) with productive trials banked in GCS that did NOT auto-upload (HF repo missing/stale), rescue automatically — no need to ask: rsync gs://marin-models-{us,eu}/ot-agent/<job>/<job>/ → /tmp/<job>_traces, then /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python /Users/benjaminfeuer/Documents/OpenThoughts-Agent/scripts/harbor/make_and_upload_trace_dataset.py --job_dir /tmp/<job>_traces --repo_id penfever/<slug>-qwen3.5-122b-32k-traces --episodes last --filter none --skip_register (source /Users/benjaminfeuer/Documents/secrets.env first). Report final row count + update the tracker.
5. KEEP TWO DATAGEN IN-FLIGHT: if the count of active DATAGEN jobs (`qwen3.5-122b-32k-%`, state 1/2/3) is < 2, auto-launch the next `pending` dataset from /Users/benjaminfeuer/Documents/experiments/active/datagen/qwen3.5-122b-tt/tracker.md using the datagen launch template (S1 config, ctx32k_verified.yaml, --tpu v5p-8 --preemptible, --gcs-output-dir gs://marin-models-us/ot-agent [unpinned], repo penfever/<slug>-qwen3.5-122b-32k-traces). Flip its tracker row to RUNNING. If a launch hits Daytona SnapshotCapExceeded, delete only MISSING harbor__ snapshots (memory daytona_snapshot_cap) then retry. (Eval jobs do NOT count toward this 2 and are never auto-launched.)

6. NEVER kill/restart/bounce a RUNNING job or the cluster without express user permission. The only autonomous write actions are datagen rescue + datagen refill-launch. If a job is stuck PENDING (no capacity), report it and surface the unpinned-relaunch option — do not kill a running/placed job unprompted.
```

If you change the cadence or scope, update BOTH the `cron`/`prompt` above and
the live job (delete + recreate), so this skill stays the canonical copy.
