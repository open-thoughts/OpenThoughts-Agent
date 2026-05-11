# Resume Handoff — Lessons & Fixes from Jupiter

This doc transfers hard-won lessons about the eval-listener resume path from
Jupiter to other clusters running the same stack (Leonardo, Perlmutter, …).
If you operate a continuous-eval-listener workload elsewhere, read this
before you trust `harbor jobs resume` in production.

The TL;DR: out-of-the-box, `harbor jobs resume` on this stack **re-runs the
entire dataset, not just unfinished trials**, and silently retries trials it
should treat as final. Four code fixes (one of them upstream) plus an
operational protocol turn it into a real resume.

---

## Architecture recap

- vLLM port is derived from SLURM JID: `VLLM_PORT = 10000 + (SLURM_JOB_ID % 50000)`.
  Every fresh fire / resume gets a new port.
- Harbor stores agent configs in `<run_dir>/config.json` (job-level) **and**
  one per trial in `<run_dir>/<task>__<id>/config.json`.
- On resume, harbor regenerates the job-level config from CLI args, then
  compares it against the on-disk per-trial configs via `TrialConfig.__eq__`.
  Trials whose stored config matches the new one are skipped. Mismatches get
  the whole trial dir rmtree'd and re-run.
- DB row in `sandbox_jobs` is found via `get_sandbox_job_by_name(run_tag)`
  and flipped Pending → Started → Finished by the sbatch + upload script.

These three facts produce four landmines.

---

## Bug G10 — `api_base` port mismatch (FIXED, in-repo sbatch)

**Symptom**: every trial fails with `Cannot connect to host localhost:NNNNN`
where NNNNN is the *original* fire's vLLM port. All retries logged as
`LiteLLMInternalServerError`. Dir ends up worse than before resume.

**Cause**: `config.json::agents[0].kwargs.api_base` is frozen at the original
fire's port. The new fire's vLLM listens on a *different* port.

**Fix** (already in `eval/unified_eval_harbor.sbatch:829-834`): sed-rewrite
the api_base in `<run_dir>/config.json` to the current `${VLLM_PORT}`
immediately before `harbor jobs resume`. Backup written to `config.json.bak-port`.

```bash
sed -i.bak-port -E \
  "s|localhost:[0-9]+|localhost:${VLLM_PORT}|g" \
  "${RUN_DIR}/config.json"
```

**On any other cluster**: verify this sed runs before `harbor jobs resume` in
your sbatch. Without it, resume cannot work at all.

---

## Bug G12 — resume re-runs every trial (FIXED, in-repo sbatch)

**Symptom**: harbor resume reports `Existing trial X not found in generated
configs; skipping` for hundreds of trials, then schedules the entire dataset
anyway. Trial directory count *doubles* (e.g. 305 old + 267 new = 572 dirs).
Compute wasted.

**Cause**: G10's sed patch only fixes the *job-level* `config.json`. Each
per-trial `<task>__<id>/config.json` still has the stale port. When harbor
resume compares per-trial configs to the regenerated one, `TrialConfig.__eq__`
fails on every trial → trial is treated as "from a different config" and
re-scheduled.

**Fix** (commit a762aea7 on this repo): extend the sed patch to walk every
per-trial `config.json` as well:

```bash
# Job-level (G10):
sed -i.bak-port -E "s|localhost:[0-9]+|localhost:${VLLM_PORT}|g" \
  "${RUN_DIR}/config.json"

# Per-trial (G12):
find "${RUN_DIR}" -mindepth 2 -maxdepth 2 -name config.json -print0 \
  | xargs -0 -I{} sed -i.bak-port -E \
      "s|localhost:[0-9]+|localhost:${VLLM_PORT}|g" "{}"
echo "[resume] Patching api_base port in $N per-trial config.json files"
```

**Verification after a resume fires** (do this every time):
```bash
grep -c "not found in generated configs; skipping" eval/logs/data_<NEW_JID>.out
# expect: 0
grep "Patching api_base port in [0-9]+ per-trial" eval/logs/data_<NEW_JID>.out
# expect: count matches the dataset size
grep -c "^Starting trial" eval/logs/data_<NEW_JID>.out
# expect: matches the *in-flight* count from inspector, NOT the dataset total
```

If `count == dataset total`, G12 has resurfaced.

---

## Bug G13 — DB row stays Pending after resume (FIXED, in-repo utils)

**Symptom**: resume runs to completion but the `sandbox_jobs` row never
flips to Started. Stays Pending forever. Upload script then can't find the
right row, or worse, finds a stale older row from a prior fire.

**Cause**: `get_sandbox_job_by_name` returned an arbitrary row when multiple
exist (e.g. an old Pending from a prior fire + the current one). Listener
saw the old already-Started row, short-circuited as `already_started`, and
never flipped the current row.

**Fix** (commit a762aea7 on this repo): add deterministic ordering to
`database/unified_db/utils.py::get_sandbox_job_by_name`:

```python
.select("*").eq("job_name", name) \
    .order("submitted_at", desc=True).limit(1).execute()
```

**Verification**:
```python
from database.unified_db.utils import get_sandbox_job_by_name
print(get_sandbox_job_by_name("<run_tag>")["job_status"])
# expect: "Started" within 60s of sbatch start, "Finished" on completion
```

---

## Harbor Bug #1 — `--retry-exclude` silently overrides YAML (FIXED locally, filed upstream as #1617)

**Symptom**: 9-10 hour tail trials, log filled with
`ContextLengthExceededError. Retrying (1/10)`, `(2/10)`, … This exception
type is in your harbor YAML's `exclude_exceptions`, but harbor retries it
anyway up to `max_retries=10`.

**Cause**: `harbor jobs start --retry-exclude` CLI flag default was a
non-empty list, which silently *replaced* the YAML's `exclude_exceptions`
instead of merging. The YAML loses.

**Fix** (in `harbor-fix/src/harbor/cli/jobs.py:304`): default the CLI flag
to `None` so the YAML wins when no override is passed.

**Upstream**: filed as harbor #1617. Watch passively for merge; until then,
every cluster needs the local harbor-fix patch.

---

## Harbor Bug #2 — proactive ContextLengthExceeded wrap (FIXED locally, BEN-ONLY)

**Symptom**: even after the #1 fix, you still see retry loops, but the
cause is now `OutputLengthExceededError` (from the SubagentSummarizer Step 3
"Answer providing" path) being wrapped as `ContextLengthExceededError`.

**Cause**: `_check_context_length` in `terminus_2.py` proactively wraps
non-timeout summarization failures as `ContextLengthExceededError` *before*
the retry path sees them. Even with exclude_exceptions configured correctly,
the wrong type slips through.

**Important caveat**: this code path is **Ben-only divergence** on
`penfever/temp-override`. It is **NOT on harbor main**. If your cluster
uses harbor main, you don't have this bug. If your cluster uses
`harbor-fix` (the penfever branch), you need the patch.

**Fix** (in `harbor-fix/src/harbor/agents/terminus_2/terminus_2.py:953`):
wrap as `SummarizationTimeoutError` (which IS in the default exclude list)
instead of `ContextLengthExceededError`.

**Verification post-patch**:
```bash
grep -c "ContextLengthExceededError. Retrying" eval/logs/data_<JID>.out
# expect: 0
grep -c "SummarizationTimeoutError .* not retrying" eval/logs/data_<JID>.out
# expect: > 0 if any Step 3 failures occurred
```

---

## Operational patterns

### Pre-walltime cancel (zombie Daytona avoidance)

Daytona sandboxes run remotely in Daytona's cloud, **not on your compute
node**. When SLURM SIGTERM fires at walltime, harbor on the compute node
dies but the sandboxes keep running and **keep writing back to the run dir
on shared FS for 30-60 minutes** after sbatch death.

Observed on Jupiter: a 267-task dir went from 267 trial_dirs to 352 after
walltime, `n_completed` went 267 → 305 (overshooting `n_total`).

This breaks the resume scanner — `n_completed > n_total` doesn't match any
classification branch, dir gets silently skipped.

**Mitigation**: cancel cleanly *before* walltime hits. The rule of thumb
that worked on Jupiter (12h cap):

```bash
# At 11h45m elapsed (TIME_LEFT ≤ 15min), scancel proactively.
squeue -u $USER --format='%.10i %.10M %.10L' --noheader \
  | awk '$3 ~ /^0:[0-3][0-9]:/ { print $1 }' | xargs -r scancel
```

After a clean scancel, wait ~60s, then run inspector.
After a walltime-killed dir, wait ~60min for zombies to finish writing.

### Resume of "stuck-at-full DONE" dirs

A walltime-killed dir at full trial coverage (267/267) but with
`finished_at=None` classifies as DONE in the inspector and gets hidden from
`--needs-resume-only`. The inspector logic is "DONE means no work needed",
but in this case the run dir really *does* need a resume — to write
`finished_at` and to fill in 4-10 missing reward.txt files from interrupted
trials.

**Override**: pass `--resume-error-threshold -1` to the listener. This
promotes infra_errors=0 dirs from DONE → DONE_WITH_ERRORS, making them
eligible for resume.

```bash
python eval/resume_chunked.py \
  --priority-file /tmp/priority.txt \
  --preset tb2 --org eval \
  --jobs-dir $JOBS_DIR \
  --tag-prefix r_$(date -u +%H%M%S) \
  --tp-size 2 --dp-size 2 --timeout-multiplier 16.0 \
  --conda-env <your_env> \
  --resume-error-threshold -1 \
  --chunk-size N --sleep-between 0
```

### n_fires cap (24h hard limit)

Default `--max-total-fires=2`. Original fire + 1 resume = 2 fires. After
that, dir is `AT_RESUME_LIMIT` and should be uploaded as-is, not resumed
again. Daytona auth tokens have a 24h validity window in this org config;
beyond that the resumes start failing in interesting ways anyway.

If a dir genuinely needs a third fire (rare), use `--max-total-fires 3
--max-resume-count 3` with the listener directly.

---

## Verification checklist (run after every resume fire)

For each new JID:

```bash
# 1. squeue confirms RUNNING within 30s
squeue -j <new_jid> --format='%.10i %.10T'

# 2. G12 verification — per-trial sed patch ran
grep "Patching api_base port in [0-9]+ per-trial" eval/logs/data_<new_jid>.out

# 3. G12 verification — zero "skipping" warnings
grep -c "not found in generated configs; skipping" eval/logs/data_<new_jid>.out
# expect: 0

# 4. Harbor scheduled only the in-flight trials, not the whole dataset
grep -c "^Starting trial" eval/logs/data_<new_jid>.out
# expect: matches in_flight count from inspector pre-resume

# 5. G13 verification — DB row flipped to Started
python -c "from database.unified_db.utils import get_sandbox_job_by_name; \
  print(get_sandbox_job_by_name('<run_tag>')['job_status'])"
# expect: "Started"
```

---

## Files of interest

| Path | Purpose |
|---|---|
| `eval/check_resume_needed.py` | Inspector — classifier, reject mechanism |
| `eval/resume_chunked.py` | Orchestrator — chunked fires per (preset, org) |
| `eval/unified_eval_harbor.sbatch:828-859` | G10/G12 sed patch + harbor jobs resume call |
| `eval/unified_eval_listener.py:910-913` | DONE classification (the line `--resume-error-threshold -1` overrides) |
| `database/unified_db/utils.py::get_sandbox_job_by_name` | G13 ORDER BY fix |
| `harbor-fix/src/harbor/cli/jobs.py:304` | Harbor Bug #1 patch (default None) |
| `harbor-fix/src/harbor/agents/terminus_2/terminus_2.py:953` | Harbor Bug #2 patch (Ben-only) |

## Open issues

- **harbor #1617**: filed upstream, awaiting maintainer response.
- **SubagentSummarizer Step 3 OOM dominance**: empirical disk audit on
  Jupiter showed Step 3 (Answer providing) fails ~118× vs Step 2's 40×
  across all run dirs. Mitigation candidate: cap question-subagent
  response length at `summarizers.py:139-142`. Not patched as of this
  handoff — track tail latency post-patches before investing in this.
- **Inspector misclassifies stuck-at-full as DONE**: known limitation,
  use `--resume-error-threshold -1` workaround. Long-term fix would be a
  new classification branch like `DONE_NEEDS_FINALIZE`.

---

*Originally captured during the Jupiter eval-ops 3-agent split (firing /
resume / upload), 2026-05-06 → 2026-05-11. Apply with cluster-specific
adjustments (env paths, conda env name, jobs dir location).*
