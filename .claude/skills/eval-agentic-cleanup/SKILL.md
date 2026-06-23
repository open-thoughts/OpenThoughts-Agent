---
name: eval-agentic-cleanup
description: >-
  Audit + recover a finished agentic eval. ALWAYS start with the read-only, idempotent completeness/health
  audit (§0): job finished? score present + non-zero + not obviously broken? HF traces present + linked?
  trial count ≈ n_rep × benchmark_size? — it writes nothing and recommends an action per check. Then run
  only the flagged remediations: manual HF-trace upload + Supabase DB-registration via manual_db_eval_push.py,
  the vLLM-numeric-ID → real-HF-model-name fix (with cross-user FK safety), verify, free disk. Use to verify
  an eval is truly complete, when a sweep finds an eval that didn't upload/register or has a broken/zero
  score or short trial count, or to re-register/correct an eval's model/traces. Distinct from the model-
  publishing cleanups (rl-job-cleanup / sft-job-cleanup) and datagen-job-cleanup — this is the EVAL path.
---

# eval-agentic-cleanup

Normal evals auto-upload their traces + auto-register in Supabase on completion — but "the score is in the
leaderboard" does NOT mean the eval is complete. **Always start with the read-only audit (§0)** to verify
the four things that actually matter, then run only the remediations it flags. The auto-harvest path is
known to land a score while silently skipping trace upload, or to record a bogus 0 from an eval that never
reached the model — the audit catches both.

## 0. Read-only completeness + health AUDIT — ALWAYS run first (idempotent, ZERO writes)
Safe to run any time, repeatedly — it only reads (sacct, run-dir `result.json`, HF API, Supabase **reads**).
It does NOT upload, register, mutate, or delete. For each eval `(slurm_job_id, model, benchmark)` check the
four conditions and emit a per-eval verdict + recommended action; only then run the flagged remediation(s).

**Pre-gate — is it EXEMPT?** grid / throughput / OOM-test **measurement** runs targeting `DCAgent2/*` →
audit not needed (upload only production winners). Skip.

| # | Check | How (read-only) | PASS | If it FAILS → recommend |
|---|---|---|---|---|
| 1 | **Job finished** | `sacct -j <id> -X -o State` | `COMPLETED` | `RUNNING`/`PENDING` → wait & re-audit later; `FAILED`/`TIMEOUT`/`CANCELLED` → diagnose + relaunch (see `eval-agentic-launch` gotchas #1–#5: `jobs_dir` perm / `hosted_vllm` 2-slash / `model_info` / reservation / stale-row) |
| 2 | **Score present & not obviously broken** | Supabase `sandbox_jobs` score (read), or run-dir `result.json` → `stats.evals.<k>.metrics[0].mean` | non-null real number; a `0` is OK *only* if check 4 shows trials actually ran (POSTs flowed) | null / missing / **`0` with ~0 trials or ~0 vLLM POSTs** = infra-broken (the eval never reached the model — classic `jobs_dir`/`hosted_vllm`/`model_info` failure) → **re-run**, don't trust the 0. **Also suspect INFLATION:** see the `VerificationNotCompletedError` note in check 4 — a high rate there means the score is biased UP. |

> **⚠️ `VerificationNotCompletedError` (no-reward trials) — the bias depends on WHICH score path you read (verified empirically 2026-06-17):**
> - **harbor's internal `JobStats` mean** (`stats.evals.*.metrics[0].mean_reward`; `JobStats.increment` only counts reward-bearing trials in `n_trials`) **DROPS no-reward trials from the denominator → biased UP** (looks better than reality).
> - **the DB/leaderboard auto-harvest path**, however, was observed to register with the **full /300 denominator** (masked trials counted as **0**) — so the leaderboard number was NOT inflated in the 2026-06-17 cohort. Don't assume the leaderboard is up-biased; check whether the registered `n_trials`/denominator is the full count or the reward-bearing count for the path that produced the row.
> - **Either way, resuming is worth it** — but the payoff is usually **recovering real passes hidden as 0s**, not "fixing inflation": the 2026-06-17 VNC-resume of 5 swebench evals re-ran 198 relabeled trials, found real passes (e.g. symclip 13/46), and corrected accuracy **ROSE** in all five (lrboost .300→.317, symclip .270→.313, stageB .307→.320, stageD .300→.327, swesmith .213→.223), VNC-residual 0. So: a non-trivial VNC rate means the score is **untrustworthy** (could be inflated in harbor-mean terms, or just hiding recoverable passes) → resume on a healthy sandbox.
> This trial state (harbor 9203989f) = the verifier never produced a result (e.g. `AddTestsDirError` on a sandbox left degraded by a 3000 s agent timeout); reward is absent. **Pre-9203989f these were mislabeled `AgentTimeoutError`** and conflated with the legitimately-scored-after-timeout trials (which DO have a reward); discriminate by **reward presence**, not the label.
>
> **The bias direction depends on WHY the trial is missing — handle no-reward classes differently:**
> - **`AgentTimeoutError`/`VerificationNotCompletedError` = *informative* missingness (MNAR).** Missing *because* the agent burned its full budget without solving → true score ≈ 0 with high probability (SWE-bench is ~pass/fail; a timeout rarely hides a passing patch). Dropping these removes near-certain 0s → **directional up-bias.** Correct repair: **impute 0 *or* re-run** (both land ≈0).
> - **Infra errors (`DaytonaAuthenticationError`/`DaytonaValidationError`/etc.) = ~*ignorable* missingness (MAR/MCAR).** The failure fires at a point uncorrelated with task solvability → the dropped trials' true scores are ~population-distributed → dropping them is **approximately unbiased** (mostly just lowers N). Repair: **re-run for completeness only — do NOT impute these as 0** (that would *introduce* a down-bias, the opposite mistake). Lower priority than the MNAR class.
| 3 | **HF traces present + linked** | trace HF repo exists + non-empty (`hf` API 200) AND the eval's DB/LB row has the trace/url field set | repo 200 **and** linked | repo missing/empty OR unlinked → **upload traces + register the link** (§1–§3) |
| 4 | **VALID trial count (PARSE, don't file-count)** | **Parse** each per-trial `result.json` for a numeric reward — do NOT `ls`/`find | wc -l` (an errored trial STILL writes a `result.json`, just with `exception_info` and no reward, so a file count overstates coverage). Count VALID vs ERRORED per eval — see the snippet below. | VALID ≥ ~90% of `n_rep_eval × benchmark_size` | materially short on VALID (<~90%, a whole rep missing, OR a high hard-error rate like 15–25%) → **resume the errored trials** by re-submitting the SAME eval (same run-tag): the Jupiter sbatch auto-calls `harbor jobs resume --filter-error-type …` (keeps valid trials, re-runs the filtered error classes). First check the error TYPES (aggregate `result.json` → `exception_stats`); if they're outside the Jupiter sbatch's filter list (`EnvironmentStartTimeoutError`/`DaytonaError`/`DaytonaRateLimitError`/**`VerificationNotCompletedError`**), widen it — see `.claude/projects/harbor/harbor.md` "Resume". Expected: swebench-verified-random-100=100, dev_set_v2=100, terminal_bench_2=89; × `n_rep_eval` (default 3) → ~300/~300/~267. **`AgentTimeoutError` WITH a reward is a passthrough VALID 0 (verifier scored it) — don't filter/re-run those.** But an `AgentTimeoutError` (or, post-9203989f, `VerificationNotCompletedError`) **with NO reward** is the verifier-never-completed state from the check-2 note — it **biases accuracy UP**, so it IS retriable: resume it. Discriminate by reward presence, not the bare label. |

Output one row per eval: `✅/⚠️` per check + the single recommended next action. **Re-run the audit after any
remediation** to confirm it flipped to ✅ (that's what makes the whole skill idempotent — the audit is the
source of truth, the remediations below are the only side-effecting parts and you run them deliberately).

### Check 4 — counting VALID trials (parse, never file-count) — STANDARD report element
Trial dirs live at `eval_jobs/eval-<safe_model>_<safe_dataset>/<task>__<id>/result.json` (depth-1 under the
run dir; the run-dir-root `result.json` is the aggregate — exclude it via the `*/` glob). Each per-trial
`result.json` has `verifier_result.rewards.reward` (the metric) and `exception_info` (set on error). A trial
is **VALID** iff `reward` is a finite number; otherwise **ERRORED** (no reward — Daytona/infra/non-passthrough
exception). One pass over all evals (run from the `otagent` env python, read-only, ~1 min for 21×~300):
```python
import glob, json
EJ='/e/data1/datasets/playground/ot-baf/eval_jobs'
EVALS=[('eval-laion_<model>_<safe_dataset>', 300), ...]  # (run-tag, n_rep*bench_size) per eval
for tag, exp in EVALS:
    valid=err=0
    for f in glob.glob(f'{EJ}/{tag}/*/result.json'):          # */ = per-trial only, skips root aggregate
        try: d=json.load(open(f))
        except Exception: err+=1; continue
        r=((d.get('verifier_result') or {}).get('rewards') or {}).get('reward')
        if isinstance(r,(int,float)): valid+=1   # numeric reward = VALID (incl. AgentTimeout passthrough)
        else: err+=1                             # no reward = ERRORED (hard infra/exception)
    print(f'{tag}: valid={valid} err={err} total={valid+err} / exp {exp}')
```
**Always include the valid/errored/expected matrix in the agent report** — a bare file count is NOT
acceptable evidence of completeness (it hides errored trials). Flag any eval whose ERRORED rate is high
(≳10–15%) as a re-run candidate + note the score may be deflated if those errors scored 0.

---
# Remediations — the §0 audit scopes which to run (idempotent; re-run §0 after to confirm ✅)
The §0 audit is read-only; the steps below **do write** — and that's expected: **HF trace upload + Supabase
registration are normal, sanctioned operations of this skill**, not something to avoid. The audit just tells
you *which* are needed so you don't redo finished work or clobber a good row. (Check-4 "re-run/resume missing
trials" is a relaunch — see `eval-agentic-launch` — not one of the steps below.)

## 1. Manual upload + DB register — `manual_db_eval_push.py`
Pass the **`trace_jobs/<RUN_TAG>`** path (where Harbor writes the `<task>__<id>` trial dirs), **NOT**
`eval_jobs/<RUN_TAG>` (that only has `meta.env`). The script auto-resolves nested trial subdirs and
auto-detects agent/model/benchmark from job metadata.
```bash
source ~/secrets.env   # needs SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN
cd <OpenThoughts-Agent>
python scripts/database/manual_db_eval_push.py --job-dir trace_jobs/<RUN_TAG> --verbose
#   --hf-repo DCAgent2/<RUN_TAG>-traces   # explicit HF repo
#   --skip-hf                             # DB only (traces already uploaded)
#   --forced-update                       # overwrite existing records
```

### HF trace dataset — use the MEMORY-EFFICIENT uploader (don't hand-extract episodes)
For the HF trace **dataset** itself, use the same streamed, **last-episode-per-trial** uploader the RL
cleanup uses (see `rl-job-cleanup` §8 for the canonical invocation) — naive per-conversation extraction
loads every episode of every trial into RAM and is brutally I/O-heavy on GPFS at eval scale (300 trials ×
hundreds of episode files each):
```bash
# otagent env; ALWAYS in tmux; `hf upload`, NEVER `hf upload-large-folder` (deprecated + LFS-429 deadlocks)
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job-dir trace_jobs/<RUN_TAG> --repo_id <org>/<RUN_TAG>-traces --episodes last
```
`--episodes last` keeps only the final (scored) episode per trial — the same convention as the RL trace
datasets, and what keeps memory flat. **Eval-vs-RL difference:** RL passes `--skip_register`; for EVALS you
DO want the trace repo linked, so register/link it onto the eval's existing `sandbox_jobs` row (§2/§3 — set
the trace/url field only, do NOT create a second row or touch the score). On **Leonardo**, login-node
`hf upload` is SIGKILLed at ~100s → use the sbatch+tunnel pattern (`ops/leonardo/ops.md`).

## 2. ⚠️ CRITICAL — verify + fix the model name (with cross-user FK safety)
The script auto-detects the model from trial `result.json` → `agent_info.model_info.name`. For vLLM-served
models that field is the **vLLM served-model name** (a numeric ID like `1774950145766573`), NOT the HF repo
— so it can register a **bogus numeric `models` row**. Get the real name from the eval config:
```bash
python3 -c "import json; d=json.load(open('experiments/<RUN_TAG>/configs/<RUN_TAG>_eval_config.json')); print(d['model_hf_name'])"
```
Then check what got registered:
```python
c.table("sandbox_jobs").select("model_id,username").eq("id", "<JOB_ID>").execute()
c.table("models").select("name").eq("id", "<MODEL_ID>").execute()
```
If it's a numeric ID, repoint to the real model — **but FIRST the cross-user FK safety pre-check**
(`feedback_supabase_filter_username`): you are about to UPDATE `sandbox_jobs` / `sandbox_trial_model_usage`
and DELETE a `models` row. **Only touch rows you OWN.** If the `sandbox_jobs` row (or any FK'd
`sandbox_trial_model_usage` row) belongs to ANOTHER user, **STOP** and surface it — do NOT repoint or delete
it. (Mutating another user's eval rows is exactly what broke `zhuang1`'s eval jobs on 2026-05-26.)
```python
import os; me = os.environ.get("USER")
job = c.table("sandbox_jobs").select("id,username,model_id").eq("id", "<JOB_ID>").execute().data[0]
assert job["username"] == me, f"FK-SAFETY STOP: job owned by {job['username']}, not {me} — do not mutate"
correct = c.table("models").select("id").eq("name", "laion/<real-model-name>").execute()
c.table("sandbox_jobs").update({"model_id": correct.data[0]["id"]}).eq("id", "<JOB_ID>").eq("username", me).execute()
c.table("sandbox_trial_model_usage").update({"model_id": correct.data[0]["id"]}).eq("model_id", "<BOGUS_ID>").execute()  # only if all FK'd rows are yours
c.table("models").delete().eq("id", "<BOGUS_ID>").execute()  # only if no other-user row FKs it
```

## 3. Verify it landed
Confirm `sandbox_jobs.<JOB_ID>.model_id` now points to the real `laion/<name>` row and the trial scores are
attached. If `--skip-hf` wasn't used, confirm the trace dataset (`DCAgent2/<RUN_TAG>-traces`) is non-empty on HF.

## 4. Clean up disk (only after upload + register verified)
Remove the local eval run dir once the traces are on HF + the DB row is correct (the `trace_jobs` tree is
the bulk). Detach a large GPFS `rm` (nohup/tmux); never `du`/`find` to size it first. Do NOT delete shared
canonical task dirs.

---
Sibling cleanups: **`rl-job-cleanup`** (RL model), **`sft-job-cleanup`** (SFT model), **`datagen-job-cleanup`** (trace dataset). Launching evals → **`eval-agentic-launch`**. Per-cluster particulars → `.claude/ops/<cluster>/`.

---

## Operating notes (folded from memory 2026-06-14)

- **`notes/ot-agent/task_repos/rl_to_check.md` is a QUEUE file, not documentation** — bare HF repo URLs, one per line, consumed line-by-line by the smoke-test runner (processed entries move to `rl_checked.md`, same flat format). "Update rl_to_check.md with the fixes" = **append the newly-uploaded repo URLs**, one per line. Do NOT add markdown tables / sections / writeups (breaks the parser). Fix notes belong in the chat response or a dedicated `rl_fixes_<date>.md`. Same caution for all `task_repos/*.md` (flat URL/path lists feeding tooling).
