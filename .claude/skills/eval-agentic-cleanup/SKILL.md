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
  publishing cleanups (rl-agentic-job-cleanup / sft-job-cleanup) and datagen-job-cleanup — this is the EVAL path.
---

# eval-agentic-cleanup

Always start with the read-only audit (§0): it verifies the four conditions that actually matter, then run only the remediations it flags. "The score is in the leaderboard" does NOT mean the eval is complete — the auto-harvest path can land a score while silently skipping trace upload, or record a bogus 0 from an eval that never reached the model.

## 0. Read-only completeness + health AUDIT — run first (idempotent, ZERO writes)

Reads only (sacct, run-dir `result.json`, HF API, Supabase reads). Emits a per-eval verdict + recommended action per check; run only the flagged remediations.

**Pre-gate — EXEMPT:** grid / throughput / OOM-test measurement runs targeting `DCAgent2/*` → skip (upload only production winners).

| # | Check | How (read-only) | PASS | If it FAILS → recommend |
|---|---|---|---|---|
| 1 | **Job finished** | `sacct -j <id> -X -o State` | `COMPLETED` | RUNNING/PENDING → wait & re-audit; FAILED/TIMEOUT/CANCELLED → diagnose + relaunch (see `eval-agentic-launch` gotchas #1–#5: `jobs_dir` perm / `hosted_vllm` 2-slash / `model_info` / reservation / stale-row) |
| 2 | **Score present & not broken** | Supabase `sandbox_jobs` score, or run-dir `result.json` → `stats.evals.<k>.metrics[0].mean` | non-null real number; `0` OK only if check 4 shows trials ran | null/missing/`0` with ~0 trials/POSTs = infra-broken (eval never reached the model — classic `jobs_dir`/`hosted_vllm`/`model_info` failure) → re-run. High `VerificationNotCompletedError` rate → score may be biased (note below). |
| 3 | **HF traces present + linked** | trace HF repo exists + non-empty (HF API 200) AND DB/LB row has trace/url field set | repo 200 AND linked | missing/empty OR unlinked → upload traces + register link (§1–§3) |
| 4 | **VALID trial count (PARSE, don't file-count)** | Parse each per-trial `result.json` for a numeric reward (snippet below). An errored trial STILL writes `result.json` with `exception_info` and no reward, so file-count overstates coverage. | VALID ≥ ~90% of `n_rep_eval × benchmark_size` | short on VALID (<~90%, a missing rep, or high hard-error 15–25%) → resume errored trials (re-submit SAME eval/run-tag; Jupiter sbatch auto-calls `harbor jobs resume --filter-error-type …`). Expected: swebench-verified-random-100=100, dev_set_v2=100, terminal_bench_2=89; × `n_rep_eval` (default 3) → ~300/~300/~267. |

Output one row per eval: `✅/⚠️` per check + the single recommended next action. **Re-run the audit after any remediation** to confirm ✅.

### No-reward trial classes (VNC / infra errors) — bias direction
- **`VerificationNotCompletedError` (harbor 9203989f) / pre-fix `AgentTimeoutError` WITH NO reward** = the verifier never produced a result. Missing *because* the agent burned its budget → true score ≈ 0 (SWE-bench ~pass/fail). Harbor's `JobStats` mean DROPS these from the denominator → biased UP; the DB/leaderboard harvest counts them as 0 → unbiased. Either way retriable: resume on a healthy sandbox — the payoff is usually recovering real passes hidden as 0s.
- **Infra errors (`DaytonaAuthenticationError`/`DaytonaValidationError`/etc.)** = failure uncorrelated with solvability → dropping is ~unbiased (lowers N). Re-run for completeness only; do NOT impute 0.
- **`AgentTimeoutError` WITH a reward** = verifier scored it → passthrough VALID 0. Don't filter/re-run.
- **Discriminate by reward presence, not the error label.**

### <90% COMPLETE → RESUME, do NOT REGISTER (and DE-REGISTER if it slipped through)
An eval whose VALID-trial count is < ~90% of planned `n_rep × benchmark_size` (swebench-100=300, tb2=267, dev_set_v2=300) is INCOMPLETE → RESUME to completion (Check-4 resume), NOT register. A partial score silently MIS-RANKS the model. If the auto-harvest already registered a partial (`Pending`/`Started`, or a prematurely-`Finished` partial) → DE-REGISTER that ONE row (cross-user FK-safe per §Remediations); it re-registers cleanly on resume. ≥90% / a ~99% tail-short is fine to finalize. **Never run §1's register/upload on a <90% eval.** Skip a candidate whose slurm job is still RUNNING.

### >10% error-fraction HARD GATE — the authoritative metric
Compute via the ONE utility `scripts/database/eval_guardrail.py` (`guardrail_counts(stats, planned).invalid_error_count`), the Python port of the leaderboard guardrail (`OT-Agent-Leaderboard server/storage.ts:416-449`). It EXCLUDES the benign passthrough error types — never hand-restate the BENIGN set (that's how sibling copies drifted). **error-fraction = invalid_error_count / total_attempted; a row is an incomplete partial (→ de-register + resume) iff > 10%.** There is NO "borderline"/"close enough" discretion: 10.1% fails like 30%. Do NOT re-derive as a naive "no-reward / valid-reward<90%" count — that folds in the benign passthroughs and over-flags near-complete evals. Exception: an explicit, logged user grandfather of a specific already-pushed row.

### Check 4 — counting VALID trials (parse, never file-count) — STANDARD report element
Trial dirs at `eval_jobs/eval-<safe_model>_<safe_dataset>/<task>__<id>/result.json` (depth-1; the run-dir-root `result.json` is the aggregate — exclude via the `*/` glob). A trial is VALID iff `reward` is finite; else ERRORED. One pass over all evals (otagent env, read-only, ~1 min):
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
**Always include the valid/errored/expected matrix in the report.** Flag any eval with ERRORED rate ≳10–15% as a re-run candidate.

### Check 4 — resuming errored trials (`resume_chunked.py`)
Resume re-runs only the errored trials of the SAME run dir (keeps valid trials). Standard driver: **`eval/resume_chunked.py`** → one `unified_eval_listener.py --resume-only --force-reeval` per chunk. **Restate the same sizing/yaml/conda-env/Pinggy from the original fire** or Harbor errors on the `config.json` conflict.

- **`--force-reeval` (NOT `--force-eval`)** — resume-mode flag; distinct from launch-time `--force-eval` (which bypasses dedup in `should_start_job`). Bypasses the DB status check AND the `active_pairs` resume filter (needed when an active job for the same pair uses a different scaffold).
- **`--once` + `--batch-size`** — in `--once` mode, freshly-submitted JIDs fold into `active_ids` as they submit, so the sliding-window `--batch-size` chain takes effect.

**Cat-3 swe-agent (installed-harness) resume — MUST start a Pinggy tunnel.** Cat-3 = preferred-harness reproduction (`dcagent_eval_config_swe_agent.yaml`, swe-agent/openhands/mini-swe-agent) where the sandboxed agent calls back to the served model over a public **Pinggy** tunnel. The resume branch of `eval/jupiter/eval_harbor.sbatch` starts the Pinggy SSH tunnel (gated on `EVAL_PINGGY_URL`) and exports `OPENAI_API_BASE`/`OPENAI_BASE_URL` (+ openhands `LLM_BASE_URL`/`LLM_API_KEY`, mini-swe-agent `MSWEA_*`/`HOSTED_VLLM_API_BASE`). A Cat-3 resume MUST pass the four passthroughs (note hyphens — distinct from launch-mode `--pinggy_persistent_url`/`--pinggy_token`):
```bash
# otagent env, in tmux. One Pinggy pair PER invocation (one chunk per free pair for multi-model batches).
python eval/resume_chunked.py \
  --csv /tmp/resume_cands.csv --preset swebench --org eval \
  --tp-size 2 --dp-size 2 --timeout-multiplier 16.0 \
  --jobs-dir <EVAL_JOBS_DIR> --conda-env <env> \
  --tag-prefix <orig_run_tag_prefix> \
  --pinggy-url <free-pair URL, e.g. dadccqeqqf.a.pinggy.link> \
  --pinggy-token <free-pair token> \
  --config-yaml dcagent_eval_config_swe_agent.yaml \
  --agent-parser '' \
  --chunk-size 4 --sleep-between 120
```
- `--config-yaml` = Harbor config (scaffold/parser); `--agent-parser ''` = disable parser for swe-agent.
- Pinggy pair URL+token privileged — read a FREE pair from `.claude/secret.md`/`pinggy_bank.md`.
- **terminus-2 resumes leave `EVAL_PINGGY_URL` empty** → tunnel block is a no-op (they use the normal sbatch path); only Cat-3 installed-harness resumes set the Pinggy flags.

> **swe-agent retry-policy** (`dcagent_eval_config_swe_agent.yaml`, aligned to M1 SERA-32B 48.67%): `ContextLengthExceededError`/`BadRequestError`/`AgentEnvironmentTimeoutError`/`SummarizationTimeout` are RETRIED; `VerifierOutputParseError`/`RewardFileEmptyError`/`RewardFileNotFoundError` are terminal. A resume scored under this policy is NOT comparable to one scored under the old drifted config.

> **Offline-first pre-download:** sbatch tries `snapshot_download(local_files_only=True)` first (HF cache may be read-only when HEAD advanced past the cached snapshot), falls back to online. Pinggy SSH-out is prefixed with proxychains to egress from no-internet nodes. (etag_timeout=120, 600s hard cap on the pre-download wedge.)

---
# Remediations — the §0 audit scopes which to run (re-run §0 after to confirm ✅)
HF trace upload + Supabase registration are normal, sanctioned operations — run the ones the audit flags.

### Cross-user FK safety — REQUIRED before EVERY write (update / delete)
The same `(model × benchmark)` pair can have MULTIPLE `sandbox_jobs` rows (reruns, resumes, sibling `model_id`s, AND other users' rows). Before ANY mutation (`sandbox_jobs`/`sandbox_trial_model_usage`/`models`/`benchmarks`): **query ALL rows for the pair, disambiguate by `created_at` + `username`, assert the target row is yours** (`username == me`; your usernames: feuer1/bfeuer00/penfever/benjaminfeuer). Scope every update/delete to `id`+`username`. **NEVER mutate/delete another user's row.** When you can't tell which row is yours → STOP and surface, don't guess. (An INSERT of a new own row never touches theirs — see §1.)

## 1. Manual upload + DB register — `manual_db_eval_push.py`
Pass the **`trace_jobs/<RUN_TAG>`** path (where Harbor writes `<task>__<id>` trial dirs), NOT `eval_jobs/<RUN_TAG>` (that only has `meta.env`). Auto-resolves nested trial subdirs and auto-detects agent/model/benchmark.
```bash
set -a; source "${DC_AGENT_SECRET_ENV:?set DC_AGENT_SECRET_ENV to the secrets file first}"; set +a   # SUPABASE_URL, SUPABASE_ANON_KEY, HF_TOKEN
cd <OpenThoughts-Agent>
python scripts/database/manual_db_eval_push.py --job-dir trace_jobs/<RUN_TAG> --verbose
#   --hf-repo DCAgent2/<RUN_TAG>-traces   # explicit HF repo
#   --skip-hf                              # DB only (traces already uploaded)
#   --forced-update                        # overwrite existing records
#   --benchmark-name terminal_bench_2      # EXPLICIT benchmark — PREVENTS the §2b hash bug. Pass when
#                                          # benchmark AND model both contain underscores (e.g.
#                                          # terminal_bench_2_Qwen3_5_35B_A3B_…) — derive_benchmark_from_job_dir
#                                          # can't find the split → mints a HASH benchmark.
```

- **A cross-user-owned (model × benchmark) pair does NOT block registration.** When a gate-passing re-fire's pair is already owned by another user, **REGISTER A NEW `penfever` row alongside it with the corrected score** — don't stop, don't ask. The guardrail forbids DELETING/MUTATING a cross-user row; an INSERT never touches theirs (multiple rows per model×benchmark are allowed).

### HF trace dataset — use the memory-efficient uploader
Use the streamed, last-episode-per-trial uploader (canonical invocation in `rl-agentic-job-cleanup` §8) — naive per-conversation extraction loads every episode into RAM (I/O-heavy on GPFS at 300 trials × hundreds of episodes):
```bash
# otagent env; ALWAYS in tmux; `hf upload`, NEVER `hf upload-large-folder` (LFS-429 deadlocks)
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job-dir trace_jobs/<RUN_TAG> --repo_id <org>/<RUN_TAG>-traces --episodes last
```
`--episodes last` keeps only the scored episode per trial. **Eval-vs-RL:** RL passes `--skip_register`; for EVALS register/link the trace repo onto the eval's existing `sandbox_jobs` row (§2/§3 — set trace/url only, do NOT create a second row or touch the score). On **Leonardo**, login-node `hf upload` is SIGKILLed at ~100s → use the sbatch+tunnel pattern (`ops/leonardo/ops.md`).

## 2. ⚠️ Verify + fix the MODEL name (with cross-user FK safety)
Auto-detect reads `agent_info.model_info.name`. For vLLM-served models that field is the **vLLM served-model name** (numeric, e.g. `1774950145766573`), NOT the HF repo → can register a bogus numeric `models` row. Get the real name:
```bash
python3 -c "import json; d=json.load(open('experiments/<RUN_TAG>/configs/<RUN_TAG>_eval_config.json')); print(d['model_hf_name'])"
```
Then check + repoint (FK-safe; the `assert` enforces it; `models` DELETE only if no other-user row FKs it):
```python
c.table("sandbox_jobs").select("model_id,username").eq("id", "<JOB_ID>").execute()
c.table("models").select("name").eq("id", "<MODEL_ID>").execute()
import os; me = os.environ.get("USER")
job = c.table("sandbox_jobs").select("id,username,model_id").eq("id", "<JOB_ID>").execute().data[0]
assert job["username"] == me, f"FK-SAFETY STOP: job owned by {job['username']}, not {me} — do not mutate"
correct = c.table("models").select("id").eq("name", "laion/<real-model-name>").execute()
c.table("sandbox_jobs").update({"model_id": correct.data[0]["id"]}).eq("id", "<JOB_ID>").eq("username", me).execute()
c.table("sandbox_trial_model_usage").update({"model_id": correct.data[0]["id"]}).eq("model_id", "<BOGUS_ID>").execute()  # only if all FK'd rows are yours
c.table("models").delete().eq("id", "<BOGUS_ID>").execute()  # only if no other-user row FKs it
```

## 2b. ⚠️ ALSO verify + fix the BENCHMARK name (the version-hash mis-registration)
Same failure mode as the model FK. The harness can register the row under a **raw 40-hex dataset version-hash** (`0c553bd6d05d…`/`377118ff…`/`693231ec…`) instead of the benchmark name — bites when the launch passed a local hash-named snapshot dir as the dataset. A hash-named benchmark silently MIS-FILES the score and spawns orphan rows. (Fixed for new launches; old legs, the auto-harvest path, and manual registers off such a run dir can still land under a hash.)

**Detect** (in §0 check-2/3 AND after any register): if the row's benchmark NAME matches `^[0-9a-f]{32,64}$` → mis-keyed.
**Resolve** from the run-dir name (`<benchmark>_<model>_<ts>` — the leading segment IS the benchmark) or trace repo (`DCAgent2/<benchmark>_…`), then map: `swebench-verified-random-100-folders`=`cc1aca76…` · `dev_set_v2`=`b94dfab2…` · `terminal_bench_2`=`34ab93c4…`.
**Repoint FK-safe:**
```python
me = ...  # your username ∈ feuer1/bfeuer00/penfever/benjaminfeuer
job = c.table("sandbox_jobs").select("id,username,benchmark_id").eq("id","<JOB_ID>").execute().data[0]
assert job["username"] == me, "FK-SAFETY STOP — not your row; surface, do not mutate"
proper = c.table("benchmarks").select("id").eq("name","terminal_bench_2").execute().data[0]["id"]
c.table("sandbox_jobs").update({"benchmark_id": proper}).eq("id","<JOB_ID>").eq("username",me).execute()
```
**Clean the orphan hash benchmark** — ONLY if **0 `sandbox_jobs` reference it across ALL users** (cross-user pre-check): delete `sandbox_benchmark_tasks` CHILDREN first (the FK that blocks the delete; that table has **no `id`** column — delete by `.eq("benchmark_id", <id>)`), then delete the benchmark. LEAVE it if ANY row still references it.

## 3. Verify it landed
Confirm `sandbox_jobs.<JOB_ID>.model_id` → real `laion/<name>` row and trial scores attached. If `--skip-hf` wasn't used, confirm the trace dataset (`DCAgent2/<RUN_TAG>-traces`) is non-empty on HF.

## 4. Clean up disk (only after upload + register verified)
Remove the local eval run dir once traces are on HF + the DB row is correct (the `trace_jobs` tree is the bulk). Detach a large GPFS `rm` (nohup/tmux); never `du`/`find` to size it first. Do NOT delete shared canonical task dirs.

---
## 5. IRIS TPU evals — the idempotent path (GCS-backed, no SLURM)

Iris (marin v6e TPU/GPU) agentic evals run through Marin's standalone
`experiments/agentic_evals` package, NOT SLURM — so the SLURM mechanics (sacct,
cluster run-dirs, sbatch resume, GPFS `rm`) DON'T apply. **The audit logic is
identical** (same 4 checks + >10% HARD gate + <90%→resume-not-register +
cross-user FK safety); only the data source + resume/register mechanics differ.

- **Enumerate** terminal evals (no sacct): the marin jobs table.
  `iris=/Users/benjaminfeuer/Documents/marin/.venv/bin/iris`
  `$iris --cluster=marin query "SELECT job_id,state FROM jobs WHERE job_id LIKE '%eval-%' ORDER BY job_id DESC" -f csv`
  **State codes: 4=COMPLETED, 5=FAILED, 6=KILLED, 1/2/3=pending/running (SKIP — live).** EXEMPT `DCAgent2/*` measurement runs (§0 pre-gate).
- **Results live in GCS, not a filesystem.** The aggregate is `<job_output_dir>/<job>/result.json`. **Resolve the prefix — never guess/scan buckets:** `OUT=$(python -m hpc.iris.job_output_resolver <job> --cluster …/marin.yaml)` (registry-first, iris-fallback). It returns whatever bucket the job wrote to (new single-region `gs://marin-us-east5/…`, legacy multi-region `gs://marin-models-us/…`, or the `gs://marin-eu-west4` static default). A single `gsutil ls "$OUT/<job>/"` finds the `result.json` regardless of region.
- **Audit the GCS `result.json`** — schema is `stats.evals.<key>` (NOT top-level `metrics`). Per eval key: `score = metrics[0].mean`; `n_trials`/`n_total_trials` (swe/v2=300, tb2=267); `exception_stats[<name>]` is a **LIST of trial ids → use `len()`** (Σlen over NON-benign names / n_trials = error-fraction; same benign set as §0 check-4). `n_cache_tokens=0` = prefix-cache-off fingerprint.
- **Resume (<90% OR >10% err)** — no sbatch; relaunch `python -m agentic_evals.launch` from the Marin package with the SAME `--job-name` (Harbor resumes incomplete/errored trials). Follow that package's Iris runbook for region, retry, and durable-output options. Iris eval = MAIN Daytona org (`DAYTONA_API_KEY`), `force_build: true` (no snapshot pre-build / cap).
- **Register (≥90%, not auto-registered)** — Iris evals auto-register via `--upload_to_database` on completion, so most complete legs ARE registered; for one that isn't, `gsutil -m rsync -r gs://<bucket>/ot-agent/<job>/<job>/ <local_tmp>/` then `manual_db_eval_push.py --job-dir <local_tmp>` (+ §2/§2b FK fixes, own-rows-only). Then remove the local tmp (GCS is the durable store — no cluster-filesystem cleanup).

Sibling cleanups: **`rl-agentic-job-cleanup`** (RL model), **`sft-job-cleanup`** (SFT model), **`datagen-job-cleanup`** (trace dataset). Launching evals → **`eval-agentic-launch`** (+ the `*-iris` variant for TPU). Per-cluster particulars → `.claude/ops/<cluster>/`.

---

## Operating notes

- **Stale >12h non-Finished (Started/Running/Pending) rows owned by us are cosmetic cruft → DELETE them** (FK-cascade, own-rows-only); registration is create-if-missing so this never blocks a real score landing later.
- **`notes/ot-agent/task_repos/rl_to_check.md` is a QUEUE file** (flat URL list, one per line, consumed line-by-line by the smoke-test runner; processed entries → `rl_checked.md`). "Update rl_to_check.md with the fixes" = **append new repo URLs**, one per line. Do NOT add markdown tables/sections (breaks the parser). Same caution for all `task_repos/*.md`.
