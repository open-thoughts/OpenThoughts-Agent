---
name: crud-otagent-supabase
description: Read, aggregate, and (carefully) write OT-Agent eval/model data in the Supabase registry. Use when asked to look up a model's ID/OOD benchmark scores, build/refresh an ablation or paper table from eval results, find unevaluated models, register a model/eval, or reconcile duplicate rows. Covers the sandbox_jobs/models/benchmarks schema, the metrics-field shape gotchas, the ID/OOD benchmark master list + per-benchmark task counts (for binomial SE), and the multiple-entries-per-model rule.
---

# crud-otagent-supabase

The OT-Agent Supabase is the source of truth for **model eval results** (the
numbers behind every ablation/paper table). The scores are NOT in any markdown
file — they live in `sandbox_jobs`. This skill is how to query them correctly,
aggregate ID/OOD means + SE the way the tables do, and write rows safely.

## Connect (run LOCALLY — faster than ssh-ing a cluster)

Run from the Mac with the `otagent` env; source the local secrets. Per the
team convention, run Supabase queries locally (≈10 s faster than ssh+python on a
cluster), and **filter by your own username for any write** (see Guardrails).

```bash
cd /Users/benjaminfeuer/Documents
set -a; source secrets.env; set +a      # SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY (+ ANON_KEY, HF_TOKEN)
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python - <<'PY'
import os
from supabase import create_client
c = create_client(os.environ["SUPABASE_URL"],
                  os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"])
# ... queries ...
PY
```

- `SUPABASE_SERVICE_ROLE_KEY` bypasses RLS (full read/write). Prefer it for reads;
  for writes it means **nothing stops you mutating other users' rows** — so the
  cross-user pre-check in Guardrails is mandatory.
- Schema DDL lives at `/Users/benjaminfeuer/Documents/OpenThoughts-Agent/schema/`
  (`sandbox_jobs`, `models`, `benchmarks`, …). Read it when unsure of a column.

## Schema (the tables you'll touch)

| table | what it holds | key columns |
|---|---|---|
| `sandbox_jobs` | **one row per (model × benchmark) eval job** — the scores | `model_id`, `benchmark_id`, `metrics` (jsonb), `stats` (jsonb), `job_status`, `n_rep_eval`, `n_trials`, `username`, `hf_traces_link`, `slurm_job_id` |
| `models` | registered models | `id`, `name` (HF repo, e.g. `laion/<run>-<step>-<size>`) |
| `benchmarks` | benchmark catalog | `id`, `name`, `duplicate_of` (→ canonical benchmark id, for families) |
| `agents` | harness/agent rows | `id`, `name` |
| `sandbox_trial_model_usage` | per-trial model usage (FK→models) | `model_id` — **watch for cross-user FKs before deleting a model** |

`job_status` is an enum: `Pending` / `Started` / `Finished` (+ failure states).
**Only `Finished` rows with a non-null accuracy are real scores** — `Pending`/`Started`
rows have `metrics=None`.

## GOTCHA 1 — the `metrics` field has TWO shapes

`metrics` is jsonb and appears as **either** a list of `{"name","value"}` dicts
**or** a plain dict. Always extract through a shape-robust helper, never
`metrics["accuracy"]` directly:

```python
def get_metric(metrics, key="accuracy"):   # key: "accuracy" or "accuracy_stderr"
    if metrics is None: return None
    if isinstance(metrics, dict):           # {"accuracy": 0.25, ...}
        return metrics.get(key)
    if isinstance(metrics, list):           # [{"name":"accuracy","value":0.25}, {"name":"accuracy_stderr",...}]
        for e in metrics:
            if isinstance(e, dict):
                if e.get("name") == key:    return e.get("value")
                if e.get(key) is not None:  return e.get(key)
    return None
```
The list-of-dicts form also carries `accuracy_stderr` (the eval's own reported SE)
— available if you want it, but the ablation tables recompute a pooled binomial SE
(see "Aggregation").

## GOTCHA 2 — multiple entries per model; sibling model rows

A single logical model can have **more than one score per benchmark**, and even
**more than one `models` row**. Three rules:

1. **Multiple entries per (model, benchmark):** there are often a `Pending`/`Started`
   row (acc `None`) *and* a `Finished` row for the same benchmark, or two reruns.
   - Drop non-`Finished`/null rows.
   - **When ≥2 *complete* entries exist with identical evaluation settings, AVERAGE
     them** (don't pick max, don't pick first). Different settings (e.g. a different
     `n_rep_eval` or harness) are not "identical settings" — keep them separate / pick
     the canonical one.
2. **Sibling model rows:** the same HF model may be registered under multiple
   `models.id` (trainer auto-push + a manual `-<step>-<size>` row, or a duplicate).
   **Query models by `ilike` on a name stub, not exact match**, and union the
   `sandbox_jobs` across all matching `model_id`s before aggregating — a benchmark
   "missing" on one row may be `Finished` on a sibling.
3. **Benchmark families (`duplicate_of`):** a benchmark may have family members
   (e.g. `dev_set_v2` → `DCAgent_dev_set_v2`, `dev_set_v2_2.0x`). Resolve via the
   `duplicate_of` field (follow it to the canonical row) the way
   `scripts/database/query_unevaled_models.py` does, when a score seems absent under
   the exact name.

```python
def get_model_scores(name_stub):
    bm = {b["id"]: b["name"] for b in c.table("benchmarks").select("id,name").execute().data}
    mods = c.table("models").select("id,name").ilike("name", f"%{name_stub}%").execute().data   # sibling rows
    perb = {}                                   # benchmark name -> list of (acc, status)
    for m in mods:
        for j in c.table("sandbox_jobs").select("benchmark_id,metrics,job_status").eq("model_id", m["id"]).execute().data:
            bn = bm.get(j["benchmark_id"], j["benchmark_id"])
            perb.setdefault(bn, []).append((get_metric(j["metrics"]), j["job_status"]))
    scores = {}                                 # benchmark -> averaged fraction over complete entries
    for bn, entries in perb.items():
        done = [a for a, s in entries if a is not None and s == "Finished"]
        if done:
            scores[bn] = sum(done) / len(done)  # AVERAGE identical-setting complete repeats
    return scores, mods, perb
```

## ID / OOD benchmark master list (memorize — easy to get wrong)

The ablation tables split the agent benchmarks into **ID ("Core")** and **OOD**.
**The split is NOT stored in the schema** and is NOT the `CORE_BENCHMARKS` list in
`eval/check_progress.py` (that grouping is display-ordering only and gives wrong
table numbers). Use exactly this:

| set | benchmarks | task count `N` (for SE) |
|---|---|---|
| **ID (Core)** | `swebench-verified-random-100-folders` | 100 |
| | `terminal_bench_2` | 89 |
| | `dev_set_v2` (**partial-credit** — see SE note below) | — (excluded from binomial SE) |
| **OOD** | `swebench-verified` (the **full** 500-task set) | 500 |
| | `aider_polyglot` | 225 |
| | `bfcl-parity` | 123 |
| | `financeagent_terminal` | 50 |
| | `gaia_127` | 127 |
| | `medagentbench` | 300 |

> **`dev_set_v2` is ID as of 2026-06-16** (3-member ID set, matching the `eval-agentic-launch` "ID evals"
> launch shorthand `{swebench, v2, tb2}`). BUT `dev_set_v2` is **partial-credit** (no clean binomial `N`,
> see `analyze-rl-behavior`): it contributes to the ID **mean** but is **EXCLUDED from the pooled binomial
> SE** (SE pools only over the binary ID benchmarks: `swebench-verified-random-100-folders` + `terminal_bench_2`).
> For any **model-vs-model ranking**, still compare on a shared **binary** benchmark (swebench-100 or tb2),
> never on `dev_set_v2`.

**The #1 trap:** `swebench-verified` (full, 500) is **OOD**, while the
`swebench-verified-random-100-folders` subset (100) is **ID**. They are different
benchmarks with similar accuracy — so using the wrong one barely moves the *mean*
on most models but (a) flips ID↔OOD membership, (b) changes the SE `N` a lot
(100 vs 500), and (c) can wrongly report a model "ID-incomplete" when only the
full-swebench (OOD) eval is still running. Always check **which** swebench row is
`Finished`.

```python
ID  = {"swebench-verified-random-100-folders", "terminal_bench_2", "dev_set_v2"}
OOD = {"swebench-verified", "aider_polyglot", "bfcl-parity",
       "financeagent_terminal", "gaia_127", "medagentbench"}
# dev_set_v2 is partial-credit → in ID for the MEAN, but NOT in NTASK / the binomial SE.
ID_SE_BENCHMARKS = {"swebench-verified-random-100-folders", "terminal_bench_2"}  # SE pools over these only
NTASK = {"swebench-verified-random-100-folders": 100, "terminal_bench_2": 89,
         "swebench-verified": 500, "aider_polyglot": 225, "bfcl-parity": 123,
         "financeagent_terminal": 50, "gaia_127": 127, "medagentbench": 300}
```

## Aggregation (reproduces the ablation/paper tables exactly)

Per set (ID or OOD): the cell **mean** is the unweighted average of the
per-benchmark averaged accuracies; the **SE** is a pooled **binomial** over the
total tasks of the *present* benchmarks. (Verified: this reproduces the existing
`rl_ablation_table.tex` rows.) **For ID, the SE pools over `ID_SE_BENCHMARKS` only**
(swebench-100 + tb2) — `dev_set_v2` is in the ID **mean** but partial-credit, so it
has no clean binomial `N` and is excluded from the SE pool.

```python
import math
def set_stat(S, scores):                        # scores from get_model_scores()
    present = {b: scores[b] for b in S if scores.get(b) is not None}
    missing = sorted(set(S) - set(present))
    if not present: return None
    mean = sum(present.values()) / len(present) # fraction
    N    = sum(NTASK[b] for b in present)
    se   = math.sqrt(mean * (1 - mean) / N) * 100
    return round(mean*100, 1), round(se, 1), sorted(present), missing   # (mean%, SE, present, missing)
```
- **Completeness:** a set is complete only if `missing == []`. Report partial
  means with the missing-benchmark list (e.g. a paper-table dagger). A model can
  be ID-complete but OOD-incomplete (or vice-versa) — check each set separately.
- **SE scales with `N`:** ID SE ≈ 2.6–3.0 (N=189), OOD SE ≈ 1.1–1.2 (N=1325 when
  all 6 present). If you see ID SE ≈ 1.7 you (or a prior table) used the full
  500-task swebench in ID by mistake — the membership trap above.

## CREATE / UPDATE (registration)

Do **not** hand-insert rows; use the maintained scripts (they fill 9–12 metadata
fields and resolve FKs). Full flags are in `OpenThoughts-Agent/CLAUDE.md`.

- **Register a model** (after an HF upload): `scripts/database/manual_db_push.py`
  `--hf-model-id <repo> --base-model <hf> --dataset-name <name|comma,list> [--training-type RL]`
  (defaults to SFT; pass `RL` for RL models). Multi-dataset → comma-list sets
  `dataset_names`.
- **Register/repair an eval job** (auto-upload failed): `scripts/database/manual_db_eval_push.py`
  `--job-dir trace_jobs/<RUN_TAG> [--hf-repo …] [--skip-hf] [--forced-update]`.
  **Verify the registered model name afterward** — for vLLM-served models the
  auto-detected name can be a numeric served-id, not the HF repo; the CLAUDE.md
  "Manual Eval Upload" section has the fix-up queries.
- **Per-series exceptions:** some series are HF-upload-ONLY (e.g. the Delphi RL
  scaling-laws SFT grid) — do **not** register those. Honor any
  `enable_db_registration: false` and documented no-DB series.

## Cross-linking eval traces to the leaderboard (`hf_traces_link`)

After an eval's HF trace dataset is uploaded, the leaderboard's **trace icon** keys off the
**`sandbox_jobs.hf_traces_link`** column (a `text` URL on the **job** row, NOT on `models`) —
see `.claude/projects/ota-leaderboard/ota-leaderboard.md` (DB-contract §5: missing link →
grey/red icon, and the `hideNoTraceLink` filter drops the row). Cross-linking = setting that
**one column** to the trace repo URL. Touch **nothing else** (never scores/`metrics`,
`model_id`, status).

**Discover the eval → HF-repo mapping** (don't guess the repo from the run-tag — the auto-push
often fails or truncates). For each eval, the run dir is `$EVAL_JOBS_DIR/eval-<safe_model>_<safe_dataset>`
(`safe_*` = `/`→`_`); read its two files:
- **`meta.env`** → `DB_JOB_ID` (the `sandbox_jobs.id` to update), `SLURM_JOB_ID`, `MODEL`, `REPO_ID`.
- **`upload.log`** → the `[uploader] upload_eval_results(... hf_repo_id='<org>/<run_tag>')` line is the
  repo the auto-harvest *attempted*. **It is frequently NOT the live repo** — the auto-push targets
  `DCAgent2/<run_tag>` (no `-traces` suffix) and on the shared `DCAgent2` org it commonly **403s on
  storage quota** (`exceeded your public storage space`). The real, manually re-uploaded dataset lands
  in **`laion/<run_tag>-traces`** (the canonical `<org>/<run_tag>-traces` convention; `rl-job-cleanup`
  §8 / `eval-agentic-cleanup` §1). **Always confirm on HF which repo actually exists + is non-empty
  before linking** — do not link the failed `DCAgent2` attempt.

```python
from huggingface_hub import HfApi; api = HfApi(token=os.environ["HF_TOKEN"])
rid = f"laion/{run_tag}-traces"                 # run_tag = the eval run-dir basename
info = api.dataset_info(rid, files_metadata=True)   # raises if missing
assert any(s.rfilename.endswith(".parquet") for s in info.siblings), f"{rid} empty"
url  = f"https://huggingface.co/datasets/{rid}"
```
> **96-char trap:** HF repo names cap at **96 chars after the `org/`**. A long run-tag (e.g.
> `eval-laion_swesmith-coldstart-complete-lt32k-2ep-8B_DCAgent2_swebench-verified-random-100-folders-traces`,
> 109 chars) **cannot be created** — the uploader truncates+hashes it (`…swebench-verified80dacb91`) or
> the push silently never lands. If `dataset_info` raises `HFValidationError` on length, the canonical-named
> repo does not exist → **flag that eval as un-linkable, do NOT invent a link.**

**FK-safe, idempotent, single-column update** (same cross-user rule as DELETE/MUTATE below — assert
ownership *before* writing, scope the write to `id` + `username`, never overwrite a good link):

```python
ME = "feuer1"   # the cluster username that OWNS the rows — NOT the Mac's os.environ["USER"]
row = c.table("sandbox_jobs").select("id,username,hf_traces_link").eq("id", job_id).execute().data[0]
assert row["username"] == ME, f"FK-SAFETY STOP: owned by {row['username']} != {ME} — do NOT mutate"
if row["hf_traces_link"] == url:    pass                      # idempotent: already correct → skip
elif row["hf_traces_link"]:         pass                      # a DIFFERENT good link exists → don't clobber
else:
    c.table("sandbox_jobs").update({"hf_traces_link": url}).eq("id", job_id).eq("username", ME).execute()
```
Then **re-read** each row to confirm the link set and that `metrics`/`model_id`/`job_status` are
unchanged. Report any eval whose trace repo was missing/empty/over-length (couldn't link). This is the
field `analyze-rl-behavior` waits on — its Q1 behavioral/judge steps auto-resolve to "no post-RL eval
traces selected" while `hf_traces_link` is `None`.

## DELETE / MUTATE — cross-user FK safety (MANDATORY pre-check)

Before deleting or updating ANY row, confirm no **other user's** rows depend on
it. Restrict every write to rows you own; if an FK forces touching someone else's
row, **STOP and ask** — do not repoint/delete it. (A past cleanup repointed
another user's eval jobs without authorization; don't repeat.)

```python
import os
me = os.environ.get("USER", "<your_user>")
fk = (c.table("sandbox_jobs").select("id,username,model_id")
        .eq("model_id", target_model_id).neq("username", me).execute().data)
if fk:
    print(f"STOP: {len(fk)} other-user sandbox_jobs FK this model — leave it, surface to user.")
else:
    ...  # safe to delete the model row / your own jobs
```
Also check `sandbox_trial_model_usage` (FK→models) the same way. Reads are always
safe; the danger is exclusively delete/update.

## Quick recipes

- **"What are model X's ID/OOD scores?"** → `get_model_scores(stub)` → `set_stat(ID,…)`,
  `set_stat(OOD,…)`; report mean±SE + any missing benchmarks.
- **"Build/refresh an ablation row."** → same, then format `mean_{SE}`. Means use the
  averaged-complete-entries rule; SE uses the binomial `N` from `NTASK`.
- **"Find models not yet evaluated on benchmark Y."** → `scripts/database/query_unevaled_models.py
  --benchmark <family> --size <8|32>` (resolves families via `duplicate_of`).
- **"Reconcile a duplicate model row."** → find sibling rows by `ilike`, run the
  cross-user FK pre-check, then dedup only your own rows.

## Recipe: per-task timeouts + turn counts + outcome breakdown (model × benchmark)

To answer "what timeout/turns did model X actually get on benchmark Y, and how did
its trials end?" — the numbers are split across **three** sources; no single column
has them:

1. **The multiplier** is in `sandbox_jobs.config` (jsonb). Pull the job row and read:
   `config.timeout_multiplier` (the scalar applied to every task's declared timeout),
   `config.agents[0].max_timeout_sec` (the hard ceiling — effective timeout is capped
   here), `config.verifier.max_timeout_sec`, `n_attempts`, `orchestrator.n_concurrent_trials`.
   ```python
   j = c.table("sandbox_jobs").select("config,stats,metrics,hf_traces_link") \
         .eq("model_id", model_id).eq("benchmark_id", tb2_id).eq("job_status","Finished").execute().data[0]
   mult = j["config"]["timeout_multiplier"]           # e.g. 2.0 for the eval team's real tb2 runs
   cap  = j["config"]["agents"][0]["max_timeout_sec"]  # 7200 ceiling
   ```
   Note: the eval team runs **terminal_bench_2 at `timeout_multiplier: 2.0`** (the
   `terminal_bench_2_2.0x` benchmark family + the listener's per-bench EVAL_TIMEOUT_MULTIPLIER);
   the sbatch *default* of 1.0 is NOT what TB2 actually uses.

2. **The real per-task timeout (seconds)** = each task's declared `agent.timeout_sec`
   **× `timeout_multiplier`**, hard-capped at `agents[0].max_timeout_sec`. The declared
   base is in the benchmark's task set (one `task.toml` per task), NOT in supabase. For
   TB2 the task set is `DCAgent2/terminal_bench_2`:
   ```python
   import tomllib; from huggingface_hub import HfApi, hf_hub_download
   api=HfApi()
   tomls=[s.rfilename for s in api.dataset_info("DCAgent2/terminal_bench_2").siblings if s.rfilename.endswith("task.toml")]
   base={tl.split("/")[0]: tomllib.load(open(hf_hub_download("DCAgent2/terminal_bench_2",tl,repo_type="dataset"),"rb"))["agent"]["timeout_sec"] for tl in tomls}
   eff = {t: min(b*mult, cap) for t,b in base.items()}   # real per-task effective timeout
   ```
   (TB2 base spans 600–12000s, median 900s → at 2.0× the median task gets 1800s/30min,
   long tail capped at 7200s.)

3. **Turn count + per-trial outcome** are in the **consolidated traces** at
   `sandbox_jobs.hf_traces_link` (an HF dataset). The traces do **NOT** carry per-trial
   timing/config (the `result` field is a short status string, not the full result.json) —
   so don't look for `agent_execution` seconds there; derive the timeout from #1×#2.
   What the trace columns DO give:
   - `conversations` (list) → **turn count** = number of `role=="assistant"` messages
     (fallback: `len(conversations)`); report min/median/mean/max.
   - `result` (string) → **per-trajectory terminal status**: a reward (`"1.0"`/`"0.0"`)
     **or** an exception type (`SummarizationTimeoutError`, `AgentTimeoutError`,
     `VerifierTimeoutError`, …). `collections.Counter` over it = the outcome breakdown
     (e.g. how many timed out vs passed, and *which* timeout dominates).
   - `episode`, `trial_name`, `task` → group turns/outcomes per task.
   ```python
   import pyarrow.parquet as pq, collections
   t=pq.read_table(hf_hub_download(traces_repo,"data/train-00000-of-00001.parquet",repo_type="dataset")).to_pylist()
   turns=[sum(1 for m in r["conversations"] if m.get("role")=="assistant") for r in t]
   outcome=collections.Counter(str(r["result"]) for r in t)   # rewards + exception types
   ```

Aggregate-only accuracy/error counts are also in `sandbox_jobs.stats`
(`stats.evals.<key>.n_errors / n_trials / reward_stats.reward` lists tasks by reward),
but `stats` does NOT have per-task timeouts or turns.

## Guardrails

- **Reads are free; writes are dangerous.** The service-role key bypasses RLS —
  always run the cross-user FK pre-check before any delete/update, and never
  mutate another user's rows.
- **Average complete identical-setting reruns; never silently pick one.** Check
  ALL entries per (model, benchmark) AND all sibling `models` rows by `ilike`.
- **ID vs OOD:** `swebench-verified-random-100-folders` = ID; full `swebench-verified`
  = OOD; `dev_set_v2` = neither. Don't trust `check_progress.py`'s CORE list.
- **`metrics` is list-OR-dict** — always go through `get_metric`.
- **Honor no-DB series** (`enable_db_registration: false`, Delphi grid, etc.).
- A `Finished` SLURM/eval job is necessary but not sufficient — confirm a numeric
  accuracy in `metrics` (evalchemy can exit 0 with empty `results`).

---

## Operating notes (folded from memory 2026-06-14)

- **Always filter bulk sandbox_jobs deletes/updates by username** — `.eq("username", "bfeuer00")`. NEVER delete all rows matching a status without the username filter (once deleted 95 Pending/Started rows across ALL users). Never assume a shared table's rows all belong to one user.
- **Run Supabase queries on the LOCAL Mac, not via `ssh Leonardo`** (~10s/query SSH round-trip saved): `source /Users/benjaminfeuer/Documents/secrets.env` for `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY`, run with the `supabase` client (fall back to `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python` if not installed locally). Only use the cluster when the data lives there.
