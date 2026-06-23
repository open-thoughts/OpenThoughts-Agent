# OT-Agent Leaderboard — dependency overview

The public web leaderboard that **consumes** OT-Agent eval results out of Supabase and renders the
model × benchmark accuracy table. Written 2026-06-17 from the live repo at
`/Users/benjaminfeuer/Documents/OT-Agent-Leaderboard` (remote
`github.com/richardzhuang0412/OT-Agent-Leaderboard`, deployed on **Replit autoscale** at
**`https://ot-agent-leaderboard.replit.app`**). Deploy/env/debug specifics live alongside this in `ops.md`;
this is the architecture/facts overview. The doc answers two questions: **how results are POPULATED** (our
eval pipeline → Supabase) and **how they are CONSUMED** (Supabase → rendered cells), plus **the DB contract**
our pipeline must satisfy for a score to show up correctly.

- **It is READ-ONLY against Supabase.** The leaderboard writes nothing. All population is done by the OT-Agent
  Python side (`scripts/database/manual_db_eval_push.py` + the eval auto-harvest; see
  `.claude/skills/eval-agentic-cleanup` and `crud-otagent-supabase`). The leaderboard only `select('*')`s.
- **Same Supabase project** the eval team writes to (`SUPABASE_URL` + `SUPABASE_*_KEY`). Tables:
  `agents`, `models`, `benchmarks`, `sandbox_jobs`, `sandbox_trials`, `sandbox_tasks`. It reads through one
  view: **`leaderboard_results`** (defined in `create_leaderboard_view.sql`, installed by hand in the Supabase
  SQL editor — it is NOT in `migrations/`, which only has the dead legacy Drizzle table).

> **Docs-vs-code drift warning.** The repo's own `CLAUDE.md`/`README.md`/`SUPABASE_SETUP.md` describe an
> older design where the SQL view did `DISTINCT ON` dedup and the frontend hit `/api/benchmark-results`. **That
> is stale.** The live code (below) is: the view returns **every** job row (no dedup), and **all** selection /
> dedup / improvement logic happens in JS in `server/storage.ts`; the frontend's only live endpoint is
> `/api/leaderboard-pivoted-with-improvement`. Trust the code, not those three docs.

---

## Consumption — the app

- **Stack:** Vite + **React 18** (TypeScript) client in `client/`, **Express** (TypeScript, run via `tsx` in
  dev / esbuild-bundled in prod) server in `server/`, **Supabase JS client** (`@supabase/supabase-js`) for DB
  access. `package.json` name is `rest-express`; shadcn/ui + Radix + Tailwind for UI; **TanStack Query** for
  fetch/cache; **wouter** for routing. Single page: `client/src/pages/Leaderboard.tsx`.
- **One server process serves both** the API and the static client on **one port (`PORT`, default 5000)** —
  `server/index.ts` mounts Vite middleware in dev / serves `dist/public` in prod. Drizzle ORM is a **dead
  dependency** (legacy `benchmark_results` table in `shared/schema.ts` + `drizzle.config.ts`); the live read
  path never touches it — all queries go through the Supabase client.
- **Data fetch is one shot, client-side everything after.** `Leaderboard.tsx` issues a single TanStack Query
  (`staleTime: Infinity`, manual refresh only) to
  **`GET /api/leaderboard-pivoted-with-improvement?mode=${selectionMode}&hideNoTraceLink=${bool}`**
  (`client/src/pages/Leaderboard.tsx:243`). All search / filter / sort / pagination is then done in-browser
  over the full result set (no server-side paging; fine to ~1000 rows). `selectionMode` ∈
  `oldest|latest|highest|all` (default **`oldest`**).
- **Server endpoints** (`server/routes.ts`): `/api/leaderboard-pivoted-with-improvement` (the only one the UI
  calls), plus `/api/leaderboard-pivoted` and `/api/benchmark-results` (older, still wired but unused), and
  unimplemented legacy CRUD stubs that throw.

### Supabase row → rendered cell (the full path)
1. `DbStorage.fetchAllRawRows()` (`server/storage.ts:207`) → `supabase.from('leaderboard_results').select('*')`
   — pulls **every** job row (one per `sandbox_jobs` row, joined to model/agent/benchmark; see the view below).
2. `buildGroupIndex()` groups rows by **`canonical_agent_id ||| model_id ||| benchmark_name`** into pools.
3. `selectResult(pool, mode)` (`server/storage.ts:135`) picks **one** row per pool (the dedup, done in JS):
   - Split into Finished-with-accuracy vs the rest. **Prefer Finished**, and **prefer non-overlong** over
     overlong (`is_overlong`).
   - `highest` → max accuracy. `oldest`/`latest` → first **prefer rows with accuracy > 1.0** (deprioritize
     "glitchy" 0% runs), then sort by `ended_at` ascending/descending. `all` → keep every row (cell shows ◀▶).
   - If no Finished rows: pick best non-Finished by status priority `Finished>Started>Pending`, latest first.
4. `getAllBenchmarkResultsWithImprovement()` (`server/storage.ts:300`) also builds a **resolved-accuracy map**
   keyed by `modelName ||| agentName ||| benchmarkName` (with `hosted_vllm/` prefix stripped/added as aliases
   and benchmark-duplicate aliases merged) so each trained model's **improvement (pp)** = its accuracy minus
   its **base model's** accuracy *on the same agent + same benchmark*. It also parses `config`/`stats` for
   per-cell badges: timeout multiplier, Daytona overrides, auto-snapshot, **`isIncomplete`** (stats trials <
   `n_trials`), **`isHighErrors`** (>10 non-benign trial errors), trial counts.
5. `routes.ts` **pivots** the flat rows: groups by `modelName ||| agentName`, benchmarks become columns
   (`{accuracy, standardError, improvement, hfTracesLink, …}` per cell). Models in `models` with **zero** eval
   rows are emitted as a synthetic **`NO EVAL`** agent row so untested models are still visible.
6. `LeaderboardTableWithImprovement.tsx` renders the pivot: frozen Model/Agent columns, one column per
   benchmark showing `acc% ± SE` (+ improvement pp, green/red), missing combos as "—", and the HF traces link
   per cell. Client-side filtering then hides rows/cols and applies the blacklist/exclusions below.

### What the cell numbers are (aggregation / normalization)
- **Accuracy** = `accuracy` metric × 100 (a **percentage**, 0–100). **No** z-score / pass@k / cross-trial
  re-aggregation happens in the leaderboard — it takes the single `accuracy` value the eval already computed
  and stored in `sandbox_jobs.metrics`. (Trial-level pooling / pass@k, if any, happened upstream in Harbor.)
- **Standard error** = `accuracy_stderr` metric × 100 (0 if absent).
- **Improvement (pp)** = this row's accuracy − base-model accuracy (same agent+benchmark), computed in JS.
- Multiple eval runs for the same (model, agent, benchmark) are **NOT averaged** — `selectResult` picks ONE by
  the mode. (Contrast `crud-otagent-supabase`, which *averages* identical-setting reruns for paper tables. The
  leaderboard and the paper-table aggregation are different consumers of the same rows.)

### Filtering / what's shown (allow-lists & exclusions — all client-side)
- **Benchmark ordering/visibility** — `client/src/config/benchmarkConfig.ts`: `CORE_BENCHMARKS` =
  `dev_set_v2`, `swebench-verified-random-100-folders`, `terminal_bench_2` (shown by default); `OOD_BENCHMARKS`
  = `aider_polyglot`, `bfcl-parity`, `medagentbench`, `gaia_127`, `swebench-verified`, `financeagent_terminal`;
  everything else is "other". Column order is Core → OOD → other(alpha). **Non-core benchmarks exist in the
  data but are hidden by default** (available via the filter).
- **Model blacklist** — `client/src/config/blacklistedModels.ts`: a hard-coded `Set` of ~400 model names
  (mostly `DCAgent*/…`, `laion/…`, `mlfoundations-dev/…`, `penfever/…` trace/ablation models) that are
  filtered OUT of the table. A model that lands in Supabase but is on this list will **not** appear on the
  leaderboard even though its row exists.
- The page has curated "scale tier" tabs (`~8B / ~32B / Larger / …`) with explicit model allow-lists in
  `Leaderboard.tsx` (`TABLE_1_SECTIONS`) for the paper-style views.

---

## Population — the `leaderboard_results` view (the read contract)

`create_leaderboard_view.sql` — installed manually in the Supabase SQL editor (`DROP VIEW IF EXISTS … CASCADE`
+ `CREATE VIEW` + `GRANT SELECT ON leaderboard_results TO anon, authenticated`). Key facts:

- **Returns ALL `sandbox_jobs` rows, no dedup** (the header comment says so explicitly). It UNIONs
  `finished_jobs` (`metrics IS NOT NULL`) with `pending_started_jobs` (`metrics IS NULL`).
- **Accuracy parsing** (the load-bearing JSONB extraction):
  ```sql
  (SELECT (elem->>'value')::float * 100
   FROM jsonb_array_elements(sj.metrics) elem
   WHERE elem->>'name' = 'accuracy' LIMIT 1)        -- → computed_accuracy
  ```
  and the same for `accuracy_stderr`. **This only handles the LIST-of-`{name,value}` shape of `metrics`.**
  `crud-otagent-supabase` GOTCHA 1 warns `metrics` can ALSO be a plain dict (`{"accuracy": …}`) — **a
  dict-shaped `metrics` row yields `computed_accuracy = NULL` and the view treats the job as
  pending/no-score.** If a Finished eval shows no number on the leaderboard, check the metrics shape first.
- **Joins:** `INNER JOIN agents`, `INNER JOIN models`, `INNER JOIN benchmarks` on the job's FK columns — so a
  job missing any of `agent_id` / `model_id` / `benchmark_id` (or pointing at a deleted entity) is **dropped
  entirely** (it never reaches the view). LEFT JOINs resolve `duplicate_of` → canonical names for
  agent/model/benchmark and the base-model chain.
- **Columns the JS layer relies on** (`RawLeaderboardRow`, `server/storage.ts:82`): `model_id`, `model_name`,
  `canonical_model_name`, `base_model_id`/`base_model_name`/`canonical_base_model_name`, `agent_id`/`agent_name`/
  `canonical_agent_id`/`canonical_agent_name`, `benchmark_name`/`canonical_benchmark_name`/`source_benchmark_*`,
  `accuracy`, `standard_error`, `hf_traces_link`, `ended_at` (`COALESCE(ended_at, created_at)`), `config`,
  `stats`, `n_trials`, `is_overlong`, `job_status`, `username`, `slurm_job_id`, `training_type`, `model_size_b`,
  `notes`.

---

## The DB contract — what a freshly-completed eval MUST have to appear correctly

For a finished eval to render as a correct cell (this is the downstream reason
`eval-agentic-cleanup`'s DB-registration matters):

1. **A `sandbox_jobs` row** with valid FK `model_id` / `agent_id` / `benchmark_id` (INNER JOINs — any missing
   FK = the row is invisible, no error).
2. **`metrics` in LIST shape** `[{"name":"accuracy","value":<frac>}, {"name":"accuracy_stderr","value":<frac>}]`
   (fraction 0–1; the view ×100s it). **Dict-shaped metrics → no score shows** (the #1 silent footgun).
3. **`model_id` points at the REAL HF model name**, not a numeric vLLM served-id. This is the
   `eval-agentic-cleanup §2` fix: vLLM evals can auto-register a bogus numeric `models` row (e.g.
   `1774950145766573`). Such a row joins fine but renders a garbage model name AND breaks base-model /
   improvement matching (the resolved-accuracy map keys on `model_name`), so the improvement column goes blank
   and scale-tier/allow-list tabs (which match on real HF names) won't pick it up.
4. **`benchmark_id`** must resolve to a known benchmark; the displayed benchmark name is the **canonical** name
   (`duplicate_of` resolved), and whether it shows by default depends on `benchmarkConfig.ts` (Core/OOD shown,
   others filter-only). A benchmark family member (e.g. `terminal_bench_2_2.0x`) merges into its canonical
   column via the `source_benchmark`/alias logic.
5. **`hf_traces_link`** set on the row for the trace icon to be blue/linked (and for the `hideNoTraceLink`
   filter to keep the row). Missing link → grey/red icon; with `hideNoTraceLink=true` the row is dropped.
6. **The model name must NOT be in `blacklistedModels.ts`** and (for the curated tabs) should match an
   allow-listed HF name.

### Footguns / schema coupling (flag these when an entry is wrong or missing)
- **Numeric vLLM model-name row** → wrong name + no improvement; fix per `eval-agentic-cleanup §2` (with the
  cross-user FK safety pre-check before repointing/deleting).
- **Dict-shaped `metrics`** → silently NULL accuracy (view only parses the list shape).
- **`?` trial-count numerator** (badge shows `?/X`, score renders fine; diagnosed + fixed 2026-06-17). The LB
  builds the incomplete-trials badge in `server/storage.ts` as `completedTrials = stats.n_trials` (numerator,
  a **top-level** key in the `sandbox_jobs.stats` JSONB) over `totalTrials = n_trials` (the column = expected =
  n_rep×bench_size); when `stats.n_trials` is `undefined` it renders `?`. **Current Harbor renamed the
  legacy top-level `stats.n_trials` → `n_completed_trials`** (`harbor/src/harbor/models/job/result.py` `JobStats`),
  so rows written by current Harbor have no top-level `stats.n_trials` → every numerator shows `?` (older rows
  that kept the legacy key render fine). We can't patch the LB, so the fix is **registration-side**:
  `database/unified_db/utils.py:_extract_job_metadata` (commit `5adf8861`) synthesizes top-level `stats.n_trials`
  when missing = **sum of `stats.evals.*.n_trials`** (the VALID-trial count — reward-bearing only, matching the
  `eval-agentic-cleanup` standard, NOT the error-inclusive `n_completed_trials`). New evals populate it
  automatically; the 21 existing `feuer1` rows were backfilled FK-safe. If a NEW Harbor rename breaks it again,
  re-check which top-level key `storage.ts` reads vs what `JobStats` serializes.
- **A `0%` "glitchy" run** can outrank a real run in `oldest`/`latest` only if no run clears the 1.0% threshold;
  `selectResult` already deprioritizes sub-1% runs, but if ALL runs are sub-1% the earliest 0% is shown.
- **Missing FK** (model/agent/benchmark deleted or never registered) → job vanishes from the view with no error.
- **Blacklist / default-hidden benchmark** → row/column present in data but not rendered; check the two config
  files before assuming a population bug.
- A **new model registered** in `models` fires a Supabase webhook → `supabase/functions/notify-new-model`
  (Resend email "New model registered — Fire Eval" to the maintainer); informational, not part of the read path.

---

## File map (for "where do I change X")
- **View / read contract:** `create_leaderboard_view.sql` (+ `setup-view.ts` helper to install it).
- **Selection / dedup / improvement / badges (JS):** `server/storage.ts` (`selectResult`, the 3-pass resolved-
  accuracy logic).
- **Pivot + NO-EVAL rows + endpoints:** `server/routes.ts`.
- **Supabase client / env:** `server/db.ts` (service-role key preferred, falls back to anon).
- **What's shown:** `client/src/config/benchmarkConfig.ts` (Core/OOD/default-visible),
  `client/src/config/blacklistedModels.ts` (hidden models), `client/src/pages/Leaderboard.tsx` (tabs,
  allow-lists, filter state), `LeaderboardTableWithImprovement.tsx` (rendering, duplicate-aware improvement).
- **Dead/legacy:** `shared/schema.ts` Drizzle tables, `drizzle.config.ts`, `migrations/`,
  `/api/benchmark-results*` (kept for compat, not on the live path).

> Deploy steps, env vars, view install/refresh, and "debug a missing entry" runbook live in `ops.md` here.
