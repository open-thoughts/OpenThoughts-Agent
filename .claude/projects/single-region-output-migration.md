# Single-region transient-output migration

**Status:** proposal for human review — no code changes yet.
**Author:** architecture pass, 2026-07-13.
**Scope:** route OT-Agent iris TPU *transient outputs* (datagen trace dirs, eval
outputs, `xla_cache`) from the launcher-pinned **multi-region** buckets
`gs://marin-models-{us,eu}` to **single-region** buckets co-located with the
worker (e.g. `us-east5 -> gs://marin-us-east5`). Multi-region storage costs
~2x single-region; the outputs are transient, so co-locating them in a
single-region bucket is a pure cost win with no read-locality regression (the
job is already region-pinned to one region at submit time).

## Non-goals / out of scope (do NOT touch)

- **Model-weight mirror** (`scripts/iris/mirror_hf_to_gcs.py`,
  `scripts/iris/launch_hf_mirror.py`). Weights live at
  `gs://marin-models-{us,eu}/ot-agent/models/...`. Re-mirroring is an expensive
  internet pull; these are **inputs that stay put** on multi-region. Follow-up
  only (see Risks §7).
- **Source datasets / task parquets** — inputs, stay put.
- **Bulk data move.** A separate cleanup agent is already deleting dead
  transient data from `marin-models-us`. This migration is **forward-only**:
  new launches write single-region; old multi-region data drains via that
  cleanup plus a lifecycle TTL (§5). No `gsutil cp`/`rsync` of existing data,
  no Storage Transfer Service.

## Verified facts (checked against the live repo + GCS, 2026-07-13)

- Single-region buckets **exist and are ours** (spot-checked with `gsutil`):
  `marin-us-east5` (Location type: **region**, US-EAST5), `marin-us-central1`,
  `marin-us-central2`, `marin-us-east1`, `marin-us-west4`, `marin-eu-west4`.
  For contrast `marin-models-us` is Location type: **multi-region** (US) — the
  ~2x bucket we are moving *off* of for outputs.
- Canonical single-region map already exists in marin:
  `config/marin.yaml` `data.region_buckets` and
  `lib/marin/src/marin/rl/placement.py:123 marin_prefix_for_region`.
- The launcher already **pins to exactly one region at submit time**
  (`hpc/iris/launcher.py` `run()` ~358-410): `discover_region_for_tpu` ->
  `gcs_bucket_for_region(region)` -> `args.gcs_output_dir =
  "{bucket}/ot-agent"`; the iris job constraint is `regions=(pinned_region,)`
  (a *single* region, not a continent), and iris honors it across
  preempt-retries. So a single-region output bucket co-located with that region
  is always read/write-local for the job's whole lifetime, and Harbor's resume
  invariant (a stable submit-time `jobs_dir`) is preserved.
- **The read-side linchpin is already recorded per job.** Every GCS-mode
  submission writes a row to the local registry
  `~/.ot-agent/state/iris_jobs.db` via
  `hpc/iris_job_registry.py:register_submission(..., gcs_output_dir=remote_output_dir)`.
  The fetch daemon (`hpc/iris_fetch_daemon.py:301,371`) *already* resolves a
  job's output location from `record.gcs_output_dir` — **not** a hardcoded
  bucket. So the daemon is region-agnostic today and needs **no change**. The
  hardcoded readers are the analysis script and the rescue *skills* (§3).
- Iris also stores the job's baked entrypoint command (which carries the
  `--jobs-dir`/`--gcs-output-dir` string) + env, so iris is a fallback
  source-of-truth when the local registry lacks a job (e.g. launched from
  another host). This is the top risk to nail down (§7).

## 1. Region -> single-region output bucket map

Adopt exactly the marin canonical map (`config/marin.yaml data.region_buckets`),
duplicated as a local dict in `hpc/iris/regions.py`. Duplication (not importing
marin) is consistent with the existing module design: its docstring notes
`regions.py` is imported on workers that may only have the base OT-Agent env,
not the `datagen-tpu` extra that pulls in `marin-rigging`.

| iris region   | single-region output bucket | exists |
|---------------|------------------------------|--------|
| us-central1   | `gs://marin-us-central1`     | yes    |
| us-central2   | `gs://marin-us-central2`     | yes    |
| us-east1      | `gs://marin-us-east1`        | yes    |
| us-east5      | `gs://marin-us-east5`        | yes    |
| us-west4      | `gs://marin-us-west4`        | yes    |
| europe-west4  | `gs://marin-eu-west4`        | yes    |

**Fail-fast for unmapped regions** (e.g. `asia-*`, or an EU region other than
`europe-west4`): the new lookup returns `None`; `discover_region_for_tpu`
already drops candidate regions with no known bucket, and the runtime helper
`region_local_output_prefix` already raises `ValueError` on an unmapped region.
Keep both behaviors — never silently emit a wrong-continent bucket. Add a unit
test asserting `asia-northeast1 -> None` and `europe-west1 -> None`.

### Critical distinction — do NOT reuse `gcs_bucket_for_region` for outputs

`gcs_bucket_for_region` (multi-region) is used in **two** places today, and only
one of them should move to single-region:

1. **Output pin** (launcher line 382, `region_local_output_prefix`,
   `discover_region_for_tpu` candidate filter) — **MOVE to single-region.**
2. **Model-weight YAML region guard**
   (`regions.py:assert_yaml_regions_match_pin` -> `gcs_bucket_for_region`, and
   the `_GCS_URI_RE = gs://marin-models-(us|eu)` scanner). This validates that a
   YAML's **model_path weights** live in the pinned continent. Weights stay on
   the **multi-region** `marin-models-{us,eu}` mirror (out of scope). If this
   guard were switched to single-region it would start *rejecting* every YAML
   whose weights point at `gs://marin-models-us/...` — i.e. it would break
   correct launches. **This guard must keep using the multi-region map.**

Therefore the write-side change is **additive**, not a rename:

- Keep `_REGION_PREFIX_TO_BUCKET` + `gcs_bucket_for_region` (multi-region) for
  the weight-mirror guard, unchanged.
- Add `_REGION_TO_OUTPUT_BUCKET` (exact per-region single-region map above) +
  `output_bucket_for_region(region) -> str | None`.
- Point the three **output** call sites at `output_bucket_for_region`.

## 2. Write-side change (launcher + regions + xla_cache)

**`hpc/iris/regions.py`**
- Add the exact map + `output_bucket_for_region(region)` (returns single-region
  bucket or `None`).
- `region_local_output_prefix(subpath="ot-agent")`: switch its internal
  `gcs_bucket_for_region` call to `output_bucket_for_region`. (This is the
  "already-built region-local util" — it is the runtime, metadata-server-based
  path; flipping the one lookup makes any runtime writer single-region too.)
- `discover_region_for_tpu`: change the candidate filter (line ~271) from
  `gcs_bucket_for_region(r["region"])` to `output_bucket_for_region(r["region"])`
  so a region lacking a single-region output bucket is never picked.
- Leave `gcs_bucket_for_region` / `assert_yaml_regions_match_pin` /
  `_GCS_URI_RE` **untouched** (weight guard, multi-region).

**`hpc/iris/launcher.py`** (`run()`, line ~382)
- `bucket = gcs_bucket_for_region(region)` -> `bucket = output_bucket_for_region(region)`.
- The submit-time pin stays stable exactly as today: `args.gcs_output_dir =
  f"{bucket}/ot-agent"`, `args._pinned_region = region`,
  `regions=(pinned_region,)` in `build_job_constraints`. No runtime
  recomputation of `jobs_dir` — the resolved value is baked into the worker
  command and recorded in the registry once, at submit.
- The fallback message strings that mention `DEFAULT_GCS_OUTPUT_ROOT` remain
  (unmapped/undiscovered region path).

**`hpc/iris/outputs.py`**
- `DEFAULT_GCS_OUTPUT_ROOT = "gs://marin-eu-west4/ot-agent"` — this is *already*
  a single-region bucket, so the static default needs **no change**. (It is the
  opt-out / discovery-failed fallback; leaving it single-region is correct.)

**`hpc/iris/env.py`** (`apply_iris_runtime_env`, line ~106) — xla_cache
co-location:
- No code change required. `OT_AGENT_XLA_CACHE_BASE` is derived from
  `args.gcs_output_dir` by stripping `/ot-agent`
  (`cache_root = args.gcs_output_dir.rstrip("/").rsplit("/ot-agent",1)[0]` ->
  `{cache_root}/ot-agent/xla_cache`). Once `gcs_output_dir` is
  `gs://marin-us-east5/ot-agent`, the cache automatically lands at
  `gs://marin-us-east5/ot-agent/xla_cache` — co-located, single-region. Add a
  test asserting this derivation for a single-region root. (This preserves the
  per-CPU-tag/per-model xla_cache namespacing that lives *below* this base.)

Net write-side effort: ~1 new small function + 3 one-line call-site swaps + 2
unit tests. Low risk.

## 3. Read-side change (the bulk) — resolve bucket from recorded URI, per reader

Principle: **every reader learns a job's output bucket from that job's recorded
output URI, never from a hardcoded scan bucket.** Introduce one shared resolver
and route all readers through it.

### 3.0 New shared resolver (enables everything below)

Add `hpc/iris/job_output_resolver.py` (name TBD) with:

```
resolve_job_output_dir(job_ref: str, *, cluster_config: str | None = None) -> str
```

Resolution order:
1. **Local registry** `iris_jobs.db` — `get_latest_by_job_name(job_ref)` (or
   `get(job_id)`); return `record.gcs_output_dir`. Authoritative for anything
   launched from this host. (Covers the overwhelming majority: the cron
   launches from the one operator laptop.)
2. **Iris fallback** — if not in the registry, `iris --config <cfg> query` the
   job's entrypoint command (or env) and extract the `--gcs-output-dir` /
   `--jobs-dir` argument. This is the cross-host path and the "verify iris
   retrievability" item in §7.
3. **Explicit override** — a `--output-dir gs://...` flag on each reader for the
   escape hatch (already how the uploader effectively works — it takes an
   explicit `--job_dir`).

This resolver returns the OUTER `.../ot-agent/<job>/` prefix. Because old jobs
recorded a multi-region URI and new jobs record a single-region URI, the
resolver transparently returns the correct bucket for **both** — this is what
makes the cutover flag-day-free (§4).

### 3.1 `scripts/iris/analyze_iris_harbor_job.py` — hardcoded, MUST change

- `GCS_ROOT = "gs://marin-models-us/ot-agent"` (line 70) is hardcoded, and
  `list_trial_trajectories` (line 958: `root = f"{GCS_ROOT}/{job_name}/{job_name}"`)
  and `fetch_harbor_result` (line 1036:
  `f"{GCS_ROOT}/{job_name}/{job_name}/result.json"`) build paths off it.
- Change: replace the module-global `GCS_ROOT` with a per-job resolution.
  `main()` already has the `job_id` and `--cluster`; resolve the job's output
  prefix via `resolve_job_output_dir(job_id, cluster_config=cluster)` once, pass
  it into `list_trial_trajectories`/`fetch_harbor_result` instead of the global.
  For a US-multi-region legacy job it returns `gs://marin-models-us/ot-agent/<job>`;
  for a new single-region job `gs://marin-us-east5/ot-agent/<job>`. Same code,
  right bucket. (Note the double-`<job>/<job>/` nesting the script already
  encodes stays as-is; only the root changes.)
- Effort: small (one resolver call + thread a param through two helpers).

### 3.2 Rescue / cleanup **skills** — hardcoded `gs://marin-models-{us,eu}`, MUST change

These are Markdown playbooks that instruct an agent to
`gsutil rsync gs://marin-models-{us,eu}/ot-agent/<job>/ /tmp/<job>_traces`,
probing *both* continent buckets by hand. With N single-region buckets, "probe
both" no longer finds the data. Update each to **resolve the bucket from the
registry/iris first** (a one-liner the skill runs), then rsync that exact URI:

- `.claude/skills/datagen-job-cleanup/SKILL.md` (§1 locate trace_jobs; the
  gs:// rescue trap at ~89-97).
- `.claude/skills/monitor-cron-sweep-iris/SKILL.md` (§ "Rescue mechanics
  (both)" ~115; §5 launch template ~127).
- `.claude/skills/monitor-restore-iris/SKILL.md` (rescue mechanics ~106).
- `.claude/skills/datagen-launch-iris/SKILL.md` (rescue ~129; the model-path
  guard at ~186 refers to *weights* -> leave pointing at `marin-models-us`).
- `.claude/skills/eval-agentic-launch-iris/SKILL.md` (~248, 278, 292, 303) and
  `.claude/skills/eval-agentic-cleanup/SKILL.md` (~307 "CHECK BOTH BUCKETS").
- `.claude/ops/iris/ops.md` (~45, 54, 142, 274, 290).

Recommended skill snippet (single source of truth), e.g.:

```
JOB=<job_name>
OUT=$(/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m hpc.iris.job_output_resolver "$JOB")
mkdir -p /tmp/${JOB}_traces
gsutil -m rsync -r "$OUT/" /tmp/${JOB}_traces/     # OUTER <job>/ — carries logs/*_literal.jsonl
```

The skills should stop saying "check both `{us,eu}`" and instead say "resolve
the recorded output URI; it already encodes the right (single- or multi-region)
bucket." Keep the existing `--served_model`, literal-yield, and
`count_populated_literal_rows>0` guidance verbatim.

### 3.3 `make_and_upload_trace_dataset.py` + `literal_correlator.py` — NO change

Both take an explicit `--job_dir` (local path or `gs://` URI) and the correlator
takes explicit `--literal_log`/auto-discovers `logs/*_literal.jsonl` relative to
`--job_dir`. Neither hardcodes a bucket. They stay correct as long as the
**caller** (the skill in §3.2) passes the resolver's URI. No code change; only
the calling skills change.

### 3.4 `hpc/iris_fetch_daemon.py` — NO change

Already resolves from `record.gcs_output_dir` (registry). Region-agnostic today.
It will fetch single-region outputs for new jobs and multi-region for old jobs
automatically. (Confirm the watchdog `result.json` probe at line 371 also uses
`record.gcs_output_dir` — it does.)

### 3.5 mirror — NO change (out of scope)

`mirror_hf_to_gcs.py` and `launch_hf_mirror.py` operate on the **model-weight**
mirror (`marin-models-{us,eu}`), which stays multi-region. Leave
`FANOUT_BUCKETS`/`DEFAULT_BUCKET`/`gcs_bucket_for_region` as-is. The former
OT-Agent Iris eval pre-cache helper was retired with the standalone Marin
agentic-evals package.

### 3.6 Dashboards / misc

No W&B/Grafana dashboard reads these GCS prefixes directly (outputs are pulled
to `~/.ot-agent/runs` by the daemon, or read via the resolver). If any ad-hoc
dashboard or script hardcodes `gs://marin-models-us/ot-agent`, route it through
the resolver. (Grep gate for CI: fail if a new `gs://marin-models-(us|eu)/ot-agent/<...>/`
**output** path literal appears outside the weight-mirror modules.)

## 4. Forward-only cutover (no flag day)

1. Land the write-side change (§2) + the resolver (§3.0) + reader changes
   (§3.1, §3.2) **together**. The resolver handles both URI shapes, so readers
   are updated *before or with* the first single-region launch — no window
   where a reader can't find data.
2. From the first launch after deploy, new jobs pin to single-region and record
   a single-region `gcs_output_dir`. In-flight/old jobs keep their **already
   recorded** multi-region URI; the resolver returns it; the fetch daemon and
   rescue skills keep working against multi-region for them. **No job is
   stranded and there is no cutover instant.**
3. `--resume-from` is inherently safe: it reuses `prev.gcs_output_dir` from the
   registry (launcher ~432), so a resumed old job stays on its original
   multi-region bucket (Harbor's stable-`jobs_dir` invariant holds); a resumed
   new job stays single-region. Either way the recorded URI is authoritative.
4. Old multi-region transient data drains via the existing cleanup agent + the
   lifecycle TTL (§5). We never move it.

## 5. Lifecycle / TTL recommendation

Prevent regrowth of transient prefixes and bound the drain of legacy data:

- Apply a GCS **Object Lifecycle** delete rule scoped to the transient output
  prefixes only — `ot-agent/` on each single-region bucket **and** on the legacy
  `marin-models-{us,eu}` (to auto-drain what the cleanup agent doesn't get). Do
  **not** apply it to `ot-agent/models/` or `ot-agent/datasets/` (weights +
  source data are durable inputs).
- Because GCS lifecycle rules are per-bucket (not per-prefix in the delete
  action selector beyond `matchesPrefix`), use a condition:
  `age > <TTL_days>` **AND** `matchesPrefix: ["ot-agent/<job-namespace>/"]`
  while **excluding** `ot-agent/models/` and `ot-agent/datasets/`. Since
  lifecycle has no "exclude prefix", structure transient outputs under a
  dedicated sub-prefix (they already are: job dirs are `ot-agent/<job>/`, models
  are `ot-agent/models/...`) and set `matchesPrefix` to the job namespace
  pattern — or, cleaner, add a `matchesSuffix`/naming convention. Simplest
  robust option: keep the `matchesPrefix: ot-agent/` rule but *also* keep the
  model/dataset mirror under a bucket where the rule does not apply. **Recommend
  a follow-up to confirm the exact prefix layout** so the TTL cannot reap
  weights.
- Suggested TTL: **30 days** for datagen/eval outputs (long enough to rescue a
  stranded job; the cron rescues within hours normally), **7 days** for
  `ot-agent/xla_cache/` (compile caches are cheap to regenerate and churn with
  every image/model change). Both are single-region, so even un-reaped data is
  at the cheaper rate.
- This is an ops action (bucket config), not a code change; capture the exact
  `gcloud storage buckets update --lifecycle-file=...` JSON in a follow-up ops
  log once prefixes are confirmed.

### 5.1 Exact TTL commands — NOT YET APPLIED (parent runs after review)

**Implementation note (2026-07-13):** the implementation pass did NOT apply any
lifecycle rule to a live bucket. Verified prefix layout with `gsutil ls`:

- `gs://marin-eu-west4/ot-agent/` contains **BOTH** `models/` (a live weight
  mirror, e.g. `.../ot-agent/models/Qwen/…`) **and** `xla_cache/`. So even a
  single-region output bucket can hold weights.
- `gs://marin-models-{us,eu}/ot-agent/` contain `models/` + `datasets/` +
  `xla_cache/` (weights + source data — durable, must never be reaped).
- `gs://marin-us-east5/ot-agent/`, `gs://marin-us-central1/ot-agent/` (spot-check)
  contain **no** `models/`/`datasets/` — transient outputs + `xla_cache/` only.

**Consequence:** GCS lifecycle has no "exclude prefix", so a broad
`matchesPrefix: ["ot-agent/"]` delete rule is **UNSAFE** on any bucket that also
holds `ot-agent/models/` (marin-eu-west4, marin-models-us, marin-models-eu) — it
would delete the weight mirror. Only two things are safe to automate today:

**(A) xla_cache 7d — UNAMBIGUOUS, safe on every bucket.** `matchesPrefix:
["ot-agent/xla_cache/"]` cannot overlap `models/`/`datasets/`. Write
`/tmp/lifecycle_xla_cache_7d.json`:

```json
{ "rule": [ { "action": {"type": "Delete"},
             "condition": {"age": 7, "matchesPrefix": ["ot-agent/xla_cache/"]} } ] }
```

Apply to ALL output + legacy buckets (lifecycle set REPLACES the whole config —
confirm each bucket has no other lifecycle rule you need first with
`gsutil lifecycle get gs://<bucket>`):

```bash
for b in marin-us-central1 marin-us-central2 marin-us-east1 marin-us-east5 \
         marin-us-west4 marin-eu-west4 marin-models-us marin-models-eu; do
  gsutil lifecycle set /tmp/lifecycle_xla_cache_7d.json gs://$b
done
```

**(B) job-output 30d — ONLY on verified weight-free single-region buckets.** These
buckets hold transient outputs + xla_cache only, so a broad `ot-agent/` 30d rule is
safe (xla_cache is caught earlier by rule (A) — GCS deletes when ANY rule matches).
Write `/tmp/lifecycle_outputs_30d.json`:

```json
{ "rule": [
    { "action": {"type": "Delete"}, "condition": {"age": 7,  "matchesPrefix": ["ot-agent/xla_cache/"]} },
    { "action": {"type": "Delete"}, "condition": {"age": 30, "matchesPrefix": ["ot-agent/"]} }
] }
```

**Pre-check each bucket is weight-free before applying (fail-closed):**

```bash
for b in marin-us-central1 marin-us-central2 marin-us-east1 marin-us-east5 marin-us-west4; do
  if gsutil ls gs://$b/ot-agent/models/ >/dev/null 2>&1 || gsutil ls gs://$b/ot-agent/datasets/ >/dev/null 2>&1; then
    echo "SKIP $b — holds models/ or datasets/, broad TTL unsafe"
  else
    gsutil lifecycle set /tmp/lifecycle_outputs_30d.json gs://$b
    echo "applied 30d+7d to $b"
  fi
done
```

**Do NOT put rule (B) on `marin-eu-west4`, `marin-models-us`, or `marin-models-eu`**
— they hold `ot-agent/models/` weights. Their legacy transient job outputs drain
via the existing cleanup agent (§Non-goals), not a broad TTL. A safe 30d TTL for
those requires first relocating transient outputs under a dedicated
`ot-agent/jobs/` sub-prefix (future code change; would also change recorded URIs,
so out of scope for this forward-only migration).

## 6. Canary test plan

Run one real datagen job end-to-end and verify each leg:

1. **Launch** a small datagen job on a US TPU variant with the region pin
   active (no `--gcs-output-dir` override), e.g. `--tpu v5p-8 --preemptible`
   with `ctx32k_verified.yaml`. Capture the `[iris] Region pin: ... (bucket
   gs://marin-us-east5)` line and the `[iris] Output:` line — assert it is a
   **single-region** bucket (`marin-us-east5`, not `marin-models-us`).
2. **Output lands single-region:** `gsutil ls gs://marin-us-east5/ot-agent/<job>/`
   shows the trial dirs + `logs/`; assert nothing was written to
   `gs://marin-models-us/ot-agent/<job>/`.
3. **xla_cache co-locates:** confirm `OT_AGENT_XLA_CACHE_BASE` in the job env /
   `gsutil ls gs://marin-us-east5/ot-agent/xla_cache/<cpu_tag>/<model_tag>/`
   is populated after the first compile.
4. **Registry records single-region URI:**
   `python -c "from hpc.iris_job_registry import get_latest_by_job_name as g;
   print(g('<job>').gcs_output_dir)"` -> `gs://marin-us-east5/ot-agent/<job>`.
5. **Resolver returns it:** `python -m hpc.iris.job_output_resolver <job>` ->
   same URI (and, with the registry row deleted, the iris fallback returns the
   same — proves §7 retrievability).
6. **Rescue finds + uploads:** run the datagen-job-cleanup flow using the
   resolver URI; assert the HF repo gets a non-zero row count and, for a
   `--record_literal` job, `Literal yield: X/Y` with `X>0` and
   `count_populated_literal_rows>0`.
7. **analyze_iris_harbor_job** against `<job>` produces §2 trace-progress stats
   (i.e. it read the single-region bucket via the resolver, not the hardcoded
   `marin-models-us`).
8. **Legacy still works:** run the resolver + analyze + a dry rescue against one
   *old* multi-region job; assert it still resolves to and reads
   `gs://marin-models-{us,eu}` — proves no flag day.

## 7. Risks

- **[TOP] Iris output-URI retrievability for cross-host jobs.** The local
  registry only has jobs launched on *this* host. If a job was launched
  elsewhere, resolution must fall back to iris (§3.0 step 2). **Verify** the
  exact iris query that returns the job's entrypoint command/env and that the
  `--gcs-output-dir`/`--jobs-dir` arg is parseable from it, before relying on
  it. Mitigation until verified: the cron launches from one laptop, so the
  registry covers the operational path; the iris fallback is belt-and-suspenders
  and the explicit `--output-dir` override is the escape hatch.
- **EU coverage gap.** Only `europe-west4 -> marin-eu-west4` is mapped (matches
  marin.yaml). A job pinned to any other EU region fails-fast (no single-region
  bucket). Today EU TPUs are effectively `europe-west4`, so this is acceptable,
  but if EU capacity expands to another region the map must be extended or those
  jobs will fall back to `DEFAULT_GCS_OUTPUT_ROOT` (`gs://marin-eu-west4`, still
  single-region and still EU — a safe fallback, but potentially cross-region
  within EU). Flag for review when EU regions grow.
- **Unmapped regions (`asia-*`).** Fail-fast by design; a job there can't get a
  co-located output bucket. Acceptable — we don't run there. Keep the explicit
  `ValueError`/`None` rather than a silent wrong-continent default.
- **Cross-region read if a reader guesses wrong.** Fully avoided *if* every
  reader uses the recorded URI. The residual risk is a *new* hardcoded path
  slipping in. Mitigation: the CI grep gate (§3.6) + deleting all
  `GCS_ROOT`-style module constants in favor of the resolver.
- **Weight-guard confusion.** Because `gcs_bucket_for_region` (multi-region)
  and `output_bucket_for_region` (single-region) now coexist, a future edit
  could wire the wrong one. Mitigation: clear docstrings + a unit test asserting
  `assert_yaml_regions_match_pin` still expects `marin-models-us` for a
  us-east5 pin (weights), while the launcher output is `marin-us-east5`.
- **TTL reaping weights.** If the lifecycle prefix is set too broad it could
  delete the model mirror. Mitigation: confirm prefix layout, scope the rule to
  the job namespace, exclude `ot-agent/models` + `ot-agent/datasets` (§5).

## File-by-file change list + rough effort

| File | Change | Effort |
|------|--------|--------|
| `hpc/iris/regions.py` | Add `_REGION_TO_OUTPUT_BUCKET` + `output_bucket_for_region`; switch `region_local_output_prefix` + `discover_region_for_tpu` filter to it; keep `gcs_bucket_for_region`/weight-guard untouched | S |
| `hpc/iris/launcher.py` | line ~382: `gcs_bucket_for_region` -> `output_bucket_for_region` | XS |
| `hpc/iris/outputs.py` | none (`DEFAULT_GCS_OUTPUT_ROOT` already single-region `marin-eu-west4`) | none |
| `hpc/iris/env.py` | none (xla_cache derives from `gcs_output_dir` automatically); add a regression test | XS |
| `hpc/iris/job_output_resolver.py` (new) | registry-first, iris-fallback, explicit-override resolver + `__main__` CLI | M |
| `scripts/iris/analyze_iris_harbor_job.py` | drop hardcoded `GCS_ROOT`; resolve per-job via resolver; thread root into `list_trial_trajectories`/`fetch_harbor_result` | S |
| `.claude/skills/datagen-job-cleanup/SKILL.md` | rescue: resolve URI, not "both {us,eu}" | S |
| `.claude/skills/monitor-cron-sweep-iris/SKILL.md` | rescue mechanics + launch template guidance | S |
| `.claude/skills/monitor-restore-iris/SKILL.md` | rescue mechanics | XS |
| `.claude/skills/datagen-launch-iris/SKILL.md` | rescue rsync (leave weight model-path guard on `marin-models-us`) | XS |
| `.claude/skills/eval-agentic-launch-iris/SKILL.md`, `eval-agentic-cleanup/SKILL.md` | resolve URI; drop "CHECK BOTH BUCKETS" | S |
| `.claude/ops/iris/ops.md` | update the output-bucket narrative to single-region + resolver | S |
| tests (`tests/hpc/...`) | `output_bucket_for_region` map + unmapped fail-fast; xla_cache derivation; weight-guard unchanged | S |
| GCS lifecycle (ops, no code) | TTL on transient prefixes, single- + legacy multi-region | S (ops) |

Effort legend: XS < 30min, S ~1-2h, M ~half day. Total: ~1-1.5 days incl. canary.
No production code is written by this document.
