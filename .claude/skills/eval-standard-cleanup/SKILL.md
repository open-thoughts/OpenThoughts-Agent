---
name: eval-standard-cleanup
description: >-
  Consolidate FINISHED standard / lm_eval (evalchemy) math-suite eval jobs — the Delphi #6279
  MATH-500 / AIME24 / gsm8k grid launched via eval-standard-launch — into the SCORES.md tracker.
  Per job: confirm a NON-EMPTY seed42 result (a crash leaves it empty — don't record it as a score),
  rsync the results_*.json into the local per-model dir, write scalar partials, parse the scalars
  (MATH-500 acc×100 / AIME24 10-seed mean±sd / gsm8k strict+flex / Raw), and flip the SCORES.md row
  🚀 eval submitted → ✅ done. The artifact is SCALAR SCORES IN A TRACKER — HF-upload-only, NEVER DB.
  Use when asked to consolidate / harvest finished standard math-eval jobs or fill the scaling-laws
  score grid. DISTINCT from eval-agentic-cleanup (the Harbor-trace + Supabase DB path).
---

# eval-standard-cleanup

Post-run consolidation for **STANDARD evals** (lm_eval / evalchemy): the Delphi #6279 RL-scaling-laws
math suite — **MATH-500 (1 seed) + AIME24 (10-seed mean±sd) + gsm8k (strict+flex)** — launched by the
**`eval-standard-launch`** skill on Leonardo. The deliverable is **scalar scores in a tracker**: no HF
trace upload, no model upload, and **NEVER a DB registration**. This is the consolidation counterpart of
that launcher; for the agentic terminal-bench path (Harbor traces + Supabase) use **`eval-agentic-cleanup`**
instead — do **not** cross the two.

## 0. Reference files (source of truth — derive the flow from these)
Local notes dir: `/Users/benjaminfeuer/Documents/experiments/active/delphi/rl-scaling-laws-6279/`
- **`eval-standard-launch`** skill — the launch + `delphi_eval.sbatch` + the per-model output layout it
  produces (`…/experiments/delphi-eval/<RUN_NAME>/seed{42..51}/`). Read it for what's already on the cluster.
- **`main_sft_evals/SCORES.md`** — the master tracker for the 54-run main grid; its header describes the
  consolidation flow (per-model dirs `main_sft_evals/<basename>/`, partials `main_sft_evals/.partial/`).
- **`EVAL_CONVENTION.md`** §3.4 / §5.2-E — the per-cluster naming, the idempotent-skip `seed42` dir, the
  exact scalar-parse method, and the "COMPLETED is NOT proof — verify numeric scores" rule.
- The earlier cold-start grid is the same flow but a **different tracker** (`eval/SCORES.md`, artifacts
  `eval/<RUN>/`) — don't cross-file; the 54-run main grid lands in `main_sft_evals/`.

## 1. Per finished job — verify a real, non-empty result (do NOT record a crash as a score)
A SLURM `COMPLETED` state is necessary but **not sufficient**: evalchemy catches engine errors, exits 0,
and writes an **empty `results: {}` JSON** (silent drop). For each job:
1. Confirm terminal state via `sacct -j <jobid> --format=State -n -P | head -1` (the source of truth, not
   log-tailing).
2. Confirm the job produced a **NON-EMPTY `seed42` result** — the §3.4 idempotent-skip dir
   (`…/delphi-eval/<RUN_NAME>/seed42/`) must exist and its `results_*.json` must carry **numeric** scores
   (a crashed/timed-out run leaves an empty or missing `seed42`). An empty/missing seed42 → leave the row
   **pending** and note it; **do not fabricate a score**. (If gsm8k failed *after* MATH-500 wrote seed42,
   the row's seed42 exists but is partial — note it; the launcher's fix is to delete seed42 and re-run, not
   a cleanup action here.)

## 2. Consolidate — rsync results into the local per-model dir + write scalar partials
Pull **only** the per-task `results_*.json` (the `examples` payloads live inside the JSON, ~1-2 MB each;
skip the multi-MB trace subdirs). For the main grid the destination is `main_sft_evals/<basename>/`:
```bash
RUN=delphi-9e19-p33m67-k0p20-lr83-a002-magpie_lr1e5-sft   # = the SFT model basename, maps 1:1 to the row
DEST=/Users/benjaminfeuer/Documents/experiments/active/delphi/rl-scaling-laws-6279/main_sft_evals/$RUN
mkdir -p "$DEST"
rsync -avz --prune-empty-dirs --include='*/' --include='results_*.json' --exclude='*' \
  Leonardo:/leonardo_work/AIFAC_5C0_290/bfeuer00/experiments/delphi-eval/$RUN/ "$DEST/"
```
This pulls the MATH-500 `seed42` JSON, the gsm8k `seed42` JSON (written by the plain `lm_eval` step), and
the AIME24 `seed42..seed51` JSONs. Write the extracted scalars to `main_sft_evals/.partial/<basename>.json`
(the durable scalar record the table is consolidated from). Keep the rsynced JSONs in `<basename>/` as the
archive. Parse by loading the JSON and reading **only numeric keys** — never print the huge `examples` list.

## 3. Parse the scalars (the §5.2-E method — round to 1 decimal to match the table)
- **MATH-500** = `results["MATH500"]["accuracy"] × 100` (it's a fraction; 0.018 → 1.8). From `seed42`.
- **AIME24** = `results["AIME24"]["accuracy_avg"] × 100`, taken over the **10 seed dirs** (`seed42..seed51`);
  report **mean ± sample-stdev** of the per-seed values (×100). (e.g. `0.1±0.1`.)
- **gsm8k** (from the `lm_eval` output under `<basename>/seed42`): `exact_match,strict-match` → **gsm8k-S**
  and `exact_match,flexible-extract` → **gsm8k-F**, each ×100.
- **Raw** = `mean(MATH-500, AIME24, gsm8k-strict)`.
- Note any format-fail cell (empty `\boxed{}` / `model_answer:""`) — that's a real signal, not an eval bug.

The marin compiler `marin:experiments/evals/evalchemy_results_compiler.py` documents the exact JSON shape
(`results[task]…`) — reuse its extraction shape rather than rolling a new parse. (verify the precise key
names against the rsynced JSON on first use; evalchemy versions have drifted.)

## 4. Update the SCORES.md row — 🚀 eval submitted → ✅ done
Fill that row's `MATH-500 | AIME24 (mean±se) | gsm8k-S | gsm8k-F | Raw` cells from the partial and flip
`status` from `🚀 eval submitted` to `✅ done` (keep the eval job id in its column). Edit the **single row**
(rows are independent lines) and preserve the table format exactly. A crashed/empty-seed42 job stays
pending with a note (step 1) — never invent numbers.

## 5. HF-upload only — NEVER DB
These are **scalar scores in a tracker**, not a model/trace artifact. Do **NOT** call
`manual_db_eval_push.py` / touch Supabase — that is the **`eval-agentic-cleanup`** (agentic / Harbor-trace)
path. Results live in this experiment's docs + (optionally) as a `training_logs/`-style JSON on the model's
HF repo; there is no leaderboard row. (Per `project_delphi_sft_hf_only_no_db`.)

## 6. Disk / hygiene
The local artifacts are small (~1-2 MB JSONs); no large delete is needed. On the cluster, leave the
`delphi-eval/<RUN_NAME>/` dirs in place — they back the §3.4 idempotent skip, so a re-harvest or re-submit
stays safe. No `find` / `du` / `rglob` on GPFS; use canonical paths.

---
Launcher: **`eval-standard-launch`**. Contrast: **`eval-agentic-cleanup`** (Harbor-trace + Supabase DB —
the OTHER eval-cleanup path; this one is scores-in-a-tracker, HF-only, NO DB). Model-publishing cleanups:
**`rl-job-cleanup`** / **`sft-job-cleanup`**. Per-cluster particulars (ssh, the step-ca cert refresh for
rsync, the `/leonardo_work/AIFAC_5C0_290/bfeuer00/…` paths) → `.claude/ops/<cluster>/`.
