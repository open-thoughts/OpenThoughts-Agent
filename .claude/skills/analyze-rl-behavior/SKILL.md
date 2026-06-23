---
name: analyze-rl-behavior
description: >-
  Run the full RL behavioral-analysis pipeline (scripts/analysis/analyze_rl_behavior.py)
  on a trained RL model to understand WHAT changed vs its pre-RL baseline, WHY, whether
  it PERSISTS, and its EVAL impact. Use when asked to "analyze RL behavior", "compare
  pre/post RL", "what did RL change", or to produce the Q1–Q4 behavioral report + GPT-5
  judge for an `laion/...` (or any) RL checkpoint. Runs LOCALLY on the Mac (no GPU).
---

# analyze-rl-behavior

Orchestrates `scripts/analysis/analyze_rl_behavior.py` — a local pipeline that pulls a
trained RL model's eval traces + training logs from HF/Supabase and answers four research
questions, each writing into `--output-dir/<step>/`:

- **Q1 (what changed):** `behavioral_delta` (macro metrics + behavioral features) + `llm_judge_diff` (GPT-5 same-task pairwise) + optional `annotate_failure_modes`.
- **Q2 (attribution):** `temporal_trace_analysis` + `parse_skyrl_metrics` (RL reward/KL/grad-norm over time).
- **Q3 (persistence):** `eval_temporal_overlay` + `trace_pair_render` (side-by-side same-task pairs).
- **Q4 (eval impact):** `solve_rate_by_context`.

## Step 0 — preflight artifact check (ALWAYS do this FIRST)

Before running, verify the model's artifacts are all present — a missing one silently downgrades the run (skipped Q2/Q3) or wastes a full pass. For `laion/<MODEL>`:

```bash
source /Users/benjaminfeuer/Documents/secrets.env
# (a) model repo exists + has weights + training_logs + README
curl -s -H "Authorization: Bearer $HF_TOKEN" "https://huggingface.co/api/models/laion/<MODEL>" \
 | python3 -c "import sys,json;d=json.load(sys.stdin);s=[x['rfilename'] for x in d.get('siblings',[])] if 'error' not in d else None;print('MISSING/404') if s is None else print('files',len(s),'| safetensors',sum(f.endswith('.safetensors') for f in s),'| training_logs',sum(f.startswith('training_logs/') for f in s),'| README','README.md' in s)"
```
Checklist (decide BEFORE launching):
1. **Repo exists + ≥1 `.safetensors`** — else the model itself never landed (an RL-cleanup Step-6 miss); fix that first (re-upload weights from the Jupiter export), don't analyze a 404.
2. **`training_logs/` present** — required for Q2 `parse_skyrl_metrics`. If absent, either complete RL-cleanup Step 9 first (upload training_logs) or accept Q2-metrics will skip.
3. **RL-trace dataset exists** — find `<job_name>` from the model repo's `rl_config.json`, check `penfever/<job_name>` exists on HF (`/api/datasets/penfever/<job_name>`). If yes → pass `--rl-traces penfever/<job_name>` (enables Q2-temporal + Q3-overlay). If 404 → those two steps just won't plan (fine, note it).
4. **`--list-evals`** resolves a baseline/post-RL pair (run it — confirms Supabase has the eval jobs; pick/pin the benchmark if needed).
5. **Eval-repo write access for `--annotate-failure-modes`** — if the eval repos are under an org you can't write (e.g. `DCAgent2/3` as `penfever`), OMIT that flag (it 403s, wasted; see Cost section).

Only proceed to the run once 1 is satisfied; 2–3 determine which `--rl-traces`/Q2 steps you'll get; 5 determines whether to include `--annotate-failure-modes`.

## TL;DR invocation

Run from the repo root `/Users/benjaminfeuer/Documents/OpenThoughts-Agent`, otagent env, secrets sourced:

```bash
source /Users/benjaminfeuer/Documents/secrets.env

# 0. Preview what auto-resolve will pick (exits without running, no API spend):
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m scripts.analysis.analyze_rl_behavior \
  --model-repo laion/<MODEL> \
  --list-evals \
  --output-dir /Users/benjaminfeuer/Documents/notes/RL/<run>/<MODEL>

# 1. Dry-run (confirm the planned step list resolves cleanly — still no spend):
#    same as the full command below + --dry-run

# 2. FULL run (cost-incurring steps ON by default here — see "Cost" to disable):
/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m scripts.analysis.analyze_rl_behavior \
  --model-repo laion/<MODEL> \
  --rl-traces penfever/<RL_TRACE_DATASET> \
  --annotate-failure-modes --llm-judge \
  --llm-judge-max-pairs 30 --llm-judge-concurrent 4 \
  --output-dir /Users/benjaminfeuer/Documents/notes/RL/<run>/<MODEL> \
  > /Users/benjaminfeuer/Documents/notes/RL/<run>/<MODEL>/_run.log 2>&1
```

- **Always run from repo root**, with `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python -m scripts.analysis.analyze_rl_behavior` (the symlinked `python` doesn't work in the sandbox).
- **One model per `--output-dir`.** Running multiple models into the same dir collides their `<step>/` outputs — give each its own subdir.

## Environment / secrets (mandatory)

`source /Users/benjaminfeuer/Documents/secrets.env` first (NOT `~/secrets.env`). The pipeline needs:
- `SUPABASE_URL` + `SUPABASE_SERVICE_ROLE_KEY` — `--model-repo` auto-resolution (models + sandbox_jobs tables).
- `HF_TOKEN` — eval-trace datasets + `training_logs/` snapshots.
- `OPENAI_API_KEY` — BOTH GPT-5 steps (no `LITELLM_API_KEY` needed; `llm_judge` falls back to the OpenAI SDK).

## `--model-repo` auto-resolution (the clean entry point)

Given just `--model-repo`, the orchestrator calls `scripts.analysis.auto_resolve.resolve()` and autofills:
`--post-rl-eval`, `--baseline-eval`, their `*-ts`, and `--training-log-dir` (snapshotted from the model repo's `training_logs/` on HF). Explicit CLI values win on conflict. It does NOT resolve `--rl-traces`.

- Default `--eval-selection=largest-delta` picks, among matched benchmark pairs, the one with the biggest positive post−baseline score gain. Other modes: `largest-abs-delta` (catches regressions), `latest`, `benchmark` (pin via `--eval-benchmark`).
- `--list-evals` prints the matched / post-only / baseline-only pairs and exits — run it first to see (and, if needed, pin) the benchmark.
- Baseline-eval timestamp defaults to the base model's `training_end` in Supabase.

> ### ⚠️ CROSS-MODEL COMPARISON: PIN ONE BENCHMARK (`--eval-benchmark`) — do NOT use the default
> `largest-delta` is correct for **single-model** analysis ("what's this model's best eval pair?"), but it is **WRONG for comparing models to each other**: it picks a *different* benchmark per model — each model's *best-looking* one — so you end up comparing models on different yardsticks (and silently flattering each: a model that *regressed* on the shared benchmark can be surfaced via a different benchmark where it happened to gain). This is a real glitch that corrupted a 9-model study (2026-06-13): `arm0-tis-15` showed **+0.0156 on its cherry-picked dev_set_v2** but was **−0.0300 on swebench_verified_random_100**, while the hero was compared on swebench — not the same axis at all.
>
> **Rule:** whenever you run `analyze_rl_behavior` across ≥2 models for comparison, **pin every model to the same benchmark**:
> `--eval-selection benchmark --eval-benchmark <uuid>`
> Pick a benchmark **all** the models share (run `--list-evals` per model to confirm coverage) and that is **binary pass/fail** (clean binomial SE + paired McNemar) — e.g. `swebench_verified_random_100` (`cc1aca76-98f5-4964-8d0b-efcb716b39c5`) or `terminal_bench_2` (`34ab93c4-…`); avoid partial-credit sets like `dev_set_v2` (no clean SE). If no benchmark is universal, pin the max-coverage one and explicitly list the excluded models. Compare each model's delta only to its OWN benchmark's noise floor; never mix benchmarks in one ranking.

### `--rl-traces` is NOT auto-resolved — pass it for Q2/Q3

Without `--rl-traces`, the Q2 `temporal_trace_analysis` and Q3 `eval_temporal_overlay` steps are silently **not planned** (only `parse_skyrl_metrics` covers Q2, and only if `training_logs/` exists in the model repo). Find the right RL-trace dataset from the model repo's `rl_config.json` (`job_name` field) → `penfever/<job_name>`; verify it exists on HF (some models have none → 404, then Q2-temporal/Q3-overlay just won't run). Don't guess from HF search — same-recipe older runs have similar names.

## Cost-incurring steps (ON in the TL;DR; how to disable)

Two GPT-5 steps. **To disable, simply omit the flag** (both are opt-in):

- **`--llm-judge`** — GPT-5 pairwise same-task classification. Robust (per-pair JSON-repair fallback). Default model `openai/gpt-5-2025-08-07`, `--llm-judge-max-pairs 30`, `--llm-judge-concurrent 4`. Caches per-pair verdicts to `<out>/Q1_llm_judge_diff/llm_judge_cache.json` → re-runs are free. ~30 calls, ~3 min. **Keep this on** — it's the headline Q1 signal and cheap.
- **`--annotate-failure-modes`** — GPT-5 (`update_hf_failure_modes`, hardcoded default `gpt-5.1`) annotates failure modes on the baseline + post-RL eval rows, then **pushes the annotations back to the eval HF repo**. Populates `behavioral_delta`'s "Failure-mode distribution" section.

  **⚠️ Two real failure modes (observed on all three 2026-06-12 ablation runs):**
  1. **Write-back 403 on `DCAgent2`/`DCAgent3` eval repos** — the auth'd HF user is `penfever`, which can't write those orgs (and they're over public-storage quota). The GPT-5 work completes, the `--push` 403s, and because the step is `optional=True` it's logged "failed (rc=1) — non-fatal" and skipped — **so the annotations are never persisted and `behavioral_delta` shows 0% failure-mode coverage. The full GPT-5 annotation budget (~tens-to-100+ batch calls over ~640 rows) is spent for zero usable output.** It only pays off for a model whose eval repos YOU own/can write.
  2. **Crashes on malformed GPT-5 JSON** — `update_hf_failure_modes.py:~191` does `json.loads(content)` with no per-batch try/except and no client timeout; one bad response (e.g. `Invalid \escape`) aborts the whole step. (A guard + retry + timeout there would fix both #1's wasted-spend visibility and this.)

  **Recommendation:** include `--annotate-failure-modes` only when the eval repos are writable by the authed HF user; otherwise omit it (saves the bulk of the runtime + cost; the failure-mode diff won't populate anyway).

## Step / Q mapping (what to expect)

| Step | Needs | Output |
|---|---|---|
| Q0.annotate_failure_modes.{baseline,post-rl} | `--annotate-failure-modes` + writable eval repo | annotations pushed to eval repo; local `Q0_failure_mode_*/done.txt` only on rc=0 |
| Q1.behavioral_delta | always | `Q1_behavioral_delta/report.{md,json}` |
| Q1.llm_judge_diff | `--llm-judge` | `Q1_llm_judge_diff/report.{md,json}`, `llm_judge_cache.json` |
| Q2.parse_skyrl_metrics | `training_logs/` in model repo (auto-snapshotted) | `Q2_skyrl_metrics/` CSVs + report + reward_vs_steps.png |
| Q2.temporal_trace_analysis | `--rl-traces` | temporal plots |
| Q3.eval_temporal_overlay | `--rl-traces` | overlay.png |
| Q3.trace_pair_render | always | `Q3_trace_pairs/pairs.html` (multi-MB) |
| Q4.solve_rate_by_context | always | `Q4_solve_rate_by_context/solve_rate.png` |

Top-level always: `INDEX.md` (cross-links every step — **written LAST; it is the reliable completion marker**), `pipeline_plan.json`, `auto_resolve.json`, `_orchestrator_run.log`.

## Re-running to fill skipped steps (partial-run / late-arriving inputs)

Common case: the first run skipped Q2 (`parse_skyrl_metrics`, `temporal_trace_analysis`) and/or Q3 (`eval_temporal_overlay`) because `training_logs/` wasn't in the model repo yet or `--rl-traces` wasn't passed. Once those inputs land (e.g. the RL-cleanup Step-9 upload finishes, or you locate the trace dataset), **re-run the same command with the missing inputs supplied** to fill the gaps:

- Steps that were **never planned** (Q2-temporal / Q3-overlay when `--rl-traces` was absent; `parse_skyrl_metrics` when no `training_logs/`) wrote **no marker**, so they run on the re-run automatically — no `--force` needed. Just pass `--rl-traces <hf-id>` and make sure `training_logs/` now exists in the model repo.
- Steps that already produced output (`behavioral_delta`, `llm_judge_diff`, `trace_pair_render`, `solve_rate_by_context`) are **skipped** (marker exists) — fine, they don't depend on the late inputs. `llm_judge_diff` re-hits its cache (free) if it does re-run.
- Use `--force` ONLY to refresh a step whose marker exists but whose inputs changed — chiefly `behavioral_delta` after a successful `annotate_failure_modes` (the stale-cache trap below).
- **For pure fill re-runs, omit `--annotate-failure-modes`** — on eval repos you can't write (e.g. `DCAgent2`/`DCAgent3` as `penfever`) it only re-burns GPT-5 budget and 403s without populating anything. Keep `--llm-judge` (cached → free on re-run).

**Output-dir durability:** write `--output-dir` to a **dedicated per-model subdir**, NOT the `~/Documents/notes/...` *root* of a shared folder. A root-level run on this Mac (iCloud-synced `~/Documents`) was observed to not persist its Q-dirs/INDEX.md even after the orchestrator reported success — use `.../ablation_exploration_in_rl/<model>/` per model.

## Operational gotchas

- **Completion signal = the per-`--output-dir` `INDEX.md` file appearing.** Do NOT rely on process-liveness — on this Mac (`~/Documents` iCloud + sandbox `/tmp` namespace) `pgrep`/`kill -0`/`ps` from background/monitor shells return phantom "process gone", and tqdm/logging stdout is block-buffered and lags minutes. Poll for `INDEX.md` (or the terminal `Q4_solve_rate_by_context/solve_rate.png`) via a foreground loop, not the `Monitor` tool.
- **Do NOT pipe the orchestrator stdout through `grep`/`tail`** — that triggers auto-backgrounding and the output is lost. Use a plain `> log 2>&1` redirect.
- **Do NOT launch duplicate concurrent runs** against the same `--output-dir`/eval repo — they share the OpenAI budget + HF `--resume` state and clobber the same log.
- **Resumable:** skip-if-output-marker-exists per step + the llm_judge cache. Killing and relaunching is safe; a re-run skips completed steps and the judge is 30/30 cache hits (free). **Stale-cache trap:** if `Q1_behavioral_delta/report.md` already exists, behavioral_delta is skipped — so a *later* successful annotation won't refresh the failure-mode diff without `--force` (or deleting the report).
- **Runtime:** ~3–10 min with `--llm-judge` only; ~10–45 min with `--annotate-failure-modes` (it annotates ALL eval rows sequentially via GPT-5 — the slow part, even when it ultimately 403s).

## Worked example (2026-06-12)

Three pymethods2test ablation checkpoints, each `--model-repo laion/<m> --annotate-failure-modes --llm-judge` into its own subdir under `notes/RL/ablation_exploration_in_rl/`:
- `ablation-pymethods2test-seqmean-arm0-tis-15-8B` → judge **66.7%** post-RL win (Q2 ran — had training_logs).
- `ablation-pymethods2test-shaped-45-8B` → judge **53.3%** post-RL win.
- `ablation-pymethods2test-seqmean-arm0-30-8B` → judge **50/50**; behavioral_delta showed reward dipped 0.457→0.379.
All three: `--annotate-failure-modes` 403'd on the DCAgent2/3 eval repos (failure-mode section empty); `--llm-judge` succeeded 30/30. `--rl-traces` had to be passed/located via `rl_config.json`; absent for two → Q2-temporal/Q3-overlay skipped.

---

## Operating notes (folded from memory 2026-06-14)

- **teacher_hint marker:** the `teacher_hint` PRM (`prm/teacher_hint.py`) injects hints into the student's **observation text** — NOT a separate `steps[].source` in the ATIF trajectory. Grep for the literal `[HINT FROM TEACHER]:` (wrapped by `\n\n[HINT FROM TEACHER]: ` … `\n\n`). Appears in `agent/trajectory.json` (substring of an agent-source step), `agent/episode-N/prompt.txt` (the prompt AFTER the hint fired), and maybe `episode-N/debug.json`. Fires every `check_interval` turns (default 5; prod used 8), skipped if `turn < min_turns` (default 3; prod 4), and **silently returns None if the teacher engine fails to init/generate** — so absence of the marker doesn't distinguish "not eligible" from "engine failed"; cross-reference turn count + `trial.log`/`exception.txt`.
