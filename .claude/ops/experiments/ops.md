# Experiments workspace — `/Users/benjaminfeuer/Documents/experiments`

The local per-experiment workspace on this Mac. **Each experiment / experiment-series gets its own
subdirectory here**, and each subdirectory typically holds **its own tracker(s)** — the source-of-truth
state for that experiment (queue, status table, per-row results, skipped items, plots, reports).

> **active/ vs complete/ split (2026-06):** experiments are now bucketed one level down —
> in-flight series live under **`active/`** (e.g. `active/delphi/`, `active/datagen/`,
> `active/ablation_exploration_in_rl/`) and finished series under **`complete/`**
> (e.g. `complete/a1/`, `complete/a3/`, `complete/gsm8k_grid_leonardo/`, …). So an experiment's
> subdir is `experiments/active/<name>/` or `experiments/complete/<name>/`. New / running work goes
> under `active/`; move a series to `complete/` once it concludes.

> Distinct from other local dirs: this is **per-experiment working state + trackers**.
> `notes/` is the broader knowledge base; `agent_logs/` is dated failure/remediation logs; the cluster-side
> `experiments/` dirs (under each repo checkout on Jupiter/Leonardo) hold the actual run artifacts
> (`logs/`, `configs/`, `sbatch/`, `checkpoints/`). This Mac dir is where the human-readable trackers live.

## Convention
- **One subdirectory per experiment or series**, named for the experiment (e.g. `a3/`, `ablation_exploration_in_rl/`, `gsm8k_grid_leonardo/`, `iris_capacity/`, `delphi/`, `datagen/`, `chat_templating/`, `cluster_timing_comparison/`).
- **Trackers live inside the subdir**, usually `*.md` — and a series often has several: a queue/plan, a status/results tracker, a skipped-list, a report, plus subfolders for plots/per-run dirs. Examples seen in the tree:
  - `a3/` → `a3_rl_tracker.md` (status), `a3_rl_experiments.md` (launch log), `a3_skipped_datasets.md`, `reward_plots/`, a PDF report.
  - `ablation_exploration_in_rl/` → `HERO_LEARNED_BEHAVIORS.md`, `COMPARISON_SWEBENCH_PINNED.md` + per-run subdirs (`hero_rl_run/`, `explore-tis-*-8B/`, `shaped-45-8B/`, …).
  - `gsm8k_grid_leonardo/` → `grid.md` (plan), `accuracy_grid.md` (results), `grid_experiment_log.md`.
  - `iris_capacity/` → `iris_capacity_analysis.md` + interim/batch trackers.
- **Tracker naming is not rigid** — `*_tracker.md` / `grid.md` / `notes.md` / `DESIGN.md` / `*_log.md` all appear. When working an experiment, **read the subdir's `*.md` files first** to find its tracker; treat the one the user points at (or the most status-like) as source of truth.

## How to use it
- **Starting a new experiment:** create `experiments/active/<name>/` and a tracker inside it; record the queue/plan and update status as runs land. (Some series keep their canonical tracker elsewhere — e.g. the MiniMax datagen tracker lives under `experiments/active/datagen/...` per `.claude/projects/daytona` / the datagen skills — follow the pointer the experiment itself gives.)
- **During a cron sweep / cleanup:** when a run for an experiment completes or changes state, update that experiment's tracker here (status table, results, reward plots) in addition to the global experiment log (`notes/claude/claude_experiments.md`).
- **This is local working state** — not git-tracked in the OT-Agent repo; don't pull large artifacts/checkpoints here (keep those on cluster scratch). Trackers + small plots/reports only.
