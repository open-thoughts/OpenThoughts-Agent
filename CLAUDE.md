# CLAUDE.md

Guidance for Claude Code when working in this repository.

**ot-agent** is a distributed LLM training + evaluation system for HPC clusters, with four subsystems —
**datagen** (Harbor/Daytona traces + standard vLLM generation), **SFT** (LLaMA-Factory), **RL** (SkyRL/GRPO),
**eval** (terminal-bench / agentic). One unified launcher: `python -m hpc.launch --job_type <type>`.

## Source of truth = `.claude/`

This file is a thin index. The real, maintained documentation lives under **`.claude/`** — read the relevant
piece for the task at hand (skills are also invocable by name via the Skill tool):

- **`.claude/skills/<name>/SKILL.md`** — operational how-tos, one per task. By prefix:
  - **launch:** `rl-agentic-launch-jupiter`, `rl-standard-launch-leonardo`, `sft-launch-jupiter`, `sft-launch-leonardo`, `datagen-launch` (agentic Harbor trace-gen), `datagen-standard-launch` (non-agentic: Curator sharded + `generate.py`/`generate_abstract.py`), `eval-agentic-launch`, `eval-standard-launch` (+ the `*-iris` variants).
  - **cleanup:** `rl-job-cleanup`, `sft-job-cleanup`, `datagen-job-cleanup`, `eval-agentic-cleanup`, `eval-standard-cleanup`.
  - **monitor:** `monitor-cron-sweep` (the sweep procedure), `monitor-job-tables` (the status-table formats + metrics/red-flags), `monitor-restore` (re-create the 3-hourly Jupiter+Leonardo sweep loop), `monitor-restore-iris-cron`.
  - **analysis / data / db:** `analyze-rl-behavior`, `analyze-dataset-token-length`, `datagen-reduce-dataset-snapshots`, `crud-otagent-supabase`.
  - **code (staged change workflow):** `code-create-staged-plan` (design a multi-stage codebase change → `notes/<codebase>/`), `code-execute-staged-plan` (run it gate-by-gate, log progress → `agent_logs/`).
  - **role / bootstrap:** `supervisor-init` — assume the supervisor role at session start (orient, load env, take custody of secrets + codebase ground truth, survey in-flight work, stand up monitoring).
- **`.claude/projects/<dep>/`** — what each codebase/dependency is + its facts & gotchas: `ot-agent/` (this repo's branches + launcher map), `marinskyrl/`, `harbor/`, `vllm/`, `llama-factory/`, `axolotl/`, `daytona/`, `ajudge/`.
- **`.claude/ops/<target>/`** — machine/cluster particulars (access, paths, env/SIF map, gotchas): `jupiter/`, `leonardo/`, `torch/`, `iris/`, `local/` (this Mac), `all/` (cross-cluster HF/tmux), `experiments/` (the per-experiment tracker workspace `~/Documents/experiments`), `data/` (dataset trackers — e.g. `tasktrove.md`, the full TaskTrove inventory).
- **`.claude/secret.md`** — untracked, gitignored; holds privileged values (pinggy bank, etc.) pulled out of the committable docs. Referenced by name from skills/ops.

## Always (apply before any skill loads)

- **Run Python via the otagent env's full interpreter path** (symlinks don't work in the sandbox): `/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python`. (`curator` env only for Curator datagen.)
- **Syntax/lint check** with the IDE MCP tool `mcp__ide__getDiagnostics`, NOT `python -m py_compile`/`flake8` (bash output capture is unreliable here).
- **Local clones are ground truth; clusters never diverge.** All code/config edits go in the local checkouts (`harbor`, `MarinSkyRL`, `OpenThoughts-Agent` on `penfever/working`; `vllm` fork) → commit → push → `git pull` on the cluster (the three Python repos are editable installs, live after pull). **No untracked/divergent changes on a cluster, ever; no patch-by-rsync; no hand-editing on the cluster.** **vLLM** (compiled) is **built from source on each cluster from the committed fork** — not rsync'd edits; some envs may run vanilla vLLM. (Details: `.claude/projects/<dep>/`, `.claude/ops/local/ops.md`, and the `supervisor-init` skill's codebases section.)
- **Standing ML-ops guardrails** (full statements in `monitor-restore` / the cleanup skills): `enable_db_registration: false` in YAMLs (manual DB register only); ≤6 RUNNING RL jobs per cluster (Daytona); a3 series CONCLUDED; Daytona snapshot caps are HARD (clean stale, never raise); cross-user FK safety pre-check before any Supabase delete/mutate; HF uploads default PUBLIC to `laion/`; never kill a RUNNING job without explicit permission.
