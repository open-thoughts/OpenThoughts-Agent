# TaskTrove — dataset tracker

Comprehensive inventory of **`open-thoughts/TaskTrove`** (HF `repo_type="dataset"`), the OpenThoughts-Agent
task-dataset collection (complement to `open-thoughts/AgentTrove` traces). Verified 2026-06-14.

- **Version: v3.2** · **116 subdirs** · **~2,362,708 tasks** total.
- **Structure:** one subdir per source dataset, named `org__name/` (the source HF repo `org/name` with `/`→`__`), containing **`tasks.parquet`** (columns `path`: str, `task_binary`: gzip-tar bytes). So a subdir maps back to its source repo by replacing the first `__` with `/` (e.g. `laion__exp_rpt_stack-csharp-v5` → `laion/exp_rpt_stack-csharp-v5`).
- **Prior versions** resolvable at git tags `v1` / `v2` / (code-contests fix `v3.1`); v3.2 = swegym v2→v5 swap.
- **Usage:** `python -m scripts.datagen.extract_tasks_from_parquet --parquet open-thoughts/TaskTrove --output_dir $SCRATCH/tasks/tasktrove --on_exist overwrite`.

> **Caveats:** (1) `laion__nemotron-gym-agent-workplace-v2` does NOT follow the convention — its data is at `data/train-00000-of-00001.parquet` (297 rows, same schema), so `extract_tasks_from_parquet` may not pick it up like the other 115. (2) `task_binary` is stored as parquet `binary`; the brief calls it "gzip tar bytes".

---

## A. Coding / SWE agentic task sets (63)

The Harbor agentic coding/SWE task sets — the `exp_rpt_*` (repo-PR-test), `exp_rle_*` (reverse-loop-eng),
`exp_flat25_*` styles + the patched-validated SWE sets. Used for agentic RL/eval.

| Subdir | Rows | Notes |
|---|---|---|
| DCAgent2__nl2bash-tasks-cleaned-oracle | 1,570 | |
| DCAgent__code-contests-noblock | 8,728 | competitive coding |
| DCAgent__exp_rle_adversarial | 5,000 | |
| DCAgent__exp_rpt_crosscodeeval-java | 2,139 | |
| DCAgent__exp_rpt_curriculum-easy | 514 | |
| DCAgent__exp_rpt_curriculum-hard | 506 | |
| DCAgent__exp_rpt_curriculum-medium | 512 | |
| DCAgent__exp_rpt_e2egit-large | 5,000 | |
| DCAgent__exp_rpt_e2egit-v2 | 500 | |
| DCAgent__exp_rpt_issue | 4,830 | |
| DCAgent__exp_rpt_multifile | 4,907 | |
| DCAgent__exp_rpt_nemotron-cpp | 5,000 | |
| DCAgent__exp_rpt_nemotron-junit | 5,000 | |
| DCAgent__exp_rpt_pr | 4,793 | |
| DCAgent__exp_rpt_pymethods2test-large | 5,000 | the a3/explore-tis RL family base set |
| DCAgent__exp_rpt_pymethods2test-v3 | 500 | |
| DCAgent__exp_rpt_stack-dockerfile-v2 | 497 | |
| DCAgent__exp_rpt_stack-jest-large | 5,000 | |
| DCAgent__exp_rpt_stack-jest-v2 | 500 | |
| DCAgent__exp_rpt_stack-pytest-large | 5,000 | |
| DCAgent__exp_rpt_stack-pytest-v2 | 500 | |
| DCAgent__exp_rpt_unitsyn-python-large | 5,000 | |
| DCAgent__exp_rpt_unitsyn-python-v3 | 500 | |
| DCAgent__inferredbugs-sandboxes-verifier | 10,000 | |
| DCAgent__llm-verifier-freelancer | 10,000 | LLM-judge verifier |
| DCAgent__r2egym-patched-full-oracle | 3,328 | snapshot-optimized variant |
| DCAgent__selfinstruct-naive-sandboxes-2-verified | 9,638 | |
| DCAgent__swe_rebench_patched_oracle | 3,787 | |
| DCAgent__swe_rebench_v2_patched_oracle | 18,341 | |
| laion__exp_flat25_pseudocode-v2 | 728 | |
| laion__exp_flat25_speed_bonus-v2 | 764 | |
| laion__exp_flat25_stackoverflow-v2 | 765 | |
| laion__exp_flat25_subtle_debug-v3 | 289 | |
| laion__exp_rle_detailed-v3 | 413 | |
| laion__exp_rle_error_report-v3 | 261 | |
| laion__exp_rle_github_issue-v3 | 264 | |
| laion__exp_rle_heavy_padding-v2 | 784 | |
| laion__exp_rle_minimal_instructions-v3 | 233 | |
| laion__exp_rpt_codenet-python-v2 | 10,000 | |
| laion__exp_rpt_crosscodeeval-csharp-v4 | 1,768 | |
| laion__exp_rpt_defects4j-v3-v4 | 216 | |
| laion__exp_rpt_exercism-python-v2 | 133 | |
| laion__exp_rpt_ghactions-v3 | 9,930 | |
| laion__exp_rpt_methods2test-large-v2 | 4,472 | |
| laion__exp_rpt_methods2test-large-v3 | 4,472 | |
| laion__exp_rpt_pr-v2 | 4,793 | |
| laion__exp_rpt_scaffold-v2 | 4,861 | |
| laion__exp_rpt_stack-bash-v3 | 9,384 | |
| laion__exp_rpt_stack-bash-withtests-gpt5mini-v2 | 8,922 | |
| laion__exp_rpt_stack-bash-withtests-v2 | 8,922 | |
| laion__exp_rpt_stack-csharp-v5 | 9,485 | |
| laion__exp_rpt_stack-dockerfile-gpt5mini-v3 | 4,137 | |
| laion__exp_rpt_stack-go-v4 | 2,313 | |
| laion__exp_rpt_stack-junit-v6 | 872 | |
| laion__exp_rpt_stack-php-large-v6 | 3,789 | |
| laion__exp_rpt_stack-php-v2-v6 | 438 | |
| laion__exp_rpt_stack-ruby-v2 | 8,627 | |
| laion__exp_rpt_stack-rust-v2 | 9,987 | |
| laion__exp_rpt_taco-v2 | 10,000 | |
| laion__freelancer-projects-sandboxes-ta-rl-gpt-5-mini-v2 | 9,999 | |
| laion__freelancer-projects-sandboxes-ta-rl-gpt-5-nano-v2 | 10,000 | |
| laion__openswe-tasks-patched-v5-oracle-success | 17,504 | |
| **laion__swegym-tasks-patched-validated-v5** | **2,438** | **v3.2 (2026-06-14): replaced `laion__swegym-tasks-patched-validated-v2` (989 tasks)** |

## B. Curriculum mixes (15)

Blended/weighted task mixes for curriculum + reward-shaping ablations (the `mix_h*` hypotheses).

| Subdir | Rows |
|---|---|
| DCAgent__mix_h2_language_proportional | 4,135 |
| DCAgent__mix_h4_binary_easy | 2,010 |
| DCAgent__mix_h6_test_quality_top25 | 2,747 |
| laion__mix_baseline_uniform-v2 | 3,718 |
| laion__mix_h1_struggle_zone-v2 | 3,116 |
| laion__mix_h2_language_balanced-v2 | 4,506 |
| laion__mix_h5_skill_diverse-v2 | 3,166 |
| laion__mix_h7_raw_volume_5k-v2 | 3,718 |
| laion__mix_h8_adversarial_tests-v2 | 2,873 |
| laion__mix_h8_original_tests-v2 | 2,862 |
| laion__mix_h10_reward_binary-v2 | 2,862 |
| laion__mix_h10_reward_proportional-v2 | 2,873 |
| laion__mix_h10_reward_staged-v2 | 3,873 |
| laion__mix_h11_compositional_gradient-v2 | 3,873 |
| laion__mix_h11_single_skill_only-v2 | 2,873 |

## C. SankalpKJ oracle-filtered (3)

Large oracle-filtered nemotron/swesmith sets.

| Subdir | Rows |
|---|---|
| SankalpKJ__nemotron-code-oracle-filtered | 15,165 |
| SankalpKJ__nemotron-math-oracle-filtered | 114,280 |
| SankalpKJ__swesmith-oracle-filtered | 12,942 |

## D. Nemotron-Gym RLVR conversions (35)

The v3 additions — converted from `nvidia/Nemotron-Post-Training-v3` via the `data.nemotron_gym` framework
(instruction-following, math, science, knowledge, reasoning, multi-turn, safety, single-step agentic pivots).
Self-contained verifiers where a deterministic gold exists; LLM-judge where grading is subjective.

| Subdir | Rows |
|---|---|
| laion__nemotron-gym-agent-calendar | 3,358 |
| laion__nemotron-gym-agent-workplace-v2 | 297 ⚠ |
| laion__nemotron-gym-agentic-conversational-tool-use-pivot-v2 | 96,965 |
| laion__nemotron-gym-agentic-function-calling-pivot-v2 | 9,579 |
| laion__nemotron-gym-agentic-indirect-prompt-injection-v2 | 1,272 |
| laion__nemotron-gym-agentic-swe-pivot-v2 | 3,978 |
| laion__nemotron-gym-arc-agi-python-inductive | 10,000 |
| laion__nemotron-gym-arc-agi-transductive-v2 | 10,000 |
| laion__nemotron-gym-cfbench-v2 | 1,105 |
| laion__nemotron-gym-competitive-coding | 15,713 |
| laion__nemotron-gym-identity-following-v2 | 21,660 |
| laion__nemotron-gym-instruction-following-adversarial-v3 | 1,000 |
| laion__nemotron-gym-instruction-following-calendar | 8,387 |
| laion__nemotron-gym-instruction-following-citation | 9,033 |
| laion__nemotron-gym-instruction-following-freeform | 8,869 |
| laion__nemotron-gym-instruction-following-multiturnchat-v2 | 2,011 |
| laion__nemotron-gym-instruction-following-structured | 9,437 |
| laion__nemotron-gym-instruction-following-v2 | 46,391 |
| laion__nemotron-gym-inverse-ifeval-v2 | 1,000 |
| laion__nemotron-gym-knowledge-mcqa | 616,888 |
| laion__nemotron-gym-knowledge-openqa-v2 | 122,357 |
| laion__nemotron-gym-knowledge-web-search-mcqa | 2,915 |
| laion__nemotron-gym-litmus-bench | 5,232 |
| laion__nemotron-gym-math-advanced-calculations-v3 | 5,291 |
| laion__nemotron-gym-math-openmathreasoning | 112,867 |
| laion__nemotron-gym-math-stack-overflow | 436,307 |
| laion__nemotron-gym-math-v4 | 6,534 |
| laion__nemotron-gym-multichallenge-advanced-v2 | 1,068 |
| laion__nemotron-gym-multichallenge-vanilla-v2 | 1,050 |
| laion__nemotron-gym-qa-abstention-v2 | 3,150 |
| laion__nemotron-gym-reasoning-gym | 14,259 |
| laion__nemotron-gym-safety-v2 | 89,066 |
| laion__nemotron-gym-science-so-openq | 150,644 |
| laion__nemotron-gym-structured-outputs-v4 | 53,870 |
| laion__nemotron-gym-sysbench-v2 | 1,010 |

⚠ `agent-workplace-v2` stores its parquet at `data/train-00000-of-00001.parquet`, not `tasks.parquet`.

---

## Maintaining TaskTrove (add / replace a dataset)

Pattern (see the scripts in `.claude/projects/tasktrove/`, e.g. `_tasktrove_v3_add.py`): stage
`<org__name>/tasks.parquet` in a staging dir, update the root `README.md` version note (dot-bump, e.g.
v3.2→v3.3), then `HfApi().upload_folder(repo_id="open-thoughts/TaskTrove", repo_type="dataset", ...)`.
- **Additive** (new dataset): no `delete_patterns`.
- **Replace** (e.g. the swegym v2→v5 swap): pass `delete_patterns=["<old_subdir>/**"]` scoped to ONLY the old subdir, in the same commit. Always re-list the repo afterward to confirm the new subdir is present, the old is gone, and the count is as expected.
- Secrets: `source secrets.env` for `HF_TOKEN` (env var only). penfever has `write` on `open-thoughts`.
- This is an outward-facing write to a shared org repo — verify before declaring done (supervisor discipline).
