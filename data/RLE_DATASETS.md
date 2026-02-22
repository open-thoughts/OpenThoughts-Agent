# All Datasets on HuggingFace

**Updated**: 2026-02-21
**Organization**: DCAgent
**Source**: DCAgent/exp_rpt_stack-pytest (9,985 pytest-based coding tasks)
**Agent**: terminus-2 via harbor/Daytona
**Models**: gpt-5-nano-2025-08-07, gpt-5-mini-2025-08-07, gpt-5-2025-08-07

## 5,000-Task Datasets (18 datasets, 90,000 total tasks)

Scaled from 500 to 5,000 tasks per dataset on 2026-02-21. Pass rates from earlier 25-task harbor tests (on the 500-task versions).

### Truly Distinct (different tests, structure, or rewards)

| # | HF Repo | Tasks | Nano | Mini | GPT-5 | Type |
|---|---------|-------|------|------|-------|------|
| 1 | `DCAgent/exp_rle_proportional` | 5,000 | 40% | 60% | 80% | Proportional reward |
| 2 | `DCAgent/exp_rle_adversarial` | 5,000 | 20% | 20% | 40% | Different tests (edge cases) |
| 3 | `DCAgent/exp_rle_2skill` | 5,000 | 20% | 40% | 40% | Different tests (compositional) |
| 4 | `DCAgent/exp_rle_partial_ambiguity` | 5,000 | 20% | 40% | 40% | Different tests (property-based) |
| 5 | `DCAgent/exp_rle_structural_debug` | 5,000 | 20% | 20% | 20% | Debug broken code |
| 6 | `DCAgent/exp_rle_curated` | 5,000 | 40% | 60% | 60% | Top-scored source tasks |
| 7 | `DCAgent/exp_flat25_baseline` | 5,000 | 12% | 24% | 28% | Raw source (control) |
| 8 | `DCAgent/exp_flat25_subtle_debug` | 5,000 | 20% | 20% | 20% | Subtle bugs to fix |

### Instruction Paraphrases (same tasks pass/fail — redundant with each other)

| # | HF Repo | Tasks | Nano | Mini | GPT-5 | Type |
|---|---------|-------|------|------|-------|------|
| 9 | `DCAgent/exp_rle_error_report` | 5,000 | 20% | 60% | 60% | Bug report style |
| 10 | `DCAgent/exp_rle_minimal_instructions` | 5,000 | 0% | 20% | 20% | Bare minimum |
| 11 | `DCAgent/exp_rle_github_issue` | 5,000 | 40% | 60% | 60% | GitHub issue style |
| 12 | `DCAgent/exp_rle_detailed` | 5,000 | 40% | 40% | 60% | Step-by-step |
| 13 | `DCAgent/exp_rle_expert` | 5,000 | 40% | 60% | 60% | Terse expert |
| 14 | `DCAgent/exp_rle_heavy_padding` | 5,000 | 40% | 60% | 60% | Verbose noise |
| 15 | `DCAgent/exp_rle_moderate_padding` | 5,000 | 40% | 60% | 60% | Moderate noise |
| 16 | `DCAgent/exp_flat25_pseudocode` | 5,000 | 20% | 20% | 20% | Pseudocode only |
| 17 | `DCAgent/exp_flat25_stackoverflow` | 5,000 | 20% | 28% | 20% | SO question style |
| 18 | `DCAgent/exp_flat25_speed_bonus` | 5,000 | 16% | 24% | 28% | Speed bonus suffix |

### 25-Task Results Not Uploaded (redundant — same pass/fail pattern)

| Experiment | Nano | Mini | GPT-5 |
|---|---|---|---|
| 1.2_rich | 20% | 24% | 28% |
| 7.1_d5_bare | 24% | 28% | 28% |
| 8.10_intermediate | 20% | 28% | 28% |
| 8.10_novice | 24% | 28% | 28% |
| 8.1_code_review | 20% | 28% | 24% |
| 8.1_slack_message | 24% | 24% | 28% |
| 8.2_failing_test | 24% | 28% | 28% |
| 8.2_nl_prose | 20% | 24% | 20% |
| 8.5_mild | 24% | 28% | 28% |

The same 7 tasks (0000, 0001, 0002, 0008, 0011, 0012, 0023) pass across virtually all of these. Differences are 1-task noise at 4% granularity.

## Summary

**Total on HuggingFace: 18 datasets (90,000 tasks)**
- 8 truly distinct datasets x 5,000 tasks = 40,000
- 10 instruction paraphrase datasets x 5,000 tasks = 50,000