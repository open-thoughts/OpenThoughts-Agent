# Experiment Results Summary (V5 — Proportional Reward Fix)

**Generated**: 2026-02-17
**Tasks per experiment**: 5 (except 1.1_70_20_10=2, 1.1_goldilocks=3, 1.3_top50=10, 8.9_pairs=2)
**Models tested**: gpt-5-nano-2025-08-07, gpt-5-mini-2025-08-07, gpt-5-2025-08-07
**Agent**: terminus-2 via harbor/Daytona
**Source dataset**: DCAgent/exp_rpt_stack-pytest (pytest-based coding tasks)
**Comparison**: All models run on identical task directories with identical patched test.sh files

## Infrastructure Fixes Applied

### V3 Fixes
- **Removed 3 broken source tasks**: scopez_server (Go binary), smlb (missing package), TestPlotBase (removed pandas API)
- **Added hypothesis to test.sh whitelist**: Property-based tests now work
- **Fixed 7.1_d5_bare Dockerfile**: Added python3, pip, venv
- **Removed 7.5_staged and 7.1_d1_full**: Still broken / empty after regeneration

### V4 Fixes
- **Added `requests_mock` to test.sh whitelist**: Task 0004 tests using `requests_mock` can now install it
- **Regenerated 8.8_devops, 8.8_ecommerce**: LLM-generated test files had syntax errors (escaped quotes, literal `\n`)
- **Regenerated 5.4_2skill**: Test file had escaped quote syntax errors
- **Regenerated 8.4_2constraints**: Test file had indentation error
- **Fixed 8.4_1constraint, 8.4_3constraints, 8.4_4constraints**: Added missing `import pytest`
- **Added `_clean_llm_test_code()` to exp_8_utils.py**: Auto-strips markdown fences, adds pytest imports, validates syntax

### V5 Fixes
- **Fixed 2.1_proportional test.sh**: Custom proportional-reward test.sh was missing venv setup, PYTHONPATH, pytest install, and whitelist dependency installer. Added all standard infrastructure while preserving proportional reward logic. Result: **0% → 40/60/80%** across nano/mini/GPT-5.

## Overall Results

| Metric | gpt-5-nano | gpt-5-mini | gpt-5 |
|---|---|---|---|
| Experiments tested | 46 | 46 | 46 |
| Non-zero pass rate | 38 (83%) | 40 (87%) | 38 (83%) |
| Overall avg pass rate | **38%** | **42%** | **42%** |
| vs Nano wins/ties/losses | — | 11 / 34 / 1 | 12 / 32 / 2 |
| vs Mini wins/ties/losses | — | — | 2 / 42 / 2 |
| Errors (avg per 5 trials) | 1.6 | 0.3 | 0.0 |

> **Key finding**: GPT-5 and GPT-5-mini produce virtually identical pass rates (42% vs 42%). The model size difference between mini and full GPT-5 does not meaningfully affect task completion on this benchmark. GPT-5-nano is slightly weaker (38%) and has significantly more errors.
>
> **Underlying task difficulty dominates**: Tasks 0000-0002 pass ~60% of the time regardless of model or transform. Tasks 0003-0004 almost never pass (~5%). This causes most experiments to cluster at 60% (3/5 easy tasks pass) or 0% (transform breaks even easy tasks).

---

## Experiment Descriptions & Pass Rates

### Theme 1: Data Curation
*How data selection and preprocessing affect agent performance.*

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **baseline_source** | Unmodified source tasks — control group | 60% | 60% | 60% |
| **1.1_70_20_10** | 70-20-10 difficulty split (2 tasks) | 100% | 100% | 100% |
| **1.1_goldilocks** | Goldilocks medium-difficulty selection (3 tasks) | 67% | 67% | 67% |
| **1.2_minimal** | Minimal instructions — bare minimum detail | 0% | 20% | 20% |
| **1.2_rich** | Rich instructions — extra context and hints | 60% | 60% | 60% |
| **1.3_top50** | Top 50% by LLM-scored test quality (10 tasks) | 40% | 40% | 40% |
| **1.4_deduplicated** | Deduplicated tasks | 60% | 60% | 60% |

### Theme 2: Reward Design

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **2.1_proportional** | Proportional reward — fraction of tests passed | **40%** | **60%** | **80%** |

### Theme 4: Skill Distillation

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **4.3_curated_1k** | LLM-curated high-quality 1K subset | 40% | 60% | 60% |

### Theme 5: Data Verification

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **5.2_adversarial** | Adversarial test cases probing edge cases | 20% | 20% | **40%** |
| **5.4_2skill** | 2-skill compositional tasks | **20%** | **40%** | **40%** |

### Theme 7: Curriculum & Format

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **7.1_d5_bare** | Bare Dockerfile — minimal Python only | 60% | 60% | 60% |
| **7.2_random_30** | 30% info dropout from instructions | 0% | 0% | 0% |
| **7.6_speed_bonus** | Bonus reward for fast completion | 60% | 60% | 60% |

### Theme 8.1: Style Transfer

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.1_code_review** | Code review comment style | 60% | 60% | 60% |
| **8.1_error_report** | Bug report style | 20% | 60% | 60% |
| **8.1_github_issue** | GitHub issue style | 40% | 60% | 60% |
| **8.1_slack_message** | Casual Slack message style | 60% | 60% | 60% |
| **8.1_stackoverflow** | StackOverflow question style | 60% | 60% | 60% |

### Theme 8.2: Specification Modality

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.2_failing_test** | "Make these tests pass" style | 60% | 60% | 60% |
| **8.2_io_examples** | Only input/output examples, no prose | 0% | 0% | 0% |
| **8.2_nl_prose** | Clear natural language paragraph | 60% | 60% | 60% |
| **8.2_pseudocode** | Pseudocode only, no natural language | 60% | 60% | 60% |
| **8.2_type_signatures** | Python type signatures and docstrings only | 60% | 20% | 20% |

### Theme 8.3: Instruction Granularity

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.3_bullets** | 3-5 requirement bullets | 40% | 40% | 40% |
| **8.3_detailed** | Step-by-step with file structure hints | 40% | 40% | **60%** |
| **8.3_vague** | Single vague sentence, no details | 0% | 0% | 0% |

### Theme 8.4: Constraint Ladder

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.4_1constraint** | stdlib only | 20% | 20% | 20% |
| **8.4_2constraints** | + single file | 20% | 20% | 20% |
| **8.4_3constraints** | + memory limit | 40% | 40% | 40% |
| **8.4_4constraints** | + time limit | 40% | 40% | 40% |

### Theme 8.5: Context Padding

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.5_mild** | 1-2 irrelevant sentences | 60% | 60% | 60% |
| **8.5_moderate** | Paragraph of context + red herrings | 40% | 60% | 60% |
| **8.5_heavy** | Verbose logs, misleading hints | 40% | 60% | 60% |

### Theme 8.6: Error State

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.6_structural** | Core logic broken | 40% | 60% | 60% |
| **8.6_subtle** | 1-2 subtle bugs | 60% | 60% | 60% |

### Theme 8.7: Ambiguity

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.7_minimal** | Only high-level goal + property tests | 20% | 20% | 20% |
| **8.7_partial** | Some details removed + property tests | 20% | 40% | 40% |

### Theme 8.8: Domain Shift

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.8_devops** | DevOps vocabulary | 0% | **20%** | 0% |
| **8.8_ecommerce** | E-commerce vocabulary | 0% | 0% | 0% |
| **8.8_financial** | Financial vocabulary | 20% | 20% | 0% |
| **8.8_medical** | Medical vocabulary | 20% | 20% | 20% |

### Theme 8.9: Task Stacking

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.9_pairs** | 2 tasks combined, all tests must pass (2 tasks) | 0% | 0% | 0% |

### Theme 8.10: Knowledge Gradient

| Experiment | Description | Nano | Mini | GPT5 |
|---|---|---|---|---|
| **8.10_expert** | Terse, assumes complete domain knowledge | 40% | 60% | 60% |
| **8.10_intermediate** | Explains algorithm steps | 60% | 60% | 60% |
| **8.10_novice** | Fully explained with pseudocode | 60% | 60% | 60% |

---

## Fix Impact History

### V4: Test file fixes

| Experiment | Fix Applied | Before | After (Nano/Mini/GPT5) |
|---|---|---|---|
| **5.4_2skill** | Regenerated (syntax error in test file) | 0% / 0% / 0% | **20% / 40% / 40%** |
| **8.8_devops** | Regenerated (syntax errors in 3/5 test files) | 0% / 0% / 0% | 0% / 20% / 0% |

### V5: Proportional reward test.sh fix

| Experiment | Fix Applied | Before | After (Nano/Mini/GPT5) | Avg Reward |
|---|---|---|---|---|
| **2.1_proportional** | Added venv, PYTHONPATH, pytest, whitelist to custom test.sh | 0% / 0% / 0% | **40% / 60% / 80%** | 0.40 / 0.49 / 0.56 |

> **V5 highlight**: 2.1_proportional is now the most interesting experiment — it shows clear monotone scaling (40→60→80%) AND provides proportional rewards. The average reward (0.40→0.49→0.56) captures partial progress that binary pass/fail misses.

---

## Key Findings

### Model Comparison
1. **GPT-5 ≈ GPT-5-mini >> GPT-5-nano**: Full GPT-5 and mini produce nearly identical results (42% vs 42%). Nano trails at 38%.
2. **GPT-5 has zero errors**: Nano averages 1.6 errors per 5 trials, mini 0.3, GPT-5 0.0. The bigger model is more reliable.
3. **Scaling doesn't help on this benchmark**: The gap between nano (38%) and GPT-5 (42%) is only 4 percentage points. Task difficulty, not model capability, is the bottleneck.

### What Makes Tasks Harder (transforms that drop below 60% baseline)
4. **Domain shift is devastating (0-20%)**: Replacing standard vocabulary with domain-specific terms causes near-total failure for all models.
5. **Compound tasks are hard (0-40%)**: 8.9_pairs (0%) require solving 2 tasks combined. 5.4_2skill (20-40%) adds file I/O requirements.
6. **IO-only specs fail (0%)**: Without any prose, just examples isn't enough.
7. **Vague instructions fail (0%)**: A single vague sentence is too underspecified.
8. **Minimal instructions hurt (0-20%)**: Stripping detail from instructions significantly hurts.
9. **30% info dropout fails (0%)**: Randomly removing information is too destructive.
10. **Constraints hurt modestly (20-40%)**: Adding stdlib/file/memory/time constraints reduces pass rates uniformly across models.

### What Doesn't Matter (stays near 60% baseline)
11. **Style transfer is robust (60%)**: Rewriting as GitHub issue, Slack, SO, etc. doesn't change outcomes (for mini/GPT-5; nano drops on some).
12. **Context padding doesn't hurt (60%)**: Even heavy noise doesn't degrade mini/GPT-5 performance.
13. **Knowledge level barely matters (60%)**: Expert, intermediate, novice all similar.
14. **Debugging works (60%)**: Starting with broken code doesn't prevent success.
15. **Speed bonus is neutral (60%)**: Adding time incentives doesn't change pass rates.

### Still 0% for All Models (4 experiments)

| Experiment | Likely Cause |
|---|---|

| gpt-5-mini | 0.3 | 37/46 (80%) |
| gpt-5 | 0.0 | 45/46 (98%) |

> GPT-5's main advantage over mini is reliability, not capability. It solves the same tasks but crashes/times out far less often.
