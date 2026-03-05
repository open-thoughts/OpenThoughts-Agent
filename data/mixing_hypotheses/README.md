# RL Data Mixing Strategy Hypotheses

Generated: 2026-03-04
All datasets uploaded to HuggingFace as parquet files under the `DCAgent/` org.

---

## Source Dataset Inventory

We draw from 40 working datasets across 12 languages. Pass rates measured on gpt-5-mini (25-task samples).

### By Dockerfile Group

Datasets sharing a Dockerfile can be mixed with zero additional container build cost.

**Python base (20 datasets, 1 Dockerfile)**

| Dataset | HF Repo | Pass Rate | Language |
|---------|---------|-----------|----------|
| r2egym-easy | DCAgent/exp-rdb-r2egym-easy | 65% | Python |
| r2egym-trivial | DCAgent/exp-rdb-r2egym-trivial | 60% | Python |
| r2egym-medium | DCAgent/exp-rdb-r2egym-medium | 56% | Python |
| r2egym-hard | DCAgent/exp-rdb-r2egym-hard | 56% | Python |
| bigcodebench | DCAgent/exp_rpt_bigcodebench-v3 | 54% | Python |
| crosscodeeval-python | DCAgent/exp_rpt_crosscodeeval-python-v2 | 48% | Python |
| e2egit | DCAgent/exp_rpt_e2egit-v2 | 48% | Python |
| pymethods2test | DCAgent/exp_rpt_pymethods2test-v3 | 40% | Python |
| stack-pytest | DCAgent/exp_rpt_stack-pytest-v2 | 40% | Python |
| unitsyn-python | DCAgent/exp_rpt_unitsyn-python-v3 | 36% | Python |
| nemotron-pytest | DCAgent/exp_rpt_nemotron-pytest-gpt5mini-v2 | 30% | Python |
| r2egym-very_hard | DCAgent/exp-rdb-r2egym-very_hard | 28% | Python |
| codereval-python | DCAgent/exp_rpt_codereval-python-v2 | 24% | Python |
| exercism-python | DCAgent/exp_rpt_exercism-python | 24% | Python |
| codenet-python | DCAgent/exp_rpt_codenet-python | 24% | Python |
| stack-selfdoc | DCAgent/exp_rpt_stack-selfdoc-v2 | 16% | Python |
| softwareheritage | DCAgent/exp_rpt_softwareheritage-v2 | 12% | Python |
| stack-pytest-gpt5mini | DCAgent/exp_rpt_stack-pytest-gpt5mini | 8% | Python |
| bugsinpy | DCAgent/exp_rpt_bugsinpy-v4 | 8% | Python |
| bugsinpy-mf | DCAgent/exp_rpt_bugsinpy-mf | 8% | Python |

**Java base (4 datasets, 1 Dockerfile)**

| Dataset | HF Repo | Pass Rate |
|---------|---------|-----------|
| stack-junit | DCAgent/exp_rpt_stack-junit | 84% |
| crosscodeeval-java | DCAgent/exp_rpt_crosscodeeval-java | 48% |
| methods2test | DCAgent/exp_rpt_methods2test-v2 | 28% |
| defects4j | DCAgent/exp_rpt_defects4j-v3 | 4% |

**Bash base (4 datasets, 1 Dockerfile)**

| Dataset | HF Repo | Pass Rate |
|---------|---------|-----------|
| nemotron-bash | DCAgent/exp_rpt_nemotron-bash-v2 | 33% |
| stack-bash-withtests-gpt5mini | DCAgent/exp_rpt_stack-bash-withtests-gpt5mini | 32% |
| stack-bash-withtests | DCAgent/exp_rpt_stack-bash-withtests | 24% |
| stack-bash | DCAgent/exp_rpt_stack-bash | 20% |

**C# base (2 datasets, 1 Dockerfile)**

| Dataset | HF Repo | Pass Rate |
|---------|---------|-----------|
| crosscodeeval-csharp | DCAgent/exp_rpt_crosscodeeval-csharp | 68% |
| stack-csharp | DCAgent/exp_rpt_stack-csharp | 12% |

**Single-dataset groups (1 Dockerfile each)**

| Dataset | HF Repo | Pass Rate | Language |
|---------|---------|-----------|----------|
| stack-php | DCAgent/exp_rpt_stack-php-v2 | 100% | PHP |
| codeelo | DCAgent/exp_rpt_codeelo-v2 | 52% | Multi |
| crosscodeeval-typescript | DCAgent/exp_rpt_crosscodeeval-typescript | 48% | TypeScript |
| stack-cpp | DCAgent/exp_rpt_stack-cpp | 24% | C++ |
| taco | DCAgent/exp_rpt_taco | 24% | Multi |
| stack-dockerfile | DCAgent/exp_rpt_stack-dockerfile-v2 | 16% | Dockerfile |
| stack-ruby | DCAgent/exp_rpt_stack-ruby | 12% | Ruby |
| stack-rust | DCAgent/exp_rpt_stack-rust | 12% | Rust |
| stack-jest | DCAgent/exp_rpt_stack-jest-v2 | 8% | JavaScript |
| manybugs | DCAgent/exp_rpt_manybugs-v2 | 7% | C |

### Difficulty Tiers (used by H1)

| Tier | Pass Rate Range | Python Datasets | Count |
|------|----------------|-----------------|-------|
| Easy | >= 50% | r2egym-easy, r2egym-trivial, r2egym-medium, r2egym-hard, bigcodebench | 5 |
| Medium | 20-49% | r2egym-very_hard, crosscodeeval-python, e2egit, pymethods2test, stack-pytest, unitsyn-python, codereval-python, exercism-python, codenet-python, nemotron-pytest | 10 |
| Hard | < 20% | stack-selfdoc, softwareheritage, stack-pytest-gpt5mini, bugsinpy, bugsinpy-mf | 5 |

---

## Generated Mixes — All Uploaded to HuggingFace

### Summary Table

| HF Repo | Hypothesis | Arm | Tasks | Dockerfiles | Reward Type |
|---------|-----------|-----|-------|-------------|-------------|
| `DCAgent/mix_baseline_uniform` | Baseline | — | 3,575 | 1 (Python) | Binary |
| `DCAgent/mix_h1_struggle_zone` | H1 Struggle Zone | test | 3,618 | 1 | Binary |
| `DCAgent/mix_h2_language_balanced` | H2 Language Diversity | test | 4,590 | 6 | Binary |
| `DCAgent/mix_h2_language_proportional` | H2 Language Diversity | baseline | 4,135 | 6 | Binary |
| `DCAgent/mix_h4_dense_rewards_hard` | H4 Dense Rewards | test | 2,503 | 1 | Proportional |
| `DCAgent/mix_h4_binary_easy` | H4 Dense Rewards | baseline | 697 | 1 | Binary |
| `DCAgent/mix_h5_skill_diverse` | H5 Skill Coverage | test | 3,499 | 1 | Binary |
| `DCAgent/mix_h6_test_quality_top25` | H6 Test Quality | test | 3,927 | 1 | Binary |
| `DCAgent/mix_h7_raw_volume_5k` | H7 Dedup vs Volume | baseline | 3,575 | 1 | Binary |
| `DCAgent/mix_h7_dedup_diverse_1k` | H7 Dedup vs Volume | test | 1,990 | 1 | Binary |
| `DCAgent/mix_h8_original_tests` | H8 Adversarial Tests | baseline | 4,873 | 1 | Binary |
| `DCAgent/mix_h8_adversarial_tests` | H8 Adversarial Tests | test | 4,873 | 1 | Binary |
| `DCAgent/mix_h10_reward_binary` | H10 Reward Ablation | Arm A | 4,873 | 1 | Binary |
| `DCAgent/mix_h10_reward_proportional` | H10 Reward Ablation | Arm B | 4,873 | 1 | Proportional |
| `DCAgent/mix_h10_reward_staged` | H10 Reward Ablation | Arm C | 4,872 | 1 | Staged |
| `DCAgent/mix_h11_single_skill_only` | H11 Compositional | baseline | 4,873 | 1 | Binary |
| `DCAgent/mix_h11_compositional_gradient` | H11 Compositional | test | 4,873 | 1 | Binary |

**Total: 17 datasets, ~63,000 tasks across all mixes.**

---

## Hypothesis Details

### Baseline: Uniform Random Mix

- **HF**: `DCAgent/mix_baseline_uniform` (3,575 tasks)
- **Method**: Sample uniformly from all 20 Python-base datasets (equal weight per dataset = 5% each). Binary reward.
- **Purpose**: Every hypothesis is compared against this. Represents naive "just throw everything together" approach.
- **Source pass rate mix**: Weighted average ~33% (spans 8% to 65%).

---

### H1: Struggle Zone (Difficulty Band Weighting)

- **Claim**: Oversampling 20-50% pass-rate tasks produces better RL signal than uniform sampling.
- **Theory**: RL learns fastest when the agent succeeds ~30-50% of the time. Tasks at 100% give no gradient; tasks at 4% give no reward. The 20-49% band maximizes informative rollouts.
- **HF**: `DCAgent/mix_h1_struggle_zone` (3,618 tasks)
- **Mix**: 70% medium tier (20-49% pass rate), 20% easy (>=50%), 10% hard (<20%)
- **Weighting**: Equal weight within each tier. Medium: 10 datasets × 7% each. Easy: 5 datasets × 4% each. Hard: 5 datasets × 2% each.
- **Baseline**: `DCAgent/mix_baseline_uniform` (uniform from same datasets)
- **Dockerfiles**: 1 (all Python-base)

**Source datasets by tier:**
- Easy (20%): r2egym-easy (65%), r2egym-trivial (60%), r2egym-medium (56%), r2egym-hard (56%), bigcodebench (54%)
- Medium (70%): r2egym-very_hard (28%), crosscodeeval-python (48%), e2egit (48%), pymethods2test (40%), stack-pytest (40%), unitsyn-python (36%), codereval-python (24%), exercism-python (24%), codenet-python (24%), nemotron-pytest (30%)
- Hard (10%): stack-selfdoc (16%), softwareheritage (12%), stack-pytest-gpt5mini (8%), bugsinpy (8%), bugsinpy-mf (8%)

---

### H2: Language Diversity Bonus

- **Claim**: Language-balanced sampling improves generalization vs Python-heavy sampling.
- **Theory**: 17/40 datasets are Python. Language-balanced forces language-agnostic reasoning. Cross-language transfer is established in code LLMs.
- **Test**: `DCAgent/mix_h2_language_balanced` (4,590 tasks) — equal weight per language family
- **Baseline**: `DCAgent/mix_h2_language_proportional` (4,135 tasks) — weight proportional to dataset count (Python-dominated)
- **Dockerfiles**: 6 (one per language family)

**Language families (1-2 datasets each):**

| Family | Datasets | Pass Rates | Weight (test) | Weight (baseline) |
|--------|----------|------------|---------------|-------------------|
| Python | stack-pytest, bigcodebench | 40%, 54% | 16.7% | ~22% |
| Java | stack-junit, methods2test | 84%, 28% | 16.7% | ~22% |
| Bash | stack-bash-withtests | 24% | 16.7% | ~11% |
| C# | crosscodeeval-csharp | 68% | 16.7% | ~11% |
| TypeScript | crosscodeeval-typescript | 48% | 16.7% | ~11% |
| C++ | stack-cpp | 24% | 16.7% | ~11% |

---

### H4: Dense Rewards on Hard Tasks > Binary Rewards on Easy Tasks

- **Claim**: Proportional rewards (passed/total tests) on hard datasets produce better learning than binary rewards on easy datasets.
- **Theory**: At 8% binary pass rate, 92% of rollouts yield zero signal. Proportional reward means passing 3/10 tests → reward 0.3, turning hard tasks into medium-difficulty learning.
- **Test**: `DCAgent/mix_h4_dense_rewards_hard` (2,503 tasks) — hard Python datasets with proportional reward
- **Baseline**: `DCAgent/mix_h4_binary_easy` (697 tasks) — easy Python datasets with binary reward
- **Dockerfiles**: 1

**Test arm sources** (proportional reward `= passed_tests / total_tests`):
- bugsinpy (8%), softwareheritage (12%), stack-selfdoc (16%), stack-pytest-gpt5mini (8%), bugsinpy-mf (8%)

**Baseline arm sources** (binary reward `= 1 if all_pass else 0`):
- r2egym-easy (65%), bigcodebench (54%), crosscodeeval-python (48%), r2egym-trivial (60%), r2egym-medium (56%)

**Reward implementation**: Custom `test.sh` that counts individual pytest PASSED/FAILED lines and writes `passed/total` to `/logs/verifier/reward.txt`.

**Note**: Baseline is smaller (697) because easy datasets have fewer available tasks. The test arm has more tasks because 5 hard datasets each contribute ~500.

---

### H5: Skill-Diverse Sampling

- **Claim**: Equalizing skill coverage across 10 categories beats random sampling.
- **Theory**: Random code datasets over-index on string/algorithm tasks. Deliberately covering rare skills (database, concurrency, API) increases problem diversity.
- **HF**: `DCAgent/mix_h5_skill_diverse` (3,499 tasks)
- **Baseline**: `DCAgent/mix_baseline_uniform` (naturally imbalanced)
- **Dockerfiles**: 1

**Method**:
1. Loaded up to 1,000 tasks from each of the 20 Python-base datasets (8,087 total pool)
2. LLM classifier (gpt-5-nano) tagged each task with primary skill from: `file_io`, `parsing`, `algorithm`, `data_structure`, `api`, `database`, `testing`, `cli`, `concurrency`, `string_processing`
3. Sampled equally from each skill category (500/skill target)
4. Final: 3,499 tasks with balanced skill distribution

---

### H6: Test Quality Filter

- **Claim**: Keeping only top-25% quality tests produces better RL agents than training on all tasks.
- **Theory**: Weak tests (`assert isinstance(x, list)`) reward trivially correct but wrong solutions. High-quality tests force genuinely correct code. Reward quality → policy quality.
- **HF**: `DCAgent/mix_h6_test_quality_top25` (3,927 tasks)
- **Baseline**: `DCAgent/mix_baseline_uniform` (all tasks unfiltered)
- **Dockerfiles**: 1

**Method**:
1. Loaded up to 1,000 tasks from each Python-base dataset (8,087 total pool)
2. LLM scorer (gpt-5-nano) rated test quality 0-100 based on:
   - Number and variety of assertions
   - Edge case coverage (empty, large, boundary)
   - Specificity (would wrong solutions still pass?)
   - Independence (tests check distinct behaviors)
   - Correctness (expected values are reasonable)
3. Kept tasks with score >= 75 (top ~48% passed this threshold)
4. Final: 3,927 high-quality-test tasks

---

### H7: Dedup + Diversity > Raw Volume

- **Claim**: ~2,000 deduplicated, skill-balanced tasks outperform ~3,500 raw tasks.
- **Theory**: Duplicates waste compute and cause overfitting. MinHash dedup + skill balance = compact but maximally diverse.
- **Test**: `DCAgent/mix_h7_dedup_diverse_1k` (1,990 tasks) — deduplicated + skill-balanced
- **Baseline**: `DCAgent/mix_h7_raw_volume_5k` (3,575 tasks) — raw uniform sample (no dedup)
- **Dockerfiles**: 1

**Method (test arm)**:
1. Loaded 8,087 tasks from all 20 Python-base datasets
2. **MinHash deduplication**: Computed 128-hash MinHash signatures from 5-character shingles of each instruction. Greedy removal of pairs with Jaccard similarity >= 0.8
3. **Skill classification**: LLM (gpt-5-nano) classified remaining tasks into 10 skill categories
4. **Skill-balanced sampling**: Equal sample per skill category, target 2,500
5. Final: 1,990 unique, skill-balanced tasks

**Method (baseline arm)**: Uniform random sample from same 20 datasets, no dedup.

---

### H8: Adversarial Test Augmentation

- **Claim**: Augmenting tasks with adversarial tests makes agents more robust.
- **Theory**: Standard tests miss edge cases. Adversarial tests lower initial pass rates but force the model to handle real-world failure modes.
- **Test**: `DCAgent/mix_h8_adversarial_tests` (4,873 tasks) — original + adversarial tests
- **Baseline**: `DCAgent/mix_h8_original_tests` (4,873 tasks) — original tests only
- **Dockerfiles**: 1

**Source**: Medium-tier Python datasets (20-49% pass rate): r2egym-very_hard, crosscodeeval-python, e2egit, pymethods2test, stack-pytest, unitsyn-python, codereval-python, exercism-python, codenet-python, nemotron-pytest.

**Method (test arm)**:
1. Same tasks as baseline
2. LLM (gpt-5-nano) generated 5 adversarial pytest functions per task targeting:
   - Off-by-one / boundary conditions
   - Empty input / missing data
   - Unicode / special characters
   - Type errors (wrong types, None values)
   - Large input / performance edge cases
3. Adversarial tests appended to existing test files (or added as `test_adversarial.py`)

**Both arms use identical instructions** — only the test suites differ.

---

### H10: Reward Structure Ablation (3-arm)

- **Claim**: Staged rewards outperform both proportional and binary rewards.
- **Theory**: Tests have natural ordering (basic → edge → robustness). Staged rewards capture this curriculum within each episode.
- **Arm A**: `DCAgent/mix_h10_reward_binary` (4,873 tasks) — pass/fail all tests
- **Arm B**: `DCAgent/mix_h10_reward_proportional` (4,873 tasks) — passed/total tests
- **Arm C**: `DCAgent/mix_h10_reward_staged` (4,872 tasks) — max_stage_reached/total_stages
- **Dockerfiles**: 1

**Source**: Same medium-tier Python datasets as H8 (10 datasets, 20-49% pass rate).

**All three arms use identical tasks and instructions** — only the reward structure differs:

| Arm | Reward | Example (3/5 basic pass, 0/3 edge pass) |
|-----|--------|----------------------------------------|
| A (Binary) | `1 if all_pass else 0` | 0 |
| B (Proportional) | `passed / total` | 0.375 |
| C (Staged) | `max_stage / total_stages` | 0.33 (stage 1 of 3) |

**Arm C method**: LLM (gpt-5-nano) decomposed each test suite into 3-5 ordered stages:
1. Code exists and is syntactically valid
2. Basic/happy-path tests pass
3. Edge cases pass
4. Performance/robustness tests pass

Custom `test.sh` runs stages sequentially, stops at first failure, reports `max_stage / total_stages`.

---

### H11: Compositional Complexity Gradient

- **Claim**: Mixing single-skill + multi-skill tasks produces better generalization than single-skill only.
- **Theory**: Real tasks require composing capabilities. Single-skill tasks teach individual skills; multi-skill tasks teach integration.
- **Test**: `DCAgent/mix_h11_compositional_gradient` (4,873 tasks) — 50/30/20 single/2-skill/3-skill
- **Baseline**: `DCAgent/mix_h11_single_skill_only` (4,873 tasks) — 100% single-skill
- **Dockerfiles**: 1

**Source**: Medium-tier Python datasets (same as H8/H10).

**Method (test arm)**:
- 50% original single-skill tasks (unchanged)
- 30% rewritten to require 2 skills: original + file I/O (read from file, write to file)
- 20% rewritten to require 3 skills: original + file I/O + JSON parsing

**Rewriting**: LLM (gpt-5-nano) rewrote both instructions and test suites. Tasks that failed rewriting (no valid `def test_` in output) fell back to original.

---

## Priority Ranking

| Rank | Hypothesis | Dockerfiles | Rationale |
|------|-----------|-------------|-----------|
| 1 | H1 (Struggle Zone) | 1 | Most testable, strong theory, cheap |
| 2 | H4 (Dense Rewards) | 1 | High potential, underexplored in RL |
| 3 | H10 (Reward Ablation) | 1 | Clean 3-arm, answers fundamental question |
| 4 | H6 (Test Quality) | 1 | Data quality is king |
| 5 | H5 (Skill Coverage) | 1 | Diversity signal |
| 6 | H7 (Dedup + Diversity) | 1 | Quality vs quantity |
| 7 | H2 (Language Diversity) | 6 | Cross-lingual transfer |
| 8 | H8 (Adversarial Tests) | 1 | Depends on LLM test quality |
| 9 | H11 (Compositional) | 1 | Depends on LLM rewrite quality |

---

## How to Load

```python
from datasets import load_dataset

# Load any mix
ds = load_dataset("DCAgent/mix_h1_struggle_zone")

# Each row has: instruction, test_sh, dockerfile, task_toml, metadata, test_files_json, task_dir_name
row = ds["train"][0]
print(row["instruction"][:200])
```

## Regeneration

```bash
# Requires HF_TOKEN and OPENAI_API_KEY in environment
source secret.env

# Generate a single hypothesis
python data/mixing_hypotheses/generate_h1_struggle_zone.py

# Multi-arm scripts accept --arm flag
python data/mixing_hypotheses/generate_h10_reward_ablation.py --arm C  # just staged
python data/mixing_hypotheses/generate_h2_language_diversity.py --arm both
```
