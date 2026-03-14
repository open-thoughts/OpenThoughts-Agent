# DCLM-Mine: Task Generation Scripts

This directory contains scripts for generating RL training datasets from various code sources. Each script transforms raw code/test data into standardized **harbor-format** tasks.

---

## Benchmark Summary (January 2026)

### Stack-Based Generators (The Stack)

Pass rates measured with **terminus-2** agent using **gpt-5-nano** model on 25 sample tasks.
Model used for task generation: **gpt-5-nano-2025-08-07**

| Dataset | Pass Rate | Tasks | HuggingFace Repository | Status |
|---------|-----------|-------|------------------------|--------|
| **PHP** | 100% | 10,000 | `DCAgent/exp_rpt_stack-php` | ✅ Working |
| **JUnit** | 80% | 10,000 | `DCAgent/exp_rpt_stack-junit` | ✅ Working |
| **Rust** | 70% | 9,987 | `DCAgent/exp_rpt_stack-rust` | ✅ Working |
| **C++** | 44% | 9,943 | `DCAgent/exp_rpt_stack-cpp` | ✅ Working |
| **C#** | 24% | 9,989 | `DCAgent/exp_rpt_stack-csharp` | ✅ Working |
| **Dockerfile** | 20% | 10,000 | `DCAgent/exp_rpt_stack-dockerfile` | ✅ Working |
| **Go** | 20% | 10,000 | `DCAgent/exp_rpt_stack-go` | ✅ Working |
| **Ruby** | 20% | 10,000 | `DCAgent/exp_rpt_stack-ruby` | ✅ Working |
| **Jest** | 10% | 10,000 | `DCAgent/exp_rpt_stack-jest` | ✅ Working |
| **Self-documented** | 10% | 9,999 | `DCAgent/exp_rpt_stack-selfdoc` | ✅ Working |
| **Bash+Tests** | 8% | 10,000 | `DCAgent/exp_rpt_stack-bash-withtests` | ✅ Working |
| **pytest** | 10% | 10,000 | `DCAgent/exp_rpt_stack-pytest` | ✅ Working |
| **Bash** | 8% | 10,000 | `DCAgent/exp_rpt_stack-bash` | ✅ Working |

**Total Stack-based Tasks: ~129,918**

#### GPT-5-Mini Variants (Teacher Model Experiment)

These are duplicate datasets generated with **gpt-5-mini-2025-08-07** instead of gpt-5-nano, for comparing teacher model quality.

| Dataset | Tasks | HuggingFace Repository | Notes |
|---------|-------|------------------------|-------|
| **Dockerfile (gpt5mini)** | 10,000 | `DCAgent/exp_rpt_stack-dockerfile-gpt4o` | Legacy naming* |
| **Self-documented (gpt5mini)** | 10,000 | `DCAgent/exp_rpt_stack-selfdoc-gpt4o` | Legacy naming* |
| **pytest (gpt5mini)** | 10,000 | `DCAgent/exp_rpt_stack-pytest-gpt5mini` | 🔄 Generating |
| **Bash+Tests (gpt5mini)** | 10,000 | `DCAgent/exp_rpt_stack-bash-withtests-gpt5mini` | ✅ Done |

\* Repos named "gpt4o" actually contain gpt-5-mini generated tasks (verified via metadata).

---

### External Benchmark Datasets

Pass rates measured with **terminus-2** agent using **gpt-5-nano** model on 25 sample tasks.

| Dataset | Pass Rate | Tasks Generated | HuggingFace Repository | Status |
|---------|-----------|-----------------|------------------------|--------|
| **Methods2Test** | 96% | 10,000 | `DCAgent/exp_rpt_methods2test` | ✅ Working |
| **CodeNet** | 88% | 10,000 | `DCAgent/exp_rpt_codenet-python` | ✅ Working (fallback) |
| **UniTSyn** | 84% | 10,000 | `DCAgent/exp_rpt_unitsyn-python` | ✅ Working (fallback) |
| **GHActions** | 80% | 10,000 | `DCAgent/exp_rpt_ghactions` | ✅ Working |
| **SoftwareHeritage** | 72% | 10,000 | `DCAgent/exp_rpt_softwareheritage` | ✅ Working |
| **TravisTorrent** | 72% | 10,000 | `DCAgent/exp_rpt_travistorrent` | ✅ Working |
| **BugSwarm** | 71% | 2 | `DCAgent/exp_rpt_bugswarm` | ⚠️ Needs manual download |
| **ManyBugs** | 68% | 155 | `DCAgent/exp_rpt_manybugs-v2` | ✅ v2: minimal Dockerfile |
| **CrossCodeEval** | 64% | 9,928 | `DCAgent/exp_rpt_crosscodeeval-*` | ✅ Working (4 languages) |
| **CodeElo** | 60% | 165 | `DCAgent/exp_rpt_codeelo` | ✅ Max available |
| **E2EGit** | 60% | 8,601 | `DCAgent/exp_rpt_e2egit` | ✅ Max available |
| **BugsInPy-MF** | 60% | 2 | `DCAgent/exp_rpt_bugsinpy-mf` | ⚠️ Needs manual download |
| **BigCodeBench** | 57% / 40% nano | 1,140 | `DCAgent/exp_rpt_bigcodebench-v3` | ✅ v3: fat Dockerfile |
| **SWE-bench** | 56% | 2,294 | `DCAgent/exp_rpt_swebench` | ✅ Max available |
| **CoderEval** | 48% | 230 | `DCAgent/exp_rpt_codereval-python` | ✅ Max available |
| **BugsInPy** | 48% | 1,215 | `DCAgent/exp_rpt_bugsinpy-v4` | ✅ v4: no COPY in Dockerfile |
| **Defects4J** | 42% | 465 | `DCAgent/exp_rpt_defects4j` | ✅ Max available |
| **TACO** | 4% | 10,000 | `DCAgent/exp_rpt_taco` | ⚠️ Low pass rate |
| **Exercism** | 4% | 133 | `DCAgent/exp_rpt_exercism-python` | ⚠️ Low pass rate |
| **QuixBugs** | 0% | 40 | - | ❌ Task format issue |
| **PyMethods2Test** | 0% | - | - | ❌ Fallback inadequate |

**Total Tasks Generated: ~83,353**

### Notes on Datasets

#### Datasets Using Fallbacks
- **CodeNet**: Uses `deepmind/code_contests` (4k samples) since IBM/CodeNet was removed from HuggingFace
- **UniTSyn**: Uses `KAKA22/CodeRM-UnitTest` since original dataset requires manual generation

#### Datasets Requiring Manual Download
- **BugSwarm**: Not on HuggingFace. Download from [bugswarm.org](https://www.bugswarm.org/dataset/)
- **BugsInPy-MF**: Dataset not available on HuggingFace

#### Known Issues
- **QuixBugs** (0% pass): Tasks missing buggy code - it's a bug-fixing benchmark but generated tasks just say "Implement X" without the buggy code to fix
- **PyMethods2Test** (0% pass): Uses same fallback as UniTSyn with minimal instructions and hidden edge cases
- **TACO/Exercism** (4% pass): Competitive programming problems may be too hard for current agent

### CrossCodeEval Language Breakdown
| Language | Tasks | Repository |
|----------|-------|------------|
| Python | 2,482 | `DCAgent/exp_rpt_crosscodeeval-python` |
| Java | 2,482 | `DCAgent/exp_rpt_crosscodeeval-java` |
| TypeScript | 2,482 | `DCAgent/exp_rpt_crosscodeeval-typescript` |
| C# | 2,482 | `DCAgent/exp_rpt_crosscodeeval-csharp` |

---

## Setup

```bash
# ZIH:
module load release/25.06 GCCcore/14.2.0
module load Python/3.10.8

# Or use conda:
source /scratch/10000/eguha3/old-dc-agent/secret.env
conda activate dataagent

# Run any generation script:
python generate_pytest_tasks_gpt5nano.py
```

---

## Overview

All scripts follow a common pipeline:
1. **Load/Filter** data from source (The Stack, HuggingFace datasets)
2. **Generate** task instructions (via LLM synthesis or extraction)
3. **Create** harbor-format task directories
4. **Upload** to HuggingFace (`DCAgent/exp_rpt_*`)

---

## Experiments & Hypotheses

### Hypothesis 1: Teacher Model Quality
**Question:** Does a better teacher model produce higher quality synthetic instructions?

Comparing **gpt-5-nano** (base) vs **gpt-5-mini** (upgraded) for task generation.

| Script | Teacher Model | HuggingFace Repo |
|--------|---------------|------------------|
| `generate_dockerfile_tasks.py` | gpt-5-nano-2025-08-07 | `stack-dockerfile` |
| `generate_dockerfile_tasks_gpt5mini.py` | gpt-5-mini-2025-08-07 | `stack-dockerfile-gpt4o`* |
| `generate_pytest_tasks.py` | gpt-5-nano-2025-08-07 | `stack-pytest` |
| `generate_pytest_tasks_gpt5mini.py` | gpt-5-mini-2025-08-07 | `stack-pytest-gpt5mini` |
| `generate_self_documented_tasks.py` | gpt-5-nano-2025-08-07 | `stack-selfdoc` |
| `generate_self_documented_tasks_gpt5mini.py` | gpt-5-mini-2025-08-07 | `stack-selfdoc-gpt4o`* |
| `generate_bash_tasks_with_tests.py` | gpt-5-nano-2025-08-07 | `stack-bash-withtests` |
| `generate_bash_tasks_with_tests_gpt5mini.py` | gpt-5-mini-2025-08-07 | `stack-bash-withtests-gpt5mini` |

\* **Note on naming:** Repos with "gpt4o" in the name are legacy naming. They actually contain tasks generated with **gpt-5-mini-2025-08-07** (verified via metadata.json).

### Hypothesis 2: Synthetic vs Extraction
**Question:** Is it better to generate instructions purely from test code, or extract from existing documentation?

| Approach | Script | Description |
|----------|--------|-------------|
| **Synthetic** | `generate_pytest_tasks_gpt5nano.py` | LLM generates instructions by analyzing test code |
| **Extraction** | `generate_self_documented_tasks.py` | LLM extracts/cleans instructions from existing docstrings |

### Hypothesis 3: Test Visibility in Instructions
**Question:** Does showing test code in the instruction help or hurt agent performance?

| Variant | Script | Instruction Contains |
|---------|--------|---------------------|
| Hidden tests | `generate_bash_tasks.py` | Task description only |
| Visible tests | `generate_bash_tasks_with_tests.py` | Task description + test code in markdown |

---

## Scripts by Data Source

### The Stack (bigcode/the-stack)

#### Bash/Shell Scripts
| Script | Generation Type | What LLM Does |
|--------|-----------------|---------------|
| `generate_bash_tasks.py` | Synthetic | Generates task from verifier script |
| `generate_bash_tasks_with_tests.py` | Synthetic | Generates task, includes test in instruction |
| `generate_bash_tasks_with_tests_gpt4o.py` | Synthetic (gpt-5-nano) | Same, better model |

#### Python/Pytest
| Script | Generation Type | What LLM Does |
|--------|-----------------|---------------|
| `generate_pytest_tasks.py` | Synthetic | Generates task from pytest file |
| `generate_pytest_tasks_with_tests.py` | Synthetic | Generates task, includes pytest in instruction |
| `generate_pytest_tasks_gpt5nano.py` | Synthetic (gpt-5-nano) | Generates task from pytest (better model) |
| `generate_self_documented_tasks.py` | Extraction | Extracts task from existing docstrings |
| `generate_self_documented_tasks_gpt4o.py` | Extraction (gpt-5-nano) | Same, better model |

#### Dockerfiles
| Script | Generation Type | What LLM Does |
|--------|-----------------|---------------|
| `generate_dockerfile_tasks.py` | Synthetic (2-stage) | 1) Generates task from Dockerfile 2) Generates test.sh |
| `generate_dockerfile_tasks_gpt4o.py` | Synthetic (gpt-5-nano) | Same, better model |

#### Other Languages
| Script | Language | Framework |
|--------|----------|-----------|
| `generate_go_test_tasks.py` | Go | `testing` package |
| `generate_jest_tasks.py` | JavaScript | Jest |
| `generate_junit_tasks.py` | Java | JUnit |
| `generate_rspec_tasks.py` | Ruby | RSpec |
| `generate_phpunit_tasks.py` | PHP | PHPUnit |
| `generate_cpp_test_tasks.py` | C++ | Catch2/GoogleTest |
| `generate_csharp_test_tasks.py` | C# | xUnit/NUnit |
| `generate_rust_test_tasks.py` | Rust | #[test] |

### External Benchmarks (HuggingFace)

#### Code Generation Benchmarks
| Script | Dataset | Size | Description |
|--------|---------|------|-------------|
| `generate_bigcodebench_tasks.py` | BigCodeBench | 1,140 | Multi-library tasks, 7 domains |
| `generate_swebench_tasks.py` | SWE-bench | 2,294 | Real GitHub issues |
| `generate_codeelo_tasks.py` | CodeElo | 387 | Elo-rated CodeForces problems |
| `generate_codenet_tasks.py` | CodeNet | 14M | IBM competitive programming |
| `generate_exercism_tasks.py` | Exercism | - | Educational coding problems |

#### Bug Fixing Benchmarks
| Script | Dataset | Language | Description |
|--------|---------|----------|-------------|
| `generate_defects4j_tasks.py` | Defects4J | Java | 854 real Java bugs |
| `generate_bugswarm_tasks.py` | BugSwarm | Python/Java | 3,000+ CI pairs |
| `generate_bugsinpy_tasks.py` | BugsInPy | Python | Python bugs |
| `generate_bugsinpy_mf_tasks.py` | BugsInPy | Python | Multi-file variant |
| `generate_manybugs_tasks.py` | ManyBugs | C | 1,600+ C bugs |
| `generate_quixbugs_tasks.py` | QuixBugs | Multi | 80 buggy programs |

#### Cross-File / Context Tasks
| Script | Dataset | Description |
|--------|---------|-------------|
| `generate_crosscodeeval_tasks.py` | CrossCodeEval | Multi-file context understanding |
| `generate_codereval_tasks.py` | CoderEval | 460 real-world tasks |
| `generate_e2egit_tasks.py` | E2EGit | 43,670 E2E web tests |

---

## Generation Approaches

### 1. Synthetic Generation (Test -> Instruction)
The LLM analyzes test/verifier code and generates a task description.

```
Input: pytest/bash test file
       |
LLM Prompt: "Given this test, describe what needs to be implemented"
       |
Output: Natural language task description
```

**Used by:** `generate_bash_tasks.py`, `generate_pytest_tasks_gpt5nano.py`, `generate_dockerfile_tasks.py`

### 2. Extraction (Docs -> Cleaned Instruction)
The LLM extracts and cleans up existing documentation.

```
Input: Python file with docstrings + tests
       |
LLM Prompt: "Extract the task description from the documentation"
       |
Output: Cleaned task description
```

**Used by:** `generate_self_documented_tasks.py`

### 3. Direct (Dataset -> Task)
No LLM generation; uses the dataset's existing instruction field.

```
Input: BigCodeBench instruct_prompt field
       |
Output: Same text as instruction.md
```

**Used by:** `generate_bigcodebench_tasks.py`, `generate_swebench_tasks.py`

---

## Output Format (Harbor-Format)

Every script outputs tasks in this structure:

```
{task_dir}/
├── instruction.md          # Task description for the agent
├── task.toml               # Standard task metadata
├── metadata.json           # Dataset-specific metadata
├── environment/
│   └── Dockerfile          # Execution environment
└── tests/
    ├── test.sh             # Test runner (writes to /logs/verifier/reward.txt)
    └── test_*.py           # Actual test file(s)
```

### test.sh Contract
All test.sh scripts must:
1. Create `/logs/verifier/` directory
2. Write `"1"` to `/logs/verifier/reward.txt` if tests pass
3. Write `"0"` to `/logs/verifier/reward.txt` if tests fail

---

## Configuration

All scripts have these configuration variables:

```python
LIMIT = 10_000              # Number of tasks to generate
MODEL = "gpt-4o-mini"       # or "gpt-5-nano-2025-08-07"
```

### Rate Limiting
```python
max_requests_per_minute=500
max_tokens_per_minute=1_000_000
```

---

## HuggingFace Repositories

All outputs are uploaded to: `DCAgent/exp_rpt_<dataset_name>`

| Script | HF Repository |
|--------|---------------|
| `generate_dockerfile_tasks_gpt4o.py` | DCAgent/exp_rpt_stack-dockerfile-gpt4o |
| `generate_pytest_tasks_gpt5nano.py` | DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano |
| `generate_self_documented_tasks_gpt4o.py` | DCAgent/exp_rpt_stack-selfdoc-gpt4o |
| `generate_bash_tasks_with_tests_gpt4o.py` | DCAgent/exp_rpt_stack-bash-withtests-gpt4o |

---

## Key Experiments Summary

| Experiment | Comparison | Scripts |
|------------|------------|---------|
| **Teacher Model** | gpt-4o-mini vs gpt-5-nano | `*_gpt4o.py` / `*_gpt5nano.py` vs base scripts |
| **Synthetic vs Extract** | Generate from tests vs extract from docs | `pytest_gpt5nano` vs `self_documented` |
| **Test Visibility** | Hidden vs visible tests | `bash_tasks` vs `bash_tasks_with_tests` |
| **Language Coverage** | Python, Go, Java, JS, etc. | Language-specific scripts |
| **Task Complexity** | Simple tests vs multi-file bugs | The Stack vs SWE-bench |

---

## Legacy Pipeline (DCLM-Baseline)

The original pipeline for extracting tasks from DCLM baseline:

### Step 1: Filter DCLM baseline
Download DCLM-baseline, find sequences with shell-like commands, upload to: `DCAgent2/dclm-baseline-terminal-candidates-100k`

### Step 2: Classify sequences
```bash
python data/dclm-mine/Step2_classify/classify_tasks.py \
  --input-dataset DCAgent2/dclm-baseline-terminal-candidates-100k \
  --output-dataset DCAgent2/dclm-baseline-terminal-candidates-classified \
  --model gpt-5-nano-2025-08-07 \
  --max-sequences 50000
```

### Step 3: Extract tasks
```bash
python data/dclm-mine/Step3_extract/extract_tasks.py \
  --input-dataset DCAgent2/dclm-baseline-terminal-candidates-classified \
  --output-dataset DCAgent2/dclm-baseline-terminal-candidates-filtered-extracted-tasks
```

### Step 4: Generate sandboxes
```bash
python data/dclm-mine/generate.py \
  --input-dataset DCAgent2/dclm-baseline-terminal-candidates-filtered-extracted-tasks \
  --n-concurrent 8 \
  --env-type docker
```

Notes:
- ~25.7% of sequences contain a shell task
- ~82.9% of sandboxes work
- Need to classify ~47k sequences for 10k working sandboxes
