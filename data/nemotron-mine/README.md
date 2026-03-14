# Nemotron-Mine: Task Generators from NVIDIA Nemotron Datasets

## Overview

`data/nemotron-mine/` contains 9 task generators that mine NVIDIA's Nemotron datasets for code samples, generate coding task descriptions via LLM, and produce harbor-format task directories. This mirrors the existing `data/dclm-mine/` generators that source from `bigcode/the-stack`.

## Data Sources

We use **two** NVIDIA datasets because no single one covers all languages with actual code:

### 1. `nvidia/Nemotron-Pretraining-Code-v2` (Synthetic configs)
- **Synthetic-Rewriting**: Python only — LLM-rewritten code with `content` column
- **Synthetic-Transpilation**: C++ only — transpiled from Python, `content` column
- These configs have direct code content ready to use

### 2. `nvidia/Nemotron-CC-Code-v1` (Common Crawl code pages)
- 216M web pages containing code, cleaned by Phi-4
- Multi-language: bash, java, python, ruby, C#, C++, rust, JS, etc.
- Code is **embedded in markdown code blocks** (`` ```lang ... ``` ``) within web pages
- We extract code blocks matching the target language using regex

### Why two datasets?
The Nemotron v2 `Nemotron-Code-Metadata` config (377M records) is **metadata-only** — it has `repo`, `commit_id`, `rel_path`, `language` columns but NO code content. The synthetic configs only cover Python and C++. For all other languages (bash, java, C#, ruby, rust), we use CC-Code-v1's code block extraction.

## Architecture

```
nemotron_loader.py          # Shared utility: load_nemotron_stream(), normalize_sample(), extract_code_from_sample()
generate_bash_tasks.py      # Bash verifier scripts → coding tasks
generate_bash_tasks_with_tests.py       # Bash with inline test content
generate_bash_tasks_with_tests_gpt5mini.py  # Same, gpt5-mini variant
generate_cpp_test_tasks.py  # C++ test files → coding tasks (via Synthetic-Transpilation)
generate_csharp_test_tasks.py  # C# xUnit/NUnit → coding tasks (via CC-Code-v1)
generate_junit_tasks.py     # JUnit test files → coding tasks (via CC-Code-v1)
generate_pytest_tasks_gpt5mini.py  # pytest files → coding tasks (via Synthetic-Rewriting)
generate_rspec_tasks.py     # Ruby RSpec → coding tasks (via CC-Code-v1)
generate_rust_test_tasks.py # Rust #[test] → coding tasks (via CC-Code-v1)
test_all_nemotron.py        # Test harness: run any/all generators + harbor evaluation
```

### Pipeline per generator

1. **Filter**: Stream Nemotron dataset, apply language-specific filters (e.g., `has_shebang()`, `has_junit_imports()`, complexity checks)
2. **Generate instructions**: Use `gpt-5-nano` via `bespokelabs.curator` to turn code samples into task descriptions
3. **Create harbor tasks**: Build task directories with `instruction.md`, `task.toml`, `environment/Dockerfile`, `tests/test.sh`, `metadata.json`
4. **Evaluate**: Run `harbor jobs start` with terminus-2 agent on Daytona environments

### Language → Dataset Routing

| Language | Dataset | Config / Method |
|----------|---------|-----------------|
| Python (pytest) | Nemotron-Pretraining-Code-v2 | `Synthetic-Rewriting` (direct `content` column) |
| C++ | Nemotron-Pretraining-Code-v2 | `Synthetic-Transpilation` (direct `content` column) |
| Bash | Nemotron-CC-Code-v1 | Extract `` ```bash `` / `` ```sh `` code blocks |
| Java (JUnit) | Nemotron-CC-Code-v1 | Extract `` ```java `` code blocks |
| C# | Nemotron-CC-Code-v1 | Extract `` ```csharp `` / `` ```cs `` code blocks |
| Ruby (RSpec) | Nemotron-CC-Code-v1 | Extract `` ```ruby `` / `` ```rb `` code blocks |
| Rust | Nemotron-CC-Code-v1 | Extract `` ```rust `` / `` ```rs `` code blocks |

## Pass Rate Results (25-task harbor evaluation)

Harbor evaluation with `terminus-2` agent using `gpt-5-nano-2025-08-07` on Daytona environments, 25 tasks per generator.

### Nemotron Results

| Generator | Source | Scanned | Pass Rate | Passed/Total |
|-----------|--------|---------|-----------|--------------|
| **C++ (cpp)** | V2 Synthetic-Transpilation | direct | **60%** | 15/25 |
| **JUnit (junit)** | CC-Code-v1 | 132,858 | **24%** | 6/25 |
| **Bash (bash)** | CC-Code-v1 | 80,900 | **33% (v2)** / 12% (v1) | 3/9 (v2) |
| **pytest** | V2 Synthetic-Rewriting | direct | **30% nano / 40% gpt-5 (v2)** / 12% (v1) | 3/10 nano, 4/10 gpt-5 (v2) |
| **C# (csharp)** | CC-Code-v1 | 406,025 | **0%** | 0/25 (20 infra errors) |
| **Ruby (rspec)** | CC-Code-v1 | 755,998 | **0%** | 0/25 |
| **Rust (rust)** | CC-Code-v1 | 6,603,487 | **0%** | 0/25 |

### Comparison vs The Stack (dclm-mine)

| Generator | Nemotron | The Stack | Winner |
|-----------|----------|-----------|--------|
| C++ | **60%** | 36% | **Nemotron** |
| JUnit | **24%** | 24% | Tie |
| Bash | **12%** | 4% | **Nemotron** |
| pytest | **12%** | 0-10% | **Nemotron** |
| C# | 0% | **12%** | Stack |
| Ruby | 0% | 0% | Tie |
| Rust | 0% | 0% | Tie |

**Key finding**: Nemotron wins or ties on 6/7 languages. C++ is the standout at 60% (vs Stack's 36%). C# is the only language where Stack outperforms, though Nemotron C# had 20/25 infrastructure errors.

### Error breakdown
Many "failures" are infrastructure errors (DaytonaError, EnvironmentStartTimeoutError, AgentTimeoutError), not test failures. The actual pass rate among tasks that ran successfully is higher. C# was particularly affected (20/25 errors).

## Usage

### Run a single generator
```bash
cd data/nemotron-mine
python generate_bash_tasks.py       # Generate bash tasks (filter → LLM → harbor tasks)
python generate_junit_tasks.py      # Generate JUnit tasks
```

### Test all generators with harbor
```bash
# All generators
python test_all_nemotron.py

# Specific generators
python test_all_nemotron.py bash junit rust --limit 25

# Generate only (skip harbor evaluation)
python test_all_nemotron.py bash --generate-only
```

### Environment variables
- `OPENAI_API_KEY`: Required for LLM instruction generation
- `HARBOR_API_KEY`: Required for harbor evaluation
- `HF_TOKEN`: Required for dataset access (Nemotron-CC-Code-v1 is gated)

## Key Design Decisions

1. **CC-Code-v1 code block extraction**: Since CC-Code-v1 contains web pages (not raw source files), we regex-extract `` ```lang `` fenced code blocks. We take the longest matching block per sample. Minimum 50 chars to skip trivial snippets.

2. **Pytest complexity filtering**: Nemotron's Synthetic-Rewriting produces overly complex pytest files (Django, Selenium, rasterio). We added:
   - Heavy framework blocklist (40+ packages)
   - AST parse validation
   - Max 2 non-stdlib imports
   - Max 15 test functions, max 1 test class

3. **max_scan increased to 50M**: CC-Code-v1 mixes all languages. Finding language-specific test code requires scanning many more samples than The Stack's pre-filtered subsets.

## Fixes (2026-03-01)

### Bash v2 (`exp_rpt_nemotron-bash-v2`)
- **Problem**: Verifier scripts could hang indefinitely (infinite loops, long sleeps, background process forks).
- **Fix 1**: Added `timeout 120` wrapper around verifier execution in `adapt_verifier_for_harbor()`.
- **Fix 2**: Added hang-detection filtering in `is_verifier_script()`: rejects scripts with `while true`, `sleep > 10s`, and background process forks (`&`).
- **Result**: Pass rate improved from 12% to **33%** (nano). Uploaded to `DCAgent/exp_rpt_nemotron-bash-v2`.

### Pytest v2 (`exp_rpt_nemotron-pytest-gpt5mini-v2`)
- **Problem**: Invalid task.toml (leading whitespace) + test.sh missing PYTHONPATH + agent couldn't find correct file structure.
- **Fix 1**: `textwrap.dedent()` in `commons.py` for clean TOML output.
- **Fix 2**: Skip tasks with empty or short (< 50 chars) instructions.
- **Fix 3**: Added `export PYTHONPATH=/app:$PYTHONPATH` to test.sh so pytest finds agent-created modules under `/app/`.
- **Fix 4**: Added `_build_import_hint()` — parses test file imports and appends a "File structure required by the tests" block to each instruction showing exact file paths, package `__init__.py` files, and exported names.
- **Result**: Pass rate improved from 0% to **30% nano / 40% gpt-5**. Uploaded to `DCAgent/exp_rpt_nemotron-pytest-gpt5mini-v2`.

## Long-Horizon Task Generators (2026-03-02)

Six new generators for longer-horizon, multi-step tasks (RL agent training):

| # | Generator | File | Data Source | Expected Pass Rate |
|---|-----------|------|-------------|-------------------|
| 1 | Multi-File Composition | `generate_multifile_tasks.py` | `DCAgent/exp_rpt_stack-pytest-v2` (triples) | 15-35% |
| 2 | PR-Based Mining | `generate_pr_tasks.py` | `bigcode/commitpackft` (Python commits) | 10-25% |
| 3 | Repo Scaffolding | `generate_scaffold_tasks.py` | `bigcode/the-stack` (Python repos w/ tests) | 20-40% |
| 4 | Feature from Issue | `generate_issue_tasks.py` | `princeton-nlp/SWE-bench_Lite` | 10-30% |
| 5 | Curriculum Variants | `generate_curriculum_tasks.py` | Multiple existing HF datasets | Easy 40-60%, Hard 5-15% |
| 6 | Refactoring Tasks | `generate_refactoring_tasks.py` | `bigcode/the-stack` (messy Python) | 10-25% |

### Usage

```bash
# Run any generator with a small limit first
LIMIT=10 python generate_curriculum_tasks.py --no-upload
LIMIT=10 python generate_multifile_tasks.py --no-upload
LIMIT=10 python generate_scaffold_tasks.py --no-upload
LIMIT=10 python generate_issue_tasks.py --no-upload
LIMIT=10 python generate_pr_tasks.py --no-upload
LIMIT=10 python generate_refactoring_tasks.py --no-upload

# Generate full dataset and upload
python generate_curriculum_tasks.py --limit 500
```

### Key design constraints
- All tasks within a generator share 1 Dockerfile (for RL image caching)
- All use `create_harbor_task_directory_generic()` from `data/commons.py`
- All use `run_completions()` from `data/completions.py` with `{{column}}` templates
- LLM-generated tests are validated with `_clean_llm_test_code()` + `ast.parse()`

## Status (2026-03-01)

All 7 original generators evaluated. Results saved to `nemotron_test_results.json`.

**Recommended for training data**: C++ (60%), Bash (33% v2), JUnit (24%), pytest (needs hints).
