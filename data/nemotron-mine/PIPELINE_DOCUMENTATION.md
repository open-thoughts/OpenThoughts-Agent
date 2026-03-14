# Dataset Generation Pipeline: Complete Documentation

This document describes the end-to-end pipeline for generating coding task datasets from source code repositories, evaluating them with AI agents, analyzing failures, and iterating until task quality is high. This is written so that another agent can replicate the entire process.

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Data Sources](#2-data-sources)
3. [Generator Architecture](#3-generator-architecture)
4. [Harbor Task Format](#4-harbor-task-format)
5. [LLM Instruction Generation](#5-llm-instruction-generation)
6. [Harbor Evaluation System](#6-harbor-evaluation-system)
7. [Result Parsing](#7-result-parsing)
8. [Quality Improvement Loop](#8-quality-improvement-loop)
9. [HuggingFace Upload](#9-huggingface-upload)
10. [Running the Pipeline](#10-running-the-pipeline)
11. [Subagent Strategy](#11-subagent-strategy)
12. [File Reference](#12-file-reference)
13. [Lessons Learned](#13-lessons-learned)

---

## 1. Pipeline Overview

The pipeline converts raw source code into verified coding tasks for training AI agents:

```
Source Code (HuggingFace datasets)
    │
    ▼
[Step 1] FILTER — Language-specific heuristics extract test files
    │
    ▼
[Step 2] SYNTHESIZE — LLM generates task descriptions from test code
    │
    ▼
[Step 3] PACKAGE — Create harbor-format task directories
    │
    ▼
[Step 4] EVALUATE — Harbor runs AI agent on tasks, measures pass rate
    │
    ▼
[Step 5] ANALYZE — Categorize failures (infra vs task quality vs genuine difficulty)
    │
    ▼
[Step 6] FIX — Improve filters, prompts, Dockerfiles, test.sh
    │
    ▼
[Step 7] UPLOAD — Push to HuggingFace when quality is acceptable
    │
    ▼
    Repeat Steps 4-6 until pass rate stabilizes
```

**Core idea**: We take test code (pytest, JUnit, etc.) from open-source datasets, use an LLM to write a task description ("implement code that passes these tests"), and package it as a harbor task. The agent then tries to solve it. If the agent fails because the task is genuinely hard, that's good training data. If it fails because the Dockerfile is broken or the test.sh has bugs, we fix that.

---

## 2. Data Sources

### 2.1 The Stack (bigcode/the-stack)

Used by `data/dclm-mine/` generators. Pre-filtered by language on HuggingFace.

```python
from datasets import load_dataset
ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
```

- Direct source files from GitHub repositories
- Pre-organized by language subdirectory
- `content` column has the full file text
- High quality: complete, self-contained files

### 2.2 Nemotron-Pretraining-Code-v2 (nvidia/Nemotron-Pretraining-Code-v2)

Used by `data/nemotron-mine/` for Python and C++ only.

```python
ds = load_dataset("nvidia/Nemotron-Pretraining-Code-v2", name="Synthetic-Rewriting", split="train", streaming=True)
```

**Available configs** (only 2 have usable code):
- `Synthetic-Rewriting` — Python only. LLM-rewritten code. `content` column.
- `Synthetic-Transpilation` — C++ only. Transpiled from Python. `content` column.
- `Synthetic-Code-Review` — Python/C++ code reviews (not raw code)
- `Synthetic-Question-Answering` — Q&A format (not raw code)
- `Synthetic-Student-Teacher` — Educational format (not raw code)
- `Nemotron-Code-Metadata` — **Metadata only** (repo, commit_id, rel_path, language). NO code content.

### 2.3 Nemotron-CC-Code-v1 (nvidia/Nemotron-CC-Code-v1)

Used by `data/nemotron-mine/` for bash, java, C#, ruby, rust.

```python
ds = load_dataset("nvidia/Nemotron-CC-Code-v1", split="train", streaming=True)
```

- 216M web pages from Common Crawl, cleaned by Phi-4
- Code is embedded in markdown code blocks: ` ```lang\n...\n``` `
- `text` column has the full web page (not just code)
- Must extract code blocks via regex — see `nemotron_loader.py`
- **Gated dataset** — requires HuggingFace access approval

**Code block extraction** (from `nemotron_loader.py`):
```python
def _extract_code_blocks(text: str, fence_labels: list[str]) -> list[str]:
    labels_pattern = "|".join(re.escape(label) for label in fence_labels)
    pattern = rf"```(?:{labels_pattern})\s*\n(.*?)```"
    blocks = []
    for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
        code = match.group(1).strip()
        if len(code) > 50:  # Skip trivial snippets
            blocks.append(code)
    return blocks
```

**Language → fence label mapping**:
```python
CC_CODE_FENCE_LABELS = {
    "shell":   ["bash", "sh", "shell"],
    "python":  ["python", "py", "python3"],
    "c++":     ["cpp", "c++", "cxx"],
    "c-sharp": ["csharp", "cs", "c#"],
    "java":    ["java"],
    "ruby":    ["ruby", "rb"],
    "rust":    ["rust", "rs"],
    "javascript": ["javascript", "js", "typescript", "ts"],
}
```

### 2.4 Language → Dataset Routing

```python
# In nemotron_loader.py
V2_LANGUAGE_CONFIG = {
    "python": "Synthetic-Rewriting",   # Direct content column
    "c++": "Synthetic-Transpilation",  # Direct content column
}
# Everything else → CC-Code-v1 with code block extraction
```

| Language | Dataset | Method | Scan Required |
|----------|---------|--------|---------------|
| Python | V2 Synthetic-Rewriting | Direct `content` | ~6M for 5000 |
| C++ | V2 Synthetic-Transpilation | Direct `content` | ~892K for 5000 |
| Bash | CC-Code-v1 | Extract ` ```bash ` blocks | ~81K for 25 |
| Java | CC-Code-v1 | Extract ` ```java ` blocks | ~133K for 25 |
| C# | CC-Code-v1 | Extract ` ```csharp ` blocks | ~406K for 25 |
| Ruby | CC-Code-v1 | Extract ` ```ruby ` blocks | ~756K for 25 |
| Rust | CC-Code-v1 | Extract ` ```rust ` blocks | ~6.6M for 25 |

---

## 3. Generator Architecture

Every generator follows the same 4-step pattern. Here's the full anatomy using C++ as an example.

### 3.1 Step 1: Filter Source Content

Each generator has a filter function that streams the dataset and applies language-specific heuristics.

**Example: C++ test file detection** (`generate_cpp_test_tasks.py`):
```python
def is_cpp_test_file(content: str) -> tuple[bool, dict]:
    """Detect Google Test, Catch2, or Doctest files."""
    # Must have test macros
    has_gtest = bool(re.search(r'\bTEST(_F)?\s*\(', content))
    has_catch = bool(re.search(r'\bTEST_CASE\s*\(', content))
    has_doctest = bool(re.search(r'\bDOCTEST_', content))

    if not (has_gtest or has_catch or has_doctest):
        return False, {}

    # Count test functions (minimum 2)
    test_count = len(re.findall(r'\bTEST(_F|_CASE)?\s*\(', content))
    if test_count < 2:
        return False, {}

    # Must have assertions
    has_assert = bool(re.search(r'\b(EXPECT_|ASSERT_|REQUIRE|CHECK)\w*\s*\(', content))
    if not has_assert:
        return False, {}

    return True, {"test_count": test_count, "framework": "gtest/catch2/doctest"}
```

**Example: pytest file detection** (`generate_pytest_tasks_gpt5mini.py`):
```python
HEAVY_FRAMEWORKS = {
    'django', 'flask', 'fastapi', 'tornado', 'aiohttp', 'starlette',
    'selenium', 'playwright', 'rasterio', 'gdal', 'tensorflow', 'torch',
    'pandas', 'numpy', 'sqlalchemy', 'celery', 'redis', 'boto3', ...
}

def is_pytest_file(content: str) -> tuple[bool, dict]:
    # Must have pytest markers
    if 'def test_' not in content:
        return False, {}

    # AST parse validation
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return False, {}

    # Reject heavy framework imports
    imports = extract_imports(tree)
    if imports & HEAVY_FRAMEWORKS:
        return False, {}

    # Max 2 non-stdlib imports
    if len(non_stdlib_imports) > 2:
        return False, {}

    # Complexity limits
    if count_test_functions(tree) > 15:
        return False, {}

    return True, {"test_count": count_test_functions(tree)}
```

**Scanning pattern** (all generators):
```python
def filter_from_nemotron(limit: int = 5000) -> list[dict]:
    ds = load_nemotron_stream(language)
    samples = []
    scanned = 0
    for raw_sample in tqdm(ds, total=limit):
        scanned += 1
        # For CC-Code-v1: extract code blocks first
        if "text" in raw_sample and "content" not in raw_sample:
            content = extract_code_from_sample(raw_sample, language_key)
            if not content:
                continue
        else:
            content = raw_sample["content"]

        is_valid, metadata = is_test_file(content)
        if is_valid:
            samples.append({"text": content, **metadata})
            if len(samples) >= limit:
                break

    print(f"Scanned {scanned:,} files, found {len(samples)} test files")
    return samples
```

### 3.2 Step 2: Generate Task Descriptions via LLM

Uses `data/completions.py` which wraps `bespokelabs-curator` for batch LLM completions.

```python
from data.completions import run_completions
from datasets import Dataset

# Convert samples to HuggingFace Dataset
dataset = Dataset.from_list(samples)

# Run LLM completions
result = run_completions(
    dataset,
    model="gpt-5-nano-2025-08-07",
    map_type="chat",
    map_config={
        "user_message": PROMPT_TEMPLATE,     # Template with {{text}} placeholder
        "output_column": "task_description", # Column name for LLM output
    },
    max_requests_per_minute=500,
    max_tokens_per_minute=1_000_000,
    require_all_responses=False,  # Allow partial results
)

task_descriptions = result.dataset["task_description"]
```

**How templating works**: The prompt template contains `{{column_name}}` placeholders that get replaced with the corresponding column value from each dataset row. For example, `{{text}}` gets replaced with the test code content.

**Example prompt** (C++):
```
You are an expert at creating coding tasks from C++ test code.
Given the following C++ test file, create a task description that asks the developer
to implement the code that these tests verify.

Test code:
{{text}}

Write a clear task description including:
1. What to implement
2. Expected behavior based on the tests
3. Any constraints visible in the test assertions
```

### 3.3 Step 3: Create Harbor Task Directories

```python
def create_harbor_task(task_dir, index, description, test_code, prefix, metadata):
    task_name = f"{prefix}-{index:04d}"
    task_path = Path(task_dir) / task_name

    # Create directory structure
    task_path.mkdir(parents=True)
    (task_path / "environment").mkdir()
    (task_path / "tests").mkdir()

    # Write instruction.md
    (task_path / "instruction.md").write_text(description)

    # Write task.toml (standard config)
    (task_path / "task.toml").write_text(create_standard_task_toml())

    # Write Dockerfile (language-specific)
    (task_path / "environment" / "Dockerfile").write_text(DOCKERFILE_TEMPLATE)

    # Write test files
    (task_path / "tests" / "test.sh").write_text(create_test_sh(test_code))
    (task_path / "tests" / "TestSolution.java").write_text(test_code)  # or test_solution.py etc.

    # Write metadata
    (task_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
```

### 3.4 Step 4: Upload to HuggingFace

```python
from data.commons import upload_tasks_to_hf

repo_url = upload_tasks_to_hf(task_dir, "DCAgent/exp_rpt_nemotron-cpp")
# Returns: https://huggingface.co/datasets/DCAgent/exp_rpt_nemotron-cpp
```

**What upload_tasks_to_hf does**:
1. Creates HF repo if it doesn't exist
2. Converts task directories to Parquet format
3. Uploads via `HfApi.upload_folder()`
4. Supports `SKIP_HF_UPLOAD=1` env var to skip

---

## 4. Harbor Task Format

Every task is a directory with this exact structure:

```
nemotron-cpp-0042/
├── instruction.md          # What the agent should do
├── task.toml               # Timeout and config
├── metadata.json           # Source tracking
├── environment/
│   └── Dockerfile          # Runtime environment
└── tests/
    ├── test.sh             # Verification script (MUST write reward.txt)
    └── test_solution.cpp   # Original test code
```

### 4.1 instruction.md

Plain text/markdown task description. This is what the agent sees.

```markdown
### Task

Implement a `JpegCompression` class in C++ with the following:

1. Constructor takes `quality_lower` and `quality_upper` parameters
2. Validate: 0 <= quality_lower <= quality_upper <= 100
3. Method `get_param()` returns random quality in range
4. Method `apply(image)` applies JPEG compression

Place your implementation in `/app/solution.cpp`.
```

### 4.2 task.toml

Standard configuration:

```toml
[agent]
timeout_sec = 900.0        # 15 minutes for agent to work

[verifier]
timeout_sec = 720.0        # 12 minutes for tests to run
restart_environment = false
```

### 4.3 Dockerfile

Language-specific. Examples:

**Python (pytest)**:
```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y python3 python3-pip python3-venv bsdutils bash && rm -rf /var/lib/apt/lists/*
WORKDIR /app
```

**C++**:
```dockerfile
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y g++ cmake make libgtest-dev bash bsdutils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
```

**Java (JUnit)**:
```dockerfile
FROM eclipse-temurin:17-jdk-jammy
RUN apt-get update && apt-get install -y maven bash bsdutils && rm -rf /var/lib/apt/lists/*
WORKDIR /app
```

**C# (xUnit/NUnit)**:
```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:8.0
WORKDIR /app
RUN apt-get update && apt-get install -y bash && rm -rf /var/lib/apt/lists/*
```

**Note**: `bsdutils` is required for the `script` command used by terminus-2 agent.

### 4.4 test.sh

**Critical**: test.sh MUST write the exit code to `/logs/verifier/reward.txt`. This is how Harbor determines pass/fail.

```bash
#!/bin/bash
set -e

# Create log directory
mkdir -p /logs/verifier

# Trap: write exit code as reward on exit
cleanup() {
    echo $? > /logs/verifier/reward.txt
}
trap cleanup EXIT

# Run the actual tests (language-specific)
pytest /tests/test_solution.py -v --tb=short
```

The pattern is always:
1. `mkdir -p /logs/verifier`
2. `trap` to write exit code to `reward.txt`
3. Run tests — if they pass, exit 0 (reward=0, which means success). If they fail, exit non-zero.

**Wait, important clarification**: The reward file contains the exit code. `0` = tests passed = reward 1.0 in Harbor. Non-zero = tests failed = reward 0.0. Harbor inverts the convention internally.

### 4.5 metadata.json

Tracks provenance:
```json
{
  "source_path": "path/to/original/file.py",
  "source_repo": "github-user/repo-name",
  "test_count": 5,
  "framework": "pytest",
  "target_module": "mymodule"
}
```

---

## 5. LLM Instruction Generation

### 5.1 The completions.py System

Located at `data/completions.py`. Wraps `bespokelabs-curator` for batch LLM API calls.

**Key class: ChatMap**
- Takes a prompt template with `{{column_name}}` placeholders
- For each row in the dataset, substitutes values and sends to LLM
- Collects responses into a new column

**Core function: run_completions()**
```python
def run_completions(
    dataset,                        # HuggingFace Dataset
    model="gpt-5-nano-2025-08-07",  # LLM model
    map_type="chat",                # "chat" or "list"
    map_config={
        "user_message": "...",      # Prompt template with {{placeholders}}
        "output_column": "...",     # Column to store responses
    },
    max_requests_per_minute=500,
    max_tokens_per_minute=1_000_000,
    require_all_responses=False,
) -> CuratorResponse:
```

### 5.2 Prompt Design Principles

1. **Show the test code**: Always include `{{text}}` so the LLM sees the actual tests
2. **Be specific about output format**: "Write a task description", "Place solution in /app/"
3. **Constrain complexity**: "Keep it simple", "Only use standard library"
4. **Include file paths**: Tell the agent exactly where to put files

### 5.3 Cost and Performance

For gpt-5-nano with 5000 tasks:
- ~$0.05 per batch (25 tasks)
- ~$8-10 for 5000 tasks
- ~2 minutes per batch of 25
- ~15-20 minutes for 5000 tasks

---

## 6. Harbor Evaluation System

### 6.1 What is Harbor?

Harbor is a framework for evaluating AI agents on coding tasks. It:
1. Builds a Docker environment from the task's Dockerfile
2. Gives the agent the instruction.md
3. Lets the agent work (write code, run commands)
4. Runs test.sh to verify the solution
5. Records pass/fail as reward

### 6.2 Key Components

**terminus-2**: The AI agent that attempts to solve tasks. It:
- Reads instruction.md
- Plans an approach
- Writes code files
- Runs build/test commands
- Has a 900-second timeout

**daytona**: Sandboxed container execution environment:
- Builds Docker images per task
- Isolated per-task environments
- Managed sandbox lifecycle

### 6.3 Running Harbor Evaluation

```bash
harbor jobs start \
  -p /path/to/task/directory \     # Directory containing task subdirs
  --n-concurrent 25 \              # Parallel task evaluations
  --agent terminus-2 \             # Agent type
  --model openai/gpt-5-nano-2025-08-07 \  # Model for the agent
  --env daytona \                  # Execution environment
  --n-attempts 1 \                 # Attempts per task
  --max-retries 0 \                # Don't retry on failure
  --job-name my_eval_job \         # Job identifier
  --config eval/tacc/dcagent_eval_config.yaml  # Config file
```

### 6.4 Configuration File

`eval/tacc/dcagent_eval_config.yaml`:
```yaml
jobs_dir: jobs
n_attempts: 3
timeout_multiplier: 1.0
orchestrator:
  type: local
  n_concurrent_trials: 4
  quiet: false
  retry:
    max_retries: 100
    exclude_exceptions:
      - AgentTimeoutError
      - VerifierTimeoutError
environment:
  type: daytona
  force_build: true
  delete: false
agents:
  - name: terminus-2
```

### 6.5 What Happens During Evaluation

For each task:
1. Harbor reads `environment/Dockerfile` and builds the container
2. Container starts with Daytona
3. Agent receives `instruction.md` content
4. Agent executes actions (file writes, shell commands) inside the container
5. After agent finishes (or times out at 900s), verifier runs `tests/test.sh`
6. test.sh writes exit code to `/logs/verifier/reward.txt`
7. Harbor reads reward.txt: 0 = pass (reward 1.0), non-zero = fail (reward 0.0)
8. Container is destroyed

### 6.6 Concurrency Recommendations

| Tasks | Recommended --n-concurrent |
|-------|---------------------------|
| 25 | 25 (all parallel) |
| 100 | 25-50 |
| 500 | 50-128 |

Higher concurrency = faster but more Daytona infrastructure pressure. With 25 concurrent, a 25-task evaluation takes ~20-45 minutes depending on task complexity.

---

## 7. Result Parsing

### 7.1 Result File Location

```
jobs/{job_name}/result.json
```

### 7.2 Result JSON Structure

```json
{
  "stats": {
    "n_trials": 25,
    "n_errors": 5,
    "evals": {
      "terminus-2__gpt-5-nano-2025-08-07__task_set": {
        "n_trials": 25,
        "n_errors": 5,
        "metrics": [
          {"mean": 0.24}
        ],
        "reward_stats": {
          "reward": {
            "0.0": ["task-id-1", "task-id-2", ...],
            "1.0": ["task-id-3", "task-id-4", ...]
          }
        },
        "exception_stats": {
          "DaytonaError": ["task-id-a", ...],
          "AgentTimeoutError": ["task-id-b", ...],
          "EnvironmentStartTimeoutError": ["task-id-c", ...]
        }
      }
    }
  }
}
```

### 7.3 Parsing Code

From `test_all_nemotron.py`:
```python
def parse_harbor_results(result_file):
    with open(result_file) as f:
        data = json.load(f)

    stats = data.get("stats", {})
    total = stats.get("n_trials", 0)
    n_errors = stats.get("n_errors", 0)

    evals = stats.get("evals", {})
    passed = 0
    mean_reward = 0.0
    for eval_key, eval_data in evals.items():
        metrics = eval_data.get("metrics", [{}])
        mean_reward = metrics[0].get("mean", 0.0) if metrics else 0.0
        reward_stats = eval_data.get("reward_stats", {}).get("reward", {})
        for reward_val, task_list in reward_stats.items():
            if float(reward_val) > 0:
                passed += len(task_list)

    pass_rate = (passed / total * 100) if total > 0 else 0
    return {"total": total, "passed": passed, "errors": n_errors, "pass_rate": pass_rate}
```

### 7.4 Error Types

| Error | Meaning | Fixable? |
|-------|---------|----------|
| DaytonaError | Sandbox creation/management failed | Infrastructure issue, retry |
| EnvironmentStartTimeoutError | Docker build took >600s | Simplify Dockerfile |
| AgentTimeoutError | Agent exceeded 900s | Task may be too complex |
| RewardFileNotFoundError | test.sh didn't write reward.txt | Fix test.sh |

---

## 8. Quality Improvement Loop

This is the most important part. Raw generators produce tasks with 0-60% pass rates. The improvement loop raises them.

### 8.1 Failure Categories

When analyzing failures, categorize each into:

**Category A: Infrastructure errors** (fixable, not the agent's fault)
- Dockerfile missing dependencies
- test.sh bugs (wrong paths, missing reward.txt)
- Environment timeout (Docker image too large)
- Daytona sandbox failures

**Category B: Task quality issues** (fixable, task is poorly defined)
- Unclear instructions (agent doesn't know what to do)
- Impossible tasks (test code references external systems)
- Overly complex tasks (too many files, too many dependencies)
- Wrong test expectations (tests test the wrong thing)

**Category C: Genuine difficulty** (KEEP THESE — this is good training data)
- Agent understands the task but can't implement it
- Algorithm/logic challenges
- Multi-step reasoning required

### 8.2 Analysis Process

For each failing task:

1. **Read the agent trajectory**: `jobs/{job_name}/{task_id}/agent/trajectory.json`
2. **Check test.sh output**: Did tests run? What error?
3. **Check Dockerfile**: Did the environment build?
4. **Read instruction.md**: Is the task clear?
5. **Classify the failure**: Category A, B, or C

### 8.3 Common Fixes (with examples from GOOD_DATASETS.md)

**Fix: Dockerfile dependencies**
- stack-php: 0% → 100%. PHPUnit needed PHAR download, not Composer
- methods2test: 0% → 28%. Java image changed to eclipse-temurin:17-jdk-jammy
- codeelo: 0% → 52%. Removed broken openjdk dependency

**Fix: test.sh harness**
- stack-pytest: 0% → 40%. Added PYTHONPATH=/app, fixed import resolution
- pymethods2test: 0% → 40%. Auto-copy any .py to solution.py (agents name files differently)
- stack-jest: 0% → 8%. Added bsdutils for `script` command

**Fix: Filter quality**
- pytest (Nemotron): 0% → 12%. Added heavy framework blocklist, AST validation, import limits
- bigcodebench: 0% → 54%. Added stdlib-only filter, fixed libs parsing

**Fix: Instruction clarity**
- 8 datasets improved by +62% average. Changed prompts from variable names to actual values.
  Example: "parse_workflow(wf)" → "parse_workflow('on: push\njobs:\n  build: {}')"

**Fix: LLM prompt improvement**
- e2egit: 0% → 48%. Changed from E2E Selenium tests to "convert to pytest unit tests"
- stack-selfdoc: 0% → 16%. Added Python3 validation + dependency detection

### 8.4 Iteration History Example

**stack-pytest**:
- v1: 0% pass rate — tests couldn't import solution files
- Investigation: Agent creates `my_module.py` but tests do `from solution import *`
- Fix: test.sh now auto-copies any .py to solution.py + PYTHONPATH fix
- v2: 40% pass rate — acceptable

**Nemotron pytest**:
- v1: 0% pass rate — all tasks were enterprise-level complexity (Django, Selenium)
- Investigation: Synthetic-Rewriting produces overly complex code
- Fix v2: Added heavy framework blocklist + max 2 non-stdlib imports
- v2: 12% pass rate
- Fix v3: Tightened too much (120 lines, 10 tests max)
- v3: 8% pass rate — worse! Over-filtering.
- Reverted to v2 settings: 12% — kept this

---

## 9. HuggingFace Upload

### 9.1 Upload Function

From `data/commons.py`:
```python
def upload_tasks_to_hf(task_dir: str, repo_id: str) -> str:
    """Upload task directory to HuggingFace Hub."""
    if os.environ.get("SKIP_HF_UPLOAD") == "1":
        return f"Skipped (SKIP_HF_UPLOAD=1)"

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=task_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"
```

### 9.2 Naming Convention

```
DCAgent/exp_rpt_{source}-{language}[-{version}]
```

Examples:
- `DCAgent/exp_rpt_stack-pytest-v2`
- `DCAgent/exp_rpt_nemotron-cpp`
- `DCAgent/exp_rpt_nemotron-junit`

### 9.3 Environment Variables

- `HF_TOKEN`: HuggingFace authentication token
- `SKIP_HF_UPLOAD=1`: Skip upload (for testing)

---

## 10. Running the Pipeline

### 10.1 Environment Setup

```bash
# Activate conda environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6

# Set API keys
source /scratch/10000/eguha3/old-dc-agent/secret.env

# Set Python path
export PYTHONPATH="/scratch/10000/eguha3/dc-agent:$PYTHONPATH"
```

### 10.2 Generate Tasks (single generator)

```bash
cd /scratch/10000/eguha3/dc-agent

# Generate 5000 C++ tasks and upload to HF
python data/nemotron-mine/generate_cpp_test_tasks.py

# Generate 5000 pytest tasks
python data/nemotron-mine/generate_pytest_tasks_gpt5mini.py
```

### 10.3 Test with Harbor (small sample)

```bash
# Test 25 tasks from specific generators
python data/nemotron-mine/test_all_nemotron.py cpp pytest bash junit --limit 25

# Generate only (no harbor eval)
python data/nemotron-mine/test_all_nemotron.py cpp --limit 25 --generate-only
```

### 10.4 Run All Generators in Parallel

```bash
# Launch 4 generators simultaneously (each in background)
python data/nemotron-mine/generate_cpp_test_tasks.py &
python data/nemotron-mine/generate_junit_tasks.py &
python data/nemotron-mine/generate_bash_tasks.py &
python data/nemotron-mine/generate_pytest_tasks_gpt5mini.py &
wait
```

### 10.5 Adjusting Parameters

Each generator has these configurable constants at the top:
```python
LIMIT = 5_000              # Number of tasks to generate
MODEL = "gpt-5-nano-2025-08-07"  # LLM for instruction generation
```

In `test_all_nemotron.py`:
```python
N_CONCURRENT = 25  # Harbor concurrent evaluations
```

---

## 11. Subagent Strategy

When using Claude Code to execute this pipeline, subagents were used for:

### 11.1 Parallel Generation

Launch multiple generators as background tasks:
```
# Each runs independently — no dependencies between languages
Background task 1: python generate_cpp_test_tasks.py
Background task 2: python generate_junit_tasks.py
Background task 3: python generate_bash_tasks.py
Background task 4: python generate_pytest_tasks_gpt5mini.py
```

### 11.2 Failure Analysis

Use Explore subagents to investigate failures:
```
Subagent prompt: "Compare the C# generators between dclm-mine and nemotron-mine.
Check filter functions, Dockerfiles, test.sh templates, LLM prompts, and sample
task quality. Identify what's different and what could explain 0% pass rate."
```

### 11.3 Codebase Research

Use Explore subagents to understand existing patterns before creating new generators:
```
Subagent prompt: "Read data/dclm-mine/generate_cpp_test_tasks.py completely.
Document the filter function, prompt template, Dockerfile, test.sh, and
upload process. I need to create a Nemotron version of this."
```

### 11.4 Monitoring Long-Running Tasks

Harbor evaluations take 20-65 minutes. Monitor with periodic checks:
```bash
# Check progress
grep -E "(pass_rate|Mean|Scanned)" /path/to/task/output

# Check harbor job progress
tail -5 jobs/{job_name}/job.log
```

---

## 12. File Reference

### 12.1 Core Files

| File | Purpose |
|------|---------|
| `data/completions.py` | LLM completion pipeline (wraps bespokelabs-curator) |
| `data/commons.py` | Shared utilities: Dockerfiles, task.toml, HF upload |
| `eval/tacc/dcagent_eval_config.yaml` | Harbor evaluation config |

### 12.2 Nemotron-Mine Files

| File | Purpose |
|------|---------|
| `data/nemotron-mine/nemotron_loader.py` | Dataset loading + code block extraction |
| `data/nemotron-mine/generate_cpp_test_tasks.py` | C++ generator (V2 Synthetic-Transpilation) |
| `data/nemotron-mine/generate_pytest_tasks_gpt5mini.py` | pytest generator (V2 Synthetic-Rewriting) |
| `data/nemotron-mine/generate_junit_tasks.py` | JUnit generator (CC-Code-v1) |
| `data/nemotron-mine/generate_bash_tasks.py` | Bash generator (CC-Code-v1) |
| `data/nemotron-mine/generate_rspec_tasks.py` | Ruby RSpec generator (CC-Code-v1) |
| `data/nemotron-mine/generate_csharp_test_tasks.py` | C# generator (CC-Code-v1) |
| `data/nemotron-mine/generate_rust_test_tasks.py` | Rust generator (CC-Code-v1) |
| `data/nemotron-mine/test_all_nemotron.py` | Unified test harness |
| `data/nemotron-mine/nemotron_test_results.json` | Evaluation results |

### 12.3 Reference Files

| File | Purpose |
|------|---------|
| `data/GOOD_DATASETS.md` | Quality tracking for all 38 working datasets |
| `data/all_good_datasets.json` | Machine-readable dataset manifest |
| `data/dclm-mine/README.md` | Stack pipeline documentation |

---

## 13. Lessons Learned

### 13.1 Data Source Quality Matters Most

The pipeline (filter, prompt, Dockerfile, test.sh) is identical between Stack and Nemotron for each language. The **only** difference is the data source. Results:

| Language | Nemotron | Stack | Why |
|----------|----------|-------|-----|
| C++ | **60%** | 36% | V2 Synthetic has clean, complete code |
| JUnit | 24% | 24% | Tie — CC-Code-v1 Java is decent |
| Bash | **12%** | 4% | CC-Code-v1 bash snippets work |
| pytest | **12%** | 0-10% | V2 Synthetic + complexity filtering |
| C# | 0% | **12%** | CC-Code-v1 extracts incomplete code |
| Ruby | 0% | 0% | Both sources produce hard Ruby tasks |
| Rust | 0% | 0% | Both sources produce hard Rust tasks |

**CC-Code-v1 web page extraction** produces lower quality than Stack's repository files because:
- Code blocks from tutorials are often incomplete
- Missing external dependencies and helper classes
- Truncated or fragmentary snippets

### 13.2 Don't Over-Filter

When pytest v2 (12%) was tightened to v3 (max 120 lines, max 10 tests, deep import rejection), pass rate **dropped** to 8%. Over-filtering removes good tasks along with bad ones. Find the sweet spot.

### 13.3 Infrastructure Errors Dominate

For many generators, 50-80% of "failures" are DaytonaError, EnvironmentStartTimeoutError, or AgentTimeoutError — not actual test failures. The true pass rate among tasks that actually ran is much higher. When reporting results, note the error count.

### 13.4 Parallelism is Free

All generators are independent. Always run them in parallel. With 4 generators, this cuts wall-clock time by 4x. Similarly, set `--n-concurrent` to match your task count (25 tasks → 25 concurrent).

### 13.5 The Improvement Loop Has Diminishing Returns

Most improvement comes from fixing obvious infrastructure issues (wrong Dockerfile, broken test.sh). After that, improvements come from filter quality. After that, you're limited by the data source quality and the agent's capability. Know when to stop iterating.
