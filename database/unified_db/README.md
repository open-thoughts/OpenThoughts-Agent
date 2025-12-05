# OT-Agents Dataset and Model Registry

A comprehensive registration system for managing datasets, ML models, and sandbox evaluation results with support for HuggingFace and local files.

## Features

### Dataset Registration
- ✅ **HuggingFace Integration** - Register datasets directly from HuggingFace Hub
- ✅ **Local Parquet Support** - Register local parquet files with automatic analysis
- ✅ **Auto-filling** - Automatically populate 10+ metadata fields
- ✅ **Type Validation** - Support for SFT and RL dataset types
- ✅ **Comprehensive Metadata** - Rich metadata including file analysis and HF fingerprints

### Model Registration
- ✅ **HuggingFace Models** - Register models from HuggingFace Hub with metadata extraction
- ✅ **Local Models** - Register local model directories with framework detection
- ✅ **Auto-detection** - Automatically detect PyTorch, TensorFlow, ONNX, and Transformers models
- ✅ **Training Tracking** - Link models to datasets, agents, and training runs
- ✅ **Forced Updates** - Update existing entries with new metadata

### Benchmark Registration
- ✅ **Internal Benchmarks** - Register custom evaluation benchmarks
- ✅ **External Benchmarks** - Register external benchmarks with links
- ✅ **Version Tracking** - Support for benchmark version hashing
- ✅ **Minimal Fields** - Only name required, all other fields optional
- ✅ **Forced Updates** - Update existing benchmark entries

### Sandbox Evaluation
- ✅ **Task Management** - Register sandbox tasks with checksum-based deduplication
- ✅ **Benchmark-Task Linking** - Many-to-many relationships between benchmarks and tasks
- ✅ **Evaluation Jobs** - Track complete evaluation runs with agent, model, and benchmark
- ✅ **Trial Tracking** - Individual task execution results with detailed timing
- ✅ **Model Usage** - Token consumption tracking per trial
- ✅ **Minimal Auto-filling** - Only system fields auto-populated

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Database Setup

1. Create a Supabase project
2. Copy your credentials to `.env`:
```bash
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
```

3. Apply the database schema:
```bash
# Apply complete schema (all tables including sandbox)
psql $DATABASE_URL -f complete_schema.sql

# Or apply individual schemas:
psql $DATABASE_URL -f dataset_schema.sql    # Datasets table only
psql $DATABASE_URL -f model_schema.sql      # Models table only
psql $DATABASE_URL -f agents_schema.sql     # Agents table only
psql $DATABASE_URL -f benchmarks_schema.sql # Benchmarks table only
psql $DATABASE_URL -f sandbox_schema.sql    # Sandbox evaluation tables only
```

## Python API

### HuggingFace Dataset Registration

```python
from unified_db import register_hf_dataset

# Register a new dataset
result = register_hf_dataset(
    repo_name="openai/gsm8k",
    dataset_type="SFT",
    name="GSM8K Math Problems",
    created_by="OpenAI"
)

if result['success']:
    dataset = result['dataset']
    print(f"Registered: {dataset['name']} with {dataset['num_tasks']} tasks")
    if result.get('exists'):
        print("Dataset already existed")

# Force update an existing dataset
result = register_hf_dataset(
    repo_name="openai/gsm8k",
    dataset_type="SFT",
    name="GSM8K Math Problems",
    created_by="OpenAI Team",
    forced_update=True  # This will update the existing entry
)

if result['success'] and result.get('updated'):
    print("Dataset updated successfully")
```

### Local Parquet Registration

```python
from unified_db import register_local_parquet

# Register a new dataset
result = register_local_parquet(
    file_path="/path/to/data.parquet",
    name="My Custom Dataset",
    created_by="researcher@company.com",
    dataset_type="RL"
)

if result['success']:
    dataset = result['dataset']
    print(f"Registered: {dataset['name']} ({dataset['num_tasks']} rows)")

# Force update an existing dataset
result = register_local_parquet(
    file_path="/path/to/updated_data.parquet",
    name="My Custom Dataset",
    created_by="researcher@company.com",
    dataset_type="RL",
    forced_update=True  # This will update the existing entry
)

if result['success'] and result.get('updated'):
    print("Dataset updated successfully")
```

### Getting Dataset Information

```python
from unified_db import get_dataset_by_name

dataset = get_dataset_by_name("GSM8K Math Problems")
if dataset:
    print(f"Found dataset: {dataset['name']}")
    print(f"Created by: {dataset['created_by']}")
    print(f"Number of tasks: {dataset['num_tasks']}")
```

### HuggingFace Model Registration

```python
from unified_db import register_hf_model
from datetime import datetime, timezone
from uuid import uuid4

# Register a new model
result = register_hf_model(
    repo_name="microsoft/phi-2",
    agent_id=str(uuid4()),  # Your agent's ID
    training_start=datetime.now(timezone.utc),
    name="Fine-tuned Phi-2",
    created_by="researcher@company.com",
    training_type="SFT",
    training_parameters={
        "learning_rate": 1e-5,
        "batch_size": 32,
        "epochs": 3
    }
)

if result['success']:
    model = result['model']
    print(f"Registered: {model['name']}")
    print(f"Weights: {model['weights_location']}")
    print(f"Status: {model['training_status']}")

# Force update an existing model
result = register_hf_model(
    repo_name="microsoft/phi-2",
    agent_id=str(uuid4()),
    training_start=datetime.now(timezone.utc),
    name="Fine-tuned Phi-2",
    created_by="updated@company.com",
    forced_update=True  # Update the existing entry
)
```

### Local Model Registration

```python
from unified_db import register_local_model

# Register a local model
result = register_local_model(
    model_path="/path/to/model/directory",
    name="Custom BERT Model",
    created_by="ml-engineer@company.com",
    agent_id=str(uuid4()),
    training_start=datetime.now(timezone.utc),
    training_type="SFT",
    training_parameters={
        "optimizer": "adamw",
        "learning_rate": 5e-5,
        "warmup_steps": 500
    }
)

if result['success']:
    model = result['model']
    print(f"Registered: {model['name']}")
    print(f"Framework: {model['training_parameters']['model_framework']}")
    print(f"Location: {model['weights_location']}")
```

### Getting Model Information

```python
from unified_db import get_model_by_name

model = get_model_by_name("Fine-tuned Phi-2")
if model:
    print(f"Found model: {model['name']}")
    print(f"Created by: {model['created_by']}")
    print(f"Training type: {model['training_type']}")
    print(f"Status: {model['training_status']}")
```

### Agent Registration

```python
from unified_db import register_agent

# Register a new agent (only manual fields required)
result = register_agent(
    name="My Evaluation Agent",
    agent_version_hash="abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd",  # 64-char SHA-256 hash
    description="Agent for evaluating coding tasks"
)

if result['success']:
    agent = result['agent']
    print(f"Registered: {agent['name']}")
    print(f"Agent ID: {agent['id']}")
    print(f"Version: {agent['agent_version_hash']}")

# Register minimal agent (only name required)
result = register_agent(
    name="Minimal Agent",
    description="Agent with minimal configuration"
    # agent_version_hash is optional
)

# Force update an existing agent
result = register_agent(
    name="My Evaluation Agent",
    agent_version_hash="new_hash_1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    description="Updated agent description",
    forced_update=True  # Update the existing entry
)
```

### Getting Agent Information

```python
from unified_db import get_agent_by_name

agent = get_agent_by_name("My Evaluation Agent")
if agent:
    print(f"Found agent: {agent['name']}")
    print(f"Version hash: {agent['agent_version_hash']}")
    print(f"Description: {agent['description']}")
    print(f"Last updated: {agent['updated_at']}")
```

### Benchmark Registration

```python
from unified_db import register_benchmark

# Basic benchmark (only name required)
result = register_benchmark(name="MMLU Benchmark")

if result['success']:
    benchmark = result['benchmark']
    print(f"Registered: {benchmark['name']}")
    print(f"Benchmark ID: {benchmark['id']}")
    print(f"Is external: {benchmark['is_external']}")

# Full benchmark with all fields
result = register_benchmark(
    name="Custom Coding Benchmark",
    benchmark_version_hash="abcd1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcd",  # 64-char SHA-256 hash
    is_external=False,
    description="Internal coding evaluation benchmark with 500 problems"
)

# External benchmark
result = register_benchmark(
    name="HumanEval",
    is_external=True,
    external_link="https://github.com/openai/human-eval",
    description="OpenAI's HumanEval coding benchmark"
)

# Force update an existing benchmark
result = register_benchmark(
    name="MMLU Benchmark",
    benchmark_version_hash="new_hash_1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    description="Updated MMLU benchmark with additional subjects",
    forced_update=True  # Update the existing entry
)
```

### Getting Benchmark Information

```python
from unified_db import get_benchmark_by_name

benchmark = get_benchmark_by_name("MMLU Benchmark")
if benchmark:
    print(f"Found benchmark: {benchmark['name']}")
    print(f"Version hash: {benchmark['benchmark_version_hash']}")
    print(f"Is external: {benchmark['is_external']}")
    print(f"Description: {benchmark['description']}")
    print(f"Last updated: {benchmark['updated_at']}")
```

## Sandbox Evaluation API

### Sandbox Task Registration

```python
from unified_db import register_sandbox_task
import hashlib

# Generate checksum for task deduplication
def generate_checksum(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()

# Register a sandbox task
checksum = generate_checksum("task_unique_content")
result = register_sandbox_task(
    checksum=checksum,
    name="Coding Task 1",
    path="/benchmarks/coding/task1",
    agent_timeout_sec=300,
    verifier_timeout_sec=60,
    source="internal_benchmark",
    instruction="Implement a binary search algorithm",
    git_url="https://github.com/org/tasks",
    git_commit_id="abc123"
)

if result['success']:
    task = result['task']
    print(f"Registered task: {task['name']}")
    print(f"Checksum: {task['checksum'][:16]}...")
```

### Benchmark-Task Linking

```python
from unified_db import link_benchmark_to_task, unlink_benchmark_from_task

# Link a benchmark to multiple tasks
result = link_benchmark_to_task(
    benchmark_id="benchmark-uuid",
    task_checksum=checksum,
    benchmark_name="Coding Benchmark v1",
    benchmark_version_hash="b" * 64
)

if result['success']:
    print("Benchmark linked to task")

# Unlink when needed
unlink_result = unlink_benchmark_from_task(
    benchmark_id="benchmark-uuid",
    task_checksum=checksum
)
```

### Sandbox Job Registration

```python
from unified_db import register_sandbox_job
from datetime import datetime, timezone

# Register an evaluation job (run)
result = register_sandbox_job(
    job_name="Eval Run 2024-01-15",
    username="researcher@example.com",
    agent_id="agent-uuid",
    model_id="model-uuid",
    benchmark_id="benchmark-uuid",
    n_trials=100,
    n_rep_eval=3,
    config={
        "temperature": 0.7,
        "max_tokens": 2048,
        "batch_size": 16
    },
    git_commit_id="abc123def456",  # OR package_version (at least one required)
    metrics={"accuracy": 0.85, "pass_rate": 0.78},
    stats={"mean_time": 45.2, "std_time": 12.3}
)

if result['success']:
    job = result['job']
    print(f"Created job: {job['job_name']}")
    print(f"Job ID: {job['id']}")
```

### Sandbox Trial Registration

```python
from unified_db import register_sandbox_trial

# Register individual trial execution
result = register_sandbox_trial(
    trial_name="trial_task1_rep1",
    trial_uri="/results/eval_run/trial_001",
    task_checksum=task_checksum,
    config={"seed": 42, "temperature": 0.7},
    job_id=job_id,
    reward=1.0,  # Success score
    started_at=datetime.now(timezone.utc),
    ended_at=datetime.now(timezone.utc),
    agent_execution_started_at=datetime.now(timezone.utc),
    agent_execution_ended_at=datetime.now(timezone.utc),
    verifier_started_at=datetime.now(timezone.utc),
    verifier_ended_at=datetime.now(timezone.utc)
)

if result['success']:
    trial = result['trial']
    print(f"Created trial: {trial['trial_name']}")
    print(f"Reward: {trial['reward']}")
```

### Trial Model Usage Tracking

```python
from unified_db import register_trial_model_usage

# Track model token usage for a trial
result = register_trial_model_usage(
    trial_id=trial_id,
    model_id=model_id,
    model_provider="openai",
    n_input_tokens=1500,
    n_output_tokens=500
)

if result['success']:
    usage = result['usage']
    print(f"Recorded usage: {usage['n_input_tokens']} in, {usage['n_output_tokens']} out")
```

### Getting Sandbox Data

```python
from unified_db import (
    get_sandbox_task_by_checksum,
    get_sandbox_job_by_id,
    get_sandbox_trial_by_name
)

# Get task by checksum
task = get_sandbox_task_by_checksum(checksum)
if task:
    print(f"Task: {task['name']}")
    print(f"Timeouts: agent={task['agent_timeout_sec']}s, verifier={task['verifier_timeout_sec']}s")

# Get job by ID
job = get_sandbox_job_by_id(job_id)
if job:
    print(f"Job: {job['job_name']}")
    print(f"Metrics: {job['metrics']}")
    print(f"Stats: {job['stats']}")

# Get trial by name
trial = get_sandbox_trial_by_name("trial_task1_rep1")
if trial:
    print(f"Trial: {trial['trial_name']}")
    print(f"Reward: {trial['reward']}")
    print(f"Config: {trial['config']}")
```

### Complete Evaluation Workflow Example

```python
from unified_db import (
    register_agent, register_benchmark, register_hf_model,
    register_sandbox_task, link_benchmark_to_task,
    register_sandbox_job, register_sandbox_trial,
    register_trial_model_usage
)
import hashlib
from datetime import datetime, timezone

# 1. Register agent, model, and benchmark (prerequisites)
agent_result = register_agent(
    name="CodeAgent v1.0",
    agent_version_hash="a" * 64
)
agent_id = agent_result['agent']['id']

model_result = register_hf_model(
    repo_name="microsoft/DialoGPT-small",
    agent_id=agent_id,
    training_start=datetime.now(timezone.utc)
)
model_id = model_result['model']['id']

benchmark_result = register_benchmark(
    name="Coding Tasks v1",
    benchmark_version_hash="b" * 64
)
benchmark_id = benchmark_result['benchmark']['id']

# 2. Register tasks and link to benchmark
tasks = []
for i in range(3):
    checksum = hashlib.sha256(f"task_{i}".encode()).hexdigest()
    task_result = register_sandbox_task(
        checksum=checksum,
        name=f"Task {i+1}",
        path=f"/tasks/task_{i+1}",
        agent_timeout_sec=300,
        verifier_timeout_sec=60,
        source="coding_benchmark"
    )
    tasks.append(checksum)

    # Link to benchmark
    link_benchmark_to_task(
        benchmark_id=benchmark_id,
        task_checksum=checksum,
        benchmark_name="Coding Tasks v1",
        benchmark_version_hash="b" * 64
    )

# 3. Create evaluation job
job_result = register_sandbox_job(
    job_name="Eval Run 2024-01-15",
    username="researcher@example.com",
    agent_id=agent_id,
    model_id=model_id,
    benchmark_id=benchmark_id,
    n_trials=len(tasks),
    n_rep_eval=1,
    config={"temperature": 0.7},
    git_commit_id="abc123"
)
job_id = job_result['job']['id']

# 4. Run trials and record results
for task_checksum in tasks:
    trial_result = register_sandbox_trial(
        trial_name=f"trial_{task_checksum[:8]}",
        trial_uri=f"/results/{task_checksum[:8]}",
        task_checksum=task_checksum,
        config={"seed": 42},
        job_id=job_id,
        reward=0.85
    )
    trial_id = trial_result['trial']['id']

    # Record model usage
    register_trial_model_usage(
        trial_id=trial_id,
        model_id=model_id,
        model_provider="openai",
        n_input_tokens=1000,
        n_output_tokens=300
    )

print("✅ Complete evaluation workflow registered!")
```

## Auto-Filled Fields

### Datasets

#### HuggingFace Datasets (12 auto-filled fields)
- System: `id`, `creation_time`, `updated_at`
- HuggingFace API: `hf_fingerprint`, `hf_commit_hash`, `num_tasks`
- Smart defaults: `data_location`, `generation_status`, `creation_location`
- Extracted: `created_by` (from repo name), `name` (from repo)
- Rich metadata: `generation_parameters` with HF metadata

#### Local Parquet Files (9 auto-filled fields)
- System: `id`, `creation_time`, `updated_at`
- File analysis: `num_tasks`, `last_modified`, file size and column info
- Smart defaults: `data_location`, `generation_status`, `creation_location`
- Rich metadata: `generation_parameters` with file metadata

### Models

#### HuggingFace Models (10-12 auto-filled fields)
- System: `id`, `creation_time`, `updated_at`
- HuggingFace API: model config, tags, pipeline info
- Smart defaults: `weights_location`, `creation_location`, `is_external`, `training_status`
- Extracted: `created_by` (from repo name), `description` (from model card)
- Rich metadata: `training_parameters` with model config and HF metadata

#### Local Models (8-10 auto-filled fields)
- System: `id`, `creation_time`, `updated_at`
- Filesystem: `weights_location` (absolute path), file inventory
- Framework detection: PyTorch, TensorFlow, ONNX, Transformers
- Smart defaults: `creation_location`, `is_external`, `training_status`
- Config extraction: from `config.json` if present
- Description: from `README.md` if present

### Benchmarks

#### Manual Benchmark Registration (2 auto-filled fields)
- System: `id`, `updated_at`
- Manual: `name` (required), `benchmark_version_hash`, `is_external`, `external_link`, `description`
- All fields except `name` are optional
- Supports both internal and external benchmarks

### Sandbox Evaluation

#### Sandbox Tasks (1 auto-filled field)
- System: `created_at`
- Manual: `checksum` (PK), `name`, `path`, `agent_timeout_sec`, `verifier_timeout_sec`, `source`, `instruction`, `git_url`, `git_commit_id`
- Checksum-based deduplication (SHA-256)
- All fields except system fields are manual

#### Sandbox Jobs (3 auto-filled fields)
- System: `id`, `created_at`, `started_at` (defaults to now if not provided)
- Manual: `job_name`, `username`, `agent_id`, `model_id`, `benchmark_id`, `n_trials`, `n_rep_eval`, `config`
- Optional: `ended_at`, `metrics`, `stats`, `git_commit_id`, `package_version`
- Version constraint: `git_commit_id` OR `package_version` required

#### Sandbox Trials (2 auto-filled fields)
- System: `id`, `created_at`
- Manual: All timing fields, `trial_name`, `trial_uri`, `task_checksum`, `config`, `reward`, `exception_info`
- 10 optional timing fields for detailed execution tracking
- All fields except system fields are manual

#### Trial Model Usage (1 auto-filled field)
- System: `created_at`
- Manual: `trial_id`, `model_id`, `model_provider`, `n_input_tokens`, `n_output_tokens`
- Composite PK: (trial_id, model_id, model_provider)
- Token counts are optional (can be null)

## Schema

The system uses nine main tables organized into two groups:

### Datasets Table
```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    created_by TEXT NOT NULL,
    dataset_type TEXT CHECK (dataset_type IN ('SFT', 'RL')),
    data_location TEXT NOT NULL,
    generation_parameters JSONB NOT NULL,
    num_tasks INTEGER,
    hf_fingerprint TEXT,
    hf_commit_hash TEXT,
    -- ... additional fields
);
```

### Models Table
```sql
CREATE TABLE models (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    agent_id UUID REFERENCES agents(id) NOT NULL,
    dataset_id UUID REFERENCES datasets(id),
    base_model_id UUID REFERENCES models(id),
    created_by TEXT NOT NULL,
    weights_location TEXT NOT NULL,
    training_type TEXT CHECK (training_type IN ('SFT', 'RL')),
    training_parameters JSONB NOT NULL,
    training_start TIMESTAMP WITH TIME ZONE NOT NULL,
    training_end TIMESTAMP WITH TIME ZONE,
    -- ... additional fields
);
```

### Agents Table
```sql
CREATE TABLE agents (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    agent_version_hash CHAR(64),
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

### Benchmarks Table
```sql
CREATE TABLE benchmarks (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    benchmark_version_hash CHAR(64),
    is_external BOOLEAN NOT NULL DEFAULT false,
    external_link TEXT,
    description TEXT,
    updated_at TIMESTAMP WITH TIME ZONE
);
```

### Sandbox Tables

#### Sandbox Tasks Table
```sql
CREATE TABLE sandbox_tasks (
    checksum TEXT NOT NULL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    source TEXT,
    name TEXT NOT NULL,
    instruction TEXT NOT NULL DEFAULT '',
    agent_timeout_sec NUMERIC NOT NULL,
    verifier_timeout_sec NUMERIC NOT NULL,
    git_url TEXT,
    git_commit_id TEXT,
    path TEXT NOT NULL,
    CONSTRAINT sandbox_task_source_name_key UNIQUE (source, name)
);
```

#### Sandbox Benchmark Tasks Table (Many-to-Many)
```sql
CREATE TABLE sandbox_benchmark_tasks (
    benchmark_id UUID REFERENCES benchmarks(id) NOT NULL,
    benchmark_name TEXT NOT NULL,
    benchmark_version_hash CHAR(64) NOT NULL,
    task_checksum TEXT REFERENCES sandbox_tasks(checksum) NOT NULL,
    CONSTRAINT sandbox_benchmark_id_task_checksum UNIQUE (benchmark_id, task_checksum)
);
```

#### Sandbox Jobs Table
```sql
CREATE TABLE sandbox_jobs (
    id UUID NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    job_name TEXT NOT NULL,
    username TEXT NOT NULL,
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    git_commit_id TEXT,
    package_version TEXT,
    n_trials INTEGER NOT NULL,
    config JSONB NOT NULL,
    metrics JSONB,
    stats JSONB,
    agent_id UUID REFERENCES agents(id) NOT NULL,
    model_id UUID REFERENCES models(id) NOT NULL,
    benchmark_id UUID REFERENCES benchmarks(id) NOT NULL,
    n_rep_eval INTEGER NOT NULL,
    CONSTRAINT sandbox_job_version_check CHECK (
        git_commit_id IS NOT NULL OR package_version IS NOT NULL
    ),
    CONSTRAINT sandbox_job_agent_model_benchmark_key UNIQUE (agent_id, model_id, benchmark_id)
);
```

#### Sandbox Trials Table
```sql
CREATE TABLE sandbox_trials (
    id UUID NOT NULL PRIMARY KEY DEFAULT uuid_generate_v4(),
    trial_name TEXT NOT NULL,
    trial_uri TEXT NOT NULL,
    job_id UUID REFERENCES sandbox_jobs(id),
    task_checksum TEXT REFERENCES sandbox_tasks(checksum) NOT NULL,
    reward NUMERIC,
    -- Detailed timing fields (10 timestamp fields)
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    environment_setup_started_at TIMESTAMP WITH TIME ZONE,
    environment_setup_ended_at TIMESTAMP WITH TIME ZONE,
    agent_setup_started_at TIMESTAMP WITH TIME ZONE,
    agent_setup_ended_at TIMESTAMP WITH TIME ZONE,
    agent_execution_started_at TIMESTAMP WITH TIME ZONE,
    agent_execution_ended_at TIMESTAMP WITH TIME ZONE,
    verifier_started_at TIMESTAMP WITH TIME ZONE,
    verifier_ended_at TIMESTAMP WITH TIME ZONE,
    config JSONB NOT NULL,
    exception_info JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);
```

#### Sandbox Trial Model Usage Table
```sql
CREATE TABLE sandbox_trial_model_usage (
    trial_id UUID REFERENCES sandbox_trials(id) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    model_id UUID REFERENCES models(id) NOT NULL,
    model_provider TEXT NOT NULL,
    n_input_tokens INTEGER,
    n_output_tokens INTEGER,
    PRIMARY KEY (trial_id, model_id, model_provider)
);
```

## Auto-Filling Summary

| Registration Type | Manual Fields | Auto-Filled Fields | Key Features |
|-------------------|---------------|-------------------|-------------|
| **HF Dataset** | 2-4 fields | 12 fields | HF API metadata, smart extraction |
| **Local Parquet** | 4-5 fields | 9 fields | File analysis, size calculation |
| **HF Model** | 4-6 fields | 10-12 fields | Model config, framework detection |
| **Local Model** | 6-8 fields | 8-10 fields | Framework detection, config parsing |
| **Agent** | 1-3 fields | 2 fields | Minimal auto-filling by design |
| **Benchmark** | 1-5 fields | 2 fields | Minimal auto-filling, external support |
| **Sandbox Task** | 6-10 fields | 1 field | Checksum-based deduplication |
| **Benchmark-Task Link** | 4 fields | 0 fields | Many-to-many association |
| **Sandbox Job** | 8+ fields | 3 fields | Version constraint validation |
| **Sandbox Trial** | 4+ fields | 2 fields | Detailed timing tracking (10 fields) |
| **Trial Usage** | 3-5 fields | 1 field | Token consumption tracking |

## Dependencies

- **supabase** - Database and storage
- **datasets** - HuggingFace dataset loading
- **huggingface_hub** - HuggingFace API access
- **pandas** - Parquet file analysis
- **pydantic** - Data validation

## Testing

Run the test scripts to verify functionality:

```bash
# Test complete system (datasets, models, agents, benchmarks)
python tests/test_complete_system.py

# Test sandbox operations (tasks, jobs, trials, usage)
python tests/test_sandbox_operations.py

# Run with options
python tests/test_sandbox_operations.py --verbose       # Detailed output
python tests/test_sandbox_operations.py --no-cleanup    # Keep test data
python tests/test_sandbox_operations.py --section tasks # Run specific section
```

## License

Open source - see repository for details.