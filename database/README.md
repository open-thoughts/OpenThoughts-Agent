# OT-Agents Dataset and Model Registry

A comprehensive registration system for managing datasets and ML models with support for HuggingFace and local files.

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
# Apply complete schema (datasets, models, and agents tables)
psql $DATABASE_URL -f complete_schema.sql

# Or apply individual schemas:
psql $DATABASE_URL -f schema.sql         # Datasets table only
psql $DATABASE_URL -f model_schema.sql   # Models table only
psql $DATABASE_URL -f agents_schema.sql  # Agents table only
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

## Schema

The system uses three main tables:

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
    name TEXT NOT NULL UNIQUE,
    created_by TEXT NOT NULL,
    configuration JSONB,
    -- ... additional fields
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

## Dependencies

- **supabase** - Database and storage
- **datasets** - HuggingFace dataset loading
- **huggingface_hub** - HuggingFace API access
- **pandas** - Parquet file analysis
- **pydantic** - Data validation

## Testing

Smoke-test the Supabase integration by running the manual registration helper (edit the module-level constants first so they describe your run):

```bash
source hpc/dotenv/tacc.env  # or any dotenv that exports the Supabase + W&B creds
python scripts/database/manual_db_push.py
```

This script pulls timestamps from W&B and calls `register_trained_model`, exercising the same database APIs that dataset/model registration uses.

## License

Open source - see repository for details.
