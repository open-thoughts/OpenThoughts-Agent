# Trace Analysis Pipeline

We add [Docent](https://docent.transluce.org) integration for trace analysis:

---

## Prerequisites

# Get a Docent API KEY:

```bash
export DOCENT_API_KEY=your_api_key_here
```

---

## Step 1: Upload Traces to Docent

```bash
uv run python eval/trace_analysis/upload_data.py \
    --dataset <HF_DATASET_ID>
```

### Examples

```bash
# Upload a dataset
uv run python eval/trace_analysis/upload_data.py \
    --dataset DCAgent2/DCAgent_dev_set_71_tasks_laion_exp_tas_full_thinking_traces_20260102_073856

# Dry run (validate without uploading)
uv run python eval/trace_analysis/upload_data.py \
    --dataset DCAgent2/dataset1 --dry-run

# Add to existing collection
uv run python eval/trace_analysis/upload_data.py \
    --dataset DCAgent2/dataset1 --collection-id <EXISTING_COLLECTION_ID>
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--dataset`, `-d` | HuggingFace dataset name(s) to upload (required) |
| `--split` | Dataset split (default: train) |
| `--collection-name`, `-n` | Name for Docent collection |
| `--collection-id`, `-c` | Use existing Docent collection |
| `--dry-run` | Process without uploading |
| `--max-rows` | Limit rows processed (0 = no limit) |
| `--verbose`, `-v` | Enable debug logging |

### Output

The script outputs a **Docent Collection ID**:
```
Success! Collection ID: a5a7fc29-8838-4930-b5cb-e897ecde79a8
```

---

## Step 2: Evaluate Traces in Docent Dashboard

Open your collection in the Docent dashboard to create and run rubrics:

```
https://docent.transluce.org/dashboard/<COLLECTION_ID>/rubric
```

1. Create rubrics using natural language prompts
2. Run the rubrics on your traces
3. View results in the dashboard

---

## Supported Dataset Formats

### 1. Trajectory Format
```json
{
  "trajectory": {
    "steps": [
      {"action": "...", "observation": "..."}
    ]
  }
}
```

### 2. Conversations Format (ShareGPT)
```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool", "content": "..."}
  ]
}
```

---

## Trace Datasets Location

Search HuggingFace for evaluation trace datasets:
- [DCAgent datasets](https://huggingface.co/DCAgent)
- [DCAgent2 datasets](https://huggingface.co/DCAgent2)

---

## Directory Structure

```
eval/trace_analysis/
├── README.md              # This file
├── upload_data.py         # Upload traces to Docent
└── download_rubric.py     # Download rubric definitions
```
