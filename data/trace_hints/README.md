# Generic Trace Hints Pipeline

Extract hints from agent episode traces and create augmented training datasets. Works with any dataset that has `trial_name`, `episode`, and `conversations` fields.

## Quick Start

```bash
# Install dependencies
pip install openai datasets bespokelabs-curator huggingface_hub

# Set API key
export OPENAI_API_KEY='your-key'

# Run with default dataset (taskmaster2)
python augment_dataset.py \
    --dataset DCAgent/taskmaster2-gpt5mini \
    --repo-name YOUR_USERNAME/taskmaster2-hints

# Run with any custom dataset
python augment_dataset.py \
    --dataset YOUR_ORG/your-traces-dataset \
    --repo-name YOUR_USERNAME/your-dataset-hints

# Test with limited trials
python augment_dataset.py \
    --dataset DCAgent/taskmaster2-gpt5mini \
    --repo-name YOUR_USERNAME/taskmaster2-hints \
    --max-trials 10
```

## How It Works

**Dataset Structure (Generic):**
- Each row = 1 episode (agent attempt)
- Rows with same `trial_name` = multiple attempts at same task
- Required fields: `trial_name`, `episode`, `conversations`

**Pipeline:**
1. **Group**: Group dataset rows by `trial_name`
2. **Extract**: Extract hints from each episode using Curator (parallel LLM calls)
3. **Aggregate**: Combine hints from all episodes of a trial (union of tools, de-duplicate strategies)
4. **Generate**: Create augmented tasks with hints section

**Example:**
```
Trial: taskmaster2-6777__dytMiiS (Book flight)
  ├─ episode-0: Failed (loops, wrong tools)
  ├─ episode-1: Failed (missing info)
  ├─ episode-2: Success (used curl, filtered by WiFi)
  └─ ...

Aggregated Hints:
  - Tools: curl (from successful episodes)
  - Strategies: Steps from episode-2
  - Pitfalls: Mistakes from episode-0, episode-1
```

## Output

**Before** (original task):
```
User: Book flight to SFO with WiFi
Task: Do whatever the user asked.
```

**After** (with aggregated hints from 8 episodes):
```
User: Book flight to SFO with WiFi
Task: Do whatever the user asked.

---

# Hints

## Available Tools
### curl
**Purpose**: Query flight API
**Example**: `curl -X POST api.flights.com`

## Suggested Strategy
1. Extract requirements
2. Query API
3. Filter by WiFi

## Common Pitfalls
- Don't loop on missing info (from episode-0)
- Don't use fake searches (from episode-1)

## Success Criteria
Flight booked with WiFi, confirmation shown
```

## Options

- `--dataset` - HuggingFace dataset (default: DCAgent/taskmaster2-gpt5mini)
- `--model` - LLM model (default: gpt-4o-mini)
- `--repo-name` - Output HF repo (required)
- `--dataset-prefix` - Prefix for task directories (default: auto from dataset name)
- `--task-markers` - Strings indicating task description (default: "Task Description:" "Task:" "## Task")
- `--batch-size` - Parallel batch size (default: 100)
- `--max-rpm` - Rate limit (default: 500)
- `--max-trials` - Limit trials for testing (default: all)
- `--private` - Make repo private

**Custom Dataset Example:**
```bash
python augment_dataset.py \
    --dataset YOUR_ORG/your-dataset \
    --repo-name YOUR_USERNAME/your-hints \
    --task-markers "# Task" "## Objective" \
    --dataset-prefix "my-custom-hints"
```

## Cost & Performance

- **Speed**: 30-60 minutes for 10K episodes with Curator parallel processing
- **Cost**: ~$10-15 for 10K episodes with gpt-4o-mini
- **Output**: One augmented task per trial (e.g., 10K rows with 8 episodes each → ~1.25K tasks)

## Code Structure

Clean single-file pipeline (similar to `generate.py`):

```python
# Step 1: Group episodes by trial
trials = group_episodes_by_trial(dataset)
# {trial_name: [episode_0, ..., episode_7]}

# Step 2: Extract hints from each episode
trial_hints = extract_hints_from_trials(trials)
# {trial_name: [hints_0, ..., hints_7]}

# Step 3: Aggregate hints across episodes
augmented_instructions = create_augmented_instructions(trials, trial_hints)
# [(task + aggregated_hints, metadata), ...]

# Step 4: Generate tasks and upload
generate_tasks_from_questions(augmented_instructions)
upload_tasks_to_hf(dataset_dir, repo_name)
```

## Files

- `augment_dataset.py` - Complete generic pipeline (537 lines)
- `README.md` - This file

## Requirements

Your dataset must have these fields per row:
- `trial_name` - Groups episodes for the same task
- `episode` - Episode identifier (e.g., "episode-0001")
- `conversations` - List of `{"role": str, "content": str}` messages

Task description can be marked with any string (configurable via `--task-markers`).
