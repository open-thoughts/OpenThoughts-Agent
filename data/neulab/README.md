# NeuLab Agent Data Collection

This directory contains generate scripts for all subsets of the `neulab/agent-data-collection` HuggingFace dataset.

## Dataset

The NeuLab agent-data-collection dataset contains 18 different subsets of agent interaction data in conversational format. Each subset contains conversations between humans and agents performing various tasks.

Dataset: [neulab/agent-data-collection](https://huggingface.co/datasets/neulab/agent-data-collection)

## Available Subsets

The following 18 subsets are available, each with its own generate script:

1. `agenttuning_os` - OS interaction tasks
2. `agenttuning_kg` - Knowledge graph tasks
3. `openhands` - OpenHands trajectories
4. `SWE-Gym_OpenHands-Sampled-Trajectories` - SWE-Gym trajectories
5. `nebius_SWE-agent-trajectories` - Nebius SWE agent trajectories
6. `mind2web` - Mind2Web web navigation tasks
7. `synatra` - Synatra dataset
8. `nnetnav-live` - NNetNav live navigation
9. `agenttuning_alfworld` - ALFWorld tasks
10. `nnetnav-wa` - NNetNav web agent
11. `codeactinstruct` - CodeAct instruction dataset
12. `SWE-smith_5kTrajectories` - SWE-smith trajectories
13. `orca_agentinstruct` - Orca agent instruct
14. `agenttuning_db` - Database tasks
15. `agenttuning_mind2web` - Agent tuning Mind2Web
16. `code_feedback` - Code feedback dataset
17. `agenttuning_webshop` - WebShop tasks
18. `go-browse-wa` - Go browse web agent

## Shared Utilities

### `common.py`

Contains shared utilities for all generate scripts:

- `extract_questions_from_subset()` - Extracts the first human message from each conversation as the instruction

All scripts use the shared `generate_tasks_from_questions()`, `upsample_tasks_directory()`, and `subsample_tasks_directory()` functions from `data.commons`.

## Usage

Each generate script is simple and takes no arguments. All scripts:
1. Extract questions (first human message) from the subset
2. Generate tasks using `generate_tasks_from_questions()`
3. **Upsample to 10k tasks** (if < 10k)
4. **Subsample to 10k tasks** (if > 10k)
5. Upload to HuggingFace at `DCAgent/neulab-<subset>-sandboxes`
6. Generate and upload traces

```bash
# Run the full pipeline for any subset
python3 -m data.neulab.generate_<subset_name>

# Example: Generate mind2web dataset
python3 -m data.neulab.generate_mind2web

# Example: Generate codeactinstruct dataset
python3 -m data.neulab.generate_codeactinstruct
```

## Examples

### Generate mind2web dataset (10k upsampled)

```bash
python3 -m data.neulab.generate_mind2web
```

### Generate codeactinstruct dataset (10k upsampled)

```bash
python3 -m data.neulab.generate_codeactinstruct
```

### Generate openhands dataset (10k upsampled)

```bash
python3 -m data.neulab.generate_openhands
```

## Task Format

Each generated task includes:

- `task.toml` - Task metadata and configuration
- `instruction.md` - The first human message as the task instruction
- `solution/solve.sh` - Basic solution script (generated via `generate_tasks_from_questions`)
- `environment/Dockerfile` - Ubuntu 24.04 environment with Python

## Process

1. **Extract Questions**: Load the subset and extract the first human message from each conversation
2. **Generate Tasks**: Use `data.commons.generate_tasks_from_questions()` to create task directories
3. **Upsample to 10k**: Upsample to 10,000 tasks using `data.commons.upsample_tasks_directory()` (if dataset has < 10k tasks)
4. **Subsample to 10k**: Subsample to 10,000 tasks using `data.commons.subsample_tasks_directory()` (if dataset has > 10k tasks)
5. **Upload to HF**: Upload to `DCAgent/neulab-<subset>-sandboxes`
6. **Generate Traces**: Run terminus-2 agent and upload traces to `DCAgent/neulab-<subset>-sandboxes-traces-terminus-2`

## Notes

- All scripts **always** produce exactly 10,000 tasks (or fewer if the original dataset is smaller)
- Upsample step repeats tasks if needed to reach 10k
- Subsample step randomly samples if dataset is larger than 10k
- The first human message from each conversation is extracted as the task instruction
- Scripts use the shared `generate_tasks_from_questions()`, `upsample_tasks_directory()`, and `subsample_tasks_directory()` functions from `data.commons`
- Some subsets may have loading issues (e.g., `SWE-smith_5kTrajectories`)
- No argparse - scripts are simple and require no configuration
