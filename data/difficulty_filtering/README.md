# R2E-Gym Difficulty Filtering Experiment

This experiment filters the R2E-Gym dataset into difficulty bands based on empirical pass rates from agent evaluations.

## Overview

We evaluated all 4,578 R2E-Gym tasks using the `terminus-2` agent with `gpt-5-nano` model, running 8 attempts per task (pass@8). Tasks were then filtered into difficulty bands based on their pass rates.

## Methodology

### Evaluation Setup
| Parameter | Value |
|-----------|-------|
| Dataset | R2E-Gym (4,578 tasks) |
| Agent | `terminus-2` |
| Model | `openai/gpt-5-nano` |
| Attempts per task | 8 (pass@8) |
| Concurrent trials | 512 |
| Environment | Daytona sandboxes |
| Total trials | 36,624 |
| Runtime | ~16 hours |

### Pass Rate Calculation
- Each task was attempted 8 times
- Timeouts and errors count as failures
- Pass rate = (successful attempts) / (total attempts)

### Difficulty Bands

| Band | Pass Rate Range | Description |
|------|-----------------|-------------|
| `impossible` | = 0% | Never solved - broken or too hard |
| `very_hard` | (0%, 25%] | Rarely solved |
| `hard` | (25%, 50%] | Occasionally solved |
| `medium` | (50%, 75%] | Usually solved |
| `easy` | (75%, 100%) | Almost always solved |
| `trivial` | = 100% | Always solved - no learning signal |

## Results

| Band | Tasks | Percentage | HuggingFace Dataset |
|------|-------|------------|---------------------|
| impossible | 2,667 | 58.3% | [DCAgent/exp-rdb-r2egym-impossible](https://huggingface.co/datasets/DCAgent/exp-rdb-r2egym-impossible) |
| very_hard | 1,210 | 26.4% | [DCAgent/exp-rdb-r2egym-very_hard](https://huggingface.co/datasets/DCAgent/exp-rdb-r2egym-very_hard) |
| hard | 514 | 11.2% | [DCAgent/exp-rdb-r2egym-hard](https://huggingface.co/datasets/DCAgent/exp-rdb-r2egym-hard) |
| medium | 162 | 3.5% | [DCAgent/exp-rdb-r2egym-medium](https://huggingface.co/datasets/DCAgent/exp-rdb-r2egym-medium) |
| easy | 20 | 0.4% | [DCAgent/exp-rdb-r2egym-easy](https://huggingface.co/datasets/DCAgent/exp-rdb-r2egym-easy) |
| trivial | 5 | 0.1% | [DCAgent/exp-rdb-r2egym-trivial](https://huggingface.co/datasets/DCAgent/exp-rdb-r2egym-trivial) |

### Key Observations

1. **Majority are impossible (58.3%)**: Most tasks were never solved by gpt-5-nano in 8 attempts
2. **Long tail distribution**: Only 4% of tasks have >50% pass rate
3. **Very few trivial tasks (0.1%)**: Only 5 tasks were solved 100% of the time

### Error Distribution
During evaluation, 53.5% of trials resulted in errors:
- `AgentTimeoutError`: Agents hitting 900s timeout limit
- `DaytonaError`: Infrastructure/sandbox issues
- `EnvironmentStartTimeoutError`: Environment startup failures
- `RewardFileNotFoundError`: Verifier output issues

## Usage

### Loading a Dataset
```python
from datasets import load_dataset

# Load the "hard" difficulty band
dataset = load_dataset("DCAgent/exp-rdb-r2egym-hard")
```

### Recommended Use Cases

| Use Case | Recommended Bands |
|----------|-------------------|
| RL Training (curriculum) | Start with `easy`/`medium`, progress to `hard`/`very_hard` |
| Benchmarking | `hard` or `very_hard` for meaningful signal |
| Stress testing | `impossible` for frontier model evaluation |
| Quick validation | `easy` or `trivial` for sanity checks |

## Reproduction

To reproduce this experiment:

```bash
cd /scratch/10000/eguha3/dc-agent
conda activate dataagent
source /scratch/10000/eguha3/old-dc-agent/secret.env
python -m data.difficulty_filtering.generate_all_bands
```

### Configuration
Edit `data/difficulty_filtering/generate_all_bands.py` to modify:
- `N_ATTEMPTS`: Number of attempts per task (default: 8)
- `N_CONCURRENT`: Parallel trial limit (default: 512)
- `MODEL`: Evaluation model (default: `openai/gpt-5-nano`)
- `AGENT`: Agent type (default: `terminus-2`)

## Files

```
data/difficulty_filtering/
├── __init__.py
├── README.md                # This file
├── utils.py                 # Shared utilities (pass@k computation, filtering)
├── generate_all_bands.py    # Main script - evaluates and uploads all bands
└── generate_hard_test.py    # Test script for small-scale validation
```

## Notes

- Pass rates are specific to the model/agent combination used
- Different models will produce different difficulty distributions
- The `impossible` band may contain both genuinely hard tasks and broken/malformed tasks
- Consider re-evaluating with stronger models for more nuanced filtering
- Timeouts (900s) count as failures in pass@k calculation
