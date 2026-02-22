# Benchmark Instruction Improvement Analysis

## Overview

This document describes the process of improving benchmark task instructions to be more well-specified, and the methodology for testing the improvements.

## Problem Identified

Initial benchmark analysis showed ~77% of failures were due to **underspecified instructions**. The main issues were:

1. **Variable names instead of actual values**: Instructions showed `parse_workflow(wf)` instead of actual input like `parse_workflow('on: push\njobs:\n  build: {}')`
2. **Missing input format specification**: Domain-specific functions like `parse_commit`, `parse_errors` didn't show what the input format looked like
3. **Generic argument names**: Using `arg1, arg2` instead of meaningful names
4. **Domain-specific unclear functions**: Functions like "parse_workflow" need context, unlike self-explanatory names like "fibonacci"

## Solution Approach

### 1. Regeneration Script

Created `/scratch/10000/eguha3/dc-agent/data/regenerate_with_examples.py` that:

- Extracts actual test values from test code using regex patterns
- Substitutes variable names with their actual assigned values
- Handles multiple assertion patterns:
  - `assert func(var) == expected`
  - `assert func(var)['key'] == expected` (dict access)
  - `assert func(var).attr == expected` (attribute access)
- Generates instructions with real example values

### 2. Example of Improvement

**BEFORE** (ghactions - 0% pass rate):
```markdown
# Task: Parse Workflow

Implement `parse_workflow(wf)` that returns a `dict`.

**Example**:
parse_workflow(wf) == dict with ['on'] == 'push'
```

**AFTER** (ghactions - 84% pass rate):
```markdown
# Task: Parse Workflow

Implement `parse_workflow(wf)` that returns a `dict`.

**Example**:
parse_workflow('on: push
jobs:
  build: {}') => dict with ['on'] == 'push'
```

The key difference: The improved instruction shows the **actual YAML input format** instead of just a variable name.

## Testing Methodology

### Running Benchmarks

Harbor benchmarks are run using:

```bash
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6
source /scratch/10000/eguha3/old-dc-agent/secret.env

harbor jobs start \
    -p /scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset/<dataset> \
    --n-concurrent 10 \
    --agent terminus-2 \
    --model openai/gpt-5-nano-2025-08-07 \
    --env daytona \
    --n-attempts 1 \
    --job-name dataset_<dataset>_v2
```

### Configuration
- **Agent**: terminus-2
- **Model**: openai/gpt-5-nano-2025-08-07
- **Environment**: Daytona (sandboxed)
- **Concurrency**: 10 concurrent trials
- **Attempts**: 1 attempt per task

### Results Location
Results are stored in:
- `jobs/dataset_<name>_v2/result.json` - Summary with pass rates
- `jobs/dataset_<name>_v2/<task_id>/` - Individual task trajectories

### Parsing Results

```python
import json
with open('jobs/dataset_<name>_v2/result.json') as f:
    data = json.load(f)

stats = data['stats']
evals = stats['evals']
eval_key = list(evals.keys())[0]
metrics = evals[eval_key]['metrics']
pass_rate = metrics[0]['mean'] * 100  # Percentage
```

## Results Summary

### Before Improvement (Jan 24, 2026)

| Dataset | Pass Rate |
|---------|-----------|
| ghactions | 0% |
| travistorrent | 4% |
| e2egit | 8% |
| manybugs | 0% |
| bugswarm | 0% |
| bugsinpy_mf | 0% |

### After Improvement (Jan 26, 2026)

| Dataset | Before | After | Improvement |
|---------|--------|-------|-------------|
| ghactions | 0% | 84% | +84% |
| travistorrent | 4% | 80% | +76% |
| softwareheritage | 16% | 90% | +74% |
| bugswarm | 0% | 70% | +70% |
| manybugs | 0% | 67% | +67% |
| bugsinpy_mf | 0% | 62% | +62% |
| defects4j | 8% | 48% | +40% |
| e2egit | 8% | 33% | +25% |

**Average improvement: +62%**

### Underspecified Rate Analysis

After improvement, the underspecified instruction rate dropped dramatically:
- **Before**: ~77% of failures due to underspecified instructions
- **After**: ~8.4% of failures due to underspecified instructions

The remaining failures are now primarily due to:
1. Complex multi-step logic requirements
2. Edge cases not covered in examples
3. Domain-specific knowledge requirements

## Files Reference

| File | Purpose |
|------|---------|
| `/scratch/10000/eguha3/dc-agent/data/regenerate_with_examples.py` | Main regeneration script |
| `/scratch/10000/eguha3/dc-agent/data/benchmark_tasks_by_dataset/` | All 21 benchmark datasets |
| `/scratch/10000/eguha3/dc-agent/jobs/` | Benchmark results |
| `/scratch/10000/eguha3/dc-agent/eval/tacc/dcagent_eval_config.yaml` | Harbor config |

## Key Insights

1. **Self-explanatory function names work with minimal specification**: Datasets like `codenet` (96%) and `methods2test` (88%) perform well because function names like `fibonacci`, `string_length` are universally understood.

2. **Domain-specific functions need explicit input examples**: Datasets like `ghactions`, `travistorrent` require showing the actual input format (YAML, log format, etc.) in the instruction.

3. **The fix is simple but impactful**: Just replacing variable names with actual values in examples dramatically improves model understanding.
