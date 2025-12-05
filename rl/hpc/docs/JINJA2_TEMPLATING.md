# Jinja2 Templating for TACC Sbatch

This document describes the Jinja2 templating feature added to the `launch.py` script for TACC sbatch generation.

## Overview

The `launch.py` script now supports Jinja2 templating specifically for TACC clusters. When a TACC cluster is detected and Jinja2 is available, the script will automatically use Jinja2 templating instead of the original simple string replacement method.

## Features

- **Automatic Detection**: The script automatically detects TACC clusters and uses Jinja2 templating when available
- **Fallback Support**: If Jinja2 is not available, the script falls back to the original templating method
- **Template File Support**: The script looks for `.j2` template files (e.g., `tacc_train.j2`) and falls back to the original `.sbatch` files if not found

## Template Files

### Jinja2 Template
- `tacc_train.j2` - Jinja2 template using `{{ variable }}` syntax

## Usage

The Jinja2 templating is automatically used when:
1. The cluster name contains "tacc"
2. Jinja2 is available in the Python environment
3. A `.j2` template file exists (optional, falls back to `.sbatch` if not found)

## Template Variables

The following variables are available in TACC Jinja2 templates:

### SLURM Parameters
- `partition` - SLURM partition
- `time_limit` - Job time limit
- `num_nodes` - Number of nodes
- `cpus_per_node` - CPUs per node
- `node_exclusion_list` - Nodes to exclude
- `account` - SLURM account
- `experiments_dir` - Experiments directory
- `job_name` - Job name

### TACC-Specific Variables
- `vllm_cache_root` - VLLM cache directory
- `vllm_config_root` - VLLM config directory
- `triton_dump_dir` - Triton dump directory
- `triton_override_dir` - Triton override directory
- `triton_cache_dir` - Triton cache directory
- `flashinfer_workspace_base` - FlashInfer workspace base
- `uv_cache_dir` - UV cache directory
- `secret_env_path` - Secret environment file path
- `conda_activate_path` - Conda activation script path
- `conda_env_path` - Conda environment path
- `skyrl_home` - SkyRL home directory
- `dc_agent_path` - DC Agent path

## Jinja2 Features

The templates support standard Jinja2 features:

### Variable Substitution
```jinja2
#SBATCH --job-name={{ job_name }}
export VLLM_CACHE_ROOT={{ vllm_cache_root }}
```

### Conditional Blocks
```jinja2
{% if vllm_cache_root %}
export VLLM_CACHE_ROOT={{ vllm_cache_root }}
{% endif %}
```

### Comments
```jinja2
{# This is a Jinja2 comment #}
```

## Installation

To use Jinja2 templating, install the jinja2 package:

```bash
pip install jinja2
```

## Testing

The implementation has been tested with:
- Basic Jinja2 template rendering
- TACC-specific template variables
- Fallback to original method when Jinja2 is not available
- Template file detection (`.j2` vs `.sbatch`)

## Benefits

1. **More Powerful Templating**: Jinja2 provides advanced features like conditionals, loops, and filters
2. **Better Error Handling**: Jinja2 provides clear error messages for template issues
3. **Extensibility**: Easy to add new template features and logic
4. **Backward Compatibility**: Falls back to original method when Jinja2 is not available
