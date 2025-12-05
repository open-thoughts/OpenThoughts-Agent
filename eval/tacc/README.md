# Harbor Evaluation Setup for vLLM models on TACC

This directory contains scripts and configuration files for running Harbor evaluations with vLLM models on TACC systems.

## Overview

The Harbor evaluation system evaluates agent performance on coding tasks using vLLM-served models. The main workflow:

1. Loads required modules and sets up environment
2. Starts a vLLM inference server with your model
3. Downloads/locates the evaluation dataset from Hugging Face
4. Runs Harbor evaluation with concurrent sandboxed environments
5. Uploads results to the DCAgents leaderboard database

## Prerequisites

### Required Access

- TACC account with access to the DataComp allocation
- Access to shared conda environment at `/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5`

### Required Files

Before running evaluations, you need a `secret.env` file containing:

```bash
# Database credentials
export DB_HOST="your_db_host"
export DB_PORT="your_db_port"
export DB_NAME="your_db_name"
export DB_USER="your_db_user"
export DB_PASSWORD="your_db_password"

# Hugging Face token for dataset download and result upload
export HF_TOKEN="your_hf_token"

# Optional: Upload configuration
export UPLOAD_USERNAME="your_username"
export UPLOAD_MODE="skip_on_error"  # or "fail_on_error"
```

Place this file at: `/scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env`

## Environment Setup

The evaluation scripts automatically handle most environment setup, including:

### Modules Loaded

- `gcc/15.1.0` - GCC compiler toolchain
- `cuda/12.8` - CUDA toolkit for GPU support

### Environment Variables

Key environment variables configured by the script:

```bash
# VLLM configuration
VLLM_USE_V1=1
VLLM_CACHE_ROOT=/scratch/10000/eguha3/vllm_cache
VLLM_CONFIG_ROOT=/scratch/10000/eguha3/vllm_config

# Triton compiler cache
TRITON_DUMP_DIR=/scratch/10000/eguha3/triton_dump_dir
TRITON_OVERRIDE_DIR=/scratch/10000/eguha3/triton_override_dir
TRITON_CACHE_DIR=/scratch/10000/eguha3/triton_cache_dir

# FlashInfer cache
FLASHINFER_WORKSPACE_BASE=/scratch/08002/gsmyrnis/flashinfer_cache

# UV package manager cache
UV_CACHE_DIR=/scratch/10000/eguha3/uv_cache_dir

# Hugging Face cache
HF_CACHE_DIR=/scratch/08134/negin/dc-agent-shared/.hf_cache
HF_HUB_CACHE=$SCRATCH/hub

# Ray configuration
RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
```

### Conda Environment

The script activates the shared conda environment:
```bash
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5
```

## Running Evaluations

### Basic Usage

The main evaluation script is `tacc_eval_harbor.sbatch`. Submit it using:

```bash
sbatch tacc_eval_harbor.sbatch [MODEL_ID] [DATASET_REPO_ID]
```

### Parameters

1. **MODEL_ID** (optional, default: `mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1`)
   - Hugging Face model ID to evaluate
   - Must be accessible to your HF token

2. **DATASET_REPO_ID** (optional, default: `mlfoundations-dev/dev_set_71_tasks`)
   - Hugging Face dataset repository containing evaluation tasks
   - Must contain valid task directories with `instruction.md` files

### Examples

**Example 1: Use default model and dataset**
```bash
sbatch tacc_eval_harbor.sbatch
```

**Example 2: Evaluate a custom model**
```bash
sbatch tacc_eval_harbor.sbatch mlfoundations-dev/my-custom-model
```

**Example 3: Evaluate with custom model and dataset**
```bash
sbatch tacc_eval_harbor.sbatch mlfoundations-dev/my-model mlfoundations-dev/my-custom-tasks
```

### Job Configuration

The SLURM job is configured with:

- **Partition**: `gh` (GPU partition)
- **Time limit**: 24 hours
- **Resources**: 1 node, 72 CPUs
- **Account**: CCR24067
- **Output logs**: `experiments/logs/<job-name>_<job-id>.out`

### Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View live output:
```bash
tail -f experiments/logs/eval_<job_id>.out
```

Check VLLM server logs:
```bash
tail -f experiments/logs/vllm_<job_id>.log
```

## Configuration

### dcagent_eval_config.yaml

The evaluation behavior is controlled by `dcagent_eval_config.yaml`:

```yaml
jobs_dir: jobs                    # Directory for job outputs
n_attempts: 3                     # Number of attempts per task
timeout_multiplier: 1.0           # Timeout scaling factor

orchestrator:
  type: local                     # Use local orchestrator
  n_concurrent_trials: 4          # Concurrent tasks (overridden by CLI)
  quiet: false                    # Verbose output
  retry:
    max_retries: 10                # Max retries on failure
    exclude_exceptions:           # Don't retry these errors
      - AgentTimeoutError
      - VerifierTimeoutError
    wait_multiplier: 1.0
    min_wait_sec: 1.0
    max_wait_sec: 60.0

environment:
  type: daytona                   # Environment type (TACC does not support docker)
  force_build: true               # Force rebuild environments
  delete: false                   # Keep environments after run

agents:
  - name: terminus-2              # Agent configuration name
```

### Harbor CLI Arguments

The script passes these arguments to `harbor jobs start`:

```bash
harbor jobs start \
  -p "$DATASET_PATH" \               # Path to evaluation tasks
  --n-concurrent 128 \               # 128 concurrent evaluations
  --agent terminus-2 \               # Agent type
  --model "hosted_vllm/$MODEL" \     # Model endpoint. Make sure to append "hosted_vllm/" 
  --env "daytona" \                  # Environment type
  --agent-kwarg "api_base=http://localhost:8000/v1" \  # VLLM endpoint
  --agent-kwarg "key=fake_key" \     # API key (not validated for local)
  --k-attempts 3 \                   # Attempts per task
  --job-name "$RUN_TAG" \            # Unique job identifier
  --config "dcagent_eval_config.yaml"  # Config file
```

### Customizing Concurrency

To change the number of concurrent evaluations, modify line 109 in `tacc_eval_harbor.sbatch`:

```bash
--n-concurrent 128  \  # Change this value
```

**Note**: Higher concurrency requires more resources. Monitor CPU and memory usage.

## Understanding Outputs

### Directory Structure

After a successful run, outputs are organized as:

```
jobs/
└── <RUN_TAG>/
    ├── meta.env                # Job metadata (model, dataset, timestamp)
    ├── results.json            # Evaluation results summary
    ├── <task_id_1>/           # Individual task results
    │   ├── trajectory.json    # Agent trajectory
    │   ├── environment.log    # Environment logs
    │   └── ...
    ├── <task_id_2>/
    └── ...

experiments/logs/
├── eval_<job_id>.out          # Main job output
├── vllm_<job_id>.log          # VLLM server logs
└── upload_<job_id>.log        # Result upload logs
```

### Run Tag Format

The run tag follows this pattern:
```
<DATASET_NAME>_<MODEL_NAME>_<TIMESTAMP>
```

Example: `DCAgent_dev_set_71_tasks_mlfoundations-dev_my-model_20250112_143022`

### Metadata File (meta.env)

Contains original parameters for reproducibility:

```bash
MODEL=mlfoundations-dev/my-model
REPO_ID=mlfoundations-dev/dev_set_71_tasks
TIMESTAMP=20250112_143022
SLURM_JOB_ID=123456
```

### Results Upload

If the evaluation succeeds, results are automatically:

1. Uploaded to the DCAgents leaderboard database
2. Published to Hugging Face as a dataset
3. Registered with the benchmark system

The HF dataset will be at: `mlfoundations-dev/<sanitized_run_tag>`

## Troubleshooting

### Common Issues

#### 1. VLLM Server Fails to Start

**Symptoms**: Job exits with "VLLM server failed to start"

**Solutions**:
- Check VLLM logs: `experiments/logs/vllm_<job_id>.log`
- Verify model ID is correct and accessible
- Check GPU availability: `nvidia-smi`
- Ensure cache directories are writable
- Check if model fits in GPU memory (adjust `--gpu-memory-utilization`)

#### 2. Dataset Download Fails

**Symptoms**: "Failed to get dataset path"

**Solutions**:
- Verify dataset repository exists on Hugging Face
- Check HF_TOKEN is valid in secret.env
- Ensure dataset has valid task structure (directories with `instruction.md`)
- Check network connectivity to Hugging Face

#### 3. Out of Memory Errors

**Symptoms**: CUDA OOM or killed processes

**Solutions**:
- Reduce concurrency: `--n-concurrent 4` (line 109)
- Lower GPU memory utilization in VLLM (line 65):
  ```bash
  --gpu-memory-utilization 0.85 \  # Reduce from 0.95
  ```
- Use a smaller model
- Request more GPU resources

#### 4. Upload Fails

**Symptoms**: "Upload failed with exit code"

**Solutions**:
- Check database credentials in secret.env
- Verify HF_TOKEN has write permissions
- Check network connectivity to database
- Review upload logs: `experiments/logs/upload_<job_id>.log`
- Set `UPLOAD_MODE=skip_on_error` to continue despite upload issues

#### 5. Sandbox Environment Issues

**Symptoms**: Tasks fail with environment errors

**Solutions**:
- Verify sandbox CLI is installed: `harbor --help`
- Check Daytona environment is properly configured
- Review individual task logs in `jobs/<RUN_TAG>/<task_id>/`
- Try setting `force_build: true` in config (already default)

#### 6. Module Load Errors

**Symptoms**: Module not found or version conflicts

**Solutions**:
- Ensure you're on a TACC login/compute node
- Check module availability: `module avail gcc cuda`
- Update module versions in script if needed (lines 27-29)

#### 7. Conda Environment Issues

**Symptoms**: Command not found, import errors

**Solutions**:
- Verify conda environment exists:
  ```bash
  ls /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5
  ```
- Check conda is initialized properly
- Manually activate environment to test:
  ```bash
  source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
  conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5
  sb --help  # Test if sandbox is available
  ```

### Getting Help

If you encounter issues not covered here:

1. Check the main project README: `../../README.md`
2. Contact Negin Raoof for eval-related questions
3. Review TACC documentation at https://docs.tacc.utexas.edu/
