# OT-Agent HPC Training System

(NOTE: this README, along with the docs in the `docs/` folder can be outdated. Use with caution.)

This directory contains a unified command-line interface for launching dc-agent training jobs on multiple HPC clusters, similar to the DCFT HPC system.
## OT-Agent Directories
Locations for codebases:

JSC: ```/p/project/laionize/marianna/terminal_bench/dc-agent```

TACC: ```/scratch/08134/negin/dc-agent-shared/dc-agent```
## Quick Start
All you need to begin is 
```bash
cd rl
source hpc/setup.sh
```
### Main command
```bash
python -m rl.hpc.launch     --job_name gsm8k_test     --time_limit 01:00:00     --num_nodes 4     --train_data [mlfoundations-dev/sandboxes-tasks-hello-world]  --val_data [mlfoundations-dev/sandboxes-tasks-hello-world]     --model_path Qwen/Qwen2.5-7B-Instruct --partition gh-dev     --epochs 1     --learning_rate 1.0e-6 [additional arguments] --final_model_name test_run
```

> [!TIP]  
> Add `--dry_run` to see the outputs (sbatch, yaml, etc.) without launching the job
>
> Run this from the repository root so `python -m rl.hpc.launch` resolves to the RL launcher even if another `hpc` package is on your `PYTHONPATH`.

## Examples
The only mandatory fields are train_data, val_data, job_name, and final_model_name
```bash
python -m rl.hpc.launch     --job_name gsm8k_test     --time_limit 01:00:00     --num_nodes 4     --train_data [mlfoundations-dev/sandboxes-tasks-hello-world]  --val_data [mlfoundations-dev/sandboxes-tasks-hello-world]     --model_path Qwen/Qwen2.5-7B-Instruct --partition gh-dev     --epochs 1     --learning_rate 1.0e-6 [additional arguments] --final_model_name test_run
```


## Setup Instructions

### Environment Setup

Add these to your `~/.bashrc`:

```bash
source <LOCATION OF DCAGENT>/rl/hpc/setup.sh
```

### Cluster-Specific Setup

#### Quick Setup (Recommended)
```bash
cd rl
source hpc/setup.sh
```


**Note**: JSC clusters use a different training approach with SSH tunnels and Ray setup, similar to the original `jsc_train_daytona.sh` script. The system automatically pre-downloads datasets and models on the login node before submitting jobs to compute nodes (which have no internet access).

## Supported Clusters

### TACC Clusters
- **Vista**: `vista.tacc.utexas.edu` - GH200 96GB GPUs
- **Lonestar**: `ls6.tacc.utexas.edu` - A100 40GB GPUs

### JSC Clusters
- **Jureca**: `jureca` - H100 94GB GPUs
- **Jupiter**: `jupiter.internal` - GH200 96GB GPUs  
- **Juwels**: `juwels` - A100 40GB GPUs

## Command Line Arguments

### Launch Arguments
- `--job_name`: Job name (auto-generated if not provided)
- `--time_limit`: Time limit for the job (e.g., "24:00:00")
- `--num_nodes`: Number of nodes to use
- `--experiments_dir`: Output directory for experiments
- `--dry_run`: Preview job without submitting

### SkyRL Training Arguments
- `--train_data`: List of training datasets
- `--val_data`: List of validation datasets
- `--model_path`: Path to the model
- `--algorithm`: RL algorithm (default: "grpo")
- `--strategy`: Training strategy (default: "fsdp2")
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--backend`: Inference backend (default: "vllm")
- `--agent_name`: Agent name (default: "terminus")
- `--n_samples_per_prompt`: Number of samples per prompt
- `--max_prompt_length`: Maximum prompt length
- `--max_generate_length`: Maximum generation length
- `--gpu_memory_utilization`: GPU memory utilization

### Placement Arguments
- `--policy_num_nodes`: Number of nodes for policy
- `--ref_num_nodes`: Number of nodes for reference model
- `--reward_num_nodes`: Number of nodes for reward model
- `--critic_num_nodes`: Number of nodes for critic
- `--policy_num_gpus_per_node`: Number of GPUs per node for policy
- `--ref_num_gpus_per_node`: Number of GPUs per node for reference model
- `--reward_num_gpus_per_node`: Number of GPUs per node for reward model
- `--critic_num_gpus_per_node`: Number of GPUs per node for critic

## Monitoring Your Jobs

### Job Status
```bash
# Show your jobs
sqme

# Show job status with logs
status

# Show failed jobs
sfail

# Show completed jobs
scompleted

# Show cancelled jobs
scancelled
```

### Log Files
SLURM logs are written to `{experiments_dir}/logs/{job_name}_{job_id}.out`

```bash
# View latest log
tail -f $DC_AGENT_TRAIN/experiments/logs/latest_job.out

# View specific job log
tail -f $DC_AGENT_TRAIN/experiments/logs/gsm8k_test_12345.out
```

### Checkpoints
Checkpoints are saved to `{checkpoints_dir}/{job_name}/`

## Helper Scripts

### Monitoring Scripts
- `status [lines]`: Show job status and recent logs
- `sfail [hours]`: Show failed jobs in last N hours
- `scompleted [hours]`: Show completed jobs in last N hours
- `scancelled [hours]`: Show cancelled jobs in last N hours
- `scancelall`: Cancel all your jobs
- `rmlogs [threshold]`: Remove old log files

### Utility Scripts
- `sinf`: Show cluster information
- `sqme`: Show your queued jobs
- `sqteam`: Show team jobs
- `sqthem <user>`: Show specific user's jobs

## Configuration Files

### Environment Files
- `dotenv/tacc.env`: TACC cluster environment variables
- `dotenv/jsc.env`: JSC cluster environment variables

### SBatch Jinja Templates
- `sbatch/tacc_train.j2`: TACC cluster job template
- `sbatch/jsc_train.j2`: TO BE ADDED

## Directory Structure

```
hpc/
├── README.md                 # This file
├── hpc.py                   # Cluster configurations
├── arguments.py             # Command line argument definitions
├── launch.py                # Main job submission logic
├── sbatch/                  # SBatch job templates
│   ├── tacc_train.j2
│   └── jsc_train.j2         # TO BE ADDED
├── dotenv/                  # Environment variable files
│   ├── tacc.env
│   └── jsc.env
└── scripts/                 # Helper scripts
    ├── common.sh
    ├── status.sh
    ├── sfail.sh
    ├── scompleted.sh
    ├── scancelled.sh
    ├── scancelall.sh
    └── rmlogs.sh
```

## Examples

```bash
python -m rl.hpc.launch \
    --job_name custom_experiment \
    --time_limit 48:00:00 \
    --num_nodes 8 \
    --train_data mlfoundations-dev/sandboxes-tasks \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --epochs 3 \
    --learning_rate 5.0e-7 \
    --n_samples_per_prompt 8 \
    --max_prompt_length 32000
```

## Troubleshooting

### Common Issues

1. **Job fails immediately**: Check logs for environment setup issues
2. **Out of memory**: Reduce `gpu_memory_utilization` or `max_prompt_length`
3. **Job times out**: Increase `time_limit` or reduce `epochs`
4. **No GPUs available**: Check cluster status with `sinf`

### Debug Mode
```bash
python -m rl.hpc.launch --dry_run --job_name debug_test
```

### Log Analysis
```bash
# Check for errors
grep -i error $DC_AGENT_TRAIN/experiments/logs/latest.out

# Check for warnings
grep -i warning $DC_AGENT_TRAIN/experiments/logs/latest.out

# Monitor real-time
tail -f $DC_AGENT_TRAIN/experiments/logs/latest.out
```

## Advanced Usage

### JSC Pre-Download Feature
For JSC clusters, the system automatically pre-downloads datasets and models on the login node before submitting jobs to compute nodes (which have no internet access).

The pre-download happens automatically and uses the HuggingFace cache directory specified in the environment variables.

### Custom SBatch Templates
You can modify the SBatch templates in `sbatch/` directory to customize job behavior.

### Custom Environment Variables
Modify the environment files in `dotenv/` directory to add cluster-specific variables.

## Support

For issues or questions:
1. Check the logs first: `status 50`
2. Review this README
3. Check cluster-specific documentation
4. Contact the development team
