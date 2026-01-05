#!/bin/bash
{partition_directive}
{account_directive}
#SBATCH --time={time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
{gpu_directive}
{mem_directive}
#SBATCH --job-name={job_name}
#SBATCH --output={output_path}

module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate $SCRATCH/miniconda3/envs/dcagent

source $DCFT/hpc/dotenv/tacc.env
export NCCL_TIMEOUT=1800  # 30 minutes
export NCCL_IB_TIMEOUT=23  # InfiniBand timeout

export CONSOLIDATE_INPUT="{consolidate_input}"
export CONSOLIDATE_INPUT_KIND="{consolidate_input_kind}"
export CONSOLIDATE_OUTPUT_REPO="{consolidate_output_repo}"
export CONSOLIDATE_BASE_REPO="{consolidate_base_repo}"
export CONSOLIDATE_WORKDIR="{consolidate_workdir}"
export CONSOLIDATE_COMMIT_MESSAGE="{consolidate_commit_message}"
export CONSOLIDATE_PUSH_TO_HUB="{consolidate_push_to_hub}"

python3 {python_script} \
  --input "${{CONSOLIDATE_INPUT}}" \
  --input-kind "${{CONSOLIDATE_INPUT_KIND}}" \
  --output-repo "${{CONSOLIDATE_OUTPUT_REPO}}" \
  --base-repo "${{CONSOLIDATE_BASE_REPO}}" \
  --workdir "${{CONSOLIDATE_WORKDIR}}" \
  --commit-message "${{CONSOLIDATE_COMMIT_MESSAGE}}" \
  --push-to-hub "${{CONSOLIDATE_PUSH_TO_HUB}}" \
  --project-root "{project_root}"
