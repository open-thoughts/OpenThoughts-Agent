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

export CONSOLIDATE_REPO_ID="{consolidate_repo_id}"
export CONSOLIDATE_BASE_REPO="{consolidate_base_repo}"
export CONSOLIDATE_WORKDIR="{consolidate_workdir}"
export CONSOLIDATE_COMMIT_MESSAGE="{consolidate_commit_message}"

python3 {python_script} \
  --target-repo "${{CONSOLIDATE_REPO_ID}}" \
  --base-repo "${{CONSOLIDATE_BASE_REPO}}" \
  --workdir "${{CONSOLIDATE_WORKDIR}}" \
  --commit-message "${{CONSOLIDATE_COMMIT_MESSAGE}}" \
  --project-root "{project_root}"
