#!/bin/bash
#SBATCH --job-name=dclm_test
#SBATCH --output=/data/horse/ws/rehe951g-dcagent/dclm-terminal-like/logs/test_%j.out
#SBATCH --error=/data/horse/ws/rehe951g-dcagent/dclm-terminal-like/logs/test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=104
#SBATCH --mem-per-cpu=4G
#SBATCH --time=05:00:00
#SBATCH --account=p_agents_finetuning

# Create logs directory if it doesn't exist
mkdir -p /data/horse/ws/rehe951g-dcagent/dclm-terminal-like/logs

# Print job information
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Load required modules (adjust as needed)
# module load Python/3.11.3-GCCcore-12.3.0  # Uncomment and adjust version if needed

# Navigate to the script directory
cd $SLURM_SUBMIT_DIR

# Create and activate virtual environment (if needed)
# python3 -m venv venv
# source venv/bin/activate

# Install requirements
pip install -r requirements.txt --user

# Run the test script with 32 workers, threshold 4.0, 200 samples
echo "=========================================="
echo "Starting run_all_shards.sh test..."
echo "Workers: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# test:
#./run_pipeline.sh 10 100 8 $SLURM_CPUS_PER_TASK
# process all shards:
./run_all_shards.sh $SLURM_CPUS_PER_TASK 8.0 200

# Print completion information
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
