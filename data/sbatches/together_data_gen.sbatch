#!/bin/bash
#SBATCH -p gg
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=64
#SBATCH --account CCR24067
#SBATCH --output=${DCAGENT_DIR}/data/sbatches/logs/together_data_gen_%j.out
#SBATCH --job-name=together_data_gen

set -eo pipefail

# Check for required argument
if [ -z "$1" ]; then
    echo "Usage: sbatch together_data_gen.sbatch <hf_dataset_repo_id>"
    echo "Example: sbatch together_data_gen.sbatch DCAgent/staqc_10_tasks"
    exit 1
fi

INPUT_REPO_ID="$1"
# Extract dataset name by splitting on "/" (e.g., "DCAgent/staqc_10_tasks" -> "staqc_10_tasks")
DATASET_NAME="${INPUT_REPO_ID##*/}"
# Always upload to DCAgent org with glm_4.6_traces suffix
OUTPUT_REPO_ID="DCAgent/${DATASET_NAME}_glm_4.6_traces_together_again"

LOGS_DIR="${DCAGENT_DIR}/data/sbatches/logs"
HARBOR_CONFIG="${DCAGENT_DIR}/eval/tacc/dcagent_eval_config.yaml"

echo "============================================"
echo "Harbor + Together AI Data Generation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Input Dataset: $INPUT_REPO_ID"
echo "Output Traces: $OUTPUT_REPO_ID"
echo "============================================"

mkdir -p "$LOGS_DIR"

# Environment Setup
echo ""
echo "=== Step 1: Environment Setup ==="

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

export HYDRA_FULL_ERROR=1
# Source secrets (Together AI key, HF token, etc.)
source /scratch/10000/eguha3/old-dc-agent/secret.env

# Link libstdc++ for tacc_rl_v6


# Activate tacc_rl_v6 environment
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6

# Verify harbor is available
harbor --help >/dev/null
echo "Harbor is ready"

# === Step 2: Download Dataset ===
echo ""
echo "=== Step 2: Downloading Dataset ==="

# Add dc-agent/scripts/harbor to PYTHONPATH for tasks_parquet_converter.py
export PYTHONPATH="${DCAGENT_DIR}/scripts/harbor:$PYTHONPATH"

DATASET_PATH=$(LD_LIBRARY_PATH="" /scratch/10000/eguha3/tacc_rl_v6/bin/python3 -c "
from tasks_parquet_converter import from_hf_dataset
path = from_hf_dataset('$INPUT_REPO_ID')
print(path)
")

if [ -z "$DATASET_PATH" ] || [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Failed to download dataset or path does not exist: $DATASET_PATH"
    exit 1
fi
echo "Dataset downloaded to: $DATASET_PATH"

# === Step 3: Run Harbor with Together AI ===
echo ""
echo "=== Step 3: Running Harbor with Together AI ==="

TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
RUN_TAG="${DATASET_NAME}_glm46_${TIMESTAMP}"
HARBOR_LOG="${LOGS_DIR}/harbor_${DATASET_NAME}_${SLURM_JOB_ID}.log"

echo "  Dataset: $DATASET_PATH"
echo "  Model: together_ai/zai-org/GLM-4.6"
echo "  Run tag: $RUN_TAG"
echo "  Log: $HARBOR_LOG"

harbor jobs start \
    -p "$DATASET_PATH" \
    --n-concurrent 256 \
    --agent terminus-2 \
    --model "together_ai/zai-org/GLM-4.6" \
    --env "daytona" \
    --n-attempts 1 \
    --max-retries 0 \
    --job-name "$RUN_TAG" \
    --config "$HARBOR_CONFIG" \
    --disable-verification \
    --agent-kwarg 'trajectory_config={"raw_content":true,"linear_history":true}' \
    --ak 'extra_body={"chat_template_kwargs": {"enable_thinking": true}}' \
    2>&1 | tee "$HARBOR_LOG"

HARBOR_EXIT=$?

# === Step 4: Export and Upload Expert Traces ===
echo ""
echo "=== Step 4: Exporting and Uploading Expert Traces ==="

# Parse job directory from harbor log (format: "Results written to jobs/<job_name>/result.json")
JOB_DIR=$(grep "Results written to" "$HARBOR_LOG" | sed 's/.*Results written to \(jobs\/[^/]*\).*/\1/' | head -1)
if [ -z "$JOB_DIR" ]; then
    echo "ERROR: Could not find job directory in harbor log"
    exit 1
fi
echo "Job directory: $JOB_DIR"

export JOB_DIR
export HF_REPO_ID="$OUTPUT_REPO_ID"

# Source HF token
source /scratch/10000/eguha3/old-dc-agent/secret.env

echo "Uploading traces to: $HF_REPO_ID"

# Clear LD_LIBRARY_PATH to avoid torch CUDA init issues on GH200
LD_LIBRARY_PATH="" TRANSFORMERS_NO_TORCH=1 /scratch/10000/eguha3/tacc_rl_v6/bin/python3 -c "
import sys
import os
from pathlib import Path

from harbor.utils.traces_utils import export_traces

# Get job directory from environment (parsed from harbor log)
job_dir = Path(os.environ.get('JOB_DIR', ''))
if not job_dir.exists():
    print(f'ERROR: Job directory does not exist: {job_dir}')
    sys.exit(1)

repo_id = os.environ.get('HF_REPO_ID', '')
print(f'Exporting traces from {job_dir} to {repo_id}')

export_traces(
    root=job_dir,
    recursive=True,
    episodes='last',
    repo_id=repo_id,
    push=True,
    verbose=True,
)
"

EXPORT_EXIT=$?
if [ $EXPORT_EXIT -eq 0 ]; then
    echo "Expert traces uploaded successfully to: $HF_REPO_ID"
else
    echo "WARNING: Failed to upload expert traces (exit code: $EXPORT_EXIT)"
fi

conda deactivate || true

echo ""
echo "============================================"
if [ $HARBOR_EXIT -eq 0 ]; then
    echo "Experiment COMPLETED"
    echo "  Input: $INPUT_REPO_ID"
    echo "  Output: $OUTPUT_REPO_ID"
else
    echo "Experiment FAILED (exit code: $HARBOR_EXIT)"
fi
echo "============================================"

exit $HARBOR_EXIT
