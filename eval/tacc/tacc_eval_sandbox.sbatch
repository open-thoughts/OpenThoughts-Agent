#!/bin/bash
#SBATCH -p gh
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c637-091,c610-111
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=eval

# Create timestamp and safe names before setting job name/output
TIMESTAMP=$(date +'%Y%m%d_%H%M%S')

MODEL="${1:-mlfoundations-dev/claude_3_7_20250219_tbench_traces_sharegptv1}"
REPO_ID="${2:-mlfoundations-dev/clean-sandboxes-tasks-eval-set}"

# Strip slashes and special chars for file-safe names
SAFE_MODEL=$(echo "$MODEL" | tr '/:' '_')
SAFE_REPO=$(echo "$REPO_ID" | tr '/:' '_')

echo "Using model: $MODEL"
echo "Using repository: $REPO_ID"

module purge
module load gcc/15.1.0
module load cuda/12.8
module load tacc-apptainer

# Set up environment variables
export VLLM_USE_V1=1
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook
export RAY_ADDRESS=${RAY_ADDRESS:-}
export VLLM_CACHE_ROOT=/scratch/10000/eguha3/vllm_cache
export VLLM_CONFIG_ROOT=/scratch/10000/eguha3/vllm_config
export TRITON_DUMP_DIR=/scratch/10000/eguha3/triton_dump_dir
export TRITON_OVERRIDE_DIR=/scratch/10000/eguha3/triton_override_dir
export TRITON_CACHE_DIR=/scratch/10000/eguha3/triton_cache_dir
export FLASHINFER_WORKSPACE_BASE=/scratch/08002/gsmyrnis/flashinfer_cache
export UV_CACHE_DIR=/scratch/10000/eguha3/uv_cache_dir
export HYDRA_FULL_ERROR=1
export HF_CACHE_DIR=/scratch/08134/negin/dc-agent-shared/.hf_cache
export HF_HUB_CACHE=$SCRATCH/hub

# DB/API secrets etc.
source /scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/secret.env

# Toolchain fixes
ln -sf /home1/apps/gcc/15.1.0/lib64/libstdc++.so.6 /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v4/lib/libstdc++.so.6
export LD_LIBRARY_PATH=/home1/apps/gcc/15.1.0/lib64:/scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v4/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

# Conda env
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v4

# Verify sandbox installation
sb --help >/dev/null

# Start VLLM server
mkdir -p experiments/logs
vllm serve "$MODEL" \
  --host 0.0.0.0 --port 8000 \
  --served-model-name "$MODEL" \
  --device cuda --gpu-memory-utilization 0.95 \
  > "experiments/logs/vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

cleanup() {
    echo "Cleaning up..."
    kill $VLLM_PID 2>/dev/null || true
    conda deactivate || true
}
trap cleanup EXIT

# Wait for VLLM server to start with healthcheck
MAX_RETRIES=20
RETRY_INTERVAL=100
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "VLLM server is ready"
        break
    fi
    echo "Waiting for VLLM server to start (attempt $i/$MAX_RETRIES)..."
    sleep $RETRY_INTERVAL
    if [ $i -eq $MAX_RETRIES ]; then
        echo "VLLM server failed to start"
        exit 1
    fi
done

# Get the dataset path using the specified repo_id
echo "Downloading/locating dataset: $REPO_ID"
DATASET_PATH=$(python "./snapshot_download.py"  "$REPO_ID" | grep DATASET_PATH | tail -n 1 | cut -d'=' -f2)
if [ -z "${DATASET_PATH:-}" ]; then
    echo "Failed to get dataset path"
    exit 1
fi
echo "Using dataset path: $DATASET_PATH"

# Construct run dir (sb honors --job-name in jobs/<job-name>)
RUN_TAG="${SAFE_REPO}_${SAFE_MODEL}_${TIMESTAMP}"
RUN_DIR="jobs/${RUN_TAG}"

# Run sandbox evaluation
set +e
sb run \
  --dataset-path "$DATASET_PATH" \
  --n-concurrent 8  \
  --agent terminus-2 \
  --model "hosted_vllm/$MODEL" \
  --env "daytona" \
  --agent-kwarg "api_base=http://localhost:8000/v1" \
  --agent-kwarg "key=fake_key" \
  --n-attempts 3 \
  --job-name "$RUN_TAG" \
  --config "dcagent_eval_config.yaml"
SB_EXIT=$?
set -e

# Save originals for exact round-trip later
mkdir -p "$RUN_DIR"
{
  echo "MODEL=$MODEL"
  echo "REPO_ID=$REPO_ID"
  echo "TIMESTAMP=$TIMESTAMP"
  echo "SLURM_JOB_ID=$SLURM_JOB_ID"
} > "$RUN_DIR/meta.env"

# ---- Upload results to DB ----

# If eval failed, don't attempt upload
if [ ${SB_EXIT:-0} -ne 0 ]; then
  echo "sb run exited with non-zero status: ${SB_EXIT}. Skipping upload."
  exit ${SB_EXIT}
fi

# Ensure run dir exists; no fallback
if [ ! -d "$RUN_DIR" ]; then
  echo "Expected run directory not found: $RUN_DIR"
  exit 2
fi

# Point PYTHONPATH at your uploader package
export PYTHONPATH="/scratch/08134/negin/dc-agent-shared/dcagents-leaderboard:${PYTHONPATH:-}"

export RUN_DIR="$RUN_DIR"
export UPLOAD_USERNAME="${UPLOAD_USERNAME:-negin}"
export UPLOAD_MODE="${UPLOAD_MODE:-skip_on_error}"
export RUN_TAG="$RUN_TAG"
UPLOAD_LOG="experiments/logs/upload_${SLURM_JOB_ID}.log"
mkdir -p "$(dirname "$UPLOAD_LOG")"

echo "Uploading results from: $RUN_DIR" | tee -a "$UPLOAD_LOG"
echo "Using username=${UPLOAD_USERNAME}, mode=${UPLOAD_MODE}" | tee -a "$UPLOAD_LOG"

# Run the uploader (from dcagents-leaderboard)
python - <<'PY' 2>&1 | tee -a "$UPLOAD_LOG"
import os, sys
from unified_db.utils import upload_eval_results

import re
import hashlib


def sanitize_hf_repo_id(repo_id: str, max_length: int = 96) -> str:
    """
    Sanitizes a Hugging Face repo_id to comply with naming rules.
    Keeps org prefix (e.g. 'mlfoundations-dev/') and cleans up the rest.
    """
    # Separate org / repo part
    if "/" in repo_id:
        org, name = repo_id.split("/", 1)
    else:
        org, name = None, repo_id

    # Remove illegal chars
    name = re.sub(r"[^A-Za-z0-9._-]", "-", name)

    # Collapse multiple hyphens/dots
    name = re.sub(r"[-.]{2,}", "-", name)

    # Strip leading/trailing hyphen or dot
    name = name.strip("-.")

    # If too long, shorten using hash suffix
    if len(name) > (max_length - (len(org) + 1 if org else 0)):
        digest = hashlib.sha1(name.encode()).hexdigest()[:8]
        base = name[:keep].rstrip("-.")
        if not base:
            base = "r"
        name = f"{base}{digest}"  # no '-' before hash
        name = collapse(name).strip("-.")

    name = collapse(name).strip("-.")
    if name[0] in "-.":
        name = "r" + name[1:]
    if name[-1] in "-.":
        name = name[:-1] + "0"

    # Combine
    clean_repo_id = f"{org}/{name}" if org else name
    return clean_repo_id


run_dir   = os.environ["RUN_DIR"]
run_tag   = os.environ["RUN_TAG"]
username  = os.environ.get("UPLOAD_USERNAME", "negin")
error_mode= os.environ.get("UPLOAD_MODE", "skip_on_error")
hf_repo_id = sanitize_hf_repo_id(f"mlfoundations-dev/{run_tag}")

print(f"[uploader] upload_eval_results(path={run_dir!r}, username={username!r}, error_mode={error_mode!r}, hf_repo_id={hf_repo_id!r})")
upload_eval_results(run_dir, username=username, error_mode=error_mode, hf_token=hf_token, hf_repo_id=hf_repo_id, register_benchmark=True)
print("[uploader] done.")
PY
UPLOAD_EXIT=${PIPESTATUS[0]}

if [ $UPLOAD_EXIT -ne 0 ]; then
  echo "Upload failed with exit code: $UPLOAD_EXIT"
  exit $UPLOAD_EXIT
fi

echo "Eval and upload finished successfully."
