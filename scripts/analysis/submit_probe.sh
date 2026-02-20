#!/bin/bash
# Submit probe_model_thinking.py as a SLURM job.
#
# Usage:
#   # Basic (uses defaults for model, dataset, cluster settings)
#   ./scripts/analysis/submit_probe.sh
#
#   # Custom model
#   ./scripts/analysis/submit_probe.sh --model laion/my-custom-model
#
#   # Custom everything
#   ./scripts/analysis/submit_probe.sh \
#       --model laion/my-model \
#       --partition gpu-h100 \
#       --time 01:00:00 \
#       --num-prompts 10 \
#       --max-new-tokens 8192
#
# Environment:
#   Expects hpc/dotenv/<cluster>.env to be sourced, or at minimum:
#     - A working Python environment with transformers, datasets, torch
#     - HF_TOKEN set for gated models

set -euo pipefail

# ---- Defaults ----
MODEL="laion/GLM-4_7-swesmith-sandboxes-with_tests-oracle_verified_120s-maxeps-131k"
DATASET="DCAgent2/DCAgent_dev_set_v2_laion_exp_tas_timeout_multiplier_4_0_traces_20260211_064438"
NUM_PROMPTS=5
MAX_NEW_TOKENS=4096
DTYPE="bfloat16"
PARTITION="${PARTITION:-}"
TIME="00:30:00"
JOB_NAME="probe_thinking"
OUTPUT_DIR="${CHECKPOINTS_DIR:-./probe_results}"
EXTRA_SBATCH_ARGS=""

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)         MODEL="$2";          shift 2 ;;
        --dataset)       DATASET="$2";        shift 2 ;;
        --num-prompts)   NUM_PROMPTS="$2";    shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --dtype)         DTYPE="$2";          shift 2 ;;
        --partition)     PARTITION="$2";       shift 2 ;;
        --time)          TIME="$2";           shift 2 ;;
        --job-name)      JOB_NAME="$2";       shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2";     shift 2 ;;
        --sbatch-args)   EXTRA_SBATCH_ARGS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---- Resolve paths ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/${JOB_NAME}_$(date +%Y%m%d_%H%M%S).json"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# ---- Build SBATCH directives ----
PARTITION_LINE=""
if [[ -n "$PARTITION" ]]; then
    PARTITION_LINE="#SBATCH --partition=$PARTITION"
fi

# ---- Generate and submit ----
SBATCH_SCRIPT=$(mktemp /tmp/probe_XXXXXX.sbatch)

cat > "$SBATCH_SCRIPT" << SBATCH_EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --output=$LOG_DIR/%x_%j.out
$PARTITION_LINE
$EXTRA_SBATCH_ARGS

set -euo pipefail

echo "=== Probe Model Thinking ==="
echo "Model:    $MODEL"
echo "Dataset:  $DATASET"
echo "Prompts:  $NUM_PROMPTS"
echo "MaxTok:   $MAX_NEW_TOKENS"
echo "Output:   $OUTPUT_FILE"
echo "Node:     \$(hostname)"
echo "GPU:      \$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

cd "$REPO_ROOT"

python -m scripts.analysis.probe_model_thinking \\
    --model "$MODEL" \\
    --dataset "$DATASET" \\
    --num-prompts $NUM_PROMPTS \\
    --max-new-tokens $MAX_NEW_TOKENS \\
    --dtype "$DTYPE" \\
    --output "$OUTPUT_FILE"

echo ""
echo "=== Done ==="
SBATCH_EOF

echo "Submitting probe job:"
echo "  Model:   $MODEL"
echo "  Dataset: $DATASET"
echo "  Output:  $OUTPUT_FILE"
echo "  Script:  $SBATCH_SCRIPT"
echo ""

sbatch "$SBATCH_SCRIPT"
