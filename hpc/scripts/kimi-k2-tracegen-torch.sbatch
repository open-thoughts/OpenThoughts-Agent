#!/bin/bash
#SBATCH --job-name=kimi-k2-tracegen
#SBATCH --account=torch_pr_40_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint="h200"
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --mem=1536G
#SBATCH --time=47:59:00
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --mail-type=END,TIME_LIMIT,FAIL
#SBATCH --mail-user=bf996@nyu.edu

set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: sbatch kimi-k2-tracegen-torch.sbatch [options]

Options (override defaults used by this job):
  --experiments-dir PATH         Experiments directory (default: $SCRATCH/OpenThoughts-Agent/experiments/kimi-k2-freelancer)
  --datagen-config PATH          Datagen YAML (default: hpc/datagen_yaml/kimi_k2_vllm_serve_tacc_ray.yaml)
  --trace-script PATH            Trace generation script (default: data/freelancer/generate_abstract.py)
  --trace-target-repo ID         Hugging Face repo for traces (default: DCAgent2/kimi-k2-freelancer-traces)
  --trace-harbor-config PATH     Harbor YAML template (default: hpc/harbor_yaml/trace_adaptive.yaml)
  --trace-model NAME             Trace model identifier (default: moonshotai/Kimi-K2-Thinking)
  --trace-engine NAME            Trace engine (default: vllm_local)
  --trace-backend NAME           Trace backend (default: ray)
  --tasks-repo ID                HF dataset repo to download (default: mlfoundations-dev/freelancer-projects-sandboxes)
  --tasks-revision REV           Optional dataset revision/commit
  --parquet-name FILE            Specific parquet filename inside the snapshot
  --gpus-per-node N              GPUs per node to allocate to vLLM (default: 8)
  --time-limit HH:MM:SS          Logical time limit recorded in metadata (default: 167:00:00)
  --trace-agent-name NAME        Override Harbor agent name
  --trace-agent-kwargs JSON      JSON object of extra agent kwargs
  --trace-env NAME               Override Harbor environment type
  --trace-agent-timeout-sec SEC  Override Harbor agent timeout
  --trace-n-concurrent N         Override concurrency for the trace orchestrator
  --trace-episodes MODE          Trace episodes parameter (default: last)
  --trace-export-filter FILTER   Trace export filter (default: none)
  --trace-dataset-type TYPE      Trace dataset type label (default: SFT)
  --trace-chunk-size N           Maximum tasks per chunk when splitting traces
  --overwrite-tasks              Re-extract tasks even if directory already exists
  -h, --help                     Show this help message
USAGE
}

# Defaults
EXPERIMENTS_DIR_DEFAULT="${SCRATCH:-$HOME}/OpenThoughts-Agent/experiments/kimi-k2-r2egym_sandboxes-maxeps-32k"
DATAGEN_CONFIG_DEFAULT="hpc/datagen_yaml/kimi_k2_vllm_serve_torch_h200.yaml"
TRACE_SCRIPT_DEFAULT="data/freelancer/generate_abstract.py"
TRACE_TARGET_REPO_DEFAULT="DCAgent2/kimi-k2-r2egym_sandboxes-maxeps-32k"
TRACE_HARBOR_CONFIG_DEFAULT="hpc/harbor_yaml/trace_16concurrency_ctx32k.yaml"
TRACE_MODEL_DEFAULT="moonshotai/Kimi-K2-Thinking"
TRACE_ENGINE_DEFAULT="vllm_local"
TRACE_BACKEND_DEFAULT="ray"
TASKS_REPO_DEFAULT="DCAgent2/r2egym_sandboxes"
TASKS_REVISION_DEFAULT=""
PARQUET_NAME_DEFAULT=""
GPUS_PER_NODE_DEFAULT=8
TIME_LIMIT_DEFAULT="47:59:00"
TRACE_AGENT_NAME_DEFAULT=""
TRACE_AGENT_KWARGS_DEFAULT=""
TRACE_ENV_DEFAULT=""
TRACE_AGENT_TIMEOUT_SEC_DEFAULT=""
TRACE_N_CONCURRENT_DEFAULT=""
TRACE_EPISODES_DEFAULT="last"
TRACE_EXPORT_FILTER_DEFAULT="none"
TRACE_DATASET_TYPE_DEFAULT="SFT"
TRACE_CHUNK_SIZE_DEFAULT=""
OVERWRITE_TASKS_DEFAULT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiments-dir) EXPERIMENTS_DIR_DEFAULT="$2"; shift 2;;
    --datagen-config) DATAGEN_CONFIG_DEFAULT="$2"; shift 2;;
    --trace-script) TRACE_SCRIPT_DEFAULT="$2"; shift 2;;
    --trace-target-repo) TRACE_TARGET_REPO_DEFAULT="$2"; shift 2;;
    --trace-harbor-config) TRACE_HARBOR_CONFIG_DEFAULT="$2"; shift 2;;
    --trace-model) TRACE_MODEL_DEFAULT="$2"; shift 2;;
    --trace-engine) TRACE_ENGINE_DEFAULT="$2"; shift 2;;
    --trace-backend) TRACE_BACKEND_DEFAULT="$2"; shift 2;;
    --tasks-repo) TASKS_REPO_DEFAULT="$2"; shift 2;;
    --tasks-revision) TASKS_REVISION_DEFAULT="$2"; shift 2;;
    --parquet-name) PARQUET_NAME_DEFAULT="$2"; shift 2;;
    --gpus-per-node) GPUS_PER_NODE_DEFAULT="$2"; shift 2;;
    --time-limit) TIME_LIMIT_DEFAULT="$2"; shift 2;;
    --trace-agent-name) TRACE_AGENT_NAME_DEFAULT="$2"; shift 2;;
    --trace-agent-kwargs) TRACE_AGENT_KWARGS_DEFAULT="$2"; shift 2;;
    --trace-env) TRACE_ENV_DEFAULT="$2"; shift 2;;
    --trace-agent-timeout-sec) TRACE_AGENT_TIMEOUT_SEC_DEFAULT="$2"; shift 2;;
    --trace-n-concurrent) TRACE_N_CONCURRENT_DEFAULT="$2"; shift 2;;
    --trace-episodes) TRACE_EPISODES_DEFAULT="$2"; shift 2;;
    --trace-export-filter) TRACE_EXPORT_FILTER_DEFAULT="$2"; shift 2;;
    --trace-dataset-type) TRACE_DATASET_TYPE_DEFAULT="$2"; shift 2;;
    --trace-chunk-size) TRACE_CHUNK_SIZE_DEFAULT="$2"; shift 2;;
    --overwrite-tasks) OVERWRITE_TASKS_DEFAULT=1; shift;;
    -h|--help) usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

EXPERIMENTS_DIR="${EXPERIMENTS_DIR_DEFAULT}"
DATAGEN_CONFIG="${DATAGEN_CONFIG_DEFAULT}"
TRACE_SCRIPT="${TRACE_SCRIPT_DEFAULT}"
TRACE_TARGET_REPO="${TRACE_TARGET_REPO_DEFAULT}"
TRACE_HARBOR_CONFIG="${TRACE_HARBOR_CONFIG_DEFAULT}"
TRACE_MODEL="${TRACE_MODEL_DEFAULT}"
TRACE_ENGINE="${TRACE_ENGINE_DEFAULT}"
TRACE_BACKEND="${TRACE_BACKEND_DEFAULT}"
TASKS_REPO="${TASKS_REPO_DEFAULT}"
TASKS_REVISION="${TASKS_REVISION_DEFAULT}"
PARQUET_NAME="${PARQUET_NAME_DEFAULT}"
GPUS_PER_NODE="${GPUS_PER_NODE_DEFAULT}"
TIME_LIMIT="${TIME_LIMIT_DEFAULT}"
TRACE_AGENT_NAME="${TRACE_AGENT_NAME_DEFAULT}"
TRACE_AGENT_KWARGS="${TRACE_AGENT_KWARGS_DEFAULT}"
TRACE_ENV="${TRACE_ENV_DEFAULT}"
TRACE_AGENT_TIMEOUT_SEC="${TRACE_AGENT_TIMEOUT_SEC_DEFAULT}"
TRACE_N_CONCURRENT="${TRACE_N_CONCURRENT_DEFAULT}"
TRACE_EPISODES="${TRACE_EPISODES_DEFAULT}"
TRACE_EXPORT_FILTER="${TRACE_EXPORT_FILTER_DEFAULT}"
TRACE_DATASET_TYPE="${TRACE_DATASET_TYPE_DEFAULT}"
TRACE_CHUNK_SIZE="${TRACE_CHUNK_SIZE_DEFAULT}"
OVERWRITE_TASKS="${OVERWRITE_TASKS_DEFAULT}"

# Export variables for direct bash (no container) and Apptainer (legacy)
export EXPERIMENTS_DIR
export DATAGEN_CONFIG
export TRACE_SCRIPT
export TRACE_TARGET_REPO
export TRACE_HARBOR_CONFIG
export TRACE_MODEL
export TRACE_ENGINE
export TRACE_BACKEND
export TASKS_REPO
export TASKS_REVISION
export PARQUET_NAME
export GPUS_PER_NODE
export TIME_LIMIT
export TRACE_AGENT_NAME
export TRACE_AGENT_KWARGS
export TRACE_ENV
export TRACE_AGENT_TIMEOUT_SEC
export TRACE_N_CONCURRENT
export TRACE_EPISODES
export TRACE_EXPORT_FILTER
export TRACE_DATASET_TYPE
export TRACE_CHUNK_SIZE
export OVERWRITE_TASKS

# Legacy Apptainer exports (kept for compatibility if container needed later)
export APPTAINERENV_EXPERIMENTS_DIR="$EXPERIMENTS_DIR"
export APPTAINERENV_DATAGEN_CONFIG="$DATAGEN_CONFIG"
export APPTAINERENV_TRACE_SCRIPT="$TRACE_SCRIPT"
export APPTAINERENV_TRACE_TARGET_REPO="$TRACE_TARGET_REPO"
export APPTAINERENV_TRACE_HARBOR_CONFIG="$TRACE_HARBOR_CONFIG"
export APPTAINERENV_TRACE_MODEL="$TRACE_MODEL"
export APPTAINERENV_TRACE_ENGINE="$TRACE_ENGINE"
export APPTAINERENV_TRACE_BACKEND="$TRACE_BACKEND"
export APPTAINERENV_TASKS_REPO="$TASKS_REPO"
export APPTAINERENV_TASKS_REVISION="$TASKS_REVISION"
export APPTAINERENV_PARQUET_NAME="$PARQUET_NAME"
export APPTAINERENV_GPUS_PER_NODE="$GPUS_PER_NODE"
export APPTAINERENV_TIME_LIMIT="$TIME_LIMIT"
export APPTAINERENV_TRACE_AGENT_NAME="$TRACE_AGENT_NAME"
export APPTAINERENV_TRACE_AGENT_KWARGS="$TRACE_AGENT_KWARGS"
export APPTAINERENV_TRACE_ENV="$TRACE_ENV"
export APPTAINERENV_TRACE_AGENT_TIMEOUT_SEC="$TRACE_AGENT_TIMEOUT_SEC"
export APPTAINERENV_TRACE_N_CONCURRENT="$TRACE_N_CONCURRENT"
export APPTAINERENV_TRACE_EPISODES="$TRACE_EPISODES"
export APPTAINERENV_TRACE_EXPORT_FILTER="$TRACE_EXPORT_FILTER"
export APPTAINERENV_TRACE_DATASET_TYPE="$TRACE_DATASET_TYPE"
export APPTAINERENV_TRACE_CHUNK_SIZE="$TRACE_CHUNK_SIZE"
export APPTAINERENV_OVERWRITE_TASKS="$OVERWRITE_TASKS"

# Disable any overlay settings that might point to missing files
unset APPTAINER_OVERLAY SINGULARITY_OVERLAY SINGULARITYENV_OVERLAY 2>/dev/null || true

# Find the run script - SLURM_SUBMIT_DIR is where sbatch was invoked
# The script should be at hpc/scripts/kimi-k2-tracegen-run-v2.sh relative to repo root
REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
RUN_SCRIPT_PATH="$REPO_ROOT/hpc/scripts/kimi-k2-tracegen-run-v2.sh"

# Fallback: try SCRATCH location
if [[ ! -f "$RUN_SCRIPT_PATH" ]]; then
  RUN_SCRIPT_PATH="${SCRATCH:-$HOME}/OpenThoughts-Agent/hpc/scripts/kimi-k2-tracegen-run-v2.sh"
fi

if [[ ! -f "$RUN_SCRIPT_PATH" ]]; then
  echo "ERROR: Expected helper script at $RUN_SCRIPT_PATH but it was not found." >&2
  echo "       SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" >&2
  echo "       SCRATCH=$SCRATCH" >&2
  exit 1
fi

echo ">>> Running trace generation script: $RUN_SCRIPT_PATH"

# Don't use singularity container - run directly with conda environment
# The vista approach uses srun directly without container wrapping
/bin/bash "$RUN_SCRIPT_PATH"
