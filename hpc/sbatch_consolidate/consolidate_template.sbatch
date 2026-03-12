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
#SBATCH --mail-type=END,TIME_LIMIT,FAIL

set -euo pipefail

# --- Environment setup (auto-generated from HPC config) ---
{environment_preamble}

# --- Source dotenv and secrets ---
if [ -n "${{DCFT:-}}" ] && [ -f "$DCFT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DCFT/hpc/dotenv/{cluster_env_file}"
elif [ -n "${{DC_AGENT:-}}" ] && [ -f "$DC_AGENT/hpc/dotenv/{cluster_env_file}" ]; then
  source "$DC_AGENT/hpc/dotenv/{cluster_env_file}"
fi

if [ -n "${{DCFT_ACTIVATE_ENV:-}}" ]; then
  set +u
  eval "$DCFT_ACTIVATE_ENV"
  set -u
fi

SECRET_FILE="${{DC_AGENT_SECRET_ENV:-${{KEYS:-}}}}"
if [[ -n "${{SECRET_FILE}}" && -f "${{SECRET_FILE}}" ]]; then
  set -a
  source "${{SECRET_FILE}}"
  set +a
fi

export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1

if [ -z "${{DCFT:-}}" ]; then
  if [ -n "${{DC_AGENT:-}}" ]; then
    export DCFT="$DC_AGENT"
  else
    export DCFT="$PWD"
  fi
fi

# --- SSH Tunneling (JSC clusters only) ---
{ssh_tunnel_setup}

# --- SOCKS5 Proxy Setup (JSC clusters - alternative to SSH tunnel) ---
{proxy_setup}

# --- Proxy status and PROXY_CMD setup ---
# Jupiter/Leonardo: PROXYCHAINS_BIN wraps commands (set by ssh_tunnel_setup, LD_PRELOAD unreliable on ARM)
# PROXYCHAINS_BINARY is the equivalent from get_proxy_setup() (shared SOCKS5 proxy)
# Jureca/Juwels: LD_PRELOAD inherited by child processes automatically
PROXY_CMD=""
_PC_BIN="${{PROXYCHAINS_BINARY:-${{PROXYCHAINS_BIN:-}}}}"
if [ -n "$_PC_BIN" ] && [ -n "${{PROXYCHAINS_CONF_FILE:-}}" ]; then
    PROXY_CMD="$_PC_BIN -f ${{PROXYCHAINS_CONF_FILE}}"
    echo "Proxy: ENABLED via wrapped binary"
    echo "  PROXY_CMD: $PROXY_CMD"
elif [ -n "${{LD_PRELOAD:-}}" ]; then
    echo "Proxy: ENABLED via LD_PRELOAD"
    echo "  LD_PRELOAD: $LD_PRELOAD"
else
    echo "Proxy: DISABLED (direct internet or not configured)"
fi

mkdir -p "{experiments_dir}"
mkdir -p "{experiments_dir}/logs"

export CONSOLIDATE_INPUT="{consolidate_input}"
export CONSOLIDATE_INPUT_KIND="{consolidate_input_kind}"
export CONSOLIDATE_OUTPUT_REPO="{consolidate_output_repo}"
export CONSOLIDATE_BASE_REPO="{consolidate_base_repo}"
export CONSOLIDATE_WORKDIR="{consolidate_workdir}"
export CONSOLIDATE_COMMIT_MESSAGE="{consolidate_commit_message}"
export CONSOLIDATE_PUSH_TO_HUB="{consolidate_push_to_hub}"

$PROXY_CMD python3 {python_script} \
  --input "${{CONSOLIDATE_INPUT}}" \
  --input-kind "${{CONSOLIDATE_INPUT_KIND}}" \
  --output-repo "${{CONSOLIDATE_OUTPUT_REPO}}" \
  --base-repo "${{CONSOLIDATE_BASE_REPO}}" \
  --workdir "${{CONSOLIDATE_WORKDIR}}" \
  --commit-message "${{CONSOLIDATE_COMMIT_MESSAGE}}" \
  --push-to-hub "${{CONSOLIDATE_PUSH_TO_HUB}}" \
  --project-root "{project_root}"
