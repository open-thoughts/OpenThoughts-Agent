#!/bin/bash
#SBATCH -J tb-vllm
#SBATCH -o sbatch_logs/tb-vllm.%j.out
#SBATCH -e sbatch_logs/tb-vllm.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -p gh                     # uncomment/adjust partition if needed
#SBATCH -A CCR24067              # CCR24067, CGAI24022
##SBATCH --mem=64G                # add memory if your site requires explicit mem

set -euo pipefail

# ---- USER VARS: update these paths/values for your env ----
HF_HOME="/scratch/10672/wenxuand/hf_home/"
# CONDA_ENV="/scratch/08002/gsmyrnis/miniconda3/envs/dcft"
CONDA_ENV="/work/10672/wenxuand/anaconda3/envs/vllm"
APPTAINER_IMG="python_3.13.sif"
APPTAINER_BINDS=(
  "--bind $(pwd)/docker:/usr/local/bin/docker"
  "--bind $(pwd)/.docker:/root/.docker:ro"
  "--bind /tmp:/tmp"
)
GROUP_NAME="G-827553"
VLLM_PORT=8000
VLLM_HOST="127.0.0.1"            # connect via loopback; server binds 0.0.0.0
MODEL_NAME="Qwen/Qwen3-8B"
GPU_MEM_UTIL="0.85"

# Remote Docker VM SSH
SSH_KEY="/scratch/10672/wenxuand/dc-agent-shared/.ssh/docker_vm_key"
# SSH_KEY="/home/08134/negin/.ssh/docker_vm_key"
SSH_USER_HOST="negin_raoof@34.42.235.32"
DOCKER_REMOTE_TCP="127.0.0.1:23750"
DOCKER_LOCAL_TCP="127.0.0.1:23751"

# Terminal-Bench repo dir (inside the batch job working dir)
TB_DIR="$PWD/terminal-bench"

# If your TB wrapper isn't patched to always include max_tokens & ignore response_format,
# keep this; vLLM requires max_tokens.
TB_EXTRA_ARGS=("--agent-kwarg" "temperature=0.2" )
# -----------------------------------------------------------

echo "[INFO] Node: $HOSTNAME"
date

module load cuda
# source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
source /work/10672/wenxuand/anaconda3/etc/profile.d/conda.sh 

conda activate "$CONDA_ENV"

# --- Start vLLM server in background ---
LOG_DIR="$PWD/_job_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"

echo "[INFO] Starting vLLM..."
export LD_LIBRARY_PATH=/work/10672/wenxuand/anaconda3/envs/vllm/lib:$LD_LIBRARY_PATH
nohup env HF_HOME="/scratch/10672/wenxuand/hf_home/" \
         HF_HUB_CACHE="/scratch/10672/wenxuand/hf_home/" \
         HUGGINGFACE_HUB_CACHE="/scratch/10672/wenxuand/hf_home/" \
         TRANSFORMERS_CACHE="/scratch/10672/wenxuand/hf_home/" \
         python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --model "$MODEL_NAME" \
  --served-model-name "$MODEL_NAME" \
  --device cuda \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  >"$VLLM_LOG" 2>&1 &

VLLM_PID=$!

LD_PRELOAD="/usr/lib64/libssl.so.3.0.7:/usr/lib64/libcrypto.so.3.0.7"

cleanup() {
  echo "[CLEANUP] Killing vLLM (PID $VLLM_PID) and SSH tunnel (if any)..."
  if kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID" || true; fi
  if [[ -f "$LOG_DIR/ssh_tun.pid" ]]; then
    TUN_PID=$(cat "$LOG_DIR/ssh_tun.pid" || true)
    if [[ -n "${TUN_PID:-}" ]] && kill -0 "$TUN_PID" 2>/dev/null; then kill "$TUN_PID" || true; fi
  fi
}
trap cleanup EXIT

# --- Wait for vLLM readiness ---
echo "[INFO] Waiting for vLLM on http://${VLLM_HOST}:${VLLM_PORT}/v1/models ..."
for i in {1..60}; do
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "[INFO] vLLM is up."
    break
  fi
  sleep 2
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM died during startup. Last 50 lines:"
    tail -n 50 "$VLLM_LOG" || true
    exit 1
  fi
  if [[ $i -eq 60 ]]; then
    echo "[ERROR] vLLM did not become ready. Log tail:"
    tail -n 100 "$VLLM_LOG" || true
    exit 1
  fi
done

# --- Bring up SSH tunnel to remote Docker daemon ---
echo "[INFO] Starting SSH tunnel to $SSH_USER_HOST (Docker $DOCKER_REMOTE_TCP -> local $DOCKER_LOCAL_TCP)..."
# Ensure key perms are strict
chmod 600 "$SSH_KEY"
# ExitOnForwardFailure fails fast if bind/listen fails
ssh -fN -o ExitOnForwardFailure=yes -i "$SSH_KEY" \
  -L "${DOCKER_LOCAL_TCP}:${DOCKER_REMOTE_TCP}" \
  "$SSH_USER_HOST"
echo $! > "$LOG_DIR/ssh_tun.pid"
chmod 777 "$SSH_KEY"

# Verify Docker proxy reachable
echo "[INFO] Verifying Docker API via tunnel..."
for i in {1..10}; do
  if curl -s "http://${DOCKER_LOCAL_TCP}/version" | grep -q '"ApiVersion"'; then
    echo "[INFO] Docker tunnel OK."
    break
  fi
  sleep 1
  if [[ $i -eq 10 ]]; then
    echo "[ERROR] Docker tunnel verification failed."
    exit 1
  fi
done

# --- Run Terminal-Bench inside Apptainer non-interactively ---
module load tacc-apptainer

# Use group G-827553 for access: use sg to set primary gid for the contained process
echo "[INFO] Running Terminal-Bench inside Apptainer..."
SG_CMD="source ./tb-env/bin/activate; \
export DOCKER_HOST=tcp://${DOCKER_LOCAL_TCP}; \
cd '${TB_DIR}'; \
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt; \
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt; \
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt; \
echo '[INFO] Running: terminal-bench run ...'; \
tb run \
  --agent terminus \
  --model 'hosted_vllm/${MODEL_NAME}' \
  --agent-kwarg api_base=http://${VLLM_HOST}:${VLLM_PORT}/v1 \
  --dataset-name terminal-bench-core \
  --dataset-version 0.1.1 \
  ${TB_EXTRA_ARGS[*]} \
  --n-concurrent 32"
  # --task-id hello-world" # eval-mteb
  
    

  
# tb run \
#   --agent terminus \
#   --model  o4-mini \
#   --dataset-name terminal-bench-core \
#   --dataset-version 0.1.1 \
#   --n-concurrent 16"


# apptainer exec + sg to set group; bash -lc to get a shell for env sourcing
sg "$GROUP_NAME" -c "apptainer exec ${APPTAINER_BINDS[*]} '$APPTAINER_IMG' bash -lc \"$SG_CMD\""

echo "[INFO] Terminal-Bench completed."

# tb run --agent terminus --model 'hosted_vllm/hosted_vllm/Qwen/Qwen2.5-Coder-0.5B' --agent-kwarg api_base=http://127.0.0.1:8000/v1 --agent-kwarg max_tokens=512 --agent-kwarg temperature=0.2 --task-id hello-world