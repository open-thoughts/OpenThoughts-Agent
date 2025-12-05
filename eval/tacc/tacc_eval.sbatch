#!/bin/bash
#SBATCH -p gh-dev
#SBATCH --time=02:00:00
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=c610-021,c611-011,c640-041,c611-041,c611-122,c637-082,c637-091,c610-111
#SBATCH --account CCR24067
#SBATCH --output=experiments/logs/%x_%j.out
#SBATCH --job-name=eval

module load cuda/12.4 nccl/12.4 nvidia_math/12.4
module load tacc-apptainer

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08002/gsmyrnis/miniconda3/envs/dcft

export IMAGE="/scratch/08134/negin/dc-agent-shared/dc-agent/images/python_3.13.sif"
chmod 666 -R /scratch/10000/eguha3/ssh_keys/etash_docker_v2
ssh -L 127.0.0.1:23753:127.0.0.1:23750 -N -f -i  /scratch/10000/eguha3/ssh_keys/etash_docker_v2 eguha3@34.174.178.159 &

python -m vllm.entrypoints.openai.api_server \
 --host 0.0.0.0 --port 8000 \
 --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
 --served-model-name Qwen/Qwen2.5-Coder-0.5B \
 --device cuda --gpu-memory-utilization 0.85 &

sleep 30
chmod 777 -R /scratch/10000/eguha3/ssh_keys/etash_docker_v2
apptainer exec --nv --bind /scratch/08134/negin/dc-agent-shared/docker:/usr/local/bin/docker --bind /scratch/08134/negin/dc-agent-shared/.docker:/root/.docker:ro --bind /tmp:/tmp \
  ${IMAGE} \
  bash -c "
  source /scratch/08134/negin/dc-agent-shared/tb-env/bin/activate
  export DOCKER_HOST=tcp://127.0.0.1:23753
  docker ps
  rm -f ~/.docker/cli-plugins/docker-compose
  # Create the directory if it doesn't exist
  mkdir -p ~/.docker/cli-plugins

  # Download the correct architecture version
  curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" -o ~/.docker/cli-plugins/docker-compose

  # Make it executable
  chmod +x ~/.docker/cli-plugins/docker-compose
  docker system prune -a -f
  mkdir -p ~/.docker/cli-plugins
  ## INSTALLING FOR ARM64
  curl -SL https://github.com/docker/buildx/releases/download/v0.11.2/buildx-v0.11.2.linux-arm64 -o ~/.docker/cli-plugins/docker-buildx
  chmod +x ~/.docker/cli-plugins/docker-buildx
  python /scratch/08134/negin/dc-agent-shared/dc-agent/eval/example_tbench.py"
