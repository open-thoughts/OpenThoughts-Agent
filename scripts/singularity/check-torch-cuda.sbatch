#!/bin/bash
#SBATCH --job-name=torch-cuda-check
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_40_tandon_advanced

set -euo pipefail

singularity exec --nv /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif /bin/bash <<'EOF'
source ~/.bashrc
source ~/secrets.env
cd $SCRATCH/OpenThoughts-Agent
source hpc/dotenv/nyutorch.env
git pull
conda activate dcagent
python - <<'PY'
import torch

print("torch", torch.__version__)
cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)
if cuda_available:
    for idx in range(torch.cuda.device_count()):
        print(f"CUDA device {idx}:", torch.cuda.get_device_name(idx))
PY
EOF
