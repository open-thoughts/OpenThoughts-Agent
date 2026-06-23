## NYU Torch Access

**SSH**: `ssh torch` (alias in ~/.ssh/config). User: `bf996`. Requires `-o StrictHostKeyChecking=no` (host keys rotate).

**Pre-launch preamble** (run before launching any new job):
```bash
cd ~/harbor && git pull && \
cd /scratch/bf996/SkyRL && git pull && \
cd /scratch/bf996/OpenThoughts-Agent && \
conda activate dcagent312 && \
source hpc/dotenv/nyutorch.env && source ~/secrets.env && \
git pull && git submodule update --init --remote sft/llamafactory
```

**Key paths**:
- Code: `/scratch/bf996/OpenThoughts-Agent`
- SkyRL: `/scratch/bf996/SkyRL`
- Harbor: `~/harbor`
- Conda env: `dcagent312` (Python 3.12, PyTorch 2.9+cu128, vLLM 0.16+)
- Conda python: `/scratch/bf996/miniconda3/envs/dcagent312/bin/python`

**Cluster details**: H200 141GB GPUs (8/node, 29 nodes = 232 GPUs) + L40S 48GB GPUs (4/node, 68 nodes = 272 GPUs). SLURM scheduler. Internet on compute nodes. NVIDIA driver 580.82, CUDA 13.0.

**GPU partitions** (use `--partition`):
- `h200_tandon` — primary H200 partition (up to 112 GPUs)
- `h200_tandon,h200_public` — fallback combo for H200
- `h200_public` — shared H200 (up to 24 GPUs)
- `l40s_public` — shared L40S (up to 208 GPUs)
- `l40s_courant` — Courant L40S (up to 52 GPUs)

**QOS limits** (wall time):
- `gpu48` — 2 day max, 2000 job limit
- `gpu168` — 7 day max, 50 job limit
- `gpuplus` — 7 day max, 50 job limit
- `interactive` — 6 hour max, 20 job limit

**SLURM account**: `torch_pr_40_tandon_advanced`

**Interactive session**:
```bash
srun --gres=gpu:h200:1 --nodes=1 --tasks-per-node=1 --cpus-per-task=8 --mem=32GB --time=04:00:00 --account=torch_pr_40_tandon_advanced --pty /bin/bash
```

**Package management**: Use `uv pip install` for all installs on Torch.

**Datagen on Torch**:

Before launching datagen, you must first extract tasks from the source parquet dataset:
```bash
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet mlfoundations-dev/ling-coder-sft-sandboxes-1 \
  --output_dir $SCRATCH/tasks/ling-coder-sft-sandboxes-1 \
  --on_exist overwrite
```

Then launch the datagen job:
```bash
python3 -m hpc.launch \
  --job_type datagen \
  --trace_harbor_config "./hpc/harbor_yaml/datagen/ctx32k.yaml" \
  --datagen_config kimi_k2_5_vllm_serve_torch_h200.yaml \
  --tasks_input_path "$SCRATCH/tasks/stackexchange-tezos-sandboxes" \
  --trace_target_repo DCAgent2/Kimi-2.5-stackexchange-tezos-sandboxes-maxeps-32k \
  --time_limit 47:59:00 \
  --num_nodes 1 \
  --gpus_per_node 8 \
  --trace-n-concurrent 20
```

Key flags: `--datagen_config` selects the vLLM serving config (model, TP, etc.), `--tasks_input_path` points to the extracted tasks dir, `--trace_target_repo` is the HF repo for output traces.

**Rsync files to local** (from Mac):
```bash
rsync -avz --progress -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null" \
  bf996@login.torch.hpc.nyu.edu:/scratch/bf996/path /local/path
```
