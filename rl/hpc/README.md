# OT-Agent HPC Training System

This is the respository of running RL experiments on clusters. Currently we only support TACC, with JSC coming soon.

## On TACC:

Be inside your `OpenThoughts-Agent` repo.

```bash
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/08134/negin/dc-agent-shared/SkyRL/envs/tacc_rl_v5
cd rl/
```

Modify keys in `rl/hpc/dotenv/tacc.env`, adding `OT_AGENT`

Populate `rl/hpc/dotenv/secret.env`:

```
export DAYTONA_API_KEY=YOUR_KEY
export HF_TOKEN=YOUR_KEY
export WANDB_API_KEY=YOUR_KEY
```

Then run:

```bash
source hpc/setup.sh
source hpc/setup.sh  # due to some bug, need to run twice TODO(Charlie): fix
bash hpc/scripts/sync_rl_scripts/run_nl2bash_gpt5codex_cleaned.sh
```

## Supported Clusters

### TACC Clusters
- **Vista**: `vista.tacc.utexas.edu` - GH200 96GB GPUs
- **Lonestar**: `ls6.tacc.utexas.edu` - A100 40GB GPUs


<!-- ### JSC Clusters
- **Jureca**: `jureca` - H100 94GB GPUs
- **Jupiter**: `jupiter.internal` - GH200 96GB GPUs  
- **Juwels**: `juwels` - A100 40GB GPUs -->

## Configuration Files

### Environment Files
- `dotenv/tacc.env`: TACC cluster environment variables
- `dotenv/jsc.env`: JSC cluster environment variables

### SBatch Jinja Templates
- `sbatch/tacc_train.j2`: TACC cluster job template
- `sbatch/jsc_train.j2`: TO BE ADDED

### Other clusters
To support other clusters, the main things to implement are the sbatch jinja template, and an instantiation of `HPC` in `hpc.py`.
