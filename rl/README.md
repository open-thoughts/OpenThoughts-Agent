# OT-Agent HPC Training System for RL

This is the respository of running RL experiments on clusters. Currently we only support TACC and JSC.

## Reproducing OpenThinker-Agent-v1
If you want to re-create the OpenThoughts-Agent experiment, but you're not on a cluster, you can check out the README here: https://github.com/mlfoundations/SkyRL

You can follow the steps to:
- Use [open-thoughts/OpenThinker-Agent-v1-SFT](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1-SFT) as base
- GRPO with the data [open-thoughts/OpenThoughts-Agent-v1-RL](https://huggingface.co/datasets/open-thoughts/OpenThoughts-Agent-v1-RL), while
- Evaluate with [open-thoughts/OpenThoughts-TB-dev](https://huggingface.co/datasets/open-thoughts/OpenThoughts-TB-dev), and 
- Get the final [open-thoughts/OpenThinker-Agent-v1](https://huggingface.co/open-thoughts/OpenThinker-Agent-v1)

While we are using a fork for now, we will soon merge changes needed to main branch of SkyRL.

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

## On JSC (JUWELS / Jureca / Jupiter)

Be inside your `OpenThoughts-Agent` repo.

```bash
source /p/project1/ccstdl/nezhurina1/miniconda/miniconda/bin/activate
conda activate /p/scratch/laionize/zorlu1/envs/dc-agent-skyrl
```

Install SkyRL inside `$SKYRL_HOME`
```
cd $SKYRL_HOME
git clone https://github.com/NovaSky-AI/SkyRL.git
cd SkyRL/skyrl-train && pip install -e .
```

Modify keys in `rl/hpc/dotenv/jsc.env`, if necessary.

Populate `rl/hpc/dotenv/secret.env`:

```
export DAYTONA_API_KEY=YOUR_KEY
export HF_TOKEN=YOUR_KEY
export WANDB_API_KEY=YOUR_KEY
```


JSC clusters have **no outbound internet from compute nodes**, so RL runs talk to remote containers via an SSH-based SOCKS proxy and `proxychains4`.

- **Proxychains install (one-time):**

  ```bash
  git clone https://github.com/rofl0r/proxychains-ng.git
  cd proxychains-ng
  ./configure --prefix=$HOME/.local
  make
  make install
  export PATH=$HOME/.local/bin:$PATH
  ```

  This provides `proxychains4` and creates `~/.proxychains/proxychains.conf`; our `rl/hpc/sbatch/ensure_proxychains.sh` checks for both before jobs run.

- **SSH key + tunnel setup:**
  - Log into a JUWELS login node (for example `jwlogin21.juwels`) using your JuDoor SSH key.
  - `python -m hpc.launch` renders `jsc_train.j2`, which in turn calls:
    - `start_tunnel.sh` – sets up a Docker host tunnel for Ray / vLLM.
    - `start_proxy_tunnel.sh` – sets up a SOCKS proxy via an internet-facing login node (e.g. `jwlogin22i`).
  - All network traffic from compute nodes to remote containers is routed through that SOCKS proxy.
  - To make this work:
    1. Generate (or reuse) an ED25519 key locally:

       ```bash
       ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519
       ```

    2. Upload `~/.ssh/id_ed25519.pub` to JuDoor under the JUWELS section and add a `from=` clause that allows both your home IP(s) and internal JUWELS hosts (e.g. `from="*.juwels,*.kfa-juelich.de,*.fz-juelich.de,134.94.52.0/24"`).
    3. On JUWELS, append the same public key into `~/.ssh/authorized_keys` and lock down permissions:

       ```bash
       cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
       chmod 600 ~/.ssh/id_ed25519
       ```

       Re-check `authorized_keys` if you later change keys in JuDoor.
    4. Point OT-Agent at this key in `rl/hpc/dotenv/secret.env`, for example:

       ```bash
       export SSH_KEY=/p/home/jusers/<user>/juwels/.ssh/id_ed25519
       ```

  Once these are in place, `start_tunnel.sh` / `start_proxy_tunnel.sh` will open the reverse proxy from login/compute nodes to the remote containers, and the JSC launch path will behave like other clusters from the user’s perspective.

Finally run:

```bash
source hpc/setup.sh
source hpc/setup.sh  # due to some bug, need to run twice TODO(Charlie): fix
bash hpc/scripts/sync_rl_scripts/run_nl2bash_gpt5codex_cleaned.sh
```

After your run is finished, don't forget to sync wandb.

```
export WANDB_API_KEY=xxx
wandb sync path/to/wandb_logs
```


## Supported Clusters

### TACC Clusters
- **Vista**: `vista.tacc.utexas.edu` - GH200 96GB GPUs
- **Lonestar**: `ls6.tacc.utexas.edu` - A100 40GB GPUs

### JSC Clusters
- **Jureca**: `jureca` - H100 94GB GPUs
- **Jupiter**: `jupiter.internal` - GH200 96GB GPUs  
- **Juwels**: `juwels` - A100 40GB GPUs

## Configuration Files

### Environment Files
- `dotenv/tacc.env`: TACC cluster environment variables
- `dotenv/jsc.env`: JSC cluster environment variables

### SBatch Jinja Templates
- `sbatch/tacc_train.j2`: TACC cluster job template
- `sbatch/jsc_train.j2`: JSC cluster job template.

### Other clusters
To support other clusters, the main things to implement are the sbatch jinja template, and an instantiation of `HPC` in `hpc.py`.
