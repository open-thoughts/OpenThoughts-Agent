# OT-Agent: Data Recipes for Training Agentic Models

Welcome to OT-Agent, a large-scale research project dedicated to creating the best tooling and finding the best data for training small agentic models.

## Links

Project Website: Coming Soon

[OT-Agent Leaderboard](https://ot-agent-leaderboard.replit.app/)

[OT-Agent Trace Viewer](https://dcagents-trace-viewer.replit.app/)

## Warning!

OT-Agent is a research codebase! Conventions will change, files will move and workflows will break as we continue to grow. Please bear with us and open an issue if you discover a bug.

## Getting Started

If you are new to the project, start here to get up and running.

### Installation

Start by creating a clean Python 3.12 virtual environment using your favorite environment manager (we use `conda + mamba`).

From the root directory, you can then install the core HPC + data infrastructure dependencies with:

`pip install .`

As this project contains many dependencies, we recommend the use of a package and project management tool such as `uv` within your virtual environment.

Optional extras:

* **HPC datagen runtime** (Ray clusters + vLLM serving):
  `pip install .[datagen]`
* **SweSmith-specific datagen helpers** (extends the above with bespoke tools):
  `pip install .[datagen,datagen-swesmith]` (or `pip install .[datagen-swesmith]` if you already pulled the base datagen extra)
* **RL training stack** (installs the latest SkyRL straight from GitHub):
  `pip install .[rl]`

* **SFT stack**:  
  * Ensure the git submodule is initialized:  
    `git submodule update --init --recursive sft/llamafactory`
  * Install LLaMA Factory directly from the submodule since OT-Agent does not ship an `[sft]` extra:  
    ```bash
    cd sft/llamafactory
    pip install -e .[train,liger-kernel,deepspeed]  # select the extras you need
    cd -
    ```
  * Training configs that pair with OT-Agent live under `sft/lf_configs/**`; refer to `sft/llamafactory/README.md` for detailed flags and dependency notes.
* **Data stack**
  * Dataset tooling docs live under `data/README.md`; install per-generator requirements in addition to the `datagen` extras above when needed.

#### Notes on CPP

Many OT-Agent launch modes JIT-compile CUDA/C++ extensions (e.g., `flash-infer`, `flash-attn`, `triton`). Those builds are sensitive to compiler and CUDA versions, so verify that the toolchain you expose to Python matches the version of PyTorch you installed (`python - <<<'import torch; print(torch.version.cuda)'`). We primarily test on CUDA 12.8/12.9 with GCC ≥12.

**Cluster modules.** If your HPC environment exposes the right stack, loading modules is the path of least resistance:

```bash
module load gcc/14.2.0
module load cuda/12.8
```

**Container shells.** Some centers publish pre-baked CUDA images. Binding your workspace into one of those containers often guarantees a clean toolchain:

```bash
singularity shell --nv \
  --bind $SCRATCH/ot-agent \
  $SCRATCH/cuda-img/cuda-cudnn-12.8-ubuntu22.sif
```

**Conda-provisioned toolchains.** When neither modules nor containers provide what you need, install the compilers and sysroot via mamba. Keep the packages pinned so minor upgrades don’t silently change ABI compatibility:

```bash
mamba install -c conda-forge c-compiler cxx-compiler -y
mamba install -c conda-forge gcc_linux-64 gxx_linux-64 sysroot_linux-64 -y
mamba install -c conda-forge libstdcxx-ng=12 libgcc-ng=12 gcc_impl_linux-64 \
    gxx_impl_linux-64 sysroot_linux-64 -y
```

**Environment variables.** Point CUDA- and GCC-aware tools at the locations you provisioned. Adjust the paths below if your install lives somewhere else:

```bash
GCC_ROOT="$(dirname "$(dirname "$(which gcc)")")"
export CUDA_HOME=/usr/local/cuda
export CPATH="$CUDA_HOME/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CUDA_HOME/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$GCC_ROOT/lib64:$GCC_ROOT/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="$CUDA_HOME/bin${PATH:+:$PATH}"
```

**Heavyweight builds.** Once the toolchain is stable, the JIT pieces compile automatically on import. Some packages (like `flash-attn`) still require manual builds—install those last so you know the rest of the environment is steady, and make sure `TORCH_CUDA_ARCH_LIST`, `NVCC_THREADS`, etc. match your hardware:

```bash
UV_COMPILE_THREADS=4 MAX_JOBS=4 NVCC_THREADS=4 TORCH_CUDA_ARCH_LIST="9.0" \
  pip install -v --no-build-isolation "flash-attn==2.8.1"
```

### Secrets and API Keys

Most scripts expect credentials (HF tokens, Daytona keys, W&B API keys, Supabase creds, etc.) to live in a private `env` file that is **not** committed to this repo. Point OT-Agent at your private file by exporting:

```bash
export DC_AGENT_SECRET_ENV=/secure/path/to/my_dc_agent_secrets.env
```

That file should `export DAYTONA_API_KEY=...`, `export HF_TOKEN=...`, `export WANDB_API_KEY=...`, `export SUPABASE_*`, etc. The launcher and auxiliary scripts now read `DC_AGENT_SECRET_ENV`; legacy `KEYS`/`SECRET_ENV_PATH` variables are still accepted for backward compatibility but will be removed once everyone migrates.

### Launching a Job

OT-Agent's job launchers are designed to work with HPC (high-performance computing) clusters. Different launchers exist for different job types. OT-Agent's launchers are modular, making it relatively straightforward to add your own preferred cluster.

#### How to Launch a Datagen Job

Datagen jobs are launched via the generic HPC launcher and use `--job_type datagen` plus a generator script.

1. Ensure your cluster environment is set up (dotenv, conda env, etc.). For TACC/Vista-style machines, follow the checklist in `hpc/README.md` and use `hpc/dotenv/tacc.env` as a starting point for your environment variables.
2. Activate your environment and source the dotenv:
   ```bash
   source hpc/dotenv/<your-cluster>.env
   $DCFT_ACTIVATE_ENV
   cd "$DCFT"
   ```
3. Choose or write a datagen script under `data/...` implementing `BaseDataGenerator` (see `data/generation/base.py` and existing generators for examples).
4. Run the launcher from a login node:
   ```bash
   python -m hpc.launch \
     --job_type datagen \
     --datagen_script data/<dataset>/generate.py \
     --datagen_target_repo <org/dataset-tasks> \
     --datagen_engine vllm_local \
     --datagen_extra_args "--stage both --limit 200" \
     --experiments_dir "$DCFT/experiments" \
     --time_limit 12:00:00
   ```
5. To also generate traces, add:
   - `--enable_trace_gen`  
   - `--trace_target_repo <org/dataset-traces>`  
   - `--trace_harbor_config path/to/harbor_job.yaml`  
   and any of the `trace_*` overrides documented in `hpc/README.md`.

The launcher will synthesize and submit one or more `sbatch` scripts under `"$experiments_dir/sbatch_scripts"` and write configs to `"$experiments_dir/configs"`. Use `--dry_run` to inspect scripts without actually calling `sbatch`.

#### How to Launch an SFT Job

SFT jobs are also launched via `hpc.launch` with `--job_type train` and a LLaMA Factory config.

1. Pull and install the SFT submodule (once per checkout) and install its dependencies in-place:
   ```bash
   git submodule update --init --remote sft/llamafactory
   cd sft/llamafactory
   pip install -e .[train,liger-kernel,deepspeed]  # pick the extras you need
   cd -
   ```
2. Configure your cluster dotenv and environment as in the Datagen section.
3. Pick a training config under `sft/lf_configs` or create your own YAML alongside the existing presets.
4. From a login node, run:
   ```bash
   python -m hpc.launch \
     --job_type train \
     --train_config_path sft/lf_configs/<path-to-config>.yaml \
     --dataset <org/dataset> \
     --num_nodes 8 \
     --time_limit 24:00:00 \
     --experiments_dir "$DCFT/experiments"
   ```
5. Optionally override LLaMA Factory flags via `--train_extra_args "..."` (see `hpc/README.md` and `sft/llamafactory/README.md` for full argument lists).

The launcher will construct a per-run YAML in `"$experiments_dir/configs"`, generate an sbatch script, and then submit the job. Training metadata and summaries are written into the run’s `output_dir`.

#### How to Launch an RL Job

RL training currently uses cluster-specific scripts under `rl/` rather than the generic `hpc.launch` entry point.

1. Make sure you have access to the shared RL environment and Ray/vLLM backend described in the TACC docs and comments inside `rl/tacc/tacc_train_rl_tbench.sh`.
2. Log into the target cluster (e.g., TACC Vista) and load the required modules (CUDA, Apptainer, GCC) as shown in the script.
3. Edit `rl/tacc/tacc_train_rl_tbench.sh` to point to your:
   - data directories
   - checkpoint/output paths
   - base model ID (e.g. `Qwen/Qwen2.5-7B-Instruct`)
   - sandboxes / trace storage locations
4. From a login node, submit the job:
   ```bash
   sbatch rl/tacc/tacc_train_rl_tbench.sh
   ```
5. Monitor logs under the `experiments/logs` directory configured in the script and resume/tune hyperparameters via the `skyrl_train.entrypoints.main_base` arguments inside the sbatch file.

#### How to add your cluster to OT-Agent

Adding a new cluster involves defining its resources, sbatch templates, and a dotenv file so `hpc.launch` can target it.

1. **Create a dotenv for your cluster** under `hpc/dotenv/`, following `tacc.env` as a template. At a minimum, define:
   - `DCFT` (path to your dc-agent checkout on the cluster)
   - `DCFT_ACTIVATE_ENV` (command to activate the Python env)
   - paths for `EXPERIMENTS_DIR`, `DATASETS_DIR`, `MODELS_DIR`, and any cluster-specific SIF/Apptainer images.
2. **Register basic cluster metadata** by exporting `HPC_NAME` and related fields in your dotenv or by passing them on the CLI:
   - `--name`, `--account`, `--partition`, `--gpus_per_node`, `--cpus_per_node`, etc. (see `hpc/README.md` and `hpc/hpc.py`).
3. **Create sbatch templates** in `hpc/sbatch_data/` for your cluster:
   - Copy an existing template for a similar machine (GPU type / internet access) and adjust `#SBATCH` headers and module loads.
   - Keep placeholders like `{time_limit}`, `{job_name}`, `{experiments_dir}` etc. intact; they will be filled by `hpc.launch`.
4. **Declare required templates** in `hpc/sbatch_data_requirements.json` so `_validate_sbatch_templates` can verify your cluster has all needed sbatch files for datagen and training.
5. **Test with a dry run**:
   ```bash
   source hpc/dotenv/<your-cluster>.env
   $DCFT_ACTIVATE_ENV
   cd "$DCFT"
   python -m hpc.launch \
     --job_type datagen \
     --datagen_script data/<dataset>/generate.py \
     --datagen_target_repo test-org/test-dataset \
     --experiments_dir "$DCFT/experiments" \
     --dry_run
   ```
6. Once sbatch scripts look correct, drop `--dry_run` to submit real jobs. If your cluster needs special handling (login vs compute nodes, proxies, etc.), add it to `hpc/hpc.py` and, if necessary, `hpc/launch.py` (for example, see the existing logic for JURECA/JUWELS internet nodes).

#### Learn More about HPC Launch

To learn more about the details of how HPC Launch works, please refer to `hpc/README.md`.

### Notes on Container Management

OT-Agent relies on [Harbor](https://github.com/laude-institute/harbor) to launch containerized tools for datagen and eval. Harbor supports multiple backends (Docker, Daytona, Modal, e2b, etc.), but most HPC centers either forbid Docker outright or only allow Apptainer/Singularity. In practice this means:

- **Remote container providers are the default.** We run large-scale datagen on managed platforms (Daytona, Modal, e2b) where Docker is available and network egress is unrestricted. Daytona is our primary provider; their infrastructure has handled millions of launches/day for OT-Agent workloads.
- **HPC clusters stay lean.** Login/compute nodes focus on scheduling, storage, and GPU time. When those jobs need containerized helpers (e.g., Harbor trace agents), they call out to the remote provider rather than trying to build Docker images locally.
- **Configuration lives in Harbor YAMLs.** Pick a template under `hpc/harbor_yaml/`, set the `type` field to your provider (e.g., `daytona`, `modal`, `modal-ray`), and make sure any required secrets/API keys are present in your runtime env (`DC_AGENT_SECRET_ENV` is sourced automatically).
- **Bring-your-own backend is fine.** Any Harbor-compatible provider works as long as its CLI/API is reachable from the cluster you launch jobs on. If you need Podman or another backend we don’t support yet, open an issue/PR—Harbor makes it straightforward to add.

Once the Harbor YAML points at the right backend and credentials, OT-Agent’s launch scripts will provision containers, stream logs, and tear everything down automatically.

### Who to contact if you get stuck

Please reach out to someone on the [terminal-bench Discord](https://discord.gg/6xWPKhGDbA) if you need help.

* For RL: Please contact Charlie Ruan
* For SFT: Please contact Benjamin Feuer
* For Data: Please contact Etash Guha
* For Eval: Please contact Negin Raoof
* For Project Management (includes cluster and account access): Please Contact Ryan Marten

## OT-Agent is Built On

[Llama Factory](https://github.com/hiyouga/LLaMA-Factory)

[SkyRL](https://github.com/NovaSky-AI/SkyRL)

[vLLM](https://github.com/vllm-project/vllm)

[Harbor](https://github.com/laude-institute/harbor)

## Friends of OT-Agent

[![Daytona Startup Grid](https://img.shields.io/badge/SPONSORED%20BY-DAYTONA%20STARTUP%20GRID-2ECC71?style=for-the-badge)](https://daytona.io/startups?utm_source=datacomp.ai)

[Laude Institute](https://www.laude.org/)

[Bespoke Labs](https://www.bespokelabs.ai/)

[Oumi](https://oumi.ai/)

## Citation

Coming Soon
