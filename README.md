<p align="center">
  <img src="assets/ot-agent-logo.png" alt="OT-Agent Logo" width="480">
</p>

# OpenThoughts-Agent: Data Recipes for Agentic Models

Welcome to OpenThoughts-Agent (OT-Agent for short), a large-scale research project dedicated to creating the best tooling and finding the best data for training small agentic models.

## Links

[Project Website](https://www.openthoughts.ai/)

[Leaderboard](https://ot-agent-leaderboard.replit.app/)

[Trace Viewer](https://ot-agent-trace-viewer.replit.app/)

[Notebook](/notebook/datagen_sft_tutorial.ipynb)

## Warning!

OT-Agent is a research codebase! Conventions will change, files will move and workflows will break as we continue to grow. Please bear with us and open an issue if you discover a bug.

## Getting Started

If you are new to the project, start here to get up and running.

### Installation

1. Create a clean Python 3.12 virtual environment using your preferred manager (conda, venv, pixi, etc.).
2. Install [uv](https://github.com/astral-sh/uv) if you don’t already have it:
   ```bash
   pip install --upgrade uv
   ```
3. From the repo root, install the base OT-Agent stack:
   ```bash
   uv pip install -e .
   ```

Optional extras (append the extra names to the command above, e.g. `uv pip install -e ".[datagen]"`):

* **HPC datagen runtime** (Ray clusters + vLLM serving):
  `.[datagen]` pulls CUDA-heavy wheels on Linux/Windows and automatically falls back to CPU-friendly packages on macOS.
* **SweSmith-specific datagen helpers** (extends the above with bespoke tools):
  `.[datagen,datagen-swesmith]` (or `.[datagen-swesmith]` if you already pulled the base datagen extra)
* **Cloud orchestration helpers** (SkyPilot, Docker tooling):
  `.[cloud]`

**Fresh Ubuntu/GCP quickstart**

```bash
sudo snap install astral-uv --classic     # installs uv system-wide
uv venv --python 3.12                     # creates .venv with CPython 3.12
source .venv/bin/activate
uv pip install -e ".[datagen]"            # add ,cloud or other extras as needed
```

* **SFT stack**:  
* Install the new `[sft]` extra to auto-sync the submodule and pull its heavy dependencies in one go (runs `git submodule update --init --remote sft/llamafactory` unless you set `OT_AGENT_SKIP_SFT_SYNC=1`):
    ```bash
    uv pip install -e "[sft]"                              # only SFT runtime
    uv pip install -e "[datagen,sft]"                      # convenient combined env
    uv pip install -e "[datagen,sft,cloud]"                # pull whatever extras you need
    ```
  * Under the hood we install LLaMA-Factory from `sft/llamafactory` with the `hf-kernels,liger-kernel,deepspeed,bitsandbytes` extras. If you need additional LLaMA-Factory extras, continue to `cd sft/llamafactory && uv pip install -e .[...more...]`.
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

OT-Agent's job launchers are designed to work with HPC (high-performance computing) clusters. Different launchers exist for different job types. OT-Agent's launchers are modular, making it relatively straightforward to add your own preferred cluster. Every invocation of `python -m hpc.launch` must explicitly set `--job_type`; use `sft` for standard finetuning jobs, `sft_mca` when you need the Megatron Core Adapter sbatch templates, `pretokenize` for tokenization-only runs, `datagen` for generator/trace work, and `consolidate` for ZeRO merges.

#### How to Launch a Datagen Job

Datagen jobs are launched via the generic HPC launcher and use `--job_type datagen` plus a generator script.

### Local Eval Runner (`eval/local/run_eval.py`)

Need to verify a Harbor eval locally before burning queue time? `eval/local/run_eval.py` spins up a single-node Ray cluster, launches a vLLM controller, and executes a Harbor job against the newly created endpoint.

Prereqs:

* Python environment with the base install plus the `datagen` extra (Ray + vLLM).
* CUDA GPUs (or Apple Silicon if you just need a dry run) and the expected driver stack.
* Harbor/Daytona/ Supabase credentials exposed via the usual `DC_AGENT_SECRET_ENV`.

Example:

```bash
python eval/local/run_eval.py \
  --datagen_config hpc/datagen_yaml/qwen3_coder_30b_a3b_vllm_serve_131k_1xH200.yaml \
  --harbor_config hpc/harbor_yaml/trace_16concurrency_eval_ctx131k.yaml \
  --dataset terminal-bench@2.0 \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --agent terminus-2 \
  --gpus 1 \
  --eval_benchmark_repo DCAgent2/Qwen3Coder30B-terminus2-terminal-bench-2.0-test-lambda
```

What happens:

1. Datagen defaults (tensor parallelism, vLLM overrides, etc.) are pulled from `--datagen_config`.
2. Ray and vLLM are booted locally; endpoint metadata lands in `eval_runs/vllm_endpoint.json`.
3. The script builds the Harbor command, injects required agent kwargs, and runs it while streaming output to `eval_runs/logs/harbor.log`.
4. Traces, logs, and Harbor artifacts accumulate under `eval_runs/` (change via `--experiments_dir`).

Handy flags:

* `--agent_kwarg foo=bar` (repeatable) to forward Harbor agent settings.
* `--harbor_extra_arg ...` for advanced Harbor CLI knobs (e.g., filtering datasets).
* `--harbor_env daytona|docker|modal` to select the Harbor sandbox backend (default: `daytona`).
* `--harbor_log /tmp/harbor.log` to redirect the live Harbor TUI.
* `--dry_run` to validate configs without launching Ray/Harbor.

Once your eval behaves locally, promote the same Harbor YAML/datagen config to your HPC launcher or the cloud wrappers (see **Cloud Launchers** below).

### Cloud Launchers

OT-Agent provides SkyPilot-based cloud launchers for running trace generation and eval jobs on cloud VMs (GCP, AWS, Lambda, Kubernetes, etc.) without SLURM. These live under `data/cloud/` and `eval/cloud/`.

#### Cloud Trace Generation (`data/cloud/launch_tracegen_cloud.py`)

Launches `data/local/run_tracegen.py` on a cloud GPU node:

```bash
python data/cloud/launch_tracegen_cloud.py \
  --harbor_config hpc/harbor_yaml/trace_16concurrency_ctx131k.yaml \
  --datagen_config hpc/datagen_yaml/gpt_oss_120b_vllm_serve_131k_1xH200.yaml \
  --tasks_input_path DCAgent/stackexchange-tor-sandboxes \
  --model openai/gpt-oss-120b \
  --secrets_env /path/to/secrets.env \
  --accelerator "H100:1" \
  --cloud_provider gcp \
  --region us-central1
```

#### Cloud Eval (`eval/cloud/launch_eval_cloud.py`)

Launches `eval/local/run_eval.py` on a cloud GPU node:

```bash
python eval/cloud/launch_eval_cloud.py \
  --harbor_config hpc/harbor_yaml/trace_16concurrency_eval_ctx131k.yaml \
  --datagen_config hpc/datagen_yaml/qwen3_coder_30b_a3b_vllm_serve_131k_1xH200.yaml \
  --dataset terminal-bench@2.0 \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --agent terminus-2 \
  --secrets_env /path/to/secrets.env \
  --accelerator "H100:1" \
  --cloud_provider gcp
```

#### Common Cloud Flags

* `--cloud_provider` - Cloud backend: `gcp`, `aws`, `lambda`, `kubernetes`, etc. (comma-separated for fallbacks)
* `--accelerator` - GPU spec, e.g., `H100:1`, `A100:4` (comma-separated for fallbacks)
* `--region` - Preferred region(s)
* `--harbor_env` - Harbor environment backend: `daytona` (default), `docker`, or `modal`
  * **Note**: Use `daytona` (default) for cloud VMs. The `docker` backend requires Docker-in-Docker which is not available in SkyPilot's container runtime. The `docker` and `modal` backends are primarily for local development.
* `--secrets_env` - Path to secrets file sourced inside the container
* `--autostop N` - Auto-stop cluster after N minutes idle (set to 0 for Kubernetes)
* `--retry_until_up` - Keep retrying until resources are available (useful for scarce GPUs)
* `--down` - Tear down cluster after job completes
* `--list_providers` - Show available cloud providers and exit

Logs and outputs sync periodically to `--local_sync_dir` during execution.

1. Ensure your cluster environment is set up (dotenv, conda env, etc.). For TACC/Vista-style machines, follow the checklist in `hpc/README.md` and use `hpc/dotenv/tacc.env` as a starting point for your environment variables.
2. Activate your environment and source the dotenv:
```bash
source hpc/dotenv/<your-cluster>.env
eval "$DCFT_ACTIVATE_ENV"
cd "$DCFT"
```
The dotenvs now export `PYTHONPATH="${DCFT_PRIVATE:-$DCFT}:$PYTHONPATH"` so `python -m hpc.launch` resolves even on clusters that strip the working directory from `sys.path`. If you maintain a custom dotenv, mirror this line to keep the launcher importable.
1. Choose or write a datagen script under `data/...` implementing `BaseDataGenerator` (see `data/generation/base.py` and existing generators for examples).
2. Run the launcher from a login node:
```bash
python -m hpc.launch \
  --job_type datagen \
  --datagen_script data/<dataset>/generate_abstract.py \
  --datagen_config hpc/datagen_yaml/<model>_vllm_serve.yaml \
  --datagen_target_repo <org/dataset-tasks> \
  --enable_task_gen True \
  --experiments_dir "$DCFT/experiments" \
  --time_limit 12:00:00
```
1. To also generate traces, add:
   - `--enable_trace_gen True`
   - `--trace_target_repo <org/dataset-traces>`
   - `--trace_harbor_config path/to/harbor_job.yaml`
   and any of the `trace_*` overrides documented in `hpc/README.md`.

The launcher will synthesize and submit one or more `sbatch` scripts under `"$experiments_dir/sbatch_scripts"` and write configs to `"$experiments_dir/configs"`. Use `--dry_run` to inspect scripts without actually calling `sbatch`.

#### How to Launch an SFT Job

SFT jobs are also launched via `hpc.launch` with `--job_type sft` and a LLaMA Factory config.

1. Pull and install the SFT submodule (once per checkout) and install its dependencies in-place:
```bash
git submodule update --init --remote sft/llamafactory
cd sft/llamafactory
pip install -e .[train,liger-kernel,deepspeed]  # pick the extras you need
cd -
```
1. Configure your cluster dotenv and environment as in the Datagen section.
2. Pick a training config under `sft/lf_configs` or create your own YAML alongside the existing presets.
3. From a login node, run:
```bash
python -m hpc.launch \
 --job_type sft \
  --train_config_path sft/lf_configs/<path-to-config>.yaml \
  --dataset <org/dataset> \
  --num_nodes 8 \
  --time_limit 24:00:00 \
  --experiments_dir "$DCFT/experiments"
```
1. Optionally override LLaMA Factory flags via `--train_extra_args "..."` (see `hpc/README.md` and `sft/llamafactory/README.md` for full argument lists).

The launcher will construct a per-run YAML in `"$experiments_dir/configs"`, generate an sbatch script, and then submit the job. Training metadata and summaries are written into the run’s `output_dir`.

#### How to Launch an Eval Job

Everything you need to evaluate models lives under `eval/`. Pick the mode that fits how much infrastructure you have available:

1. **Terminal-Bench smoke tests (local or Daytona).** `eval/example_tbench.py` wraps the `terminal_bench` CLI so you can point Harbor at a hosted vLLM endpoint. Point it at an already-running (or SSH-tunneled) OpenAI-compatible vLLM server, then run:
   ```bash
   python eval/example_tbench.py \
     # tweak dataset_name/version, backend, agent, model_name, agent_kwargs, n_concurrent_trials
   ```
   This creates a run with Daytona sandboxes and prints the aggregated score. Use it to verify that your model + Harbor wiring works before touching HPC.

1. **Cluster-scale Harbor eval (unified listener).** The canonical surface is the root `eval/unified_eval_listener.py` driven by `--cluster-config eval/clusters/<cluster>.yaml`. It serves the model with vLLM inside SLURM, runs trials through Harbor + Daytona, and uploads to Supabase + HuggingFace. See [`docs/EVAL_GUIDE.md`](docs/EVAL_GUIDE.md) for the full fire templates, the five firing categories, the failure-mode catalog, and recovery procedures; [`eval/README.md`](eval/README.md) for the quickstart. To target a new cluster, copy `eval/clusters/example.yaml`.

2. **Launcher-driven evals.** The unified listener (§1 above) is also reachable through the HPC launcher: `python -m hpc.launch --job_type eval_listener ...` runs the `hpc.launch` preamble (auto-detects the cluster, sources `hpc/dotenv/<cluster>.env`, sets up Supabase + hosted-vLLM keys) and then forwards your flags verbatim to `eval/unified_eval_listener.py`. Use the SAME listener flags as §1 (`--cluster-config`, `--preset`, `--priority-file`, `--require-priority-list`, `--dry-run`, `--once`, …) — the launcher adds the dotenv preamble for you and otherwise stays out of the way. `--job_type eval` is kept as a deprecated alias for `--job_type eval_listener`.

   Example (ID-eval one-shot, dry-run):
   ```bash
   python -m hpc.launch --job_type eval_listener \
     --cluster-config eval/clusters/leonardo.yaml \
     --preset tb2 \
     --priority-file eval/lists/models_8b_tb2.txt \
     --require-priority-list \
     --dry-run --once
   ```

   The single-shot `--job_type eval` engine path (`--datagen_config` / `--trace_harbor_config` / `--harbor_dataset`) was removed — it was strictly subsumed by the listener and had no live consumers.

Regardless of the path you choose, make sure `DC_AGENT_SECRET_ENV` or the cluster-specific `secret.env` is exported so Harbor can read HF, Daytona, and database credentials. Use `--dry-run`/small `--n-concurrent` first to validate that Harbor, the sandbox provider, and your model endpoint all respond as expected.

##### Eval Lifecycle & Result Uploads

- **Job bookkeeping starts before Harbor launches.** The sbatch driver (`eval/jupiter/eval_harbor.sbatch`) calls `database/unified_db/utils.py:create_job_entry_started` to write a Supabase `sandbox_jobs` row with `job_status="Started"`, the intended `n_trials` (Harbor tasks) and `n_rep_eval` (copied from `config.n_attempts`, default 3). This dedupes model/benchmark pairs and records agent/model provenance before GPUs are consumed.
- **Runs must finish cleanly before we touch the database.** After Harbor exits, the sbatch script inspects `jobs/<RUN_TAG>/result.json` and bails if too many Daytona errors show up (TACC skips the upload if there are >3). Only when the run directory exists and passes that gate do we proceed.
- **Averaged metrics come straight from Harbor.** `_extract_job_metadata()` (see `database/unified_db/utils.py`) parses `stats.evals.*.metrics` inside `result.json`, grabs the `mean` values that Harbor already computed across repeated attempts, and stores them with `n_total_trials`. We never recompute scores—what Harbor reports is what lands in Supabase.
- **Trial completeness is enforced.** `_extract_trial_metadata()` refuses to register a trial unless both `agent_execution` and `verifier_result` blocks are present, so partially executed tasks never pollute the leaderboard averages. Harbor’s retry knobs (`--n-attempts`, `n_rep_eval`) ensure enough fully verified trials exist to compute the mean.
- **Uploads are two-phased.** `upload_eval_results()` first pushes traces to HuggingFace (if `hf_repo_id` is supplied) and then updates the pre-created job row with `job_status="Finished"`, metrics, stats, and the HF dataset URL via `upload_job_and_trial_records()`. Trial + usage rows are inserted in the same pass, so Supabase has both the aggregate score and every per-task attempt. `error_mode` controls whether a failed HF upload rolls everything back or simply marks the job with warnings.
- **Listeners keep things consistent.** The `eval/unified_eval_listener.py` daemon polls Supabase for recent models, detects stale `sandbox_jobs` stuck in `Started`, and only submit new sbatch runs when a model/benchmark pair still needs coverage. That feedback loop guarantees the averages you see on the leaderboard come from completed Harbor jobs with trace artifacts uploaded.

#### How to Launch an RL Job

Please check `rl/README.md`.

#### How to add your cluster to OT-Agent

Adding a new cluster involves defining its resources, sbatch templates, and a dotenv file so `hpc.launch` can target it.

1. **Create a dotenv for your cluster** under `hpc/dotenv/`, following `tacc.env` as a template. At a minimum, define:
   - `DCFT` (path to your OpenThoughts-Agent checkout on the cluster)
   - `DCFT_ACTIVATE_ENV` (command to activate the Python env)
   - paths for `EXPERIMENTS_DIR`, `DATASETS_DIR`, `MODELS_DIR`, and any cluster-specific SIF/Apptainer images.
2. **Register basic cluster metadata** by exporting `HPC_NAME` and related fields in your dotenv or by passing them on the CLI:
   - `--name`, `--account`, `--partition`, `--gpus_per_node`, `--cpus_per_node`, etc. (see `hpc/README.md` and `hpc/hpc.py`).
3. **Create sbatch templates** in `hpc/sbatch_data/` for your cluster:
   - Copy an existing template for a similar machine (GPU type / internet access) and adjust `#SBATCH` headers and module loads.
   - Keep placeholders like `{time_limit}`, `{job_name}`, `{experiments_dir}` etc. intact; they will be filled by `hpc.launch`.
4. **Test with a dry run**:
```bash
source hpc/dotenv/<your-cluster>.env
eval "$DCFT_ACTIVATE_ENV"
cd "$DCFT"
python -m hpc.launch \
  --job_type datagen \
  --datagen_script data/<dataset>/generate.py \
  --datagen_target_repo test-org/test-dataset \
  --experiments_dir "$DCFT/experiments" \
  --dry_run
```
1. Once sbatch scripts look correct, drop `--dry_run` to submit real jobs. If your cluster needs special handling (login vs compute nodes, proxies, etc.), add it to `hpc/hpc.py` and, if necessary, `hpc/launch.py` (for example, see the existing logic for JURECA/JUWELS internet nodes).

#### Learn More about HPC Launch

To learn more about the details of how HPC Launch works, please refer to `hpc/README.md`.

#### Tutorial Notebook

For non-HPC users, we provided a tutorial notebook under `notebook/datagen_sft_tutorial.ipynb` with an example of how we generate data from the inferredbugs dataset and perform SFT.

### Notes on Container Management

OT-Agent relies on [Harbor](https://github.com/laude-institute/harbor) to launch containerized tools for datagen and eval. Harbor supports multiple backends (Docker, Daytona, Modal, e2b, etc.), but most HPC centers either forbid Docker outright or only allow Apptainer/Singularity. In practice this means:

- **Remote container providers are the default.** We run large-scale datagen on managed platforms (Daytona, Modal, e2b) where Docker is available and network egress is unrestricted. Daytona is our primary provider; their infrastructure has handled millions of launches/day for OT-Agent workloads.
- **HPC clusters stay lean.** Login/compute nodes focus on scheduling, storage, and GPU time. When those jobs need containerized helpers (e.g., Harbor trace agents), they call out to the remote provider rather than trying to build Docker images locally.
- **Configuration lives in Harbor YAMLs.** Pick a template under `hpc/harbor_yaml/`, set the `type` field to your provider (e.g., `daytona`, `modal`, `modal-ray`), and make sure any required secrets/API keys are present in your runtime env (`DC_AGENT_SECRET_ENV` is sourced automatically).
- **Bring-your-own backend is fine.** Any Harbor-compatible provider works as long as its CLI/API is reachable from the cluster you launch jobs on. If you need Podman or another backend we don’t support yet, open an issue/PR—Harbor makes it straightforward to add.

Once the Harbor YAML points at the right backend and credentials, OT-Agent’s launch scripts will provision containers, stream logs, and tear everything down automatically.

### About Us

We are a collaboration led by researchers and engineers from Stanford, UC Berkeley, UT Austin, NYU, University of Washington, UCSD, ASU, CMU, UCLA, UNC Chapel Hill, TUM, LAION, and other partners focused on building the best datasets (and therefore the best models). See our previous work at <a href="https://www.datacomp.ai/">datacomp.ai</a> and <a href="https://github.com/mlfoundations">mlfoundations</a>.

We currently organize via the [terminal-bench Discord](https://discord.gg/6xWPKhGDbA); go there if you need help.

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

```
@misc{raoof2026openthoughtsagentdatarecipesagentic,
      title={{OpenThoughts-Agent: Data Recipes for Agentic Models}},
      author={Negin Raoof and Richard Zhuang and Marianna Nezhurina and Etash Guha and Atula Tejaswi and Ryan Marten and Charlie F. Ruan and Tyler Griggs and Alexander Glenn Shaw and Hritik Bansal and E. Kelly Buchanan and Artem Gazizov and Reinhard Heckel and Chinmay Hegde and Sankalp Jajee and Daanish Khazi and Emmanouil Koukoumidis and Xiangyi Li and Hange Liu and Shlok Natarajan and Harsh Raj and Nicholas Roberts and Ethan Shen and Nishad Singhi and Michael Siu and Ashima Suvarna and Hanwen Xing and Patrick Yubeaton and Robert Zhang and Leon Liangyu Chen and Xiaokun Chen and Steven Dillmann and Saadia Gabriel and Xunyi Jiang and Anurag Kashyap and Boxuan Li and Yein Park and Minh Pham and Sujay Sanghavi and Lin Shi and Ke Sun and Yixin Wang and Zhiwei Xu and Erica Zhang and Siyan Zhao and Wanjia Zhao and Jenia Jitsev and Alex Dimakis and Benjamin Feuer and Ludwig Schmidt},
      year={2026},
      eprint={2606.24855},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2606.24855}, 
}
```
