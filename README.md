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

1. **Terminal-Bench smoke tests (local or Daytona).** `eval/example_tbench.py` wraps the `terminal_bench` CLI so you can point Harbor at a hosted vLLM endpoint. Start (or SSH-tunnel to) a VLLM server first (`python eval/start_ray.py` spins up a Ray Serve-backed OpenAI-compatible endpoint bound to `127.0.0.1:8000`). Then run:
   ```bash
   python eval/example_tbench.py \
     # tweak dataset_name/version, backend, agent, model_name, agent_kwargs, n_concurrent_trials
   ```
   This creates a run with Daytona sandboxes and prints the aggregated score. Use it to verify that your model + Harbor wiring works before touching HPC.

2. **HF RL task batches.** `eval/example_rltasks_eval.py` downloads a HuggingFace dataset, optionally slices it, and launches Harbor with your preferred agent + model:
```bash
python eval/example_rltasks_eval.py \
  --model openai/gpt-5-chat-latest \
  --dataset DCAgent/nl2bash \
  --agent terminus-2 \
  --n-tasks 100 \
  --n-concurrent 8 \
  --n-attempts 3 \
  --env daytona
```
The script converts parquet task dumps via `scripts/sandboxes/tasks_parquet_converter.py`, stages results under `./jobs`, and accepts `--job-name`/`--output` so you can checkpoint multiple experimental runs.

1. **Cluster-scale Harbor eval (TACC template).** The `eval/tacc/` subtree packages the scripts we use on TACC GH nodes:
   - Drop database + HF credentials into `eval/tacc/secret.env` (see `eval/tacc/README.md` for the expected keys) and make sure the shared conda env in that README is accessible from your account.
   - Submit `sbatch eval/tacc/tacc_eval_harbor.sbatch <hf_model_id> <hf_dataset_repo>`; both arguments default to the canonical OT-Agent eval model/dataset if omitted.
   - The sbatch file loads CUDA/GCC modules, boots a VLLM server, hydrates datasets via `snapshot_download.py`, and then calls `harbor jobs start` with the concurrency specified in line 109 (set `--n-concurrent` here to scale up/down). Behavior is controlled by `eval/tacc/dcagent_eval_config.yaml`.
   - Logs land in `experiments/logs/` and full Harbor outputs go under `jobs/<RUN_TAG>/`. `eval/tacc/README.md` documents troubleshooting tips plus alternate sbatch variants (`tacc_eval_multi_gpu.sbatch`, `tb2_eval_harbor.sbatch`, SweBench listeners, etc.).

2. **Other clusters/benchmarks.** The remaining files in `eval/` are copyable templates:
   - `run_tb_wx.sbatch` and the `run_tb_vllm_wx_qwen3_t*.sbatch` scripts schedule Terminal-Bench runs with hosted vLLM models.
   - `JSC/launch_tbench.sh` shows how to adapt the flow to JSC (interactive SLURM allocation + reverse tunnel + Apptainer).
   Reuse these when porting eval to a new site—each script labels the environment variables and SLURM resources you will likely need to swap out.

3. **Launcher-driven evals.** Prefer `python -m hpc.launch --job_type eval ...` whenever you want the same CLI ergonomics as SFT/datagen jobs. Key flags:
   - `--datagen_config hpc/datagen_yaml/<model>.yaml` (required). This is the same engine file you would pass to `--job_type datagen` and determines how the vLLM/Ray controller boots. `--trace_model` optionally overrides both `engine.model` and `vllm_server.model_path` so a single YAML can serve multiple finetunes.
   - `--trace_harbor_config hpc/harbor_yaml/<cluster>_eval_*.yaml` (filename must include `_eval_`)
   - Either `--trace_input_path /path/to/tasks` *or* `--harbor_dataset terminal-bench@2.0` (mutually exclusive)
   - `--eval_benchmark_repo <org/dataset>` so Supabase rows can track the benchmark
   - Agent knobs now default to the Harbor YAML: `--trace_agent_name`, `--trace_env`, and `--trace_n_concurrent` are optional overrides when you need to deviate.
   - `--trace_agent_kwargs '{"temperature":0.2,"max_tokens":2048}'` overlays JSON on top of the Harbor + datagen defaults. When the datagen config uses `vllm_local`, the launcher injects the computed `api_base`/`metrics_endpoint` from `vllm_endpoint.json` automatically unless you explicitly provide them.

   When a `vllm_local` engine is selected, the eval launcher reuses the datagen hosting flow: it spins up the Ray/vLLM server, waits for the generated endpoint JSON to pass health checks, feeds the derived OpenAI-compatible URL to Harbor, and tears the server down once the eval finishes. No ad-hoc `--trace_eval_only` hacks are needed—the vLLM bootstrap, Supabase bookkeeping, and HF uploads all run in one job.

   Example (local dataset path):
   ```bash
   python -m hpc.launch \
     --job_type eval \
     --job_name tacc-qwen2-eval \
     --datagen_config hpc/datagen_yaml/qwen3_coder_30b_a3b_vllm_serve.yaml \
     --trace_harbor_config hpc/harbor_yaml/tacc_eval_daytona_eval.yaml \
     --trace_input_path $SCRATCH/dev_set_71_tasks \
     --eval_benchmark_repo mlfoundations-dev/dev_set_71_tasks \
     --trace_model hosted_vllm/qwen2.5-7b-instruct \
     --trace_agent_name terminus-2 \
     --trace_agent_kwargs '{"api_base":"http://127.0.0.1:8000/v1","key":"fake_key"}' \
     --trace_n_concurrent 128
   ```

   Example (Harbor registry dataset slug):
   ```bash
   python -m hpc.launch \
     --job_type eval \
     --job_name tb2-claude-eval \
     --datagen_config hpc/datagen_yaml/qwen3_coder_30b_a3b_vllm_serve.yaml \
     --trace_harbor_config hpc/harbor_yaml/tacc_eval_daytona_eval.yaml \
     --harbor_dataset terminal-bench@2.0 \
     --eval_benchmark_repo DCAgent/dev_set_71_tasks \
     --trace_model anthropic/claude-opus-4-1 \
     --trace_agent_name claude-code \
     --trace_env daytona \
     --trace_n_concurrent 64
   ```

   Valid `--harbor_dataset` slugs come from the upstream registry (`../harbor/registry.json`). Popular options include `terminal-bench@2.0`, `terminal-bench-pro@head`, and `hello-world@head`; run `harbor datasets list` for the rest. The launcher validates slugs against the registry when it is available locally.

Regardless of the path you choose, make sure `DC_AGENT_SECRET_ENV` or the cluster-specific `secret.env` is exported so Harbor can read HF, Daytona, and database credentials. Use `--dry-run`/small `--n-concurrent` first to validate that Harbor, the sandbox provider, and your model endpoint all respond as expected.

##### Eval Lifecycle & Result Uploads

- **Job bookkeeping starts before Harbor launches.** The sbatch drivers (for example `eval/tacc/tacc_eval_harbor.sbatch`) call `database/unified_db/utils.py:create_job_entry_started` to write a Supabase `sandbox_jobs` row with `job_status="Started"`, the intended `n_trials` (Harbor tasks) and `n_rep_eval` (copied from `config.n_attempts`, default 3). This dedupes model/benchmark pairs and records agent/model provenance before GPUs are consumed.
- **Runs must finish cleanly before we touch the database.** After Harbor exits, the sbatch script inspects `jobs/<RUN_TAG>/result.json` and bails if too many Daytona errors show up (TACC skips the upload if there are >3). Only when the run directory exists and passes that gate do we proceed.
- **Averaged metrics come straight from Harbor.** `_extract_job_metadata()` (see `database/unified_db/utils.py`) parses `stats.evals.*.metrics` inside `result.json`, grabs the `mean` values that Harbor already computed across repeated attempts, and stores them with `n_total_trials`. We never recompute scores—what Harbor reports is what lands in Supabase.
- **Trial completeness is enforced.** `_extract_trial_metadata()` refuses to register a trial unless both `agent_execution` and `verifier_result` blocks are present, so partially executed tasks never pollute the leaderboard averages. Harbor’s retry knobs (`--n-attempts`, `n_rep_eval`) ensure enough fully verified trials exist to compute the mean.
- **Uploads are two-phased.** `upload_eval_results()` first pushes traces to HuggingFace (if `hf_repo_id` is supplied) and then updates the pre-created job row with `job_status="Finished"`, metrics, stats, and the HF dataset URL via `upload_job_and_trial_records()`. Trial + usage rows are inserted in the same pass, so Supabase has both the aggregate score and every per-task attempt. `error_mode` controls whether a failed HF upload rolls everything back or simply marks the job with warnings.
- **Listeners keep things consistent.** Automations like `eval/tacc/tb2_eval_listener.py` poll Supabase for recent models, detect stale `sandbox_jobs` stuck in `Started`, and only submit new sbatch runs when a model/benchmark pair still needs coverage. That feedback loop guarantees the averages you see on the leaderboard come from completed Harbor jobs with trace artifacts uploaded.

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
@misc{openthoughts-agent,
  author = {Team, OpenThoughts-Agent},
  month = Dec,
  title = {{OpenThoughts-Agent}},
  howpublished = {https://open-thoughts.ai/agent},
  year = {2025}
}
```
