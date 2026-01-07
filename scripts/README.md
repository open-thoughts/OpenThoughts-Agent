# Scripts

Utility entrypoints that support data generation, trace analysis, Harbor uploads, Daytona orchestration, and benchmarking live in this directory. Run commands from the `OpenThoughts-Agent/` root so relative imports resolve correctly, and append `--help` to any script for full CLI details.

## Directory overview
- `analysis/` – post-processing helpers for trace JSONL files and eval runtimes.
- `consolidate/` – DeepSpeed tooling (e.g., converting ZeRO checkpoints to fp32).
- `database/` – Supabase + Hugging Face registration utilities.
- `datagen/` – helpers for launching and inspecting data-generation jobs.
- `daytona/` – sandbox management, validation, and Supabase queries for Daytona runs.
- `docker_ray/`, `ray/`, `vllm/` – Ray/vLLM cluster bring-up scripts.
- `harbor/` – adapters that convert Harbor jobs/tasks into upload-ready datasets.
- `terminal_bench/` – wrappers for running and cleaning up terminal-bench evaluations.

## Commonly used scripts

### Analysis & reporting
- `analysis/filter_latest_episodes.py` – reduce a HF dataset (or local Harbor job) to the newest episode per task.  
  Example:  
  ```bash
  python scripts/analysis/filter_latest_episodes.py DCAgent/dev_set --split train --output-jsonl outputs/dev.jsonl
  ```
- `analysis/summarize_conversations.py` – tokenize/summarize the JSONL produced above, reporting average turns, tokens, rewards, and marker counts.  
  ```bash
  python scripts/analysis/summarize_conversations.py outputs/dev.jsonl --tokenizer-id Qwen/Qwen2.5-7B
  ```
- `analysis/batch_filter_and_summarize.py` – iterate many Harbor job dirs, emit filtered JSONLs, then summarize each.  
  ```bash
  python scripts/analysis/batch_filter_and_summarize.py --root /scratch/jobs --out_dir ~/eval-jsonl --skip_existing
  ```
- `analysis/eval_runtime_stats.py` & `analysis/trace_runtime_report.py` – crawl `evaltraces/`, compute stage runtimes, generate JSON summaries, and optionally render plots.  
  ```bash
  python scripts/analysis/trace_runtime_report.py --root ~/evaltraces --output-json ~/evaltraces/summary.json
  ```
- `analysis/episode_distribution.py` – download one or more Hugging Face trace datasets, normalize their `episode` labels to integers, and plot smoothed per-episode counts to compare run lengths.  
  ```bash
  python scripts/analysis/episode_distribution.py my-org/datasetA my-org/datasetB --sigma 3.0 --output plots/episodes.png
  ```

### Data generation helpers
- `datagen/gsm8k_terminal_bench_traces.py` – BaseDataGenerator entrypoint for GSM8K Terminal Bench traces; reruns the standard datagen CLI with dataset-specific flags.  
  ```bash
  python scripts/datagen/gsm8k_terminal_bench_traces.py --tasks-repo mlfoundations-dev/gsm8k-terminal-bench --output-dir /tmp/gsm8k-traces
  ```
- `datagen/extract_tasks_from_parquet.py` – resolve a local parquet file (or Hugging Face dataset repo), then materialize Harbor-style task folders for downstream trace jobs.  
  ```bash
  python scripts/datagen/extract_tasks_from_parquet.py \
    --parquet my-org/tasks-parquet \
    --output_dir $SCRATCH/experiments/tasks_extracted \
    --parquet_name tasks/train-00000-of-00001.parquet \
    --on_exist overwrite
  ```
- `datagen/print_trace_contents.py` – quickly preview the conversations inside exported trace JSONL files.  
  ```bash
  python scripts/datagen/print_trace_contents.py trace_jobs/chunk_000/2024-11-01__12-00-00
  ```

### Harbor dataset uploaders
- `harbor/make_and_upload_task_dataset.py` – convert a directory of Harbor tasks into a Parquet snapshot and push it to Hugging Face.  
  ```bash
  python scripts/harbor/make_and_upload_task_dataset.py \
    --repo_id my-org/my-tasks \
    --tasks_dir data/tasks_to_upload \
    --private
  ```
- `harbor/make_and_upload_trace_dataset.py` – take a completed Harbor job directory, export traces, and upload them as a dataset repo.  
  ```bash
  python scripts/harbor/make_and_upload_trace_dataset.py \
    --job_dir /scratch/jobs/codecontests_glm46 \
    --repo_id my-org/codecontests-glm46-traces \
    --episodes last
  ```
- `harbor/run_and_export_traces.py` – programmatic helper that loads a Harbor job config, runs it in-process, and returns a Hugging Face `Dataset` (import and call from Python, or build your own wrapper).

### Daytona & Supabase tooling
- `daytona/inspect_daytona_data.py` – build a sandbox from a local task and dump the staged files to inspect what the orchestrator uploads.  
  ```bash
  python scripts/daytona/inspect_daytona_data.py --dockerfile path/to/task/environment/Dockerfile
  ```
- `daytona/validate_and_upload_from_hf.py` – download sandbox tasks from HF, validate they build/run via Daytona + Harbor, then push only successful tasks to a new dataset.  
  ```bash
  python scripts/daytona/validate_and_upload_from_hf.py \
    --repo_id my-org/raw-tasks \
    --extract_dir ./tmp/tasks \
    --target_repo my-org/validated-tasks \
    --timeout 900
  ```
- `daytona/search_sandbox_jobs.py` – query the Supabase `sandbox_jobs` table with include/exclude filters and dump metrics to CSV.  
  ```bash
  python scripts/daytona/search_sandbox_jobs.py --include terminus --include vista --output vista_runs.csv
  ```
- `database/manual_db_push.py` – register a trained model with Supabase after a run; edit the constants at the top, source your `hpc/dotenv/*.env`, then run:  
  ```bash
  python scripts/database/manual_db_push.py
  ```
- `huggingface/list_public_models.py` and `huggingface/reset_hf_repo.py` provide quick HF org utilities (list models, wipe repo contents).

### Ray / vLLM / Terminal Bench
- `docker_ray/start_ray_cluster.py` – spin up a Ray Serve deployment backed by vLLM directly from your workstation (handy for local testing).  
  ```bash
  python scripts/docker_ray/start_ray_cluster.py --model meta-llama/Llama-3.1-8B-Instruct --min-replicas 1 --max-replicas 2 --tensor-parallel-size 1
  ```
- `ray/wait_for_cluster.py` – block until a Ray head reports the desired nodes/GPUs (used by HPC sbatch templates).  
  ```bash
  python scripts/ray/wait_for_cluster.py --address ${RAY_ADDRESS} --expected-gpus 8 --expected-nodes 2 --timeout 900
  ```
- `vllm/start_vllm_ray_controller.py` – bring up a vLLM OpenAI-compatible endpoint on top of Ray; pair with `vllm/wait_for_endpoint.py` to poll readiness.  
  - Note: on some SLURM clusters, Ray's default `CUDA_VISIBLE_DEVICES` rewriting can cause vLLM to crash with `CUDA error: invalid device ordinal`; `start_vllm_ray_controller.py` sets `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` by default (override by exporting it yourself).  
  ```bash
  python scripts/vllm/start_vllm_ray_controller.py --model /checkpoint/qwen --ray-address auto --tensor-parallel-size 2
  ```
- `terminal_bench/run_terminal_bench.py` – convenience wrapper for launching terminal-bench evaluations via `uv run`, pointing at a Ray-hosted vLLM endpoint.  
  ```bash
  python scripts/terminal_bench/run_terminal_bench.py \
    --llm-name glm46-vllm \
    --ray-endpoint http://127.0.0.1:9000 \
    --dataset-name terminal_bench_dev \
    --dataset-version v0.2 \
    --max-replicas 8
  ```
- `terminal_bench/tbench_cleanup.sh` – kill lingering terminal-bench, Ray, and Docker artifacts after aborting a run.  
  ```bash
  bash scripts/terminal_bench/tbench_cleanup.sh
  ```

### Model checkpoint utilities
- `consolidate/zero_to_fp32.py` – convert DeepSpeed ZeRO stage checkpoints into a single fp32 weight file. Invoke from the checkpoint directory:  
  ```bash
  python scripts/consolidate/zero_to_fp32.py . output_fp32/ --safe_serialization
  ```

These examples cover the tasks we reach for most often; inspect each script (or run with `--help`) for the full set of switches, expected environment variables, and pre-requisites (HF tokens, Supabase keys, Daytona credentials, etc.).
