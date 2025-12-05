# Data Generation

The `ot-agent/data` directory houses every data pipeline used by ot-agent, with each subdirectory representing a unique, named data pipeline. Data pipeline directories contain one or more generation strategies, which turn some upstream data source into:
- **tasks** – runnable [Harbor](https://github.com/laude-institute/harbor) tasks that agents train on
- **traces** – execution logs produced by running agents over those tasks, typically from the default Terminus-2 agent in Harbor

using some documented, reproducible strategy.

Two complementary abstractions are used to keep data pipelines reproducible and maintainable.

## Strategy 1 – Declarative scripts (`generate.py`)

All data pipelines should ship a single `generate.py` script that completely specifies how to generate tasks and traces. These scripts should be simple and self-contained, so that rerunning them is as simple as executing the file directly.

### Characteristics
- Pure Python module with a `main()` function that orchestrates the run.
- Pulls in shared helpers from `data/commons.py` for Hugging Face uploads, task directory management, templated question creation, etc.
- Encodes every parameter needed to reproduce the run (source dataset identifiers, sampling logic, upload destinations, agent configuration for traces) directly in the script.
- Ideal for local iteration or one-off data drops where the whole pipeline fits on a single machine.

### Running
```bash
cd ot-agent
python data/<dataset>/generate.py --optional-flags
```

The exact CLI surface is script-specific, but common patterns include:
- Loading a raw dataset (e.g., Hugging Face `load_dataset`)
- Converting records into questions/instructions (`generate_tasks_from_questions`)
- Down-sampling or shuffling (`subsample_tasks_directory`)
- Optionally invoking trace generation inline with `run_dataset_to_traces`
- Uploading results (`upload_tasks_to_hf`, `upload_traces_to_hf`)

Because everything is captured in the file, version controlling the script is enough to reproduce the dataset later.

## Strategy 2 – Class-based generators (`generate_abstract.py`)

For larger runs, especially those launched on shared HPC hardware, or where we want to vary the base model or quantity of traces generated via the command line, it is possible to instead use `generate_abstract.py`, the class-based abstraction exported from `data/generation`. Each `generate_abstract.py` defines a subclass of `BaseDataGenerator` that describes how to build tasks (and optionally traces). The runtime wiring, CLI parsing, engine bootstrapping, and trace orchestration are provided by the shared framework.

### Lifecycle
1. `BaseDataGenerator` constructs a unified `argparse` CLI (see `data/generation/utils.py`).
2. Parsed arguments are normalised into a `GenerationRequest`, and an `InferenceEngine` is created if the stage requires one.
3. `run_task_generation` (implemented by the subclass) returns a `GenerationResult`. `BaseDataGenerator` can automatically limit tasks (`limit_tasks_directory`) and forward the output to trace generation.
4. `run_trace_generation` (in `BaseDataGenerator`) handles Daytona trace jobs, exporting and uploading traces when `--stage traces` or `--stage both` is requested.

Subclasses can override hooks such as:
- `add_arguments` – extend the CLI with dataset-specific options.
- `_build_request_metadata` – record custom metadata for downstream logging.
- `run_trace_generation` – replace the default trace flow if needed.

Decorate `run_task_generation` with `@gpu_required` when the task stage must start only after GPU-backed inference engines are ready.

### Running via HPC launcher
These generators are typically invoked through `hpc.launch` so that sbatch templates, VLLM endpoints, and environment activation happen automatically. From the repository root:

```bash
python -m hpc.launch \
  --job_type datagen \
  --datagen_script data/<dataset>/generate_abstract.py \
  --datagen_target_repo <org/repo> \
  --datagen_extra_args "--stage both --limit 2000" \
  --datagen_engine vllm_local \
  --experiments_dir $DCFT/experiments
```

`--datagen_extra_args` is passed verbatim to the generator, so any options added in `add_arguments` (for example, sampling bounds or input dataset handles) are available. When running locally you can call the script directly: `python data/<dataset>/generate_abstract.py --stage tasks`.

### When to choose this strategy
- You need the reusable CLI surface provided by `BaseDataGenerator`.
- The pipeline will run under `hpc.launch` with managed VLLM endpoints or Daytona traces.
- Multiple datasets share trace infrastructure and benefit from the consistent `GenerationResult` metadata.

For more information on the flags in HPC Launch, please refer to `hpc/README.md`

## Shared abstractions in `data/generation`

The `data/generation` package centralises the reusable building blocks used by every class-based generator:

- `base.py` – Defines `BaseDataGenerator`, orchestrating CLI parsing, engine setup, stage dispatch (`tasks`, `traces`, `both`), limit enforcement, and default trace generation via sandboxes.
- `schemas.py` – Dataclasses (`GenerationRequest`, `GenerationContext`, `GenerationResult`, `GenerationStatus`, etc.) and typed errors (`GenerationError`, `InputValidationError`, `InferenceFailure`, `TraceFailure`, `UploadFailure`) used across the pipeline.
- `engines.py` – Abstractions for inference backends (`InferenceEngine`) plus concrete implementations for OpenAI, Anthropic, OpenAI-compatible/vLLM endpoints, and a `NoOpEngine`. Includes health-check logic for long-running jobs.
- `utils.py` – Utility functions that extend CLIs with standard flags (`--engine`, `--model`, `--stage`, `--trace-*`) and create engines from parsed arguments.
- `decorators.py` – `@gpu_required` marker to signal that `run_task_generation` cannot proceed until a GPU-capable backend is healthy.
- `limits.py` – Helpers that copy a limited number of task or trace directories into a temporary workspace so downstream steps can operate on a capped sample.
- `task_generation.py` – Concurrency helpers for Daytona-backed generation, including `DaytonaTaskGenerationExecutor`, `TaskGenerationSpec`, and `TaskGenerationResult`, making it easier to schedule many seeds with bounded parallelism.
- `__init__.py` – Re-exports the public surface so generators can import everything via `from data.generation import ...`.

Understanding these modules is key when building new datasets or extending existing ones—all class-based generators rely on them.

## Choosing an approach
- **Use `generate.py`** when the pipeline is simple, you want a single entry point that anyone can run locally, or you are prototyping without HPC infrastructure.
- **Use `generate_abstract.py`** when the pipeline needs the shared datagen CLI, automatic trace support, or has to run under the `hpc.launch` workflow.

Many datasets maintain both: a lightweight `generate.py` for quick local reproductions and a `generate_abstract.py` for the production HPC run that writes the authoritative artifacts.

## Scripts

### `scripts/datagen/launch_trace_from_parquet.py`
- Convenience entry point for trace-only jobs that start from a Parquet snapshot of tasks.
- Workflow: downloads a parquet dataset from Hugging Face (`--tasks_repo` / `--tasks_revision`), extracts tasks to a local directory (default `<experiments_dir>/tasks_extracted`), then launches `python -m hpc.launch` with trace generation enabled and task generation disabled.
- Select a specific parquet file from the downloaded snapshot with `--parquet_name`; otherwise the first `*.parquet` discovered is used.
- Supports dry runs (`--dry_run`) and safe re-extraction via `--overwrite`.
- Passes through vLLM, sandbox, and trace configuration flags so the launched job matches your desired hardware footprint.

#### Example
```bash
python scripts/datagen/launch_trace_from_parquet.py \
  --experiments_dir $DCFT/experiments \
  --trace_target_repo org/traces-repo \
  --vllm_model_path meta-llama/Llama-3-70b-instruct \
  --tasks_repo DCAgent/nl2bash \
  --parquet_name tasks/train-00000-of-00001.parquet
```

## Adding a new dataset
1. Start with the declarative version: create `data/<dataset>/generate.py` that uses `data/commons.py` to build tasks and (optionally) traces.
2. When you need to move the pipeline to HPC:
   - Create `generate_abstract.py`, subclassing `BaseDataGenerator`.
   - Implement `run_task_generation` (and override `add_arguments` or `run_trace_generation` if necessary).
   - Point `hpc.launch` at the new script via `--datagen_script`.
   - Document any required environment variables or additional assets in the module docstring.
