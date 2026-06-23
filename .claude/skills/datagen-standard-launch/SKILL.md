---
name: datagen-standard-launch
description: >-
  Launch NON-AGENTIC (standard) data generation — plain vLLM/API completion generation with NO Harbor agent
  loop or Daytona sandboxes. Two paths: (1) Curator sharded datagen — the multi-node data-parallel
  run_curator_datagen_sharded.sbatch (one vLLM server per node, disjoint dataset slices, auto-resume, manual
  afterany restart chain, `curator` conda env, --account=reformo); (2) the declarative `generate.py` /
  class-based `generate_abstract.py` generators under data/ (data/generation BaseDataGenerator + InferenceEngine
  for OpenAI/Anthropic/vLLM). Use for bulk completion/synthetic-data generation. The AGENTIC trace-generation
  path (Harbor + Daytona rollouts, MiniMax/GLM trace sets) is the SEPARATE `datagen-launch` skill.
---

# datagen-standard-launch

**Standard = non-agentic data generation**: generate completions/synthetic data directly from a model, no
Harbor agent loop, no Daytona sandboxes. Contrast with **`datagen-launch`** (agentic Harbor trace-gen over
Daytona — the production RL/SFT trace sets). Two paths here.

---

## Path 1 — Curator sharded datagen (multi-node DP) — the primary standard path

`data/sbatches/run_curator_datagen_sharded.sbatch`: runs **N independent vLLM servers (one per node)**, each
processing a **disjoint slice** of the input dataset via Curator's async datagen; results merge at the end.
Default **32 nodes** (`#SBATCH --nodes=32`, 4 GPUs/node), `--account=reformo` on Jupiter (NOT the default
`jureap59`). Runs from the **`curator` conda env**, not `otagent`.

**Auto-resume:** each shard's output dir is **stable** (no timestamps/job-IDs in the path —
`data/sbatches/curator_runs/<model>__<dataset>__<N>shards/shard_<i>/checkpoint_*.parquet`), so a SLURM
TIMEOUT kill + an `afterany` restart picks up where it left off.

**Pre-req (login node, has internet):** cache the input dataset first —
```bash
conda activate curator
python -c "from datasets import load_dataset; ds=load_dataset('<dataset>',split='train'); print(len(ds))"
```

**Launch — positional args** `<model> <input_dataset> <output_repo> [limit] [save_every]`:
```bash
# Simple (no restarts):
sbatch data/sbatches/run_curator_datagen_sharded.sbatch <model> <input_dataset> <output_repo> [limit] [save_every]

# With restart chain (recommended for long datasets) — build the afterany chain MANUALLY:
FIRST=$(sbatch data/sbatches/run_curator_datagen_sharded.sbatch \
  <model> <input_dataset> <output_repo> [limit] [save_every] | awk '{print $4}')
PREV=$FIRST; for i in $(seq 1 6); do
  PREV=$(sbatch --dependency=afterany:$PREV \
    data/sbatches/run_curator_datagen_sharded.sbatch \
    <model> <input_dataset> <output_repo> [limit] [save_every] | awk '{print $4}')
done
```
- **`MAX_RESTARTS` in the sbatch header comment is NOT implemented** — you MUST build the `--dependency=afterany:` chain by hand as above. (Setting the env var does nothing.)
- **`save_every`: pass `700`, not the default 200** → ~4 checkpoints per 12 h job instead of ~14 (each checkpoint is a serial GPU-wasting pause). It's the 5th positional arg.
- `limit` (4th arg) caps rows for a smoke run; omit for the full set.

---

## Path 2 — declarative / class-based generator scripts (`data/`)

`data/` holds named pipeline dirs. Two styles:

- **Declarative `generate.py`** — self-contained scripts for local / one-off runs:
  ```bash
  python data/<dataset>/generate.py [--flags]
  ```
- **Class-based `generate_abstract.py`** — subclass `BaseDataGenerator` for HPC runs with launcher-managed vLLM endpoints, submitted through the unified launcher:
  ```bash
  python -m hpc.launch --job_type datagen \
    --datagen_script data/<dataset>/generate_abstract.py \
    --datagen_target_repo <org/repo> \
    --datagen_extra_args "--stage both --limit 2000"
  ```

Core modules in **`data/generation/`**: `base.py` (`BaseDataGenerator`), `schemas.py`
(`GenerationRequest`/`GenerationResult`), `engines.py` (`InferenceEngine` implementations for
OpenAI / Anthropic / vLLM). Use this path for custom synthetic-data recipes that aren't agent rollouts.

---

## Cleanup / verification
Standard datagen pushes a plain HF dataset (merged parquet shards → `<output_repo>`). Verify the row count
and that the repo self-populated; no Daytona/trace-export step applies. (The agentic-trace cleanup +
MiniMax auto-advance live in `datagen-job-cleanup` and apply to the `datagen-launch` path, not this one.)
Cluster particulars (account, paths, the `curator` env) → `.claude/ops/jupiter/`; the launcher overview →
`.claude/projects/ot-agent/ot-agent.md`.
