# RL Training Pipeline — End-to-End Documentation

**Repo**: `/e/scratch/jureap59/feuer1/OpenThoughts-Agent` (branch: `penfever/working`)
**Harbor**: `/e/scratch/jureap59/feuer1/harbor` (branch: `penfever/temp-override`)
**Cluster**: Jupiter (JSC) — GH200 Grace-Hopper, 4 GPUs per node, 4 nodes = 16 GPUs total

---

## 0. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOGIN NODE                                │
│                                                                   │
│  1. Download HF dataset → local task dirs                        │
│  2. Pre-build Daytona snapshots (US region → RL region)          │
│  3. Generate sbatch script from template + config                │
│  4. sbatch --dependency=afterany:PREV_JOB submit                 │
│                                                                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ SLURM
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     COMPUTE NODES (4 nodes × 4 GPUs)             │
│                                                                   │
│  5. Activate RL venv (Python 3.12, torch 2.8, vllm 0.11)        │
│  6. Start Ray cluster (head + 3 workers)                         │
│  7. Setup container runtime (Daytona for sandboxes)              │
│  8. SSH tunnel + proxychains (no internet on compute)            │
│  9. Launch SkyRL training loop:                                  │
│     ├─ vLLM inference engines (rollout generation)               │
│     ├─ Harbor terminal_bench trials (Daytona sandboxes)          │
│     ├─ RLOO-N advantage estimation                               │
│     └─ FSDP2 distributed policy update                           │
│ 10. Checkpoint every 3 steps → {exp_dir}/checkpoints/            │
│ 11. Export model every 5 steps → {exp_dir}/exports/              │
│ 12. Upload traces to HF Hub (post-training)                      │
│ 13. Register model in Supabase DB                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Environment Setup

### 1.1 RL Virtual Environment

**Script**: `hpc/setup_rl_env.sh`
**Location**: `$DCFT/envs/rl/` (created by the script)

The RL env is **separate** from the datagen/SFT env due to dependency conflicts:
- RL: torch 2.8 + vllm 0.11.0
- Datagen: torch 2.9 + vllm 0.11.2

```bash
# Create the RL environment (one-time setup)
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent
./hpc/setup_rl_env.sh
```

Installs: PyTorch, Flash-Attention 2, Ray, vLLM, SkyRL (penfever/working branch), Harbor, uv package manager.

**Python 3.12 path** (for daytona SDK on login node):
```
/e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl/bin/python3
```

### 1.2 SkyRL Repositories

Cloned automatically by `setup_rl_env.sh`:
- `skyrl-train` → `$SCRATCH/SkyRL/skyrl-train` (penfever/working branch)
- `skyrl-gym` → `$SCRATCH/SkyRL/skyrl-gym` (penfever/working branch)

---

## 2. Task Dataset Download & Resolution

### 2.1 Data Sources

Tasks come from HuggingFace datasets or local directories. Each task has:
```
task_dir/
├── task.toml          # Task config (instructions, verifier, env settings)
├── environment/
│   ├── Dockerfile     # Container definition
│   └── fixtures/      # Optional: workspace files, test data
└── workspace/         # Optional: initial workspace state
```

### 2.2 Downloading Datasets (Hands-On)

There are two ways to get task data: from HuggingFace or from a local path.

#### Option A: From HuggingFace (automatic via launcher)

When you pass `--train_data org/dataset-name` to the launcher, it automatically:
1. Detects it's an HF dataset (exactly one `/`, not a filesystem path)
2. Downloads the parquet via `scripts.datagen.extract_tasks_from_parquet`
3. Extracts task directories to `$SCRATCH/tasks/{dataset_name}/`
4. Fixes permissions: `chmod -R a+rX` (for multi-node access)

```bash
# The launcher handles this automatically:
python -m hpc.launch --job_type rl --train_data penfever/my-dataset ...

# Or extract manually on login node:
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent
source envs/rl/bin/activate
python -m scripts.datagen.extract_tasks_from_parquet \
  --parquet org/dataset-name \
  --output_dir /e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/tasks/my-dataset \
  --on_exist skip
```

The extraction script auto-detects format:
- **Parquet with `task_binary` column** → extracts compressed archives into task dirs
- **Raw task directories** → copies directory structure directly

#### Option B: Local directory (already extracted)

If tasks are already on the shared filesystem, pass the path directly:
```bash
python -m hpc.launch --job_type rl \
  --train_data /e/data1/.../tasks/exp_rpt_nemotron-cpp ...
```

**Critical**: The path MUST be on a shared filesystem (not `/tmp`) for multi-node jobs.

#### Multiple datasets

```bash
python -m hpc.launch --job_type rl \
  --train_data '["org/dataset1", "/local/path/dataset2"]' ...
```

### 2.3 Resolution Flow (Internal)

**Function**: `resolve_rl_train_data()` in `hpc/rl_launch_utils.py`

```
Input: ["penfever/my-dataset"] or ["/local/path/to/tasks"]
  ↓
If HF dataset (detected by is_hf_dataset_path: one "/" + not a fs prefix):
  → Extract via scripts.datagen.extract_tasks_from_parquet
  → Store at: $SCRATCH/tasks/{dataset_name}/
  → Fix permissions: chmod -R a+rX (for multi-node access)
  ↓
If local path:
  → Fix permissions
  → Use as-is
  ↓
Output: ["/absolute/path/to/task/dirs"]
```

Scratch directory selection (in order): `$SCRATCH` → `$DCFT` → `$DCFT_PRIVATE` → `$HOME` → `/tmp` (last resort, warns)

### 2.4 Our Task Datasets

Located at: `/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/tasks/`

Two prefixes:
- `exp_rpt_*` — "repeat" datasets (standard task sets)
- `exp_rle_*` — "rle" datasets (curated/specialized)

| Dataset | Prefix | Description |
|---------|--------|-------------|
| nemotron-cpp | exp_rpt | C++ coding tasks |
| nemotron-junit | exp_rpt | Java/JUnit tasks |
| stack-csharp | exp_rpt | C# tasks |
| stack-jest-v2 | exp_rpt | JavaScript/Jest tasks (small) |
| stack-jest-large | exp_rpt | JavaScript/Jest tasks (large) |
| stack-pytest-large | exp_rpt | Python/pytest tasks (large) |
| stack-selfdoc-v2 | exp_rpt | Self-documenting tasks |
| exercism-python | exp_rpt | Exercism Python exercises |
| methods2test-v2 | exp_rpt | Method-to-test generation |
| 2skill | exp_rle | Two-skill tasks |
| adversarial | exp_rle | Adversarial tasks |
| curated | exp_rle | Curated task mix |
| github_issue | exp_rle | GitHub issue resolution |
| structural_debug | exp_rle | Structural debugging tasks |

---

## 3. Daytona Snapshot Registry

### 3.1 What Are Snapshots?

Daytona snapshots are pre-built Docker container images. Without pre-built snapshots, every trial builds the Dockerfile from scratch → rate limiting (ThrottlerException: Too Many Requests).

### 3.2 Snapshot Naming

Harbor computes a deterministic hash of the `environment/` directory:

```python
# harbor/src/harbor/utils/container_cache.py
def environment_dir_hash_truncated(env_dir, truncate=12):
    """SHA256 of all files in environment/ dir, truncated to 12 hex chars."""
    # Processes files in sorted order for determinism
    # Hashes: relative_path + file_contents for each file
```

Snapshot name format:
- Standard: `harbor__{hash}__snapshot`
- RL target: `harbor__{hash}__RL__snapshot` (when `DAYTONA_TARGET=RL`)

### 3.3 Snapshot Pre-Creation

**Function**: `prebuild_daytona_snapshots()` in `hpc/rl_launch_utils.py`

Called automatically during `construct_rl_sbatch_script()` if:
- `DAYTONA_API_KEY` is set
- `harbor_env == "daytona"`
- `resolved_train_data` is not empty

Flow:
1. Discover task directories (find `task.toml` files)
2. Analyze unique Dockerfile environments (hash each `environment/` dir)
3. Check safety limits (max 5 new, max 40 total in org)
4. Build missing snapshots in "us" region (has build runners)
5. Register for "RL" region

### 3.4 Manual Snapshot Pre-Creation

When the launcher's auto-prebuild isn't available (e.g., resubmitting existing jobs):

```python
#!/usr/bin/env python3
"""Pre-create Daytona snapshots for RL region."""
# Use Python 3.12: /e/scratch/jureap59/feuer1/OpenThoughts-Agent/envs/rl/bin/python3

import asyncio, os
from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources

async def main():
    # Use the RL Daytona key (the one with RL region access)
    api_key = os.environ["DAYTONA_API_KEY"]
    client = AsyncDaytona(DaytonaConfig(api_key=api_key, target="RL"))
    await client.snapshot.create(CreateSnapshotParams(
        name="harbor__{hash}__RL__snapshot",
        image=Image.from_dockerfile("/path/to/environment/Dockerfile"),
        resources=Resources(cpu=1, memory=1, disk=3),
    ))
    # Wait for ACTIVE state...
    await client.close()

asyncio.run(main())
```

### 3.5 Daytona API Keys

There are three API keys stored in the secrets env (`DC_AGENT_SECRET_ENV`):

| Key | Usage | Has RL Region? |
|-----|-------|---------------|
| RL key | **ALWAYS** for RL jobs | Yes |
| org1 key | Data generation + eval | No |
| org2 key | Eval only | No |

**Critical**: Using org1/org2 keys for RL jobs → "Region not found" error. Always use the RL key for anything involving `DAYTONA_TARGET=RL`.

### 3.6 Snapshot Cleanup

To list and delete snapshots (e.g., stale ERROR-state snapshots):

```python
#!/usr/bin/env python3
"""List and clean up Daytona snapshots."""
import asyncio, os
from daytona import AsyncDaytona, DaytonaConfig
from daytona._async.snapshot import SnapshotState

async def cleanup(target="RL"):
    api_key = os.environ["DAYTONA_API_KEY"]
    client = AsyncDaytona(DaytonaConfig(api_key=api_key, target=target))
    try:
        page = 1
        while True:
            result = await client.snapshot.list(page=page, limit=100)
            print(f"Page {page}/{result.total_pages} ({result.total} total snapshots)")
            for snap in result.items:
                print(f"  {snap.name} - state: {snap.state}")
                # Delete ERROR snapshots:
                if snap.state == SnapshotState.ERROR:
                    await client.snapshot.delete(snap)
                    print(f"    DELETED")
            if page >= result.total_pages:
                break
            page += 1
    finally:
        await client.close()

asyncio.run(cleanup())
```

**SDK methods**: `client.snapshot.list(page, limit)` → `PaginatedSnapshots`, `.get(name)` → `Snapshot`, `.delete(snapshot)` → None, `.create(params)` → `Snapshot`

**Snapshot states**: `ACTIVE`, `PENDING`, `BUILDING`, `PULLING`, `ERROR`, `BUILD_FAILED`, `INACTIVE`, `REMOVING`

### 3.7 Computing Snapshot Hashes Manually

```python
import hashlib, os

def environment_dir_hash_truncated(env_dir, truncate=12):
    h = hashlib.sha256()
    for root, dirs, files in os.walk(env_dir):
        dirs.sort()
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, env_dir)
            h.update(relpath.encode())
            with open(fpath, 'rb') as f:
                h.update(f.read())
    return h.hexdigest()[:truncate]
```

### 3.8 RL Region Pipeline (End-to-End)

The RL region is a special Daytona deployment for RL training. Here's the complete pipeline to get snapshots working:

**Step 1: Identify unique environment hashes in your task dataset**
```bash
# Using harbor's hash function:
python3 -c "
from harbor.utils.container_cache import environment_dir_hash_truncated
import os, glob
task_dir = '/path/to/tasks/exp_rpt_nemotron-cpp'
hashes = set()
for task in sorted(glob.glob(f'{task_dir}/*/environment')):
    h = environment_dir_hash_truncated(task, truncate=12)
    hashes.add(h)
print(f'{len(hashes)} unique hashes: {hashes}')
"
```

**Step 2: Pre-create snapshots in RL region**
```bash
# Must use the RL Daytona key + target="RL"
export DAYTONA_API_KEY=<rl-key>

python3 -c "
import asyncio, os
from daytona import AsyncDaytona, DaytonaConfig, CreateSnapshotParams, Image, Resources

async def create(name, dockerfile):
    client = AsyncDaytona(DaytonaConfig(
        api_key=os.environ['DAYTONA_API_KEY'],
        target='RL'
    ))
    await client.snapshot.create(CreateSnapshotParams(
        name=name,
        image=Image.from_dockerfile(dockerfile),
        resources=Resources(cpu=1, memory=1, disk=3),
    ))
    await client.close()

# For each unique hash:
asyncio.run(create('harbor__HASH__RL__snapshot', '/path/to/environment/Dockerfile'))
"
```

**Step 3: Verify snapshots are ACTIVE**
```python
snap = await client.snapshot.get("harbor__HASH__RL__snapshot")
assert snap.state == SnapshotState.ACTIVE
```

**Step 4: Submit RL job with DAYTONA_TARGET=RL**
- The sbatch sets `export DAYTONA_TARGET="RL"` and `DAYTONA_API_KEY_OVERRIDE=<rl-key>`
- Harbor auto-generates snapshot names with `__RL__` suffix
- If snapshot exists → fast sandbox creation (~2s)
- If snapshot missing → tries to build from Dockerfile → may hit `_environment_definition_path` bug (see Troubleshooting)

**Key**: The RL region has no build runners. Snapshot builds are routed through `"us"` region, then registered for RL via `region_id`. This is handled by harbor's `_create_snapshot_with_retry()` but has a known bug (see section 12).

---

## 4. RL Configuration

### 4.1 YAML Config System

**Location**: `hpc/skyrl_yaml/jupiter/`

**Deployed YAML**: `/e/scratch/jureap59/etash/16GPU_tp4_deployed.yaml` (all overrides baked in)
**Base YAML**: `hpc/skyrl_yaml/jupiter/24GPU_base.yaml` (original, needs 20+ `--skyrl_override` flags)
**Extra configs**: `hpc/skyrl_yaml/jupiter/extra/` (variant experiments)

The deployed YAML has all overrides baked in, so you can use it directly without `--skyrl_override` flags. The config JSON at `{exp_dir}/configs/{job_name}_rl_config.json` contains the final resolved hydra args.

**Actual deployed values** (16 GPUs across 4 nodes):

```yaml
entrypoint: examples.terminal_bench.entrypoints.main_tbench

terminal_bench:
  harbor:
    name: terminus-2           # Agent name
    n_concurrent_trials: 256   # Parallel sandbox trials
    auto_snapshot: true        # Hash-based snapshot caching
    override_timeout_sec: 900  # Per-trial timeout (seconds)
    interleaved_thinking: true # Enable thinking mode
    max_retries: 3             # Daytona retry on transient errors
    mask_exceptions: [DaytonaError, EnvironmentStartTimeoutError, ...]
    zero_exceptions: [AgentTimeoutError, ContextLengthExceededError]
  model_info:
    max_input_tokens: 32768
    max_output_tokens: 8192

trainer:
  strategy: fsdp2              # Fully Sharded Data Parallel v2
  algorithm:
    advantage_estimator: rloo_n  # RLOO-N (Reward LOO with N samples)
    use_kl_loss: true
    kl_loss_coef: 0.01
    eps_clip_low: 0.2
    eps_clip_high: 0.2
    loss_reduction: token_mean
  epochs: 10                   # Gradient epochs per batch
  update_epochs_per_batch: 1
  train_batch_size: 64
  policy_mini_batch_size: 64
  micro_train_batch_size_per_gpu: 1
  ckpt_interval: 3            # Checkpoint every 3 steps
  hf_save_interval: 10        # Export to HF every 10 steps
  resume_mode: latest          # Auto-resume from latest checkpoint
  policy:
    optimizer_config:
      lr: 1e-5
      weight_decay: 0.01
      adam_betas: [0.9, 0.999]
      max_grad_norm: 1.0
    fsdp_config:
      fsdp_size: 4
      cpu_offload: false
  placement:
    policy_num_nodes: 2        # 8 GPUs for policy (2 nodes × 4 GPUs)
    ref_num_nodes: 2           # 8 GPUs for reference model
    policy_num_gpus_per_node: 4
    ref_num_gpus_per_node: 4
  fully_async:
    max_staleness_steps: 8
    num_parallel_generation_workers: 128

generator:
  backend: vllm
  inference_engine_tensor_parallel_size: 4  # TP4 per engine
  num_inference_engines: 2                   # 2 engines × TP4 = 8 GPUs
  n_samples_per_prompt: 8                    # 8 rollouts per prompt
  gpu_memory_utilization: 0.9
  max_num_seqs: 64
  max_num_batched_tokens: 65536              # 64k token batch
  enable_prefix_caching: true
  enable_chunked_prefill: true
  weight_sync_backend: nccl
  async_engine: true
  engine_init_kwargs:
    max_model_len: 32768                     # 32k context window
    kv_cache_dtype: fp8                      # FP8 KV cache for memory savings
  sampling_params:
    max_generate_length: 8192
    temperature: 1.0
    top_p: 0.95
    top_k: -1                                # Disabled
```

**GPU allocation** (16 GPUs = 4 nodes × 4 GPUs):
- Policy model: 8 GPUs (2 nodes × 4, FSDP2)
- Reference model: 8 GPUs (2 nodes × 4, FSDP2)
- vLLM inference: 2 engines × TP4 = 8 GPUs (colocated with policy/ref GPUs)

**Key overrides from base YAML** (applied via `--skyrl_override`):

| Parameter | Base YAML | Deployed Override |
|-----------|-----------|-------------------|
| `inference_engine_tensor_parallel_size` | 1 | **4** |
| `num_inference_engines` | 16 | **2** |
| `max_num_seqs` | 24 | **64** |
| `gpu_memory_utilization` | 0.75 | **0.9** |
| `temperature` | 0.7 | **1.0** |
| `top_k` | 20 | **-1** (disabled) |
| `epochs` | 2 | **10** |
| `ckpt_interval` | 999999 | **3** |
| `hf_save_interval` | 5 | **10** |
| `max_staleness_steps` | 16 | **8** |
| `num_parallel_generation_workers` | 768 | **128** |
| `n_concurrent_trials` | 280 | **256** |
| `lr` | 5e-6 | **1e-5** |
| `weight_decay` | 0.0 | **0.01** |
| `max_grad_norm` | 10.0 | **1.0** |
| `use_kl_loss` | false | **true** (coef=0.01) |
| `reshard_after_forward` | true | **false** |
| `timeout_multiplier` | 1.0 | **8.0** |
| `override_timeout_sec` | 1800 | **900** |

**Job name decoding**: `rl_v1_tp4s64_8x_nemotron-cpp`
- `tp4` = `inference_engine_tensor_parallel_size=4`
- `s64` = `max_num_batched_tokens=65536` (64k)
- `8x` = `n_samples_per_prompt=8` (8 rollouts)

### 4.2 Config Resolution

**Function**: `parse_rl_config()` in `hpc/rl_config_utils.py`

```
--rl_config terminal_bench.yaml
  ↓
1. Check absolute path
2. Check hpc/skyrl_yaml/{name}
3. Check hpc/skyrl_yaml/{name}.yaml
  ↓
Parse YAML → ParsedRLConfig dataclass
  ↓
Build Hydra args → List[str] for SkyRL CLI
```

### 4.3 Key Hydra Arguments

These get passed to SkyRL as command-line overrides:
```
+terminal_bench_config=terminal_bench
trainer.epochs=2
trainer.max_steps=80
data.train_data=["/path/to/tasks"]
data.val_data=["open-thoughts/OpenThoughts-TB-dev"]
generator.num_inference_engines=16
++trainer.hf_hub_repo_id=laion/{job_name}
```

---

## 5. Job Launch

### 5.1 CLI Entry Point

Use the deployed YAML (`16GPU_tp4_deployed.yaml`) which has all overrides baked in — no `--skyrl_override` flags needed:

```bash
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent
source envs/rl/bin/activate

python -m hpc.launch \
  --job_type rl \
  --rl_config /e/scratch/jureap59/etash/16GPU_tp4_deployed.yaml \
  --model_path laion/r2egym-nl2bash-stack-bugsseq-fixthink-again \
  --train_data /path/to/tasks/exp_rpt_nemotron-cpp \
  --num_nodes 4 \
  --gpus_per_node 4 \
  --time_limit 12:00:00 \
  --job_name rl_v3_tp4s64_8x_nemotron-cpp \
  --daytona_api_key $DAYTONA_API_KEY \
  --dependency "afterany:12345"
```

The `--daytona_api_key` must be the **RL key** (the one with RL region access). It's passed via CLI and gets baked into the sbatch script as `DAYTONA_API_KEY_OVERRIDE`. The key is stored in the secrets env file (`DC_AGENT_SECRET_ENV`).

<details>
<summary>Alternative: using base YAML with --skyrl_override flags</summary>

If using `24GPU_base.yaml` directly, you need these 20 overrides (all 18 jobs use the same set):

```bash
python -m hpc.launch --job_type rl --rl_config jupiter/24GPU_base.yaml ... \
  --skyrl_override generator.inference_engine_tensor_parallel_size=4 \
  --skyrl_override generator.num_inference_engines=2 \
  --skyrl_override generator.gpu_memory_utilization=0.9 \
  --skyrl_override generator.max_num_seqs=64 \
  --skyrl_override generator.timeout_multiplier=8.0 \
  --skyrl_override generator.sampling_params.temperature=1.0 \
  --skyrl_override generator.sampling_params.top_k=-1 \
  --skyrl_override trainer.epochs=10 \
  --skyrl_override trainer.ckpt_interval=3 \
  --skyrl_override trainer.hf_save_interval=10 \
  --skyrl_override trainer.algorithm.use_kl_loss=true \
  --skyrl_override trainer.algorithm.kl_loss_coef=0.01 \
  --skyrl_override trainer.policy.optimizer_config.lr=1e-5 \
  --skyrl_override trainer.policy.optimizer_config.weight_decay=0.01 \
  --skyrl_override trainer.policy.optimizer_config.max_grad_norm=1.0 \
  --skyrl_override trainer.policy.fsdp_config.reshard_after_forward=false \
  --skyrl_override trainer.ref.fsdp_config.reshard_after_forward=false \
  --skyrl_override trainer.fully_async.max_staleness_steps=8 \
  --skyrl_override trainer.fully_async.num_parallel_generation_workers=128 \
  --skyrl_override trainer.micro_forward_batch_size_per_gpu=1
```
</details>

### 5.2 Launch Flow

```
hpc/launch.py:main()
  ↓
1. detect_hpc() → Jupiter config (account, partition, modules)
2. parse_args() → CLI arguments merged with defaults
3. get_job_name() → auto-derived or --job_name
4. launch_rl_job(exp_args, hpc)
   ├─ check_rl_environment() → envs/rl/
   ├─ construct_rl_sbatch_script(exp_args, hpc)
   │  ├─ parse_rl_config(yaml)
   │  ├─ resolve_rl_train_data() → local task paths
   │  ├─ prebuild_daytona_snapshots() → RL region snapshots
   │  ├─ pre_download_model() → local model cache
   │  ├─ build_skyrl_hydra_args() → CLI args for SkyRL
   │  ├─ RLJobConfig → JSON (serialized config)
   │  ├─ Load universal_rl.sbatch template
   │  ├─ Substitute 30+ template variables
   │  └─ Write {exp_dir}/sbatch/{job_name}_rl.sbatch
   └─ launch_sbatch(sbatch_path, dependency=dependency)
      → returns SLURM job ID
```

### 5.3 Job Name Convention

Format: `rl_{version}_tp{tp}s{seq}_{nodes}x_{dataset}`

Example: `rl_v3_tp4s64_8x_nemotron-cpp`
- `rl` — job type
- `v3` — model version (v1=bugsseq, v3=fixthink-again)
- `tp4` — tensor_parallel_size=4 (but actually 1 per engine, 4 GPUs per node)
- `s64` — sequence length 64k
- `8x` — 8 rollouts
- `nemotron-cpp` — dataset name

### 5.4 Generated Sbatch Script

**Template**: `hpc/sbatch_rl/universal_rl.sbatch`
**Output**: `{experiments_dir}/{job_name}/sbatch/{job_name}_rl.sbatch`

Key substitutions:
```bash
#SBATCH --time=12:00:00
#SBATCH --nodes=4
#SBATCH --job-name=rl_v3_tp4s64_8x_nemotron-cpp
#SBATCH -p booster --account jureap59 --gres=gpu:4

export DAYTONA_TARGET="RL"      # RL region for snapshots
DAYTONA_API_KEY_OVERRIDE="$DAYTONA_API_KEY"  # Must be RL key

python -m hpc.rl_launch_utils --config {config_path}
```

### 5.5 Dependency Chains

```bash
# Submit with dependency on previous job
python -m hpc.launch --job_type rl ... --dependency "afterany:12345"

# Or manually:
sbatch --dependency=afterany:12345:12346:12347 script.sbatch
```

**Gotcha**: If you `scancel` a dependency, all dependent jobs are **released** (start running). This can cause >6 jobs running simultaneously. Always cancel the full chain or hold jobs first.

### 5.6 Experiments Directory Structure

```
/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces/
└── rl_v3_tp4s64_8x_nemotron-cpp/
    ├── logs/                    # Top-level logs
    ├── ray_logs/                # Preserved Ray logs
    └── rl_v3_tp4s64_8x_nemotron-cpp/
        ├── configs/
        │   └── rl_v3_tp4s64_8x_nemotron-cpp_rl_config.json
        ├── sbatch/
        │   └── rl_v3_tp4s64_8x_nemotron-cpp_rl.sbatch
        ├── logs/
        │   └── rl_v3_tp4s64_8x_nemotron-cpp_{SLURM_ID}.out
        ├── rl_v3_tp4s64_8x_nemotron-cpp/   # Inner job dir
        │   ├── checkpoints/
        │   │   ├── global_step_3/
        │   │   ├── global_step_6/
        │   │   └── latest_ckpt_global_step.txt
        │   ├── exports/          # HF-format model exports
        │   └── trace_jobs/       # Harbor trial results
        ├── wandb/
        └── ray_logs/
```

---

## 6. Job Execution (Inside Sbatch)

### 6.1 Sbatch Startup Sequence

```
1. Clean /tmp (tmux, ray, containers from previous jobs)
2. Load cluster modules
3. Set WORKDIR, source dotenv
4. Source secrets (DC_AGENT_SECRET_ENV)
5. Set DAYTONA_API_KEY override
6. Set DAYTONA_TARGET=RL
7. Activate RL venv (deactivate conda first)
8. CUDA/NCCL setup
9. Set env vars (PYTHONFAULTHANDLER, VLLM_USE_V1, etc.)
10. Setup Triton/TorchInductor cache (node-local /tmp)
11. SSH tunnel + proxychains (JSC has no internet on compute)
12. Setup container runtime (Daytona)
13. Create experiment directories
14. python -m hpc.rl_launch_utils --config {config_path}
```

### 6.2 RLJobRunner

**Class**: `RLJobRunner` in `hpc/rl_launch_utils.py`

```
RLJobRunner.run():
  ├─ _setup_environment()
  │  ├─ Set TENSOR_PARALLEL_SIZE, NUM_INFERENCE_ENGINES
  │  ├─ Ensure WANDB_DIR writable
  │  └─ Pass HF_TOKEN
  ├─ _run_with_ray()
  │  ├─ RayCluster.from_slurm() → start head + 3 workers
  │  ├─ Wait for cluster ready (all 16 GPUs)
  │  ├─ Set RAY_ADDRESS
  │  └─ _run_skyrl()
  │     └─ python -m examples.terminal_bench.entrypoints.main_tbench {hydra_args}
  └─ _launch_trace_upload() (post-training)
```

### 6.3 Ray Cluster Setup

**Module**: `hpc/ray_utils.py`

```
1. Parse SLURM_JOB_NODELIST → list of hostnames
2. Get head node IP via srun
3. Start Ray head: ray start --head --port 6379
4. Start Ray workers on remaining nodes: ray start --address HEAD:6379
5. Wait for cluster: poll until expected nodes/GPUs available
6. Set RAY_ADDRESS=HEAD_IP:6379
```

**Memory**: Auto-computed from SLURM allocation minus 5% headroom (or 32GB min for large nodes).

### 6.4 SkyRL Training Loop

Once Ray is ready, SkyRL runs the RL training:

```
SkyRL entrypoint: examples.terminal_bench.entrypoints.main_tbench
  ├─ Initialize FSDP2 policy + reference model (8 GPUs each)
  ├─ Initialize vLLM inference engines (16 engines on remaining GPUs)
  ├─ For each step (target: 80):
  │   ├─ Generate rollouts (vLLM → Harbor terminal_bench trials)
  │   │   ├─ Create Daytona sandboxes (using pre-built snapshots)
  │   │   ├─ Run agent (terminus-2) in sandbox
  │   │   ├─ Verify output → binary reward (0 or 1)
  │   │   └─ Collect trajectories
  │   ├─ Compute advantages (RLOO-N estimator)
  │   ├─ Policy gradient update (FSDP2, 2 epochs per step)
  │   ├─ Log metrics (reward/avg_raw_reward, loss, timing)
  │   └─ Checkpoint every ckpt_interval steps
  └─ Export final model
```

### 6.5 SSH Tunnel / Proxy (Jupiter-specific)

Compute nodes on Jupiter have no internet. The sbatch template sets up:
1. SSH tunnel to login node (SOCKS5 proxy)
2. proxychains library via `LD_PRELOAD`
3. All external traffic (Daytona API) routes through proxy
4. Internal traffic (Ray, NCCL) goes direct

---

## 7. Checkpoint Resumption

### 7.1 Automatic Resume

SkyRL automatically resumes from the latest checkpoint:

```yaml
trainer:
  resume_mode: latest    # Auto-detect and resume
  ckpt_path: null        # Derived from experiments_dir
  ckpt_interval: 999999  # Checkpoint save interval
```

When resubmitting an sbatch script, SkyRL:
1. Looks for `checkpoints/latest_ckpt_global_step.txt`
2. Loads the latest `global_step_N/` checkpoint
3. Resumes training from step N+1

### 7.2 Checkpoint Structure

```
checkpoints/
├── global_step_3/
│   ├── optimizer.pt
│   ├── model.safetensors
│   └── trainer_state.json
├── global_step_6/
├── global_step_9/
└── latest_ckpt_global_step.txt  # Contains "9"
```

Each checkpoint is ~92GB for Qwen3-8B.

### 7.3 Starting Fresh

To reset a job and start from scratch:
```bash
EXP_DIR="/e/data1/.../rl_v3_tp4s64_8x_nemotron-cpp/rl_v3_tp4s64_8x_nemotron-cpp"
rm -rf "$EXP_DIR/rl_v3_tp4s64_8x_nemotron-cpp/checkpoints"
rm -rf "$EXP_DIR/rl_v3_tp4s64_8x_nemotron-cpp/trace_jobs"
rm -rf "$EXP_DIR/rl_v3_tp4s64_8x_nemotron-cpp/exports"
rm -rf "$EXP_DIR/wandb"
rm -rf "$EXP_DIR/ray_logs"
rm -f "$EXP_DIR/logs/"*.out
```

Then resubmit the same sbatch script.

---

## 8. Model Upload & Trace Export

### 8.1 HuggingFace Hub Upload (Automatic During Training)

SkyRL has two automatic callbacks that handle model upload:

1. **HFModelSaveCallback** — saves models in HF format to `{exp_dir}/exports/step_N/`
2. **HFHubUploadCallback** — uploads saved models to Hub asynchronously

Config (in YAML or via `--skyrl_override`):
```yaml
trainer:
  hf_hub_repo_id: laion/{job_name}  # Auto-derived from job_name if null
  hf_hub_private: false
  hf_hub_revision: main
  hf_save_interval: 10              # Export every 10 steps (deployed value)
  export_path: {exp_dir}/exports
```

Upload goes to: `{repo_id}/checkpoints/step_{N}/` on HuggingFace Hub.

**Requires**: `HF_TOKEN` env var or `huggingface-cli login`.

### 8.2 Manual Model Upload

For uploading a specific checkpoint manually:

```bash
# Option A: Using the helper script
bash rl/hpc/scripts/helpers/manual_upload.sh /path/to/checkpoint DCAgent2/my-model

# Option B: Using Python directly
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='/path/to/exports/step_60',
    repo_id='laion/rl_v1_tp4s64_8x_nemotron-cpp',
    repo_type='model',
    commit_message='Upload step 60 checkpoint',
)
"
```

The manual upload script auto-detects checkpoint structure, finds the policy directory, and copies only HF format files (.safetensors, config.json, tokenizer files), excluding FSDP checkpoint files.

### 8.3 Trace Upload (Automatic Post-Training)

**Function**: `_launch_trace_upload()` in `RLJobRunner`

After training completes (or times out), a subprocess runs:
```bash
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job_dir {exp_dir}/{job_name} \
  --repo_id DCAgent/{job_name} \
  --episodes last \
  --dataset_type SFT
```

This:
1. Reads trial results from `trace_jobs/`
2. Extracts agent trajectories (conversations + tool calls)
3. Sanitizes data (unicode, subagent merging)
4. Converts to HF Dataset format
5. Pushes to Hub (chunked: 100MB max shards)
6. Registers dataset in Supabase DB (unless `--skip_register`)

### 8.4 Manual Trace Upload

```bash
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent
source envs/rl/bin/activate

python -m scripts.harbor.make_and_upload_trace_dataset \
  --job_dir /path/to/trace_jobs \
  --repo_id DCAgent/my-traces \
  --episodes last \
  --filter success \
  --dataset_type SFT

# Arguments:
#   --job_dir       Path to Harbor job directory (contains trace_jobs/)
#   --repo_id       Target HF dataset repo (org/name)
#   --episodes      all|last (default: last)
#   --filter        success|failure|none (default: none)
#   --dataset_type  SFT|RL (default: SFT)
#   --skip_register Skip Supabase registration
#   --to_sharegpt   Export in ShareGPT format
```

---

## 9. Database Registry

### 9.1 Supabase Integration

**Config**: `database/unified_db/config.py`

Credentials loaded from `DC_AGENT_SECRET_ENV` or `KEYS` env var pointing to a credentials file:
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY` (optional, for admin ops)

### 9.2 Automatic Registration (During Training)

When `trainer.enable_db_registration=true`, a `DatabaseRegistrationCallback` runs at training end:
- Collects: agent name, base model, dataset names, HF repo ID, WandB link, hyperparams
- Calls `register_trained_model(record)` to insert into Supabase

### 9.3 Manual Model Registration

```bash
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent
source envs/rl/bin/activate

# Option A: Register HF model
python scripts/database/manual_db_push.py \
  --hf-model-id laion/rl_v1_tp4s64_8x_nemotron-cpp \
  --dataset-name exp_rpt_nemotron-cpp \
  --base-model laion/r2egym-nl2bash-stack-bugsseq \
  --training-type RL \
  --agent-name terminus-2

# Option B: Register from Python
python3 -c "
from database.unified_db.utils import register_hf_model
from datetime import datetime
result = register_hf_model(
    repo_name='laion/rl_v1_tp4s64_8x_nemotron-cpp',
    agent_id='<agent-uuid>',
    training_start=datetime(2026, 3, 8),
    training_type='RL',
    dataset_names='exp_rpt_nemotron-cpp',
)
print(result)
"
```

### 9.4 Manual Dataset Registration

```bash
# Registers a HF dataset in Supabase (also done automatically by trace upload)
python3 -c "
from database.unified_db.utils import register_hf_dataset
register_hf_dataset(
    repo_name='DCAgent/rl_v1_tp4s64_8x_nemotron-cpp',
    dataset_type='SFT',
)
"
```

### 9.5 Manual Eval Results Push

```bash
python scripts/database/manual_db_eval_push.py \
  --job-dir /path/to/trace_jobs/eval-run-dir \
  --hf-repo DCAgent2/my-eval-traces \
  --hf-episodes last
```

### 9.6 Registration Flow

```
Training completes (or times out)
  → DatabaseRegistrationCallback (if enable_db_registration=true)
    → register_trained_model() → Supabase models table
  → _launch_trace_upload() subprocess:
    → export_traces() → HF Dataset
    → push_to_hub() → HuggingFace Hub
    → register_hf_dataset() → Supabase datasets table
```

### 9.7 Required Environment Variables

| Variable | Purpose |
|----------|---------|
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_ANON_KEY` | Public anonymous key |
| `HF_TOKEN` | HuggingFace API token |
| `KEYS` | Path to credentials file (alternative to individual vars) |

---

## 10. Monitoring & Debugging

### 10.1 Log Locations

| Log | Path | Content |
|-----|------|---------|
| SBATCH output | `{exp_dir}/logs/{job_name}_{SLURM_ID}.out` | Main training log |
| Trace upload | `{exp_dir}/logs/{job_name}_trace_upload.log` | Post-training upload |
| Ray logs | `{exp_dir}/ray_logs/` | Ray worker/scheduler logs |
| WandB | `{exp_dir}/wandb/` | Metrics and artifacts |

### 10.2 Finding Log Files for a Job

```bash
BASE="/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"
JOB="rl_v1_tp4s64_8x_nemotron-cpp"

# Log files (one per SLURM submission, named {job_name}_{slurm_id}.out):
ls -lh "$BASE/$JOB/$JOB/logs/"

# Latest log:
ls -t "$BASE/$JOB/$JOB/logs/"*.out | head -1

# Latest checkpoint step:
cat "$BASE/$JOB/$JOB/$JOB/checkpoints/latest_ckpt_global_step.txt"
```

### 10.3 Extracting Rewards from Logs

```python
#!/usr/bin/env python3
"""Extract reward/avg_raw_reward from RL training logs."""
import re, glob, os

reward_pattern = re.compile(r"'reward/avg_raw_reward':\s*'([\d.]+)'")

BASE = "/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"
JOB = "rl_v1_tp4s64_8x_nemotron-cpp"

log_dir = f"{BASE}/{JOB}/{JOB}/logs"
all_rewards = []

for logfile in sorted(glob.glob(f"{log_dir}/*.out")):
    # Read in 1MB chunks (logs can be 15-65MB)
    file_size = os.path.getsize(logfile)
    with open(logfile, 'r', errors='replace') as f:
        for chunk_start in range(0, file_size, 1024*1024):
            f.seek(chunk_start)
            chunk = f.read(1024*1024 + 1000)
            for m in reward_pattern.finditer(chunk):
                all_rewards.append(float(m.group(1)))

print(f"{JOB}: {len(all_rewards)} steps")
if all_rewards:
    print(f"  First: {all_rewards[0]:.4f}, Last: {all_rewards[-1]:.4f}")
    print(f"  Last 5: {[round(r, 4) for r in all_rewards[-5:]]}")
```

### 10.4 Plotting Reward Curves

```python
#!/usr/bin/env python3
"""Plot reward curves for all RL jobs."""
import re, glob, os
import matplotlib.pyplot as plt

reward_pattern = re.compile(r"'reward/avg_raw_reward':\s*'([\d.]+)'")
BASE = "/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"

# Collect rewards for all jobs
jobs = {}
for exp_dir in sorted(glob.glob(f"{BASE}/rl_v*")):
    job = os.path.basename(exp_dir)
    log_dir = f"{exp_dir}/{job}/logs"
    rewards = []
    for logfile in sorted(glob.glob(f"{log_dir}/*.out")):
        with open(logfile, 'r', errors='replace') as f:
            content = f.read()
        rewards.extend(float(m) for m in reward_pattern.findall(content))
    if rewards:
        jobs[job] = rewards

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# v1 jobs
ax = axes[0]
ax.set_title("v1 (bugsseq)")
for name, rewards in sorted(jobs.items()):
    if "_v1_" in name:
        label = name.replace("rl_v1_tp4s64_8x_", "")
        ax.plot(rewards, label=f"{label} ({len(rewards)} steps)")
ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.legend(); ax.grid(True)

# v3 jobs
ax = axes[1]
ax.set_title("v3 (fixthink-again)")
for name, rewards in sorted(jobs.items()):
    if "_v3_" in name:
        label = name.replace("rl_v3_tp4s64_8x_", "")
        ax.plot(rewards, label=f"{label} ({len(rewards)} steps)")
ax.set_xlabel("Step"); ax.set_ylabel("Reward"); ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig("/e/scratch/jureap59/etash/rl_reward_curves.png", dpi=150)
print("Saved to /e/scratch/jureap59/etash/rl_reward_curves.png")
```

### 10.5 Key Metrics in Logs

- `reward/avg_raw_reward` — mean reward per step (0-1)
- `reward/avg_pass_at_8` — pass@8 metric
- `loss/avg_final_rewards` — same as raw_reward (different name)
- `timing/fwd_logprobs_values_reward` — forward pass time
- `generate/avg_tokens_non_zero_rewards` — token count for successful trials
- `generate/avg_tokens_zero_rewards` — token count for failed trials

### 10.6 Checking Job Health

```bash
# Current queue status
squeue -u guha1 --format="%.10i %.40j %.8T %.10M" | grep rl_v

# Job exit status
sacct -j JOB_ID --format=JobID,JobName,State,Elapsed,ExitCode

# Snapshot miss warnings (should be 0)
grep "not found (not global)" {logfile} | wc -l

# Current reward
tail -5000 {logfile} | grep "reward/avg_raw_reward"
```

---

## 11. Batch Submission Workflow

### 11.1 Manual Batch Submission

Since we can't edit the OT Agent repo, we submit by reusing existing sbatch scripts:

```bash
BASE="/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"

# Submit batch of 6 with dependency
DEPS="--dependency=afterany:JOB1:JOB2:JOB3:JOB4:JOB5:JOB6"
for name in rl_v3_tp4s64_8x_nemotron-cpp rl_v3_tp4s64_8x_stack-csharp ...; do
    sbatch $DEPS "$BASE/$name/$name/sbatch/${name}_rl.sbatch"
done
```

### 11.2 Rules

1. **Max 6 concurrent jobs** — use `--dependency=afterany:...` chains
2. **Always use RL Daytona key** — the key with RL region access, set in sbatch scripts
3. **Pre-create snapshots** before first submission
4. **Don't scancel middle of chain** — releases all dependents (cascade)
5. **12h wall time** → ~10-14 steps per run → need 4-6 runs per job to reach 60 steps

### 11.3 Current Models

| Alias | HF Repo | Local Path |
|-------|---------|------------|
| v1 | `laion/r2egym-nl2bash-stack-bugsseq` | `.../models--laion--r2egym-nl2bash-stack-bugsseq/snapshots/1e2611c5...` |
| v3 | `laion/r2egym-nl2bash-stack-bugsseq-fixthink-again` | `.../models--laion--r2egym-nl2bash-stack-bugsseq-fixthink-again/snapshots/2f4f59f0...` |

---

## 12. Troubleshooting

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ThrottlerException: Too Many Requests` | Missing snapshots → Dockerfile builds | Pre-create snapshots |
| `Region not found` | Wrong API key (org1/org2 for RL) | Use RL key |
| `snapshot not found (not global)` | Snapshot doesn't exist for this hash | Pre-create with correct hash |
| `TypeError: str \| None` | Python 3.9 used for daytona SDK | Use Python 3.12 from RL env |
| `ActorDiedError: node terminated (SIGTERM)` | 12h wall time reached | Normal — resubmit to continue |
| All rewards 0.0 | Broken verifier or env setup | Check logs for specific error; may need to delete checkpoints and restart fresh |
| `'DaytonaEnvironment' has no attribute '_environment_definition_path'` | Harbor bug: property renamed to `_dockerfile_path` but `_create_snapshot_with_retry` (line 1276) still uses old name | Fix line 1276 in `harbor/src/harbor/environments/daytona.py`: change `self._environment_definition_path` → `self._dockerfile_path`. Workaround: pre-create all snapshots so the fallback code path is never hit |
| `CUDA out of memory` | Ray memory too high | Reduce gpu_memory_utilization |
| >6 jobs running | Cancelled dependency released chain | Cancel ALL dependents, rebuild chain |

### Debugging Steps

1. **Check snapshot misses**: `grep "not found" logfile | head`
2. **Check rewards**: Extract `reward/avg_raw_reward` from log
3. **Check errors**: `grep "ERROR\|Exception\|Traceback" logfile | tail -20`
4. **Check Daytona**: `grep "DaytonaError\|ThrottlerException" logfile | wc -l`
5. **Check proxy**: Look for `Proxy: ENABLED via LD_PRELOAD` near start of log

### Monkey-Patching Harbor Bugs (Without Write Access)

When a bug exists in feuer1's harbor repo and you can't edit it directly, use a `sitecustomize.py` monkey-patch:

1. **Create the patch directory and script**:
```bash
mkdir -p /e/scratch/jureap59/etash/harbor_patch
cat > /e/scratch/jureap59/etash/harbor_patch/sitecustomize.py << 'PATCH'
"""
Monkey-patch for harbor bugs. Python auto-imports sitecustomize.py at startup.
This patches DaytonaEnvironment after it's imported via an import hook.
"""
def _apply_harbor_patch():
    try:
        _patched = False
        import builtins
        _original_import = builtins.__import__

        def _patching_import(name, *args, **kwargs):
            nonlocal _patched
            result = _original_import(name, *args, **kwargs)
            if not _patched and name == 'harbor.environments.daytona':
                _patched = True
                try:
                    cls = result.DaytonaEnvironment
                    if not hasattr(cls, '_environment_definition_path'):
                        cls._environment_definition_path = property(
                            lambda self: self._dockerfile_path
                        )
                        print("[PATCH] Applied _environment_definition_path → _dockerfile_path alias")
                except Exception as e:
                    print(f"[PATCH] Failed: {e}")
            return result

        builtins.__import__ = _patching_import
    except Exception:
        pass

_apply_harbor_patch()
PATCH
```

2. **Inject into sbatch files** — prepend the patch dir to PYTHONPATH:
```bash
# Patch all sbatch files at once
BASE="/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"
PATCH_DIR="/e/scratch/jureap59/etash/harbor_patch"
for sbatch_file in ${BASE}/rl_v*/*/sbatch/*.sbatch; do
  if ! grep -q "harbor_patch" "$sbatch_file"; then
    sed -i "s|export PYTHONPATH=\"\$WORKDIR:\${PYTHONPATH:-}\"|export PYTHONPATH=\"${PATCH_DIR}:\$WORKDIR:\${PYTHONPATH:-}\"|" "$sbatch_file"
  fi
done
```

3. **Verify** — in job logs, look for: `[PATCH] Applied _environment_definition_path → _dockerfile_path alias`

### Rolling Back Poisoned Checkpoints

When a buggy run writes checkpoints with 0.0 reward on top of healthy ones, you need to delete the bad checkpoints and reset `latest_ckpt_global_step.txt` so `resume_mode: latest` picks up the last healthy checkpoint.

**How to identify poisoned checkpoints**:
- Check checkpoint dates: `ls -la {ckpt_dir}/global_step_*` — healthy runs are before the bug, poisoned ones are after
- Cross-reference with log files: find which Slurm ID produced 0.0 rewards, check its date range

**How to roll back**:
```bash
BASE="/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"
JOB="rl_v1_tp4s64_8x_nemotron-cpp"
CKPT="${BASE}/${JOB}/${JOB}/${JOB}/checkpoints"

# 1. List checkpoints with dates to identify the cutoff
ls -la ${CKPT}/global_step_*

# 2. Delete poisoned checkpoints (example: steps 39,42,45 from bugged batch)
rm -rf ${CKPT}/global_step_39 ${CKPT}/global_step_42 ${CKPT}/global_step_45

# 3. Reset the latest pointer to the last healthy step
echo "36" > ${CKPT}/latest_ckpt_global_step.txt

# 4. Verify
cat ${CKPT}/latest_ckpt_global_step.txt  # Should show the healthy step
ls ${CKPT}/  # Should only show healthy checkpoints
```

**Example from 2026-03-11**: The `_environment_definition_path` bug caused batch 4 (Slurm 270441-270446) to produce 0.0 rewards but still write checkpoints. Rollback was:

| Job | Healthy last step | Deleted steps | Reset to |
|-----|------------------|---------------|----------|
| v1_nemotron-cpp | 36 (Mar 9) | 39, 42, 45 (Mar 10-11) | 36 |
| v1_stack-csharp | 18 (Mar 9) | 21, 24, 27 (Mar 10-11) | 18 |
| v1_stack-jest-v2 | 18 (Mar 9) | 21, 24 (Mar 10-11) | 18 |
| v1_stack-selfdoc-v2 | 12 (Mar 9) | 15, 18 (Mar 10-11) | 12 |

---

## 13. Quick Reference

### Submit a New RL Job (via launcher)

```bash
cd /e/scratch/jureap59/feuer1/OpenThoughts-Agent
source envs/rl/bin/activate

python -m hpc.launch \
  --job_type rl \
  --rl_config /e/scratch/jureap59/etash/16GPU_tp4_deployed.yaml \
  --model_path laion/r2egym-nl2bash-stack-bugsseq-fixthink-again \
  --train_data /path/to/tasks/exp_rpt_nemotron-cpp \
  --num_nodes 4 \
  --job_name rl_v3_tp4s64_8x_nemotron-cpp \
  --daytona_api_key $DAYTONA_API_KEY  # Must be the RL key
```

### Resubmit an Existing Job

```bash
sbatch --dependency=afterany:PREV_JOB_ID \
  /e/data1/.../rl_v3_tp4s64_8x_nemotron-cpp/rl_v3_tp4s64_8x_nemotron-cpp/sbatch/rl_v3_tp4s64_8x_nemotron-cpp_rl.sbatch
```

### Delete Checkpoints and Restart Fresh

```bash
BASE="/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"
NAME="rl_v3_tp4s64_8x_nemotron-cpp"
rm -rf "$BASE/$NAME/$NAME/$NAME/checkpoints"
rm -rf "$BASE/$NAME/$NAME/$NAME/trace_jobs"
rm -rf "$BASE/$NAME/$NAME/$NAME/exports"
rm -rf "$BASE/$NAME/$NAME/wandb"
rm -rf "$BASE/$NAME/$NAME/ray_logs"
rm -f  "$BASE/$NAME/$NAME/logs/"*.out
# Then resubmit sbatch
```

### Check All Job Rewards

```bash
BASE="/e/data1/datasets/playground/mmlaion/shared/guha1_glm47_traces"
for job in "$BASE"/rl_v*/; do
    name=$(basename "$job")
    latest=$(cat "$job/$name/$name/checkpoints/latest_ckpt_global_step.txt" 2>/dev/null || echo "?")
    reward=$(grep -oh "'reward/avg_raw_reward': '[^']*'" "$job/$name/logs/"*.out 2>/dev/null | tail -1)
    echo "$name: step=$latest, $reward"
done
```

### Plot Reward Curves

```bash
# See section 10.4 for the full plotting script
# Current plot at:
/e/scratch/jureap59/etash/rl_reward_curves.png
```

### Upload Model + Register in DB

```bash
# Upload checkpoint to HF Hub
bash rl/hpc/scripts/helpers/manual_upload.sh /path/to/exports/step_60 laion/my-model

# Register model in Supabase
python scripts/database/manual_db_push.py \
  --hf-model-id laion/my-model \
  --dataset-name exp_rpt_nemotron-cpp \
  --base-model laion/r2egym-nl2bash-stack-bugsseq \
  --training-type RL

# Upload traces to HF
python -m scripts.harbor.make_and_upload_trace_dataset \
  --job_dir $BASE/my-job/my-job/my-job/trace_jobs \
  --repo_id DCAgent/my-traces --episodes last --dataset_type SFT
```
