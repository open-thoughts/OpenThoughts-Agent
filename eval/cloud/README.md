# Cloud Eval Launcher

Launch OpenThoughts evaluations on cloud GPUs via [SkyPilot](https://skypilot.readthedocs.io/).

## Overview

The cloud launcher wraps `eval/local/run_eval.py` and runs it on cloud GPU instances. It handles:

- Provisioning GPU instances across multiple cloud providers
- Syncing your local codebase to the remote instance
- Running evaluations inside a Docker container
- Syncing results back to your local machine

## Installation

```bash
# Basic installation (GCP)
uv pip install -e ".[cloud]"

# Provider-specific installations
uv pip install -e ".[cloud-aws]"       # AWS
uv pip install -e ".[cloud-vast]"      # Vast.ai
uv pip install -e ".[cloud-all]"       # All providers
```

## Supported Providers

| Provider | Docker Runtime | Spot Instances | Region Selection |
|----------|---------------|----------------|------------------|
| `gcp` | Yes | Yes | Yes |
| `aws` | Yes | Yes | Yes |
| `azure` | Yes | Yes | Yes |
| `lambda` | Yes | No | No |
| `vast` | Yes | Yes | No |
| `kubernetes` | Yes | No | No |
| `runpod` | No | Yes | No |
| `cudo` | Yes | No | Yes |
| `paperspace` | Yes | No | Yes |
| `fluidstack` | Yes | No | No |

List all providers:
```bash
python eval/cloud/launch_eval_cloud.py --list-providers
```

## Quick Start

### 1. Configure your cloud provider

```bash
# Show setup instructions
python -m eval.cloud.providers --setup gcp
python -m eval.cloud.providers --setup lambda
python -m eval.cloud.providers --setup vast
```

For GCP:
```bash
brew install --cask google-cloud-sdk  # macOS
gcloud init
gcloud auth application-default login
```

For Lambda Cloud:
```bash
mkdir -p ~/.lambda_cloud
echo "api_key = YOUR_API_KEY" > ~/.lambda_cloud/lambda_keys
```

For Vast.ai:
```bash
pip install 'vastai-sdk>=0.1.12'
echo 'YOUR_API_KEY' > ~/.vast_api_key
```

### 2. Build and push the Docker image (one-time)

```bash
# Authenticate to GitHub Container Registry
docker login ghcr.io

# Build and push
./docker/build_and_push.sh
```

### 3. Launch an evaluation

```bash
python eval/cloud/launch_eval_cloud.py \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --datagen-config hpc/datagen_yaml/qwen3_8b_vllm_serve_32k_1xH200.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/your-eval-repo \
  --accelerator "H100:1,A100-80GB:1" \
  --cloud-provider gcp \
  --secrets-env /path/to/secrets.env
```

## Command Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--harbor-config` | Path to Harbor YAML configuration |
| `--model` | Model identifier for evaluation |
| `--eval-benchmark-repo` | Supabase repo ID for eval tracking |
| `--dataset` or `--dataset-path` | Dataset to evaluate on |

### Cloud Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--cloud-provider` | `gcp` | Cloud provider(s). Comma-separated for fallbacks (e.g., `gcp,aws,lambda`). |
| `--accelerator` | `A100:1` | GPU type and count. Comma-separated for fallbacks (e.g., `H100:1,H200:1,A100-80GB:1`). |
| `--region` | auto | Preferred region(s). Comma-separated for fallbacks (e.g., `us-central1,us-west1`). |
| `--zone` | auto | Preferred zone |
| `--use-spot` | false | Use spot/preemptible instances |
| `--docker-image` | auto | Docker image (auto-selects gpu-1x/4x/8x) |
| `--no-sync` | false | Skip syncing local code to VM |
| `--autostop` | `30` | Auto-stop cluster after N minutes idle. Set to `-1` to disable. |
| `--down` | false | Tear down cluster after task completes (don't keep for reuse) |

### Eval Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--agent` | `terminus-2` | Harbor agent to run |
| `--n-concurrent` | `16` | Concurrent eval tasks |
| `--n-attempts` | `3` | Retry attempts per task |
| `--gpus` | `1` | GPUs for run_eval |
| `--dry-run` | false | Dry run mode |

### Output Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--task-name` | `ot-eval-cloud` | SkyPilot task name |
| `--cluster-name` | auto | SkyPilot cluster name |
| `--remote-output-subdir` | `cloud_runs` | Subdirectory under workdir for outputs |
| `--local-sync-dir` | `./cloud_runs` | Local path to sync results |
| `--secrets-env` | none | Path to secrets.env file |

## Docker Images

The launcher uses pre-built Docker images from GitHub Container Registry:

- `ghcr.io/open-thoughts/openthoughts-agent:gpu-1x` - Single GPU
- `ghcr.io/open-thoughts/openthoughts-agent:gpu-4x` - Up to 4 GPUs
- `ghcr.io/open-thoughts/openthoughts-agent:gpu-8x` - Up to 8 GPUs

The image is auto-selected based on accelerator count. Build images with:

```bash
./docker/build_and_push.sh           # All variants
./docker/build_and_push.sh gpu-1x    # Single variant
./docker/build_and_push.sh --build-only  # Build without pushing
```

## Code Syncing

By default, your local codebase is synced to the remote VM on every launch via SkyPilot's `file_mounts`. This means:

- Code changes are reflected immediately without rebuilding Docker images
- Docker images only need rebuilding when dependencies change

Disable with `--no-sync` to use the code baked into the Docker image (faster but won't pick up local changes).

## Cluster Lifecycle Management

SkyPilot clusters persist after task completion by default, enabling fast reruns. The launcher provides several options to manage cluster lifecycle:

### Autostop (default: 30 minutes)

Clusters automatically stop after being idle:

```bash
--autostop 30    # Stop after 30 min idle (default)
--autostop 60    # Stop after 1 hour idle
--autostop -1    # Never auto-stop (careful: keeps billing!)
```

Stopped clusters retain their disk and can be restarted quickly with `sky start <cluster>`.

### Auto-teardown

Delete the cluster completely after the task finishes:

```bash
--down    # Tear down cluster after task completes
```

Use this for one-off jobs or when you won't need the cluster again.

### Cluster Reuse

Using the same `--cluster-name` reuses an existing cluster:

```bash
# First run: creates cluster
python eval/cloud/launch_eval_cloud.py --cluster-name my-eval ...

# Second run: reuses cluster (skips provisioning)
python eval/cloud/launch_eval_cloud.py --cluster-name my-eval ...
```

### Manual Cluster Management

```bash
sky status                  # List all clusters
sky stop <cluster>          # Stop cluster (pause billing, keep disk)
sky start <cluster>         # Restart stopped cluster
sky down <cluster>          # Terminate cluster completely
sky down -a                 # Terminate all clusters
```

## Examples

### GCP with H100

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider gcp \
  --accelerator H100:1 \
  --region us-central1 \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

### Lambda Cloud

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider lambda \
  --accelerator A100:1 \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

### Vast.ai with spot instances

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider vast \
  --accelerator RTX4090:1 \
  --use-spot \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

### Kubernetes cluster

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider kubernetes \
  --accelerator A100:1 \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

### Multi-GPU evaluation

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider gcp \
  --accelerator A100-80GB:4 \
  --gpus 4 \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

### Fallback accelerators (use first available)

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider gcp \
  --accelerator "H100:1,H200:1,A100-80GB:1" \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

SkyPilot will try each accelerator in order of cost and use the first available.

### Multi-cloud fallback (use cheapest available)

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider "gcp,aws,lambda" \
  --accelerator "H100:1,A100-80GB:1" \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

SkyPilot will try all 6 combinations (3 providers × 2 accelerators) and use the cheapest available.

### One-off job with auto-teardown

```bash
python eval/cloud/launch_eval_cloud.py \
  --cloud-provider gcp \
  --accelerator "H100:1,A100-80GB:1" \
  --autostop 15 \
  --down \
  --harbor-config hpc/harbor_yaml/trace_32concurrency_eval_ctx32k.yaml \
  --dataset terminal-bench@2.0 \
  --model your-org/your-model \
  --eval-benchmark-repo YourOrg/eval-repo
```

This will auto-stop after 15 minutes idle and delete the cluster when the task completes.

To see all available accelerators:
```bash
sky show-gpus        # Common GPUs
sky show-gpus --all  # All GPUs with pricing
```

## Provider Utilities

The `providers.py` module provides utilities for working with cloud providers:

```bash
# List all providers
python -m eval.cloud.providers --list

# Check credential status
python -m eval.cloud.providers --check

# Show setup instructions
python -m eval.cloud.providers --setup lambda
```

## Troubleshooting

### SkyPilot can't find gcloud/kubectl

Restart the SkyPilot API server after installing CLI tools:
```bash
sky api stop
sky api start
sky check
```

### Docker push denied

Authenticate to GitHub Container Registry:
```bash
docker login ghcr.io
# Username: your GitHub username
# Password: GitHub Personal Access Token with write:packages scope
```

### Docker pull unauthorized on cloud VM

If the cloud VM can't pull the Docker image:

1. **Make the package public** (recommended):
   - Go to https://github.com/orgs/open-thoughts/packages
   - Click on the package → Package settings → Danger Zone → Change visibility → Public

2. **Or use a public base image** as a workaround:
   ```bash
   --docker-image nvcr.io/nvidia/pytorch:24.01-py3
   ```

### GCP service account not found

If you see `Service account skypilot-v1@<project>.iam.gserviceaccount.com does not exist`:

```bash
# Create the service account
gcloud iam service-accounts create skypilot-v1 \
  --display-name="SkyPilot Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:skypilot-v1@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/compute.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:skypilot-v1@YOUR_PROJECT.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"
```

### Accelerator not found / ResourcesUnavailableError

If you see `Catalog does not contain any instances satisfying the request`:

1. Check valid accelerator names:
   ```bash
   sky show-gpus
   ```

2. Common accelerator names:
   - `H100` (not `H100-80GB`)
   - `A100` (40GB version)
   - `A100-80GB` (80GB version)
   - `L4`, `L40S`, `T4`, `V100`

3. Try fallback accelerators:
   ```bash
   --accelerator "H100:1,H200:1,A100-80GB:1"
   ```

### Provider credentials not found

Check credential status and get setup instructions:
```bash
python -m eval.cloud.providers --check
python -m eval.cloud.providers --setup <provider>
```

### Code changes not reflected

Make sure `--no-sync` is not set. By default, local code is synced on every launch.

## Output Syncing

After the eval task completes, outputs are automatically synced from the remote cluster to your local machine using rsync. The remote output directory is `<workdir>/cloud_runs/` where workdir is `/sky/workdir` (with code sync) or `/opt/openthoughts` (with `--no-sync`).

```
[cloud-sync] Syncing outputs from cluster...
[cloud-sync]   Remote: glm46-tb2-smoke:/sky/workdir/cloud_runs/
[cloud-sync]   Local:  /Users/you/OpenThoughts-Agent/cloud_runs/
[cloud-sync] Successfully synced outputs to /Users/you/OpenThoughts-Agent/cloud_runs/...
```

If `--down` is used, the sync happens before the cluster is torn down.

**Manual sync** (if automatic sync fails):
```bash
# SkyPilot clusters are accessible via SSH using the cluster name as hostname
rsync -avz --progress CLUSTER:/sky/workdir/cloud_runs/ ./cloud_runs/
# or
scp -r CLUSTER:/sky/workdir/cloud_runs/* ./cloud_runs/
```

## Architecture

```
eval/cloud/
├── launch_eval_cloud.py  # Main launcher script
├── providers.py          # Provider configurations and utilities
├── sync_utils.py         # Output syncing utilities (rsync-based)
└── README.md             # This file
```

The launcher:
1. Validates arguments and provider credentials
2. Builds file mounts for code sync
3. Creates a SkyPilot Task with the eval command
4. Launches on the specified provider
5. Streams logs and waits for completion
6. Syncs results back to local machine via rsync
7. Tears down cluster if `--down` was specified
