# Beam Cluster Setup Scripts

Scripts for provisioning a self-hosted Beta9 (Beam) cluster on GKE with Pinggy or LoadBalancer exposure.

## Prerequisites

- **gcloud CLI**: [Install](https://cloud.google.com/sdk/docs/install) and authenticate with `gcloud auth login`
- **kubectl**: [Install](https://kubernetes.io/docs/tasks/tools/)
- **Helm**: [Install](https://helm.sh/docs/intro/install/)
- **Pinggy account** (optional): For tunnel-based exposure, get a persistent URL from [pinggy.io](https://pinggy.io)

## Quick Start

### 1. Create cluster with Pinggy tunnel

```bash
# Set your credentials
export PINGGY_TOKEN="your-pinggy-token"

# Create cluster (interactive - keeps tunnel running)
python -m scripts.beam.cluster_setup create \
  --project-id my-gcp-project \
  --num-nodes 8 \
  --pinggy-url mybeam.a.pinggy.link \
  --pinggy-token $PINGGY_TOKEN
```

### 2. Create cluster with LoadBalancer

```bash
# Create cluster (non-interactive - always-on exposure)
python -m scripts.beam.cluster_setup create \
  --project-id my-gcp-project \
  --expose-method loadbalancer
```

### 3. Dry run to preview commands

```bash
python -m scripts.beam.cluster_setup create \
  --project-id my-gcp-project \
  --dry-run
```

## Commands

### create

Provision GKE cluster, deploy Beta9, expose to internet, and validate.

```bash
python -m scripts.beam.cluster_setup create \
  --project-id PROJECT_ID \
  [--cluster-name NAME] \
  [--region REGION] \
  [--num-nodes N] \
  [--machine-type TYPE] \
  [--expose-method {pinggy,loadbalancer}] \
  [--pinggy-url URL] \
  [--pinggy-token TOKEN] \
  [--skip-gke] \
  [--skip-beta9] \
  [--skip-expose] \
  [--skip-validation] \
  [--dry-run]
```

### destroy

Tear down everything.

```bash
python -m scripts.beam.cluster_setup destroy \
  --project-id PROJECT_ID \
  [--cluster-name NAME] \
  [--skip-gke] \
  [--skip-beta9] \
  [--dry-run]
```

### status

Show current cluster status.

```bash
python -m scripts.beam.cluster_setup status \
  --project-id PROJECT_ID \
  [--cluster-name NAME]
```

### validate

Run sandbox health checks against an existing cluster.

```bash
python -m scripts.beam.cluster_setup validate \
  --gateway-url https://mybeam.a.pinggy.link \
  [--num-tests N] \
  [--skip-isolation]
```

## Configuration

### Default Values

| Option | Default | Description |
|--------|---------|-------------|
| `--cluster-name` | `beam-sandbox-cluster` | GKE cluster name |
| `--region` | `us-central1` | GCP region |
| `--num-nodes` | `8` | Number of CPU-only nodes |
| `--machine-type` | `e2-standard-4` | GCE machine type (4 vCPU, 16GB RAM) |
| `--expose-method` | `pinggy` | How to expose cluster |

### Cost Estimates

- **e2-standard-4**: ~$0.13/hour per node
- **8 nodes**: ~$1.04/hour (~$750/month if running 24/7)
- **LoadBalancer**: ~$18/month additional

## Architecture

```
GCP Project
├── GKE Cluster (N x e2-standard-4 nodes)
│   └── Beta9 Namespace
│       ├── Gateway (HTTP:1994, gRPC:1993)
│       ├── Redis (scheduling)
│       ├── PostgreSQL (metadata)
│       └── Worker Pool (CPU sandboxes)
│
└── Exposure Method
    ├── Option A: Pinggy Tunnel
    │   └── kubectl port-forward → SSH tunnel → Public URL
    └── Option B: LoadBalancer
        └── GKE LoadBalancer → Public IP
```

## Using with Harbor

After cluster setup, configure Harbor's BeamEnvironment to use your cluster:

```bash
export BETA9_GATEWAY_HOST=mybeam.a.pinggy.link
```

Or in Harbor config:

```yaml
environment:
  type: beam
  # Beta9 will use BETA9_GATEWAY_HOST environment variable
```

## Troubleshooting

### Cluster creation fails

```bash
# Check GKE quotas
gcloud compute project-info describe --project PROJECT_ID

# Check for existing clusters
gcloud container clusters list --project PROJECT_ID
```

### Beta9 deployment fails

```bash
# Check Helm status
helm status beta9 -n beta9

# Check pod logs
kubectl logs -n beta9 -l app.kubernetes.io/name=beta9-gateway
```

### Pinggy tunnel disconnects

The tunnel script includes auto-reconnect. If issues persist:

```bash
# Test SSH connectivity
ssh -p 443 -v TOKEN@pro.pinggy.io

# Check if port-forward is running
ps aux | grep port-forward
```
