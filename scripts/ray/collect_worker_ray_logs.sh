#!/usr/bin/env bash
# collect_worker_ray_logs.sh — Collect Ray logs from SLURM worker nodes.
#
# Usage: source this script, then call collect_worker_ray_logs <dest_dir>
#
# Prerequisites:
#   - SSH_KEY env var pointing to an SSH private key accepted by worker nodes
#   - SLURM_JOB_NODELIST set (automatic inside SLURM jobs)
#
# This script is cluster-specific: it requires intra-cluster SSH access
# from the head node to worker nodes, which is only available on clusters
# where SSH_KEY is configured (e.g. Jupiter/JSC).

collect_worker_ray_logs() {
  local DEST_DIR="$1"
  if [[ -z "$DEST_DIR" ]]; then
    echo "Usage: collect_worker_ray_logs <dest_dir>" >&2
    return 1
  fi

  if [[ -z "${SSH_KEY:-}" ]]; then
    echo "  Skipping worker log collection (SSH_KEY not set)"
    return 0
  fi
  if [[ -z "${SLURM_JOB_NODELIST:-}" ]]; then
    echo "  Skipping worker log collection (SLURM_JOB_NODELIST not set)"
    return 0
  fi
  if [[ ! -f "$SSH_KEY" ]]; then
    echo "  WARNING: SSH_KEY=$SSH_KEY does not exist, skipping worker log collection"
    return 0
  fi

  local WORKERS_DIR="$DEST_DIR/ray_${SLURM_JOB_ID}_workers"
  mkdir -p "$WORKERS_DIR"

  local HEAD_NODE
  HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

  for node in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    [[ "$node" == "$HEAD_NODE" ]] && continue  # head node already copied by caller
    echo "  Collecting Ray logs from worker $node..."
    mkdir -p "$WORKERS_DIR/$node"
    rsync -az --ignore-errors \
      -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i $SSH_KEY" \
      "$node:/tmp/ray/" "$WORKERS_DIR/$node/" 2>/dev/null || \
      echo "  WARNING: Failed to collect logs from $node"
  done
}
