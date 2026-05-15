#!/bin/bash
# ==============================================================================
# SSH Tunnel Proxy for Leonardo Compute Nodes
#
# Creates a SOCKS5 proxy via SSH dynamic port forwarding from compute → login.
# Based on Marianna's start_proxy_tunnel_lrdn.sh.
#
# Prerequisites:
#   1. Generate SSH cert via step-ca (from local machine):
#      step ssh certificate 'bfeuer00' --provisioner cineca-hpc ~/.ssh/leonardo_daytona --no-password --insecure
#   2. Sync keys to Leonardo:
#      rsync -avz -e 'ssh -i ~/.ssh/leonardo_daytona -o IdentitiesOnly=yes -o StrictHostKeyChecking=no' \
#        ~/.ssh/leonardo_daytona ~/.ssh/leonardo_daytona.pub ~/.ssh/leonardo_daytona-cert.pub \
#        bfeuer00@login.leonardo.cineca.it:~/.ssh/
#   3. Set SSH_KEY env var (or use default path)
#
# Usage (from sbatch script):
#   CMD_PREFIX=$(bash eval/leonardo/start_proxy_tunnel.sh)
#   $CMD_PREFIX python my_script.py   # runs through proxy
#
# The ONLY stdout output is the proxychains command prefix.
# All diagnostic messages go to stderr.
# ==============================================================================

set -e

NODE_HOST=$(hostname -s)

# Determine login node based on compute node hostname
if [[ $NODE_HOST == lrdn* ]]; then
    LOGIN_NODE="login05"
else
    echo "ERROR: Not a Leonardo compute node (hostname: $NODE_HOST)" >&2
    exit 1
fi

TUNNEL_PORT="${TUNNEL_PORT:-27003}"

# SSH key for intra-cluster tunneling (cert-based, no passphrase)
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/leonardo_daytona}"

if [ ! -f "$SSH_KEY" ]; then
    echo "ERROR: SSH key not found at $SSH_KEY" >&2
    echo "Sync keys from local machine (see Prerequisites above)" >&2
    exit 1
fi

echo "Using SSH key: $SSH_KEY" >&2

# --- Open SSH tunnel (dynamic SOCKS5 port forwarding) ---
NODE_IP=$(nslookup "$NODE_HOST" 2>/dev/null | grep 'Address' | tail -n1 | awk '{print $2}')
if [ -z "$NODE_IP" ]; then
    NODE_IP="$NODE_HOST"
fi

ssh -g -f -N -D "${TUNNEL_PORT}" \
    -i "$SSH_KEY" \
    -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=no \
    -o BatchMode=yes \
    -o ConnectTimeout=30 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=15 \
    -o TCPKeepAlive=no \
    -o ExitOnForwardFailure=yes \
    "${USER}@${LOGIN_NODE}"

echo "SSH tunnel established: localhost:${TUNNEL_PORT} → ${LOGIN_NODE}" >&2

# --- Generate proxychains config ---
CFG_PATH="${HOME}/.proxychains/proxychains_${SLURM_JOB_ID:-local}.conf"
mkdir -p "$(dirname "$CFG_PATH")"

cat > "$CFG_PATH" <<PCEOF
strict_chain
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${NODE_IP} ${TUNNEL_PORT}
PCEOF

echo "Proxychains config at $CFG_PATH (socks5://${NODE_IP}:${TUNNEL_PORT})" >&2

# Export for child processes that source this script
export PROXYCHAINS_CONF_FILE="$CFG_PATH"
export PROXYCHAINS_SOCKS5_HOST="${NODE_IP}"
export PROXYCHAINS_SOCKS5_PORT="${TUNNEL_PORT}"

# The ONLY stdout line — captured by CMD_PREFIX=$(bash this_script.sh)
echo "proxychains4 -q -f $CFG_PATH"
