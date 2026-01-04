#!/bin/bash

NODE_HOST=$(hostname -s)

if [[ $NODE_HOST == jrc* ]]; then
    LOGIN_NODE="jrlogin05i"
elif [[ $NODE_HOST == jwb* ]]; then
    LOGIN_NODE="jwlogin22i"
else
    echo "This script is intended to be run on JSC compute nodes." >&2
    exit 1
fi

TUNNEL_PORT=${TUNNEL_PORT:-7003}

if [ -z "${SSH_KEY}" ]; then
    echo "SSH_KEY is not set. Please set SSH_KEY to the path of your SSH private key." >&2
    exit 1
fi

NODE_HOST="${NODE_HOST}i"
NODE_IP=$(nslookup "$NODE_HOST" | grep 'Address' | tail -n1 | awk '{print $2}')

ssh -g -f -N -D "${TUNNEL_PORT}" \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=1000 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=15 \
    -o TCPKeepAlive=no \
    -o ExitOnForwardFailure=yes \
    -o BatchMode=yes \
    -i "${SSH_KEY}" \
    "${USER}@${LOGIN_NODE}"

export PROXYCHAINS_SOCKS5_HOST=${NODE_IP}
export PROXYCHAINS_SOCKS5_PORT=${TUNNEL_PORT}

# Use a per-job config so concurrent jobs don't clobber each other.
CONF_DIR="${HOME}/.proxychains"
mkdir -p "${CONF_DIR}"

JOB_ID="${SLURM_JOB_ID:-$$}"
CFG_PATH="${CONF_DIR}/proxychains_${JOB_ID}.conf"

# Note: this script is typically executed via command-substitution in sbatch templates,
# so exports here won't propagate. We still set it for completeness in case the script
# is sourced / run interactively.
export PROXYCHAINS_CONF_FILE="${CFG_PATH}"

cat > "${CFG_PATH}" <<EOF
strict_chain
tcp_read_time_out  30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${PROXYCHAINS_SOCKS5_HOST} ${PROXYCHAINS_SOCKS5_PORT}
EOF

chmod 600 "${CFG_PATH}" 2>/dev/null || true

# This will be the ONLY stdout output captured by CMD_PREFIX in the sbatch template.
echo "proxychains4 -q -f ${CFG_PATH}"



