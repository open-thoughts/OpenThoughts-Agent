
NODE_HOST=$(hostname -s)

if [[ $NODE_HOST == jrc* ]]; then
    LOGIN_NODE="jrlogin05i"
elif [[ $NODE_HOST == jwb* ]]; then
    LOGIN_NODE="jwlogin22i"
elif [[ $NODE_HOST == lrdn* ]]; then
    LOGIN_NODE="login05"
else
    echo "This script is intended to be run on JSC compute nodes."
    exit 1
fi

TUNNEL_PORT=27003

if [ -z "${SSH_KEY}" ]; then
    echo "SSH_KEY is not set. Please set SSH_KEY to the path of your SSH private key."
    exit 1
fi

# NODE_HOST="${NODE_HOST}-interconnect-1.leonardo.local"
NODE_IP=$(nslookup $NODE_HOST | grep 'Address' | tail -n1 | awk '{print $2}')
# NODE_IP=$NODE_HOST

ssh -g -f -N -D ${TUNNEL_PORT} \
    -o StrictHostKeyChecking=no \
    -o ConnectTimeout=1000 \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=15 \
    -o TCPKeepAlive=no \
    -o ExitOnForwardFailure=yes \
    -o BatchMode=yes \
    -i ${SSH_KEY} \
    ${USER}@${LOGIN_NODE}

export PROXYCHAINS_SOCKS5_HOST=${NODE_IP}
export PROXYCHAINS_SOCKS5_PORT=${TUNNEL_PORT}

SLURM_JOB_ID=${SLURM_JOB_ID:-"local"}

CFG_PATH=~/.proxychains/proxychains_${SLURM_JOB_ID}.conf
export PROXYCHAINS_CONF_FILE=$CFG_PATH
mkdir -p ~/.proxychains

cat > "$CFG_PATH" <<EOF
strict_chain
tcp_read_time_out 30000
tcp_connect_time_out 15000
localnet 127.0.0.0/255.0.0.0
localnet 127.0.0.1/255.255.255.255
localnet 10.0.0.0/255.0.0.0
localnet 172.16.0.0/255.240.0.0
localnet 192.168.0.0/255.255.0.0
[ProxyList]
socks5 ${PROXYCHAINS_SOCKS5_HOST} ${PROXYCHAINS_SOCKS5_PORT}
EOF

# This will be the ONLY output captured by CMD_PREFIX
echo "proxychains4 -q -f $CFG_PATH"
