#!/bin/bash

NODE_HOST=$(hostname -s)

if [[ $NODE_HOST == jrc* ]]; then
    LOGIN_NODE="jrlogin05i"
elif [[ $NODE_HOST == jwb* ]]; then
    LOGIN_NODE="jwlogin22i"
else
    echo "This script is intended to be run on JSC compute nodes."
    exit 1
fi

TUNNEL_PORT=${TUNNEL_PORT:-7003}

if [ -z "${SSH_KEY}" ]; then
    echo "SSH_KEY is not set. Please set SSH_KEY to the path of your SSH private key."
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

CFG_PATH="${HOME}/.proxychains/proxychains.conf"
mkdir -p "$(dirname "${CFG_PATH}")"

{
  echo "strict_chain"
  echo "tcp_read_time_out  30000"
  echo "tcp_connect_time_out 15000"
  echo "localnet 127.0.0.0/255.0.0.0"
  echo "localnet 127.0.0.1/255.255.255.255"
  echo "localnet 10.0.0.0/255.0.0.0"
  echo "localnet 172.16.0.0/255.240.0.0"
  echo "localnet 192.168.0.0/255.255.0.0"
  echo "[ProxyList]"
  echo "socks5 ${PROXYCHAINS_SOCKS5_HOST} ${PROXYCHAINS_SOCKS5_PORT}"
} > "${CFG_PATH}"

echo "proxychains4 -f ${CFG_PATH}"



