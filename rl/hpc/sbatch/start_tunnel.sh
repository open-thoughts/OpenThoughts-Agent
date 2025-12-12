#!/bin/bash

NODE_HOST=$(hostname -s)

# if jrc in the hostname, we are on a jureca
if [[ $NODE_HOST == jrc* ]]; then
    LOGIN_NODE="jrlogin05i"
    TUNNEL_PORT=7001
    NODE_HOST="${NODE_HOST}i"
    NODE_IP=$(nslookup "$NODE_HOST" | grep 'Address' | tail -n1 | awk '{print $2}')
    export DOCKER_HOST=tcp://$NODE_IP:$TUNNEL_PORT
elif [[ $NODE_HOST == jwb* ]]; then
    LOGIN_NODE="jwlogin22i"
    TUNNEL_PORT=7001
    NODE_HOST="${NODE_HOST}1"
    NODE_IP=$(nslookup "$NODE_HOST" | grep 'Address' | tail -n1 | awk '{print $2}')
    export DOCKER_HOST=tcp://$NODE_IP:$TUNNEL_PORT
else
    echo "This script is intended to be run on JSC compute nodes."
    exit 1
fi

if [ -z "${SSH_KEY}" ]; then
    echo "SSH_KEY is not set. Please set SSH_KEY to the path of your SSH private key."
    exit 1
fi

ssh -i "${SSH_KEY}" \
    -o ExitOnForwardFailure=yes \
    -o StrictHostKeyChecking=no \
    -L 127.0.0.1:${TUNNEL_PORT}:127.0.0.1:${TUNNEL_PORT} \
    -fN "${USER}@${LOGIN_NODE}"

# return the DOCKER_HOST for use in the sbatch script
echo "$DOCKER_HOST"



