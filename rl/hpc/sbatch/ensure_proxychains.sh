#!/bin/bash

# Lightweight check that proxychains-ng is installed and environment is wired.
# Installation itself is a one-off step; see JSC_setup/README.md in dc-agent.

if ! command -v proxychains4 >/dev/null 2>&1; then
  echo "[proxychains] proxychains4 not found in PATH."
  echo "[proxychains] Please install proxychains-ng as described in dc-agent/JSC_setup/README.md."
  exit 1
fi

CONF_DIR="${HOME}/.proxychains"
CONF_PATH="${CONF_DIR}/proxychains.conf"

mkdir -p "${CONF_DIR}"

if [ ! -f "${CONF_PATH}" ]; then
  echo "[proxychains] ${CONF_PATH} does not exist."
  echo "[proxychains] Please create it according to dc-agent/JSC_setup/README.md before launching."
  exit 1
fi

export LD_PRELOAD="${HOME}/.local/lib/libproxychains4.so"
export PROXYCHAINS_CONF="${CONF_PATH}"


