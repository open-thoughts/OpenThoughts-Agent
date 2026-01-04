#!/bin/bash

# Lightweight check that proxychains-ng is installed and environment is wired.
# Installation itself is a one-off step; see JSC_setup/README.md in dc-agent.

if ! command -v proxychains4 >/dev/null 2>&1; then
  echo "[proxychains] proxychains4 not found in PATH."
  echo "[proxychains] Please install proxychains-ng as described in dc-agent/JSC_setup/README.md."
  exit 1
fi

export LD_PRELOAD="${HOME}/.local/lib/libproxychains4.so"
# Some scripts generate a per-job config and pass it via `-f ...`.
# If the default config exists, expose it via the standard env var as a convenience.
CONF_PATH="${HOME}/.proxychains/proxychains.conf"
if [ -f "${CONF_PATH}" ]; then
  export PROXYCHAINS_CONF_FILE="${CONF_PATH}"
fi


