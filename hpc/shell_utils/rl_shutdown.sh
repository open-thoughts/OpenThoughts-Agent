#!/bin/bash

forward_termination_to_rl_runner() {
  local runner_pid="${1:-}"
  if [[ -z "$runner_pid" ]] || ! kill -0 "$runner_pid" 2>/dev/null; then
    return 1
  fi
  kill -TERM "$runner_pid"
}
