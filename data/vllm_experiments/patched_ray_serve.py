#!/usr/bin/env python3
"""Wrapper to patch vLLM envs before running Ray Serve."""

import os
import sys

# Monkey-patch vLLM envs before any imports that might use it
import vllm.envs as vllm_envs

# Add the missing VLLM_USE_V1 attribute (set to False to use non-V1 engine)
if not hasattr(vllm_envs, 'VLLM_USE_V1'):
    vllm_envs.VLLM_USE_V1 = False
    print("[patched_ray_serve] Added VLLM_USE_V1 = False to vllm.envs")

# Now run the original script
sys.path.insert(0, os.path.expandvars('${DCAGENT_DIR}'))
from scripts.vllm.start_vllm_ray_serve import main

if __name__ == "__main__":
    main()
