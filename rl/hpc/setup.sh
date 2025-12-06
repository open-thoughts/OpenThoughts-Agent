#!/bin/bash
# Setup script for OT-Agent HPC system

echo "=== OT-Agent HPC Setup ==="

# Check if we're in the right directory
if [ ! -f "hpc/launch.py" ]; then
    echo "Error: Please run this script from the OpenThoughts-Agent/train directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

source hpc/scripts/common.sh  # Load common functions

# TODO(Charlie) First time running it will exit because in `common.sh` we do `exec newgrp`
# Only second time running will go pass this line. Fix this.

# Detect cluster and load appropriate environment
echo "Detecting cluster..."
if [[ $(hostname) == *"tacc"* ]]; then
    echo "Detected TACC cluster"
    source hpc/dotenv/tacc.env
elif [[ $(hostname) == *"jureca"* ]] || [[ $(hostname) == *"jupiter"* ]] || [[ $(hostname) == *"juwels"* ]]; then
    echo "Detected JSC cluster"
    source hpc/dotenv/jsc.env
else
    echo "Warning: Unknown cluster, using TACC defaults"
    source hpc/dotenv/tacc.env
fi

pip install huggingface_hub datasets Jinja2
source hpc/scripts/common.sh  # Load common functions

echo "✓ Environment loaded"
# Test the setup
echo "Testing HPC system..."
python3 hpc/test_hpc.py

echo ""
echo "=== Setup Complete ==="
echo "You can now use the following commands:"
echo "  status             - Check job status"
echo "  sfail              - Show failed jobs"
echo "  scompleted         - Show completed jobs"
echo "  scancelled         - Show cancelled jobs"
echo "  scancelall         - Cancel all jobs"
echo "  rmlogs             - Remove old log files"
echo "  sinf               - Show cluster information"
echo "  sqme               - Show your queued jobs"
echo "  sqteam             - Show team jobs"
echo "  sqthem <user>      - Show specific user's jobs"
echo ""
echo "For more information, see hpc/README.md"
