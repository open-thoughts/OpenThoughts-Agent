#!/bin/bash

# Source the original env file to load all variables
source /Users/ryan/dcft_private/hpc/dotenv/zih.env

# Output the expanded variables to a new file
cat > /Users/ryan/dcft_private/hpc/dotenv/zih.env.expanded << EOF
export DCFT=$DCFT
export DCFT_CONDA=$DCFT_CONDA
export DCFT_PRIVATE=$DCFT_PRIVATE
export DCFT_PRIVATE_ENV=$DCFT_PRIVATE_ENV
export DCFT_PRIVATE_ACTIVATE_ENV="$DCFT_PRIVATE_ACTIVATE_ENV"
export EVALCHEMY=$EVALCHEMY
export EVALCHEMY_ENV=$EVALCHEMY_ENV
export HF_HUB_CACHE=$HF_HUB_CACHE
export CHECKPOINTS_DIR=$CHECKPOINTS_DIR
export MODELS_DIR=$MODELS_DIR
export DATASETS_DIR=$DATASETS_DIR
EOF

echo "Expanded variables written to zih.env.expanded"
