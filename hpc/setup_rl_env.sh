#!/bin/bash
# =============================================================================
# RL Environment Setup Script
# =============================================================================
# Creates a standalone Python 3.12 virtual environment for SkyRL training.
# This environment is separate from the main project environment due to
# dependency conflicts between RL (torch 2.8, vllm 0.11.0) and datagen
# (torch 2.9, vllm 0.11.2).
#
# Usage:
#   ./hpc/setup_rl_env.sh [--force]
#
# Options:
#   --force    Remove existing RL environment and recreate
#
# The environment is created at: $DCFT/envs/rl or ./envs/rl
# The RL launcher (hpc/launch.py --job_type rl) will automatically use this.
# =============================================================================

set -euo pipefail

# Configuration
RL_ENV_NAME="rl"
PYTHON_VERSION="3.12"

# Determine base directory (project root with pyproject.toml)
# DCFT_PRIVATE is the project dir on clusters where DCFT is a parent scratch dir
if [[ -n "${DCFT_PRIVATE:-}" ]]; then
    BASE_DIR="$DCFT_PRIVATE"
elif [[ -n "${DCFT:-}" ]]; then
    BASE_DIR="$DCFT"
else
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

RL_ENV_DIR="$BASE_DIR/envs/$RL_ENV_NAME"
RL_REQUIREMENTS="$BASE_DIR/hpc/rl_requirements.txt"

# Parse arguments
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--force]"
            exit 1
            ;;
    esac
done

echo "=== RL Environment Setup ==="
echo "Base directory: $BASE_DIR"
echo "Environment directory: $RL_ENV_DIR"
echo "Python version: $PYTHON_VERSION"
echo ""

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Install it with: pip install uv"
    exit 1
fi

# Check Python 3.12 availability
if ! uv python find "$PYTHON_VERSION" &> /dev/null; then
    echo "Python $PYTHON_VERSION not found. Attempting to install..."
    uv python install "$PYTHON_VERSION"
fi

# Handle existing environment
if [[ -d "$RL_ENV_DIR" ]]; then
    if [[ "$FORCE" == "true" ]]; then
        echo "Removing existing environment (--force)..."
        rm -rf "$RL_ENV_DIR"
    else
        echo "RL environment already exists at: $RL_ENV_DIR"
        echo "Use --force to recreate, or activate with:"
        echo "  source $RL_ENV_DIR/bin/activate"
        exit 0
    fi
fi

# Create environment directory
mkdir -p "$(dirname "$RL_ENV_DIR")"

echo "Creating Python $PYTHON_VERSION virtual environment..."
uv venv "$RL_ENV_DIR" --python "$PYTHON_VERSION"

echo "Activating environment..."
source "$RL_ENV_DIR/bin/activate"

echo "Installing RL dependencies..."

# Install from requirements file if it exists, otherwise use pyproject.toml
if [[ -f "$RL_REQUIREMENTS" ]]; then
    echo "Using requirements file: $RL_REQUIREMENTS"
    uv pip install -r "$RL_REQUIREMENTS"
else
    echo "Using pyproject.toml [rl] extra..."
    # Install the project with rl extra
    # Note: We install in editable mode so changes to hpc/ are reflected
    uv pip install -e "$BASE_DIR[rl]"
fi

# =============================================================================
# Clone SkyRL repository (required for examples/terminal_bench entrypoints)
# =============================================================================
# The terminal_bench entrypoint lives in examples/, which is NOT part of the
# installed skyrl-train package. We need the full repo cloned.

SKYRL_REPO="https://github.com/penfever/SkyRL.git"
SKYRL_BRANCH="penfever/working"

# Determine SKYRL_HOME location
if [[ -n "${SKYRL_HOME:-}" ]]; then
    SKYRL_DIR="$SKYRL_HOME"
elif [[ -n "${SCRATCH:-}" ]]; then
    SKYRL_DIR="$SCRATCH/SkyRL"
else
    SKYRL_DIR="$BASE_DIR/SkyRL"
fi

echo ""
echo "=== SkyRL Repository Setup ==="
echo "Target directory: $SKYRL_DIR"

if [[ -d "$SKYRL_DIR" ]]; then
    echo "SkyRL repo already exists at: $SKYRL_DIR"
    echo "Updating to latest $SKYRL_BRANCH..."
    pushd "$SKYRL_DIR" > /dev/null
    git fetch origin
    git checkout "$SKYRL_BRANCH" 2>/dev/null || git checkout -b "$SKYRL_BRANCH" "origin/$SKYRL_BRANCH"
    git pull origin "$SKYRL_BRANCH" || echo "Warning: Could not pull latest changes"
    popd > /dev/null
else
    echo "Cloning SkyRL from $SKYRL_REPO..."
    git clone --branch "$SKYRL_BRANCH" "$SKYRL_REPO" "$SKYRL_DIR"
fi

echo "SkyRL repo ready at: $SKYRL_DIR"

# =============================================================================
# Install SkyRL packages (skyrl-train and skyrl-gym)
# =============================================================================
# skyrl-train: Core RL training framework
# skyrl-gym: Environment implementations (GSM8K, AIME, etc.) - required by skyrl-train

echo ""
echo "=== Installing SkyRL Packages ==="

echo "Installing skyrl-gym (environment implementations)..."
uv pip install -e "$SKYRL_DIR/skyrl-gym"

echo "Installing skyrl-train (RL training framework)..."
uv pip install -e "$SKYRL_DIR/skyrl-train"

echo ""
echo "IMPORTANT: Add this to your shell or source your cluster's dotenv file:"
echo "  export SKYRL_HOME=\"$SKYRL_DIR\""
echo "  export PYTHONPATH=\"\$SKYRL_HOME/skyrl-train:\$PYTHONPATH\""

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')" || echo "Warning: vLLM not installed (may be CPU-only system)"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import skyrl_gym; print(f'skyrl-gym: installed')"
python -c "import skyrl_train; print(f'skyrl-train: installed')"

echo ""
echo "=== RL Environment Setup Complete ==="
echo ""
echo "Environment location: $RL_ENV_DIR"
echo ""
echo "To activate manually:"
echo "  source $RL_ENV_DIR/bin/activate"
echo ""
echo "The RL launcher will use this environment automatically when you run:"
echo "  python -m hpc.launch --job_type rl --rl_config terminal_bench.yaml ..."
echo ""

# Create activation helper script
ACTIVATE_SCRIPT="$BASE_DIR/hpc/activate_rl_env.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Source this script to activate the RL environment
# Usage: source hpc/activate_rl_env.sh

export RL_ENV_DIR="$RL_ENV_DIR"
source "\$RL_ENV_DIR/bin/activate"
echo "Activated RL environment: \$RL_ENV_DIR"
EOF
chmod +x "$ACTIVATE_SCRIPT"
echo "Created activation helper: $ACTIVATE_SCRIPT"
