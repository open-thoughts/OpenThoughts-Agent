#!/usr/bin/env bash
# Reproducible setup: create eval venv with vLLM + Harbor on Hazy from a
# pinned lockfile.
#
# Usage:
#   ./modules/OpenThoughts-Agent/eval/hazy/setup_env.sh
#
# To regenerate the lockfile after intentionally bumping deps, run:
#   VIRTUAL_ENV=/data/home/macs/eval/venv uv pip freeze --exclude-editable \
#     > modules/OpenThoughts-Agent/eval/hazy/requirements.lock.txt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
# Eval artifacts root. Default: <repo>/eval. Override with HARBOR_EVAL_ROOT to
# put venv/jobs/logs/cache outside the repo on a different cluster.
EVAL_ROOT="${HARBOR_EVAL_ROOT:-$REPO_ROOT/eval}"
VENV_DIR="$EVAL_ROOT/venv"
LOCK_FILE="$SCRIPT_DIR/requirements.lock.txt"

if [ ! -f "$LOCK_FILE" ]; then
    echo "ERROR: lockfile not found at $LOCK_FILE"
    exit 1
fi

echo "Creating eval environment at $VENV_DIR"
echo "Using lockfile: $LOCK_FILE"

# Create directory structure
mkdir -p "$EVAL_ROOT"/{datasets,jobs,logs,cache/vllm}

# Create venv (Python 3.12 to match the lockfile)
uv venv "$VENV_DIR" --python 3.12

# Install all transitive deps from the pinned lockfile.
# CUDA 12.8 wheels live on the PyTorch index — needed for torch>=2.10.
uv pip install --python "$VENV_DIR/bin/python" \
    -r "$LOCK_FILE" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install Harbor as an editable install from the submodule. Editable so future
# submodule edits propagate without reinstall, and --no-deps because the
# transitive deps are already pinned by the lockfile above.
uv pip install --python "$VENV_DIR/bin/python" \
    --no-deps \
    -e "$REPO_ROOT/modules/harbor"

echo ""
echo "Done. Verify with:"
echo "  $VENV_DIR/bin/python -c 'import vllm; print(vllm.__version__)'"
echo "  $VENV_DIR/bin/python -c 'import harbor; print(harbor.__file__)'  # should point at modules/harbor/src"
echo "  $VENV_DIR/bin/harbor --help"
