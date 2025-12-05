#!/bin/bash
# Flatten task directory structure for sandboxes compatibility
#
# Transforms:
#   tasks/task_0/nl2bash-0-0000/ -> tasks_flat/nl2bash-0-0000/
#   tasks/task_1/nl2bash-1-0000/ -> tasks_flat/nl2bash-1-0000/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use provided run directory or default to local_runs
RUN_DIR="$1"

TASKS_DIR="$RUN_DIR/tasks"
FLAT_DIR="$RUN_DIR/tasks_flat"

echo "Flattening task directory structure..."
echo "Source: $TASKS_DIR"
echo "Target: $FLAT_DIR"
echo ""

# Create flat directory if it doesn't exist
mkdir -p "$FLAT_DIR"

# Counter for valid tasks
VALID_COUNT=0
INVALID_COUNT=0

# Process each task_* directory
for task_wrapper in "$TASKS_DIR"/task_*/; do
    if [ ! -d "$task_wrapper" ]; then
        continue
    fi
    
    # Find the nl2bash-* subdirectory
    for task_dir in "$task_wrapper"nl2bash-*/; do
        if [ ! -d "$task_dir" ]; then
            continue
        fi
        
        task_name=$(basename "$task_dir")
        
        # Check if it's a valid task (has task.toml)
        if [ -f "$task_dir/task.toml" ]; then
            # Create symlink in flat directory
            ln -sfn "$(cd "$task_dir" && pwd)" "$FLAT_DIR/$task_name"
            ((VALID_COUNT++))
            echo "✓ Linked: $task_name"
        else
            ((INVALID_COUNT++))
            echo "✗ Skipped (no task.toml): $task_name"
        fi
    done
done

echo ""
echo "Summary:"
echo "  Valid tasks linked: $VALID_COUNT"
echo "  Invalid tasks skipped: $INVALID_COUNT"
echo ""
echo "Flattened tasks directory: $FLAT_DIR"

