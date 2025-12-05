#!/bin/bash
#
# Manual upload script for HuggingFace with CLI arguments
#
# Usage:
#   bash manual_upload.sh <checkpoint_dir> <repo_name>
#
# Examples:
#   bash manual_upload.sh /path/to/checkpoint DCAgent2/my-model
#   bash manual_upload.sh /path/to/checkpoint/global_step_2/policy DCAgent2/my-model
#

set -e  # Exit on error

# Parse arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint_dir> <repo_name>"
    echo ""
    echo "Arguments:"
    echo "  checkpoint_dir  - Path to checkpoint directory (can be export path or policy dir)"
    echo "  repo_name       - HuggingFace repo (e.g., DCAgent2/my-model)"
    echo ""
    echo "Examples:"
    echo "  $0 /scratch/user/exports/my-run DCAgent2/my-model"
    echo "  $0 /scratch/user/exports/my-run/global_step_2/policy DCAgent2/my-model"
    exit 1
fi

CHECKPOINT_DIR="$1"
REPO_NAME="$2"
TEMP_DIR="/tmp/hf_upload_$$"

# Validate inputs
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "✗ Error: Checkpoint directory does not exist: $CHECKPOINT_DIR"
    exit 1
fi

if [ -z "$REPO_NAME" ]; then
    echo "✗ Error: Repository name cannot be empty"
    exit 1
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "⚠ Warning: HF_TOKEN environment variable not set"
    echo "  You may need to set it for authentication:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
fi

echo "=================================================="
echo "Manual HuggingFace Upload (Filtered)"
echo "=================================================="
echo "Source: $CHECKPOINT_DIR"
echo "Destination: $REPO_NAME"
echo "=================================================="

# Auto-detect checkpoint structure
# Check if this is already a policy directory or if we need to find global_step_*
if [ -f "$CHECKPOINT_DIR/config.json" ] || [ -f "$CHECKPOINT_DIR/model.safetensors.index.json" ]; then
    # Already in policy directory
    POLICY_DIR="$CHECKPOINT_DIR"
    echo "✓ Using provided directory (contains model files)"
elif [ -d "$CHECKPOINT_DIR/policy" ]; then
    # policy subdirectory exists
    POLICY_DIR="$CHECKPOINT_DIR/policy"
    echo "✓ Found policy subdirectory: $POLICY_DIR"
else
    # Search for global_step_* directories
    echo "Searching for global_step_* directories..."
    LATEST_STEP=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -1)
    
    if [ -z "$LATEST_STEP" ]; then
        echo "✗ Error: No global_step_* directories found in $CHECKPOINT_DIR"
        echo "  Directory contents:"
        ls -la "$CHECKPOINT_DIR" | head -10
        exit 1
    fi
    
    echo "✓ Found latest checkpoint: $LATEST_STEP"
    
    # Check for policy subdirectory
    if [ -d "$LATEST_STEP/policy" ]; then
        POLICY_DIR="$LATEST_STEP/policy"
        echo "✓ Found policy directory: $POLICY_DIR"
    else
        POLICY_DIR="$LATEST_STEP"
        echo "✓ Using checkpoint root: $POLICY_DIR"
    fi
fi

echo ""

# Create temp directory for HF files only
mkdir -p "$TEMP_DIR"

# Copy ONLY HuggingFace format files (exclude FSDP checkpoints)
echo "Copying HuggingFace format files from: $POLICY_DIR"
echo ""
cd "$POLICY_DIR"

# Count files
TOTAL_FILES=$(ls -1 | wc -l)
echo "Total files in source: $TOTAL_FILES"
echo ""

# Copy safetensors models
cp model-*.safetensors "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied model shards" || echo "  ⊘ No model shards found"
cp model.safetensors.index.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied model index" || echo "  ⊘ No model index found"

# Copy config files
cp config.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied config.json" || echo "  ⊘ No config.json found"
cp generation_config.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied generation_config.json" || true
cp fsdp_config.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied fsdp_config.json" || true

# Copy tokenizer files
cp tokenizer.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied tokenizer.json" || echo "  ⊘ No tokenizer.json found"
cp tokenizer_config.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied tokenizer_config.json" || true
cp special_tokens_map.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied special_tokens_map.json" || true
cp added_tokens.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied added_tokens.json" || true
cp vocab.json "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied vocab.json" || true
cp merges.txt "$TEMP_DIR/" 2>/dev/null && echo "  ✓ Copied merges.txt" || true

echo ""

# Check if we copied anything
COPIED_FILES=$(ls -1 "$TEMP_DIR" 2>/dev/null | wc -l)
if [ "$COPIED_FILES" -eq 0 ]; then
    echo "✗ Error: No HuggingFace format files found to upload!"
    echo "  Make sure the directory contains .safetensors, config.json, etc."
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "Files prepared in: $TEMP_DIR"
echo "Files to upload: $COPIED_FILES"
echo "Total size: $(du -sh $TEMP_DIR | cut -f1)"
echo ""

# Show what will be uploaded
echo "Files being uploaded:"
ls -lh "$TEMP_DIR" | tail -n +2 | awk '{print "  - " $9 " (" $5 ")"}'
echo ""

# Upload using Python
python3 << UPLOAD_SCRIPT
from huggingface_hub import HfApi, login
import os
import sys

token = os.environ.get("HF_TOKEN")
repo_name = "$REPO_NAME"
temp_dir = "$TEMP_DIR"

try:
    if token:
        login(token=token)
        print("✓ Logged in to HuggingFace")
    else:
        print("⚠ No HF_TOKEN set, attempting to use cached credentials")
    
    api = HfApi()
    
    # Create repo
    api.create_repo(repo_id=repo_name, exist_ok=True)
    print(f"✓ Repository ready: {repo_name}")
    
    # Upload
    print("\nUploading to HuggingFace...")
    print("This may take a few minutes depending on file size...")
    
    api.upload_folder(
        folder_path=temp_dir,
        repo_id=repo_name,
        commit_message="Upload HuggingFace format model (FSDP checkpoints excluded)"
    )
    
    print("\n" + "=" * 60)
    print("✓ Upload completed successfully!")
    print(f"✓ Model available at: https://huggingface.co/{repo_name}")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ Upload failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

UPLOAD_SCRIPT

UPLOAD_STATUS=$?

# Cleanup
rm -rf "$TEMP_DIR"
echo ""
echo "✓ Cleaned up temporary directory"

if [ $UPLOAD_STATUS -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "SUCCESS!"
    echo "=================================================="
    echo "Repository: https://huggingface.co/$REPO_NAME"
    exit 0
else
    echo ""
    echo "=================================================="
    echo "FAILED - See error above"
    echo "=================================================="
    exit 1
fi