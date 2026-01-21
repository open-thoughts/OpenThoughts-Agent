#!/bin/bash
# fix_permissions.sh - Set safe permissions on a directory tree
#
# Makes all files readable by everyone, writable only by owner,
# and preserves execute permissions only where needed (bin/ directories).
#
# Usage: ./fix_permissions.sh /path/to/directory

set -u

TARGET_DIR="$1"

if [[ -z "${TARGET_DIR:-}" ]]; then
    echo "Usage: $0 <target_directory>"
    echo "Example: $0 ./miniconda3"
    exit 1
fi

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: '$TARGET_DIR' is not a directory"
    exit 1
fi

echo "Fixing permissions for: $TARGET_DIR"

echo "  [1/3] Setting directories to 755 (rwxr-xr-x)..."
find "$TARGET_DIR" -type d -exec chmod 755 {} +

echo "  [2/3] Setting files to 644 (rw-r--r--)..."
find "$TARGET_DIR" -type f -exec chmod 644 {} +

echo "  [3/3] Setting executables in bin/ to 755 (rwxr-xr-x)..."
find "$TARGET_DIR" -type f -path "*/bin/*" -exec chmod 755 {} +

echo "Done."
