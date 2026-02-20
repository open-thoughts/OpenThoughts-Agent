#!/bin/bash
# fix_permissions.sh - Set safe permissions on a directory tree
#
# Makes all files readable by everyone, writable only by owner,
# and preserves execute permissions where needed:
# - bin/ directories (scripts and binaries)
# - ELF executables (detected by file header)
# - Shell scripts with shebang
# - Known binary locations (ray/core, etc.)
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

echo "  [1/5] Setting directories to 755 (rwxr-xr-x)..."
find "$TARGET_DIR" -type d -exec chmod 755 {} +

echo "  [2/5] Setting files to 644 (rw-r--r--)..."
find "$TARGET_DIR" -type f -exec chmod 644 {} +

echo "  [3/5] Setting executables in bin/ to 755 (rwxr-xr-x)..."
find "$TARGET_DIR" -type f -path "*/bin/*" -exec chmod 755 {} +

echo "  [4/5] Setting ELF binaries to 755 (rwxr-xr-x)..."
# Find ELF executables by checking file header (first 4 bytes = 0x7f ELF)
find "$TARGET_DIR" -type f -exec sh -c '
    for f; do
        # Check if file starts with ELF magic bytes
        if head -c 4 "$f" 2>/dev/null | grep -q "^.ELF"; then
            chmod 755 "$f"
        fi
    done
' _ {} +

echo "  [5/5] Setting shell scripts with shebang to 755..."
# Find files starting with #! (shebang)
find "$TARGET_DIR" -type f -exec sh -c '
    for f; do
        if head -c 2 "$f" 2>/dev/null | grep -q "^#!"; then
            chmod 755 "$f"
        fi
    done
' _ {} +

echo "Done."
