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

# Ensure all ancestor directories up to / are traversable (o+x).
# Without this, other users can't reach the target even if it's 755.
echo "  [0/5] Ensuring parent directories are traversable (o+x)..."
_dir="$TARGET_DIR"
while [[ "$_dir" != "/" ]]; do
    _dir="$(dirname "$_dir")"
    # Only fix dirs owned by us — don't touch system dirs
    if [[ -O "$_dir" ]]; then
        _perms=$(stat -c '%a' "$_dir" 2>/dev/null || stat -f '%Lp' "$_dir" 2>/dev/null)
        if [[ $(( 0$_perms & 0005 )) -eq 0 ]]; then
            echo "    Setting $_dir to 755 (was $_perms)"
            chmod 755 "$_dir"
        fi
    fi
done

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
