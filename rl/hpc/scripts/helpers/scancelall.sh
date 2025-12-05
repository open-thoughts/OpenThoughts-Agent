#!/bin/bash
# Cancel all pending and running jobs for the user

echo "=== Current Jobs ==="
squeue -u $USER --format="%.10i %.9P %.40j %.8u %.1T %.8M %.9l %.6D %.15r"

echo ""
echo "Cancelling all jobs for user $USER..."
scancel -u $USER

echo "Done."
