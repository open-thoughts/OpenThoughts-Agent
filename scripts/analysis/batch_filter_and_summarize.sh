#!/usr/bin/env bash
# Loop over subdirectories in a root folder, run filter_latest_episodes.py on each,
# and then summarize all resulting JSONL files.
#
# Usage:
#   ./scripts/analysis/batch_filter_and_summarize.sh \
#     /scratch/.../to_upload \
#     ../eval-jsonl
#
# Defaults if not provided:
#   ROOT=/scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/jobs/to_upload
#   OUT_DIR=../eval-jsonl

set -euo pipefail

ROOT=${1:-/scratch/08134/negin/dc-agent-shared/dc-agent/eval/tacc/jobs/to_upload}
OUT_DIR=${2:-../eval-jsonl}

mkdir -p "$OUT_DIR"

echo "[batch] Root: $ROOT"
echo "[batch] Output dir: $OUT_DIR"

for d in "$ROOT"/*/; do
  [ -d "$d" ] || continue
  name="$(basename "$d")"
  # Adjust mapping to repo_id here if needed
  repo_id="$name"
  # Example alternative if your folder name encodes org/repo as org_repo:
  # repo_id="${name/_//}"

  out_jsonl="$OUT_DIR/$name.jsonl"
  echo "[batch] Processing (local job dir): $name"
  python scripts/analysis/filter_latest_episodes.py --job-dir "$d" --output-jsonl "$out_jsonl" || {
    echo "[warn] Failed to process $name" >&2
    continue
  }
done

echo "[batch] Summarizing JSONL files in $OUT_DIR"
for f in "$OUT_DIR"/*.jsonl; do
  [ -f "$f" ] || continue
  echo "[summary] $f"
  python scripts/analysis/summarize_conversations.py "$f" || true
done

echo "[batch] Done."
