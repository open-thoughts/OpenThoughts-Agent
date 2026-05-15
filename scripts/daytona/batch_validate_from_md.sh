#!/usr/bin/env bash
# Batch-run validate_and_upload_from_hf.py (Harbor smoke-test stage only) over
# every HF dataset listed in an input markdown file. Each line in the file may
# be either a huggingface.co/datasets/<org>/<name> URL or a bare <org>/<name>
# (one repo per line, optionally prefixed with `- `/`* `/`+ `). For each dataset:
#   - take a random sample of N tasks (default 500)
#   - run the Harbor smoke test
#   - skip the HF upload
#   - copy the staged successes + failure dirs under a per-dataset folder
#   - record success/fail counts in a summary TSV
#
# Usage:
#   batch_validate_from_md.sh [INPUT_MD] [SECRETS_ENV] [TARGET_DIR]
#
# Defaults:
#   INPUT_MD     /Users/benjaminfeuer/Documents/notes/task_repos/rl_to_check.md
#   SECRETS_ENV  /Users/benjaminfeuer/Documents/secrets.env
#   TARGET_DIR   /Users/benjaminfeuer/Documents/agent-traces-analysis
#
# Env overrides:
#   SAMPLE_SIZE  (default 200)
#   SAMPLE_SEED  (default 42)
#   PYTHON       (default /Users/benjaminfeuer/miniconda3/envs/otagent/bin/python)
#   VALIDATE_PY  (default sibling validate_and_upload_from_hf.py)
#
# Per-dataset artifacts written under TARGET_DIR/<org>__<name>/:
#   run.log      — full validator stdout (noisy: includes 5+ Hz progress refreshes)
#   clean.log    — same, minus progress-bar refreshes and ANSI escapes
#   metrics.tsv  — single-row TSV with the parsed Harbor metrics
#   traces/      — Harbor agent traces synced from /var/folders/…/harbor_jobs_*/harbor-validate-*/
#                  (per-trial dirs with agent/trajectory.json, verifier/, result.json, …)
#   failures/    — copies of tasks Harbor rejected (per-stage subdirs; only present when n_fail > 0)

set -euo pipefail

INPUT_MD="${1:-/Users/benjaminfeuer/Documents/notes/task_repos/rl_to_check.md}"
SECRETS_ENV="${2:-/Users/benjaminfeuer/Documents/secrets.env}"
TARGET_DIR="${3:-/Users/benjaminfeuer/Documents/agent-traces-analysis}"

SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
PYTHON="${PYTHON:-/Users/benjaminfeuer/miniconda3/envs/otagent/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATE_PY="${VALIDATE_PY:-$SCRIPT_DIR/validate_and_upload_from_hf.py}"

[[ -f "$INPUT_MD"   ]] || { echo "Input MD not found: $INPUT_MD"   >&2; exit 1; }
[[ -f "$SECRETS_ENV" ]] || { echo "Secrets file not found: $SECRETS_ENV" >&2; exit 1; }
[[ -f "$VALIDATE_PY" ]] || { echo "Validator not found: $VALIDATE_PY"    >&2; exit 1; }
[[ -x "$PYTHON"      ]] || { echo "Python not executable: $PYTHON"        >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$SECRETS_ENV"
set +a

mkdir -p "$TARGET_DIR"
SUMMARY="$TARGET_DIR/summary.tsv"
EXPECTED_HEADER=$'dataset\ttotal\tinfra_ok\tinfra_rate\tsolved\tsolve_rate'
if [[ ! -s "$SUMMARY" ]]; then
    printf "%s\n" "$EXPECTED_HEADER" > "$SUMMARY"
elif [[ "$(head -1 "$SUMMARY")" != "$EXPECTED_HEADER" ]]; then
    # Existing file has stale header (older 4-col schema). Rotate it out so we don't keep
    # appending mismatched rows.
    mv "$SUMMARY" "${SUMMARY%.tsv}.stale_$(date +%Y%m%d_%H%M%S).tsv"
    printf "%s\n" "$EXPECTED_HEADER" > "$SUMMARY"
    echo "Rotated stale-schema summary.tsv → ${SUMMARY%.tsv}.stale_$(date +%Y%m%d_%H%M%S).tsv"
fi

# Extract HF dataset repos from $INPUT_MD. Two branches feed into a dedup pass:
#   1. huggingface.co/datasets/<org>/<name> URLs — sed strips the prefix.
#   2. bare <org>/<name> lines, optionally prefixed with a markdown list bullet
#      "- ", "* ", or "+ ". Anything containing "://", spaces, or extra "/"s
#      is excluded so file paths and URLs do not leak in.
# Each branch is `|| true`-wrapped: under `set -euo pipefail`, a grep with no
# matches exits 1 and would otherwise abort the whole extraction.
extract_repos() {
    {
        { grep -oE 'huggingface\.co/datasets/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+' "$INPUT_MD" \
            | sed 's|huggingface\.co/datasets/||'; } || true
        { sed -E 's/^[[:space:]]*[-*+]?[[:space:]]*//; s/[[:space:]]*$//' "$INPUT_MD" \
            | grep -E '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$'; } || true
    } | awk '!seen[$0]++'
}

REPOS=()
while IFS= read -r line; do
    [[ -n "$line" ]] && REPOS+=("$line")
done < <(extract_repos)

if [[ ${#REPOS[@]} -eq 0 ]]; then
    echo "No HF dataset repos found in $INPUT_MD" >&2
    echo "Accepted formats per line: a huggingface.co/datasets/<org>/<name> URL, or a bare <org>/<name>." >&2
    exit 1
fi

echo "Found ${#REPOS[@]} dataset(s); sample_size=$SAMPLE_SIZE seed=$SAMPLE_SEED"
echo "Target dir: $TARGET_DIR"
echo

for REPO in "${REPOS[@]}"; do
    SAFE="${REPO//\//__}"
    DEST="$TARGET_DIR/$SAFE"
    mkdir -p "$DEST"
    LOG="$DEST/run.log"

    echo "=== $REPO ==="

    rc=0
    "$PYTHON" "$VALIDATE_PY" \
        --repo_id "$REPO" \
        --stages harbor \
        --sample_size "$SAMPLE_SIZE" \
        --sample_seed "$SAMPLE_SEED" \
        --skip_upload \
        --keep_failed_dir "$DEST/failures" \
        > >(tee "$LOG") 2>&1 || rc=$?

    # Strip ANSI escapes + convert CR-overwrites into newlines, then drop progress-bar refreshes.
    # rich.Progress emits one line per refresh ending in CR (`\r`) and uses the "task/s" rate column,
    # so filtering on "task/s" is a robust way to remove only progress noise.
    CLEAN="$DEST/clean.log"
    perl -pe 's/\e\[[0-9;]*[a-zA-Z]//g; s/\r/\n/g' "$LOG" \
        | grep -v -E 'task/s|Sampling params probe OK' \
        | awk 'NF || (prev_blank=0)' > "$CLEAN" 2>/dev/null || true

    if [[ $rc -ne 0 ]]; then
        echo "  -> validator exited $rc (see $CLEAN)"
        printf "%s\tERROR\tERROR\tERROR\tERROR\tERROR\n" "$REPO" >> "$SUMMARY"
        continue
    fi

    # Sync Harbor's per-trial trace dir (agent/trajectory.json, verifier/, result.json, …)
    # into <DEST>/traces/. The validate.py prints the path on its `[harbor]` retention line;
    # we extract from the ANSI-stripped clean.log.
    HARBOR_DIR=$(grep -oE '/(var/folders|tmp)/[^[:space:]]+/harbor_jobs_[A-Za-z0-9_]+/harbor-validate-[0-9-]+' "$CLEAN" 2>/dev/null | tail -1 || true)
    if [[ -n "$HARBOR_DIR" && -d "$HARBOR_DIR" ]]; then
        mkdir -p "$DEST/traces"
        # rsync is fast and handles large per-trial dirs well; --remove-source-files frees
        # the /tmp/var/folders space afterwards (the trace dir is no longer needed locally).
        rsync -a "$HARBOR_DIR/" "$DEST/traces/" 2>/dev/null
        rm -rf "$HARBOR_DIR" 2>/dev/null
    else
        echo "  -> WARNING: could not locate Harbor traces dir (looked for /var/folders/…/harbor_jobs_*/harbor-validate-* in $CLEAN)"
    fi

    # Match the metrics line whether or not rich emitted the `[harbor]` markup. The substantive
    # signature is `infra_ok: N/N` and `solved: N/N` co-occurring on a single line.
    line=$(grep -E 'infra_ok: [0-9]+/[0-9]+.*solved: [0-9]+/[0-9]+' "$CLEAN" | tail -1 || true)
    if [[ -n "$line" ]]; then
        infra_ok=$(echo "$line" | sed -nE 's|.*infra_ok: ([0-9]+)/[0-9]+.*|\1|p')
        total=$(echo "$line"    | sed -nE 's|.*infra_ok: [0-9]+/([0-9]+).*|\1|p')
        solved=$(echo "$line"   | sed -nE 's|.*solved: ([0-9]+)/[0-9]+.*|\1|p')
        if [[ ${total:-0} -gt 0 ]]; then
            infra_rate=$(awk -v s="$infra_ok" -v t="$total" 'BEGIN{printf "%.3f", s/t}')
            solve_rate=$(awk -v s="$solved"   -v t="$total" 'BEGIN{printf "%.3f", s/t}')
        else
            infra_rate="N/A"; solve_rate="N/A"
        fi
        printf "  -> n=%s infra_ok=%s (%s)  solved=%s (%s)\n" \
            "$total" "$infra_ok" "$infra_rate" "$solved" "$solve_rate"
        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$REPO" "$total" "$infra_ok" "$infra_rate" "$solved" "$solve_rate" >> "$SUMMARY"
        {
            printf "dataset\ttotal\tinfra_ok\tinfra_rate\tsolved\tsolve_rate\n"
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
                "$REPO" "$total" "$infra_ok" "$infra_rate" "$solved" "$solve_rate"
        } > "$DEST/metrics.tsv"
    else
        echo "  -> could not parse Harbor metrics line (see $CLEAN)"
        printf "%s\tPARSE_FAIL\tPARSE_FAIL\tPARSE_FAIL\tPARSE_FAIL\tPARSE_FAIL\n" "$REPO" >> "$SUMMARY"
        {
            printf "dataset\ttotal\tinfra_ok\tinfra_rate\tsolved\tsolve_rate\n"
            printf "%s\tPARSE_FAIL\tPARSE_FAIL\tPARSE_FAIL\tPARSE_FAIL\tPARSE_FAIL\n" "$REPO"
        } > "$DEST/metrics.tsv"
    fi
    echo
done

echo "Done. Summary at $SUMMARY"
echo
column -t -s $'\t' "$SUMMARY"
