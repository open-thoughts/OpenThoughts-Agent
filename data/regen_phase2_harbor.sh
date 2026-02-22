#!/bin/bash
# Phase 2: Run harbor on all generated datasets (4 at a time)
source /scratch/10000/eguha3/old-dc-agent/secret.env

MANIFEST="data/regen_results/manifest.json"
MAX_CONCURRENT=4

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: No manifest found at $MANIFEST. Run phase1 first."
    exit 1
fi

# Read manifest and launch harbor jobs
PIDS=()
NAMES=()
RUNNING=0

launch_harbor() {
    local name="$1"
    local task_dir="$2"
    local job_name="regen_${name}"

    # Skip if already done
    if [ -f "jobs/${job_name}/result.json" ]; then
        echo "[$(date +%H:%M:%S)] SKIP $name (already has results)"
        return
    fi

    # Remove partial job dir
    rm -rf "jobs/${job_name}" 2>/dev/null

    echo "[$(date +%H:%M:%S)] LAUNCH $name -> $task_dir"
    harbor jobs start -p "$task_dir" \
        --n-concurrent 5 \
        --agent terminus-2 \
        --model openai/gpt-5-nano-2025-08-07 \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config eval/tacc/dcagent_eval_config.yaml \
        > "data/regen_results/${name}_harbor.log" 2>&1 &

    PIDS+=($!)
    NAMES+=("$name")
    RUNNING=$((RUNNING + 1))
}

wait_for_slot() {
    while [ $RUNNING -ge $MAX_CONCURRENT ]; do
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] DONE ${NAMES[$i]} (PID ${PIDS[$i]})"
                unset PIDS[$i]
                unset NAMES[$i]
                RUNNING=$((RUNNING - 1))
            fi
        done
        # Re-index arrays
        PIDS=("${PIDS[@]}")
        NAMES=("${NAMES[@]}")
        if [ $RUNNING -ge $MAX_CONCURRENT ]; then
            sleep 10
        fi
    done
}

# Parse manifest and launch jobs
python3 << 'PYEOF' | while IFS='|' read -r name task_dir; do
import json
with open("data/regen_results/manifest.json") as f:
    manifest = json.load(f)
for name, info in manifest.items():
    task_dir = info.get("task_dir", "FAILED")
    if task_dir != "FAILED":
        print(f"{name}|{task_dir}")
PYEOF
    wait_for_slot
    launch_harbor "$name" "$task_dir"
done

# Wait for remaining jobs
echo "[$(date +%H:%M:%S)] Waiting for remaining ${#PIDS[@]} jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null
done

echo ""
echo "====== All harbor jobs complete ======"

# Collect results
echo ""
echo "====== RESULTS ======"
python3 << 'PYEOF'
import json, os

OLD_RATES = {
    "bigcodebench": 60, "codeelo": 84, "exercism_python": 80,
    "swebench": 64, "manybugs": 60, "crosscodeeval_python": 60,
    "bugsinpy_mf": 52, "bugswarm": 72, "codenet_python": 92,
    "codereval_python": 52, "defects4j": 48, "methods2test": 92,
    "bugsinpy": 40, "taco": 72, "unitsyn_python": 80,
    "softwareheritage": 72, "e2egit": 56, "ghactions": 84,
    "pymethods2test": 76,
}

with open("data/regen_results/manifest.json") as f:
    manifest = json.load(f)

results = {}
print(f"{'Dataset':<25} {'Old':>5} {'New':>5} {'Diff':>6} {'Errors':>7} {'Status'}")
print("-" * 70)

for name, info in manifest.items():
    job_name = f"regen_{name}"
    result_path = f"jobs/{job_name}/result.json"
    old_rate = OLD_RATES.get(name, -1)

    if not os.path.exists(result_path):
        print(f"{name:<25} {old_rate:>4}% {'N/A':>5} {'N/A':>6} {'N/A':>7} HARBOR_FAILED")
        results[name] = {"status": "HARBOR_FAILED"}
        continue

    with open(result_path) as f:
        data = json.load(f)
    stats = data.get("stats", {})
    new_rate = None
    for k, v in stats.get("evals", {}).items():
        metrics = v.get("metrics", [])
        if metrics:
            new_rate = metrics[0].get("mean", 0) * 100
            break

    n_errors = stats.get("n_errors", 0)
    n_trials = stats.get("n_trials", 0)

    if new_rate is not None:
        diff = new_rate - old_rate if old_rate >= 0 else 0
        status = "GOOD" if abs(diff) < 25 else "STILL_BAD"
        print(f"{name:<25} {old_rate:>4}% {new_rate:>4.0f}% {diff:>+5.0f}% {n_errors:>3}/{n_trials:<3} {status}")
        results[name] = {"status": status, "new_rate": new_rate, "old_rate": old_rate, "errors": n_errors, "trials": n_trials}
    else:
        print(f"{name:<25} {old_rate:>4}% {'ERR':>5} {'N/A':>6} {n_errors:>3}/{n_trials:<3} PARSE_ERROR")
        results[name] = {"status": "PARSE_ERROR"}

with open("data/regen_results/final_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to data/regen_results/final_results.json")
PYEOF
