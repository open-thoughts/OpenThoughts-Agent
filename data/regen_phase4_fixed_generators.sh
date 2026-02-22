#!/bin/bash
# Phase 4: Regenerate tasks with fixed generators and test with harbor
set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
export SKIP_HF_UPLOAD=1

cd /scratch/10000/eguha3/dc-agent

echo "=== Phase 4: Regenerating with fixed generators ==="

# 1. pymethods2test
echo "[1/5] Regenerating pymethods2test..."
python3 data/dclm-mine/generate_pymethods2test_tasks.py --limit 25 2>&1 | tail -5
PYMETHODS_DIR=$(ls -td /scratch/10000/eguha3/data_cache/pymethods2test_tasks_* 2>/dev/null | head -1)
echo "  Dir: $PYMETHODS_DIR ($(ls -d "$PYMETHODS_DIR"/*/ 2>/dev/null | wc -l) tasks)"

# 2. unitsyn-python
echo "[2/5] Regenerating unitsyn-python..."
python3 data/dclm-mine/generate_unitsyn_tasks.py --limit 25 --lang python 2>&1 | tail -5
UNITSYN_DIR=$(ls -td /scratch/10000/eguha3/data_cache/unitsyn-python_tasks_* 2>/dev/null | head -1)
echo "  Dir: $UNITSYN_DIR ($(ls -d "$UNITSYN_DIR"/*/ 2>/dev/null | wc -l) tasks)"

# 3. codenet-python
echo "[3/5] Regenerating codenet-python..."
python3 data/dclm-mine/generate_codenet_tasks.py --limit 25 2>&1 | tail -5
CODENET_DIR=$(ls -td /scratch/10000/eguha3/data_cache/codenet-python_tasks_* 2>/dev/null | head -1)
echo "  Dir: $CODENET_DIR ($(ls -d "$CODENET_DIR"/*/ 2>/dev/null | wc -l) tasks)"

# 4. e2egit
echo "[4/5] Regenerating e2egit..."
python3 data/dclm-mine/generate_e2egit_tasks.py --limit 25 --skip-clone 2>&1 | tail -5
E2EGIT_DIR=$(ls -td /scratch/10000/eguha3/data_cache/e2egit_tasks_* 2>/dev/null | head -1)
echo "  Dir: $E2EGIT_DIR ($(ls -d "$E2EGIT_DIR"/*/ 2>/dev/null | wc -l) tasks)"

# 5. ghactions
echo "[5/5] Regenerating ghactions..."
python3 data/dclm-mine/generate_ghactions_tasks.py --limit 25 --skip-clone 2>&1 | tail -5
GHACTIONS_DIR=$(ls -td /scratch/10000/eguha3/data_cache/ghactions_tasks_* 2>/dev/null | head -1)
echo "  Dir: $GHACTIONS_DIR ($(ls -d "$GHACTIONS_DIR"/*/ 2>/dev/null | wc -l) tasks)"

echo ""
echo "=== Generation complete. Launching harbor tests ==="

# Clean old job dirs
for name in v2_pymethods2test v2_unitsyn_python v2_codenet_python v2_e2egit v2_ghactions; do
    rm -rf "jobs/$name" 2>/dev/null
done

# Launch all harbor tests in parallel
for entry in "v2_pymethods2test:$PYMETHODS_DIR" "v2_unitsyn_python:$UNITSYN_DIR" "v2_codenet_python:$CODENET_DIR" "v2_e2egit:$E2EGIT_DIR" "v2_ghactions:$GHACTIONS_DIR"; do
    name="${entry%%:*}"
    dir="${entry##*:}"
    if [ -z "$dir" ] || [ ! -d "$dir" ]; then
        echo "SKIP: $name - no task dir"
        continue
    fi
    echo "Launching harbor: $name ($dir)"
    harbor jobs start \
        -p "$dir" \
        --n-concurrent 5 \
        --agent terminus-2 \
        --model openai/gpt-5-nano-2025-08-07 \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$name" \
        --config eval/tacc/dcagent_eval_config.yaml &
done

echo "Waiting for all harbor jobs..."
wait

echo ""
echo "=== Phase 4 Results ==="
for name in v2_pymethods2test v2_unitsyn_python v2_codenet_python v2_e2egit v2_ghactions; do
    result_file="jobs/$name/result.json"
    if [ -f "$result_file" ]; then
        mean=$(python3 -c "
import json
with open('$result_file') as f:
    d = json.load(f)
for k,v in d['stats']['evals'].items():
    print(f'{v[\"metrics\"][0][\"mean\"]:.0%}')
    break
" 2>/dev/null || echo "ERROR")
        echo "  $name: $mean"
    else
        echo "  $name: NO RESULT"
    fi
done
