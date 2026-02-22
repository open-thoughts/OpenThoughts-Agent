#!/bin/bash
set -uo pipefail
source /scratch/10000/eguha3/old-dc-agent/secret.env
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
cd /scratch/10000/eguha3/dc-agent

echo "=== Phase 1: Regenerate all Theme 8 experiments (5 tasks each) ==="
python data/test_all_exp8.py --generate-only 2>&1 | tee /tmp/regen_exp8.log
echo "Theme 8 generation done: $(date)"

echo ""
echo "=== Phase 2: Regenerate other themes ==="
# baseline, 1.x, 2.x, 4.x, 5.x, 7.x
python data/test_all_other_themes.py --generate-only 2>&1 | tee /tmp/regen_other.log
echo "Other themes generation done: $(date)"

echo ""
echo "=== Phase 3: Collect all task directories ==="
# Read manifests
python3 -c "
import json
exp8 = json.load(open('data/exp8_test_manifest.json'))
other = json.load(open('data/other_themes_manifest.json'))
all_dirs = {**exp8, **other}
print(f'Total experiments: {len(all_dirs)}')
for k, v in sorted(all_dirs.items()):
    print(f'  {k}: {v}')
json.dump(all_dirs, open('/tmp/all_experiments_manifest.json', 'w'), indent=2)
"

echo ""
echo "=== Phase 4: Launch harbor tests for both models ==="
MODEL_NANO="openai/gpt-5-nano-2025-08-07"
MODEL_MINI="openai/gpt-5-mini-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
PID=$$

launch() {
    local model_prefix="$1"
    local model="$2"
    local name="$3"
    local task_dir="$4"
    local job_name="${model_prefix}_${name}_${PID}"
    echo "Launching: ${job_name}"
    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent 8 \
        --agent terminus-2 \
        --model "$model" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config "$CONFIG" &
}

# Launch all experiments for both models
python3 -c "
import json
manifest = json.load(open('/tmp/all_experiments_manifest.json'))
for name, task_dir in sorted(manifest.items()):
    print(f'{name}\t{task_dir}')
" | while IFS=$'\t' read -r name task_dir; do
    launch "nano" "$MODEL_NANO" "$name" "$task_dir"
    launch "mini" "$MODEL_MINI" "$name" "$task_dir"
done

echo "All harbor jobs launched. Waiting..."
wait
echo "All done: $(date)"
