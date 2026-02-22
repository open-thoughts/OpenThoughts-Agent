#!/bin/bash
# Download all HF datasets, sample 25 tasks each, and launch harbor jobs in parallel.
# Results are collected by verify_collect_results.py after all jobs complete.

set -eo pipefail

DCAGENT_DIR="/scratch/10000/eguha3/dc-agent"
cd "$DCAGENT_DIR"

# Source secrets
source "$DCAGENT_DIR/rl/hpc/dotenv/secret.env"

# Setup conda
source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/10000/eguha3/tacc_rl_v6

export PYTHONPATH="$DCAGENT_DIR:$DCAGENT_DIR/scripts/harbor:$PYTHONPATH"

MODEL="openai/gpt-5-nano-2025-08-07"
CONFIG="eval/tacc/dcagent_eval_config.yaml"
NCONCURRENT=8
SAMPLE_DIR="$DCAGENT_DIR/data/verify_samples"

echo "============================================="
echo "HF Dataset Verification - Parallel Launch"
echo "Date: $(date)"
echo "============================================="

# Step 1: Download and sample all datasets (fast, ~30 sec per dataset)
echo ""
echo "=== Phase 1: Download & Sample ==="
python3 -c "
import json, random, io, tarfile, sys, os, tempfile
from pathlib import Path, PurePosixPath
from datasets import load_dataset

DATASETS = [
    'DCAgent/exp_rpt_stack-bash',
    'DCAgent/exp_rpt_stack-bash-withtests',
    'DCAgent/exp_rpt_stack-bash-withtests-gpt5mini',
    'DCAgent/exp_rpt_stack-cpp',
    'DCAgent/exp_rpt_stack-csharp',
    'DCAgent/exp_rpt_stack-dockerfile',
    'DCAgent/exp_rpt_stack-dockerfile-gpt5mini',
    'DCAgent/exp_rpt_stack-go',
    'DCAgent/exp_rpt_stack-jest',
    'DCAgent/exp_rpt_stack-junit',
    'DCAgent/exp_rpt_stack-php',
    'DCAgent/exp_rpt_stack-pytest',
    'DCAgent/exp_rpt_stack-pytest-gpt5mini',
    'DCAgent/exp_rpt_stack-pytest-synthetic-gpt5nano',
    'DCAgent/exp_rpt_stack-pytest-withtests',
    'DCAgent/exp_rpt_stack-ruby',
    'DCAgent/exp_rpt_stack-rust',
    'DCAgent/exp_rpt_stack-selfdoc',
    'DCAgent/exp_rpt_stack-selfdoc-gpt5mini',
    'DCAgent/exp_rpt_bigcodebench',
    'DCAgent/exp_rpt_bugsinpy',
    'DCAgent/exp_rpt_bugsinpy-mf',
    'DCAgent/exp_rpt_bugswarm',
    'DCAgent/exp_rpt_codeelo',
    'DCAgent/exp_rpt_codenet-python',
    'DCAgent/exp_rpt_codereval-python',
    'DCAgent/exp_rpt_crosscodeeval-python',
    'DCAgent/exp_rpt_crosscodeeval-java',
    'DCAgent/exp_rpt_crosscodeeval-typescript',
    'DCAgent/exp_rpt_crosscodeeval-csharp',
    'DCAgent/exp_rpt_defects4j',
    'DCAgent/exp_rpt_e2egit',
    'DCAgent/exp_rpt_exercism-python',
    'DCAgent/exp_rpt_ghactions',
    'DCAgent/exp_rpt_manybugs',
    'DCAgent/exp_rpt_methods2test',
    'DCAgent/exp_rpt_pymethods2test',
    'DCAgent/exp_rpt_quixbugs-python-10k',
    'DCAgent/exp_rpt_softwareheritage',
    'DCAgent/exp_rpt_swebench',
    'DCAgent/exp_rpt_taco',
    'DCAgent/exp_rpt_unitsyn-python',
    'DCAgent/exp_rpt_bigcodebench-10k',
    'DCAgent/exp_rpt_bugsinpy-10k',
    'DCAgent/exp_rpt_codeelo-10k',
    'DCAgent/exp_rpt_exercism-python-10k',
    'DCAgent/exp_rpt_swebench-10k',
    'DCAgent/exp-rdb-r2egym-impossible',
    'DCAgent/exp-rdb-r2egym-very_hard',
    'DCAgent/exp-rdb-r2egym-hard',
    'DCAgent/exp-rdb-r2egym-medium',
    'DCAgent/exp-rdb-r2egym-easy',
    'DCAgent/exp-rdb-r2egym-trivial',
    'DCAgent2/logsmith-easy-500',
]

sample_base = Path('$SAMPLE_DIR')
sample_base.mkdir(parents=True, exist_ok=True)

def extract_tar(archive_bytes, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(archive_bytes)
    with tarfile.open(fileobj=buf, mode='r:*') as tf:
        for member in tf.getmembers():
            name = member.name
            while name.startswith('./') or name.startswith('/'):
                name = name[1:] if name.startswith('/') else name[2:]
            if not name or '..' in name.split('/'):
                continue
            target = dest_dir / name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tf.extractfile(member)
                if f is not None:
                    target.write_bytes(f.read())

manifest = {}
for repo_id in DATASETS:
    safe = repo_id.replace('/', '_').replace('-', '_')
    out_dir = sample_base / safe

    # Skip if already sampled
    if out_dir.exists() and any(out_dir.iterdir()):
        n = len(list(out_dir.iterdir()))
        print(f'CACHED {repo_id}: {n} tasks at {out_dir}')
        manifest[repo_id] = {'path': str(out_dir), 'total': -1, 'sampled': n, 'status': 'OK'}
        continue

    try:
        dataset = load_dataset(repo_id)
        train = dataset['train']
        total = len(train)

        if total == 0:
            print(f'EMPTY {repo_id}: 0 tasks')
            manifest[repo_id] = {'path': None, 'total': 0, 'sampled': 0, 'status': 'EMPTY'}
            continue

        random.seed(42)
        n = min(25, total)
        indices = random.sample(range(total), n)

        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in indices:
            row = train[idx]
            rel_path = row['path']
            task_name = PurePosixPath(rel_path).name if '/' in rel_path else rel_path
            extract_tar(bytes(row['task_binary']), out_dir / task_name)

        actual = len(list(out_dir.iterdir()))
        print(f'OK {repo_id}: {actual}/{total} tasks sampled')
        manifest[repo_id] = {'path': str(out_dir), 'total': total, 'sampled': actual, 'status': 'OK'}

    except Exception as e:
        err = str(e)[:200]
        print(f'ERROR {repo_id}: {err}')
        manifest[repo_id] = {'path': None, 'total': 0, 'sampled': 0, 'status': f'ERROR: {err}'}

with open('$SAMPLE_DIR/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)
print(f'\nManifest saved. {sum(1 for v in manifest.values() if v[\"status\"]==\"OK\")} datasets ready for harbor.')
"

echo ""
echo "=== Phase 2: Launch Harbor Jobs ==="

# Read manifest and launch harbor for each OK dataset
python3 -c "
import json
with open('$SAMPLE_DIR/manifest.json') as f:
    manifest = json.load(f)
for repo_id, info in manifest.items():
    if info['status'] == 'OK' and info['path']:
        safe = repo_id.replace('/', '_').replace('-', '_')
        job_name = f'verify_{safe}'
        print(f'{job_name}|{info[\"path\"]}')
" | while IFS='|' read job_name task_dir; do
    # Clean old job
    rm -rf "jobs/${job_name}" 2>/dev/null || true

    echo "Launching: $job_name ($task_dir)"
    harbor jobs start \
        -p "$task_dir" \
        --n-concurrent "$NCONCURRENT" \
        --agent terminus-2 \
        --model "$MODEL" \
        --env daytona \
        --n-attempts 1 \
        --max-retries 0 \
        --job-name "$job_name" \
        --config "$CONFIG" &
done

echo ""
echo "All harbor jobs launched. Waiting for completion..."
wait
echo "All harbor jobs complete! $(date)"
echo ""
echo "=== Phase 3: Collect Results ==="
python3 data/verify_collect_results.py
