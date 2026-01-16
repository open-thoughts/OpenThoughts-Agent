#!/usr/bin/env python3
"""Generate resume sbatch files by copying originals and replacing harbor command."""

import os
import re
from pathlib import Path

SBATCH_DIR = Path(os.path.expandvars("${DCAGENT_DIR}/data/ablation_experiments/sbatch"))
JOBS_DIR = Path(os.path.expandvars("${DCAGENT_DIR}/data/ablation_experiments/sbatch/jobs"))
RESUME_SBATCH_DIR = Path(os.path.expandvars("${DCAGENT_DIR}/data/ablation_experiments/resume_sbatch"))

# Map experiment names to their job timestamps (updated 2025-12-28/29/30)
EXPERIMENTS = {
    "frequency_penalty_0.25": "frequency_penalty_0.25_20251228_213021",
    "frequency_penalty_0.5": "frequency_penalty_0.5_20251228_213348",
    "frequency_penalty_1.0": "frequency_penalty_1.0_20251228_233149",
    "full_thinking": "full_thinking_20251228_233315",
    "high_diversity": "high_diversity_20251228_233420",
    "interleaved_thinking_on": "interleaved_thinking_on_20251229_052611",
    "linear_history_off": "linear_history_off_20251229_080621",
    "low_diversity": "low_diversity_20251229_083725",
    "max_episodes_256": "max_episodes_256_20251229_121742",
    "max_episodes_32": "max_episodes_32_20251229_132120",
    "max_episodes_512": "max_episodes_512_20251229_135258",
    "max_episodes_64": "max_episodes_64_20251229_150418",
    "max_tokens_1024": "max_tokens_1024_20251229_151819",
    "max_tokens_2048": "max_tokens_2048_20251229_151943",
    "max_tokens_4096": "max_tokens_4096_20251229_173342",
    "max_tokens_8192": "max_tokens_8192_20251229_180142",
    "min_p_0.01": "min_p_0.01_20251229_180734",
    "min_p_0.05": "min_p_0.05_20251229_180729",
    "min_p_0.1": "min_p_0.1_20251229_193442",
    "parser_xml": "parser_xml_20251229_193902",
    "presence_penalty_0.25": "presence_penalty_0.25_20251229_232624",
    "presence_penalty_0.5": "presence_penalty_0.5_20251230_001106",
    "presence_penalty_1.0": "presence_penalty_1.0_20251230_002505",
    "raw_content_off": "raw_content_off_20251230_005631",
    "repetition_penalty_1.05": "repetition_penalty_1.05_20251230_030631",
    "repetition_penalty_1.1": "repetition_penalty_1.1_20251230_033011",
    "repetition_penalty_1.2": "repetition_penalty_1.2_20251230_074632",
    "summarize_off": "summarize_off_20251230_084647",
    "summarize_threshold_16384": "summarize_threshold_16384_20251230_085339",
    "summarize_threshold_2048": "summarize_threshold_2048_20251230_090934",
    "summarize_threshold_32768": "summarize_threshold_32768_20251230_094521",
}


def generate_resume_sbatch(exp_name: str, full_job_name: str) -> str | None:
    """Copy original sbatch and replace harbor command with resume."""

    orig_sbatch = SBATCH_DIR / f"{exp_name}.sbatch"
    if not orig_sbatch.exists():
        print(f"Skipping {exp_name} - original sbatch not found")
        return None

    # Check job dirs exist
    first_shard = JOBS_DIR / f"{full_job_name}_shard0"
    if not first_shard.exists():
        print(f"Skipping {exp_name} - job directories not found at {first_shard}")
        return None

    with open(orig_sbatch) as f:
        content = f.read()

    # Change job name in header
    content = re.sub(r'#SBATCH --job-name=abl_', '#SBATCH --job-name=res_', content)

    # Change output log name
    content = re.sub(
        rf'--output=(.+){re.escape(exp_name)}_%j\.out',
        rf'--output=\g<1>{exp_name}_resume_%j.out',
        content
    )

    # Add job path variables after NODES_PER_SHARD line
    job_paths_block = f'''
# Hardcoded job paths for resume
JOB_PATH_SHARD0="{JOBS_DIR}/{full_job_name}_shard0"
JOB_PATH_SHARD1="{JOBS_DIR}/{full_job_name}_shard1"
JOB_PATH_SHARD2="{JOBS_DIR}/{full_job_name}_shard2"
JOB_PATH_SHARD3="{JOBS_DIR}/{full_job_name}_shard3"
JOB_PATH_SHARD4="{JOBS_DIR}/{full_job_name}_shard4"
JOB_PATH_SHARD5="{JOBS_DIR}/{full_job_name}_shard5"
JOB_PATH_SHARD6="{JOBS_DIR}/{full_job_name}_shard6"
JOB_PATH_SHARD7="{JOBS_DIR}/{full_job_name}_shard7"
'''
    content = re.sub(
        r'(NUM_SHARDS=8\nNODES_PER_SHARD=4)',
        rf'\1{job_paths_block}',
        content
    )

    new_function = '''run_harbor_for_shard() {
    local shard_idx=$1
    local start_node_idx=$((shard_idx * NODES_PER_SHARD))
    local head_node=${ALL_NODES_ARRAY[$start_node_idx]}
    local harbor_log="${EXPERIMENTS_DIR}/logs/harbor_${EXPERIMENT_NAME}_resume_shard${shard_idx}_${SLURM_JOB_ID}.log"

    # Get the job path for this shard
    local job_path
    case $shard_idx in
        0) job_path="$JOB_PATH_SHARD0" ;;
        1) job_path="$JOB_PATH_SHARD1" ;;
        2) job_path="$JOB_PATH_SHARD2" ;;
        3) job_path="$JOB_PATH_SHARD3" ;;
        4) job_path="$JOB_PATH_SHARD4" ;;
        5) job_path="$JOB_PATH_SHARD5" ;;
        6) job_path="$JOB_PATH_SHARD6" ;;
        7) job_path="$JOB_PATH_SHARD7" ;;
    esac

    HARBOR_LOGS[$shard_idx]="$harbor_log"

    echo "  Starting Harbor RESUME for shard $shard_idx..."
    echo "    Job path: $job_path"
    echo "    Filter: DaytonaRateLimitError, DaytonaError, DaytonaNotFoundError"

    srun --export="$SRUN_EXPORT_ENV" --nodes=1 --ntasks=1 --overlap -w "$head_node" \\
        harbor jobs resume --job-path "$job_path" -f DaytonaRateLimitError -f DaytonaError -f DaytonaNotFoundError \\
        >> "$harbor_log" 2>&1 &

    HARBOR_PIDS[$shard_idx]=$!
    echo "    PID: ${HARBOR_PIDS[$shard_idx]}, Log: $harbor_log"
}'''

    # Find and replace the function block, and remove dataset download/sharding steps
    lines = content.split('\n')
    new_lines = []
    i = 0
    skip_until_step = None  # Track which step we're skipping

    while i < len(lines):
        line = lines[i]

        # Skip Step 2: Downloading Dataset
        if '=== Step 2: Downloading Dataset ===' in line:
            skip_until_step = 'Step 3'
            i += 1
            continue

        # Skip Step 3: Sharding Dataset
        if '=== Step 3: Sharding Dataset ===' in line:
            skip_until_step = 'Step 4'
            i += 1
            continue

        # If we're skipping, check for the next step marker
        if skip_until_step:
            if f'=== {skip_until_step}' in line:
                skip_until_step = None
                # Continue to process this line normally
            else:
                i += 1
                continue

        if 'run_harbor_for_shard()' in line:
            brace_count = 0
            # Add the new function
            new_lines.append(new_function)
            # Skip until we find the closing brace
            while i < len(lines):
                if '{' in lines[i]:
                    brace_count += lines[i].count('{')
                if '}' in lines[i]:
                    brace_count -= lines[i].count('}')
                    if brace_count == 0:
                        i += 1
                        break
                i += 1
            continue

        new_lines.append(line)
        i += 1

    return '\n'.join(new_lines)


def main():
    RESUME_SBATCH_DIR.mkdir(parents=True, exist_ok=True)

    for exp_name, full_job_name in EXPERIMENTS.items():
        content = generate_resume_sbatch(exp_name, full_job_name)
        if content is None:
            continue

        sbatch_path = RESUME_SBATCH_DIR / f"{exp_name}_resume.sbatch"
        with open(sbatch_path, 'w') as f:
            f.write(content)

        print(f"Generated: {sbatch_path}")

    # Create run_all_resumes.sh
    run_all_content = '''#!/bin/bash
# Run all resume sbatch files
cd ${DCAGENT_DIR}/data/ablation_experiments/resume_sbatch

for sbatch_file in *.sbatch; do
    echo "Submitting $sbatch_file..."
    sbatch "$sbatch_file"
    sleep 2
done

echo "All resume jobs submitted!"
'''

    run_all_path = RESUME_SBATCH_DIR.parent / "run_all_resumes.sh"
    with open(run_all_path, 'w') as f:
        f.write(run_all_content)
    os.chmod(run_all_path, 0o755)
    print(f"\nGenerated: {run_all_path}")


if __name__ == "__main__":
    main()
