#!/usr/bin/env python3
"""
Generate and run an sbatch file with a job name derived from the module name.
Usage: python generate_and_run_sbatch.py <module_name>
Example: python generate_and_run_sbatch.py data.processing.dataset
         -> job name will be "dtst" (last part "dataset" with vowels removed)
"""
import sys
import subprocess
import re
from pathlib import Path


def remove_vowels(text):
    """Remove vowels (a, e, i, o, u) from text, case-insensitive."""
    return re.sub(r'[aeiouAEIOU]', '', text)


def get_job_name(module_name):
    """Extract name after last period and remove vowels."""
    # Get the part after the last period
    last_part = module_name.split('.')[-1]
    # Remove vowels
    job_name = remove_vowels(last_part)
    return job_name if job_name else "job"  # Fallback if all vowels


def generate_sbatch(module_name, job_name):
    """Generate sbatch file content."""
    sbatch_content = f"""#!/bin/bash
#SBATCH -p gg
#SBATCH --time=24:00:00
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task=72
#SBATCH --exclude=
#SBATCH --account CCR24067
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --job-name={job_name}

source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
conda activate /work/10000/eguha3/vista/miniconda3/envs/dataagent
source /scratch/10000/eguha3/old-ot-agent/secret.env
cd /scratch/10000/eguha3/ot-agent
python -m {module_name}
"""
    return sbatch_content


def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_and_run_sbatch.py <module_name>")
        print("Example: python generate_and_run_sbatch.py data.processing.dataset")
        sys.exit(1)

    module_name = sys.argv[1]
    job_name = get_job_name(module_name)

    print(f"Module: {module_name}")
    print(f"Job name: {job_name}")

    # Ensure logs directory exists
    logs_dir = Path("/scratch/10000/eguha3/ot-agent/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Generate sbatch content
    sbatch_content = generate_sbatch(module_name, job_name)

    # Write to temporary sbatch file
    sbatch_path = Path(f"/scratch/10000/eguha3/ot-agent/data/data_gen_{job_name}.sbatch")
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_content)

    print(f"Generated sbatch file: {sbatch_path}")

    # Submit the job
    result = subprocess.run(['sbatch', str(sbatch_path)],
                          capture_output=True,
                          text=True)

    if result.returncode == 0:
        print(f"Job submitted successfully!")
        print(result.stdout.strip())
    else:
        print(f"Error submitting job:")
        print(result.stderr.strip())
        sys.exit(1)


if __name__ == "__main__":
    main()
