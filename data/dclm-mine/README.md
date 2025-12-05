### dclm-mine: Constructing sequences from DCLM baseline

## Setup

# ZIH:
module load release/25.06  GCCcore/14.2.0
module load Python/3.10.8

python -m venv ~/venvs/ot-agent-venv
source ~/venvs/ot-agent-venv/bin/activate
pip install openai pydantic datasets supabase dotenv google-cloud-storage daytona-sdk daytona harbor h5py
export $(grep -v '^#' ~/ot-agent/data/dclm-mine/.env | xargs)
export PYTHONPATH=./

## Step 1: Filter dclm baseline

Download DCLM-baseline, finds sequences that contain shell-like commands and tasks, upload 100k sample to hf as: DCAgent2/dclm-baseline-terminal-candidates-100k

## Step 2: Classify as containing a suitable task

Classify the task and upload results:

python data/dclm-mine/Step2_classify/classify_tasks.py \
  --input-dataset DCAgent2/dclm-baseline-terminal-candidates-100k \
  --output-dataset DCAgent2/dclm-baseline-terminal-candidates-classified \
  --model gpt-5-nano-2025-08-07 \
  --max-sequences 50000

With gpt nano, classifying 10k sequences with gpt-5-nano is about $10

### Step 3: Extracting the task

python data/dclm-mine/Step3_extract/extract_tasks.py \
  --input-dataset DCAgent2/dclm-baseline-terminal-candidates-classified \
  --output-dataset DCAgent2/dclm-baseline-terminal-candidates-filtered-extracted-tasks

Extracting tasks from sequences; extracting 2k tasks should be around $10 with gpt-5-mini

### Step 4: Generating sandboxes and traces for SFT

# Use a specific dataset and limit
python data/dclm-mine/generate.py \
  --input-dataset DCAgent2/dclm-baseline-terminal-candidates-filtered-extracted-tasks \
  --n-concurrent 8 \
  --env-type docker

Notes:
- about 25.7 of the task contain a shell task
- about 82.9 of the sandboxes work; 
-> need to classify about 47k sequences for 10k working sandboxes 
