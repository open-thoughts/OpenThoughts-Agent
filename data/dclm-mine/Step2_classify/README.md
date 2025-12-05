### Classifying and extracting tasks from the sequences

## Test script for extraction

# Install dependencies
python3 -m venv ~/venvs/ot-agent-venv
source ~/venvs/ot-agent-venv/bin/activate
pip install openai pydantic
export $(grep -v '^#' ~/ot-agent/data/dclm-mine/.env | xargs)

# Run the script
set -a && source .env && set +a
python /home/rehe951g/ot-agent/data/dclm-mine/Step2_classify/classify_tasks.py

# Inspect lines in the bash
sed -n '1p' classified_tasks.jsonl | jq .