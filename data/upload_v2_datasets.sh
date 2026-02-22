#!/bin/bash
# Upload v2 datasets to HuggingFace (500 tasks each)
# For all 12 datasets with improved pass rates
set -e
source /scratch/10000/eguha3/old-dc-agent/secret.env
# Do NOT set SKIP_HF_UPLOAD - we want to upload

cd /scratch/10000/eguha3/dc-agent

echo "=== Uploading 12 improved datasets to HF (500 tasks each) ==="

# 1. bigcodebench (0% -> 54%)
echo "[1/12] bigcodebench..."
python3 data/dclm-mine/generate_bigcodebench_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 2. codeelo (0% -> 52%)
echo "[2/12] codeelo..."
python3 data/dclm-mine/generate_codeelo_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 3. crosscodeeval-python (28% -> 48%)
echo "[3/12] crosscodeeval-python..."
python3 data/dclm-mine/generate_crosscodeeval_tasks.py --language python --limit 500 2>&1 | tail -3
echo ""

# 4. e2egit (0% -> 48%)
echo "[4/12] e2egit..."
python3 data/dclm-mine/generate_e2egit_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 5. methods2test (0% -> 28%)
echo "[5/12] methods2test..."
python3 data/dclm-mine/generate_methods2test_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 6. codereval-python (0% -> 24%)
echo "[6/12] codereval-python..."
python3 data/dclm-mine/generate_codereval_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 7. softwareheritage (0% -> 12%)
echo "[7/12] softwareheritage..."
python3 data/dclm-mine/generate_softwareheritage_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 8. stack-php (0% -> 100%)
echo "[8/12] stack-php..."
python3 data/dclm-mine/generate_phpunit_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 9. stack-pytest (0% -> 40%)
echo "[9/12] stack-pytest..."
python3 data/dclm-mine/generate_pytest_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 10. stack-dockerfile (0% -> 16%)
echo "[10/12] stack-dockerfile..."
python3 data/dclm-mine/generate_dockerfile_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 11. stack-selfdoc (0% -> 16%)
echo "[11/12] stack-selfdoc..."
python3 data/dclm-mine/generate_self_documented_tasks.py --limit 500 2>&1 | tail -3
echo ""

# 12. stack-jest (0% -> 8%)
echo "[12/12] stack-jest..."
python3 data/dclm-mine/generate_jest_tasks.py --limit 500 2>&1 | tail -3
echo ""

echo "=== All 12 datasets uploaded ==="
