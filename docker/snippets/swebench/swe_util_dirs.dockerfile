# SWE-Bench Utility Directories Snippet
# Purpose: Create directory structure used by SWE-Bench evaluation
# Source: SkyRL-Agent utils.py:189
#
# Creates /swe_util directory structure for:
# - eval_data/instances: Storage for SWE-Bench instance JSON files

RUN mkdir -p /swe_util/eval_data/instances
