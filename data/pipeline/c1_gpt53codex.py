#!/usr/bin/env python3
"""
C1 Trace Generation: GPT-5.3 Codex teacher model via opencode agent.

Usage:
    python c1_gpt53codex.py --sandbox-datasets DCAgent/exp_rpt_stack-bash DCAgent/exp_rpt_crosscodeeval-python-v2 --push --submit
"""

from _c1_common import run_c1_pipeline

TEACHER_MODEL = "gpt-5.3-codex"
AGENT = "opencode"
C1_PREFIX = "c1_gpt53codex"

if __name__ == "__main__":
    run_c1_pipeline(teacher_model=TEACHER_MODEL, agent=AGENT, c1_prefix=C1_PREFIX)
