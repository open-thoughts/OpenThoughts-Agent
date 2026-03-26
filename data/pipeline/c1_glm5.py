#!/usr/bin/env python3
"""
C1 Trace Generation: GLM-5 teacher model via terminus-2 agent.

Usage:
    python c1_glm5.py --sandbox-datasets DCAgent/exp_rpt_stack-bash DCAgent/exp_rpt_crosscodeeval-python-v2 --push --submit
"""

from _c1_common import run_c1_pipeline

TEACHER_MODEL = "glm-5"
AGENT = "terminus-2"
C1_PREFIX = "c1_glm5"

if __name__ == "__main__":
    run_c1_pipeline(teacher_model=TEACHER_MODEL, agent=AGENT, c1_prefix=C1_PREFIX)
