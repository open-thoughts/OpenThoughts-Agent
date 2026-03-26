#!/usr/bin/env python3
"""
C1 Trace Generation: Kimi-K2 teacher model via kimi-cli agent.

Usage:
    python c1_kimik2.py --sandbox-datasets DCAgent/exp_rpt_stack-bash DCAgent/exp_rpt_crosscodeeval-python-v2 --push --submit
"""

from _c1_common import run_c1_pipeline

TEACHER_MODEL = "kimi-k2"
AGENT = "kimi-cli"
C1_PREFIX = "c1_kimik2"

if __name__ == "__main__":
    run_c1_pipeline(teacher_model=TEACHER_MODEL, agent=AGENT, c1_prefix=C1_PREFIX)
