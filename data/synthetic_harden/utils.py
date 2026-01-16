#!/usr/bin/env python3
"""
Shared utilities for synthetic hardening of tasks.
"""

from datasets import Dataset
from data.completions import run_completions


HARDEN_PROMPT = """You are an expert at making coding tasks more challenging and complex.

Given the following task/instruction, create a harder version of it. The harder version should:
1. Add additional requirements or constraints
2. Require more edge cases to be handled
3. Introduce additional complexity (e.g., performance requirements, error handling, testing requirements)
4. Be more specific about expected behavior in corner cases
5. Potentially require integration with additional components

Keep the core task the same but make it significantly more challenging to complete correctly.

Original task:
{{instruction}}

Provide ONLY the harder version of the task. Do not include any explanation or preamble."""


def harden_instructions(
    instructions: list[str],
    model: str = "gpt-4o-mini",
) -> list[str]:
    """
    Use an LLM to create harder versions of the instructions.

    Args:
        instructions: List of original instructions
        model: Model to use for hardening

    Returns:
        List of hardened instructions
    """
    dataset = Dataset.from_dict({"instruction": instructions})

    map_config = {
        "user_message": HARDEN_PROMPT,
        "output_column": "hardened_instruction",
    }

    result_dataset = run_completions(
        dataset=dataset,
        model=model,
        map_type="chat",
        map_config=map_config,
        temperature=0.7,
    )

    return result_dataset.dataset["hardened_instruction"]
