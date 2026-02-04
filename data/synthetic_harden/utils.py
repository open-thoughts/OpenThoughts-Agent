#!/usr/bin/env python3
"""
Shared utilities for synthetic hardening of tasks.
"""

import itertools
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
        map_config=map_config
    )

    return result_dataset.dataset["hardened_instruction"]


MIX_PROMPT = """You are an expert at creating challenging tasks by combining concepts from multiple problems.

Given the following two tasks from different sources, create a SINGLE harder task that combines elements from both. The new task should:
1. Integrate concepts/requirements from both original tasks
2. Be coherent and make sense as a single unified problem
3. Be significantly more challenging than either task alone
4. Require the solver to understand both domains

Original Task A (from {{source_a}}):
{{instruction_a}}

Original Task B (from {{source_b}}):
{{instruction_b}}

Provide ONLY the combined harder task. Do not include any explanation or preamble."""


def mix_questions(
    instructions_a: list[str],
    instructions_b: list[str],
    source_a: str,
    source_b: str,
    model: str = "gpt-5-nano",
) -> list[str]:
    """
    Mix questions from two sources to create harder combined tasks.

    Args:
        instructions_a: List of instructions from source A
        instructions_b: List of instructions from source B
        source_a: Name of source A (e.g., "Tezos StackExchange")
        source_b: Name of source B (e.g., "StackOverflow")
        model: Model to use for mixing

    Returns:
        List of mixed instructions
    """
    # Determine the target length (use the longer list)
    target_len = max(len(instructions_a), len(instructions_b))

    # Cycle the shorter list to match the longer one
    if len(instructions_a) < target_len:
        instructions_a = list(itertools.islice(itertools.cycle(instructions_a), target_len))
    if len(instructions_b) < target_len:
        instructions_b = list(itertools.islice(itertools.cycle(instructions_b), target_len))

    # Create dataset with paired instructions
    dataset = Dataset.from_dict({
        "instruction_a": instructions_a,
        "instruction_b": instructions_b,
        "source_a": [source_a] * target_len,
        "source_b": [source_b] * target_len,
    })

    map_config = {
        "user_message": MIX_PROMPT,
        "output_column": "mixed_instruction",
    }

    result_dataset = run_completions(
        dataset=dataset,
        model=model,
        map_type="chat",
        map_config=map_config
    )

    return result_dataset.dataset["mixed_instruction"]


CONSTRAINT_PROMPT = """You are an expert at adding artificial constraints to coding tasks to make them more challenging.

Given the following task/instruction, come up with creative artificial constraints to make it harder. Examples include tool restrictions, step limits, brevity requirements, testing requirements, etc. Be creative!

Original task:
{{instruction}}

Add the constraints naturally to the task description.

Provide ONLY the task with the added constraints. Do not include any explanation or preamble."""


def add_constraints(
    instructions: list[str],
    model: str = "gpt-5-nano",
) -> list[str]:
    """
    Use an LLM to add artificial constraints to instructions.

    Args:
        instructions: List of original instructions
        model: Model to use for adding constraints

    Returns:
        List of instructions with added constraints
    """
    dataset = Dataset.from_dict({"instruction": instructions})

    map_config = {
        "user_message": CONSTRAINT_PROMPT,
        "output_column": "constrained_instruction",
    }

    result_dataset = run_completions(
        dataset=dataset,
        model=model,
        map_type="chat",
        map_config=map_config
    )

    return result_dataset.dataset["constrained_instruction"]
