#!/usr/bin/env python3
"""Generate 25-task versions of all flat-60% experiments to check if pass rates differentiate."""

import json
import sys
import concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from data.exp_8_utils import (
    generate_instruction_transform,
    generate_error_state,
)
from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
import tempfile
from tqdm import tqdm

LIMIT = 25
HF_ORG = "DCAgent"

# ─── Prompts (same as test_all_exp8.py) ─────────────────────

STYLE_PROMPT = """Rewrite the following coding task as a {{variant}}.

Styles:
- "github_issue": GitHub issue with title, description, steps to reproduce
- "slack_message": Casual Slack message from a coworker
- "stackoverflow_question": SO question with context and code
- "code_review_comment": Code review comment on what needs to change
- "error_report": Bug report with observed vs expected behavior

ORIGINAL TASK:
{{instruction}}

Rewrite as a {{variant}}. Preserve ALL technical requirements. Output only the rewritten instruction:"""

MODALITY_PROMPT = """Rewrite the following coding task using ONLY the "{{variant}}" specification modality.

- "nl_prose": Clear natural language paragraph
- "pseudocode": Pseudocode only, no natural language
- "io_examples": Only input/output examples (3+ examples, no prose)
- "type_signatures": Only Python type signatures and docstrings
- "failing_test_description": Describe as "write code that makes these tests pass" with tests described in words

ORIGINAL TASK:
{{instruction}}

Rewrite using ONLY "{{variant}}" modality. Output only the rewritten instruction:"""

PADDING_PROMPT = """Add irrelevant context/noise to this task at "{{variant}}" level. Real requirements must stay intact but be buried in noise.

- "mild": 1-2 irrelevant sentences of backstory
- "moderate": Paragraph of project context + red herring requirements
- "heavy": Verbose logs, unrelated code snippets, misleading hints. Real task buried deep.

ORIGINAL TASK:
{{instruction}}

Add "{{variant}}" padding. Output only the padded instruction:"""

BROKEN_CODE_PROMPT = """Generate a PARTIALLY BROKEN implementation for this task at "{{variant}}" severity.

- "subtle": 90% correct, 1-2 subtle bugs (off-by-one, wrong import, missing edge case)
- "structural": Structure correct but core logic broken (wrong algorithm, missing steps)

TASK:
{{instruction}}

Output only the broken Python code:"""

ERROR_INSTR_PROMPT = """Rewrite this task to say the workspace contains a partially broken implementation at /app/starter_code.py. The agent must fix the bugs and complete the implementation.

ORIGINAL TASK:
{{instruction}}

Output only the rewritten instruction:"""

KNOWLEDGE_PROMPT = """Rewrite this task at "{{variant}}" prerequisite knowledge level.

- "expert": Terse, assumes complete domain knowledge. No algorithm explanation.
- "intermediate": Explains algorithm steps but assumes programming competence.
- "novice": Fully explained with pseudocode or step-by-step algorithm.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" level. Output only the rewritten instruction:"""

RICH_PROMPT = """Enrich the following coding task with additional context, hints, examples, and edge case descriptions. Keep all original requirements but add helpful information.

ORIGINAL TASK:
{{instruction}}

Output only the enriched instruction:"""


# ─── Generator functions ─────────────────────────────────────

def _cat_a(prompt, prefix, variant_col, variant_value):
    return generate_instruction_transform(
        prompt=prompt, prefix=prefix, hf_repo=None,
        variant_col=variant_col, variant_value=variant_value, limit=LIMIT,
    )

def gen_baseline():
    """baseline_source: Unmodified source tasks."""
    print("=== Generating baseline_source (25 tasks) ===")
    tasks = load_harbor_tasks_as_dicts("DCAgent/exp_rpt_stack-pytest", limit=LIMIT)
    out = Path(tempfile.mkdtemp(prefix="flat25-baseline_"))
    for i, row in enumerate(tqdm(tasks, desc="baseline")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix="flat25-baseline",
            metadata={"source": row["task_dir_name"]},
        )
    return str(out)

def gen_rich():
    print("=== Generating 1.2_rich (25 tasks) ===")
    return generate_instruction_transform(
        prompt=RICH_PROMPT, prefix="flat25-rich", hf_repo=None,
        variant_col="variant", variant_value="rich", limit=LIMIT,
    )

def gen_dedup():
    """1.4_deduplicated: Same as baseline (dedup only matters at scale)."""
    print("=== Generating 1.4_deduplicated (25 tasks) ===")
    return gen_baseline()  # identical at 25 tasks

def gen_code_review():
    print("=== Generating 8.1_code_review (25 tasks) ===")
    return _cat_a(STYLE_PROMPT, "flat25-cr", "variant", "code_review_comment")

def gen_slack():
    print("=== Generating 8.1_slack_message (25 tasks) ===")
    return _cat_a(STYLE_PROMPT, "flat25-sl", "variant", "slack_message")

def gen_stackoverflow():
    print("=== Generating 8.1_stackoverflow (25 tasks) ===")
    return _cat_a(STYLE_PROMPT, "flat25-so", "variant", "stackoverflow_question")

def gen_failing_test():
    print("=== Generating 8.2_failing_test (25 tasks) ===")
    return _cat_a(MODALITY_PROMPT, "flat25-ft", "variant", "failing_test_description")

def gen_nl_prose():
    print("=== Generating 8.2_nl_prose (25 tasks) ===")
    return _cat_a(MODALITY_PROMPT, "flat25-nl", "variant", "nl_prose")

def gen_pseudocode():
    print("=== Generating 8.2_pseudocode (25 tasks) ===")
    return _cat_a(MODALITY_PROMPT, "flat25-ps", "variant", "pseudocode")

def gen_mild():
    print("=== Generating 8.5_mild (25 tasks) ===")
    return _cat_a(PADDING_PROMPT, "flat25-mi", "variant", "mild")

def gen_subtle():
    print("=== Generating 8.6_subtle (25 tasks) ===")
    from data.exp_8_utils import generate_error_state
    return generate_error_state(
        broken_code_prompt=BROKEN_CODE_PROMPT, instruction_prompt=ERROR_INSTR_PROMPT,
        prefix="flat25-sub", hf_repo=None,
        variant_col="variant", variant_value="subtle", limit=LIMIT,
    )

def gen_intermediate():
    print("=== Generating 8.10_intermediate (25 tasks) ===")
    return _cat_a(KNOWLEDGE_PROMPT, "flat25-int", "variant", "intermediate")

def gen_novice():
    print("=== Generating 8.10_novice (25 tasks) ===")
    return _cat_a(KNOWLEDGE_PROMPT, "flat25-nov", "variant", "novice")

def gen_d5_bare():
    """7.1_d5_bare: Bare Dockerfile — just copy baseline with minimal Dockerfile."""
    print("=== Generating 7.1_d5_bare (25 tasks) ===")
    tasks = load_harbor_tasks_as_dicts("DCAgent/exp_rpt_stack-pytest", limit=LIMIT)
    bare_dockerfile = "FROM ubuntu:24.04\nWORKDIR /app\nRUN apt-get update && apt-get install -y python3 python3-pip python3-venv bsdutils && rm -rf /var/lib/apt/lists/*\n"
    out = Path(tempfile.mkdtemp(prefix="flat25-d5bare_"))
    for i, row in enumerate(tqdm(tasks, desc="d5_bare")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=bare_dockerfile,
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix="flat25-d5bare",
            metadata={"source": row["task_dir_name"]},
        )
    return str(out)

def gen_speed_bonus():
    """7.6_speed_bonus: Same tasks with speed bonus instruction appended."""
    print("=== Generating 7.6_speed_bonus (25 tasks) ===")
    tasks = load_harbor_tasks_as_dicts("DCAgent/exp_rpt_stack-pytest", limit=LIMIT)
    out = Path(tempfile.mkdtemp(prefix="flat25-speed_"))
    for i, row in enumerate(tqdm(tasks, desc="speed_bonus")):
        instruction = row["instruction"] + "\n\nBONUS: Complete this task as quickly as possible. Faster solutions are preferred."
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=instruction,
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix="flat25-speed",
            metadata={"source": row["task_dir_name"]},
        )
    return str(out)


# ─── Main ────────────────────────────────────────────────────

GENERATORS = {
    "baseline_source": gen_baseline,
    "1.2_rich": gen_rich,
    "8.1_code_review": gen_code_review,
    "8.1_slack_message": gen_slack,
    "8.1_stackoverflow": gen_stackoverflow,
    "8.2_failing_test": gen_failing_test,
    "8.2_nl_prose": gen_nl_prose,
    "8.2_pseudocode": gen_pseudocode,
    "8.5_mild": gen_mild,
    "8.6_subtle": gen_subtle,
    "8.10_intermediate": gen_intermediate,
    "8.10_novice": gen_novice,
    "7.1_d5_bare": gen_d5_bare,
    "7.6_speed_bonus": gen_speed_bonus,
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", help="Subset to generate")
    args = parser.parse_args()

    gens = GENERATORS
    if args.experiments:
        gens = {k: v for k, v in gens.items() if any(k.startswith(p) for p in args.experiments)}

    print(f"Generating {len(gens)} experiments, {LIMIT} tasks each")
    print(f"Running in parallel...\n")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(gens), 8)) as pool:
        futures = {pool.submit(fn): name for name, fn in gens.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                task_dir = future.result()
                n_tasks = len(list(Path(task_dir).glob("*/instruction.md"))) if task_dir else 0
                results[name] = task_dir
                print(f"\n  DONE {name}: {n_tasks} tasks at {task_dir}")
            except Exception as e:
                results[name] = None
                print(f"\n  FAILED {name}: {e}")
                import traceback; traceback.print_exc()

    # Save manifest
    manifest_path = PROJECT_ROOT / "data" / "flat25_manifest.json"
    manifest = {k: v for k, v in results.items() if v}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path} ({len(manifest)} datasets)")
