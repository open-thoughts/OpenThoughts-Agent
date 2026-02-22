#!/usr/bin/env python3
"""Generate 500-task RL experiment datasets — batch 2 (remaining 7 scaling experiments).

Remaining experiments with monotone pass rate progression:
  1.2_minimal:       0% → 20% → 20%  (+20pp)
  4.3_curated_1k:   40% → 60% → 60%  (+20pp)
  8.10_expert:      40% → 60% → 60%  (+20pp)
  8.1_github_issue: 40% → 60% → 60%  (+20pp)
  8.3_detailed:     40% → 40% → 60%  (+20pp)
  8.5_heavy:        40% → 60% → 60%  (+20pp)
  8.5_moderate:     40% → 60% → 60%  (+20pp)
"""

import sys
import json
import concurrent.futures
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "data"))

from data.exp_8_utils import (
    generate_instruction_transform,
    generate_instruction_and_tests_transform,
)
from data.exp_general_utils import llm_score_and_filter

LIMIT = 500
HF_ORG = "DCAgent"

# ─── Prompts ─────────────────────────────────────────────────

MINIMAL_PROMPT = """Rewrite the following coding task using ONLY the bare minimum information. Strip away all examples, context, hints, and details. Keep only the core requirement in 1-2 sentences.

ORIGINAL TASK:
{{instruction}}

Output only the minimal instruction:"""

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

GRANULARITY_PROMPT = """Rewrite the following coding task at "{{variant}}" granularity level.

- "vague": Single vague sentence. No details, no file hints.
- "bullets": 3-5 requirement bullets. Key requirements but no implementation hints.
- "detailed": Step-by-step with file structure hints, function signatures, edge cases.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" granularity. Output only the rewritten instruction:"""

PADDING_PROMPT = """Add irrelevant context/noise to this task at "{{variant}}" level. Real requirements must stay intact but be buried in noise.

- "mild": 1-2 irrelevant sentences of backstory
- "moderate": Paragraph of project context + red herring requirements
- "heavy": Verbose logs, unrelated code snippets, misleading hints. Real task buried deep.

ORIGINAL TASK:
{{instruction}}

Add "{{variant}}" padding. Output only the padded instruction:"""

KNOWLEDGE_PROMPT = """Rewrite this task at "{{variant}}" prerequisite knowledge level.

- "expert": Terse, assumes complete domain knowledge. No algorithm explanation.
- "intermediate": Explains algorithm steps but assumes programming competence.
- "novice": Fully explained with pseudocode or step-by-step algorithm.

ORIGINAL TASK:
{{instruction}}

Rewrite at "{{variant}}" level. Output only the rewritten instruction:"""

CURATE_PROMPT = """Rate this coding task on a scale of 1-5 for training quality:
- Clarity of requirements (1-5)
- Testability (1-5)
- Appropriate difficulty (1-5)

TASK:
{{instruction}}

TESTS:
{{test_files_json}}

Output ONLY a single number 1-5 (the average):"""


# ─── Generator functions ─────────────────────────────────────

def gen_minimal():
    """1.2_minimal: Minimal instructions."""
    print("=== Generating 1.2_minimal (500 tasks) ===")
    return generate_instruction_transform(
        prompt=MINIMAL_PROMPT,
        prefix="exp-rle-minimal",
        hf_repo=f"{HF_ORG}/exp_rle_minimal_instructions",
        variant_col="variant",
        variant_value="minimal",
        limit=LIMIT,
    )

def gen_curated_1k():
    """4.3_curated_1k: LLM-scored top tasks."""
    print("=== Generating 4.3_curated_1k (500 tasks) ===")
    # Score all source tasks, keep top 500
    from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
    from datasets import Dataset
    from data.completions import run_completions
    import tempfile
    from tqdm import tqdm

    tasks = load_harbor_tasks_as_dicts("DCAgent/exp_rpt_stack-pytest", limit=2000)
    print(f"  Loaded {len(tasks)} tasks for scoring")

    dataset = Dataset.from_list(tasks)
    result = run_completions(
        dataset, model="gpt-5-nano-2025-08-07", map_type="chat",
        map_config={"user_message": CURATE_PROMPT, "output_column": "quality_score"},
        require_all_responses=False,
    )
    scored = result.dataset
    print(f"  Scored {len(scored)} tasks")

    # Parse scores and filter top 500
    scored_list = []
    for row in scored:
        try:
            score = float(row["quality_score"].strip()[:1])
            scored_list.append((score, row))
        except (ValueError, IndexError):
            continue

    scored_list.sort(key=lambda x: -x[0])
    top_tasks = [row for _, row in scored_list[:LIMIT]]
    print(f"  Selected top {len(top_tasks)} tasks (score >= {scored_list[min(LIMIT-1, len(scored_list)-1)][0]:.0f})")

    out = Path(tempfile.mkdtemp(prefix="exp-rle-curated_"))
    for i, row in enumerate(tqdm(top_tasks, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i,
            instruction=row["instruction"],
            dockerfile=row["dockerfile"],
            test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix="exp-rle-curated",
            metadata={"source": row.get("task_dir_name", ""), "quality_score": row["quality_score"]},
        )

    repo_url = upload_tasks_to_hf(str(out), f"{HF_ORG}/exp_rle_curated")
    print(f"  Uploaded: {repo_url}")
    return str(out)

def gen_expert():
    """8.10_expert: Expert-level instructions."""
    print("=== Generating 8.10_expert (500 tasks) ===")
    return generate_instruction_transform(
        prompt=KNOWLEDGE_PROMPT,
        prefix="exp-rle-expert",
        hf_repo=f"{HF_ORG}/exp_rle_expert",
        variant_col="variant",
        variant_value="expert",
        limit=LIMIT,
    )

def gen_github_issue():
    """8.1_github_issue: GitHub issue style."""
    print("=== Generating 8.1_github_issue (500 tasks) ===")
    return generate_instruction_transform(
        prompt=STYLE_PROMPT,
        prefix="exp-rle-github-issue",
        hf_repo=f"{HF_ORG}/exp_rle_github_issue",
        variant_col="variant",
        variant_value="github_issue",
        limit=LIMIT,
    )

def gen_detailed():
    """8.3_detailed: Detailed step-by-step instructions."""
    print("=== Generating 8.3_detailed (500 tasks) ===")
    return generate_instruction_transform(
        prompt=GRANULARITY_PROMPT,
        prefix="exp-rle-detailed",
        hf_repo=f"{HF_ORG}/exp_rle_detailed",
        variant_col="variant",
        variant_value="detailed",
        limit=LIMIT,
    )

def gen_heavy():
    """8.5_heavy: Heavy context padding."""
    print("=== Generating 8.5_heavy (500 tasks) ===")
    return generate_instruction_transform(
        prompt=PADDING_PROMPT,
        prefix="exp-rle-heavy",
        hf_repo=f"{HF_ORG}/exp_rle_heavy_padding",
        variant_col="variant",
        variant_value="heavy",
        limit=LIMIT,
    )

def gen_moderate():
    """8.5_moderate: Moderate context padding."""
    print("=== Generating 8.5_moderate (500 tasks) ===")
    return generate_instruction_transform(
        prompt=PADDING_PROMPT,
        prefix="exp-rle-moderate",
        hf_repo=f"{HF_ORG}/exp_rle_moderate_padding",
        variant_col="variant",
        variant_value="moderate",
        limit=LIMIT,
    )


# ─── Main ────────────────────────────────────────────────────

GENERATORS = {
    "1.2_minimal": gen_minimal,
    "4.3_curated_1k": gen_curated_1k,
    "8.10_expert": gen_expert,
    "8.1_github_issue": gen_github_issue,
    "8.3_detailed": gen_detailed,
    "8.5_heavy": gen_heavy,
    "8.5_moderate": gen_moderate,
}

if __name__ == "__main__":
    print(f"Generating {len(GENERATORS)} datasets, {LIMIT} tasks each")
    print(f"Running in parallel...\n")

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GENERATORS)) as pool:
        futures = {pool.submit(fn): name for name, fn in GENERATORS.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                task_dir = future.result()
                n_tasks = len(list(Path(task_dir).glob("*/instruction.md"))) if task_dir else 0
                results[name] = {"task_dir": task_dir, "n_tasks": n_tasks, "status": "ok"}
                print(f"\n  DONE {name}: {n_tasks} tasks at {task_dir}")
            except Exception as e:
                results[name] = {"error": str(e), "status": "failed"}
                print(f"\n  FAILED {name}: {e}")
                import traceback; traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    for name in sorted(results):
        r = results[name]
        if r["status"] == "ok":
            print(f"  {name:<25} {r['n_tasks']:>4} tasks  {r['task_dir']}")
        else:
            print(f"  {name:<25} FAILED: {r['error']}")

    # Merge with batch 1 manifest
    manifest_path = PROJECT_ROOT / "data" / "rle_manifest.json"
    existing = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
    existing.update({k: v["task_dir"] for k, v in results.items() if v["status"] == "ok"})
    with open(manifest_path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nManifest saved to {manifest_path} ({len(existing)} total datasets)")
