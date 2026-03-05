#!/usr/bin/env python3
"""H5: Skill-Diverse Sampling — equalize skill coverage across 10 categories.

Claim: Equalizing skill coverage (file_io, algorithm, parsing, api, database, cli,
concurrency) beats random sampling.

Why: Random code datasets over-index on string/algorithm tasks. Deliberately covering
rare skills (database, concurrency, API) increases problem diversity.

Mix: Apply skill classifier to Python-base datasets, sample to equalize 10 categories.
1 Dockerfile (all Python).
"""
import json, random, sys, tempfile
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import DEFAULT_MODEL
from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
from data.completions import run_completions
from datasets import Dataset
from tqdm import tqdm
from mixing_hypotheses.dataset_registry import DATASETS, PYTHON_BASE

PREFIX = "mix-h5-skill-diverse"
HF_REPO = "DCAgent/mix_h5_skill_diverse"
TOTAL = 5000
LOAD_PER_DATASET = 1000  # load up to this many per source dataset

SKILL_CATEGORIES = [
    "file_io", "parsing", "algorithm", "data_structure", "api",
    "database", "testing", "cli", "concurrency", "string_processing",
]

SKILL_PROMPT = """Classify the primary skills required for this coding task.
Pick 1-3 from this list:
- file_io: Reading/writing files
- parsing: Parsing data formats (JSON, CSV, XML, etc.)
- algorithm: Core algorithm implementation
- data_structure: Custom data structure implementation
- api: HTTP/REST API interaction
- database: Database operations
- testing: Test writing/debugging
- cli: Command-line tool building
- concurrency: Threading/async
- string_processing: Regex/text manipulation

TASK:
{{instruction}}

Output only the skill names, comma-separated:"""


def main():
    # Load tasks from all Python-base datasets
    all_tasks = []
    for name in PYTHON_BASE:
        info = DATASETS[name]
        tasks = load_harbor_tasks_as_dicts(info.hf_repo, limit=LOAD_PER_DATASET)
        for t in tasks:
            t["_source_dataset"] = name
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} from {name}")

    print(f"Total pool: {len(all_tasks)} tasks from {len(PYTHON_BASE)} datasets")

    # Classify skills via LLM
    dataset = Dataset.from_list(all_tasks)
    result = run_completions(
        dataset, model=DEFAULT_MODEL, map_type="chat",
        map_config={"user_message": SKILL_PROMPT, "output_column": "skills"},
        require_all_responses=False,
    )

    # Group by primary skill (first skill listed)
    skill_buckets = defaultdict(list)
    for row in result.dataset:
        skills_raw = row.get("skills", "").strip().lower()
        # Extract first valid skill
        primary = "unknown"
        for s in skills_raw.replace(",", " ").split():
            s = s.strip()
            if s in SKILL_CATEGORIES:
                primary = s
                break
        skill_buckets[primary].append(row)

    print(f"\nSkill distribution:")
    for skill in sorted(skill_buckets.keys()):
        print(f"  {skill}: {len(skill_buckets[skill])} tasks")

    # Sample equally from each skill to equalize coverage
    valid_skills = [s for s in skill_buckets if s != "unknown"]
    per_skill = max(1, TOTAL // len(valid_skills))
    sampled = []
    for skill in valid_skills:
        pool = skill_buckets[skill]
        random.shuffle(pool)
        n = min(per_skill, len(pool))
        sampled.extend(pool[:n])
        print(f"  Sampled {n} from {skill}")

    # Include some "unknown" if under target
    if len(sampled) < TOTAL and skill_buckets.get("unknown"):
        deficit = TOTAL - len(sampled)
        pool = skill_buckets["unknown"]
        random.shuffle(pool)
        sampled.extend(pool[:deficit])

    random.shuffle(sampled)
    sampled = sampled[:TOTAL]
    print(f"\nFinal dataset: {len(sampled)} tasks")

    out = Path(tempfile.mkdtemp(prefix=f"{PREFIX}_"))
    for i, row in enumerate(tqdm(sampled, desc="Creating tasks")):
        create_harbor_task_directory_generic(
            output_dir=out, task_id=i, instruction=row["instruction"],
            dockerfile=row["dockerfile"], test_sh=row["test_sh"],
            test_files=json.loads(row["test_files_json"]),
            dataset_prefix=PREFIX,
            metadata={"source": row.get("task_dir_name", ""), "skills": row.get("skills", "")},
        )

    print(f"\nUploading to {HF_REPO}...")
    upload_tasks_to_hf(str(out), HF_REPO)


if __name__ == "__main__":
    main()
