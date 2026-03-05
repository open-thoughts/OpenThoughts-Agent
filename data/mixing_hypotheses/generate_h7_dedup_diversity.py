#!/usr/bin/env python3
"""H7: Dedup + Diversity > Raw Volume.

Claim: 1000 deduplicated, skill-balanced tasks outperform 5000 raw tasks.

Why: Duplicates waste compute and cause overfitting. MinHash dedup + skill balance
= compact but maximally diverse. Quality > quantity is established in SFT; test for RL.

Mix: Dedup Python-base (MinHash, Jaccard >= 0.8), then skill-balance sample 1000.
1 Dockerfile.
Baseline: 5000 random tasks from same datasets (no dedup).
"""
import hashlib, json, random, sys, tempfile
from collections import defaultdict
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from exp_general_utils import DEFAULT_MODEL, mix_datasets
from data.commons import load_harbor_tasks_as_dicts, create_harbor_task_directory_generic, upload_tasks_to_hf
from data.completions import run_completions
from datasets import Dataset
from tqdm import tqdm
from mixing_hypotheses.dataset_registry import DATASETS, PYTHON_BASE

PREFIX_TEST = "mix-h7-dedup-diverse"
PREFIX_BASELINE = "mix-h7-raw-volume"
HF_REPO_TEST = "DCAgent/mix_h7_dedup_diverse_1k"
HF_REPO_BASELINE = "DCAgent/mix_h7_raw_volume_5k"
TARGET_DEDUP = 2500
TARGET_RAW = 5000
LOAD_PER_DATASET = 1000
SIMILARITY_THRESHOLD = 0.8
NUM_HASHES = 128

SKILL_PROMPT = """Classify the primary skills required for this coding task.
Pick 1-3 from this list: file_io, parsing, algorithm, data_structure, api,
database, testing, cli, concurrency, string_processing

TASK:
{{instruction}}

Output only the skill names, comma-separated:"""

SKILL_CATEGORIES = [
    "file_io", "parsing", "algorithm", "data_structure", "api",
    "database", "testing", "cli", "concurrency", "string_processing",
]


def _shingles(text, k=5):
    text = text.lower().strip()
    return {text[i:i+k] for i in range(len(text) - k + 1)} if len(text) >= k else {text}


def _minhash(shingles, num_hashes=NUM_HASHES):
    sig = []
    for i in range(num_hashes):
        min_val = float('inf')
        for s in shingles:
            h = int(hashlib.md5(f"{i}:{s}".encode()).hexdigest(), 16)
            min_val = min(min_val, h)
        sig.append(min_val)
    return sig


def _jaccard_minhash(sig1, sig2):
    return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["test", "baseline", "both"], default="both")
    args = parser.parse_args()

    # Load all Python-base tasks
    all_tasks = []
    for name in PYTHON_BASE:
        info = DATASETS[name]
        tasks = load_harbor_tasks_as_dicts(info.hf_repo, limit=LOAD_PER_DATASET)
        all_tasks.extend(tasks)
        print(f"  Loaded {len(tasks)} from {name}")
    print(f"Total pool: {len(all_tasks)} tasks")

    # ── Baseline arm: 5000 raw tasks, no dedup ──
    if args.arm in ("baseline", "both"):
        print(f"\n[BASELINE] {TARGET_RAW} raw tasks (no dedup)")
        n = len(PYTHON_BASE)
        weight = 1.0 / n
        repos_weights = [(DATASETS[d].hf_repo, weight) for d in PYTHON_BASE]
        mix_datasets(
            source_repos_with_weights=repos_weights,
            total_limit=TARGET_RAW,
            prefix=PREFIX_BASELINE,
            hf_repo=HF_REPO_BASELINE,
        )

    # ── Test arm: dedup + skill-balance → 1000 tasks ──
    if args.arm in ("test", "both"):
        print(f"\n[TEST] Dedup + skill-balanced → {TARGET_DEDUP} tasks")

        # Step 1: MinHash dedup
        print("Computing MinHash signatures...")
        sigs = []
        for t in tqdm(all_tasks, desc="MinHash"):
            sigs.append(_minhash(_shingles(t["instruction"])))

        keep = [True] * len(all_tasks)
        for i in tqdm(range(len(all_tasks)), desc="Deduplicating"):
            if not keep[i]:
                continue
            for j in range(i + 1, len(all_tasks)):
                if not keep[j]:
                    continue
                if _jaccard_minhash(sigs[i], sigs[j]) >= SIMILARITY_THRESHOLD:
                    keep[j] = False

        deduped = [t for t, k in zip(all_tasks, keep) if k]
        print(f"After dedup: {len(deduped)} / {len(all_tasks)} tasks")

        # Step 2: Skill classification
        dataset = Dataset.from_list(deduped)
        result = run_completions(
            dataset, model=DEFAULT_MODEL, map_type="chat",
            map_config={"user_message": SKILL_PROMPT, "output_column": "skills"},
            require_all_responses=False,
        )

        # Step 3: Skill-balanced sampling
        skill_buckets = defaultdict(list)
        for row in result.dataset:
            skills_raw = row.get("skills", "").strip().lower()
            primary = "unknown"
            for s in skills_raw.replace(",", " ").split():
                s = s.strip()
                if s in SKILL_CATEGORIES:
                    primary = s
                    break
            skill_buckets[primary].append(row)

        valid_skills = [s for s in skill_buckets if s != "unknown"]
        per_skill = max(1, TARGET_DEDUP // len(valid_skills))
        sampled = []
        for skill in valid_skills:
            pool = skill_buckets[skill]
            random.shuffle(pool)
            sampled.extend(pool[:min(per_skill, len(pool))])

        random.shuffle(sampled)
        sampled = sampled[:TARGET_DEDUP]
        print(f"Final skill-balanced set: {len(sampled)} tasks")

        out = Path(tempfile.mkdtemp(prefix=f"{PREFIX_TEST}_"))
        for i, row in enumerate(tqdm(sampled, desc="Creating tasks")):
            create_harbor_task_directory_generic(
                output_dir=out, task_id=i, instruction=row["instruction"],
                dockerfile=row["dockerfile"], test_sh=row["test_sh"],
                test_files=json.loads(row["test_files_json"]),
                dataset_prefix=PREFIX_TEST,
                metadata={"source": row.get("task_dir_name", ""), "skills": row.get("skills", "")},
            )

        print(f"\nUploading to {HF_REPO_TEST}...")
        upload_tasks_to_hf(str(out), HF_REPO_TEST)


if __name__ == "__main__":
    main()
