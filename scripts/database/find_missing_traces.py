#!/usr/bin/env python3
"""
Find eval jobs missing HF trace links and attempt to match them against
existing HuggingFace trace datasets.

Workflow:
  1. Query Supabase for sandbox_jobs where hf_traces_link is NULL/empty
  2. For each, search HF orgs (DCAgent, DCAgent2, penfever, laion) for
     fuzzy-match candidates based on model name + benchmark name
  3. Exclude HF datasets already linked to other sandbox_jobs
  4. Verify candidates by spot-matching trial_names from sandbox_trials
     against the HF dataset rows
  5. Print strong matches (>50% trial overlap)

Usage:
    source ~/secrets.env  # or source /path/to/secrets.env

    # Find all missing traces and attempt matching
    python scripts/database/find_missing_traces.py

    # Only check specific job statuses
    python scripts/database/find_missing_traces.py --status Finished

    # Limit to N jobs (for testing)
    python scripts/database/find_missing_traces.py --limit 10

    # Auto-update: write matched HF links back to Supabase
    python scripts/database/find_missing_traces.py --update

Required environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY: Supabase service role key
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

# HF orgs to search for trace datasets
HF_ORGS = ["DCAgent", "DCAgent2", "penfever", "laion"]

# Minimum trial overlap ratio to consider a strong match
MATCH_THRESHOLD = 0.5

# Max HF datasets to check per job (avoid API rate limits)
MAX_CANDIDATES_PER_JOB = 10

# Number of DB trials to sample for matching (0 = all)
TRIAL_SAMPLE_SIZE = 50


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find eval jobs missing HF trace links and match against HF datasets.",
    )
    parser.add_argument(
        "--status",
        default="Finished",
        help="Only check jobs with this status (default: Finished)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of jobs to check (0 = all)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Write matched HF links back to Supabase",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed matching info",
    )
    return parser.parse_args()


def _sanitize_for_search(name: str) -> str:
    """Extract searchable tokens from a job/model name.

    Strips common prefixes, suffixes, and hash suffixes to produce
    a search string that will fuzzy-match HF dataset repo names.
    """
    # Remove common prefixes
    for prefix in ("eval-", "pending_", "laion/", "laion_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    # Remove trailing hash (8+ hex chars at end)
    name = re.sub(r"[_-][0-9a-f]{8,}$", "", name)
    # Replace underscores with hyphens for search
    return name


def _extract_model_token(job_name: str) -> str | None:
    """Extract the model identifier portion from a job name for searching."""
    # Common patterns: eval-<benchmark>__<model>__<config> or <benchmark>_<model>_<date>
    parts = re.split(r"__|_DCAgent|_openthoughts", job_name, maxsplit=1)
    if len(parts) >= 2:
        return parts[0]
    return job_name


def main() -> None:
    args = _parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set.")
        sys.exit(1)

    from supabase import create_client

    client = create_client(url, key)

    # Step 1: Find jobs missing HF traces (paginate to get all)
    print("Step 1: Finding eval jobs with missing HF trace links...")
    all_matching = []
    offset = 0
    page_size = 1000
    while True:
        batch = (
            client.table("sandbox_jobs")
            .select("id,job_name,model_id,benchmark_id,n_trials,hf_traces_link,job_status")
            .eq("job_status", args.status)
            .is_("hf_traces_link", "null")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        all_matching.extend(batch.data)
        if len(batch.data) < page_size:
            break
        offset += page_size
    missing = all_matching

    if args.limit > 0:
        missing = missing[: args.limit]

    print(f"  Found {len(missing)} {args.status} jobs with no HF traces link")
    if not missing:
        print("Nothing to do.")
        return

    # Build a set of already-linked HF datasets (to exclude from candidates)
    print("\nStep 2: Building index of already-linked HF trace datasets...")
    all_jobs = client.table("sandbox_jobs").select("hf_traces_link").not_.is_("hf_traces_link", "null").execute()
    linked_urls = set()
    for j in all_jobs.data:
        link = j.get("hf_traces_link", "")
        if link:
            # Normalize: extract repo ID from URL
            if "huggingface.co/datasets/" in link:
                repo_id = link.split("huggingface.co/datasets/")[-1].rstrip("/")
                linked_urls.add(repo_id)
            else:
                linked_urls.add(link)
    print(f"  {len(linked_urls)} HF datasets already linked to jobs")

    # Step 3: For each missing job, search HF for candidates
    print("\nStep 3: Searching HF for trace dataset candidates...")
    from huggingface_hub import HfApi

    api = HfApi()

    # Cache HF dataset listings per org (avoid repeated API calls)
    org_datasets: dict[str, list] = {}
    for org in HF_ORGS:
        try:
            datasets = list(api.list_datasets(author=org, limit=10000))
            org_datasets[org] = datasets
            print(f"  {org}: {len(datasets)} datasets")
        except Exception as e:
            print(f"  {org}: ERROR listing datasets: {e}")
            org_datasets[org] = []

    # Flatten to a searchable index: repo_id -> dataset_info
    all_hf_datasets = {}
    for org, datasets in org_datasets.items():
        for ds in datasets:
            all_hf_datasets[ds.id] = ds

    # Step 4: Match jobs to HF datasets
    print(f"\nStep 4: Matching {len(missing)} jobs against HF datasets...")
    matches = []
    no_candidates = []
    no_match = []

    for i, job in enumerate(missing):
        job_name = job["job_name"]
        job_id = job["id"]
        n_trials = job.get("n_trials", 0)

        if args.verbose:
            print(f"\n  [{i+1}/{len(missing)}] {job_name[:60]} (trials={n_trials})")

        # Generate search tokens from job name
        search_str = _sanitize_for_search(job_name)
        # Split into meaningful tokens (at least 4 chars)
        tokens = [t for t in re.split(r"[_\-]+", search_str) if len(t) >= 4]
        # Use the longest 3-4 tokens for matching
        tokens = sorted(tokens, key=len, reverse=True)[:4]

        if not tokens:
            if args.verbose:
                print(f"    No search tokens extracted")
            no_candidates.append(job_name)
            continue

        # Find candidate HF datasets that contain most tokens in their name
        candidates = []
        for repo_id, ds_info in all_hf_datasets.items():
            # Skip already-linked datasets
            if repo_id in linked_urls:
                continue
            repo_lower = repo_id.lower().replace("-", "_")
            # Count token matches
            matched_tokens = sum(1 for t in tokens if t.lower().replace("-", "_") in repo_lower)
            if matched_tokens >= min(2, len(tokens)):  # At least 2 tokens match (or all if <2)
                candidates.append((repo_id, matched_tokens))

        candidates.sort(key=lambda x: -x[1])
        candidates = candidates[:MAX_CANDIDATES_PER_JOB]

        if not candidates:
            if args.verbose:
                print(f"    No HF candidates found (tokens: {tokens})")
            no_candidates.append(job_name)
            continue

        if args.verbose:
            print(f"    {len(candidates)} candidates (tokens: {tokens})")

        # Step 5: Verify by trial name matching
        # Get trial names from DB
        trial_query = client.table("sandbox_trials").select("trial_name").eq("job_id", job_id)
        if TRIAL_SAMPLE_SIZE > 0:
            trial_query = trial_query.limit(TRIAL_SAMPLE_SIZE)
        db_trials = trial_query.execute()
        db_trial_names = set(t["trial_name"] for t in db_trials.data)

        if not db_trial_names:
            if args.verbose:
                print(f"    No trials in DB to match against")
            no_candidates.append(job_name)
            continue

        best_match = None
        best_overlap = 0

        for repo_id, token_score in candidates:
            try:
                from datasets import load_dataset

                ds = load_dataset(repo_id, split="train")
                if "trial_name" not in ds.column_names:
                    if args.verbose:
                        print(f"    {repo_id}: no trial_name column, skipping")
                    continue
                hf_trial_names = set(row["trial_name"] for row in ds)
                overlap = len(db_trial_names & hf_trial_names)
                ratio = overlap / len(db_trial_names) if db_trial_names else 0

                if args.verbose:
                    print(f"    {repo_id}: {overlap}/{len(db_trial_names)} match ({ratio:.0%})")

                if ratio > best_overlap:
                    best_overlap = ratio
                    best_match = (repo_id, overlap, len(db_trial_names), len(hf_trial_names), ratio)
            except Exception as e:
                if args.verbose:
                    print(f"    {repo_id}: ERROR loading - {e}")

        if best_match and best_match[4] >= MATCH_THRESHOLD:
            matches.append((job, best_match))
        else:
            no_match.append(job_name)

    # Report results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if matches:
        print(f"\nStrong matches ({len(matches)}):")
        print(f"{'Job Name':<60} {'HF Dataset':<60} {'Overlap':>10}")
        print("-" * 130)
        for job, (repo_id, overlap, db_total, hf_total, ratio) in matches:
            print(
                f"{job['job_name'][:60]:<60} {repo_id[:60]:<60} "
                f"{overlap}/{db_total} ({ratio:.0%})"
            )

        if args.update:
            print(f"\nUpdating {len(matches)} jobs with HF trace links...")
            for job, (repo_id, *_) in matches:
                hf_url = f"https://huggingface.co/datasets/{repo_id}"
                client.table("sandbox_jobs").update(
                    {"hf_traces_link": hf_url}
                ).eq("id", job["id"]).execute()
                print(f"  Updated {job['job_name'][:50]} -> {hf_url}")
    else:
        print("\nNo strong matches found.")

    if no_candidates:
        print(f"\nNo HF candidates found ({len(no_candidates)}):")
        for name in no_candidates[:10]:
            print(f"  {name[:80]}")
        if len(no_candidates) > 10:
            print(f"  ... and {len(no_candidates) - 10} more")

    if no_match:
        print(f"\nCandidates found but no trial overlap ({len(no_match)}):")
        for name in no_match[:10]:
            print(f"  {name[:80]}")
        if len(no_match) > 10:
            print(f"  ... and {len(no_match) - 10} more")

    print(f"\nSummary: {len(matches)} matched, {len(no_candidates)} no candidates, {len(no_match)} no trial match")


if __name__ == "__main__":
    main()
