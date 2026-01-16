#!/usr/bin/env python3
"""
Manually upload eval/trace job results to Supabase + HuggingFace.

This script replicates the upload flow from `run_eval.py --upload_to_database`,
allowing you to push traces from a pre-existing job directory.

Usage (from OpenThoughts-Agent/):
    source hpc/dotenv/tacc.env  # or otherwise export the Supabase + HF env vars

    # Basic usage - auto-detects agent/model/benchmark from job metadata
    python scripts/database/manual_db_eval_push.py \\
        --job-dir trace_jobs/eval-terminal-bench@2.0-gpt-5-nano-20260113_145348

    # With explicit HuggingFace repo
    python scripts/database/manual_db_eval_push.py \\
        --job-dir trace_jobs/my-eval-job \\
        --hf-repo DCAgent2/my-eval-traces

    # Skip HF upload (database only)
    python scripts/database/manual_db_eval_push.py \\
        --job-dir trace_jobs/my-eval-job \\
        --skip-hf

Required environment variables:
    SUPABASE_URL: Supabase project URL
    SUPABASE_ANON_KEY: Supabase anonymous key
    HF_TOKEN: HuggingFace token (for trace uploads)
"""

import argparse
import os
import sys
from pathlib import Path

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Also add database/unified_db to path
_db_path = _repo_root / "database" / "unified_db"
if str(_db_path) not in sys.path:
    sys.path.insert(0, str(_db_path))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload eval job results to Supabase + HuggingFace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(__doc__ or "").split("Usage")[0],  # Show description only
    )

    # Required arguments
    parser.add_argument(
        "--job-dir",
        required=True,
        help="Path to the Harbor job directory (e.g., trace_jobs/eval-terminal-bench@2.0-gpt-5-nano-20260113_145348)",
    )

    # Optional metadata overrides
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Override agent name (default: auto-detected from job metadata)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name (default: auto-detected from job metadata)",
    )
    parser.add_argument(
        "--benchmark-name",
        default=None,
        help="Override benchmark name (default: derived from job directory name)",
    )
    parser.add_argument(
        "--benchmark-version-hash",
        default=None,
        help="Override benchmark version hash (default: auto-generated from benchmark name)",
    )

    # HuggingFace options
    parser.add_argument(
        "--hf-repo",
        default=None,
        help="HuggingFace repo ID (e.g., DCAgent2/my-traces). Default: auto-derived from job name.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create HuggingFace repository as private (default: public)",
    )
    parser.add_argument(
        "--hf-episodes",
        default="last",
        choices=["all", "last"],
        help="Which episodes to export: 'all' or 'last' (default: last)",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip HuggingFace upload (database only)",
    )

    # Database options
    parser.add_argument(
        "--username",
        default=None,
        help="Username for job registration (default: UPLOAD_USERNAME env or current user)",
    )
    parser.add_argument(
        "--error-mode",
        default="skip_on_error",
        choices=["rollback_on_error", "skip_on_error"],
        help="Error handling: 'rollback_on_error' (atomic) or 'skip_on_error' (continue on failures)",
    )
    parser.add_argument(
        "--register-benchmark",
        action="store_true",
        default=True,
        help="Auto-register benchmark if not found (default: True)",
    )
    parser.add_argument(
        "--no-register-benchmark",
        action="store_false",
        dest="register_benchmark",
        help="Do not auto-register benchmark if not found",
    )
    parser.add_argument(
        "--forced-update",
        action="store_true",
        help="Allow updating existing job records",
    )

    # Debug options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually uploading",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    return parser.parse_args()


def derive_benchmark_name(job_dir: Path) -> str:
    """Derive benchmark name from job config or directory name.

    Delegates to the shared utility in hpc.launch_utils for consistency.
    """
    from hpc.launch_utils import derive_benchmark_from_job_dir
    return derive_benchmark_from_job_dir(job_dir)


def derive_hf_repo_id(job_name: str, org: str = "DCAgent2") -> str:
    """Derive HuggingFace repo ID from job name.

    Note: Sanitization happens in launch_utils.sync_eval_to_database().
    """
    sanitized = job_name.replace("@", "-").replace(" ", "-")
    return f"{org}/{sanitized}-traces"


def main() -> None:
    args = _parse_args()

    # Validate job directory
    job_dir = Path(args.job_dir).expanduser().resolve()
    if not job_dir.exists():
        print(f"Error: Job directory does not exist: {job_dir}")
        sys.exit(1)
    if not job_dir.is_dir():
        print(f"Error: Path is not a directory: {job_dir}")
        sys.exit(1)

    # Check for required environment variables
    if not os.environ.get("SUPABASE_URL"):
        print("Warning: SUPABASE_URL not set - database upload will fail")
    if not os.environ.get("HF_TOKEN") and not args.skip_hf:
        print("Warning: HF_TOKEN not set - HuggingFace upload will be skipped")

    # Derive benchmark name if not provided
    benchmark_name = args.benchmark_name or derive_benchmark_name(job_dir)

    # Derive HF repo ID if not provided and not skipping HF
    # (sanitization happens in launch_utils.sync_eval_to_database)
    hf_repo_id = None
    if not args.skip_hf:
        if args.hf_repo:
            hf_repo_id = args.hf_repo
        else:
            hf_repo_id = derive_hf_repo_id(job_dir.name)

    if args.verbose:
        print(f"Job directory: {job_dir}")
        print(f"Benchmark name: {benchmark_name}")
        print(f"HF repo ID: {hf_repo_id or '(skipped)'}")
        print(f"Agent name: {args.agent_name or '(auto-detect)'}")
        print(f"Model name: {args.model_name or '(auto-detect)'}")
        print()

    if args.dry_run:
        print("DRY RUN - would upload with:")
        print(f"  job_dir: {job_dir}")
        print(f"  benchmark_name: {benchmark_name}")
        print(f"  hf_repo_id: {hf_repo_id}")
        print(f"  agent_name: {args.agent_name}")
        print(f"  model_name: {args.model_name}")
        print(f"  error_mode: {args.error_mode}")
        print(f"  register_benchmark: {args.register_benchmark}")
        print(f"  forced_update: {args.forced_update}")
        return

    # Import and call the sync function
    from hpc.launch_utils import sync_eval_to_database

    result = sync_eval_to_database(
        job_dir=job_dir,
        username=args.username,
        error_mode=args.error_mode,
        agent_name=args.agent_name,
        model_name=args.model_name,
        benchmark_name=benchmark_name,
        benchmark_version_hash=args.benchmark_version_hash,
        register_benchmark=args.register_benchmark,
        hf_repo_id=hf_repo_id,
        hf_private=args.hf_private,
        hf_token=os.environ.get("HF_TOKEN"),
        hf_episodes=args.hf_episodes,
        forced_update=args.forced_update,
        dry_run=False,
    )

    # Report results
    if result.get("success"):
        job_id = result.get("job_id")
        n_trials = result.get("n_trials_uploaded", 0)
        hf_url = result.get("hf_dataset_url")

        print()
        print("Upload completed successfully!")
        print(f"  Job ID: {job_id}")
        print(f"  Trials uploaded: {n_trials}")
        if hf_url:
            print(f"  HuggingFace: {hf_url}")
    else:
        error = result.get("error", "Unknown error")
        print(f"Upload failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
