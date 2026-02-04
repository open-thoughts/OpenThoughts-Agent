#!/usr/bin/env python3
"""
Download all datasets required for task generation.

This script downloads datasets from:
- Zenodo (E2EGit, GHActions, PyMethods2Test)
- GitHub (CrossCodeEval, ManyBugs)
- BugSwarm (requires manual download or API token)

Usage:
    python download_all_datasets.py                    # Download all
    python download_all_datasets.py --datasets e2egit ghactions  # Download specific
    python download_all_datasets.py --list             # List available datasets
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from dataset_utils import (
    download_all_datasets,
    download_crosscodeeval,
    download_manybugs,
    download_pymethods2test,
    download_e2egit,
    download_ghactions,
    download_bugswarm,
    get_cache_dir,
)


DATASETS_INFO = {
    "crosscodeeval": {
        "source": "GitHub (amazon-science/cceval)",
        "size": "~50MB",
        "description": "Code completion across languages",
    },
    "manybugs": {
        "source": "GitHub (squaresLab/ManyBugs)",
        "size": "~100MB",
        "description": "185 bugs in C programs",
    },
    "pymethods2test": {
        "source": "Zenodo (10.5281/zenodo.14264519)",
        "size": "~2GB",
        "description": "2M+ Python focal-test pairs",
    },
    "e2egit": {
        "source": "Zenodo (10.5281/zenodo.14234731)",
        "size": "~2.4GB",
        "description": "43K E2E web tests (SQLite)",
    },
    "ghactions": {
        "source": "Zenodo (10.5281/zenodo.10566003)",
        "size": "~560MB",
        "description": "160K+ GitHub Actions workflows",
    },
    "bugswarm": {
        "source": "bugswarm.org (manual or API)",
        "size": "~10MB JSON + Docker images",
        "description": "3000+ CI bug-fix pairs",
    },
}


def list_datasets():
    """Print information about available datasets."""
    print("\nAvailable datasets:")
    print("=" * 80)
    for name, info in DATASETS_INFO.items():
        print(f"\n{name}:")
        print(f"  Source: {info['source']}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
    print("\n" + "=" * 80)
    print(f"\nCache directory: {get_cache_dir()}")


def main():
    parser = argparse.ArgumentParser(description="Download datasets for task generation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS_INFO.keys()) + ["all"],
        help="Datasets to download (default: all)",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument(
        "--skip-large",
        action="store_true",
        help="Skip large datasets (pymethods2test, e2egit)",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    # Determine which datasets to download
    if args.datasets is None or "all" in args.datasets:
        datasets = list(DATASETS_INFO.keys())
    else:
        datasets = args.datasets

    if args.skip_large:
        datasets = [d for d in datasets if d not in ["pymethods2test", "e2egit"]]

    print(f"\nDownloading datasets: {datasets}")
    print(f"Cache directory: {get_cache_dir()}")
    print()

    results = download_all_datasets(datasets)

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    success = []
    failed = []

    for name, path in results.items():
        if path is not None:
            success.append(name)
            print(f"✓ {name}: {path}")
        else:
            failed.append(name)
            print(f"✗ {name}: FAILED")

    print()
    print(f"Successful: {len(success)}/{len(results)}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
