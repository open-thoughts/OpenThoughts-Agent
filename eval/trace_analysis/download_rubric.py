"""
Download rubrics from the Docent dashboard and save them to a JSON file.

Usage:
    # Download all rubrics for a collection
    python download_rubric.py --collection_id <COLLECTION_ID>

    # Download rubrics and save to custom output file
    python download_rubric.py --collection_id <COLLECTION_ID> --output rubrics/my_rubrics.json

Example:
    uv run python eval/trace_analysis/download_rubric.py \
        --collection_id a5a7fc29-8838-4930-b5cb-e897ecde79a8
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from docent import Docent

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RUBRICS_DIR = Path(__file__).parent / "rubrics"


def download_rubrics(collection_id: str) -> list[dict[str, Any]]:
    api_key = os.getenv("DOCENT_API_KEY")
    if not api_key:
        raise ValueError("DOCENT_API_KEY environment variable not set")

    client = Docent(api_key=api_key)
    logger.info(f"Fetching rubrics for collection: {collection_id}")
    rubrics = client.list_rubrics(collection_id)
    logger.info(f"Found {len(rubrics)} rubrics")
    return rubrics


def main():
    parser = argparse.ArgumentParser(
        description="Download rubrics from a Docent collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download rubrics for a collection
  uv run python download_rubric.py --collection_id a5a7fc29-8838-4930-b5cb-e897ecde79a8

  # Save to custom file
  uv run python download_rubric.py --collection_id abc123 --output my_rubrics.json
        """,
    )
    parser.add_argument(
        "--collection_id",
        type=str,
        required=True,
        help="Docent collection ID to download rubrics from",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: rubrics/<collection_id>.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    output_path = Path(args.output) if args.output else RUBRICS_DIR / f"{args.collection_id}.json"

    try:
        rubrics = download_rubrics(args.collection_id)
    except Exception as e:
        logger.error(f"Failed to download rubrics: {e}")
        return 1
    if not rubrics:
        logger.warning("No rubrics found for this collection")
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(rubrics, f, indent=2, default=str)
    logger.info(f"Saved {len(rubrics)} rubrics to {output_path}")

    logger.info("\nRubric summary:")
    for i, rubric in enumerate(rubrics, 1):
        rubric_text = rubric.get("rubric_text", "")
        preview = rubric_text[:100].replace("\n", " ")
        if len(rubric_text) > 100:
            preview += "..."
        logger.info(f"  {i}. {rubric.get('id', 'unknown')}: {preview}")


if __name__ == "__main__":
    sys.exit(main())
    