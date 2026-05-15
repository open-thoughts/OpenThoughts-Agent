"""
Scan all datasets in the DCAgent HuggingFace org for a 'task_binary' column.
Writes results to dcagent_task_binary_datasets.md in the same directory.

Usage:
    python find_task_binary_datasets.py
"""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

ORG = "DCAgent2"
API_BASE = "https://huggingface.co/api/datasets"
ROWS_API = "https://datasets-server.huggingface.co/first-rows"
OUTPUT = Path(__file__).parent / "dcagent2_task_binary_datasets.md"


def list_all_datasets(org: str) -> list[str]:
    """Paginate through the HF API to get all dataset IDs for an org."""
    datasets = []
    url = f"{API_BASE}?author={org}&limit=1000&full=false"
    while url:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        for ds in data:
            datasets.append(ds["id"])
        # HF API uses Link header for pagination
        link = resp.headers.get("Link", "")
        url = None
        if 'rel="next"' in link:
            for part in link.split(","):
                if 'rel="next"' in part:
                    url = part.split("<")[1].split(">")[0]
    return datasets


def has_task_binary(dataset_id: str) -> bool | None:
    """Check if a dataset has a task_binary column via the first-rows API.

    Returns True/False, or None if the API call fails.
    """
    url = f"{ROWS_API}?dataset={dataset_id}&config=default&split=train"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        features = data.get("features", [])
        for f in features:
            if f.get("name") == "task_binary":
                return True
        return False
    except Exception:
        pass

    # Try without config param (some datasets don't have "default" config)
    url = f"{ROWS_API}?dataset={dataset_id}&split=train"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        features = data.get("features", [])
        for f in features:
            if f.get("name") == "task_binary":
                return True
        return False
    except Exception:
        return None


def main():
    print(f"Listing all datasets in {ORG}...")
    datasets = list_all_datasets(ORG)
    print(f"Found {len(datasets)} datasets. Checking for task_binary column...")

    found = []
    failed = []

    for i, ds_id in enumerate(datasets):
        result = has_task_binary(ds_id)
        status = "✓" if result else ("?" if result is None else "·")
        if (i + 1) % 50 == 0 or result or result is None:
            print(f"  [{i+1}/{len(datasets)}] {status} {ds_id}")
        if result is True:
            found.append(ds_id)
        elif result is None:
            failed.append(ds_id)

    print(f"\nDone. {len(found)} datasets with task_binary, {len(failed)} failed to check.")

    # Write results
    lines = [
        "# DCAgent Datasets with `task_binary` Field\n",
        f"\nGenerated from https://huggingface.co/datasets/{ORG} ({len(datasets)} total datasets checked).\n",
        f"\n## Datasets ({len(found)})\n\n",
    ]
    for i, ds in enumerate(found, 1):
        lines.append(f"{i}. {ds}\n")

    if failed:
        lines.append(f"\n## Could Not Verify ({len(failed)})\n\n")
        lines.append("These datasets returned errors from the first-rows API (pending processing, private, etc.):\n\n")
        for ds in failed:
            lines.append(f"- {ds}\n")

    OUTPUT.write_text("".join(lines))
    print(f"Results written to {OUTPUT}")


if __name__ == "__main__":
    main()
