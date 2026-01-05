#!/usr/bin/env python3
"""
Utility for exporting all public Hugging Face models owned by an organization.

Example:
    python scripts/database/list_public_models.py --org DCAgent --output hf_models.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import list_models


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the list of public Hugging Face models for an organization."
    )
    parser.add_argument(
        "--org",
        default="DCAgent",
        help="Hugging Face organization ID to inspect (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public_models.txt"),
        help="Path to the text file that will receive the model list (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    org = args.org.strip()
    if not org:
        raise ValueError("--org must be a non-empty string")

    # Hugging Face org listings are public; this narrows results to the owner.
    models = list_models(author=org, full=False)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    public_model_ids: list[str] = []
    for model in models:
        # `model.private` is False for public repositories.
        if getattr(model, "private", False):
            continue
        model_id = getattr(model, "modelId", None) or getattr(model, "id", None)
        if model_id:
            public_model_ids.append(model_id)

    public_model_ids.sort()

    output_path.write_text("\n".join(public_model_ids) + ("\n" if public_model_ids else ""))
    print(
        f"Wrote {len(public_model_ids)} public model(s) "
        f"from organization '{org}' to {output_path}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
