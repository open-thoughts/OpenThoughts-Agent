#!/usr/bin/env python3
"""BaseDataGenerator integration for the Negotiation pipeline."""

from __future__ import annotations

import argparse
import random
import sys
import tempfile
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    # Allow executing the script directly via `python data/negotiation/generate_abstract.py`.
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from data.commons import (
    BaseDataGenerator,
    GenerationContext,
    GenerationRequest,
    GenerationResult,
    GenerationStatus,
    finalize_dataset_output,
    upload_tasks_to_hf,
)

if __package__ in (None, ""):
    from data.negotiation.generate import (  # type: ignore
        DEFAULT_ITEMS,
        build_instruction,
        build_scenario,
        create_negotiation_task,
        load_items,
    )
else:
    from .generate import (
        DEFAULT_ITEMS,
        build_instruction,
        build_scenario,
        create_negotiation_task,
        load_items,
    )


class NegotiationGenerator(BaseDataGenerator):
    """BaseDataGenerator-compatible wrapper for the negotiation task pipeline."""

    parser_description = "Generate multi-round bilateral negotiation sandboxes dataset"
    default_target_repo: Optional[str] = "OpenThoughts-Agent/negotiation-tasks"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed for reproducibility (default: 42).",
        )
        parser.add_argument(
            "--source", "-s",
            type=str,
            default=None,
            help="Path to CraigslistBargain CSV or directory containing train.csv. "
                 "If omitted, uses built-in synthetic items.",
        )
        parser.add_argument(
            "--zopa-mode",
            type=str,
            choices=("filter", "resample"),
            default="filter",
            help="filter: skip rows without ZOPA; "
                 "resample: regenerate reservations from list_price for missing/no-ZOPA rows "
                 "(default: filter).",
        )
        parser.add_argument(
            "--dataset-prefix",
            type=str,
            default="negotiation",
            help="Prefix for generated task directories (default: negotiation).",
        )

    def run_task_generation(
        self,
        request: GenerationRequest,
        context: GenerationContext,
    ) -> GenerationResult:
        args = context.args
        seed: int = getattr(args, "seed", 42)
        source: Optional[str] = getattr(args, "source", None)
        zopa_mode: str = getattr(args, "zopa_mode", "filter")
        dataset_prefix: str = getattr(args, "dataset_prefix", "negotiation")

        limit = self._resolve_limit(request)
        num_instances: int = limit if limit is not None and limit > 0 else 10

        items = load_items(source, DEFAULT_ITEMS, zopa_mode=zopa_mode, seed=seed)
        if not items:
            raise ValueError(
                "No items loaded. Check --source or use the default synthetic items."
            )

        rng = random.Random(seed)
        roles = ["buyer", "seller"]

        with tempfile.TemporaryDirectory() as tmp:
            tasks_dir = Path(tmp) / dataset_prefix
            tasks_dir.mkdir()

            for i in range(num_instances):
                item = rng.choice(items)
                role = rng.choice(roles)
                instance_seed = seed + i
                scenario = build_scenario(item, role, instance_seed)
                instruction = build_instruction(role, item, scenario)
                create_negotiation_task(tasks_dir, i, scenario, instruction)

            output_dir = getattr(args, "output_dir", None)
            final_path = finalize_dataset_output(str(tasks_dir), output_dir)

        uploaded_repo: Optional[str] = None
        if not getattr(args, "no_upload", False):
            target_repo = getattr(args, "target_repo", None) or self.default_target_repo
            if target_repo:
                upload_tasks_to_hf(str(final_path), target_repo)
                uploaded_repo = target_repo
            else:
                raise ValueError(
                    "No target repo specified; pass --target-repo or set default_target_repo."
                )

        return GenerationResult(
            dataset_path=str(final_path),
            status=GenerationStatus.SUCCESS,
            uploaded_repo=uploaded_repo,
            artifacts={
                "num_instances": num_instances,
                "seed": seed,
                "source": source or "default_synthetic",
                "zopa_mode": zopa_mode,
                "output_dir": str(final_path),
            },
        )


def main() -> None:
    NegotiationGenerator().run()


if __name__ == "__main__":
    main()
