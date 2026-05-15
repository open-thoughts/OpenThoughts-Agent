#!/usr/bin/env python3
"""
Manually register a trained model with Supabase.

Usage (from OpenThoughts-Agent/):
    source hpc/dotenv/tacc.env  # or otherwise export the Supabase + WANDB env vars

    # Single dataset:
    python scripts/database/manual_db_push.py --hf-model-id org/model --dataset-name DCAgent/my-dataset

    # Multiple datasets (comma-separated → sets dataset_names instead of dataset_id):
    python scripts/database/manual_db_push.py --hf-model-id org/model \
        --dataset-name "DCAgent/dataset-a,DCAgent/dataset-b,DCAgent/dataset-c"
"""

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add repo root to sys.path for imports
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from database.unified_db.utils import register_trained_model  # noqa: E402


DEFAULT_HF_MODEL_ID = "laion/Kimi-K2T-swesmith-32ep-131k"
DEFAULT_WANDB_RUN = "dogml/dc-agent/34m5gsp5"
DEFAULT_DATASET_NAME = "penfever/Kimi-K2T-swesmith-32ep-131k"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TRAINING_TYPE = "SFT"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register a trained model with Supabase.")
    parser.add_argument(
        "--hf-model-id",
        default=DEFAULT_HF_MODEL_ID,
        help="Hugging Face repo ID for the trained model (default: %(default)s)",
    )
    parser.add_argument(
        "--wandb-run",
        default=None,
        help="Weights & Biases run path (entity/project/run_id). Optional — timestamps default to now if omitted.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name(s) used during training. For multi-dataset models, "
        "pass comma-separated names (e.g. 'DCAgent/ds-a,DCAgent/ds-b'). "
        "Single dataset → sets dataset_id. Multiple → sets dataset_names. "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model name that was finetuned (default: %(default)s)",
    )
    parser.add_argument(
        "--training-type",
        default=DEFAULT_TRAINING_TYPE,
        help="Training type label to store (default: %(default)s)",
    )
    parser.add_argument(
        "--agent-name",
        default=None,
        help="Optional agent name override (default: derived from dataset slug)",
    )
    return parser.parse_args()


def _derive_agent_name(dataset_name: str) -> str:
    slug = dataset_name.split("/")[-1]
    return slug or "agent"


def main() -> None:
    args = _parse_args()

    # 1. Pull timestamps from W&B (if provided), otherwise default to now
    if args.wandb_run:
        import wandb

        api = wandb.Api()
        run = api.run(args.wandb_run)

        created = getattr(run, "created_at", None)
        finished = getattr(run, "finished_at", None) or getattr(run, "stopped_at", None)
        if finished is None:
            attrs = getattr(run, "_attrs", {})
            if isinstance(attrs, dict):
                finished = attrs.get("finishedAt")
        if finished is None:
            finished = getattr(run, "updated_at", None)

        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))
        if isinstance(finished, str):
            finished = datetime.fromisoformat(finished.replace("Z", "+00:00"))

        if created is None:
            raise RuntimeError(f"W&B run {args.wandb_run} does not have created_at populated yet")

        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if finished is not None and finished.tzinfo is None:
            finished = finished.replace(tzinfo=timezone.utc)

        if finished is None:
            finished = datetime.now(timezone.utc)

        training_start = created.astimezone(timezone.utc).isoformat()
        training_end = finished.astimezone(timezone.utc).isoformat() if finished else None
    else:
        now = datetime.now(timezone.utc)
        training_start = now.isoformat()
        training_end = now.isoformat()

    # 2. Detect single vs multi-dataset
    is_multi = "," in args.dataset_name
    if is_multi:
        ds_list = [d.strip() for d in args.dataset_name.split(",") if d.strip()]
        print(f"Multi-dataset mode: {len(ds_list)} datasets")
        for ds in ds_list:
            print(f"  - {ds}")

    # 3. Shape the record
    record = {
        "agent_name": args.agent_name or _derive_agent_name(args.dataset_name),
        "training_start": training_start,
        "training_end": training_end,
        "created_by": args.hf_model_id.split("/", 1)[0],  # -> org name
        "base_model_name": args.base_model,
        "dataset_name": args.dataset_name,
        "dataset_id": None,
        "training_type": args.training_type,
        "training_parameters": {
            "config_blob": f"https://huggingface.co/{args.hf_model_id}/blob/main/config.json",
            "hf_repo": args.hf_model_id,
        },
        "wandb_link": f"https://wandb.ai/{args.wandb_run}" if args.wandb_run else None,
        "traces_location_s3": os.environ.get("TRACE_S3_PATH"),
        "model_name": args.hf_model_id,
    }

    # 4. Insert / upsert into Supabase
    result = register_trained_model(record, forced_update=True)
    if result.get("success"):
        model = result["model"]
        status = "updated" if result.get("updated") else "created"
        print(f"✅ Supabase registration {status}: {model['id']} ({model['name']})")
    else:
        raise RuntimeError(f"Supabase registration failed: {result.get('error')}")

if __name__ == "__main__":
    main()
