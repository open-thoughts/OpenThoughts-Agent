#!/usr/bin/env python3
"""
Manually register a trained model with Supabase.

Usage (from dc-agent/):
    source hpc/dotenv/tacc.env       # or otherwise export the Supabase + WANDB env vars
    python scripts/database/manual_db_push.py
"""

from datetime import datetime, timezone
import os
import sys
from pathlib import Path

import wandb

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.unified_db.utils import register_trained_model  # noqa: E402


HF_MODEL_ID = "DCAgent2/swesmith-stackseq"
WANDB_RUN = "dogml/dc-agent/81en31zw"
DATASET_NAME = "penfever/GLM-4.6-stackexchange-overflow-sandboxes-32eps-65k,penfever/GLM-4.6-swesmith-32ep-131k-nosumm-reasoning"
BASE_MODEL = "Qwen/Qwen3-8B"
TRAINING_TYPE = "SFT"

def main() -> None:
    # 1. Pull timestamps from W&B
    api = wandb.Api()
    run = api.run(WANDB_RUN)

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
        raise RuntimeError(f"W&B run {WANDB_RUN} does not have created_at populated yet")

    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    if finished is not None and finished.tzinfo is None:
        finished = finished.replace(tzinfo=timezone.utc)

    if finished is None:
        finished = datetime.now(timezone.utc)

    training_start = created.astimezone(timezone.utc).isoformat()
    training_end = finished.astimezone(timezone.utc).isoformat() if finished else None

    # 2. Shape the record exactly the way Llama-Factory expects
    record = {
        "agent_name": "terminus-2",  # derived from the dataset slug
        "training_start": training_start,
        "training_end": training_end,
        "created_by": HF_MODEL_ID.split("/", 1)[0],  # -> org name
        "base_model_name": BASE_MODEL,
        "dataset_name": DATASET_NAME,
        "dataset_id": None,
        "training_type": TRAINING_TYPE,
        "training_parameters": {
            "config_blob": f"https://huggingface.co/{HF_MODEL_ID}/blob/main/config.json",
            "hf_repo": HF_MODEL_ID,
        },
        "wandb_link": f"https://wandb.ai/{WANDB_RUN}",
        "traces_location_s3": os.environ.get("TRACE_S3_PATH"),
        "model_name": HF_MODEL_ID,
    }

    # 3. Insert / upsert into Supabase
    result = register_trained_model(record, forced_update=True)
    if result.get("success"):
        model = result["model"]
        status = "updated" if result.get("updated") else "created"
        print(f"✅ Supabase registration {status}: {model['id']} ({model['name']})")
    else:
        raise RuntimeError(f"Supabase registration failed: {result.get('error')}")

if __name__ == "__main__":
    main()
