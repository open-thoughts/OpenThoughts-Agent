from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import wandb


def wandb_init(kwargs: dict[str, Any]) -> None:
    """Initialize a wandb run using a normalized run name."""

    wandb_run_name = "_".join([str(value) for value in kwargs.values()])
    wandb_run_name = wandb_run_name.replace("/", "_")
    wandb_project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    wandb.init(project=wandb_project, name=wandb_run_name, config=kwargs)


def fetch_wandb_times(entity: str, project: str, run_name: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ISO timestamps for a wandb run, if accessible."""

    if not (entity and project and run_name):
        return None, None

    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}", filters={"display_name": run_name})
        for run in runs:
            run_display = getattr(run, "display_name", None)
            run_name_attr = getattr(run, "name", None)
            if run_display == run_name or run_name_attr == run_name:
                start = getattr(run, "created_at", None)
                end = getattr(run, "finished_at", None) or getattr(run, "updated_at", None)
                start_iso = start.isoformat() if hasattr(start, "isoformat") else start
                end_iso = end.isoformat() if hasattr(end, "isoformat") else end
                return start_iso, end_iso
    except ValueError:
        return None, None
    return None, None


def collect_wandb_metadata(exp_args: dict, train_config: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return wandb link plus training start/end timestamps if wandb logging is enabled."""

    report_to = train_config.get("report_to", "")
    wandb_enabled = False
    if isinstance(report_to, str):
        wandb_enabled = report_to.lower() == "wandb"
    elif isinstance(report_to, (list, tuple, set)):
        wandb_enabled = any(str(item).lower() == "wandb" for item in report_to)

    if not wandb_enabled:
        return None, None, None

    project = os.path.expandvars(os.environ.get("WANDB_PROJECT", "dcft"))
    entity = (
        os.environ.get("WANDB_ENTITY")
        or os.environ.get("WANDB_USERNAME")
        or exp_args.get("job_creator")
    )
    run_name_value = train_config.get("run_name") or exp_args.get("job_name")
    run_name = str(run_name_value) if run_name_value else None

    wandb_link = None
    if entity and project and run_name:
        wandb_link = f"https://wandb.ai/{entity}/{project}/runs/{run_name}"

    training_start, training_end = fetch_wandb_times(entity, project, run_name)
    return wandb_link, training_start, training_end
