from __future__ import annotations

from typing import Any, Callable

from .arguments import JobType


def should_run_pretokenize(exp_args: dict, job_type: str | None = None) -> bool:
    """Return True when the current launch should schedule a pretokenization job."""

    resolved_job_type = (job_type or str(exp_args.get("job_type") or JobType.default_value())).strip().lower()
    return bool(exp_args.get("pretokenize")) or resolved_job_type == JobType.PRETOKENIZE.value


def schedule_pretokenize(
    exp_args: dict,
    *,
    update_exp_args_fn: Callable[[dict, dict], dict],
    construct_config_yaml_fn: Callable[[dict], tuple[dict, str]],
    construct_sbatch_script_fn: Callable[[dict], str],
    submit_job_fn: Callable[..., str],
) -> str:
    """Launch the pretokenization job using the provided helper callables."""

    pretok_args = exp_args.copy()
    pretok_args.pop("dependency", None)
    pretok_args = update_exp_args_fn(pretok_args, {"job_name": f"{exp_args['job_name']}_pretokenize"})
    pretok_args = update_exp_args_fn(pretok_args, {"num_nodes": 1})

    pretok_train_config, pretok_train_config_path_out = construct_config_yaml_fn(pretok_args)
    if exp_args.get("pretok_large"):
        if exp_args["name"] != "leonardo":
            raise ValueError("Large pretokenization is only supported on leonardo")
        pretok_args = update_exp_args_fn(
            pretok_args,
            {
                "time_limit": "03:00:00",
                "qos": "normal",
                "max_restarts": 0,
                "node_exclusion_list": "",
                "job_name": f"{exp_args['job_name']}_pretokenize",
            },
        )
    else:
        pretok_args = update_exp_args_fn(
            pretok_args,
            {
                "partition": exp_args["pretok_partition"],
                "qos": exp_args["pretok_qos"],
                "time_limit": exp_args["pretok_time_limit"],
                "max_restarts": 0,
            },
        )

    pretok_args = update_exp_args_fn(pretok_args, pretok_train_config)
    pretok_args = update_exp_args_fn(pretok_args, {"train_config_path_out": pretok_train_config_path_out})
    pretok_sbatch_path_out = construct_sbatch_script_fn(pretok_args)
    pretok_args = update_exp_args_fn(pretok_args, {"train_sbatch_path_out": pretok_sbatch_path_out})
    pretok_job_id = submit_job_fn(exp_args=pretok_args, dependency=None)
    return pretok_job_id
