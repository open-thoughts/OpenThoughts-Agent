"""Regression tests for launcher artifact collision handling.

Collision-renamed directories own launch artifacts such as configs, logs, and
sbatch scripts. Durable RL state is resolved separately by ``RLPathManager``.

Run from the OT-Agent repo root with:
    .venv/bin/python -m pytest tests/hpc/test_launch_utils_collision.py -v
"""

from __future__ import annotations

import json
from pathlib import Path


from hpc.launch_utils import setup_experiments_dir


def _seed_existing_experiment(experiments_root: Path) -> None:
    """Seed ``experiments_root/configs/<file>.json`` so collision detection fires."""
    configs = experiments_root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "prior_run_config.json").write_text(json.dumps({"job_name": "prior"}))


def test_no_collision_leaves_path_unchanged(tmp_path: Path) -> None:
    """When no prior dir exists, exp_args["experiments_dir"] stays at canonical form."""
    target = tmp_path / "ot-baf" / "myjob"
    exp_args = {"experiments_dir": str(target), "job_type": "rl"}

    paths = setup_experiments_dir(exp_args, job_name="myjob")

    assert paths.root == target
    assert exp_args["experiments_dir"] == str(target)
    assert paths.configs == target / "configs"
    assert paths.sbatch == target / "sbatch"
    assert paths.logs == target / "logs"


def test_collision_propagates_renamed_path_to_exp_args(tmp_path: Path) -> None:
    """A real collision must bump exp_args["experiments_dir"] to the _2 form.

    This is the core regression: prior to the fix, exp_args was NOT
    updated, so downstream derivations like trainer.trials_dir read
    the un-suffixed canonical path while sbatch/configs/logs lived
    at <name>_2.
    """
    canonical = tmp_path / "ot-baf" / "myjob"
    _seed_existing_experiment(canonical)

    exp_args = {"experiments_dir": str(canonical), "job_type": "rl"}
    paths = setup_experiments_dir(exp_args, job_name="myjob")

    expected_renamed = tmp_path / "ot-baf" / "myjob_2"
    assert paths.root == expected_renamed
    assert exp_args["experiments_dir"] == str(expected_renamed)
    # Inner artifact subdirs must also land at the renamed root.
    assert paths.configs == expected_renamed / "configs"
    assert paths.sbatch == expected_renamed / "sbatch"
    assert paths.logs == expected_renamed / "logs"


def test_collision_chain_increments_to_3(tmp_path: Path) -> None:
    """Two prior dirs (_canonical_ and ``_2``) force a bump to ``_3``."""
    canonical = tmp_path / "ot-baf" / "myjob"
    _seed_existing_experiment(canonical)
    _seed_existing_experiment(tmp_path / "ot-baf" / "myjob_2")

    exp_args = {"experiments_dir": str(canonical), "job_type": "rl"}
    paths = setup_experiments_dir(exp_args, job_name="myjob")

    expected_renamed = tmp_path / "ot-baf" / "myjob_3"
    assert paths.root == expected_renamed
    assert exp_args["experiments_dir"] == str(expected_renamed)


def test_disable_dedup_skips_rename_and_still_writes_back(tmp_path: Path) -> None:
    """When the resume manager disables dedup, no rename happens AND
    exp_args["experiments_dir"] is still set to the canonical absolute
    path (so callers can rely on the field shape unconditionally)."""
    canonical = tmp_path / "ot-baf" / "myjob"
    _seed_existing_experiment(canonical)

    exp_args = {"experiments_dir": str(canonical), "job_type": "datagen"}
    paths = setup_experiments_dir(exp_args, job_name="myjob", disable_dedup=True)

    assert paths.root == canonical
    assert exp_args["experiments_dir"] == str(canonical)
