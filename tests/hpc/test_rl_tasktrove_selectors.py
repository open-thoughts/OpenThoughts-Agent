import importlib
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from hpc import hf_utils, rl_launch_utils
from hpc.hf_utils import HfDatasetSelector, parse_hf_dataset_selector
from hpc.rl_launch_utils import RLJobConfig


def test_cache_names_do_not_confuse_repo_suffixes_with_subdirectories():
    repo_suffix = HfDatasetSelector("fixture-org/a__b", revision="commit").cache_name()
    subdirectory = HfDatasetSelector(
        "fixture-org/a", revision="commit", subdir="b"
    ).cache_name()

    assert repo_suffix != subdirectory


def test_revision_resolution_honors_the_requested_revision(monkeypatch):
    calls = []

    class FakeApi:
        def dataset_info(self, repo_id, revision):
            calls.append((repo_id, revision))
            return SimpleNamespace(sha="immutable-sha")

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    selector = hf_utils.resolve_hf_dataset_selector(
        "fixture-org/fixture-trove@rev-1::a"
    )

    assert selector.canonical() == "fixture-org/fixture-trove@immutable-sha::a"
    assert calls == [("fixture-org/fixture-trove", "rev-1")]


def test_task_selectors_use_distinct_revision_pinned_caches(monkeypatch, tmp_path):
    commits = {"a": "commit-a", "b": "commit-b"}
    commands = []

    def resolve(value):
        selector = parse_hf_dataset_selector(value)
        assert selector is not None and selector.subdir is not None
        return HfDatasetSelector(
            selector.repo_id, commits[selector.subdir], selector.subdir
        )

    def run(command, **_kwargs):
        commands.append(command)
        output = Path(command[command.index("--output_dir") + 1])
        (output / "task-1").mkdir(parents=True)
        (output / "task-1" / "instruction.md").write_text("task")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(rl_launch_utils, "resolve_hf_dataset_selector", resolve)
    monkeypatch.setattr(rl_launch_utils.subprocess, "run", run)
    monkeypatch.setattr(
        rl_launch_utils, "_fix_task_permissions", lambda *_args, **_kwargs: None
    )

    resolved = rl_launch_utils.resolve_rl_train_data_with_sources(
        ["fixture-org/fixture-trove::a", "fixture-org/fixture-trove::b"],
        scratch_dir=str(tmp_path),
        verbose=False,
    )

    assert resolved.sources == (
        "fixture-org/fixture-trove@commit-a::a",
        "fixture-org/fixture-trove@commit-b::b",
    )
    assert resolved.paths[0] != resolved.paths[1]
    assert all(
        (Path(path) / "task-1" / "instruction.md").read_text() == "task"
        for path in resolved.paths
    )
    assert [command[command.index("--parquet") + 1] for command in commands] == list(
        resolved.sources
    )


def test_rendered_job_config_keeps_canonical_sources():
    config = RLJobConfig(
        job_name="job",
        experiments_dir="experiments",
        cluster_name="jupiter",
        skyrl_entrypoint="fully_async",
        trials_dir="trials",
        train_data=["/tasks/a"],
        train_data_sources=["fixture-org/fixture-trove@immutable-sha::a"],
    )

    assert asdict(config)["train_data_sources"] == [
        "fixture-org/fixture-trove@immutable-sha::a"
    ]


def test_hf_extractor_import_does_not_load_google_cloud_storage():
    for name in tuple(sys.modules):
        if name == "google.cloud.storage" or name.startswith("google.cloud.storage."):
            del sys.modules[name]

    importlib.import_module("scripts.datagen.extract_tasks_from_parquet")

    assert "google.cloud.storage" not in sys.modules
