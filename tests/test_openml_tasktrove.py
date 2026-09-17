"""Behavior tests; the opt-in end-to-end test runs real Harbor containers."""

import importlib.util
import json
import os
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq
import pytest

from data.openml_tasktrove import common, discover, generate, package, verify
from scripts.harbor.tasks_parquet_converter import safe_extract_tar


def fixture_source(tmp_path, task_type="binary", missing_target=False):
    labels = [str(i % (3 if task_type == "multiclass" else 2)) for i in range(60)]
    task_type_id = 1
    task_type_name = "Supervised Classification"
    if task_type == "regression":
        labels = [i / 10 for i in range(60)]
        task_type_id = 2
        task_type_name = "Supervised Regression"
    if missing_target:
        labels[0] = None
    df = pd.DataFrame(
        {
            "x": [i % 6 for i in range(60)],
            "category": ["a", "b"] * 30,
            "empty": [None] * 60,
            "target": labels,
        }
    )
    df = pd.concat([df, df.iloc[:6]], ignore_index=True)
    _, _, checksum = common.canonicalize(df, "target")
    row = dict.fromkeys(common.MANIFEST_FIELDS)
    row.update(
        openml_task_id=999999999,
        openml_task_type_id=task_type_id,
        openml_task_type=task_type_name,
        openml_dataset_id=999999999,
        dataset_version=1,
        dataset_name="synthetic-test-fixture",
        target="target",
        estimation_procedure="holdout",
        evaluation_measure="predictive_accuracy",
        n_rows=len(df),
        n_features=3,
        original_license="CC0-1.0",
        license_id="CC0-1.0",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution="Locally constructed converter test fixture.",
        source_url="urn:openml-tasktrove:test-fixture",
        original_data_url="urn:openml-tasktrove:original-test-fixture",
        openml_dataset_url="https://www.openml.org/d/999999999",
        openml_task_url="https://www.openml.org/t/999999999",
        data_url="urn:openml-tasktrove:test-fixture-data",
        license_evidence_url="urn:openml-tasktrove:test-license-evidence",
        eligible=True,
        reason="",
        split_seed=42,
        test_fraction=0.2,
        dataset_checksum=checksum,
        canonical_key=f"999999999:v1:t{task_type_id}:target",
        duplicate_task_ids=[],
    )
    manifest = tmp_path / "manifest.parquet"
    common.write_manifest(manifest, [row])
    cache = tmp_path / "cache"
    cache.mkdir()
    df.to_parquet(common.cache_path(cache, 999999999, 1), index=False)
    return row, df, manifest, cache


def import_file(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task_type", ["binary", "multiclass", "regression"])
def test_generation_is_order_independent_and_complete(tmp_path, task_type):
    row, df, manifest, cache = fixture_source(tmp_path, task_type)
    first = generate.build_tasks(manifest, tmp_path / "one", 42, cache)[0]
    second = generate.render_task(
        row, df.sample(frac=1, random_state=37), tmp_path / "two"
    )
    assert verify.file_hashes(first) == verify.file_hashes(second)
    verify.check_determinism(first, cache)
    canonical, _, checksum = common.canonicalize(df.iloc[::-1], "target")
    assert checksum == row["dataset_checksum"]
    split_type = "regression" if task_type == "regression" else "classification"
    train, test = common.deterministic_split(canonical, "target", split_type)
    assert {record["row_id"] for record in train}.isdisjoint(
        record["row_id"] for record in test
    )
    assert {record["row_id"] for record in train + test} == {
        record["row_id"] for record in canonical
    }
    assert len({record["row_id"] for record in canonical}) == len(df)
    assert not list((first / "environment").rglob("*hidden*"))
    solution = import_file(first / "solution/solution.py")
    verifier = import_file(first / "tests/verifier.py")
    submission = tmp_path / "submission.csv"
    solution.solve(first / "environment/data", submission)
    rewards = verifier.score(submission, first / "tests/hidden_labels.csv")
    assert rewards["valid"] == 1
    assert rewards.get("accuracy", 1) > 0


def test_missing_features_and_targets_are_retained_or_recorded(tmp_path):
    row, df, _, _ = fixture_source(tmp_path, missing_target=True)
    task = generate.render_task(row, df, tmp_path / "tasks")
    metadata = json.loads((task / "provenance.json").read_text())
    train = pd.read_csv(task / "environment/data/train.csv")
    test = pd.read_csv(task / "environment/data/test.csv")
    assert metadata["missing_target_rows"] == 2
    assert len(train) + len(test) == len(df) - 2
    assert "empty" in train and train["empty"].isna().all()


def test_generation_loads_shared_dataset_once(tmp_path, monkeypatch):
    row, df, manifest, _ = fixture_source(tmp_path)
    regression = dict(row)
    regression.update(
        openml_task_id=1000000000,
        openml_task_type_id=2,
        openml_task_type="Supervised Regression",
        canonical_key="999999999:v1:t2:target",
    )
    common.write_manifest(manifest, [row, regression])
    calls = []

    def load_once(candidate, source_cache):
        calls.append((candidate["openml_dataset_id"], candidate["dataset_version"]))
        return df

    monkeypatch.setattr(generate, "load_dataset", load_once)
    tasks = generate.build_tasks(manifest, tmp_path / "tasks", 42)
    assert len(tasks) == 2
    assert calls == [(999999999, 1)]


def test_discovery_filters_licenses_and_collapses_duplicates(tmp_path, monkeypatch):
    frame = pd.DataFrame({"x": range(20), "target": ["a", "b"] * 10})
    loads = []

    class Dataset:
        version = 1
        name = "fixture"
        licence = "CC0"
        citation = "Fixture authors"
        original_data_url = "https://example.test/source"

        def get_data(self, dataset_format):
            assert dataset_format == "dataframe"
            loads.append(dataset_format)
            return frame, None, None, None

    def metadata(task_id):
        task_type_id = 1 if task_id < 3 else 2
        return Dataset(), {
            "openml_task_id": task_id,
            "openml_task_type_id": task_type_id,
            "openml_task_type": "classification" if task_type_id == 1 else "regression",
            "openml_dataset_id": 10,
            "dataset_version": 1,
            "dataset_name": "fixture",
            "target": "target",
            "estimation_procedure": "holdout",
            "evaluation_measure": "accuracy",
            "original_license": "CC0",
            "attribution": "Fixture authors",
            "source_url": "https://example.test/source",
            "canonical_key": f"10:v1:t{task_type_id}:target",
        }

    monkeypatch.setattr(discover, "_task_ids", lambda *args: [1, 2, 3])
    monkeypatch.setattr(discover, "_metadata", metadata)
    output = tmp_path / "catalog.parquet"
    rows = discover.run(output, limit=3)
    assert len(rows) == 2
    assert rows[0]["duplicate_task_ids"] == [2]
    assert all(row["eligible"] for row in rows)
    assert len({row["canonical_key"] for row in rows}) == 2
    assert loads == ["dataframe"]
    assert common.read_manifest(output)[0]["license_id"] == "CC0-1.0"
    assert pq.read_schema(output) == common.MANIFEST_SCHEMA


def test_task_listing_balances_requested_types(monkeypatch):
    task_ids = {1: [11, 12, 13], 2: [21, 22, 23]}

    def list_tasks(task_type, offset, size, **kwargs):
        values = task_ids[task_type.value][offset : offset + size]
        return pd.DataFrame({"tid": values})

    monkeypatch.setattr(discover.openml.tasks, "list_tasks", list_tasks)
    assert discover._task_ids(4, task_types=[1, 2], page_size=2) == [11, 12, 21, 22]


def test_task_listing_all_stops_at_last_page(monkeypatch):
    task_ids = [31, 32, 33]
    calls = []

    def list_tasks(offset, size, **kwargs):
        calls.append((offset, size))
        return pd.DataFrame({"tid": task_ids[offset : offset + size]})

    monkeypatch.setattr(discover.openml.tasks, "list_tasks", list_tasks)
    assert discover._task_ids(None, page_size=2) == task_ids
    assert calls == [(0, 2), (2, 2)]


def test_discovery_excludes_unknown_license_without_manual_review(
    tmp_path, monkeypatch
):
    frame = pd.DataFrame({"x": range(20), "target": ["a", "b"] * 10})

    class Dataset:
        def get_data(self, dataset_format):
            return frame, None, None, None

    values = dict(fixture_source(tmp_path)[0])
    values.update(
        openml_task_id=1,
        original_license="Public",
        canonical_key="10:v1:t1:target",
    )
    monkeypatch.setattr(discover, "_task_ids", lambda *args: [1])
    monkeypatch.setattr(discover, "_metadata", lambda task_id: (Dataset(), values))
    rows = discover.run(tmp_path / "catalog.parquet", limit=1)
    assert not rows[0]["eligible"]
    assert rows[0]["reason"] == "license_not_allowlisted"


@pytest.mark.parametrize(
    "malformation",
    [
        "duplicate",
        "missing",
        "extra",
        "nan",
        "inf",
        "columns",
        "fraction",
        "range",
        "no_file",
    ],
)
def test_malformed_submissions_fail(tmp_path, malformation):
    row, df, _, _ = fixture_source(tmp_path)
    task = generate.render_task(row, df, tmp_path / "tasks")
    verifier = import_file(task / "tests/verifier.py")
    labels = task / "tests/hidden_labels.csv"
    submission = tmp_path / "submission.csv"
    predictions = pd.read_csv(labels).rename(columns={"target": "prediction"})
    if malformation == "duplicate":
        predictions = pd.concat([predictions, predictions.iloc[:1]])
    elif malformation == "missing":
        predictions = predictions.iloc[1:]
    elif malformation == "extra":
        predictions.loc[len(predictions)] = ["extra", 0]
    elif malformation in {"nan", "inf", "fraction", "range"}:
        predictions["prediction"] = predictions["prediction"].astype(float)
        predictions.loc[0, "prediction"] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "fraction": 0.5,
            "range": 3,
        }[malformation]
    elif malformation == "columns":
        predictions = predictions.rename(columns={"prediction": "wrong"})
    if malformation != "no_file":
        predictions.to_csv(submission, index=False)
    with pytest.raises(ValueError):
        verifier.score(submission, labels)


def test_source_seed_and_duplicate_drift_fail(tmp_path):
    row, df, manifest, cache = fixture_source(tmp_path)
    with pytest.raises(ValueError, match="seed"):
        generate.build_tasks(manifest, tmp_path / "seed", 123, cache)
    df.loc[0, "x"] = 100
    with pytest.raises(ValueError, match="content changed"):
        generate.render_task(row, df, tmp_path / "changed")
    row["eligible"] = False
    common.write_manifest(manifest, [row])
    with pytest.raises(ValueError, match="No automatically eligible"):
        generate.build_tasks(manifest, tmp_path / "ineligible", 42, cache)
    row["eligible"] = True
    common.write_manifest(manifest, [row, row])
    with pytest.raises(ValueError, match="Duplicate canonical"):
        generate.build_tasks(manifest, tmp_path / "duplicate", 42, cache)


def test_package_refuses_unverified_task(tmp_path):
    row, df, _, _ = fixture_source(tmp_path)
    generate.render_task(row, df, tmp_path / "tasks")
    with pytest.raises(FileNotFoundError):
        package.run(tmp_path / "tasks", tmp_path / "tasks.parquet")


def test_canonicalization_preserves_large_integer_changes():
    df = pd.DataFrame({"x": [2**60] * 10, "target": ["a", "b"] * 5})
    before = common.canonicalize(df, "target")[2]
    df.loc[0, "x"] += 1
    assert common.canonicalize(df, "target")[2] != before


@pytest.mark.skipif(
    os.environ.get("OPENML_HARBOR_E2E") != "1",
    reason="Set OPENML_HARBOR_E2E=1 to run real Docker trials",
)
def test_pipeline_twice_with_real_harbor(tmp_path):
    row, _, manifest, cache = fixture_source(tmp_path)
    outputs = [tmp_path / "run1", tmp_path / "run2"]
    for output in outputs:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "data.openml_tasktrove.generate",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(output),
                "--source-cache",
                str(cache),
                "--split-seed",
                "42",
            ],
            check=True,
        )
    task_name = generate.task_id_for(row)
    first = outputs[0] / "accepted" / task_name
    second = outputs[1] / "accepted" / task_name
    assert verify.file_hashes(first) == verify.file_hashes(second)
    for relative in common.DATA_FILES[:2]:
        assert verify.row_ids(first / relative) == verify.row_ids(second / relative)
    tables = [pq.read_table(output / "tasks.parquet") for output in outputs]
    assert tables[0]["path"].to_pylist() == tables[1]["path"].to_pylist() == [task_name]
    assert [table.num_rows for table in tables] == [1, 1]
    roundtrip = tmp_path / "roundtrip"
    safe_extract_tar(tables[0]["task_binary"][0].as_py(), roundtrip)
    assert verify.file_hashes(roundtrip) == verify.file_hashes(first)
    receipt = json.loads((first / "verification.json").read_text())
    assert receipt["result"]["oracle_passed"] and receipt["result"]["empty_failed"]
    (first / "instruction.md").write_text("tampered")
    with pytest.raises(ValueError, match="changed after verification"):
        package.run(first.parent, tmp_path / "tampered.parquet")
