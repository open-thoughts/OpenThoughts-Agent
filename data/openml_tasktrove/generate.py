"""Generate, verify, and package Harbor tasks from an OpenML task catalog."""

import argparse
import hashlib
import json
from itertools import groupby
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .common import (
    canonical_json,
    canonicalize,
    deterministic_split,
    eligible_rows,
    load_dataset,
    write_csv,
)
from .task_types import handler_for

TEMPLATES = Path(__file__).parent / "templates"


def task_id_for(row):
    target_hash = hashlib.sha256(row["target"].encode()).hexdigest()[:8]
    return (
        f"openml_d{int(row['openml_dataset_id'])}"
        f"_v{int(row['dataset_version'])}"
        f"_t{int(row['openml_task_type_id'])}_{target_hash}"
    )


def render_task(row, df, output_dir):
    handler = handler_for(row["openml_task_type_id"])
    rows, columns, checksum, stats = canonicalize(
        df, row["target"], include_stats=True
    )
    if checksum != row["dataset_checksum"]:
        raise ValueError(
            f"Dataset content changed for OpenML dataset {row['openml_dataset_id']}"
        )
    if len(df) != int(row["n_rows"]) or len(columns) - 1 != int(row["n_features"]):
        raise ValueError("Dataset dimensions differ from manifest")
    seed, fraction = int(row["split_seed"]), float(row["test_fraction"])
    train, test = deterministic_split(
        rows, row["target"], handler.split_type, fraction, seed
    )
    features = columns[:-1]
    numeric = [column for column in features if pd.api.types.is_numeric_dtype(df[column])]
    classes = (
        sorted({record[row["target"]] for record in rows})
        if handler.encode_classes
        else []
    )
    if handler.encode_classes and len(classes) < 2:
        raise ValueError("Classification needs at least two target classes")
    mapping = {label: str(index) for index, label in enumerate(classes)}

    def record(source, labeled):
        value = {"row_id": source["row_id"], **{c: source[c] for c in features}}
        if labeled:
            value[row["target"]] = (
                mapping[source[row["target"]]] if classes else source[row["target"]]
            )
        return value

    task_id = task_id_for(row)
    task_dir = Path(output_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=False)
    train_records = [record(source, True) for source in train]
    test_records = [record(source, True) for source in test]
    write_csv(
        task_dir / "environment/data/train.csv",
        train_records,
        ["row_id", *columns],
    )
    write_csv(
        task_dir / "environment/data/test.csv",
        [
            {key: value for key, value in source.items() if key != row["target"]}
            for source in test_records
        ],
        ["row_id", *features],
    )
    write_csv(
        task_dir / "tests/hidden_labels.csv",
        [
            {"row_id": source["row_id"], "target": source[row["target"]]}
            for source in test_records
        ],
        ["row_id", "target"],
    )
    metadata = dict(
        row,
        task_id=task_id,
        task_type=handler.key,
        metric=handler.metric,
        train_rows=len(train),
        test_rows=len(test),
        missing_target_rows=stats["missing_target_rows"],
    )
    (task_dir / "provenance.json").write_text(canonical_json(metadata) + "\n")
    attribution = (
        f"{row['attribution']}\n\nSource: {row['source_url']}\n"
        f"OpenML dataset: {row['openml_dataset_url']}\n"
        f"OpenML task: {row['openml_task_url']}\n"
        f"License: {row['license_id']} ({row['license_url']})\n"
        f"License metadata: {row['license_evidence_url']}\n\n"
        "Changes: canonicalized rows and columns, normalized missing values, "
        "excluded rows without targets, added row IDs, created a deterministic "
        "train/test split, and encoded classification targets as integer codes. "
        "Test labels are withheld from the agent.\n"
    )
    (task_dir / "environment/data/ATTRIBUTION.md").write_text(
        attribution, encoding="utf-8"
    )
    config = {
        "target": row["target"],
        "features": features,
        "numeric": numeric,
        "classes": classes,
        "metric": handler.metric,
        "task_type": handler.key,
        "prediction_kind": handler.prediction_kind,
        "estimator": handler.estimator,
        "seed": seed,
    }
    environment = Environment(
        loader=FileSystemLoader(TEMPLATES),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    context = {
        "config": config,
        "config_json": json.dumps(config),
        "task_id": task_id,
        "metric": handler.metric,
        "target": row["target"],
        "classes": classes,
        "task_type": handler.key,
        "instruction_template": handler.instruction_template,
    }
    for template, destination in {
        "instruction.md.j2": "instruction.md",
        "task.toml.j2": "task.toml",
        "Dockerfile.j2": "environment/Dockerfile",
        handler.verifier_template: "tests/verifier.py",
        handler.solution_template: "solution/solution.py",
        "verifier.Dockerfile.j2": "tests/Dockerfile",
    }.items():
        path = task_dir / destination
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            environment.get_template(template).render(**context), encoding="utf-8"
        )
    for destination, command in {
        "solution/solve.sh": "python -I /solution/solution.py",
        "tests/test.sh": "python -I /tests/verifier.py",
    }.items():
        path = task_dir / destination
        path.write_text(f"#!/bin/sh\nset -eu\n{command}\n")
        path.chmod(0o755)
    return task_dir


def build_tasks(manifest, output_dir, split_seed=None, source_cache=None):
    """Build tasks while loading each pinned dataset only once."""
    rows = eligible_rows(manifest, split_seed)
    tasks = []
    def dataset_key(row):
        return int(row["openml_dataset_id"]), int(row["dataset_version"])
    for _, task_rows in groupby(sorted(rows, key=dataset_key), key=dataset_key):
        task_rows = list(task_rows)
        dataset = load_dataset(task_rows[0], source_cache)
        tasks.extend(render_task(row, dataset, output_dir) for row in task_rows)
        del dataset
    return tasks


def run(
    manifest,
    output_dir,
    split_seed=42,
    source_cache=None,
    jobs_dir=None,
    harbor="harbor",
    workers=1,
):
    """Run the complete OpenML-to-TaskTrove pipeline."""
    from . import package, verify

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    cache = Path(source_cache) if source_cache else output_dir / "source_cache"
    build_tasks(manifest, output_dir / "generated", split_seed, cache)
    accepted, results = verify.run(
        output_dir / "generated",
        output_dir,
        cache,
        jobs_dir,
        harbor,
        workers,
    )
    if any(result["status"] == "accepted" for result in results):
        package.run(accepted, output_dir / "tasks.parquet")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--jobs-dir", type=Path)
    parser.add_argument("--harbor", default="harbor")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    results = run(
        args.manifest,
        args.output_dir,
        args.split_seed if args.split_seed is not None else 42,
        args.source_cache,
        args.jobs_dir,
        args.harbor,
        args.workers,
    )
    if any(result["status"] != "accepted" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
