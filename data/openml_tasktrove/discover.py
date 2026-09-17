"""Discover OpenML tasks and write a deduplicated, policy-filtered Parquet catalog."""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openml

from .common import MANIFEST_FIELDS, canonicalize, deterministic_split, write_manifest
from .licenses import load_policy, resolve_license
from .task_types import handler_for, supports


def _empty_row(task_id):
    row = dict.fromkeys(MANIFEST_FIELDS)
    row.update(
        openml_task_id=int(task_id),
        eligible=False,
        reason="",
        split_seed=42,
        test_fraction=0.2,
        duplicate_task_ids=[],
    )
    return row


def _task_ids(limit, ids=None, task_types=None, page_size=1000):
    if ids:
        selected = sorted(set(int(task_id) for task_id in ids))
        return selected if limit is None else selected[:limit]
    found = []
    active = [(task_type, 0) for task_type in (task_types or [None])]
    while active and (limit is None or len(set(found)) < limit):
        next_page = []
        for index, (task_type, offset) in enumerate(active):
            filters_left = len(active) - index
            size = page_size
            if limit is not None:
                remaining = limit - len(set(found))
                size = min(page_size, max(1, math.ceil(remaining / filters_left)))
            kwargs = {
                "offset": offset,
                "size": size,
                "status": "active",
                "output_format": "dataframe",
            }
            if task_type is not None:
                kwargs["task_type"] = openml.tasks.TaskType(int(task_type))
            frame = openml.tasks.list_tasks(**kwargs)
            if "tid" not in frame or frame.empty:
                continue
            found.extend(int(task_id) for task_id in frame["tid"].tolist())
            if len(frame) >= size:
                next_page.append((task_type, offset + len(frame)))
            if limit is not None and len(set(found)) >= limit:
                break
        active = next_page
    selected = sorted(set(found))
    return selected if limit is None else selected[:limit]


def _metadata(task_id):
    task = openml.tasks.get_task(
        int(task_id),
        download_splits=False,
        download_data=False,
        download_qualities=True,
        download_features_meta_data=True,
    )
    dataset = task.get_dataset()
    target = getattr(task, "target_name", None)
    task_type_id = int(getattr(task.task_type_id, "value", task.task_type_id))
    dataset_id = int(task.dataset_id)
    version = int(dataset.version)
    canonical_key = f"{dataset_id}:v{version}:t{task_type_id}:{target or ''}"
    openml_dataset_url = f"https://www.openml.org/d/{dataset_id}"
    original_data_url = dataset.original_data_url or None
    return dataset, {
        "openml_task_id": int(task_id),
        "openml_task_type_id": task_type_id,
        "openml_task_type": str(task.task_type),
        "openml_dataset_id": dataset_id,
        "dataset_version": version,
        "dataset_name": dataset.name,
        "target": target,
        "estimation_procedure": task.estimation_procedure.get("type"),
        "evaluation_measure": getattr(task, "evaluation_measure", None),
        "original_license": dataset.licence or "",
        "attribution": dataset.citation or f"OpenML dataset {dataset_id}",
        "source_url": original_data_url or openml_dataset_url,
        "original_data_url": original_data_url,
        "openml_dataset_url": openml_dataset_url,
        "openml_task_url": f"https://www.openml.org/t/{task_id}",
        "data_url": getattr(dataset, "url", None),
        "license_evidence_url": openml_dataset_url,
        "canonical_key": canonical_key,
    }


def _inspect(row, df, policy, load_error=None):
    license_match = resolve_license(row["original_license"], policy)
    if license_match:
        row.update(license_match)
    reasons = []
    if not license_match:
        reasons.append("license_not_allowlisted")
    if not supports(row["openml_task_type_id"]):
        reasons.append("unsupported_task_type")
    if not row["target"]:
        reasons.append("missing_target_definition")
    try:
        if load_error is not None:
            raise load_error
        row.update(n_rows=len(df), n_features=max(0, len(df.columns) - 1))
        if row["target"]:
            rows, columns, checksum, stats = canonicalize(
                df, row["target"], include_stats=True
            )
            row.update(n_features=len(columns) - 1, dataset_checksum=checksum)
            if supports(row["openml_task_type_id"]):
                handler = handler_for(row["openml_task_type_id"])
                deterministic_split(
                    rows,
                    row["target"],
                    handler.split_type,
                    row["test_fraction"],
                    row["split_seed"],
                )
            if stats["missing_target_rows"]:
                reasons.append(
                    f"info:excluded_{stats['missing_target_rows']}_missing_target_rows"
                )
    except Exception as exc:
        reasons.append(f"data_unusable:{type(exc).__name__}:{exc}")
    blocking = [reason for reason in reasons if not reason.startswith("info:")]
    row["eligible"] = not blocking
    row["reason"] = ";".join(reasons)
    return row


def _candidate(task_id):
    row = _empty_row(task_id)
    try:
        dataset, values = _metadata(task_id)
        row.update(values)
        return row, dataset
    except Exception as exc:
        row["reason"] = f"metadata_unavailable:{type(exc).__name__}:{exc}"
        return row, None


def run(output, limit=20, ids=None, task_types=None, page_size=1000, workers=1):
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    policy = load_policy()
    task_ids = _task_ids(limit, ids, task_types, page_size)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        candidates = list(executor.map(_candidate, task_ids))

    grouped = {}
    failures = []
    for row, dataset in candidates:
        key = row.get("canonical_key")
        if not key:
            failures.append(row)
            continue
        grouped.setdefault(key, []).append((row, dataset))
    representatives = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda pair: pair[0]["openml_task_id"])
        row, dataset = group[0]
        row["duplicate_task_ids"] = [
            pair[0]["openml_task_id"] for pair in group[1:]
        ]
        representatives.append((row, dataset))
    by_dataset = {}
    for row, dataset in representatives:
        dataset_key = (row["openml_dataset_id"], row["dataset_version"])
        by_dataset.setdefault(dataset_key, []).append((row, dataset))
    rows = []
    for dataset_key in sorted(by_dataset):
        dataset_group = by_dataset[dataset_key]
        try:
            df = dataset_group[0][1].get_data(dataset_format="dataframe")[0]
            load_error = None
        except Exception as exc:
            df, load_error = None, exc
        for row, _ in dataset_group:
            rows.append(_inspect(row, df, policy, load_error))
        del df
    rows.extend(failures)
    rows.sort(key=lambda row: row["openml_task_id"])
    write_manifest(output, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--all", action="store_true", help="crawl all matching active OpenML tasks"
    )
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--task-types", type=int, nargs="+")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    run(
        args.output,
        None if args.all else args.limit,
        args.task_ids,
        args.task_types,
        args.page_size,
        args.workers,
    )


if __name__ == "__main__":
    main()
