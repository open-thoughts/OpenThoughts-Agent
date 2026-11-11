"""Typed manifests, canonical source data, caching, and stable splits."""

import csv
import hashlib
import hmac
import json
import math
from datetime import date, datetime
from numbers import Integral, Real
from pathlib import Path

import openml
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MANIFEST_FIELDS = [
    "openml_task_id",
    "openml_task_type_id",
    "openml_task_type",
    "openml_dataset_id",
    "dataset_version",
    "dataset_name",
    "target",
    "estimation_procedure",
    "evaluation_measure",
    "n_rows",
    "n_features",
    "original_license",
    "license_id",
    "license_url",
    "attribution",
    "source_url",
    "original_data_url",
    "openml_dataset_url",
    "openml_task_url",
    "data_url",
    "license_evidence_url",
    "eligible",
    "reason",
    "split_seed",
    "test_fraction",
    "dataset_checksum",
    "canonical_key",
    "duplicate_task_ids",
]
MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("openml_task_id", pa.int64()),
        pa.field("openml_task_type_id", pa.int64()),
        pa.field("openml_task_type", pa.string()),
        pa.field("openml_dataset_id", pa.int64()),
        pa.field("dataset_version", pa.int64()),
        pa.field("dataset_name", pa.string()),
        pa.field("target", pa.string()),
        pa.field("estimation_procedure", pa.string()),
        pa.field("evaluation_measure", pa.string()),
        pa.field("n_rows", pa.int64()),
        pa.field("n_features", pa.int64()),
        pa.field("original_license", pa.string()),
        pa.field("license_id", pa.string()),
        pa.field("license_url", pa.string()),
        pa.field("attribution", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("original_data_url", pa.string()),
        pa.field("openml_dataset_url", pa.string()),
        pa.field("openml_task_url", pa.string()),
        pa.field("data_url", pa.string()),
        pa.field("license_evidence_url", pa.string()),
        pa.field("eligible", pa.bool_()),
        pa.field("reason", pa.string()),
        pa.field("split_seed", pa.int64()),
        pa.field("test_fraction", pa.float64()),
        pa.field("dataset_checksum", pa.string()),
        pa.field("canonical_key", pa.string()),
        pa.field("duplicate_task_ids", pa.list_(pa.int64())),
    ]
)
DATA_FILES = [
    "environment/data/train.csv",
    "environment/data/test.csv",
    "tests/hidden_labels.csv",
    "instruction.md",
    "task.toml",
]


def canonical_json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(path, rows):
    """Write the authoritative, typed discovery catalog."""
    path = Path(path)
    if path.suffix != ".parquet":
        raise ValueError("OpenML manifests must use the .parquet extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    def clean(value):
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if value is None or (not isinstance(value, dict) and pd.isna(value)):
            return None
        return value

    records = [
        {field: clean(row.get(field)) for field in MANIFEST_FIELDS} for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(records, schema=MANIFEST_SCHEMA), path)


def read_manifest(path):
    path = Path(path)
    if path.suffix != ".parquet":
        raise ValueError("OpenML manifests must use the .parquet extension")
    table = pq.read_table(path)
    missing = set(MANIFEST_FIELDS) - set(table.column_names)
    if missing:
        raise ValueError(f"Manifest lacks columns: {sorted(missing)}")
    return table.select(MANIFEST_FIELDS).to_pylist()


def normalize(value):
    if value is None or (
        not isinstance(value, (list, tuple, dict)) and pd.isna(value)
    ):
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Infinite source value")
        return format(number if number else 0.0, ".17g")
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if not isinstance(value, (list, tuple, dict, set)):
        return str(value)
    raise ValueError(f"Unsupported nested feature value: {type(value).__name__}")


def canonicalize(df, target, *, include_stats=False):
    """Canonicalize labeled rows while checksumming the complete source table."""
    if not df.columns.is_unique or not all(isinstance(c, str) for c in df.columns):
        raise ValueError("Columns must be unique strings")
    if target not in df or "row_id" in df:
        raise ValueError("Missing target or reserved row_id column")
    if len(df) < 2 or len(df.columns) < 2:
        raise ValueError("A supervised task needs at least two rows and one feature")
    columns = sorted(c for c in df if c != target) + [target]
    all_payloads = []
    labeled_payloads = []
    for row in df[columns].itertuples(index=False, name=None):
        values = [normalize(value) for value in row]
        payload = canonical_json(values)
        all_payloads.append(payload)
        if values[-1] is not None:
            labeled_payloads.append(payload)
    all_payloads.sort()
    labeled_payloads.sort()
    checksum = hashlib.sha256(
        canonical_json([columns, all_payloads]).encode()
    ).hexdigest()
    rows, counts = [], {}
    for payload in labeled_payloads:
        fingerprint = hashlib.sha256(payload.encode()).hexdigest()
        occurrence = counts.get(fingerprint, 0)
        counts[fingerprint] = occurrence + 1
        source_id = f"{fingerprint}:{occurrence}"
        row = dict(zip(columns, json.loads(payload), strict=True))
        row["source_id"] = source_id
        row["row_id"] = hmac.new(
            bytes.fromhex(checksum), source_id.encode(), "sha256"
        ).hexdigest()
        rows.append(row)
    if len(rows) < 2:
        raise ValueError("Fewer than two rows have a target value")
    result = (sorted(rows, key=lambda row: row["source_id"]), columns, checksum)
    if include_stats:
        stats = {
            "source_rows": len(df),
            "missing_target_rows": len(df) - len(rows),
        }
        return (*result, stats)
    return result


def deterministic_split(rows, target, task_type, test_fraction=0.2, seed=42):
    if task_type not in {"classification", "regression"}:
        raise ValueError(f"Unsupported split task type: {task_type}")
    if not 0 < test_fraction < 1:
        raise ValueError("Invalid test fraction")
    groups = {}
    for row in rows:
        key = row[target] if task_type == "classification" else "all"
        groups.setdefault(key, []).append(row)
    train, test = [], []
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda row: (
                hashlib.sha256(f"{seed}:{row['source_id']}".encode()).hexdigest(),
                row["source_id"],
            ),
        )
        if len(group) == 1:
            train.extend(group)
            continue
        count = max(1, min(len(group) - 1, math.floor(len(group) * test_fraction)))
        test.extend(group[:count])
        train.extend(group[count:])
    if not train or not test:
        raise ValueError("Could not create nonempty train and test partitions")
    return sorted(train, key=lambda row: row["row_id"]), sorted(
        test, key=lambda row: row["row_id"]
    )


def cache_path(source_cache, dataset_id, version):
    return Path(source_cache) / f"{int(dataset_id)}_v{int(version)}.parquet"


def load_dataset(row, source_cache=None):
    """Load one pinned dataset, populating a shared on-disk cache when requested."""
    dataset_id = int(row["openml_dataset_id"])
    version = int(row["dataset_version"])
    cached = cache_path(source_cache, dataset_id, version) if source_cache else None
    if cached and cached.is_file():
        return pd.read_parquet(cached)
    dataset = openml.datasets.get_dataset(
        dataset_id,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    if int(dataset.version) != version:
        raise ValueError("Pinned OpenML version mismatch")
    if (dataset.licence or "") != (row.get("original_license") or ""):
        raise ValueError("OpenML license metadata changed")
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        temporary = cached.with_suffix(".tmp.parquet")
        df.to_parquet(temporary, index=False)
        temporary.replace(cached)
    return df


def eligible_rows(manifest, split_seed=None):
    rows = [row for row in read_manifest(manifest) if bool(row["eligible"])]
    if not rows:
        raise ValueError("No automatically eligible OpenML tasks in manifest")
    keys = set()
    for row in rows:
        for field in (
            "openml_task_id",
            "openml_task_type_id",
            "openml_dataset_id",
            "dataset_version",
            "target",
            "dataset_checksum",
            "license_id",
            "license_url",
            "license_evidence_url",
            "attribution",
            "source_url",
            "openml_dataset_url",
            "openml_task_url",
            "canonical_key",
        ):
            if row.get(field) is None or row.get(field) == "":
                raise ValueError(f"Eligible task {row['openml_task_id']} lacks {field}")
        if len(bytes.fromhex(row["dataset_checksum"])) != 32:
            raise ValueError("Expected SHA-256 dataset_checksum")
        if row["canonical_key"] in keys:
            raise ValueError(f"Duplicate canonical task key: {row['canonical_key']}")
        keys.add(row["canonical_key"])
        if split_seed is not None and int(row["split_seed"]) != split_seed:
            raise ValueError("CLI split seed must match the manifest")
    return sorted(rows, key=lambda row: row["canonical_key"])
