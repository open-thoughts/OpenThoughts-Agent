# OpenML → Harbor → TaskTrove

This source deterministically converts OpenML tasks into Harbor tasks and packages
verified tasks with the repository's existing TaskTrove converter. Generation
does not call an LLM.

OpenML tasks are the discovery unit because they define the dataset, problem
type, and target. Pinned OpenML datasets are the download and cache unit. If
several tasks use the same dataset version, the source is downloaded and loaded
once and then shared by their task-type adapters.

## Setup

Development targets these checkouts:

- OpenThoughts-Agent: `3bd1917e62c9d03d73063b433f5c442c279c0563`
- marin-community/harbor: `4e468d45c91a4aadb6de398a6d8b323aa6223bdf`

Use Python 3.12+ and a working Docker daemon with Docker Compose. From the
OpenThoughts-Agent root, with Harbor cloned beside it:

```bash
uv sync --project ../harbor --frozen --no-dev
uv pip install --python ../harbor/.venv/bin/python -r data/openml_tasktrove/requirements.txt
export OPENML_PYTHON="$PWD/../harbor/.venv/bin/python"
export PATH="$PWD/../harbor/.venv/bin:$PATH"
```

Generated tasks use Harbor schema 1.2 and a separate verifier environment. The
agent image contains train data and unlabeled test data. Hidden labels are copied
only into the verifier image after the agent has stopped.

## Discover tasks

Discovery writes a typed Parquet catalog:

```bash
$OPENML_PYTHON -m data.openml_tasktrove.discover \
  --output /tmp/openml_tasks.parquet \
  --limit 100 \
  --workers 4
```

Use `--task-ids 363290 363291` for specific tasks, or `--task-types 1 2` to
limit the crawl to OpenML task-type IDs. With no task-type filter, discovery
records every active type. Types without an implemented generator remain in the
catalog with `eligible=false` and `reason=unsupported_task_type`.
Task listings are fetched in bounded pages; tune `--page-size` for a large crawl.
Metadata lookups use bounded `--workers`. Dataset contents are loaded once for
each dataset/version group and released before the next group.
Use `--all` for a complete paginated crawl instead of supplying an artificial
large limit.

The checked-in `openml_tasks.parquet` is a small review sample. It contains five
automatically eligible supervised classification tasks and one unsupported or
unlicensed task to demonstrate filtering.

### License policy

`allowed_licenses.toml` is the repository-owned redistribution policy. Discovery
normalizes OpenML's declared dataset license against that allowlist. This is an
automated policy check of the publisher-supplied OpenML metadata, not an
independent legal audit. A matching license is accepted automatically; missing,
ambiguous, and unknown values are ineligible. There are no per-dataset reviewer
or manually edited acceptance fields. Expanding the allowlist changes policy for
the full crawl and should be reviewed as code.

The catalog retains the declared license, canonical license ID and URL,
attribution, original source URL when supplied, OpenML dataset and task pages,
download URL, license evidence URL, dataset checksum, task type, target, and
evaluation metadata. These URLs are separate fields so an OpenML download URL is
never presented as the original source.

### Deduplication

Discovery emits at most one row for each:

```text
(dataset_id, dataset_version, task_type_id, target)
```

When OpenML has several tasks for that key, the lowest task ID is the canonical
representative and the remaining IDs are stored in `duplicate_task_ids`.
Evaluation-procedure variants are intentionally collapsed because this converter
creates its own deterministic holdout split.

## Generate, verify, and package

`generate.py` is the normal entry point and runs the complete pipeline:

```bash
$OPENML_PYTHON -m data.openml_tasktrove.generate \
  --manifest data/openml_tasktrove/openml_tasks.parquet \
  --output-dir /tmp/openml_tasktrove \
  --split-seed 42 \
  --workers 2
```

Use a fresh output directory. The command:

1. reads automatically eligible catalog rows;
2. downloads each pinned dataset version once into `source_cache/`;
3. constructs all tasks that use that dataset;
4. regenerates each task to verify deterministic files and membership;
5. runs Harbor `nop` and `oracle` trials;
6. copies tasks to `accepted/` or `rejected/`;
7. writes `pipeline_summary.csv`; and
8. packages accepted tasks as `tasks.parquet` through
   `scripts/harbor/tasks_parquet_converter.py`.

For offline reproduction, pass `--source-cache DIR`. Cached files are named
`<dataset_id>_v<version>.parquet`; their canonical SHA-256 must still match the
catalog. On Colima, place `--jobs-dir` under a host directory shared with its VM.
`--workers` bounds concurrent Harbor verification; keep it at 1 on small Docker
installations.

Verification and packaging remain independently runnable for debugging:

```bash
$OPENML_PYTHON -m data.openml_tasktrove.verify \
  --tasks-dir /tmp/openml_tasktrove/generated
$OPENML_PYTHON -m data.openml_tasktrove.package \
  --tasks-dir /tmp/openml_tasktrove/accepted \
  --output /tmp/openml_tasktrove/tasks.parquet
```

## Task types and imperfect data

Task construction is selected by the OpenML task-type ID through
`task_types.py`. Each handler selects its split behavior, metric, target
encoding, estimator, instruction partial, solution template, and verifier
template. The current adapters are:

- supervised classification, with deterministic class-stratified splitting and
  accuracy;
- supervised regression, with deterministic global splitting and RMSE.

New OpenML types must define their own instruction, split behavior, prediction
schema, metric, verifier, and reference solution. Merely adding an ID to the
registry is insufficient for types such as clustering or survival analysis.

Missing feature values and columns containing only missing values remain in the
task. The reference solution imputes numeric and categorical features. Rows with
missing targets cannot be graded, so they are deterministically excluded and the
count is recorded in `provenance.json`. A dataset fails only when the task
contract cannot be built, for example when no usable target rows remain.

## Determinism and grading

Columns are sorted, scalar values and missing values are normalized, and all
source rows are serialized as canonical JSON. The dataset checksum covers rows
with and without target values. SHA-256 fingerprints plus duplicate occurrence
indices provide stable source identities. Split membership ranks
SHA-256(seed + source identity), within each class for classification.

Stable row IDs use HMAC-SHA-256 keyed by the complete dataset checksum. CSV files
have fixed column order, UTF-8 encoding, and LF line endings. Classification
targets are encoded as deterministic integer codes. Regression reports raw RMSE
alongside a separate validity reward.

The verifier rejects missing files, wrong columns, duplicate, missing, or extra
row IDs, non-finite predictions, and invalid class codes. The reference solution
uses the same public files as the agent.

## Tests

```bash
uv run --with openml==0.15.1 --with jinja2==3.1.6 \
  pytest tests/test_openml_tasktrove.py -q
OPENML_HARBOR_E2E=1 $OPENML_PYTHON -m pytest \
  tests/test_openml_tasktrove.py -k pipeline_twice -s
```

The unit suite covers typed manifest round trips, automatic license filtering,
task deduplication, missing features and targets, deterministic generation,
malformed submissions, checksum drift, and package gating. The opt-in test runs
real Harbor containers twice and compares task membership and file hashes.
