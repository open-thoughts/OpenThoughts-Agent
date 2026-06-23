"""Region discovery and region-fit checks for Iris launchers."""

from __future__ import annotations

import json as _json
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional


# Maps the multi-region prefix in a single-region zone name
# ("us-east5" → "us") to the GCS multi-region bucket that's free/cheap
# from any single-region in that prefix. We use multi-region buckets
# (gs://marin-models-us, gs://marin-models-eu) rather than per-zone
# buckets so a job that gets re-scheduled into a sibling zone within
# the same continent (e.g. us-east5 → us-central1) still reads/writes
# locally.
_REGION_PREFIX_TO_BUCKET = {
    "us": "gs://marin-models-us",
    "europe": "gs://marin-models-eu",
}


def _region_prefix(region: str) -> Optional[str]:
    """Return the multi-region prefix ('us', 'europe') for a single-region.

    >>> _region_prefix("us-east5")
    'us'
    >>> _region_prefix("europe-west4")
    'europe'
    >>> _region_prefix("asia-northeast1")  # unmapped
    None
    """
    for prefix in _REGION_PREFIX_TO_BUCKET:
        if region.startswith(prefix + "-"):
            return prefix
    return None


def gcs_bucket_for_region(region: str) -> Optional[str]:
    """Return the cheapest GCS bucket for workers in ``region``.

    Returns the multi-region bucket (``gs://marin-models-us`` /
    ``gs://marin-models-eu``) matching the region's continent. Returns
    None for unmapped regions — callers should fall back rather than
    silently emit a wrong bucket.
    """
    prefix = _region_prefix(region)
    return _REGION_PREFIX_TO_BUCKET.get(prefix) if prefix else None


_GCS_URI_RE = re.compile(r"gs://(marin-models-(?:us|eu))(?:/|$)")


def _scan_yaml_for_gcs_paths(yaml_path: Path) -> List[tuple]:
    """Return ``[(field_dotted_path, gcs_uri, bucket), ...]`` for every gs://marin-models-{us,eu}/...
    string in the YAML at ``yaml_path``.

    Walks the parsed YAML recursively. ``field_dotted_path`` is the
    dotted location ('vllm_server.model_path', 'engine.model', etc.) for
    error messages. ``bucket`` is the matched bucket name (the part used
    for region-fit checks). Returns [] if the file isn't readable or
    isn't valid YAML.
    """
    try:
        import yaml  # PyYAML is already a project dep
    except ImportError:
        return []
    try:
        text = yaml_path.read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
    except (OSError, yaml.YAMLError):
        return []

    matches: List[tuple] = []

    def walk(node, prefix):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{prefix}[{i}]")
        elif isinstance(node, str):
            m = _GCS_URI_RE.search(node)
            if m:
                matches.append((prefix, node, m.group(1)))

    walk(doc, "")
    return matches


def assert_yaml_regions_match_pin(yaml_paths: List[Path], pinned_region: str) -> None:
    """Fail-fast when any YAML hardcodes a GCS bucket in the wrong region.

    Iris pins the job to ``pinned_region`` at submit time. If a YAML
    points at ``gs://marin-models-us/...`` and the worker will live in
    ``europe-west4``, every model-weight read crosses continents at GCS
    egress prices. We refuse to submit in that case and tell the user
    which path is wrong + which bucket to use.

    The launcher already pins the *output* bucket automatically; weight
    paths inside config YAMLs are advisory (we don't auto-rewrite them
    on purpose — silent rewrites mask which mirror is canonical and have
    failed before when the EU mirror was incomplete). The contract is:
    keep your YAML's bucket consistent with the variant's natural
    region, or run with ``--gcs-output-dir`` explicitly to opt out of
    the region pin entirely.
    """
    expected_bucket = gcs_bucket_for_region(pinned_region)
    if not expected_bucket:
        return  # Unmapped region (e.g. asia-*); can't validate.
    expected_bucket_name = expected_bucket.removeprefix("gs://")  # "marin-models-us"
    violations: List[str] = []
    for path in yaml_paths:
        if not path or not Path(path).is_file():
            continue
        for field, uri, bucket in _scan_yaml_for_gcs_paths(Path(path)):
            if bucket != expected_bucket_name:
                violations.append(
                    f"  {path}: {field} = {uri!r}\n"
                    f"    bucket {bucket!r} but iris pinned region {pinned_region} "
                    f"expects {expected_bucket_name!r}"
                )
    if violations:
        raise SystemExit(
            "[iris] YAML model paths point at the wrong region for the iris-pinned "
            f"worker (region={pinned_region}, expected bucket={expected_bucket_name}). "
            "Cross-region GCS reads are expensive; refusing to submit.\n\n"
            + "\n".join(violations)
            + f"\n\nFix: swap the bucket in the YAML to {expected_bucket!r}, or "
            "pass --gcs-output-dir explicitly to opt out of the region pin."
        )


def _iris_query(cluster_config: str, sql: str, timeout: int = 30) -> List[dict]:
    """Run a SQL query against the iris controller, return parsed rows.

    Shells out to the ``iris query -f json`` CLI rather than wrangling
    the connectrpc client directly: it handles tunnel setup, auth, and
    the protobuf round-trip for us. ``shutil.which`` picks up the iris
    binary from whichever venv the launcher is running in (e.g. the
    otagent conda env on the launch host).
    """
    iris_bin = shutil.which("iris")
    if iris_bin is None or not Path(iris_bin).exists():
        raise RuntimeError(
            "iris CLI not found on PATH; run from the Marin/Iris launch "
            "environment before using dynamic region discovery."
        )
    cmd = [iris_bin, "--config", str(cluster_config), "query", "-f", "json", sql]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(
            f"iris query failed (rc={result.returncode}): {result.stderr.strip()}"
        )
    out = result.stdout.strip()
    if not out:
        return []
    return _json.loads(out)


def discover_region_for_tpu(
    cluster_config: str, tpu_spec: str
) -> tuple[Optional[str], List[dict]]:
    """Pick the region with the most v5p/v6e capacity for ``tpu_spec``.

    Queries iris's worker_attributes table to find every region that has
    workers of the requested device variant, ranks them by unassigned-
    worker count (warm/idle workers first), and returns the top region
    plus the full ranked breakdown.

    Returns ``(region, rows)`` where ``region`` may be ``None`` if no
    workers are visible (e.g., scale-group exists but has been scaled to
    zero). Caller falls back to the static default in that case.

    The pin happens at submit time and iris's scheduler honors region
    constraints across preempt-retries, so a job picked into us-east5
    stays in us-east5 (or another us-* zone iris adds later if the
    constraint is multi-region) for its whole lifetime — preserving
    harbor's resume invariant on a stable ``jobs_dir`` location.
    """
    # slice_id is NULL/empty for unassigned (warm) workers. Rank regions
    # by unassigned count (more warm = better odds of avoiding a queue)
    # then by total (tie-breaker for hot regions still likely to recycle
    # soonest).
    sql = (
        "SELECT wa.str_value AS region, "
        "COUNT(*) AS total, "
        "SUM(CASE WHEN w.slice_id IS NULL OR w.slice_id = '' THEN 1 ELSE 0 END) AS unassigned "
        f"FROM workers w JOIN worker_attributes wa ON w.worker_id = wa.worker_id "
        f"WHERE w.device_variant = '{tpu_spec}' AND wa.key = 'region' "
        "GROUP BY wa.str_value "
        "ORDER BY unassigned DESC, total DESC, region"
    )
    rows = _iris_query(cluster_config, sql)
    # Drop rows for regions we can't route to a known bucket — picking
    # one would just re-create the cross-continent problem.
    candidates = [r for r in rows if r.get("region") and gcs_bucket_for_region(r["region"])]
    if not candidates:
        return None, rows
    return candidates[0]["region"], rows


def parse_tpu_vm_count(tpu_spec: Optional[str]) -> int:
    """Return the host-VM count for a TPU variant via iris's topology table.

    Delegates to iris ``get_tpu_topology`` — the same source iris uses to
    auto-set replicas — so the count matches reality across families. The
    old ``chips / 4`` arithmetic was wrong on two axes: chips-per-host
    varies (v6e-8 packs 8 chips on a single host, v5p hosts have 4), and
    v5p suffixes count cores, not chips (``v5p-32`` = 32 cores = 16 chips =
    4 hosts, not ``32 / 4 = 8``). Returns 1 when no TPU is requested; an
    unrecognized variant raises ``ValueError`` from ``get_tpu_topology``
    (fail fast rather than guess).
    """
    if not tpu_spec:
        return 1
    from iris.cli.job import get_tpu_topology

    return get_tpu_topology(tpu_spec).vm_count
