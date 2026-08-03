"""Pre-cache models + datasets to region-local cloud storage, then run HF-offline.

Motivation
----------
At campaign bring-up, dozens of eval pods each do HF-hub online HEAD/etag
validation on the SAME model + dataset files, get rate-limited (HTTP 429,
~271s retry stalls), and stall — even though the bytes are (or can be) served
region-locally from our GCS mirror. This helper makes a launcher, at submit
time, *confirm* the job's model + dataset(s) are present in the region-local
GCS mirror (mirroring HF->GCS only when absent), so the job can run with
``HF_HUB_OFFLINE=1`` + ``TRANSFORMERS_OFFLINE=1`` and never touch HF.

Design contract (safety-first, for shared launch infra)
-------------------------------------------------------
* **Idempotent + fast on a cache-HIT** (the common case): a presence check on
  the region bucket. Zero HF calls, no re-upload of multi-GB weights.
* **Offline is gated on PRESENCE.** ``HF_HUB_OFFLINE`` is emitted ONLY when the
  helper has confirmed every required artifact is present in the region mirror.
  Never emit offline for an artifact that isn't provably there (that is the
  silent-offline-not-found-at-runtime failure we must avoid).
* **Models are NOT mirrored inline from the launch host** by default: a model
  mirror streams tens of GB and belongs on an in-cluster iris worker
  (``scripts/iris/launch_mirror.py hf-to-gcs``), not the Mac. On a model MISS the
  helper reports ``present=False`` (the caller decides: fail loud in ``strict``
  mode, or fall back to today's online behavior in ``auto`` mode). Datasets are
  small (task parquets) and ARE mirrored inline on a miss.
* **Never crashes a launch.** Any cloud-access failure (no GCS creds on the
  launch host, transient error) is caught and reported as "could not verify" ->
  offline is NOT enabled -> the launch proceeds exactly as it does today
  (online). The win only materializes where presence can be confirmed.

GCS layout (reuses ``scripts/iris/mirror_hf_to_gcs.py`` for models):
    gs://marin-models-{us,eu}/ot-agent/models/<org>/<repo>/      (config, tokenizer, *.safetensors, *.py)
    gs://marin-models-{us,eu}/ot-agent/datasets/<org>/<repo>/    (dataset snapshot: *.parquet, ...)

runai_streamer reads the model over the GCS S3-compatible endpoint, so the
serve URI is the ``s3://`` form of the ``gs://`` mirror dir; the launcher sets
``AWS_ENDPOINT_URL=https://storage.googleapis.com`` so ``hpc.iris.env`` aliases
the ``MARIN_HMAC_*`` creds onto ``AWS_*`` for the streamer.
"""

from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from hpc.iris.regions import gcs_bucket_for_region

# Prefixes under each region bucket. Models match mirror_hf_to_gcs's default
# callsite (launch_mirror.py hf-to-gcs -> gs://marin-models-{us,eu}/ot-agent/models).
MODELS_PREFIX = "ot-agent/models"
DATASETS_PREFIX = "ot-agent/datasets"

# The two multi-region buckets that together cover every iris worker region
# with zero cross-region read egress (same fan-out as launch_mirror.py hf-to-gcs).
FANOUT_BUCKETS = ("gs://marin-models-us", "gs://marin-models-eu")

# Fallback region bucket when the launcher couldn't pin a region (unmapped
# zone / no workers visible). US multi-region is the safe default (most slices).
DEFAULT_BUCKET = "gs://marin-models-us"

DATASET_MANIFEST = ".dataset_mirror_manifest.json"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArtifact:
    repo: str
    gs_uri: str            # gs://<bucket>/ot-agent/models/<repo>
    s3_uri: str            # s3://<bucket>/ot-agent/models/<repo>/  (runai_streamer serve path)
    present: bool
    mirrored: bool = False


@dataclass(frozen=True)
class DatasetArtifact:
    repo: str
    gs_uri: str            # gs://<bucket>/ot-agent/datasets/<repo>
    present: bool
    mirrored: bool = False


@dataclass
class PrecacheResult:
    """Outcome of a pre-cache pass. ``offline_ok`` gates the offline flags."""
    model: Optional[ModelArtifact]
    datasets: List[DatasetArtifact]
    offline_ok: bool
    env: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def model_serve_uri(self) -> Optional[str]:
        return self.model.s3_uri if (self.model and self.model.present) else None

    @property
    def dataset_uris(self) -> List[str]:
        return [d.gs_uri for d in self.datasets if d.present]


# ---------------------------------------------------------------------------
# Cloud helpers
# ---------------------------------------------------------------------------


def _bucket_for_region(region: Optional[str]) -> str:
    """Region-local mirror bucket (falls back to US multi-region)."""
    if region:
        b = gcs_bucket_for_region(region)
        if b:
            return b
    return DEFAULT_BUCKET


def gs_to_s3(gs_uri: str) -> str:
    """Rewrite ``gs://bucket/key`` -> ``s3://bucket/key/`` for runai_streamer.

    The trailing slash matches the datagen runai configs (the streamer treats
    the URI as a model DIRECTORY, not a single object).
    """
    assert gs_uri.startswith("gs://"), gs_uri
    return "s3://" + gs_uri[len("gs://"):].rstrip("/") + "/"


def model_gcs_dir(bucket: str, repo: str) -> str:
    return f"{bucket.rstrip('/')}/{MODELS_PREFIX}/{repo}"


def dataset_gcs_dir(bucket: str, repo: str) -> str:
    return f"{bucket.rstrip('/')}/{DATASETS_PREFIX}/{repo}"


def _gcs_fs():
    import gcsfs
    return gcsfs.GCSFileSystem()


def _ls(fs, gs_dir: str) -> List[str]:
    """List file basenames under a gs:// dir; [] if missing/unreadable."""
    try:
        entries = fs.ls(gs_dir, detail=True)
    except (FileNotFoundError, OSError):
        return []
    out = []
    for e in entries:
        name = e.get("name", "") if isinstance(e, dict) else str(e)
        out.append(name.rstrip("/").rsplit("/", 1)[-1])
    return out


def _model_present(fs, gs_dir: str) -> bool:
    """A model mirror is usable iff config.json + >=1 .safetensors are present."""
    names = _ls(fs, gs_dir)
    if not names:
        return False
    has_config = "config.json" in names
    has_weights = any(n.endswith(".safetensors") for n in names)
    return has_config and has_weights


def _dataset_present(fs, gs_dir: str) -> bool:
    """A dataset mirror is usable iff it holds >=1 .parquet (or a raw task tree).

    We recurse one level for parquet because some HF dataset snapshots nest
    parquet under ``data/`` — a top-level ``ls`` alone can miss them.
    """
    names = _ls(fs, gs_dir)
    if not names:
        return False
    if any(n.endswith(".parquet") for n in names):
        return True
    try:
        found = fs.glob(f"{gs_dir.rstrip('/')}/**/*.parquet")
        if found:
            return True
    except (FileNotFoundError, OSError):
        pass
    # Raw task directories (each task = a subdir) also count as present.
    return DATASET_MANIFEST in names


# ---------------------------------------------------------------------------
# Ensure-cached primitives
# ---------------------------------------------------------------------------


def ensure_model_cached(
    repo: str,
    region: Optional[str],
    *,
    allow_mirror: bool = False,
    verbose: bool = True,
) -> ModelArtifact:
    """Confirm ``repo`` is present in the region model mirror (mirror iff allowed).

    ``allow_mirror`` defaults False: on a MISS this returns ``present=False``
    rather than streaming tens of GB from the launch host. Pass True only in a
    context that can afford a full model mirror (an in-cluster iris worker).
    """
    bucket = _bucket_for_region(region)
    gs_uri = model_gcs_dir(bucket, repo)
    s3_uri = gs_to_s3(gs_uri)

    try:
        fs = _gcs_fs()
        present = _model_present(fs, gs_uri)
    except Exception as e:  # noqa: BLE001 - never crash a launch on a cloud hiccup
        if verbose:
            print(f"[precache] model {repo}: could not verify GCS presence ({e}); "
                  "treating as NOT cached.", file=sys.stderr, flush=True)
        return ModelArtifact(repo=repo, gs_uri=gs_uri, s3_uri=s3_uri, present=False)

    if present:
        if verbose:
            print(f"[precache] model {repo}: cache HIT at {gs_uri}", flush=True)
        return ModelArtifact(repo=repo, gs_uri=gs_uri, s3_uri=s3_uri, present=True)

    if not allow_mirror:
        if verbose:
            print(f"[precache] model {repo}: cache MISS at {gs_uri} "
                  "(not mirroring inline from the launch host).", flush=True)
        return ModelArtifact(repo=repo, gs_uri=gs_uri, s3_uri=s3_uri, present=False)

    # Inline mirror (fan out to both region buckets), then re-verify.
    from scripts.iris.mirror_hf_to_gcs import mirror as mirror_model
    dest_prefixes = [f"{b}/{MODELS_PREFIX}" for b in FANOUT_BUCKETS]
    if verbose:
        print(f"[precache] model {repo}: MISS -> mirroring HF -> {dest_prefixes}", flush=True)
    mirror_model(repo, dest_prefixes, verbose=verbose)
    present = _model_present(_gcs_fs(), gs_uri)
    return ModelArtifact(repo=repo, gs_uri=gs_uri, s3_uri=s3_uri, present=present, mirrored=present)


def ensure_dataset_cached(
    repo: str,
    region: Optional[str],
    *,
    allow_mirror: bool = True,
    verbose: bool = True,
) -> DatasetArtifact:
    """Confirm ``repo`` (an HF *dataset*) is present in the region dataset mirror.

    Datasets are small (task parquets), so on a MISS we mirror inline: HF
    ``snapshot_download`` -> upload the snapshot to both region buckets. On a
    HIT this is a cheap presence check (zero HF calls).
    """
    bucket = _bucket_for_region(region)
    gs_uri = dataset_gcs_dir(bucket, repo)

    try:
        fs = _gcs_fs()
        present = _dataset_present(fs, gs_uri)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[precache] dataset {repo}: could not verify GCS presence ({e}); "
                  "treating as NOT cached.", file=sys.stderr, flush=True)
        return DatasetArtifact(repo=repo, gs_uri=gs_uri, present=False)

    if present:
        if verbose:
            print(f"[precache] dataset {repo}: cache HIT at {gs_uri}", flush=True)
        return DatasetArtifact(repo=repo, gs_uri=gs_uri, present=True)

    if not allow_mirror:
        return DatasetArtifact(repo=repo, gs_uri=gs_uri, present=False)

    try:
        _mirror_dataset(repo, verbose=verbose)  # writes to both buckets
        present = _dataset_present(_gcs_fs(), gs_uri)
        return DatasetArtifact(repo=repo, gs_uri=gs_uri, present=present, mirrored=present)
    except Exception as e:  # noqa: BLE001 - a mirror failure must not crash the launch
        if verbose:
            print(f"[precache] dataset {repo}: inline mirror FAILED ({e}); "
                  "will fall back to online.", file=sys.stderr, flush=True)
        return DatasetArtifact(repo=repo, gs_uri=gs_uri, present=False)


def _mirror_dataset(repo: str, *, verbose: bool = True) -> str:
    """HF snapshot_download a dataset repo and upload it to both region buckets.

    One-shot: datasets are MB-scale, so a plain snapshot + recursive put is fine
    (no per-shard streaming needed). Writes a manifest so ``_dataset_present``
    treats a raw-task-tree dataset (no parquet) as present too.
    """
    import json

    from huggingface_hub import snapshot_download

    if verbose:
        print(f"[precache] dataset {repo}: MISS -> snapshot_download from HF", flush=True)
    with tempfile.TemporaryDirectory(prefix="ds_mirror_") as tmp:
        local = Path(snapshot_download(repo_id=repo, repo_type="dataset", local_dir=tmp))
        files = [p for p in local.rglob("*") if p.is_file()]
        manifest = {
            "hf_repo": repo,
            "repo_type": "dataset",
            "mirrored_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(files),
            "iris_job_id": os.environ.get("IRIS_JOB_ID"),
        }
        (local / DATASET_MANIFEST).write_text(json.dumps(manifest, indent=2))
        files.append(local / DATASET_MANIFEST)

        fs = _gcs_fs()
        first_uri = ""
        for bucket in FANOUT_BUCKETS:
            dest = dataset_gcs_dir(bucket, repo)
            first_uri = first_uri or dest
            for f in files:
                rel = f.relative_to(local).as_posix()
                gcs_uri = f"{dest}/{rel}"
                try:
                    info = fs.info(gcs_uri)
                    if int(info.get("size", 0)) == f.stat().st_size:
                        continue  # idempotent: already there at matching size
                except (FileNotFoundError, OSError):
                    pass
                fs.put(str(f), gcs_uri)
            if verbose:
                print(f"[precache] dataset {repo}: uploaded {len(files)} files -> {dest}",
                      flush=True)
        return first_uri


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def precache_for_eval(
    model: str,
    datasets: List[str],
    region: Optional[str],
    *,
    mode: str = "auto",
    allow_model_mirror: bool = False,
    verbose: bool = True,
) -> PrecacheResult:
    """Ensure ``model`` + ``datasets`` are region-cached; return the offline plan.

    ``mode``:
      * ``auto``   — cache-HIT => offline; any MISS => online fallback (no abort).
                     Safest for cron-fired campaign refills: a launch is never
                     blocked, and offline is only enabled when everything is present.
      * ``strict`` — any MISS raises SystemExit at launch (fail loud) with the
                     exact remediation command. Use when you want a hard guarantee
                     that the job runs offline.
      * ``off``    — no-op (byte-identical to today's online behavior).

    Returns a :class:`PrecacheResult`. When ``offline_ok`` is True the caller
    should point the vLLM serve path at ``model_serve_uri``, the dataset(s) at
    ``dataset_uris``, and merge ``env`` (the offline + endpoint vars).
    """
    notes: List[str] = []
    if mode == "off":
        return PrecacheResult(model=None, datasets=[], offline_ok=False,
                              notes=["precache mode=off (online, unchanged)"])

    m = ensure_model_cached(model, region, allow_mirror=allow_model_mirror, verbose=verbose)
    ds = [ensure_dataset_cached(d, region, verbose=verbose) for d in datasets]

    missing: List[str] = []
    if not m.present:
        missing.append(f"model {model}")
    missing += [f"dataset {d.repo}" for d in ds if not d.present]

    if missing:
        bucket = _bucket_for_region(region)
        remediation = (
            f"pre-mirror it, then relaunch:\n"
            f"    python -m scripts.iris.launch_mirror hf-to-gcs --repo {model}\n"
            f"  (region bucket: {bucket}); datasets mirror inline on launch when "
            f"the launch host has GCS write creds."
        )
        msg = ("[precache] NOT offline-eligible; missing from the region mirror: "
               + ", ".join(missing))
        if mode == "strict":
            raise SystemExit(
                msg + "\n" + remediation +
                "\n  (or relaunch with --hf-offline-mode=auto to run online instead)"
            )
        # auto: never block the launch — run online exactly as today.
        notes.append(msg + " -> running ONLINE (unchanged). To fix: " + remediation)
        if verbose:
            print(msg + " -> ONLINE fallback (unchanged behavior).", flush=True)
        return PrecacheResult(model=m, datasets=ds, offline_ok=False, notes=notes)

    # Everything present -> safe to go offline.
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        # runai_streamer reads the model over the GCS S3-compatible endpoint;
        # hpc.iris.env aliases MARIN_HMAC_* -> AWS_* when this endpoint is set.
        "AWS_ENDPOINT_URL": "https://storage.googleapis.com",
    }
    notes.append("all artifacts present in region mirror -> HF_HUB_OFFLINE=1")
    if verbose:
        print(f"[precache] OFFLINE OK: model={m.s3_uri} datasets="
              f"{[d.gs_uri for d in ds]}", flush=True)
    return PrecacheResult(model=m, datasets=ds, offline_ok=True, env=env, notes=notes)
