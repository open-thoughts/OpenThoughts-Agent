#!/usr/bin/env python3
"""Mirror a HuggingFace model repo to GCS, one file at a time.

Designed to run on a small iris CPU worker without depending on
fitting the full model on the worker's ephemeral disk: each safetensors
shard is downloaded, uploaded to GCS, then deleted before the next
shard starts. Tokenizer + config files are tiny and processed first.

Usage::

    python -m scripts.iris.mirror_hf_to_gcs \\
        --repo cyankiwi/MiniMax-M2.7-AWQ-4bit \\
        --gcs-prefix gs://marin-eu-west4/ot-agent/models/

The resulting GCS layout::

    gs://marin-eu-west4/ot-agent/models/cyankiwi/MiniMax-M2.7-AWQ-4bit/
        config.json
        tokenizer.json
        model-00001-of-00050.safetensors
        ...

Idempotent: files already present in GCS with a matching size are
skipped. Re-run to resume an interrupted mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from scripts.iris.hf_model_files import select_model_files, selection_policy

# Filename for the per-repo manifest written to the GCS prefix at the
# end of a successful mirror. Picked up by
# ``hpc.model_mirror_registry.refresh_from_gcs`` to populate the local
# catalog.
MANIFEST_FILENAME = ".mirror_manifest.json"


def _gcs_fs():
    """Return a cached fsspec GCS filesystem instance.

    Uses gcsfs's default credential discovery — picks up workload-identity
    creds on iris workers and ADC creds locally. We require gcsfs (and
    transitively fsspec) which are already in the OT-Agent venv via the
    [datagen-tpu] extra (datasets/huggingface_hub pull them in).
    """
    import gcsfs
    return gcsfs.GCSFileSystem()


def _gcs_size(uri: str) -> int | None:
    """Return the size in bytes of an existing GCS object, or None if missing.

    Tolerates any gcsfs failure mode (auth, missing key, transient errors)
    by returning None — the caller treats that as "need to upload".
    """
    try:
        info = _gcs_fs().info(uri)
        return int(info.get("size", 0))
    except (FileNotFoundError, OSError):
        return None
    except Exception as e:
        # Be lenient so a transient hiccup doesn't make the whole mirror
        # skip uploads; treat as "unknown / re-upload".
        print(f"[mirror] _gcs_size({uri}) unexpected error: {e}",
              file=sys.stderr, flush=True)
        return None


def _upload(local_path: Path, gcs_uri: str) -> None:
    """Stream-upload one file via gcsfs.put. Raise on any error."""
    fs = _gcs_fs()
    # gcsfs.put handles chunked upload for large files.
    fs.put(str(local_path), gcs_uri)


def _write_manifest(
    dest_prefix: str,
    *,
    repo_id: str,
    files_mirrored: list[tuple[str, int]],
    iris_job_id: str | None,
) -> None:
    """Write the per-repo manifest to GCS for the local registry to discover."""
    manifest = {
        "hf_repo": repo_id,
        "mirrored_at": datetime.now(timezone.utc).isoformat(),
        "mirror_script": "scripts/iris/mirror_hf_to_gcs.py",
        "file_count": len(files_mirrored),
        "size_bytes": sum(sz for _, sz in files_mirrored),
        "files": [{"name": n, "size": sz} for n, sz in files_mirrored],
        # The POLICY, not just the resulting file list: a manifest that records only
        # what was copied cannot answer "why is this file missing?" later.
        "selection_policy": selection_policy(),
        "iris_job_id": iris_job_id,
    }
    manifest_uri = f"{dest_prefix}/{MANIFEST_FILENAME}"
    fs = _gcs_fs()
    with fs.open(manifest_uri, "w") as f:
        json.dump(manifest, f, indent=2)


def mirror(repo_id: str, gcs_prefixes: list[str], *, verbose: bool = True,
           iris_job_id: str | None = None) -> None:
    """Mirror ``repo_id`` to ``<prefix>/<repo_id>/`` for every prefix in ``gcs_prefixes``.

    Mirrors one file at a time via huggingface_hub.hf_hub_download so the
    local disk never holds more than one shard at once. Each downloaded
    file is uploaded to every prefix before being deleted. Writes a
    ``.mirror_manifest.json`` to every prefix on successful completion
    so ``hpc.model_mirror_registry.refresh_from_gcs`` can index each.

    The multi-prefix fan-out is how we keep one canonical copy in every
    GCS region that iris workers run in — currently ``marin-models-us``
    (US multi-region) and ``marin-models-eu`` (EU multi-region), which
    together cover all 6 iris regions with zero cross-region egress at
    read time.
    """
    from huggingface_hub import HfApi, hf_hub_download

    if isinstance(gcs_prefixes, str):
        gcs_prefixes = [gcs_prefixes]
    gcs_prefixes = [p.rstrip("/") for p in gcs_prefixes]
    dest_prefixes = [f"{p}/{repo_id}" for p in gcs_prefixes]

    api = HfApi()
    files = sorted(api.list_repo_files(repo_id, repo_type="model"))

    if verbose:
        for dest in dest_prefixes:
            print(f"[mirror] {repo_id} -> {dest}", flush=True)
        print(f"[mirror] repo has {len(files)} files", flush=True)

    # Selection and metadata-first ordering both live in hf_model_files, shared with the
    # S3 route so the two mirrors cannot drift.
    keep = select_model_files(files)
    if verbose:
        print(f"[mirror] mirroring {len(keep)} files to {len(dest_prefixes)} prefix(es)",
              flush=True)
        skipped = sorted(set(files) - set(keep))
        if skipped:
            print(f"[mirror] skipping {len(skipped)}: {', '.join(skipped)}", flush=True)

    files_mirrored: list[tuple[str, int]] = []

    for idx, fname in enumerate(keep, 1):
        # Per-destination idempotency: only upload to prefixes that
        # don't already have the file. This lets a re-run skip work in
        # the bucket where the previous attempt finished and fill the
        # other from HF.
        targets = []
        skipped_size = None
        for dest in dest_prefixes:
            gcs_uri = f"{dest}/{fname}"
            remote_size = _gcs_size(gcs_uri)
            if remote_size is not None and remote_size > 0:
                if verbose:
                    print(f"[mirror] [{idx}/{len(keep)}] skip "
                          f"(already in GCS, {remote_size} bytes): {gcs_uri}",
                          flush=True)
                skipped_size = remote_size
            else:
                targets.append(gcs_uri)

        if not targets:
            # Already in every destination — record canonical size and continue.
            if skipped_size is not None:
                files_mirrored.append((fname, skipped_size))
            continue

        with tempfile.TemporaryDirectory(prefix="hf_mirror_") as tmp:
            tmp_path = Path(tmp)
            if verbose:
                print(f"[mirror] [{idx}/{len(keep)}] download: {fname}",
                      flush=True)
            local_file = hf_hub_download(
                repo_id=repo_id,
                filename=fname,
                local_dir=str(tmp_path),
                # Skip cache because we delete on temp-dir exit.
                local_dir_use_symlinks=False,
            )
            local_file = Path(local_file)
            local_size = local_file.stat().st_size

            for gcs_uri in targets:
                if verbose:
                    print(f"[mirror] [{idx}/{len(keep)}] upload "
                          f"({local_size} bytes): {gcs_uri}", flush=True)
                _upload(local_file, gcs_uri)
            files_mirrored.append((fname, local_size))
            # TemporaryDirectory will rm everything on exit, but be
            # explicit on the unlink so failure modes are clearer.
            try:
                local_file.unlink(missing_ok=True)
            except OSError:
                pass

    # Write the per-repo manifest last so a partial / interrupted mirror
    # is detectable (manifest missing -> incomplete). Write one per dest.
    for dest in dest_prefixes:
        _write_manifest(
            dest,
            repo_id=repo_id,
            files_mirrored=files_mirrored,
            iris_job_id=iris_job_id,
        )

    if verbose:
        total = sum(sz for _, sz in files_mirrored)
        print(f"[mirror] done: {repo_id} -> {len(dest_prefixes)} dest(s) "
              f"({len(files_mirrored)} files, {total} bytes per dest); "
              f"manifests at <dest>/{MANIFEST_FILENAME}", flush=True)


def _legacy_main() -> int:
    p = argparse.ArgumentParser(
        description="Stream one or more HuggingFace model repos into a GCS prefix.",
    )
    p.add_argument("--repo", action="append", required=True,
                   help="HF model repo id (org/name); repeatable.")
    p.add_argument("--gcs-prefix", "--gcs_prefix", action="append", required=True,
                   help="GCS prefix; each repo lands under <prefix>/<repo>/. "
                        "Repeatable — every prefix gets a full mirror, so a "
                        "single HF download fans out to every region. Default "
                        "callsite (launch_mirror.py hf-to-gcs) passes both "
                        "gs://marin-models-us/ot-agent/models and "
                        "gs://marin-models-eu/ot-agent/models so iris workers "
                        "in any region read locally.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-file progress lines.")
    p.add_argument("--iris-job-id", default=os.environ.get("IRIS_JOB_ID"),
                   help="Record this job id in each repo's manifest. "
                        "Defaults to $IRIS_JOB_ID if set on the worker.")
    args = p.parse_args()

    bad = [p for p in args.gcs_prefix if not p.startswith("gs://")]
    if bad:
        print(f"error: --gcs-prefix must start with gs:// (got {bad!r})",
              file=sys.stderr)
        return 2

    for repo in args.repo:
        mirror(repo, args.gcs_prefix, verbose=not args.quiet,
               iris_job_id=args.iris_job_id)
    return 0


def main() -> int:
    """Compatibility entrypoint; use ``mirror_models hf-to-gcs`` for new calls."""
    from scripts.iris.mirror_models import main as unified_main

    return unified_main(["hf-to-gcs", *sys.argv[1:]])


if __name__ == "__main__":
    sys.exit(main())
