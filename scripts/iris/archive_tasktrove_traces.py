#!/usr/bin/env python3
"""Archive TaskTrove trace prefixes into durable CoreWeave object storage."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from typing import Callable, Iterable

import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from huggingface_hub import HfApi, hf_hub_download


R2_ENDPOINT = "https://cwobject.com"
GCS_ENDPOINT = "https://storage.googleapis.com"
R2_BUCKET = "marin-us-east-02a"
MANIFEST_NAME = "archive-manifest.json"
MAX_PENDING_FUTURES = 512
COPY_WORKERS = 64
STREAM_WORKERS = 256
CLIENT_LOCAL = threading.local()
STREAM_TRANSFER_CONFIG = TransferConfig(
    multipart_chunksize=16 * 1024 * 1024,
    max_concurrency=1,
    use_threads=False,
)


def s3_client(
    *,
    endpoint: str,
    access_key: str,
    secret_key: str,
    addressing_style: str,
):
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(
            connect_timeout=15,
            read_timeout=180,
            max_pool_connections=512,
            retries={"max_attempts": 10, "mode": "adaptive"},
            s3={"addressing_style": addressing_style},
        ),
    )


def r2_client():
    client = getattr(CLIENT_LOCAL, "r2", None)
    if client is None:
        client = s3_client(
            endpoint=os.environ.get("AWS_ENDPOINT_URL", R2_ENDPOINT),
            access_key=os.environ["AWS_ACCESS_KEY_ID"],
            secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            addressing_style="virtual",
        )
        CLIENT_LOCAL.r2 = client
    return client


def gcs_client():
    client = getattr(CLIENT_LOCAL, "gcs", None)
    if client is None:
        client = s3_client(
            endpoint=GCS_ENDPOINT,
            access_key=os.environ["MARIN_HMAC_ACCESS_ID"],
            secret_key=os.environ["MARIN_HMAC_SECRET"],
            addressing_style="path",
        )
        CLIENT_LOCAL.gcs = client
    return client


def objects(client, bucket: str, prefix: str) -> Iterable[dict]:
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix.rstrip("/") + "/"
    ):
        yield from page.get("Contents", [])


def existing_sizes(prefix: str) -> dict[str, int]:
    normalized = prefix.rstrip("/") + "/"
    return {
        item["Key"][len(normalized) :]: int(item["Size"])
        for item in objects(r2_client(), R2_BUCKET, normalized)
    }


def drain_bounded(
    executor: ThreadPoolExecutor,
    work: Iterable[tuple[Callable, tuple]],
) -> None:
    pending: set[Future] = set()
    for function, arguments in work:
        pending.add(executor.submit(function, *arguments))
        if len(pending) < MAX_PENDING_FUTURES:
            continue
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            future.result()
    for future in pending:
        future.result()


def copy_r2_object(source_key: str, destination_key: str) -> None:
    r2_client().copy_object(
        Bucket=R2_BUCKET,
        Key=destination_key,
        CopySource={"Bucket": R2_BUCKET, "Key": source_key},
    )


def archive_r2_prefix(source_prefix: str, destination_prefix: str) -> dict:
    source_prefix = source_prefix.rstrip("/") + "/"
    destination_prefix = destination_prefix.rstrip("/") + "/"
    destination = existing_sizes(destination_prefix)
    source_count = 0
    source_bytes = 0
    skipped = 0

    def work_items():
        nonlocal source_count, source_bytes, skipped
        for item in objects(r2_client(), R2_BUCKET, source_prefix):
            relative = item["Key"][len(source_prefix) :]
            size = int(item["Size"])
            source_count += 1
            source_bytes += size
            if destination.get(relative) == size:
                skipped += 1
                continue
            yield copy_r2_object, (item["Key"], destination_prefix + relative)

    with ThreadPoolExecutor(max_workers=COPY_WORKERS) as executor:
        drain_bounded(executor, work_items())
    result = verify_prefix(
        source_client=r2_client(),
        source_bucket=R2_BUCKET,
        source_prefix=source_prefix,
        destination_prefix=destination_prefix,
    )
    result.update(
        {
            "kind": "r2",
            "source": f"s3://{R2_BUCKET}/{source_prefix}",
            "destination": f"s3://{R2_BUCKET}/{destination_prefix}",
            "source_count": source_count,
            "source_bytes": source_bytes,
            "skipped_existing": skipped,
        }
    )
    return result


def stream_object(
    source_bucket: str,
    source_key: str,
    destination_key: str,
) -> None:
    response = gcs_client().get_object(Bucket=source_bucket, Key=source_key)
    try:
        r2_client().upload_fileobj(
            response["Body"],
            R2_BUCKET,
            destination_key,
            Config=STREAM_TRANSFER_CONFIG,
        )
    finally:
        response["Body"].close()


def archive_gcs_prefix(source_uri: str, destination_prefix: str) -> dict:
    source_bucket, _, source_prefix = source_uri.removeprefix("gs://").partition("/")
    source_prefix = source_prefix.rstrip("/") + "/"
    destination_prefix = destination_prefix.rstrip("/") + "/"
    destination = existing_sizes(destination_prefix)
    source_count = 0
    source_bytes = 0
    skipped = 0

    def work_items():
        nonlocal source_count, source_bytes, skipped
        for item in objects(gcs_client(), source_bucket, source_prefix):
            relative = item["Key"][len(source_prefix) :]
            size = int(item["Size"])
            source_count += 1
            source_bytes += size
            if destination.get(relative) == size:
                skipped += 1
                continue
            yield (
                stream_object,
                (
                    source_bucket,
                    item["Key"],
                    destination_prefix + relative,
                ),
            )

    with ThreadPoolExecutor(max_workers=STREAM_WORKERS) as executor:
        drain_bounded(executor, work_items())
    result = verify_prefix(
        source_client=gcs_client(),
        source_bucket=source_bucket,
        source_prefix=source_prefix,
        destination_prefix=destination_prefix,
    )
    result.update(
        {
            "kind": "gcs",
            "source": source_uri.rstrip("/") + "/",
            "destination": f"s3://{R2_BUCKET}/{destination_prefix}",
            "source_count": source_count,
            "source_bytes": source_bytes,
            "skipped_existing": skipped,
        }
    )
    return result


def upload_hf_file(repo_id: str, filename: str, destination_key: str) -> None:
    with tempfile.TemporaryDirectory(prefix="tasktrove-hf-archive-") as temporary_dir:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=temporary_dir,
            token=os.environ.get("HF_TOKEN"),
        )
        r2_client().upload_file(local_path, R2_BUCKET, destination_key)


def archive_hf_repo(repo_id: str, destination_prefix: str) -> dict:
    destination_prefix = destination_prefix.rstrip("/") + "/"
    info = HfApi(token=os.environ.get("HF_TOKEN")).repo_info(
        repo_id,
        repo_type="dataset",
        files_metadata=True,
    )
    files = [
        (sibling.rfilename, int(sibling.size or 0))
        for sibling in info.siblings
        if sibling.rfilename and not sibling.rfilename.endswith("/")
    ]
    destination = existing_sizes(destination_prefix)

    def work_items():
        for filename, size in files:
            if destination.get(filename) == size:
                continue
            yield upload_hf_file, (repo_id, filename, destination_prefix + filename)

    with ThreadPoolExecutor(max_workers=STREAM_WORKERS) as executor:
        drain_bounded(executor, work_items())
    archived = existing_sizes(destination_prefix)
    expected = {filename: size for filename, size in files}
    mismatches = {
        filename: {"expected": size, "actual": archived.get(filename)}
        for filename, size in expected.items()
        if archived.get(filename) != size
    }
    if mismatches:
        raise RuntimeError(
            f"HF archive verification failed for {repo_id}: {mismatches}"
        )
    return {
        "kind": "huggingface",
        "source": f"https://huggingface.co/datasets/{repo_id}/tree/{info.sha}",
        "source_revision": info.sha,
        "destination": f"s3://{R2_BUCKET}/{destination_prefix}",
        "source_count": len(files),
        "source_bytes": sum(size for _, size in files),
        "verified": True,
    }


def verify_prefix(
    *,
    source_client,
    source_bucket: str,
    source_prefix: str,
    destination_prefix: str,
) -> dict:
    source_prefix = source_prefix.rstrip("/") + "/"
    destination_prefix = destination_prefix.rstrip("/") + "/"
    destination = existing_sizes(destination_prefix)
    source_count = 0
    source_bytes = 0
    mismatches: list[dict] = []
    for item in objects(source_client, source_bucket, source_prefix):
        relative = item["Key"][len(source_prefix) :]
        size = int(item["Size"])
        source_count += 1
        source_bytes += size
        if destination.get(relative) != size:
            mismatches.append(
                {
                    "key": relative,
                    "expected": size,
                    "actual": destination.get(relative),
                }
            )
            if len(mismatches) >= 20:
                break
    if mismatches:
        raise RuntimeError(f"archive verification failed: {mismatches}")
    return {
        "verified": True,
        "verified_count": source_count,
        "verified_bytes": source_bytes,
    }


def parse_route(raw: str) -> tuple[str, str]:
    source, separator, destination = raw.rpartition("=")
    if not separator or not source or not destination:
        raise argparse.ArgumentTypeError("routes must use SOURCE=DESTINATION")
    return source, destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2-route", action="append", default=[], type=parse_route)
    parser.add_argument("--gcs-route", action="append", default=[], type=parse_route)
    parser.add_argument("--hf-route", action="append", default=[], type=parse_route)
    parser.add_argument("--manifest-prefix", required=True)
    args = parser.parse_args()

    results = []
    for source, destination in args.r2_route:
        results.append(archive_r2_prefix(source, destination))
    for source, destination in args.gcs_route:
        results.append(archive_gcs_prefix(source, destination))
    for source, destination in args.hf_route:
        results.append(archive_hf_repo(source, destination))

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "bucket": R2_BUCKET,
        "routes": results,
        "verified": all(result["verified"] for result in results),
    }
    manifest_key = f"{args.manifest_prefix.rstrip('/')}/{MANIFEST_NAME}"
    r2_client().put_object(
        Bucket=R2_BUCKET,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode(),
        ContentType="application/json",
    )
    print(json.dumps(manifest, indent=2))
    print(f"manifest: s3://{R2_BUCKET}/{manifest_key}")


if __name__ == "__main__":
    main()
