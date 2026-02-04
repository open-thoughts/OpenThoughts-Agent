"""S3/MinIO storage utilities for Beam cluster artifact caching.

This module provides functions to interact with S3-compatible object storage
(MinIO, AWS S3, etc.) for caching built container artifacts.

Environment variables:
    LAION_BUCKET_NAME: S3 bucket name
    LAION_ACCESS_KEY: S3 access key
    LAION_SECRET_KEY: S3 secret key
    LAION_ENDPOINT: S3 endpoint URL (e.g., https://just-object.fz-juelich.de)

Usage:
    from scripts.beam.s3_storage import S3Storage, S3StorageConfig

    config = S3StorageConfig.from_env()
    storage = S3Storage(config)

    # Check if artifact exists
    if storage.artifact_exists("my-image:latest"):
        storage.download_artifact("my-image:latest", "/tmp/image.tar")
    else:
        # Build and upload
        storage.upload_artifact("/tmp/image.tar", "my-image:latest")
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class S3StorageConfig:
    """Configuration for S3/MinIO storage backend."""

    bucket_name: str
    access_key: str
    secret_key: str
    endpoint_url: str
    namespace: str = "beam-artifacts"  # Dedicated prefix/namespace for beam storage
    region: str = "us-east-1"  # Default region (may not matter for MinIO)

    @classmethod
    def from_env(cls) -> "S3StorageConfig":
        """Create config from environment variables.

        Required environment variables:
            LAION_BUCKET_NAME: S3 bucket name
            LAION_ACCESS_KEY: S3 access key
            LAION_SECRET_KEY: S3 secret key
            LAION_ENDPOINT: S3 endpoint URL

        Optional:
            BEAM_S3_NAMESPACE: Namespace/prefix for beam artifacts (default: beam-artifacts)
        """
        bucket_name = os.environ.get("LAION_BUCKET_NAME")
        access_key = os.environ.get("LAION_ACCESS_KEY")
        secret_key = os.environ.get("LAION_SECRET_KEY")
        endpoint_url = os.environ.get("LAION_ENDPOINT")

        missing = []
        if not bucket_name:
            missing.append("LAION_BUCKET_NAME")
        if not access_key:
            missing.append("LAION_ACCESS_KEY")
        if not secret_key:
            missing.append("LAION_SECRET_KEY")
        if not endpoint_url:
            missing.append("LAION_ENDPOINT")

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        # Auto-add https:// prefix if missing from endpoint
        if endpoint_url and not endpoint_url.startswith(("http://", "https://")):
            endpoint_url = f"https://{endpoint_url}"
            logger.debug(f"Added https:// prefix to endpoint: {endpoint_url}")

        namespace = os.environ.get("BEAM_S3_NAMESPACE", "beam-artifacts")

        return cls(
            bucket_name=bucket_name,
            access_key=access_key,
            secret_key=secret_key,
            endpoint_url=endpoint_url,
            namespace=namespace,
        )

    @classmethod
    def from_env_optional(cls) -> Optional["S3StorageConfig"]:
        """Create config from environment variables, returning None if not configured."""
        try:
            return cls.from_env()
        except ValueError:
            return None


class S3Storage:
    """S3/MinIO storage client for artifact caching.

    This class provides methods to upload, download, and check for cached
    container artifacts in S3-compatible object storage.

    Artifacts are stored with the following structure:
        {namespace}/
            manifests/
                images.json          # Index of all cached images
            images/
                {image_hash}.tar     # Container image tarballs
            metadata/
                {image_hash}.json    # Metadata for each image
    """

    MANIFEST_KEY = "manifests/images.json"
    IMAGES_PREFIX = "images/"
    METADATA_PREFIX = "metadata/"

    def __init__(self, config: S3StorageConfig):
        """Initialize S3 storage client.

        Args:
            config: S3 storage configuration.
        """
        self.config = config
        self._client = None
        self._s3fs = None

    @property
    def client(self):
        """Lazy-load boto3 S3 client."""
        if self._client is None:
            import boto3

            # Unset AWS_SESSION_TOKEN to prevent conflicts with explicit credentials.
            # boto3 will use session token from env even when explicit creds are provided,
            # causing "InvalidTokenId" errors with S3-compatible storage (MinIO, JSC S3).
            if "AWS_SESSION_TOKEN" in os.environ:
                logger.debug("Unsetting AWS_SESSION_TOKEN to use explicit S3 credentials")
                del os.environ["AWS_SESSION_TOKEN"]

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key,
                aws_secret_access_key=self.config.secret_key,
                region_name=self.config.region,
            )
        return self._client

    @property
    def fs(self):
        """Lazy-load s3fs filesystem interface."""
        if self._s3fs is None:
            import s3fs

            # Unset AWS_SESSION_TOKEN to prevent conflicts (same reason as boto3 client)
            if "AWS_SESSION_TOKEN" in os.environ:
                logger.debug("Unsetting AWS_SESSION_TOKEN to use explicit S3 credentials")
                del os.environ["AWS_SESSION_TOKEN"]

            self._s3fs = s3fs.S3FileSystem(
                endpoint_url=self.config.endpoint_url,
                key=self.config.access_key,
                secret=self.config.secret_key,
            )
        return self._s3fs

    def _full_key(self, key: str) -> str:
        """Get full S3 key with namespace prefix."""
        return f"{self.config.namespace}/{key}"

    def _full_path(self, key: str) -> str:
        """Get full S3 path (bucket + namespace + key) for s3fs."""
        return f"{self.config.bucket_name}/{self._full_key(key)}"

    @staticmethod
    def _image_hash(image_name: str, dockerfile_content: Optional[str] = None) -> str:
        """Generate a hash for an image based on name and optional Dockerfile content.

        Args:
            image_name: Image name/tag (e.g., "my-image:latest")
            dockerfile_content: Optional Dockerfile content for content-based hashing.

        Returns:
            SHA256 hash prefix (first 16 chars).
        """
        content = image_name
        if dockerfile_content:
            content += dockerfile_content
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def ensure_namespace_exists(self) -> bool:
        """Ensure the namespace (prefix) exists by creating a marker file.

        Returns:
            True if namespace is accessible.
        """
        try:
            # Create a marker file to ensure the namespace exists
            marker_key = self._full_key(".namespace_marker")
            self.client.put_object(
                Bucket=self.config.bucket_name,
                Key=marker_key,
                Body=json.dumps({
                    "created_at": time.time(),
                    "namespace": self.config.namespace,
                    "description": "Beam cluster artifact cache",
                }).encode(),
                ContentType="application/json",
            )
            logger.info(f"Namespace '{self.config.namespace}' is ready in bucket '{self.config.bucket_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to access/create namespace: {e}")
            return False

    def list_artifacts(self) -> list[dict[str, Any]]:
        """List all cached artifacts.

        Returns:
            List of artifact metadata dictionaries.
        """
        try:
            manifest = self._load_manifest()
            return manifest.get("artifacts", [])
        except Exception as e:
            logger.warning(f"Failed to list artifacts: {e}")
            return []

    def _load_manifest(self) -> dict[str, Any]:
        """Load the artifact manifest from S3."""
        try:
            response = self.client.get_object(
                Bucket=self.config.bucket_name,
                Key=self._full_key(self.MANIFEST_KEY),
            )
            return json.loads(response["Body"].read().decode())
        except self.client.exceptions.NoSuchKey:
            return {"artifacts": [], "updated_at": None}
        except Exception as e:
            logger.debug(f"Failed to load manifest: {e}")
            return {"artifacts": [], "updated_at": None}

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Save the artifact manifest to S3."""
        manifest["updated_at"] = time.time()
        self.client.put_object(
            Bucket=self.config.bucket_name,
            Key=self._full_key(self.MANIFEST_KEY),
            Body=json.dumps(manifest, indent=2).encode(),
            ContentType="application/json",
        )

    def artifact_exists(
        self,
        image_name: str,
        dockerfile_content: Optional[str] = None,
    ) -> bool:
        """Check if an artifact exists in the cache.

        Args:
            image_name: Image name/tag to check.
            dockerfile_content: Optional Dockerfile content for content-based lookup.

        Returns:
            True if artifact exists and is valid.
        """
        image_hash = self._image_hash(image_name, dockerfile_content)
        tar_key = self._full_key(f"{self.IMAGES_PREFIX}{image_hash}.tar")

        try:
            self.client.head_object(
                Bucket=self.config.bucket_name,
                Key=tar_key,
            )
            logger.debug(f"Artifact exists: {image_name} (hash: {image_hash})")
            return True
        except Exception:
            return False

    def get_artifact_url(
        self,
        image_name: str,
        dockerfile_content: Optional[str] = None,
        expiry: int = 3600,
    ) -> Optional[str]:
        """Get a presigned URL for downloading an artifact.

        Args:
            image_name: Image name/tag.
            dockerfile_content: Optional Dockerfile content for content-based lookup.
            expiry: URL expiry time in seconds (default: 1 hour).

        Returns:
            Presigned URL if artifact exists, None otherwise.
        """
        if not self.artifact_exists(image_name, dockerfile_content):
            return None

        image_hash = self._image_hash(image_name, dockerfile_content)
        tar_key = self._full_key(f"{self.IMAGES_PREFIX}{image_hash}.tar")

        try:
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket_name, "Key": tar_key},
                ExpiresIn=expiry,
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None

    def upload_artifact(
        self,
        local_path: Path | str,
        image_name: str,
        dockerfile_content: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Upload a container artifact to S3.

        Args:
            local_path: Path to the local tarball.
            image_name: Image name/tag for the artifact.
            dockerfile_content: Optional Dockerfile content for content-based hashing.
            metadata: Optional metadata to store with the artifact.

        Returns:
            True if upload successful.
        """
        local_path = Path(local_path)
        if not local_path.exists():
            logger.error(f"Local file does not exist: {local_path}")
            return False

        image_hash = self._image_hash(image_name, dockerfile_content)
        tar_key = self._full_key(f"{self.IMAGES_PREFIX}{image_hash}.tar")
        metadata_key = self._full_key(f"{self.METADATA_PREFIX}{image_hash}.json")

        try:
            # Upload tarball
            logger.info(f"Uploading artifact: {image_name} -> s3://{self.config.bucket_name}/{tar_key}")
            file_size = local_path.stat().st_size
            with open(local_path, "rb") as f:
                self.client.upload_fileobj(
                    f,
                    self.config.bucket_name,
                    tar_key,
                    ExtraArgs={"ContentType": "application/x-tar"},
                )

            # Upload metadata
            artifact_metadata = {
                "image_name": image_name,
                "image_hash": image_hash,
                "dockerfile_hash": hashlib.sha256((dockerfile_content or "").encode()).hexdigest()
                if dockerfile_content
                else None,
                "size_bytes": file_size,
                "uploaded_at": time.time(),
                **(metadata or {}),
            }
            self.client.put_object(
                Bucket=self.config.bucket_name,
                Key=metadata_key,
                Body=json.dumps(artifact_metadata, indent=2).encode(),
                ContentType="application/json",
            )

            # Update manifest
            manifest = self._load_manifest()
            # Remove existing entry for this image if present
            manifest["artifacts"] = [
                a for a in manifest.get("artifacts", []) if a.get("image_hash") != image_hash
            ]
            manifest["artifacts"].append(artifact_metadata)
            self._save_manifest(manifest)

            logger.info(f"Artifact uploaded successfully: {image_name} ({file_size / 1024 / 1024:.1f} MB)")
            return True

        except Exception as e:
            logger.error(f"Failed to upload artifact: {e}")
            return False

    def download_artifact(
        self,
        image_name: str,
        local_path: Path | str,
        dockerfile_content: Optional[str] = None,
    ) -> bool:
        """Download a container artifact from S3.

        Args:
            image_name: Image name/tag to download.
            local_path: Local path to save the tarball.
            dockerfile_content: Optional Dockerfile content for content-based lookup.

        Returns:
            True if download successful.
        """
        if not self.artifact_exists(image_name, dockerfile_content):
            logger.error(f"Artifact does not exist: {image_name}")
            return False

        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        image_hash = self._image_hash(image_name, dockerfile_content)
        tar_key = self._full_key(f"{self.IMAGES_PREFIX}{image_hash}.tar")

        try:
            logger.info(f"Downloading artifact: {image_name} -> {local_path}")
            with open(local_path, "wb") as f:
                self.client.download_fileobj(
                    self.config.bucket_name,
                    tar_key,
                    f,
                )

            # Set 777 permissions on downloaded file
            local_path.chmod(0o777)

            logger.info(f"Artifact downloaded successfully: {local_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to download artifact: {e}")
            return False

    def delete_artifact(
        self,
        image_name: str,
        dockerfile_content: Optional[str] = None,
    ) -> bool:
        """Delete an artifact from S3.

        Args:
            image_name: Image name/tag to delete.
            dockerfile_content: Optional Dockerfile content for content-based lookup.

        Returns:
            True if deletion successful.
        """
        image_hash = self._image_hash(image_name, dockerfile_content)
        tar_key = self._full_key(f"{self.IMAGES_PREFIX}{image_hash}.tar")
        metadata_key = self._full_key(f"{self.METADATA_PREFIX}{image_hash}.json")

        try:
            # Delete tarball and metadata
            self.client.delete_object(Bucket=self.config.bucket_name, Key=tar_key)
            self.client.delete_object(Bucket=self.config.bucket_name, Key=metadata_key)

            # Update manifest
            manifest = self._load_manifest()
            manifest["artifacts"] = [
                a for a in manifest.get("artifacts", []) if a.get("image_hash") != image_hash
            ]
            self._save_manifest(manifest)

            logger.info(f"Artifact deleted: {image_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete artifact: {e}")
            return False

    def get_artifact_metadata(
        self,
        image_name: str,
        dockerfile_content: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        """Get metadata for a cached artifact.

        Args:
            image_name: Image name/tag.
            dockerfile_content: Optional Dockerfile content for content-based lookup.

        Returns:
            Artifact metadata dictionary, or None if not found.
        """
        image_hash = self._image_hash(image_name, dockerfile_content)
        metadata_key = self._full_key(f"{self.METADATA_PREFIX}{image_hash}.json")

        try:
            response = self.client.get_object(
                Bucket=self.config.bucket_name,
                Key=metadata_key,
            )
            return json.loads(response["Body"].read().decode())
        except Exception:
            return None

    def cleanup_old_artifacts(self, max_age_days: int = 30) -> int:
        """Clean up artifacts older than the specified age.

        Args:
            max_age_days: Maximum age in days for artifacts to keep.

        Returns:
            Number of artifacts deleted.
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        manifest = self._load_manifest()
        deleted_count = 0

        for artifact in list(manifest.get("artifacts", [])):
            uploaded_at = artifact.get("uploaded_at", 0)
            if uploaded_at < cutoff_time:
                image_name = artifact.get("image_name", "unknown")
                image_hash = artifact.get("image_hash")
                if image_hash:
                    # Delete the artifact
                    tar_key = self._full_key(f"{self.IMAGES_PREFIX}{image_hash}.tar")
                    metadata_key = self._full_key(f"{self.METADATA_PREFIX}{image_hash}.json")
                    try:
                        self.client.delete_object(Bucket=self.config.bucket_name, Key=tar_key)
                        self.client.delete_object(Bucket=self.config.bucket_name, Key=metadata_key)
                        manifest["artifacts"].remove(artifact)
                        deleted_count += 1
                        logger.info(f"Deleted old artifact: {image_name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete old artifact {image_name}: {e}")

        if deleted_count > 0:
            self._save_manifest(manifest)

        return deleted_count


def check_s3_configured() -> bool:
    """Check if S3 storage is configured via environment variables.

    Returns:
        True if all required environment variables are set.
    """
    required_vars = ["LAION_BUCKET_NAME", "LAION_ACCESS_KEY", "LAION_SECRET_KEY", "LAION_ENDPOINT"]
    return all(os.environ.get(var) for var in required_vars)


def init_s3_storage() -> Optional[S3Storage]:
    """Initialize S3 storage from environment variables.

    Returns:
        S3Storage instance if configured, None otherwise.
    """
    config = S3StorageConfig.from_env_optional()
    if config is None:
        return None
    return S3Storage(config)


def cleanup_juicefs_storage(
    bucket_name: Optional[str] = None,
    juicefs_prefix: str = "beta9-fs",
) -> bool:
    """Clean up JuiceFS storage in S3 to allow fresh cluster deployments.

    JuiceFS refuses to format if the storage path is not empty. This function
    deletes all objects under the JuiceFS prefix to allow a fresh deployment.

    Args:
        bucket_name: S3 bucket name (default: from LAION_BUCKET_NAME env var).
        juicefs_prefix: JuiceFS storage prefix (default: "beta9-fs").

    Returns:
        True if cleanup successful, False otherwise.
    """
    import boto3

    # Get config from environment
    config = S3StorageConfig.from_env_optional()
    if config is None:
        logger.error("S3 storage not configured. Set LAION_* environment variables.")
        return False

    bucket = bucket_name or config.bucket_name

    # Unset AWS_SESSION_TOKEN to prevent credential conflicts
    if "AWS_SESSION_TOKEN" in os.environ:
        logger.debug("Unsetting AWS_SESSION_TOKEN to use explicit S3 credentials")
        del os.environ["AWS_SESSION_TOKEN"]

    try:
        client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region,
        )

        # List all objects under the JuiceFS prefix
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=f"{juicefs_prefix}/")

        objects_to_delete = []
        for page in pages:
            for obj in page.get("Contents", []):
                objects_to_delete.append({"Key": obj["Key"]})

        if not objects_to_delete:
            logger.info(f"JuiceFS storage s3://{bucket}/{juicefs_prefix}/ is already empty")
            return True

        # Delete in batches of 1000 (S3 limit)
        total_deleted = 0
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i : i + 1000]
            response = client.delete_objects(
                Bucket=bucket,
                Delete={"Objects": batch, "Quiet": True},
            )
            deleted = len(batch) - len(response.get("Errors", []))
            total_deleted += deleted

            if response.get("Errors"):
                for error in response["Errors"]:
                    logger.warning(f"Failed to delete {error['Key']}: {error['Message']}")

        logger.info(
            f"Cleaned up JuiceFS storage: deleted {total_deleted} objects from "
            f"s3://{bucket}/{juicefs_prefix}/"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to clean up JuiceFS storage: {e}")
        return False
