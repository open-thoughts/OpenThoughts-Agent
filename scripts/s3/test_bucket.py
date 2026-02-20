#!/usr/bin/env python3
"""
Quick S3/MinIO bucket connectivity test.

Usage:
    source ../secrets.env
    python scripts/s3/test_bucket.py

Environment variables required:
    LAION_BUCKET_NAME - S3 bucket name
    LAION_ACCESS_KEY  - S3 access key
    LAION_SECRET_KEY  - S3 secret key
    LAION_ENDPOINT    - S3 endpoint URL (e.g., https://just-object.fz-juelich.de:9000)
"""

import os
import sys
from datetime import datetime


def get_env_vars():
    """Load and validate environment variables."""
    vars_needed = {
        "LAION_BUCKET_NAME": os.environ.get("LAION_BUCKET_NAME"),
        "LAION_ACCESS_KEY": os.environ.get("LAION_ACCESS_KEY"),
        "LAION_SECRET_KEY": os.environ.get("LAION_SECRET_KEY"),
        "LAION_ENDPOINT": os.environ.get("LAION_ENDPOINT"),
    }

    print("=" * 60)
    print("S3/MinIO Bucket Connectivity Test")
    print("=" * 60)
    print()
    print("Environment Variables:")
    for name, value in vars_needed.items():
        if value:
            # Mask secrets
            if "KEY" in name and "SECRET" in name:
                display = value[:4] + "***" + value[-4:] if len(value) > 8 else "***"
            elif "ACCESS" in name:
                display = value[:4] + "***" if len(value) > 4 else "***"
            else:
                display = value
            print(f"  {name}: {display}")
        else:
            print(f"  {name}: NOT SET")

    missing = [k for k, v in vars_needed.items() if not v]
    if missing:
        print()
        print(f"ERROR: Missing environment variables: {', '.join(missing)}")
        print("Please run: source ../secrets.env")
        return None

    return vars_needed


def test_endpoint_format(endpoint: str) -> str:
    """Check and fix endpoint format."""
    print()
    print("Checking endpoint format...")

    original = endpoint

    # Check if protocol is missing
    if not endpoint.startswith("http://") and not endpoint.startswith("https://"):
        print(f"  WARNING: Endpoint missing protocol: {endpoint}")
        # Try https first
        endpoint = f"https://{endpoint}"
        print(f"  Adding https:// prefix: {endpoint}")

    print(f"  Final endpoint: {endpoint}")
    return endpoint


def test_boto3_connection(bucket_name: str, access_key: str, secret_key: str, endpoint: str):
    """Test connection using boto3."""
    print()
    print("Testing boto3 connection...")

    try:
        import boto3
        from botocore.exceptions import ClientError, EndpointConnectionError, NoCredentialsError
    except ImportError:
        print("  ERROR: boto3 not installed. Run: pip install boto3")
        return False

    # Try with the endpoint as-is first, then with protocol prefix if needed
    endpoints_to_try = [endpoint]
    if not endpoint.startswith("http"):
        endpoints_to_try = [f"https://{endpoint}", f"http://{endpoint}"]

    for ep in endpoints_to_try:
        print(f"  Trying endpoint: {ep}")

        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=ep,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="us-east-1",  # Required but often ignored by MinIO
            )

            # Test 1: List buckets
            print("  [1/4] Listing buckets...")
            response = s3.list_buckets()
            buckets = [b["Name"] for b in response.get("Buckets", [])]
            print(f"        Found {len(buckets)} buckets: {buckets[:5]}{'...' if len(buckets) > 5 else ''}")

            # Test 2: Check if our bucket exists
            print(f"  [2/4] Checking bucket '{bucket_name}'...")
            if bucket_name in buckets:
                print(f"        Bucket '{bucket_name}' exists")
            else:
                print(f"        WARNING: Bucket '{bucket_name}' not found in list")
                print(f"        Available buckets: {buckets}")

            # Test 3: List objects in bucket (first 10)
            print(f"  [3/4] Listing objects in '{bucket_name}'...")
            try:
                response = s3.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
                objects = response.get("Contents", [])
                print(f"        Found {len(objects)} objects (showing max 10)")
                for obj in objects[:5]:
                    print(f"          - {obj['Key']} ({obj['Size']} bytes)")
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                print(f"        ERROR listing objects: {error_code}")
                if error_code == "NoSuchBucket":
                    print(f"        Bucket '{bucket_name}' does not exist")
                    return False

            # Test 4: Test write access (create and delete test file)
            print("  [4/4] Testing write access...")
            test_key = f"beam-artifacts/_connection_test_{datetime.now().isoformat()}.txt"
            test_content = f"Connection test at {datetime.now().isoformat()}"

            try:
                s3.put_object(Bucket=bucket_name, Key=test_key, Body=test_content.encode())
                print(f"        Created test object: {test_key}")

                # Read it back
                response = s3.get_object(Bucket=bucket_name, Key=test_key)
                read_content = response["Body"].read().decode()
                if read_content == test_content:
                    print("        Read/write verification: OK")
                else:
                    print("        WARNING: Content mismatch!")

                # Delete it
                s3.delete_object(Bucket=bucket_name, Key=test_key)
                print("        Deleted test object")

            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                print(f"        ERROR writing: {error_code} - {e}")
                if error_code == "AccessDenied":
                    print("        You may have read-only access to this bucket")

            print()
            print("  SUCCESS: boto3 connection working!")
            print(f"  Working endpoint: {ep}")
            return True

        except EndpointConnectionError as e:
            print(f"        Connection failed: {e}")
            continue
        except NoCredentialsError:
            print("        ERROR: Invalid credentials")
            return False
        except ClientError as e:
            print(f"        Client error: {e}")
            continue
        except Exception as e:
            print(f"        Unexpected error: {type(e).__name__}: {e}")
            continue

    print()
    print("  FAILED: Could not connect with any endpoint format")
    return False


def test_s3fs_connection(bucket_name: str, access_key: str, secret_key: str, endpoint: str):
    """Test connection using s3fs."""
    print()
    print("Testing s3fs connection...")

    try:
        import s3fs
    except ImportError:
        print("  ERROR: s3fs not installed. Run: pip install s3fs")
        return False

    # Try with the endpoint as-is first, then with protocol prefix if needed
    endpoints_to_try = [endpoint]
    if not endpoint.startswith("http"):
        endpoints_to_try = [f"https://{endpoint}", f"http://{endpoint}"]

    for ep in endpoints_to_try:
        print(f"  Trying endpoint: {ep}")

        try:
            fs = s3fs.S3FileSystem(
                key=access_key,
                secret=secret_key,
                endpoint_url=ep,
                client_kwargs={"region_name": "us-east-1"},
            )

            # Test: List top-level of bucket
            print(f"  Listing '{bucket_name}/'...")
            items = fs.ls(bucket_name, detail=False)
            print(f"        Found {len(items)} items")
            for item in items[:5]:
                print(f"          - {item}")
            if len(items) > 5:
                print(f"          ... and {len(items) - 5} more")

            print()
            print("  SUCCESS: s3fs connection working!")
            print(f"  Working endpoint: {ep}")
            return True

        except Exception as e:
            print(f"        Error: {type(e).__name__}: {e}")
            continue

    print()
    print("  FAILED: Could not connect with any endpoint format")
    return False


def main():
    # Load env vars
    env = get_env_vars()
    if not env:
        return 1

    bucket_name = env["LAION_BUCKET_NAME"]
    access_key = env["LAION_ACCESS_KEY"]
    secret_key = env["LAION_SECRET_KEY"]
    endpoint = env["LAION_ENDPOINT"]

    # Test endpoint format
    endpoint = test_endpoint_format(endpoint)

    # Test boto3
    boto3_ok = test_boto3_connection(bucket_name, access_key, secret_key, endpoint)

    # Test s3fs
    s3fs_ok = test_s3fs_connection(bucket_name, access_key, secret_key, endpoint)

    # Summary
    print()
    print("=" * 60)
    print("Summary:")
    print(f"  boto3: {'OK' if boto3_ok else 'FAILED'}")
    print(f"  s3fs:  {'OK' if s3fs_ok else 'FAILED'}")
    print("=" * 60)

    if boto3_ok:
        print()
        print("Recommended endpoint format for your secrets.env:")
        if not env["LAION_ENDPOINT"].startswith("http"):
            print(f"  LAION_ENDPOINT=https://{env['LAION_ENDPOINT']}")
        else:
            print(f"  LAION_ENDPOINT={env['LAION_ENDPOINT']}")

    return 0 if boto3_ok else 1


if __name__ == "__main__":
    sys.exit(main())
