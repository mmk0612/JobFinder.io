from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import boto3


DEFAULT_AWS_REGION = "us-east-1"
DEFAULT_AWS_S3_BUCKET = "job-finder-resume-s3-bucket"


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise EnvironmentError(f"Missing required AWS setting: {name}")
    return value


def _sanitize_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", filename).strip("._")
    return cleaned or "resume.pdf"


@lru_cache(maxsize=1)
def _get_s3_client():
    region = (os.environ.get("AWS_REGION") or "").strip() or DEFAULT_AWS_REGION
    access_key = _require_env("AWS_ACCESS_KEY_ID")
    secret_key = _require_env("AWS_SECRET_ACCESS_KEY")

    # Endpoint override is optional; useful for testing but defaults to AWS S3.
    endpoint_url = (os.environ.get("AWS_S3_ENDPOINT_URL") or "").strip() or None

    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
    )


def _build_resume_object_key(original_filename: str) -> str:
    safe_name = _sanitize_filename(original_filename)
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else "pdf"
    ts = datetime.now(timezone.utc)
    return (
        f"resumes/{ts.year:04d}/{ts.month:02d}/{ts.day:02d}/"
        f"{ts.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:12]}.{ext}"
    )


def upload_resume_bytes(*, original_filename: str, content_bytes: bytes, content_type: str) -> str:
    bucket = (os.environ.get("AWS_S3_BUCKET") or "").strip() or DEFAULT_AWS_S3_BUCKET
    key = _build_resume_object_key(original_filename)

    client = _get_s3_client()
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=content_bytes,
        ContentType=content_type or "application/pdf",
        ServerSideEncryption="AES256",
    )

    return f"s3://{bucket}/{key}"


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    raw = str(s3_uri or "").strip()
    if not raw.startswith("s3://"):
        raise ValueError(f"Unsupported resume path {raw!r}. Expected s3://bucket/key")

    without_scheme = raw[5:]
    if "/" not in without_scheme:
        raise ValueError(f"Invalid S3 URI {raw!r}. Missing object key")

    bucket, key = without_scheme.split("/", 1)
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI {raw!r}. Missing bucket or key")
    return bucket, key


def download_s3_uri_to_path(*, s3_uri: str, destination_path: str | Path) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    client = _get_s3_client()
    client.download_file(bucket, key, str(destination))
    return destination
