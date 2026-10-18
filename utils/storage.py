"""Where uploaded files are kept: local disk (default), S3, or nowhere."""
import os
import shutil
from pathlib import Path


class NoStorage:
    backend = "none"

    def save(self, tmp_path, user_id, media_type, name):
        return {"backend": "none"}


class LocalStorage:
    backend = "local"

    def __init__(self, base_dir):
        self.root = Path(base_dir) / "uploads"

    def save(self, tmp_path, user_id, media_type, name):
        try:
            dest = self.root / str(user_id) / media_type
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path, dest / name)
            return {"backend": "local", "path": f"uploads/{user_id}/{media_type}/{name}"}
        except Exception as e:
            return {"backend": "local", "error": str(e)}


class S3Storage:
    backend = "s3"

    def __init__(self, bucket, region):
        import boto3
        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)  # credentials come from AWS_* env vars

    def save(self, tmp_path, user_id, media_type, name):
        key = f"users/{user_id}/{media_type}/{name}"
        try:
            self.client.upload_file(tmp_path, self.bucket, key)
            return {"backend": "s3", "bucket": self.bucket, "key": key}
        except Exception as e:
            return {"backend": "s3", "error": str(e)}


def get_storage(base_dir):
    backend = os.environ.get("STORAGE_BACKEND", "local").lower()
    if backend == "none":
        return NoStorage()
    if backend == "s3":
        bucket = os.environ.get("S3_BUCKET")
        if bucket:
            try:
                return S3Storage(bucket, os.environ.get("S3_REGION", "us-east-1"))
            except Exception as e:
                print(f"[storage] S3 unavailable ({e.__class__.__name__}); using local disk instead.")
        else:
            print("[storage] STORAGE_BACKEND=s3 but S3_BUCKET is not set; using local disk instead.")
    return LocalStorage(base_dir)
